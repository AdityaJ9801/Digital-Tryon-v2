"""
Hugging Face Hub-backed dataset for TryOnDiffusion.

Replaces the old workflow of manually downloading VITON-HD, hand-building a
`tryon_mapping.csv`, and organizing folders on disk. Instead, this streams
directly from any dataset on the Hugging Face Hub that has (at minimum) a
person-image column and a garment/cloth-image column — the clothing-agnostic
image, person pose, and garment pose are derived automatically if the
dataset doesn't already provide them (see tryondiffusion/preprocessing.py).

Default dataset: `SaffalPoosh/VITON-HD-test` (2,032 samples, the standard
VITON-HD benchmark set, packaged with `image`/`cloth`/`openpose_json`
columns). Point `repo_id` at any other dataset with a compatible schema —
e.g. a larger private VITON-HD-style dataset — for production-scale
training.
"""
import io
from typing import Optional

import torch
from PIL import Image
from torchvision.transforms import v2 as T

from tryondiffusion.preprocessing import (
    default_garment_keypoints,
    estimate_person_pose,
    generate_agnostic_image,
    keypoints_from_openpose_json,
)


class _HFTryOnBase:
    """Shared field setup + sample-building logic for both dataset variants below."""

    def __init__(
        self,
        repo_id: str,
        split: str,
        image_size,
        max_keypoints: int,
        person_image_column: str,
        cloth_image_column: str,
        pose_column: Optional[str],
        agnostic_image_column: Optional[str],
        streaming: bool,
        hf_token: Optional[str],
    ):
        from datasets import load_dataset  # local import: keep `datasets` an optional dependency at import time

        self.image_size = image_size
        self.max_keypoints = max_keypoints
        self.person_image_column = person_image_column
        self.cloth_image_column = cloth_image_column
        self.pose_column = pose_column
        self.agnostic_image_column = agnostic_image_column
        self.streaming = streaming

        self.dataset = load_dataset(repo_id, split=split, streaming=streaming, token=hf_token)

        if not streaming:
            columns = self.dataset.column_names
            missing = [c for c in (person_image_column, cloth_image_column) if c not in columns]
            if missing:
                raise ValueError(
                    f"Dataset '{repo_id}' is missing required column(s) {missing}. "
                    f"Available columns: {columns}. Pass person_image_column= / "
                    f"cloth_image_column= to match this dataset's schema."
                )

        self.transforms = T.Compose(
            [
                T.ToImage(),
                T.Resize(image_size),
                T.ToDtype(torch.float32, scale=True),
            ]
        )

    def __len__(self):
        if self.streaming:
            raise TypeError("len() is not supported for a streaming HFTryOnDataset")
        return len(self.dataset)

    @staticmethod
    def _to_pil(value) -> Image.Image:
        if isinstance(value, Image.Image):
            return value.convert("RGB")
        if isinstance(value, (bytes, bytearray)):
            return Image.open(io.BytesIO(value)).convert("RGB")
        if isinstance(value, dict) and "bytes" in value and value["bytes"] is not None:
            return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
        if isinstance(value, str):
            return Image.open(value).convert("RGB")
        raise TypeError(f"Unsupported image field type: {type(value)}")

    def _build_sample(self, row):
        person_image = self._to_pil(row[self.person_image_column])
        garment_image = self._to_pil(row[self.cloth_image_column])

        # Person pose: prefer the dataset's own OpenPose annotations (accurate,
        # free); fall back to on-the-fly MediaPipe estimation otherwise.
        raw_pose = row.get(self.pose_column) if self.pose_column else None
        if raw_pose:
            person_pose = keypoints_from_openpose_json(raw_pose, self.max_keypoints)
            keypoint_format = "openpose"
        else:
            person_pose = estimate_person_pose(person_image, self.max_keypoints)
            keypoint_format = "mediapipe"

        # Clothing-agnostic image: prefer the dataset's own, else derive it.
        raw_agnostic = row.get(self.agnostic_image_column) if self.agnostic_image_column else None
        if raw_agnostic:
            ca_image = self._to_pil(raw_agnostic)
        else:
            ca_image = generate_agnostic_image(person_image, person_pose, keypoint_format=keypoint_format)

        garment_pose = default_garment_keypoints(self.max_keypoints)

        # Rescale keypoints (originally in the source image's pixel space)
        # into the resized tensor's coordinate space.
        ow, oh = person_image.size
        th, tw = self.image_size
        scale = torch.tensor([tw / ow, th / oh], dtype=torch.float32)

        return {
            "person_images": self.transforms(person_image),
            "ca_images": self.transforms(ca_image),
            "garment_images": self.transforms(garment_image),
            "person_poses": person_pose * scale,
            "garment_poses": garment_pose,
        }

    def __getitem__(self, idx):
        return self._build_sample(self.dataset[idx])

    def __iter__(self):
        for row in self.dataset:
            yield self._build_sample(row)


class HFTryOnDataset(_HFTryOnBase, torch.utils.data.Dataset):
    """Map-style dataset (random access, works with `shuffle=True`). Requires streaming=False."""

    def __init__(self, *args, **kwargs):
        kwargs["streaming"] = False
        super().__init__(*args, **kwargs)


class HFTryOnIterableDataset(_HFTryOnBase, torch.utils.data.IterableDataset):
    """
    Streaming dataset: no random-access indexing or `len()`, but starts
    training immediately without downloading the full dataset first — useful
    for very large production datasets. Pass `shuffle=False` to the
    DataLoader (shuffling happens via `.shuffle(buffer_size=...)` on the
    underlying HF dataset instead, see `build_hf_tryon_dataset`).
    """

    def __init__(self, *args, **kwargs):
        kwargs["streaming"] = True
        super().__init__(*args, **kwargs)


def build_hf_tryon_dataset(
    repo_id: str = "SaffalPoosh/VITON-HD-test",
    split: str = "train",
    image_size=(256, 256),
    max_keypoints: int = 25,
    person_image_column: str = "image",
    cloth_image_column: str = "cloth",
    pose_column: Optional[str] = "openpose_json",
    agnostic_image_column: Optional[str] = None,
    streaming: bool = False,
    shuffle_buffer_size: int = 1000,
    hf_token: Optional[str] = None,
):
    """Factory that returns a map-style or streaming HF dataset depending on `streaming`."""
    klass = HFTryOnIterableDataset if streaming else HFTryOnDataset
    ds = klass(
        repo_id=repo_id,
        split=split,
        image_size=image_size,
        max_keypoints=max_keypoints,
        person_image_column=person_image_column,
        cloth_image_column=cloth_image_column,
        pose_column=pose_column,
        agnostic_image_column=agnostic_image_column,
        hf_token=hf_token,
    )
    if streaming:
        ds.dataset = ds.dataset.shuffle(buffer_size=shuffle_buffer_size, seed=42)
    return ds
