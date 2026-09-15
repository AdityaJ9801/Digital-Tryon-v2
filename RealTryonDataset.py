"""
Local-disk dataset for TryOnDiffusion.

Only two things are required per sample: a person image and a garment
image. If a precomputed clothing-agnostic image and/or OpenPose JSON pose
file also happen to be present on disk they are used directly (faster and
more accurate); otherwise both are derived automatically from the person
image, exactly like `tryondiffusion.hf_dataset.HFTryOnDataset` does for
Hugging Face Hub datasets.

Expected layout under `root`:
    root/
      tryon_mapping.csv        # columns: person_images, garment_images[, ca_images, person_pose_path]
      person_images/<file>
      garment_images/<file>
      ca_images/<file>              (optional)
      person_pose_path/<file>.json  (optional, OpenPose format)

Use csv_mapping.py to auto-generate tryon_mapping.csv from folder contents.
"""
import json
import os

import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import v2 as T

from tryondiffusion.preprocessing import (
    default_garment_keypoints,
    estimate_person_pose,
    generate_agnostic_image,
    keypoints_from_openpose_json,
)


class RealTryonDataset(torch.utils.data.Dataset):
    def __init__(self, root, image_size=(256, 256), mapping_file="tryon_mapping.csv", max_keypoints=25):
        self.root = root
        self.image_size = image_size
        self.max_keypoints = max_keypoints
        self.mapping_file_path = os.path.join(root, mapping_file)

        if not os.path.exists(self.mapping_file_path):
            raise FileNotFoundError(f"Mapping file not found at: {self.mapping_file_path}")

        self.data_map = pd.read_csv(self.mapping_file_path)

        required_cols = ["person_images", "garment_images"]
        missing = [c for c in required_cols if c not in self.data_map.columns]
        if missing:
            raise ValueError(f"Mapping file must contain columns: {required_cols} (missing {missing})")

        self.has_ca = "ca_images" in self.data_map.columns
        self.has_pose = "person_pose_path" in self.data_map.columns

        self.transforms = T.Compose(
            [
                T.ToImage(),
                T.Resize(image_size),
                T.ToDtype(torch.float32, scale=True),
            ]
        )

    def __len__(self):
        return len(self.data_map)

    def __getitem__(self, idx):
        try:
            row = self.data_map.iloc[idx]

            person_path = os.path.join(self.root, "person_images", row["person_images"])
            garment_path = os.path.join(self.root, "garment_images", row["garment_images"])

            if not os.path.exists(person_path):
                raise FileNotFoundError(f"Image not found: {person_path}")
            if not os.path.exists(garment_path):
                raise FileNotFoundError(f"Image not found: {garment_path}")

            person_image = Image.open(person_path).convert("RGB")
            garment_image = Image.open(garment_path).convert("RGB")

            # Pose: use precomputed OpenPose JSON if available, else estimate it.
            pose_value = row.get("person_pose_path") if self.has_pose else None
            if self.has_pose and isinstance(pose_value, str) and pose_value:
                pose_path = os.path.join(self.root, "person_pose_path", pose_value)
                with open(pose_path, "r") as f:
                    person_pose = keypoints_from_openpose_json(json.load(f), self.max_keypoints)
                keypoint_format = "openpose"
            else:
                person_pose = estimate_person_pose(person_image, self.max_keypoints)
                keypoint_format = "mediapipe"

            # Clothing-agnostic image: use precomputed if available, else derive it.
            ca_value = row.get("ca_images") if self.has_ca else None
            if self.has_ca and isinstance(ca_value, str) and ca_value:
                ca_path = os.path.join(self.root, "ca_images", ca_value)
                ca_image = Image.open(ca_path).convert("RGB")
            else:
                ca_image = generate_agnostic_image(person_image, person_pose, keypoint_format=keypoint_format)

            garment_pose = default_garment_keypoints(self.max_keypoints)

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

        except Exception as e:
            print(f"[ERROR] Failed to load index {idx}: {e}")
            raise

    def __repr__(self):
        lines = [f"<RealTryonDataset | Root: {self.root} | Samples: {len(self)}>\n"]
        lines.append(f"{'person':<24}{'garment'}")
        for idx in range(min(5, len(self.data_map))):
            row = self.data_map.iloc[idx]
            lines.append(f"{row['person_images']:<24}{row['garment_images']}")
        return "\n".join(lines)
