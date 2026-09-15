"""
Shared auto-derivation utilities for TryOnDiffusion.

TryOnDiffusion's architecture (tryondiffusion/tryondiffusion.py) conditions
on four signals: the clothing-agnostic person image, the garment image, the
person's pose keypoints, and the garment's pose keypoints. Historically this
repo required all four to be precomputed and supplied by hand per sample.

This module lets every data path (Hugging Face dataset, local dataset,
production inference pipeline) accept just a person image and a garment
image, and derive the remaining conditioning signals automatically:

- `estimate_person_pose`    : pose keypoints via MediaPipe Pose (CPU-friendly)
- `keypoints_from_openpose_json` : parses OpenPose-format JSON when a dataset
                                    already ships pose annotations (cheaper
                                    and more accurate than re-estimating)
- `generate_agnostic_image` : cheap, model-free clothing-agnostic image by
                               blanking out the torso region implied by pose
- `generate_agnostic_image_segmentation` : optional, higher-quality agnostic
                               image using a pretrained clothes-segmentation
                               model from the Hugging Face Hub
- `default_garment_keypoints` : canonical placeholder garment pose
"""
from typing import Optional

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter

_POSE_ESTIMATOR = None
_SEGMENTATION_PIPELINE = None

# MediaPipe's legacy `mp.solutions.pose` API was removed upstream in favor of
# the Tasks API (`mediapipe.tasks.vision.PoseLandmarker`), which needs a
# small model file downloaded once and cached locally.
_POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)

# MediaPipe Pose landmark indices for the torso corners we need to blank out.
_MEDIAPIPE_TORSO_IDX = {"r_shoulder": 12, "l_shoulder": 11, "l_hip": 23, "r_hip": 24}

# OpenPose BODY_25 / COCO-18 indices for the same corners, used when pose
# keypoints come from a dataset's precomputed OpenPose JSON instead of
# MediaPipe (the two formats index joints differently).
_OPENPOSE_TORSO_IDX = {"r_shoulder": 2, "l_shoulder": 5, "l_hip": 12, "r_hip": 9}


def _ensure_pose_landmarker_model() -> str:
    """Downloads (once) and caches the MediaPipe PoseLandmarker model file."""
    import os
    import urllib.request

    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "tryondiffusion")
    os.makedirs(cache_dir, exist_ok=True)
    model_path = os.path.join(cache_dir, "pose_landmarker_lite.task")

    if not os.path.exists(model_path):
        urllib.request.urlretrieve(_POSE_MODEL_URL, model_path)

    return model_path


def get_pose_estimator():
    """Lazily constructs a MediaPipe PoseLandmarker (Tasks API). Loaded once per process."""
    global _POSE_ESTIMATOR
    if _POSE_ESTIMATOR is None:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_ensure_pose_landmarker_model()),
            running_mode=RunningMode.IMAGE,
            num_poses=1,
        )
        try:
            _POSE_ESTIMATOR = PoseLandmarker.create_from_options(options)
        except OSError as e:
            if "libEGL" in str(e) or "libGL" in str(e):
                raise OSError(
                    "MediaPipe's native library needs libEGL/libGL, which is missing on this "
                    "(likely headless/minimal-container) machine. Install it with:\n"
                    "    sudo apt-get update && sudo apt-get install -y libegl1 libgl1 libgbm1\n"
                    "then retry."
                ) from e
            raise
    return _POSE_ESTIMATOR


def _pad_or_truncate(keypoints, max_keypoints: int) -> torch.Tensor:
    kp = torch.tensor(keypoints, dtype=torch.float32) if len(keypoints) else torch.zeros((0, 2))
    n = kp.shape[0]
    if n >= max_keypoints:
        return kp[:max_keypoints].contiguous()
    pad = torch.zeros((max_keypoints - n, 2), dtype=torch.float32)
    return torch.cat([kp, pad], dim=0)


def estimate_person_pose(person_image: Image.Image, max_keypoints: int = 25) -> torch.Tensor:
    """
    Runs MediaPipe Pose on a PIL person image and returns a
    `(max_keypoints, 2)` tensor of pixel-space (x, y) keypoints,
    zero-padded/truncated to `max_keypoints`. Used whenever a dataset (or a
    user, at inference time) does not already provide pose annotations.
    """
    import mediapipe as mp

    pose_landmarker = get_pose_estimator()
    arr = np.array(person_image.convert("RGB"))
    h, w = arr.shape[:2]

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=arr)
    result = pose_landmarker.detect(mp_image)

    keypoints = []
    if result.pose_landmarks:
        for lm in result.pose_landmarks[0]:
            keypoints.append([lm.x * w, lm.y * h])

    return _pad_or_truncate(keypoints, max_keypoints)


def keypoints_from_openpose_json(pose_data, max_keypoints: int = 25) -> torch.Tensor:
    """Parses OpenPose-format JSON (dict, or JSON string) into a keypoints tensor."""
    import json

    if isinstance(pose_data, (str, bytes)):
        pose_data = json.loads(pose_data)

    if "keypoints" in pose_data:
        keypoints = pose_data["keypoints"]
    elif "people" in pose_data and len(pose_data["people"]) > 0:
        raw = pose_data["people"][0].get("pose_keypoints_2d", [])
        keypoints = [[raw[i], raw[i + 1]] for i in range(0, len(raw), 3)]
    else:
        raise ValueError("Unrecognized OpenPose JSON schema (expected 'keypoints' or 'people' key)")

    return _pad_or_truncate(keypoints, max_keypoints)


def default_garment_keypoints(max_keypoints: int = 25) -> torch.Tensor:
    """
    Canonical placeholder garment pose. The paper's Parallel-UNet treats
    garment pose as an optional conditioning signal (it is dropped during
    classifier-free-guidance training), so a fixed zero prior is a safe
    default when a dataset has no per-garment landmark annotations.
    """
    return torch.zeros((max_keypoints, 2), dtype=torch.float32)


def _torso_polygon(w: int, h: int, keypoints: torch.Tensor, idx_map: dict, expand: float):
    pts = keypoints.numpy()

    def valid(i):
        return i < len(pts) and not (pts[i][0] == 0 and pts[i][1] == 0)

    required = [idx_map["r_shoulder"], idx_map["l_shoulder"], idx_map["l_hip"], idx_map["r_hip"]]
    if not all(valid(i) for i in required):
        # Fallback: generous centered box covering the typical torso area.
        return [(w * 0.2, h * 0.12), (w * 0.8, h * 0.12), (w * 0.8, h * 0.78), (w * 0.2, h * 0.78)]

    r_sh, l_sh, l_hip, r_hip = (pts[i] for i in required)
    cx = (r_sh[0] + l_sh[0] + l_hip[0] + r_hip[0]) / 4
    cy = (r_sh[1] + l_sh[1] + l_hip[1] + r_hip[1]) / 4
    return [(cx + (x - cx) * expand, cy + (y - cy) * expand) for x, y in (r_sh, l_sh, l_hip, r_hip)]


def generate_agnostic_image(
    person_image: Image.Image,
    keypoints: torch.Tensor,
    expand: float = 1.35,
    keypoint_format: str = "mediapipe",
) -> Image.Image:
    """
    Cheap, model-free clothing-agnostic image: blanks out the torso polygon
    (shoulders -> hips) implied by the pose keypoints with neutral gray, so
    the network never sees the original garment. This approximates the
    paper's "agnostic RGB" input without requiring a human-parsing/
    segmentation network in the data-loading hot path. For higher fidelity,
    use `generate_agnostic_image_segmentation` instead.
    """
    w, h = person_image.size
    idx_map = _MEDIAPIPE_TORSO_IDX if keypoint_format == "mediapipe" else _OPENPOSE_TORSO_IDX
    polygon = _torso_polygon(w, h, keypoints, idx_map, expand)

    agnostic = person_image.convert("RGB").copy()
    mask = Image.new("L", agnostic.size, 0)
    ImageDraw.Draw(mask).polygon(polygon, fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=max(2, int(min(w, h) * 0.01))))

    gray = Image.new("RGB", agnostic.size, (127, 127, 127))
    return Image.composite(gray, agnostic, mask)


def get_segmentation_pipeline(model_id: str = "mattmdjaga/segformer_b2_clothes"):
    """Lazily loads a pretrained clothes-segmentation model from the HF Hub."""
    global _SEGMENTATION_PIPELINE
    if _SEGMENTATION_PIPELINE is None:
        from transformers import pipeline

        _SEGMENTATION_PIPELINE = pipeline("image-segmentation", model=model_id)
    return _SEGMENTATION_PIPELINE


# Labels from mattmdjaga/segformer_b2_clothes that correspond to worn garments.
_GARMENT_LABELS = {"Upper-clothes", "Dress", "Skirt", "Pants", "Belt"}


def generate_agnostic_image_segmentation(person_image: Image.Image, model_id: Optional[str] = None) -> Image.Image:
    """
    Higher-quality clothing-agnostic image using a pretrained human/clothes
    parsing model from the Hugging Face Hub. Slower than
    `generate_agnostic_image` (needs a forward pass through a segmentation
    network) but produces a tighter, more realistic garment-removal mask —
    recommended for production data pipelines when throughput allows it.
    """
    seg = get_segmentation_pipeline(model_id) if model_id else get_segmentation_pipeline()
    person_image = person_image.convert("RGB")
    results = seg(person_image)

    mask = Image.new("L", person_image.size, 0)
    for r in results:
        if r["label"] in _GARMENT_LABELS:
            mask = Image.composite(Image.new("L", person_image.size, 255), mask, r["mask"])

    mask = mask.filter(ImageFilter.GaussianBlur(radius=max(2, int(min(person_image.size) * 0.01))))
    gray = Image.new("RGB", person_image.size, (127, 127, 127))
    return Image.composite(gray, person_image, mask)
