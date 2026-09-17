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

# Arm indices (shoulder/elbow/wrist) for masking sleeves - a torso-only mask
# leaves long sleeves fully visible (confirmed visually: a mask covering only
# the shoulder-to-hip box left both sleeves of a long-sleeve top untouched),
# which gives the model every reason to reproduce the original garment
# instead of the new one it's supposed to be conditioned on.
_MEDIAPIPE_ARM_IDX = {
    "l_shoulder": 11, "l_elbow": 13, "l_wrist": 15,
    "r_shoulder": 12, "r_elbow": 14, "r_wrist": 16,
}
_OPENPOSE_ARM_IDX = {
    "l_shoulder": 5, "l_elbow": 6, "l_wrist": 7,
    "r_shoulder": 2, "r_elbow": 3, "r_wrist": 4,
}


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
    """
    Lazily constructs a MediaPipe pose estimator, preferring whichever API the
    installed `mediapipe` version supports. Returns `("solutions", estimator)`
    or `("tasks", estimator)` so `estimate_person_pose` knows how to call it.

    mediapipe<1.0's legacy `solutions.pose.Pose` runs a CPU-only graph with no
    OpenGL/EGL dependency - the right choice on headless/minimal/no-root
    servers. mediapipe>=1.0 dropped `solutions` in favor of the Tasks API
    (`PoseLandmarker`), whose native library unconditionally links against
    libEGL/libGL even for CPU inference; that requires installing those
    system libraries (root/apt access), so it's only used as a fallback.
    """
    global _POSE_ESTIMATOR
    if _POSE_ESTIMATOR is not None:
        return _POSE_ESTIMATOR

    import mediapipe as mp

    if hasattr(mp, "solutions"):
        estimator = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1)
        _POSE_ESTIMATOR = ("solutions", estimator)
        return _POSE_ESTIMATOR

    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=_ensure_pose_landmarker_model()),
        running_mode=RunningMode.IMAGE,
        num_poses=1,
    )
    try:
        estimator = PoseLandmarker.create_from_options(options)
    except OSError as e:
        if "libEGL" in str(e) or "libGL" in str(e):
            raise OSError(
                "MediaPipe's Tasks API needs libEGL/libGL, which is missing on this "
                "(likely headless/minimal-container) machine.\n"
                "If you have root/sudo:\n"
                "    sudo apt-get update && sudo apt-get install -y libegl1 libgl1 libgbm1\n"
                "If you don't have root, pin mediapipe to the last version with the older "
                "'solutions' API instead (pip-only, no root needed, no EGL/GL dependency) - see "
                "requirements.txt and README -> Setup for the exact two-command sequence "
                "(a plain `pip install mediapipe==0.10.14` alone breaks wandb via a protobuf "
                "conflict, so a specific follow-up command is required too):\n"
                "    pip install mediapipe==0.10.14 && pip install -U 'protobuf>=5,<6'"
            ) from e
        raise
    _POSE_ESTIMATOR = ("tasks", estimator)
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
    kind, estimator = get_pose_estimator()
    arr = np.array(person_image.convert("RGB"))
    h, w = arr.shape[:2]

    keypoints = []
    if kind == "solutions":
        result = estimator.process(arr)
        if result.pose_landmarks:
            for lm in result.pose_landmarks.landmark:
                keypoints.append([lm.x * w, lm.y * h])
    else:
        import mediapipe as mp

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=arr)
        result = estimator.detect(mp_image)
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
    polygon = [(cx + (x - cx) * expand, cy + (y - cy) * expand) for x, y in (r_sh, l_sh, l_hip, r_hip)]

    # Nudge the top two corners (shoulders) further up, so a crew-neck/high
    # collar sitting slightly above the shoulder line still gets covered.
    shoulder_width = abs(l_sh[0] - r_sh[0])
    neck_lift = shoulder_width * 0.15
    polygon[0] = (polygon[0][0], polygon[0][1] - neck_lift)
    polygon[1] = (polygon[1][0], polygon[1][1] - neck_lift)
    return polygon


def _arm_lines(keypoints: torch.Tensor, idx_map: dict):
    """Returns a list of (point_a, point_b) segments tracing each visible arm
    (shoulder->elbow, elbow->wrist), skipping any joint pose estimation
    didn't find (reported as (0, 0))."""
    pts = keypoints.numpy()

    def valid(i):
        return i < len(pts) and not (pts[i][0] == 0 and pts[i][1] == 0)

    segments = []
    for side in ("l", "r"):
        joint_names = [f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist"]
        joint_idx = [idx_map.get(name) for name in joint_names]
        joints = [pts[i] for i in joint_idx if i is not None and valid(i)]
        for a, b in zip(joints, joints[1:]):
            segments.append((tuple(a), tuple(b)))
    return segments


def generate_agnostic_image(
    person_image: Image.Image,
    keypoints: torch.Tensor,
    expand: float = 1.35,
    keypoint_format: str = "mediapipe",
    arm_width_frac: float = 0.16,
) -> Image.Image:
    """
    Cheap, model-free clothing-agnostic image: blanks out the torso polygon
    (shoulders -> hips) AND both arms (shoulder->elbow->wrist, as thick
    capsules) implied by the pose keypoints with neutral gray, so the network
    never sees the original garment - including long sleeves, which a
    torso-only mask leaves fully visible. This approximates the paper's
    "agnostic RGB" input without requiring a human-parsing/segmentation
    network in the data-loading hot path. For higher fidelity, use
    `generate_agnostic_image_segmentation` instead.
    """
    w, h = person_image.size
    idx_map_torso = _MEDIAPIPE_TORSO_IDX if keypoint_format == "mediapipe" else _OPENPOSE_TORSO_IDX
    idx_map_arms = _MEDIAPIPE_ARM_IDX if keypoint_format == "mediapipe" else _OPENPOSE_ARM_IDX
    polygon = _torso_polygon(w, h, keypoints, idx_map_torso, expand)
    arm_segments = _arm_lines(keypoints, idx_map_arms)

    agnostic = person_image.convert("RGB").copy()
    mask = Image.new("L", agnostic.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon(polygon, fill=255)

    arm_width = max(4, int(min(w, h) * arm_width_frac))
    for a, b in arm_segments:
        draw.line([a, b], fill=255, width=arm_width)
        # round the joints so consecutive segments don't leave visible notches
        for cx, cy in (a, b):
            draw.ellipse(
                [cx - arm_width / 2, cy - arm_width / 2, cx + arm_width / 2, cy + arm_width / 2], fill=255
            )

    mask = mask.filter(ImageFilter.GaussianBlur(radius=max(2, int(min(w, h) * 0.01))))

    gray = Image.new("RGB", agnostic.size, (127, 127, 127))
    return Image.composite(gray, agnostic, mask)


def get_segmentation_pipeline(model_id: str = "mattmdjaga/segformer_b2_clothes", device=None):
    """Lazily loads a pretrained clothes-segmentation model from the HF Hub."""
    global _SEGMENTATION_PIPELINE
    if _SEGMENTATION_PIPELINE is None:
        from transformers import pipeline

        pipeline_kwargs = {}
        if device is not None:
            # transformers' `device` kwarg wants an int GPU index (or -1 for
            # CPU), not a torch.device - normalize either form.
            pipeline_kwargs["device"] = device.index if isinstance(device, torch.device) and device.type == "cuda" else (
                -1 if (isinstance(device, torch.device) and device.type == "cpu") else device
            )
        _SEGMENTATION_PIPELINE = pipeline("image-segmentation", model=model_id, **pipeline_kwargs)
    return _SEGMENTATION_PIPELINE


# Labels from mattmdjaga/segformer_b2_clothes corresponding to the WORN UPPER
# GARMENT specifically - this repo's task is upper-body try-on only (matching
# the "cloth" folder convention in VITON-HD/Zalando-style datasets), so
# deliberately excludes Skirt/Pants/Belt/shoes/etc: those should stay visible
# and unchanged in both the agnostic input and the generated output, not be
# masked out and left for the model to hallucinate back in.
_GARMENT_LABELS = {"Upper-clothes", "Dress"}


def generate_agnostic_image_segmentation(
    person_image: Image.Image,
    model_id: Optional[str] = None,
    device=None,
    fallback_keypoints: Optional[torch.Tensor] = None,
    fallback_keypoint_format: str = "mediapipe",
) -> Image.Image:
    """
    Higher-quality clothing-agnostic image using a pretrained human/clothes
    parsing model from the Hugging Face Hub - a real per-pixel garment mask,
    not a geometric approximation. Slower than `generate_agnostic_image`
    (needs a forward pass through a segmentation network per image) but
    produces much tighter, more accurate garment removal.

    If segmentation finds no upper-garment region at all (unusual image, or
    a model hiccup), falls back to the geometric heuristic
    (`generate_agnostic_image`) rather than silently returning the person
    image completely unmasked - pass `fallback_keypoints` (and matching
    `fallback_keypoint_format`) to enable this safety net.
    """
    seg = get_segmentation_pipeline(model_id, device=device) if model_id else get_segmentation_pipeline(device=device)
    person_image = person_image.convert("RGB")
    results = seg(person_image)

    mask = Image.new("L", person_image.size, 0)
    found_garment = False
    for r in results:
        if r["label"] in _GARMENT_LABELS:
            mask = Image.composite(Image.new("L", person_image.size, 255), mask, r["mask"])
            found_garment = True

    if not found_garment:
        if fallback_keypoints is not None:
            return generate_agnostic_image(person_image, fallback_keypoints, keypoint_format=fallback_keypoint_format)
        return person_image

    # Dilate slightly before blurring, so small gaps right at the segmentation
    # boundary (e.g. a sliver of collar the model didn't quite classify as
    # garment) don't leave a thin strip of the original clothing visible.
    dilate_size = max(3, int(min(person_image.size) * 0.01) | 1)  # MaxFilter needs an odd size
    mask = mask.filter(ImageFilter.MaxFilter(dilate_size))
    mask = mask.filter(ImageFilter.GaussianBlur(radius=max(2, int(min(person_image.size) * 0.01))))
    gray = Image.new("RGB", person_image.size, (127, 127, 127))
    return Image.composite(gray, person_image, mask)


def derive_agnostic_image(
    person_image: Image.Image,
    keypoints: torch.Tensor,
    method: str = "segmentation",
    keypoint_format: str = "mediapipe",
    device=None,
) -> Image.Image:
    """
    Single entry point used by RealTryonDataset, HFTryOnDataset, and
    TryOnPipeline to auto-derive the clothing-agnostic image, so all three
    switch between methods (TryOnConfig.agnostic_method) the same way.
    """
    if method == "segmentation":
        return generate_agnostic_image_segmentation(
            person_image, device=device, fallback_keypoints=keypoints, fallback_keypoint_format=keypoint_format
        )
    if method == "heuristic":
        return generate_agnostic_image(person_image, keypoints, keypoint_format=keypoint_format)
    raise ValueError(f"Unknown agnostic_method: {method!r} (expected 'segmentation' or 'heuristic')")
