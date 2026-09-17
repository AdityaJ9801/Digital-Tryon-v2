"""
Production inference pipeline: person image + garment image -> try-on image.

This is the one class app.py (Gradio demo) and any custom serving code
should use. Only two images are required from the caller; the
clothing-agnostic representation and pose keypoints the model was trained
with are derived automatically (see tryondiffusion/preprocessing.py).

CLI usage:
    python tryon_pipeline.py --checkpoint ./checkpoints/checkpoint.50000.pt \\
        --person ./examples/person.jpg --garment ./examples/garment.jpg \\
        --output ./out.png
"""
import argparse
import glob
import os

import torch
from PIL import Image
from torchvision.transforms import v2 as T

from config import TryOnConfig
from tryondiffusion import TryOnImagen, TryOnImagenTrainer, get_unet_by_name
from tryondiffusion.preprocessing import default_garment_keypoints, estimate_person_pose, generate_agnostic_image


def find_latest_checkpoint(path: str) -> str:
    """
    Resolves `path` to a checkpoint file. If `path` is already a .pt file,
    returns it as-is. If it's a directory, finds the highest-step
    checkpoint.<N>.pt inside it (same naming/sorting TryOnImagenTrainer's
    own checkpointing uses), so callers can just point at a checkpoint
    directory and always get the newest one without tracking step numbers.
    """
    if os.path.isfile(path):
        return path

    if not os.path.isdir(path):
        raise FileNotFoundError(f"Checkpoint path not found: {path}")

    candidates = glob.glob(os.path.join(path, "checkpoint.*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint.<N>.pt files found in directory: {path}")

    def step_of(p):
        try:
            return int(os.path.basename(p).split(".")[-2])
        except (IndexError, ValueError):
            return -1

    latest = max(candidates, key=step_of)
    print(f"Auto-selected newest checkpoint: {latest} (of {len(candidates)} found in {path})")
    return latest


def find_globally_latest_checkpoint(root: str = ".", dir_pattern: str = "checkpoints*") -> str:
    """
    Scans every directory under `root` matching `dir_pattern` (by default,
    every training lineage: checkpoints_base, checkpoints_kaggle_hf_zalando,
    checkpoints_sr_stage1, etc.) for checkpoint.<N>.pt files, and returns
    whichever single file was modified most recently.

    Uses file modification time, not step number, because step counters are
    independent per lineage (e.g. a base-only lineage's step 47400 and an
    SR-stage lineage's step 3000 aren't comparable) - the most recently
    *written* checkpoint, whichever folder it's in, is "the latest model".

    Note: different lineages can differ in unet_number/resolution (e.g. a
    base-only checkpoint vs. an SR-stage one). This function only picks the
    freshest *file* - it does not know which config it needs, so the caller
    should print which directory got picked (see app.py) so a mismatch is
    obvious rather than silently loading with the wrong assumptions.
    """
    candidates = glob.glob(os.path.join(root, dir_pattern, "checkpoint.*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint.<N>.pt files found under {root}/{dir_pattern}/")

    latest = max(candidates, key=os.path.getmtime)
    print(f"Auto-selected globally newest checkpoint (by modification time): {latest}")
    return latest


class TryOnPipeline:
    """Loads a trained TryOnDiffusion checkpoint once and runs repeated inference."""

    def __init__(self, checkpoint_path: str, config: TryOnConfig = None, device: str = None):
        self.config = config or TryOnConfig()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        checkpoint_path = find_latest_checkpoint(checkpoint_path)

        base_unet = get_unet_by_name(
            "base", image_size=self.config.base_image_size, max_keypoints_len=self.config.max_keypoints
        )
        unets = (base_unet,)
        image_sizes = (self.config.base_image_size,)
        # Inference-only sampling-step count, independent of training - see
        # TryOnConfig.inference_timesteps for why this is safe to differ.
        timesteps = self.config.inference_timesteps

        if self.config.use_sr_unet:
            sr_unet = get_unet_by_name(
                "sr", image_size=self.config.sr_image_size, max_keypoints_len=self.config.max_keypoints
            )
            unets = (base_unet, sr_unet)
            image_sizes = (self.config.base_image_size, self.config.sr_image_size)
            timesteps = (self.config.inference_timesteps, self.config.inference_timesteps)

        imagen = TryOnImagen(
            unets=unets,
            image_sizes=image_sizes,
            timesteps=timesteps,
            pred_objectives="noise" if not self.config.use_sr_unet else ("noise", "noise"),
            noise_schedules="cosine",
            auto_normalize_img=True,
            dynamic_thresholding=True,
        )

        print("Loading trainer / checkpoint...")
        # bf16 autocast during sampling is a free speedup on GPU (Ampere+
        # tensor cores) with negligible quality impact; irrelevant on CPU.
        precision = "bf16" if self.device.type == "cuda" else None
        self.trainer = TryOnImagenTrainer(imagen=imagen, use_ema=True, precision=precision)
        self.trainer.to(self.device)

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        self.trainer.load(checkpoint_path)

        self.image_transforms = T.Compose(
            [
                T.ToImage(),
                T.Resize(self.config.base_image_size),
                T.ToDtype(torch.float32, scale=True),
            ]
        )
        print("Pipeline ready.")

    @staticmethod
    def _prep_image(image) -> Image.Image:
        if isinstance(image, str):
            return Image.open(image).convert("RGB")
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        raise TypeError("image must be a file path or a PIL.Image")

    @torch.no_grad()
    def generate(
        self,
        person_image,
        garment_image,
        cond_scale: float = None,
        output_path: str = None,
        return_debug: bool = False,
    ):
        """
        Args:
            person_image: file path or PIL.Image of the person.
            garment_image: file path or PIL.Image of the garment to try on.
            cond_scale: classifier-free-guidance scale (higher = stronger
                garment/pose adherence, typical range 2.0-5.0).
            output_path: if given, saves the generated image there.
            return_debug: if True, returns (result_image, debug_dict) instead
                of just result_image. debug_dict["ca_image"] is the
                auto-generated clothing-agnostic image actually fed to the
                model - inspect this when generated garments look wrong or
                the original clothing doesn't seem removed: if the mask
                isn't fully covering the worn garment in this image, the
                model can still see the original clothing and has every
                reason to reproduce it instead of the new garment.
        """
        cond_scale = cond_scale if cond_scale is not None else self.config.cond_scale

        person_image = self._prep_image(person_image)
        garment_image = self._prep_image(garment_image)

        person_pose = estimate_person_pose(person_image, self.config.max_keypoints)
        ca_image = generate_agnostic_image(person_image, person_pose, keypoint_format="mediapipe")
        garment_pose = default_garment_keypoints(self.config.max_keypoints)

        ow, oh = person_image.size
        th, tw = self.config.base_image_size
        scale = torch.tensor([tw / ow, th / oh], dtype=torch.float32)

        conditioning = dict(
            ca_images=self.image_transforms(ca_image).unsqueeze(0).to(self.device),
            garment_images=self.image_transforms(garment_image).unsqueeze(0).to(self.device),
            person_poses=(person_pose * scale).unsqueeze(0).to(self.device),
            garment_poses=garment_pose.unsqueeze(0).to(self.device),
        )

        images = self.trainer.sample(
            batch_size=1,
            **conditioning,
            cond_scale=cond_scale,
            return_pil_images=True,
            use_tqdm=True,
        )

        result = images[0]

        if return_debug:
            debug = {"ca_image": ca_image, "person_pose": person_pose}

        if output_path:
            out_dir = os.path.dirname(output_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            result.save(output_path)

        if return_debug:
            return result, debug
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to a trainer checkpoint .pt file, OR a checkpoint directory - if a directory, "
             "the highest-step checkpoint.<N>.pt inside it is used automatically."
    )
    parser.add_argument(
        "--inference-steps", type=int, default=50,
        help="Denoising sampling steps at inference - independent of the training timesteps "
             "(this model uses continuous-time diffusion; see TryOnConfig.inference_timesteps). "
             "Lower = faster, less detail (try 20-30 for quick iteration); higher = slower, more "
             "detail (try 100-250 for a final render)."
    )
    parser.add_argument("--person", required=True, help="Path to the person image")
    parser.add_argument("--garment", required=True, help="Path to the garment image")
    parser.add_argument("--output", default="./output.png", help="Where to save the generated image")
    parser.add_argument("--cond-scale", type=float, default=3.0, help="Classifier-free-guidance scale")
    parser.add_argument("--use-sr-unet", action="store_true", help="Run the two-stage base+SR cascade")
    parser.add_argument(
        "--base-image-size", type=int, nargs=2, default=[256, 256],
        help="MUST match the --base-image-size the checkpoint was trained with, or loading will silently "
             "partial-load (mismatched-shape layers get skipped) and generation will look broken/random."
    )
    parser.add_argument(
        "--sr-image-size", type=int, nargs=2, default=[512, 512],
        help="MUST match the --sr-image-size the checkpoint was trained with (only used with --use-sr-unet)."
    )
    parser.add_argument("--max-keypoints", type=int, default=25, help="Must match the training --max-keypoints")
    args = parser.parse_args()

    config = TryOnConfig(
        unet_number=2 if args.use_sr_unet else 1,
        base_image_size=tuple(args.base_image_size),
        sr_image_size=tuple(args.sr_image_size),
        max_keypoints=args.max_keypoints,
        inference_timesteps=args.inference_steps,
    )
    pipeline = TryOnPipeline(checkpoint_path=args.checkpoint, config=config)
    pipeline.generate(args.person, args.garment, cond_scale=args.cond_scale, output_path=args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
