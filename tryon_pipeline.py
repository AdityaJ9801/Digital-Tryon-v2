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
import os

import torch
from PIL import Image
from torchvision.transforms import v2 as T

from config import TryOnConfig
from tryondiffusion import TryOnImagen, TryOnImagenTrainer, get_unet_by_name
from tryondiffusion.preprocessing import default_garment_keypoints, estimate_person_pose, generate_agnostic_image


class TryOnPipeline:
    """Loads a trained TryOnDiffusion checkpoint once and runs repeated inference."""

    def __init__(self, checkpoint_path: str, config: TryOnConfig = None, device: str = None):
        self.config = config or TryOnConfig()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        base_unet = get_unet_by_name(
            "base", image_size=self.config.base_image_size, max_keypoints_len=self.config.max_keypoints
        )
        unets = (base_unet,)
        image_sizes = (self.config.base_image_size,)
        timesteps = self.config.timesteps[0]

        if self.config.use_sr_unet:
            sr_unet = get_unet_by_name(
                "sr", image_size=self.config.sr_image_size, max_keypoints_len=self.config.max_keypoints
            )
            unets = (base_unet, sr_unet)
            image_sizes = (self.config.base_image_size, self.config.sr_image_size)
            timesteps = self.config.timesteps

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
        self.trainer = TryOnImagenTrainer(imagen=imagen, use_ema=True)
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
    def generate(self, person_image, garment_image, cond_scale: float = None, output_path: str = None) -> Image.Image:
        """
        Args:
            person_image: file path or PIL.Image of the person.
            garment_image: file path or PIL.Image of the garment to try on.
            cond_scale: classifier-free-guidance scale (higher = stronger
                garment/pose adherence, typical range 2.0-5.0).
            output_path: if given, saves the generated image there.
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
        if output_path:
            out_dir = os.path.dirname(output_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            result.save(output_path)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Path to a trainer checkpoint .pt file")
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
    )
    pipeline = TryOnPipeline(checkpoint_path=args.checkpoint, config=config)
    pipeline.generate(args.person, args.garment, cond_scale=args.cond_scale, output_path=args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
