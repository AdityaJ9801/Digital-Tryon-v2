"""
Central configuration for TryOnDiffusion training and inference.

Every runnable script (trainer.py, tryon_pipeline.py, app.py) builds a
`TryOnConfig` and optionally overrides fields from the CLI, so resolution,
data source, and hardware settings never have to be edited by hand inside
multiple files at once.
"""
import argparse
from dataclasses import dataclass, field, fields
from typing import Optional, Tuple


@dataclass
class TryOnConfig:
    # ---------------------------------------------------------------- data
    # Only a person-image column and a garment/cloth-image column are
    # required from the source dataset. Pose and the clothing-agnostic
    # image are derived automatically (see tryondiffusion/preprocessing.py)
    # unless the dataset already provides them.
    data_source: str = "huggingface"  # "huggingface" | "local"

    hf_dataset_id: str = "SaffalPoosh/VITON-HD-test"
    hf_split: str = "train"
    hf_streaming: bool = False
    hf_person_image_column: str = "image"
    hf_cloth_image_column: str = "cloth"
    hf_pose_column: Optional[str] = "openpose_json"
    hf_agnostic_image_column: Optional[str] = None

    local_root: str = "./data"
    local_mapping_file: str = "tryon_mapping.csv"
    # Subfolder names for the local data path - override these to match a
    # dataset that uses different naming (e.g. Kaggle VITON-HD-style dumps
    # use `image`/`cloth` instead of `person_images`/`garment_images`).
    local_person_folder: str = "person_images"
    local_garment_folder: str = "garment_images"
    local_ca_folder: str = "ca_images"
    local_pose_folder: str = "person_pose_path"

    # ------------------------------------------------------------ resolution
    # Base U-Net produces the low-resolution "structure" image; the SR U-Net
    # upscales + refines it. Both were bumped up from the original 128/256
    # defaults for sharper, more photorealistic output.
    base_image_size: Tuple[int, int] = (256, 256)
    sr_image_size: Tuple[int, int] = (512, 512)
    max_keypoints: int = 25

    # ------------------------------------------------------------- training
    unet_number: int = 1  # 1 = train base U-Net, 2 = train SR U-Net
    use_sr_unet: bool = False  # set True automatically when unet_number == 2

    epochs: int = 100
    max_steps: Optional[int] = None
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    timesteps: Tuple[int, int] = (1000, 1000)
    cond_scale: float = 3.0

    # Sampling-step count used at INFERENCE only, independent of training.
    # This model uses a continuous-time diffusion formulation (see
    # GaussianDiffusionContinuousTimes in tryondiffusion/modules/imagen.py):
    # training samples a continuous random t ~ Uniform(0,1), so `timesteps`
    # above never affects what the model learns - it only controls how many
    # discrete denoising steps the *sampling loop* takes. A checkpoint
    # trained with timesteps=1000 can be sampled with far fewer steps for a
    # large speedup, at some cost to fine detail. 50 is a reasonable
    # quality/speed default; push lower (e.g. 20-30) for faster iteration,
    # higher (e.g. 100-250) for a final higher-quality render.
    inference_timesteps: int = 50

    ema: bool = True
    max_grad_norm: float = 1.0
    num_workers: int = 16

    checkpoint_path: str = "./checkpoints"
    checkpoint_every: int = 1000
    max_checkpoints_keep: int = 5
    init_checkpoint_path: Optional[str] = None  # e.g. trained base model, when training the SR unet
    validate_every: int = 500

    project_name: str = "digital-tryon-v2"
    use_wandb: bool = False
    wandb_entity: Optional[str] = "aditya_1976-shri-ramdeobaba-college-of-engineering-and-m"
    wandb_run_name: Optional[str] = None  # omit to let wandb auto-generate one
    log_every: int = 10  # steps between wandb/console scalar logs
    wandb_sample_every: int = 0  # steps between logging generated sample images to wandb; 0 = disabled

    # ------------------------------------------------ hardware / precision
    # Defaults tuned for NVIDIA Hopper/Blackwell (incl. B200) data-center GPUs.
    mixed_precision: str = "bf16"  # "bf16" (recommended on B200/H100), "fp16", or "no"
    allow_tf32: bool = True
    compile_model: bool = True
    compile_mode: str = "default"  # "max-autotune" is faster once warmed up but far more
                                    # prone to CUDA-graphs/NVML issues on some cloud GPU
                                    # containers; trainer.py also explicitly disables inductor's
                                    # cudagraphs capture regardless of mode (see configure_hardware)
    seed: int = 42

    def __post_init__(self):
        self.use_sr_unet = self.unet_number == 2

    @property
    def image_size(self) -> Tuple[int, int]:
        return self.sr_image_size if self.unet_number == 2 else self.base_image_size


# Fields whose Optional[...] inner type argparse can't infer from a None
# default (the default alone doesn't tell us it should parse as int).
_EXPLICIT_TYPES = {
    "max_steps": int,
}


def add_config_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Registers one `--flag` per TryOnConfig field, using its default as the CLI default."""
    import typing

    hints = typing.get_type_hints(TryOnConfig)

    for f in fields(TryOnConfig):
        if f.name in ("use_sr_unet",):
            continue
        flag = f"--{f.name.replace('_', '-')}"
        hint = hints.get(f.name)

        if hint is bool:
            parser.add_argument(flag, dest=f.name, action=argparse.BooleanOptionalAction, default=f.default)
        elif f.name in ("base_image_size", "sr_image_size", "timesteps"):
            parser.add_argument(flag, dest=f.name, type=int, nargs=2, default=f.default)
        else:
            arg_type = _EXPLICIT_TYPES.get(f.name, type(f.default) if f.default is not None else str)
            parser.add_argument(flag, dest=f.name, type=arg_type, default=f.default)
    return parser


def config_from_args(args: argparse.Namespace) -> TryOnConfig:
    kwargs = {f.name: getattr(args, f.name) for f in fields(TryOnConfig) if f.name != "use_sr_unet" and hasattr(args, f.name)}
    if isinstance(kwargs.get("base_image_size"), list):
        kwargs["base_image_size"] = tuple(kwargs["base_image_size"])
    if isinstance(kwargs.get("sr_image_size"), list):
        kwargs["sr_image_size"] = tuple(kwargs["sr_image_size"])
    if isinstance(kwargs.get("timesteps"), list):
        kwargs["timesteps"] = tuple(kwargs["timesteps"])
    return TryOnConfig(**kwargs)
