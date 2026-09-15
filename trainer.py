"""
Unified production training script for TryOnDiffusion.

Trains either the base U-Net (--unet-number 1, low-res structure model) or
the SR U-Net (--unet-number 2, high-res refiner model, requires a trained
base checkpoint via --init-checkpoint-path). Replaces the old
trainer.py / trainer01.py / trainer02.py split — pass --unet-number instead
of maintaining three near-duplicate scripts.

Data comes from the Hugging Face Hub by default (no manual dataset download
or CSV mapping needed); pass --data-source local to use a folder on disk
via RealTryonDataset instead. Only a person image and a garment image are
required either way (see tryondiffusion/preprocessing.py for how the
clothing-agnostic image and pose keypoints are derived automatically).

Examples
--------
Train the base U-Net at 256x256 on the default HF dataset, single GPU:
    python trainer.py --unet-number 1

Train the SR U-Net at 512x512, resuming garment/pose weights from a
trained base checkpoint, on a multi-GPU node (e.g. 8x B200):
    accelerate launch trainer.py --unet-number 2 \\
        --init-checkpoint-path ./checkpoints/checkpoint.50000.pt

Train from a local folder instead of the Hub:
    python trainer.py --data-source local --local-root ./data_small

See `README.md` -> "Training on NVIDIA B200 / Blackwell" for multi-GPU
launch and environment-variable recommendations.
"""
import argparse

import torch
from torch.utils.data import DataLoader

from config import TryOnConfig, add_config_arguments, config_from_args
from tryondiffusion import TryOnImagen, TryOnImagenTrainer, get_unet_by_name


def tryondiffusion_collate_fn(batch):
    return {
        "person_images": torch.stack([item["person_images"] for item in batch]),
        "ca_images": torch.stack([item["ca_images"] for item in batch]),
        "garment_images": torch.stack([item["garment_images"] for item in batch]),
        "person_poses": torch.stack([item["person_poses"] for item in batch]),
        "garment_poses": torch.stack([item["garment_poses"] for item in batch]),
    }


def configure_hardware(config: TryOnConfig):
    """Enables TF32 matmuls and high-precision fp32 matmul mode; both are
    safe accuracy/throughput wins on Ampere+ and especially Hopper/Blackwell
    (B200) tensor cores, and are otherwise off by default in PyTorch."""
    if config.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")


def build_dataset(config: TryOnConfig):
    image_size = config.image_size

    if config.data_source == "huggingface":
        from tryondiffusion.hf_dataset import build_hf_tryon_dataset

        dataset = build_hf_tryon_dataset(
            repo_id=config.hf_dataset_id,
            split=config.hf_split,
            image_size=image_size,
            max_keypoints=config.max_keypoints,
            person_image_column=config.hf_person_image_column,
            cloth_image_column=config.hf_cloth_image_column,
            pose_column=config.hf_pose_column,
            agnostic_image_column=config.hf_agnostic_image_column,
            streaming=config.hf_streaming,
        )
    elif config.data_source == "local":
        from RealTryonDataset import RealTryonDataset

        dataset = RealTryonDataset(
            root=config.local_root,
            image_size=image_size,
            mapping_file=config.local_mapping_file,
            max_keypoints=config.max_keypoints,
        )
    else:
        raise ValueError(f"Unknown data_source: {config.data_source!r} (expected 'huggingface' or 'local')")

    return dataset


def build_dataloaders(config: TryOnConfig, dataset):
    common_kwargs = dict(
        batch_size=config.batch_size,
        collate_fn=tryondiffusion_collate_fn,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    if config.hf_streaming and config.data_source == "huggingface":
        # IterableDataset: no shuffle=True (shuffling already happened via
        # .shuffle(buffer_size=...) on the underlying HF dataset) and no
        # train/valid split (validate on a small held-out streaming subset
        # of the same source instead, if you need periodic eval).
        train_dl = DataLoader(dataset, **common_kwargs)
        return train_dl, None

    train_size = int(0.975 * len(dataset))
    valid_size = len(dataset) - train_size
    train_ds, valid_ds = torch.utils.data.random_split(
        dataset, [train_size, valid_size], generator=torch.Generator().manual_seed(config.seed)
    )

    train_dl = DataLoader(train_ds, shuffle=True, **common_kwargs)
    valid_dl = DataLoader(valid_ds, shuffle=False, **common_kwargs)
    return train_dl, valid_dl


def _maybe_compile(unet, config: TryOnConfig):
    # Compile the bound `forward` method in place (rather than wrapping the
    # whole module with torch.compile) so `unet` stays a real ParallelUNet
    # instance: the library relies on `isinstance(unet, ParallelUNet)` and
    # on methods like `cast_model_parameters`/`forward_with_cond_scale`
    # elsewhere, which a torch.compile module wrapper would break.
    if config.compile_model and torch.cuda.is_available():
        unet.forward = torch.compile(unet.forward, mode=config.compile_mode)
    return unet


def build_imagen(config: TryOnConfig):
    base_unet = get_unet_by_name("base", image_size=config.base_image_size, max_keypoints_len=config.max_keypoints)
    base_unet = _maybe_compile(base_unet, config)

    if not config.use_sr_unet:
        return TryOnImagen(
            unets=(base_unet,),
            image_sizes=(config.base_image_size,),
            timesteps=config.timesteps[0],
            pred_objectives="noise",
            noise_schedules="cosine",
            auto_normalize_img=True,
            dynamic_thresholding=True,
        )

    sr_unet = get_unet_by_name("sr", image_size=config.sr_image_size, max_keypoints_len=config.max_keypoints)
    sr_unet = _maybe_compile(sr_unet, config)

    return TryOnImagen(
        unets=(base_unet, sr_unet),
        image_sizes=(config.base_image_size, config.sr_image_size),
        timesteps=config.timesteps,
        auto_normalize_img=True,
        dynamic_thresholding=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_arguments(parser)
    args = parser.parse_args()
    config = config_from_args(args)

    configure_hardware(config)
    torch.manual_seed(config.seed)

    print(f"Config: {config}")

    print("Building dataset...")
    dataset = build_dataset(config)
    train_dataloader, validation_dataloader = build_dataloaders(config, dataset)

    print("Instantiating U-Net(s) and Imagen...")
    imagen = build_imagen(config)

    print("Instantiating trainer...")
    wandb_kwargs = {}
    if config.use_wandb:
        # log_with="wandb" tells accelerate to actually spin up a wandb run;
        # without it, project_name alone would try (and by default fail
        # silently into a no-op) to initialize a tracker.
        wandb_kwargs["accelerate_log_with"] = "wandb"
        if config.wandb_entity:
            wandb_kwargs["wandb_entity"] = config.wandb_entity
        if config.wandb_run_name:
            wandb_kwargs["wandb_name"] = config.wandb_run_name

    trainer = TryOnImagenTrainer(
        imagen=imagen,
        lr=config.learning_rate,
        max_grad_norm=config.max_grad_norm,
        accelerate_gradient_accumulation_steps=config.gradient_accumulation_steps,
        accelerate_mixed_precision=config.mixed_precision,
        use_ema=config.ema,
        max_checkpoints_keep=config.max_checkpoints_keep,
        checkpoint_every=config.checkpoint_every,
        checkpoint_path=config.checkpoint_path,
        init_checkpoint_path=config.init_checkpoint_path,
        only_train_unet_number=config.unet_number,
        project_name=config.project_name if config.use_wandb else None,
        verbose=True,
        **wandb_kwargs,
    )

    trainer.add_train_dataloader(train_dataloader)
    if validation_dataloader is not None:
        trainer.add_valid_dataloader(validation_dataloader)

    if config.hf_streaming and config.data_source == "huggingface":
        if not config.max_steps:
            raise ValueError("--max-steps is required when --hf-streaming is set (dataset length is unknown)")
        total_steps = config.max_steps
    else:
        steps_per_epoch = len(train_dataloader) // config.gradient_accumulation_steps
        total_steps = config.max_steps or steps_per_epoch * config.epochs

    print(f"unet_number={config.unet_number}  image_size={config.image_size}  "
          f"batch_size={config.batch_size}  grad_accum={config.gradient_accumulation_steps}  "
          f"effective_batch_size={config.batch_size * config.gradient_accumulation_steps}  "
          f"total_steps={total_steps}  mixed_precision={config.mixed_precision}")

    if config.use_wandb:
        print(f"Weights & Biases logging enabled  project={config.project_name}"
              f"{'  entity=' + config.wandb_entity if config.wandb_entity else ''}"
              f"{'  run=' + config.wandb_run_name if config.wandb_run_name else ''}")
        print("Run `wandb login` first if you haven't already (or set the WANDB_API_KEY env var).")

    print("Starting training loop...")
    step = trainer.num_steps_taken(unet_number=config.unet_number)
    while step < total_steps:
        loss = trainer.train_step(unet_number=config.unet_number)
        step += 1

        if step % config.log_every == 0:
            trainer.log_metrics(
                {"train/loss": float(loss.detach()), "train/lr": trainer.get_lr(config.unet_number)}, step=step
            )

        if step % 50 == 0:
            print(f"step {step}/{total_steps}  loss={loss:.4f}")

        if validation_dataloader is not None and trainer.is_main and step % config.validate_every == 0:
            valid_loss = trainer.valid_step(unet_number=config.unet_number)
            print(f"step {step}  valid_loss={valid_loss:.4f}")
            trainer.log_metrics({"valid/loss": float(valid_loss)}, step=step)

            if config.use_wandb and config.wandb_sample_every and step % config.wandb_sample_every == 0:
                log_sample_images(trainer, validation_dataloader, config, step)

    print("Training finished.")
    trainer.save_to_checkpoint_folder()


def log_sample_images(trainer, validation_dataloader, config: TryOnConfig, step: int, num_samples: int = 4):
    """Generates a few try-on images from the current (EMA) weights and logs
    them to the active wandb run, so you can watch generation quality evolve
    alongside the loss curves instead of only seeing numbers."""
    import wandb

    if not trainer.is_main:
        return

    batch = next(iter(validation_dataloader))
    batch = {k: v[:num_samples] for k, v in batch.items()}
    _ = batch.pop("person_images")

    images = trainer.sample(
        batch_size=batch["ca_images"].shape[0],
        **batch,
        cond_scale=config.cond_scale,
        return_pil_images=True,
        use_tqdm=False,
    )
    trainer.log_metrics({"samples": [wandb.Image(img, caption=f"step {step}") for img in images]}, step=step)


if __name__ == "__main__":
    main()
