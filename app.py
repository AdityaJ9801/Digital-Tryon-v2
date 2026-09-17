"""
Gradio demo UI for TryOnDiffusion.

Run: python app.py
Requires a trained checkpoint at CHECKPOINT_PATH (see README.md ->
"Running the demo UI"). Only a person photo and a garment photo are needed
from the user — the clothing-agnostic image and pose keypoints are derived
automatically by TryOnPipeline (tryon_pipeline.py).

TRYON_CHECKPOINT, if set, points at a specific checkpoint file OR directory
(auto-picks the highest-step checkpoint.<N>.pt inside it). If NOT set (the
default), this scans every ./checkpoints* directory (every training
lineage you have - checkpoints_base, checkpoints_kaggle_hf_zalando,
checkpoints_sr_stage1, etc.) and uses whichever single checkpoint file was
modified most recently - "the latest model" across all your training runs,
not just one folder.

Auto-detection also guesses unet_number from the picked checkpoint's
directory name (a folder name containing "sr" -> assumes it's an SR-stage
checkpoint) - this is a naming-convention heuristic, not something read
from the checkpoint file itself, so it can be wrong if you name folders
differently. Set TRYON_USE_SR_UNET explicitly (0 or 1) to override it.
The picked path and the unet_number it's using are both printed at
startup - check that output matches what you expect.

The resolution env vars below MUST match whatever --base-image-size /
--sr-image-size the checkpoint was actually trained with, or the checkpoint
will silently partial-load (mismatched-shape layers get skipped) and
generation will look broken. Example, for a checkpoint trained with
`trainer.py --base-image-size 128 128`:
    TRYON_BASE_SIZE=128,128 python app.py

On a remote server (SSH, no local browser), set TRYON_SHARE=1 to get a
public gradio.live URL you can open from any browser, instead of needing
SSH port-forwarding:
    TRYON_BASE_SIZE=128,128 TRYON_SHARE=1 python app.py

TRYON_INFERENCE_STEPS (default 50) controls generation speed vs detail -
this model uses continuous-time diffusion, so sampling step count is
independent of training and safe to lower for faster generation (see
TryOnConfig.inference_timesteps for why). Try 20-30 for faster iteration
while testing, or 100-250 for a final higher-detail render.
"""
import os

import gradio as gr

from config import TryOnConfig
from tryon_pipeline import TryOnPipeline, find_globally_latest_checkpoint, find_latest_checkpoint

BASE_SIZE = tuple(int(x) for x in os.environ.get("TRYON_BASE_SIZE", "256,256").split(","))
SR_SIZE = tuple(int(x) for x in os.environ.get("TRYON_SR_SIZE", "512,512").split(","))
SHARE = os.environ.get("TRYON_SHARE", "0") == "1"
INFERENCE_STEPS = int(os.environ.get("TRYON_INFERENCE_STEPS", "50"))

PIPELINE = None


def resolve_checkpoint_and_unet_number():
    env_checkpoint = os.environ.get("TRYON_CHECKPOINT")
    checkpoint_path = find_latest_checkpoint(env_checkpoint) if env_checkpoint else find_globally_latest_checkpoint()

    env_use_sr = os.environ.get("TRYON_USE_SR_UNET")
    if env_use_sr is not None:
        use_sr_unet = env_use_sr == "1"
    else:
        checkpoint_dir_name = os.path.basename(os.path.dirname(os.path.abspath(checkpoint_path))).lower()
        use_sr_unet = "sr" in checkpoint_dir_name.split("_")

    print(f"Using checkpoint: {checkpoint_path}  (unet_number={2 if use_sr_unet else 1}, "
          f"{'auto-detected from folder name' if env_use_sr is None else 'set via TRYON_USE_SR_UNET'})")
    return checkpoint_path, use_sr_unet


def load_pipeline():
    global PIPELINE
    if PIPELINE is None:
        checkpoint_path, use_sr_unet = resolve_checkpoint_and_unet_number()
        config = TryOnConfig(
            unet_number=2 if use_sr_unet else 1,
            base_image_size=BASE_SIZE,
            sr_image_size=SR_SIZE,
            inference_timesteps=INFERENCE_STEPS,
        )
        PIPELINE = TryOnPipeline(checkpoint_path=checkpoint_path, config=config)
    return PIPELINE


def run_tryon(person_image, garment_image, cond_scale):
    if person_image is None or garment_image is None:
        raise gr.Error("Please provide both a person image and a garment image.")

    pipeline = load_pipeline()
    result, debug = pipeline.generate(person_image, garment_image, cond_scale=cond_scale, return_debug=True)
    return result, debug["ca_image"]


with gr.Blocks(title="TryOn Diffusion") as demo:
    gr.Markdown("## 👕 Virtual Try-On (TryOnDiffusion)")
    gr.Markdown("Upload a person photo and a garment photo. Pose and clothing-agnostic masking are computed automatically.")

    with gr.Row():
        person_image = gr.Image(label="Person Photo", type="filepath")
        garment_image = gr.Image(label="Garment Photo", type="filepath")

    if os.path.exists("test_person.jpg") and os.path.exists("test_garment.jpg"):
        gr.Examples(
            examples=[["test_person.jpg", "test_garment.jpg"]],
            inputs=[person_image, garment_image],
            label="Example (click to load)",
        )

    cond_scale = gr.Slider(minimum=1.0, maximum=10.0, value=3.0, step=0.1, label="CFG Scale")

    run_button = gr.Button("Generate Try-On")

    with gr.Row():
        output_image = gr.Image(label="Generated Image")
        debug_ca_image = gr.Image(
            label="Debug: auto-generated clothing-agnostic mask (what the model actually sees as 'person')",
        )
    gr.Markdown(
        "If the generated garment looks wrong or the original clothing seems to still be there, check the "
        "debug image above: the gray-masked region is all of the person's original clothing the model can "
        "no longer see. If the mask isn't fully covering the worn garment, the model can still see the "
        "original clothing and will tend to reproduce it instead of the new one."
    )

    run_button.click(
        fn=run_tryon, inputs=[person_image, garment_image, cond_scale], outputs=[output_image, debug_ca_image]
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=SHARE)
