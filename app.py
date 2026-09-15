"""
Gradio demo UI for TryOnDiffusion.

Run: python app.py
Requires a trained checkpoint at CHECKPOINT_PATH (see README.md ->
"Running the demo UI"). Only a person photo and a garment photo are needed
from the user — the clothing-agnostic image and pose keypoints are derived
automatically by TryOnPipeline (tryon_pipeline.py).

The resolution env vars below MUST match whatever --base-image-size /
--sr-image-size the checkpoint was actually trained with, or the checkpoint
will silently partial-load (mismatched-shape layers get skipped) and
generation will look broken. Example, for a checkpoint trained with
`trainer.py --base-image-size 128 128`:
    TRYON_CHECKPOINT=./checkpoints_base/checkpoint.6200.pt \
    TRYON_BASE_SIZE=128,128 python app.py

On a remote server (SSH, no local browser), set TRYON_SHARE=1 to get a
public gradio.live URL you can open from any browser, instead of needing
SSH port-forwarding:
    TRYON_CHECKPOINT=./checkpoints_base/checkpoint.6200.pt \
    TRYON_BASE_SIZE=128,128 TRYON_SHARE=1 python app.py
"""
import os

import gradio as gr

from config import TryOnConfig
from tryon_pipeline import TryOnPipeline

CHECKPOINT_PATH = os.environ.get("TRYON_CHECKPOINT", "./model/checkpoint.11900/checkpoint.11900.pt")
BASE_SIZE = tuple(int(x) for x in os.environ.get("TRYON_BASE_SIZE", "256,256").split(","))
SR_SIZE = tuple(int(x) for x in os.environ.get("TRYON_SR_SIZE", "512,512").split(","))
USE_SR_UNET = os.environ.get("TRYON_USE_SR_UNET", "0") == "1"
SHARE = os.environ.get("TRYON_SHARE", "0") == "1"

PIPELINE = None


def load_pipeline():
    global PIPELINE
    if PIPELINE is None:
        config = TryOnConfig(
            unet_number=2 if USE_SR_UNET else 1,
            base_image_size=BASE_SIZE,
            sr_image_size=SR_SIZE,
        )
        PIPELINE = TryOnPipeline(checkpoint_path=CHECKPOINT_PATH, config=config)
    return PIPELINE


def run_tryon(person_image, garment_image, cond_scale):
    if person_image is None or garment_image is None:
        raise gr.Error("Please provide both a person image and a garment image.")

    pipeline = load_pipeline()
    return pipeline.generate(person_image, garment_image, cond_scale=cond_scale)


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
    output_image = gr.Image(label="Generated Image")

    run_button.click(fn=run_tryon, inputs=[person_image, garment_image, cond_scale], outputs=output_image)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=SHARE)
