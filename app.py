"""
Gradio demo UI for TryOnDiffusion.

Run: python app.py
Requires a trained checkpoint at CHECKPOINT_PATH (see README.md ->
"Running the demo UI"). Only a person photo and a garment photo are needed
from the user — the clothing-agnostic image and pose keypoints are derived
automatically by TryOnPipeline (tryon_pipeline.py).
"""
import os

import gradio as gr

from config import TryOnConfig
from tryon_pipeline import TryOnPipeline

CHECKPOINT_PATH = os.environ.get("TRYON_CHECKPOINT", "./model/checkpoint.11900/checkpoint.11900.pt")

PIPELINE = None


def load_pipeline():
    global PIPELINE
    if PIPELINE is None:
        PIPELINE = TryOnPipeline(checkpoint_path=CHECKPOINT_PATH, config=TryOnConfig())
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

    cond_scale = gr.Slider(minimum=1.0, maximum=10.0, value=3.0, step=0.1, label="CFG Scale")

    run_button = gr.Button("Generate Try-On")
    output_image = gr.Image(label="Generated Image")

    run_button.click(fn=run_tryon, inputs=[person_image, garment_image, cond_scale], outputs=output_image)


if __name__ == "__main__":
    demo.launch()
