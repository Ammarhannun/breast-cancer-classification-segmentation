"""Gradio web app: upload a breast ultrasound -> class + confidence + tumour outline.

Run locally: python app.py
"""
import os
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from PIL import Image

from bcus.models import CLASS_NAMES, get_classifier, get_segmenter
from bcus.preprocess import cls_eval_transform, seg_eval_input

ROOT = Path(__file__).parent
WEIGHTS = ROOT / "weights"
EXAMPLES = ROOT / "examples"
torch.set_num_threads(max(1, os.cpu_count() or 1))


def load(model, path):
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.eval()


classifier = load(get_classifier(pretrained=False), WEIGHTS / "classifier.pt")
segmenter = load(get_segmenter(pretrained=False), WEIGHTS / "segmenter.pt")


def overlay(image, mask, color=(230, 40, 40), alpha=0.45):
    """Tint the tumour pixels and draw a solid outline around them."""
    base = np.array(image).astype(np.float32)
    out = base.copy()
    out[mask] = (1 - alpha) * base[mask] + alpha * np.array(color, dtype=np.float32)
    # Outline = mask pixels with at least one non-mask neighbour.
    padded = np.pad(mask, 1)
    inner = (padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:])
    out[mask & ~inner] = color
    return Image.fromarray(out.astype(np.uint8))


@torch.inference_mode()
def analyse(image):
    if image is None:
        raise gr.Error("Please upload an ultrasound image first.")
    image = image.convert("RGB")

    probs = torch.softmax(classifier(cls_eval_transform(image).unsqueeze(0)), dim=1)[0]
    confidences = {name.capitalize(): float(p) for name, p in zip(CLASS_NAMES, probs)}
    top = CLASS_NAMES[int(probs.argmax())]

    logits = segmenter(seg_eval_input(image))[0, 0]
    mask_small = (torch.sigmoid(logits) > 0.5).numpy().astype(np.uint8) * 255
    mask = np.array(Image.fromarray(mask_small).resize(image.size, Image.NEAREST)) > 127
    area = 100 * mask.mean()

    if top == "normal" or area < 0.5:
        note = "No clear lesion region found." if area < 0.5 else (
            f"Classifier says normal, but the segmenter marked {area:.1f}% of the image. Treat the outline with caution.")
    else:
        note = f"Highlighted region covers about {area:.1f}% of the image."
    return confidences, overlay(image, mask), note


examples = sorted(str(p) for p in EXAMPLES.glob("*.png")) if EXAMPLES.exists() else []

with gr.Blocks(title="Breast Ultrasound Analyser") as demo:
    gr.Markdown(
        "# Breast Ultrasound Analyser\n"
        "Upload a breast ultrasound image. A **ResNet-50** classifies it as benign, malignant or normal, "
        "and a **U-Net** outlines the suspected lesion.\n\n"
        "> ⚠️ Research demo only. **Not a medical device** and not for diagnosis."
    )
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="pil", label="Ultrasound image", height=360)
            btn = gr.Button("Analyse", variant="primary")
            if examples:
                gr.Examples(examples=examples, inputs=inp, label="Try a held-out test image")
        with gr.Column():
            label = gr.Label(num_top_classes=3, label="Prediction")
            seg = gr.Image(type="pil", label="Suspected lesion (red)", height=360)
            note = gr.Markdown()
    btn.click(analyse, inputs=inp, outputs=[label, seg, note])
    inp.upload(analyse, inputs=inp, outputs=[label, seg, note])

if __name__ == "__main__":
    demo.launch()
