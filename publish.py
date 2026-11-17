"""Publish the trained models and the in-browser demo to Hugging Face (both free).

1. Converts weights/*.pt to ONNX (export_onnx.py)
2. Uploads weights + ONNX + metrics to a model repo:   <user>/breast-ultrasound-models
3. Uploads web/index.html + examples to a static Space: <user>/breast-ultrasound-analyser

Needs a Hugging Face write token (HF_TOKEN env var, or `hf auth login`).
Usage: python publish.py
"""
import json
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

import export_onnx

ROOT = Path(__file__).parent
MODEL_NAME = "breast-ultrasound-models"
SPACE_NAME = "breast-ultrasound-analyser"
GITHUB = "https://github.com/Ammarhannun/breast-cancer-classification-segmentation"


def metrics_text():
    try:
        c = json.loads((ROOT / "weights/classifier_metrics.json").read_text())
        s = json.loads((ROOT / "weights/segmenter_metrics.json").read_text())
        mal = c["report"]["malignant"]["recall"]
        return (f"| Classifier (ResNet-50) | Test accuracy | {100 * c['test_acc']:.1f}% |\n"
                f"| Classifier | Malignant recall | {100 * mal:.1f}% |\n"
                f"| Segmenter (U-Net, ResNet-34) | Test IoU / Dice (lesion images) | "
                f"{s['test']['iou']:.3f} / {s['test']['dice']:.3f} |\n"
                f"| Segmenter | Normal images left blank | {100 * s['test']['normal_clean_rate']:.1f}% |\n")
    except (FileNotFoundError, KeyError):
        return ""


def main():
    api = HfApi()
    user = api.whoami()["name"]
    model_repo, space_repo = f"{user}/{MODEL_NAME}", f"{user}/{SPACE_NAME}"

    print("1/3 Converting to ONNX…")
    export_onnx.main()

    print(f"2/3 Uploading models to https://huggingface.co/{model_repo}")
    api.create_repo(model_repo, repo_type="model", exist_ok=True)
    card = f"""---
license: mit
tags: [medical-imaging, ultrasound, image-classification, image-segmentation, onnx]
---
# Breast ultrasound classification + lesion segmentation

Trained on the BUSI dataset (780 images). Code: {GITHUB}
Live demo (runs in your browser): https://huggingface.co/spaces/{space_repo}

| Model | Metric | Score |
|---|---|---|
{metrics_text()}
Files: `classifier.pt` / `segmenter.pt` (PyTorch state dicts, see `bcus/models.py` in the repo),
`classifier.onnx` / `segmenter.onnx` (input `1x3xHxW` ImageNet-normalised, H=W=224 / 256; output logits).
Classes: benign, malignant, normal.

**Research demo only. Not a medical device.**
"""
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "README.md").write_text(card)
        api.upload_file(path_or_fileobj=str(Path(tmp) / "README.md"), path_in_repo="README.md", repo_id=model_repo)
    api.upload_folder(folder_path=str(ROOT / "weights"), repo_id=model_repo,
                      allow_patterns=["*.pt", "*.onnx", "*.json"], commit_message="Upload trained models")

    print(f"3/3 Publishing static Space https://huggingface.co/spaces/{space_repo}")
    api.create_repo(space_repo, repo_type="space", space_sdk="static", exist_ok=True)
    space_card = f"""---
title: Breast Ultrasound Analyser
emoji: 🩺
colorFrom: pink
colorTo: indigo
sdk: static
app_file: index.html
pinned: false
short_description: Ultrasound classification + lesion outline, in your browser
models: [{model_repo}]
---
"""
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        (t / "README.md").write_text(space_card)
        html = (ROOT / "web/index.html").read_text().replace(
            'const MODEL_REPO = "AmmarHannun/breast-ultrasound-models";', f'const MODEL_REPO = "{model_repo}";')
        (t / "index.html").write_text(html)
        (t / "examples").mkdir()
        for p in (ROOT / "examples").glob("*.png"):
            (t / "examples" / p.name).write_bytes(p.read_bytes())
        api.upload_folder(folder_path=tmp, repo_id=space_repo, repo_type="space", commit_message="Publish demo")

    print(f"\nDone!\n  Demo:   https://huggingface.co/spaces/{space_repo}\n  Models: https://huggingface.co/{model_repo}")


if __name__ == "__main__":
    main()
