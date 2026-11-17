"""Upload the app + trained weights to a Hugging Face Space.

Before running: train both models (weights/ must exist) and sign in once with
    huggingface-cli login      (or: hf auth login)

Usage: python deploy_space.py <your-hf-username>/<space-name>
e.g.   python deploy_space.py ammarhannun/breast-ultrasound-analyser
"""
import sys
import tempfile
from pathlib import Path

import gradio
from huggingface_hub import HfApi

ROOT = Path(__file__).parent
REQUIRED = ["weights/classifier.pt", "weights/segmenter.pt"]

SPACE_CARD = f"""---
title: Breast Ultrasound Analyser
emoji: 🩺
colorFrom: pink
colorTo: indigo
sdk: gradio
sdk_version: {gradio.__version__}
app_file: app.py
pinned: false
short_description: ResNet-50 classification + U-Net segmentation of ultrasound
---

Breast ultrasound classification (benign / malignant / normal) and lesion segmentation.

Code and training: https://github.com/Ammarhannun/breast-cancer-classification-segmentation

**Research demo only. Not a medical device.**
"""


def main():
    if len(sys.argv) != 2 or "/" not in sys.argv[1]:
        sys.exit(__doc__)
    repo_id = sys.argv[1]

    missing = [p for p in REQUIRED if not (ROOT / p).exists()]
    if missing:
        sys.exit(f"Missing {missing}. Train the models first (see README).")

    api = HfApi()
    api.create_repo(repo_id, repo_type="space", space_sdk="gradio", exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        card = Path(tmp) / "README.md"
        card.write_text(SPACE_CARD)
        api.upload_file(path_or_fileobj=str(card), path_in_repo="README.md",
                        repo_id=repo_id, repo_type="space")

    api.upload_folder(
        folder_path=str(ROOT),
        repo_id=repo_id,
        repo_type="space",
        allow_patterns=["app.py", "requirements.txt", "bcus/*.py",
                        "weights/*.pt", "weights/*.json", "examples/*.png"],
        commit_message="Deploy app and weights",
    )
    print(f"Done. Your app will be live in a few minutes at https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    main()
