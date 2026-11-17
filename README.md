# Breast Ultrasound Classification & Segmentation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ammarhannun/breast-cancer-classification-segmentation/blob/deploy-app/train_and_deploy_colab.ipynb)

**Live demo (runs in your browser):** https://huggingface.co/spaces/AmmarHannun/breast-ultrasound-analyser  
**Trained models:** https://huggingface.co/AmmarHannun/breast-ultrasound-models

Upload a breast ultrasound image and the app:
1. **Classifies** it as benign, malignant or normal (ResNet-50, transfer learning)
2. **Outlines** the suspected lesion (U-Net with a ResNet-34 encoder)

> ⚠️ Research and portfolio project. **Not a medical device** and not for diagnosis.

---

## Results

All scores are on the held-out **test set** (156 images, never used for training or for choosing the best epoch).

| Task | Model | Metric | Score |
|---|---|---|---|
| Classification | ResNet-50 (layer4 + new head fine-tuned) | Accuracy | **84.6%** |
| | | Macro F1 | 0.835 |
| | | Malignant recall (sensitivity) | 81.0% |
| Segmentation | U-Net, ResNet-34 encoder | IoU / Dice (lesion images) | **0.673 / 0.759** |
| | | Normal scans correctly left blank | 88.9% |

Most classification errors are benign images predicted as normal (11 of 87) and malignant images predicted as benign (7 of 42). In both models' scores, the run-to-run variation on a GPU is about ±1–2 points.

## Dataset

[`gymprathap/Breast-Cancer-Ultrasound-Images-Dataset`](https://huggingface.co/datasets/gymprathap/Breast-Cancer-Ultrasound-Images-Dataset) on Hugging Face: This is the BUSI dataset: 780 ultrasound images (437 benign / 210 malignant / 133 normal), each with a hand-drawn tumour mask. Images are paired with their masks by filename. Where an image has several lesion masks, they are merged. Both models use the **same stratified 64/16/20 train/val/test split**, so neither has seen the test images.

## Project structure

```
web/index.html              In-browser demo (ONNX Runtime Web): what the Space serves
app.py                      Optional Gradio app for running locally with PyTorch
bcus/models.py              ResNet-50 classifier + U-Net segmenter definitions
bcus/preprocess.py          Shared preprocessing (training and app use the same code)
bcus/data.py                Dataset loading, mask filtering, image/mask pairing
train_classifier.py         Trains the classifier -> weights/classifier.pt
train_segmenter.py          Trains the segmenter  -> weights/segmenter.pt
export_onnx.py              Converts the weights to ONNX for the browser
publish.py                  Uploads models + demo to Hugging Face (free model repo + static Space)
train_and_deploy_colab.ipynb  One-click train + deploy on a free Colab GPU
breastCancerClassificationAndSegmentation.ipynb  Original exploratory notebook
```

## Run it yourself

**Train** (GPU recommended; on Colab use the badge above):
```bash
pip install -r requirements-train.txt
python train_classifier.py
python train_segmenter.py
```

**Run the app locally:**
```bash
python app.py
```

**Publish to Hugging Face** (free; the demo is a static page, so the models run in the visitor's browser):
```bash
hf auth login
python publish.py
```

## Method

- **Classification:** 224×224 inputs, ImageNet normalisation, flip/rotation/colour-jitter augmentation. Only `layer4` and a new head (512 → ReLU → Dropout 0.4 → 3) are trained. Uses Adam (lr 1e-4) and cross-entropy. The best validation epoch is kept. Splits are stratified 64/16/20.
- **Segmentation:** 256×256 inputs, flip and brightness/contrast augmentation, BCE-with-logits loss, Adam (lr 1e-4). The best epoch is chosen on validation IoU. IoU and Dice are averaged per image, over lesion images only. Normal images (empty masks) are scored separately, by how often the model correctly predicts no lesion.

## Future work
- Combine Dice and BCE loss to sharpen the segmentation
- Grad-CAM heatmaps to show what the classifier is looking at
- Try EfficientNet / ConvNeXt backbones

## References
- He et al. (2016). *Deep Residual Learning for Image Recognition.* CVPR.
- Ronneberger et al. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI.
