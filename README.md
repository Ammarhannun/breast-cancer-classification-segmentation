# Breast Ultrasound Classification & Segmentation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ammarhannun/breast-cancer-classification-segmentation/blob/deploy-app/train_and_deploy_colab.ipynb)

**Live demo:** _coming soon (Hugging Face Spaces)_

Upload a breast ultrasound image and the app:
1. **Classifies** it as benign, malignant or normal (ResNet-50, transfer learning)
2. **Outlines** the suspected lesion (U-Net with a ResNet-34 encoder)

> ⚠️ Research and portfolio project. **Not a medical device** and not for diagnosis.

---

## Results

| Task | Model | Metric | Score |
|---|---|---|---|
| Classification | ResNet-50 (layer4 + new head fine-tuned) | Test accuracy | _TBD after retrain_ |
| Segmentation | U-Net, ResNet-34 encoder | Mean IoU / Dice | _TBD after retrain_ |

The original notebook run reached **83.6% test accuracy** and **0.64 IoU**.

## Dataset

[`gymprathap/Breast-Cancer-Ultrasound-Images-Dataset`](https://huggingface.co/datasets/gymprathap/Breast-Cancer-Ultrasound-Images-Dataset) on Hugging Face: This is the BUSI dataset: 780 ultrasound images (437 benign / 210 malignant / 133 normal), each with a hand-drawn tumour mask. Images are paired with their masks by filename. Where an image has several lesion masks, they are merged. Both models use the **same stratified 64/16/20 train/val/test split**, so neither has seen the test images.

## Project structure

```
app.py                      Gradio web app (what the Space runs)
bcus/models.py              ResNet-50 classifier + U-Net segmenter definitions
bcus/preprocess.py          Shared preprocessing (training and app use the same code)
bcus/data.py                Dataset loading, mask filtering, image/mask pairing
train_classifier.py         Trains the classifier -> weights/classifier.pt
train_segmenter.py          Trains the segmenter  -> weights/segmenter.pt
deploy_space.py             Uploads app + weights to a Hugging Face Space
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

**Deploy to Hugging Face Spaces:**
```bash
huggingface-cli login
python deploy_space.py <hf-username>/breast-ultrasound-analyser
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
