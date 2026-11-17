"""Model definitions shared by training and the web app."""
import torch.nn as nn
import segmentation_models_pytorch as smp
from torchvision import models

# Label order of the Hugging Face dataset (ClassLabel names, in index order).
CLASS_NAMES = ["benign", "malignant", "normal"]

CLS_IMAGE_SIZE = 224
SEG_IMAGE_SIZE = 256
SEG_ENCODER = "resnet34"


def get_classifier(num_classes=len(CLASS_NAMES), pretrained=True):
    """ResNet-50 with only layer4 and a new head trainable."""
    weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet50(weights=weights)

    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = True

    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, num_classes),
    )
    return model


def get_segmenter(pretrained=True):
    """U-Net with a ResNet-34 encoder, one output channel (tumour mask logits)."""
    return smp.Unet(
        encoder_name=SEG_ENCODER,
        encoder_weights="imagenet" if pretrained else None,
        in_channels=3,
        classes=1,
    )
