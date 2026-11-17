"""Image preprocessing shared by training and the web app, so both see identical inputs."""
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .models import CLS_IMAGE_SIZE, SEG_IMAGE_SIZE

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

cls_eval_transform = transforms.Compose([
    transforms.Resize((CLS_IMAGE_SIZE, CLS_IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

cls_train_transform = transforms.Compose([
    transforms.Resize((CLS_IMAGE_SIZE, CLS_IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def seg_normalize(image_uint8):
    """HxWx3 uint8 array -> 3xHxW float32 tensor, ImageNet-normalised.

    Same maths as segmentation_models_pytorch's resnet34/imagenet preprocessing_fn.
    """
    x = image_uint8.astype(np.float32) / 255.0
    x = (x - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(IMAGENET_STD, dtype=np.float32)
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


def seg_eval_input(pil_image):
    """PIL image -> 1x3x256x256 tensor ready for the U-Net."""
    img = pil_image.convert("RGB").resize((SEG_IMAGE_SIZE, SEG_IMAGE_SIZE), Image.BILINEAR)
    return seg_normalize(np.array(img)).unsqueeze(0)
