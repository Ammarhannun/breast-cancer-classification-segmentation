"""Dataset loading for classification and segmentation (training only).

The Hugging Face dataset is the BUSI dataset as one zip:
    Dataset_BUSI_with_GT/<class>/<class> (N).png          ultrasound image
    Dataset_BUSI_with_GT/<class>/<class> (N)_mask.png     tumour mask
    Dataset_BUSI_with_GT/<class>/<class> (N)_mask_1.png   extra mask (some images have 2+ lesions)
We read the zip directly and pair images with masks by filename.
"""
import io
import random
import re
import zipfile
from collections import defaultdict

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from .models import CLASS_NAMES, SEG_IMAGE_SIZE
from .preprocess import seg_normalize

HF_DATASET = "gymprathap/Breast-Cancer-Ultrasound-Images-Dataset"
HF_FILE = "Breast-Cancer-Ultrasound-Images-Dataset.zip"
SEED = 42
_MASK_RE = re.compile(r"^(?P<stem>.+?)_mask(?:_\d+)?\.png$")


def load_records():
    """Return a list of {"name", "label", "image" (PIL RGB), "mask" (PIL L, 0/255)}."""
    path = hf_hub_download(HF_DATASET, HF_FILE, repo_type="dataset")
    images, masks = {}, defaultdict(list)
    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            if not name.lower().endswith(".png"):
                continue
            folder, fname = name.split("/")[-2], name.split("/")[-1]
            if folder not in CLASS_NAMES:
                continue
            img = Image.open(io.BytesIO(zf.read(name)))
            img.load()  # read now, before the zip closes
            m = _MASK_RE.match(fname)
            if m:
                masks[(folder, m["stem"])].append(img)
            else:
                images[(folder, fname[:-4])] = img

    records = []
    for (folder, stem), img in sorted(images.items()):
        img = img.convert("RGB")
        merged = np.zeros((img.height, img.width), dtype=bool)
        for m in masks.get((folder, stem), []):  # union of all lesion masks
            if m.size == img.size:
                merged |= np.array(m.convert("L")) > 127
        records.append({"name": stem, "label": CLASS_NAMES.index(folder), "image": img,
                        "mask": Image.fromarray(merged.astype(np.uint8) * 255)})
    return records


def split_records(records):
    """Stratified 64/16/20 train/val/test. Both models use the same split."""
    labels = [r["label"] for r in records]
    train_val, test = train_test_split(records, test_size=0.2, random_state=SEED, stratify=labels)
    tv_labels = [r["label"] for r in train_val]
    train, val = train_test_split(train_val, test_size=0.2, random_state=SEED, stratify=tv_labels)
    return train, val, test


class ClassificationDataset(Dataset):
    def __init__(self, records, transform):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        return self.transform(r["image"]), r["label"]


class SegmentationDataset(Dataset):
    def __init__(self, records, augment=False):
        self.records = records
        self.augment = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        size = (SEG_IMAGE_SIZE, SEG_IMAGE_SIZE)
        image = np.array(r["image"].resize(size, Image.BILINEAR))
        mask = np.array(r["mask"].resize(size, Image.NEAREST)) > 127

        if self.augment:
            if random.random() < 0.5:
                image, mask = image[:, ::-1], mask[:, ::-1]
            if random.random() < 0.2:
                alpha = random.uniform(0.8, 1.2)  # contrast
                beta = random.uniform(-25, 25)    # brightness
                image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        x = seg_normalize(np.ascontiguousarray(image))
        y = torch.from_numpy(np.ascontiguousarray(mask)).float().unsqueeze(0)
        return x, y, r["label"]
