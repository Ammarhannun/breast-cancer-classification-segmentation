"""Train the U-Net (ResNet-34 encoder) tumour segmenter and save weights/segmenter.pt.

Usage: python train_segmenter.py [--epochs 20]
"""
import argparse
import json
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from bcus.data import SegmentationDataset, load_records, split_records
from bcus.models import CLASS_NAMES, get_segmenter
from train_classifier import pick_device

NORMAL = CLASS_NAMES.index("normal")


def batch_iou_dice(logits, target):
    """Per-image IoU and Dice, so small tumours count as much as big ones."""
    pred = torch.sigmoid(logits) > 0.5
    target = target > 0.5
    inter = (pred & target).flatten(1).sum(1).float()
    union = (pred | target).flatten(1).sum(1).float()
    total = pred.flatten(1).sum(1).float() + target.flatten(1).sum(1).float()
    iou = (inter + 1e-6) / (union + 1e-6)
    dice = (2 * inter + 1e-6) / (total + 1e-6)
    return iou.tolist(), dice.tolist()


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    """Returns loss, plus IoU/Dice on lesion images only (benign + malignant).

    Normal images have empty masks, so they're scored separately as
    'clean rate' = share of normal images where the model predicts (almost) nothing.
    """
    model.eval()
    loss, ious, dices, clean = 0.0, [], [], []
    for x, y, labels in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss += loss_fn(logits, y).item()
        iou, dice = batch_iou_dice(logits, y)
        pred_area = (torch.sigmoid(logits) > 0.5).flatten(1).float().mean(1).tolist()
        for i, lab in enumerate(labels.tolist()):
            if lab == NORMAL:
                clean.append(pred_area[i] < 0.005)
            else:
                ious.append(iou[i])
                dices.append(dice[i])
    return {"loss": loss / len(loader),
            "iou": sum(ious) / len(ious),
            "dice": sum(dices) / len(dices),
            "normal_clean_rate": sum(clean) / max(1, len(clean))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", default="weights/segmenter.pt")
    args = ap.parse_args()

    torch.manual_seed(42)
    device = pick_device()
    print(f"Device: {device}")

    train_ds, val_ds, test_ds = split_records(load_records())
    print(f"Train {len(train_ds)} | Val {len(val_ds)} | Test {len(test_ds)}")
    workers = 2 if device.type == "cuda" else 0
    train_loader = DataLoader(SegmentationDataset(train_ds, augment=True),
                              batch_size=args.batch_size, shuffle=True, num_workers=workers)
    val_loader = DataLoader(SegmentationDataset(val_ds), batch_size=args.batch_size, num_workers=workers)
    test_loader = DataLoader(SegmentationDataset(test_ds), batch_size=args.batch_size, num_workers=workers)

    model = get_segmenter().to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    best_iou, best_epoch = -1.0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        v = evaluate(model, val_loader, loss_fn, device)
        print(f"Epoch {epoch}/{args.epochs}  train loss {train_loss / len(train_loader):.4f}  "
              f"val loss {v['loss']:.4f}  val IoU {v['iou']:.4f}  val Dice {v['dice']:.4f}")
        if v["iou"] > best_iou:  # pick the epoch on validation, report on test
            best_iou, best_epoch = v["iou"], epoch
            torch.save(model.state_dict(), args.out)

    model.load_state_dict(torch.load(args.out, map_location=device))
    t = evaluate(model, test_loader, loss_fn, device)
    print(f"\nBest val epoch {best_epoch} (val IoU {best_iou:.4f})")
    print(f"Test (lesion images): IoU {t['iou']:.4f}  Dice {t['dice']:.4f}")
    print(f"Test (normal images): predicted no lesion on {100 * t['normal_clean_rate']:.1f}%")
    with open("weights/segmenter_metrics.json", "w") as f:
        json.dump({"best_epoch": best_epoch, "val_iou": best_iou, "test": t}, f, indent=2)
    print(f"Saved weights to {args.out}")


if __name__ == "__main__":
    main()
