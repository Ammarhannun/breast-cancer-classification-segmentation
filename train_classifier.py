"""Train the ResNet-50 classifier and save weights/classifier.pt.

Usage: python train_classifier.py [--epochs 15]
"""
import argparse
import json
import os

import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from bcus.data import ClassificationDataset, load_records, split_records
from bcus.models import CLASS_NAMES, get_classifier
from bcus.preprocess import cls_eval_transform, cls_train_transform


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, labels = [], []
    for x, y in loader:
        preds += model(x.to(device)).argmax(1).cpu().tolist()
        labels += y.tolist()
    return preds, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", default="weights/classifier.pt")
    args = ap.parse_args()

    torch.manual_seed(42)
    device = pick_device()
    print(f"Device: {device}")

    train_ds, val_ds, test_ds = split_records(load_records())
    workers = 2 if device.type == "cuda" else 0
    print(f"Train {len(train_ds)} | Val {len(val_ds)} | Test {len(test_ds)}")

    train_loader = DataLoader(ClassificationDataset(train_ds, cls_train_transform),
                              batch_size=args.batch_size, shuffle=True, num_workers=workers)
    val_loader = DataLoader(ClassificationDataset(val_ds, cls_eval_transform),
                            batch_size=args.batch_size, num_workers=workers)
    test_loader = DataLoader(ClassificationDataset(test_ds, cls_eval_transform),
                             batch_size=args.batch_size, num_workers=workers)

    model = get_classifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    best_val = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, correct, seen = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y.size(0)
            correct += (out.argmax(1) == y).sum().item()
            seen += y.size(0)

        vp, vl = predict(model, val_loader, device)
        val_acc = sum(p == t for p, t in zip(vp, vl)) / len(vl)
        print(f"Epoch {epoch}/{args.epochs}  loss {total_loss / seen:.4f}  "
              f"train acc {correct / seen:.4f}  val acc {val_acc:.4f}")
        if val_acc > best_val:  # keep the best epoch, not the last one
            best_val = val_acc
            torch.save(model.state_dict(), args.out)

    model.load_state_dict(torch.load(args.out, map_location=device))
    tp, tl = predict(model, test_loader, device)
    test_acc = sum(p == t for p, t in zip(tp, tl)) / len(tl)
    report = classification_report(tl, tp, target_names=CLASS_NAMES, output_dict=True)
    cm = confusion_matrix(tl, tp).tolist()

    print(f"\nBest val acc: {best_val:.4f}")
    print(f"Test acc:     {test_acc:.4f}")
    print(classification_report(tl, tp, target_names=CLASS_NAMES, digits=3))
    print("Confusion matrix (rows = true, cols = predicted):", CLASS_NAMES)
    for name, row in zip(CLASS_NAMES, cm):
        print(f"  {name:<10} {row}")

    with open("weights/classifier_metrics.json", "w") as f:
        json.dump({"val_acc": best_val, "test_acc": test_acc, "report": report,
                   "confusion_matrix": cm, "labels": CLASS_NAMES}, f, indent=2)

    # Save one held-out test image per class as clickable examples for the app.
    os.makedirs("examples", exist_ok=True)
    saved = set()
    for item in test_ds:
        name = CLASS_NAMES[item["label"]]
        if name not in saved:
            item["image"].save(f"examples/{name}.png")
            saved.add(name)
        if len(saved) == len(CLASS_NAMES):
            break
    print(f"Saved weights to {args.out} and examples to examples/")


if __name__ == "__main__":
    main()
