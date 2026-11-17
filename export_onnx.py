"""Convert the trained PyTorch weights to ONNX so they can run in a web browser.

Reads  weights/classifier.pt, weights/segmenter.pt
Writes weights/classifier.onnx, weights/segmenter.onnx
and checks that ONNX Runtime gives the same outputs as PyTorch.
"""
import numpy as np
import onnxruntime as ort
import torch

from bcus.models import CLS_IMAGE_SIZE, SEG_IMAGE_SIZE, get_classifier, get_segmenter


def export(model, size, path):
    model.eval()
    dummy = torch.randn(1, 3, size, size)
    torch.onnx.export(model, dummy, path, input_names=["input"], output_names=["logits"],
                      opset_version=17, dynamo=False)
    with torch.no_grad():
        expected = model(dummy).numpy()
    got = ort.InferenceSession(path, providers=["CPUExecutionProvider"]).run(None, {"input": dummy.numpy()})[0]
    diff = float(np.abs(expected - got).max())
    print(f"{path}: max difference vs PyTorch = {diff:.2e}")
    assert diff < 1e-3, "ONNX output does not match PyTorch"


def main():
    cls = get_classifier(pretrained=False)
    cls.load_state_dict(torch.load("weights/classifier.pt", map_location="cpu"))
    export(cls, CLS_IMAGE_SIZE, "weights/classifier.onnx")

    seg = get_segmenter(pretrained=False)
    seg.load_state_dict(torch.load("weights/segmenter.pt", map_location="cpu"))
    export(seg, SEG_IMAGE_SIZE, "weights/segmenter.onnx")


if __name__ == "__main__":
    main()
