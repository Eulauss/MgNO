#!/usr/bin/env python
"""Evaluate a saved Helmholtz checkpoint with relative H1 metrics."""

import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utilities3 import (
    HSloss_d,
    HSloss_d_relative,
    LpLoss,
    getDataSize,
    getHelmDataset,
)


def _load_model(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _prepare_loader(x, y, batch_size, device):
    if x.ndim == 3:
        x = x[:, None, ...]
    dataset = TensorDataset(x.contiguous().to(device), y.contiguous().to(device))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False), len(dataset)


@torch.no_grad()
def evaluate(model, loader, num_samples):
    l2loss = LpLoss(size_average=False)
    relative_h1_weight10 = HSloss_d_relative(l2_weight=10.0)
    relative_h1_weight1 = HSloss_d_relative(l2_weight=1.0)
    legacy_h1 = HSloss_d()
    total_l2 = 0.0
    total_relative_h1_weight10 = 0.0
    total_relative_h1_weight1 = 0.0
    total_legacy_h1_log = 0.0

    model.eval()
    for x, y in loader:
        out = model(x)
        total_l2 += l2loss(out, y).item()
        total_relative_h1_weight10 += relative_h1_weight10(out, y)[0].item()
        total_relative_h1_weight1 += relative_h1_weight1(out, y)[0].item()
        total_legacy_h1_log += legacy_h1(out, y)[0].item()

    return {
        "relative_l2": total_l2 / num_samples,
        "relative_h1_weight10": total_relative_h1_weight10 / num_samples,
        "relative_h1_weight1": total_relative_h1_weight1 / num_samples,
        "legacy_h1_log_scaled": total_legacy_h1_log / num_samples,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--normalizer_type", default="GN", choices=["GN", "PGN"])
    parser.add_argument("--no_GN", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    data_opt = {
        "data": "helm",
        "sampling_rate": 1,
        "sample_x": False,
        "batch_size": args.batch_size,
        "normalizer_type": args.normalizer_type,
        "GN": not args.no_GN,
        "data_root": args.data_root,
    }
    data_opt = getDataSize(data_opt)
    _, _, x_test, y_test, x_val, y_val, _, _ = getHelmDataset(data_opt)

    model = _load_model(args.checkpoint, device).to(device)
    test_loader, num_test = _prepare_loader(x_test, y_test, args.batch_size, device)
    val_loader, num_val = _prepare_loader(x_val, y_val, args.batch_size, device)

    result = {
        "checkpoint": args.checkpoint,
        "device": str(device),
        "test": evaluate(model, test_loader, num_test),
        "val": evaluate(model, val_loader, num_val),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
