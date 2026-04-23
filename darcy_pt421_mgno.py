"""
darcy_pt421_mgno.py — Train MgNO_DC_smooth on the neuralop Darcy PT-421 dataset.

Data source: darcy_train_421.pt / darcy_test_421.pt (n=4000/1000, res=421x421).
This matches the darcy_pt421.yaml experiment in unigen-hrm exactly:
  - n_train=1000, n_test=100
  - same eval set as unigen-hrm (darcy_test_421.pt[0:100])

Output layout (under --run_root / --experiment_name / <timestamp>/):
  train.log        — same format as unigen-hrm
  metrics.jsonl    — one JSON record per epoch
  config.yaml      — saved run config
  checkpoints/
    last.pt
    best.pt
    epoch_NNN.pt   (every --save_every epochs)

NOTE on resolution:
  MgNO_DC_smooth's V-cycle uses alternating ConvTranspose2d kernel sizes
  [3,4,3,3,4] that are tuned for 211×211 input (same as --sample_x in the
  README command).  The PT-421 data is 421×421; we subsample by 2 (::2,::2)
  before training, yielding 211×211 — identical to the README smooth-darcy
  experiment.  Both x and y are subsampled, matching getDarcyDataSet() in
  utilities3.py.

Usage:
  python darcy_pt421_mgno.py \
    --data_root /vepfs-dev/pep_design/miniconda3/envs/fno/lib/python3.10/site-packages/neuralop/data/datasets/data \
    --run_root  /nas-dev-slow/pep_design/PDE/MgNO-runs \
    --device    cuda:1
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent))
# torchinfo is used inside models.py; fall back to the fno env's copy if needed
_fno_site = "/vepfs-dev/pep_design/miniconda3/envs/fno/lib/python3.10/site-packages"
if os.path.isdir(_fno_site):
    sys.path.append(_fno_site)

from models import MgNO_DC_smooth
from utilities3 import (
    HsLoss,
    LpLoss,
    UnitGaussianNormalizer,
    GaussianNormalizer,
    count_params,
)
from Adam import Adam

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_pt(path: Path, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Load x, y from a neuralop Darcy PT file.  Returns [n, H, W] tensors."""
    data = torch.load(str(path), weights_only=False)
    x = data["x"][:n].float()   # [N, H, W]
    y = data["y"][:n].float()   # [N, H, W]
    return x, y


def load_darcy_pt421(
    data_root: Path,
    n_train: int = 1000,
    n_test: int = 100,
    n_val: int = 100,
) -> tuple[torch.Tensor, ...]:
    """Return (x_train, y_train, x_test, y_test, x_val, y_val) — all [N, H, W]."""
    train_file = data_root / "darcy_train_421.pt"
    test_file  = data_root / "darcy_test_421.pt"
    for p in (train_file, test_file):
        if not p.exists():
            raise FileNotFoundError(f"Missing data file: {p}")

    x_tr, y_tr = _load_pt(train_file, n_train)
    data_test = torch.load(str(test_file), weights_only=False)
    x_all = data_test["x"].float()
    y_all = data_test["y"].float()
    x_te = x_all[:n_test]
    y_te = y_all[:n_test]
    x_val = x_all[n_test: n_test + n_val]
    y_val = y_all[n_test: n_test + n_val]
    return x_tr, y_tr, x_te, y_te, x_val, y_val


# ---------------------------------------------------------------------------
# Checkpoint / metrics helpers  (matches unigen-hrm format)
# ---------------------------------------------------------------------------

class CheckpointManager:
    def __init__(self, ckpt_dir: Path, save_every: int = 50, keep_last_k: int = 2) -> None:
        self.dir = ckpt_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.save_every = save_every
        self.keep_last_k = keep_last_k
        self._periodic: list[Path] = []

    def save(self, obj: dict, name: str) -> None:
        torch.save(obj, self.dir / name)

    def save_periodic(self, obj: dict, epoch: int) -> None:
        path = self.dir / f"epoch_{epoch:03d}.pt"
        torch.save(obj, path)
        self._periodic.append(path)
        while len(self._periodic) > self.keep_last_k:
            old = self._periodic.pop(0)
            if old.exists():
                old.unlink()


class MetricsWriter:
    def __init__(self, path: Path) -> None:
        self._f = open(path, "a")

    def write(self, record: dict) -> None:
        self._f.write(json.dumps(record) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    h1loss: HsLoss,
    l2loss: LpLoss,
    n_samples: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_h1, total_l2 = 0.0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        total_h1 += h1loss(out, y)[0].item()
        total_l2 += l2loss(out, y).item()
    return {
        "h1": total_h1 / n_samples,
        "l2": total_l2 / n_samples,
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args: dict) -> None:
    # ------------------------------------------------------------------ paths
    data_root = Path(args["data_root"])
    run_root  = Path(args["run_root"])
    exp_name  = args["experiment_name"]
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_dir   = run_root / exp_name / timestamp
    ckpt_dir  = exp_dir / "checkpoints"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------- logging
    log_fmt = "%(asctime)s | %(levelname)s | %(message)s"
    log_datefmt = "%Y-%m-%d %H:%M:%S,%f"[:-3]   # match unigen-hrm
    logging.basicConfig(
        level=logging.INFO,
        format=log_fmt,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(exp_dir / "train.log"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    logger = logging.getLogger("darcy_pt421_mgno")

    # ---------------------------------------------------------------- device
    device = torch.device(
        args["device"] or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)

    # --------------------------------------------------------------- config
    config = {
        "experiment": {
            "name": exp_name,
            "output_root": str(run_root),
            "timestamp": timestamp,
            "device": str(device),
        },
        "data": {
            "data_root": str(data_root),
            "dataset": "darcy_pt421",
            "n_train": args["n_train"],
            "n_test": args["n_test"],
            "n_val": args["n_val"],
            "batch_size": args["batch_size"],
        },
        "model": {
            "model_type": "MgNO_DC_smooth",
            "num_layer": args["num_layer"],
            "num_channel_u": args["num_channel_u"],
            "num_channel_f": 1,
            "num_iteration": args["num_iteration"],
            "normalizer": "UnitGaussian (PGN)",
            "GN": True,
            "input_resolution": f"{421 // args['sampling_rate'] + 421 % args['sampling_rate']}",
            "sampling_rate": args["sampling_rate"],
        },
        "train": {
            "epochs": args["epochs"],
            "lr": args["lr"],
            "weight_decay": args["weight_decay"],
            "final_div_factor": args["final_div_factor"],
            "loss_type": args["loss_type"],
        },
        "logging": {
            "save_every": args["save_every"],
            "keep_last_k": args["keep_last_k"],
        },
    }
    with open(exp_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    ckpt_mgr = CheckpointManager(
        ckpt_dir, save_every=args["save_every"], keep_last_k=args["keep_last_k"]
    )
    metrics_writer = MetricsWriter(exp_dir / "metrics.jsonl")

    logger.info("Experiment directory: %s", exp_dir)
    logger.info("Device: %s", device)

    # ------------------------------------------------------------------ data
    logger.info("Loading data from %s", data_root)
    x_tr, y_tr, x_te, y_te, x_val, y_val = load_darcy_pt421(
        data_root,
        n_train=args["n_train"],
        n_test=args["n_test"],
        n_val=args["n_val"],
    )
    # MgNO_DC_smooth's V-cycle requires 211×211 (see module docstring).
    # Apply the same subsampling as --sample_x --sampling_rate 2 in README.
    r = args["sampling_rate"]
    x_tr  = x_tr[:,  ::r, ::r]   # [N, 211, 211]
    x_te  = x_te[:,  ::r, ::r]
    x_val = x_val[:, ::r, ::r]
    y_tr  = y_tr[:,  ::r, ::r]   # y also subsampled, matching utilities3.py
    y_te  = y_te[:,  ::r, ::r]
    y_val = y_val[:, ::r, ::r]

    # Add channel dim for convolutions
    x_tr  = x_tr.unsqueeze(1)   # [N, 1, H, W]
    x_te  = x_te.unsqueeze(1)
    x_val = x_val.unsqueeze(1)
    # y stays [N, H, W] — matches MgNO's convention

    # Input normalizer (PGN = UnitGaussianNormalizer, per-pixel)
    x_normalizer = UnitGaussianNormalizer(x_tr)
    x_tr  = x_normalizer.encode(x_tr)
    x_te  = x_normalizer.encode(x_te)
    x_val = x_normalizer.encode(x_val)

    # Output normalizer — passed to model for internal decode
    y_normalizer = UnitGaussianNormalizer(y_tr)

    logger.info(
        "Data: train=%d  test=%d  val=%d  res=%s  (subsampled from 421 by rate %d)",
        len(x_tr), len(x_te), len(x_val), str(tuple(x_tr.shape[2:])), r,
    )

    x_tr  = x_tr.contiguous().to(device)
    y_tr  = y_tr.contiguous().to(device)
    x_te  = x_te.contiguous().to(device)
    y_te  = y_te.contiguous().to(device)
    x_val = x_val.contiguous().to(device)
    y_val = y_val.contiguous().to(device)

    train_loader = DataLoader(
        TensorDataset(x_tr, y_tr), batch_size=args["batch_size"], shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(x_te, y_te), batch_size=args["batch_size"], shuffle=False
    )
    val_loader = DataLoader(
        TensorDataset(x_val, y_val), batch_size=args["batch_size"], shuffle=False
    )

    # ----------------------------------------------------------------- model
    num_iteration_parsed = [
        [int(v) for v in pair] for pair in args["num_iteration"]
    ]
    model = MgNO_DC_smooth(
        num_layer=args["num_layer"],
        num_channel_u=args["num_channel_u"],
        num_channel_f=1,
        num_classes=1,
        num_iteration=num_iteration_parsed,
        in_chans=1,
        normalizer=y_normalizer,
        output_dim=1,
        activation="gelu",
    ).to(device)
    y_normalizer.to(device)

    n_params = count_params(model)
    logger.info("Model: MgNO_DC_smooth  |  parameters: %d", n_params)

    # ------------------------------------------------------------- optimizer
    optimizer = Adam(model.parameters(), lr=args["lr"], weight_decay=args["weight_decay"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args["lr"],
        div_factor=2,
        final_div_factor=args["final_div_factor"],
        pct_start=0.1,
        steps_per_epoch=1,
        epochs=args["epochs"],
    )

    # ---------------------------------------------------------------- losses
    H = y_tr.size(1)   # spatial resolution
    h1loss_train = HsLoss(d=2, p=2, k=1, size_average=False, res=H)
    if device.type == "cuda":
        h1loss_train.cuda(device)
    else:
        h1loss_train.cpu()
    l2loss = LpLoss(size_average=False)

    # ---------------------------------------------------------------- train
    logger.info(
        "Training: epochs=%d  lr=%.1e  batch_size=%d  loss=%s",
        args["epochs"], args["lr"], args["batch_size"], args["loss_type"],
    )

    best_val_h1 = float("inf")
    total_start = time.time()

    for epoch in range(1, args["epochs"] + 1):
        epoch_start = time.time()
        model.train()
        train_h1_sum, train_l2_sum = 0.0, 0.0

        for x, y in train_loader:
            optimizer.zero_grad()
            out = model(x)
            if args["loss_type"] == "h1":
                h1_val, _, _, _ = h1loss_train(out, y)
                h1_val.backward()
                with torch.no_grad():
                    l2_val = l2loss(out, y)
            else:
                l2_val = l2loss(out, y)
                l2_val.backward()
                with torch.no_grad():
                    h1_val, _, _, _ = h1loss_train(out, y)
            optimizer.step()
            train_h1_sum += h1_val.item()
            train_l2_sum += l2_val.item()

        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        train_h1 = train_h1_sum / args["n_train"]
        train_l2 = train_l2_sum / args["n_train"]
        train_loss = train_h1 if args["loss_type"] == "h1" else train_l2

        # evaluate on val (primary) and test
        val_m  = evaluate(model, val_loader,  h1loss_train, l2loss, args["n_val"],  device)
        test_m = evaluate(model, test_loader, h1loss_train, l2loss, args["n_test"], device)

        epoch_time = time.time() - epoch_start
        total_time = time.time() - total_start

        # primary metric = val H1 (matches unigen-hrm primary_metric: h1)
        val_loss = val_m["h1"]

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_h1": train_h1,
            "train_l2": train_l2,
            "val_loss": val_loss,
            "val_h1": val_m["h1"],
            "val_l2": val_m["l2"],
            "test_h1": test_m["h1"],
            "test_l2": test_m["l2"],
            "lr": lr,
            "epoch_time_sec": epoch_time,
            "total_time_sec": total_time,
        }
        metrics_writer.write(record)

        logger.info(
            "Epoch %03d/%03d | train_loss=%.6f | val_loss(h1)=%.6f"
            " | val_h1=%.6f | val_l2=%.6f | test_h1=%.6f | test_l2=%.6f"
            " | lr=%.6e | epoch_time=%.2fs | total_time=%.2fs",
            epoch, args["epochs"],
            train_loss, val_loss,
            val_m["h1"], val_m["l2"],
            test_m["h1"], test_m["l2"],
            lr, epoch_time, total_time,
        )

        # -------------------------------------------------------- checkpoints
        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "metrics": record,
            "best_val_loss": min(best_val_h1, val_loss),
            "config": config,
        }
        ckpt_mgr.save(ckpt, "last.pt")

        if val_loss < best_val_h1:
            best_val_h1 = val_loss
            ckpt["best_val_loss"] = best_val_h1
            ckpt_mgr.save(ckpt, "best.pt")

        if epoch % args["save_every"] == 0:
            ckpt_mgr.save_periodic(ckpt, epoch)

    metrics_writer.close()

    # load best checkpoint and report its test metrics
    best_ckpt = torch.load(str(ckpt_dir / "best.pt"), weights_only=False)
    best_epoch = best_ckpt["epoch"]
    best_metrics = best_ckpt["metrics"]
    logger.info(
        "Training finished. Best epoch=%d | val_h1=%.6f | test_h1=%.6f | test_l2=%.6f",
        best_epoch,
        best_metrics["val_h1"],
        best_metrics["test_h1"],
        best_metrics["test_l2"],
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train MgNO_DC_smooth on neuralop Darcy PT-421 dataset"
    )
    parser.add_argument("--data_root",  type=str,   required=True)
    parser.add_argument("--run_root",   type=str,   default="/nas-dev-slow/pep_design/PDE/MgNO-runs")
    parser.add_argument("--experiment_name", type=str, default="darcy_pt421_mgno")
    parser.add_argument("--device",     type=str,   default=None)

    # data
    parser.add_argument("--n_train",    type=int,   default=1000)
    parser.add_argument("--n_test",     type=int,   default=100)
    parser.add_argument("--n_val",      type=int,   default=100)
    parser.add_argument("--batch_size",    type=int,   default=8)
    parser.add_argument("--sampling_rate", type=int,   default=2,
                        help="Subsample factor applied to x and y (same as --sample_x in README)")

    # model  (README smooth darcy defaults)
    parser.add_argument("--num_layer",     type=int,   default=5)
    parser.add_argument("--num_channel_u", type=int,   default=24)
    parser.add_argument(
        "--num_iteration", type=int, nargs="+",
        default=[10, 10, 10, 10, 10, 20],
        help="Flattened list of V-cycle iterations per level; "
             "consecutive pairs form [[a,b], ...] layers.",
    )

    # optimizer  (README smooth darcy defaults)
    parser.add_argument("--epochs",           type=int,   default=500)
    parser.add_argument("--lr",               type=float, default=5e-4)
    parser.add_argument("--weight_decay",     type=float, default=1e-4)
    parser.add_argument("--final_div_factor", type=float, default=100.0)
    parser.add_argument("--loss_type",        type=str,   default="h1",
                        help="Training loss: h1 or l2")

    # logging
    parser.add_argument("--save_every",  type=int, default=50)
    parser.add_argument("--keep_last_k", type=int, default=2)

    raw = parser.parse_args()
    args = vars(raw)

    # Reshape num_iteration from flat list [a0,b0,a1,b1,...] to [[a0,b0],...]
    flat = args["num_iteration"]
    if len(flat) % 2 != 0:
        flat = flat + [0]   # pad to even length
    args["num_iteration"] = [[flat[i], flat[i + 1]] for i in range(0, len(flat), 2)]

    train(args)
