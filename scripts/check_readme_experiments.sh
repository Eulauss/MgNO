#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${MGNO_RUN_ROOT:-/nas-dev-slow/pep_design/PDE/MgNO-runs}"
CONDA_ENV="${MGNO_CONDA_ENV:-fno}"

cd "$REPO_ROOT"

echo "== Python compile check =="
CUDA_VISIBLE_DEVICES="" conda run -n "$CONDA_ENV" python -m py_compile \
  utilities3.py darcy.py navier.py helm.py models.py Adam.py

echo "== Data path and dependency check =="
CUDA_VISIBLE_DEVICES="" conda run -n "$CONDA_ENV" python - <<'PY'
import os
import sys
import torch
import numpy
import scipy
import h5py

from utilities3 import getPath

root = os.environ.get("MGNO_DATA_ROOT", "/vepfs-dev/tianzt/pde_data")
checks = [
    ("darcy smooth train", getPath("darcy", "train", data_root=root)),
    ("darcy smooth test", getPath("darcy", "test", data_root=root)),
    ("darcy rough train", getPath("darcy20c6", "train", data_root=root)),
    ("darcy rough test", getPath("darcy20c6", "test", data_root=root)),
    ("navier stokes", getPath("1e-5", None, data_root=root)),
    ("pipe x", getPath("pipe", "x", data_root=root)),
    ("pipe y", getPath("pipe", "y", data_root=root)),
    ("pipe q", getPath("pipe", "q", data_root=root)),
    ("helm inputs", getPath("helm", "x", data_root=root)),
    ("helm outputs", getPath("helm", "y", data_root=root)),
]
missing = []
for name, path in checks:
    exists = os.path.exists(path)
    print(f"{name}: {path} exists={exists}")
    if not exists:
        missing.append((name, path))

print(f"python={sys.executable}")
print(f"torch={torch.__version__}")
print(f"cuda_visible={os.environ.get('CUDA_VISIBLE_DEVICES')!r}")
print(f"torch_cuda_available={torch.cuda.is_available()}")
print(f"numpy={numpy.__version__} scipy={scipy.__version__} h5py={h5py.__version__}")

if missing:
    raise SystemExit("missing required data files: " + repr(missing))
PY

echo "== README command dry-run check =="
MGNO_DRY_RUN=1 \
MGNO_DEVICE=cpu \
MGNO_EXPERIMENT_PREFIX=check_ \
MGNO_RUN_ROOT="$RUN_ROOT" \
bash scripts/run_readme_experiments.sh

echo "All dry-run checks passed."
