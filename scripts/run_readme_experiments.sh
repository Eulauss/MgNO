#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${MGNO_DATA_ROOT:-/vepfs-dev/tianzt/pde_data}"
RUN_ROOT="${MGNO_RUN_ROOT:-/nas-dev-slow/pep_design/PDE/MgNO-runs}"
DEVICE="${MGNO_DEVICE:-cuda:1}"
CONDA_ENV="${MGNO_CONDA_ENV:-fno}"
EXPERIMENT_PREFIX="${MGNO_EXPERIMENT_PREFIX:-}"
DRY_RUN="${MGNO_DRY_RUN:-0}"

DARCY_EPOCHS="${MGNO_DARCY_EPOCHS:-500}"
NAVIER_EPOCHS="${MGNO_NAVIER_EPOCHS:-500}"
PIPE_EPOCHS="${MGNO_PIPE_EPOCHS:-500}"
HELM_EPOCHS="${MGNO_HELM_EPOCHS:-100}"

mkdir -p "$RUN_ROOT"
cd "$REPO_ROOT"

print_command() {
  printf '%q ' "$@"
  printf '\n'
}

run_one() {
  local experiment="$1"
  shift
  local experiment_name="${EXPERIMENT_PREFIX}${experiment}"
  local stdout_dir="$RUN_ROOT/$experiment_name/stdout"
  local stdout_log="$stdout_dir/$(date +%Y%m%d_%H%M%S).stdout.log"
  local cmd=(
    conda run -n "$CONDA_ENV" python "$@"
    --data_root "$DATA_ROOT"
    --run_root "$RUN_ROOT"
    --experiment_name "$experiment_name"
    --device "$DEVICE"
  )

  mkdir -p "$stdout_dir"
  echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] START $experiment_name"
  echo "stdout: $stdout_log"
  echo -n "command: "
  print_command "${cmd[@]}"

  if [[ "$DRY_RUN" == "1" ]]; then
    "${cmd[@]}" --dry_run 2>&1 | tee "$stdout_log"
  else
    "${cmd[@]}" 2>&1 | tee "$stdout_log"
  fi

  echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] END $experiment_name"
}

run_one darcy_smooth darcy.py \
  --data darcy --model_type MgNO_DC_smooth \
  --sample_x --normalizer --normalizer_type PGN --GN \
  --num_channel_u 24 --num_layer 5 \
  --num_iteration 10 10 10 10 10 20 \
  --lr 5e-4 --batch_size 8 --epochs "$DARCY_EPOCHS"

run_one darcy_rough darcy.py \
  --data darcy20c6 --model_type MgNO_DC \
  --sample_x --normalizer --normalizer_type GN \
  --num_channel_u 24 --num_layer 4 \
  --num_iteration 10 10 10 10 10 20 \
  --lr 5e-4 --batch_size 8 --epochs "$DARCY_EPOCHS"

run_one navier_stokes_1e-5 navier.py \
  --data 1e-5 --model_type MgNO \
  --num_iteration 10 10 10 20 20 --num_layer 5 \
  --num_channel_u 32 --num_channel_f 1 \
  --final_div_factor 50 --weight_decay 1e-5 \
  --lr 1e-3 --batch_size 50 --epochs "$NAVIER_EPOCHS" --bias

run_one pipe darcy.py \
  --data pipe --model_type MgNO_DC \
  --sample_x --num_channel_f 2 \
  --num_channel_u 24 --num_layer 5 \
  --num_iteration 10 10 10 10 11 20 \
  --lr 3e-4 --batch_size 4 --epochs "$PIPE_EPOCHS" --loss_type l2

run_one helmholtz helm.py \
  --data helm --model_type MgNO_helm \
  --num_layer 4 --lr 3e-4 \
  --final_div_factor 100 --batch_size 10 \
  --weight_decay 1e-5 --normalizer --GN \
  --num_channel_u 20 --num_iteration 1 1 1 1 2 \
  --epochs "$HELM_EPOCHS"
