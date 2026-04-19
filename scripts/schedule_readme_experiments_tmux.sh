#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${MGNO_RUN_ROOT:-/nas-dev-slow/pep_design/PDE/MgNO-runs}"
SESSION="${MGNO_TMUX_SESSION:-mgno_readme_repro}"
DELAY_SECONDS="${MGNO_START_DELAY_SECONDS:-14400}"
RUNNER_LOG="$RUN_ROOT/scheduler/$(date +%Y%m%d_%H%M%S).runner.log"

mkdir -p "$(dirname "$RUNNER_LOG")"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  echo "Attach with: tmux attach -t $SESSION" >&2
  exit 1
fi

tmux new-session -d -s "$SESSION" \
  "cd '$REPO_ROOT' && \
   echo '[schedule] start utc: '\"\$(date -u '+%Y-%m-%d %H:%M:%S UTC')\" && \
   echo '[schedule] sleeping ${DELAY_SECONDS}s before launch' && \
   echo '[schedule] runner log: $RUNNER_LOG' && \
   sleep '$DELAY_SECONDS' && \
   echo '[schedule] launching utc: '\"\$(date -u '+%Y-%m-%d %H:%M:%S UTC')\" && \
   bash scripts/run_readme_experiments.sh 2>&1 | tee -a '$RUNNER_LOG'"

echo "Created tmux session: $SESSION"
echo "Delay seconds: $DELAY_SECONDS"
echo "Runner log: $RUNNER_LOG"
echo "Attach with: tmux attach -t $SESSION"
