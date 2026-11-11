#!/usr/bin/env bash
# Score every non-local event of every analysed epoch with the decode's stored
# likelihood, one process per epoch. This is the spike-evidence run that the
# events summary and measure_ripple_class_structure.py read.
#
# Usage, inside tmux so it survives a disconnect:
#     tmux new -s event_spike_evidence 'scripts/detection/run_event_spike_evidence.sh 8'
#     scripts/detection/run_event_spike_evidence.sh [NJOBS] [options passed to
#         analyze_event_spike_evidence.py, such as --cache-dir <folder>]
#
# Each epoch's tables go to event_spike_evidence/<epoch>/ under the
# processed-data root ($CSR_PROCESSED_DATA_DIR, else the checkout's
# Processed-Data/). An epoch whose metrics file exists is skipped, so a stopped
# run is resumed by running this again; delete an epoch's folder to redo it.
# --n-sample is larger than any epoch's event count, so every event is scored.
# One line per epoch is printed; each epoch's output is in
# logs/event_spike_evidence/<epoch>.log.
set -uo pipefail

NJOBS="${1:-8}"
shift || true
cd "$(dirname "$0")/../.."
source .venv/bin/activate
# One epoch is a serial CPU job: keep each process to two threads.
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
OUT="$(python -c 'from continuum_swr_replay.paths import event_spike_evidence_dir; print(event_spike_evidence_dir())')"
export OUT
mkdir -p logs/event_spike_evidence

run_one() {
  local epoch="$1"
  shift
  if [ -f "$OUT/$epoch/event_spike_evidence_metrics.parquet" ]; then
    echo "$epoch skip"
    return
  fi
  local start
  start=$(date +%s)
  if python scripts/detection/analyze_event_spike_evidence.py \
      --epoch-identifiers "$epoch" --n-sample 1000000 --likelihood stored \
      --output-dir "$OUT/$epoch" "$@" > "logs/event_spike_evidence/$epoch.log" 2>&1; then
    echo "$epoch ok $(( $(date +%s) - start ))s"
  else
    echo "$epoch FAILED $(( $(date +%s) - start ))s"
  fi
}
export -f run_one

python -c 'from scripts.detection.analyze_event_spike_evidence import find_sessions; print("\n".join(find_sessions(None)))' \
  | xargs -P "$NJOBS" -I {} bash -c 'run_one "$@"' _ {} "$@"

echo "EVENT_SPIKE_EVIDENCE_DONE"
