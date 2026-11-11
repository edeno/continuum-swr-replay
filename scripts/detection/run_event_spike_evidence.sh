#!/usr/bin/env bash
# Score every non-local event of every analysed epoch with the decode's stored
# likelihood, one process per epoch. This is the spike-evidence run that the
# events summary and measure_ripple_class_structure.py read.
#
# Usage, inside tmux so it survives a disconnect:
#     tmux new -s event_spike_evidence 'scripts/detection/run_event_spike_evidence.sh 8'
#     scripts/detection/run_event_spike_evidence.sh NJOBS [options passed to
#         analyze_event_spike_evidence.py, such as --cache-dir <folder>]
# NJOBS is the number of epochs run at once (8 when no argument is given).
#
# Each epoch's tables go to event_spike_evidence/<epoch>/ under the
# processed-data root ($CSR_PROCESSED_DATA_DIR, else the checkout's
# Processed-Data/). An epoch is written beside them first, under .partial/, and
# moved into place when its metrics file is complete, so a stopped run never
# leaves half an epoch where the readers look. An epoch whose metrics file is in
# place is skipped: a stopped run is resumed by running this again, and an
# epoch is redone by deleting its folder.
# --n-sample is larger than any epoch's event count, so sampling leaves no event
# out.
# One line per epoch is printed (ok, skip or FAILED); each epoch's output is in
# logs/event_spike_evidence/<epoch>.log. The last line is
# EVENT_SPIKE_EVIDENCE_DONE and the exit status 0 only when no epoch failed.
set -uo pipefail

NJOBS="${1:-8}"
shift || true
cd "$(dirname "$0")/../.." || exit 1
source .venv/bin/activate || exit 1
# One epoch is a serial CPU job: keep each process to two threads.
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
OUT="$(python -c 'from continuum_swr_replay.paths import event_spike_evidence_dir; print(event_spike_evidence_dir())')" || exit 1
EPOCHS="$(python -c 'from scripts.detection.analyze_event_spike_evidence import find_sessions; print("\n".join(find_sessions(None)))')" || exit 1
if [ -z "$OUT" ] || [ -z "$EPOCHS" ]; then
  echo "no output folder or no analysed epoch with a decode under the data root" >&2
  exit 1
fi
export OUT
mkdir -p logs/event_spike_evidence

run_one() {
  local epoch="$1"
  shift
  local metrics=event_spike_evidence_metrics.parquet
  local partial="$OUT/.partial/$epoch"
  if [ -f "$OUT/$epoch/$metrics" ]; then
    echo "$epoch skip"
    return 0
  fi
  if [ -e "$OUT/$epoch" ]; then
    echo "$epoch FAILED: $OUT/$epoch exists without $metrics; delete it"
    return 1
  fi
  rm -rf "$partial"
  local start
  start=$(date +%s)
  # The analysis script exits 0 when an epoch's analysis raises, so an epoch
  # is done only when its metrics file was written.
  if python scripts/detection/analyze_event_spike_evidence.py \
      --epoch-identifiers "$epoch" --n-sample 1000000 --likelihood stored \
      --output-dir "$partial" "$@" > "logs/event_spike_evidence/$epoch.log" 2>&1 \
      && [ -f "$partial/$metrics" ] && mv "$partial" "$OUT/$epoch"; then
    echo "$epoch ok $(( $(date +%s) - start ))s"
  else
    echo "$epoch FAILED $(( $(date +%s) - start ))s"
    return 1
  fi
}
export -f run_one

if printf '%s\n' "$EPOCHS" \
    | xargs -P "$NJOBS" -I {} bash -c 'run_one "$@"' _ {} "$@"; then
  echo "EVENT_SPIKE_EVIDENCE_DONE"
else
  echo "EVENT_SPIKE_EVIDENCE_FAILED: see the FAILED lines and logs/event_spike_evidence/" >&2
  exit 1
fi
