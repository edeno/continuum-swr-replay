#!/usr/bin/env bash
# Render the per-event decoding gallery in parallel across epochs (tmux).
#
# Phase 1 (catalog) must already have run:
#     python scripts/figures/generate_event_gallery.py catalog
#
# Usage:
#     scripts/figures/run_event_gallery.sh [NJOBS]
# Default NJOBS=6. Each epoch is loaded once (Spyglass ~9 min) and all its
# selected events rendered; epochs run NJOBS at a time. Per-epoch logs land
# in logs/event_gallery/. Run inside tmux so it survives disconnects:
#     tmux new -s event_gallery_render 'scripts/figures/run_event_gallery.sh 6'
set -uo pipefail

NJOBS="${1:-6}"
cd "$(dirname "$0")/.."
source .venv/bin/activate
mkdir -p logs/event_gallery

echo "Rendering gallery with NJOBS=$NJOBS over $(python scripts/figures/generate_event_gallery.py list-epochs | wc -l) epochs"

python scripts/figures/generate_event_gallery.py list-epochs \
  | xargs -P "$NJOBS" -I {} bash -c \
    'python scripts/figures/generate_event_gallery.py render --epoch-id "{}" \
       > logs/event_gallery/{}.log 2>&1 \
     && echo "done {}" || echo "FAILED {}"'

echo "RENDER_ALL_DONE"
