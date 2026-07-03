#!/usr/bin/env bash
# Fetch the canonical parsed behavior CSVs for the spatial bandit task from
# LorenFrankLab/SpatialBanditTask into Processed-Data/spatial_bandit_behavior/.
# These per-trial CSVs (choice=leaf, patch=stem, reward, contingency, date,
# session, trial) were the input to the (now-retired) difflib nominal-value
# pipeline. The canonical per-stop value covariates now come from the
# DataJoint-exact `qvals_table` (analysis/behavior/qvals_table.py); these CSVs
# are kept only as a raw reference / cross-check of the behavior parse.
#
# The data is gitignored (lives under Processed-Data/); this script makes the
# pull reproducible. Pinned to a commit for reproducibility.
set -euo pipefail

REPO="LorenFrankLab/SpatialBanditTask"
COMMIT="96dfed1b7efbce1c2d478968396f729b0e437a85"   # pinned 2026-06-28
DEST="${PROCESSED_DATA_DIR:-Processed-Data}/spatial_bandit_behavior"
BASE="https://raw.githubusercontent.com/${REPO}/${COMMIT}"

mkdir -p "$DEST"
paths=$(gh api "repos/${REPO}/git/trees/${COMMIT}?recursive=1" \
  --jq '.tree[] | select(.type=="blob" and (.path|test("\\.(csv|m)$"))) | .path')

n=0
for p in $paths; do
  mkdir -p "$DEST/$(dirname "$p")"
  curl -sfL "$BASE/$p" -o "$DEST/$p" && n=$((n+1)) || echo "FAILED: $p" >&2
done
printf "Source: https://github.com/%s\nCommit: %s\nFetched per-trial bandit behavior for chimi/j16/peanut/senor/wilbur.\n" \
  "$REPO" "$COMMIT" > "$DEST/PROVENANCE.txt"
echo "fetched $n files -> $DEST"
