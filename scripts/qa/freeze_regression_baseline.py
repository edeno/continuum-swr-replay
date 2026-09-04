#!/usr/bin/env python
"""Freeze the current canonical tables and model outputs as a read-only baseline.

The frozen copy is the reference a later refactor is diffed against: any change
that is supposed to preserve results must regenerate to a byte-identical copy of
this baseline (see ``compare_to_golden_baseline.py``). The tables and model
directories are copied verbatim and a ``manifest.json`` records each file's
sha256 and, for canonical tables, its row count, so the freeze itself is
auditable.

Usage:
    uv run python scripts/qa/freeze_regression_baseline.py                 # default label
    uv run python scripts/qa/freeze_regression_baseline.py --label v5      # explicit label
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import pyarrow.parquet as pq

from continuum_swr_replay.parameters import PROCESSED_DATA_DIR
from continuum_swr_replay.table_io import parquet_path, sha256_file
from continuum_swr_replay.table_parity import CANONICAL_TABLE_BASENAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)
RC = PROCESSED_DATA_DIR / "replay_content"

# qvals lives one level down; the rest sit at the replay_content root.
_TABLE_RELPATHS = {
    name: (
        "models_per_bin_rate/qvals_table" if name == "qvals_table" else name
    )
    for name in CANONICAL_TABLE_BASENAMES
}
_MODEL_DIRS = (
    "models", "models_swr_T", "models_swr_F", "models_patch",
    "models_patch_swr_T", "models_patch_swr_F", "models_event_grain",
    "models_per_stop_rate", "models_per_bin_rate", "uncertainty_content",
    "event_source_comparison",
)


def freeze(label: str) -> Path:
    out = RC / f"_golden_baseline_{label}"
    if out.exists():
        raise FileExistsError(f"{out} already exists; refusing to overwrite a baseline")
    out.mkdir(parents=True)
    manifest: dict[str, object] = {"tables": {}, "model_dirs": []}

    for name, relpath in _TABLE_RELPATHS.items():
        src = parquet_path(RC / relpath)
        if not src.exists():
            raise FileNotFoundError(f"canonical table missing: {src}")
        dst = out / f"{name}.parquet"
        shutil.copy2(src, dst)
        # Row count from the parquet footer -- no whole-file column load.
        n_rows = pq.read_metadata(dst).num_rows
        manifest["tables"][name] = {"sha256": sha256_file(dst), "n_rows": n_rows}
        logger.info("froze %s (%d rows)", name, n_rows)

    for model_dir in _MODEL_DIRS:
        src = RC / model_dir
        if src.exists():
            shutil.copytree(src, out / model_dir)
            manifest["model_dirs"].append(model_dir)
            logger.info("froze model dir %s", model_dir)

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    # Make the baseline read-only so a stray regeneration cannot overwrite it.
    for path in out.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
    logger.info("baseline frozen -> %s", out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="v5", help="baseline label (dir suffix)")
    args = parser.parse_args()
    freeze(args.label)


if __name__ == "__main__":
    main()
