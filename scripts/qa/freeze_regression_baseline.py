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
import sys
from pathlib import Path

import pyarrow.parquet as pq

from continuum_swr_replay.parameters import PROCESSED_DATA_DIR
from continuum_swr_replay.table_io import parquet_path, sha256_file
from continuum_swr_replay.table_parity import CANONICAL_TABLE_BASENAMES

# Sibling-script import: the leaf bundle's name is derived from the fitting
# runner's own helper rather than spelled out here, so the freeze cannot go on
# skipping a ladder whose directory name changed. Run as a script, only this
# file's directory is on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.content.fit_replay_content_models import (  # noqa: E402
    belief_chain_output_subdir,
)

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
# Every fitted bundle the baseline is supposed to cover. ``models_event_grain``
# is not among them: the event-grain ladder is no longer refit (the fraction
# ladder serves its purpose) and survives only in backups, so freezing it would
# copy a stale bundle into a fresh baseline. The leaf-only belief chain's
# directory is derived, not spelled out, because it is keyed by the rungs it
# fits.
# NB the Dirichlet-multinomial ``*_dm`` / ``*_dm_len`` contrast artifacts that
# live INSIDE ``models`` are copied along with that directory (``copytree`` has
# no ignore filter); add one here if that ever matters.
_MODEL_DIRS = (
    "models",
    belief_chain_output_subdir(),
    "models_per_stop_rate",
    "models_per_bin_rate",
    "uncertainty_content",
    "event_source_comparison",
)


def freeze(label: str) -> Path:
    out = RC / f"_golden_baseline_{label}"
    if out.exists():
        raise FileExistsError(f"{out} already exists; refusing to overwrite a baseline")
    out.mkdir(parents=True)
    manifest: dict[str, object] = {
        "tables": {},
        "model_dirs": [],
        "missing_model_dirs": [],
    }

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
        if not src.exists():
            # Named, not silent: a baseline missing a ladder is a baseline a
            # later comparison cannot check that ladder against, and the freeze
            # is the only place that fact is visible.
            manifest["missing_model_dirs"].append(model_dir)
            logger.warning(
                "model dir %s is absent from %s; the baseline will not cover it",
                model_dir,
                RC,
            )
            continue
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
