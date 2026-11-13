"""Compute raw-data QC stats for hippocampal clusterless decoding.

For each epoch, loads raw spike times and waveform features via
``load_data()`` and the fitted detector pickle, then computes:

- Per-tetrode spike counts and rates (``n_tetrodes``, ``rate_hz_per_tetrode``, ...)
- Waveform absolute-amplitude distribution (``wf_abs_median``, ``wf_abs_p99``,
  ``wf_per_tet_abs_median_max``, ...)
- Flattened 4x4 discrete state transition matrix (``T_local_local``, ...,
  ``T_nlf_nlf``)

Results are written incrementally to a CSV so a crash mid-run does not lose
progress. Sessions already present in the CSV are skipped unless
``--overwrite`` is passed.

The resulting ``hpc_raw_stats.csv`` is merged into ``quality_summary.csv``
by :mod:`scripts.run_quality_metrics`.

Usage
-----
    uv run python scripts/checks/compute_hpc_raw_stats.py
    uv run python scripts/checks/compute_hpc_raw_stats.py --animal wilbur
    uv run python scripts/checks/compute_hpc_raw_stats.py --overwrite
    uv run python scripts/checks/compute_hpc_raw_stats.py --limit 5   # quick smoke test
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.decode_quality.hpc_raw_stats import (
    RAW_STAT_COLUMNS,
    TRANSITION_MATRIX_COLUMNS,
    compute_hpc_raw_stats_single,
    extract_discrete_transition_matrix,
)
from continuum_swr_replay.data_loaders.epoch_register import analysed_epochs
from continuum_swr_replay.data_types import DataLoadingPreset
from continuum_swr_replay.decoding.files import decode_paths
from continuum_swr_replay.decoding.identifiers import format_epoch_identifier
from continuum_swr_replay.decoding.results import load_saved_detector
from continuum_swr_replay.paths import processed_data_dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger(__name__)

KEY_COLS = ["animal", "day", "epoch", "nwb_file_name"]
ALL_STAT_COLS = list(RAW_STAT_COLUMNS) + list(TRANSITION_MATRIX_COLUMNS)


def _load_detector(epoch_identifier: str):
    """Load a fitted HPC detector pickle for an epoch, or None if missing."""
    path = decode_paths(epoch_identifier, processed_data_dir()).detector
    if not path.exists():
        return None
    try:
        return load_saved_detector(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load detector %s: %s", path.name, exc)
        return None


def _nan_row() -> dict[str, float]:
    return dict.fromkeys(ALL_STAT_COLS, np.nan)


def _compute_one_session(
    nwb_file_name: str,
    epoch_name: str,
) -> dict[str, float]:
    """Compute all raw stats + transition matrix for one epoch."""
    from continuum_swr_replay.data_loaders.bandit_task import load_data

    row: dict[str, float] = _nan_row()

    try:
        data = load_data(
            nwb_file_name,
            epoch_name,
            include=DataLoadingPreset.MINIMAL,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("load_data failed for %s %s: %s", nwb_file_name, epoch_name, exc)
        return row

    hpc_times = data.get("spike_times", {}).get("HPC", [])
    hpc_feats = data.get("spike_waveform_features", {}).get("HPC", [])

    pos = data.get("position_info")
    if pos is None or len(pos) == 0:
        logger.warning("No position for %s %s", nwb_file_name, epoch_name)
        return row

    epoch_dur_s = float(pos.index[-1] - pos.index[0])

    raw = compute_hpc_raw_stats_single(hpc_times, hpc_feats, epoch_dur_s)
    row.update(raw)

    detector = _load_detector(format_epoch_identifier(nwb_file_name, epoch_name, "HPC"))
    if detector is not None:
        row.update(extract_discrete_transition_matrix(detector))
        # Free JAX/GPU-backed state eagerly; detectors can hold onto device
        # memory that accumulates across 218 sessions otherwise.
        del detector

    return row


def _build_session_list(animal: str | None, limit: int | None) -> pd.DataFrame:
    """Return a DataFrame with one row per epoch, columns animal/day/epoch/nwb.

    The epochs are the epoch register's analysed ones.
    """
    ei = analysed_epochs().reset_index()  # index: (nwb_file_name, interval_list_name)
    df = pd.DataFrame(
        {
            "nwb_file_name": ei["nwb_file_name"],
            "animal": ei["animal"],
            "day": ei["date"].astype(str),
            "epoch": ei["interval_list_name"],
        }
    )
    if animal is not None:
        df = df.loc[df["animal"] == animal]
    df = df.sort_values(["animal", "day", "epoch"]).reset_index(drop=True)
    if limit is not None:
        df = df.head(limit)
    return df


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output",
        type=str,
        default="hpc_raw_stats.csv",
        help="Output CSV filename (saved in paths.processed_data_dir())",
    )
    parser.add_argument("--animal", type=str, default=None, help="Filter to one animal")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute all sessions; default skips rows already in the CSV",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N sessions (for testing)",
    )
    args = parser.parse_args(argv)

    output_path = processed_data_dir() / args.output
    sessions = _build_session_list(args.animal, args.limit)
    logger.info("Processing %d sessions", len(sessions))

    # Resume-from-CSV behavior
    if output_path.exists() and not args.overwrite:
        existing = pd.read_csv(output_path)
        existing["day"] = existing["day"].astype(str)
        already = set(
            zip(existing["animal"], existing["day"], existing["epoch"], strict=True)
        )
        before = len(sessions)
        sessions = sessions.loc[
            ~sessions.apply(
                lambda r: (r["animal"], r["day"], r["epoch"]) in already,
                axis=1,
            )
        ].reset_index(drop=True)
        logger.info(
            "Resuming: %d sessions already in %s, %d remaining",
            before - len(sessions),
            output_path.name,
            len(sessions),
        )
        rows_accum: list[dict] = existing.to_dict("records")
    else:
        rows_accum = []

    for i, sess in sessions.iterrows():
        logger.info(
            "[%d/%d] %s %s %s",
            i + 1,
            len(sessions),
            sess["animal"],
            sess["day"],
            sess["epoch"],
        )
        stats = _compute_one_session(sess["nwb_file_name"], sess["epoch"])
        row = {
            "animal": sess["animal"],
            "day": sess["day"],
            "epoch": sess["epoch"],
            "nwb_file_name": sess["nwb_file_name"],
            **stats,
        }
        rows_accum.append(row)

        # Incremental checkpoint every session
        out_df = pd.DataFrame(rows_accum)
        col_order = KEY_COLS + [c for c in ALL_STAT_COLS if c in out_df.columns]
        out_df = out_df[col_order]
        out_df.to_csv(output_path, index=False)

    logger.info("Saved %d rows to %s", len(rows_accum), output_path)


if __name__ == "__main__":
    main()
