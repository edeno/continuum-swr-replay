"""Measure each analysed epoch's decoder quality: Local-state error, state fractions, calibration.

The inputs of the events summary's decoder-validation values, so the summary
reads them without the decode frames. From each analysed epoch's decode
(read through the SWR source):

- ``sessions.parquet``, one row per epoch:
  ``quality_metrics.session_decoder_quality`` (the Local-state decoding error
  and the decoder state fractions, moving and immobile);
- ``calibration.parquet``, one row per epoch and Local-probability bin:
  ``quality_metrics.compute_calibration_curve``.

Each row carries the epoch id and its decode's sha256. Written to
``Processed-Data/decoder_quality/`` (or ``--out-dir``) with a
``manifest.json``.

Usage::

    uv run python scripts/decoding/measure_decoder_quality.py --n-workers 8 \\
        [--epochs wilbur20210329_02_r1_HPC] [--out-dir <folder>]
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from continuum_swr_replay.analysis.events import analysed_time
from continuum_swr_replay.analysis.hippocampus.quality_metrics import (
    compute_calibration_curve,
    session_decoder_quality,
)
from continuum_swr_replay.data_loaders.epoch_register import analysed_decode_files
from continuum_swr_replay.decoding.files import (
    data_over_time_suffix,
    epoch_identifier_of,
)
from continuum_swr_replay.paths import DECODER_QUALITY_DIR, processed_data_dir
from continuum_swr_replay.runs.io import sha256_file, write_json
from continuum_swr_replay.runs.provenance import described_commit, produced_by


def measure_epoch(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One epoch's session row and calibration rows, with its id and decode hash."""
    sha256 = sha256_file(path)
    dot = analysed_time.read_verified_dot_with_swr_source(path, sha256)
    epoch = {
        "epoch_id": epoch_identifier_of(path, data_over_time_suffix()),
        "decode_sha256": sha256,
    }
    return (
        session_decoder_quality(dot).assign(**epoch),
        compute_calibration_curve(dot).assign(**epoch),
    )


def _write_parquet(table: pd.DataFrame, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    table.to_parquet(tmp, index=False)
    tmp.replace(path)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument(
        "--epochs",
        nargs="+",
        help="measure only these epoch ids (a trial; the events summary refuses "
        "an output that lacks an analysed epoch)",
    )
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args(argv)

    paths = analysed_decode_files(
        processed_data_dir(), data_over_time_suffix(), require_all=True
    )
    if args.epochs:
        wanted = {f"{epoch}{data_over_time_suffix()}" for epoch in args.epochs}
        paths = [path for path in paths if path.name in wanted]
        if len(paths) != len(wanted):
            raise ValueError(f"not all of {args.epochs} are analysed epochs")
    with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
        measured = list(pool.map(measure_epoch, paths))
    sessions = pd.concat([s for s, _ in measured], ignore_index=True)
    calibration = pd.concat([c for _, c in measured], ignore_index=True)

    out_dir = args.out_dir or processed_data_dir() / DECODER_QUALITY_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_parquet(sessions, out_dir / "sessions.parquet")
    _write_parquet(calibration, out_dir / "calibration.parquet")
    write_json(
        out_dir / "manifest.json",
        {
            "produced_by": produced_by(__file__),
            "commit": described_commit(),
            "n_epochs": len(sessions),
            "decodes": dict(
                zip(sessions["epoch_id"], sessions["decode_sha256"], strict=True)
            ),
            "local_error": "time bins with Local_prob above 0.5 outside "
            "excluded time; abs(ahead_behind_distance), cm",
            "state_fractions": "mean state probability over bins faster "
            "(moving) or slower (immobile) than 4 cm/s, outside excluded time",
            "calibration": "median abs(ahead_behind_distance) per 0.1-wide "
            "Local_prob bin, outside excluded time",
        },
        indent=1,
        final_newline=True,
    )
    print(f"wrote {len(sessions)} epochs to {out_dir}")


if __name__ == "__main__":
    main()
