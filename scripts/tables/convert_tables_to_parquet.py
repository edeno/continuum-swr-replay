#!/usr/bin/env python
"""Convert the materialized canonical tables from pickle to parquet.

One-time migration for tables already on disk. Regenerating them from Spyglass
takes hours and would change nothing but the container, so the existing
artifacts are re-encoded in place instead.

Every table is verified after writing -- read back and compared to the source
with ``DataFrame.equals``, plus an explicit dtype and ``attrs`` check -- and the
pickle is only retired once its parquet twin has been proven identical. Retired
pickles move to a ``_pickle_archive`` sibling rather than being deleted, so the
migration is reversible; nothing reads them there, which is the point, since a
pickle left beside its parquet is a stale read waiting to happen.

Backup directories (``_prephase1_backup`` and friends) are deliberately left
alone. They are historical snapshots, and ``read_table`` still falls back to
pickle, so they stay readable exactly as they are.

Usage
-----
    python scripts/tables/convert_tables_to_parquet.py            # dry run
    python scripts/tables/convert_tables_to_parquet.py --apply
"""

from __future__ import annotations

import argparse
import logging
import shutil
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from continuum_swr_replay.paths import (
    contfrag_replay_content_dir,
    replay_content_dir,
    table_path,
)
from continuum_swr_replay.tables.io import parquet_path, read_table, write_table

LOGGER = logging.getLogger(__name__)


def canonical_tables() -> tuple[Path, ...]:
    """The retired pickles of the canonical tables, under the data root."""
    rc, cf = replay_content_dir(), contfrag_replay_content_dir()
    return (
        table_path("event_table", rc, fmt="pkl"),
        table_path("event_arm_table", rc, fmt="pkl"),
        table_path("event_patch_table", rc, fmt="pkl"),
        table_path("event_sequence_table", rc, fmt="pkl"),
        table_path("stop_table", rc, fmt="pkl"),
        table_path("event_onset_table", rc, fmt="pkl"),
        table_path("qvals_table", rc, fmt="pkl"),
        *(
            cf / f"{name}_{source}_HPC.pkl"
            for name in (
                "event_table",
                "event_arm_table",
                "event_patch_table",
                "stop_table",
            )
            for source in ("swr", "mua")
        ),
    )


def convert(path: Path, *, apply: bool) -> tuple[bool, str]:
    """Convert one table, verifying the round-trip before retiring the pickle.

    Returns
    -------
    (ok, message)
        ``ok`` is False if the table was missing or the round-trip differed.
    """
    if not path.exists():
        return True, "absent (skipped)"

    source = pd.read_pickle(path)
    if not apply:
        return (
            True,
            f"{len(source):>9,d} rows, {path.stat().st_size / 1e6:6.1f} MB -> would convert",
        )

    out = write_table(source, path)
    back = read_table(out)

    drift = [
        (c, str(source[c].dtype), str(back[c].dtype))
        for c in source.columns
        if str(source[c].dtype) != str(back[c].dtype)
    ]
    if drift:
        out.unlink()
        return False, f"DTYPE DRIFT {drift[:3]} -- parquet removed, pickle kept"
    if not source.equals(back):
        out.unlink()
        return False, "VALUES DIFFER on read-back -- parquet removed, pickle kept"
    if source.attrs != back.attrs:
        out.unlink()
        return False, "ATTRS DIFFER on read-back -- parquet removed, pickle kept"

    archive = path.parent / "_pickle_archive"
    archive.mkdir(exist_ok=True)
    shutil.move(str(path), str(archive / path.name))
    shrink = 100 * (1 - out.stat().st_size / (archive / path.name).stat().st_size)
    return (
        True,
        f"{len(source):>9,d} rows, {out.stat().st_size / 1e6:6.1f} MB ({shrink:.0f}% smaller), verified",
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="perform the conversion (default is a dry run)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not args.apply:
        LOGGER.info("DRY RUN -- pass --apply to convert\n")

    failures = []
    for path in canonical_tables():
        if args.apply and parquet_path(path).exists() and not path.exists():
            LOGGER.info("%-46s already parquet", path.name)
            continue
        ok, message = convert(path, apply=args.apply)
        LOGGER.info("%-46s %s", path.name, message)
        if not ok:
            failures.append(path)

    if failures:
        raise SystemExit(
            f"\n{len(failures)} table(s) failed verification and were left as pickle: "
            + ", ".join(p.name for p in failures)
        )
    LOGGER.info("\nall tables verified")


if __name__ == "__main__":
    main()
