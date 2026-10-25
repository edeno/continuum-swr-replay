#!/usr/bin/env python
"""Regenerate the canonical tables into a scratch root and diff two roots.

The parity check of a change that must not move any table. ``regen`` rebuilds
every canonical table -- the replay tables (stop, qvals, event, event-arm,
event-patch, event-sequence, event-onset, stop-timebin) and the ContFrag SWR
and MUA tables -- into a new root, with the same builders, flags and order as
the 2026-10-11 rebuild the live tables come from
(``.claude/research/refresh_r13/r13c_report.md``): stop bootstrap -> qvals ->
one replay shard per animal (beside the time-bin build) -> assemble -> ContFrag
SWR and MUA. ``diff`` compares two roots table by table on each table's row
key, exactly (``tables.parity.diff_table`` with ``atol=0``), and reports every
table before failing.

``--sessions`` restricts ``regen`` to a few epochs. Every builder takes its
sessions from the epoch register, so each builder runs with
``epoch_register.REGISTER_PATH`` at a copy of the register in which every other
analysed epoch is marked excluded. Nothing else changes, so the slice runs the
full cohort's code path. Pooled quantities (the event-arm and qvals ``_z``
columns and their standardization references) are pooled over the slice, so a
slice is compared with a slice, not with the live tables.

A root holds ``replay_content/`` and ``contfrag_replay_content/``, the layout of
``Processed-Data`` itself, so the live tables are a valid ``--reference`` for a
full-cohort ``regen``. Decodes are read from ``PROCESSED_DATA_DIR``; nothing is
written outside ``--root``. The behavioral cache entries are copied from the
canonical cache into the root, so a missing entry falls back to Spyglass and is
cached in the root, not in the live cache.

Usage::

    # slice: regenerate three epochs, then diff against the frozen slice
    uv run python scripts/qa/table_parity.py regen \
        --root Processed-Data/_scratch_simplify_20261024/slice_<label> \
        --sessions chimi20200217_06_r3_HPC j1620210712_10_r5_HPC \
                   wilbur20210330_04_r2_HPC
    uv run python scripts/qa/table_parity.py diff \
        --candidate Processed-Data/_scratch_simplify_20261024/slice_<label> \
        --reference Processed-Data/_scratch_simplify_20261024/slice_reference

    # full cohort, diffed against the live tables
    uv run python scripts/qa/table_parity.py regen \
        --root Processed-Data/_scratch_simplify_20261024/full_<label>
    uv run python scripts/qa/table_parity.py diff \
        --candidate Processed-Data/_scratch_simplify_20261024/full_<label> \
        --reference Processed-Data

    # a slice against the live tables restricted to the slice's sessions
    uv run python scripts/qa/table_parity.py diff --restrict \
        --candidate <slice root> --reference Processed-Data
"""

from __future__ import annotations

import argparse
import filecmp
import json
import os
import runpy
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RC = "replay_content"
CF = "contfrag_replay_content"
CACHE = "_session_cache"
#: The ContFrag tables, ``<table>_table_<source>_HPC``, by registry table key.
CONTFRAG_TABLES = tuple(
    (f"{table}_table_{source}_HPC", table)
    for source in ("swr", "mua")
    for table in ("event", "event_arm", "event_patch", "stop")
)
#: Files compared byte for byte, beside the keyed table comparison.
REFERENCE_FILES = (
    f"{RC}/standardization_reference.json",
    f"{RC}/qvals_standardization_reference.json",
)
_PREPARE = "scripts/tables/prepare_replay_content_data.py"
_CONTFRAG = "scripts/tables/prepare_contfrag_replay_content_data.py"


def _slice_register(sessions: list[str], path: Path) -> list[str]:
    """Write the register with every analysed epoch outside `sessions` excluded.

    Returns the animals of `sessions`, in register order.
    """
    from continuum_swr_replay.data_loaders.epoch_register import (
        Reason,
        Status,
        analysed_animals,
        analysed_epoch_ids,
        read_register,
    )
    from continuum_swr_replay.data_loaders.utils import format_epoch_identifier

    register = read_register()
    unknown = sorted(set(sessions) - set(analysed_epoch_ids(register)))
    if unknown:
        raise ValueError(f"not analysed epochs of the register: {unknown}")
    ids = [
        format_epoch_identifier(name, epoch, "HPC") for name, epoch in register.index
    ]
    drop = (register["status"] == Status.ANALYSED) & ~pd.Series(
        ids, index=register.index
    ).isin(sessions)
    register.loc[drop, "status"] = Status.EXCLUDED.value
    register.loc[drop, "reason"] = Reason.MISSING_INPUT.value
    register.loc[drop, "evidence"] = "outside the table-parity slice"
    register.reset_index().to_csv(path, index=False)
    sliced = read_register(path)  # the copy obeys the register's contract
    assert sorted(analysed_epoch_ids(sliced)) == sorted(sessions)
    return list(analysed_animals(sliced))


def _copy_cache(sessions: list[str] | None, root: Path) -> None:
    """Copy the canonical behavioral-cache entries the run reads into `root`."""
    from continuum_swr_replay.paths import REPLAY_CONTENT_DIR
    from continuum_swr_replay.tables.versions import BEHAVIORAL_CONTEXT_CACHE_VERSION

    src = REPLAY_CONTENT_DIR / CACHE / BEHAVIORAL_CONTEXT_CACHE_VERSION
    dst = root / CACHE / BEHAVIORAL_CONTEXT_CACHE_VERSION
    dst.mkdir(parents=True)
    names = (
        [f"{s}.pkl" for s in sessions]
        if sessions
        else sorted(p.name for p in src.glob("*.pkl"))
    )
    for name in names:
        if (src / name).exists():
            shutil.copy2(src / name, dst / name)


def _launch(step: str, script: str, args: list[str], root: Path, register: Path | None):
    """Start one builder as ``_step`` in a child process, logging to ``logs/``."""
    cmd = [sys.executable, str(Path(__file__).resolve()), "_step"]
    if register is not None:
        cmd += ["--register", str(register)]
    cmd += [script, *args]
    log = (root / "logs" / f"{step}.log").open("w")
    log.write(" ".join(cmd) + "\n")
    log.flush()
    proc = subprocess.Popen(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
    return step, proc, time.time()


def _wait(running: list, timings: dict[str, float]) -> None:
    """Wait for every launched step; raise naming the steps that failed."""
    failed, pending = [], list(running)
    while pending:
        time.sleep(1)
        for item in [item for item in pending if item[1].poll() is not None]:
            step, proc, start = item
            pending.remove(item)
            timings[step] = round(time.time() - start, 1)
            print(
                f"{step}: exit {proc.returncode} after {timings[step]:.0f} s",
                flush=True,
            )
            if proc.returncode:
                failed.append(step)
    if failed:
        raise RuntimeError(f"step(s) failed: {failed}; see the root's logs/")


def regen(root: Path, sessions: list[str] | None) -> None:
    """Rebuild every canonical table into `root` (refuses an existing root)."""
    root = root.resolve()
    root.mkdir(parents=True)  # FileExistsError on a used root, by design
    (root / "logs").mkdir()
    register = None
    if sessions:
        register = root / "epoch_register.csv"
        animals = _slice_register(sessions, register)
    else:
        from continuum_swr_replay.data_loaders.epoch_register import analysed_animals

        animals = list(analysed_animals())
    _copy_cache(sessions, root)
    rc, cf, cache = root / RC, root / CF, str(root / CACHE)
    rc.mkdir()
    cf.mkdir()
    timings: dict[str, float] = {}
    t0 = time.time()

    def run(*steps) -> None:
        _wait([_launch(*step, root, register) for step in steps], timings)

    run(
        (
            "stop",
            _PREPARE,
            [
                "--stop-only",
                "--from-cache",
                "--output-dir",
                str(rc),
                "--behavior-cache-dir",
                cache,
            ],
        )
    )
    run(
        (
            "qvals",
            "scripts/behavior/build_qvals_table.py",
            ["--replay-content-dir", str(rc), "--write"],
        )
    )
    run(
        (
            "timebin",
            "scripts/tables/build_timebin_tables.py",
            ["--output-dir", str(rc)],
        ),
        *(
            (
                f"shard_{a}",
                _PREPARE,
                [
                    "--from-cache",
                    "--animal",
                    a,
                    "--output-dir",
                    str(rc / f"shard_{a}"),
                    "--behavior-cache-dir",
                    cache,
                    "--contingency-dir",
                    str(rc),
                ],
            )
            for a in animals
        ),
    )
    run(
        (
            "assemble",
            "scripts/tables/assemble_replay_content_shards.py",
            [
                *(str(rc / f"shard_{a}") for a in animals),
                "--output-dir",
                str(rc),
                "--contingency-dir",
                str(rc),
            ],
        )
    )
    run(
        *(
            (
                f"contfrag_{source}",
                _CONTFRAG,
                [
                    "--event-source",
                    source,
                    "--from-cache",
                    "--output-dir",
                    str(cf),
                    "--replay-content-dir",
                    str(rc),
                    "--behavior-cache-dir",
                    cache,
                ],
            )
            for source in ("SWR", "MUA")
        )
    )
    timings["total"] = round(time.time() - t0, 1)
    (root / "timings.json").write_text(
        json.dumps({"sessions": sessions or "all", "seconds": timings}, indent=2)
    )
    print(f"regenerated -> {root} in {timings['total']:.0f} s", flush=True)


def _session_restriction(candidate: Path):
    """The candidate's sessions and stops, to restrict a full reference to them."""
    from continuum_swr_replay.tables.io import read_table
    from continuum_swr_replay.tables.session_keys import SESSION_COLUMNS

    stops = read_table(candidate / RC / "stop_table")
    columns = list(SESSION_COLUMNS)
    sessions = pd.MultiIndex.from_frame(stops[columns].drop_duplicates())
    stop_ids = set(stops["stop_id"])

    def restrict(frame: pd.DataFrame) -> pd.DataFrame:
        if set(columns) <= set(frame.columns):
            return frame[pd.MultiIndex.from_frame(frame[columns]).isin(sessions)]
        return frame[frame["stop_id"].isin(stop_ids)]  # qvals: no session columns

    return restrict


def diff(candidate: Path, reference: Path, restrict: bool) -> int:
    """Diff every canonical table of two roots; return the number that differ."""
    from continuum_swr_replay.paths import BASENAME_TO_TABLE, CANONICAL_TABLE_BASENAMES
    from continuum_swr_replay.tables.covariate_registry import row_keys
    from continuum_swr_replay.tables.io import read_table
    from continuum_swr_replay.tables.parity import diff_table

    keep = _session_restriction(candidate) if restrict else (lambda frame: frame)
    tables = [
        (f"{RC}/{name}", BASENAME_TO_TABLE[name]) for name in CANONICAL_TABLE_BASENAMES
    ]
    tables += [(f"{CF}/{name}", table) for name, table in CONTFRAG_TABLES]
    n_bad = 0
    for relative, table in tables:
        cand = read_table(candidate / relative)
        ref_all = read_table(reference / relative)
        ref = keep(ref_all).reset_index(drop=True)
        try:
            diff_table(
                cand, ref, keys=list(row_keys(table=table)), label=relative, atol=0.0
            )
        except ValueError as error:
            n_bad += 1
            print(f"DIFFERS  {relative}: {error}", flush=True)
            continue
        # The attrs (schema stamp, SWR source, cohort sums) must match too; a
        # restricted reference's cohort sums are the full cohort's, so there
        # an attrs difference is shown, not counted.
        moved = sorted(
            k
            for k in set(cand.attrs) | set(ref_all.attrs)
            if cand.attrs.get(k) != ref_all.attrs.get(k)
        )
        n_bad += bool(moved) and not restrict
        attrs = f"attrs DIFFER {moved}" if moved else "attrs equal"
        print(f"same     {relative} ({len(cand)} rows, {attrs})", flush=True)
    for relative in REFERENCE_FILES:
        same = filecmp.cmp(candidate / relative, reference / relative, shallow=False)
        print(f"{'same    ' if same else 'DIFFERS '} {relative} (bytes)", flush=True)
        n_bad += not same
    print(f"{n_bad} difference(s)", flush=True)
    return n_bad


def _step(register: str | None, script: str, args: list[str]) -> None:
    """Run one builder script in this process, with the register patched."""
    if register is not None:
        from continuum_swr_replay.data_loaders import epoch_register

        epoch_register.REGISTER_PATH = Path(register)
    sys.argv = [script, *args]
    sys.path.insert(0, str(REPO))
    runpy.run_path(str(REPO / script), run_name="__main__")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p_regen = sub.add_parser(
        "regen", help="rebuild every canonical table into a new root"
    )
    p_regen.add_argument("--root", type=Path, required=True)
    p_regen.add_argument(
        "--sessions",
        nargs="+",
        default=None,
        help="HPC epoch ids (e.g. chimi20200217_06_r3_HPC); default all",
    )
    p_diff = sub.add_parser("diff", help="diff two roots table by table, exactly")
    p_diff.add_argument("--candidate", type=Path, required=True)
    p_diff.add_argument("--reference", type=Path, required=True)
    p_diff.add_argument(
        "--restrict",
        action="store_true",
        help="restrict the reference to the candidate's sessions",
    )
    p_step = sub.add_parser("_step", help="internal: run one builder")
    p_step.add_argument("--register", default=None)
    p_step.add_argument("script")
    p_step.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command == "regen":
        regen(args.root, args.sessions)
    elif args.command == "diff":
        sys.exit(1 if diff(args.candidate, args.reference, args.restrict) else 0)
    else:
        _step(args.register, args.script, args.args)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
