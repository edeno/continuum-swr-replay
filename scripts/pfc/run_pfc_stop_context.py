#!/usr/bin/env python3
"""Stop-level context of the mPFC replay response, pooled across sessions (Part 4).

Two steps (the model and its estimators are described in
``analysis/pfc/stop_context.py``):

``assemble`` builds, per epoch of the PFC cohort (the analysed epochs less the
ones ``pfc_cohort_mPFC.json`` leaves out; each must have a canonical mPFC
bundle), the event-level table (unit x replay event: post-onset count, the
stop's event-free baseline, the stop's reward outcome and next-trip patch
switch, overlapping events) from the canonical Figure 4 session assembly, and
caches it as ``<out>/sessions/<identifier>.parquet`` with a provenance JSON
beside it. A cached session is reused only when its provenance records the
current canonical event and stop tables. CPU only, about 2 min per session;
``--workers`` runs sessions in parallel.

``fit`` reads the cached tables and the canonical bundles (unit response sign
and held-out p-value per class) and fits, per class x sign x unit set, the
pooled Poisson model with per-unit intercepts and the two stop context terms
plus overlap, by both estimators (conditional: the stop baseline estimated
jointly; plugin: plugged in as an offset), with standard errors clustered by
recording day. A class's units are chosen by that class's own peak sign and
held-out p-value (:func:`select_units`). Writes ``<out>/results.csv`` and
``<out>/results.provenance.json``.

Examples
--------
One session::

    JAX_PLATFORMS=cpu uv run python scripts/pfc/run_pfc_stop_context.py assemble \
        --animal j16 --date 20210710 --epoch 08_r4

All sessions, four processes, then the fit::

    JAX_PLATFORMS=cpu uv run python scripts/pfc/run_pfc_stop_context.py assemble --workers 4
    JAX_PLATFORMS=cpu uv run python scripts/pfc/run_pfc_stop_context.py fit
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import json
import multiprocessing
import sys
import time
import traceback
from argparse import ArgumentParser, Namespace
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.pfc.load_population import (
    CALLED_ALPHA,
    HELDOUT_P_COLUMNS,
)
from continuum_swr_replay.analysis.pfc.population_statistics import called_units
from continuum_swr_replay.analysis.pfc.stop_context import (
    ESTIMATORS,
    STOP_CONTEXT_CLASSES,
    model_frame,
    summarize_stop_context,
)
from continuum_swr_replay.decoding.identifiers import format_epoch_identifier
from continuum_swr_replay.paths import (
    processed_data_dir,
    replay_content_dir,
    table_path,
)
from continuum_swr_replay.runs.io import json_ready, sha256_file, write_json
from continuum_swr_replay.runs.provenance import git_commit, git_dirty
from continuum_swr_replay.tables.io import read_table, write_table

REPO = Path(__file__).resolve().parents[2]
BRAIN_AREA = "mPFC"
SIGNS = ("increase", "decrease")
UNIT_SETS = ("called", "all")


def default_output_dir() -> Path:
    """``pfc_response_followups/stop_context`` under the processed-data root."""
    return processed_data_dir() / "pfc_response_followups" / "stop_context"


def bundle_path(nwb_file_name: str, epoch_name: str) -> Path:
    """The session's canonical mPFC modulation bundle."""
    identifier = format_epoch_identifier(nwb_file_name, epoch_name, BRAIN_AREA)
    return processed_data_dir() / f"{identifier}_modulation.pkl"


def pfc_left_out() -> set[tuple[str, str]]:
    """The analysed epochs the PFC cohort record leaves out, as (file, epoch)."""
    record = json.loads(
        (processed_data_dir() / f"pfc_cohort_{BRAIN_AREA}.json").read_text()
    )
    return {tuple(key.split(" ")) for key in record["left_out"]}


def selected_sessions(args: Namespace) -> list[tuple[str, str]]:
    """The PFC cohort's epochs, after the CLI filters.

    Raises
    ------
    FileNotFoundError
        If a cohort epoch has no canonical mPFC bundle.
    """
    from continuum_swr_replay.analysis.pfc.runner_support import filter_epoch_info
    from continuum_swr_replay.data_loaders.epoch_register import analysed_epochs

    epochs = filter_epoch_info(
        analysed_epochs(), animal=args.animal, date=args.date, epoch=args.epoch
    )
    left_out = pfc_left_out()
    keys = [(index[0], index[1]) for index in epochs.index]
    cohort = [key for key in keys if key not in left_out]
    missing = [key for key in cohort if not bundle_path(*key).exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} PFC-cohort epoch(s) have no canonical {BRAIN_AREA} "
            f"bundle: {missing}"
        )
    return cohort


def current_table_hashes() -> dict[str, str]:
    """The SHA-256 of the canonical event and stop tables, by the names a
    session's provenance records them under (``input_provenance``)."""
    return {
        "canonical_event_table_sha256": sha256_file(
            table_path("event_table", replay_content_dir())
        ),
        "stop_table_sha256": sha256_file(
            table_path("stop_table", replay_content_dir())
        ),
    }


def cached_is_current(provenance_path: Path, current: dict[str, str]) -> bool:
    """Whether a cached session exists and was assembled from ``current`` tables."""
    if not provenance_path.exists():
        return False
    recorded = json.loads(provenance_path.read_text())["input_provenance"]
    return all(recorded.get(name) == sha for name, sha in current.items())


def _assemble_one(
    nwb_file_name: str, epoch_name: str, output_dir: Path
) -> tuple[str, float, int]:
    """Assemble and write one session; returns its identifier, seconds, rows."""
    from continuum_swr_replay.analysis.pfc.stop_context import (
        prepare_session_stop_context,
    )

    start = time.perf_counter()
    identifier = format_epoch_identifier(nwb_file_name, epoch_name, BRAIN_AREA)
    table, report = prepare_session_stop_context(
        nwb_file_name,
        epoch_name,
        brain_area=BRAIN_AREA,
        bundle_path=bundle_path(nwb_file_name, epoch_name),
    )
    for column in ("nwb_file_name", "epoch", "animal", "event_class"):
        table[column] = table[column].astype("category")
    out = write_table(table, output_dir / "sessions" / f"{identifier}.parquet")
    seconds = time.perf_counter() - start
    write_json(
        output_dir / "sessions" / f"{identifier}.provenance.json",
        json_ready(
            {
                "nwb_file_name": nwb_file_name,
                "epoch": epoch_name,
                "brain_area": BRAIN_AREA,
                "commit": git_commit(REPO),
                "dirty": git_dirty(REPO),
                "table": str(out),
                "table_sha256": sha256_file(out),
                "n_rows": int(len(table)),
                "runtime_s": seconds,
                **report,
            }
        ),
        indent=1,
        default=str,
    )
    return identifier, seconds, int(len(table))


def assemble(args: Namespace) -> None:
    """Assemble every selected session not yet cached (or all, ``--overwrite``)."""
    output_dir = args.output_dir
    sessions = selected_sessions(args)
    if args.max_sessions is not None:
        sessions = sessions[: args.max_sessions]
    current = current_table_hashes()
    todo = [
        key
        for key in sessions
        if args.overwrite
        or not cached_is_current(
            output_dir
            / "sessions"
            / f"{format_epoch_identifier(*key, BRAIN_AREA)}.provenance.json",
            current,
        )
    ]
    print(
        f"{len(sessions)} session(s) selected, {len(todo)} to assemble, "
        f"{args.workers} worker(s); output {output_dir}",
        flush=True,
    )
    failures: list[str] = []
    started = time.perf_counter()
    if args.workers <= 1:
        for key in todo:
            try:
                identifier, seconds, n_rows = _assemble_one(*key, output_dir)
                print(f"done {identifier}: {n_rows} rows, {seconds:.0f} s", flush=True)
            except Exception:
                failures.append(f"{key}: {traceback.format_exc()}")
                print(f"FAILED {key}\n{traceback.format_exc()}", flush=True)
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(args.workers, mp_context=context) as pool:
            futures = {
                pool.submit(_assemble_one, *key, output_dir): key for key in todo
            }
            for future in as_completed(futures):
                key = futures[future]
                try:
                    identifier, seconds, n_rows = future.result()
                    print(
                        f"done {identifier}: {n_rows} rows, {seconds:.0f} s",
                        flush=True,
                    )
                except Exception:
                    failures.append(f"{key}: {traceback.format_exc()}")
                    print(f"FAILED {key}\n{traceback.format_exc()}", flush=True)
    print(
        f"assembled {len(todo) - len(failures)} session(s) in "
        f"{time.perf_counter() - started:.0f} s; {len(failures)} failed",
        flush=True,
    )
    if failures:
        raise RuntimeError("assembly failures:\n" + "\n".join(failures))


def unit_table(sessions: list[tuple[str, str]]) -> pd.DataFrame:
    """Per unit of every session: response sign and called flag per class.

    Returns
    -------
    pd.DataFrame
        ``nwb_file_name``, ``epoch``, ``unit`` (bundle row) and, per class,
        ``<class>: peak_log_modulation`` and ``<class>: p_heldout``.
    """
    frames = []
    for nwb_file_name, epoch_name in sessions:
        bundle = pd.read_pickle(bundle_path(nwb_file_name, epoch_name))
        columns = {
            column: bundle[column].to_numpy(dtype=float)
            for event_class in STOP_CONTEXT_CLASSES
            for column in (
                f"{event_class}: peak_log_modulation",
                HELDOUT_P_COLUMNS[event_class],
            )
        }
        frames.append(
            pd.DataFrame(
                {
                    "nwb_file_name": nwb_file_name,
                    "epoch": epoch_name,
                    "unit": np.arange(len(bundle)),
                    **columns,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def select_units(
    units: pd.DataFrame, event_class: str, sign: str, unit_set: str
) -> np.ndarray:
    """Which units one fit uses: their peak for ``event_class`` has ``sign``
    and, for ``unit_set == "called"``, they are called for ``event_class`` at
    :data:`~continuum_swr_replay.analysis.pfc.load_population.CALLED_ALPHA`
    (:func:`~continuum_swr_replay.analysis.pfc.population_statistics.called_units`).
    Another class's peak or p never enters.

    Returns
    -------
    np.ndarray of bool, shape (n_units,)
    """
    peak = units[f"{event_class}: peak_log_modulation"].to_numpy(dtype=float)
    has_sign = peak > 0 if sign == "increase" else peak < 0
    if unit_set == "all":
        return has_sign
    return has_sign & called_units(units, event_class, CALLED_ALPHA)


def fit(args: Namespace) -> None:
    """Fit every class x sign x unit set x estimator and write the results."""
    output_dir = args.output_dir
    started = time.perf_counter()
    sessions = selected_sessions(args)
    current = current_table_hashes()
    paths = []
    for key in sessions:
        identifier = format_epoch_identifier(*key, BRAIN_AREA)
        path = output_dir / "sessions" / f"{identifier}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing; run assemble first")
        if not cached_is_current(path.with_suffix(".provenance.json"), current):
            raise ValueError(f"{path} was assembled from other tables; reassemble")
        paths.append(path)
    table = pd.concat([read_table(path) for path in paths], ignore_index=True)
    for column in ("nwb_file_name", "epoch", "animal", "event_class"):
        table[column] = table[column].astype(str)
    units = unit_table(sessions)
    # The bundle rows are the units counted: same number per session.
    counted = table.groupby(["nwb_file_name", "epoch"])["unit"].max() + 1
    bundled = units.groupby(["nwb_file_name", "epoch"])["unit"].size()
    mismatch = counted.index[counted.to_numpy() > bundled.loc[counted.index].to_numpy()]
    if len(mismatch):
        raise ValueError(f"units counted beyond the bundle in {list(mismatch)}")
    print(
        f"{len(sessions)} sessions, {len(table)} unit-event rows, {len(units)} units",
        flush=True,
    )
    stops = table.drop_duplicates(["nwb_file_name", "epoch", "stop_idx"])
    known = stops[["rewarded", "next_trip_switches_patch"]].dropna()
    print(
        "stops with an analysed event: "
        f"{len(stops)}; context correlation (rewarded, next switch) over "
        f"{len(known)} known: {known.corr().iloc[0, 1]:.3f}",
        flush=True,
    )
    rows = []
    for event_class in STOP_CONTEXT_CLASSES:
        peak = units[f"{event_class}: peak_log_modulation"].to_numpy()
        n_no_sign = int((~np.isfinite(peak) | (peak == 0)).sum())
        for sign in SIGNS:
            for unit_set in UNIT_SETS:
                chosen = select_units(units, event_class, sign, unit_set)
                frame, drops = model_frame(
                    table, event_class=event_class, units=units.loc[chosen]
                )
                for estimator in ESTIMATORS:
                    summary = summarize_stop_context(frame, estimator=estimator)
                    summary.insert(0, "estimator", estimator)
                    summary.insert(0, "unit_set", unit_set)
                    summary.insert(0, "sign", sign)
                    summary.insert(0, "event_class", event_class)
                    summary["n_units_selected"] = int(chosen.sum())
                    summary["n_units_no_sign"] = n_no_sign
                    for name, value in drops.items():
                        summary[name] = value
                    rows.append(summary)
                    pooled = summary.set_index("term")
                    print(
                        f"{event_class} {sign} {unit_set} {estimator}: "
                        + ", ".join(
                            f"{term} {pooled.loc[term, 'exp_coef']:.3f} "
                            f"[{pooled.loc[term, 'ci_low']:.3f}, "
                            f"{pooled.loc[term, 'ci_high']:.3f}]"
                            for term in pooled.index
                        )
                        + f" (units {int(pooled['n_units'].iloc[0])}, "
                        f"events {int(pooled['n_events'].iloc[0])})",
                        flush=True,
                    )
    results = pd.concat(rows, ignore_index=True)
    results_path = output_dir / "results.csv"
    results.to_csv(results_path, index=False)
    write_json(
        output_dir / "results.provenance.json",
        json_ready(
            {
                "commit": git_commit(REPO),
                "dirty": git_dirty(REPO),
                "results": str(results_path),
                "results_sha256": sha256_file(results_path),
                "called_p": CALLED_ALPHA,
                "se_cluster": "recording day (nwb_file_name); CR1 factor, t "
                "intervals with G - 1 degrees of freedom",
                "input_tables": current,
                "session_tables": {str(path): sha256_file(path) for path in paths},
                "bundles": {
                    str(bundle_path(*key)): sha256_file(bundle_path(*key))
                    for key in sessions
                },
                "runtime_s": time.perf_counter() - started,
            }
        ),
        indent=1,
    )
    print(f"wrote {results_path} in {time.perf_counter() - started:.0f} s", flush=True)


def get_command_line_arguments() -> Namespace:
    """Parse the CLI."""
    parser = ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("step", choices=["assemble", "fit"])
    parser.add_argument("--animal")
    parser.add_argument("--date", type=int)
    parser.add_argument("--epoch")
    parser.add_argument("--max-sessions", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = default_output_dir()
    return args


def main() -> None:
    """Run the requested step."""
    args = get_command_line_arguments()
    if args.step == "assemble":
        assemble(args)
    else:
        fit(args)


if __name__ == "__main__":
    sys.exit(main())
