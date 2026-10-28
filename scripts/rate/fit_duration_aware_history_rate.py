#!/usr/bin/env python
"""Duration-aware self- and cross-history of the per-bin event rate.

The shipped per-bin design treats each event as an instant onset: every 15 ms
bin of a stop is at risk, including the bins inside an ongoing event, and
self-/cross-history count past onsets. On simulated data that alone produces a
short-lag "refractory" dip, a self-history burst near 90-120 ms and two-way
cross coupling (``.claude/docs/reviews/2026-10-26-duration-aware-history-
simulation.md``). This runner refits the history on the at-risk design:

- a bin's exposure is the time no event of the response's detector covers (bins
  inside an ongoing event drop out): Kay SWR intervals for the detected-SWR
  population, every non-local event for the three non-local populations;
- history is measured from past events' onsets, their ends, or both.

Per session and population it fits the ``MODEL_CROSS_HISTORY_DURATION_RATE``
rungs on ONE frozen cohort: ``Mnohist_dur``, ``Mself_onset_dur``,
``Mself_offset_dur``, ``Mself_dur`` for every population (``Mself_dur`` is
``M7r_time_histcos_rew`` on the at-risk design plus the offset block), and the
three cross rungs for the two non-local streams (cross = the other stream).
Fixed-block contrasts are ``Mnohist_dur -> Mself_dur`` and ``Mself_dur ->
Mselfcross_dur``; the one-anchor rungs give the onset-vs-offset contrasts. Each
row also carries the lag support (expected onsets per lag) and the onset/offset
canonical correlation, so a kernel is read only where it is identified.

The design otherwise matches ``fit_per_bin_rate_models.py``: same stop cohort
(``--stop-cohort``, default at-well), bins, TMAX, reward basis with the outcome
shift, excluded time and history window. No CV and no GOF.

Writes ``models_per_bin_rate/duration_aware_history_models{suffix}.pkl`` and its
``.provenance.json``; it never touches ``cross_history_models.pkl``.

Usage::

    python scripts/rate/fit_duration_aware_history_rate.py [--workers N]
        [--stop-cohort at_well] [--max-per-animal N]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import warnings
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.events.analysed_time import require_swr_source
from continuum_swr_replay.analysis.events.cohort_selection import (
    STOP_COHORTS,
    select_stop_cohort,
    stop_cohort_suffix,
)
from continuum_swr_replay.analysis.events.swr_source import (
    read_swr_inventory,
    swr_epoch_id_of_session,
    swr_source_name,
)
from continuum_swr_replay.analysis.replay import (
    BIN_S,
    N_HISTORY_BASIS,
    N_HISTORY_LAGS,
    POPULATIONS,
    TMAX,
    aggregate_to_timebins,
    attach_is_swr_to_onsets,
    fit_per_bin_rate_model,
    freeze_per_bin_cohort,
)
from continuum_swr_replay.analysis.replay.covariate_timing import (
    assert_predictor_timing,
)
from continuum_swr_replay.analysis.replay.duration_history_simulation import (
    lag_support,
    onset_offset_collinearity,
)
from continuum_swr_replay.analysis.replay.fit_provenance import (
    record_inputs,
    write_fit_provenance,
)
from continuum_swr_replay.analysis.replay.model_specs import (
    MODEL_CROSS_HISTORY_DURATION_RATE,
    MODEL_CROSS_HISTORY_RATE,
    STOP_COVARIATES,
    longest_history_window_s,
)
from continuum_swr_replay.parameters import SAMPLING_FREQUENCY
from continuum_swr_replay.paths import model_dir, table_path
from continuum_swr_replay.tables.covariate_registry import require_covariate_schema
from continuum_swr_replay.tables.io import read_table
from continuum_swr_replay.tables.session_keys import SESSION_COLUMNS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

LADDER = MODEL_CROSS_HISTORY_DURATION_RATE
SELF_RUNGS = ("Mnohist_dur", "Mself_onset_dur", "Mself_offset_dur", "Mself_dur")
CROSS_RUNGS = ("Mselfcross_onset_dur", "Mselfcross_offset_dur", "Mselfcross_dur")
HISTORY_WINDOW_S = longest_history_window_s(LADDER.values())
END_COLUMN = "t_end_since_stop_onset"
#: The decode's sample (2 ms at 500 Hz).
DECODE_SAMPLE_S = 1 / SAMPLING_FREQUENCY
#: An SWR onset matches its Kay event when their starts differ by at most this
#: (one 2 ms sample, with rounding).
SWR_START_TOLERANCE_S = 0.0025

#: Population -> (event type whose events set the at-risk time, cross population).
POPULATION_PLAN: dict[str, tuple[str, str | None]] = {
    "swr": ("SWR", None),
    "non_local": ("non_local", None),
    "non_local_swr": ("non_local", "non_local_no_swr"),
    "non_local_no_swr": ("non_local", "non_local_swr"),
}


def attach_event_ends(onsets: pd.DataFrame, event_table: pd.DataFrame) -> pd.DataFrame:
    """Add ``t_end_since_stop_onset`` to the non-local and SWR onset rows.

    Non-local: the matched ``event_table`` event's duration (``event_end_s -
    event_start_s``) added to its onset, matched on (session, stop, onset time
    == ``event_start_s``) exactly. SWR: the Kay inventory event whose start is
    nearest the onset (within :data:`SWR_START_TOLERANCE_S`). MUA rows keep NaN.

    Raises
    ------
    ValueError
        If a non-local onset has no event, or an SWR onset no Kay event within
        tolerance.
    """
    out = onsets.copy()
    out[END_COLUMN] = np.nan
    keys = [*SESSION_COLUMNS, "stop_id"]
    nl = out["event_type"] == "non_local"
    ev = event_table[[*keys, "event_start_s", "event_end_s"]].rename(
        columns={"event_start_s": "onset_time"}
    )
    merged = (
        out.loc[nl, [*keys, "onset_time"]]
        .reset_index()
        .merge(ev, on=[*keys, "onset_time"], how="left", validate="one_to_one")
        .set_index("index")
    )
    if merged["event_end_s"].isna().any():
        raise ValueError(
            f"{int(merged['event_end_s'].isna().sum())} non-local onsets have no "
            "event_table event"
        )
    out.loc[merged.index, END_COLUMN] = (
        out.loc[merged.index, "t_since_stop_onset"].to_numpy()
        + (merged["event_end_s"] - merged["onset_time"]).to_numpy()
    )

    swr = out["event_type"] == "SWR"
    for session, g in out.loc[swr].groupby(list(SESSION_COLUMNS), sort=False):
        inv = read_swr_inventory(swr_epoch_id_of_session(*session))
        starts = inv["start_time"].to_numpy(dtype=float)
        ends = inv["end_time"].to_numpy(dtype=float)
        t = g["onset_time"].to_numpy(dtype=float)
        right = np.clip(np.searchsorted(starts, t), 0, len(starts) - 1)
        left = np.clip(right - 1, 0, len(starts) - 1)
        pick = np.where(
            np.abs(starts[left] - t) < np.abs(starts[right] - t), left, right
        )
        err = np.abs(starts[pick] - t)
        if (err > SWR_START_TOLERANCE_S).any():
            raise ValueError(
                f"{session}: {int((err > SWR_START_TOLERANCE_S).sum())} SWR onsets "
                f"have no Kay event within {SWR_START_TOLERANCE_S} s"
            )
        out.loc[g.index, END_COLUMN] = (
            g["t_since_stop_onset"].to_numpy() + ends[pick] - starts[pick]
        )
    return out


def build_design(
    keys: tuple,
    stops_g: pd.DataFrame,
    onsets_g: pd.DataFrame,
    outcome: str,
    *,
    duration_aware: bool = True,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """One session's frozen design for ``outcome`` and the rungs it carries.

    ``duration_aware=False`` builds the shipped design on the same arguments
    (every bin at risk, onset history only), for a like-for-like comparison;
    its frame still carries the cross block for the non-local streams.

    Raises
    ------
    ValueError
        If the freeze drops a stop (every required column is complete by
        construction).
    """
    event_type, cross_pop = POPULATION_PLAN[outcome]
    response = POPULATIONS[outcome](onsets_g)
    # An event occupies its samples start..end (end is its last sample); the
    # sample in which it starts is the sample it was at risk in. So it covers
    # [start + one sample, end + one sample): otherwise an onset on a 15 ms bin
    # edge would leave its own bin no at-risk time.
    at_risk = onsets_g.loc[
        onsets_g["event_type"] == event_type,
        ["stop_id", "t_since_stop_onset", END_COLUMN],
    ].copy()
    at_risk[["t_since_stop_onset", END_COLUMN]] += DECODE_SAMPLE_S
    cross_kwargs: dict = {}
    rungs = SELF_RUNGS
    if cross_pop is not None:
        cross_kwargs = {
            "cross_onset_times": POPULATIONS[cross_pop](onsets_g),
            "cross_history_lags": N_HISTORY_LAGS,
            "cross_history_basis": N_HISTORY_BASIS,
        }
        rungs = SELF_RUNGS + CROSS_RUNGS
    design = aggregate_to_timebins(
        stops_g,
        response,
        bin_s=BIN_S,
        tmax=TMAX,
        reward_basis=True,
        self_history_lags=N_HISTORY_LAGS,
        self_history_basis=N_HISTORY_BASIS,
        stop_covariates=STOP_COVARIATES,
        history_window_s=HISTORY_WINDOW_S,
        arrival_clocks=False,
        at_risk_events=at_risk if duration_aware else None,
        offset_history=duration_aware,
        **cross_kwargs,
    )
    specs = (
        [LADDER[name] for name in rungs]
        if duration_aware
        else [
            MODEL_CROSS_HISTORY_RATE["Mselfcross" if cross_pop is not None else "Mself"]
        ]
    )
    design, freeze = freeze_per_bin_cohort(design, specs)
    if freeze["n_stops_before"] != freeze["n_stops_after"]:
        raise ValueError(
            f"{keys}/{outcome}: freezing dropped "
            f"{freeze['n_stops_before'] - freeze['n_stops_after']} stops"
        )
    return design, rungs


def _fit_one_session(
    keys: tuple, stops_g: pd.DataFrame, onsets_g: pd.DataFrame
) -> tuple[tuple, list[dict], float]:
    """Fit every population's rungs for one session. Top-level for workers."""
    warnings.filterwarnings("ignore")
    t0 = time.perf_counter()
    rows: list[dict] = []
    for outcome, (event_type, cross_pop) in POPULATION_PLAN.items():
        design, rungs = build_design(keys, stops_g, onsets_g, outcome)
        specs = [LADDER[name] for name in rungs]
        shared = {
            "outcome": outcome,
            **dict(zip(SESSION_COLUMNS, keys, strict=True)),
            "at_risk_event_type": event_type,
            "cross_population": cross_pop,
            "n_bins_not_at_risk": design.attrs.get("n_bins_not_at_risk"),
            "n_onsets_not_at_risk": design.attrs.get("n_onsets_not_at_risk"),
            "n_bins_excluded": design.attrs.get("n_bins_excluded", 0),
            **{
                f"support_{block}": support.round(3).tolist()
                for block, support in lag_support(design).items()
            },
            **onset_offset_collinearity(design),
        }
        for name, spec in zip(rungs, specs, strict=True):
            fit = fit_per_bin_rate_model(design, spec)
            row = {**shared, "model": name, "fit_ok": fit is not None}
            if fit is not None:
                row.update({k: v for k, v in fit.items() if k != "predicted"})
            rows.append(row)
    return keys, rows, time.perf_counter() - t0


def fit_duration_aware_history(
    stop_table: pd.DataFrame, onsets: pd.DataFrame, *, workers: int = 1
) -> pd.DataFrame:
    """One row per (session, population, rung)."""
    onset_groups = dict(tuple(onsets.groupby(list(SESSION_COLUMNS), sort=False)))
    items = [
        (keys, stops_g, onset_groups[keys])
        for keys, stops_g in stop_table.groupby(list(SESSION_COLUMNS), sort=False)
        if keys in onset_groups
    ]
    rows: list[dict] = []
    done = 0

    def _collect(keys, r, elapsed):
        nonlocal done
        done += 1
        rows.extend(r)
        logger.info(
            "  [%d/%d] %s: %d fits in %.1fs",
            done,
            len(items),
            "_".join(map(str, keys)),
            len(r),
            elapsed,
        )

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_fit_one_session, *item) for item in items]
            for fut in as_completed(futures):
                _collect(*fut.result())
    else:
        for item in items:
            _collect(*_fit_one_session(*item))
    return pd.DataFrame(rows)


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-per-animal", type=int, default=None, help="smoke subset")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--stop-cohort", choices=STOP_COHORTS, default="at_well")
    argv = sys.argv[1:] if argv is None else list(argv)
    args = p.parse_args(argv)

    for spec in LADDER.values():
        assert_predictor_timing(c for c in spec.design_columns() if c != "count")

    inputs = {
        name: table_path(name)
        for name in ("stop_table", "event_onset_table", "event_table")
    }
    stop = read_table(inputs["stop_table"])
    onset_table = read_table(inputs["event_onset_table"])
    event_table = read_table(inputs["event_table"])
    require_covariate_schema(stop, table="stop")
    require_covariate_schema(onset_table, table="event_onset")
    require_covariate_schema(event_table, table="event")
    for name, frame in (
        ("stop_table", stop),
        ("event_onset_table", onset_table),
        ("event_table", event_table),
    ):
        require_swr_source(frame, what=str(inputs[name]))
    fit_inputs = record_inputs(inputs)
    onsets = attach_event_ends(
        attach_is_swr_to_onsets(onset_table, event_table), event_table
    )

    stop, cohort_report = select_stop_cohort(stop, cohort=args.stop_cohort)
    logger.info(
        "stop cohort = %s: %d -> %d stops",
        cohort_report["cohort"],
        cohort_report["n_stops_before"],
        cohort_report["n_stops_after"],
    )
    if args.max_per_animal is not None:
        keep = (
            stop[list(SESSION_COLUMNS)]
            .drop_duplicates()
            .groupby("animal")
            .head(args.max_per_animal)
        )
        stop = stop.merge(keep, on=list(SESSION_COLUMNS))
    n_sessions = stop.groupby(list(SESSION_COLUMNS)).ngroups
    logger.info(
        "Fitting %s over %d sessions with %d worker(s)",
        list(LADDER),
        n_sessions,
        args.workers,
    )
    t0 = time.perf_counter()
    models = fit_duration_aware_history(stop, onsets, workers=args.workers)
    models["stop_cohort"] = args.stop_cohort

    out_dir = model_dir("per_bin_rate")
    out_dir.mkdir(parents=True, exist_ok=True)
    smoke = args.max_per_animal is not None
    out = out_dir / (
        f"duration_aware_history_models{stop_cohort_suffix(args.stop_cohort, smoke=smoke)}.pkl"
    )
    models.to_pickle(out)
    write_fit_provenance(
        out,
        fit_inputs,
        "poisson_glm",
        [sys.argv[0], *argv],
        extra={
            "stop_cohort": args.stop_cohort,
            "swr_source": swr_source_name(),
            "design": "at-risk time + onset/offset history (duration-aware)",
        },
    )
    logger.info(
        "Saved %d rows (%d sessions) to %s in %.1fs",
        len(models),
        n_sessions,
        out,
        time.perf_counter() - t0,
    )


if __name__ == "__main__":
    main()
