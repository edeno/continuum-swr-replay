"""The visit rate model's scenario list for the simulation engine.

Every number a run of this list writes is a simulation number: the counts are
drawn from a planted rate on the model's real rows (``regression.simulation``);
the model's real outcome is never fitted.

``scripts/checks/run_simulations.py --model visit_rate --config CONFIG`` runs
it. :func:`simulation` builds one run for one kind of event and one clock
penalty target:

- the rows: model A's rows of the cohort step (``visit_rate.cohort_rows``) of
  the analyzed sessions of the canonical stop and event tables, with
  ``sessions_per_animal`` sessions of each animal spread over its sessions
  (:func:`spread_sessions`), or all; the quick configuration uses a synthetic
  cohort instead (:func:`synthetic_visit_rows`), so a test runs it without the
  data;
- the planted curves (:func:`shaped_clocks`): the arrival and leaving curves
  of the model-free event rates stored by
  ``scripts/replay_models/rate/describe_visit_clocks.py``
  (``models_convention/descriptive/visit_clock_curves.parquet``, read only),
  pooled over the animals and smoothed (:func:`smoothed_log_rate`); each
  session's expected total of events is its real number of events of the kind
  (the only use of the real counts but the rate table);
- the scenarios (:data:`SCENARIOS`, from ``visit_rate_scenarios``), each an
  outside scenario of the engine: its expected counts and its planted values;
- the models: model A with the weak penalty at each prior SD of
  :data:`PRIOR_SDS`, all fitted to the same data sets. The estimands are Q1 with the declared reference
  window (0 to 1 s), Q1b (the same with 1 to 2 s), Q2 and Q7a;
- the clocks' log penalties: each clock calibrated on the run's rows by
  ``visit_rate.calibrate_log_penalty`` to the target median edf, on one data
  set of the calibration scenario (:data:`CALIBRATION_SEED`; the rows' real
  counts are not used), in two rounds (each clock with the others at the
  previous round's values; the first round starts from the module's
  provisional values). The values are kept in ``calibration_path`` so that a
  rerun reads them. A run with no target (the quick configuration) fits at
  the provisional values with ``allow_provisional="simulation"``.

The planted values (each scenario's truth for each model):

- the session level: each session's truth of the coefficient form
  (``Planted.pooled``, the Poisson projection onto that form) reduced as the
  estimates are, session -> recording day -> animal, the animals' mean;
- beside them (``table``): the omission side's truth of Q2 (the time-varying
  form's), reduced as at the session level.

A scenario with no planted ratio (the later onset) has Q2 NaN in every truth.

Which truths depend on the real outcome. On the real cohort, every scenario's
planted clock curves are shaped by the stored model-free curves of the real
event starts, and each session's expected total is its real number of events;
those set the curves and the baseline, not a reward effect. The reward effect
of "no effect" (zero) and of "constant" (:data:`LOG_RATIO`) does not depend
on the real outcome, nor do the fading, differing, later-onset and drop
effects. The rate table's does (:data:`TRUTH_FROM_REAL_COUNTS`): its planted
rate is a tabulation of the rows' REAL counts by outcome and both clocks
(:func:`_rate_table`), and its "planted" Q1, Q2 and Q7a are an unpenalized
Poisson projection of the model's coefficient form onto that tabulation
(``visit_rate_scenarios.rate_table_truth``): in substance the real data's
pooled model-free values, not a chosen truth. They are not a result and are
not to be quoted; the table carries ``truth_from_real_counts`` on its rows
and the run's ``attrs`` the same note.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.regression.likelihoods import Poisson
from continuum_swr_replay.analysis.regression.newton import (
    CONVERGED,
)
from continuum_swr_replay.analysis.regression.pspline import PSplineBasis
from continuum_swr_replay.analysis.regression.sessions import SessionKey, fit_sessions
from continuum_swr_replay.analysis.regression.simulation import (
    Model,
    OutsideScenario,
    Simulation,
    SimulationResult,
)
from continuum_swr_replay.analysis.regression.terms import (
    Offset,
    Smooth,
    build_design,
)
from continuum_swr_replay.analysis.shared_statistics import animal_day_session_values
from continuum_swr_replay.runs.threads import pool_context

from ..visit_design import REWARD_LAG_COLUMNS
from . import visit_rate as vr
from . import visit_rate_scenarios as sc

__all__ = [
    "CALIBRATION_SEED",
    "DATA_SEED",
    "PRIOR_SDS",
    "SCENARIOS",
    "TARGETS",
    "TRUTH_FROM_REAL_COUNTS",
    "Config",
    "calibrate_clocks",
    "real_cohort",
    "shaped_clocks",
    "simulation",
    "smoothed_log_rate",
    "spread_sessions",
    "table",
]

#: The prior SDs of the weak penalty at the session level.
PRIOR_SDS: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0)
#: The clocks' target median edfs.
TARGETS: tuple[float, ...] = (6.0, 10.0)
#: The run's seed (``regression.simulation.data_seed``).
DATA_SEED = 20261124
#: The seed of the calibration data set.
CALIBRATION_SEED = 999_999
#: The seed of the session and animal log ratios of ``differing_ratio``.
DIFFERING_SEED = 20261123
#: The rate ratio after a reward of the constant scenario, per kind.
LOG_RATIO: dict[str, float] = {"SWR": float(np.log(2.5)), "No SWR": float(np.log(1.3))}
#: The fading ratio: constant to 6 s since arrival, linear to 1 at 10 s.
FADE_S = (6.0, 10.0)
#: The later onset after an omission (s).
DELAY_S = 0.5
#: The drops before leaving and before small movements: log rate and width (s).
DROP = 1.0
DROP_WIDTH_S = 0.3
#: The animal and session SDs of the differing log ratios.
DIFFERING_SD = (0.2, 0.3)
#: The rate table's cells (s) and its shrinkage and floor (``rate_table_from_bins``).
RATE_TABLE_ARRIVAL_EDGES = np.append(np.arange(0.0, 10.0 + 1e-9, 0.25), np.inf)
RATE_TABLE_LEAVING_EDGES = np.array([0.0, 0.25, 0.5, 1.0, 2.0, 3.0, np.inf])
RATE_TABLE_SHRINK_S = 50.0
RATE_TABLE_FLOOR = 5e-4
#: The smoothing of the stored model-free curves: knot spacing (s) and log
#: penalty of each clock's P-spline.
SMOOTHING = {vr.ARRIVAL: (0.5, 4.0), vr.LEAVING: (0.1, 4.0)}
#: The stored model-free curves, under the replay-content root.
DESCRIPTIVE = Path("models_convention") / "descriptive" / "visit_clock_curves.parquet"

#: The scenarios whose planted values are read from the real counts (the
#: module docstring): their rows of :func:`table` carry
#: ``truth_from_real_counts``.
TRUTH_FROM_REAL_COUNTS: tuple[str, ...] = ("rate_table",)

#: The scenarios of each kind, in the list's order (the order fixes the seeds).
SCENARIOS: dict[str, tuple[str, ...]] = {
    "SWR": (
        "no_effect",
        "constant_ratio",
        "fading_ratio",
        "rate_table",
        "differing_ratio",
        "later_onset",
        "drop_before_leaving",
        "drop_before_small_movements",
    ),
    "No SWR": (
        "no_effect",
        "constant_ratio",
        "rate_table",
        "drop_before_small_movements",
    ),
}

#: The quick configuration's synthetic cohort: animals, sessions per animal,
#: visits per session and events per session.
QUICK_COHORT = (5, 2, 40, 150.0)
#: The quick configuration's planted curves: a rise after arrival and a drop
#: before leaving.
QUICK_CLOCKS = sc.PlantedClocks(
    arrival=sc.interpolated_curve(
        np.array([0.0, 1.0, 3.0]), np.array([-2.0, 0.0, 0.5])
    ),
    leaving=sc.interpolated_curve(
        np.array([0.0, 0.5, 1.0]), np.array([-1.0, -0.2, 0.0])
    ),
)


@dataclass(frozen=True)
class Config:
    """One run of the list.

    Attributes
    ----------
    kind : {"SWR", "No SWR"}
    target_edf : float or None
        The clocks' target median edf; None fits at the provisional values.
    sessions_per_animal : int or None
        Sessions of each animal (:func:`spread_sessions`); None takes all.
    n_data_sets, n_draws : int
    prior_sds : tuple of float
        The weak penalty's prior SDs of the session-level models.
    calibration_path : str or None
        Where the calibrated log penalties are kept (read when it exists).
    quick : bool
        The synthetic cohort (:data:`QUICK_COHORT`) and :data:`QUICK_CLOCKS`.
    """

    kind: str = "SWR"
    target_edf: float | None = None
    sessions_per_animal: int | None = None
    n_data_sets: int = 20
    n_draws: int = 1000
    prior_sds: tuple[float, ...] = PRIOR_SDS
    calibration_path: str | None = None
    quick: bool = False


QUICK = Config(n_data_sets=2, n_draws=200, prior_sds=(2.0,), quick=True)


# --- The rows --------------------------------------------------------------------


def synthetic_visit_rows(
    key: tuple[str, str, str],
    *,
    n_visits: int,
    seed: int,
    bin_s: float = 0.05,
    p_reward: float = 0.6,
    two_stretch_share: float = 0.5,
    length_s: tuple[float, float] = (1.5, 8.0),
    omission_length_s: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """One session's synthetic in-visit bins, as the cohort step returns them.

    Visits of random length, each one or two immobile stretches (with a 0.3 s
    movement between them), bins of ``bin_s`` from each stretch's start, the
    three clocks at the bin center, and an outcome history with ten known
    outcomes in every bin. The counts are zero.

    Parameters
    ----------
    key : tuple of str
        (animal, session_date, epoch).
    n_visits : int
    seed : int
    bin_s : float
        Bin width (s).
    p_reward : float
        The chance that a visit is rewarded.
    two_stretch_share : float
        The share of visits with two stretches; 0 makes the stretch clock the
        leaving clock in every bin.
    length_s : tuple of float
        The range of the visits' lengths (uniform).
    omission_length_s : tuple of float, optional
        The range of the lengths of the visits whose last outcome is an
        omission; None draws them from ``length_s`` too.

    Returns
    -------
    pd.DataFrame
        One row per bin.
    """
    gap_s = 0.3
    rng = np.random.default_rng(seed)
    outcomes = (rng.random(n_visits + len(REWARD_LAG_COLUMNS)) < p_reward).astype(float)
    parts = []
    for visit in range(n_visits):
        history = outcomes[visit : visit + len(REWARD_LAG_COLUMNS)][::-1]
        omitted = history[0] == 0.0 and omission_length_s is not None
        length = rng.uniform(*(omission_length_s if omitted else length_s))
        if rng.random() < two_stretch_share:
            cut = rng.uniform(0.3, length - gap_s - 0.3)
            stretches = [(0.0, cut), (cut + gap_s, length)]
        else:
            stretches = [(0.0, length)]
        for index, (start, end) in enumerate(stretches):
            left = np.arange(start, end - 1e-9, bin_s)
            right = np.minimum(left + bin_s, end)
            center = (left + right) / 2
            part = pd.DataFrame(
                {
                    "well_visit_id": visit,
                    "visit_stretch_index": index,
                    "bin_start_s": 100.0 * visit + left,
                    "bin_end_s": 100.0 * visit + right,
                    "exposure": right - left,
                    "count": 0.0,
                    vr.ARRIVAL: center,
                    vr.LEAVING: length - center,
                    vr.STRETCH: end - center,
                }
            )
            for column, outcome in zip(REWARD_LAG_COLUMNS, history, strict=True):
                part[column] = outcome
            parts.append(part)
    rows = pd.concat(parts, ignore_index=True)
    rows[vr.REWARD] = rows[REWARD_LAG_COLUMNS[0]]
    rows["visit_context"] = "in_visit"
    rows["well_visit_id"] = pd.array(rows["well_visit_id"], dtype="Int64")
    for column, value in zip(("animal", "session_date", "epoch"), key, strict=True):
        rows.insert(0, column, value)
    return rows[["animal", "session_date", "epoch", *rows.columns[3:]]]


def _synthetic_cohort(kind: str) -> dict[SessionKey, vr.CohortRows]:
    """The quick configuration's cohort (:data:`QUICK_COHORT`); each
    session's counts are drawn once from :data:`QUICK_CLOCKS` with a ratio of
    1.5, so that the rate table and the totals have counts to read."""
    n_animals, n_sessions, n_visits, total = QUICK_COHORT
    rng = np.random.default_rng(0)
    out = {}
    for a in range(n_animals):
        for e in range(n_sessions):
            key = (f"animal{a}", f"2026{e + 1:04d}", "02_r1")
            rows = synthetic_visit_rows(key, n_visits=n_visits, seed=n_sessions * a + e)
            truth = sc.constant_ratio(
                rows, QUICK_CLOCKS, log_ratio=float(np.log(1.5)), total=total
            )
            rows["count"] = rng.poisson(truth.expected).astype(float)
            out[key] = vr.CohortRows(kind, {"A": rows, "D": rows.iloc[:0]}, {})
    return out


def spread_sessions(keys: Sequence[SessionKey], per_animal: int) -> list[SessionKey]:
    """``per_animal`` sessions of each animal, spread evenly over its
    sessions in sorted order (all of an animal with fewer)."""
    out = []
    for animal in sorted({key[0] for key in keys}):
        own = sorted(key for key in keys if key[0] == animal)
        if len(own) <= per_animal:
            out += own
            continue
        picks = np.unique(np.round(np.linspace(0, len(own) - 1, per_animal)))
        out += [own[int(i)] for i in picks]
    return out


#: A worker's tables, set once by the pool's initializer.
_TABLES: dict = {}


def _load_tables() -> None:
    from continuum_swr_replay.analysis.event_definition.analyzed_time import (
        excluded_intervals_by_session,
        read_swr_table,
    )
    from continuum_swr_replay.paths import table_path
    from continuum_swr_replay.tables.session_keys import SESSION_COLUMNS

    stops = read_swr_table(table_path("stop_table"))
    events = read_swr_table(table_path("event_table"))
    for table in (stops, events):
        for column in SESSION_COLUMNS:
            table[column] = table[column].astype(str)
    keys = list(SESSION_COLUMNS)
    excluded = excluded_intervals_by_session(stops)
    _TABLES["stops"] = dict(tuple(stops.groupby(keys, sort=True, observed=True)))
    _TABLES["events"] = dict(tuple(events.groupby(keys, sort=False, observed=True)))
    _TABLES["excluded"] = dict(tuple(excluded.groupby(keys, sort=False, observed=True)))
    _TABLES["empty"] = (events.iloc[:0], excluded.iloc[:0])


def _session_rows(key: SessionKey, kind: str) -> vr.CohortRows:
    from continuum_swr_replay.analysis.event_tables import prepare_data
    from continuum_swr_replay.decoding.identifiers import (
        format_epoch_identifier,
        nwb_file_name,
    )

    empty_events, empty_excluded = _TABLES["empty"]
    experiences = prepare_data.cached_port_experiences(
        format_epoch_identifier(nwb_file_name(key[0], key[1]), key[2], "HPC")
    )
    return vr.cohort_rows(
        _TABLES["stops"][key],
        _TABLES["events"].get(key, empty_events),
        experiences,
        kind=kind,
        settings=vr.VisitRateSettings(),
        excluded_intervals=_TABLES["excluded"].get(key, empty_excluded),
    )


def real_cohort(
    kind: str, sessions_per_animal: int | None, workers: int = 1
) -> tuple[dict[SessionKey, vr.CohortRows], dict]:
    """Model A's rows of one kind in the analyzed sessions of the canonical
    tables (``visit_rate.cohort_rows`` at the default settings), and their
    record.

    Each session's port experiences are read from the session cache only.

    Returns
    -------
    cohort : dict of (animal, session_date, epoch) to CohortRows
    record : dict
        The tables' SHA-256, the commit, the sessions and the rows.
    """
    from continuum_swr_replay.paths import table_path
    from continuum_swr_replay.runs.io import sha256_file
    from continuum_swr_replay.runs.provenance import described_commit

    _load_tables()
    keys = list(_TABLES["stops"])
    if sessions_per_animal is not None:
        keys = spread_sessions(keys, sessions_per_animal)
    if workers > 1:
        with pool_context().Pool(workers, _load_tables) as pool:
            rows = pool.starmap(_session_rows, [(key, kind) for key in keys])
    else:
        rows = [_session_rows(key, kind) for key in keys]
    cohort = dict(zip(keys, rows, strict=True))
    record = {
        "tables": {
            name: sha256_file(table_path(name))
            for name in ("stop_table", "event_table")
        },
        "commit": described_commit(),
        "sessions": ["/".join(key) for key in keys],
        "n_sessions": len(keys),
        "n_rows": int(sum(len(c.rows["A"]) for c in cohort.values())),
    }
    return cohort, record


# --- The planted curves ------------------------------------------------------------


def smoothed_log_rate(
    starts: np.ndarray,
    at_risk_s: np.ndarray,
    x: np.ndarray,
    lo: float,
    hi: float,
    spacing_s: float,
    log_penalty: float,
) -> sc.LogCurve:
    """A smooth log rate through binned event counts, by the shared fitter.

    A Poisson P-spline (``terms.Smooth`` over ``[lo, hi]``, knots
    ``spacing_s`` apart, at ``log_penalty``) with the intercept and the
    offset ``log(at_risk_s)``, fitted to the bins' event starts at their
    clock values ``x``; returned as :func:`.visit_rate_scenarios.interpolated_curve`
    on a 0.01 s grid (flat beyond the range).

    Parameters
    ----------
    starts, at_risk_s, x : np.ndarray, shape (n_bins,)
        Each bin's event starts, at-risk seconds and clock value (s).
    lo, hi, spacing_s, log_penalty : float
    """
    rows = pd.DataFrame(
        {"x": x, "exposure": at_risk_s, "count": starts, "unit": np.arange(len(x))}
    )
    rows = rows[rows["exposure"] > 0].reset_index(drop=True)
    n_basis = PSplineBasis.from_spacing(lo, hi, spacing_s).n_basis
    design = build_design(
        [Smooth("x", lo, hi, n_basis, log_penalty), Offset("exposure")],
        rows,
        intercept=True,
    )
    (record,) = fit_sessions(
        design,
        {("curve", "", ""): rows},
        Poisson(),
        model="smoothed_log_rate",
        kind="",
        response="count",
        unit="unit",
        table_sha256={},
    )
    if record.status != CONVERGED:
        raise ValueError(f"the smoothing fit did not converge: {record.status}")
    grid = np.round(np.arange(lo, hi + 1e-9, 0.01), 10)
    X = design.build(pd.DataFrame({"x": grid, "exposure": 1.0})).X
    return sc.interpolated_curve(grid, X @ np.asarray(record.coef))


def shaped_clocks(kind: str, curves: pd.DataFrame) -> sc.PlantedClocks:
    """The planted arrival and leaving curves of a kind, shaped like the
    stored model-free curves (module docstring): each clock's event starts
    and at-risk seconds summed over the animals, smoothed by
    :func:`smoothed_log_rate` (:data:`SMOOTHING`), and centered (any constant
    is the session's baseline).

    Parameters
    ----------
    kind : {"SWR", "No SWR"}
    curves : pd.DataFrame
        The stored curves (``describe_visit_clocks.py``).
    """
    out = {}
    for clock in (vr.ARRIVAL, vr.LEAVING):
        part = curves[(curves["kind"] == kind) & (curves["clock"] == clock)]
        pooled = (
            part.groupby(["clock_bin_start_s", "clock_bin_end_s"], as_index=False)[
                ["n_starts", "at_risk_s"]
            ]
            .sum()
            .sort_values("clock_bin_start_s")
        )
        lo, hi, _ = vr.CLOCK_BASES[clock]
        spacing, log_penalty = SMOOTHING[clock]
        x = 0.5 * (pooled["clock_bin_start_s"] + pooled["clock_bin_end_s"])
        out[clock] = smoothed_log_rate(
            pooled["n_starts"].to_numpy(dtype=float),
            pooled["at_risk_s"].to_numpy(dtype=float),
            x.to_numpy(dtype=float),
            lo,
            hi,
            spacing,
            log_penalty,
        )
    return sc.PlantedClocks(out[vr.ARRIVAL], out[vr.LEAVING])


# --- The scenarios -------------------------------------------------------------------


def _scenario_truth(
    name: str,
    rows: pd.DataFrame,
    clocks: sc.PlantedClocks,
    kind: str,
    total: float,
    q1_reference_s: tuple[float, float],
    log_ratios: Mapping[SessionKey, float],
    table: sc.RateTable,
) -> sc.Planted:
    """One session's planted scenario."""
    common = {"total": total, "q1_reference_s": q1_reference_s}
    if name == "rate_table":
        return sc.rate_table_truth(rows, table, **common)
    if name == "no_effect":
        return sc.no_effect(rows, clocks, **common)
    if name == "constant_ratio":
        return sc.constant_ratio(rows, clocks, log_ratio=LOG_RATIO[kind], **common)
    if name == "fading_ratio":
        return sc.fading_ratio(
            rows,
            clocks,
            log_ratio=LOG_RATIO[kind],
            fade_start_s=FADE_S[0],
            fade_end_s=FADE_S[1],
            **common,
        )
    if name == "differing_ratio":
        return sc.differing_ratio(rows, clocks, log_ratios=log_ratios, **common)
    if name == "later_onset":
        return sc.later_onset_after_omission(rows, clocks, delay_s=DELAY_S, **common)
    if name == "drop_before_leaving":
        return sc.drop_before_leaving(
            rows, clocks.arrival, drop=DROP, width_s=DROP_WIDTH_S, **common
        )
    if name == "drop_before_small_movements":
        return sc.drop_before_small_movements(
            rows, clocks.arrival, drop=DROP, width_s=DROP_WIDTH_S, **common
        )
    raise ValueError(f"no scenario {name!r}")


def _session_planted(
    rows: pd.DataFrame,
    names: Sequence[str],
    clocks: sc.PlantedClocks,
    kind: str,
    total: float,
    log_ratios: Mapping[SessionKey, float],
    table: sc.RateTable,
) -> dict[str, dict]:
    """Each scenario's expected counts and truths in one session:
    the pooled and omission-side truths with Q1 at both reference windows."""
    out = {}
    for name in names:
        planted = _scenario_truth(
            name, rows, clocks, kind, total, (0.0, 1.0), log_ratios, table
        )
        later = _scenario_truth(
            name, rows, clocks, kind, total, (1.0, 2.0), log_ratios, table
        )
        out[name] = {
            "expected": planted.expected,
            "pooled": {**planted.pooled, "Q1b": later.pooled["Q1"]},
            "omission_side": {
                **planted.omission_side,
                "Q1b": later.omission_side["Q1"],
            },
        }
    return out


def _balanced(values: Mapping[SessionKey, float]) -> float:
    """The session -> day -> animal reduction of one value per session: the
    mean of the animals' values (as ``estimand_summary``'s estimate); NaN
    when no session has a finite value."""
    keys = [key for key, value in values.items() if np.isfinite(value)]
    if not keys:
        return np.nan
    index = pd.RangeIndex(len(keys))
    per_animal = animal_day_session_values(
        pd.Series([values[k] for k in keys], index=index, dtype=float),
        *(pd.Series([k[i] for k in keys], index=index) for i in range(3)),
    )
    return _snapped(float(per_animal.mean()))


def _snapped(value: float) -> float:
    """A truth that is zero in exact arithmetic comes out of the projection
    as a rounding error (below 1e-12): it is zero, so that its bias has no
    share."""
    return 0.0 if abs(value) < 1e-12 else value


def _expected_column(
    sessions: Mapping[SessionKey, pd.DataFrame],
    rng: np.random.Generator,
    *,
    column: str,
) -> dict[SessionKey, np.ndarray]:
    """An outside scenario's expected counts, read from the rows' column."""
    return {key: rows[column].to_numpy(dtype=float) for key, rows in sessions.items()}


def _planted_of(model: Model, *, values: Mapping[str, float]) -> float:
    """A planted value that depends on the model: its level's truth."""
    return values[model.name]


# --- The models ------------------------------------------------------------------


def _estimands(design, settings: vr.VisitRateSettings) -> dict[str, np.ndarray]:
    """Q1 (declared window), Q1b (1 to 2 s), Q2 (at the session level) and
    Q7a on a design."""
    c = vr.declared_contrasts(design, replace(settings, q1_reference_s=(0.0, 1.0)))
    later = vr.declared_contrasts(design, replace(settings, q1_reference_s=(1.0, 2.0)))
    out = {"Q1": c["Q1"], "Q1b": later["Q1"], "Q7a": c["Q7a"]}
    if "Q2" in c:
        out["Q2"] = c["Q2"]
    return out


def _models(
    kind: str,
    pooled: pd.DataFrame,
    config: Config,
    calibrated: Mapping[tuple[str, str], vr.CalibratedPenalty] | None,
) -> list[Model]:
    """The session-level models, one per prior SD."""
    base = vr.VisitRateSettings(calibrated_penalties=calibrated)
    models = []
    for sd in config.prior_sds:
        settings = replace(base, weak_prior_sd=sd)
        design = vr.build_visit_rate_design("A", kind, pooled, settings)
        models.append(
            Model(
                f"session_sd{sd:g}",
                design,
                Poisson(),
                vr.RESPONSE,
                vr.UNIT,
                _estimands(design, settings),
            )
        )
    return models


# --- The calibration ---------------------------------------------------------------


def _calibrate_one(
    cohort: Mapping[SessionKey, vr.CohortRows],
    kind: str,
    curve: str,
    target_edf: float,
    log_penalties: Mapping[str, float],
) -> tuple[float, float]:
    """One clock's calibrated log penalty and median edf, the others at
    ``log_penalties``."""
    found = vr.calibrate_log_penalty(
        cohort,
        model="A",
        kind=kind,
        curve=curve,
        target_edf=target_edf,
        settings=vr.VisitRateSettings(log_penalties=dict(log_penalties)),
    )
    return float(found.log_penalty), float(found.median_edf)


def calibrate_clocks(
    cohort: Mapping[SessionKey, vr.CohortRows],
    kind: str,
    target_edf: float,
    clocks: sc.PlantedClocks,
    record: Mapping[str, object],
    workers: int = 1,
) -> dict[tuple[str, str], vr.CalibratedPenalty]:
    """Each clock's log penalty at the target median edf, on one data set of
    the calibration scenario (the constant ratio of the kind, drawn with
    :data:`CALIBRATION_SEED`), in two rounds (module docstring).

    Returns
    -------
    dict of (kind, clock) to CalibratedPenalty
        Each with its record: ``record`` (the rows), the calibration
        scenario and seed, and the median edf of each round.
    """
    rng = np.random.default_rng(CALIBRATION_SEED)
    drawn = {}
    for key, rows in cohort.items():
        a = rows.rows["A"]
        planted = sc.constant_ratio(
            a, clocks, log_ratio=LOG_RATIO[kind], total=float(a[vr.RESPONSE].sum())
        )
        drawn[key] = replace(
            rows, rows={"A": a.assign(**{vr.RESPONSE: rng.poisson(planted.expected)})}
        )
    curves = vr.VisitRateSettings().clocks()
    values = {c: vr.CLOCK_LOG_PENALTIES[kind]["declared"][c] for c in curves}
    rounds = []
    for _ in range(2):
        tasks = [(drawn, kind, c, target_edf, values) for c in curves]
        if workers > 1:
            with pool_context().Pool(min(workers, len(tasks))) as pool:
                found = pool.starmap(_calibrate_one, tasks)
        else:
            found = [_calibrate_one(*task) for task in tasks]
        values = {c: f[0] for c, f in zip(curves, found, strict=True)}
        rounds.append(
            {
                c: {"log_penalty": f[0], "median_edf": f[1]}
                for c, f in zip(curves, found, strict=True)
            }
        )
    return {
        (kind, c): vr.CalibratedPenalty(
            values[c],
            target_edf,
            {
                **record,
                "calibration_scenario": f"constant_ratio log {np.exp(LOG_RATIO[kind]):g}",
                "calibration_seed": CALIBRATION_SEED,
                "rounds": rounds,
            },
        )
        for c in curves
    }


def _calibrated(
    cohort, kind: str, config: Config, clocks, record, workers: int
) -> dict[tuple[str, str], vr.CalibratedPenalty] | None:
    """The calibrated penalties of the run: read from ``calibration_path``
    when it exists, else calibrated and written there; None without a target."""
    if config.target_edf is None:
        return None
    path = Path(config.calibration_path) if config.calibration_path else None
    if path is not None and path.exists():
        stored = json.loads(path.read_text())
        return {
            (kind, c): vr.CalibratedPenalty(
                v["log_penalty"], v["target_edf"], v["rows"]
            )
            for c, v in stored.items()
        }
    found = calibrate_clocks(cohort, kind, config.target_edf, clocks, record, workers)
    if path is not None:
        from continuum_swr_replay.runs.io import json_ready
        from continuum_swr_replay.tables.io import write_atomically

        text = json.dumps(
            {
                c: {
                    "log_penalty": p.log_penalty,
                    "target_edf": p.target_edf,
                    "rows": json_ready(dict(p.rows)),
                }
                for (_, c), p in found.items()
            },
            indent=1,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        write_atomically(path, lambda tmp: Path(tmp).write_text(text))
    return found


# --- The run ---------------------------------------------------------------------


def _rate_table(sessions: Mapping[SessionKey, pd.DataFrame]) -> sc.RateTable:
    """The rate table of the scenario taken from a table of rates: the rows'
    counts by outcome and both clocks, pooled over every session (a
    model-free tabulation; ``rate_table_from_bins``). On the real cohort
    these are the REAL counts, so the scenario's planted rate and its truths
    are the real data's pooled model-free values
    (:data:`TRUTH_FROM_REAL_COUNTS`)."""
    pooled = pd.concat(
        [
            rows[[vr.ARRIVAL, vr.LEAVING, vr.REWARD, vr.EXPOSURE, vr.RESPONSE]]
            for rows in sessions.values()
        ],
        ignore_index=True,
    )
    return sc.rate_table_from_bins(
        pooled,
        RATE_TABLE_ARRIVAL_EDGES,
        RATE_TABLE_LEAVING_EDGES,
        shrink_s=RATE_TABLE_SHRINK_S,
        floor=RATE_TABLE_FLOOR,
    )


def _planted_task(args) -> dict[str, dict]:
    return _session_planted(*args)


def simulation(
    quick: bool, workers: int, config: Mapping[str, object] | None = None
) -> Simulation:
    """One run of the list (module docstring).

    Parameters
    ----------
    quick : bool
        The quick configuration (:data:`QUICK`), whatever ``config`` says of
        the cohort.
    workers : int
        Processes for building the rows, the planted values and the
        calibration.
    config : mapping, optional
        The fields of :class:`Config`; None is the default :class:`Config`.

    Returns
    -------
    Simulation
        Its ``attrs`` hold the run's configuration, the rows' record, the
        calibration, and each scenario's truths (``truths``).
    """
    config = Config(**dict(config or {}))
    if quick:
        config = replace(QUICK, kind=config.kind)
    kind = config.kind
    names = SCENARIOS[kind]
    if config.quick:
        cohort, record = _synthetic_cohort(kind), {"rows": "synthetic"}
        clocks = QUICK_CLOCKS
    else:
        from continuum_swr_replay.paths import replay_content_dir

        cohort, record = real_cohort(kind, config.sessions_per_animal, workers)
        clocks = shaped_clocks(
            kind, pd.read_parquet(replay_content_dir() / DESCRIPTIVE)
        )
    sessions = {key: rows.rows["A"] for key, rows in cohort.items()}
    totals = {key: float(rows[vr.RESPONSE].sum()) for key, rows in sessions.items()}
    table_rates = _rate_table(sessions)
    keys_frame = pd.DataFrame(
        list(sessions), columns=["animal", "session_date", "epoch"]
    )
    log_ratios = sc.session_log_ratios(
        keys_frame,
        mean=LOG_RATIO[kind],
        animal_sd=DIFFERING_SD[0],
        session_sd=DIFFERING_SD[1],
        seed=DIFFERING_SEED,
    )
    tasks = [
        (rows, names, clocks, kind, totals[key], log_ratios, table_rates)
        for key, rows in sessions.items()
    ]
    if workers > 1 and not config.quick:
        with pool_context().Pool(workers) as pool:
            planted_list = pool.map(_planted_task, tasks, chunksize=1)
    else:
        planted_list = [_planted_task(task) for task in tasks]
    planted = dict(zip(sessions, planted_list, strict=True))
    calibrated = _calibrated(cohort, kind, config, clocks, record, workers)
    pooled = pd.concat(list(sessions.values()), ignore_index=True)
    models = _models(kind, pooled, config, calibrated)
    rows_out = {}
    for key, rows in sessions.items():
        rows_out[key] = rows.assign(
            **{f"expected_{n}": planted[key][n]["expected"] for n in names}
        )
    scenarios, truths = [], {}
    for name in names:
        by_session = {key: planted[key][name] for key in sessions}
        truth = {
            "session": {
                e: _balanced({k: v["pooled"][e] for k, v in by_session.items()})
                for e in ("Q1", "Q1b", "Q2", "Q7a")
            },
            "omission_side": {
                e: _balanced({k: v["omission_side"][e] for k, v in by_session.items()})
                for e in ("Q1", "Q1b", "Q2", "Q7a")
            },
        }
        truths[name] = truth
        planted_values = {
            e: partial(
                _planted_of,
                values={m.name: truth["session"][e] for m in models},
            )
            for e in ("Q1", "Q1b", "Q2", "Q7a")
        }
        scenarios.append(
            OutsideScenario(
                name,
                partial(_expected_column, column=f"expected_{name}"),
                planted=planted_values,
                columns=(f"expected_{name}",),
            )
        )
    attrs = {
        "config": {
            **dict(config.__dict__),
            "prior_sds": list(config.prior_sds),
        },
        "rows": record,
        "calibration": {
            c: {"log_penalty": p.log_penalty, "target_edf": p.target_edf}
            for (_, c), p in (calibrated or {}).items()
        },
        "allow_provisional": None if calibrated else "simulation",
        "truths": truths,
        "truth_from_real_counts": {
            "scenarios": [n for n in names if n in TRUTH_FROM_REAL_COUNTS],
            "note": (
                "the planted rate is a tabulation of the rows' counts (the real "
                "counts on the real cohort) and its planted values a Poisson "
                "projection onto it: the real data's pooled model-free values, "
                "not a result"
            ),
        },
        "planted_constants": {
            "log_ratio": LOG_RATIO[kind],
            "fade_s": list(FADE_S),
            "delay_s": DELAY_S,
            "drop": DROP,
            "drop_width_s": DROP_WIDTH_S,
            "differing_sd": list(DIFFERING_SD),
            "differing_seed": DIFFERING_SEED,
        },
    }
    return Simulation(
        models,
        rows_out,
        scenarios,
        config.n_data_sets,
        config.n_draws,
        DATA_SEED,
        attrs,
    )


def table(result: SimulationResult, simulation: Simulation) -> pd.DataFrame:
    """The engine's summary with the run's kind, target and each model's
    prior SD, and Q2's truths beside the planted value:
    ``truth_session`` (the pooled truth) and
    ``truth_omission_side`` (the time-varying form's), and
    ``truth_from_real_counts`` (:data:`TRUTH_FROM_REAL_COUNTS`)."""
    config = simulation.attrs["config"]
    truths = simulation.attrs["truths"]
    out = result.summary.copy()
    out.insert(0, "kind", config["kind"])
    out.insert(1, "target_edf", config["target_edf"])
    out["prior_sd"] = [
        float(m.removeprefix("session_sd")) if m.startswith("session_sd") else np.nan
        for m in out["model"]
    ]
    out["truth_from_real_counts"] = out["scenario"].isin(TRUTH_FROM_REAL_COUNTS)
    for name in ("session", "omission_side"):
        out[f"truth_{name}"] = [
            truths[s][name][e]
            for s, e in zip(out["scenario"], out["estimand"], strict=True)
        ]
    return out
