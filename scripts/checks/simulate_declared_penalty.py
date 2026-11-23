#!/usr/bin/env python
"""The declared-penalty rule for a per-session smooth, in simulation.

Every number this script writes is a simulation number.

The rule: a per-session smooth term is the cubic P-spline at one fixed log
penalty, declared in the model and the same in every session, the value at
which the median edf of the term over the sessions is a target
(``regression.calibration.log_penalty_for_edf``). This module is a scenario
list of the simulation engine (``regression.simulation``): it builds the
cohort, the model at each target's declared penalty and the two scenarios,
and ``scripts/checks/run_simulations.py --model declared_penalty`` runs them.
The engine draws the counts and runs the real fit
(``regression.sessions.fit_sessions``), the curve contrast
(``regression.contrasts``) and the reduction session -> recording day ->
animal (``shared_statistics.estimand_summary``).

The simulation:

- the cohort: 5 animals of :data:`DAYS_PER_ANIMAL` recording days and
  :data:`SESSIONS_PER_DAY` sessions a day, the shape of the real cohort as
  ``simulate_interval_coverage.py`` draws it (45 days; 256 sessions, drawn
  once with :data:`STRUCTURE_SEED`, with their bins and event numbers);
- a session: :data:`N_VISITS` visits whose durations mix short, ordinary and
  long stays (:data:`DURATIONS`), cut into 15 ms bins from the visit's start
  (median 35,900 bins a session, visits of median 2.2 s, deciles 0.26 and
  6.1 s); its expected number of events is lognormal around
  :data:`EVENTS_MEDIAN`, clipped to :data:`EVENTS_RANGE`. These follow the
  real sessions' in-visit bins and SWR counts (median 181 visits, 33,900
  bins and 110 events, range 33 to 782; visits of median 2.3 s, deciles 0.26
  and 5.9 s; measured on 2026-11-23 from the at-risk bins of
  ``.claude/research/2026-11-20-regression-prototypes/proto_rate_smoothing``);
- the count of a bin is Poisson with mean ``exposure * exp(b_s + f(s))``,
  ``s`` the time since the visit's start and ``b_s`` set so that the expected
  events of the session are its number; :data:`CURVES` gives ``f``: a
  threefold rise over the first 3 s (``rise``) and none (``flat``), the same
  in every session (no session, day or animal effects). Both are scenarios
  outside the model (``regression.simulation.OutsideScenario``): the rise is
  not a curve of the smooth's span, so each states its planted contrast;
- the model: an intercept, ``Smooth(time_since_visit_start_s, 0, 20, 43)`` (the
  planned basis, knots 0.5 s apart) and the exposure's offset, Poisson;
- the estimand: the Q1 contrast, the curve's mean 2 to 4 s after arrival less
  its mean in the first second (:data:`CONTRAST`).

The scope: every session shares one curve and one rate, with no session, day
or animal effect, and its events are independent Poisson counts, not
clustered in time. The bias, spread, coverage and rejection rates therefore
speak only to sampling noise and the penalty's shrinkage, not to the
variation between sessions, days and animals that the project's interval is
built for. Without that variation the interval is known to be wide: it
covers 0.999 of the truth with no animal or day effect
(``simulate_interval_coverage.py``, setting ``sd 0/0/1`` of
``tests/scripts/checks/data/interval_coverage_full.csv``), so a coverage near
1 here is expected and is not a finding about real data.

The bins of a session are summed over those with the same time since the
visit's start (every full bin of a given index, and each last partial bin
alone). The design row of such bins is the same, so the summed counts are
sufficient and the fit, the edf and the contrast are those of the bins
(``test_simulate_declared_penalty.py`` checks it on one session).

The declared log penalty of each target edf (:data:`TARGETS`) is found once,
on one data set of the rise (:data:`CALIBRATION_SEED`), and held for every
data set of both curves: one model per target, all fitted to the same data
sets. The table (:func:`table`), one row per curve and target:

- ``target_edf``, ``log_penalty``, ``calibration_median_edf``;
- ``median_edf``: the median over every session and data set;
- ``planted``: the planted contrast; ``mean_estimate``, ``bias`` (the mean of
  the animal-balanced estimate less ``planted``), ``bias_share`` (``bias`` as
  a share of ``planted``; NaN for the flat curve), ``sd`` (over data sets),
  ``bias_se``;
- the project's inference for a scalar (``estimand_summary``): ``coverage``,
  the share of data sets whose 95 % interval (the nested bootstrap,
  t-corrected) holds ``planted``; ``excludes_zero``, the share whose interval
  excludes zero; ``reject_day``, the share whose day-level signed-rank p is
  below 0.05, a NaN p (fewer than six non-zero day values) counted as no
  rejection. For the flat curve ``excludes_zero`` and ``reject_day`` are
  false positives, for the rise power. Each has its binomial standard error
  (``*_se``);
- ``reject_t_reference``: the share whose t test over the 5 animal values
  (``animal_day_session_values``) against zero has p below 0.05, with
  ``reject_t_reference_se``. It is a reference only, not the project's test:
  no claim rests on a p over five animals;
- ``share_converged``: the session fits that converged. A session without a
  converged fit leaves that data set's estimate: ``n_sessions_used_min`` and
  ``n_sessions_used_mean`` are the sessions that entered the estimates over
  the data sets, ``n_sessions`` the cohort's; ``n_data_sets``, ``n_draws``.

Data set ``i`` of curve ``k`` (0 the rise, 1 the flat curve) is drawn from
``SeedSequence([DATA_SEED, k, i])``; the bootstrap seed of each target's row
is a child spawned from it (``regression.simulation.bootstrap_seeds``).

::

    python scripts/checks/run_simulations.py --model declared_penalty --output-dir OUT
    python scripts/checks/run_simulations.py --model declared_penalty --quick
    python scripts/checks/simulate_declared_penalty.py --output-dir OUT --workers 16

``--quick`` runs a cohort of 5 animals of 1 day of 2 sessions and 2 data sets
per curve; its table is the fixed-seed fixture
``tests/analysis/regression/data/numpy_reference/simulation_declared_penalty.json``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from continuum_swr_replay.runs.threads import pin_blas_threads

pin_blas_threads()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.stats  # noqa: E402

from continuum_swr_replay.analysis.regression.calibration import (  # noqa: E402
    log_penalty_for_edf,
)
from continuum_swr_replay.analysis.regression.contrasts import (  # noqa: E402
    range_grid,
    range_mean_contrast,
)
from continuum_swr_replay.analysis.regression.likelihoods import Poisson  # noqa: E402
from continuum_swr_replay.analysis.regression.newton import CONVERGED  # noqa: E402
from continuum_swr_replay.analysis.regression.simulation import (  # noqa: E402
    Model,
    OutsideScenario,
    Simulation,
    SimulationResult,
    draw_data_set,
)
from continuum_swr_replay.analysis.regression.terms import (  # noqa: E402
    Design,
    Offset,
    Smooth,
    build_design,
)
from continuum_swr_replay.analysis.shared_statistics import (  # noqa: E402
    animal_day_session_values,
)

SessionKey = tuple[str, str, str]

BIN_S = 0.015
COLUMN = "time_since_visit_start_s"
TERM = f"s({COLUMN})"
TERM_RANGE = (0.0, 20.0)
N_BASIS = 43
#: Q1: the mean 2 to 4 s after arrival less the mean in the first second.
CONTRAST = ((2.0, 4.0), (0.0, 1.0))
ESTIMAND = "Q1"
TARGETS = (4.0, 6.0, 8.0)


def _rise(s: np.ndarray) -> np.ndarray:
    """A threefold rise over the first 3 s (a smoothstep), flat after."""
    u = np.clip(np.asarray(s, dtype=float) / 3.0, 0.0, 1.0)
    return np.log(3.0) * u**2 * (3.0 - 2.0 * u)


def _flat(s: np.ndarray) -> np.ndarray:
    """No change over the visit."""
    return np.zeros(np.shape(s))


CURVES = {"rise": _rise, "flat": _flat}

DAYS_PER_ANIMAL = (6, 8, 9, 10, 12)
SESSIONS_PER_DAY = (4, 7)  # inclusive range
N_VISITS = 181
#: Share, median (s) and log SD of the short, ordinary and long stays.
DURATIONS = ((0.49, 0.4, 0.5), (0.50, 4.5, 0.35), (0.01, 30.0, 0.7))
EVENTS_MEDIAN = 110.0
EVENTS_LOG_SD = 0.4
EVENTS_RANGE = (33.0, 782.0)
STRUCTURE_SEED = 0
CALIBRATION_SEED = 1
DATA_SEED = 20261123


@dataclass(frozen=True)
class Config:
    """The cohort's shape, the data sets per curve and the bootstrap draws."""

    days_per_animal: tuple[int, ...] = DAYS_PER_ANIMAL
    sessions_per_day: tuple[int, int] = SESSIONS_PER_DAY
    n_data_sets: int = 200
    n_draws: int = 1000


QUICK = Config(
    days_per_animal=(1,) * 5, sessions_per_day=(2, 2), n_data_sets=2, n_draws=200
)


def visit_bins(duration: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``BIN_S`` bins of visits of the given durations, from each visit's start.

    Parameters
    ----------
    duration : np.ndarray, shape (n_visits,)
        Each visit's duration (s).

    Returns
    -------
    visit : np.ndarray of int, shape (n_bins,)
    time : np.ndarray, shape (n_bins,)
        The time since the visit's start at the bin's middle (s).
    exposure : np.ndarray, shape (n_bins,)
        The bin's length (s); the last bin of a visit is partial.
    """
    n_bins = np.ceil(duration / BIN_S).astype(int)
    visit = np.repeat(np.arange(duration.size), n_bins)
    k = np.arange(n_bins.sum()) - np.repeat(np.cumsum(n_bins) - n_bins, n_bins)
    start = k * BIN_S
    end = np.minimum(start + BIN_S, duration[visit])
    return visit, 0.5 * (start + end), end - start


def durations(rng: np.random.Generator, n_visits: int) -> np.ndarray:
    """Visit durations (s), shape ``(n_visits,)``, from :data:`DURATIONS`."""
    shares, medians, sds = (np.array(column) for column in zip(*DURATIONS, strict=True))
    kind = rng.choice(len(DURATIONS), size=n_visits, p=shares / shares.sum())
    return medians[kind] * np.exp(sds[kind] * rng.normal(size=n_visits))


def cells(
    time: np.ndarray, exposure: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The bins summed over those with the same time: each distinct time and
    its total exposure, shape ``(n_cells,)``, and each bin's cell, shape
    ``(n_bins,)``."""
    unique, cell = np.unique(time, return_inverse=True)
    return unique, np.bincount(cell, weights=exposure), cell


def cohort(
    config: Config, seed: int = STRUCTURE_SEED
) -> dict[SessionKey, pd.DataFrame]:
    """The sessions of the cohort, their bins summed (module docstring).

    Returns
    -------
    dict of (animal, day, epoch) to pd.DataFrame
        Each session's cells: :data:`COLUMN`, ``exposure`` (s), ``cell`` (the
        unit) and ``expected_events``, the session's expected number of
        events, the same on every row.
    """
    rng = np.random.default_rng(seed)
    lo, hi = config.sessions_per_day
    out = {}
    for a, n_days in enumerate(config.days_per_animal):
        animal = f"animal{a}"
        for d in range(n_days):
            day = f"2026{d + 1:04d}"
            for e in range(int(rng.integers(lo, hi + 1))):
                _, time, exposure = visit_bins(durations(rng, N_VISITS))
                time, exposure, _ = cells(time, exposure)
                n_events = np.clip(
                    EVENTS_MEDIAN * np.exp(EVENTS_LOG_SD * rng.normal()), *EVENTS_RANGE
                )
                out[(animal, day, f"{2 * e + 2:02d}_r{e + 1}")] = pd.DataFrame(
                    {
                        COLUMN: time,
                        "exposure": exposure,
                        "cell": np.arange(time.size),
                        "expected_events": float(n_events),
                    }
                )
    return out


def expected_counts(
    sessions: dict[SessionKey, pd.DataFrame],
    rng: np.random.Generator,
    *,
    curve: str,
) -> dict[SessionKey, np.ndarray]:
    """Each cell's expected count under ``curve``: ``exposure * exp(f(s))``,
    scaled so that the session's sum is its ``expected_events``. A cell's
    count is one Poisson of its bins' summed mean, which is the sum of their
    independent counts. ``rng`` is not read: the curve is the same in every
    data set."""
    out = {}
    for key, rows in sessions.items():
        mean = rows["exposure"].to_numpy() * np.exp(
            CURVES[curve](rows[COLUMN].to_numpy())
        )
        mean *= rows["expected_events"].iloc[0] / mean.sum()
        out[key] = mean
    return out


def planted_contrast(curve: str) -> float:
    """:data:`CONTRAST` of the planted curve, on the same grids."""
    f = CURVES[curve]
    (a, b), (c, d) = CONTRAST
    return float(f(range_grid(a, b)).mean() - f(range_grid(c, d)).mean())


def scenarios() -> list[OutsideScenario]:
    """The rise, then the flat curve, each with its planted contrast."""
    return [
        OutsideScenario(
            curve,
            partial(expected_counts, curve=curve),
            planted={ESTIMAND: planted_contrast(curve)},
        )
        for curve in CURVES
    ]


def design_at(pooled: pd.DataFrame, log_penalty: float) -> Design:
    """The model's design (module docstring) at a declared log penalty."""
    return build_design(
        [Smooth(COLUMN, *TERM_RANGE, N_BASIS, log_penalty), Offset("exposure")],
        pooled,
        intercept=True,
    )


def contrast_vector(design: Design) -> np.ndarray:
    """The ``c`` of :data:`CONTRAST` on the smooth, shape ``(n_coef,)``."""
    columns = design.term_columns[TERM]

    def curve_rows(x: np.ndarray) -> np.ndarray:
        X = design.build(pd.DataFrame({COLUMN: x, "exposure": 1.0})).X
        rows = np.zeros_like(X)
        rows[:, columns] = X[:, columns]
        return rows

    return range_mean_contrast(curve_rows, *CONTRAST, term_range=TERM_RANGE)


def model_at(name: str, pooled: pd.DataFrame, log_penalty: float) -> Model:
    """The model at a declared log penalty, with the Q1 estimand."""
    design = design_at(pooled, log_penalty)
    return Model(
        name, design, Poisson(), "count", "cell", {ESTIMAND: contrast_vector(design)}
    )


def calibrate(
    target: float, sessions: dict[SessionKey, pd.DataFrame]
) -> tuple[float, float]:
    """The declared log penalty of a target edf, found on the calibration
    data set of the rise: ``(log penalty, median edf there)``."""
    pooled = pd.concat(sessions.values(), ignore_index=True)
    data = draw_data_set(
        model_at("calibration", pooled, 0.0),
        sessions,
        scenarios()[0],
        np.random.default_rng(CALIBRATION_SEED),
    )
    found = log_penalty_for_edf(
        design_at(pooled, 0.0),
        data,
        Poisson(),
        term=TERM,
        target_edf=target,
        response="count",
        unit="cell",
    )
    return found.log_penalty, found.median_edf


def model_name(target: float) -> str:
    """The name of the model at the declared penalty of ``target``."""
    return f"target_edf_{target:g}"


def simulation(quick: bool, config: Config | None = None) -> Simulation:
    """The run: the cohort of ``config`` (by default :data:`QUICK` or the
    full :class:`Config`), one model per target at its declared penalty, and
    the two curves. Each target's calibration is in ``attrs``."""
    config = config or (QUICK if quick else Config())
    sessions = cohort(config)
    pooled = pd.concat(sessions.values(), ignore_index=True)
    calibration, models = {}, []
    for target in TARGETS:
        log_penalty, median_edf = calibrate(target, sessions)
        name = model_name(target)
        calibration[name] = {
            "target_edf": target,
            "log_penalty": log_penalty,
            "calibration_median_edf": median_edf,
        }
        models.append(model_at(name, pooled, log_penalty))
    return Simulation(
        models,
        sessions,
        scenarios(),
        config.n_data_sets,
        config.n_draws,
        DATA_SEED,
        {"calibration": calibration},
    )


def _share(hits: np.ndarray, name: str) -> dict[str, float]:
    """The share of data sets in ``hits`` and its binomial standard error."""
    p = float(np.mean(hits))
    return {name: p, f"{name}_se": float(np.sqrt(p * (1 - p) / len(hits)))}


def _t_reference(sessions: pd.DataFrame) -> np.ndarray:
    """Per data set, whether the t test over the animal values has p below
    0.05, shape ``(n_data_sets,)``."""
    hits = []
    for _, group in sessions.groupby("data_set", sort=False):
        per_animal = animal_day_session_values(
            group["estimate"], group["animal"], group["session_date"], group["epoch"]
        )
        hits.append(scipy.stats.ttest_1samp(per_animal.to_numpy(), 0.0).pvalue < 0.05)
    return np.array(hits, dtype=float)


def table(result: SimulationResult, simulation: Simulation) -> pd.DataFrame:
    """One row per curve and target (module docstring), from the engine's
    tables."""
    calibration = simulation.attrs["calibration"]
    sessions = result.sessions.groupby(["scenario", "model"], sort=False)
    out = []
    for row in result.summary.itertuples():
        fits = sessions.get_group((row.scenario, row.model))
        declared = calibration[row.model]
        out.append(
            {
                "target_edf": declared["target_edf"],
                "curve": row.scenario,
                "log_penalty": declared["log_penalty"],
                "calibration_median_edf": declared["calibration_median_edf"],
                "median_edf": float(
                    np.nanmedian([edf.get(TERM, np.nan) for edf in fits["edf"]])
                ),
                "planted": row.planted,
                "mean_estimate": row.mean_estimate,
                "bias": row.bias,
                "bias_share": row.bias_share,
                "sd": row.sd,
                "bias_se": row.bias_se,
                "coverage": row.coverage,
                "coverage_se": row.coverage_se,
                "excludes_zero": row.excludes_zero,
                "excludes_zero_se": row.excludes_zero_se,
                "reject_day": row.reject_day,
                "reject_day_se": row.reject_day_se,
                **_share(_t_reference(fits), "reject_t_reference"),
                "share_converged": float(np.mean(fits["status"] == CONVERGED)),
                "n_sessions_used_min": row.n_sessions_used_min,
                "n_sessions_used_mean": row.n_sessions_used_mean,
                "n_data_sets": row.n_data_sets,
                "n_sessions": row.n_sessions,
                "n_draws": row.n_draws,
            }
        )
    return pd.DataFrame(out)


def main(argv: list[str] | None = None) -> None:
    """``run_simulations.py --model declared_penalty`` with ``argv``."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.checks import run_simulations

    run_simulations.main(["--model", "declared_penalty", *(argv or [])])


if __name__ == "__main__":
    main(sys.argv[1:])
