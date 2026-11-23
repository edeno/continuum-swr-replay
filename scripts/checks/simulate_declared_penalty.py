#!/usr/bin/env python
"""The declared-penalty rule for a per-session smooth, in simulation.

Every number this script writes is a simulation number.

The rule: a per-session smooth term is the cubic P-spline at one fixed log
penalty, declared in the model and the same in every session, the value at
which the median edf of the term over the sessions is a target
(``regression.calibration.log_penalty_for_edf``). Each session is fitted alone
(``regression.sessions.fit_sessions``); the scalar estimand is a curve
contrast of each session's fit (``regression.contrasts``), reduced session ->
recording day -> animal (``shared_statistics``).

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
  in every session (no session, day or animal effects);
- the model: an intercept, ``Smooth(time_since_visit_start_s, 0, 20, 43)`` (the
  planned basis, knots 0.5 s apart) and the exposure's offset, Poisson;
- the estimand: the Q1 contrast, the curve's mean 2 to 4 s after arrival less
  its mean in the first second (:data:`CONTRAST`).

The bins of a session are summed over those with the same time since the
visit's start (every full bin of a given index, and each last partial bin
alone). The design row of such bins is the same, so the summed counts are
sufficient and the fit, the edf and the contrast are those of the bins
(``test_simulate_declared_penalty.py`` checks it on one session).

The declared log penalty of each target edf (:data:`TARGETS`) is found once,
on one data set of the rise (:data:`CALIBRATION_SEED`), and held for every
data set of both curves. Each data set is fitted at every target. Written,
one row per target and curve (``declared_penalty.parquet`` in
``--output-dir``):

- ``target_edf``, ``log_penalty``, ``calibration_median_edf``;
- ``median_edf``: the median over every session and data set;
- ``planted``: the planted contrast; ``mean_estimate``, ``bias`` (the mean of
  the animal-balanced estimate less ``planted``), ``bias_share`` (``bias`` as
  a share of ``planted``; NaN for the flat curve), ``sd`` (over data sets),
  ``bias_se``;
- ``reject_t``: the share of data sets whose t test over the 5 animal values
  (``animal_day_session_values``) against zero has p below 0.05 (false
  positives for the flat curve, power for the rise), ``reject_t_se``;
- ``coverage``: the share whose 95 % interval (``estimand_summary``: the
  nested bootstrap, t-corrected) holds ``planted``, ``coverage_se``;
- ``share_converged``: the session fits that converged; ``n_data_sets``,
  ``n_sessions``, ``n_draws``.

::

    python scripts/checks/simulate_declared_penalty.py --output-dir OUT
    python scripts/checks/simulate_declared_penalty.py --output-dir OUT --quick
    python scripts/checks/simulate_declared_penalty.py --output-dir OUT --workers 16

``--quick`` runs a cohort of 5 animals of 2 days of 2 sessions and 4 data
sets per curve.
"""

from __future__ import annotations

import argparse
import multiprocessing
import time
from dataclasses import dataclass, replace
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
    evaluate_contrast,
    range_grid,
    range_mean_contrast,
)
from continuum_swr_replay.analysis.regression.likelihoods import Poisson  # noqa: E402
from continuum_swr_replay.analysis.regression.newton import CONVERGED  # noqa: E402
from continuum_swr_replay.analysis.regression.sessions import (  # noqa: E402
    fit_sessions,
)
from continuum_swr_replay.analysis.regression.terms import (  # noqa: E402
    Design,
    Offset,
    Smooth,
    build_design,
)
from continuum_swr_replay.analysis.shared_statistics import (  # noqa: E402
    animal_day_session_values,
    estimand_summary,
)
from continuum_swr_replay.runs.device import default_jax_platforms  # noqa: E402
from continuum_swr_replay.tables.io import write_table  # noqa: E402

BIN_S = 0.015
COLUMN = "time_since_visit_start_s"
TERM = f"s({COLUMN})"
TERM_RANGE = (0.0, 20.0)
N_BASIS = 43
#: Q1: the mean 2 to 4 s after arrival less the mean in the first second.
CONTRAST = ((2.0, 4.0), (0.0, 1.0))
TARGETS = (4.0, 6.0, 8.0)


def _rise(s: np.ndarray) -> np.ndarray:
    """A threefold rise over the first 3 s (a smoothstep), flat after."""
    u = np.clip(np.asarray(s, dtype=float) / 3.0, 0.0, 1.0)
    return np.log(3.0) * u**2 * (3.0 - 2.0 * u)


CURVES = {"rise": _rise, "flat": lambda s: np.zeros(np.shape(s))}

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
OUTPUT_NAME = "declared_penalty"


@dataclass(frozen=True)
class Config:
    """The cohort's shape, the data sets per curve and the bootstrap draws."""

    days_per_animal: tuple[int, ...] = DAYS_PER_ANIMAL
    sessions_per_day: tuple[int, int] = SESSIONS_PER_DAY
    n_data_sets: int = 200
    n_draws: int = 1000


QUICK = Config(days_per_animal=(2,) * 5, sessions_per_day=(2, 2), n_data_sets=4)


@dataclass(frozen=True)
class Session:
    """One session's identity and its summed bins (cells)."""

    key: tuple[str, str, str]
    animal: str
    day: str
    time: np.ndarray  # (n_cells,) the time since the visit's start, s
    exposure: np.ndarray  # (n_cells,) s
    n_events: float  # expected


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


def cohort(config: Config, seed: int = STRUCTURE_SEED) -> list[Session]:
    """The sessions of the cohort, their bins summed (module docstring)."""
    rng = np.random.default_rng(seed)
    lo, hi = config.sessions_per_day
    out = []
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
                out.append(
                    Session(
                        (animal, day, f"{2 * e + 2:02d}_r{e + 1}"),
                        animal,
                        day,
                        time,
                        exposure,
                        float(n_events),
                    )
                )
    return out


def simulate(
    sessions: list[Session], curve: str, rng: np.random.Generator
) -> dict[tuple[str, str, str], pd.DataFrame]:
    """Each session's cells with a Poisson count at the planted rate.

    A cell's count is drawn as one Poisson of the cell's summed mean: the
    sum of its bins' independent counts, whose rate is the same.

    Returns
    -------
    dict of session key -> pd.DataFrame
        Columns :data:`COLUMN`, ``exposure``, ``count`` and ``cell``.
    """
    out = {}
    for session in sessions:
        mean = session.exposure * np.exp(CURVES[curve](session.time))
        mean *= session.n_events / mean.sum()
        out[session.key] = pd.DataFrame(
            {
                COLUMN: session.time,
                "exposure": session.exposure,
                "count": rng.poisson(mean).astype(float),
                "cell": np.arange(session.time.size),
            }
        )
    return out


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


def planted_contrast(curve: str) -> float:
    """:data:`CONTRAST` of the planted curve, on the same grids."""
    f = CURVES[curve]
    (a, b), (c, d) = CONTRAST
    return float(f(range_grid(a, b)).mean() - f(range_grid(c, d)).mean())


def fit_contrasts(
    design: Design, data: dict[tuple[str, str, str], pd.DataFrame]
) -> pd.DataFrame:
    """Each session's contrast, edf and status at the design's penalty.

    Returns
    -------
    pd.DataFrame, shape (n_sessions, 3)
        ``estimate`` (NaN without a converged fit), ``edf`` and
        ``converged``, indexed by session key.
    """
    c = contrast_vector(design)
    records = fit_sessions(
        design,
        data,
        Poisson(),
        model="declared_penalty",
        kind="simulated",
        response="count",
        unit="cell",
        table_sha256={},
    )
    rows = []
    for record in records:
        converged = record.status == CONVERGED
        estimate = (
            evaluate_contrast(c, np.array(record.coef), np.array(record.vcov)).estimate
            if converged
            else np.nan
        )
        rows.append((estimate, record.edf.get(TERM, np.nan), converged))
    return pd.DataFrame(
        rows, columns=["estimate", "edf", "converged"], index=pd.Index(list(data))
    )


def summarize_data_set(
    estimates: pd.Series,
    sessions: list[Session],
    planted: float,
    n_draws: int,
    seed: int,
) -> dict[str, float]:
    """One data set's animal-balanced estimate, its interval's coverage of
    ``planted`` and the t test over animals."""
    animal = pd.Series([s.animal for s in sessions], index=estimates.index)
    day = pd.Series([s.day for s in sessions], index=estimates.index)
    session = pd.Series([s.key[2] for s in sessions], index=estimates.index)
    summary = estimand_summary(
        estimates, animal, day, session, n_draws=n_draws, seed=seed
    )
    per_animal = animal_day_session_values(estimates, animal, day, session)
    return {
        "estimate": summary["estimate"],
        "covered": float(summary["low"] <= planted <= summary["high"]),
        "t_p": float(scipy.stats.ttest_1samp(per_animal.to_numpy(), 0.0).pvalue),
    }


def run_data_set(
    task: tuple[str, int],
    sessions: list[Session],
    penalties: dict[float, float],
    n_draws: int,
) -> list[dict]:
    """One data set of a curve fitted at every target's declared penalty.

    Returns
    -------
    list of dict
        One per target: the data set's estimate, its interval's coverage, the
        t test's p, and each session's edf.
    """
    curve, index = task
    seed = np.random.SeedSequence([DATA_SEED, list(CURVES).index(curve), index])
    data = simulate(sessions, curve, np.random.default_rng(seed))
    pooled = pd.concat(data.values(), ignore_index=True)
    planted = planted_contrast(curve)
    out = []
    for target, log_penalty in penalties.items():
        fits = fit_contrasts(design_at(pooled, log_penalty), data)
        row = summarize_data_set(
            fits["estimate"], sessions, planted, n_draws, seed=DATA_SEED + index
        )
        out.append(
            {
                "target_edf": target,
                "curve": curve,
                "data_set": index,
                **row,
                "median_edf": float(np.nanmedian(fits["edf"])),
                "edf": fits["edf"].to_numpy(),
                "share_converged": float(fits["converged"].mean()),
            }
        )
    return out


def calibrate(target: float, sessions: list[Session]) -> tuple[float, float, float]:
    """The declared log penalty of a target edf, found on the calibration
    data set of the rise: ``(target, log penalty, median edf there)``."""
    data = simulate(sessions, "rise", np.random.default_rng(CALIBRATION_SEED))
    pooled = pd.concat(data.values(), ignore_index=True)
    found = log_penalty_for_edf(
        design_at(pooled, 0.0),
        data,
        Poisson(),
        term=TERM,
        target_edf=target,
        response="count",
        unit="cell",
    )
    return target, found.log_penalty, found.median_edf


def summarize(
    rows: list[dict],
    calibration: list[tuple[float, float, float]],
    config: Config,
    n_sessions: int,
) -> pd.DataFrame:
    """One row per target and curve (module docstring)."""
    frame = pd.DataFrame(rows)
    declared = {t: (rho, edf) for t, rho, edf in calibration}
    out = []
    for (target, curve), group in frame.groupby(["target_edf", "curve"], sort=False):
        planted = planted_contrast(curve)
        n = len(group)
        estimate = group["estimate"].to_numpy()
        bias = float(np.mean(estimate) - planted)
        reject = float(np.mean(group["t_p"] < 0.05))
        coverage = float(group["covered"].mean())
        out.append(
            {
                "target_edf": target,
                "curve": curve,
                "log_penalty": declared[target][0],
                "calibration_median_edf": declared[target][1],
                "median_edf": float(
                    np.nanmedian(np.concatenate(group["edf"].to_list()))
                ),
                "planted": planted,
                "mean_estimate": float(np.mean(estimate)),
                "bias": bias,
                "bias_share": bias / planted if planted != 0 else np.nan,
                "sd": float(np.std(estimate, ddof=1)),
                "bias_se": float(np.std(estimate, ddof=1) / np.sqrt(n)),
                "reject_t": reject,
                "reject_t_se": float(np.sqrt(reject * (1 - reject) / n)),
                "coverage": coverage,
                "coverage_se": float(np.sqrt(coverage * (1 - coverage) / n)),
                "share_converged": float(group["share_converged"].mean()),
                "n_data_sets": n,
                "n_sessions": n_sessions,
                "n_draws": config.n_draws,
            }
        )
    return pd.DataFrame(out)


#: The cohort, set before the worker processes fork, so that it is not sent
#: with every task.
_COHORT: list[Session] = []


def _calibrate(target: float) -> tuple[float, float, float]:
    return calibrate(target, _COHORT)


def _run_data_set(
    task: tuple[str, int], penalties: dict[float, float], n_draws: int
) -> list[dict]:
    return run_data_set(task, _COHORT, penalties, n_draws)


def run(config: Config, workers: int = 1) -> pd.DataFrame:
    """The table of the module docstring for ``config``, over ``workers``
    processes."""
    _COHORT[:] = cohort(config)
    tasks = [(curve, i) for curve in CURVES for i in range(config.n_data_sets)]
    with multiprocessing.get_context("fork").Pool(workers) as pool:
        calibration = pool.map(_calibrate, TARGETS)
        penalties = {target: rho for target, rho, _ in calibration}
        work = partial(_run_data_set, penalties=penalties, n_draws=config.n_draws)
        rows = [row for chunk in pool.map(work, tasks, chunksize=1) for row in chunk]
    return summarize(rows, calibration, config, len(_COHORT))


def main(argv: list[str] | None = None) -> None:
    default_jax_platforms("cpu")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--n-data-sets", type=int, default=None)
    args = parser.parse_args(argv)
    config = QUICK if args.quick else Config()
    if args.n_data_sets is not None:
        config = replace(config, n_data_sets=args.n_data_sets)
    started = time.perf_counter()
    table = run(config, args.workers)
    table.attrs["seconds"] = time.perf_counter() - started
    path = write_table(table, args.output_dir / OUTPUT_NAME)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(table.round(4).to_string(index=False))
    print(f"wrote {path} in {table.attrs['seconds']:.0f} s")


if __name__ == "__main__":
    main()
