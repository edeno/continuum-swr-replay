"""Summary statistics of the per-bin event-rate ladder's fits.

Each function takes per-session fits of the per-bin conditional-intensity
ladder (``MODEL_PER_BIN_RATE``; one row per session, rung and population, as
``scripts/rate/fit_per_bin_rate_models.py`` writes them) and returns named
numbers: the frozen cohort, the baseline rate, the reconstructed reward kernel,
each rung's fit quality, and the fitted time course on the stop and arrival
clocks.

The reward effect is the kernel reconstructed per session from the three
reward-basis coefficients, read over the sessions whose three reward terms all
pass the per-term gate
(:func:`~continuum_swr_replay.analysis.replay.contrast_aggregation.gate_term_sessions`);
a single basis coefficient is not the effect of a reward at any lag. Rates are
events per at-risk second, kernels are rate ratios, times are in seconds.
Summaries of one coefficient, and of one rung against another, are in
:mod:`~continuum_swr_replay.analysis.replay.coefficient_statistics`.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from patsy import dmatrix
from scipy.stats import wilcoxon

from continuum_swr_replay.analysis.replay.coefficient_statistics import (
    require_one_fit_per_session,
    separated_fraction,
    session_median_statistics,
)
from continuum_swr_replay.analysis.replay.contrast_aggregation import (
    gate_term_sessions,
)
from continuum_swr_replay.analysis.replay.model_specs import (
    BIN_S,
    REWARD_HISTORY_COLUMNS,
    REWARD_HISTORY_N_BASIS,
    REWARD_HISTORY_N_LAGS,
    time_course_spline,
)
from continuum_swr_replay.analysis.replay.per_bin_design import (
    reconstruct_reward_kernels,
)
from continuum_swr_replay.tables.session_keys import SESSION_COLUMNS, n_sessions

TIME_COURSE_GRID_POINTS = 401
"""The evenly spaced times a fitted time course is evaluated at."""

FIRST_LATE_REWARD_LAG = 3
"""The first lag of the reward kernel's late part, read apart from lags 1 and 2."""

_REWARD_COEFFICIENTS = [f"coef_{column}" for column in REWARD_HISTORY_COLUMNS]

# --- the cohort and the baseline rate -------------------------------------------------


def frozen_cohort_statistics(null_fits: pd.DataFrame) -> dict[str, int]:
    """The size of the common cohort every rung of the ladder is fit on.

    Every population is fit on the same stops. The at-risk bins are the
    detector's: the non-local populations share theirs (no bin inside a
    non-local event), and detected SWR (``outcome`` ``"swr"``) has its own (no
    bin inside an SWR).

    Parameters
    ----------
    null_fits : pd.DataFrame
        The constant-rate rung's fits, one row per session and population, with
        ``outcome``, ``n_stops`` and ``n_bins``.

    Returns
    -------
    statistics : dict
        ``n_sessions``; ``n_stops``, the stops summed over sessions;
        ``n_bins_non_local`` and ``n_bins_swr``, the at-risk bins of the
        non-local populations and of detected SWR, summed over sessions.

    Raises
    ------
    ValueError
        If the populations' stops, summed over sessions, differ, or the
        non-local populations' summed bins. The totals are compared, not each
        session's counts.
    """
    stops = null_fits.groupby("outcome")["n_stops"].sum()
    bins = null_fits.groupby("outcome")["n_bins"].sum()
    non_local_bins = bins.drop("swr")
    if stops.nunique() != 1 or non_local_bins.nunique() != 1:
        raise ValueError("populations differ in cohort")
    return {
        "n_sessions": n_sessions(null_fits),
        "n_stops": int(stops.iloc[0]),
        "n_bins_non_local": int(non_local_bins.iloc[0]),
        "n_bins_swr": int(bins["swr"]),
    }


def baseline_rate_statistics(null_fits: pd.DataFrame) -> dict[str, float]:
    """The constant-rate rung's rate over sessions, for one population.

    Parameters
    ----------
    null_fits : pd.DataFrame
        The constant-rate rung's fits of one population, one row per session,
        with the session columns and ``coef_Intercept`` (log events per
        at-risk second).

    Returns
    -------
    statistics : dict
        ``median``, ``lower_quartile`` and ``upper_quartile`` of the sessions'
        ``exp(intercept)``, in events per at-risk second, and
        ``events_per_bin``, the median rate times the bin width.
    """
    require_one_fit_per_session(null_fits)
    rate = np.exp(null_fits["coef_Intercept"].dropna())
    return {
        "median": float(rate.median()),
        "lower_quartile": float(rate.quantile(0.25)),
        "upper_quartile": float(rate.quantile(0.75)),
        "events_per_bin": float(rate.median() * BIN_S),
    }


def baseline_rate_ratio_statistics(
    null_a: pd.DataFrame, null_b: pd.DataFrame
) -> dict[str, float]:
    """One population's constant rate over another's, within session.

    Parameters
    ----------
    null_a, null_b : pd.DataFrame
        The constant-rate rung's fits of each population, with the session
        columns and ``coef_Intercept``; at most one row per session.

    Returns
    -------
    statistics : dict
        Over the sessions with both fits: ``n_sessions``; ``median_ratio``,
        ``exp`` of the median difference of the log rates (a over b); and
        ``signed_rank_p``, the paired Wilcoxon signed-rank p of the log rates.
    """
    session = list(SESSION_COLUMNS)
    columns = [*session, "coef_Intercept"]
    merged = (
        null_a[columns]
        .merge(null_b[columns], on=session, suffixes=("_a", "_b"), validate="1:1")
        .dropna()
    )
    difference = merged["coef_Intercept_a"] - merged["coef_Intercept_b"]
    return {
        "n_sessions": len(merged),
        "median_ratio": float(np.exp(difference.median())),
        "signed_rank_p": float(
            wilcoxon(merged["coef_Intercept_a"], merged["coef_Intercept_b"]).pvalue
        ),
    }


# --- the reward kernel ----------------------------------------------------------------


def reward_gated_fits(fits: pd.DataFrame) -> pd.DataFrame:
    """The fits whose three reward-basis coefficients all pass the per-term gate.

    A fit is left out when any of the three terms is flagged separated or ran
    away, since a kernel reconstructed from it inherits that term. Convergence
    is not part of the gate.

    Parameters
    ----------
    fits : pd.DataFrame
        Per-session fits of one rung and population that carries the reward
        block, one row per session, with the session columns.

    Returns
    -------
    gated : pd.DataFrame
        The kept rows of ``fits``.
    """
    require_one_fit_per_session(fits)
    kept = np.ones(len(fits), dtype=bool)
    for term in REWARD_HISTORY_COLUMNS:
        kept &= gate_term_sessions(fits, term).kept.to_numpy()
    return fits[kept]


def _gated_reward_kernels(fits: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """The reward-gated fits (:func:`reward_gated_fits`) and each one's
    reconstructed kernel: log rate ratios, shape (n_gated_sessions, n_lags)."""
    gated = reward_gated_fits(fits)
    kernels = reconstruct_reward_kernels(
        gated[_REWARD_COEFFICIENTS].to_numpy(dtype=float),
        REWARD_HISTORY_N_LAGS,
        REWARD_HISTORY_N_BASIS,
    )
    return gated, kernels


def lag_one_reward_kernels(fits: pd.DataFrame) -> pd.DataFrame:
    """Each gated session's reconstructed reward kernel at lag 1.

    Parameters
    ----------
    fits : pd.DataFrame
        Per-session fits of one rung and population (:func:`reward_gated_fits`),
        with the session columns and ``converged``.

    Returns
    -------
    kernels : pd.DataFrame, shape (n_gated_sessions, 5)
        The session columns, ``converged`` and ``lag_one``: the log rate ratio
        of a reward one trial back.
    """
    gated, kernels = _gated_reward_kernels(fits)
    return gated[[*SESSION_COLUMNS, "converged"]].assign(lag_one=kernels[:, 0])


def reward_kernel_statistics(fits: pd.DataFrame) -> dict[str, float]:
    """The reconstructed reward kernel of one rung and population.

    Parameters
    ----------
    fits : pd.DataFrame
        Per-session fits of one rung and population that carries the reward
        block (:func:`lag_one_reward_kernels`).

    Returns
    -------
    statistics : dict
        Over the sessions the reward gate keeps (:func:`reward_gated_fits`):

        - ``n_sessions``: those sessions.
        - ``n_dropped``: fitted sessions the gate left out.
        - ``n_nonconverged``: kept sessions whose fit did not converge.
        - ``rate_ratio_lag_<k>``, for each lag ``k`` from 1: the median over
          sessions of the session's kernel at that lag, as a rate ratio.
        - ``late_rate_ratio_min``, ``late_rate_ratio_max``: the lowest and
          highest of those medians from lag :data:`FIRST_LATE_REWARD_LAG` on.
        - ``lag_one_positive_fraction``: the share of sessions whose lag-1
          kernel is above a rate ratio of 1.
        - ``lag_one_signed_rank_p``: the Wilcoxon signed-rank p of the
          sessions' log lag-1 kernel against 0.
        - ``lag_one_animal_rate_ratio``: ``exp`` of the median over animals of
          each animal's median log lag-1 kernel.
        - ``n_animals``: the animals with a kept session.
        - ``n_animals_lag_one_positive``: animals whose median lag-1 kernel is
          above a rate ratio of 1.
    """
    gated, kernels = _gated_reward_kernels(fits)
    lag_one = pd.Series(kernels[:, 0], index=gated.index)
    lag_one_statistics = session_median_statistics(lag_one, gated["animal"])
    rate_ratios = np.median(np.exp(kernels), axis=0)
    late = rate_ratios[FIRST_LATE_REWARD_LAG - 1 :]
    n_fitted = int(fits[_REWARD_COEFFICIENTS].notna().all(axis=1).sum())
    return {
        "n_sessions": len(gated),
        "n_dropped": n_fitted - len(gated),
        "n_nonconverged": int((gated["converged"] == False).sum()),  # noqa: E712
        **{
            f"rate_ratio_lag_{lag}": float(rate_ratio)
            for lag, rate_ratio in enumerate(rate_ratios, start=1)
        },
        "late_rate_ratio_min": float(late.min()),
        "late_rate_ratio_max": float(late.max()),
        "lag_one_positive_fraction": lag_one_statistics["positive_fraction"],
        "lag_one_signed_rank_p": float(wilcoxon(lag_one).pvalue),
        "lag_one_animal_rate_ratio": float(np.exp(lag_one_statistics["animal_median"])),
        "n_animals": lag_one_statistics["n_animals"],
        "n_animals_lag_one_positive": lag_one_statistics["n_animals_positive"],
    }


def paired_lag_one_kernel_statistics(
    fits_a: pd.DataFrame, fits_b: pd.DataFrame
) -> dict[str, float]:
    """The lag-1 reward kernel of one population against another's, within session.

    Parameters
    ----------
    fits_a, fits_b : pd.DataFrame
        Per-session fits of one rung, each of one population
        (:func:`lag_one_reward_kernels`).

    Returns
    -------
    statistics : dict
        Over the sessions in which both populations' fits pass the reward gate:
        ``n_sessions``; ``positive_fraction``, the share whose log lag-1 kernel
        is larger in a than in b; ``n_animals``; ``n_animals_positive``, the
        animals whose median a - b difference is positive; and
        ``signed_rank_p``, the paired Wilcoxon signed-rank p.
    """
    merged = lag_one_reward_kernels(fits_a).merge(
        lag_one_reward_kernels(fits_b),
        on=list(SESSION_COLUMNS),
        suffixes=("_a", "_b"),
        validate="1:1",
    )
    difference = session_median_statistics(
        merged["lag_one_a"] - merged["lag_one_b"], merged["animal"]
    )
    return {
        "n_sessions": difference["n_sessions"],
        "positive_fraction": difference["positive_fraction"],
        "n_animals": difference["n_animals"],
        "n_animals_positive": difference["n_animals_positive"],
        "signed_rank_p": float(
            wilcoxon(merged["lag_one_a"], merged["lag_one_b"]).pvalue
        ),
    }


# --- each rung's fits -----------------------------------------------------------------


def ladder_fit_statistics(fits: pd.DataFrame) -> pd.DataFrame:
    """How many sessions each rung fit, and how well, for each population.

    Parameters
    ----------
    fits : pd.DataFrame
        Per-session fits of the ladder, with ``model``, ``outcome``,
        ``coef_Intercept``, ``aic``, ``separated``, ``cv_bits_per_event`` (the
        held-out score) and ``ks_within_band`` (whether the rescaled waiting
        times stay inside the KS band).

    Returns
    -------
    statistics : pd.DataFrame
        One row per rung and population, indexed by ``model`` and ``outcome``:

        - ``n_fitted``: sessions with a fit (an intercept).
        - ``n_finite_aic``: sessions with a finite AIC.
        - ``aic_median``: the median AIC over the sessions whose AIC is not
          missing; an infinite AIC is one of them.
        - ``separated_fraction``: the share of all the fits flagged separated.
        - ``n_cv_finite``: sessions with a finite held-out score.
        - ``cv_bits_median``: the median held-out bits per event over those.
        - ``cv_not_finite_fraction``: the share of all the fits whose held-out
          score is not finite.
        - ``ks_band_fraction``: of the fits that carry the KS flag, the share
          inside the band. A fit without the flag is left out of this share,
          though it counts in the two shares above.
    """
    rows = {}
    for rung, rung_fits in fits.groupby(["model", "outcome"]):
        cv = rung_fits["cv_bits_per_event"]
        finite_cv = cv[np.isfinite(cv)]
        rows[rung] = {
            "n_fitted": int(rung_fits["coef_Intercept"].notna().sum()),
            "n_finite_aic": int(np.isfinite(rung_fits["aic"]).sum()),
            "aic_median": rung_fits["aic"].median(),
            "separated_fraction": separated_fraction(rung_fits),
            "n_cv_finite": len(finite_cv),
            "cv_bits_median": finite_cv.median(),
            "cv_not_finite_fraction": float((~np.isfinite(cv)).mean()),
            "ks_band_fraction": float(rung_fits["ks_within_band"].mean()),
        }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis(
        ["model", "outcome"]
    )


def arrival_window_statistics(cohort: Mapping[str, int]) -> dict[str, float]:
    """What the arrival clock and its window keep of one population's cohort.

    Parameters
    ----------
    cohort : mapping of str to int
        The per-bin runner's arrival-clock record of one population:
        ``n_bins_frozen`` and ``n_event_bins_frozen`` (bins and event bins of
        the frozen common cohort), ``n_bins_dropped_arrival_clock`` (bins the
        arrival clock's missingness removed from that cohort), and
        ``n_bins_arrival_window`` and ``n_event_bins_arrival_window`` (those of
        the frozen cohort inside the arrival window).

    Returns
    -------
    statistics : dict
        - ``event_share``: the share of the frozen cohort's event bins inside
          the arrival window.
        - ``bin_share``: the share of its bins inside the window.
        - ``n_bins_before``: the bins before the arrival clock removed any.
        - ``dropped_bin_fraction``: the share of those the clock removed.
    """
    before = cohort["n_bins_frozen"] + cohort["n_bins_dropped_arrival_clock"]
    return {
        "event_share": cohort["n_event_bins_arrival_window"]
        / cohort["n_event_bins_frozen"],
        "bin_share": cohort["n_bins_arrival_window"] / cohort["n_bins_frozen"],
        "n_bins_before": before,
        "dropped_bin_fraction": cohort["n_bins_dropped_arrival_clock"] / before,
    }


# --- the fitted time course -----------------------------------------------------------


def fitted_time_course(
    fits: pd.DataFrame, time_clock: str
) -> tuple[np.ndarray, np.ndarray, int]:
    """The across-session median fitted rate over a clock's supported interior.

    The curve is the fitted intercept and time-course spline of the ladder
    (:func:`~continuum_swr_replay.analysis.replay.model_specs.time_course_spline`),
    evaluated on a grid. The interior is the span of the spline's outermost
    interior knots: outside it the spline is held by few bins. A session
    flagged separated, or whose curve is not finite, is left out.

    Parameters
    ----------
    fits : pd.DataFrame
        Per-session fits of one time rung and population, one row per session,
        with the session columns, ``coef_Intercept``, ``separated`` and the
        coefficient of each basis column of the clock's spline
        (``coef_<term>[k]``).
    time_clock : str
        ``"stop"`` or ``"arrival"``: the clock the rung's spline is on.

    Returns
    -------
    grid : np.ndarray, shape (TIME_COURSE_GRID_POINTS,)
        Times on the clock from the first to the last interior knot, in seconds.
    curve : np.ndarray, shape (TIME_COURSE_GRID_POINTS,)
        The median over sessions of the fitted rate at each time, in events per
        at-risk second.
    n_curves : int
        The session curves the median is over.

    Raises
    ------
    ValueError
        If no session is left with a curve: there is no time course to
        summarize, and its peak would read as the start of the interval.
    """
    require_one_fit_per_session(fits)
    term, column, knots = time_course_spline(time_clock)
    grid = np.linspace(knots[0], knots[-1], TIME_COURSE_GRID_POINTS)
    basis = np.asarray(dmatrix(f"{term} - 1", {column: grid}, return_type="dataframe"))
    beta = fits[[f"coef_{term}[{k}]" for k in range(basis.shape[1])]].to_numpy(
        dtype=float
    )
    intercept = fits["coef_Intercept"].to_numpy(dtype=float)
    ok = np.isfinite(beta).all(axis=1) & np.isfinite(intercept)
    ok &= ~(fits["separated"] == True).to_numpy()  # noqa: E712 (NaN must not count)
    with np.errstate(over="ignore"):
        curves = np.exp(intercept[ok, None] + beta[ok] @ basis.T)
    curves = curves[np.isfinite(curves).all(axis=1)]
    if not len(curves):
        raise ValueError(f"no session has a usable {time_clock}-clock time course")
    return grid, np.median(curves, axis=0), len(curves)


def _shape_cdf(grid: np.ndarray, curve: np.ndarray) -> np.ndarray:
    """The curve's cumulative area over the grid (trapezoids), scaled to end at 1."""
    increments = np.diff(grid) * 0.5 * (curve[1:] + curve[:-1])
    cdf = np.concatenate([[0.0], np.cumsum(increments)])
    return cdf / cdf[-1]


def clock_shape_statistics(
    stop_fits: pd.DataFrame, arrival_fits: pd.DataFrame
) -> dict[str, float]:
    """Where the fitted rate peaks on the stop clock and on the arrival clock.

    Parameters
    ----------
    stop_fits : pd.DataFrame
        Per-session fits of the stop-clock time rung for one population
        (:func:`fitted_time_course`).
    arrival_fits : pd.DataFrame
        Per-session fits of the arrival-clock time rung for that population.

    Returns
    -------
    statistics : dict
        For the stop clock (``time``) and the arrival clock (``arrival``):

        - ``<clock>_peak_s``: the time at which the median fitted time course
          is highest, in seconds on that clock.
        - ``<clock>_peak_edge``: 1 if that time is an end of the supported
          interval (the peak is then bounded, not located), else 0.
        - ``<clock>_peak_sessions``: the session curves behind the median.
        - ``<clock>_interval_start_s``, ``<clock>_interval_end_s``: the ends
          of the supported interval the time course is read on, in seconds on
          that clock.

        And ``clock_ks``: the largest distance between the two time courses'
        cumulative shapes, each normalized over its own supported interval and
        read on relative position.
    """
    grid, stop_curve, n_stop = fitted_time_course(stop_fits, "stop")
    arrival_grid, arrival_curve, n_arrival = fitted_time_course(arrival_fits, "arrival")
    ks = np.max(
        np.abs(_shape_cdf(grid, stop_curve) - _shape_cdf(arrival_grid, arrival_curve))
    )
    stop_i, arrival_i = int(np.argmax(stop_curve)), int(np.argmax(arrival_curve))
    return {
        "time_peak_s": float(grid[stop_i]),
        "time_peak_edge": int(stop_i in (0, len(grid) - 1)),
        "time_peak_sessions": n_stop,
        "time_interval_start_s": float(grid[0]),
        "time_interval_end_s": float(grid[-1]),
        "arrival_peak_s": float(arrival_grid[arrival_i]),
        "arrival_peak_edge": int(arrival_i in (0, len(arrival_grid) - 1)),
        "arrival_peak_sessions": n_arrival,
        "arrival_interval_start_s": float(arrival_grid[0]),
        "arrival_interval_end_s": float(arrival_grid[-1]),
        "clock_ks": float(ks),
    }
