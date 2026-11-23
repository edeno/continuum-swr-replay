"""Planted scenarios of the visit rate model, as pure functions of the rows.

Each scenario maps one session's rows of the model (``visit_rate.cohort_rows``:
the clocks, ``exposure``, ``last_outcome_rewarded``) to the expected count of
each bin, and gives the planted value of each declared estimand of the model
(Q1, Q2, Q7a) under it. A scenario draws nothing and fits nothing: a
simulation draws the counts from the expected ones and runs the real fit.

The planted rate of a bin is ``exposure * exp(log rate)``. In every scenario
but the rate table, the log rate is ``b0 + f_s(s) + f_u(u) [+ f_m(m)] + g *
rewarded``, with the clock curves of :class:`PlantedClocks` and ``b0`` set so
that the session's expected total is ``total`` (when given; else ``b0 = 0``).

The planted values:

- Q1 and Q7a, from the planted curves: the range means of
  ``visit_rate.declared_contrasts``'s windows on the same grid
  (``contrasts.range_grid``), so the curves' additive constants cancel. Q1's
  reference window is an argument (the declared ``(0, 1)`` by default).
- Q2, the planted coefficient where one is planted (no effect: 0; a constant
  ratio: its log). Where the effect changes with time since arrival there is
  no coefficient, and Q2 is defined as the log of the rewarded bins' expected
  events over their expected events without the effect,
  ``log(sum mu / sum mu0)`` over the rewarded bins (:func:`fading_ratio`).
  Where the outcomes differ only in their arrival curve (a later onset after
  an omission), there is no planted reward ratio: NaN.
- The rate table (:func:`rate_table_truth`) plants no curve and no
  coefficient; each value is defined from the planted rates of the session's
  bins (see there).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.regression.contrasts import range_grid
from continuum_swr_replay.tables.session_keys import SESSION_COLUMNS

from ..declared_tests import estimand
from .visit_rate import ARRIVAL, EXPOSURE, LEAVING, RESPONSE, REWARD, STRETCH

__all__ = [
    "Planted",
    "PlantedClocks",
    "RateTable",
    "constant_ratio",
    "differing_ratio",
    "fading_ratio",
    "interpolated_curve",
    "later_onset_after_omission",
    "no_effect",
    "rate_table_from_bins",
    "rate_table_truth",
    "session_log_ratios",
]

LogCurve = Callable[[np.ndarray], np.ndarray]

#: The declared windows of Q1 and Q7a, read from ``declared_tests``.
_Q1 = estimand("Q1").contrast
_Q7A = estimand("Q7a").contrast


def interpolated_curve(x: np.ndarray, log_rate: np.ndarray) -> LogCurve:
    """A log-rate curve through points, linear between them and flat beyond.

    Parameters
    ----------
    x : np.ndarray, shape (n_points,)
        Ascending clock values (s).
    log_rate : np.ndarray, shape (n_points,)

    Returns
    -------
    callable
        Clock values -> log rate, the same shape.
    """
    x = np.asarray(x, dtype=float)
    log_rate = np.asarray(log_rate, dtype=float)
    if x.ndim != 1 or x.shape != log_rate.shape or np.any(np.diff(x) <= 0):
        raise ValueError("x ascending, one log rate per point")
    return lambda values: np.interp(np.asarray(values, dtype=float), x, log_rate)


@dataclass(frozen=True)
class PlantedClocks:
    """The planted log-rate curves of the clocks.

    Attributes
    ----------
    arrival : callable
        ``f_s``, over time since arrival (s).
    leaving : callable
        ``f_u``, over time until leaving (s).
    stretch : callable or None
        ``f_m``, over time until the end of the stretch (s); None is flat.
    """

    arrival: LogCurve
    leaving: LogCurve
    stretch: LogCurve | None = None

    def log_rate(self, rows: pd.DataFrame, arrival_s: np.ndarray | None = None):
        """``f_s + f_u [+ f_m]`` at each row, shape ``(n_rows,)``; the arrival
        curve at ``arrival_s`` in place of the rows' clock when given."""
        s = rows[ARRIVAL].to_numpy(dtype=float) if arrival_s is None else arrival_s
        out = self.arrival(s) + self.leaving(rows[LEAVING].to_numpy(dtype=float))
        if self.stretch is not None:
            out = out + self.stretch(rows[STRETCH].to_numpy(dtype=float))
        return out

    def curve_values(
        self, q1_reference_s: tuple[float, float] = _Q1.second
    ) -> dict[str, float]:
        """Planted Q1 and Q7a from the curves (module docstring)."""
        return {
            "Q1": _range_contrast(self.arrival, _Q1.first, q1_reference_s),
            "Q7a": _range_contrast(self.leaving, _Q7A.first, _Q7A.second),
        }


def _range_contrast(
    curve: LogCurve, first: tuple[float, float], second: tuple[float, float]
) -> float:
    """The mean of ``curve`` over ``first`` less its mean over ``second``,
    on ``contrasts.range_grid``."""
    return float(
        np.mean(curve(range_grid(*first))) - np.mean(curve(range_grid(*second)))
    )


class Planted(NamedTuple):
    """A scenario in one session.

    Attributes
    ----------
    expected : np.ndarray, shape (n_rows,)
        The expected count of each bin.
    values : dict of str to float
        The planted Q1, Q2 and Q7a; NaN where the scenario has none.
    """

    expected: np.ndarray
    values: dict[str, float]


def _scaled(rows: pd.DataFrame, log_rate: np.ndarray, total: float | None):
    """``exposure * exp(log_rate)``, scaled to ``total`` when given."""
    expected = rows[EXPOSURE].to_numpy(dtype=float) * np.exp(log_rate)
    if total is not None:
        if not (total > 0 and expected.sum() > 0):
            raise ValueError("a positive total needs rows with exposure")
        expected = expected * (total / expected.sum())
    return expected


def _rewarded(rows: pd.DataFrame) -> np.ndarray:
    reward = rows[REWARD].to_numpy(dtype=float)
    if not np.isin(reward, (0.0, 1.0)).all():
        raise ValueError(f"{REWARD} must be 0 or 1 on every row")
    return reward


def _with_ratio(
    rows: pd.DataFrame,
    clocks: PlantedClocks,
    g: np.ndarray,
    q2: float,
    total: float | None,
    q1_reference_s: tuple[float, float],
) -> Planted:
    log_rate = clocks.log_rate(rows) + g * _rewarded(rows)
    return Planted(
        _scaled(rows, log_rate, total),
        {**clocks.curve_values(q1_reference_s), "Q2": float(q2)},
    )


def no_effect(
    rows: pd.DataFrame,
    clocks: PlantedClocks,
    *,
    total: float | None = None,
    q1_reference_s: tuple[float, float] = _Q1.second,
) -> Planted:
    """No reward effect: the clocks only (planted Q2 is 0)."""
    return constant_ratio(
        rows, clocks, log_ratio=0.0, total=total, q1_reference_s=q1_reference_s
    )


def constant_ratio(
    rows: pd.DataFrame,
    clocks: PlantedClocks,
    *,
    log_ratio: float,
    total: float | None = None,
    q1_reference_s: tuple[float, float] = _Q1.second,
) -> Planted:
    """A constant rate ratio after a reward: ``g = log_ratio`` (planted Q2)."""
    g = np.full(len(rows), float(log_ratio))
    return _with_ratio(rows, clocks, g, log_ratio, total, q1_reference_s)


def fading_ratio(
    rows: pd.DataFrame,
    clocks: PlantedClocks,
    *,
    log_ratio: float,
    fade_start_s: float,
    fade_end_s: float,
    total: float | None = None,
    q1_reference_s: tuple[float, float] = _Q1.second,
) -> Planted:
    """A ratio that fades with time since arrival: ``g(s) = log_ratio`` up to
    ``fade_start_s``, falling linearly to 0 at ``fade_end_s``, 0 after.

    Planted Q2 is ``log(sum mu / sum mu0)`` over the rewarded bins, ``mu0``
    their expected counts without the effect (module docstring).
    """
    if not fade_end_s > fade_start_s:
        raise ValueError("the fade ends after it starts")
    s = rows[ARRIVAL].to_numpy(dtype=float)
    g = log_ratio * np.clip((fade_end_s - s) / (fade_end_s - fade_start_s), 0.0, 1.0)
    rewarded = _rewarded(rows) == 1.0
    mu0 = rows[EXPOSURE].to_numpy(dtype=float) * np.exp(clocks.log_rate(rows))
    q2 = (
        float(np.log(np.sum(mu0[rewarded] * np.exp(g[rewarded])) / mu0[rewarded].sum()))
        if mu0[rewarded].sum() > 0
        else np.nan
    )
    return _with_ratio(rows, clocks, g, q2, total, q1_reference_s)


def session_log_ratios(
    sessions: pd.DataFrame,
    *,
    mean: float,
    animal_sd: float,
    session_sd: float,
    seed: int,
) -> dict[tuple[str, str, str], float]:
    """Per-session log ratios that differ between sessions and animals:
    ``mean + a_animal + e_session``, ``a ~ N(0, animal_sd^2)`` and ``e ~ N(0,
    session_sd^2)``, one draw (``numpy.random.default_rng(seed)``: the animals
    in sorted order, then the sessions in sorted order).

    Parameters
    ----------
    sessions : pd.DataFrame
        The session columns, one row per session.

    Returns
    -------
    dict of (animal, session_date, epoch) to float
    """
    keys = sorted(
        set(sessions[list(SESSION_COLUMNS)].astype(str).itertuples(index=False))
    )
    rng = np.random.default_rng(seed)
    animals = sorted({key[0] for key in keys})
    animal_term = dict(
        zip(animals, rng.normal(0.0, animal_sd, len(animals)), strict=True)
    )
    session_term = rng.normal(0.0, session_sd, len(keys))
    return {
        tuple(key): float(mean + animal_term[key[0]] + e)
        for key, e in zip(keys, session_term, strict=True)
    }


def differing_ratio(
    rows: pd.DataFrame,
    clocks: PlantedClocks,
    *,
    log_ratios: Mapping[tuple[str, str, str], float],
    total: float | None = None,
    q1_reference_s: tuple[float, float] = _Q1.second,
) -> Planted:
    """An effect that differs between sessions and animals: the session's own
    constant log ratio (:func:`session_log_ratios`), its planted Q2."""
    keys = set(rows[list(SESSION_COLUMNS)].astype(str).itertuples(index=False))
    if len(keys) != 1:
        raise ValueError(f"the rows of one session, not {len(keys)}")
    return constant_ratio(
        rows,
        clocks,
        log_ratio=log_ratios[tuple(keys.pop())],
        total=total,
        q1_reference_s=q1_reference_s,
    )


def later_onset_after_omission(
    rows: pd.DataFrame,
    clocks: PlantedClocks,
    *,
    delay_s: float,
    total: float | None = None,
    q1_reference_s: tuple[float, float] = _Q1.second,
) -> Planted:
    """A later onset after an omission: after an omission the arrival curve
    is the reward side's ``delay_s`` later, ``f_s(max(s - delay_s, 0))``; no
    reward ratio. Planted Q1 and Q7a are the reward side's curves'; Q2 is NaN
    (no ratio is planted)."""
    s = rows[ARRIVAL].to_numpy(dtype=float)
    omitted = _rewarded(rows) == 0.0
    arrival = np.where(omitted, np.maximum(s - delay_s, 0.0), s)
    log_rate = clocks.log_rate(rows, arrival_s=arrival)
    return Planted(
        _scaled(rows, log_rate, total),
        {**clocks.curve_values(q1_reference_s), "Q2": np.nan},
    )


@dataclass(frozen=True)
class RateTable:
    """Events per at-risk second by outcome and the two clocks.

    Attributes
    ----------
    rates : np.ndarray, shape (2, n_arrival, n_leaving)
        Index 0 after an omission, 1 after a reward.
    arrival_edges, leaving_edges : np.ndarray
        Ascending cell edges (s), shape ``(n_arrival + 1,)`` and
        ``(n_leaving + 1,)``; the last edge may be ``inf``. A cell is
        ``[edges[i], edges[i + 1])``.
    """

    rates: np.ndarray
    arrival_edges: np.ndarray
    leaving_edges: np.ndarray

    def cells(self, rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Each row's (outcome, arrival cell, leaving cell)."""
        out = []
        for column, edges in (
            (ARRIVAL, self.arrival_edges),
            (LEAVING, self.leaving_edges),
        ):
            value = rows[column].to_numpy(dtype=float)
            index = np.searchsorted(edges, value, side="right") - 1
            if np.any((index < 0) | (index >= edges.size - 1)):
                raise ValueError(f"{column} outside the table's edges")
            out.append(index)
        return _rewarded(rows).astype(int), out[0], out[1]


def rate_table_from_bins(
    bins: pd.DataFrame,
    arrival_edges: np.ndarray,
    leaving_edges: np.ndarray,
    *,
    shrink_s: float,
    floor: float,
) -> RateTable:
    """A rate table from bins: each cell's events per at-risk second, shrunk
    toward its outcome and arrival cell's rate over the leaving cells with
    weight ``t / (t + shrink_s)`` on its own rate (``t`` its at-risk
    seconds), then floored at ``floor``.

    Parameters
    ----------
    bins : pd.DataFrame
        The clocks, ``exposure``, ``count`` and ``last_outcome_rewarded``.
    arrival_edges, leaving_edges : np.ndarray
        As :class:`RateTable`.
    shrink_s, floor : float

    Returns
    -------
    RateTable
    """
    arrival_edges = np.asarray(arrival_edges, dtype=float)
    leaving_edges = np.asarray(leaving_edges, dtype=float)
    table = RateTable(np.empty(0), arrival_edges, leaving_edges)
    shape = (2, arrival_edges.size - 1, leaving_edges.size - 1)
    flat = np.ravel_multi_index(table.cells(bins), shape)
    n_cells = int(np.prod(shape))
    n = np.bincount(flat, bins[RESPONSE].to_numpy(dtype=float), n_cells).reshape(shape)
    t = np.bincount(flat, bins[EXPOSURE].to_numpy(dtype=float), n_cells).reshape(shape)
    margin_t = t.sum(axis=2, keepdims=True)
    margin = np.divide(
        n.sum(axis=2, keepdims=True),
        margin_t,
        out=np.zeros_like(margin_t),
        where=margin_t > 0,
    )
    own = np.divide(n, t, out=np.zeros_like(n), where=t > 0)
    weight = t / (t + shrink_s)
    rates = np.maximum(weight * own + (1.0 - weight) * margin, floor)
    return RateTable(rates, arrival_edges, leaving_edges)


def _log_rate_ratio(expected: np.ndarray, exposure: np.ndarray, a, b) -> float:
    """``log`` of the planted rate (expected over exposure) on rows ``a``
    less on rows ``b``; NaN when either holds no exposure."""
    if exposure[a].sum() <= 0 or exposure[b].sum() <= 0:
        return np.nan
    return float(
        np.log(expected[a].sum() / exposure[a].sum())
        - np.log(expected[b].sum() / exposure[b].sum())
    )


def _window(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (x >= lo) & (x <= hi)


def rate_table_truth(
    rows: pd.DataFrame,
    table: RateTable,
    *,
    total: float | None = None,
    q1_reference_s: tuple[float, float] = _Q1.second,
    reward_steps_s: tuple[float, float] = (1.0, 6.0),
    step_s: float = 0.25,
) -> Planted:
    """A truth taken from a table of rates by outcome and both clocks.

    Each bin's planted rate is its cell's (scaled to ``total`` when given).
    The planted values are defined from the planted rates of the session's
    bins, the rate of a set of bins being its expected events over its
    at-risk seconds (windows closed at both ends):

    - Q1: the log rate of the rewarded bins 2 to 4 s after arrival less that
      of the rewarded bins in the reference window;
    - Q7a: among the rewarded bins at least 2 s after arrival, the log rate
      in the last 0.25 s before leaving less that 1 to 2 s before;
    - Q2: the mean, over the steps of ``step_s`` from ``reward_steps_s[0]``
      to ``reward_steps_s[1]`` s since arrival (1 to 6 s), of the log rate
      after a reward less after an omission; a step without at-risk time on
      a side is left out.

    These differ from the model's coefficient definitions even where the
    model is right (a log of a mean is not a mean of logs, and the outcomes
    differ in their time until leaving).
    """
    outcome, arrival_cell, leaving_cell = table.cells(rows)
    expected = _scaled(
        rows, np.log(table.rates[outcome, arrival_cell, leaving_cell]), total
    )
    exposure = rows[EXPOSURE].to_numpy(dtype=float)
    s = rows[ARRIVAL].to_numpy(dtype=float)
    u = rows[LEAVING].to_numpy(dtype=float)
    rewarded = outcome == 1
    q1 = _log_rate_ratio(
        expected,
        exposure,
        rewarded & _window(s, *_Q1.first),
        rewarded & _window(s, *q1_reference_s),
    )
    late = rewarded & (s >= 2.0)
    q7a = _log_rate_ratio(
        expected,
        exposure,
        late & _window(u, *_Q7A.first),
        late & _window(u, *_Q7A.second),
    )
    steps = [
        _log_rate_ratio(
            expected,
            exposure,
            rewarded & (s >= lo) & (s < lo + step_s),
            ~rewarded & (s >= lo) & (s < lo + step_s),
        )
        for lo in np.arange(*reward_steps_s, step_s)
    ]
    steps = [x for x in steps if np.isfinite(x)]
    q2 = float(np.mean(steps)) if steps else np.nan
    return Planted(expected, {"Q1": q1, "Q2": q2, "Q7a": q7a})
