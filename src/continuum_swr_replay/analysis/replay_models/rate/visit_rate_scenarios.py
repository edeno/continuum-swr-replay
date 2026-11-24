"""Planted scenarios of the visit rate model, as pure functions of the rows.

Each scenario maps one session's rows of the model (``visit_rate.cohort_rows``:
the clocks, ``exposure``, ``last_outcome_rewarded``) to the expected count of
each bin, and gives the planted value of each declared estimand of the model
(Q1, Q2, Q7a) under it. A scenario draws nothing and fits nothing: a
simulation draws the counts from the expected ones and runs the real fit.

In every scenario but the rate table, the log rate of a bin is ``b0 +
f_s(s) + f_u(u) [+ f_m(m)] + g(s) * rewarded``: the omission side's clock
curves (:class:`PlantedClocks`), and the effect of a reward ``g``, which may
change with time since arrival. ``b0`` is set so that the session's expected
total is ``total`` (when given; else ``b0 = 0``).

Two truths
----------
The two forms of the model's reward effect (``visit_rate.REWARD_FORMS``)
estimate different things when the effect changes with time since arrival, so
each scenario gives both (:class:`Planted`):

- ``omission_side``, the truth of ``reward_form="time_varying"``. That form
  fits one arrival curve for the rows after an omission and a second one for
  the rows after a reward, so its arrival curve is the omission side's; the
  leaving curve is shared, and is ``f_u`` because the effect changes with
  time since arrival only. Q1 and Q7a are the range contrasts of ``f_s`` and
  ``f_u``. Q2 is the model's coefficient: the rewarded arrival curve sums to
  zero over the rewarded rows of the design, so the coefficient is the mean of
  ``g`` over those rows (one weight per row, the arrival clock clamped at its
  basis range). It is computed on the rows given, so it equals the model's
  value for a design built on these rows alone (one session's fit).
- ``pooled``, the truth of ``reward_form="coefficient"``. That form fits ONE
  arrival curve and ONE leaving curve for both outcomes and one coefficient,
  so where ``g`` changes with time the shared curves absorb the part of the
  effect that the coefficient cannot hold, in proportion to the share of
  rewarded expected events at each clock value. The pooled truth is the
  Poisson projection of the planted expected counts onto that form
  (:func:`poisson_projection`): ``log mu = log mu_ref + a(arrival cell) +
  b(leaving cell) + beta * rewarded``, fitted exactly to the session's own
  expected counts, with ``mu_ref`` the omission side's planted rate and ``a``,
  ``b`` steps on cells of ``ARRIVAL_CELL_S`` and ``LEAVING_CELL_S``. It is the
  limit of the model's fit on unlimited data, with the curves' flexibility
  set by the cells (the stretch clock, if planted, held at its curve).
  Pooled Q1 is the range contrast of ``f_s + a``, pooled Q7a that of ``f_u +
  b``, pooled Q2 is ``beta``. Where ``g`` is constant, both truths are the
  planted curves' contrasts and ``g``.

:meth:`Planted.for_form` gives the truth of one reward form
(:data:`TRUTH_OF_FORM`).

The range contrasts are the means of ``visit_rate.declared_contrasts``'s
windows on the same grid (``contrasts.range_grid``), so a curve's additive
constant cancels. Q1's reference window is an argument (the declared ``(0,
1)`` by default). Where the outcomes differ only in their arrival curve (a
later onset after an omission), there is no planted reward ratio: Q2 is NaN in
both truths. The rate table (:func:`rate_table_truth`) plants no curve: both
of its truths are projections (see there).
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
from .visit_rate import (
    ARRIVAL,
    CLOCK_BASES,
    EXPOSURE,
    LEAVING,
    RESPONSE,
    REWARD,
    STRETCH,
)

__all__ = [
    "ARRIVAL_CELL_S",
    "LEAVING_CELL_S",
    "TRUTH_OF_FORM",
    "Planted",
    "PlantedClocks",
    "Projection",
    "RateTable",
    "constant_ratio",
    "differing_ratio",
    "drop_before_leaving",
    "drop_before_small_movements",
    "fading_ratio",
    "interpolated_curve",
    "later_onset_after_omission",
    "no_effect",
    "poisson_projection",
    "rate_table_from_bins",
    "rate_table_truth",
    "session_log_ratios",
]

LogCurve = Callable[[np.ndarray], np.ndarray]

#: The declared windows of Q1 and Q7a, read from ``declared_tests``.
_Q1 = estimand("Q1").contrast
_Q7A = estimand("Q7a").contrast

#: The width of the projection's arrival and leaving cells (s).
ARRIVAL_CELL_S = 0.25
LEAVING_CELL_S = 0.05

#: The truth each reward form of the model estimates (module docstring).
TRUTH_OF_FORM: dict[str, str] = {
    "coefficient": "pooled",
    "time_varying": "omission_side",
}

#: The projection's convergence: the largest log ratio of a margin's planted
#: to fitted expected events, and the sweeps allowed.
_PROJECTION_TOL = 1e-11
_PROJECTION_MAX_SWEEPS = 20_000


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

    def log_rate(self, rows: pd.DataFrame) -> np.ndarray:
        """``f_s + f_u [+ f_m]`` at each row, shape ``(n_rows,)``."""
        out = self.arrival(rows[ARRIVAL].to_numpy(dtype=float)) + self.leaving(
            rows[LEAVING].to_numpy(dtype=float)
        )
        if self.stretch is not None:
            out = out + self.stretch(rows[STRETCH].to_numpy(dtype=float))
        return out

    def curve_values(
        self, q1_reference_s: tuple[float, float] = _Q1.second
    ) -> dict[str, float]:
        """Q1 and Q7a of the curves: their range contrasts (module docstring)."""
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
    pooled : dict of str to float
        Q1, Q2 and Q7a of one shared arrival curve, one shared leaving curve
        and one coefficient: the truth of ``reward_form="coefficient"``.
    omission_side : dict of str to float
        Q1 and Q7a of the omission side's arrival curve and the shared
        leaving curve, Q2 the time-varying form's coefficient: the truth of
        ``reward_form="time_varying"``. NaN where the scenario has none.
    """

    expected: np.ndarray
    pooled: dict[str, float]
    omission_side: dict[str, float]

    def for_form(self, reward_form: str = "coefficient") -> dict[str, float]:
        """The planted Q1, Q2 and Q7a of one reward form (:data:`TRUTH_OF_FORM`)."""
        if reward_form not in TRUTH_OF_FORM:
            raise ValueError(f"reward_form {reward_form!r} not in {set(TRUTH_OF_FORM)}")
        return getattr(self, TRUTH_OF_FORM[reward_form])


def _scaled(
    rows: pd.DataFrame, log_rate: np.ndarray, total: float | None
) -> np.ndarray:
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


# --- The projection onto the shared form ---------------------------------------


def _cell_edges(clock: str, cell_s: float) -> np.ndarray:
    """Cells of ``cell_s`` over the clock's basis range, and one beyond it
    (where the model clamps the clock)."""
    lo, hi, _ = CLOCK_BASES[clock]
    n_cells = (hi - lo) / cell_s
    if not (cell_s > 0 and np.isclose(n_cells, round(n_cells))):
        raise ValueError(f"{cell_s} s does not split {clock}'s range [{lo}, {hi}]")
    return np.append(np.linspace(lo, hi, round(n_cells) + 1), np.inf)


def _cell(edges: np.ndarray, x: np.ndarray) -> np.ndarray:
    index = np.searchsorted(edges, np.asarray(x, dtype=float), side="right") - 1
    if np.any((index < 0) | (index >= edges.size - 1)):
        raise ValueError("a clock value below its range")
    return index


class Projection(NamedTuple):
    """The Poisson projection of planted expected counts (:func:`poisson_projection`).

    Attributes
    ----------
    arrival : np.ndarray, shape (n_outcomes, n_arrival_cells)
        The arrival adjustment of each cell: one row (shared) or two (after
        an omission, after a reward); NaN in a cell without exposure.
    leaving : np.ndarray, shape (n_leaving_cells,)
        The leaving adjustment of each cell; NaN in a cell without exposure.
    beta : float
        The reward coefficient; NaN when the arrival steps are by outcome.
    arrival_edges, leaving_edges : np.ndarray
        The cells' edges (s), the last ``inf``.
    """

    arrival: np.ndarray
    leaving: np.ndarray
    beta: float
    arrival_edges: np.ndarray
    leaving_edges: np.ndarray

    def arrival_at(self, x: np.ndarray, outcome: int = 0) -> np.ndarray:
        """The arrival adjustment at clock values ``x`` (row ``outcome``)."""
        return self.arrival[outcome][_cell(self.arrival_edges, x)]

    def leaving_at(self, x: np.ndarray) -> np.ndarray:
        """The leaving adjustment at clock values ``x``."""
        return self.leaving[_cell(self.leaving_edges, x)]


def poisson_projection(
    rows: pd.DataFrame,
    expected: np.ndarray,
    reference: np.ndarray,
    *,
    by_outcome: bool,
    arrival_cell_s: float = ARRIVAL_CELL_S,
    leaving_cell_s: float = LEAVING_CELL_S,
) -> Projection:
    """The Poisson projection of planted expected counts onto steps of the
    two clocks and the reward.

    The model ``mu = reference * exp(a(arrival cell) + b(leaving cell) [+
    beta * rewarded])`` fitted to ``expected`` by maximum likelihood, by
    iterative proportional fitting: its score equations say that the fitted
    and planted expected events agree in every arrival cell (of each outcome
    when ``by_outcome``), every leaving cell and, with one shared arrival
    step, on the rewarded rows. ``a`` and ``b`` are identified up to one
    constant, which no contrast reads.

    Parameters
    ----------
    rows : pd.DataFrame
        One session's rows: the two clocks and ``last_outcome_rewarded``.
    expected : np.ndarray, shape (n_rows,)
        The planted expected counts.
    reference : np.ndarray, shape (n_rows,)
        The rate the steps multiply (positive where ``expected`` is).
    by_outcome : bool
        False: one arrival step for both outcomes and ``beta`` (the coefficient
        form); True: an arrival step for each outcome (the time-varying form).
    arrival_cell_s, leaving_cell_s : float
        The cells' widths (s); each splits its clock's basis range.

    Returns
    -------
    Projection

    Raises
    ------
    RuntimeError
        If the fit does not converge in the sweeps allowed.
    """
    expected = np.asarray(expected, dtype=float)
    reference = np.asarray(reference, dtype=float)
    rewarded = _rewarded(rows).astype(int)
    arrival_edges = _cell_edges(ARRIVAL, arrival_cell_s)
    leaving_edges = _cell_edges(LEAVING, leaving_cell_s)
    n_arrival = arrival_edges.size - 1
    n_outcomes = 2 if by_outcome else 1
    arrival_cell = _cell(arrival_edges, rows[ARRIVAL].to_numpy(dtype=float))
    arrival_group = arrival_cell + (rewarded * n_arrival if by_outcome else 0)
    leaving_cell = _cell(leaving_edges, rows[LEAVING].to_numpy(dtype=float))
    groups = [
        (arrival_group, n_outcomes * n_arrival),
        (leaving_cell, leaving_edges.size - 1),
    ]
    if not by_outcome:
        groups.append((rewarded, 2))
    planted = [np.bincount(g, expected, n) for g, n in groups]
    steps = [np.zeros(n) for _, n in groups]
    for _ in range(_PROJECTION_MAX_SWEEPS):
        largest = 0.0
        for k, (group, n) in enumerate(groups):
            log_fitted = np.log(reference) + sum(
                step[g] for step, (g, _) in zip(steps, groups, strict=True)
            )
            fitted = np.bincount(group, np.exp(log_fitted), n)
            update = np.zeros(n)
            has = planted[k] > 0
            update[has] = np.log(planted[k][has] / fitted[has])
            if k == 2:  # beta: only the rewarded level moves
                update[0] = 0.0
            steps[k] += update
            largest = max(largest, float(np.max(np.abs(update))))
        if largest < _PROJECTION_TOL:
            break
    else:
        raise RuntimeError("the projection did not converge")
    arrival = np.where(planted[0] > 0, steps[0], np.nan).reshape(n_outcomes, n_arrival)
    leaving = np.where(planted[1] > 0, steps[1], np.nan)
    beta = np.nan if by_outcome else float(steps[2][1])
    return Projection(arrival, leaving, beta, arrival_edges, leaving_edges)


def _projected_contrasts(
    projection: Projection,
    arrival: LogCurve,
    leaving: LogCurve,
    q1_reference_s: tuple[float, float],
) -> dict[str, float]:
    """Q1 and Q7a of the curves plus the projection's steps (row 0 of the
    arrival steps); NaN when a window's cell holds no exposure."""

    def shifted_arrival(x: np.ndarray) -> np.ndarray:
        return arrival(x) + projection.arrival_at(x)

    def shifted_leaving(x: np.ndarray) -> np.ndarray:
        return leaving(x) + projection.leaving_at(x)

    return PlantedClocks(shifted_arrival, shifted_leaving).curve_values(q1_reference_s)


def _zero(x: np.ndarray) -> np.ndarray:
    return np.zeros(np.shape(x))


# --- The scenarios on planted curves --------------------------------------------


def _from_curves(
    rows: pd.DataFrame,
    clocks: PlantedClocks,
    effect: LogCurve,
    *,
    total: float | None,
    q1_reference_s: tuple[float, float],
    planted_ratio: bool = True,
) -> Planted:
    """The scenario ``log mu = b0 + f_s + f_u [+ f_m] + g(s) * rewarded``,
    ``clocks`` the omission side's curves and ``effect`` ``g``, with both
    truths (module docstring)."""
    rewarded = _rewarded(rows)
    s = rows[ARRIVAL].to_numpy(dtype=float)
    g = effect(s) * rewarded
    expected = _scaled(rows, clocks.log_rate(rows) + g, total)
    reference = expected * np.exp(-g)
    projection = poisson_projection(rows, expected, reference, by_outcome=False)
    hi = CLOCK_BASES[ARRIVAL][1]
    on_rewarded = effect(np.minimum(s[rewarded == 1.0], hi))
    omission_q2 = float(np.mean(on_rewarded)) if on_rewarded.size else np.nan
    pooled_q2 = projection.beta
    if not planted_ratio:
        omission_q2 = pooled_q2 = np.nan
    return Planted(
        expected,
        {
            **_projected_contrasts(
                projection, clocks.arrival, clocks.leaving, q1_reference_s
            ),
            "Q2": pooled_q2,
        },
        {**clocks.curve_values(q1_reference_s), "Q2": omission_q2},
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
    """A constant rate ratio after a reward, ``g = log_ratio``: both truths
    are the curves' contrasts and ``log_ratio``."""

    def effect(x: np.ndarray) -> np.ndarray:
        return np.full(np.shape(x), float(log_ratio))

    return _from_curves(
        rows, clocks, effect, total=total, q1_reference_s=q1_reference_s
    )


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

    ``clocks`` are the omission side's curves. The time-varying form's truth
    (``omission_side``): their Q1 and Q7a, and Q2 the mean of ``g`` over the
    rewarded rows. The coefficient form's (``pooled``): the projection's
    (module docstring), whose shared arrival curve takes part of the fading
    effect, so its Q1 differs from the omission side's.
    """
    if not fade_end_s > fade_start_s:
        raise ValueError("the fade ends after it starts")

    def effect(s: np.ndarray) -> np.ndarray:
        return log_ratio * np.clip(
            (fade_end_s - np.asarray(s)) / (fade_end_s - fade_start_s), 0.0, 1.0
        )

    return _from_curves(
        rows, clocks, effect, total=total, q1_reference_s=q1_reference_s
    )


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
    is ``clocks.arrival`` ``delay_s`` later, ``f_s(max(s - delay_s, 0))``;
    after a reward it is ``f_s(s)``.

    The omission side's arrival curve is the delayed one, so the time-varying
    form's Q1 is the delayed curve's; the coefficient form's is the
    projection's (module docstring). No reward ratio is planted: Q2 is NaN
    in both truths.
    """

    def delayed(s: np.ndarray) -> np.ndarray:
        return clocks.arrival(np.maximum(np.asarray(s, dtype=float) - delay_s, 0.0))

    def effect(s: np.ndarray) -> np.ndarray:
        return clocks.arrival(np.asarray(s, dtype=float)) - delayed(s)

    omission = PlantedClocks(delayed, clocks.leaving, clocks.stretch)
    return _from_curves(
        rows,
        omission,
        effect,
        total=total,
        q1_reference_s=q1_reference_s,
        planted_ratio=False,
    )


def drop_before_leaving(
    rows: pd.DataFrame,
    arrival: LogCurve,
    *,
    drop: float,
    width_s: float = 0.3,
    total: float | None = None,
    q1_reference_s: tuple[float, float] = _Q1.second,
) -> Planted:
    """A drop of the log rate by ``drop`` in the last ``width_s`` before
    leaving, on the leaving clock only, with no reward effect and no stretch
    effect.

    ``f_u(u) = -drop`` for ``u < width_s``, 0 after; planted Q7a is that
    step's range contrast (``-drop`` when the last window lies inside
    ``width_s`` and the reference window beyond it), Q1 the arrival curve's
    and Q2 0. A model with the stretch clock may move part of the drop to
    ``f_m``, since in a visit's last stretch the two clocks are equal.
    """

    def leaving(u: np.ndarray) -> np.ndarray:
        return np.where(np.asarray(u) < width_s, -float(drop), 0.0)

    return no_effect(
        rows,
        PlantedClocks(arrival, leaving),
        total=total,
        q1_reference_s=q1_reference_s,
    )


def drop_before_small_movements(
    rows: pd.DataFrame,
    arrival: LogCurve,
    *,
    drop: float,
    width_s: float = 0.3,
    total: float | None = None,
    q1_reference_s: tuple[float, float] = _Q1.second,
) -> Planted:
    """A drop of the log rate by ``drop`` in the last ``width_s`` of every
    immobile stretch, on the stretch clock only, with a flat leaving curve and
    no reward effect.

    ``f_m(m) = -drop`` for ``m < width_s``; ``f_u`` is flat, so planted Q7a
    is 0, Q1 the arrival curve's and Q2 0. The visit's last stretch ends at
    the final poke-out, so the rate drops before leaving too: a model must
    give the drop to ``f_m``, or the third clock's rule must catch that it
    cannot.
    """

    def stretch(m: np.ndarray) -> np.ndarray:
        return np.where(np.asarray(m) < width_s, -float(drop), 0.0)

    return no_effect(
        rows,
        PlantedClocks(arrival, _zero, stretch),
        total=total,
        q1_reference_s=q1_reference_s,
    )


# --- The rate table --------------------------------------------------------------


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


def rate_table_truth(
    rows: pd.DataFrame,
    table: RateTable,
    *,
    total: float | None = None,
    q1_reference_s: tuple[float, float] = _Q1.second,
    arrival_cell_s: float = ARRIVAL_CELL_S,
    leaving_cell_s: float = LEAVING_CELL_S,
) -> Planted:
    """A truth taken from a table of rates by outcome and both clocks.

    Each bin's planted rate is its cell's (scaled to ``total`` when given).
    The table plants no curve and no coefficient; both truths are Poisson
    projections of the planted expected counts on the session's own rows
    (:func:`poisson_projection`, reference the exposure, cells of
    ``arrival_cell_s`` and ``leaving_cell_s``), with Q1 and Q7a the range
    contrasts of the projection's steps:

    - ``pooled``: one arrival step for both outcomes, one leaving step and
      ``beta``; Q2 is ``beta``.
    - ``omission_side``: an arrival step for each outcome and one leaving
      step; Q1 is the omission rows' arrival step's, and Q2 the mean over the
      rewarded rows of their arrival step less the omission rows' in the same
      cell (the time-varying form's coefficient).

    A value whose window or rows fall in a cell without exposure is NaN.
    """
    outcome, arrival_cell, leaving_cell = table.cells(rows)
    expected = _scaled(
        rows, np.log(table.rates[outcome, arrival_cell, leaving_cell]), total
    )
    reference = rows[EXPOSURE].to_numpy(dtype=float)
    cells = {"arrival_cell_s": arrival_cell_s, "leaving_cell_s": leaving_cell_s}
    pooled = poisson_projection(rows, expected, reference, by_outcome=False, **cells)
    by_outcome = poisson_projection(rows, expected, reference, by_outcome=True, **cells)
    rewarded = outcome == 1
    s = rows[ARRIVAL].to_numpy(dtype=float)[rewarded]
    step = by_outcome.arrival_at(s, 1) - by_outcome.arrival_at(s, 0)
    return Planted(
        expected,
        {
            **_projected_contrasts(pooled, _zero, _zero, q1_reference_s),
            "Q2": pooled.beta,
        },
        {
            **_projected_contrasts(by_outcome, _zero, _zero, q1_reference_s),
            "Q2": float(np.mean(step)) if step.size else np.nan,
        },
    )
