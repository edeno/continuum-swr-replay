"""Synthetic sessions for the convention's tools (terms, support, the loop),
and the likelihoods' JAX functions called on NumPy rows (:func:`family_kernel`).

Two kinds of rows, each built from a fixed seed:

- rate rows: 15 ms bins of the visits of one session (the mgcv comparison's
  ``bins_of``), with a Poisson count of
  event starts, the bin's exposure, two clocks, the last outcome (0/1), ten
  lagged outcomes and one continuous covariate;
- content rows: one row per candidate arm of each event (four candidates: the
  current arm, the previous arm, another leaf arm and a trunk arm), with the
  event's share on each, a clock of each candidate and a reward fraction
  (missing on the trunk).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.regression.padding import pad_values
from continuum_swr_replay.analysis.regression.reml import PenaltyBlock
from continuum_swr_replay.analysis.regression.runtime import float64_kernel, in_float64
from continuum_swr_replay.analysis.regression.terms import Category
from scripts.checks.compare_regression_to_mgcv import bins_of

N_LAGS = 10
ROLES = ("current", "previous", "other", "trunk")
#: The content rows' role category, the "other" leaf arm its reference.
ROLE = Category("role", ROLES, reference="other", penalty="none")


def relative_error(analytic: np.ndarray, numeric: np.ndarray) -> float:
    """``||analytic - numeric|| / ||numeric||``; zero when both are zero."""
    gap = float(np.linalg.norm(np.asarray(analytic) - np.asarray(numeric)))
    return gap / float(np.linalg.norm(numeric)) if gap > 0 else 0.0


def numeric_gradient(f, point: np.ndarray, h: float) -> np.ndarray:
    """Central differences of ``f`` at ``point`` with step ``h``, one per
    coordinate of ``point`` (each the shape of ``f``'s value)."""
    unit = np.eye(point.size)
    return np.array([(f(point + h * e) - f(point - h * e)) / (2 * h) for e in unit])


def numeric_hessian(f, point: np.ndarray, h: float) -> np.ndarray:
    """Central second differences of a scalar ``f`` at ``point`` with step
    ``h``, shape ``(m, m)``."""
    unit = np.eye(point.size)
    return np.array(
        [
            [
                (
                    f(point + h * (a + c))
                    - f(point + h * (a - c))
                    - f(point - h * (a - c))
                    + f(point - h * (a + c))
                )
                / (4 * h * h)
                for c in unit
            ]
            for a in unit
        ]
    )


def stack_smooths(*parts) -> tuple[np.ndarray, list[PenaltyBlock]]:
    """A design of ``parts`` side by side, and a penalty block on each smooth.

    Parameters
    ----------
    *parts : np.ndarray or tuple
        A column, shape ``(n_rows,)``, or columns, shape ``(n_rows, k)``; or a
        smooth, the pair ``(design, penalty)`` of a constrained P-spline.

    Returns
    -------
    X : np.ndarray, shape (n_rows, p)
    blocks : list of PenaltyBlock
        One per smooth, on its columns.
    """
    columns, blocks, start = [], [], 0
    for part in parts:
        if isinstance(part, tuple):
            design, penalty = part
            blocks.append(
                PenaltyBlock(np.arange(start, start + design.shape[1]), penalty)
            )
        else:
            design = np.asarray(part, dtype=float).reshape(np.shape(part)[0], -1)
        columns.append(design)
        start += design.shape[1]
    return np.column_stack(columns), blocks


@float64_kernel(static_argnames=("lik", "name"))
def jax_family(lik, name, *arrays):
    """``lik._jax_<name>(*arrays)``, jitted as the fitter's kernels are."""
    return getattr(lik, f"_jax_{name}")(*arrays)


@in_float64
def family_kernel(lik, name, X, eta, y, aux, *directions):
    """A family's JAX function ``_jax_<name>`` on NumPy rows, in float64.

    The rows are padded by ``lik.pad``; ``eta`` and each direction (``d``, or
    ``d1`` and ``d2``) by ``padding.pad_values``.

    Parameters
    ----------
    lik : Likelihood
    name : {"loglik", "score", "information", "dinformation", "d2information"}
    X : np.ndarray, shape (n_rows, p), or None
        None for ``loglik`` and ``score``, which take no design.
    eta, y : np.ndarray, shape (n_rows,)
    aux : object
        The family's aux, as ``lik.pad`` takes it.
    *directions : np.ndarray, shape (n_rows,)

    Returns
    -------
    float or np.ndarray
        The log-likelihood; the score on the real rows, shape ``(n_rows,)``;
        or a matrix, shape ``(p, p)``.
    """
    rows = lik.pad(np.empty((np.size(y), 0)) if X is None else X, y, aux)
    eta, *directions = (
        pad_values(v, rows.n_real, rows.y.size) for v in (eta, *directions)
    )
    design = () if X is None else (rows.X,)
    out = jax_family(
        lik, name, *design, eta, rows.y, rows.aux, rows.weight, *directions
    )
    if name == "loglik":
        return float(out)
    return np.array(out)[: rows.n_real] if name == "score" else np.array(out)


def rate_rows(seed: int, n_visits: int = 60) -> pd.DataFrame:
    """One session's visit bins (columns described in the module docstring).

    Visit durations are lognormal (median 2 s); the log rate rises over the
    first 2 s of a visit and is 0.4 higher after a rewarded last outcome.

    Returns
    -------
    pd.DataFrame, shape (n_bins, 9 + N_LAGS)
        ``animal``, ``session_date``, ``epoch``, ``well_visit_id``,
        ``exposure`` (s), ``count``, ``time_since_s``, ``time_until_s``,
        ``last_rewarded`` (0/1), ``x`` (N(0, 1) per visit) and
        ``lag_1`` .. ``lag_10`` (0/1).
    """
    rng = np.random.default_rng(seed)
    duration = 2.0 * np.exp(0.6 * rng.normal(size=n_visits))
    visit, _, since, until, exposure = bins_of(duration)
    outcomes = (rng.uniform(size=(n_visits, N_LAGS)) < 0.6).astype(float)
    x = rng.normal(size=n_visits)
    log_rate = (
        np.log(0.5)
        + np.log(3.0) * (1.0 - np.exp(-since / 0.7))
        + 0.4 * outcomes[visit, 0]
        + 0.2 * x[visit]
    )
    frame = pd.DataFrame(
        {
            "animal": "sim",
            "session_date": f"2026{seed:04d}",
            "epoch": "02_r1",
            "well_visit_id": visit,
            "exposure": exposure,
            "count": rng.poisson(exposure * np.exp(log_rate)).astype(float),
            "time_since_s": since,
            "time_until_s": until,
            "last_rewarded": outcomes[visit, 0],
            "x": x[visit],
        }
    )
    for lag in range(N_LAGS):
        frame[f"lag_{lag + 1}"] = outcomes[visit, lag]
    return frame


def content_rows(seed: int, n_events: int = 150) -> pd.DataFrame:
    """One session's candidate rows (columns in the module docstring).

    Each event's shares are a Dirichlet draw around a softmax that favors the
    current arm and rises with the reward fraction.

    Returns
    -------
    pd.DataFrame, shape (4 * n_events, 8)
        ``animal``, ``session_date``, ``epoch``, ``event_id``, ``role``,
        ``share``, ``clock_s`` (the candidate's, s) and
        ``reward_fraction`` (NaN on the trunk row).
    """
    rng = np.random.default_rng(10_000 + seed)
    event = np.repeat(np.arange(n_events), len(ROLES))
    role = np.tile(np.array(ROLES, dtype=object), n_events)
    fraction = rng.uniform(0.0, 1.0, size=event.size)
    fraction[role == "trunk"] = np.nan
    clock = rng.uniform(0.0, 20.0, size=event.size)
    eta = np.select(
        [role == "current", role == "previous", role == "other"],
        [1.0, 0.3, 0.0],
        default=-0.5,
    ) + 0.8 * np.nan_to_num(fraction)
    weight = np.exp(eta).reshape(n_events, len(ROLES))
    p = weight / weight.sum(axis=1, keepdims=True)
    shares = np.concatenate([rng.dirichlet(20.0 * row) for row in p])
    return pd.DataFrame(
        {
            "animal": "sim",
            "session_date": f"2026{seed:04d}",
            "epoch": "02_r1",
            "event_id": event,
            "role": role,
            "share": shares,
            "clock_s": clock,
            "reward_fraction": fraction,
        }
    )


def draw_response(
    rng: np.random.Generator,
    family: str,
    eta: np.ndarray,
    total: float = 0.0,
    group_size: int = 4,
) -> np.ndarray:
    """A response drawn at the linear predictor ``eta`` (an offset included).

    Parameters
    ----------
    rng : np.random.Generator
    family : {"poisson", "bernoulli", "counts", "shares"}
        Poisson counts of mean ``exp(eta)``; 0/1 with probability
        ``expit(eta)``; under the grouped softmax over groups of
        ``group_size`` rows with probabilities ``softmax(eta)``, ``total``
        counts per group
        (multinomial) or shares from a Dirichlet of concentration ``total``
        times the probabilities.
    eta : np.ndarray, shape (n_rows,)
    total : float
        The softmax's counts or concentration per group.
    group_size : int
        The softmax's rows per group.

    Returns
    -------
    np.ndarray, shape (n_rows,)
    """
    if family == "poisson":
        return rng.poisson(np.exp(eta)).astype(float)
    if family == "bernoulli":
        return (rng.uniform(size=eta.size) < 1 / (1 + np.exp(-eta))).astype(float)
    weight = np.exp(eta).reshape(-1, group_size)
    p = weight / weight.sum(axis=1, keepdims=True)
    if family == "counts":
        return np.concatenate([rng.multinomial(int(total), row) for row in p]).astype(
            float
        )
    return np.concatenate([rng.dirichlet(total * row) for row in p])


def by_session(frames: list[pd.DataFrame]) -> dict[tuple[str, str, str], pd.DataFrame]:
    """Session key ``(animal, session_date, epoch)`` -> its rows."""
    return {
        tuple(frame.loc[0, ["animal", "session_date", "epoch"]]): frame
        for frame in frames
    }


@pytest.fixture
def rate_sessions() -> dict[tuple[str, str, str], pd.DataFrame]:
    """Six rate sessions (seeds 0 to 5)."""
    return by_session([rate_rows(seed) for seed in range(6)])


@pytest.fixture
def content_sessions() -> dict[tuple[str, str, str], pd.DataFrame]:
    """Six content sessions (seeds 0 to 5)."""
    return by_session([content_rows(seed) for seed in range(6)])
