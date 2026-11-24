"""Reference cases of the shared regression fitter, and the rule that compares them.

Each case builds its inputs from a fixed seed, runs one public entry of
``continuum_swr_replay.analysis.regression`` and returns its outputs as plain
Python values (lists, floats, ints, strings, bools, None). The cases are in
groups, one stored file each (:data:`GROUPS`):

- ``newton``: :func:`.newton.penalized_newton` for each likelihood (Poisson with
  an offset, Bernoulli, Gaussian, grouped softmax), with and without a penalty,
  with penalized simple terms (a slope and the levels of a category, each with
  a weak identity penalty), a P-spline of 40 columns at a weak penalty for each
  likelihood, and the statuses other than converged.
- ``reml``: :func:`.reml.reml_and_derivatives` (the score, its gradient and
  Hessian in the log penalties, the coefficients, covariance and edf) for each
  likelihood with one and two smooths at several log penalties, with a smooth
  beside penalized simple terms and as a P-spline of 40 columns (each
  likelihood), and a design not identified beside a penalty block.
- ``refit``: :func:`.reml.refit_session` on sessions with no finite maximum:
  columns left out over rounds, along a penalty, along the intercept.
- ``select``: :func:`.reml.select_penalties` on small sets of sessions, with a
  session set aside and refitted, one left without a fit, one whose fit fails
  at a grid point, a search of six outer steps, and four smooths (the grid of
  one term at a time).
- ``sessions``: :func:`.sessions.fit_sessions`, every field of every record,
  on simulated rate, content, Bernoulli and Gaussian sessions.
- ``calibration``: :func:`.calibration.log_penalty_for_edf` on rate sessions.

The rule (:func:`differences`): a float ``a`` matches its reference ``b`` when
``|a - b| <= ATOL + RTOL |b|``; an int, a bool, a string or None must be
equal, of the same type. A float reference needs a float and an int reference
an int, alone or in a list: ``3`` does not match ``3.0``, nor ``3.0`` ``3``. A float that is not finite is stored as a string
(:func:`plain`), so NaN must be NaN and an infinity the same infinity.
So statuses, reasons, step counts, the columns left out and every count are
exact. A dict must have the same keys and a list the same length.

Not a script: ``write_regression_reference.py`` writes the cases' outputs and
``tests/analysis/regression/test_reference_fixtures.py`` compares the fitter
with them; both import it as ``scripts.checks._regression_reference``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.regression.calibration import log_penalty_for_edf
from continuum_swr_replay.analysis.regression.constants import NEWTON_DECREMENT_TOL
from continuum_swr_replay.analysis.regression.likelihoods import (
    Bernoulli,
    Gaussian,
    GroupedSoftmax,
    Poisson,
)
from continuum_swr_replay.analysis.regression.newton import (
    NO_FINITE_MAXIMUM,
    NewtonFit,
    penalized_newton,
)
from continuum_swr_replay.analysis.regression.pspline import PSplineBasis
from continuum_swr_replay.analysis.regression.reml import (
    PenaltyBlock,
    PooledFit,
    RemlScore,
    SessionData,
    SessionRefit,
    assemble_penalty,
    free_basis,
    refit_session,
    reml_and_derivatives,
    select_penalties,
)
from continuum_swr_replay.analysis.regression.sessions import fit_sessions
from continuum_swr_replay.analysis.regression.terms import (
    Category,
    Design,
    Offset,
    Slope,
    Smooth,
    build_design,
    log_penalty_from_prior_sd,
)

__all__ = ["ATOL", "CASES", "GROUPS", "RTOL", "differences", "plain"]

#: The acceptance rule: ``|a - b| <= ATOL + RTOL |b|`` for a float.
RTOL = 1e-9
ATOL = 1e-11

#: The weak penalty of the penalized simple terms: a Normal prior of SD 2.
WEAK = log_penalty_from_prior_sd(2.0)


# --------------------------------------------------------------------------
# The rule


def plain(value: object) -> object:
    """``value`` as strict JSON holds it: NumPy arrays and scalars as lists
    and Python numbers, tuples as lists, dict keys and string enumerations as
    strings, and a float that is not finite as the string ``"NaN"``,
    ``"Infinity"`` or ``"-Infinity"``."""
    if isinstance(value, str):
        return str(value)
    if isinstance(value, float | np.floating):
        value = float(value)
        if math.isfinite(value):
            return value
        return (
            "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
        )
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def _is_float(value: object) -> bool:
    return isinstance(value, float)


def _floats_differ(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Elementwise, True where the float ``a`` does not match ``b`` (both
    finite, as :func:`plain` leaves every float) by the rule."""
    return ~(np.abs(a - b) <= ATOL + RTOL * np.abs(b))


def _float_array(value: object) -> np.ndarray | None:
    """A list of floats (nested, rectangular) as an array; None otherwise."""
    try:
        array = np.asarray(value)
    except ValueError:  # ragged
        return None
    if array.dtype != float or array.ndim == 0:
        return None
    flat = value
    while isinstance(flat, list) and flat and isinstance(flat[0], list):
        flat = [x for row in flat for x in row]
    return array if all(_is_float(x) for x in flat) else None


def differences(actual: object, reference: object, where: str = "") -> list[str]:
    """Where ``actual`` does not match ``reference`` by the rule (module
    docstring); empty when it matches.

    Parameters
    ----------
    actual : object
        Plain values (:func:`plain`).
    reference : object
        The stored values, as read from JSON.
    where : str
        The path of ``reference`` in its file, for the messages.

    Returns
    -------
    list of str
        One line per mismatch, naming its path, the values and, for floats,
        the largest ratio of the difference to the allowed difference.
    """
    if isinstance(reference, dict):
        if not isinstance(actual, dict) or set(actual) != set(reference):
            got = sorted(actual) if isinstance(actual, dict) else type(actual)
            return [f"{where}: keys {got}, reference {sorted(reference)}"]
        return [
            line
            for key in reference
            for line in differences(actual[key], reference[key], f"{where}.{key}")
        ]
    if isinstance(reference, list):
        if not isinstance(actual, list) or len(actual) != len(reference):
            got = len(actual) if isinstance(actual, list) else type(actual)
            return [f"{where}: length {got}, reference {len(reference)}"]
        b = _float_array(reference)
        a = None if b is None else _float_array(actual)
        if b is None or a is None or a.shape != b.shape:
            return [
                line
                for i, (x, y) in enumerate(zip(actual, reference, strict=True))
                for line in differences(x, y, f"{where}[{i}]")
            ]
        bad = _floats_differ(a, b)
        if not bad.any():
            return []
        ratio = np.abs(a - b) / (ATOL + RTOL * np.abs(b))
        worst = np.unravel_index(np.argmax(ratio), b.shape)
        return [
            f"{where}: {int(bad.sum())} of {b.size} floats differ; at {worst} "
            f"{a[worst]!r} against {b[worst]!r} (ratio {ratio[worst]:.3g})"
        ]
    if _is_float(reference):
        if not _is_float(actual):
            return [f"{where}: {actual!r}, reference float {reference!r}"]
        if _floats_differ(np.array(float(actual)), np.array(reference)):
            ratio = abs(actual - reference) / (ATOL + RTOL * abs(reference))
            return [f"{where}: {actual!r} against {reference!r} (ratio {ratio:.3g})"]
        return []
    if type(actual) is not type(reference) or actual != reference:
        return [f"{where}: {actual!r}, reference {reference!r}"]
    return []


# --------------------------------------------------------------------------
# Outputs of the fitter's results


def _newton(fit: NewtonFit, with_A: bool = True) -> dict:
    """The outputs of a Newton fit (``A`` only ``with_A``).

    The decrement is stored only as below the tolerance or not (at the stop
    it is rounding), and the combinations the design does not identify as
    their projector (the basis's signs are not defined). A fit with no finite
    maximum keeps its status, steps, log-likelihood and the columns running
    away, not its coefficients, ``A`` or step: along the direction of
    unbounded increase these depend on the rounding (between two BLAS kernels
    of one machine a coefficient of 22 moved by 3e-6, a step of 1 by 2e-5).
    """
    projector = None
    if fit.unidentified is not None:
        projector = fit.unidentified @ fit.unidentified.T
    runs_away = fit.status == NO_FINITE_MAXIMUM
    return {
        **({} if runs_away else {"beta": fit.beta}),
        **({"A": fit.A} if with_A and not runs_away else {}),
        "status": fit.status,
        "n_iter": fit.n_iter,
        "loglik": fit.loglik,
        "decrement_below_tol": bool(fit.newton_decrement < NEWTON_DECREMENT_TOL),
        "running_away": fit.running_away,
        "unidentified_projector": projector,
    }


def _score(score: RemlScore) -> dict:
    """A session's REML score and its fit; ``A`` is left out, its inverse
    (the covariance) is kept."""
    return {
        "score": score.score,
        "gradient": score.gradient,
        "hessian": score.hessian,
        "scale": score.scale,
        "covariance": score.covariance,
        "edf": score.edf,
        "edf_total": score.edf_total,
        "fit": _newton(score.fit, with_A=False),
        "running_away_directions": [str(d) for d in score.running_away_directions],
        "unidentified_directions": [str(d) for d in score.unidentified_directions],
    }


def _refit(refit: SessionRefit) -> dict:
    return {
        "refits": [_score(r) for r in refit.refits],
        "left_out": refit.left_out,
        "rounds": refit.rounds,
        "ended": str(refit.ended),
        "reason": refit.reason,
    }


def _pooled(fit: PooledFit) -> dict:
    search = fit.search
    return {
        "rho": fit.rho,
        "score": fit.score,
        "gradient": fit.gradient,
        "sessions": [_score(s) for s in fit.sessions],
        "search": None
        if search is None
        else {
            "reference_rho": search.reference_rho,
            "in_choice": search.in_choice,
            "set_aside": [
                {
                    "index": s.index,
                    "name": s.name,
                    "reference": _score(s.reference),
                    "refits": [_score(r) for r in s.refits],
                    "left_out": s.left_out,
                    "reason": s.reason,
                }
                for s in search.set_aside
            ],
            "start_score": search.start_score,
            "n_grid_points": search.n_grid_points,
            "grid_failures": search.grid_failures,
            "n_outer_steps": search.n_outer_steps,
            "at_boundary": search.at_boundary,
        },
    }


# --------------------------------------------------------------------------
# Inputs


def _smooth(x: np.ndarray, lo: float, hi: float, n_basis: int):
    """A constrained cubic P-spline of ``x``: its design and penalty."""
    smooth = PSplineBasis(lo, hi, n_basis).constrained(x)
    return smooth.design, smooth.penalty


def _rows(seed: int, n: int = 600) -> pd.DataFrame:
    """Rows of a diagonal-family design: a clock in [0, 20] (most rows early,
    so the late basis functions see few rows), a continuous ``x``, a 0/1
    ``flag``, a three-level ``level`` and an exposure."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "clock": np.minimum(rng.exponential(3.0, n), 20.0),
            "x": rng.normal(size=n),
            "flag": (rng.uniform(size=n) < 0.4).astype(float),
            "level": rng.choice(np.array(["a", "b", "c"], dtype=object), n),
            "exposure": rng.uniform(0.05, 0.2, n),
        }
    )


LEVEL = Category("level", ("a", "b", "c"), reference="a", penalty=WEAK)


def _diagonal_design(rows: pd.DataFrame, n_basis: int, rho: float) -> Design:
    """Intercept, a smooth of the clock at ``rho``, the weakly penalized slope
    of ``x`` and category ``level``, the unpenalized 0/1 ``flag``."""
    return build_design(
        [
            Smooth("clock", 0.0, 20.0, n_basis, rho),
            Slope("x", WEAK),
            LEVEL,
            Slope("flag", "none"),
            Offset("exposure"),
        ],
        rows,
        intercept=True,
    )


def _eta(rows: pd.DataFrame) -> np.ndarray:
    level = rows["level"].map({"a": 0.0, "b": 0.3, "c": -0.4}).to_numpy()
    return (
        np.log(2.0) * (1.0 - np.exp(-rows["clock"].to_numpy() / 2.0))
        + 0.3 * rows["x"].to_numpy()
        - 0.5 * rows["flag"].to_numpy()
        + level
    )


def _diagonal_inputs(family: str, seed: int, n_basis: int, rho: float):
    """A likelihood, its design matrix, response, aux, penalty and free basis."""
    rows = _rows(seed)
    rng = np.random.default_rng(100 + seed)
    design = _diagonal_design(rows, n_basis, rho)
    built = design.build(rows)
    X, eta = built.X, _eta(rows)
    if family == "poisson":
        lik, aux = Poisson(), built.offset
        y = rng.poisson(np.exp(1.0 + eta + aux)).astype(float)
    elif family == "bernoulli":
        lik, aux = Bernoulli(), None
        y = (rng.uniform(size=eta.size) < 1 / (1 + np.exp(-eta))).astype(float)
    else:
        lik, aux = Gaussian(), 0.7
        y = 1.0 + eta + rng.normal(scale=0.8, size=eta.size)
    p = X.shape[1]
    S = assemble_penalty(design.blocks, design.log_penalties, p)
    return lik, X, y, aux, S, free_basis(design.blocks, p)


def _content_frame(
    seed: int, n_events: int = 120, early_clock: bool = False
) -> pd.DataFrame:
    """Content rows: four candidate rows per event (roles), a clock and a
    reward fraction (missing on the trunk row), shares from a Dirichlet. The
    clock is uniform on [0, 20]; ``early_clock``: exponential of mean 3, cut
    at 20, so that the late basis functions of a wide spline see few rows."""
    rng = np.random.default_rng(1000 + seed)
    roles = ("current", "previous", "other", "trunk")
    event = np.repeat(np.arange(n_events), len(roles))
    role = np.tile(np.array(roles, dtype=object), n_events)
    fraction = rng.uniform(0.0, 1.0, event.size)
    fraction[role == "trunk"] = np.nan
    clock = (
        np.minimum(rng.exponential(3.0, event.size), 20.0)
        if early_clock
        else rng.uniform(0.0, 20.0, event.size)
    )
    eta = (
        np.select([role == "current", role == "previous"], [1.0, 0.3], default=0.0)
        + 0.8 * np.nan_to_num(fraction)
        - 0.04 * clock
    )
    weight = np.exp(eta).reshape(n_events, len(roles))
    share = np.concatenate([rng.dirichlet(20.0 * row / row.sum()) for row in weight])
    return pd.DataFrame(
        {
            "animal": "sim",
            "session_date": f"2026{seed:04d}",
            "epoch": "02_r1",
            "event_id": event,
            "role": role,
            "share": share,
            "clock_s": clock,
            "reward_fraction": fraction,
        }
    )


ROLE = Category(
    "role", ("current", "previous", "other", "trunk"), reference="other", penalty=WEAK
)


def _content_design(
    pooled: pd.DataFrame, rho: float = 2.0, n_basis: int = 10
) -> Design:
    """The role category (weakly penalized), a slope of the reward fraction by
    leaf role (weakly penalized), a smooth of the clock (``n_basis`` basis
    functions at log penalty ``rho``)."""
    return build_design(
        [
            ROLE,
            Slope(
                "reward_fraction",
                WEAK,
                by=ROLE,
                levels=("current", "previous", "other"),
            ),
            Smooth("clock_s", 0.0, 20.0, n_basis, rho),
        ],
        pooled,
        intercept=False,
    )


def _softmax_frame(seed: int, early_clock: bool = False) -> pd.DataFrame:
    frame = _content_frame(seed, early_clock=early_clock)
    frame["reward_fraction"] = frame["reward_fraction"].fillna(0.0)
    return frame


def _softmax_inputs(seed: int, n_basis: int = 10, rho: float = 2.0):
    frame = _softmax_frame(seed, early_clock=n_basis > 10)
    design = _content_design(frame, rho, n_basis)
    X = design.build(frame).X
    p = X.shape[1]
    S = assemble_penalty(design.blocks, design.log_penalties, p)
    return (
        GroupedSoftmax(),
        X,
        frame["share"].to_numpy(),
        frame["event_id"].to_numpy(),
        S,
        free_basis(design.blocks, p),
    )


def _logistic(seed: int = 5, n: int = 200):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    X = np.column_stack([np.ones(n), x])
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-(0.3 + 1.2 * x)))).astype(float)
    return X, y


# --------------------------------------------------------------------------
# Group newton


def _fit(inputs, beta0=None, **kw) -> dict:
    lik, X, y, aux, S, basis = inputs
    start = np.zeros(X.shape[1]) if beta0 is None else np.asarray(beta0, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        return _newton(
            penalized_newton(lik, X, y, aux, S, start, free_basis=basis, **kw)
        )


def _unpenalized(lik, X, y, aux):
    p = X.shape[1]
    return lik, X, y, aux, np.zeros((p, p)), None


def _separated(name: str, seed: int = 1):
    """A logistic 0/1 cell whose rows are all y = 1, a Poisson cell with no
    events, a logistic column that quasi-separates the outcome.

    Along the direction of unbounded increase the decrement falls by ``e``
    per step, so the stop lies between a decrement within a factor ``e`` above
    the tolerance and one below it. The logistic cell uses seed 6, where these
    are 1.60 and 0.59 times it; from seed 1 they were 1.18 and 0.43."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=300)
    high = (x > 1.5).astype(float)
    y = (rng.uniform(size=300) < 0.3).astype(float)
    y[high == 1] = 1.0
    if name == "logistic_cell":
        return _unpenalized(
            Bernoulli(), np.column_stack([np.ones(300), x, high]), y, None
        )
    counts = rng.poisson(1.0, size=300).astype(float)
    counts[high == 1] = 0.0
    if name == "poisson_cell":
        X = np.column_stack([np.ones(300), high])
        return _unpenalized(Poisson(), X, counts, np.zeros(300))
    away = rng.choice([-1.0, 1.0], size=200) * rng.uniform(0.5, 2.0, size=200)
    z = np.r_[away, np.zeros(10)]
    quasi_y = np.r_[(z[:200] > 0).astype(float), np.tile([0.0, 1.0], 5)]
    return _unpenalized(Bernoulli(), np.column_stack([np.ones(210), z]), quasi_y, None)


def _far_maximum():
    """A Poisson slope of 2e3 on a column of units 1e-3. Seed 4: with seed 3
    the converged fit's last decrement was 1.08 times the tolerance; here the
    closest is 871 times it, on the same path (step limit at 40 steps,
    converged in about 190)."""
    rng = np.random.default_rng(4)
    x = rng.uniform(0.0, 1e-3, 400)
    X = np.column_stack([np.ones(400), x])
    y = rng.poisson(np.exp(0.2 + 2e3 * x)).astype(float)
    return _unpenalized(Poisson(), X, y, np.zeros(400))


def _collinear():
    rng = np.random.default_rng(0)
    x = rng.uniform(size=300)
    X = np.column_stack([np.ones(300), x, 3.0 * x])
    y = rng.poisson(np.exp(0.5 + np.sin(3.0 * x))).astype(float)
    return _unpenalized(Poisson(), X, y, np.zeros(300))


def _softmax_constant_within_groups():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(50), 4)
    x = rng.normal(size=200)
    X = np.column_stack([x, x + np.repeat(rng.normal(size=50), 4)])
    y = rng.dirichlet(np.ones(4), size=50).ravel()
    return _unpenalized(GroupedSoftmax(), X, y, groups)


def _without_penalty(inputs):
    lik, X, y, aux, S, _ = inputs
    keep = ~(S != 0).any(axis=0)  # the unpenalized columns
    return _unpenalized(lik, X[:, keep], y, aux)


NEWTON_CASES: dict[str, Callable[[], dict]] = {
    "poisson_unpenalized": lambda: _fit(
        _without_penalty(_diagonal_inputs("poisson", 0, 12, 2.0))
    ),
    "poisson_penalized": lambda: _fit(_diagonal_inputs("poisson", 0, 12, 2.0)),
    "poisson_spline_40_weak": lambda: _fit(_diagonal_inputs("poisson", 1, 41, -6.0)),
    "bernoulli_unpenalized": lambda: _fit(
        _without_penalty(_diagonal_inputs("bernoulli", 2, 12, 2.0))
    ),
    "bernoulli_penalized": lambda: _fit(_diagonal_inputs("bernoulli", 2, 12, 2.0)),
    "gaussian_penalized": lambda: _fit(_diagonal_inputs("gaussian", 3, 12, 2.0)),
    "gaussian_unpenalized": lambda: _fit(
        _without_penalty(_diagonal_inputs("gaussian", 3, 12, 2.0))
    ),
    "softmax_penalized": lambda: _fit(_softmax_inputs(4)),
    "softmax_unpenalized": lambda: _fit(_without_penalty(_softmax_inputs(4))),
    "bernoulli_saturated_start": lambda: _fit(
        (Bernoulli(), *_logistic(), None, np.diag([0.0, 0.5]), None), [200.0, -300.0]
    ),
    "step_limit": lambda: _fit(
        _unpenalized(Bernoulli(), *_logistic(), None), [25.0, -30.0], max_iter=2
    ),
    "not_positive_definite": lambda: _fit(
        _unpenalized(
            Poisson(), np.array([[1.0], [2.0]]), np.array([1.0, 2.0]), np.zeros(2)
        ),
        [800.0],
    ),
    "not_identified": lambda: _fit(_collinear()),
    "softmax_not_identified": lambda: _fit(_softmax_constant_within_groups()),
    "no_finite_maximum_logistic_cell": lambda: _fit(_separated("logistic_cell", 6)),
    "no_finite_maximum_poisson_cell": lambda: _fit(_separated("poisson_cell")),
    "no_finite_maximum_quasi": lambda: _fit(_separated("quasi")),
    "far_maximum_step_limit": lambda: _fit(_far_maximum(), max_iter=40),
    "far_maximum_converged": lambda: _fit(_far_maximum(), max_iter=400),
    # Seed 6: from seed 1 the last decrement was 1.06 times the tolerance.
    "bernoulli_spline_40_weak": lambda: _fit(
        _diagonal_inputs("bernoulli", 6, 41, -6.0)
    ),
    "gaussian_spline_40_weak": lambda: _fit(_diagonal_inputs("gaussian", 1, 41, -8.0)),
    "softmax_spline_40_weak": lambda: _fit(_softmax_inputs(4, 41, -6.0)),
}


# --------------------------------------------------------------------------
# Group reml


def _smooth_session(
    family: str,
    n_smooths: int,
    seed: int,
    n_rows: int = 240,
    amplitudes: tuple[float, ...] | None = None,
):
    """A likelihood, a session with ``n_smooths`` smooths of 6 columns each
    (an intercept first but under the grouped softmax) and its blocks. The
    effect of smooth ``j`` is ``a_j sin(pi (j + 1) x_j)``, ``a_j`` of
    ``amplitudes`` (0.7 each by default)."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(size=(n_smooths, n_rows))
    a = (0.7,) * n_smooths if amplitudes is None else amplitudes
    truth = sum(a[j] * np.sin(np.pi * (j + 1) * x[j]) for j in range(n_smooths))
    parts = [] if family == "grouped_softmax" else [np.ones((n_rows, 1))]
    blocks, start = [], len(parts)
    for j in range(n_smooths):
        design, penalty = _smooth(x[j], 0.0, 1.0, 7)
        parts.append(design)
        blocks.append(PenaltyBlock(np.arange(start, start + design.shape[1]), penalty))
        start += design.shape[1]
    X = np.hstack(parts)
    if family == "poisson":
        offset = np.log(rng.uniform(0.5, 2.0, size=n_rows))
        y = rng.poisson(np.exp(0.7 + truth + offset)).astype(float)
        return Poisson(), SessionData(X, y, offset), blocks
    if family == "bernoulli":
        y = (rng.uniform(size=n_rows) < 1 / (1 + np.exp(-truth))).astype(float)
        return Bernoulli(), SessionData(X, y), blocks
    if family == "gaussian":
        y = 1.0 + truth + rng.normal(scale=0.5, size=n_rows)
        return Gaussian(), SessionData(X, y), blocks
    weight = np.exp(truth).reshape(-1, 4)
    y = np.concatenate([rng.dirichlet(30.0 * row / row.sum()) for row in weight])
    return (
        GroupedSoftmax(),
        SessionData(X, y, np.repeat(np.arange(n_rows // 4), 4)),
        blocks,
    )


def _reml(lik, data: SessionData, blocks, rho) -> dict:
    with np.errstate(over="ignore", invalid="ignore"):
        score = reml_and_derivatives(lik, data.X, data.y, data.aux, blocks, rho)
    return _score(score)


def _reml_case(family: str, n_smooths: int, rho: list[float]) -> Callable[[], dict]:
    def case() -> dict:
        lik, data, blocks = _smooth_session(family, n_smooths, seed=11 + n_smooths)
        return _reml(lik, data, blocks, rho)

    return case


def _reml_design_case(family: str, n_basis: int, rho: float) -> Callable[[], dict]:
    """A design's blocks (a smooth, a weak slope, two weak levels) at their
    log penalties, the smooth's at ``rho``."""

    def case() -> dict:
        lik, X, y, aux, _, _ = _diagonal_inputs(family, 7, n_basis, rho)
        design = _diagonal_design(_rows(7), n_basis, rho)
        if family == "gaussian":
            aux = None  # profiled
        return _reml(lik, SessionData(X, y, aux), design.blocks, design.log_penalties)

    return case


REML_CASES: dict[str, Callable[[], dict]] = {
    f"{family}_{m}_smooths_rho_{'_'.join(f'{r:g}' for r in rho)}": _reml_case(
        family, m, rho
    )
    for family in ("poisson", "bernoulli", "gaussian", "grouped_softmax")
    for m, points in ((1, ([-2.0], [6.0], [20.0])), (2, ([-2.0, 14.0], [20.0, 20.0])))
    for rho in points
}
REML_CASES.update(
    {
        f"{family}_design_simple_terms_rho_{rho:g}": _reml_design_case(family, 12, rho)
        for family in ("poisson", "bernoulli", "gaussian")
        for rho in (2.0,)
    }
)
REML_CASES.update(
    {
        f"{family}_spline_40_rho_{rho:g}": _reml_design_case(family, 41, rho)
        for family, rho in (("poisson", -4.0), ("bernoulli", -4.0), ("gaussian", -6.0))
    }
)


def _reml_softmax_design_case(n_basis: int, rho: float) -> Callable[[], dict]:
    """The content design's blocks (the role category and the slope by role,
    each weakly penalized, and a smooth of the clock at ``rho``) at their log
    penalties, under the grouped softmax."""

    def case() -> dict:
        frame = _softmax_frame(7, early_clock=n_basis > 10)
        design = _content_design(frame, rho, n_basis)
        data = SessionData(
            design.build(frame).X,
            frame["share"].to_numpy(),
            frame["event_id"].to_numpy(),
        )
        return _reml(GroupedSoftmax(), data, design.blocks, design.log_penalties)

    return case


REML_CASES.update(
    {
        "grouped_softmax_design_simple_terms_rho_2": _reml_softmax_design_case(10, 2.0),
        "grouped_softmax_spline_40_rho_-4": _reml_softmax_design_case(41, -4.0),
    }
)


def _line_beside_its_smooth() -> dict:
    """A Poisson session whose design is not identified beside a penalty
    block: an intercept, a smooth of ``x`` and the slope of ``x``, which lies
    in the smooth's null space (its straight line)."""
    rng = np.random.default_rng(12)
    x = rng.uniform(size=240)
    design, penalty = _smooth(x, 0.0, 1.0, 7)
    X = np.column_stack([np.ones(240), design, x - x.mean()])
    y = rng.poisson(np.exp(0.5 + 0.7 * np.sin(np.pi * x))).astype(float)
    blocks = [PenaltyBlock(np.arange(1, 1 + design.shape[1]), penalty)]
    return _reml(Poisson(), SessionData(X, y, np.zeros(240)), blocks, [2.0])


REML_CASES["poisson_not_identified_line_beside_its_smooth"] = _line_beside_its_smooth


# --------------------------------------------------------------------------
# Group refit


def _content_session(
    seed: int, n_events: int = 140, n_arms: int = 6, by_the_clock: bool = False
):
    """A grouped-softmax session: a P-spline of a clock (9 columns), two 0/1
    role columns; ``by_the_clock``: each event's response on its row of
    smallest clock, which the smooth's line separates."""
    rng = np.random.default_rng(seed)
    n_rows = n_events * n_arms
    x = rng.uniform(0.0, 5.0, n_rows)
    design, penalty = _smooth(x, 0.0, 5.0, 10)
    role = np.zeros((n_rows, 2))
    first = np.arange(n_events) * n_arms
    role[first, 0] = 1.0
    role[first + 1 + rng.integers(0, n_arms - 1, n_events), 1] = 1.0
    X = np.column_stack([design, role])
    eta = -1.2 * x + role @ np.array([1.0, -0.5])
    weight = np.exp(eta).reshape(n_events, n_arms)
    p = weight / weight.sum(axis=1, keepdims=True)
    y = np.zeros((n_events, n_arms))
    y[np.arange(n_events), [rng.choice(n_arms, p=row) for row in p]] = 1.0
    if by_the_clock:
        y = np.zeros((n_events, n_arms))
        y[np.arange(n_events), np.argmin(x.reshape(n_events, n_arms), axis=1)] = 1.0
    groups = np.repeat(np.arange(n_events), n_arms)
    return SessionData(X, y.ravel(), groups, name=f"s{seed}"), [
        PenaltyBlock(np.arange(9), penalty)
    ]


def _with_arm_columns(data: SessionData, seed: int, separated: bool) -> SessionData:
    """Two unpenalized 0/1 columns on the second and third rows; separated:
    the direction "arm down, role up" lowers only rows never chosen."""
    rng = np.random.default_rng(seed)
    n_arms = 6
    n_events = data.X.shape[0] // n_arms
    y = data.y.reshape(n_events, n_arms).copy()
    arm = np.zeros((n_events, n_arms))
    role = np.zeros((n_events, n_arms))
    if separated:
        y[:40, 0] += y[:40, 1]
        y[:40, 1] = 0.0
        arm[:40, 1] = 1.0
        arm[40:80, 2] = 1.0
        role[40:80, 2] = 1.0
    else:
        arm[rng.uniform(size=n_events) < 0.5, 2] = 1.0
        role[rng.uniform(size=n_events) < 0.3, 2] = 1.0
    X = np.column_stack([data.X, arm.ravel(), role.ravel()])
    return SessionData(X, y.ravel(), data.aux, name=data.name)


def _near_copies(seed: int, separated: bool, n_copies: int = 10):
    """A Bernoulli session: an intercept, a P-spline (9 columns) and
    ``n_copies`` near-copies of a 0/1 column; separated: every row of the
    column has y = 1."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 4.0, 600)
    design, penalty = _smooth(x, 0.0, 4.0, 10)
    indicator = (rng.uniform(size=600) < 0.2).astype(float)
    copies = [indicator * (1.0 + 0.05 * rng.normal(size=600)) for _ in range(n_copies)]
    X = np.column_stack([np.ones(600), design, *copies])
    y = (rng.uniform(size=600) < 1 / (1 + np.exp(-np.sin(x)))).astype(float)
    if separated:
        y[indicator == 1] = 1.0
    return SessionData(X, y, name=f"s{seed}"), [PenaltyBlock(np.arange(1, 10), penalty)]


def _rate_levels(seed: int, empty_reference: bool):
    """A Poisson session of 40 visits of 30 bins: an intercept, two level
    columns, a slope and a P-spline of 7 columns; ``empty_reference``: no
    event on the reference level's rows."""
    rng = np.random.default_rng(seed)
    visit = np.repeat(np.arange(40), 30)
    level = np.where(visit < 10, 0, np.where(visit < 25, 1, 2))
    x = rng.normal(size=40)[visit]
    t = np.tile(np.arange(30) * 0.1, 40)
    count = rng.poisson(0.2, visit.size).astype(float)
    if empty_reference:
        count[level == 0] = 0.0
    design, penalty = _smooth(t, 0.0, 3.0, 8)
    X = np.column_stack([np.ones(visit.size), level == 1, level == 2, x, design])
    offset = np.full(visit.size, np.log(0.1))
    data = SessionData(X, count, offset, name=f"s{seed}")
    return data, [PenaltyBlock(np.arange(4, 11), penalty)]


def _refit_case(lik, built, rho) -> Callable[[], dict]:
    def case() -> dict:
        data, blocks = built()
        with np.errstate(over="ignore", invalid="ignore"):
            return _refit(refit_session(lik, data, blocks, rho))

    return case


REFIT_CASES: dict[str, Callable[[], dict]] = {
    # Six near-copies: with ten, the copies shared a step of about 1, so each
    # column's part of it was about NO_MAXIMUM_ETA_STEP (1.06 times it).
    "columns_left_out_in_rounds": _refit_case(
        Bernoulli(), lambda: _near_copies(0, separated=True, n_copies=6), [6.0]
    ),
    "arm_and_role_left_out": _refit_case(
        GroupedSoftmax(),
        lambda: (
            _with_arm_columns(_content_session(4)[0], 4, separated=True),
            _content_session(4)[1],
        ),
        [6.0],
    ),
    "along_a_penalty": _refit_case(
        GroupedSoftmax(), lambda: _content_session(6, by_the_clock=True), [6.0]
    ),
    "along_the_intercept": _refit_case(
        Poisson(), lambda: _rate_levels(0, empty_reference=True), [6.0]
    ),
    "converged_first": _refit_case(
        Poisson(), lambda: _rate_levels(1, empty_reference=False), [6.0]
    ),
}


# --------------------------------------------------------------------------
# Group select


def _select_case(lik, sessions: Callable[[], tuple[list, list]]) -> Callable[[], dict]:
    def case() -> dict:
        datas, blocks = sessions()
        with np.errstate(over="ignore", invalid="ignore"):
            return _pooled(select_penalties(lik, datas, blocks))

    return case


def _smooth_sessions(family: str, n_smooths: int, n: int, first_seed: int = 40, **kw):
    built = [
        _smooth_session(family, n_smooths, seed=first_seed + s, **kw) for s in range(n)
    ]
    return [b[1] for b in built], built[0][2]


def _wiggly_bernoulli(seed: int, flip: float):
    """A Bernoulli session of 300 rows: an intercept and a P-spline of ``x``
    (9 columns); ``y`` is 1 where ``sin(3 pi x) > 0``, each row flipped with
    probability ``flip``. Unflipped, the spline's wiggly part separates the
    outcome: at the weakest penalty its maximum is far, and Newton from zero
    runs out of steps (117 needed at log penalty -10). The session and its
    block."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(size=300)
    y = (np.sin(3.0 * np.pi * x) > 0).astype(float)
    flips = rng.uniform(size=300) < flip
    y[flips] = 1.0 - y[flips]
    design, penalty = _smooth(x, 0.0, 1.0, 10)
    X = np.column_stack([np.ones(300), design])
    return SessionData(X, y, name=f"w{seed}"), [
        PenaltyBlock(np.arange(1, 1 + design.shape[1]), penalty)
    ]


def _one_fails_at_the_weakest_point():
    """Two noisy wiggly sessions and one separated by its spline: the grid
    point at log penalty -10 fails (step limit) in the separated session,
    the others are finite."""
    built = [_wiggly_bernoulli(3, 0.0)] + [
        _wiggly_bernoulli(100 + s, 0.05) for s in range(2)
    ]
    return [b[0] for b in built], built[0][1]


def _content_sessions_set_aside():
    sessions = []
    for seed in range(5):
        data, blocks = _content_session(seed)
        sessions.append(_with_arm_columns(data, seed, separated=seed == 2))
    return sessions, blocks


def _rate_sessions_intercept():
    built = [_rate_levels(seed, empty_reference=seed == 0) for seed in range(6)]
    return [b[0] for b in built], built[0][1]


SELECT_CASES: dict[str, Callable[[], dict]] = {
    "poisson_one_smooth": _select_case(
        Poisson(), lambda: _smooth_sessions("poisson", 1, 4)
    ),
    # Seeds 43 to 45: from seeds 40 to 42 a Newton decrement of the search
    # was 1.0009 times the tolerance; here the closest is 1.37 times it.
    "poisson_two_smooths": _select_case(
        Poisson(), lambda: _smooth_sessions("poisson", 2, 3, first_seed=43)
    ),
    # Seeds 44 to 47 (the closest decrement 1.30 times the tolerance; 0.95
    # from seeds 40 to 43).
    "bernoulli_one_smooth": _select_case(
        Bernoulli(), lambda: _smooth_sessions("bernoulli", 1, 4, first_seed=44)
    ),
    "gaussian_one_smooth": _select_case(
        Gaussian(), lambda: _smooth_sessions("gaussian", 1, 4)
    ),
    "softmax_one_set_aside_and_refitted": _select_case(
        GroupedSoftmax(), _content_sessions_set_aside
    ),
    "poisson_one_without_a_fit": _select_case(Poisson(), _rate_sessions_intercept),
    "bernoulli_one_fails_at_the_weakest_grid_point": _select_case(
        Bernoulli(), _one_fails_at_the_weakest_point
    ),
    # Six outer steps from the best grid point. With the second smooth
    # without an effect its term stops on the score's plateau (log penalty
    # 16, 11 steps), but there the stop is set by the tolerance: the last
    # decrement was 0.93 times it, and the log penalty moved with it.
    "gaussian_two_smooths_six_steps": _select_case(
        Gaussian(),
        lambda: _smooth_sessions(
            "gaussian", 2, 3, first_seed=5000, amplitudes=(0.7, 0.4)
        ),
    ),
    "poisson_four_smooths": _select_case(
        Poisson(), lambda: _smooth_sessions("poisson", 4, 3, first_seed=60, n_rows=400)
    ),
}


# --------------------------------------------------------------------------
# Group sessions


def _rate_frame(seed: int, n_visits: int = 50) -> pd.DataFrame:
    """One session's visit bins (0.1 s): exposure, a Poisson count, the clock
    since arrival, a per-visit ``x``, the last outcome (0/1) and a three-level
    ``level`` per visit."""
    rng = np.random.default_rng(2000 + seed)
    n_bins = np.maximum(
        1, np.round(20.0 * np.exp(0.6 * rng.normal(size=n_visits)))
    ).astype(int)
    visit = np.repeat(np.arange(n_visits), n_bins)
    since = np.concatenate([0.1 * np.arange(k) for k in n_bins])
    x = rng.normal(size=n_visits)[visit]
    last = (rng.uniform(size=n_visits) < 0.6).astype(float)[visit]
    level = rng.choice(np.array(["a", "b", "c"], dtype=object), n_visits)[visit]
    exposure = np.full(visit.size, 0.1)
    log_rate = (
        np.log(0.5)
        + np.log(3.0) * (1.0 - np.exp(-since / 0.7))
        + 0.4 * last
        + 0.2 * x
        + np.where(level == "b", 0.3, 0.0)
    )
    return pd.DataFrame(
        {
            "animal": "sim",
            "session_date": f"2026{seed:04d}",
            "epoch": "02_r1",
            "well_visit_id": visit,
            "exposure": exposure,
            "count": rng.poisson(exposure * np.exp(log_rate)).astype(float),
            "time_since_s": since,
            "x": x,
            "last_rewarded": last,
            "level": level,
        }
    )


def _by_session(frames: list[pd.DataFrame]) -> dict:
    return {tuple(f.loc[0, ["animal", "session_date", "epoch"]]): f for f in frames}


def _records(design, sessions, lik, response: str, unit: str) -> list[dict]:
    """The records of ``fit_sessions``, without ``fit_by`` and ``sessions``.

    The references are of the NumPy fitter, whose record had no unit of the
    fit. Every case here is a per-session fit, where the two fields repeat what
    the record already holds; that is checked here before they are left out of
    the comparison.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        records = fit_sessions(
            design,
            sessions,
            lik,
            model="reference",
            kind="SWR",
            response=response,
            unit=unit,
            table_sha256={"t": "0" * 64},
        )
    out = []
    for r in records:
        record = asdict(r)
        key = (r.animal, r.session_date, r.epoch)
        if record.pop("fit_by") != "session":
            raise AssertionError(f"session {key}: fit_by {r.fit_by!r}, not 'session'")
        entries = record.pop("sessions")
        if (
            len(entries) != 1
            or tuple(entries[0][k] for k in ("animal", "session_date", "epoch")) != key
        ):
            raise AssertionError(f"session {key}: sessions {entries!r}")
        for field in ("n_rows", "n_units", "n_events"):
            if entries[0][field] != record[field]:
                raise AssertionError(
                    f"session {key}: sessions[0][{field!r}] {entries[0][field]!r}, "
                    f"the record's {record[field]!r}"
                )
        out.append(record)
    return out


def _rate_sessions() -> dict:
    """Five rate sessions: as drawn; one with no event after a rewarded last
    outcome (the unpenalized ``last_rewarded`` has no finite maximum and is
    left out); one with no event on level ``a``, the reference (without a
    penalty on the levels, no finite maximum along the intercept); one with a
    missing clock on some rows; one with no event (not fitted)."""
    frames = [_rate_frame(seed) for seed in range(5)]
    frames[1].loc[frames[1]["last_rewarded"] == 1, "count"] = 0.0
    frames[2].loc[frames[2]["level"] == "a", "count"] = 0.0
    frames[3].loc[:9, "time_since_s"] = np.nan
    frames[4]["count"] = 0.0
    return _by_session(frames)


def _rate_records(level_penalty) -> Callable[[], list[dict]]:
    def case() -> list[dict]:
        sessions = _rate_sessions()
        pooled = pd.concat(sessions.values(), ignore_index=True)
        level = Category("level", ("a", "b", "c"), reference="a", penalty=level_penalty)
        design = build_design(
            [
                Smooth("time_since_s", 0.0, 6.0, 8, 2.0),
                Slope("x", WEAK),
                Slope("last_rewarded", "none"),
                level,
                Offset("exposure"),
            ],
            pooled.dropna(),
            intercept=True,
        )
        return _records(design, sessions, Poisson(), "count", "well_visit_id")

    return case


def _content_records() -> list[dict]:
    frames = [_content_frame(seed, n_events=80) for seed in range(3)]
    # A session whose "previous" rows are all missing: the level and its
    # slope leave with their rows.
    frames[2].loc[frames[2]["role"] == "previous", "reward_fraction"] = np.nan
    sessions = _by_session(frames)
    pooled = pd.concat(sessions.values(), ignore_index=True)
    pooled["reward_fraction"] = pooled["reward_fraction"].fillna(0.5)
    return _records(
        _content_design(pooled), sessions, GroupedSoftmax(), "share", "event_id"
    )


def _diagonal_frames(family: str) -> dict:
    frames = []
    for seed in range(3):
        rows = _rows(20 + seed, n=400)
        rng = np.random.default_rng(300 + seed)
        eta = _eta(rows)
        if family == "bernoulli":
            rows["y"] = (rng.uniform(size=eta.size) < 1 / (1 + np.exp(-eta))).astype(
                float
            )
        else:
            rows["y"] = 1.0 + eta + rng.normal(scale=0.8, size=eta.size)
        rows["unit"] = np.arange(len(rows)) // 4
        rows["animal"], rows["session_date"], rows["epoch"] = (
            "sim",
            f"2026{seed:04d}",
            "02_r1",
        )
        frames.append(rows)
    return _by_session(frames)


def _diagonal_records(family: str) -> Callable[[], list[dict]]:
    def case() -> list[dict]:
        sessions = _diagonal_frames(family)
        pooled = pd.concat(sessions.values(), ignore_index=True)
        design = build_design(
            [
                Smooth("clock", 0.0, 20.0, 8, 2.0),
                Slope("x", WEAK),
                LEVEL,
                Slope("flag", "none"),
            ],
            pooled,
            intercept=True,
        )
        lik = Bernoulli() if family == "bernoulli" else Gaussian()
        return _records(design, sessions, lik, "y", "unit")

    return case


SESSIONS_CASES: dict[str, Callable[[], list[dict]]] = {
    "rate_weak_level": _rate_records(WEAK),
    "rate_unpenalized_level": _rate_records("none"),
    "content": _content_records,
    "bernoulli": _diagonal_records("bernoulli"),
    "gaussian": _diagonal_records("gaussian"),
}


# --------------------------------------------------------------------------
# Group calibration


def _calibration(target: float) -> Callable[[], dict]:
    def case() -> dict:
        sessions = _by_session([_rate_frame(seed) for seed in range(6)])
        pooled = pd.concat(sessions.values(), ignore_index=True)
        design = build_design(
            [
                Smooth("time_since_s", 0.0, 6.0, 12, 2.0),
                Slope("x", WEAK),
                Offset("exposure"),
            ],
            pooled,
            intercept=True,
        )
        found = log_penalty_for_edf(
            design,
            sessions,
            Poisson(),
            term="s(time_since_s)",
            target_edf=target,
            response="count",
            unit="well_visit_id",
        )
        return {
            "log_penalty": found.log_penalty,
            "median_edf": found.median_edf,
            "n_sessions_used": found.n_sessions_used,
            "edf": found.edf.to_list(),
        }

    return case


CALIBRATION_CASES: dict[str, Callable[[], dict]] = {
    "rate_edf_4": _calibration(4.0),
}

#: Group name -> its cases, name -> a function that returns its outputs.
GROUPS: dict[str, dict[str, Callable[[], object]]] = {
    "newton": NEWTON_CASES,
    "reml": REML_CASES,
    "refit": REFIT_CASES,
    "select": SELECT_CASES,
    "sessions": SESSIONS_CASES,
    "calibration": CALIBRATION_CASES,
}

#: ``(group, case)`` -> the case's function.
CASES: dict[tuple[str, str], Callable[[], object]] = {
    (group, name): case
    for group, cases in GROUPS.items()
    for name, case in cases.items()
}


def run(group: str, name: str) -> object:
    """The plain outputs of one case (:func:`plain`)."""
    return plain(CASES[group, name]())
