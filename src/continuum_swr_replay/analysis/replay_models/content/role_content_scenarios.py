"""Planted truths for the role content models, as functions of a session's rows.

A scenario says, for each candidate row of a session's real cohort rows
(:func:`.role_content.role_content_cohort`), the share of its event that the
truth expects (:func:`expected_shares`), and the value of every declared
content estimand under that truth (:func:`planted_values`). A simulation draws
the response from those shares, fits the real model and compares the
estimates with the planted values; nothing here draws or fits.

A truth (:class:`Truth`) is written in the model's own coordinates, so that
its planted values are the estimands themselves:

- arm effects (a baseline);
- role levels against the other-leaf role (``current``, ``previous``,
  ``trunk``);
- slopes per pooled SD of a covariate on the rows of one leaf role, centered at
  the pooled reference, as the model centers a slope by role;
- curves of a covariate on the rows of one role, centered on the pooled rows
  of that role, as the model's sum-to-zero constraint centers a smooth;
- length terms: per unit of ``log(duration) - L0``, a role level's and a role
  slope's change, which only the length-adjusted variant holds.

The pooled quantities (each covariate's pooled mean and SD, each curve's
pooled mean, ``L0``, the reward fraction's mean and SD on current rows) are
those of the scale rows the model's design is built on: both kinds, or the
kind fitted (``role_content.SCALE_POOLINGS``; :class:`ScenarioReference`). A truth from the data, or one whose effects
vary over sessions and animals, is a :class:`Truth` per session. For those,
the target of the project's reduction (session, then recording day, then
animal, a mean of means) is :func:`planted_target` of the per-session planted
values.

The scenario list (:func:`role_content_scenarios`) names, for each scenario,
the estimands it validates and what result would be a failure of the model.
The uncertainty model's own truths are in :mod:`.uncertainty_scenarios`.

Each scenario also carries the noise of its simulated shares
(:class:`ShareNoise`): one categorical draw per event, a multinomial over the
event's decode bins, or a Dirichlet-multinomial over them. The noise is not
an assumption of the scenarios: it is a parameter of each, and the results of
the validation (the spread of the estimates, the coverage, the shrinkage by
the weak penalty) depend on it. Which noise the real shares have is open.
Nothing here draws; the simulation engine's draw function reads the choice.

"The 95 % interval covers" in a scenario's failure means the project's
interval over sessions, recording days and animals
(``shared_statistics.estimand_summary``), never the record's per-session
standard error, which is on the model's scale and not a sampling error.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.regression.contrasts import range_grid
from continuum_swr_replay.analysis.regression.sessions import SessionKey
from continuum_swr_replay.analysis.shared_statistics import animal_day_session_values

from .. import declared_tests as dt
from . import role_content as rc

__all__ = [
    "NOISES",
    "SIZES",
    "Scenario",
    "ScenarioReference",
    "ShareNoise",
    "Truth",
    "arm_effects",
    "expected_shares",
    "leaving_curve",
    "planted_target",
    "planted_values",
    "recency_curve",
    "reward_curve",
    "role_content_scenarios",
    "spread_truths",
    "thin_events",
    "truth_from_data",
]

Curve = Callable[[np.ndarray], np.ndarray]

#: The small and the large planted size of each kind of effect: a role level
#: (log share ratio), a slope (per pooled SD) and a curve's amplitude.
SIZES: Mapping[str, Mapping[str, float]] = {
    "role": {"small": 0.25, "large": 1.0},
    "slope": {"small": 0.1, "large": 0.4},
    "curve": {"small": 0.25, "large": 1.0},
}


NOISES: tuple[str, ...] = ("categorical", "multinomial", "dirichlet_multinomial")
"""OPEN (lead): the noise of the simulated shares. ``categorical``: each
event's shares are one categorical draw (one arm takes the whole event).
``multinomial``: the counts of a multinomial over the event's number of decode
bins, divided by that number. ``dirichlet_multinomial``: the same with
overdispersion, the bins' probabilities a Dirichlet draw about the expected
shares with a stated concentration."""


@dataclass(frozen=True)
class ShareNoise:
    """The noise of a scenario's simulated shares (:data:`NOISES`).

    Attributes
    ----------
    kind : str
        One of :data:`NOISES`.
    n_bins_column : str or None
        The cohort column that gives each event's number of decode bins (for
        example the event-arm table's ``n_nonlocal_bins``); required for the
        multinomial and the Dirichlet-multinomial, None for the categorical.
    concentration : float or None
        The Dirichlet's total concentration; required for, and only for, the
        Dirichlet-multinomial.
    """

    kind: str
    n_bins_column: str | None = None
    concentration: float | None = None

    def __post_init__(self) -> None:
        if self.kind not in NOISES:
            raise ValueError(f"noise {self.kind!r} not in {NOISES}")
        if (self.kind == "categorical") != (self.n_bins_column is None):
            raise ValueError(
                "a number-of-bins column is required for a multinomial noise "
                "and not read by the categorical"
            )
        if (self.kind == "dirichlet_multinomial") != (self.concentration is not None):
            raise ValueError(
                "a concentration is required for, and only for, the "
                "Dirichlet-multinomial"
            )
        if self.concentration is not None and not self.concentration > 0:
            raise ValueError("the concentration must be positive")


@dataclass(frozen=True)
class Truth:
    """A planted linear predictor, in the model's coordinates (module docstring).

    Attributes
    ----------
    arm : mapping of int to float
        Arm -> effect; an arm not named is 0.
    role : mapping of str to float
        Role -> its level against the other-leaf role.
    slopes : mapping of (covariate, role) to float
        Per pooled SD of the covariate, on the role's rows.
    curves : mapping of (covariate, role) to callable
        The curve of the covariate (on the model's scale: log trials, a
        fraction, seconds) on the role's rows.
    length : mapping of str to float
        Role -> its level's change per unit of ``log(duration) - L0``.
    length_slopes : mapping of (covariate, role) to float
        A slope's change per unit of ``log(duration) - L0``.
    """

    arm: Mapping[int, float] = field(default_factory=dict)
    role: Mapping[str, float] = field(default_factory=dict)
    slopes: Mapping[tuple[str, str], float] = field(default_factory=dict)
    curves: Mapping[tuple[str, str], Curve] = field(default_factory=dict)
    length: Mapping[str, float] = field(default_factory=dict)
    length_slopes: Mapping[tuple[str, str], float] = field(default_factory=dict)


@dataclass(frozen=True)
class ScenarioReference:
    """The pooled quantities a truth is centered on: those of the scale rows
    the model's design is built on.

    Attributes
    ----------
    pooled_rows : pd.DataFrame
        The scale rows: the cohort rows of every session of the fit, of both
        kinds or of the kind fitted (``role_content.SCALE_POOLINGS``).
    reference_log_duration : float
        ``L0``.
    """

    pooled_rows: pd.DataFrame
    reference_log_duration: float

    def slope_scale(self, covariate: str) -> tuple[float, float]:
        """The pooled mean and population SD of a slope covariate over the
        leaf rows, as the model standardizes it."""
        rows = self.pooled_rows[rc.ROLE].isin(rc.LEAF_ROLES)
        values = self.pooled_rows.loc[rows, covariate].to_numpy(dtype=float)
        return float(np.mean(values)), float(np.std(values, ddof=0))

    def curve_mean(self, curve: Curve, covariate: str, role: str) -> float:
        """The mean of ``curve`` over the pooled rows of ``role``."""
        rows = self.pooled_rows[rc.ROLE] == role
        x = _model_scale(self.pooled_rows.loc[rows, covariate], covariate)
        return float(np.mean(curve(x)))


def _model_scale(values: pd.Series | np.ndarray, covariate: str) -> np.ndarray:
    """A covariate as the model's smooth reads it: clamped to its range."""
    lo, hi = next((s[1], s[2]) for s in rc._SMOOTHS if s[0] == covariate)
    return np.clip(np.asarray(values, dtype=float), lo, hi)


def expected_shares(
    rows: pd.DataFrame, truth: Truth, reference: ScenarioReference
) -> np.ndarray:
    """Each candidate row's expected share of its event under ``truth``.

    Parameters
    ----------
    rows : pd.DataFrame
        One session's cohort rows (any order; events need not be contiguous).
    truth : Truth
    reference : ScenarioReference

    Returns
    -------
    np.ndarray, shape (n_rows,)
        Shares that sum to one over each event's rows.
    """
    eta = _linear_predictor(rows, truth, reference)
    groups = pd.factorize(rows[rc.UNIT])[0]
    top = np.full(groups.max() + 1, -np.inf)
    np.maximum.at(top, groups, eta)
    weight = np.exp(eta - top[groups])
    total = np.bincount(groups, weights=weight)
    return weight / total[groups]


def _linear_predictor(
    rows: pd.DataFrame, truth: Truth, reference: ScenarioReference
) -> np.ndarray:
    """The truth's linear predictor at each row, shape ``(n_rows,)``."""
    role = rows[rc.ROLE].to_numpy()
    arm = rows[rc.ARM].to_numpy()
    eta = np.zeros(len(rows))
    for a, effect in truth.arm.items():
        eta[arm == a] += effect
    length = (
        rows[rc.LOG_DURATION].to_numpy(dtype=float) - reference.reference_log_duration
    )
    for r, effect in truth.role.items():
        eta[role == r] += effect
    for r, effect in truth.length.items():
        eta[role == r] += effect * length[role == r]
    for (covariate, r), effect in truth.slopes.items():
        mean, sd = reference.slope_scale(covariate)
        on = role == r
        eta[on] += effect * (rows.loc[on, covariate].to_numpy(dtype=float) - mean) / sd
    for (covariate, r), effect in truth.length_slopes.items():
        mean, sd = reference.slope_scale(covariate)
        on = role == r
        z = (rows.loc[on, covariate].to_numpy(dtype=float) - mean) / sd
        eta[on] += effect * z * length[on]
    for (covariate, r), curve in truth.curves.items():
        on = role == r
        x = _model_scale(rows.loc[on, covariate], covariate)
        eta[on] += curve(x) - reference.curve_mean(curve, covariate, r)
    return eta


def planted_values(
    truth: Truth,
    reference: ScenarioReference,
    variant: str = "unadjusted",
    settings: rc.RoleContentSettings | None = None,
) -> dict[str, float]:
    """The value of every declared content estimand under ``truth``.

    The estimands are those of the two role content models
    (:func:`.role_content.model_estimands`), read as the model reads them
    (:func:`.role_content.estimand_contrasts`): a role level, a slope per
    pooled SD on the other-leaf rows, a curve contrast. A truth with a length
    term is defined at ``L0`` for the length-adjusted variant; for the
    unadjusted variant an estimand whose truth has a length term is NaN (that
    variant averages over lengths, so it has no planted value).

    Returns
    -------
    dict of str to float
        Estimand id -> planted value.
    """
    settings = settings or rc.RoleContentSettings()
    out = {}
    for model in rc.MODELS:
        for estimand in rc.model_estimands(model):
            out[estimand.estimand_id] = _planted(
                estimand, truth, reference, variant, settings
            )
    return out


def _curve_at(truth: Truth, covariate: str, role: str, x: np.ndarray) -> np.ndarray:
    curve = truth.curves.get((covariate, role))
    return np.zeros_like(x) if curve is None else curve(_model_scale(x, covariate))


def _planted(
    estimand: dt.Estimand,
    truth: Truth,
    reference: ScenarioReference,
    variant: str,
    settings: rc.RoleContentSettings,
) -> float:
    eid = estimand.estimand_id
    unadjusted = variant == "unadjusted"
    if eid in ("Q3a", "Q3b"):
        role = "current" if eid == "Q3a" else "previous"
        if unadjusted and role in truth.length:
            return float("nan")
        return float(truth.role.get(role, 0.0))
    if eid in ("Q4b", "Q5b-var", "Q5b-n"):
        covariate = {
            "Q4b": rc.VALUE,
            "Q5b-var": rc.BELIEF_VARIANCE,
            "Q5b-n": rc.SAMPLE_COUNT,
        }[eid]
        key = (covariate, rc.REFERENCE_ROLE)
        if unadjusted and key in truth.length_slopes:
            return float("nan")
        return float(truth.slopes.get(key, 0.0))
    if eid == "Q4a":
        rows = reference.pooled_rows
        current = rows.loc[rows[rc.ROLE] == "current", rc.REWARD_FRACTION]
        if settings.q4a_reading == "sd_change":
            m, sd = float(np.mean(current)), float(np.std(current, ddof=0))
            points = np.array([m + sd / 2, m - sd / 2])
        else:
            points = np.array([1.0, 0.0])
        f = _curve_at(truth, rc.REWARD_FRACTION, "current", points)
        return float(f[0] - f[1])
    if eid == "Q5a":
        points = np.log(np.asarray(settings.q5a_trials, dtype=float))
        f = _curve_at(truth, rc.RECENCY, rc.REFERENCE_ROLE, points)
        return float(f[0] - f[1])
    if eid == "Q7b":
        contrast = estimand.contrast
        first = _curve_at(truth, rc.LEAVING, "current", range_grid(*contrast.first))
        second = _curve_at(truth, rc.LEAVING, "current", range_grid(*contrast.second))
        return float(first.mean() - second.mean())
    raise ValueError(f"{eid} is not a role content estimand")


def planted_target(planted: pd.DataFrame) -> pd.Series:
    """The target of the project's reduction of per-session planted values.

    Parameters
    ----------
    planted : pd.DataFrame
        One row per session: ``animal``, ``session_date``, ``epoch`` and one
        column per estimand.

    Returns
    -------
    pd.Series
        Estimand id -> the mean over animals of each animal's mean over its
        recording days of the days' means over their sessions
        (``shared_statistics.animal_day_session_values``), what
        ``estimand_summary`` estimates.
    """
    labels = [planted[c] for c in rc.SESSION_COLUMNS]
    return pd.Series(
        {
            column: float(animal_day_session_values(planted[column], *labels).mean())
            for column in planted.columns
            if column not in rc.SESSION_COLUMNS
        }
    )


# ---------------------------------------------------------------------------
# Planted shapes
# ---------------------------------------------------------------------------


def arm_effects() -> dict[int, float]:
    """Fixed arm effects (a baseline every scenario carries): leaf arms
    between -0.3 and 0.3, trunk arms -1 (the trunk arms hold little content)."""
    return {0: 0.0, 1: 0.2, 2: -0.3, 3: 0.1, 4: 0.3, 5: -0.2, 6: -1.0, 7: -1.2, 8: -0.8}


def _recency(x: np.ndarray, amplitude: float) -> np.ndarray:
    return amplitude * (1.0 - np.exp(-(np.asarray(x) - np.log(3.0)) / 1.2))


def _reward(x: np.ndarray, amplitude: float) -> np.ndarray:
    return amplitude * np.tanh(3.0 * (np.asarray(x) - 0.5)) / np.tanh(1.5)


def _leaving(u: np.ndarray, amplitude: float) -> np.ndarray:
    return amplitude * np.exp(-np.asarray(u) / 0.3)


def recency_curve(amplitude: float) -> Curve:
    """A concave recency curve over log trials: rises from 3 trials and
    flattens, ``amplitude * (1 - exp(-(x - log 3) / 1.2))``. Each curve here
    is a partial of a module-level function, so that a process pool can send
    it."""
    return partial(_recency, amplitude=amplitude)


def reward_curve(amplitude: float) -> Curve:
    """A monotone S-shaped reward-fraction curve over [0, 1], from
    ``-amplitude`` to ``amplitude``."""
    return partial(_reward, amplitude=amplitude)


def leaving_curve(amplitude: float) -> Curve:
    """A rise toward leaving: ``amplitude * exp(-u / 0.3)``, ``u`` the time
    until the final poke-out (s)."""
    return partial(_leaving, amplitude=amplitude)


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Scenario:
    """One planted truth and how it judges the model.

    Attributes
    ----------
    name : str
    model : str
        The model that is fitted (``role_content`` or
        ``role_content_uncertainty``).
    variant : str
        The variant that is fitted.
    validates : tuple of str
        The estimands the scenario validates.
    failure : str
        What result counts as a failure of the model.
    truth_for : callable
        Session key -> its :class:`Truth`.
    noise : ShareNoise
        The noise of the simulated shares (module docstring).
    """

    name: str
    model: str
    variant: str
    validates: tuple[str, ...]
    failure: str
    truth_for: Callable[[SessionKey], Truth]
    noise: ShareNoise

    def expected_shares(
        self, key: SessionKey, rows: pd.DataFrame, reference: ScenarioReference
    ) -> np.ndarray:
        """The session's expected shares (:func:`expected_shares`)."""
        return expected_shares(rows, self.truth_for(key), reference)

    def planted(
        self,
        keys: Sequence[SessionKey],
        reference: ScenarioReference,
        settings: rc.RoleContentSettings | None = None,
    ) -> pd.DataFrame:
        """One row per session: its key and the planted value of every
        estimand (:func:`planted_values`)."""
        rows = []
        for key in keys:
            values = planted_values(
                self.truth_for(key), reference, self.variant, settings
            )
            rows.append(dict(zip(rc.SESSION_COLUMNS, key, strict=True)) | values)
        return pd.DataFrame(rows)


def _constant(truth: Truth) -> Callable[[SessionKey], Truth]:
    return lambda key: truth


_SIZE_FAILURE = (
    "the declared tests of the estimands at zero reject in more than 7 % of "
    "data sets, or their mean estimate differs from 0 by more than its "
    "Monte Carlo SE times 3"
)
_INTERVAL = (
    "the project's 95 % interval over sessions, days and animals (not the "
    "record's per-session SE)"
)
_RECOVERY_FAILURE = (
    "the planted estimand's mean estimate differs from its planted value by "
    "more than 10 % of it (or its Monte Carlo SE times 3, if larger), "
    f"{_INTERVAL} covers it in fewer than 90 % of data sets, or another "
    "estimand planted at zero rejects in more than 7 %"
)


def _single_effects() -> list[tuple[str, str, Truth, tuple[str, ...]]]:
    """(name, model, truth, estimands it plants) for each tested effect alone,
    at each size."""
    out = []
    for size in ("small", "large"):
        r, s, c = (SIZES[k][size] for k in ("role", "slope", "curve"))
        for role, eid in (("current", "Q3a"), ("previous", "Q3b")):
            out.append(
                (f"role_{role}_{size}", "role_content", Truth(role={role: r}), (eid,))
            )
        for role in rc.LEAF_ROLES:
            eid = ("Q4b",) if role == rc.REFERENCE_ROLE else ()
            out.append(
                (
                    f"value_{role}_{size}",
                    "role_content",
                    Truth(slopes={(rc.VALUE, role): s}),
                    eid,
                )
            )
            eid = ("Q4a",) if role == "current" else ()
            out.append(
                (
                    f"reward_curve_{role}_{size}",
                    "role_content",
                    Truth(curves={(rc.REWARD_FRACTION, role): reward_curve(c)}),
                    eid,
                )
            )
        out.append(
            (
                f"leaving_curve_{size}",
                "role_content",
                Truth(curves={(rc.LEAVING, "current"): leaving_curve(c)}),
                ("Q7b",),
            )
        )
        out.append(
            (
                f"recency_curve_{size}",
                "role_content_uncertainty",
                Truth(curves={(rc.RECENCY, rc.REFERENCE_ROLE): recency_curve(c)}),
                ("Q5a",),
            )
        )
        for covariate, eid in (
            (rc.BELIEF_VARIANCE, "Q5b-var"),
            (rc.SAMPLE_COUNT, "Q5b-n"),
        ):
            for role in rc.LEAF_ROLES:
                planted = (eid,) if role == rc.REFERENCE_ROLE else ()
                out.append(
                    (
                        f"{covariate}_{role}_{size}",
                        "role_content_uncertainty",
                        Truth(slopes={(covariate, role): s}),
                        planted,
                    )
                )
    return out


def _ids(model: str) -> tuple[str, ...]:
    """The declared estimands of a model."""
    return tuple(e.estimand_id for e in rc.model_estimands(model))


def role_content_scenarios(*, noise: ShareNoise) -> list[Scenario]:
    """The scenarios that need no data beyond a session's rows, each with the
    noise ``noise`` (module docstring: the results depend on it).

    - ``null`` (arm effects only): every estimand at 0; gives the
      false-positive rate of every content test, in both models.
    - One planted effect at a time (each role level; each role slope; the
      reward-fraction curve of each leaf role; the recency curve; the current
      role's leaving curve; the belief variance and the sample count on each
      leaf role), at a small and a large size, every other effect at 0: the
      planted effect is recovered and does not move to another term.
    - ``length_current`` and ``length_value``: a length interaction of the
      current role's level and of the other-leaf value slope, fitted with the
      length-adjusted variant, which must recover the planted value at ``L0``.
    - ``large_effects``: every tested role level and slope large at once, for
      the weak penalty's shrinkage at each candidate width (run it at each
      prior SD and on thin rows, :func:`thin_events`).

    The truths from the data, the effects that vary over sessions
    (:func:`truth_from_data`, :func:`spread_truths`) and the uncertainty
    model's collinear cases (:func:`.uncertainty_scenarios.uncertainty_scenarios`)
    are built from rows.
    """
    arms = arm_effects()
    out = [
        Scenario(
            f"null_{model}",
            model,
            "unadjusted",
            _ids(model),
            _SIZE_FAILURE,
            _constant(Truth(arm=arms)),
            noise,
        )
        for model in rc.MODELS
    ]
    for name, model, truth, _ in _single_effects():
        truth = Truth(
            arm=arms, role=truth.role, slopes=truth.slopes, curves=truth.curves
        )
        out.append(
            Scenario(
                name,
                model,
                "unadjusted",
                _ids(model),
                _RECOVERY_FAILURE,
                _constant(truth),
                noise,
            )
        )
    out.append(
        Scenario(
            "length_current",
            "role_content",
            "length_adjusted",
            ("Q3a",),
            _RECOVERY_FAILURE + ", at L0 in the length-adjusted variant",
            _constant(Truth(arm=arms, role={"current": 0.5}, length={"current": 0.6})),
            noise,
        )
    )
    out.append(
        Scenario(
            "length_value",
            "role_content",
            "length_adjusted",
            ("Q4b",),
            _RECOVERY_FAILURE + ", at L0 in the length-adjusted variant",
            _constant(
                Truth(
                    arm=arms,
                    slopes={(rc.VALUE, rc.REFERENCE_ROLE): 0.2},
                    length_slopes={(rc.VALUE, rc.REFERENCE_ROLE): 0.3},
                )
            ),
            noise,
        )
    )
    large = SIZES["role"]["large"], SIZES["slope"]["large"]
    out.append(
        Scenario(
            "large_effects",
            "role_content_uncertainty",
            "unadjusted",
            ("Q3a", "Q3b", "Q4b", "Q5b-var", "Q5b-n"),
            "the mean estimate is shrunk toward 0 by more than 10 % of the planted "
            "value at the chosen prior SD (record the shrinkage at each candidate "
            f"SD, {', '.join(f'{sd:g}' for sd in rc.PRIOR_SD_CANDIDATES)}), or "
            f"{_INTERVAL} covers it in fewer than 90 % of data sets",
            _constant(
                Truth(
                    arm=arms,
                    role={"current": large[0], "previous": -large[0]},
                    slopes={
                        (rc.VALUE, rc.REFERENCE_ROLE): large[1],
                        (rc.BELIEF_VARIANCE, rc.REFERENCE_ROLE): large[1],
                        (rc.SAMPLE_COUNT, rc.REFERENCE_ROLE): -large[1],
                    },
                )
            ),
            noise,
        )
    )
    return out


# ---------------------------------------------------------------------------
# Truths built from rows
# ---------------------------------------------------------------------------

#: The bands of the truth from the data: the reward fraction on the leaf
#: roles, log trials on the other-leaf role, time until leaving on the
#: current role (each band's edges, on the model's scale).
DATA_BANDS: Mapping[tuple[str, str], tuple[float, ...]] = {
    (rc.REWARD_FRACTION, "current"): (0.0, 0.35, 0.65, 1.0),
    (rc.REWARD_FRACTION, rc.REFERENCE_ROLE): (0.0, 0.35, 0.65, 1.0),
    (rc.RECENCY, rc.REFERENCE_ROLE): (
        0.0,
        np.log(5.0),
        np.log(15.0),
        np.log(40.0),
        np.log(200.0),
    ),
    (rc.LEAVING, "current"): (0.0, 0.5, 1.5, 3.0),
}

#: The pseudo-count of the light smoothing: each band's mean share is shrunk
#: toward its role's mean share as if this many events of the role's mean were
#: added to it.
DATA_PSEUDO_EVENTS = 5.0


def _interp(x: np.ndarray, centers: np.ndarray, values: np.ndarray) -> np.ndarray:
    return np.interp(np.asarray(x, dtype=float), centers, values)


def _band_curve(centers: np.ndarray, values: np.ndarray) -> Curve:
    """The piecewise-linear curve through the band centers, flat beyond."""
    return partial(_interp, centers=centers, values=values)


def truth_from_data(
    pooled_rows: pd.DataFrame, response: str = rc.RESPONSE
) -> dict[str, Truth]:
    """One truth per animal, from the observed shares of one kind.

    For each animal: the role levels are the log of the mean share per arm of
    each role over the mean share per arm of the other-leaf role (over all its
    sessions' events); each band curve (:data:`DATA_BANDS`) is the log of the
    band's mean share per arm over the role's, shrunk by
    :data:`DATA_PSEUDO_EVENTS`, joined linearly between band centers and flat
    beyond. Arm effects are 0, and no slope is planted. The truth is outside
    the model (a piecewise-linear curve is not a P-spline, the bands are
    coarse); its planted values are defined from it as for any truth
    (:func:`planted_values`): Q3a and Q3b the role levels (against the
    other-leaf rows' pooled mean curve), Q4a, Q5a and Q7b the band curves at
    the estimands' points, the slopes 0.

    Parameters
    ----------
    pooled_rows : pd.DataFrame
        The cohort rows of one kind, every session.
    response : str
        The share column.

    Returns
    -------
    dict of str to Truth
        Animal -> its truth.
    """
    out = {}
    for animal, rows in pooled_rows.groupby("animal", sort=True):
        role_mean = rows.groupby(rc.ROLE)[response].mean()
        levels = {
            role: float(np.log(role_mean[role] / role_mean[rc.REFERENCE_ROLE]))
            for role in ("current", "previous", "trunk")
            if role in role_mean
        }
        curves = {}
        for (covariate, role), edges in DATA_BANDS.items():
            on = rows[rc.ROLE] == role
            x = _model_scale(rows.loc[on, covariate], covariate)
            share = rows.loc[on, response].to_numpy(dtype=float)
            band = np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)
            n = np.bincount(band, minlength=len(edges) - 1).astype(float)
            total = np.bincount(band, weights=share, minlength=len(edges) - 1)
            mean = share.mean()
            smoothed = (total + DATA_PSEUDO_EVENTS * mean) / (n + DATA_PSEUDO_EVENTS)
            centers = 0.5 * (np.asarray(edges[:-1]) + np.asarray(edges[1:]))
            curves[(covariate, role)] = _band_curve(centers, np.log(smoothed / mean))
        out[str(animal)] = Truth(role=levels, curves=curves)
    return out


def spread_truths(
    keys: Sequence[SessionKey],
    mean: Truth,
    *,
    animal_sd: float,
    session_sd: float,
    seed: int,
) -> dict[SessionKey, Truth]:
    """Truths whose role levels and slopes vary over animals and sessions.

    Each role level and slope of ``mean`` is, in a session, its mean plus an
    animal's Normal deviation (SD ``animal_sd``) plus the session's own
    (SD ``session_sd``), drawn once with ``seed`` in the sorted order of the
    animals and the sessions. The curves, arm and length terms stay as in
    ``mean``. The target of the reduction is :func:`planted_target` of the
    sessions' planted values, not the mean.

    Returns
    -------
    dict of SessionKey to Truth
    """
    rng = np.random.default_rng(seed)
    names = [("role", k) for k in mean.role] + [("slopes", k) for k in mean.slopes]
    animals = sorted({key[0] for key in keys})
    animal_dev = {a: rng.normal(0.0, animal_sd, len(names)) for a in animals}
    out = {}
    for key in sorted(keys):
        dev = animal_dev[key[0]] + rng.normal(0.0, session_sd, len(names))
        role = dict(mean.role)
        slopes = dict(mean.slopes)
        for (where, k), d in zip(names, dev, strict=True):
            target = role if where == "role" else slopes
            target[k] = target[k] + float(d)
        out[key] = Truth(
            arm=mean.arm,
            role=role,
            slopes=slopes,
            curves=mean.curves,
            length=mean.length,
            length_slopes=mean.length_slopes,
        )
    return out


def thin_events(rows: pd.DataFrame, n_events: int, seed: int) -> pd.DataFrame:
    """A session's rows with only ``n_events`` of its events, drawn without
    replacement (all of them when it has fewer): a setting that makes every
    role level and slope rest on few events, for the weak penalty's
    shrinkage. The events kept keep their nine rows."""
    events = rows[rc.UNIT].unique()
    if events.size <= n_events:
        return rows
    keep = np.random.default_rng(seed).choice(events, n_events, replace=False)
    return rows.loc[rows[rc.UNIT].isin(keep)].reset_index(drop=True)
