"""The visit rate model: event starts on the two visit clocks, models A and D.

One fit for each unit of the fit (``VisitRateSettings.fit_by``: a recording
day or an animal, ``fit_by.BEHAVIOR_FIT_BY``, the convention's twenty-fourth
round; or a session, the earlier design, for the validation and comparison)
and each kind (SWR, No SWR), by the shared fitter (``analysis.regression``):
plain penalized maximum likelihood, Poisson with the offset
``log(exposure)``, on the in-visit at-risk bins of the visit time-bin design
(``replay_models.visit_design.visit_timebins``). Inside a unit each session
keeps its own intercept (the baseline, ``declared_tests.BASELINES``) and every
other term (the clock curves, the reward terms, model D's lag curve) is
shared.

- Model A: ``f_s(time since the first poke-in) + f_u(time until the final
  poke-out) + beta * last_outcome_rewarded`` (in the declared by-outcome form,
  ``f_s`` for each last outcome). Two clocks (DECIDED, lead, 2026-11-26: "no
  just two"); the time-bin table's ``time_until_stretch_end_s`` is not read.
- Model D: the same smooths and, in place of the 0/1 term, a smooth curve over
  the last ten known outcomes on the log-lag scale (a lag functional), on the
  bins with ten known outcomes.

Each clock curve, and model D's lag curve, is a cubic P-spline at ONE fixed,
declared log penalty, the same in every unit; no penalty is chosen from
the data. The reward coefficient, the one simple tested coefficient, carries
the weak penalty (a Normal prior of mean 0, ``terms.log_penalty_from_prior_sd``);
the intercept is not penalized. Every model's design is checked by
``support.require_tested_penalized`` with the terms its estimands read.

The model's form has open questions. Each is a named setting below, in one
place, with the values allowed; :class:`VisitRateSettings` holds one choice of
each, and the model builds and fits for every allowed choice. The defaults
are the declared form (the by-outcome form of the reward effect, Q1's
declared reference window, all in-visit bins). The log penalties of the module (:data:`CLOCK_LOG_PENALTIES`) are
provisional: :func:`fit_visit_rate` refuses them unless the caller says the
fit is a test or a simulation; a real fit carries calibrated penalties
(:class:`CalibratedPenalty`).

The cohort step (:func:`cohort_rows`): ``visit_timebins`` of a session and
kind -> ``select_visit_context`` (the only filter of the bins) -> the bins
beyond the time range, if one is set, dropped and counted -> the rows with a
missing covariate dropped and counted -> model D also drops the bins without
ten known outcomes (:func:`model_d_rows`), counted. At the fit, model D drops
every bin of a unit whose outcomes cannot identify the lag curve, judged on
the unit's rows with each session's intercept (:func:`lag_curve_counts`),
counted. A unit's counts (its sessions' counts summed, and model D's of the
unit) are added to its record's ``cohort_counts``, in model A's and model D's
records alike. The two clocks that count down reference the future
(``covariate_timing.FUTURE_REFERENCING``): this model describes; it does not
predict.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal, NamedTuple

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.behavior.design_constants import (
    RATE_BIN_S,
    REWARD_HISTORY_N_LAGS,
)
from continuum_swr_replay.analysis.event_definition.cohort_selection import (
    count_visit_context,
    select_visit_context,
)
from continuum_swr_replay.analysis.regression import contrasts
from continuum_swr_replay.analysis.regression.calibration import (
    EdfCalibration,
    log_penalty_for_edf,
)
from continuum_swr_replay.analysis.regression.likelihoods import Poisson
from continuum_swr_replay.analysis.regression.pspline import PSplineBasis
from continuum_swr_replay.analysis.regression.record import FitRecord
from continuum_swr_replay.analysis.regression.sessions import SessionKey, fit_sessions
from continuum_swr_replay.analysis.regression.support import (
    MIN_UNITS,
    require_tested_penalized,
)
from continuum_swr_replay.analysis.regression.terms import (
    Category,
    Design,
    LinearFunctional,
    Offset,
    Slope,
    Smooth,
    Term,
    build_design,
    log_penalty_from_prior_sd,
)
from continuum_swr_replay.tables.session_keys import FitBy, fit_by_columns, unit_key

from ..declared_tests import (
    BASELINES,
    CoefficientContrast,
    CurveContrast,
    OutcomeWindowContrast,
    estimand,
)
from ..fit_by import BEHAVIOR_FIT_BY
from ..visit_design import EVENT_KINDS, REWARD_LAG_COLUMNS, visit_timebins

__all__ = [
    "ARRIVAL",
    "CLOCK_LOG_PENALTIES",
    "CLOCK_PENALTY_SETTINGS",
    "EXPOSURE",
    "LAGS",
    "LAG_TERM",
    "LEAVING",
    "MODELS",
    "MODEL_NAMES",
    "Q1_REFERENCE_WINDOWS",
    "RATE_ESTIMANDS",
    "SECONDARY_ESTIMAND",
    "OMITTED_ARRIVAL_TERM",
    "ARRIVAL_BY_OUTCOME",
    "RESPONSE",
    "REWARDED_ARRIVAL_TERM",
    "REWARD",
    "REWARD_FORMS",
    "TIME_RANGE_CAPS_S",
    "UNIT",
    "WEAK_PRIOR_SD",
    "WEAK_PRIOR_SD_SENSITIVITY",
    "CohortRows",
    "VisitRateFit",
    "VisitRateSettings",
    "CalibratedPenalty",
    "PROVISIONAL_USES",
    "build_visit_rate_design",
    "calibrate_log_penalty",
    "cohort_rows",
    "curve_rows",
    "declared_contrasts",
    "fit_visit_rate",
    "fit_visit_rate_session",
    "lag_curve_counts",
    "lag_curve_rows",
    "model_d_rows",
    "model_terms",
    "require_declared_penalized",
    "tested_terms",
    "window_contrast",
    "window_reads",
    "window_support",
    "WINDOW_SUPPORT",
    "DIFFERENCE_STEP_S",
    "difference_curve_rows",
    "difference_grid",
]

# --- The columns the model reads -------------------------------------------

#: Time since the first poke-in of the visit (s).
ARRIVAL = "time_since_visit_start_s"
#: Time until the final poke-out of the visit (s); counts down.
LEAVING = "time_until_visit_end_s"
#: The last known outcome, 1 rewarded and 0 omitted (lag 1 of the history).
REWARD = "last_outcome_rewarded"
#: The bin's at-risk seconds, the Poisson offset's column.
EXPOSURE = "exposure"
#: The bin's event starts of the kind, the response.
RESPONSE = "count"
#: The unit of the support rule: the well visit.
UNIT = "well_visit_id"

#: The two models: A (last outcome, 0/1) and D (the last ten outcomes).
MODELS: tuple[str, ...] = ("A", "D")

#: The model names in the records and in ``declared_tests``.
MODEL_NAMES: dict[str, str] = {"A": "visit_rate_A", "D": "visit_rate_D"}

#: The declared estimands this model answers (``declared_tests``), all read
#: from model A's by-outcome form.
RATE_ESTIMANDS: tuple[str, ...] = ("Q1", "Q2", "Q2-onset", "Q7a")

#: The secondary value, read from model A's coefficient form: the one reward
#: coefficient (the twenty-sixth round: kept for comparison with other
#: studies, described as mixing the timing with the rate; not a declared
#: test).
SECONDARY_ESTIMAND = "reward_coefficient"

# --- The bases -------------------------------------------------------------

#: Each clock's range and knot spacing (s). Beyond its range a clock is
#: clamped (the curve is flat there).
CLOCK_BASES: dict[str, tuple[float, float, float]] = {
    ARRIVAL: (0.0, 20.0, 0.5),
    LEAVING: (0.0, 3.0, 0.1),
}

#: Model D's lag curve: basis functions over ``[log 1, log 10]``.
LAG_N_BASIS = 8

#: The lags of model D (1 the most recent known outcome).
LAGS: tuple[int, ...] = tuple(range(1, REWARD_HISTORY_N_LAGS + 1))

#: Model D's lag-curve term, as ``terms`` names it.
LAG_TERM = f"lf({REWARD_LAG_COLUMNS[0]}..{REWARD_LAG_COLUMNS[-1]})"


def _smooth_name(column: str) -> str:
    return f"s({column})"


#: The arrival curve after an omission, of the by-outcome reward form: the
#: arrival smooth fitted on the rows whose last outcome was omitted.
OMITTED_ARRIVAL_TERM = f"{_smooth_name(ARRIVAL)}:{REWARD}=0"
#: The arrival curve after a reward, of the by-outcome reward form.
REWARDED_ARRIVAL_TERM = f"{_smooth_name(ARRIVAL)}:{REWARD}=1"
#: The two arrival curves of the by-outcome form, by last outcome.
ARRIVAL_BY_OUTCOME: dict[int, str] = {0: OMITTED_ARRIVAL_TERM, 1: REWARDED_ARRIVAL_TERM}

# --- OPEN (lead): the settings of the model's form --------------------------
#
# Each setting below is an open question of the project lead, except where it
# is marked DECIDED; the model builds and fits for each allowed value. The defaults are the
# declared form and decide nothing: no real fit runs on the provisional
# penalties.

#: DECIDED (lead, 2026-11-23, twenty-fifth round): the weak penalty stays on
#: the reward coefficient and on every simple tested coefficient of the rate
#: model, at whatever unit the model is fitted (the unit of the fit is the
#: twenty-fourth round's: a recording day and an animal). The penalty is a
#: Normal prior of mean 0 on the log rate ratio after a reward; its SD is this value, the author's call
#: under the lead's "pick a penalty that's reasonable", not a number the lead
#: chose. An SD of 2 says that a rate ratio within a factor of about 50 either
#: way is plausible (``exp(1.96 * 2)``). The log penalty is ``log(1 / SD^2)``
#: (``terms.log_penalty_from_prior_sd``).
WEAK_PRIOR_SD = 2.0

#: The stiffer and looser prior SDs reported beside :data:`WEAK_PRIOR_SD`
#: (the convention's sensitivity pair).
WEAK_PRIOR_SD_SENSITIVITY: tuple[float, float] = (1.0, 4.0)

#: OPEN (lead): the clock penalty setting. Each clock's declared log penalty
#: is to be set by ``calibration.log_penalty_for_edf`` on this model's own real
#: rows, per kind, at a target median edf (:func:`calibrate_log_penalty`); the
#: fit is also reported at a stiffer and a looser value (one step of the
#: target each way). The default is ``"declared"``.
CLOCK_PENALTY_SETTINGS: tuple[str, ...] = ("declared", "stiffer", "looser")

#: OPEN (lead): the log penalty of each curve, per kind and setting.
#: PROVISIONAL (:data:`CLOCK_LOG_PENALTIES_ARE_PROVISIONAL`): copied from a
#: prototype on another design, not calibrated on this model's rows. A real
#: fit takes each curve's penalty from :func:`calibrate_log_penalty`, carried
#: by ``VisitRateSettings.calibrated_penalties`` with its target and rows;
#: :func:`fit_visit_rate` refuses these values unless the fit is a test or a
#: simulation. Model D's lag curve is 2, with 4 and 0 either side; each
#: arrival curve of the by-outcome reward form takes the arrival clock's
#: value (:meth:`VisitRateSettings.log_penalty`).
CLOCK_LOG_PENALTIES: dict[str, dict[str, dict[str, float]]] = {
    "SWR": {
        "declared": {ARRIVAL: 1.0522, LEAVING: 3.0718, LAG_TERM: 2.0},
        "stiffer": {ARRIVAL: 3.8344, LEAVING: 5.3921, LAG_TERM: 4.0},
        "looser": {ARRIVAL: -1.7300, LEAVING: 0.7515, LAG_TERM: 0.0},
    },
    "No SWR": {
        "declared": {ARRIVAL: 1.8472, LEAVING: 3.2544, LAG_TERM: 2.0},
        "stiffer": {ARRIVAL: 4.3179, LEAVING: 5.5317, LAG_TERM: 4.0},
        "looser": {ARRIVAL: -0.6235, LEAVING: 0.9771, LAG_TERM: 0.0},
    },
}

#: Every value of :data:`CLOCK_LOG_PENALTIES` is provisional.
CLOCK_LOG_PENALTIES_ARE_PROVISIONAL = True

#: The uses that may fit on provisional penalties (:func:`fit_visit_rate`'s
#: ``allow_provisional``).
PROVISIONAL_USES: tuple[str, ...] = ("test", "simulation")

#: OPEN (lead): the time range of the model. None models every in-visit bin
#: (the default); a cap keeps only the bins whose time since arrival is below
#: it (candidate 10 s), and the arrival clock's range ends at the cap. The
#: bins beyond the cap leave the model on both clocks, so Q7a is then read
#: only from the visits, or the parts of visits, that end before the cap: a
#: visit longer than the cap plus 2 s gives no bin to Q7a's windows. The
#: cohort counts say how many bins, events and visits of the last 2 s before
#: leaving the cap removes (:func:`cohort_rows`).
TIME_RANGE_CAPS_S: tuple[float | None, ...] = (None, 10.0)

#: DECIDED (lead, 2026-11-25, the convention's twenty-sixth round): the form of
#: the reward effect. ``"by_outcome"`` (the default, the declared form): an
#: arrival curve for each last outcome, each a smooth ``by`` the outcome with
#: its own sum-to-zero constraint (on the pooled rows of its outcome) and its
#: own declared penalty, the outcome coefficient (the weak penalty) and the
#: leaving curve shared by both outcomes: ``outcome +
#: session + s(arrival, by = outcome) + s(leaving) + offset(log exposure)``.
#: The outcome coefficient and the two centered curves identify the
#: difference curve ``beta + f_1(s) - f_0(s)`` (the log rate ratio after a
#: reward at time ``s`` since arrival, at the same time before leaving), so
#: the reward effect may change with time since arrival, and only along it.
#: Q2 and the onset test are its mean over declared windows, Q1 is read from
#: the arrival curve after an omission (``declared_tests``).
#: ``"coefficient"``: one arrival curve for both outcomes and one reward
#: coefficient. Its coefficient is the SECONDARY value
#: (:data:`SECONDARY_ESTIMAND`), for comparison with other studies: where
#: the effect changes with time since arrival it mixes the timing with the
#: rate. The form of a shared curve plus a difference curve
#: (``"time_varying"``) was decided against (the same round) and removed.
REWARD_FORMS: tuple[str, ...] = ("by_outcome", "coefficient")

#: OPEN (lead): the reference window of Q1 (s since arrival). ``(0, 1)`` is
#: the declared one (``declared_tests``, the default); ``(1, 2)`` the
#: alternative, for a kind whose events rarely start in the first second.
Q1_REFERENCE_WINDOWS: tuple[tuple[float, float], ...] = ((0.0, 1.0), (1.0, 2.0))


@dataclass(frozen=True)
class CalibratedPenalty:
    """A curve's log penalty found by calibration, with how it was found.

    Attributes
    ----------
    log_penalty : float
        The value (:func:`calibrate_log_penalty`'s ``log_penalty``).
    target_edf : float
        The median edf over sessions it was calibrated to.
    rows : mapping
        The rows it was found on, JSON-ready: at least the tables read and
        their hashes, the commit and the numbers of sessions and rows.
    """

    log_penalty: float
    target_edf: float
    rows: Mapping[str, object]

    def __post_init__(self) -> None:
        if not (np.isfinite(self.log_penalty) and np.isfinite(self.target_edf)):
            raise ValueError("a calibrated penalty needs a finite value and target")
        if not self.rows:
            raise ValueError("a calibrated penalty needs the record of its rows")


@dataclass(frozen=True)
class VisitRateSettings:
    """One choice of each open setting of the model's form.

    Attributes
    ----------
    weak_prior_sd : float
        The weak penalty's prior SD: the decided :data:`WEAK_PRIOR_SD`, or
        one of :data:`WEAK_PRIOR_SD_SENSITIVITY` (any positive
        value for a sensitivity analysis).
    clock_penalty : {"declared", "stiffer", "looser"}
        Which of :data:`CLOCK_LOG_PENALTIES` the curves take.
    max_time_since_arrival_s : float or None
        :data:`TIME_RANGE_CAPS_S`. With a cap, Q7a is read only from the
        visits that end before it (see there).
    reward_form : {"by_outcome", "coefficient"}
        :data:`REWARD_FORMS`: the declared form (an arrival curve for each
        outcome) or the secondary one (one coefficient).
    q1_reference_s : tuple of float
        :data:`Q1_REFERENCE_WINDOWS`.
    fit_by : {"day", "animal", "session"}
        The unit of the fit, one for each run (DECIDED, the convention's
        twenty-fourth round): one of ``fit_by.BEHAVIOR_FIT_BY``, the units every
        behavior model is fitted at (the first is the default); ``"session"``
        is the earlier design, kept for the validation and for comparison.
        Each session keeps its own intercept (the baseline,
        ``declared_tests.BASELINES``); every other term is shared inside the
        unit.
    log_penalties : mapping of str to float or None
        Curve column (or :data:`LAG_TERM`) -> a log penalty in place of the
        table's, for the calibration and the tests; None uses the table. A
        value given here has no record and is provisional.
    calibrated_penalties : mapping of (kind, curve) to CalibratedPenalty or None
        Calibrated log penalties with their record, which a real fit needs
        for every curve of the model; ``log_penalties`` takes precedence. A
        calibrated value is used at every ``clock_penalty`` setting (which
        chooses only among the provisional values): a stiffer or looser fit
        passes its own calibrated values.
    """

    weak_prior_sd: float = WEAK_PRIOR_SD
    clock_penalty: Literal["declared", "stiffer", "looser"] = "declared"
    max_time_since_arrival_s: float | None = None
    reward_form: Literal["by_outcome", "coefficient"] = "by_outcome"
    q1_reference_s: tuple[float, float] = (0.0, 1.0)
    fit_by: FitBy = BEHAVIOR_FIT_BY[0]
    log_penalties: Mapping[str, float] | None = None
    calibrated_penalties: Mapping[tuple[str, str], CalibratedPenalty] | None = None

    def __post_init__(self) -> None:
        if not (np.isfinite(self.weak_prior_sd) and self.weak_prior_sd > 0):
            raise ValueError(f"weak_prior_sd must be positive: {self.weak_prior_sd}")
        if self.clock_penalty not in CLOCK_PENALTY_SETTINGS:
            raise ValueError(
                f"clock_penalty {self.clock_penalty!r} not in {CLOCK_PENALTY_SETTINGS}"
            )
        if self.max_time_since_arrival_s not in TIME_RANGE_CAPS_S:
            raise ValueError(
                f"max_time_since_arrival_s {self.max_time_since_arrival_s} not in "
                f"{TIME_RANGE_CAPS_S}"
            )
        if self.reward_form not in REWARD_FORMS:
            raise ValueError(f"reward_form {self.reward_form!r} not in {REWARD_FORMS}")
        reference = tuple(float(x) for x in self.q1_reference_s)
        if reference not in Q1_REFERENCE_WINDOWS:
            raise ValueError(
                f"q1_reference_s {self.q1_reference_s} not in {Q1_REFERENCE_WINDOWS}"
            )
        object.__setattr__(self, "q1_reference_s", reference)
        fit_by_columns(self.fit_by)

    @property
    def weak_log_penalty(self) -> float:
        """The weak penalty's log penalty, ``log(1 / SD^2)``."""
        return log_penalty_from_prior_sd(self.weak_prior_sd)

    def log_penalty(self, kind: str, curve: str) -> float:
        """The log penalty of one curve (a clock column, an arrival curve of
        :data:`ARRIVAL_BY_OUTCOME` or :data:`LAG_TERM`). Each curve has its
        own; in :data:`CLOCK_LOG_PENALTIES` an arrival curve by outcome takes
        the arrival clock's value."""
        if self.log_penalties is not None and curve in self.log_penalties:
            return float(self.log_penalties[curve])
        calibrated = self.calibrated_penalties or {}
        if (_kind(kind), curve) in calibrated:
            return float(calibrated[(kind, curve)].log_penalty)
        table = ARRIVAL if curve in ARRIVAL_BY_OUTCOME.values() else curve
        return CLOCK_LOG_PENALTIES[kind][self.clock_penalty][table]

    def curves(self, model: str) -> tuple[str, ...]:
        """The curves of a model whose log penalty this setting gives: the
        clocks (model A's by-outcome form: the two arrival curves,
        :data:`ARRIVAL_BY_OUTCOME`, in place of the arrival clock) and, for
        model D, :data:`LAG_TERM`."""
        if _model(model) == "D":
            return (*self.clocks(), LAG_TERM)
        clocks = self.clocks()
        if self.reward_form == "by_outcome":
            return (*ARRIVAL_BY_OUTCOME.values(), *clocks[1:])
        return clocks

    def provisional_curves(self, model: str, kind: str) -> tuple[str, ...]:
        """The model's curves whose log penalty has no calibration record:
        from :data:`CLOCK_LOG_PENALTIES` or from ``log_penalties``."""
        calibrated = self.calibrated_penalties or {}
        overridden = self.log_penalties or {}
        return tuple(
            curve
            for curve in self.curves(model)
            if curve in overridden or (_kind(kind), curve) not in calibrated
        )

    def clocks(self) -> tuple[str, ...]:
        """The clock columns the model holds, in order."""
        return (ARRIVAL, LEAVING)

    def clock_range(self, clock: str) -> tuple[float, float]:
        """A clock's basis range: the arrival clock's ends at the cap."""
        lo, hi, _ = CLOCK_BASES[clock]
        if clock == ARRIVAL and self.max_time_since_arrival_s is not None:
            hi = min(hi, self.max_time_since_arrival_s)
        return lo, hi


def _kind(kind: str) -> str:
    if kind not in EVENT_KINDS:
        raise ValueError(f"kind must be one of {sorted(EVENT_KINDS)}: {kind!r}")
    return kind


def _model(model: str) -> str:
    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}: {model!r}")
    return model


# --- Model specs -------------------------------------------------------------


def _clock_smooth(
    clock: str, kind: str, settings: VisitRateSettings, curve: str | None = None, **by
) -> Smooth:
    """A clock's smooth at the log penalty of ``curve`` (the clock's own by
    default)."""
    lo, hi = settings.clock_range(clock)
    basis = PSplineBasis.from_spacing(lo, hi, CLOCK_BASES[clock][2])
    penalty = settings.log_penalty(kind, curve or clock)
    return Smooth(clock, lo, hi, basis.n_basis, penalty, **by)


#: The last outcome as the ``by`` of the by-outcome form's arrival curves (a
#: category used only as a ``by``, so not a term: its penalty is not read).
_OUTCOME = Category(REWARD, (0, 1), 0, "none")


def model_terms(model: str, kind: str, settings: VisitRateSettings) -> list[Term]:
    """The terms of model A or D for one kind (module docstring).

    Parameters
    ----------
    model : {"A", "D"}
    kind : {"SWR", "No SWR"}
    settings : VisitRateSettings

    Returns
    -------
    list of terms
        The offset, the arrival curves (model A's by-outcome form: one for
        each last outcome, :data:`ARRIVAL_BY_OUTCOME`, each at its own
        penalty; else one), the other clock smooths, then model A's reward
        coefficient or model D's lag curve. The intercept is the design's,
        unpenalized: the baseline.
    """
    _model(model)
    _kind(kind)
    terms: list[Term] = [Offset(EXPOSURE)]
    if model == "A" and settings.reward_form == "by_outcome":
        terms += [
            _clock_smooth(ARRIVAL, kind, settings, term, by=_OUTCOME, levels=(level,))
            for level, term in ARRIVAL_BY_OUTCOME.items()
        ]
    else:
        terms.append(_clock_smooth(ARRIVAL, kind, settings))
    terms += [_clock_smooth(clock, kind, settings) for clock in settings.clocks()[1:]]
    if model == "A":
        terms.append(Slope(REWARD, settings.weak_log_penalty))
    else:
        terms.append(
            LinearFunctional(
                REWARD_LAG_COLUMNS,
                LAGS,
                0.0,
                float(np.log(LAGS[-1])),
                LAG_N_BASIS,
                settings.log_penalty(kind, LAG_TERM),
            )
        )
    return terms


def tested_terms(model: str, settings: VisitRateSettings) -> tuple[str, ...]:
    """The terms and coefficients the model's estimands read: model A's
    by-outcome form, its declared tests (Q1: the arrival curve after an
    omission; Q2 and the onset test: the reward coefficient and both arrival
    curves; Q7a: the leaving curve); model A's coefficient form, its
    secondary value (the reward coefficient); model D's lag curve
    (described, not tested)."""
    if _model(model) == "D":
        return (LAG_TERM,)
    if settings.reward_form == "by_outcome":
        return (*ARRIVAL_BY_OUTCOME.values(), _smooth_name(LEAVING), REWARD)
    return (REWARD,)


def require_declared_penalized(
    design: Design, model: str, settings: VisitRateSettings
) -> None:
    """Refuse a design whose tested terms lack a fixed penalty
    (``support.require_tested_penalized`` with :func:`tested_terms`)."""
    require_tested_penalized(design, tested_terms(model, settings))


def build_visit_rate_design(
    model: str,
    kind: str,
    pooled_rows: pd.DataFrame,
    settings: VisitRateSettings,
) -> Design:
    """Model A's or D's design on the pooled rows of its sessions and kind.

    The sum-to-zero constraints are computed on ``pooled_rows``, so every
    unit fitted with this design shares them. The intercept is the model's
    baseline (``declared_tests.BASELINES``): in a fit of several sessions
    each session has its own. The guard (:func:`require_declared_penalized`)
    is called on the built design.

    Parameters
    ----------
    model : {"A", "D"}
    kind : {"SWR", "No SWR"}
    pooled_rows : pd.DataFrame
        Every session's rows of the model (``CohortRows.rows[model]``).
    settings : VisitRateSettings

    Returns
    -------
    Design
    """
    design = build_design(
        model_terms(model, kind, settings),
        pooled_rows,
        intercept=True,
        baseline=BASELINES[MODEL_NAMES[model]],
    )
    require_declared_penalized(design, model, settings)
    return design


# --- The declared tests on the model's own terms -----------------------------


def _probe(design: Design, n: int) -> pd.DataFrame:
    """Rows at which every column the model reads is 0 (exposure 1)."""
    columns = [ARRIVAL, LEAVING, REWARD, *REWARD_LAG_COLUMNS]
    frame = pd.DataFrame(0.0, index=np.arange(n), columns=columns)
    frame[EXPOSURE] = 1.0
    return frame


def curve_rows(design: Design, term: str) -> Callable[[np.ndarray], np.ndarray]:
    """The map from a smooth's variable to the rows of the full coefficient
    vector that give the curve there (``contrasts.range_mean_contrast``'s
    ``curve_rows``).

    Parameters
    ----------
    design : Design
    term : str
        A smooth term of the design (``s(<clock>)``, or a smooth fitted for
        one level of a ``by``).

    Returns
    -------
    callable
        ``x`` of shape ``(n_points,)`` -> shape ``(n_points, n_coef)``, zero
        outside the term's columns.
    """
    designed = design.by_name[term]
    if designed.kind != "smooth":
        raise ValueError(f"{term!r} is not a smooth term of the design")
    columns = designed.columns

    def rows(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        frame = _probe(design, x.size)
        frame[designed.covariate] = x
        if designed.level is not None:
            frame[designed.level[0]] = designed.level[1]
        out = np.zeros((x.size, len(design.coef_names)))
        out[:, columns] = design.build(frame).X[:, columns]
        return out

    return rows


def lag_curve_rows(design: Design) -> np.ndarray:
    """The rows of the full coefficient vector that give model D's lag curve
    ``f(log l)`` at each lag ``l`` of :data:`LAGS`, shape ``(n_lags, n_coef)``."""
    columns = design.by_name[LAG_TERM].columns
    frame = _probe(design, len(LAGS))
    frame[list(REWARD_LAG_COLUMNS)] = np.eye(len(LAGS))
    out = np.zeros((len(LAGS), len(design.coef_names)))
    out[:, columns] = design.build(frame).X[:, columns]
    return out


def window_contrast(design: Design, contrast: OutcomeWindowContrast) -> np.ndarray:
    """The ``c`` of an outcome window contrast (``declared_tests``): the
    outcome coefficient plus the mean over the window, on
    ``contrasts.range_grid``, of the rewarded arrival curve less the omitted
    one.

    Raises
    ------
    ValueError
        If the window lies outside a curve's basis range (beyond it a clamped
        smooth is flat), or the design lacks a term or the coefficient.
    """
    basis = design.basis_record()
    grid = contrasts.range_grid(*contrast.window)
    c = contrasts.coefficient_contrast(design.coef_names, (contrast.coefficient,))
    for term, sign in ((contrast.plus_term, 1.0), (contrast.minus_term, -1.0)):
        if term not in design.by_name:
            raise ValueError(f"{term!r} is not in the design")
        lo, hi = basis[term]["range"]
        a, b = contrast.window
        if a < lo - 1e-9 * (hi - lo) or b > hi + 1e-9 * (hi - lo):
            raise ValueError(
                f"the window [{a}, {b}] lies outside the term's range [{lo}, {hi}]"
            )
        c = c + sign * curve_rows(design, term)(grid).mean(axis=0)
    return c


def declared_contrasts(
    design: Design, settings: VisitRateSettings
) -> dict[str, np.ndarray]:
    """The ``c`` of each estimand of model A's form.

    The by-outcome form gives the declared estimands (:data:`RATE_ESTIMANDS`:
    Q1, Q2, Q2-onset, Q7a), each read from ``declared_tests``; Q1's
    reference window is the setting's (the declared ``(0, 1)`` by default).
    A curve contrast is the range mean of its smooth over the contrast's
    ranges (``contrasts.range_mean_contrast``), which must lie inside the
    term's basis range; an outcome window contrast is
    :func:`window_contrast`. The coefficient form gives the secondary value
    alone (:data:`SECONDARY_ESTIMAND`, the reward coefficient).

    Returns
    -------
    dict of str to np.ndarray
        Estimand id -> ``c``, shape ``(n_coef,)``.

    Raises
    ------
    ValueError
        If a declared estimand is not of model A, or names a term or a
        coefficient the design lacks.
    """
    if settings.reward_form == "coefficient":
        return {
            SECONDARY_ESTIMAND: contrasts.coefficient_contrast(
                design.coef_names, (REWARD,)
            )
        }
    out = {}
    basis = design.basis_record()
    for estimand_id in RATE_ESTIMANDS:
        declared = estimand(estimand_id)
        if declared.model != MODEL_NAMES["A"]:
            raise ValueError(f"{estimand_id} is an estimand of {declared.model}")
        contrast = declared.contrast
        if estimand_id == "Q1":
            contrast = replace(contrast, second=settings.q1_reference_s)
        if isinstance(contrast, CurveContrast):
            if contrast.term not in design.by_name:
                raise ValueError(
                    f"{estimand_id}: {contrast.term!r} is not in the design"
                )
            out[estimand_id] = contrasts.range_mean_contrast(
                curve_rows(design, contrast.term),
                contrast.first,
                contrast.second,
                term_range=tuple(basis[contrast.term]["range"]),
            )
        elif isinstance(contrast, OutcomeWindowContrast):
            out[estimand_id] = window_contrast(design, contrast)
        elif isinstance(contrast, CoefficientContrast):
            out[estimand_id] = contrasts.coefficient_contrast(
                design.coef_names, contrast.plus, contrast.minus
            )
        else:
            raise ValueError(f"{estimand_id}: a {type(contrast).__name__} contrast")
    return out


# --- The support of a window contrast -------------------------------------------

#: The key of a record's ``cohort_counts`` that holds :func:`window_support`.
WINDOW_SUPPORT = "window_support"


def window_reads(
    settings: VisitRateSettings,
) -> dict[str, tuple[tuple[int, tuple[float, float]], ...]]:
    """What each declared estimand of the by-outcome form reads along the
    arrival clock: (last outcome, range of time since arrival) pairs.

    Q1 reads the curve after an omission over its two ranges (the second the
    setting's reference window); Q2 and Q2-onset read both outcomes' curves
    over their window. Q7a reads the leaving curve, shared by both outcomes,
    and has no window read.
    """
    out = {}
    for estimand_id in RATE_ESTIMANDS:
        contrast = estimand(estimand_id).contrast
        if isinstance(contrast, OutcomeWindowContrast):
            out[estimand_id] = tuple((level, contrast.window) for level in (1, 0))
        elif contrast.term in ARRIVAL_BY_OUTCOME.values():
            level = {t: k for k, t in ARRIVAL_BY_OUTCOME.items()}[contrast.term]
            second = settings.q1_reference_s if estimand_id == "Q1" else contrast.second
            out[estimand_id] = ((level, contrast.first), (level, second))
    return out


def window_support(
    sessions: Sequence[pd.DataFrame], settings: VisitRateSettings
) -> dict[str, dict]:
    """The support of each window contrast in one unit of the fit (the
    author's rule, written for the twenty-sixth round's "under the support
    rule").

    For each (outcome, range) an estimand reads (:func:`window_reads`), the
    unit's rows of that last outcome whose time since arrival lies in the
    range (``lo <= s < hi``) are counted: the visits with such a row (the
    support rule's unit, a visit; each session's visits apart) and their
    at-risk seconds. An estimand is supported when every read has at least
    ``support.MIN_UNITS`` visits: the rule of a 0/1 slope
    (``support.binary_support``) counted on the rows inside each window, the
    smallest rule the support machinery states. No response is read. The
    rule asks for visits with a row anywhere in the window, not for rows over
    its whole range: where no visit after an omission lasts to the window's
    end, the curve there is the penalty's extrapolation and the rule does not
    see it (a synthetic cohort whose visits after an omission all ended by 8
    s gave Q2 biased by +0.14 to +0.47 with nothing planted, 2026-11-26).

    Parameters
    ----------
    sessions : sequence of pd.DataFrame
        The rows of each session in the unit's fit (a session dropped from
        the fit is not given).
    settings : VisitRateSettings

    Returns
    -------
    dict
        Estimand id -> ``{"supported": bool, "reads": [{"outcome": 0 or 1,
        "range_s": [lo, hi], "visits": int, "at_risk_s": float}, ...]}``,
        JSON-ready.
    """
    out = {}
    for estimand_id, reads in window_reads(settings).items():
        entries = []
        for level, (lo, hi) in reads:
            visits, at_risk = 0, 0.0
            for rows in sessions:
                s = rows[ARRIVAL].to_numpy(dtype=float)
                on = (
                    (rows[REWARD].to_numpy(dtype=float) == level) & (s >= lo) & (s < hi)
                )
                visits += int(rows.loc[on, UNIT].nunique())
                at_risk += float(rows.loc[on, EXPOSURE].sum())
            entries.append(
                {
                    "outcome": level,
                    "range_s": [float(lo), float(hi)],
                    "visits": visits,
                    "at_risk_s": at_risk,
                }
            )
        out[estimand_id] = {
            "supported": all(e["visits"] >= MIN_UNITS for e in entries),
            "reads": entries,
        }
    return out


# --- The difference curve ---------------------------------------------------------

#: The grid of the difference curve (s since arrival): 0.1 s steps over the
#: arrival clock's basis range. It is shown, not tested.
DIFFERENCE_STEP_S = 0.1


def difference_grid(settings: VisitRateSettings) -> np.ndarray:
    """The difference curve's grid over the arrival clock's range, shape
    ``(n_points,)``."""
    lo, hi = settings.clock_range(ARRIVAL)
    return np.round(np.arange(lo, hi + 1e-9, DIFFERENCE_STEP_S), 10)


def difference_curve_rows(design: Design, grid: np.ndarray) -> np.ndarray:
    """The rows of the coefficient vector that give the difference curve
    ``beta + f_1(s) - f_0(s)`` (the log rate ratio after a reward against
    after an omission at time ``s`` since arrival) at each grid point, shape
    ``(n_points, n_coef)``."""
    reward = contrasts.coefficient_contrast(design.coef_names, (REWARD,))
    return (
        reward[None, :]
        + curve_rows(design, REWARDED_ARRIVAL_TERM)(grid)
        - curve_rows(design, OMITTED_ARRIVAL_TERM)(grid)
    )


# --- The cohort step ----------------------------------------------------------


@dataclass(frozen=True)
class CohortRows:
    """One session's rows of one kind for models A and D, and the counts of
    what the cohort step kept and dropped.

    Attributes
    ----------
    kind : str
    rows : dict of str to pd.DataFrame
        Model (``"A"``, ``"D"``) -> its rows (the selected bins with the
        :data:`REWARD` column).
    counts : dict
        JSON-ready (:func:`cohort_rows`).
    """

    kind: str
    rows: dict[str, pd.DataFrame]
    counts: dict


def _label_counts(table: pd.DataFrame) -> dict[str, dict]:
    counted = count_visit_context(table)
    return {
        str(label): {"n": int(row["n"]), "kept": bool(row["kept"])}
        for label, row in counted.iterrows()
    }


def _counted(rows: pd.DataFrame, name: str) -> dict[str, int]:
    """``bins_<name>`` and ``events_<name>``: the bins of ``rows`` and their
    event starts."""
    return {f"bins_{name}": len(rows), f"events_{name}": int(rows[RESPONSE].sum())}


#: The end of Q7a's windows: the time before leaving the estimand reads (s).
_Q7A_SPAN_S = max(
    hi for _, hi in (estimand("Q7a").contrast.first, estimand("Q7a").contrast.second)
)


def _lag_curve_identified(sessions: Sequence[pd.DataFrame]) -> bool:
    """Whether the unit's rows identify the lag curve's unpenalized part.

    The lag curve's penalty leaves a constant and a line in log lag
    unpenalized; on a row they give ``sum_l r_l`` and ``sum_l r_l log l``.
    With each session's intercept (the baseline; one intercept for a unit of
    one session), these columns must have full rank on the unit's rows. They
    do not when, in every session of the unit, every outcome is the same
    (each lag column constant in each session), and then the fitter refuses
    the unit's design. A session of constant outcomes inside a unit whose
    other sessions vary does not stop it.
    """
    parts = [rows for rows in sessions if len(rows)]
    if not parts:
        return False
    lags = np.vstack([p[list(REWARD_LAG_COLUMNS)].to_numpy(dtype=float) for p in parts])
    sizes = [len(p) for p in parts]
    indicators = np.repeat(np.eye(len(parts)), sizes, axis=0)
    columns = np.column_stack(
        [indicators, lags.sum(axis=1), lags @ np.log(np.asarray(LAGS))]
    )
    return np.linalg.matrix_rank(columns) == columns.shape[1]


def model_d_rows(rows_a: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Model D's rows of one session, from model A's: the bins with ten known
    outcomes.

    Whether the rows identify the lag curve is judged on the rows of the unit
    of the fit, not here (:func:`lag_curve_counts`).

    Parameters
    ----------
    rows_a : pd.DataFrame
        One session's rows of model A.

    Returns
    -------
    rows : pd.DataFrame
        The bins with ten known outcomes.
    counts : dict of str to int
        ``bins_``/``events_`` ``without_ten_lags`` (of model A's rows).
    """
    without = ~rows_a[list(REWARD_LAG_COLUMNS)].notna().all(axis=1).to_numpy()
    counts = _counted(rows_a.loc[without], "without_ten_lags")
    return rows_a.loc[~without].reset_index(drop=True), counts


def lag_curve_counts(sessions: Sequence[pd.DataFrame]) -> dict[str, int]:
    """Model D's counts of one unit of the fit, from its sessions' model D
    rows (:func:`model_d_rows`).

    When the unit's rows cannot identify the lag curve
    (:func:`_lag_curve_identified`), the unit has no model D rows: its fit is
    recorded as not fitted, and the other units are fitted.

    Parameters
    ----------
    sessions : sequence of pd.DataFrame
        The model D rows of each session of the unit.

    Returns
    -------
    dict of str to int
        ``bins_``/``events_`` ``lag_curve_not_identified`` (the unit's rows
        left out for that reason; 0 or all of them) and ``D`` (the rows
        model D fits).
    """
    rows = pd.concat(list(sessions), ignore_index=True)
    not_identified = not rows.empty and not _lag_curve_identified(sessions)
    left = rows if not_identified else rows.iloc[:0]
    kept = rows.iloc[:0] if not_identified else rows
    return _counted(left, "lag_curve_not_identified") | _counted(kept, "D")


def cohort_rows(
    stop_table: pd.DataFrame,
    events: pd.DataFrame,
    port_experiences: pd.DataFrame,
    *,
    kind: str,
    settings: VisitRateSettings,
    excluded_intervals: pd.DataFrame | None = None,
    bin_s: float = RATE_BIN_S,
) -> CohortRows:
    """One session's rows of one kind, by the cohort step (module docstring).

    Parameters
    ----------
    stop_table, events, port_experiences : pd.DataFrame
        One session's stops, its events and its port experiences, as
        ``visit_timebins`` takes them.
    kind : {"SWR", "No SWR"}
    settings : VisitRateSettings
        Its time range and clocks decide the rows kept.
    excluded_intervals : pd.DataFrame, optional
        As ``visit_timebins`` takes it (None reads it from the SWR source).
    bin_s : float
        Bin width (s); default ``RATE_BIN_S``.

    Returns
    -------
    CohortRows
        ``counts`` holds, for the session and kind: ``bins_visit_context``
        and ``events_visit_context`` (``count_visit_context`` of the bins and
        of the kind's events: per label, ``n`` and ``kept``);
        ``starts_not_at_risk`` and ``starts_not_at_risk_by_reason`` (the event
        starts the design does not count, ``visit_timebins``' ``attrs``);
        ``events_on_stop_first_sample`` (those of them on a stop's first
        sample, an open question of the lead; their handling is unchanged;
        these three are over all the session's bins), and
        ``events_not_at_risk_in_default_set`` (the kind's events of the
        default set among those starts, so that the default set's events are
        ``events_selected`` plus these);
        the bins and their event starts at each step, ``bins_<step>`` and
        ``events_<step>``: ``selected`` (by ``select_visit_context``), then
        dropped ``beyond_time_range`` (time since arrival at or beyond the
        cap; a missing time since arrival is a missing covariate, not beyond
        the cap), dropped ``missing_any`` (a missing covariate;
        ``bins_missing`` counts each column's), model ``A``'s, dropped
        ``without_ten_lags`` (of model A's), dropped
        ``lag_curve_not_identified`` (:func:`model_d_rows`), and model
        ``D``'s; and, of the bins beyond the cap, those in the last 2 s before
        leaving (Q7a's windows), ``bins_``, ``events_`` and
        ``visits_beyond_time_range_before_leaving``.
    """
    _kind(kind)
    bins = visit_timebins(
        stop_table,
        events,
        port_experiences,
        kind=kind,
        bin_s=bin_s,
        excluded_intervals=excluded_intervals,
    )
    kind_events = events[events["is_swr"] == EVENT_KINDS[kind]]
    selected = select_visit_context(bins)
    selected[REWARD] = selected[REWARD_LAG_COLUMNS[0]]
    counts: dict = {
        "bins_visit_context": _label_counts(bins),
        "events_visit_context": _label_counts(kind_events),
        "starts_not_at_risk": int(bins.attrs["n_starts_not_at_risk"]),
        "starts_not_at_risk_by_reason": dict(
            bins.attrs["n_starts_not_at_risk_by_reason"]
        ),
        "events_on_stop_first_sample": int(
            bins.attrs["n_starts_not_at_risk_by_reason"]["stop_start"]
        ),
        "events_not_at_risk_in_default_set": int(
            select_visit_context(kind_events)["event_start_s"]
            .isin(bins.attrs["starts_not_at_risk_s"])
            .sum()
        ),
    }
    counts |= _counted(selected, "selected")
    cap = settings.max_time_since_arrival_s
    beyond = (
        np.zeros(len(selected), dtype=bool)
        if cap is None
        else selected[ARRIVAL].to_numpy(dtype=float) >= cap
    )
    counts |= _counted(selected.loc[beyond], "beyond_time_range")
    before_leaving = selected.loc[
        beyond & (selected[LEAVING].to_numpy(dtype=float) <= _Q7A_SPAN_S)
    ]
    counts |= _counted(before_leaving, "beyond_time_range_before_leaving")
    counts["visits_beyond_time_range_before_leaving"] = int(
        before_leaving[UNIT].nunique()
    )
    in_range = selected.loc[~beyond].reset_index(drop=True)
    read = [EXPOSURE, *settings.clocks(), REWARD]
    missing = in_range[read].isna()
    counts["bins_missing"] = {column: int(missing[column].sum()) for column in read}
    any_missing = missing.any(axis=1).to_numpy()
    counts |= _counted(in_range.loc[any_missing], "missing_any")
    rows_a = in_range.loc[~any_missing].reset_index(drop=True)
    rows_d, counts_d = model_d_rows(rows_a)
    counts |= _counted(rows_a, "A") | counts_d
    return CohortRows(kind, {"A": rows_a, "D": rows_d}, counts)


def _summed(counts: Sequence[Mapping]) -> dict:
    """The cohort counts of a unit: its sessions' counts, summed key by key
    (a count summed, a nested mapping summed by its keys, a flag that must
    agree kept once). The counts of one session are themselves."""
    out: dict = {}
    for name in dict.fromkeys(k for c in counts for k in c):
        values = [c[name] for c in counts if name in c]
        if all(isinstance(v, Mapping) for v in values):
            out[name] = _summed(values)
        elif all(isinstance(v, bool | np.bool_) for v in values):
            if len(set(values)) > 1:
                raise ValueError(f"cohort count {name!r} differs between sessions")
            out[name] = values[0]
        elif all(isinstance(v, int | np.integer) for v in values):
            out[name] = int(sum(values))
        else:
            raise ValueError(f"cohort count {name!r} cannot be summed: {values}")
    return out


# --- The fit -------------------------------------------------------------------


class VisitRateFit(NamedTuple):
    """One model's fit of one kind over a set of sessions.

    Attributes
    ----------
    model, kind : str
    settings : VisitRateSettings
    design : Design
        Built on the pooled rows of the sessions fitted.
    records : list of FitRecord
        One per unit of the fit (``settings.fit_by``), in the order of each
        unit's first session given; ``cohort_counts`` holds the cohort step's
        counts too, summed over the unit's sessions, and model D's counts of
        the unit (:func:`lag_curve_counts`).
    """

    model: str
    kind: str
    settings: VisitRateSettings
    design: Design
    records: list[FitRecord]


def _unit_rows(
    cohort: Mapping[SessionKey, CohortRows], model: str, kind: str, fit_by: FitBy
) -> tuple[dict[SessionKey, pd.DataFrame], dict[tuple, dict]]:
    """Each session's rows of the model, and each unit's cohort counts.

    The units are those of ``fit_by`` (``tables.session_keys.unit_key``). A
    unit's counts are its sessions' counts summed (:func:`_summed`) and model
    D's counts of the unit (:func:`lag_curve_counts`); model D's rows of a
    unit that cannot identify the lag curve are emptied.
    """
    sessions, units = {}, {}
    for key, rows in cohort.items():
        if rows.kind != kind:
            raise ValueError(
                f"session {'/'.join(key)}: rows of {rows.kind}, not {kind}"
            )
        sessions[key] = rows.rows[model]
        units.setdefault(unit_key(key, fit_by), []).append(key)
    counts = {}
    for unit, keys in units.items():
        counts_d = lag_curve_counts([cohort[key].rows["D"] for key in keys])
        if model == "D" and counts_d["bins_lag_curve_not_identified"]:
            for key in keys:
                sessions[key] = sessions[key].iloc[:0]
        counts[unit] = _summed([cohort[key].counts for key in keys]) | counts_d
    return sessions, counts


def fit_visit_rate(
    cohort: Mapping[SessionKey, CohortRows],
    *,
    model: str,
    kind: str,
    settings: VisitRateSettings,
    table_sha256: Mapping[str, str] | None = None,
    allow_provisional: Literal["test", "simulation"] | None = None,
) -> VisitRateFit:
    """Fit model A or D of one kind in every unit of the sessions given.

    The design is built on the pooled rows of the sessions given
    (:func:`build_visit_rate_design`), then each unit of ``settings.fit_by``
    (a recording day, an animal, or a session) is fitted at the declared log
    penalties by ``sessions.fit_sessions``, each session with its own
    intercept. No penalty is chosen from the data and nothing is selected. A curve whose log penalty
    has no calibration record (``VisitRateSettings.provisional_curves``) is
    refused unless ``allow_provisional`` says the fit is a test or a
    simulation.

    Parameters
    ----------
    cohort : mapping of (animal, session_date, epoch) to CohortRows
        Each session's rows of the kind (:func:`cohort_rows`).
    model : {"A", "D"}
    kind : {"SWR", "No SWR"}
    settings : VisitRateSettings
        The settings the rows were made with.
    table_sha256 : mapping of str to str, optional
        The tables the rows were read from -> their hashes; recorded.
    allow_provisional : {"test", "simulation"} or None
        Set only for a test or a simulation: it allows provisional log
        penalties. None (a real fit) requires calibrated ones.

    Returns
    -------
    VisitRateFit

    Raises
    ------
    ValueError
        If a curve's log penalty is provisional and ``allow_provisional`` is
        None, or ``allow_provisional`` is not one of :data:`PROVISIONAL_USES`.
    """
    _model(model)
    if allow_provisional is not None and allow_provisional not in PROVISIONAL_USES:
        raise ValueError(
            f"allow_provisional {allow_provisional!r} not in {PROVISIONAL_USES}"
        )
    provisional = settings.provisional_curves(model, _kind(kind))
    if provisional and allow_provisional is None:
        raise ValueError(
            f"model {model}, {kind}: the log penalties of {list(provisional)} are "
            "provisional (no calibration record); calibrate them "
            "(calibrate_log_penalty, VisitRateSettings.calibrated_penalties) or "
            "pass allow_provisional='test' or 'simulation'"
        )
    sessions, counts = _unit_rows(cohort, model, kind, settings.fit_by)
    pooled = pd.concat(list(sessions.values()), ignore_index=True)
    design = build_visit_rate_design(model, kind, pooled, settings)
    records = fit_sessions(
        design,
        sessions,
        Poisson(),
        model=MODEL_NAMES[model],
        kind=kind,
        response=RESPONSE,
        unit=UNIT,
        table_sha256=dict(table_sha256 or {}),
        fit_by=settings.fit_by,
    )
    records = [
        replace(r, cohort_counts={**r.cohort_counts, **counts[r.unit_key]})
        for r in records
    ]
    if model == "A" and settings.reward_form == "by_outcome":
        records = [
            replace(
                r,
                cohort_counts={
                    **r.cohort_counts,
                    WINDOW_SUPPORT: window_support(
                        [
                            sessions[(e["animal"], e["session_date"], e["epoch"])]
                            for e in r.sessions
                            if e["n_rows"] > 0
                        ],
                        settings,
                    ),
                },
            )
            for r in records
        ]
    return VisitRateFit(model, kind, settings, design, records)


def fit_visit_rate_session(
    key: SessionKey,
    rows: CohortRows,
    *,
    model: str,
    settings: VisitRateSettings,
    table_sha256: Mapping[str, str] | None = None,
    allow_provisional: Literal["test", "simulation"] | None = None,
) -> VisitRateFit:
    """Fit model A or D in one session (:func:`fit_visit_rate` of that
    session alone: its design's constraints are this session's rows')."""
    return fit_visit_rate(
        {key: rows},
        model=model,
        kind=rows.kind,
        settings=settings,
        table_sha256=table_sha256,
        allow_provisional=allow_provisional,
    )


def calibrate_log_penalty(
    cohort: Mapping[SessionKey, CohortRows],
    *,
    model: str,
    kind: str,
    curve: str,
    target_edf: float,
    settings: VisitRateSettings,
) -> EdfCalibration:
    """The log penalty of one curve at which its median edf over the units of
    the fit (``settings.fit_by``) is ``target_edf``
    (``calibration.log_penalty_for_edf``), every other penalty at the
    setting's value.

    Parameters
    ----------
    cohort : mapping of (animal, session_date, epoch) to CohortRows
    model : {"A", "D"}
    kind : {"SWR", "No SWR"}
    curve : str
        A curve of :meth:`VisitRateSettings.curves`: a clock column, an
        arrival curve of :data:`ARRIVAL_BY_OUTCOME`, or :data:`LAG_TERM` for
        model D.
    target_edf : float
    settings : VisitRateSettings

    Returns
    -------
    EdfCalibration
    """
    sessions, _ = _unit_rows(cohort, _model(model), _kind(kind), settings.fit_by)
    pooled = pd.concat(list(sessions.values()), ignore_index=True)
    design = build_visit_rate_design(model, kind, pooled, settings)
    named = (LAG_TERM, *ARRIVAL_BY_OUTCOME.values())
    term = curve if curve in named else _smooth_name(curve)
    return log_penalty_for_edf(
        design,
        sessions,
        Poisson(),
        term=term,
        target_edf=target_edf,
        response=RESPONSE,
        unit=UNIT,
        fit_by=settings.fit_by,
    )
