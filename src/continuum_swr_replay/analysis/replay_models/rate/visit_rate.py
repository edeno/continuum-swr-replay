"""The visit rate model: event starts on the three visit clocks, models A and D.

One fit for each session and each kind (SWR, No SWR), by the shared fitter
(``analysis.regression``): plain penalized maximum likelihood, Poisson with
the offset ``log(exposure)``, on the in-visit at-risk bins of the visit
time-bin design (``replay_models.visit_design.visit_timebins``).

- Model A: ``f_s(time since the first poke-in) + f_u(time until the final
  poke-out) [+ f_m(time until the end of the immobile stretch)] + beta *
  last_outcome_rewarded``.
- Model D: the same smooths and, in place of the 0/1 term, a smooth curve over
  the last ten known outcomes on the log-lag scale (a lag functional), on the
  bins with ten known outcomes.

Each clock curve, and model D's lag curve, is a cubic P-spline at ONE fixed,
declared log penalty, the same in every session; no penalty is chosen from
the data. The reward coefficient, the one simple tested coefficient, carries
the weak penalty (a Normal prior of mean 0, ``terms.log_penalty_from_prior_sd``);
the intercept is not penalized. Every model's design is checked by
``support.require_tested_penalized`` with the terms its estimands read.

The model's form has open questions. Each is a named setting below, in one
place, with the values allowed; :class:`VisitRateSettings` holds one choice of
each, and the model builds and fits for every allowed choice. The defaults
are the declared form (the coefficient form of the reward effect, Q1's
declared reference window, all in-visit bins, the third clock in the first
fit); the log penalties are provisional values until calibrated.

The cohort step (:func:`cohort_rows`): ``visit_timebins`` of a session and
kind -> ``select_visit_context`` (the only filter of the bins) -> the bins
beyond the time range, if one is set, dropped and counted -> the rows with a
missing covariate dropped and counted -> model D also drops the bins without
ten known outcomes, counted. The counts are added to every record's
``cohort_counts``. The two clocks that count down reference the future
(``covariate_timing.FUTURE_REFERENCING``): this model describes; it does not
predict.

The third clock's rule, written before any fit (:func:`third_clock_decision`):
fit A with ``f_m``; if the concurvity ``estimate`` of ``f_m`` or of ``f_u`` is
at least ``collinearity.MAX_EXPLAINED`` (0.8) in more than
``MAX_FAILING_SHARE`` (20 %) of the sessions with a fit, A is refit without
``f_m``, and the time-until-movement curve is reported only from the
model-free curves. Both fits are stored.
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
from continuum_swr_replay.analysis.regression.collinearity import separable
from continuum_swr_replay.analysis.regression.constants import MAX_FAILING_SHARE
from continuum_swr_replay.analysis.regression.likelihoods import Poisson
from continuum_swr_replay.analysis.regression.newton import CONVERGED
from continuum_swr_replay.analysis.regression.pspline import PSplineBasis
from continuum_swr_replay.analysis.regression.record import FitRecord
from continuum_swr_replay.analysis.regression.sessions import SessionKey, fit_sessions
from continuum_swr_replay.analysis.regression.support import require_tested_penalized
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

from ..declared_tests import CoefficientContrast, CurveContrast, estimand
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
    "RESPONSE",
    "REWARDED_ARRIVAL_TERM",
    "REWARD",
    "REWARD_FORMS",
    "STRETCH",
    "TIME_RANGE_CAPS_S",
    "UNIT",
    "WEAK_PRIOR_SDS",
    "CohortRows",
    "ThirdClockDecision",
    "VisitRateFit",
    "VisitRateSettings",
    "build_visit_rate_design",
    "calibrate_log_penalty",
    "cohort_rows",
    "curve_rows",
    "declared_contrasts",
    "fit_visit_rate",
    "fit_visit_rate_session",
    "lag_curve_rows",
    "model_terms",
    "require_declared_penalized",
    "tested_terms",
    "third_clock_decision",
]

# --- The columns the model reads -------------------------------------------

#: Time since the first poke-in of the visit (s).
ARRIVAL = "time_since_visit_start_s"
#: Time until the final poke-out of the visit (s); counts down.
LEAVING = "time_until_visit_end_s"
#: Time until the end of the current immobile stretch (s); counts down.
STRETCH = "time_until_stretch_end_s"
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
#: from model A.
RATE_ESTIMANDS: tuple[str, ...] = ("Q1", "Q2", "Q7a")

# --- The bases (calls of the design, "Smooth terms and bases") -------------

#: Each clock's range and knot spacing (s). Beyond its range a clock is
#: clamped (the curve is flat there).
CLOCK_BASES: dict[str, tuple[float, float, float]] = {
    ARRIVAL: (0.0, 20.0, 0.5),
    LEAVING: (0.0, 3.0, 0.1),
    STRETCH: (0.0, 1.5, 0.1),
}

#: Model D's lag curve: basis functions over ``[log 1, log 10]``.
LAG_N_BASIS = 8

#: The lags of model D (1 the most recent known outcome).
LAGS: tuple[int, ...] = tuple(range(1, REWARD_HISTORY_N_LAGS + 1))

#: Model D's lag-curve term, as ``terms`` names it.
LAG_TERM = f"lf({REWARD_LAG_COLUMNS[0]}..{REWARD_LAG_COLUMNS[-1]})"


def _smooth_name(column: str) -> str:
    return f"s({column})"


#: The term of the reward effect's own arrival curve (the time-varying form):
#: the arrival smooth fitted on the rows after a reward.
REWARDED_ARRIVAL_TERM = f"{_smooth_name(ARRIVAL)}:{REWARD}=1"

# --- OPEN (lead): the settings of the model's form --------------------------
#
# Each setting below is an open question of the lead (design, "Open after the
# twenty-third round", b and g). None is decided here; the model builds and
# fits for each allowed value. The defaults are the declared form and do not
# decide anything: no real fit is run on them.

#: OPEN (lead): the prior SD of the weak penalty on every simple tested
#: coefficient, in its reported unit (here a 0/1 step on the log rate); the
#: log penalty is ``log(1 / SD^2)`` (``terms.log_penalty_from_prior_sd``).
#: The candidates of the running simulation. The default
#: (:class:`VisitRateSettings`) is the middle one, 2; the fit is also
#: reported over the strength (a sensitivity analysis).
WEAK_PRIOR_SDS: tuple[float, ...] = (1.0, 2.0, 4.0)

#: OPEN (lead): the clock penalty setting. Each clock's declared log penalty
#: is to be set by ``calibration.log_penalty_for_edf`` on this model's own real
#: rows, per kind, at a target median edf (:func:`calibrate_log_penalty`); the
#: fit is also reported at a stiffer and a looser value (one step of the
#: target each way). The default is ``"declared"``.
CLOCK_PENALTY_SETTINGS: tuple[str, ...] = ("declared", "stiffer", "looser")

#: OPEN (lead): the log penalty of each curve, per kind and setting.
#: PROVISIONAL, NOT CALIBRATED on this model's real rows; replace each value by
#: :func:`calibrate_log_penalty`'s, with its target, commit and rows, before
#: the first real fit. Where the values come from: the clocks of time since
#: arrival and time until leaving are those the realistic simulation
#: prototype found on one simulated data set of model A (the same bases) at a
#: median edf of 6 ("declared") and 10 ("looser"); "stiffer" is the declared
#: value plus the same step in log penalty. The stretch clock (no prototype
#: value) takes the leaving clock's values (the same knot spacing). Model D's
#: lag curve has no prior value: 2, with 4 and 0 either side. The rewarded
#: arrival curve of the time-varying reward form takes the arrival clock's
#: value.
CLOCK_LOG_PENALTIES: dict[str, dict[str, dict[str, float]]] = {
    "SWR": {
        "declared": {ARRIVAL: 1.0522, LEAVING: 3.0718, STRETCH: 3.0718, LAG_TERM: 2.0},
        "stiffer": {ARRIVAL: 3.8344, LEAVING: 5.3921, STRETCH: 5.3921, LAG_TERM: 4.0},
        "looser": {ARRIVAL: -1.7300, LEAVING: 0.7515, STRETCH: 0.7515, LAG_TERM: 0.0},
    },
    "No SWR": {
        "declared": {ARRIVAL: 1.8472, LEAVING: 3.2544, STRETCH: 3.2544, LAG_TERM: 2.0},
        "stiffer": {ARRIVAL: 4.3179, LEAVING: 5.5317, STRETCH: 5.5317, LAG_TERM: 4.0},
        "looser": {ARRIVAL: -0.6235, LEAVING: 0.9771, STRETCH: 0.9771, LAG_TERM: 0.0},
    },
}

#: OPEN (lead): the time range of the model. None models every in-visit bin
#: (the default); a cap keeps only the bins whose time since arrival is below
#: it (candidate 10 s), and the arrival clock's range ends at the cap.
TIME_RANGE_CAPS_S: tuple[float | None, ...] = (None, 10.0)

#: OPEN (lead): the form of the reward effect. ``"coefficient"`` (the
#: default, the declared Q2): one coefficient. ``"time_varying"``: the same
#: coefficient and a second arrival curve for the rows after a reward (a
#: smooth ``by`` the last outcome, sum-to-zero on those rows), so the effect
#: may change with time since arrival; its coefficient is then the effect
#: averaged over the rewarded rows' arrival times. An option to build and
#: test, not the declared test.
REWARD_FORMS: tuple[str, ...] = ("coefficient", "time_varying")

#: OPEN (lead): the reference window of Q1 (s since arrival). ``(0, 1)`` is
#: the declared one (``declared_tests``, the default); ``(1, 2)`` the
#: alternative (SWR events almost never start in the first second).
Q1_REFERENCE_WINDOWS: tuple[tuple[float, float], ...] = ((0.0, 1.0), (1.0, 2.0))


@dataclass(frozen=True)
class VisitRateSettings:
    """One choice of each open setting of the model's form.

    Attributes
    ----------
    weak_prior_sd : float
        The weak penalty's prior SD (:data:`WEAK_PRIOR_SDS`; any positive
        value for a sensitivity analysis).
    clock_penalty : {"declared", "stiffer", "looser"}
        Which of :data:`CLOCK_LOG_PENALTIES` the curves take.
    max_time_since_arrival_s : float or None
        :data:`TIME_RANGE_CAPS_S`.
    reward_form : {"coefficient", "time_varying"}
        :data:`REWARD_FORMS`.
    q1_reference_s : tuple of float
        :data:`Q1_REFERENCE_WINDOWS`.
    include_stretch_clock : bool
        Whether model A and D hold ``f_m``; the third clock's rule
        (:func:`third_clock_decision`) decides it after the first fit, which
        holds it.
    log_penalties : mapping of str to float or None
        Curve column (or :data:`LAG_TERM`) -> a log penalty in place of the
        table's, for the calibration and the tests; None uses the table.
    """

    weak_prior_sd: float = 2.0
    clock_penalty: Literal["declared", "stiffer", "looser"] = "declared"
    max_time_since_arrival_s: float | None = None
    reward_form: Literal["coefficient", "time_varying"] = "coefficient"
    q1_reference_s: tuple[float, float] = (0.0, 1.0)
    include_stretch_clock: bool = True
    log_penalties: Mapping[str, float] | None = None

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

    @property
    def weak_log_penalty(self) -> float:
        """The weak penalty's log penalty, ``log(1 / SD^2)``."""
        return log_penalty_from_prior_sd(self.weak_prior_sd)

    def log_penalty(self, kind: str, curve: str) -> float:
        """The log penalty of one curve (a clock column or :data:`LAG_TERM`)."""
        if self.log_penalties is not None and curve in self.log_penalties:
            return float(self.log_penalties[curve])
        return CLOCK_LOG_PENALTIES[_kind(kind)][self.clock_penalty][curve]

    def clocks(self) -> tuple[str, ...]:
        """The clock columns the model holds, in order."""
        return (ARRIVAL, LEAVING, *((STRETCH,) if self.include_stretch_clock else ()))

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
    clock: str, kind: str, settings: VisitRateSettings, **by: object
) -> Smooth:
    lo, hi = settings.clock_range(clock)
    basis = PSplineBasis.from_spacing(lo, hi, CLOCK_BASES[clock][2])
    return Smooth(clock, lo, hi, basis.n_basis, settings.log_penalty(kind, clock), **by)


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
        The offset, the clock smooths, then model A's reward slope (and, in
        the time-varying form, the rewarded rows' arrival smooth) or model
        D's lag curve. The intercept is the design's, unpenalized.
    """
    _model(model)
    _kind(kind)
    terms: list[Term] = [Offset(EXPOSURE)]
    terms += [_clock_smooth(clock, kind, settings) for clock in settings.clocks()]
    if model == "A":
        terms.append(Slope(REWARD, settings.weak_log_penalty))
        if settings.reward_form == "time_varying":
            outcome = Category(REWARD, (0, 1), 0, "none")
            terms.append(
                _clock_smooth(ARRIVAL, kind, settings, by=outcome, levels=(1,))
            )
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
    declared tests (Q1, Q7a: the clock smooths; Q2: the reward coefficient,
    and the rewarded arrival curve in the time-varying form); model D's lag
    curve (described, not tested)."""
    if _model(model) == "D":
        return (LAG_TERM,)
    out = [_smooth_name(ARRIVAL), _smooth_name(LEAVING), REWARD]
    if settings.reward_form == "time_varying":
        out.append(REWARDED_ARRIVAL_TERM)
    return tuple(out)


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
    session fitted with this design shares them. The guard
    (:func:`require_declared_penalized`) is called on the built design.

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
        model_terms(model, kind, settings), pooled_rows, intercept=True
    )
    require_declared_penalized(design, model, settings)
    return design


# --- The declared tests on the model's own terms -----------------------------


def _probe(design: Design, n: int) -> pd.DataFrame:
    """Rows at which every column the model reads is 0 (exposure 1)."""
    columns = [ARRIVAL, LEAVING, STRETCH, REWARD, *REWARD_LAG_COLUMNS]
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


def declared_contrasts(
    design: Design, settings: VisitRateSettings
) -> dict[str, np.ndarray]:
    """The ``c`` of each declared estimand of the model (Q1, Q2, Q7a), on
    model A's design.

    Each is read from ``declared_tests``; Q1's reference window is the
    setting's (the declared ``(0, 1)`` by default). A curve contrast is the
    range mean of its smooth over the contrast's ranges
    (``contrasts.range_mean_contrast``), which must lie inside the term's
    basis range; a coefficient contrast names coefficients of the design.

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
        elif isinstance(contrast, CoefficientContrast):
            out[estimand_id] = contrasts.coefficient_contrast(
                design.coef_names, contrast.plus, contrast.minus
            )
        else:
            raise ValueError(f"{estimand_id}: a {type(contrast).__name__} contrast")
    return out


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
        dropped ``beyond_time_range``, dropped ``missing_any`` (a missing
        covariate; ``bins_missing`` counts each column's), model ``A``'s,
        dropped ``without_ten_lags`` (of model A's), and model ``D``'s.
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
        else ~(selected[ARRIVAL].to_numpy(dtype=float) < cap)
    )
    counts |= _counted(selected.loc[beyond], "beyond_time_range")
    in_range = selected.loc[~beyond].reset_index(drop=True)
    read = [EXPOSURE, *settings.clocks(), REWARD]
    missing = in_range[read].isna()
    counts["bins_missing"] = {column: int(missing[column].sum()) for column in read}
    any_missing = missing.any(axis=1).to_numpy()
    counts |= _counted(in_range.loc[any_missing], "missing_any")
    rows_a = in_range.loc[~any_missing].reset_index(drop=True)
    without = ~rows_a[list(REWARD_LAG_COLUMNS)].notna().all(axis=1).to_numpy()
    counts |= _counted(rows_a.loc[without], "without_ten_lags")
    rows_d = rows_a.loc[~without].reset_index(drop=True)
    counts |= _counted(rows_a, "A") | _counted(rows_d, "D")
    return CohortRows(kind, {"A": rows_a, "D": rows_d}, counts)


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
        One per session, in the order given; ``cohort_counts`` holds the
        cohort step's counts too.
    """

    model: str
    kind: str
    settings: VisitRateSettings
    design: Design
    records: list[FitRecord]


def _session_rows(
    cohort: Mapping[SessionKey, CohortRows], model: str, kind: str
) -> dict[SessionKey, pd.DataFrame]:
    out = {}
    for key, rows in cohort.items():
        if rows.kind != kind:
            raise ValueError(
                f"session {'/'.join(key)}: rows of {rows.kind}, not {kind}"
            )
        out[key] = rows.rows[model]
    return out


def fit_visit_rate(
    cohort: Mapping[SessionKey, CohortRows],
    *,
    model: str,
    kind: str,
    settings: VisitRateSettings,
    table_sha256: Mapping[str, str] | None = None,
) -> VisitRateFit:
    """Fit model A or D of one kind in every session given.

    The design is built on the pooled rows of the sessions given
    (:func:`build_visit_rate_design`), then every session is fitted alone at
    the declared log penalties by ``sessions.fit_sessions``. No penalty is
    chosen from the data and nothing is selected.

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

    Returns
    -------
    VisitRateFit
    """
    _model(model)
    sessions = _session_rows(cohort, model, _kind(kind))
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
    )
    records = [
        replace(r, cohort_counts={**r.cohort_counts, **cohort[key].counts})
        for key, r in zip(sessions, records, strict=True)
    ]
    return VisitRateFit(model, kind, settings, design, records)


def fit_visit_rate_session(
    key: SessionKey,
    rows: CohortRows,
    *,
    model: str,
    settings: VisitRateSettings,
    table_sha256: Mapping[str, str] | None = None,
) -> VisitRateFit:
    """Fit model A or D in one session (:func:`fit_visit_rate` of that
    session alone: its design's constraints are this session's rows')."""
    return fit_visit_rate(
        {key: rows},
        model=model,
        kind=rows.kind,
        settings=settings,
        table_sha256=table_sha256,
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
    """The log penalty of one curve at which its median edf over the sessions
    is ``target_edf`` (``calibration.log_penalty_for_edf``), every other
    penalty at the setting's value.

    Parameters
    ----------
    cohort : mapping of (animal, session_date, epoch) to CohortRows
    model : {"A", "D"}
    kind : {"SWR", "No SWR"}
    curve : str
        A clock column of the model, or :data:`LAG_TERM` for model D.
    target_edf : float
    settings : VisitRateSettings

    Returns
    -------
    EdfCalibration
    """
    sessions = _session_rows(cohort, _model(model), _kind(kind))
    pooled = pd.concat(list(sessions.values()), ignore_index=True)
    design = build_visit_rate_design(model, kind, pooled, settings)
    term = curve if curve == LAG_TERM else _smooth_name(curve)
    return log_penalty_for_edf(
        design,
        sessions,
        Poisson(),
        term=term,
        target_edf=target_edf,
        response=RESPONSE,
        unit=UNIT,
    )


# --- The third clock's rule ---------------------------------------------------


class ThirdClockDecision(NamedTuple):
    """The third clock's rule on model A's fit with ``f_m``.

    Attributes
    ----------
    keep : bool
        False when ``f_m`` leaves the model.
    share : float
        The share of the sessions with a fit in which ``f_m`` or ``f_u`` is
        not separable (concurvity ``estimate`` at least ``MAX_EXPLAINED``).
    n_sessions : int
        The sessions with a fit, in which both terms are measured.
    """

    keep: bool
    share: float
    n_sessions: int


def third_clock_decision(records: Sequence[FitRecord]) -> ThirdClockDecision:
    """The third clock's rule (module docstring) on model A's records with
    ``f_m``.

    Raises
    ------
    ValueError
        If no session has a fit with both terms measured.
    """
    pair = (_smooth_name(STRETCH), _smooth_name(LEAVING))
    flagged = []
    for record in records:
        if record.status != CONVERGED or record.no_fit:
            continue
        measured = {
            name: record.collinearity[name]
            for name in pair
            if name in record.collinearity
            and "not_measured" not in record.collinearity[name]
        }
        if len(measured) < len(pair):
            continue
        flagged.append(not all(separable(measured).values()))
    if not flagged:
        raise ValueError("no session has a fit with f_m and f_u measured")
    share = float(np.mean(flagged))
    return ThirdClockDecision(share <= MAX_FAILING_SHARE, share, len(flagged))
