"""Content on the nine arms by arm role: the cohort, the design and the fit.

The response is each event's share over its nine candidate arms (the event-arm
table's ``arm_fraction``), fitted by the grouped softmax, one event per group,
one fit for each session and each kind (``SWR``, ``No SWR``).

Roles, exclusive, per (event, arm): ``current`` (the arm of the well being
visited; for an event that crosses the final poke-out, the visit's well's arm,
as the event-arm table sets it), ``previous`` (the arm of the previous trial's
well, when it is not the current one), ``other_leaf`` (any other leaf arm, the
reference) and ``trunk`` (arms 6 to 8). A well touches exactly one arm, so the
current and the previous role each mark at most one of the nine rows of an
event.

Two models, each in two variants:

- ``role_content``: arm effects (a baseline, not penalized), the role category,
  a reward-fraction smooth per leaf role, a value slope per leaf role, the
  recency smooth and the never-visited indicator on the other-leaf rows, and
  the current role's smooth of time until leaving;
- ``role_content_uncertainty``: the same plus, per leaf role, the belief
  variance, the sample count and the trial number in the epoch.

The ``unadjusted`` variant is as above. The ``length_adjusted`` variant adds,
for the current and previous roles (penalized) and the trunk role (a baseline),
a slope on ``log(event duration) - L0``, and for every role slope a slope on the
product of its covariate with ``log(event duration) - L0``. ``L0`` is the
median log duration over the cohort's events of both kinds, fixed and stored.
An estimand of that variant is read at ``L0``. The smooths do not interact with
the length.

Each estimand of :mod:`..declared_tests` read from these models is a linear
functional of the fit (:func:`estimand_contrasts`), built from the model's own
design at stated reference rows, so it holds for both variants: a role
contrast is the current (or previous) role's row less an other-leaf row of the
same arm, every slope covariate at its pooled reference and the length at
``L0``, the smooth terms left out; a slope is the change of one other-leaf row
over one pooled SD of its covariate at ``L0``; a curve is read from the term's
own columns. In the unadjusted model these are the declared coefficient
contrasts themselves.

The recency smooth and the never-visited indicator are on the other-leaf rows
only; this is decided. The current arm is always 1 trial since its last visit
and the previous arm always 2, so a recency curve on those rows would be the
same column as their role level, and the split between the two would be set by
the penalties. The role levels therefore carry everything for the current and
previous arms: Q3a and Q3b compare them with an average other leaf arm, and in
this task "current arm" cannot be separated from "most recently visited arm".
A visited other-leaf arm is at least 3 trials since its last visit; one never
visited carries the censored count, which can be 2, and the never-visited
indicator. Q5a reads the other-leaf curve at its two points.

A call of this module, made where the plan does not fix it:

- The trunk level of the role category is a function of the arm, so it is not
  identified beside the arm effects. It is kept as a level (so that the trunk
  rows are a role of their own for every term by role and for the support
  rule) and the weak penalty of the role category sets its coefficient to 0;
  the likelihood is flat along that one direction, so every other coefficient
  and its variance are those of the model without it. The record's
  collinearity of the term ``role`` therefore counts that column, which is
  exactly collinear with the arm effects (its ``worst`` is 1) while its
  ``estimate`` can be far below the rule's limit. The collinearity rule of
  these models reads the collinearity of each tested coefficient alone
  (:func:`tested_coefficient_collinearity`, :attr:`RoleContentFit.separable`),
  never the record's term ``role``; the trunk level is not a tested
  coefficient and is left out of it, because its collinearity is a property of
  the design (a function of the arm, fitted at 0), not of the data.
- The collinearity of a tested coefficient is measured on the information of
  the penalized problem with the smooths' declared penalties in and the weak
  penalties out (:func:`tested_coefficient_collinearity` says why). On the
  information of the likelihood alone a role level is collinear with its own
  curves in most real sessions (R^2 1 to six decimals), because a curve's
  sum-to-zero constraint is computed on the scale rows and a basis function
  seen there can be unseen in a session; the curve's penalty fixes that
  direction, and the fit recovers the level.

The cohort step (:func:`role_content_cohort`) selects the events with
``select_visit_context`` (default ``STARTS_IN_WELL_VISIT``), checks the nine
candidate rows of each event, assigns the roles and marks, per model, every
event with a missing value on a row that reads one of the model's columns (an
event's choice set is never cut to fewer than nine arms), counting the events
and rows by model, kind and column. A model drops an event only for the
columns it reads: the plain model keeps an event that lacks a value only the
uncertainty model reads (:meth:`RoleContentCohort.model_rows`).

Response weights: the grouped softmax reads a response that sums to ``n_g``
over an event's rows as ``n_g`` draws (``regression.likelihoods.GroupedSoftmax``),
so an event's weight enters as its shares times the weight
(:data:`EVENT_WEIGHTS`); the fitter needs no change for it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Literal, NamedTuple

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.constants import STARTS_IN_WELL_VISIT
from continuum_swr_replay.analysis.event_definition.cohort_selection import (
    count_visit_context,
    select_visit_context,
)
from continuum_swr_replay.analysis.regression.collinearity import (
    RANK_TOL,
    concurvity,
    separable,
    weighted_design,
)
from continuum_swr_replay.analysis.regression.contrasts import range_mean_contrast
from continuum_swr_replay.analysis.regression.likelihoods import GroupedSoftmax
from continuum_swr_replay.analysis.regression.newton import CONVERGED
from continuum_swr_replay.analysis.regression.pspline import PSplineBasis
from continuum_swr_replay.analysis.regression.record import FitRecord
from continuum_swr_replay.analysis.regression.sessions import SessionKey, fit_sessions
from continuum_swr_replay.analysis.regression.support import require_tested_penalized
from continuum_swr_replay.analysis.regression.terms import (
    Category,
    Design,
    Slope,
    Smooth,
    build_design,
    log_penalty_from_prior_sd,
)
from continuum_swr_replay.data_loaders.constants import (
    ALL_ARMS,
    LEAF_ARMS,
    TRUNK_ARMS,
)
from continuum_swr_replay.tables.columns import require_columns

from .. import declared_tests as dt

__all__ = [
    "COHORT_COLUMNS",
    "DEFAULT_PRIOR_SD",
    "DEFAULT_SMOOTH_LOG_PENALTIES",
    "EVENT_WEIGHTS",
    "LEAF_ROLES",
    "MODELS",
    "MODEL_COLUMNS",
    "PRIOR_SD_CANDIDATES",
    "Q4A_READINGS",
    "REFERENCE_ROLE",
    "ROLES",
    "SCALE_POOLINGS",
    "TRIAL_NUMBER_COLUMNS",
    "TRUNK_LEVEL",
    "VARIANTS",
    "ContrastReference",
    "RoleContentCohort",
    "RoleContentFit",
    "RoleContentSettings",
    "assign_roles",
    "design_rows",
    "estimand_contrasts",
    "fit_role_content",
    "fit_role_content_session",
    "model_estimands",
    "role_content_cohort",
    "role_design",
    "role_terms",
    "session_rows",
    "tested_coefficient_collinearity",
    "tested_coefficients",
]

# ---------------------------------------------------------------------------
# The open settings, in one place. Each is a question for the lead; the
# default is a placeholder that lets the module build and fit, and no real fit
# is made with it before the lead answers.
# ---------------------------------------------------------------------------

PRIOR_SD_CANDIDATES: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0)
"""OPEN (lead): penalty width. The Normal prior SD of every simple tested
coefficient (each role level, each role slope), in the reported unit (a 0/1
step; one pooled SD of a continuous covariate): the log penalty is
``log(1 / SD**2)``. One declared number for all models; these are the widths
the simulations compare. A weak prior is not weak on this likelihood's scale:
with one unit of weight per event (:data:`EVENT_WEIGHTS`) the information is
small, and the penalty shrinks a rare role level or a slope in a small
session toward 0 (:func:`.role_content_statistics.penalized_edf` reports by
how much)."""

DEFAULT_PRIOR_SD = 2.0
"""The width used until the lead chooses one: a placeholder."""

DEFAULT_SMOOTH_LOG_PENALTIES: Mapping[str, float] = {
    "reward_fraction": 0.0,
    "log_trials_since_last_visit": -4.0,
    "time_until_visit_end_s": 0.0,
}
"""OPEN (lead): the declared fixed log penalty of each smooth, by its
covariate, the same in every session. Each is to be set by
``regression.calibration.log_penalty_for_edf`` on this model's real rows at a
target median edf (a target is not carried over from another model) and
checked by the scenario list. The defaults (recency -4, the two others 0) are
placeholders, with no measurement on this model's rows behind them."""

SCALE_POOLINGS: tuple[str, ...] = ("both_kinds", "per_kind")
"""OPEN (lead): the rows the design's scales and reference points are computed
on: each slope's pooled SD and centering mean, each smooth's sum-to-zero
constraint, the reward fraction's mean and SD of Q4a and the covariate means of
the length products. ``both_kinds``: the rows of both kinds (SWR and No SWR)
pooled, so that "per pooled SD" and every reference point are the same for
the two kinds and the kind difference is a difference of like quantities.
``per_kind``: the rows of the kind fitted, so that each kind's estimand is
per its own SD and at its own means. ``L0`` is pooled over both kinds in
either case. The default is ``both_kinds``, for the reason above; the lead
has not yet confirmed it."""

EVENT_WEIGHTS: tuple[str, ...] = ("one_per_event", "n_nonlocal_bins")
"""OPEN (lead): the weight of each event in the likelihood. ``one_per_event``
(the default, the lead's earlier decision): the response is the event's
shares, one categorical draw per event whatever the number of decode bins
behind them. ``n_nonlocal_bins``: the response is the shares times the
event's number of non-local decode bins (the event-arm table's column of
that name), as if each bin were one draw; overdispersion across bins makes
that information too large, so it bounds the other side."""


def _event_weight_column(event_weight: str) -> str | None:
    """The event-arm table's column of an event weight; None for one per
    event."""
    return None if event_weight == "one_per_event" else event_weight


TRIAL_NUMBER_COLUMNS: tuple[str, ...] = ("n_visits_arrived", "completed_trial_id")
"""OPEN (lead): which trial number in the epoch the uncertainty model holds.
``n_visits_arrived`` (the visits arrived at by the event start, the starting
one included; on the event-arm table) is the plan's reading;
``completed_trial_id`` (the event table's completed trials) needs the caller
to join it onto the event-arm rows. Per pooled SD, a slope that differs from
the other by a constant shift is the same fit."""

Q4A_READINGS: tuple[str, ...] = ("sd_change", "full_range")
"""OPEN (lead): the reading of Q4a from the current role's reward-fraction
smooth. ``sd_change`` (the plan's): ``f(m + sd/2) - f(m - sd/2)``, ``m`` and
``sd`` the pooled mean and population SD of the fraction on current rows;
``full_range``: ``f(1) - f(0)``. Both are linear functionals with an exact
variance."""

DEFAULT_Q5A_TRIALS: tuple[float, float] = (20.0, 3.0)
"""OPEN (lead): the two recency points of Q5a, trials since the last visit.
The default is the declared 20 against 3, a placeholder; 3 trials is the
lower end of a visited other-leaf arm's recency, where few rows lie."""


@dataclass(frozen=True)
class RoleContentSettings:
    """The open settings of a fit (module constants above).

    Attributes
    ----------
    prior_sd : float
        The weak penalty's prior SD, one of :data:`PRIOR_SD_CANDIDATES`.
    smooth_log_penalties : mapping of str to float
        Each smooth's covariate -> its declared log penalty.
    trial_number : str
        The column of the trial number, one of :data:`TRIAL_NUMBER_COLUMNS`.
    q4a_reading : str
        One of :data:`Q4A_READINGS`.
    q5a_trials : tuple of float
        The two recency points of Q5a, ``(first, second)`` trials.
    scale_pooling : str
        One of :data:`SCALE_POOLINGS`.
    event_weight : str
        One of :data:`EVENT_WEIGHTS`.
    """

    prior_sd: float = DEFAULT_PRIOR_SD
    smooth_log_penalties: Mapping[str, float] = field(
        default_factory=lambda: dict(DEFAULT_SMOOTH_LOG_PENALTIES)
    )
    trial_number: str = TRIAL_NUMBER_COLUMNS[0]
    q4a_reading: str = Q4A_READINGS[0]
    q5a_trials: tuple[float, float] = DEFAULT_Q5A_TRIALS
    scale_pooling: str = SCALE_POOLINGS[0]
    event_weight: str = EVENT_WEIGHTS[0]

    def __post_init__(self) -> None:
        if self.scale_pooling not in SCALE_POOLINGS:
            raise ValueError(
                f"scale_pooling {self.scale_pooling!r} not in {SCALE_POOLINGS}"
            )
        if self.event_weight not in EVENT_WEIGHTS:
            raise ValueError(
                f"event_weight {self.event_weight!r} not in {EVENT_WEIGHTS}"
            )
        if self.prior_sd not in PRIOR_SD_CANDIDATES:
            raise ValueError(
                f"prior_sd {self.prior_sd} is not a candidate width "
                f"{PRIOR_SD_CANDIDATES}"
            )
        if set(self.smooth_log_penalties) != set(DEFAULT_SMOOTH_LOG_PENALTIES):
            raise ValueError(
                "smooth_log_penalties must name exactly the smooths "
                f"{sorted(DEFAULT_SMOOTH_LOG_PENALTIES)}"
            )
        if self.trial_number not in TRIAL_NUMBER_COLUMNS:
            raise ValueError(
                f"trial_number {self.trial_number!r} not in {TRIAL_NUMBER_COLUMNS}"
            )
        if self.q4a_reading not in Q4A_READINGS:
            raise ValueError(f"q4a_reading {self.q4a_reading!r} not in {Q4A_READINGS}")
        first, second = self.q5a_trials
        lo, hi = _RECENCY_TRIALS
        if not (lo <= first <= hi and lo <= second <= hi):
            raise ValueError(
                f"q5a_trials {self.q5a_trials} outside [{lo}, {hi}] trials"
            )


# ---------------------------------------------------------------------------
# The decided design
# ---------------------------------------------------------------------------

MODELS: tuple[str, ...] = ("role_content", "role_content_uncertainty")
"""The two models, as :mod:`..declared_tests` names them."""

VARIANTS: tuple[str, ...] = ("unadjusted", "length_adjusted")

Model = Literal["role_content", "role_content_uncertainty"]
Variant = Literal["unadjusted", "length_adjusted"]

ROLES: tuple[str, ...] = ("other_leaf", "current", "previous", "trunk")
REFERENCE_ROLE = "other_leaf"
TRUNK_LEVEL = "role[trunk]"
"""The trunk level's coefficient: a function of the arm, fitted at 0."""
LEAF_ROLES: tuple[str, ...] = ("current", "previous", "other_leaf")

RESPONSE = "arm_fraction"
UNIT = "event_id"
ARM = "candidate_arm"
ROLE = "role"
KIND = "kind"
SESSION_COLUMNS = ("animal", "session_date", "epoch")

REWARD_FRACTION = "reward_fraction"
VALUE = "value"
RECENCY = "log_trials_since_last_visit"
NEVER_VISITED = "never_visited"
LEAVING = "time_until_visit_end_s"
LOG_DURATION = "log_duration"
LOG_DURATION_REL = "log_duration_rel"
BELIEF_VARIANCE = "belief_variance"
SAMPLE_COUNT = "sample_count"
TRIAL_NUMBER = "trial_number"
EVENT_WEIGHT = "event_weight"
WEIGHTED_RESPONSE = "weighted_arm_fraction"
"""The fitted response: the shares times the event's weight."""

_RECENCY_TRIALS = (1.0, 200.0)
"""Recency is clamped to this range of trials since the last visit."""

#: (covariate, lo, hi, n_basis, roles) of each smooth. The ranges are decided;
#: time until leaving has knots 0.1 s apart over [0, 3] s.
_SMOOTHS: tuple[tuple[str, float, float, int, tuple[str, ...]], ...] = (
    (REWARD_FRACTION, 0.0, 1.0, 6, LEAF_ROLES),
    (RECENCY, 0.0, float(np.log(_RECENCY_TRIALS[1])), 10, (REFERENCE_ROLE,)),
    (LEAVING, 0.0, 3.0, PSplineBasis.from_spacing(0.0, 3.0, 0.1).n_basis, ("current",)),
)

#: The slopes per leaf role of each model, each penalized (a role slope).
_ROLE_SLOPES: dict[str, tuple[str, ...]] = {
    "role_content": (VALUE,),
    "role_content_uncertainty": (VALUE, BELIEF_VARIANCE, SAMPLE_COUNT, TRIAL_NUMBER),
}

#: The event-arm table's column behind each model column, and the roles whose
#: rows read it (a missing value there drops the event). The trial number's
#: source is a setting.
_SOURCES: dict[str, tuple[str, tuple[str, ...]]] = {
    RESPONSE: (RESPONSE, ROLES),
    REWARD_FRACTION: ("reward_fraction_candidate_arm_last10_visits", LEAF_ROLES),
    VALUE: ("reward_belief_mean_candidate_arm_preoutcome", LEAF_ROLES),
    RECENCY: ("visit_recency_trials", (REFERENCE_ROLE,)),
    NEVER_VISITED: ("is_candidate_arm_never_visited", (REFERENCE_ROLE,)),
    LEAVING: ("time_until_visit_end_s", ("current",)),
    LOG_DURATION: ("event_duration_s", ROLES),
    BELIEF_VARIANCE: ("reward_belief_variance_candidate_arm_preoutcome", LEAF_ROLES),
    SAMPLE_COUNT: ("log1p_n_known_visits_candidate_arm", LEAF_ROLES),
    TRIAL_NUMBER: (None, LEAF_ROLES),
    EVENT_WEIGHT: (None, ROLES),
}

#: The cohort columns each model reads: a missing value on a row that reads
#: one of them drops the event from that model only.
MODEL_COLUMNS: dict[str, tuple[str, ...]] = {
    "role_content": (
        RESPONSE,
        REWARD_FRACTION,
        VALUE,
        RECENCY,
        NEVER_VISITED,
        LEAVING,
        LOG_DURATION,
        EVENT_WEIGHT,
    ),
}
MODEL_COLUMNS["role_content_uncertainty"] = (
    *MODEL_COLUMNS["role_content"],
    BELIEF_VARIANCE,
    SAMPLE_COUNT,
    TRIAL_NUMBER,
)


def _in_model(model: str) -> str:
    """The cohort column that marks the events kept for ``model``."""
    return f"in_{model}"


COHORT_COLUMNS: tuple[str, ...] = (
    *SESSION_COLUMNS,
    KIND,
    UNIT,
    "visit_context",
    ARM,
    ROLE,
    *_SOURCES,
    *(_in_model(model) for model in MODEL_COLUMNS),
)
"""The columns of the cohort rows (:func:`role_content_cohort`)."""


def _product(covariate: str) -> str:
    """The column of a covariate times ``log(duration) - L0``."""
    return f"{covariate}_x_{LOG_DURATION_REL}"


# ---------------------------------------------------------------------------
# The cohort step
# ---------------------------------------------------------------------------


def assign_roles(
    candidate_arm: np.ndarray, current_arm: np.ndarray, previous_arm: np.ndarray
) -> np.ndarray:
    """Each candidate row's role (module docstring).

    Parameters
    ----------
    candidate_arm, current_arm, previous_arm : np.ndarray, shape (n_rows,)
        The row's arm, the current well's arm and the previous trial's well's
        arm (NaN when there is none).

    Returns
    -------
    np.ndarray of str, shape (n_rows,)
        One of :data:`ROLES`.
    """
    arm = np.asarray(candidate_arm, dtype=float)
    current = np.asarray(current_arm, dtype=float)
    previous = np.asarray(previous_arm, dtype=float)
    role = np.full(arm.shape, REFERENCE_ROLE, dtype=object)
    role[(arm == previous) & (previous != current)] = "previous"
    role[arm == current] = "current"
    role[np.isin(arm, TRUNK_ARMS)] = "trunk"  # a trunk arm is no well's arm
    return role


class RoleContentCohort(NamedTuple):
    """The cohort rows and where the others went.

    Attributes
    ----------
    rows : pd.DataFrame
        :data:`COHORT_COLUMNS`, nine rows per event, events contiguous: every
        event kept for at least one model, with ``in_<model>`` True for the
        events kept for that model. A value missing on a row that reads it is
        NaN; read a model's rows with :meth:`model_rows`.
    counts : dict
        ``visit_context``: events per label, kept or not
        (``count_visit_context``); ``by_model``: model -> kind ->
        ``events_selected``, ``events_dropped`` (a missing value in a column
        the model reads), ``events_missing`` (each column the model reads ->
        events with a missing value there), ``rows_dropped``, ``events_kept``,
        ``rows_kept``.
    reference_log_duration : float
        ``L0``: the median of the log event duration (s) over the events of
        both kinds in ``rows``.
    """

    rows: pd.DataFrame
    counts: dict
    reference_log_duration: float

    def model_rows(self, model: Model) -> pd.DataFrame:
        """The rows of the events kept for ``model``, finite in every column
        it reads."""
        if model not in MODELS:
            raise ValueError(f"model {model!r} not in {MODELS}")
        kept = self.rows[_in_model(model)].to_numpy(dtype=bool)
        return self.rows.loc[kept].reset_index(drop=True)


def _check_candidates(table: pd.DataFrame) -> None:
    """Nine rows per event, each arm of the track once; one current leaf arm;
    shares summing to one. A junction place bin is not an arm: a candidate
    arm outside the nine is refused."""
    arms = table[ARM].to_numpy(dtype=float)
    if not np.isin(arms, ALL_ARMS).all():
        bad = sorted(set(arms[~np.isin(arms, ALL_ARMS)].tolist()))
        raise ValueError(f"candidate arms outside the nine arms {ALL_ARMS}: {bad}")
    sizes = table.groupby(UNIT, sort=False)[ARM].agg(["size", "nunique"])
    if not (
        (sizes["size"] == len(ALL_ARMS)) & (sizes["nunique"] == len(ALL_ARMS))
    ).all():
        raise ValueError(
            "every event must have exactly one row for each of the nine arms"
        )
    current = table["occupancy_well_arm"].to_numpy(dtype=float)
    if not np.isin(current, LEAF_ARMS).all():
        raise ValueError(
            "the current arm must be the leaf arm of the visited well "
            f"(one of {LEAF_ARMS}) for every selected event"
        )
    total = table.groupby(UNIT, sort=False)[RESPONSE].sum()
    if not np.allclose(total.to_numpy(dtype=float), 1.0, atol=1e-6):
        raise ValueError("an event's shares over its nine arms must sum to one")


def role_content_cohort(
    event_arm_table: pd.DataFrame,
    settings: RoleContentSettings | None = None,
    *,
    contexts: str | Sequence[str] = STARTS_IN_WELL_VISIT,
) -> RoleContentCohort:
    """The cohort rows of the role content models, from the event-arm table.

    Parameters
    ----------
    event_arm_table : pd.DataFrame
        Nine rows per event (the canonical event-arm table, or the rows of
        some sessions of it).
    settings : RoleContentSettings, optional
        Read for the trial-number column and the event weight.
    contexts : str or sequence of str
        The ``visit_context`` labels kept (``select_visit_context``).

    Returns
    -------
    RoleContentCohort

    Raises
    ------
    ValueError
        If a column is missing, a candidate arm is not one of the nine, an
        event lacks one of its nine rows, a current arm is not a leaf arm, an
        event's shares do not sum to one, or an event weight is not positive
        or not the same on an event's nine rows.
    """
    settings = settings or RoleContentSettings()
    sources = dict(_SOURCES)
    sources[TRIAL_NUMBER] = (settings.trial_number, LEAF_ROLES)
    sources[EVENT_WEIGHT] = (_event_weight_column(settings.event_weight), ROLES)
    require_columns(
        event_arm_table,
        [
            *SESSION_COLUMNS,
            UNIT,
            ARM,
            "is_swr",
            "visit_context",
            "occupancy_well_arm",
            "prev_arm",
            *(source for source, _ in sources.values() if source is not None),
        ],
        "the event-arm table",
    )
    visit_counts = count_visit_context(event_arm_table, contexts, unit_col=UNIT)
    table = select_visit_context(event_arm_table, contexts)
    _check_candidates(table)
    table = table.sort_values([UNIT, ARM], kind="stable").reset_index(drop=True)
    rows = pd.DataFrame(
        {
            "animal": table["animal"].astype(str),
            "session_date": table["session_date"].astype(str),
            "epoch": table["epoch"].astype(str),
            KIND: np.where(table["is_swr"].astype(bool), dt.KINDS[0], dt.KINDS[1]),
            UNIT: table[UNIT].to_numpy(),
            "visit_context": table["visit_context"].to_numpy(),
            ARM: table[ARM].astype(int).to_numpy(),
            ROLE: assign_roles(
                table[ARM].to_numpy(),
                table["occupancy_well_arm"].to_numpy(),
                table["prev_arm"].to_numpy(),
            ),
        }
    )
    for column, (source, _) in sources.items():
        rows[column] = 1.0 if source is None else table[source].to_numpy(dtype=float)
    _check_event_weights(rows)
    rows[RECENCY] = np.log(np.clip(rows[RECENCY].to_numpy(), *_RECENCY_TRIALS))
    rows[LOG_DURATION] = np.log(rows[LOG_DURATION].to_numpy())
    # A value a row does not read is not missing: it is set to 0 (trunk rows
    # carry no leaf covariates; the role or arm terms absorb what is constant).
    # A missing value on a row that reads it stays NaN.
    missing = {}
    for column, (_, roles) in sources.items():
        reads = rows[ROLE].isin(roles).to_numpy()
        values = rows[column].to_numpy(dtype=float)
        missing[column] = reads & ~np.isfinite(values)
        rows[column] = np.where(reads, values, 0.0)
    event_missing = {
        column: pd.Series(mask)
        .groupby(rows[UNIT].to_numpy())
        .transform("any")
        .to_numpy()
        for column, mask in missing.items()
    }
    by_model = {}
    for model, columns in MODEL_COLUMNS.items():
        drop = np.zeros(len(rows), dtype=bool)
        for column in columns:
            drop |= event_missing[column]
        rows[_in_model(model)] = ~drop
        by_model[model] = {}
        for kind in dt.KINDS:
            of_kind = (rows[KIND] == kind).to_numpy()
            by_model[model][kind] = {
                "events_selected": int(rows.loc[of_kind, UNIT].nunique()),
                "events_dropped": int(rows.loc[of_kind & drop, UNIT].nunique()),
                "events_missing": {
                    column: int(
                        rows.loc[of_kind & event_missing[column], UNIT].nunique()
                    )
                    for column in columns
                },
                "rows_dropped": int((of_kind & drop).sum()),
                "events_kept": int(rows.loc[of_kind & ~drop, UNIT].nunique()),
                "rows_kept": int((of_kind & ~drop).sum()),
            }
    in_any = rows[[_in_model(model) for model in MODEL_COLUMNS]].any(axis=1)
    rows = rows.loc[in_any, list(COHORT_COLUMNS)].reset_index(drop=True)
    durations = rows.groupby(UNIT, sort=False)[LOG_DURATION].first()
    counts = {
        "visit_context": {
            label: {"n": int(entry.n), "kept": bool(entry.kept)}
            for label, entry in visit_counts.iterrows()
        },
        "by_model": by_model,
    }
    return RoleContentCohort(rows, counts, float(np.median(durations.to_numpy())))


def _check_event_weights(rows: pd.DataFrame) -> None:
    """An event weight is positive where it is finite (a missing one drops
    the event) and the same on an event's nine rows."""
    weight = rows[EVENT_WEIGHT].to_numpy(dtype=float)
    if np.any(weight[np.isfinite(weight)] <= 0):
        raise ValueError("an event weight must be positive")
    per_event = pd.Series(weight).groupby(rows[UNIT].to_numpy())
    if np.any((per_event.max() - per_event.min()).to_numpy() > 0):
        raise ValueError("an event weight must be the same on an event's nine rows")


def session_rows(rows: pd.DataFrame) -> dict[SessionKey, pd.DataFrame]:
    """The cohort rows split by session, ``(animal, session_date, epoch)`` ->
    rows, in sorted order."""
    return {
        tuple(str(v) for v in key): part.reset_index(drop=True)
        for key, part in rows.groupby(list(SESSION_COLUMNS), sort=True)
    }


# ---------------------------------------------------------------------------
# The design
# ---------------------------------------------------------------------------


class ContrastReference(NamedTuple):
    """The pooled quantities a fit fixes beyond its design, computed on its
    scale rows (:data:`SCALE_POOLINGS`).

    Attributes
    ----------
    reference_log_duration : float
        ``L0``.
    covariate_means : dict of str to float
        Each role slope's covariate -> its pooled mean over the leaf rows, the
        center of its product with ``log_duration_rel``.
    reward_fraction_mean, reward_fraction_sd : float
        The pooled mean and population SD of the reward fraction on the
        current rows (Q4a).
    """

    reference_log_duration: float
    covariate_means: dict
    reward_fraction_mean: float
    reward_fraction_sd: float

    @classmethod
    def from_rows(
        cls, pooled_rows: pd.DataFrame, reference_log_duration: float, model: Model
    ) -> ContrastReference:
        """The reference of the scale rows: the cohort rows of both kinds or
        of one (:data:`SCALE_POOLINGS`)."""
        leaf = pooled_rows[ROLE].isin(LEAF_ROLES)
        means = {
            covariate: float(pooled_rows.loc[leaf, covariate].mean())
            for covariate in _ROLE_SLOPES[model]
        }
        current = pooled_rows.loc[pooled_rows[ROLE] == "current", REWARD_FRACTION]
        values = current.to_numpy(dtype=float)
        return cls(
            float(reference_log_duration),
            means,
            float(np.mean(values)),
            float(np.std(values, ddof=0)),
        )


def design_rows(
    rows: pd.DataFrame, model: Model, variant: Variant, reference: ContrastReference
) -> pd.DataFrame:
    """The cohort rows with the columns of the length-adjusted variant.

    ``log_duration_rel`` is ``log(duration) - L0``; each role slope's product
    column is its covariate less its pooled mean, times ``log_duration_rel``
    (centered, so that it is not the role's length slope again). The
    unadjusted variant adds nothing.
    """
    _check_model(model, variant)
    if variant == "unadjusted":
        return rows
    out = rows.copy()
    out[LOG_DURATION_REL] = out[LOG_DURATION] - reference.reference_log_duration
    for covariate in _ROLE_SLOPES[model]:
        centered = out[covariate] - reference.covariate_means[covariate]
        out[_product(covariate)] = centered * out[LOG_DURATION_REL]
    return out


def _check_model(model: str, variant: str) -> None:
    if model not in MODELS:
        raise ValueError(f"model {model!r} not in {MODELS}")
    if variant not in VARIANTS:
        raise ValueError(f"variant {variant!r} not in {VARIANTS}")


def role_terms(
    model: Model, variant: Variant, settings: RoleContentSettings | None = None
) -> list:
    """The model's terms (module docstring), at the settings' penalties.

    Returns
    -------
    list of terms
        For :func:`regression.terms.build_design` with no intercept.
    """
    _check_model(model, variant)
    settings = settings or RoleContentSettings()
    weak = log_penalty_from_prior_sd(settings.prior_sd)
    role = Category(ROLE, ROLES, reference=REFERENCE_ROLE, penalty=weak)
    terms: list = [
        Category(ARM, ALL_ARMS, reference=ALL_ARMS[0], penalty="none"),
        role,
    ]
    for covariate, lo, hi, n_basis, roles in _SMOOTHS:
        terms.append(
            Smooth(
                covariate,
                lo,
                hi,
                n_basis,
                settings.smooth_log_penalties[covariate],
                by=role,
                levels=roles,
            )
        )
    terms.append(Slope(NEVER_VISITED, "none"))
    for covariate in _ROLE_SLOPES[model]:
        terms.append(Slope(covariate, weak, by=role, levels=LEAF_ROLES))
    if variant == "length_adjusted":
        terms.append(
            Slope(LOG_DURATION_REL, weak, by=role, levels=("current", "previous"))
        )
        terms.append(Slope(LOG_DURATION_REL, "none", by=role, levels=("trunk",)))
        for covariate in _ROLE_SLOPES[model]:
            terms.append(Slope(_product(covariate), weak, by=role, levels=LEAF_ROLES))
    return terms


def role_design(
    pooled_rows: pd.DataFrame,
    model: Model,
    variant: Variant,
    settings: RoleContentSettings | None = None,
    *,
    reference: ContrastReference,
) -> Design:
    """The model's design, built on the scale rows.

    Parameters
    ----------
    pooled_rows : pd.DataFrame
        The scale rows: the cohort rows of every session of the fit, of both
        kinds or of the kind fitted (:data:`SCALE_POOLINGS`).
    reference : ContrastReference
        Of the same rows (:meth:`ContrastReference.from_rows`).
    """
    rows = design_rows(pooled_rows, model, variant, reference)
    return build_design(role_terms(model, variant, settings), rows, intercept=False)


# ---------------------------------------------------------------------------
# The estimands as functionals of the design
# ---------------------------------------------------------------------------


def model_estimands(model: Model) -> tuple[dt.Estimand, ...]:
    """The declared estimands read from ``model``, in declared order."""
    return tuple(e for e in dt.ESTIMANDS if e.model == model)


def _reference_frame(
    design: Design, model: Model, reference: ContrastReference, role: str, n: int
) -> pd.DataFrame:
    """``n`` rows of one role on arm 0 (a leaf arm), every covariate at its
    reference: a standardized slope covariate at its pooled reference, a 0/1
    one at 0, a smooth's covariate at the lower end of its range, the length
    at ``L0``."""
    record = design.standardization_record()
    frame = {ARM: np.full(n, ALL_ARMS[0]), ROLE: np.full(n, role, dtype=object)}
    for covariate, lo, *_ in _SMOOTHS:
        frame[covariate] = np.full(n, lo)
    frame[NEVER_VISITED] = np.zeros(n)
    for covariate in _ROLE_SLOPES[model]:
        frame[covariate] = np.full(n, record[covariate]["reference"])
    frame[LOG_DURATION] = np.full(n, reference.reference_log_duration)
    return pd.DataFrame(frame)


def _design_at(
    design: Design,
    model: Model,
    variant: Variant,
    reference: ContrastReference,
    frame: pd.DataFrame,
) -> np.ndarray:
    """The design's rows at a reference frame, shape ``(n, n_coef)``."""
    rows = design_rows(frame, model, variant, reference)
    X, _ = design.session(rows)
    return X


def _role_contrast(
    design: Design,
    model: Model,
    variant: Variant,
    reference: ContrastReference,
    role: str,
) -> np.ndarray:
    """A role's row less an other-leaf row of the same arm, at the reference,
    with the smooth terms left out."""
    frame = pd.concat(
        [
            _reference_frame(design, model, reference, r, 1)
            for r in (role, REFERENCE_ROLE)
        ],
        ignore_index=True,
    )
    X = _design_at(design, model, variant, reference, frame)
    c = X[0] - X[1]
    for term in design.terms:
        if term.kind == "smooth":
            c[term.columns] = 0.0
    return c


def _slope_contrast(
    design: Design,
    model: Model,
    variant: Variant,
    reference: ContrastReference,
    covariate: str,
    role: str,
) -> np.ndarray:
    """The change of one row of ``role`` over one pooled SD of ``covariate``,
    about its reference, at ``L0``."""
    entry = design.standardization_record()[covariate]
    frame = _reference_frame(design, model, reference, role, 2)
    frame[covariate] = entry["reference"] + np.array([0.5, -0.5]) * entry["sd"]
    X = _design_at(design, model, variant, reference, frame)
    return X[0] - X[1]


def _curve_contrast(
    design: Design,
    model: Model,
    variant: Variant,
    reference: ContrastReference,
    contrast: dt.CurveContrast,
) -> np.ndarray:
    """A curve contrast on a smooth term's own columns."""
    term = design.by_name[contrast.term]
    covariate, role = term.covariate, term.level[1]
    lo, hi = next((s[1], s[2]) for s in _SMOOTHS if s[0] == covariate)

    def curve_rows(x: np.ndarray) -> np.ndarray:
        frame = _reference_frame(design, model, reference, role, x.size)
        frame[covariate] = x
        X = _design_at(design, model, variant, reference, frame)
        out = np.zeros_like(X)
        out[:, term.columns] = X[:, term.columns]
        return out

    return range_mean_contrast(
        curve_rows, contrast.first, contrast.second, term_range=(lo, hi)
    )


def estimand_contrasts(
    design: Design,
    model: Model,
    variant: Variant,
    reference: ContrastReference,
    settings: RoleContentSettings | None = None,
) -> dict[str, np.ndarray]:
    """Each declared estimand of ``model`` as ``c`` over the design's columns.

    Parameters
    ----------
    design : Design
        The model's design (:func:`role_design`).
    model, variant : str
    reference : ContrastReference
    settings : RoleContentSettings, optional
        Read for the reading of Q4a and the points of Q5a.

    Returns
    -------
    dict of str to np.ndarray, shape (n_coef,)
        Estimand id -> ``c``; the estimate is ``c' beta`` and its variance
        ``c' V c``.
    """
    settings = settings or RoleContentSettings()
    out = {}
    for estimand in model_estimands(model):
        contrast = estimand.contrast
        if estimand.estimand_id == "Q5a":
            first, second = (float(np.log(t)) for t in settings.q5a_trials)
            contrast = dt.CurveContrast(contrast.term, (first, first), (second, second))
        if isinstance(contrast, dt.SDChangeContrast):
            if settings.q4a_reading == "sd_change":
                contrast = contrast.at(
                    reference.reward_fraction_mean, reference.reward_fraction_sd
                )
            else:
                contrast = dt.CurveContrast(contrast.term, (1.0, 1.0), (0.0, 0.0))
        if isinstance(contrast, dt.CurveContrast):
            c = _curve_contrast(design, model, variant, reference, contrast)
        else:
            (name,) = contrast.plus
            if contrast.minus:
                raise ValueError(f"{estimand.estimand_id}: a difference is not built")
            term = design.by_name[name]
            if term.kind == "level":
                c = _role_contrast(design, model, variant, reference, term.level[1])
            else:
                c = _slope_contrast(
                    design, model, variant, reference, term.covariate, term.level[1]
                )
        out[estimand.estimand_id] = c
    return out


def tested_coefficients(
    design: Design, contrasts: Mapping[str, np.ndarray]
) -> list[str]:
    """The coefficients the declared tests read: those with a non-zero entry
    in a contrast, in column order."""
    used = np.zeros(len(design.coef_names), dtype=bool)
    for c in contrasts.values():
        used |= np.abs(c) > 0
    return [name for name, u in zip(design.coef_names, used, strict=True) if u]


# ---------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RoleContentFit:
    """One model, variant and kind fitted to a set of sessions.

    Attributes
    ----------
    model, variant, kind : str
    design : Design
        Built on the scale rows (:data:`SCALE_POOLINGS`).
    reference : ContrastReference
    contrasts : dict of str to np.ndarray
        :func:`estimand_contrasts`.
    records : list of FitRecord
        One per session, in sorted session order.
    tested_collinearity : dict
        Session key -> :func:`tested_coefficient_collinearity` of the simple
        tested coefficients (a role level, a slope; a smooth's is the
        record's concurvity of the term); empty for a session without a kept
        fit. The trunk level is never among them (module docstring).
    settings : RoleContentSettings
    """

    model: str
    variant: str
    kind: str
    design: Design
    reference: ContrastReference
    contrasts: dict
    records: list
    tested_collinearity: dict
    settings: RoleContentSettings

    @property
    def separable(self) -> dict[SessionKey, dict[str, bool]]:
        """The collinearity rule of these models: session key -> tested
        coefficient -> True when the other columns explain less than
        ``collinearity.MAX_EXPLAINED`` of it (``collinearity.separable`` of
        :attr:`tested_collinearity`, measured on the penalized problem). The
        record's collinearity of the term ``role`` is not read: it counts the
        trunk level, exactly collinear with the arm effects by construction."""
        return {key: separable(m) for key, m in self.tested_collinearity.items()}


def fit_role_content(
    sessions: Mapping[SessionKey, pd.DataFrame],
    *,
    model: Model,
    variant: Variant,
    kind: str,
    reference_log_duration: float,
    settings: RoleContentSettings | None = None,
    scale_rows: pd.DataFrame | None = None,
    table_sha256: Mapping[str, str] | None = None,
    cohort_counts: Mapping | None = None,
) -> RoleContentFit:
    """Fit every session of one kind (``sessions.fit_sessions``, grouped softmax).

    Parameters
    ----------
    sessions : mapping of (animal, session_date, epoch) to pd.DataFrame
        Each session's cohort rows of ``kind`` for ``model``
        (:meth:`RoleContentCohort.model_rows`, :func:`session_rows`).
    model, variant, kind : str
    reference_log_duration : float
        ``L0`` of the cohort.
    settings : RoleContentSettings, optional
    scale_rows : pd.DataFrame, optional
        The model's cohort rows of both kinds over the sessions of the fit
        (:meth:`RoleContentCohort.model_rows`). Required when
        ``settings.scale_pooling`` is ``both_kinds``, where the design's
        scales and reference points are computed on them; under ``per_kind``
        their rows of ``kind`` are used, or, when not given, the sessions'
        rows.
    table_sha256 : mapping, optional
        The tables the rows were read from -> their hashes; recorded.
    cohort_counts : mapping, optional
        The cohort step's counts (:attr:`RoleContentCohort.counts`), added to
        every record's ``cohort_counts`` under ``"cohort"``.

    Returns
    -------
    RoleContentFit

    Raises
    ------
    ValueError
        If a session holds rows of another kind or a missing value in a
        column the model reads, ``scale_rows`` is missing under
        ``both_kinds``, or a tested coefficient has no fixed penalty
        (``support.require_tested_penalized``).
    """
    settings = settings or RoleContentSettings()
    if kind not in dt.KINDS:
        raise ValueError(f"kind {kind!r} not in {dt.KINDS}")
    _check_model(model, variant)
    for key, rows in sessions.items():
        if not (rows[KIND] == kind).all():
            raise ValueError(f"session {'/'.join(key)} holds rows of another kind")
        values = rows[list(MODEL_COLUMNS[model])].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(
                f"session {'/'.join(key)} holds missing values in a column "
                f"{model} reads: give it RoleContentCohort.model_rows({model!r})"
            )
    if settings.scale_pooling == "both_kinds":
        if scale_rows is None:
            raise ValueError(
                "scale_pooling 'both_kinds' needs scale_rows: the model's cohort "
                "rows of both kinds"
            )
        pooled = scale_rows
    elif scale_rows is not None:
        pooled = scale_rows.loc[scale_rows[KIND] == kind]
    else:
        pooled = pd.concat(list(sessions.values()), ignore_index=True)
    reference = ContrastReference.from_rows(pooled, reference_log_duration, model)
    design = role_design(pooled, model, variant, settings, reference=reference)
    prepared = {
        key: design_rows(rows, model, variant, reference).assign(
            **{WEIGHTED_RESPONSE: rows[RESPONSE] * rows[EVENT_WEIGHT]}
        )
        for key, rows in sessions.items()
    }
    contrasts = estimand_contrasts(design, model, variant, reference, settings)
    require_tested_penalized(design, tested_coefficients(design, contrasts))
    records = fit_sessions(
        design,
        prepared,
        GroupedSoftmax(),
        model=f"{model}:{variant}",
        kind=kind,
        response=WEIGHTED_RESPONSE,
        unit=UNIT,
        table_sha256=dict(table_sha256 or {}),
    )
    if cohort_counts is not None:
        records = [_with_cohort(r, cohort_counts) for r in records]
    simple = [
        name
        for name in tested_coefficients(design, contrasts)
        if design.column_terms[design.coef_names.index(name)].kind != "smooth"
        and name != TRUNK_LEVEL
    ]
    collinearity = {
        key: tested_coefficient_collinearity(design, prepared[key], record, simple)
        for key, record in zip(prepared, records, strict=True)
    }
    return RoleContentFit(
        model,
        variant,
        kind,
        design,
        reference,
        contrasts,
        records,
        collinearity,
        settings,
    )


def fit_role_content_session(
    key: SessionKey,
    rows: pd.DataFrame,
    *,
    model: Model,
    variant: Variant,
    kind: str,
    reference_log_duration: float,
    settings: RoleContentSettings | None = None,
    scale_rows: pd.DataFrame | None = None,
) -> RoleContentFit:
    """:func:`fit_role_content` for one session: the design, its
    standardization and its constraints are built on that session's rows
    (``scale_rows``: its rows of both kinds, under ``both_kinds``)."""
    return fit_role_content(
        {key: rows},
        model=model,
        variant=variant,
        kind=kind,
        reference_log_duration=reference_log_duration,
        settings=settings,
        scale_rows=scale_rows,
    )


def _with_cohort(record: FitRecord, cohort_counts: Mapping) -> FitRecord:
    """The record with the cohort step's counts added under ``"cohort"``."""
    return replace(
        record, cohort_counts={**record.cohort_counts, "cohort": dict(cohort_counts)}
    )


def _fitted_rows(design: Design, rows: pd.DataFrame, record: FitRecord) -> pd.DataFrame:
    """The rows a session's kept fit used: sorted by event as the fitting loop
    sorts them, less the rows of each dropped level and the events left with
    no response."""
    rows = rows.sort_values(UNIT, kind="stable").reset_index(drop=True)
    built = design.build(rows)
    keep = ~built.missing
    for name in record.cohort_counts.get("rows_dropped", {}):
        keep &= ~built.level_rows[design.by_name[name].level]
    total = (
        pd.Series(np.where(keep, rows[WEIGHTED_RESPONSE], 0.0))
        .groupby(rows[UNIT])
        .transform("sum")
    )
    keep &= total.to_numpy() > 0
    return rows.loc[keep].reset_index(drop=True)


def tested_coefficient_collinearity(
    design: Design, rows: pd.DataFrame, record: FitRecord, tested: Sequence[str]
) -> dict[str, dict[str, float]]:
    """Each tested coefficient's collinearity with every other column, alone,
    on the information of the fit's penalized problem.

    The measure is ``collinearity.concurvity`` (``r2`` and ``vif`` for one
    column) on the design weighted by the information at the fit
    (``collinearity.weighted_design``), with the square root of each smooth
    term's declared penalty appended as rows: ``X~' X~`` is then the
    information plus the smooths' penalties. The weak penalties of the
    simple terms are left out, so that a coefficient fixed only by its own
    prior, or by another tested coefficient's, still fails.

    Why the smooths' penalties are in: a smooth's sum-to-zero constraint is
    computed on the scale rows, not on a session's rows. When a basis
    function of a role's curve is seen in the scale rows and not in a
    session's rows of the kind fitted, the curve's columns hold that role's
    indicator on the session's rows, and the likelihood alone cannot tell
    the role level from the curve's level (R^2 1). The curve's declared
    penalty fixes that direction (the curve beyond the session's values is
    its penalized extrapolation), and the fit recovers the level. A
    direction the penalty does not touch (a slope that is a straight line
    in a smooth's covariate, in the penalty's null space) still fails.

    Each tested coefficient is a term of its own and each model term's other
    columns are one term. Empty for a session without a kept fit.

    Parameters
    ----------
    design : Design
    rows : pd.DataFrame
        The session's rows as given to the fit (:func:`design_rows`, with
        the weighted response).
    record : FitRecord
        The session's record.
    tested : sequence of str
        Coefficient names.

    Raises
    ------
    ValueError
        If the rows rebuilt for the fit are not as many as the record's.
    """
    if record.status != CONVERGED or record.no_fit:
        return {}
    used = _fitted_rows(design, rows, record)
    if len(used) != record.n_rows:
        raise ValueError(
            f"rebuilt {len(used)} rows for the fit, the record has {record.n_rows}"
        )
    X, _ = design.session(used)
    beta = np.asarray(record.coef, dtype=float)
    active = np.isfinite(beta)
    index = np.cumsum(active) - 1  # a column's place among the active ones
    weighted = weighted_design(
        GroupedSoftmax(),
        X[:, active],
        used[WEIGHTED_RESPONSE].to_numpy(dtype=float),
        pd.factorize(used[UNIT])[0],
        X[:, active] @ beta[active],
    )
    augmented = np.vstack([weighted, _smooth_penalty_root(design, active, index)])
    names = list(design.coef_names)
    terms: dict[str, np.ndarray] = {}
    for term, columns in design.term_columns.items():
        columns = [j for j in columns if active[j]]
        alone = [j for j in columns if names[j] in tested]
        rest = [j for j in columns if names[j] not in tested]
        for j in alone:
            terms[names[j]] = index[[j]]
        if rest:
            terms[term] = index[rest]
    measured = concurvity(augmented, terms, beta[active], intercept=None)
    out = {}
    for name in tested:
        if name not in measured:
            continue
        r2 = measured[name]["estimate"]
        vif = np.inf if 1.0 - r2 <= RANK_TOL else 1.0 / (1.0 - r2)
        out[name] = {"r2": r2, "vif": float(np.nan if np.isnan(r2) else vif)}
        if "not_measured" in measured[name]:
            out[name]["not_measured"] = measured[name]["not_measured"]
    return out


def _smooth_penalty_root(
    design: Design, active: np.ndarray, index: np.ndarray
) -> np.ndarray:
    """Rows ``L`` with ``L' L`` the smooth terms' declared penalties over the
    active columns (each block at its fixed log penalty), shape
    ``(n_rows, n_active)``."""
    out = []
    for term in design.terms:
        if term.kind != "smooth" or term.block is None:
            continue
        block = design.blocks[term.block]
        values, vectors = np.linalg.eigh(
            np.exp(design.log_penalties[term.block]) * block.S
        )
        root = np.sqrt(np.clip(values, 0.0, None))[:, None] * vectors.T
        keep = active[block.columns]
        rows = np.zeros((root.shape[0], int(active.sum())))
        rows[:, index[block.columns[keep]]] = root[:, keep]
        out.append(rows)
    return np.vstack(out) if out else np.zeros((0, int(active.sum())))
