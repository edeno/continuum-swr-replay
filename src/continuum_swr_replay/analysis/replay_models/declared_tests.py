"""The declared family of tests: one list, written before any fit.

Each member is a scalar estimand of one model, a linear functional of the
model's coefficients
(:mod:`continuum_swr_replay.analysis.regression.contrasts`), so its variance in a fit is
exact. Each estimand is tested for SWR events, for No SWR events and for their
difference (the difference of the two kinds' unit estimates, reduced like
any other estimand): :data:`DECLARED_TESTS` is every (estimand, tested) pair,
and the false discovery rate is controlled over all of them. The list is read
from here only; no other code holds its length.

A contrast is defined by its ranges or its coefficient names, never by a
basis, so it holds whatever basis and freedom a model's smooth terms are
given:

- :class:`CurveContrast`: the mean of a curve over one range less its mean
  over another, on the grid of :func:`continuum_swr_replay.analysis.regression.contrasts.range_grid` (a range whose
  ends are equal is one point);
- :class:`SDChangeContrast`: the curve at the pooled mean plus half a pooled
  standard deviation of its variable less the curve at the mean less half,
  the mean and SD taken on the stated role's rows (a change per pooled SD);
- :class:`CoefficientContrast`: a sum of named coefficients less another sum;
- :class:`OutcomeWindowContrast`: a coefficient plus the mean over a window
  of one curve less another, on the same grid (the rate model's reward
  effect read from an arrival curve for each outcome).

Names. Terms and coefficients are named as
:mod:`continuum_swr_replay.analysis.regression.terms` names a design's (the rule is
stated there once): the role category is the column ``role``, whose reference
level is ``other_leaf``, so a role's contrast against another leaf arm is its
own coefficient (``role[current]``); a smooth or a slope fitted for one role
carries ``:role=<level>``. The rate models' clocks are
``time_since_visit_start_s`` (since the first poke-in) and
``time_until_visit_end_s`` (until the final poke-out); recency is
``log_trials_since_last_visit``, a smooth on the other-leaf rows only (the
current arm is always 1 trial since its last visit and the previous arm 2, so
on their rows a recency curve is their role level); the last outcome is
``last_outcome_rewarded`` (0/1), and the arrival curve after each outcome is
``s(time_since_visit_start_s):last_outcome_rewarded=<0 or 1>``.

Models: ``visit_rate_A`` (the visit rate, its declared form: an arrival curve
for each last outcome, the shared leaving curve and the outcome coefficient,
0/1; the convention's twenty-sixth round); ``role_content`` (content on nine arms by arm role:
role category, reward-fraction smooth per role, value slope per role, recency
smooth, the current role's smooth of time until leaving); ``role_content_uncertainty``
(the same plus, per leaf role, the belief variance, the sample count and the
trial number in the epoch); ``role_content_next`` (``role_content`` with a
``next`` role for the next trial's arm when it is neither current nor
previous).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

__all__ = [
    "DECLARED_TESTS",
    "BASELINES",
    "PER_DAY",
    "DIFFERENCE",
    "ESTIMANDS",
    "KINDS",
    "ONSET_WINDOW_S",
    "Q2_WINDOW_S",
    "TESTED",
    "CoefficientContrast",
    "CurveContrast",
    "Estimand",
    "OutcomeWindowContrast",
    "SDChangeContrast",
    "estimand",
]

KINDS = ("SWR", "No SWR")
"""The two kinds of non-local event, each fit on its own."""

DIFFERENCE = "SWR minus No SWR"
"""The kind difference of an estimand."""

TESTED = (*KINDS, DIFFERENCE)
"""What each estimand is tested for."""


@dataclass(frozen=True)
class CurveContrast:
    """The mean of a smooth over ``first`` less its mean over ``second``.

    Attributes
    ----------
    term : str
        The smooth term.
    first, second : tuple of float
        The ranges ``(lo, hi)`` in the term's variable; equal ends are a point.
    """

    term: str
    first: tuple[float, float]
    second: tuple[float, float]


@dataclass(frozen=True)
class SDChangeContrast:
    """A smooth's change over one pooled SD of its variable, about its mean.

    ``f(m + sd / 2) - f(m - sd / 2)``, ``m`` and ``sd`` the pooled mean and
    population SD (``ddof = 0``) of ``variable`` on the rows of ``role``, taken
    once from the cohort rows of the model and kind.

    Attributes
    ----------
    term : str
        The smooth term.
    variable : str
        Its variable.
    role : str
        The role whose rows give ``m`` and ``sd``.
    """

    term: str
    variable: str
    role: str

    def at(self, mean: float, sd: float) -> CurveContrast:
        """The contrast at a pooled mean and SD, as two one-point ranges.

        Like every curve contrast, its points must lie inside the term's
        range: ``contrasts.range_mean_contrast`` refuses them otherwise.
        """
        high, low = mean + sd / 2, mean - sd / 2
        return CurveContrast(self.term, (high, high), (low, low))


@dataclass(frozen=True)
class CoefficientContrast:
    """The sum of the ``plus`` coefficients less the sum of the ``minus`` ones."""

    plus: tuple[str, ...]
    minus: tuple[str, ...] = ()


@dataclass(frozen=True)
class OutcomeWindowContrast:
    """A coefficient plus the mean over ``window`` of one curve less another.

    For the rate model: the outcome coefficient plus the mean over the window
    of the arrival curve after a reward less the one after an omission, the
    log rate ratio after a reward against after an omission at the same time
    since arrival (and before leaving), averaged over the window on the grid
    of ``contrasts.range_grid``. The window must lie inside both curves'
    range.

    Attributes
    ----------
    coefficient : str
        The outcome coefficient.
    plus_term, minus_term : str
        The two smooth terms, of the same variable.
    window : tuple of float
        ``(lo, hi)`` in that variable (s since arrival).
    """

    coefficient: str
    plus_term: str
    minus_term: str
    window: tuple[float, float]


@dataclass(frozen=True)
class Estimand:
    """One member of the declared family.

    Attributes
    ----------
    estimand_id : str
        Its id (``Q1``, ``Q5b-var``).
    question : str
        The question it answers.
    description : str
        The estimand, in words.
    model : str
        The model it is read from.
    contrast : CurveContrast, SDChangeContrast, CoefficientContrast or OutcomeWindowContrast
        Its definition on the model's coefficients.
    scale : {"log rate ratio", "log share ratio"}
        What the contrast is on the model's link scale.
    """

    estimand_id: str
    question: str
    description: str
    model: str
    contrast: (
        CurveContrast | SDChangeContrast | CoefficientContrast | OutcomeWindowContrast
    )
    scale: Literal["log rate ratio", "log share ratio"]


BASELINES: dict[str, tuple[str, ...]] = {
    "visit_rate_A": ("(Intercept)",),
    "visit_rate_D": ("(Intercept)",),
    "role_content": ("candidate_arm",),
    "role_content_uncertainty": ("candidate_arm",),
    "role_content_next": ("candidate_arm",),
}
"""Each declared model's baseline terms, copied for each session in a fit of
several sessions (``regression.terms``, "Baselines"): the rate models'
intercept (model D's too, which has no declared test), the content models'
arm category. No declared contrast reads one."""

PER_DAY: dict[str, tuple[str, ...]] = {
    "visit_rate_A": (),
    "visit_rate_D": (),
    "role_content": (),
    "role_content_uncertainty": (),
    "role_content_next": (),
}
"""Each declared model's tested categories whose levels are estimated for each
recording day in a fit of several days (``regression.terms``, "Day copies";
passed to ``build_design(..., per_day=...)``), beside :data:`BASELINES`. The
rule is the convention's, the same for every family: in a fit of several
recording days, each tested category that this declaration names for the model
has one copy of each coded level per day, each with the level's declared weak
penalty; every other tested term and every curve is shared, and the baselines
stay per session. A declared contrast reads such a level as the plain mean of
its active day copies (``Design.contrast_on``); a copy without support drops
that level's rows that day and is left out of the mean, and the copies left
out are counted. In a fit of one day the level is the level itself. No family
has copy logic or a mean over days of its own.

Why: the lead decided R1 for the content model's animal fit ("Agree on r1.",
2026-11-27 02:19 PST: the role levels estimated per recording day, the curves
and slopes shared in the animal, the animal's value the mean of the per-day
levels), after a simulation showed that a level that changes from day to day,
forced to one value per animal, leaks into the shared curves; and added
"Remember that we want to have consistency between the rate, contract, etc
models" (2026-11-27 02:27 PST; "contract" is the content model), so the rule
is declared here for every model and implemented once in the fitter.

Every entry is empty for now: the content models adopt ``("role",)`` in a
later change, and the rate models wait for their own simulation. Only the
coded levels of a penalized category can be named, not a slope: how a tested
0/1 term (the rate model's last outcome) enters the rule is not decided (the
coordinator's call, 2026-11-27, not the lead's)."""

_OMITTED_ARRIVAL = "s(time_since_visit_start_s):last_outcome_rewarded=0"
_REWARDED_ARRIVAL = "s(time_since_visit_start_s):last_outcome_rewarded=1"
_OUTCOME = "last_outcome_rewarded"

Q2_WINDOW_S: tuple[float, float] = (4.0, 10.0)
"""Q2's window, s since arrival (the twenty-sixth round): the reward effect on
the rate once events are going. Declared before any real arrival curve by
outcome was looked at; never moved to fit the data."""

ONSET_WINDOW_S: tuple[float, float] = (2.0, 4.0)
"""The onset test's window, s since arrival (the twenty-sixth round): the
reward effect on when events start. Declared as :data:`Q2_WINDOW_S` is."""
_LEAVING = "s(time_until_visit_end_s)"
_RECENCY = "s(log_trials_since_last_visit):role=other_leaf"

ESTIMANDS: tuple[Estimand, ...] = (
    Estimand(
        "Q1",
        "Does the rate of event starts change over the time at a well?",
        "log rate 2 to 4 s after arrival minus log rate in the first second, "
        "read from the arrival curve after an omission",
        "visit_rate_A",
        CurveContrast(_OMITTED_ARRIVAL, (2.0, 4.0), (0.0, 1.0)),
        "log rate ratio",
    ),
    Estimand(
        "Q2",
        "Does the outcome just received change the rate once events are going?",
        "log rate ratio, rewarded against omitted last outcome, at the same time "
        "since arrival and before leaving, averaged over 4 to 10 s after arrival, "
        "among visits that last into the window",
        "visit_rate_A",
        OutcomeWindowContrast(
            _OUTCOME, _REWARDED_ARRIVAL, _OMITTED_ARRIVAL, Q2_WINDOW_S
        ),
        "log rate ratio",
    ),
    Estimand(
        "Q2-onset",
        "Does the outcome just received change when events start?",
        "log rate ratio, rewarded against omitted last outcome, at the same time "
        "since arrival and before leaving, averaged over 2 to 4 s after arrival, "
        "among visits that last into the window",
        "visit_rate_A",
        OutcomeWindowContrast(
            _OUTCOME, _REWARDED_ARRIVAL, _OMITTED_ARRIVAL, ONSET_WINDOW_S
        ),
        "log rate ratio",
    ),
    Estimand(
        "Q3a",
        "Is content biased to where the animal is?",
        "log share ratio, current arm against another leaf arm",
        "role_content",
        CoefficientContrast(plus=("role[current]",)),
        "log share ratio",
    ),
    Estimand(
        "Q3b",
        "Is content biased to where the animal came from?",
        "log share ratio, previous arm against another leaf arm",
        "role_content",
        CoefficientContrast(plus=("role[previous]",)),
        "log share ratio",
    ),
    Estimand(
        "Q4a",
        "Does content follow the current arm's reward history?",
        "change of the current arm's log share over one pooled SD of its reward "
        "fraction, read from the current role's reward-fraction smooth",
        "role_content",
        SDChangeContrast(
            "s(reward_fraction):role=current", "reward_fraction", "current"
        ),
        "log share ratio",
    ),
    Estimand(
        "Q4b",
        "Does content follow another arm's value?",
        "slope of another leaf arm's log share on its value, per pooled SD",
        "role_content",
        CoefficientContrast(plus=("value:role=other_leaf",)),
        "log share ratio",
    ),
    Estimand(
        "Q5a",
        "Does content follow how long an arm has gone unvisited?",
        "log share ratio of an arm 20 trials since its last visit against one 3 "
        "trials since, read from the recency smooth of the model that holds the "
        "uncertainty measures",
        "role_content_uncertainty",
        CurveContrast(
            _RECENCY, (np.log(20.0), np.log(20.0)), (np.log(3.0), np.log(3.0))
        ),
        "log share ratio",
    ),
    Estimand(
        "Q5b-var",
        "Does uncertainty about an arm add to recency?",
        "slope of another leaf arm's log share on its belief variance, per pooled "
        "SD, with recency, the reward fraction, the trial number and the sample "
        "count in the model",
        "role_content_uncertainty",
        CoefficientContrast(plus=("belief_variance:role=other_leaf",)),
        "log share ratio",
    ),
    Estimand(
        "Q5b-n",
        "Does how often an arm was sampled add to recency?",
        "slope of another leaf arm's log share on its sample count (log of one "
        "plus the visits to its well with a known outcome earlier in the epoch), "
        "per pooled SD, in the same model",
        "role_content_uncertainty",
        CoefficientContrast(plus=("sample_count:role=other_leaf",)),
        "log share ratio",
    ),
    Estimand(
        "Q6",
        "Does content relate to where the animal goes next?",
        "log share ratio, next arm against an arm that is neither current nor previous",
        "role_content_next",
        CoefficientContrast(plus=("role[next]",)),
        "log share ratio",
    ),
    Estimand(
        "Q7a",
        "How does the rate change as the animal gets ready to leave the well?",
        "log rate in the last 0.25 s before the final poke-out minus log rate 1 "
        "to 2 s before",
        "visit_rate_A",
        CurveContrast(_LEAVING, (0.0, 0.25), (1.0, 2.0)),
        "log rate ratio",
    ),
    Estimand(
        "Q7b",
        "How does content change as the animal gets ready to leave the well?",
        "the current arm's log share in the last 0.25 s before the final poke-out "
        "minus 1 to 2 s before, from the current role's smooth of time until "
        "leaving",
        "role_content",
        CurveContrast(f"{_LEAVING}:role=current", (0.0, 0.25), (1.0, 2.0)),
        "log share ratio",
    ),
)
"""The declared estimands, in the order of the questions."""

DECLARED_TESTS: tuple[tuple[str, str], ...] = tuple(
    (e.estimand_id, tested) for e in ESTIMANDS for tested in TESTED
)
"""Every test of the family: (estimand id, what it is tested for)."""


def estimand(estimand_id: str) -> Estimand:
    """The declared estimand of an id.

    Raises
    ------
    KeyError
        If no estimand has that id.
    """
    for e in ESTIMANDS:
        if e.estimand_id == estimand_id:
            return e
    raise KeyError(f"{estimand_id} is not a declared estimand")
