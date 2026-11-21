"""The declared family of tests: one list, written before any fit.

Each member is a scalar estimand of one model, a linear functional of the
model's coefficients (:mod:`.contrasts`), so its variance in a session is
exact. Each estimand is tested for SWR events, for No SWR events and for their
difference (the difference of the two kinds' session estimates, reduced like
any other estimand): :data:`DECLARED_TESTS` is every (estimand, tested) pair,
and the false discovery rate is controlled over all of them. The list is read
from here only; no other code holds its length.

A contrast is defined by its ranges or its coefficient names, never by a
basis, so it holds whatever basis and freedom a model's smooth terms are
given:

- :class:`CurveContrast`: the mean of a curve over one range less its mean
  over another, on the grid of :func:`.contrasts.range_grid` (a range whose
  ends are equal is one point);
- :class:`SDChangeContrast`: the curve at the pooled mean plus half a pooled
  standard deviation of its variable less the curve at the mean less half,
  the mean and SD taken on the stated role's rows (a change per pooled SD);
- :class:`CoefficientContrast`: a sum of named coefficients less another sum.

Names. A smooth term is named by its variable, with ``:<role>`` when it is
one smooth of a role (``time_until_visit_end_s:current``). A slope by role is
``<covariate>:<role>``. A level of a category is ``<category>[<level>]``; the
role category's reference level is ``other_leaf``, so a role's contrast
against another leaf arm is its own coefficient. The rate models' clocks are
``time_since_visit_start_s`` (since the first poke-in) and
``time_until_visit_end_s`` (until the final poke-out); recency is
``log_trials_since_last_visit``.

Models: ``visit_rate_A`` (the visit rate with the smooths of both clocks and
the last outcome, 0/1); ``role_content`` (content on nine arms by arm role:
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
    "DIFFERENCE",
    "ESTIMANDS",
    "KINDS",
    "TESTED",
    "CoefficientContrast",
    "CurveContrast",
    "Estimand",
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
        """The contrast at a pooled mean and SD, as two one-point ranges."""
        high, low = mean + sd / 2, mean - sd / 2
        return CurveContrast(self.term, (high, high), (low, low))


@dataclass(frozen=True)
class CoefficientContrast:
    """The sum of the ``plus`` coefficients less the sum of the ``minus`` ones."""

    plus: tuple[str, ...]
    minus: tuple[str, ...] = ()


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
    contrast : CurveContrast or SDChangeContrast or CoefficientContrast
        Its definition on the model's coefficients.
    scale : {"log rate ratio", "log share ratio"}
        What the contrast is on the model's link scale.
    """

    estimand_id: str
    question: str
    description: str
    model: str
    contrast: CurveContrast | SDChangeContrast | CoefficientContrast
    scale: Literal["log rate ratio", "log share ratio"]


_ARRIVAL = "time_since_visit_start_s"
_LEAVING = "time_until_visit_end_s"
_RECENCY = "log_trials_since_last_visit"

ESTIMANDS: tuple[Estimand, ...] = (
    Estimand(
        "Q1",
        "Does the rate of event starts change over the time at a well?",
        "log rate 2 to 4 s after arrival minus log rate in the first second",
        "visit_rate_A",
        CurveContrast(_ARRIVAL, (2.0, 4.0), (0.0, 1.0)),
        "log rate ratio",
    ),
    Estimand(
        "Q2",
        "Does the outcome just received change the rate?",
        "log rate ratio, rewarded against omitted last outcome, at fixed time "
        "before leaving",
        "visit_rate_A",
        CoefficientContrast(plus=("last_outcome_rewarded",)),
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
        SDChangeContrast("reward_fraction:current", "reward_fraction", "current"),
        "log share ratio",
    ),
    Estimand(
        "Q4b",
        "Does content follow another arm's value?",
        "slope of another leaf arm's log share on its value, per pooled SD",
        "role_content",
        CoefficientContrast(plus=("value:other_leaf",)),
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
        CoefficientContrast(plus=("belief_variance:other_leaf",)),
        "log share ratio",
    ),
    Estimand(
        "Q5b-n",
        "Does how often an arm was sampled add to recency?",
        "slope of another leaf arm's log share on its sample count (log of one "
        "plus the visits to its well with a known outcome earlier in the epoch), "
        "per pooled SD, in the same model",
        "role_content_uncertainty",
        CoefficientContrast(plus=("sample_count:other_leaf",)),
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
        CurveContrast(f"{_LEAVING}:current", (0.0, 0.25), (1.0, 2.0)),
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
