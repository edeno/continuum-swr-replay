"""Planted truths for the uncertainty model: can the measures be told from recency?

The belief variance is mostly a clock of how long an arm has gone unvisited,
and the sample count drifts with the trial number. These truths, each on the
real rows of a session (:mod:`.role_content_scenarios` for the machinery),
check that the uncertainty model (``role_content_uncertainty``) keeps each
measure's test at its size when the measure has no effect, and recovers a
planted one:

- ``recency_only``: a curved recency effect and no measure's effect;
- ``belief_variance``: the same recency and a belief-variance slope;
- ``sample_count``: the same recency and a sample-count slope;
- ``trial_drift``: the same recency and a drift of the other-leaf share with
  the trial number, no sample-count effect;
- ``reward_curvature``: the same recency and a U-shaped effect of the reward
  fraction on the other-leaf rows, which neither measure may take (the reason
  the reward variability was dropped).

All on the other-leaf rows, as the declared tests read them; arm effects as
:func:`.role_content_scenarios.arm_effects`.
"""

from __future__ import annotations

import numpy as np

from . import role_content as rc
from .role_content_scenarios import (
    Scenario,
    Truth,
    arm_effects,
    recency_curve,
)

__all__ = [
    "MEASURE_SIZE",
    "RECENCY_AMPLITUDE",
    "uncertainty_scenarios",
]

#: The planted size of a measure's slope, per pooled SD (the scratch study's).
MEASURE_SIZE = 0.25

#: The amplitude of the recency curve every truth here carries.
RECENCY_AMPLITUDE = 0.6


def _u_shape(amplitude: float):
    """A U-shaped reward-fraction curve, 0 at 0.5 and ``amplitude`` at the
    ends; its change over a pooled SD about a mean near 0.5 is near 0."""
    return lambda x: amplitude * 4.0 * (np.asarray(x) - 0.5) ** 2


def uncertainty_scenarios() -> list[Scenario]:
    """The uncertainty model's truths (module docstring), each fitted with the
    unadjusted ``role_content_uncertainty``."""
    other = rc.REFERENCE_ROLE
    recency = {(rc.RECENCY, other): recency_curve(RECENCY_AMPLITUDE)}
    size = (
        "the measure's declared test rejects in more than 7 % (or fewer than 3 %) "
        "of data sets, or its mean slope differs from 0 by more than its Monte "
        "Carlo SE times 3"
    )
    power = (
        "the planted slope's mean estimate differs from its planted value by more "
        "than 20 % of it (record the bias and the power at the planted size), or "
        "the other measure's test rejects in more than 7 %"
    )
    truths = {
        "recency_only": (
            Truth(arm=arm_effects(), curves=recency),
            ("Q5b-var", "Q5b-n", "Q5a"),
            size + "; or Q5a's mean differs from its planted value by more than 10 %",
        ),
        "belief_variance": (
            Truth(
                arm=arm_effects(),
                curves=recency,
                slopes={(rc.BELIEF_VARIANCE, other): MEASURE_SIZE},
            ),
            ("Q5b-var", "Q5b-n"),
            power,
        ),
        "sample_count": (
            Truth(
                arm=arm_effects(),
                curves=recency,
                slopes={(rc.SAMPLE_COUNT, other): MEASURE_SIZE},
            ),
            ("Q5b-n", "Q5b-var"),
            power,
        ),
        "trial_drift": (
            Truth(
                arm=arm_effects(),
                curves=recency,
                slopes={(rc.TRIAL_NUMBER, other): MEASURE_SIZE},
            ),
            ("Q5b-n", "Q5b-var"),
            size,
        ),
        "reward_curvature": (
            Truth(
                arm=arm_effects(),
                curves={**recency, (rc.REWARD_FRACTION, other): _u_shape(0.5)},
            ),
            ("Q5b-var", "Q5b-n"),
            size,
        ),
    }
    return [
        Scenario(
            name,
            "role_content_uncertainty",
            "unadjusted",
            validates,
            failure,
            (lambda key, truth=truth: truth),
        )
        for name, (truth, validates, failure) in truths.items()
    ]
