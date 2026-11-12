"""What the parts of the mpfc summary share: the tags of its values, the
words its keys use for the animals and the response classes, the timing
bootstrap's draws and seed, and the adder that leaves out a statistic over an
empty set."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from continuum_swr_replay.reported.values import (
    Tags,
    Value,
)

TAGS = {
    "modulation": Tags(("4.1",)),
    "figure_4d": Tags(("4.1",), ("4d",)),
    "units": Tags(("4.1",), ("4c", "4d")),
}
"""Each group of values: every mPFC value answers question 4.1; the groups
differ in the panels of Figure 4 that show them."""


CLASS_LABELS = {
    "SWR": "SWR",
    "non_local": "all non-local events",
    "non_local_no_SWR": "non-SWR replay",
    "non_local_SWR": "SWR-coupled replay",
    "engagement": (
        "the engagement test (matched_control + non_local_no_SWR vs. matched_control)"
    ),
}

#: Animals by the word their keys use (a key holds no digits).
ANIMAL_WORDS = {
    "chimi": "chimi",
    "j16": "jsixteen",
    "peanut": "peanut",
    "senor": "senor",
    "wilbur": "wilbur",
}


#: Nested-bootstrap replicates behind the population timing intervals, and their
#: seed.
TIMING_BOOTSTRAP_DRAWS = 1_000
TIMING_BOOTSTRAP_SEED = 0
#: What those replicates resample.
TIMING_RESAMPLES = (
    "nested resamples (the animals, then each drawn animal's recording days, then "
    "each drawn day's units)"
)
#: An animal's own mean curve, in the timing values.
OWN_CURVE = "the mean over its recording days of each day's sign-aligned mean curve"


def _finite_adder(
    values: list[Value], make: Callable[..., Value]
) -> Callable[..., None]:
    """An ``add(key, value, ...)`` that appends ``make(key, value, ...)`` to
    ``values`` when the value is finite. A statistic over an empty set is
    absent, never invented."""

    def add(key: str, value: float, *args: Any, **kwargs: Any) -> None:
        if np.isfinite(float(value)):
            values.append(make(key, value, *args, **kwargs))

    return add
