"""Hand-made per-session fits the behavior summary's builder tests share."""

from __future__ import annotations

import pandas as pd

ANIMALS = ("a", "a", "b", "c", "d", "e")
"""Six sessions of five animals: the median over sessions is not the median
over the animals' own medians."""


def coefficient_fits(term: str, coefficients, se: float) -> pd.DataFrame:
    """One fit per coefficient of ``term``, in the six sessions of five animals."""
    return pd.DataFrame(
        {
            "animal": list(ANIMALS),
            "session_date": list(range(len(ANIMALS))),
            "epoch": "02_r1",
            f"coef_{term}": coefficients,
            f"sd_{term}": 0.1,
            f"bse_{term}": se,
            "separated_terms": "",
        }
    )


def values_by_key(values: list) -> dict[str, float]:
    return {value.key: value.value for value in values}
