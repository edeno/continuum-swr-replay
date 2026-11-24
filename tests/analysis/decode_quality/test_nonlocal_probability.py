"""The immobile time's spread over non-local probability, on tables small enough
to check by hand."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.decode_quality import nonlocal_probability as nlp


def _frame(probabilities, speeds=None, excluded=None) -> pd.DataFrame:
    n = len(probabilities)
    p = np.asarray(probabilities, dtype=float)
    return pd.DataFrame(
        {
            "Non-Local_Continuous_prob": p / 2,
            "Non-Local_Fragmented_prob": p / 2,
            "head_speed": np.zeros(n) if speeds is None else speeds,
            "in_excluded_time": np.zeros(n, bool) if excluded is None else excluded,
        }
    )


def test_counts_hold_the_immobile_analyzed_bins_inside_a_visit_in_closed_bins():
    """0 and 0.02 fall in the first bin, 0.8 in the bin up to 0.8 and 0.81 in
    the next; a moving bin, a bin in excluded time and a bin outside a visit
    are not counted, nor is a missing probability."""
    frame = _frame(
        [0.0, 0.02, 0.8, 0.81, 1.0, 0.5, 0.5, 0.5, np.nan],
        speeds=[0, 0, 0, 0, 0, 10.0, 0, 0, 0],
        excluded=[False] * 6 + [True, False, False],
    )
    in_visit = np.array([True] * 7 + [False, True])
    counts = nlp.probability_counts(frame, in_visit)
    assert len(counts) == 50 and counts["n_bins"].sum() == 5
    by_low = counts.set_index("bin_low")["n_bins"]
    assert by_low[0.0] == 2
    assert by_low.loc[np.isclose(by_low.index, 0.78)].item() == 1
    assert by_low.loc[np.isclose(by_low.index, 0.8)].item() == 1
    assert by_low.loc[np.isclose(by_low.index, 0.98)].item() == 1
    with pytest.raises(ValueError, match="one bool per bin"):
        nlp.probability_counts(frame, in_visit[:-1])


def _stored(epochs: dict) -> pd.DataFrame:
    """Rows as stored, from ``{(animal, epoch): {bin_low: n}}``."""
    rows = []
    for (animal, epoch), counts in epochs.items():
        for low, high in zip(
            nlp.PROBABILITY_BIN_EDGES[:-1], nlp.PROBABILITY_BIN_EDGES[1:], strict=True
        ):
            n = next((v for k, v in counts.items() if np.isclose(k, low)), 0)
            rows.append(
                {"animal": animal, "session_date": 1, "epoch": epoch,
                 "bin_low": low, "bin_high": high, "n_bins": n}
            )  # fmt: skip
    return pd.DataFrame(rows)


def test_each_animal_is_the_mean_over_its_epochs_then_the_animals():
    """Animal a: one epoch all low (share 1, 0, 0), one half low, half high
    (0.5, 0, 0.5): its mean (0.75, 0, 0.25). Animal b: one epoch, a quarter in
    the middle and three quarters high (0, 0.25, 0.75). The mean over animals
    weighs each animal once."""
    counts = _stored(
        {
            ("a", "1"): {0.0: 10},
            ("a", "2"): {0.1: 5, 0.9: 5},
            ("b", "1"): {0.5: 1, 0.98: 3},
            ("c", "1"): {},  # no time: left out
        }
    )
    shares = nlp.probability_share_statistics(counts)
    assert list(shares.index) == ["low", "middle", "high"]
    np.testing.assert_allclose(shares["animal_mean"], [0.375, 0.125, 0.5])
    np.testing.assert_allclose(shares["animal_min"], [0.0, 0.0, 0.25])
    np.testing.assert_allclose(shares["animal_max"], [0.75, 0.25, 0.75])
    assert set(shares["n_animals"]) == {2} and set(shares["n_epochs"]) == {3}
    distribution = nlp.animal_probability_distribution(counts)
    a = distribution[distribution["animal"] == "a"].set_index("bin_low")["percent"]
    assert a[0.0] == pytest.approx(50.0)
    assert a.loc[np.isclose(a.index, 0.9)].item() == pytest.approx(25.0)
    assert a.sum() == pytest.approx(100.0)
    assert set(distribution["animal"]) == {"a", "b"}
