"""Tests for the nested bootstrap and the duration-matched contrast."""

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.hippocampus import property_contrast as pc


def _design(n_animals=5, n_sessions=8, n_events=40):
    animal = np.repeat(np.arange(n_animals), n_sessions * n_events)
    session = np.repeat(np.arange(n_animals * n_sessions), n_events)
    return animal, session


def _duration_world(rng, class_effect, n=20_000):
    """Events whose property depends on duration (steeply) and, optionally, on
    the class; the class is far more likely for long events."""
    log_duration = rng.normal(np.log(0.06), 0.6, n)
    p_swr = 1 / (1 + np.exp(-3 * (log_duration - np.log(0.07))))
    is_swr = rng.random(n) < p_swr
    y = 2.0 * log_duration + class_effect * is_swr + rng.normal(0, 0.3, n)
    return np.exp(log_duration), is_swr, y


def test_matched_difference_removes_a_pure_duration_effect():
    rng = np.random.default_rng(3)
    duration, is_swr, y = _duration_world(rng, class_effect=0.0)
    raw = np.median(y[is_swr]) - np.median(y[~is_swr])
    matched = pc.matched_difference(y, is_swr, pc.duration_strata(duration, 10))
    assert raw > 1.0  # the raw contrast is mostly duration
    assert abs(matched) < 0.1 * raw


def test_matched_difference_recovers_a_class_effect():
    rng = np.random.default_rng(4)
    duration, is_swr, y = _duration_world(rng, class_effect=0.5)
    matched = pc.matched_difference(y, is_swr, pc.duration_strata(duration, 10))
    assert matched == pytest.approx(0.5, abs=0.1)


def test_bootstrap_contrasts_and_intervals():
    rng = np.random.default_rng(5)
    animal, session = _design(n_events=60)
    duration, is_swr, y = _duration_world(rng, class_effect=0.5, n=animal.size)
    out = pc.bootstrap_contrasts(
        {"y": y, "dur": duration},
        is_swr,
        animal,
        session,
        duration,
        n_replicates=60,
        seed=7,
    )
    assert set(out.replicate) == set(range(-1, 60))
    data = out[out.replicate < 0].set_index("property")
    assert data.loc["y", "median_swr"] == pytest.approx(np.median(y[is_swr]))
    assert data.loc["y", "median_non_swr"] == pytest.approx(np.median(y[~is_swr]))
    intervals = pc.percentile_intervals(out)
    row = intervals.loc["y"]
    assert row.difference_low <= row.difference <= row.difference_high
    assert row.matched_difference_low < 0.5 < row.matched_difference_high
    # fixed seed: the same replicates again
    again = pc.bootstrap_contrasts(
        {"y": y, "dur": duration},
        is_swr,
        animal,
        session,
        duration,
        n_replicates=60,
        seed=7,
    )
    pd.testing.assert_frame_equal(out, again)


def test_a_session_in_two_animals_is_refused():
    with pytest.raises(ValueError, match="more than one animal"):
        pc.bootstrap_contrasts(
            {"y": np.arange(4.0)},
            np.array([True, False, True, False]),
            np.array([0, 0, 1, 1]),
            np.array([0, 0, 0, 1]),
            np.ones(4),
            n_replicates=1,
            seed=0,
        )


def test_events_with_the_same_number_of_samples_share_a_stratum():
    # 17 samples of 2 ms, written with float noise either side; a quantile edge
    # falls inside the tie.
    rng = np.random.default_rng(8)
    seventeen = 17 * 0.002 + rng.choice([-1e-12, 1e-12], size=50)
    durations = np.concatenate([np.full(50, 0.010), seventeen, np.full(50, 0.060)])
    strata = pc.duration_strata(durations, 4)
    assert np.unique(strata[50:100]).size == 1
    # the raw floats would have split the tie across two strata
    raw = pd.qcut(durations, 4, labels=False, duplicates="drop")
    assert np.unique(raw[50:100]).size == 2


def test_skipped_strata_and_nan_replicates_are_counted():
    rng = np.random.default_rng(9)
    animal, session = _design(n_events=60)
    duration, is_swr, y = _duration_world(rng, class_effect=0.5, n=animal.size)
    out = pc.bootstrap_contrasts(
        {"y": y}, is_swr, animal, session, duration, n_replicates=40, seed=3
    )
    assert (out.loc[out.replicate < 0, "matched_strata_skipped"] == 0).all()
    intervals = pc.percentile_intervals(out)
    row = intervals.loc["y"]
    assert row.n_replicates == 40
    assert row.matched_difference_nan_replicates == 0
    assert row.matched_replicates_skipping_strata == int(
        (out.loc[out.replicate >= 0, "matched_strata_skipped"] > 0).sum()
    )
    # a replicate with no estimate is counted, and too many of them raise
    out.loc[out.replicate == 0, "matched_difference"] = np.nan
    assert (
        pc.percentile_intervals(out, max_nan_share=0.05).loc[
            "y", "matched_difference_nan_replicates"
        ]
        == 1
    )
    with pytest.raises(ValueError, match="no estimate"):
        pc.percentile_intervals(out, max_nan_share=0.01)


def test_a_replicate_drawing_no_event_of_a_class_skips_the_stratum():
    values = np.arange(8.0)
    is_swr = np.array([True, False] * 4)
    strata = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    groups = pc._SortedGroups(values, is_swr, strata)
    weights = np.array([1, 1, 1, 1, 0, 1, 0, 1])  # no SWR event of stratum 1 drawn
    value, skipped = groups.matched(weights, [0, 1])
    assert skipped == 1
    assert value == pytest.approx(-1.0)  # stratum 0 alone: median 1 - median 2
