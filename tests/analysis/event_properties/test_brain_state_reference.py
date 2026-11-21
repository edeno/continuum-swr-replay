"""brain_state_reference: running and quiet-rest levels of a decode frame's z-scores."""

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.event_properties.brain_state_reference import (
    brain_state_reference,
    matched_window_means,
    regime_masks,
)


def frame() -> pd.DataFrame:
    # 10 running bins (theta 2), 10 quiet rest bins (theta -1), 5 rest bins in an
    # event (theta 5), 5 rest bins in excluded time (theta 9), 5 bins at 6 cm/s
    speed = np.r_[np.full(10, 20.0), np.full(20, 1.0), np.full(5, 6.0)]
    theta = np.r_[
        np.full(10, 2.0),
        np.full(10, -1.0),
        np.full(5, 5.0),
        np.full(5, 9.0),
        np.zeros(5),
    ]
    event = np.r_[np.zeros(20), np.ones(5), np.zeros(10)].astype(int)
    excluded = np.r_[np.zeros(25), np.ones(5), np.zeros(5)].astype(bool)
    return pd.DataFrame(
        {
            "head_speed": speed,
            "in_excluded_time": excluded,
            "non_local_event": event,
            "SWR_event": np.zeros(35, int),
            "MUA_event": np.zeros(35, int),
            "theta_power_zscore": theta,
            "SWR_zscore": -theta,
            "MUA_zscore": np.zeros(35),
        }
    )


def everywhere(dot: pd.DataFrame) -> np.ndarray:
    """Every bin inside a well visit."""
    return np.ones(len(dot), dtype=bool)


def test_running_and_rest_exclude_events_excluded_time_and_middle_speeds():
    ref = brain_state_reference(frame(), everywhere(frame()))
    assert ref["n_running_bins"] == 10
    assert ref["n_rest_bins"] == 10
    assert ref["theta_running_median"] == 2.0
    assert ref["theta_rest_median"] == -1.0
    assert ref["ripple_rest_mean"] == 1.0


def test_a_regime_with_no_bin_reads_nan():
    dot = frame().assign(head_speed=1.0)
    ref = brain_state_reference(dot, everywhere(dot))
    assert ref["n_running_bins"] == 0
    assert np.isnan(ref["theta_running_median"])


def test_quiet_rest_is_the_rest_inside_a_well_visit():
    # the visits hold the first 15 bins: 10 running (kept, running is the
    # epoch's wherever it is) and 5 of the 10 quiet-rest bins (theta -1)
    dot = frame()
    dot.loc[15:19, "theta_power_zscore"] = 3.0  # rest outside every visit
    in_visit = np.arange(len(dot)) < 15
    masks = regime_masks(dot, in_visit)
    np.testing.assert_array_equal(np.flatnonzero(masks["running"]), np.arange(10))
    np.testing.assert_array_equal(np.flatnonzero(masks["rest"]), np.arange(10, 15))
    ref = brain_state_reference(dot, in_visit)
    assert ref["n_rest_bins"] == 5
    assert ref["n_rest_bins_outside_visit"] == 5
    assert ref["theta_rest_median"] == -1.0
    assert ref["n_running_bins"] == 10


def test_the_visit_mask_must_be_one_bool_per_bin():
    with pytest.raises(ValueError, match="one bool per bin"):
        regime_masks(frame(), np.ones(3, dtype=bool))
    with pytest.raises(ValueError, match="one bool per bin"):
        regime_masks(frame(), np.ones(len(frame())))


def test_regime_masks():
    masks = regime_masks(frame(), everywhere(frame()))
    np.testing.assert_array_equal(np.flatnonzero(masks["running"]), np.arange(10))
    np.testing.assert_array_equal(np.flatnonzero(masks["rest"]), np.arange(10, 20))


def test_matched_windows_lie_in_eligible_runs_and_average_their_bins():
    # two eligible runs: bins 2-5 (values 2..5) and 10-11 (values 10, 11)
    values = np.arange(14, dtype=float)
    eligible = np.zeros(14, dtype=bool)
    eligible[2:6] = True
    eligible[10:12] = True
    rng = np.random.default_rng(0)
    means = matched_window_means(values, eligible, np.array([1, 2, 4, 5]), 400, rng)
    # length 1: every eligible bin
    assert set(np.unique(means[0])) == {2.0, 3.0, 4.0, 5.0, 10.0, 11.0}
    # length 2: the windows (2,3), (3,4), (4,5) and (10,11)
    assert set(np.unique(means[1])) == {2.5, 3.5, 4.5, 10.5}
    # length 4: only bins 2-5 hold it
    assert set(np.unique(means[2])) == {3.5}
    # length 5: no run holds it
    assert np.isnan(means[3]).all()


def test_matched_windows_draw_starts_uniformly():
    eligible = np.ones(100, dtype=bool)
    values = np.arange(100, dtype=float)
    rng = np.random.default_rng(1)
    means = matched_window_means(values, eligible, np.array([1]), 20_000, rng)
    counts = np.bincount(means[0].astype(int), minlength=100)
    assert counts.min() > 120 and counts.max() < 280


def test_matched_windows_skip_nan_values():
    values = np.array([1.0, np.nan, 3.0, np.nan])
    rng = np.random.default_rng(2)
    means = matched_window_means(values, np.ones(4, dtype=bool), np.array([4]), 3, rng)
    np.testing.assert_allclose(means, [[2.0, 2.0, 2.0]])
