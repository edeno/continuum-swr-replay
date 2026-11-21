"""Trial folds: whole trial segments, long stays cut into pieces, balanced by class."""

from __future__ import annotations

import numpy as np
import pytest

from continuum_swr_replay.analysis.regression import folds as fo


def _trials(
    lengths_s: np.ndarray, classes: np.ndarray, bin_s: float = 0.1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A time axis and the segment starts of back-to-back trials."""
    starts = np.r_[0.0, np.cumsum(lengths_s)[:-1]]
    time = np.arange(0.0, float(np.sum(lengths_s)), bin_s)
    return time, starts, np.asarray(classes)


def test_the_rule_constants() -> None:
    assert fo.N_FOLDS == 5
    assert fo.MAX_SEGMENT_S == 30.0


def test_trial_segments_join_the_epochs_edges_to_the_first_and_last() -> None:
    time = np.arange(10.0)
    np.testing.assert_array_equal(
        fo.trial_segments(time, np.array([2.0, 5.0, 8.5])),
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 2],
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        fo.trial_segments(time, np.array([2.0, 2.0]))
    with pytest.raises(ValueError, match="no segment starts"):
        fo.trial_segments(time, np.array([]))


def test_trial_folds_rules() -> None:
    """Every row of a group in one fold; long stays cut at 30 s; classes
    balanced; deterministic; too few groups raise."""
    rng = np.random.default_rng(2)
    time, starts, classes = _trials(rng.uniform(5, 30, 40), rng.integers(0, 6, 40))
    folds = fo.trial_folds(time, starts, classes)
    segment = fo.trial_segments(time, starts)
    group = fo.capped_groups(time, segment)
    assert folds.shape == time.shape
    assert set(np.unique(folds)) == set(range(fo.N_FOLDS))
    for g in np.unique(group):
        assert np.unique(folds[group == g]).size == 1
    np.testing.assert_array_equal(folds, fo.trial_folds(time, starts, classes))

    # Classes balanced: each fold holds a fifth of each class's time.
    lengths = np.tile([8.0, 12.0, 9.0, 11.0, 10.0], 4)
    time, starts, classes = _trials(lengths, np.repeat([0, 1], 10))
    folds = fo.trial_folds(time, starts, classes)
    y = classes[fo.trial_segments(time, starts)]
    for c in (0, 1):
        share = np.bincount(folds[y == c], minlength=5) / (y == c).sum()
        np.testing.assert_allclose(share, 0.2, atol=0.05)

    # Too few groups.
    time, starts, classes = _trials([10.0] * 4, [0, 1, 0, 1])
    with pytest.raises(ValueError, match="fewer than the 5 folds"):
        fo.trial_folds(time, starts, classes)


def test_capped_groups_cut_a_long_segment_into_equal_pieces_within_the_cap() -> None:
    time = np.arange(0.0, 200.0, 0.1)  # 2000 bins
    # segments of 95 s, exactly 30 s, 10 s and 65 s
    segment = fo.trial_segments(time, np.array([0.0, 95.0, 125.0, 135.0]))
    group = fo.capped_groups(time, segment, 30.0)
    assert (np.diff(group) >= 0).all()
    sizes = np.bincount(group)
    # 95 s -> 4 pieces of 23.7-23.8 s; 30 s whole; 10 s whole; 65 s -> 3 pieces
    np.testing.assert_array_equal(sizes, [238, 237, 238, 237, 300, 100, 217, 217, 216])
    for g in np.unique(group):
        assert np.unique(segment[group == g]).size == 1
    np.testing.assert_array_equal(fo.capped_groups(time, segment, None), segment)
    with pytest.raises(ValueError, match="runs of consecutive"):
        fo.capped_groups(time, segment[::-1])


def test_task_segment_starts_add_the_time_before_and_after_the_trials() -> None:
    time = np.arange(0.0, 75.0, 0.1)
    starts, classes = fo.task_segment_starts(
        time, np.array([5.0, 30.0, 50.0]), np.array([2, 3, 2]), 70.0
    )
    np.testing.assert_array_equal(starts, [0.0, 5.0, 30.0, 50.0, 70.0])
    np.testing.assert_array_equal(
        classes, [fo.BEFORE_FIRST_TRIAL, 2, 3, 2, fo.AFTER_LAST_TRIAL]
    )
    starts, classes = fo.task_segment_starts(
        time, np.array([0.0, 30.0]), np.array([2, 3]), 200.0
    )
    np.testing.assert_array_equal(classes, [2, 3])
    with pytest.raises(ValueError, match="ends before its departure"):
        fo.task_segment_starts(time, np.array([10.0, 30.0]), np.array([2, 3]), 20.0)
    with pytest.raises(ValueError, match="needs a destination"):
        fo.task_segment_starts(time, np.array([10.0]), np.array([np.nan]), 20.0)
    with pytest.raises(ValueError, match="no trials"):
        fo.task_segment_starts(time, np.array([]), np.array([]), 20.0)


def test_long_stays_before_and_after_the_trials_are_spread_over_the_folds() -> None:
    """150 s before the first trial and 300 s after the last, around two
    hundred 10 s trials: with the 30 s cap each fold holds a fifth of the time
    before and of the time after; without it each sits in one fold."""
    departures = 150.0 + 10.0 * np.arange(200)
    time = np.arange(0.0, 2450.0, 0.1)
    starts, classes = fo.task_segment_starts(
        time, departures, np.tile([0, 1], 100), 2150.0
    )
    y = classes[fo.trial_segments(time, starts)]
    folds = fo.trial_folds(time, starts, classes)
    for c in (fo.BEFORE_FIRST_TRIAL, fo.AFTER_LAST_TRIAL, 0, 1):
        share = np.bincount(folds[y == c], minlength=5) / (y == c).sum()
        np.testing.assert_allclose(share, 0.2, atol=0.02)
    whole = fo.trial_folds(time, starts, classes, max_segment_s=None)
    for c in (fo.BEFORE_FIRST_TRIAL, fo.AFTER_LAST_TRIAL):
        assert np.unique(whole[y == c]).size == 1


def test_trial_folds_need_a_class_per_segment() -> None:
    time, starts, _ = _trials([10.0] * 6, [0] * 6)
    with pytest.raises(ValueError, match="needs a class"):
        fo.trial_folds(time, starts, np.array([0, 1, np.nan, 0, 1, 0]))
    with pytest.raises(ValueError, match="segment classes"):
        fo.trial_folds(time, starts, np.array([0, 1]))


def test_trials_off_the_rows_clock_raise() -> None:
    time = 1000.0 + np.arange(0.0, 600.0, 0.1)
    departures = 1000.0 + 10.0 * np.arange(1, 50)
    wells = np.tile([2, 3], 25)[:49]
    fo.task_segment_starts(time, departures, wells, 1595.0)
    assert fo.trial_bin_share(time, departures, 1595.0) == pytest.approx(0.975)
    with pytest.raises(ValueError, match="outside the time axis"):
        fo.task_segment_starts(time, departures * 1000.0, wells, 1595.0 * 1000.0)
    with pytest.raises(ValueError, match="cover 19.8% of the bins"):
        fo.task_segment_starts(time, departures[:12], wells[:12], 1129.0)


def test_a_long_stay_and_an_event_across_its_cut_span_two_folds() -> None:
    """A 95 s stay among 10 s trials is cut into four pieces that fall in
    different folds; the rows of an event that crosses a cut (here the 2 s
    around it) are in two folds."""
    lengths = np.r_[[10.0] * 20, 95.0, [10.0] * 20]
    time, starts, classes = _trials(lengths, np.r_[[0, 1] * 10, 2, [0, 1] * 10])
    folds = fo.trial_folds(time, starts, classes)
    segment = fo.trial_segments(time, starts)
    group = fo.capped_groups(time, segment)
    stay = segment == 20
    assert np.unique(group[stay]).size == 4
    assert np.unique(folds[stay]).size > 1
    cut = time[stay][np.flatnonzero(np.diff(group[stay]))]
    crossing = [c for c in cut if np.unique(folds[np.abs(time - c) < 1.0]).size == 2]
    assert crossing


def test_row_times_must_be_strictly_increasing() -> None:
    """Event by arm rows share their event's time: the folds are made on one
    row per event, and the caller maps them to the event's arm rows."""
    time, starts, classes = _trials([10.0] * 8, [0, 1] * 4)
    events = np.sort(np.random.default_rng(0).uniform(0.0, 80.0, 60))
    fo.trial_folds(events, starts, classes)
    for rows in (np.repeat(events, 9), events[::-1]):
        with pytest.raises(ValueError, match="one row per event or per bin"):
            fo.trial_folds(rows, starts, classes)
        with pytest.raises(ValueError, match="one row per event or per bin"):
            fo.capped_groups(rows, fo.trial_segments(rows, starts))


def test_the_cap_is_a_number_of_rows_at_the_median_spacing() -> None:
    """Rows 1 s apart then 3 s apart: the median spacing is 1 s, so a piece is
    30 rows, which spans 29 s where the rows are 1 s apart and 87 s where
    they are 3 s apart."""
    time = np.r_[np.arange(0.0, 100.0), 100.0 + 3.0 * np.arange(1, 41)]
    segment = np.zeros(time.size, dtype=np.int64)
    group = fo.capped_groups(time, segment)
    sizes = np.bincount(group)
    assert sizes.max() <= 30
    spans = [np.ptp(time[group == g]) for g in np.unique(group)]
    assert max(spans) > 30.0
