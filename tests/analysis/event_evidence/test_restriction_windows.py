"""Tests for the non-local restriction scores over panel g's windows, on tables
small enough to check by hand."""

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.event_evidence import restriction_windows as rw
from continuum_swr_replay.analysis.event_evidence.nonlocal_restriction import (
    SCORE_COLUMNS,
)

N = 10
STEP = 0.002


def _bins() -> pd.DataFrame:
    """Ten bins: ``none`` = 0, ``swr`` = bin index, ``full`` = 2 x bin index;
    the two-sided scores are the filter scores plus 1. Bins 2-3 inside an SWR,
    bin 6 moving, bin 9 excluded."""
    index = np.arange(N, dtype=np.float32)
    category = ["immobile_outside_swr"] * N
    category[2] = category[3] = "immobile_in_swr"
    category[6] = "moving"
    category[9] = "excluded"
    return pd.DataFrame(
        {
            "time": 100.0 + STEP * np.arange(N),
            "none": np.zeros(N, dtype=np.float32),
            "swr": index,
            "full": 2 * index,
            "none_two_sided": np.ones(N, dtype=np.float32),
            "swr_two_sided": index + 1,
            "full_two_sided": 2 * index + 1,
            "in_swr": np.isin(np.arange(N), [2, 3]),
            "category": pd.Categorical(category),
        }
    )


def _windows() -> pd.DataFrame:
    """Windows 0-1 (an event), 1-3 (overlapping it, inside the SWR), 5-6 (a
    moving bin) and 8-9 (an excluded bin)."""
    return pd.DataFrame(
        {
            "window_id": [0, 1, 2, 3],
            "event_id": ["a", "b", "a", "b"],
            "kind": ["event", "event", "control", "control"],
            "group": ["no SWR", "SWR", "no SWR", "SWR"],
            "s": [0, 1, 5, 8],
            "e": [1, 3, 6, 9],
        }
    )


def test_window_sums_count_and_sum_each_windows_bins():
    sums = rw.restriction_window_sums(_bins(), _windows())
    np.testing.assert_array_equal(sums["n_bins"], [2, 3, 2, 2])
    np.testing.assert_array_equal(sums["n_bins_in_swr"], [0, 2, 0, 0])
    np.testing.assert_array_equal(sums["n_bins_immobile"], [2, 3, 1, 1])
    np.testing.assert_array_equal(sums["n_bins_excluded"], [0, 0, 0, 1])
    np.testing.assert_array_equal(sums["n_bins_not_finite"], [0, 0, 0, 0])
    np.testing.assert_allclose(sums["duration_s"], np.array([2, 3, 2, 2]) * STEP)
    # swr sums the bin indices: 0+1, 1+2+3, 5+6, 8+9
    np.testing.assert_allclose(sums["swr"], [1, 6, 11, 17])
    np.testing.assert_allclose(sums["full"], [2, 12, 22, 34])
    np.testing.assert_allclose(sums["none"], 0)
    np.testing.assert_allclose(sums["full_two_sided"], [4, 15, 24, 36])
    assert all(sums[c].dtype == np.float64 for c in SCORE_COLUMNS)
    assert sums[list(rw.WINDOW_KEYS)].equals(_windows()[list(rw.WINDOW_KEYS)])


def test_window_sums_over_the_bins_outside_swrs():
    """Only the ``immobile_outside_swr`` bins: 0-1 of window 0, 1 of window 1
    (2-3 are inside the SWR), 5 of window 2 (6 moves), 8 of window 3 (9 is
    excluded); a window inside the SWR has none."""
    windows = pd.concat(
        [_windows(), pd.DataFrame([[4, "c", "event", "SWR", 2, 3]], columns=["window_id", "event_id", "kind", "group", "s", "e"])],
        ignore_index=True,
    )  # fmt: skip
    sums = rw.restriction_window_sums(_bins(), windows)
    np.testing.assert_array_equal(sums["n_bins_outside_swr"], [2, 1, 1, 1, 0])
    np.testing.assert_allclose(
        sums["duration_outside_swr_s"], np.array([2, 1, 1, 1, 0]) * STEP
    )
    np.testing.assert_allclose(sums["swr_outside_swr"], [1, 1, 5, 8, 0])
    np.testing.assert_allclose(sums["full_outside_swr"], [2, 2, 10, 16, 0])
    np.testing.assert_allclose(sums["none_outside_swr"], 0)
    np.testing.assert_allclose(sums["swr_two_sided_outside_swr"], [3, 2, 6, 9, 0])
    assert all(sums[c].dtype == np.float64 for c in rw.OUTSIDE_SWR_COLUMNS[2:])
    # the whole window's columns come first, unchanged by the new ones
    whole = [
        *rw.WINDOW_KEYS, "s", "e", "n_bins", "n_bins_in_swr", "n_bins_immobile",
        "n_bins_excluded", "n_bins_not_finite", "duration_s", *SCORE_COLUMNS,
    ]  # fmt: skip
    assert list(sums.columns) == [*whole, *rw.OUTSIDE_SWR_COLUMNS]


def test_a_score_not_finite_outside_swrs_makes_only_that_sum_nan():
    bins = _bins()
    bins.loc[2, "full"] = np.nan  # inside the SWR: the outside sums keep it out
    bins.loc[5, "swr"] = np.nan  # outside: window 2's outside sum is NaN too
    sums = rw.restriction_window_sums(bins, _windows())
    assert sums["full"].isna().tolist() == [False, True, False, False]
    np.testing.assert_allclose(sums["full_outside_swr"], [2, 2, 10, 16])
    assert sums["swr_outside_swr"].isna().tolist() == [False, False, True, False]


def test_a_score_that_is_not_finite_makes_its_windows_sum_nan_and_is_counted():
    bins = _bins()
    bins.loc[1, "full"] = np.nan
    sums = rw.restriction_window_sums(bins, _windows())
    np.testing.assert_array_equal(sums["n_bins_not_finite"], [1, 1, 0, 0])
    assert sums["full"].isna().tolist() == [True, True, False, False]
    # the other columns of those windows are kept
    np.testing.assert_allclose(sums["swr"], [1, 6, 11, 17])
    np.testing.assert_allclose(sums.loc[2:, "full"], [22, 34])


def test_a_window_outside_the_bins_raises():
    windows = _windows()
    windows.loc[3, "e"] = N
    with pytest.raises(ValueError, match="outside"):
        rw.restriction_window_sums(_bins(), windows)


def _holdout(windows: pd.DataFrame) -> pd.DataFrame:
    """Two tetrodes per window: 1 and 2 spikes in window 0, 0 and 0 in window 1,
    3 and 0 in 2, 1 and 1 in 3."""
    rows = windows.loc[np.repeat(windows.index, 2), list(rw.WINDOW_KEYS)]
    rows = rows.assign(
        tetrode=np.tile([0, 1], len(windows)),
        n_bins=np.repeat((windows["e"] - windows["s"] + 1).to_numpy(), 2),
        n_spikes=[1, 2, 0, 0, 3, 0, 1, 1],
    )
    return rows.iloc[::-1].reset_index(drop=True)


def test_spike_counts_sum_the_tetrodes_of_each_window_in_window_order():
    spikes = rw.window_spike_counts(_holdout(_windows()), _windows())
    assert spikes.tolist() == [3, 0, 3, 2]
    assert spikes.index.tolist() == [0, 1, 2, 3]


@pytest.mark.parametrize(
    "change",
    [
        lambda h: h[h["window_id"] != 2],  # a window missing
        lambda h: h.assign(group=np.where(h["window_id"] == 1, "no SWR", h["group"])),
        lambda h: h.assign(n_bins=h["n_bins"] + (h["window_id"] == 0)),  # length
    ],
)
def test_windows_that_are_not_the_holdout_runs_raise(change):
    with pytest.raises(ValueError, match="not the held-out"):
        rw.window_spike_counts(change(_holdout(_windows())), _windows())


def test_a_window_without_one_of_the_tetrodes_rows_raises():
    """Its spike count would leave that tetrode's spikes out."""
    holdout = _holdout(_windows())
    missing = holdout[~((holdout["window_id"] == 2) & (holdout["tetrode"] == 0))]
    with pytest.raises(ValueError, match="one row per tetrode"):
        rw.window_spike_counts(missing, _windows())
    twice = pd.concat([holdout, holdout.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="one row per tetrode"):
        rw.window_spike_counts(twice, _windows())


def _facts() -> pd.DataFrame:
    """Spikes per bin that give the windows the held-out counts 3, 0, 3, 2;
    bins 0-5 inside a well visit."""
    return pd.DataFrame(
        {
            "time": 100.0 + STEP * np.arange(N),
            "n_spikes": np.array([3, 0, 0, 0, 0, 2, 1, 0, 1, 1], dtype=np.int32),
            "in_well_visit": np.arange(N) < 6,
        }
    )


def test_bin_fact_sums_count_spikes_outside_swrs_and_inside_a_visit():
    facts = rw.window_bin_fact_sums(_bins(), _facts(), _windows())
    assert list(facts.columns) == ["n_spikes_bins", *rw.BIN_FACT_COLUMNS]
    np.testing.assert_array_equal(facts["n_spikes_bins"], [3, 0, 3, 2])
    # bins outside SWRs: 0-1; 1; 5 (6 moving); 8 (9 excluded)
    np.testing.assert_array_equal(facts["n_spikes_outside_swr"], [3, 0, 2, 1])
    # of those, inside the visit (bins 0-5): 0-1; 1; 5; none
    np.testing.assert_array_equal(facts["n_bins_outside_swr_in_visit"], [2, 1, 1, 0])
    np.testing.assert_array_equal(facts["n_spikes_outside_swr_in_visit"], [3, 0, 2, 0])
    # full - swr is the bin index there: 0 + 1, 1, 5, 0
    np.testing.assert_allclose(
        facts["full_outside_swr_in_visit"] - facts["swr_outside_swr_in_visit"],
        [1, 1, 5, 0],
    )
    assert facts["swr_two_sided_outside_swr_in_visit"].dtype == np.float64


def test_the_window_table_joins_the_sums_the_spikes_and_the_bin_fact_sums():
    table = rw.window_table(_bins(), _facts(), _windows())
    sums = rw.restriction_window_sums(_bins(), _windows())
    facts = rw.window_bin_fact_sums(_bins(), _facts(), _windows())
    assert list(table.columns) == [*sums.columns, "n_spikes", *rw.BIN_FACT_COLUMNS]
    pd.testing.assert_frame_equal(table[list(sums.columns)], sums)
    np.testing.assert_array_equal(table["n_spikes"], facts["n_spikes_bins"])
    pd.testing.assert_frame_equal(
        table[list(rw.BIN_FACT_COLUMNS)], facts[list(rw.BIN_FACT_COLUMNS)]
    )


def test_bin_facts_on_another_time_grid_raise():
    facts = _facts().assign(time=lambda f: f["time"] + 1.0)
    with pytest.raises(ValueError, match="time"):
        rw.window_bin_fact_sums(_bins(), facts, _windows())


def _scored() -> pd.DataFrame:
    sums = rw.restriction_window_sums(_bins(), _windows())
    facts = rw.window_bin_fact_sums(_bins(), _facts(), _windows())
    return pd.concat(
        [sums.assign(n_spikes=[3, 0, 3, 2]), facts[list(rw.BIN_FACT_COLUMNS)]],
        axis=1,
    )


def test_outside_swr_window_counts_name_the_windows_without_spikes_there():
    counts = rw.outside_swr_window_counts(_scored())
    # window 1 (an SWR event) has a bin outside its SWR and no spike in it
    assert counts.loc[("event", "SWR")].tolist() == [1, 1, 1, 0]
    assert counts.loc[("event", "no SWR")].tolist() == [1, 1, 0, 0]
    # window 3 (a control) has its bin 8 outside SWRs after the visit (bins 0-5)
    assert counts.loc[("control", "SWR")].tolist() == [1, 1, 0, 1]


def test_contrasts_per_spike_and_per_second():
    out = rw.window_contrasts(_scored())
    # full - swr filter: 1, 6, 11, (17, but excluded)
    np.testing.assert_allclose(
        out["full_minus_swr_per_spike"], [1 / 3, np.nan, 11 / 3, np.nan]
    )
    np.testing.assert_allclose(
        out["full_minus_swr_per_s"],
        [1 / (2 * STEP), 6 / (3 * STEP), 11 / (2 * STEP), np.nan],
    )
    # swr - none two-sided: (1+2) - 2 = 1, (2+3+4) - 3 = 6, (6+7) - 2 = 11
    np.testing.assert_allclose(
        out["swr_minus_none_two_sided_per_spike"], [1 / 3, np.nan, 11 / 3, np.nan]
    )
    np.testing.assert_allclose(
        out["full_minus_none_per_s"],
        [2 / (2 * STEP), 12 / (3 * STEP), 22 / (2 * STEP), np.nan],
    )


def test_contrasts_over_the_bins_outside_swrs_per_second_and_per_spike():
    out = rw.window_contrasts(_scored())
    # full - swr outside SWRs: 1 over 2 bins, 1 over 1, 5 over 1, (excluded)
    np.testing.assert_allclose(
        out["full_minus_swr_outside_swr_per_s"],
        [1 / (2 * STEP), 1 / STEP, 5 / STEP, np.nan],
    )
    # two-sided: (1+3) - (1+2) = 1, 3 - 2 = 1, 11 - 6 = 5
    np.testing.assert_allclose(
        out["full_minus_swr_two_sided_outside_swr_per_s"],
        [1 / (2 * STEP), 1 / STEP, 5 / STEP, np.nan],
    )
    # per spike there: 1 over 3 spikes, none in window 1's bin, 5 over 2
    np.testing.assert_allclose(
        out["full_minus_swr_outside_swr_per_spike"], [1 / 3, np.nan, 5 / 2, np.nan]
    )
    # a window with no bin outside SWRs has no value there
    inside = _scored().assign(n_bins_outside_swr=0, duration_outside_swr_s=0.0)
    assert rw.window_contrasts(inside)["swr_minus_none_outside_swr_per_s"].isna().all()


def _cohort() -> pd.DataFrame:
    """Two animals; animal a has two epochs. Event windows of the "no SWR"
    group with ``full - swr`` (filter) per spike of: a/e1 1 and 3 (median 2),
    a/e2 4 (median 4), b/e3 10 and 20 and 30 (median 20); one "SWR" event and
    one control per epoch with ``full - swr`` per spike 1."""
    rows = []
    for animal, epoch, gains in (
        ("a", "e1", [1, 3]),
        ("a", "e2", [4]),
        ("b", "e3", [10, 20, 30]),
    ):
        for gain in gains:
            rows.append((animal, epoch, "event", "no SWR", gain))
        rows.append((animal, epoch, "event", "SWR", 1))
        rows.append((animal, epoch, "control", "no SWR", 1))
    table = pd.DataFrame(rows, columns=["animal", "epoch_id", "kind", "group", "gain"])
    n = len(table)
    # one spike per window and one bin of 1 s; full - swr = gain, swr - none = 0;
    # outside SWRs, half a second with full - swr of the gain (2 gain per s),
    # but none in e2's SWR event
    zeros = np.zeros(n)
    gain = table["gain"].to_numpy(float)
    inside = ((table["epoch_id"] == "e2") & (table["group"] == "SWR")).to_numpy()
    return table.assign(
        n_spikes=1,
        n_bins_excluded=0,
        duration_s=1.0,
        none=zeros,
        swr=zeros,
        full=gain,
        none_two_sided=zeros,
        swr_two_sided=zeros,
        full_two_sided=2 * gain,
        n_bins_outside_swr=np.where(inside, 0, 1),
        # outside SWRs two spikes, but none in e1's SWR event
        n_spikes_outside_swr=np.where(
            inside,
            0,
            np.where((table["epoch_id"] == "e1") & (table["group"] == "SWR"), 0, 2),
        ),
        duration_outside_swr_s=np.where(inside, 0.0, 0.5),
        none_outside_swr=zeros,
        swr_outside_swr=zeros,
        full_outside_swr=np.where(inside, 0.0, gain),
        none_two_sided_outside_swr=zeros,
        swr_two_sided_outside_swr=zeros,
        full_two_sided_outside_swr=np.where(inside, 0.0, 3 * gain),
        **{
            column: zeros
            for column in rw.BIN_FACT_COLUMNS
            if column != "n_spikes_outside_swr"
        },
    )


def test_statistics_are_the_per_animal_median_of_epoch_medians_and_their_mean():
    stats = rw.restriction_window_statistics(_cohort(), n_boot=200, seed=1)
    # 2 families x 3 contrasts x 2 units, events (2 groups) and controls (1)
    assert len(stats) == 2 * 3 * 2 * 3
    assert list(stats.columns[:6]) == [
        "family", "contrast", "unit", "kind", "group", "part",
    ]  # fmt: skip
    assert set(stats["part"]) == {"whole"}
    key = stats.set_index(["family", "contrast", "unit", "kind", "group"])
    row = key.loc[("filter", "full_minus_swr", "per_spike", "event", "no SWR")]
    # a: median of epoch medians 2 and 4 = 3; b: 20
    np.testing.assert_allclose(row["animals"], [3.0, 20.0])
    assert list(row["animal_names"]) == ["a", "b"]
    assert row["estimate"] == pytest.approx(11.5)
    assert row["ci_low"] <= row["estimate"] <= row["ci_high"]
    assert row["n_events"] == 6 and row["n_scored"] == 6 and row["n_animals"] == 2
    assert row["n_animals_positive"] == 2
    two_sided = key.loc[("two_sided", "full_minus_swr", "per_s", "event", "no SWR")]
    assert two_sided["estimate"] == pytest.approx(23.0)
    control = key.loc[("filter", "full_minus_swr", "per_spike", "control", "no SWR")]
    assert control["estimate"] == pytest.approx(1.0) and control["n_events"] == 3
    null = key.loc[("filter", "swr_minus_none", "per_s", "event", "SWR")]
    assert null["estimate"] == pytest.approx(0.0)
    assert null["n_animals_positive"] == 0


def test_statistics_outside_swrs_count_windows_without_bins_or_spikes():
    keys = ["family", "contrast", "unit", "kind", "group", "part"]
    both = rw.restriction_window_statistics(
        _cohort(), n_boot=200, seed=1, parts=["whole", "outside_swr"]
    ).set_index(keys)
    # per spike and per second over the whole window and outside SWRs
    assert len(both) == 2 * 3 * (2 + 2) * 3
    whole = rw.restriction_window_statistics(_cohort(), n_boot=200, seed=1)
    for row in whole.itertuples(index=False):
        mine = both.loc[(*row[:5], "whole")]
        assert mine["estimate"] == row.estimate and mine["ci_low"] == row.ci_low
    row = both.loc[
        ("filter", "full_minus_swr", "per_s", "event", "no SWR", "outside_swr")
    ]
    # twice the per-window gain: a 3 x 2, b 20 x 2
    np.testing.assert_allclose(row["animals"], [6.0, 40.0])
    assert row["estimate"] == pytest.approx(23.0)
    two_sided = both.loc[
        ("two_sided", "full_minus_swr", "per_s", "event", "no SWR", "outside_swr")
    ]
    assert two_sided["estimate"] == pytest.approx(69.0)
    # e2's SWR event has no bin outside its SWR: counted, not scored
    swr = both.loc[("filter", "full_minus_swr", "per_s", "event", "SWR", "outside_swr")]
    assert swr["n_events"] == 3 and swr["n_scored"] == 2
    np.testing.assert_allclose(swr["animals"], [2.0, 2.0])
    # per spike: half the gain (2 spikes); e1's SWR event has none either
    row = both.loc[
        ("filter", "full_minus_swr", "per_spike", "event", "no SWR", "outside_swr")
    ]
    np.testing.assert_allclose(row["animals"], [1.5, 10.0])
    swr = both.loc[
        ("filter", "full_minus_swr", "per_spike", "event", "SWR", "outside_swr")
    ]
    assert swr["n_events"] == 3 and swr["n_scored"] == 1


def test_statistics_leave_out_and_count_windows_with_excluded_bins():
    cohort = _cohort()
    cohort.loc[cohort["gain"] == 30, "n_bins_excluded"] = 1
    stats = rw.restriction_window_statistics(cohort, n_boot=50)
    key = stats.set_index(["family", "contrast", "unit", "kind", "group"])
    row = key.loc[("filter", "full_minus_swr", "per_spike", "event", "no SWR")]
    # b: median of 10 and 20 = 15
    np.testing.assert_allclose(row["animals"], [3.0, 15.0])
    assert row["n_events"] == 6 and row["n_scored"] == 5


def test_a_chosen_row_is_the_same_row_of_the_full_table():
    """Each row's bootstrap is seeded on its own, so computing only some rows
    gives those rows exactly as the full table has them."""
    keys = ["family", "contrast", "unit", "kind", "group", "part"]
    full = rw.restriction_window_statistics(_cohort(), n_boot=200, seed=3)
    chosen = rw.restriction_window_statistics(
        _cohort(),
        n_boot=200,
        seed=3,
        families=["filter"],
        contrasts=["full_minus_none"],
        units=["per_spike"],
        kinds=["event"],
    )
    assert list(chosen.columns) == list(full.columns)
    assert list(chosen["group"]) == ["SWR", "no SWR"]
    expected = full.set_index(keys).loc[
        [tuple(row) for row in chosen[keys].itertuples(index=False)]
    ]
    for (_, mine), (_, theirs) in zip(
        chosen.set_index(keys).iterrows(), expected.iterrows(), strict=True
    ):
        for column in ("estimate", "ci_low", "ci_high", "n_events", "n_scored"):
            assert mine[column] == theirs[column]
        np.testing.assert_array_equal(mine["animals"], theirs["animals"])
        assert mine["n_animals_positive"] == theirs["n_animals_positive"]


def test_an_unknown_choice_raises():
    with pytest.raises(ValueError, match="unknown contrast"):
        rw.restriction_window_statistics(_cohort(), n_boot=10, contrasts=["x"])
    with pytest.raises(ValueError, match="no windows of kind"):
        rw.restriction_window_statistics(_cohort(), n_boot=10, kinds=["y"])
    with pytest.raises(ValueError, match="unknown part"):
        rw.restriction_window_statistics(_cohort(), n_boot=10, parts=["edges"])
