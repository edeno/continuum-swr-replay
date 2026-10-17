"""The fragment-merge recount."""

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.inventory_comparison.availability import (
    not_still_times,
    still_intervals,
)
from continuum_swr_replay.analysis.inventory_comparison.overlap import (
    Coverage,
    directional_agreement,
    overlap_pairs,
)
from continuum_swr_replay.analysis.inventory_comparison.recount import (
    FRAGMENT_GAPS_S,
    RECOUNT_DTYPES,
    merge_fragments,
)
from continuum_swr_replay.analysis.inventory_comparison.schema import (
    SessionIdentity,
    inventory_from_intervals,
)

GRID = np.arange(0, 100.001, 0.002)
EPOCH = SessionIdentity(animal="sim", day=20200101, epoch="02_r1", brain_area="HPC")
OTHER = SessionIdentity(animal="sim", day=20200101, epoch="04_r2", brain_area="HPC")
KEY = ("sim", 20200101, "02_r1")
OTHER_KEY = ("sim", 20200101, "04_r2")
FRAGMENTS = [(10.0, 11.0), (11.005, 12.0), (12.015, 13.0), (13.04, 14.0)]
ALL_STILL = {KEY: np.empty(0), OTHER_KEY: np.empty(0)}
"""No sample in either epoch was moving or of unknown speed."""


def key(detector, native):
    """The event key of an event in the test epoch."""
    return f"{detector}:sim20200101_02_r1:{native}"


def events(detector, spans, session=EPOCH):
    return inventory_from_intervals(
        np.array([a for a, _ in spans], dtype=float),
        np.array([b for _, b in spans], dtype=float),
        native_keys=np.arange(len(spans)),
        detector=detector,
        session=session,
        run_id="run",
        speed_time_s=GRID,
        speed_cm_per_s=np.zeros(GRID.size),
    )


class GridSpan:
    """Events and full coverage on one clock, for the rounding test."""

    def __init__(self, time):
        self.time = time
        self.coverage = {
            (*KEY, "A"): Coverage(
                observed=np.array([[time[0], time[-1]]]),
                exposure=np.array([[time[0], time[-1]]]),
            )
        }

    def events(self, spans):
        return inventory_from_intervals(
            np.array([a for a, _ in spans]),
            np.array([b for _, b in spans]),
            native_keys=np.arange(len(spans)),
            detector="A",
            session=EPOCH,
            run_id="run",
            speed_time_s=self.time[:2],
            speed_cm_per_s=np.zeros(2),
        )


def coverage(*detectors, exposure=((0.0, 100.0),), session_key=KEY):
    exposure = np.array(exposure)
    return {
        (*session_key, d): Coverage(
            observed=np.array([[0.0, 100.0]]), exposure=exposure
        )
        for d in detectors
    }


def test_the_grid_is_the_stated_one():
    assert FRAGMENT_GAPS_S == (0.0, 0.010, 0.020, 0.050)


def test_gap_zero_reproduces_the_native_events():
    inventory = events("A", FRAGMENTS)

    merged = merge_fragments(
        inventory,
        coverage("A"),
        detector="A",
        gap_s=0.0,
        sample_interval_s=0.001,
        not_still_s=ALL_STILL,
    )

    assert merged.event_key.tolist() == inventory.event_key.tolist()
    np.testing.assert_array_equal(merged.event_start_s, inventory.event_start_s)
    np.testing.assert_array_equal(merged.event_end_s, inventory.event_end_s)
    assert (merged.n_fragments == 1).all()


@pytest.mark.parametrize(
    ("gap_s", "expected"),
    [
        (0.010, [(10.0, 12.0), (12.015, 13.0), (13.04, 14.0)]),
        (0.020, [(10.0, 13.0), (13.04, 14.0)]),
        (0.050, [(10.0, 14.0)]),
    ],
)
def test_fragments_closer_than_the_gap_merge(gap_s, expected):
    merged = merge_fragments(
        events("A", FRAGMENTS),
        coverage("A"),
        detector="A",
        gap_s=gap_s,
        sample_interval_s=0.001,
        not_still_s=ALL_STILL,
    )

    assert list(zip(merged.event_start_s, merged.event_end_s, strict=True)) == expected
    assert merged.n_fragments.sum() == len(FRAGMENTS)


@pytest.mark.parametrize("base_s", [1234.5678, 1_581_500_000.123456])
@pytest.mark.parametrize("gap_s", FRAGMENT_GAPS_S[1:])
def test_a_gap_of_exactly_the_setting_never_merges_on_a_real_clock(gap_s, base_s):
    """Gaps on the 500 Hz grid land exactly on 10, 20 and 50 ms; whether they
    count as "less than" must not depend on how the clock rounds, at a small
    offset or at the 1.6e9 s of an NWB timestamp."""
    time = base_s + np.arange(200_000) / 500.0
    n_gap = round(gap_s * 500.0)
    starts = np.arange(100, 199_000, 997)
    pairs = [
        [(time[a], time[a + 20]), (time[a + 20 + n_gap], time[a + 40 + n_gap])]
        for a in starts
    ]
    grid = GridSpan(time)

    at_setting = [
        len(
            merge_fragments(
                grid.events(p),
                grid.coverage,
                detector="A",
                gap_s=gap_s,
                sample_interval_s=0.002,
                not_still_s=ALL_STILL,
            )
        )
        for p in pairs
    ]
    one_short = [
        len(
            merge_fragments(
                grid.events([p[0], (p[1][0] - 0.002, p[1][1])]),
                grid.coverage,
                detector="A",
                gap_s=gap_s,
                sample_interval_s=0.002,
                not_still_s=ALL_STILL,
            )
        )
        for p in pairs
    ]

    assert set(at_setting) == {2}
    assert set(one_short) == {1}


def test_gaps_off_the_sample_grid_are_refused():
    inventory = events("A", [(10.0, 11.0), (11.0055, 12.0)])

    with pytest.raises(ValueError, match="grid"):
        merge_fragments(
            inventory,
            coverage("A"),
            detector="A",
            gap_s=0.01,
            sample_interval_s=0.001,
            not_still_s=ALL_STILL,
        )


@pytest.mark.parametrize(
    "exposure",
    [
        pytest.param(
            ((0.0, 11.001), (11.003, 100.0)),
            id="movement or missing samples in the gap",
        ),
        pytest.param(
            ((0.0, 10.5), (10.6, 100.0)), id="the first fragment leaves the bout"
        ),
    ],
)
def test_never_merges_outside_one_continuous_eligible_bout(exposure):
    inventory = events("A", [(10.0, 11.0), (11.005, 12.0)])

    merged = merge_fragments(
        inventory,
        coverage("A", exposure=exposure),
        detector="A",
        gap_s=0.05,
        sample_interval_s=0.001,
        not_still_s=ALL_STILL,
    )

    assert len(merged) == 2


def test_never_merges_across_epochs():
    inventory = pd.concat(
        [events("A", [(10.0, 11.0)]), events("A", [(11.005, 12.0)], session=OTHER)],
        ignore_index=True,
    )
    both = coverage("A") | coverage("A", session_key=OTHER_KEY)

    assert (
        len(
            merge_fragments(
                inventory,
                both,
                detector="A",
                gap_s=0.05,
                sample_interval_s=0.001,
                not_still_s=ALL_STILL,
            )
        )
        == 2
    )


def test_other_detectors_pass_through_untouched():
    inventory = pd.concat(
        [events("A", FRAGMENTS), events("B", FRAGMENTS)], ignore_index=True
    )

    merged = merge_fragments(
        inventory,
        coverage("A", "B"),
        detector="A",
        gap_s=0.05,
        sample_interval_s=0.001,
        not_still_s=ALL_STILL,
    )

    b = merged[merged.detector == "B"]
    assert b.event_key.tolist() == [key("B", i) for i in range(len(FRAGMENTS))]
    assert (b.n_fragments == 1).all()


def test_the_recount_feeds_the_same_agreement_accounting():
    """Merging changes segmentation, and so the count, without changing
    which events are eligible."""
    inventory = pd.concat(
        [events("A", FRAGMENTS), events("B", [(10.0, 14.0)])], ignore_index=True
    )
    cover = coverage("A", "B")

    def a_to_b(table):
        agreement = directional_agreement(table, overlap_pairs(table), cover)
        return agreement.set_index(["source_detector", "target_detector"]).loc[
            ("A", "B")
        ]

    native = a_to_b(inventory)
    merged = a_to_b(
        merge_fragments(
            inventory,
            cover,
            detector="A",
            gap_s=0.05,
            sample_interval_s=0.001,
            not_still_s=ALL_STILL,
        )
    )

    assert (native.n_eligible, native.n_matched) == (4, 4)
    assert (merged.n_eligible, merged.n_matched) == (1, 1)


def test_merged_keys_name_their_fragments():
    merged = merge_fragments(
        events("A", FRAGMENTS[:2]),
        coverage("A"),
        detector="A",
        gap_s=0.05,
        sample_interval_s=0.001,
        not_still_s=ALL_STILL,
    )

    assert merged.event_key.tolist() == [f"{key('A', 0)}|{key('A', 1)}"]
    assert merged.interval_span_s.tolist() == [2.0]


def test_row_order_does_not_matter():
    inventory = events("A", FRAGMENTS).sample(frac=1.0, random_state=0)
    assert inventory.event_key.tolist() != sorted(inventory.event_key)

    merged = merge_fragments(
        inventory,
        coverage("A"),
        detector="A",
        gap_s=0.010,
        sample_interval_s=0.001,
        not_still_s=ALL_STILL,
    )

    assert list(zip(merged.event_start_s, merged.event_end_s, strict=True)) == [
        (10.0, 12.0),
        (12.015, 13.0),
        (13.04, 14.0),
    ]


@pytest.mark.parametrize(("moving_in_gap", "n_merged"), [(False, 1), (True, 2)])
def test_a_bridge_never_crosses_a_sample_that_was_not_still(moving_in_gap, n_merged):
    """One moving sample between still ones leaves the exposure whole, since
    each still run reaches it, but the merged event would hold it."""
    speed = np.zeros(GRID.size)
    if moving_in_gap:
        speed[np.flatnonzero(np.isclose(GRID, 11.002))] = 9.0
    exposed = still_intervals(GRID, speed)
    assert exposed.shape == (1, 2)

    merged = merge_fragments(
        events("A", [(10.0, 11.0), (11.004, 12.0)]),
        coverage("A", exposure=exposed),
        detector="A",
        gap_s=0.05,
        sample_interval_s=0.002,
        not_still_s={KEY: not_still_times(GRID, speed)},
    )

    assert len(merged) == n_merged


class TestRecountContracts:
    def test_a_negative_gap_is_refused(self):
        with pytest.raises(ValueError, match="negative"):
            merge_fragments(
                events("A", [(10.0, 10.1)]),
                coverage("A"),
                detector="A",
                gap_s=-0.01,
                sample_interval_s=0.002,
                not_still_s=ALL_STILL,
            )

    def test_a_detector_the_comparison_does_not_know_is_refused(self):
        with pytest.raises(ValueError, match="roster"):
            merge_fragments(
                events("A", [(10.0, 10.1)]),
                coverage("A"),
                detector="nobody",
                gap_s=0.0,
                sample_interval_s=0.002,
                not_still_s=ALL_STILL,
            )

    def test_events_in_an_epoch_without_coverage_are_refused_by_name(self):
        with pytest.raises(ValueError, match="no coverage"):
            merge_fragments(
                events("A", [(10.0, 10.1)]),
                coverage("A", session_key=OTHER_KEY),
                detector="A",
                gap_s=0.0,
                sample_interval_s=0.002,
                not_still_s=ALL_STILL,
            )

    def test_events_in_an_epoch_without_a_not_still_entry_are_refused(self):
        with pytest.raises(ValueError, match="not_still_s"):
            merge_fragments(
                events("A", [(10.0, 10.1)]),
                coverage("A"),
                detector="A",
                gap_s=0.0,
                sample_interval_s=0.002,
                not_still_s={OTHER_KEY: np.empty(0)},
            )

    def test_the_merged_table_is_typed_even_when_empty(self):
        merged = merge_fragments(
            events("A", []),
            coverage("A"),
            detector="A",
            gap_s=0.0,
            sample_interval_s=0.002,
            not_still_s=ALL_STILL,
        )

        assert list(merged.columns) == list(RECOUNT_DTYPES)
        for column, dtype in RECOUNT_DTYPES.items():
            assert merged[column].dtype == dtype, column
