"""The fragment-merge recount (spec 03, "Fragment-merge recount")."""

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.inventory_comparison.overlap import (
    Coverage,
    directional_agreement,
    overlap_pairs,
)
from continuum_swr_replay.analysis.inventory_comparison.recount import (
    FRAGMENT_GAPS_S,
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

    merged = merge_fragments(inventory, coverage("A"), detector="A", gap_s=0.0)

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
        events("A", FRAGMENTS), coverage("A"), detector="A", gap_s=gap_s
    )

    assert list(zip(merged.event_start_s, merged.event_end_s, strict=True)) == expected
    assert merged.n_fragments.sum() == len(FRAGMENTS)


def test_a_separation_equal_to_the_gap_is_not_merged():
    """ "Separated by less than" the gap: exact binary values, no rounding."""
    inventory = events("A", [(10.0, 10.5), (11.0, 12.0)])

    assert len(merge_fragments(inventory, coverage("A"), detector="A", gap_s=0.5)) == 2
    assert (
        len(merge_fragments(inventory, coverage("A"), detector="A", gap_s=0.625)) == 1
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
        inventory, coverage("A", exposure=exposure), detector="A", gap_s=0.05
    )

    assert len(merged) == 2


def test_never_merges_across_epochs():
    inventory = pd.concat(
        [events("A", [(10.0, 11.0)]), events("A", [(11.005, 12.0)], session=OTHER)],
        ignore_index=True,
    )
    both = coverage("A") | coverage("A", session_key=OTHER_KEY)

    assert len(merge_fragments(inventory, both, detector="A", gap_s=0.05)) == 2


def test_other_detectors_pass_through_untouched():
    inventory = pd.concat(
        [events("A", FRAGMENTS), events("B", FRAGMENTS)], ignore_index=True
    )

    merged = merge_fragments(inventory, coverage("A", "B"), detector="A", gap_s=0.05)

    b = merged[merged.detector == "B"]
    assert b.event_key.tolist() == [f"B:{i}" for i in range(len(FRAGMENTS))]
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
    merged = a_to_b(merge_fragments(inventory, cover, detector="A", gap_s=0.05))

    assert (native.n_eligible, native.n_matched) == (4, 4)
    assert (merged.n_eligible, merged.n_matched) == (1, 1)


def test_merged_keys_name_their_fragments():
    merged = merge_fragments(
        events("A", FRAGMENTS[:2]), coverage("A"), detector="A", gap_s=0.05
    )

    assert merged.event_key.tolist() == ["A:0|A:1"]
    assert merged.interval_span_s.tolist() == [2.0]


def test_row_order_does_not_matter():
    inventory = events("A", FRAGMENTS).sample(frac=1.0, random_state=0)
    assert inventory.event_key.tolist() != sorted(inventory.event_key)

    merged = merge_fragments(inventory, coverage("A"), detector="A", gap_s=0.010)

    assert list(zip(merged.event_start_s, merged.event_end_s, strict=True)) == [
        (10.0, 12.0),
        (12.015, 13.0),
        (13.04, 14.0),
    ]
