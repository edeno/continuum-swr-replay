"""The cells pairwise counting misses."""

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.inventory_comparison.cells import (
    NO_DETECTOR_DTYPES,
    membership_against,
    no_detector_time,
)
from continuum_swr_replay.analysis.inventory_comparison.overlap import (
    Coverage,
    event_matches,
    overlap_pairs,
)
from continuum_swr_replay.analysis.inventory_comparison.schema import (
    SessionIdentity,
    inventory_from_intervals,
)

GRID = np.arange(0, 100.001, 0.002)
EPOCH = SessionIdentity(animal="sim", day=20200101, epoch="02_r1", brain_area="HPC")
KEY = ("sim", 20200101, "02_r1")


def key(detector, native):
    """The event key of an event in the test epoch."""
    return f"{detector}:sim20200101_02_r1:{native}"


def events(detector, spans):
    return inventory_from_intervals(
        np.array([a for a, _ in spans], dtype=float),
        np.array([b for _, b in spans], dtype=float),
        native_keys=np.arange(len(spans)),
        detector=detector,
        session=EPOCH,
        run_id="run",
        speed_time_s=GRID,
        speed_cm_per_s=np.zeros(GRID.size),
    )


def cover(exposure=((0.0, 100.0),)):
    return Coverage(observed=np.array([[0.0, 100.0]]), exposure=np.array(exposure))


def everywhere(*detectors):
    return {(*KEY, d): cover() for d in detectors}


def cells(inventory, coverage, **kwargs):
    matches = event_matches(inventory, overlap_pairs(inventory), coverage)
    return membership_against(inventory, matches, **kwargs).set_index("event_key")


class TestMembershipAgainst:
    def test_found_by_exactly_one_detector(self):
        inventory = pd.concat(
            [
                events("A", [(10, 20), (50, 60)]),
                events("B", [(15, 25)]),
                events("C", [(80, 90)]),
            ],
            ignore_index=True,
        )

        found = cells(
            inventory,
            everywhere("A", "B", "C"),
            sources=["A", "B", "C"],
            against=["A", "B", "C"],
        )

        assert found.membership.to_dict() == {
            key("A", 0): "matched",
            key("A", 1): "unmatched",
            key("B", 0): "matched",
            key("C", 0): "unmatched",
        }
        assert found.loc[key("A", 1), "n_against"] == 2

    def test_matched_by_one_of_the_set_is_matched_even_if_another_is_unknown(self):
        inventory = pd.concat(
            [events("S", [(10, 20)]), events("D1", [(12, 14)])], ignore_index=True
        )
        coverage = everywhere("S", "D1") | {(*KEY, "D2"): cover(((0.0, 5.0),))}

        found = cells(inventory, coverage, sources=["S"], against=["D1", "D2"])

        assert found.loc[key("S", 0), "membership"] == "matched"

    def test_unmatched_needs_every_detector_of_the_set_to_have_looked(self):
        """An SWR no decoding detector matched: only if both decoders looked."""
        inventory = events("SWR", [(10, 20)])

        missing_run = cells(
            inventory, everywhere("SWR", "D1"), sources=["SWR"], against=["D1", "D2"]
        )
        not_exposed = cells(
            inventory,
            everywhere("SWR", "D1") | {(*KEY, "D2"): cover(((0.0, 15.0),))},
            sources=["SWR"],
            against=["D1", "D2"],
        )
        both_looked = cells(
            inventory,
            everywhere("SWR", "D1", "D2"),
            sources=["SWR"],
            against=["D1", "D2"],
        )

        assert missing_run.loc[key("SWR", 0), "membership"] == "unknown"
        assert not_exposed.loc[key("SWR", 0), "membership"] == "unknown"
        assert both_looked.loc[key("SWR", 0), "membership"] == "unmatched"

    def test_only_source_events_are_classified_and_only_against_the_set(self):
        """A sequence matched by a decoder but by no triggered inventory is in
        the neither cell: decoders are not in the set it is tested against."""
        inventory = pd.concat(
            [
                events("SEQ", [(10, 20)]),
                events("OTHER_SEQ", [(10, 20)]),
                events("SWR", [(50, 60)]),
            ],
            ignore_index=True,
        )

        found = cells(
            inventory,
            everywhere("SEQ", "OTHER_SEQ", "SWR"),
            sources=["SEQ"],
            against=["SWR"],
        )

        assert found.index.tolist() == [key("SEQ", 0)]
        assert found.loc[key("SEQ", 0), "membership"] == "unmatched"

    def test_a_source_is_never_tested_against_itself(self):
        with pytest.raises(ValueError, match="other than"):
            cells(
                events("A", [(10, 20)]), everywhere("A"), sources=["A"], against=["A"]
            )


class TestNoDetectorTime:
    def test_shared_exposure_minus_the_union_of_detections(self):
        inventory = pd.concat(
            [events("A", [(10, 20)]), events("B", [(15, 30)])], ignore_index=True
        )

        row = no_detector_time(
            inventory, everywhere("A", "B"), detectors=["A", "B"]
        ).iloc[0]

        assert list(row.index) == list(NO_DETECTOR_DTYPES)
        assert (row.shared_exposure_s, row.detected_s, row.undetected_s) == (
            100,
            20,
            80,
        )
        assert row.undetected_fraction == pytest.approx(0.8)

    def test_uses_every_detectors_exposure_and_clips_to_it(self):
        inventory = events("B", [(40, 60)])
        coverage = {(*KEY, "A"): cover(((0.0, 50.0),)), (*KEY, "B"): cover()}

        row = no_detector_time(inventory, coverage, detectors=["A", "B"]).iloc[0]

        assert (row.shared_exposure_s, row.detected_s, row.undetected_s) == (50, 10, 40)

    def test_nothing_found_in_complete_data_is_observed(self):
        row = no_detector_time(
            events("A", []), everywhere("A", "B"), detectors=["A", "B"]
        ).iloc[0]

        assert row.available
        assert (row.undetected_s, row.undetected_fraction) == (100, 1.0)

    def test_a_detector_that_did_not_run_makes_it_unavailable(self):
        row = no_detector_time(
            events("A", []), everywhere("A"), detectors=["A", "B"]
        ).iloc[0]

        assert not row.available
        assert np.isnan(row.undetected_fraction)

    def test_detectors_outside_the_set_are_ignored(self):
        inventory = pd.concat(
            [events("A", [(10, 20)]), events("Z", [(50, 90)])], ignore_index=True
        )

        row = no_detector_time(
            inventory, everywhere("A", "B", "Z"), detectors=["A", "B"]
        ).iloc[0]

        assert row.detected_s == 10
