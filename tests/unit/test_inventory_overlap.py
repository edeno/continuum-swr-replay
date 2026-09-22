"""Every overlap between inventories, and the counts built on it (spec 03,
"Matching and counting")."""

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.inventory_comparison.overlap import (
    DIRECTIONAL_DTYPES,
    MATCH_DTYPES,
    PAIR_DTYPES,
    Coverage,
    directional_agreement,
    event_matches,
    overlap_pairs,
    remove_bridging_events,
    time_coverage,
)
from continuum_swr_replay.analysis.inventory_comparison.schema import (
    SessionIdentity,
    inventory_from_intervals,
)

GRID = np.arange(0, 100.001, 0.002)
EPOCH = SessionIdentity(animal="sim", day=20200101, epoch="02_r1", brain_area="HPC")
OTHER = SessionIdentity(animal="sim", day=20200101, epoch="04_r2", brain_area="HPC")
KEY = ("sim", 20200101, "02_r1")


def events(detector, spans, session=EPOCH):
    starts = np.array([a for a, _ in spans], dtype=float)
    ends = np.array([b for _, b in spans], dtype=float)
    return inventory_from_intervals(
        starts,
        ends,
        native_keys=np.arange(len(spans)),
        detector=detector,
        session=session,
        run_id="run",
        speed_time_s=GRID,
        speed_cm_per_s=np.zeros(GRID.size),
    )


def everywhere(*detectors, session_key=KEY, bounds=(0.0, 100.0)):
    full = np.array([bounds])
    return {
        (*session_key, d): Coverage(observed=full, exposure=full) for d in detectors
    }


class TestPairs:
    def test_every_strict_overlap_with_its_geometry(self):
        inventory = pd.concat(
            [events("A", [(10, 20)]), events("B", [(15, 30)])], ignore_index=True
        )

        pairs = overlap_pairs(inventory)

        assert list(pairs.columns) == list(PAIR_DTYPES)
        pair = pairs.iloc[0]
        assert (pair.detector_a, pair.detector_b) == ("A", "B")
        assert pair.intersection_s == pytest.approx(5.0)
        assert pair.iou == pytest.approx(5.0 / 20.0)
        assert pair.fraction_a_covered == pytest.approx(0.5)
        assert pair.fraction_b_covered == pytest.approx(5.0 / 15.0)
        assert pair.onset_difference_s == pytest.approx(5.0)
        assert pair.offset_difference_s == pytest.approx(10.0)

    def test_touching_at_one_instant_is_not_an_overlap(self):
        """A one-sample edge contact is not agreement."""
        inventory = pd.concat(
            [events("A", [(10, 20)]), events("B", [(20, 30)])], ignore_index=True
        )

        assert overlap_pairs(inventory).empty

    def test_events_of_one_detector_are_never_paired_with_each_other(self):
        assert overlap_pairs(events("A", [(10, 20), (15, 25)])).empty

    def test_an_event_is_not_matched_across_epochs(self):
        """Same times, different epochs: no pair."""
        inventory = pd.concat(
            [events("A", [(10, 20)]), events("B", [(10, 20)], session=OTHER)],
            ignore_index=True,
        )

        assert overlap_pairs(inventory).empty

    def test_one_to_many_keeps_every_pair(self):
        inventory = pd.concat(
            [events("A", [(0, 50)]), events("B", [(5, 10), (20, 30), (40, 45)])],
            ignore_index=True,
        )

        assert len(overlap_pairs(inventory)) == 3

    def test_no_events_gives_the_declared_dtypes(self):
        pairs = overlap_pairs(events("A", []))

        assert pairs.empty
        for column, dtype in PAIR_DTYPES.items():
            assert pairs[column].dtype == dtype, column


class TestDirectionalAgreement:
    def test_each_event_counts_once_in_its_direction_however_many_it_matches(self):
        """The denominators are the easiest thing to get silently wrong."""
        inventory = pd.concat(
            [
                events("A", [(0, 50), (60, 70)]),
                events("B", [(5, 10), (20, 30), (40, 45)]),
            ],
            ignore_index=True,
        )
        coverage = everywhere("A", "B")

        raw = directional_agreement(inventory, overlap_pairs(inventory), coverage)
        agreement = raw.set_index(["source_detector", "target_detector"])

        assert list(raw.columns) == list(DIRECTIONAL_DTYPES)
        a_to_b = agreement.loc[("A", "B")]
        assert (a_to_b.n_eligible, a_to_b.n_matched) == (2, 1)
        assert a_to_b.proportion_matched == pytest.approx(0.5)
        b_to_a = agreement.loc[("B", "A")]
        assert (b_to_a.n_eligible, b_to_a.n_matched) == (3, 3)

    def test_many_to_many_counts_each_event_once(self):
        inventory = pd.concat(
            [events("A", [(0, 10), (8, 20)]), events("B", [(5, 9), (9, 15)])],
            ignore_index=True,
        )

        agreement = directional_agreement(
            inventory, overlap_pairs(inventory), everywhere("A", "B")
        ).set_index(["source_detector", "target_detector"])

        assert len(overlap_pairs(inventory)) == 4
        assert agreement.loc[("A", "B")].n_matched == 2
        assert agreement.loc[("B", "A")].n_matched == 2

    def test_an_event_crossing_the_coverage_boundary_leaves_the_denominator(self):
        """Its bounds are not clipped to qualify it; it is excluded and counted
        as excluded."""
        inventory = pd.concat(
            [events("A", [(10, 20), (45, 55)]), events("B", [(12, 14)])],
            ignore_index=True,
        )
        coverage = everywhere("A") | {
            (*KEY, "B"): Coverage(
                observed=np.array([[0.0, 50.0]]), exposure=np.array([[0.0, 50.0]])
            )
        }

        agreement = directional_agreement(
            inventory, overlap_pairs(inventory), coverage
        ).set_index(["source_detector", "target_detector"])
        matches = event_matches(inventory, overlap_pairs(inventory), coverage)

        a_to_b = agreement.loc[("A", "B")]
        assert (a_to_b.n_source_events, a_to_b.n_eligible, a_to_b.n_excluded) == (
            2,
            1,
            1,
        )
        crossing = matches[
            (matches.event_key == "A:1") & (matches.other_detector == "B")
        ]
        assert crossing.membership.item() == "unknown"
        assert inventory.set_index("event_key").loc["A:1", "event_end_s"] == 55.0

    def test_a_counterpart_crossing_the_boundary_still_confirms_an_overlap(self):
        """The counterpart was observed; it need not qualify for the reverse
        direction's denominator to confirm this one."""
        inventory = pd.concat(
            [events("A", [(40, 49)]), events("B", [(45, 55)])], ignore_index=True
        )
        coverage = everywhere("A") | {
            (*KEY, "B"): Coverage(
                observed=np.array([[0.0, 60.0]]), exposure=np.array([[0.0, 50.0]])
            )
        }

        agreement = directional_agreement(
            inventory, overlap_pairs(inventory), coverage
        ).set_index(["source_detector", "target_detector"])

        assert agreement.loc[("A", "B")].n_matched == 1
        assert agreement.loc[("B", "A")].n_eligible == 0

    def test_no_shared_exposure_is_unavailable_not_zero(self):
        inventory = pd.concat(
            [events("A", [(10, 20)]), events("B", [(60, 70)])], ignore_index=True
        )
        coverage = {
            (*KEY, "A"): Coverage(np.array([[0.0, 40.0]]), np.array([[0.0, 40.0]])),
            (*KEY, "B"): Coverage(np.array([[50.0, 90.0]]), np.array([[50.0, 90.0]])),
        }

        agreement = directional_agreement(inventory, overlap_pairs(inventory), coverage)

        assert not agreement.available.any()
        assert agreement.proportion_matched.isna().all()
        assert (agreement.shared_exposure_s == 0).all()

    def test_a_detector_that_found_nothing_but_looked_counts_as_looked(self):
        """B observed the whole epoch and found nothing: A's events are
        unmatched, not unknown, and the proportion is zero, not missing."""
        inventory = events("A", [(10, 20)])

        agreement = directional_agreement(
            inventory, overlap_pairs(inventory), everywhere("A", "B")
        ).set_index(["source_detector", "target_detector"])

        assert agreement.loc[("A", "B")].proportion_matched == 0.0
        assert agreement.loc[("B", "A")].n_source_events == 0
        assert np.isnan(agreement.loc[("B", "A")].proportion_matched)

    def test_a_single_event_cohort(self):
        inventory = pd.concat(
            [events("A", [(10, 20)]), events("B", [(12, 18)])], ignore_index=True
        )

        agreement = directional_agreement(
            inventory, overlap_pairs(inventory), everywhere("A", "B")
        )

        assert (agreement.proportion_matched == 1.0).all()

    def test_events_from_a_detector_with_no_coverage_are_refused(self):
        """Whether it looked is what the denominators need; never assume it."""
        inventory = pd.concat(
            [events("A", [(10, 20)]), events("B", [(10, 20)])], ignore_index=True
        )

        with pytest.raises(ValueError, match="no coverage"):
            directional_agreement(inventory, overlap_pairs(inventory), everywhere("A"))


class TestEventMatches:
    def test_one_row_per_event_and_other_detector(self):
        inventory = pd.concat(
            [events("A", [(10, 20), (30, 40)]), events("B", [(12, 18)])],
            ignore_index=True,
        )

        matches = event_matches(
            inventory, overlap_pairs(inventory), everywhere("A", "B")
        )

        assert list(matches.columns) == list(MATCH_DTYPES)
        assert len(matches) == 3
        by_key = matches.set_index("event_key")
        assert by_key.loc["A:0", "membership"] == "matched"
        assert by_key.loc["A:1", "membership"] == "unmatched"
        assert by_key.loc["A:0", "n_counterparts"] == 1


class TestBridgingEvents:
    def test_an_event_across_its_own_unavailable_data_is_set_aside(self):
        """Its span becomes unavailable, so discarding it cannot manufacture
        an apparent non-detection by the other detector."""
        inventory = pd.concat(
            [events("A", [(10, 30), (60, 70)]), events("B", [(20, 25)])],
            ignore_index=True,
        )
        coverage = everywhere("B") | {
            (*KEY, "A"): Coverage(
                observed=np.array([[0.0, 15.0], [18.0, 100.0]]),
                exposure=np.array([[0.0, 15.0], [18.0, 100.0]]),
            )
        }

        kept, adjusted = remove_bridging_events(inventory, coverage)

        assert kept.event_key.tolist() == ["A:1", "B:0"]
        np.testing.assert_array_equal(
            adjusted[(*KEY, "A")].exposure, [[0.0, 10.0], [30.0, 100.0]]
        )
        # B's event now sits in time A did not observe: unknown, not unmatched
        matches = event_matches(kept, overlap_pairs(kept), adjusted)
        assert matches.set_index("event_key").loc["B:0", "membership"] == "unknown"

    def test_an_event_across_movement_it_observed_is_kept(self):
        """Movement is observed time: such an event is not bridging, it just
        falls outside the exposure and so outside every denominator."""
        inventory = pd.concat(
            [events("A", [(10, 30)]), events("B", [(20, 25)])], ignore_index=True
        )
        coverage = everywhere("B") | {
            (*KEY, "A"): Coverage(
                observed=np.array([[0.0, 100.0]]),
                exposure=np.array([[0.0, 15.0], [18.0, 100.0]]),
            )
        }

        kept, adjusted = remove_bridging_events(inventory, coverage)

        assert kept.event_key.tolist() == ["A:0", "B:0"]
        assert adjusted[(*KEY, "A")] is coverage[(*KEY, "A")]


class TestTimeCoverage:
    def test_coverage_and_counts_disagree_on_a_split_event(self):
        """One detector splits an event into three: counts change, time does not."""
        inventory = pd.concat(
            [events("A", [(10, 40)]), events("B", [(10, 20), (20.5, 30), (30.5, 40)])],
            ignore_index=True,
        )
        coverage = everywhere("A", "B")

        covered = time_coverage(inventory, coverage).iloc[0]
        agreement = directional_agreement(
            inventory, overlap_pairs(inventory), coverage
        ).set_index(["source_detector", "target_detector"])

        assert covered.intersection_s == pytest.approx(29.0)
        assert covered.union_s == pytest.approx(30.0)
        assert covered.intersection_fraction == pytest.approx(29.0 / 100.0)
        assert agreement.loc[("A", "B")].n_eligible == 1
        assert agreement.loc[("B", "A")].n_eligible == 3

    def test_event_time_outside_the_shared_exposure_is_clipped_for_coverage_only(self):
        inventory = pd.concat(
            [events("A", [(40, 60)]), events("B", [(40, 60)])], ignore_index=True
        )
        coverage = everywhere("A") | {
            (*KEY, "B"): Coverage(np.array([[0.0, 50.0]]), np.array([[0.0, 50.0]]))
        }

        covered = time_coverage(inventory, coverage).iloc[0]

        assert covered.shared_exposure_s == pytest.approx(50.0)
        assert covered.intersection_s == pytest.approx(10.0)
        assert covered.union_s == pytest.approx(10.0)

    def test_no_shared_exposure_is_unavailable(self):
        coverage = {
            (*KEY, "A"): Coverage(np.array([[0.0, 40.0]]), np.array([[0.0, 40.0]])),
            (*KEY, "B"): Coverage(np.array([[50.0, 90.0]]), np.array([[50.0, 90.0]])),
        }

        covered = time_coverage(events("A", [(10, 20)]), coverage).iloc[0]

        assert not covered.available
        assert np.isnan(covered.union_fraction)
