"""Every overlap between inventories, and the counts built on it."""

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
    flag_pairs,
    overlap_pairs,
    remove_bridging_events,
    time_coverage,
    validate_pairs,
)
from continuum_swr_replay.analysis.inventory_comparison.schema import (
    SessionIdentity,
    inventory_from_intervals,
)

GRID = np.arange(0, 100.001, 0.002)
EPOCH = SessionIdentity(animal="sim", day=20200101, epoch="02_r1", brain_area="HPC")
OTHER = SessionIdentity(animal="sim", day=20200101, epoch="04_r2", brain_area="HPC")
KEY = ("sim", 20200101, "02_r1")


def key(detector, native):
    """The event key of an event in the test epoch."""
    return f"{detector}:sim20200101_02_r1:{native}"


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
        assert overlap_pairs(events("A", [(10, 20), (20, 25)])).empty

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
        """A chain in which the middle events each overlap two of the other
        detector's: four pairs, but three A events and two B events."""
        inventory = pd.concat(
            [
                events("A", [(0, 10), (10, 20), (20, 30)]),
                events("B", [(5, 15), (15, 25)]),
            ],
            ignore_index=True,
        )

        agreement = directional_agreement(
            inventory, overlap_pairs(inventory), everywhere("A", "B")
        ).set_index(["source_detector", "target_detector"])

        assert len(overlap_pairs(inventory)) == 4
        assert agreement.loc[("A", "B")].n_matched == 3
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
            (matches.event_key == key("A", 1)) & (matches.other_detector == "B")
        ]
        assert crossing.membership.item() == "unknown"
        assert inventory.set_index("event_key").loc[key("A", 1), "event_end_s"] == 55.0

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

    def test_a_detector_that_did_not_run_in_an_epoch_is_unavailable_there(self):
        """Its absence is reported, not left out of the pooled counts."""
        inventory = pd.concat(
            [events("A", [(10, 20)]), events("A", [(10, 20)], session=OTHER)],
            ignore_index=True,
        )
        coverage = everywhere("A", "B") | everywhere(
            "A", session_key=("sim", 20200101, "04_r2")
        )
        pairs = overlap_pairs(inventory)

        agreement = directional_agreement(inventory, pairs, coverage)
        covered = time_coverage(inventory, coverage)
        matches = event_matches(inventory, pairs, coverage)

        missing = agreement[(agreement.epoch == "04_r2")].set_index("source_detector")
        assert not missing.loc["A", "available"]
        assert missing.loc["A", "n_source_events"] == 1
        assert np.isnan(missing.loc["A", "proportion_matched"])
        assert not covered.set_index("epoch").loc["04_r2", "available"]
        assert matches[matches.epoch == "04_r2"].membership.tolist() == ["unknown"]

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


class TestFlagPairs:
    def test_each_side_says_whether_it_lies_within_the_shared_exposure(self):
        """A pair is kept however its events sit; the flags say which of its
        two events could enter a denominator."""
        inventory = pd.concat(
            [events("A", [(10, 20)]), events("B", [(15, 30)])], ignore_index=True
        )
        # B observed the whole epoch but was exposed (still) only to 25 s, so
        # its event is a valid event outside the shared exposure
        coverage = everywhere("A") | {
            (*KEY, "B"): Coverage(np.array([[0.0, 100.0]]), np.array([[0.0, 25.0]]))
        }
        pairs = overlap_pairs(inventory)

        flagged = flag_pairs(pairs, event_matches(inventory, pairs, coverage))

        assert list(flagged.columns) == [
            *pairs.columns,
            "a_within_shared_exposure",
            "b_within_shared_exposure",
        ]
        assert flagged[
            ["a_within_shared_exposure", "b_within_shared_exposure"]
        ].values.tolist() == [[True, False]]


class TestEventMatches:
    def test_native_keys_repeating_across_epochs_never_match_across_them(self):
        """SWR labels restart in every epoch; an overlap in one epoch must not
        mark the same-numbered events of another."""
        inventory = pd.concat(
            [
                events("A", [(10, 20)]),
                events("K", [(12, 14)]),
                events("A", [(50, 60)], session=OTHER),
                events("K", [(80, 90)], session=OTHER),
            ],
            ignore_index=True,
        )
        coverage = everywhere("A", "K") | everywhere(
            "A", "K", session_key=("sim", 20200101, "04_r2")
        )

        matches = event_matches(inventory, overlap_pairs(inventory), coverage)
        agreement = directional_agreement(inventory, overlap_pairs(inventory), coverage)

        by_epoch = matches.groupby("epoch").membership.agg(sorted).to_dict()
        assert by_epoch == {
            "02_r1": ["matched", "matched"],
            "04_r2": ["unmatched", "unmatched"],
        }
        assert agreement.groupby("epoch").n_matched.sum().to_dict() == {
            "02_r1": 2,
            "04_r2": 0,
        }

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
        assert by_key.loc[key("A", 0), "membership"] == "matched"
        assert by_key.loc[key("A", 1), "membership"] == "unmatched"
        assert by_key.loc[key("A", 0), "n_counterparts"] == 1


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

        kept, adjusted, bridging = remove_bridging_events(inventory, coverage)

        assert kept.event_key.tolist() == [key("A", 1), key("B", 0)]
        # flagged and counted, not silently dropped
        assert bridging.event_key.tolist() == [key("A", 0)]
        np.testing.assert_array_equal(
            adjusted[(*KEY, "A")].exposure, [[0.0, 10.0], [30.0, 100.0]]
        )
        # B's event now sits in time A did not observe: unknown, not unmatched
        matches = event_matches(kept, overlap_pairs(kept), adjusted)
        assert (
            matches.set_index("event_key").loc[key("B", 0), "membership"] == "unknown"
        )

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

        kept, adjusted, bridging = remove_bridging_events(inventory, coverage)

        assert kept.event_key.tolist() == [key("A", 0), key("B", 0)]
        assert bridging.empty
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
            (*KEY, "B"): Coverage(np.array([[0.0, 100.0]]), np.array([[0.0, 50.0]]))
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


class TestContracts:
    """What the counting functions refuse, so a stale or cross-wired input
    cannot inflate a count."""

    def test_coverage_is_read_only_and_never_compares_by_accident(self):
        coverage = Coverage(np.array([[0.0, 100.0]]), np.array([[0.0, 100.0]]))

        with pytest.raises(ValueError):
            coverage.exposure[0, 1] = 500.0
        with pytest.raises(ValueError, match="inverted"):
            Coverage(np.array([[10.0, 0.0]]), np.array([[0.0, 1.0]]))
        # identity, not array truth, so comparing two never raises
        assert coverage != Coverage(np.array([[0.0, 100.0]]), np.array([[0.0, 100.0]]))

    def test_a_pair_table_built_before_bridging_events_were_removed_is_refused(self):
        """Its neighbours would otherwise read as matched."""
        inventory = pd.concat(
            [events("A", [(39.9, 41.1)]), events("B", [(41.05, 41.2)])],
            ignore_index=True,
        )
        coverage = everywhere("B") | {
            (*KEY, "A"): Coverage(
                np.array([[0.0, 40.0], [41.0, 100.0]]),
                np.array([[0.0, 40.0], [41.0, 100.0]]),
            )
        }
        stale = overlap_pairs(inventory)
        kept, adjusted, _ = remove_bridging_events(inventory, coverage)

        with pytest.raises(ValueError, match="bridging"):
            event_matches(kept, stale, adjusted)
        fresh = event_matches(kept, overlap_pairs(kept), adjusted)
        assert fresh.membership.tolist() == ["unknown"]

    def test_an_event_outside_its_own_observed_time_is_refused(self):
        inventory = events("A", [(10, 20)])
        coverage = {
            (*KEY, "A"): Coverage(np.array([[0.0, 15.0]]), np.array([[0.0, 15.0]]))
        }

        with pytest.raises(ValueError, match="remove_bridging_events"):
            event_matches(inventory, overlap_pairs(inventory), coverage)

    def test_a_pair_between_two_events_of_one_detector_is_refused(self):
        inventory = pd.concat(
            [events("A", [(10, 20), (30, 40)]), events("B", [(12, 14)])],
            ignore_index=True,
        )
        pairs = overlap_pairs(inventory)
        forged = pairs.assign(
            detector_b="A", event_key_b=key("A", 1), intersection_s=1.0
        )

        with pytest.raises(ValueError, match="detector"):
            validate_pairs(forged, inventory)

    def test_the_roster_can_be_named_and_a_name_outside_it_is_refused(self):
        """A detector that produced nothing anywhere must not vanish: name it,
        or register it with empty coverage, and it reads as unavailable."""
        inventory = events("A", [(10, 20)])
        coverage = everywhere("A")

        agreement = directional_agreement(
            inventory,
            overlap_pairs(inventory),
            coverage | {(*KEY, "B"): Coverage(np.empty((0, 2)), np.empty((0, 2)))},
            detectors=["A", "B"],
        ).set_index(["source_detector", "target_detector"])

        assert not agreement.loc[("A", "B")].available
        assert agreement.loc[("A", "B")].n_source_events == 1
        with pytest.raises(ValueError, match="roster"):
            event_matches(
                inventory, overlap_pairs(inventory), coverage, detectors=["A", "b"]
            )


def brute_force_pairs_and_agreement(spans, exposures):
    """Pairs and per-direction counts by double loop, independent of the code."""
    pairs = set()
    for a, b in ((x, y) for x in spans for y in spans if x < y):
        for i, (s1, e1) in enumerate(spans[a]):
            for j, (s2, e2) in enumerate(spans[b]):
                if min(e1, e2) - max(s1, s2) > 0:
                    pairs.add((a, i, b, j))

    def inside(interval, exposure):
        return any(lo <= interval[0] and interval[1] <= hi for lo, hi in exposure)

    agreement = {}
    for source in spans:
        for target in spans:
            if source == target:
                continue
            shared = [
                (max(lo1, lo2), min(hi1, hi2))
                for lo1, hi1 in exposures[source]
                for lo2, hi2 in exposures[target]
                if min(hi1, hi2) > max(lo1, lo2)
            ]
            eligible = [
                i for i, span in enumerate(spans[source]) if inside(span, shared)
            ]
            matched = [
                i
                for i in eligible
                if any(
                    (source, i, target, j) in pairs or (target, j, source, i) in pairs
                    for j in range(len(spans[target]))
                )
            ]
            agreement[(source, target)] = (
                len(spans[source]),
                len(eligible),
                len(matched),
            )
    return pairs, agreement


@pytest.mark.parametrize("seed", range(25))
def test_pairs_and_agreement_match_a_brute_force_double_loop(seed):
    """The denominators are the easiest thing to get silently wrong; this
    guards `_counterparts` and the within/matched split against a change."""
    rng = np.random.default_rng(seed)
    spans, exposures = {}, {}
    for detector in ("A", "B", "C"):
        edges = np.sort(rng.uniform(0.0, 100.0, size=2 * rng.integers(0, 8)))
        spans[detector] = [
            (float(a), float(b))
            for a, b in zip(edges[::2], edges[1::2], strict=True)
            if b > a
        ]
        hole = np.sort(rng.uniform(0.0, 100.0, size=2))
        exposures[detector] = [(0.0, float(hole[0])), (float(hole[1]), 100.0)]
    inventory = pd.concat(
        [events(d, s) if s else events(d, []) for d, s in spans.items()],
        ignore_index=True,
    )
    coverage = {
        (*KEY, d): Coverage(np.array([[0.0, 100.0]]), np.array(exposures[d]))
        for d in spans
    }
    pairs = overlap_pairs(inventory)
    expected_pairs, expected = brute_force_pairs_and_agreement(spans, exposures)

    got_pairs = {
        (a, int(ka.rsplit(":", 1)[1]), b, int(kb.rsplit(":", 1)[1]))
        for a, ka, b, kb in zip(
            pairs.detector_a,
            pairs.event_key_a,
            pairs.detector_b,
            pairs.event_key_b,
            strict=True,
        )
    }
    assert got_pairs == expected_pairs
    agreement = directional_agreement(inventory, pairs, coverage).set_index(
        ["source_detector", "target_detector"]
    )
    for (source, target), (n_source, n_eligible, n_matched) in expected.items():
        row = agreement.loc[(source, target)]
        assert (row.n_source_events, row.n_eligible, row.n_matched) == (
            n_source,
            n_eligible,
            n_matched,
        ), (source, target)
