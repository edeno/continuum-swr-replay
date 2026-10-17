"""The inventory comparison run end to end on one simulated epoch.

Coverage is built from samples as it will be on recorded data, then every
step runs in order. The epoch holds a movement bout (20-25 s), a gap in the
spikes (40-41 s) and a gap in the LFP (50-52 s):

- state space ``SS`` needs the spikes;
- memoryless ``MS`` needs the spikes over a whole 80 ms window;
- the SWR detector ``K`` needs the LFP.

Shared exposure, worked by hand: still [0, 20.0] and [24.998, 60], each still
run reaching the first and last moving samples (20.0 and 24.998 s) beside it;
SS observes [0, 40] and [41, 60]; MS, whose 40-sample windows are stamped 20
samples in, observes [0.04, 39.962] and [41.04, 59.962] and loses [41.0, 41.2]
to its bridging event; K observes [0, 50] and [52, 60]. Their intersection is
[0.04, 20.0], [24.998, 39.962], [41.2, 50] and [52, 59.962]. SS's event during
the movement bout is set aside before any comparison.
"""

import numpy as np
import pytest

from continuum_swr_replay.analysis.inventory_comparison.availability import (
    coverage_from_observations,
    exposure,
    observation_table,
    sample_runs,
    still_intervals,
    window_supported,
)
from continuum_swr_replay.analysis.inventory_comparison.cells import (
    membership_against,
    no_detector_time,
)
from continuum_swr_replay.analysis.inventory_comparison.components import (
    overlap_components,
)
from continuum_swr_replay.analysis.inventory_comparison.overlap import (
    Coverage,
    directional_agreement,
    event_matches,
    overlap_pairs,
    remove_bridging_events,
    time_coverage,
)
from continuum_swr_replay.analysis.inventory_comparison.recount import (
    merge_fragments,
)
from continuum_swr_replay.analysis.inventory_comparison.schema import (
    MEMORYLESS,
    STATE_SPACE,
    SessionIdentity,
    combine_inventories,
    inventory_from_intervals,
)
from continuum_swr_replay.analysis.inventory_comparison.swr_matrix import (
    paired_labels,
    swr_label_matrix,
    swr_labels,
)

EPOCH = SessionIdentity(animal="sim", day=20200101, epoch="02_r1", brain_area="HPC")
KEY = ("sim", 20200101, "02_r1")
TIME = np.arange(30_001) * 0.002
SPEED = np.where((TIME >= 20.0) & (TIME < 25.0), 10.0, 1.0)
KAY = "kay"
EVENTS = {
    STATE_SPACE: [(5.0, 5.3), (10.0, 10.2), (41.1, 41.15), (51.0, 51.2), (21.0, 21.3)],
    MEMORYLESS: [(5.1, 5.4), (41.0, 41.2)],
    KAY: [(5.05, 5.1), (30.0, 30.1)],
}


def key(detector, native):
    """The event key of an event in the test epoch."""
    return f"{detector}:sim20200101_02_r1:{native}"


@pytest.fixture(scope="module")
def comparison():
    spikes = ~((TIME > 40.0) & (TIME < 41.0))
    lfp = ~((TIME > 50.0) & (TIME < 52.0))
    observed = {
        STATE_SPACE: sample_runs(TIME, spikes),
        MEMORYLESS: sample_runs(TIME, window_supported(TIME, spikes, 40)),
        KAY: sample_runs(TIME, lfp),
    }
    still = still_intervals(TIME, SPEED)
    coverage = {
        (*KEY, d): Coverage(observed=o, exposure=exposure(o, still))
        for d, o in observed.items()
    }
    inventory, moving = combine_inventories(
        [
            inventory_from_intervals(
                np.array([a for a, _ in spans]),
                np.array([b for _, b in spans]),
                native_keys=np.arange(len(spans)),
                detector=detector,
                session=EPOCH,
                run_id="run",
                speed_time_s=TIME,
                speed_cm_per_s=SPEED,
            )
            for detector, spans in EVENTS.items()
        ]
    )
    assert moving.event_key.tolist() == [key(STATE_SPACE, 4)]
    kept, coverage, bridging = remove_bridging_events(inventory, coverage)
    assert bridging.event_key.tolist() == [key(MEMORYLESS, 1)]
    pairs = overlap_pairs(kept)
    return kept, coverage, pairs, event_matches(kept, pairs, coverage)


def test_only_the_event_across_its_own_window_gap_is_set_aside(comparison):
    kept, coverage, _, _ = comparison

    assert key(MEMORYLESS, 1) not in set(kept.event_key)
    assert len(kept) == sum(map(len, EVENTS.values())) - 2  # and SS:4, moving
    np.testing.assert_allclose(
        coverage[(*KEY, MEMORYLESS)].observed, [[0.04, 39.962], [41.2, 59.962]]
    )


def test_denominators_leave_out_movement_and_unobserved_time(comparison):
    kept, coverage, pairs, _ = comparison

    agreement = directional_agreement(kept, pairs, coverage).set_index(
        ["source_detector", "target_detector"]
    )

    ss_to_ms = agreement.loc[(STATE_SPACE, MEMORYLESS)]
    # SS:2 lies where MS did not observe; SS:4, during movement, never entered
    assert (ss_to_ms.n_source_events, ss_to_ms.n_eligible, ss_to_ms.n_matched) == (
        4,
        3,
        1,
    )
    ms_to_ss = agreement.loc[(MEMORYLESS, STATE_SPACE)]
    assert (ms_to_ss.n_eligible, ms_to_ss.n_matched) == (1, 1)


def test_the_cells_separate_unmatched_from_unknown(comparison):
    kept, _, _, matches = comparison
    everyone = [STATE_SPACE, MEMORYLESS, KAY]

    only_one = membership_against(kept, matches, sources=everyone, against=everyone)
    swr_without_sequence = membership_against(
        kept, matches, sources=[KAY], against=[STATE_SPACE, MEMORYLESS]
    )
    neither = membership_against(kept, matches, sources=[STATE_SPACE], against=[KAY])

    assert only_one.set_index("event_key").membership.to_dict() == {
        key(STATE_SPACE, 0): "matched",
        key(STATE_SPACE, 1): "unmatched",
        key(STATE_SPACE, 2): "unknown",  # MS lost this time to its bridging event
        key(STATE_SPACE, 3): "unknown",  # the LFP is missing
        key(MEMORYLESS, 0): "matched",
        key(KAY, 0): "matched",
        key(KAY, 1): "unmatched",
    }
    assert swr_without_sequence.set_index("event_key").membership.to_dict() == {
        key(KAY, 0): "matched",
        key(KAY, 1): "unmatched",
    }
    # SS:2 is where MS lost coverage, but K observed it: the neither cell is
    # asked of the triggered inventory alone
    assert neither.set_index("event_key").membership.to_dict() == {
        key(STATE_SPACE, 0): "matched",
        key(STATE_SPACE, 1): "unmatched",
        key(STATE_SPACE, 2): "unmatched",
        key(STATE_SPACE, 3): "unknown",
    }


def test_no_detector_time_is_the_hand_worked_shared_exposure_less_detections(
    comparison,
):
    kept, coverage, _, _ = comparison

    row = no_detector_time(
        kept, coverage, detectors=[STATE_SPACE, MEMORYLESS, KAY]
    ).iloc[0]

    shared = (20.0 - 0.04) + (39.962 - 24.998) + (50.0 - 41.2) + (59.962 - 52.0)
    # SS:0/MS:0/K:0 cover 5.0-5.4, SS:1 0.2 s, K:1 0.1 s; the rest is outside
    assert row.shared_exposure_s == pytest.approx(shared)
    assert row.detected_s == pytest.approx(0.7)
    assert row.undetected_s == pytest.approx(shared - 0.7)


def test_missing_lfp_leaves_the_matrix_rather_than_reading_as_non_swr(comparison):
    kept, coverage, _, matches = comparison

    labels = swr_labels(
        kept,
        matches,
        coverage,
        cohorts=[STATE_SPACE, MEMORYLESS],
        swr_definitions=[KAY],
    )
    matrix = swr_label_matrix(
        labels, coverage, cohorts=[STATE_SPACE, MEMORYLESS], swr_definitions=[KAY]
    ).set_index("cohort")

    assert labels.set_index("event_key").loc[key(STATE_SPACE, 3), "label"] == "unknown"
    ss = matrix.loc[STATE_SPACE]
    assert (ss.n_events, ss.n_eligible, ss.n_swr, ss.n_non_swr) == (4, 3, 1, 2)


def test_time_coverage_is_clipped_to_the_shared_exposure(comparison):
    """SS and K share exposure over [0, 20], [24.998, 40], [41, 50] and [52, 60]:
    SS:0 with K:0, SS:1, SS:2 and K:1 lie inside it; SS:3, during the LFP
    gap, does not."""
    kept, coverage, _, _ = comparison

    covered = time_coverage(kept, coverage).set_index(["detector_a", "detector_b"])

    ss_k = covered.loc[(KAY, STATE_SPACE)]
    assert ss_k.available
    assert ss_k.union_s == pytest.approx(0.3 + 0.2 + 0.05 + 0.1)
    assert ss_k.intersection_s == pytest.approx(0.05)


def test_the_components_join_the_three_detectors_at_five_seconds(comparison):
    kept, _, pairs, _ = comparison

    labelled, components = overlap_components(kept, pairs)

    three = components[components.n_detectors == 3]
    assert len(three) == 1
    assert three.iloc[0].detectors == f"{KAY}+{MEMORYLESS}+{STATE_SPACE}"
    assert (three.iloc[0].span_start_s, three.iloc[0].span_end_s) == (5.0, 5.4)
    assert three.iloc[0].n_pairs == 2  # SS:0-MS:0 and SS:0-K:0; MS:0 and K:0 touch
    assert len(components) == len(kept) - 2


def test_the_recount_leaves_this_epoch_alone_and_stays_inside_exposure(comparison):
    """No two state-space events lie within 50 ms; the merged table is the
    native one, and it runs on the bridging-trimmed coverage without error."""
    kept, coverage, pairs, _ = comparison

    merged = merge_fragments(
        kept, coverage, detector=STATE_SPACE, gap_s=0.05, sample_interval_s=0.002
    )

    assert len(merged) == len(kept)
    assert (merged.n_fragments == 1).all()


def test_a_matched_sequence_pair_can_disagree_on_the_swr_label(comparison):
    """SS:0 overlaps K:0; MS:0 only touches it at 5.1 s, which is not an
    overlap, so under the same definition the two matched events carry
    different labels and the table says so rather than resolving it."""
    kept, coverage, pairs, matches = comparison
    labels = swr_labels(
        kept,
        matches,
        coverage,
        cohorts=[STATE_SPACE, MEMORYLESS],
        swr_definitions=[KAY],
    )

    paired = paired_labels(pairs, labels, cohorts=[STATE_SPACE, MEMORYLESS])

    assert len(paired) == 1
    row = paired.iloc[0]
    assert (row.label_a, row.label_b, row.agreement) == ("non-SWR", "SWR", "disagree")


def test_coverage_built_from_observation_tables_is_the_coverage_the_counts_read():
    """The converter reproduces the sample-built coverage before bridging
    events are removed, so both representations share one origin."""
    spikes = ~((TIME > 40.0) & (TIME < 41.0))
    observed = {
        STATE_SPACE: sample_runs(TIME, spikes),
        MEMORYLESS: sample_runs(TIME, window_supported(TIME, spikes, 40)),
    }
    still = still_intervals(TIME, SPEED)
    tables = [
        observation_table(
            EPOCH,
            detector=detector,
            run_id="run",
            epoch_bounds=(TIME[0], TIME[-1]),
            observed=runs,
            unavailable_reason="spikes missing",
        )
        for detector, runs in observed.items()
    ]

    coverage = coverage_from_observations(
        __import__("pandas").concat(tables, ignore_index=True), {KEY: still}
    )

    for detector, runs in observed.items():
        entry = coverage[(*KEY, detector)]
        np.testing.assert_allclose(entry.observed, runs)
        np.testing.assert_allclose(entry.exposure, exposure(runs, still))
