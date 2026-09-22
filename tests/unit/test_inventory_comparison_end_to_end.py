"""The inventory comparison run end to end on one simulated epoch.

Coverage is built from samples as it will be on recorded data, then every
step runs in order. The epoch holds a movement bout (20-25 s), a gap in the
spikes (40-41 s) and a gap in the LFP (50-52 s):

- state space ``SS`` needs the spikes;
- memoryless ``MS`` needs the spikes over a whole 80 ms window;
- the SWR detector ``K`` needs the LFP.

Shared exposure, worked by hand: still [0, 20) and [25, 60]; SS observes
[0, 40] and [41, 60]; MS observes [0.04, 39.96] and [41.04, 59.96] and loses
[41.0, 41.2] to its bridging event; K observes [0, 50] and [52, 60]. Their
intersection is [0.04, 19.998], [25, 39.96], [41.2, 50] and [52, 59.96].
"""

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.inventory_comparison.availability import (
    exposure,
    sample_runs,
    still_intervals,
    with_window_support,
)
from continuum_swr_replay.analysis.inventory_comparison.cells import (
    membership_against,
    no_detector_time,
)
from continuum_swr_replay.analysis.inventory_comparison.overlap import (
    Coverage,
    directional_agreement,
    event_matches,
    overlap_pairs,
    remove_bridging_events,
)
from continuum_swr_replay.analysis.inventory_comparison.schema import (
    MEMORYLESS,
    STATE_SPACE,
    SessionIdentity,
    inventory_from_intervals,
)
from continuum_swr_replay.analysis.inventory_comparison.swr_matrix import (
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
    spikes = sample_runs(TIME, ~((TIME > 40.0) & (TIME < 41.0)))
    lfp = sample_runs(TIME, ~((TIME > 50.0) & (TIME < 52.0)))
    observed = {
        STATE_SPACE: spikes,
        MEMORYLESS: with_window_support(spikes, 0.04),
        KAY: lfp,
    }
    still = still_intervals(TIME, SPEED)
    coverage = {
        (*KEY, d): Coverage(observed=o, exposure=exposure(o, still))
        for d, o in observed.items()
    }
    inventory = pd.concat(
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
        ],
        ignore_index=True,
    )
    kept, coverage = remove_bridging_events(inventory, coverage)
    pairs = overlap_pairs(kept)
    return kept, coverage, pairs, event_matches(kept, pairs, coverage)


def test_only_the_event_across_its_own_window_gap_is_set_aside(comparison):
    kept, coverage, _, _ = comparison

    assert key(MEMORYLESS, 1) not in set(kept.event_key)
    assert len(kept) == sum(map(len, EVENTS.values())) - 1
    np.testing.assert_allclose(
        coverage[(*KEY, MEMORYLESS)].observed, [[0.04, 39.96], [41.2, 59.96]]
    )


def test_denominators_leave_out_movement_and_unobserved_time(comparison):
    kept, coverage, pairs, _ = comparison

    agreement = directional_agreement(kept, pairs, coverage).set_index(
        ["source_detector", "target_detector"]
    )

    ss_to_ms = agreement.loc[(STATE_SPACE, MEMORYLESS)]
    # SS:2 lies where MS did not observe, SS:4 during movement
    assert (ss_to_ms.n_source_events, ss_to_ms.n_eligible, ss_to_ms.n_matched) == (
        5,
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
        key(STATE_SPACE, 4): "unknown",  # the animal is moving
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
        key(STATE_SPACE, 4): "unknown",
    }


def test_no_detector_time_is_the_hand_worked_shared_exposure_less_detections(
    comparison,
):
    kept, coverage, _, _ = comparison

    row = no_detector_time(
        kept, coverage, detectors=[STATE_SPACE, MEMORYLESS, KAY]
    ).iloc[0]

    shared = (19.998 - 0.04) + (39.96 - 25.0) + (50.0 - 41.2) + (59.96 - 52.0)
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
    matrix = swr_label_matrix(labels).set_index("cohort")

    assert labels.set_index("event_key").loc[key(STATE_SPACE, 3), "label"] == "unknown"
    ss = matrix.loc[STATE_SPACE]
    assert (ss.n_events, ss.n_eligible, ss.n_swr, ss.n_non_swr) == (5, 3, 1, 2)
