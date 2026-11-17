"""What the two detectors made of a labeled recording, on one small enough to check by hand.

The recording has 30 windows, 10 ms apart, decoded over 11 place bins 10 cm
apart (bin ``i`` at ``10 i`` cm). The animal sits at 50 cm until 1.2 s and at
25 cm from then on. Four candidate sequences:

====  =======  ===============================  ==========  ========
seq   windows  kept centers (bins)              dispersion  accepted
====  =======  ===============================  ==========  ========
A     1-8      5 5 . 10 . 6 7 8 (3, 5 not kept)  17.80 cm   yes (fails the shuffle)
B     10-13    0 1 2 3                          12.25 cm    no: 3 strides, too short
C     15-19    5 5 5 5 5                        0 cm        no: too little dispersion
D     21-26    2 3 4 5 6 7                      17.80 cm    yes (passes)
====  =======  ===============================  ==========  ========

A is built from two runs, windows 1-2 and 6-8; window 4 is kept but its
single-window run was dropped before the merge, so it is outside every piece.
Without it, A's dispersion is sqrt(140) = 11.83 cm.
"""

from __future__ import annotations

import numpy as np
import pytest

from continuum_swr_replay.analysis.detection.memoryless_sequence import (
    recording_statistics as rs,
)
from continuum_swr_replay.analysis.detection.memoryless_sequence.comparison_simulation import (
    FRAGMENTED_EVENT,
    REMOTE_STATIONARY_EVENT,
    SHORT_PATH_EVENT,
    SHORT_REPLAY_EVENT,
    SILENCE_EVENT,
    SWEEP_EVENT,
)
from continuum_swr_replay.analysis.detection.memoryless_sequence.config import (
    MemorylessSequenceConfig,
)
from continuum_swr_replay.analysis.detection.memoryless_sequence.detect import (
    CandidateSequences,
    SequenceAssembly,
    WindowDecisions,
    WindowGates,
    replay_mask,
    sequence_dispersion_cm,
    sequence_duration_s,
)
from continuum_swr_replay.analysis.detection.memoryless_sequence.shuffle import (
    ShuffleResult,
)
from continuum_swr_replay.analysis.detection.memoryless_sequence.window_statistics import (
    NO_CENTER,
)

# 2 samples a window, 1 a stride; a replay needs more than 3 strides
CONFIG = MemorylessSequenceConfig(
    sampling_frequency=100.0,
    window_duration_s=0.02,
    stride_s=0.01,
    minimum_duration_s=0.03,
)
N_WINDOWS = 30
MIDPOINT_S = 1.0 + 0.01 * np.arange(N_WINDOWS)
PLACE_BIN_CENTERS_CM = 10.0 * np.arange(11)
BIN_DISTANCE_CM = np.abs(np.subtract.outer(PLACE_BIN_CENTERS_CM, PLACE_BIN_CENTERS_CM))

CENTERS = {1: 5, 2: 5, 3: 5, 4: 10, 5: 6, 6: 6, 7: 7, 8: 8}
CENTERS |= {10: 0, 11: 1, 12: 2, 13: 3}
CENTERS |= dict.fromkeys(range(15, 20), 5)
CENTERS |= dict(zip(range(21, 27), range(2, 8), strict=True))
NOT_KEPT = (3, 5)
RUNS = [(1, 8), (10, 13), (15, 19), (21, 26)]
PRE_MERGE_RUNS = [(1, 2), (6, 8), (10, 13), (15, 19), (21, 26)]
PIECES = ([0, 1], [2], [3], [4])
A, B, C, D = range(4)


def _candidates() -> CandidateSequences:
    center = np.full(N_WINDOWS, NO_CENTER)
    for window, bin_index in CENTERS.items():
        center[window] = bin_index
    has_center = center != NO_CENTER
    # a window with a center is kept unless its jump fails
    jump_ok = has_center.copy()
    jump_ok[list(NOT_KEPT)] = False
    gates = WindowGates(
        posterior_usable=has_center,
        speed_eligible=np.ones(N_WINDOWS, dtype=bool),
        spread_below_threshold=np.ones(N_WINDOWS, dtype=bool),
        jump_below_threshold=jump_ok,
    )
    decisions = WindowDecisions(
        center_index=center,
        spread_cm=np.where(has_center, 1.0, np.nan),
        jump_cm=np.zeros(N_WINDOWS),
        gates=gates,
    )
    assembly = SequenceAssembly(
        runs=np.array(RUNS),
        pieces=tuple(np.array(piece) for piece in PIECES),
        pre_merge_runs=np.array(PRE_MERGE_RUNS),
        kept=gates.kept,
    )
    runs = np.array(RUNS)
    dispersion = sequence_dispersion_cm(runs, gates.kept, center, BIN_DISTANCE_CM)
    return CandidateSequences(
        runs=runs,
        duration_s=sequence_duration_s(runs, MIDPOINT_S),
        dispersion_cm=dispersion,
        is_replay=replay_mask(runs, dispersion, CONFIG),
        decisions=decisions,
        assembly=assembly,
        midpoint_s=MIDPOINT_S,
        config=CONFIG,
    )


def _shuffle(n_below_spread: int, n_below_jump: int, n: int = 19) -> ShuffleResult:
    """A shuffle with observed statistics of 5 and that many draws below them;
    the others are above, one of them tied with the observed value, which does
    not count against it."""

    def draws(n_below: int) -> np.ndarray:
        return np.array([1.0] * n_below + [5.0] + [10.0] * (n - n_below - 1))

    return ShuffleResult(5.0, 5.0, draws(n_below_spread), draws(n_below_jump))


# event rows, in the simulation's order (comparison_simulation's *_EVENT)
EVENT_TIMES = np.array(
    [
        (1.03, 1.05),  # sweep, inside A
        (1.135, 1.145),  # fragmented, between B and C: no candidate
        (1.16, 1.17),  # remote stationary, inside C
        (1.12, 1.16),  # silence, over B's end and C's start
        (1.105, 1.115),  # short path, inside B
        (1.23, 1.235),  # short replay, inside D
        (1.30, 1.31),  # far replay: no candidate, one exact state-space interval
        (1.40, 1.41),  # sparse replay: the same
        (1.50, 1.51),  # fast replay: the same
    ]
)
STATE_SPACE_INTERVALS = [
    (1.02, 1.06),
    (1.10, 1.108),
    (1.110, 1.13),
    (1.16, 1.18),
    (1.225, 1.24),
    (1.30, 1.31),
    (1.40, 1.41),
    (1.50, 1.51),
]


@pytest.fixture
def decoded() -> dict:
    position = np.where(MIDPOINT_S < 1.2 - 1e-9, 50.0, 25.0)
    return {
        "time": MIDPOINT_S,
        "position": position,
        "midpoint_s": MIDPOINT_S,
        "place_bin_centers_cm": PLACE_BIN_CENTERS_CM,
        "bin_distance_cm": BIN_DISTANCE_CM,
        "candidates": _candidates(),
        "event_times": EVENT_TIMES,
        # A has 2 draws below its spread; D none
        "shuffles": {A: _shuffle(2, 0), D: _shuffle(0, 0)},
        "state_space_intervals": STATE_SPACE_INTERVALS,
    }


def test_the_recording_is_the_one_described(decoded):
    candidates = decoded["candidates"]
    assert candidates.is_replay.tolist() == [True, False, False, True]
    assert candidates.dispersion_cm == pytest.approx(
        [np.sqrt(1900 / 6), np.sqrt(600 / 4), 0.0, np.sqrt(1900 / 6)]
    )
    assert candidates.duration_s == pytest.approx([0.07, 0.03, 0.04, 0.05])


def test_overlap_counts_any_shared_time(decoded):
    assert rs.candidates_overlapping(decoded, SWEEP_EVENT) == [A]
    assert rs.candidates_overlapping(decoded, SILENCE_EVENT) == [B, C]
    assert rs.candidates_overlapping(decoded, FRAGMENTED_EVENT) == []
    assert rs.accepted_overlapping(decoded, SILENCE_EVENT) == []
    assert rs.accepted_overlapping(decoded, SWEEP_EVENT) == [A]
    assert rs.reported_overlapping(decoded, SWEEP_EVENT) == []
    assert rs.reported_overlapping(decoded, SHORT_REPLAY_EVENT) == [D]
    assert rs.is_reported(decoded).tolist() == [False, False, False, True]


@pytest.mark.parametrize(
    ("event", "expected"),
    [
        (SWEEP_EVENT, "failed the shuffle"),
        (SHORT_REPLAY_EVENT, "found"),
        (SHORT_PATH_EVENT, "declined"),
        (REMOTE_STATIONARY_EVENT, "declined"),
        (FRAGMENTED_EVENT, "missed"),
    ],
)
def test_verdict(decoded, event, expected):
    assert rs.verdict(decoded, event) == expected


def test_duration_ratio_is_the_reported_over_the_true_duration(decoded):
    # A spans 70 ms over a 20 ms event; D 50 ms over a 5 ms one
    assert rs.duration_ratio(decoded, SWEEP_EVENT) == pytest.approx(3.5)
    assert rs.duration_ratio(decoded, SHORT_REPLAY_EVENT) == pytest.approx(10.0)


def test_state_space_duration_ratio_spans_the_reports_over_the_event(decoded):
    # one 40 ms interval over the 20 ms sweep
    assert rs.state_space_duration_ratio(decoded, SWEEP_EVENT) == pytest.approx(2.0)
    # two intervals over the 10 ms short path, 1.10 to 1.13 s
    assert rs.state_space_duration_ratio(decoded, SHORT_PATH_EVENT) == pytest.approx(
        3.0
    )
    # none over the fragmented event
    assert np.isnan(rs.state_space_duration_ratio(decoded, FRAGMENTED_EVENT))


def test_the_longest_accepted_sequence_is_the_one_read(decoded):
    # an event over both accepted sequences reads A, the longer
    decoded["event_times"] = np.array([(1.05, 1.25)])
    assert rs.duration_ratio(decoded, 0) == pytest.approx(0.07 / 0.2)
    assert rs.shuffle_of(decoded, 0) is decoded["shuffles"][A]


@pytest.mark.parametrize(
    "read",
    [rs.duration_ratio, rs.shuffle_of, rs.local_fraction_of_event],
)
def test_an_event_no_accepted_sequence_overlaps_is_refused(decoded, read):
    with pytest.raises(RuntimeError, match="No accepted sequence overlaps event 4"):
        read(decoded, SHORT_PATH_EVENT)


def test_local_fraction_counts_kept_windows_closer_than_15_cm(decoded):
    # A's kept windows decode 50, 50, 100, 60, 70, 80 cm with the animal at 50:
    # 0, 0 and 10 cm away are local, 20 cm is not; windows 3 and 5 are not kept
    assert rs.local_fraction(decoded, A) == 0.5
    # D's decode 20 to 70 cm with the animal at 25: 5 and 5 cm away are local,
    # 15 cm is not
    assert rs.local_fraction(decoded, D) == pytest.approx(2 / 6)
    assert rs.local_fraction_of_event(decoded, SHORT_REPLAY_EVENT) == (
        pytest.approx(2 / 6)
    )


def test_decoded_center_is_nan_where_a_window_has_none(decoded):
    center = rs.decoded_center_cm(decoded)
    assert center[4] == 100.0
    assert center[3] == 50.0
    assert np.isnan(center[0])
    assert np.isnan(center[9])


def test_best_dispersion_is_the_largest_over_every_candidate(decoded):
    assert rs.best_dispersion_cm(decoded, SILENCE_EVENT) == pytest.approx(np.sqrt(150))
    assert rs.best_dispersion_cm(decoded, REMOTE_STATIONARY_EVENT) == 0.0
    assert np.isnan(rs.best_dispersion_cm(decoded, FRAGMENTED_EVENT))


@pytest.mark.parametrize(
    ("n", "n_below", "expected"),
    [
        # of 19 draws a pass allows none below; of 99, four
        (19, 0, 0),
        (19, 2, 2),
        (99, 4, 0),
        (99, 6, 2),
    ],
)
def test_draws_from_passing(n, n_below, expected):
    assert rs.draws_from_passing(_shuffle(n_below, 0, n)) == expected
    # the worse half decides
    assert rs.draws_from_passing(_shuffle(0, n_below, n)) == expected


def test_bound_error_is_the_worst_edge_of_the_first_and_last_interval(decoded):
    # the short path's event, 1.105 to 1.115 s, is covered by two intervals:
    # the first starts 5 ms early, the last ends 15 ms late
    assert rs.bound_error_ms(decoded, (SHORT_PATH_EVENT,)) == pytest.approx(15.0)
    assert rs.bound_error_ms(decoded, (SWEEP_EVENT,)) == pytest.approx(10.0)
    assert rs.bound_error_ms(
        decoded,
        (SWEEP_EVENT, SHORT_REPLAY_EVENT, SHORT_PATH_EVENT, REMOTE_STATIONARY_EVENT),
    ) == pytest.approx(15.0)
    # nothing is reported over the event between B and C
    assert np.isnan(rs.bound_error_ms(decoded, (SWEEP_EVENT, FRAGMENTED_EVENT)))


def test_recording_numbers(decoded):
    numbers = rs.recording_numbers(decoded)
    expected = {
        "sweep_duration_ratio": 3.5,
        "sweep_local_fraction": 0.5,
        "sweep_p_spread": 3 / 20,
        "sweep_p_jump": 1 / 20,
        "sweep_draws_from_passing": 2,
        "short_replay_duration_ratio": 10.0,
        "short_replay_local_fraction": 2 / 6,
        "short_replay_p_spread": 1 / 20,
        "short_replay_p_jump": 1 / 20,
        "short_path_dispersion_cm": np.sqrt(150),
        "remote_stationary_dispersion_cm": 0.0,
    }
    # over every simulated replay, and the fragmented one has no report here
    assert np.isnan(numbers.pop("state_space_bound_error_ms"))
    assert set(numbers) == set(expected)
    for key, value in expected.items():
        assert numbers[key] == pytest.approx(value), key
    assert isinstance(numbers["sweep_draws_from_passing"], int)


def test_dominant_state_is_the_most_probable_over_the_event():
    decoded = {
        "time": np.array([0.0, 0.1, 0.2, 0.3]),
        "event_times": np.array([(0.1, 0.2)]),
        "state_names": ("Local", "Non-Local Fragmented"),
        # Local outside the event, mostly Fragmented inside it
        "state_probabilities": np.array(
            [[1.0, 0.0], [0.4, 0.6], [0.1, 0.9], [1.0, 0.0]]
        ),
    }
    assert rs.dominant_state(decoded, 0) == "Non-Local Fragmented"
