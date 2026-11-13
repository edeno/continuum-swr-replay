"""Baseline windows: the pool, the matching levels, the draws and the rank.

One simulated epoch on a 500 Hz grid: rows 0-2999 at 0-5.998 s, then a 0.5 s
timestamp gap, rows 3000-5999 at 6.5-12.498 s. The animal moves over rows
1000-1199 (2.0-2.4 s). Detector ``a`` observes all of it, ``b`` all but
7.0-7.5 s, and ``c`` all but an event it set aside at 9.0-9.1 s. ``a`` finds
events on rows 500-524 and at 5.0011-5.0489 s (off the grid), ``b`` on rows
4000-4049, and ``c`` a moving event at 1.95-2.10 s and one at 7.8011-7.8019 s,
between two rows.
"""

from functools import reduce

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.detection.memoryless_sequence.config import (
    MemorylessSequenceConfig,
)
from continuum_swr_replay.analysis.detection.memoryless_sequence.detect import (
    CandidateSequences,
    WindowDecisions,
    assemble_sequences,
    gate_windows,
    replay_mask,
    sequence_dispersion_cm,
    sequence_duration_s,
)
from continuum_swr_replay.analysis.detection.memoryless_sequence.window_statistics import (
    NO_CENTER,
    forward_jumps_cm,
)
from continuum_swr_replay.analysis.detector_comparison import selection
from continuum_swr_replay.analysis.detector_comparison.availability import (
    still_intervals,
)
from continuum_swr_replay.analysis.detector_comparison.evidence import (
    interval_evidence,
)
from continuum_swr_replay.analysis.detector_comparison.features import (
    RowTraces,
    interval_rows,
)
from continuum_swr_replay.analysis.detector_comparison.overlap import Coverage
from continuum_swr_replay.analysis.detector_comparison.selection import (
    BASELINE_DTYPES,
    DURATION_ONLY,
    FAILED,
    LOCATION,
    LOCATION_TOLERANCE_CM,
    N_BASELINES,
    REJECTION_COHORTS,
    REJECTION_DTYPES,
    REJECTION_STAGES,
    WINDOW_DTYPES,
    baseline_draws,
    baseline_pool,
    matched_baselines,
    rejection_stages,
    window_room,
)
from continuum_swr_replay.intervals import (
    contains,
    intersect,
    subtract,
    total_duration,
    union,
)

SESSION = ("sim", 20200101, "02_r1")
STEP = 0.002
MAX_GAP_S = 1.5 * STEP
N_TIME = 6000
TIME = np.arange(N_TIME) * STEP
TIME[3000:] += 0.5
SPEED = np.zeros(N_TIME)
SPEED[1000:1200] = 10.0
STILL = still_intervals(TIME, SPEED, 4.0, MAX_GAP_S)
SPAN = np.array([[TIME[0], TIME[-1]]])
N_BINS = 30
BIN_DISTANCE_CM = 2.0 * np.abs(np.subtract.outer(np.arange(N_BINS), np.arange(N_BINS)))
"""Bins 2 cm apart on one line."""
SEED = 7


def coverage_entry(unobserved):
    observed = subtract(SPAN, np.array(unobserved).reshape(-1, 2))
    return Coverage(observed=observed, exposure=intersect(observed, STILL))


COVERAGE = {
    (*SESSION, "a"): coverage_entry([]),
    (*SESSION, "b"): coverage_entry([[7.0, 7.5]]),
    (*SESSION, "c"): coverage_entry([[9.0, 9.1]]),
}
SHARED = reduce(intersect, (entry.exposure for entry in COVERAGE.values()))


def event_frame(rows):
    """Events from ``(detector, key, start_s, end_s)`` rows, in the epoch."""
    frame = pd.DataFrame(
        rows, columns=["detector", "event_key", "event_start_s", "event_end_s"]
    )
    for column, value in zip(("animal", "day", "epoch"), SESSION, strict=True):
        frame[column] = value
    return frame


COMPARED = event_frame(
    [
        ("a", "a:sim20200101_02_r1:0", TIME[500], TIME[524]),
        ("a", "a:sim20200101_02_r1:1", 5.0011, 5.0489),
        ("b", "b:sim20200101_02_r1:0", TIME[4000], TIME[4049]),
    ]
)
EVERY_EVENT = pd.concat(
    [
        COMPARED,
        event_frame(
            [
                ("c", "c:sim20200101_02_r1:0", 1.95, 2.10),  # moving
                ("c", "c:sim20200101_02_r1:1", 9.0, 9.1),  # set aside
                ("c", "c:sim20200101_02_r1:2", 7.8011, 7.8019),  # between two rows
            ]
        ),
    ],
    ignore_index=True,
)
POOL = baseline_pool(COVERAGE, EVERY_EVENT)


def traces():
    zeros = np.zeros(N_TIME)
    return RowTraces(
        time=TIME,
        spike_counts=np.zeros((N_TIME, 1), dtype=np.int64),
        ripple_z=zeros,
        mua_z=zeros,
        linear_position=zeros,
        track_segment_id=zeros.astype(np.int64),
        patch_id=zeros.astype(np.int64),
    )


def noise_evidence(seed=0):
    return np.random.default_rng(seed).normal(size=N_TIME)


def rows_of(frame):
    first, n_rows = interval_rows(
        TIME, frame.event_start_s.to_numpy(), frame.event_end_s.to_numpy()
    )
    return first, n_rows


def event_row_set(frame):
    first, n_rows = rows_of(frame)
    return set(
        np.concatenate(
            [np.arange(f, f + n) for f, n in zip(first, n_rows, strict=True)]
        ).tolist()
    )


def run(events=COMPARED, evidence=None, animal_bin=None, seed=SEED):
    return matched_baselines(
        POOL,
        events,
        traces(),
        noise_evidence() if evidence is None else evidence,
        np.zeros(N_TIME, dtype=np.int64) if animal_bin is None else animal_bin,
        BIN_DISTANCE_CM,
        seed=seed,
    )


class TestBaselinePool:
    def test_lies_in_the_shared_exposure_and_off_every_event(self):
        assert POOL.size
        for entry in COVERAGE.values():
            assert total_duration(subtract(POOL, entry.exposure)) == 0
        spans = EVERY_EVENT[["event_start_s", "event_end_s"]].to_numpy()
        for start, end in spans:
            assert total_duration(intersect(POOL, np.array([[start, end]]))) == 0

    def test_removes_only_event_time_from_the_shared_exposure(self):
        spans = EVERY_EVENT[["event_start_s", "event_end_s"]].to_numpy()
        removed = total_duration(subtract(SHARED, POOL))
        assert removed == pytest.approx(total_duration(intersect(SHARED, union(spans))))

    def test_the_moving_event_is_removed_where_it_was_still(self):
        moving = np.array([[1.95, 2.10]])
        assert total_duration(intersect(SHARED, moving)) > 0
        assert total_duration(intersect(POOL, moving)) == 0

    def test_a_detector_that_did_not_run_leaves_no_pool(self):
        coverage = dict(COVERAGE)
        coverage[(*SESSION, "d")] = Coverage(
            observed=np.empty((0, 2)), exposure=np.empty((0, 2))
        )
        assert baseline_pool(coverage, EVERY_EVENT).size == 0

    def test_named_detectors_set_the_comparison(self):
        pool = baseline_pool(COVERAGE, EVERY_EVENT, detectors=["a"])
        assert total_duration(pool) > total_duration(POOL)
        with pytest.raises(ValueError, match="no coverage entry knows"):
            baseline_pool(COVERAGE, EVERY_EVENT, detectors=["a", "z"])

    @pytest.mark.parametrize("bounds", [(2.6, 2.5), (np.nan, 2.5)])
    def test_refuses_an_event_with_a_bad_bound(self, bounds):
        bad = EVERY_EVENT.iloc[[0]].assign(
            event_start_s=bounds[0], event_end_s=bounds[1]
        )
        with pytest.raises(ValueError, match="non-finite or inverted"):
            baseline_pool(COVERAGE, pd.concat([EVERY_EVENT, bad]))

    def test_refuses_more_than_one_epoch(self):
        coverage = dict(COVERAGE)
        coverage[("sim", 20200101, "04_r2", "a")] = COVERAGE[(*SESSION, "a")]
        with pytest.raises(ValueError, match="one epoch"):
            baseline_pool(coverage, EVERY_EVENT)

    def test_refuses_events_of_another_epoch(self):
        other = EVERY_EVENT.assign(epoch="04_r2")
        with pytest.raises(ValueError, match="another epoch"):
            baseline_pool(COVERAGE, other)


class TestCandidateWindows:
    @pytest.mark.parametrize("n_rows", [1, 7, 25, 50, 400])
    def test_every_candidate_stays_in_the_pool_off_every_event_and_gap(self, n_rows):
        starts = np.flatnonzero(window_room(TIME, POOL) >= n_rows)
        assert starts.size
        rows = starts[:, None] + np.arange(n_rows)
        bounds_start, bounds_end = TIME[starts], TIME[starts + n_rows - 1]
        # inside the shared exposure and one pool interval
        assert contains(SHARED, bounds_start, bounds_end).all()
        assert contains(POOL, bounds_start, bounds_end).all()
        windows = np.column_stack([bounds_start, bounds_end])
        spans = EVERY_EVENT[["event_start_s", "event_end_s"]].to_numpy()
        assert total_duration(intersect(union(windows), union(spans))) == 0
        # no row of any event, moving and set-aside ones included
        assert not set(rows.ravel().tolist()) & event_row_set(EVERY_EVENT)
        # no timestamp gap inside a window
        steps = np.r_[np.diff(TIME), np.inf]
        if n_rows > 1:
            assert (steps[rows[:, :-1]] <= MAX_GAP_S).all()
        # never the grid's last sample, and exactly n_rows rows
        assert rows.max() < N_TIME - 1
        first, counted = interval_rows(TIME, bounds_start, bounds_end)
        np.testing.assert_array_equal(first, starts)
        assert (counted == n_rows).all()

    def test_the_gap_splits_the_pool(self):
        before = np.flatnonzero(TIME < 6.0)[-1]
        assert window_room(TIME, POOL)[before - 5] <= 5

    def test_an_event_between_two_rows_splits_the_pool(self):
        after = np.flatnonzero(TIME > 7.8019)[0]
        assert window_room(TIME, POOL)[after - 3] == 3

    def test_the_grids_last_sample_has_no_room(self):
        room = window_room(TIME, np.array([[TIME[-10] - 1e-4, TIME[-1] + 1.0]]))
        np.testing.assert_array_equal(room[-10:], [9, 8, 7, 6, 5, 4, 3, 2, 1, 0])


class TestMatchedBaselines:
    def test_dtypes(self):
        summary, windows = run()
        assert dict(summary.dtypes) == dict(BASELINE_DTYPES)
        assert dict(windows.dtypes) == dict(WINDOW_DTYPES)
        assert summary.index.equals(COMPARED.index)

    def test_empty_events_keep_the_dtypes(self):
        summary, windows = run(events=COMPARED.iloc[:0])
        assert summary.empty and windows.empty
        assert dict(summary.dtypes) == dict(BASELINE_DTYPES)
        assert dict(windows.dtypes) == dict(WINDOW_DTYPES)

    def test_each_window_has_its_events_row_count_and_lies_in_the_pool(self):
        summary, windows = run()
        assert (summary.baseline_level == LOCATION).all()
        assert (summary.baseline_n == N_BASELINES).all()
        assert list(windows.matched_event_key) == list(COMPARED.event_key)
        _, event_n = rows_of(COMPARED)
        _, window_n = rows_of(windows)
        np.testing.assert_array_equal(window_n, event_n)
        assert contains(
            POOL, windows.event_start_s.to_numpy(), windows.event_end_s.to_numpy()
        ).all()
        assert not event_row_set(windows) & event_row_set(EVERY_EVENT)

    def test_every_draw_has_the_events_row_count_and_stays_in_the_pool(self):
        levels, starts, n_rows = baseline_draws(
            POOL,
            COMPARED,
            traces(),
            np.zeros(N_TIME, dtype=np.int64),
            BIN_DISTANCE_CM,
            seed=SEED,
        )
        room = window_room(TIME, POOL)
        for i in range(len(COMPARED)):
            drawn = starts[i][starts[i] >= 0]
            assert drawn.size == N_BASELINES
            assert np.unique(drawn).size == drawn.size  # without replacement
            assert (room[drawn] >= n_rows[i]).all()

    def test_location_windows_lie_within_the_tolerance(self):
        # the animal's bin climbs one bin (2 cm) every 50 rows
        animal_bin = (np.arange(N_TIME) // 50) % N_BINS
        levels, starts, n_rows = baseline_draws(
            POOL, COMPARED, traces(), animal_bin, BIN_DISTANCE_CM, seed=SEED
        )
        first, _ = rows_of(COMPARED)
        event_bin = animal_bin[first + (n_rows - 1) // 2]
        for i in range(len(COMPARED)):
            assert levels[i] == LOCATION
            drawn = starts[i][starts[i] >= 0]
            window_bin = animal_bin[drawn + (n_rows[i] - 1) // 2]
            distance = BIN_DISTANCE_CM[event_bin[i], window_bin]
            assert (distance <= LOCATION_TOLERANCE_CM).all()
        summary, windows = run(animal_bin=animal_bin)
        first_window, window_n = rows_of(windows)
        window_bin = animal_bin[first_window + (window_n - 1) // 2]
        assert (BIN_DISTANCE_CM[event_bin, window_bin] <= LOCATION_TOLERANCE_CM).all()

    def test_an_event_longer_than_every_pool_interval_fails(self):
        longest = int(window_room(TIME, POOL).max())
        long_event = event_frame(
            [("a", "a:sim20200101_02_r1:9", TIME[100], TIME[100 + longest])]
        )
        summary, windows = run(events=long_event)
        assert summary.baseline_level.iloc[0] == FAILED
        assert summary.baseline_n.iloc[0] == 0
        assert (
            summary[["baseline_median_dll", "baseline_q95_dll", "evidence_rank"]]
            .isna()
            .all(axis=None)
        )
        assert windows.empty

    @pytest.mark.parametrize(
        ("n_near", "level", "n_drawn"),
        [(5, DURATION_ONLY, N_BASELINES), (30, LOCATION, 30)],
    )
    def test_too_few_location_windows_fall_back_to_duration_only(
        self, n_near, level, n_drawn
    ):
        # the animal sits at bin 29 (58 cm from bin 0) except at the event's
        # middle row and at n_near windows' middle rows, well inside the pool
        animal_bin = np.full(N_TIME, N_BINS - 1, dtype=np.int64)
        animal_bin[512] = 0
        animal_bin[3600 : 3600 + n_near] = 0
        summary, windows = run(events=COMPARED.iloc[:1], animal_bin=animal_bin)
        assert summary.baseline_level.iloc[0] == level
        assert summary.baseline_n.iloc[0] == n_drawn

    def test_the_middle_of_an_even_run_is_the_earlier_row(self):
        # 24 rows from 500: the middle is row 511, not 512, for the event and
        # for its windows, as for the event's own location features
        event = event_frame([("a", "a:even", TIME[500], TIME[523])])
        animal_bin = np.full(N_TIME, N_BINS - 1, dtype=np.int64)
        animal_bin[511] = 0
        # the middle rows of the 30 windows from the first row of a pool
        # interval: the row before it starts no window, so a middle one row
        # later would find 29
        start = 3400 + np.flatnonzero(window_room(TIME, POOL)[3400:] > 0)[0]
        animal_bin[start + 11 : start + 41] = 0
        summary, windows = run(events=event, animal_bin=animal_bin)
        assert summary.baseline_level.iloc[0] == LOCATION
        assert summary.baseline_n.iloc[0] == 30
        first_window, _ = rows_of(windows)
        assert start <= first_window[0] < start + 30

    def test_an_event_of_unknown_location_is_matched_on_duration(self):
        animal_bin = np.zeros(N_TIME, dtype=np.int64)
        animal_bin[500:525] = -1
        summary, _ = run(events=COMPARED.iloc[:1], animal_bin=animal_bin)
        assert summary.baseline_level.iloc[0] == DURATION_ONLY

    def test_draws_are_deterministic_and_independent_of_other_events(self):
        summary, windows = run()
        again_summary, again_windows = run()
        pd.testing.assert_frame_equal(summary, again_summary)
        pd.testing.assert_frame_equal(windows, again_windows)
        alone_summary, alone_windows = run(events=COMPARED.iloc[[2]])
        pd.testing.assert_frame_equal(alone_summary, summary.iloc[[2]])
        pd.testing.assert_frame_equal(
            alone_windows, windows.iloc[[2]].reset_index(drop=True)
        )
        _, other_windows = run(seed=SEED + 1)
        assert not other_windows.event_start_s.equals(windows.event_start_s)

    def test_identical_intervals_of_two_inventories_draw_identically(self):
        twin = COMPARED.iloc[[0]].assign(detector="c", event_key="c:twin")
        events = pd.concat([COMPARED, twin], ignore_index=True)
        summary, windows = run(events=events)
        pd.testing.assert_series_equal(
            summary.iloc[3], summary.iloc[0], check_names=False
        )
        assert windows.matched_event_key.iloc[3] == "c:twin"
        pd.testing.assert_series_equal(
            windows.drop(columns="matched_event_key").iloc[3],
            windows.drop(columns="matched_event_key").iloc[0],
            check_names=False,
        )

    def test_different_bounds_draw_from_different_streams(self):
        # the same rows, so the same candidates, but another start
        moved = COMPARED.iloc[[0]].assign(
            event_start_s=TIME[500] - 1e-4, event_key="a:moved"
        )
        _, windows = run(events=pd.concat([COMPARED.iloc[[0]], moved]))
        np.testing.assert_array_equal(rows_of(windows)[1], [25, 25])
        assert windows.event_start_s.iloc[0] != windows.event_start_s.iloc[1]

    def test_an_event_ending_on_the_grids_last_sample_is_matched_without_it(self):
        # rows 5980 to 5998: the grid's last sample, row 5999, is never a row
        events = event_frame(
            [
                ("a", "a:end", TIME[5980], TIME[-1]),
                ("a", "a:last", TIME[-1], TIME[-1]),
            ]
        )
        evidence = noise_evidence()
        levels, starts, n_rows = baseline_draws(
            POOL,
            events,
            traces(),
            np.zeros(N_TIME, dtype=np.int64),
            BIN_DISTANCE_CM,
            seed=SEED,
        )
        assert n_rows.tolist() == [19, 0]
        assert list(levels) == [LOCATION, FAILED]
        summary, windows = run(events=events, evidence=evidence)
        np.testing.assert_array_equal(rows_of(windows)[1], [19])
        own = evidence[5980:5999].mean()
        drawn = pd.DataFrame(
            {
                "event_start_s": TIME[starts[0]],
                "event_end_s": TIME[starts[0] + 18],
            }
        )
        scores = interval_evidence(traces(), evidence, drawn).evidence_dll
        assert summary.evidence_rank.iloc[0] == (scores < own).mean()
        assert summary.baseline_n.iloc[1] == 0

    def test_a_window_across_a_timestamp_gap_is_refused(self):
        # a pool by hand across the gap after row 2999: not still time
        pool = np.array([[TIME[2985] - 1e-4, TIME[3049] + 1e-4]])
        with pytest.raises(ValueError, match="timestamp gap"):
            matched_baselines(
                pool,
                COMPARED.iloc[:1],
                traces(),
                noise_evidence(),
                np.zeros(N_TIME, dtype=np.int64),
                BIN_DISTANCE_CM,
                seed=SEED,
            )

    def test_evidence_rank_is_one_above_every_baseline_and_zero_below(self):
        evidence = noise_evidence()
        evidence[500:525] = 50.0
        evidence[4000:4050] = -50.0
        summary, _ = run(evidence=evidence)
        assert summary.evidence_rank.tolist()[0] == 1.0
        assert summary.evidence_rank.tolist()[2] == 0.0
        assert 0.0 < summary.evidence_rank.tolist()[1] < 1.0
        assert (summary.baseline_median_dll.abs() < 1.0).all()
        assert (summary.baseline_q95_dll > summary.baseline_median_dll).all()

    def test_an_event_tied_with_every_baseline_beats_none(self):
        summary, _ = run(evidence=np.zeros(N_TIME))
        assert (summary.evidence_rank == 0.0).all()

    def test_scored_and_unscored_windows_mixed(self):
        # no evidence on rows 3000-3999: windows there are unscored
        evidence = noise_evidence()
        evidence[3000:4000] = np.nan
        levels, starts, n_rows = baseline_draws(
            POOL,
            COMPARED.iloc[:1],
            traces(),
            np.zeros(N_TIME, dtype=np.int64),
            BIN_DISTANCE_CM,
            seed=SEED,
        )
        drawn = pd.DataFrame(
            {
                "event_start_s": TIME[starts[0]],
                "event_end_s": TIME[starts[0] + n_rows[0] - 1],
            }
        )
        scores = interval_evidence(traces(), evidence, drawn).evidence_dll
        scored = scores.dropna().to_numpy()
        assert 0 < scored.size < N_BASELINES
        summary, _ = run(events=COMPARED.iloc[:1], evidence=evidence)
        own = interval_evidence(traces(), evidence, COMPARED.iloc[:1]).evidence_dll
        assert summary.baseline_n.iloc[0] == N_BASELINES
        assert summary.baseline_n_scored.iloc[0] == scored.size
        assert summary.baseline_median_dll.iloc[0] == np.median(scored)
        assert summary.baseline_q95_dll.iloc[0] == np.quantile(scored, 0.95)
        assert summary.evidence_rank.iloc[0] == (
            (scored < own.iloc[0]).sum() / scored.size
        )

    def test_windows_without_evidence_enter_no_statistic(self):
        evidence = np.full(N_TIME, np.nan)
        evidence[500:525] = 1.0
        summary, _ = run(events=COMPARED.iloc[:1], evidence=evidence)
        assert summary.baseline_n.iloc[0] == N_BASELINES
        assert summary.baseline_n_scored.iloc[0] == 0
        assert (
            summary[["baseline_median_dll", "baseline_q95_dll", "evidence_rank"]]
            .isna()
            .all(axis=None)
        )

    def test_the_statistics_are_those_of_the_drawn_windows(self):
        evidence = noise_evidence()
        levels, starts, n_rows = baseline_draws(
            POOL,
            COMPARED,
            traces(),
            np.zeros(N_TIME, dtype=np.int64),
            BIN_DISTANCE_CM,
            seed=SEED,
        )
        summary, windows = run(evidence=evidence)
        for i in range(len(COMPARED)):
            drawn = pd.DataFrame(
                {
                    "event_start_s": TIME[starts[i]],
                    "event_end_s": TIME[starts[i] + n_rows[i] - 1],
                }
            )
            # each window scored exactly as interval_evidence scores it
            scores = interval_evidence(traces(), evidence, drawn).evidence_dll
            assert summary.baseline_median_dll.iloc[i] == np.median(scores)
            assert summary.baseline_q95_dll.iloc[i] == np.quantile(scores, 0.95)
            own = interval_evidence(traces(), evidence, COMPARED.iloc[[i]])
            rank = (scores < own.evidence_dll.iloc[0]).mean()
            assert summary.evidence_rank.iloc[i] == rank
            assert windows.event_start_s.iloc[i] == TIME[starts[i, 0]]


# The rejection stages. One simulated memoryless run of 700 windows, 4 ms
# apart, on a straight track of 2 cm bins. Every window has a center (bin
# k // 2, so jumps are 0 or 2 cm), a usable posterior and an eligible speed,
# and fails the spread gate, except in these stretches, which pass it:
#   50          one kept window, a run of one, dropped;
#   100-115     a lone run of 60 ms, below the duration filter;
#   200-215, 220-235  two 60 ms runs 20 ms apart, merged into a candidate;
#   300-329     a stationary stretch (bin 150); window 329 fails its 30 cm jump;
#   400-460     a 240 ms moving run, replay;
#   600-625     exactly 100 ms, kept by assembly and declined as replay.
# Window 30 is undecoded (it holds a masked row) and window 71 is too fast.

# window counts above are at a 4 ms step: the stages read the config's stride
REJECTION_CONFIG = MemorylessSequenceConfig(stride_s=0.004, sampling_frequency=500.0)
N_WINDOWS = 700
MIDPOINT = 100.0 + REJECTION_CONFIG.stride_s * np.arange(N_WINDOWS)
TRACK_DISTANCE_CM = 2.0 * np.abs(np.subtract.outer(np.arange(400), np.arange(400)))
UNDECODED = np.zeros(N_WINDOWS, dtype=bool)
UNDECODED[30] = True
SS = "state_space"
CRITERIA = "memoryless_criteria"


def simulated_candidates():
    """The run, built with the detector's own gates, assembly and criteria."""
    raw_center = np.arange(N_WINDOWS) // 2
    spread = np.full(N_WINDOWS, 50.0)
    for first, last in [
        (50, 50),
        (100, 115),
        (200, 215),
        (220, 235),
        (300, 329),
        (400, 460),
        (600, 625),
    ]:
        spread[first : last + 1] = 1.0
    raw_center[300:330] = 150
    raw_center[30], spread[30] = NO_CENTER, np.nan
    speed_ok = np.ones(N_WINDOWS, dtype=bool)
    speed_ok[71] = False
    center = np.where(speed_ok, raw_center, NO_CENTER)
    jump = forward_jumps_cm(center, TRACK_DISTANCE_CM)
    gates = gate_windows(
        raw_center, spread, jump, REJECTION_CONFIG, speed_eligible=speed_ok
    )
    assembly = assemble_sequences(
        gates.kept, center, MIDPOINT, TRACK_DISTANCE_CM, REJECTION_CONFIG, UNDECODED
    )
    dispersion = sequence_dispersion_cm(
        assembly.runs, assembly.kept, center, TRACK_DISTANCE_CM
    )
    return CandidateSequences(
        runs=assembly.runs,
        duration_s=sequence_duration_s(assembly.runs, MIDPOINT),
        dispersion_cm=dispersion,
        is_replay=replay_mask(assembly.runs, dispersion, REJECTION_CONFIG),
        decisions=WindowDecisions(center, spread, jump, gates),
        assembly=assembly,
        midpoint_s=MIDPOINT,
        config=REJECTION_CONFIG,
    )


CANDIDATES = simulated_candidates()
MERGED, STATIONARY, REPLAY, EXACTLY_100_MS = 0, 1, 2, 3
"""The candidates, in window order."""


def candidate_table(**replay_columns):
    """The run's table, every candidate still and unshuffled but the replay one,
    which takes ``replay_columns`` (passed and still by default)."""
    runs = CANDIDATES.runs
    n = len(runs)
    table = pd.DataFrame(
        {
            "animal": SESSION[0],
            "day": SESSION[1],
            "epoch": SESSION[2],
            "ms_event_id": 1000 + np.arange(n),
            "first_window": runs[:, 0],
            "last_window": runs[:, 1],
            "event_start_s": MIDPOINT[runs[:, 0]],
            "event_end_s": MIDPOINT[runs[:, 1]],
            "is_replay": CANDIDATES.is_replay,
            "whole_event_speed_eligible": True,
            "shuffle_testable": False,
            "shuffle_pass": False,
            "n_shuffles": 0,
            "artifact_drop_reason": None,
        }
    )
    replay = {
        "whole_event_speed_eligible": True,
        "shuffle_testable": True,
        "shuffle_pass": True,
        "n_shuffles": 100,
    } | replay_columns
    for column, value in replay.items():
        table.loc[REPLAY, column] = value
    return table


def candidate_key(cohort, index):
    return f"{cohort}:sim20200101_02_r1:{1000 + index}"


def ss_events(spans):
    """State-space events from ``(first window, last window)`` spans or bounds."""
    return event_frame(
        [
            (SS, f"{SS}:sim20200101_02_r1:{i}", start, end)
            for i, (start, end) in enumerate(spans)
        ]
    )


def windows(first, last):
    return MIDPOINT[first], MIDPOINT[last]


def membership(events, labels, cohort, in_cohort=(), reasons=None):
    """Population rows: each event's membership against ``cohort`` and the
    cohort's compared events (``in_cohort``, candidate indices)."""
    reasons = reasons or {}
    rows = [
        (SS, key, f"{label}_by:{cohort}", reasons.get(key, ""))
        for key, label in zip(events.event_key, labels, strict=True)
    ]
    rows += [(cohort, candidate_key(cohort, i), "all", "") for i in in_cohort]
    rows += [(SS, key, "all", "") for key in events.event_key]
    frame = pd.DataFrame(
        rows, columns=["detector", "event_key", "population", "unknown_reason"]
    )
    for column, value in zip(("animal", "day", "epoch"), SESSION, strict=True):
        frame[column] = value
    return frame


def stages(spans, labels=None, *, cohort="memoryless", in_cohort=(), table=None):
    events = ss_events(spans)
    labels = labels or ["unmatched"] * len(events)
    return rejection_stages(
        events,
        membership(events, labels, cohort, in_cohort),
        CANDIDATES,
        candidate_table() if table is None else table,
        UNDECODED,
        cohort=cohort,
    )


class TestRejectionStages:
    def test_the_simulated_run_is_as_described(self):
        assert CANDIDATES.runs.tolist() == [
            [200, 235],
            [300, 328],
            [400, 460],
            [600, 625],
        ]
        assert CANDIDATES.is_replay.tolist() == [False, False, True, False]
        assert CANDIDATES.pre_merge_runs[:, 0].tolist() == [
            100,
            200,
            220,
            300,
            400,
            600,
        ]

    def test_a_window_failing_only_spread_counts_under_spread(self):
        row = stages([windows(10, 20)]).iloc[0]
        assert row.stage == "no_kept_window"
        assert row.n_windows == row.n_spread_failed == 11
        for name in (
            "n_undecoded",
            "n_posterior_unusable",
            "n_speed_ineligible",
            "n_jump_failed",
            "n_kept",
            "n_kept_in_runs",
            "n_candidates",
        ):
            assert row[name] == 0, name

    def test_an_undecoded_window_counts_apart(self):
        row = stages([windows(28, 32)]).iloc[0]
        assert (row.n_windows, row.n_undecoded) == (5, 1)
        assert row.n_posterior_unusable == 0
        # the decoded windows only; window 29 has no jump into window 30
        assert (row.n_spread_failed, row.n_jump_failed) == (4, 1)

    def test_a_fast_window_counts_under_speed(self):
        row = stages([windows(70, 72)]).iloc[0]
        assert (row.n_speed_ineligible, row.n_posterior_unusable) == (1, 0)
        # the blanked window 71 has no jump, nor has window 70 into it
        assert row.n_jump_failed == 2

    def test_a_single_kept_window_between_rejected_ones_is_no_run(self):
        row = stages([windows(48, 52)]).iloc[0]
        assert (row.n_kept, row.n_kept_in_runs) == (1, 0)
        assert row.stage == "no_run"

    def test_a_lone_60_ms_run_is_no_candidate(self):
        row = stages([windows(100, 115)]).iloc[0]
        assert row.n_kept == row.n_kept_in_runs == 16
        assert (row.n_candidates, row.stage) == (0, "no_candidate")

    def test_two_short_runs_merged_reach_the_criteria(self):
        row = stages([windows(205, 225)]).iloc[0]
        assert row.n_candidates == 1
        assert REJECTION_STAGES.index(row.stage) >= REJECTION_STAGES.index("duration")
        # the merged gap's windows are not kept
        assert row.n_kept == 11 + 6

    def test_touching_a_candidate_is_not_overlapping_it(self):
        result = stages([windows(190, 200), windows(235, 240), windows(190, 201)])
        assert result.n_candidates.tolist() == [0, 0, 1]
        # window 200 is kept in a run, but the candidate only touches the event
        assert result.stage.iloc[0] == "no_candidate"

    def test_a_candidate_of_exactly_100_ms_fails_duration(self):
        row = stages([windows(600, 625)]).iloc[0]
        assert row.stage == "duration"
        # every reason is kept, not only the stage's
        assert row.any_duration_failed and row.any_dispersion_failed

    def test_a_stationary_candidate_is_dispersion(self):
        row = stages([windows(300, 328)]).iloc[0]
        assert row.stage == "dispersion"
        assert not row.any_duration_failed and row.any_dispersion_failed

    def test_a_moving_replay_candidate_is_speed(self):
        table = candidate_table(whole_event_speed_eligible=False)
        for cohort in ("memoryless", CRITERIA):
            row = stages([windows(410, 450)], cohort=cohort, table=table).iloc[0]
            assert row.stage == "speed"
            assert row.any_replay and row.any_replay_not_speed_eligible

    @pytest.mark.parametrize(
        ("columns", "stage"),
        [
            ({"shuffle_pass": False}, "shuffle_failed"),
            (
                {"shuffle_pass": False, "shuffle_testable": False, "n_shuffles": 0},
                "shuffle_failed",
            ),
            ({"shuffle_pass": False, "shuffle_testable": False}, "shuffle_untestable"),
        ],
    )
    def test_a_failing_or_unshuffled_candidate_is_shuffle_failed(self, columns, stage):
        table = candidate_table(**columns)
        row = stages([windows(410, 450)], table=table).iloc[0]
        assert row.stage == stage
        assert row.any_shuffle_failed
        assert row.any_shuffle_untestable == (stage == "shuffle_untestable")
        # the criteria cohort has no shuffle
        criteria = stages(
            [windows(410, 450)],
            ["matched"],
            cohort=CRITERIA,
            in_cohort=[REPLAY],
            table=table,
        ).iloc[0]
        assert criteria.stage == "in_cohort"

    def test_an_event_matched_by_the_cohort_is_in_cohort(self):
        row = stages([windows(410, 450)], ["matched"], in_cohort=[REPLAY]).iloc[0]
        assert row.stage == "in_cohort"
        assert row.any_in_cohort and row.membership == "matched"

    def test_an_unmatched_event_is_never_in_cohort(self):
        with pytest.raises(ValueError, match="matched"):
            stages([windows(410, 450)], ["unmatched"], in_cohort=[REPLAY])
        with pytest.raises(ValueError, match="matched"):
            stages([windows(410, 450)], ["matched"], in_cohort=[])

    def test_a_candidate_through_every_gate_outside_the_cohort_is_refused(self):
        # set aside by the comparison: no unmatched event can overlap it
        with pytest.raises(
            ValueError,
            match="overlaps a candidate that passed every gate but is not among "
            "the cohort's compared events",
        ):
            stages([windows(410, 450)], ["unmatched"], in_cohort=[])

    def test_a_known_event_with_no_stage_reached_is_refused(self, monkeypatch):
        # every overlapping candidate ranked -1 (through every gate, outside
        # the cohort) without the set-aside check catching it: the stage must
        # not be read silently from the end of REJECTION_STAGES
        real = selection._candidate_gates

        def no_stage(*args, **kwargs):
            flags, set_aside, rank = real(*args, **kwargs)
            return flags, np.zeros_like(set_aside), np.full_like(rank, -1)

        monkeypatch.setattr(selection, "_candidate_gates", no_stage)
        with pytest.raises(ValueError, match="no rejection stage"):
            stages([windows(300, 328)])

    def test_an_event_over_candidates_of_different_ranks_takes_the_furthest(self):
        table = candidate_table(shuffle_pass=False)
        # the stationary (dispersion), the replay (shuffle failed) and the
        # 100 ms (duration) candidates
        row = stages([windows(300, 625)], table=table).iloc[0]
        assert row.n_candidates == 3
        assert row.stage == "shuffle_failed"
        assert row.any_duration_failed and row.any_dispersion_failed
        assert row.any_shuffle_failed and not row.any_in_cohort
        # with the replay candidate in the cohort, it is the furthest
        matched = stages([windows(300, 625)], ["matched"], in_cohort=[REPLAY])
        assert matched.stage.iloc[0] == "in_cohort"

    def test_events_out_of_time_order_get_their_own_stages(self):
        spans = [windows(600, 625), windows(10, 20), windows(410, 450)]
        labels = ["unmatched", "unmatched", "matched"]
        shuffled = stages(spans, labels, in_cohort=[REPLAY])
        ordered = stages(
            [spans[1], spans[0], spans[2]],
            [labels[1], labels[0], labels[2]],
            in_cohort=[REPLAY],
        )
        assert shuffled.stage.tolist() == ["duration", "no_kept_window", "in_cohort"]
        columns = shuffled.columns.drop("event_key")
        pd.testing.assert_frame_equal(
            shuffled.iloc[[1, 0, 2]][columns].reset_index(drop=True),
            ordered[columns].reset_index(drop=True),
        )

    def test_an_event_of_no_duration_inside_a_candidate_overlaps_none(self):
        inside = MIDPOINT[310]
        row = stages([(inside, inside)]).iloc[0]
        assert row.n_candidates == 0
        assert row.stage == "no_candidate"

    def test_an_unknown_event_has_no_rejection_reason(self):
        events = ss_events([windows(10, 20), windows(410, 450)])
        result = rejection_stages(
            events,
            membership(
                events,
                ["unknown", "unknown"],
                "memoryless",
                [REPLAY],
                {events.event_key[1]: "outside_shared_exposure"},
            ),
            CANDIDATES,
            candidate_table(),
            UNDECODED,
            cohort="memoryless",
        )
        assert result.stage.tolist() == ["unknown", "unknown"]
        assert result.unknown_reason.tolist() == ["", "outside_shared_exposure"]
        reasons = result.filter(regex="^(n_|any_)")
        assert reasons.shape[1] == 16
        assert reasons.isna().all(axis=None)

    def test_no_window_and_in_cohort_exactly_when_matched_over_every_event(self):
        spans = [
            (MIDPOINT[60] + 0.001, MIDPOINT[60] + 0.002),
            windows(10, 20),
            windows(205, 225),
            windows(300, 328),
            windows(400, 460),
            windows(600, 625),
        ]
        labels = ["unmatched"] * 4 + ["matched", "unmatched"]
        result = stages(spans, labels, in_cohort=[REPLAY])
        assert result.stage.iloc[0] == "no_window"
        assert ((result.stage == "in_cohort") == (result.membership == "matched")).all()
        assert list(result.dtypes.astype(str)) == [
            str(pd.Series(dtype=dtype).dtype) for dtype in REJECTION_DTYPES.values()
        ]

    def test_without_a_run_every_event_must_be_unknown(self):
        events = ss_events([windows(10, 20)])
        rows = membership(events, ["unknown"], "memoryless", (), {})
        result = rejection_stages(events, rows, None, None, None, cohort="memoryless")
        assert result.stage.tolist() == ["unknown"]
        with pytest.raises(ValueError, match="together"):
            rejection_stages(
                events, rows, CANDIDATES, None, UNDECODED, cohort="memoryless"
            )
        with pytest.raises(ValueError, match="unknown"):
            rejection_stages(
                events,
                membership(events, ["unmatched"], "memoryless"),
                None,
                None,
                None,
                cohort="memoryless",
            )

    def test_the_cohorts_are_the_comparisons(self):
        from continuum_swr_replay.analysis.detector_comparison.inventory_run import (
            ROSTER,
        )

        assert {name: ROSTER[name] for name in REJECTION_COHORTS} == dict(
            REJECTION_COHORTS
        )

    def test_empty_events_keep_their_dtypes(self):
        result = stages([])
        assert result.empty
        assert list(result.columns) == list(REJECTION_DTYPES)
