"""Per-event features read on the decode grid."""

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.inventory_comparison.features import (
    DURATION_STRATA_S,
    RowTraces,
    duration_stratum,
    interval_features,
    interval_rows,
    neighbour_gaps,
    stop_context,
)
from continuum_swr_replay.analysis.inventory_comparison.schema import (
    SessionIdentity,
    inventory_from_intervals,
)

SAMPLE_INTERVAL_S = 0.002
TIME = 100.0 + SAMPLE_INTERVAL_S * np.arange(1000)
"""A 500 Hz grid of 2 s."""
EPOCH = SessionIdentity(animal="sim", day=20200101, epoch="02_r1", brain_area="HPC")
OTHER_EPOCH = SessionIdentity(
    animal="sim", day=20200101, epoch="04_r2", brain_area="HPC"
)


def decode_frame(**columns) -> pd.DataFrame:
    """A decode frame on ``TIME`` with flat traces, any column replaced."""
    n_time = TIME.size
    frame = {
        "SWR_zscore": np.linspace(-1.0, 3.0, n_time),
        "MUA_zscore": np.linspace(2.0, -2.0, n_time),
        "linear_position": np.linspace(0.0, 199.8, n_time),
        "track_segment_id": np.repeat(np.arange(4), n_time // 4),
        "patch_id": np.repeat([1, 2], n_time // 2),
        "head_speed": np.zeros(n_time),
    }
    frame.update(columns)
    return pd.DataFrame(frame, index=pd.Index(TIME, name="time"))


def traces(spike_times=None, **columns) -> RowTraces:
    if spike_times is None:
        spike_times = [np.array([]), np.array([])]
    return RowTraces.from_frame(decode_frame(**columns), spike_times)


def intervals(bounds) -> pd.DataFrame:
    bounds = np.asarray(bounds, dtype=float).reshape(-1, 2)
    return pd.DataFrame({"event_start_s": bounds[:, 0], "event_end_s": bounds[:, 1]})


def inventory(bounds, detector, session=EPOCH) -> pd.DataFrame:
    bounds = np.asarray(bounds, dtype=float).reshape(-1, 2)
    return inventory_from_intervals(
        bounds[:, 0],
        bounds[:, 1],
        native_keys=np.arange(len(bounds)),
        detector=detector,
        session=session,
        run_id="run",
        speed_time_s=TIME,
        speed_cm_per_s=np.zeros(TIME.size),
    )


def features(row_traces, bounds) -> pd.DataFrame:
    return interval_features(
        row_traces, intervals(bounds), sample_interval_s=SAMPLE_INTERVAL_S
    )


# --- rows ---------------------------------------------------------------


def test_an_interval_between_two_samples_has_no_rows():
    first, n_rows = interval_rows(
        TIME, np.array([TIME[10] + 0.0005]), np.array([TIME[10] + 0.0015])
    )
    assert n_rows.tolist() == [0]


def test_a_bound_exactly_on_a_sample_includes_it():
    first, n_rows = interval_rows(TIME, np.array([TIME[10]]), np.array([TIME[14]]))
    assert first.tolist() == [10]
    assert n_rows.tolist() == [5]
    # one sample: an interval of no duration on a sample holds that sample
    first, n_rows = interval_rows(TIME, np.array([TIME[7]]), np.array([TIME[7]]))
    assert (first.tolist(), n_rows.tolist()) == ([7], [1])


def test_bounds_between_samples_take_the_samples_inside():
    first, n_rows = interval_rows(
        TIME, np.array([TIME[10] + 0.001]), np.array([TIME[14] + 0.001])
    )
    assert (first.tolist(), n_rows.tolist()) == ([11], [4])


# --- spikes -------------------------------------------------------------


def test_a_spike_counts_in_the_row_whose_sample_starts_it():
    k = 20
    spikes = [
        np.array([TIME[k], TIME[k + 1] - 1e-6, TIME[k + 1]]),
        np.array([]),
    ]
    row_traces = traces(spikes)
    one_row = features(row_traces, [[TIME[k], TIME[k]], [TIME[k + 1], TIME[k + 1]]])
    assert one_row.n_spikes.tolist() == [2, 1]
    assert one_row.n_active_tetrodes.tolist() == [1, 1]


def test_active_tetrodes_counts_the_columns_with_a_spike():
    spikes = [
        np.array([TIME[30], TIME[31]]),
        np.array([]),
        np.array([TIME[32]]),
    ]
    table = features(traces(spikes), [[TIME[30], TIME[34]], [TIME[40], TIME[44]]])
    assert table.n_spikes.tolist() == [3, 0]
    assert table.n_active_tetrodes.tolist() == [2, 0]
    assert table.n_tetrodes.tolist() == [3, 3]
    np.testing.assert_allclose(table.spike_rate_hz, [3 / (5 * 0.002), 0.0])


def test_an_interval_with_no_row_has_every_row_feature_missing():
    table = features(traces(), [[TIME[10] + 0.0005, TIME[10] + 0.0015]])
    assert table.n_rows.tolist() == [0]
    assert table.n_spikes.isna().all()
    assert table.n_active_tetrodes.isna().all()
    assert table.spike_rate_hz.isna().all()
    for column in ("ripple_z_max", "ripple_z_mean", "mua_z_max", "mua_z_mean"):
        assert table[column].isna().all()
    assert table.linear_position_cm.isna().all()
    assert table.track_segment_id.isna().all()
    assert table.patch_id.isna().all()


def test_an_interval_reaching_the_grids_last_sample_is_refused():
    with pytest.raises(ValueError, match="last sample"):
        features(traces(), [[TIME[-3], TIME[-1]]])


# --- the same interval reads the same -----------------------------------


def test_identical_intervals_from_two_inventories_give_identical_features():
    rng = np.random.default_rng(0)
    spikes = [np.sort(rng.uniform(TIME[0], TIME[-2], 400)) for _ in range(4)]
    row_traces = traces(
        spikes,
        SWR_zscore=rng.normal(size=TIME.size),
        MUA_zscore=rng.normal(size=TIME.size),
    )
    shared = [[TIME[100] + 0.0003, TIME[160] - 0.0004], [TIME[400], TIME[455]]]
    first = inventory(shared + [[TIME[600], TIME[650]]], "kay")
    second = inventory([[TIME[50], TIME[90]]] + shared, "state_space")
    together = interval_features(
        row_traces,
        pd.concat([first, second], ignore_index=True),
        sample_interval_s=SAMPLE_INTERVAL_S,
    )
    alone = interval_features(row_traces, first, sample_interval_s=SAMPLE_INTERVAL_S)
    pd.testing.assert_frame_equal(
        together.iloc[[0, 1]].reset_index(drop=True),
        together.iloc[[4, 5]].reset_index(drop=True),
        check_exact=True,
    )
    pd.testing.assert_frame_equal(
        together.iloc[:3].reset_index(drop=True),
        alone.reset_index(drop=True),
        check_exact=True,
    )


def test_ripple_and_mua_summaries_match_a_direct_reading():
    row_traces = traces()
    table = features(row_traces, [[TIME[100], TIME[110]]])
    rows = slice(100, 111)
    assert table.ripple_z_max.item() == row_traces.ripple_z[rows].max()
    np.testing.assert_allclose(table.ripple_z_mean, row_traces.ripple_z[rows].mean())
    assert table.mua_z_max.item() == row_traces.mua_z[rows].max()
    np.testing.assert_allclose(table.mua_z_mean, row_traces.mua_z[rows].mean())


def test_position_is_read_at_the_middle_row():
    row_traces = traces()
    odd, even = features(
        row_traces, [[TIME[248], TIME[252]], [TIME[248], TIME[251]]]
    ).itertuples()
    assert odd.linear_position_cm == row_traces.linear_position[250]
    assert even.linear_position_cm == row_traces.linear_position[249]
    assert (odd.track_segment_id, odd.patch_id) == (1, 1)
    table = features(row_traces, [[TIME[748], TIME[752]]])
    assert table.track_segment_id.item() == 3
    assert table.patch_id.item() == 2


# --- a masked row is never skipped ---------------------------------------


def test_a_nan_ripple_row_makes_only_the_intervals_holding_it_nan():
    ripple = np.linspace(-1.0, 3.0, TIME.size)
    ripple[300] = np.nan
    row_traces = traces(SWR_zscore=ripple)
    table = features(
        row_traces,
        [
            [TIME[290], TIME[300]],  # ends on the NaN row
            [TIME[300], TIME[310]],  # starts on it
            [TIME[280], TIME[299]],  # ends just before it
            [TIME[301], TIME[320]],  # starts just after it
        ],
    )
    assert table.ripple_z_max.isna().tolist() == [True, True, False, False]
    assert table.ripple_z_mean.isna().tolist() == [True, True, False, False]
    assert table.mua_z_max.notna().all()
    assert table.mua_z_mean.notna().all()


# --- gaps to the neighbouring events -------------------------------------


def test_three_events_give_nan_two_gaps_nan():
    events = inventory([[100.10, 100.20], [100.50, 100.60], [101.00, 101.40]], "kay")
    gaps = neighbour_gaps(events)
    np.testing.assert_allclose(gaps.previous_gap_s, [np.nan, 0.30, 0.40])
    np.testing.assert_allclose(gaps.next_gap_s, [0.30, 0.40, np.nan])


def test_another_inventorys_or_epochs_event_does_not_enter():
    kay = inventory([[100.10, 100.20], [101.00, 101.40]], "kay")
    between = inventory([[100.50, 100.60]], "state_space")
    other_epoch = inventory([[100.50, 100.60]], "kay", session=OTHER_EPOCH)
    events = pd.concat([between, kay, other_epoch], ignore_index=True)
    gaps = neighbour_gaps(events)
    np.testing.assert_allclose(gaps.previous_gap_s, [np.nan, np.nan, 0.80, np.nan])
    np.testing.assert_allclose(gaps.next_gap_s, [np.nan, 0.80, np.nan, np.nan])


def test_gaps_follow_onset_order_not_row_order():
    events = inventory([[100.10, 100.20], [100.50, 100.60], [101.00, 101.40]], "kay")
    gaps = neighbour_gaps(events.iloc[[2, 0, 1]])
    assert gaps.index.tolist() == [2, 0, 1]
    np.testing.assert_allclose(
        gaps.loc[[0, 1, 2], "previous_gap_s"], [np.nan, 0.3, 0.4]
    )


def test_gaps_refuse_an_event_that_is_not_compared():
    events = inventory([[100.10, 100.20], [100.50, 100.60]], "kay")
    events.loc[1, "whole_event_speed_eligible"] = False
    with pytest.raises(ValueError, match="compared"):
        neighbour_gaps(events)


# --- the stop holding the onset -------------------------------------------


STOPS = pd.DataFrame(
    {
        "stop_id": np.array([11, 7], dtype=np.uint64),
        "stop_start": [105.0, 100.0],
        "stop_end": [108.0, 103.0],
        "at_well": pd.array([True, pd.NA], dtype="boolean"),
        "occupancy_state": ["at_well", "unknown"],
        "occupancy_well_id": [2.0, np.nan],
        "current_patch": [2.0, np.nan],
        "is_rewarded_outcome": [1.0, np.nan],
        "last_well_arrival_time_s": [104.5, np.nan],
    }
)
"""Two stops, given out of time order."""


def test_an_onset_on_a_stops_closed_bound_is_in_that_stop():
    context = stop_context(
        intervals([[105.0, 105.2], [108.0, 108.3], [106.0, 106.1]]), STOPS
    )
    assert context.stop_id.tolist() == [11, 11, 11]
    assert context.at_well.tolist() == [True, True, True]
    assert context.occupancy_state.tolist() == ["at_well"] * 3
    np.testing.assert_allclose(context.time_in_stop_s, [0.0, 3.0, 1.0])
    np.testing.assert_allclose(context.time_since_arrival_s, [0.5, 3.5, 1.5])
    np.testing.assert_allclose(context.occupancy_well_id, [2.0] * 3)
    np.testing.assert_allclose(context.current_patch, [2.0] * 3)
    np.testing.assert_allclose(context.is_rewarded_outcome, [1.0] * 3)


def test_an_event_belongs_to_the_stop_of_its_onset_alone():
    context = stop_context(intervals([[102.9, 104.0]]), STOPS)
    assert context.stop_id.tolist() == [7]
    assert context.at_well.isna().all()
    assert context.time_since_arrival_s.isna().all()


def test_an_onset_in_no_stop_gives_missing_context():
    context = stop_context(
        intervals([[104.0, 106.0], [99.0, 100.5], [108.001, 108.2]]), STOPS
    )
    assert context.stop_id.isna().all()
    assert context.at_well.isna().all()
    assert context.occupancy_state.isna().all()
    for column in (
        "occupancy_well_id",
        "current_patch",
        "is_rewarded_outcome",
        "time_in_stop_s",
        "time_since_arrival_s",
    ):
        assert context[column].isna().all()


def test_stop_ids_keep_their_full_precision():
    stops = STOPS.assign(
        stop_id=np.array([6840840862921915817, 8785710247612500823], dtype=np.uint64)
    )
    context = stop_context(intervals([[100.5, 100.6], [101.0, 101.1]]), stops)
    assert context.stop_id.tolist() == [8785710247612500823] * 2


# --- duration strata -----------------------------------------------------


def test_duration_strata_are_closed_on_the_right():
    assert DURATION_STRATA_S == (0.0, 0.050, 0.100, 0.200, 0.500, np.inf)
    strata = duration_stratum(np.array([0.002, 0.050, 0.052, 0.100, 0.5, 0.502, 3.0]))
    assert strata.tolist() == [
        "0-50 ms",
        "0-50 ms",
        "50-100 ms",
        "50-100 ms",
        "200-500 ms",
        ">500 ms",
        ">500 ms",
    ]
    assert list(strata.categories) == [
        "0-50 ms",
        "50-100 ms",
        "100-200 ms",
        "200-500 ms",
        ">500 ms",
    ]


def test_a_grid_span_on_a_bound_lands_on_the_bound_despite_rounding():
    start = 1581571621.423794
    grid = start + SAMPLE_INTERVAL_S * np.arange(30)
    span = grid[25] - grid[0]
    assert span != 0.050  # the subtraction leaves float error
    assert duration_stratum(np.array([span])).tolist() == ["0-50 ms"]
