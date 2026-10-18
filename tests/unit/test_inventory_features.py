"""Per-event features read on the decode grid."""

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from non_local_detector.environment import Environment
from ripple_detection.core import minimum_sample_count

from continuum_swr_replay.analysis.hippocampus.event_detection import (
    get_confident_times,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.config import (
    MemorylessSequenceConfig,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.detect import (
    detect_candidate_sequences,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.likelihood import (
    interior_bin_distances_cm,
    window_midpoints_s,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.statistics import (
    dispersion_cm,
    windows_holding,
)
from continuum_swr_replay.analysis.inventory_comparison.evidence import animal_bins
from continuum_swr_replay.analysis.inventory_comparison.features import (
    DURATION_STRATA_S,
    PosteriorTraces,
    RowTraces,
    _over_rows,
    duration_stratum,
    interval_features,
    interval_rows,
    ms_features,
    neighbour_gaps,
    ss_features,
    stop_context,
)
from continuum_swr_replay.analysis.inventory_comparison.recount import (
    FRAGMENT_GAPS_S,
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


# --- posterior features: the state-space decode (ss_) --------------------

DISTANCE_CM = np.array([[0.0, 10.0, 20.0], [10.0, 0.0, 10.0], [20.0, 10.0, 0.0]])
"""Track distances of a three-bin path: bin 1 lies between bins 0 and 2."""
PATH_DISPERSION_CM = np.sqrt(200.0 / 3.0)
"""``dispersion_cm`` of the path's three bins by hand: about bin 1, the
squared distances are 100, 0 and 100 cm^2."""
STATE_COLUMNS = (
    "Local_prob",
    "No-Spike_prob",
    "Non-Local_Continuous_prob",
    "Non-Local_Fragmented_prob",
)


def posterior_frame(non_local=None, **columns) -> pd.DataFrame:
    """The decode frame's state columns and ``hpd`` on ``TIME``, as float32.

    ``non_local`` sets the continuous state's probability, the fragmented
    state's 0 and the local state's the rest.
    """
    n_time = TIME.size
    frame = {
        "Local_prob": np.full(n_time, 0.7),
        "No-Spike_prob": np.full(n_time, 0.1),
        "Non-Local_Continuous_prob": np.full(n_time, 0.15),
        "Non-Local_Fragmented_prob": np.full(n_time, 0.05),
        "hpd": np.linspace(5.0, 50.0, n_time),
    }
    if non_local is not None:
        non_local = np.asarray(non_local, dtype=float)
        frame |= {
            "Local_prob": 1.0 - non_local,
            "No-Spike_prob": np.zeros(n_time),
            "Non-Local_Continuous_prob": non_local,
            "Non-Local_Fragmented_prob": np.zeros(n_time),
        }
    frame.update(columns)
    frame = {
        name: np.asarray(values, dtype=np.float32 if name in STATE_COLUMNS else float)
        for name, values in frame.items()
    }
    return pd.DataFrame(frame, index=pd.Index(TIME, name="time"))


def posterior(non_local=None, **columns) -> PosteriorTraces:
    return PosteriorTraces.from_frame(posterior_frame(non_local, **columns))


def ss(
    table,
    row_posterior=None,
    *,
    map_bin=None,
    animal_bin=None,
    distance=DISTANCE_CM,
    state_space_events=None,
) -> pd.DataFrame:
    """``ss_features`` on the flat traces; MAP and animal on bin 1 by default."""
    if row_posterior is None:
        row_posterior = posterior()
    if map_bin is None:
        map_bin = np.ones(TIME.size, dtype=np.int64)
    if animal_bin is None:
        animal_bin = np.ones(TIME.size, dtype=np.int64)
    if state_space_events is None:
        state_space_events = (
            table[table["detector"] == "state_space"]
            if "whole_event_speed_eligible" in table
            else inventory(np.empty((0, 2)), "state_space")
        )
    return ss_features(
        traces(),
        row_posterior,
        table,
        distance,
        map_bin,
        animal_bin,
        state_space_events=state_space_events,
    )


def on_rows(first, last, detector="kay"):
    """An inventory of one event whose rows are ``first`` to ``last``."""
    return inventory([[TIME[first], TIME[last]]], detector)


def test_the_state_means_and_the_non_local_maximum_over_the_rows():
    rng = np.random.default_rng(1)
    states = rng.dirichlet(np.ones(4), size=TIME.size)
    row_posterior = posterior(
        **{name: states[:, k] for k, name in enumerate(STATE_COLUMNS)},
        hpd=rng.uniform(2.0, 80.0, TIME.size),
    )
    out = ss(on_rows(100, 140), row_posterior).iloc[0]
    rows = slice(100, 141)
    stored = states.astype(np.float32)
    for k, name in enumerate(
        ("local", "no_spike", "continuous", "fragmented"),
    ):
        assert out[f"ss_{name}_prob_mean"] == pytest.approx(
            stored[rows, k].astype(float).mean(), rel=1e-12
        )
    non_local = (stored[:, 2] + stored[:, 3]).astype(float)
    assert out["ss_nonlocal_prob_max"] == non_local[rows].max()
    assert out["ss_hpd_cm_mean"] == pytest.approx(
        row_posterior.hpd_cm[rows].mean(), rel=1e-12
    )


def test_the_fraction_of_rows_above_the_non_local_threshold():
    non_local = np.zeros(TIME.size)
    non_local[100:110] = 0.9
    out = ss(on_rows(100, 119), posterior(non_local)).iloc[0]
    assert out["ss_frac_rows_nonlocal"] == pytest.approx(10 / 20)


def test_a_row_at_the_threshold_is_not_above_it_as_in_get_confident_times():
    # the frame stores float32, and the event rule compares in float32: a row
    # that reads 0.8 there is not above 0.8, although its float64 value is
    non_local = np.zeros(TIME.size)
    non_local[100:120] = 0.8
    row_posterior = posterior(non_local)
    events = get_confident_times(
        xr.DataArray(row_posterior.nonlocal_prob, dims="time", coords={"time": TIME}),
        minimum_duration=0.0,
    )
    out = ss(on_rows(90, 130), row_posterior).iloc[0]
    assert len(events) == 0
    assert out["ss_frac_rows_nonlocal"] == 0.0
    assert out["ss_n_runs_nonlocal"] == 0


def test_the_mean_graph_distance_from_the_map_to_the_animal():
    map_bin = np.ones(TIME.size, dtype=np.int64)
    map_bin[100:102] = [0, 2]
    animal_bin = np.zeros(TIME.size, dtype=np.int64)
    out = ss(on_rows(100, 102), map_bin=map_bin, animal_bin=animal_bin).iloc[0]
    assert out["ss_distance_cm_mean"] == pytest.approx((0.0 + 20.0 + 10.0) / 3)


def test_an_unknown_map_or_animal_bin_makes_the_spatial_features_nan():
    map_bin = np.ones(TIME.size, dtype=np.int64)
    animal_bin = np.ones(TIME.size, dtype=np.int64)
    map_bin[105] = -1
    animal_bin[305] = -1
    table = inventory([[TIME[100], TIME[110]], [TIME[300], TIME[310]]], "kay")
    out = ss(table, map_bin=map_bin, animal_bin=animal_bin)
    assert np.isnan(out["ss_extent_cm"].iloc[0])
    assert np.isnan(out["ss_distance_cm_mean"]).all()
    assert out["ss_extent_cm"].iloc[1] == 0.0


def test_a_nan_posterior_row_makes_the_posterior_features_nan():
    local = np.full(TIME.size, 0.7)
    local[105] = np.nan
    table = inventory([[TIME[100], TIME[110]], [TIME[300], TIME[310]]], "kay")
    out = ss(table, posterior(Local_prob=local))
    assert np.isnan(out["ss_local_prob_mean"].iloc[0])
    assert np.isfinite(out["ss_local_prob_mean"].iloc[1])


def test_an_interval_with_no_row_has_every_ss_feature_missing():
    out = ss(intervals_of("kay", [[TIME[100] + 0.0005, TIME[100] + 0.0015]])).iloc[0]
    for name in out.index:
        if name in ("ss_nearest_event_gap_s", "ss_disagreement"):
            continue
        assert pd.isna(out[name]), name


def test_ss_features_refuse_an_interval_reaching_the_last_sample():
    with pytest.raises(ValueError, match="last sample"):
        ss(on_rows(990, TIME.size - 1))


def test_ss_features_refuse_a_bin_outside_the_distance_matrix():
    map_bin = np.ones(TIME.size, dtype=np.int64)
    map_bin[5] = DISTANCE_CM.shape[0]  # an index into the full grid, say
    with pytest.raises(ValueError, match="real place bins"):
        ss(on_rows(100, 110), map_bin=map_bin)


def intervals_of(detector, bounds) -> pd.DataFrame:
    """Intervals with a detector, which need not be events of the schema."""
    table = intervals(bounds)
    table["detector"] = detector
    return table


# --- runs of rows above the threshold, and the comparator-only classes ----


def test_runs_are_counted_on_the_rows_inside_the_interval():
    non_local = np.zeros(TIME.size)
    non_local[100:104] = 0.9  # 4 rows
    non_local[110:120] = 0.9  # 10 rows, cut by the interval's end at row 114
    out = ss(on_rows(95, 114), posterior(non_local)).iloc[0]
    assert out["ss_n_runs_nonlocal"] == 2
    # rows 110 to 114: the last sample's time less the first's
    assert out["ss_longest_run_nonlocal_s"] == TIME[114] - TIME[110]


def test_no_run_has_no_longest_run():
    out = ss(on_rows(100, 120), posterior(np.zeros(TIME.size))).iloc[0]
    assert out["ss_n_runs_nonlocal"] == 0
    assert np.isnan(out["ss_longest_run_nonlocal_s"])


def test_a_run_forms_an_event_exactly_when_get_confident_times_would_keep_it():
    # get_confident_times keeps a run of at least minimum_sample_count(time,
    # 0.015) rows above 0.8 and gives it the last sample's time less the
    # first's as its duration; the run features read the same samples
    n_min = minimum_sample_count(TIME, 0.015)
    non_local = np.zeros(TIME.size)
    non_local[100 : 100 + n_min] = 0.9
    non_local[300 : 300 + n_min - 1] = 0.9
    row_posterior = posterior(non_local)
    events = get_confident_times(
        xr.DataArray(row_posterior.nonlocal_prob, dims="time", coords={"time": TIME}),
    )
    table = inventory([[TIME[90], TIME[200]], [TIME[290], TIME[400]]], "kay")
    out = ss(table, row_posterior)
    assert list(events["start_time"]) == [TIME[100]]
    assert out["ss_longest_run_nonlocal_s"].iloc[0] == events["duration"].iloc[0]
    # one row fewer forms no event, and reads one sample shorter
    assert out["ss_longest_run_nonlocal_s"].iloc[1] == TIME[300 + n_min - 2] - TIME[300]


def test_the_comparator_only_classes():
    non_local = np.zeros(TIME.size)
    non_local[200:221] = 0.95  # our event, rows 200 to 220
    non_local[510:516] = 0.9  # a 10 ms run: rows 510 to 515
    non_local[700:731] = 0.7  # below the threshold throughout
    ours = inventory([[TIME[200], TIME[220]]], "state_space")
    theirs = inventory(
        [
            [TIME[170], TIME[180]],  # ends 40 ms before ours starts
            [TIME[500], TIME[530]],
            [TIME[700], TIME[730]],
        ],
        "kay",
    )
    table = pd.concat([ours, theirs], ignore_index=True)
    out = ss(table, posterior(non_local))
    assert np.isnan(out["ss_nearest_event_gap_s"].iloc[0])
    assert pd.isna(out["ss_disagreement"].iloc[0])
    assert out["ss_nearest_event_gap_s"].iloc[1] == pytest.approx(0.040)
    assert list(out["ss_disagreement"].iloc[1:]) == [
        "near",
        "subthreshold",
        "sustained",
    ]
    assert out["ss_longest_run_nonlocal_s"].iloc[2] == pytest.approx(0.010)


def test_the_nearest_event_is_on_either_side_and_an_overlap_is_no_gap():
    ours = inventory(
        [[TIME[100], TIME[110]], [TIME[300], TIME[310]], [TIME[600], TIME[610]]],
        "state_space",
    )
    theirs = intervals_of(
        "kay",
        [
            [TIME[250], TIME[290]],  # 10 rows before the second, 140 after the first
            [TIME[320], TIME[330]],  # 10 rows after the second
            [TIME[305], TIME[315]],  # overlaps the second
            [TIME[700], TIME[710]],  # after the last
        ],
    )
    out = ss(theirs, state_space_events=ours)
    np.testing.assert_allclose(
        out["ss_nearest_event_gap_s"], [0.020, 0.020, 0.0, TIME[700] - TIME[610]]
    )


def test_a_gap_of_exactly_the_widest_fragment_gap_is_near_despite_rounding():
    # on a clock near 1.6e9 s, 25 samples of 2 ms read slightly over 50 ms
    time = 1.6e9 + SAMPLE_INTERVAL_S * np.arange(TIME.size)
    ours = pd.DataFrame(
        {
            "event_start_s": [time[200]],
            "event_end_s": [time[220]],
            "whole_event_speed_eligible": [True],
        }
    )
    theirs = pd.DataFrame(
        {
            "event_start_s": [time[160], time[159]],
            "event_end_s": [time[175], time[174]],
            "detector": "kay",
        }
    )
    row_traces = RowTraces.from_frame(
        decode_frame().set_axis(pd.Index(time, name="time")), [np.array([])]
    )
    out = ss_features(
        row_traces,
        posterior(),
        theirs,
        DISTANCE_CM,
        np.ones(time.size, dtype=np.int64),
        np.ones(time.size, dtype=np.int64),
        state_space_events=ours,
    )
    assert out["ss_nearest_event_gap_s"].iloc[0] > max(FRAGMENT_GAPS_S)
    assert list(out["ss_disagreement"]) == ["near", "sustained"]


def test_the_nearest_event_must_be_a_compared_one():
    ours = inventory([[TIME[200], TIME[220]]], "state_space")
    ours["whole_event_speed_eligible"] = False
    with pytest.raises(ValueError, match="compared"):
        ss(on_rows(100, 110), state_space_events=ours)


# --- posterior features: the memoryless windows (ms_) --------------------

CONFIG = MemorylessSequenceConfig()
MIDPOINT_S = window_midpoints_s(TIME[:-1], CONFIG)
"""The windows' midpoints on the grid less its last sample, as the run makes
them: window ``i``'s midpoint is row ``2 i + 20``."""
N_WINDOWS = MIDPOINT_S.size


def window_of(row: int) -> int:
    """The window whose midpoint is ``row``."""
    return (row - CONFIG.window_n_samples // 2) // CONFIG.stride_n_samples


def memoryless(window_posterior=None, *, speed=None, undecoded=None):
    """The stored candidates of windows over the three-bin path.

    Every window's posterior is on bin 1 unless given; ``undecoded`` windows
    are all NaN, as the run leaves the windows holding a masked sample.
    """
    if window_posterior is None:
        window_posterior = np.zeros((N_WINDOWS, 3))
        window_posterior[:, 1] = 1.0
    window_posterior = np.array(window_posterior, dtype=float)
    if undecoded is not None:
        window_posterior[undecoded] = np.nan
    return detect_candidate_sequences(
        window_posterior,
        MIDPOINT_S,
        DISTANCE_CM,
        np.zeros(N_WINDOWS) if speed is None else speed,
        CONFIG,
        unobserved=undecoded,
    )


def ms(
    table,
    candidates=None,
    *,
    undecoded=None,
    animal_bin_at_window=None,
    distance=DISTANCE_CM,
) -> pd.DataFrame:
    if candidates is None:
        candidates = memoryless(undecoded=undecoded)
    if undecoded is None:
        undecoded = np.zeros(N_WINDOWS, dtype=bool)
    if animal_bin_at_window is None:
        animal_bin_at_window = np.ones(N_WINDOWS, dtype=np.int64)
    return ms_features(candidates, table, undecoded, distance, animal_bin_at_window)


def one_hot(centers) -> np.ndarray:
    window_posterior = np.zeros((N_WINDOWS, 3))
    window_posterior[np.arange(N_WINDOWS), centers] = 1.0
    return window_posterior


def test_the_windows_whose_midpoint_lies_in_the_interval():
    # rows 100 to 104 hold the midpoints of windows 40, 41 and 42
    out = ms(on_rows(100, 104)).iloc[0]
    assert window_of(100) == 40
    assert out["ms_n_windows"] == 3
    assert out["ms_n_windows_decoded"] == 3
    assert out["ms_n_windows_with_center"] == 3
    assert out["ms_frac_windows_kept"] == 1.0


def test_an_interval_shorter_than_the_stride_holds_no_window():
    # row 101 lies between the midpoints of windows 40 and 41
    out = ms(intervals(([TIME[101], TIME[101] + 0.0015],))).iloc[0]
    assert out["ms_n_windows"] == 0
    for name in out.index.drop("ms_n_windows"):
        assert pd.isna(out[name]), name


def test_spread_and_distance_are_means_over_the_windows():
    window_posterior = one_hot(np.ones(N_WINDOWS, dtype=int))
    window_posterior[40] = [0.25, 0.5, 0.25]  # center 1, spread sqrt(50) cm
    window_posterior[41] = [0.0, 0.0, 1.0]
    animal = np.zeros(N_WINDOWS, dtype=np.int64)
    out = ms(
        on_rows(100, 104),
        memoryless(window_posterior),
        animal_bin_at_window=animal,
    ).iloc[0]
    assert out["ms_spread_cm_mean"] == pytest.approx(np.sqrt(50.0) / 3)
    # centers 1, 2, 1 against the animal's bin 0
    assert out["ms_distance_cm_mean"] == pytest.approx((10.0 + 20.0 + 10.0) / 3)


def test_a_window_the_speed_gate_blanked_has_a_spread_but_no_center():
    speed = np.zeros(N_WINDOWS)
    speed[41] = 10.0
    out = ms(on_rows(100, 104), memoryless(speed=speed)).iloc[0]
    assert out["ms_n_windows_decoded"] == 3
    assert out["ms_n_windows_with_center"] == 2
    assert out["ms_spread_cm_mean"] == 0.0
    # window 40's jump to the blanked window is undefined, so only 42 is kept
    assert out["ms_frac_windows_kept"] == pytest.approx(1 / 3)


def test_an_undecoded_window_is_counted_apart_and_enters_no_mean():
    masked = np.zeros(TIME.size, dtype=bool)
    masked[60] = True  # held by the windows starting at rows 21 to 60
    undecoded = windows_holding(
        masked[:-1], CONFIG.window_n_samples, CONFIG.stride_n_samples
    )
    assert undecoded[window_of(80)] and not undecoded[window_of(82)]
    out = ms(on_rows(78, 84), undecoded=undecoded).iloc[0]
    assert out["ms_n_windows"] == 4
    assert out["ms_n_windows_decoded"] == 2
    assert out["ms_n_windows_with_center"] == 2
    assert out["ms_frac_windows_kept"] == 1.0
    assert out["ms_spread_cm_mean"] == 0.0


def test_ms_features_refuse_windows_of_another_length():
    with pytest.raises(ValueError, match="windows"):
        ms(
            on_rows(100, 104),
            memoryless(),
            undecoded=np.zeros(N_WINDOWS - 1, dtype=bool),
        )


# --- both namespaces ------------------------------------------------------


def test_extent_is_the_dispersion_of_a_three_bin_path_in_both_namespaces():
    assert dispersion_cm(np.array([0, 1, 2]), DISTANCE_CM) == pytest.approx(
        PATH_DISPERSION_CM
    )
    map_bin = np.ones(TIME.size, dtype=np.int64)
    map_bin[100:103] = [0, 1, 2]
    centers = np.ones(N_WINDOWS, dtype=int)
    centers[40:43] = [0, 1, 2]  # the windows whose midpoints are rows 100 to 104
    candidates = memoryless(one_hot(centers))
    for scale in (1.0, 2.0):  # one distance matrix: doubling it doubles both
        distance = scale * DISTANCE_CM
        ss_extent = ss(on_rows(100, 102), map_bin=map_bin, distance=distance)
        ms_extent = ms(on_rows(100, 104), candidates, distance=distance)
        assert ss_extent["ss_extent_cm"].iloc[0] == pytest.approx(
            scale * PATH_DISPERSION_CM
        )
        assert ms_extent["ms_extent_cm"].iloc[0] == pytest.approx(
            scale * PATH_DISPERSION_CM
        )


def test_identical_intervals_give_identical_ss_and_ms_features():
    rng = np.random.default_rng(2)
    states = rng.dirichlet(np.ones(4), size=TIME.size)
    row_posterior = posterior(
        **{name: states[:, k] for k, name in enumerate(STATE_COLUMNS)},
        hpd=rng.uniform(2.0, 80.0, TIME.size),
    )
    map_bin = rng.integers(0, 3, TIME.size)
    animal_bin = rng.integers(0, 3, TIME.size)
    candidates = memoryless(rng.dirichlet(np.ones(3), size=N_WINDOWS))
    animal_at_window = rng.integers(0, 3, N_WINDOWS)
    shared = [[TIME[100] + 0.0003, TIME[160] - 0.0004], [TIME[400], TIME[455]]]
    first = inventory(shared + [[TIME[600], TIME[650]]], "kay")
    second = inventory([[TIME[50], TIME[90]]] + shared, "carey")
    ours = inventory([[TIME[300], TIME[320]]], "state_space")
    table = pd.concat([first, second, ours], ignore_index=True)
    out = pd.concat(
        [
            ss(table, row_posterior, map_bin=map_bin, animal_bin=animal_bin),
            ms(table, candidates, animal_bin_at_window=animal_at_window),
        ],
        axis=1,
    )
    pd.testing.assert_frame_equal(
        out.iloc[[0, 1]].reset_index(drop=True),
        out.iloc[[4, 5]].reset_index(drop=True),
        check_exact=True,
    )


# --- junction placeholder bins --------------------------------------------

ARM_CM = 20.0
GAP_CM = 8.0


@pytest.fixture(scope="module")
def y_environment():
    """A fitted Y track whose arms sit ``GAP_CM`` apart in linear position."""
    graph = nx.Graph()
    graph.add_edges_from([(0, 3), (1, 3), (2, 3)])
    nx.set_edge_attributes(graph, dict.fromkeys(graph.edges, ARM_CM), "distance")
    nx.set_edge_attributes(graph, {(0, 3): 0, (1, 3): 1, (2, 3): 2}, "edge_id")
    nx.set_node_attributes(
        graph,
        {0: (0.0, 0.0), 1: (2 * ARM_CM, 0.0), 2: (ARM_CM, ARM_CM), 3: (ARM_CM, 0.0)},
        "pos",
    )
    covered = np.concatenate(
        [
            np.linspace(0.0, ARM_CM, 200),
            np.linspace(ARM_CM + GAP_CM, 2 * ARM_CM + GAP_CM, 200),
            np.linspace(2 * (ARM_CM + GAP_CM), 3 * ARM_CM + 2 * GAP_CM, 200),
        ]
    )
    return Environment(
        place_bin_size=4.0,
        track_graph=graph,
        edge_order=[(0, 3), (3, 1), (3, 2)],
        edge_spacing=GAP_CM,
    ).fit_place_grid(covered[:, None], infer_track_interior=True)


def test_a_map_in_a_junction_gap_maps_to_a_real_bin(y_environment):
    nodes = y_environment.place_bin_centers_nodes_df_
    interior = np.asarray(y_environment.is_track_interior_).ravel()
    assert (nodes["edge_id"].to_numpy() == -1).sum() > 0
    distance = interior_bin_distances_cm(y_environment)
    real_centers = np.asarray(y_environment.place_bin_centers_).ravel()[interior]
    # MAP estimates in the gap between the first and second arms
    in_gap = ARM_CM + np.array([1.0, GAP_CM / 2, GAP_CM - 1.0])
    bins = animal_bins(y_environment, in_gap)
    assert ((bins >= 0) & (bins < distance.shape[0])).all()
    # each lands on a real bin at an end of an arm, never a placeholder's center
    placeholder_centers = nodes.loc[nodes["edge_id"] == -1, "linear_position"]
    assert not np.isin(real_centers[bins], placeholder_centers.to_numpy()).any()
    map_bin = np.zeros(TIME.size, dtype=np.int64)
    map_bin[100:103] = bins
    animal_bin = np.zeros(TIME.size, dtype=np.int64)
    out = ss(
        on_rows(100, 102), map_bin=map_bin, animal_bin=animal_bin, distance=distance
    ).iloc[0]
    assert out["ss_distance_cm_mean"] == pytest.approx(distance[bins, 0].mean())
    assert out["ss_extent_cm"] == pytest.approx(dispersion_cm(bins, distance))


# --- reductions reaching the end of an array ------------------------------


@pytest.mark.parametrize(
    ("ufunc", "expected"),
    [(np.add, [3.0, 9.0, 9.0]), (np.maximum, [2.0, 4.0, 5.0])],
)
def test_over_rows_reads_an_interval_ending_at_the_arrays_end(ufunc, expected):
    values = np.arange(6, dtype=float)
    first, n_rows = np.array([0, 2, 4]), np.array([3, 3, 2])
    np.testing.assert_array_equal(_over_rows(ufunc, values, first, n_rows), expected)


def test_over_rows_carries_a_nan_in_the_last_row_to_its_interval_alone():
    values = np.array([1.0, 2.0, 3.0, np.nan])
    reduced = _over_rows(np.add, values, np.array([0, 2]), np.array([2, 2]))
    assert reduced[0] == 3.0 and np.isnan(reduced[1])


def test_ms_features_read_an_interval_holding_the_last_window():
    out = ms(inventory([[MIDPOINT_S[-3], MIDPOINT_S[-1]]], "kay")).iloc[0]
    assert out["ms_n_windows"] == 3
    assert out["ms_n_windows_decoded"] == 3
    assert out["ms_frac_windows_kept"] == pytest.approx(
        memoryless().decisions.gates.kept[-3:].mean()
    )
