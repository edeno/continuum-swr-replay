"""Every rate denominator loses excluded time: the mask widened by 0.3 s.

Excluded time is not analyzed, so it is no one's exposure:

- a stop's immobility exposure (Family A's offset, the ContFrag stop table's
  too) falls by exactly the excluded seconds inside the stop, and nothing else
  about the stop changes;
- a per-bin row that touches excluded time is dropped (Family B's design, the
  PFC fit bins), and so is one whose history or
  peri-event window reaches it, and a matched control never lands on a
  dropped event;
- with no artifact run configured, every one of these is what it was.

Offline: the artifact run is written under ``tmp_path`` and the source pointed
at it, as ``test_swr_source.py`` does.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.contfrag.tables import build_contfrag_stop_table
from continuum_swr_replay.analysis.event_definition import analyzed_time, swr_source
from continuum_swr_replay.analysis.event_tables.stop_table import build_stop_table
from continuum_swr_replay.analysis.pfc.event_sets import build_session_fit_mask
from continuum_swr_replay.analysis.replay_models.model_specs import (
    MODEL_PER_BIN_RATE,
    N_HISTORY_LAGS,
    RATE_BIN_S,
    SWR_CONTRAST_RATE_G_SPECS,
    SWR_CONTRAST_RATE_S_SPECS,
    longest_history_window_s,
)
from continuum_swr_replay.analysis.replay_models.per_bin_design import (
    NO_EXCLUDED_TIME,
    aggregate_to_timebins,
)
from continuum_swr_replay.tables.io import SWR_SOURCE_TABLES
from tests.support._port_experience_fixtures import port_experiences_from_trials
from tests.support._swr_inventory_fixtures import use_artifact_run, write_artifact_run

EPOCH_ID = "m20200101_02_r1_HPC"
TIME = np.round(np.arange(0.0, 20.0, 0.002), 6)
"""A 500 Hz grid, 0-19.998 s."""
CORES = [("S", 3.0, 4.0), ("spk", 15.95, 15.95)]
"""An S block padded by 1 s, [2.0, 5.0], and a spike-only line padded by
0.1 s, [15.85, 16.05], which straddles the second bout's start at 16.0."""
MASK = [(2.0, 5.0), (15.85, 16.05)]
EXCLUDED = [(1.7, 5.3), (15.55, 16.35)]
"""MASK widened by 0.3 s on each side."""

#: Stop columns summarized over the stop's rows outside excluded time.
LOCATION_COLUMNS = [
    "occupancy_state",
    "at_well",
    "occupancy_well_id",
    "modal_observed_track_segment_id",
    "nearest_well_distance_cm_min",
    "nearest_well_distance_cm_mean",
    "nearest_well_distance_cm_max",
    "occupancy_well_arm",
    "current_patch",
    "is_stay",
    # The well-visit label reads the same rows (event_tables/visit_context.py).
    "visit_context",
    "well_visit_id",
    "last_completed_well_visit_id",
    "time_since_last_poke_out_s",
    "time_until_next_poke_in_s",
]


def _speed() -> np.ndarray:
    speed = np.zeros(TIME.size)
    speed[(TIME > 10.0) & (TIME < 16.0)] = 10.0  # bouts [0, 10] and [16, 19.998]
    return speed


def _dot() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "animal": "m",
            "day": 20200101,
            "epoch": "02_r1",
            "head_speed": _speed(),
            "Non-Local_Continuous_prob": 0.0,
            "Non-Local_Fragmented_prob": 0.0,
            "Continuous_prob": 0.5,
            "Fragmented_prob": 0.5,
            "non_local_event": 0,
            "SWR_event": 0,
            "MUA_event": 0,
            "track_segment_id": np.nan,
            "map_estimate": np.nan,
        },
        index=pd.Index(TIME, name="time"),
    )


def _graph() -> nx.Graph:
    graph = nx.Graph()
    graph.add_node(0, pos=(0.0, 0.0))
    graph.add_node(1, pos=(1.0, 0.0))
    graph.add_edge(0, 1, edge_id=0, distance=1.0)
    return graph


def _trials() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "movement_start_time": [11.0],
            "end_time": [15.0],
            "from_well": [1],
            "to_well": [0],
            "is_reward": [True],
            "is_patch_change": [False],
            "outcome_known_time": [15.0],
        }
    )


@pytest.fixture
def masked(tmp_path, monkeypatch):
    """The source's artifact run holds CORES for the test epoch."""
    directory = write_artifact_run(tmp_path, {EPOCH_ID: CORES})
    use_artifact_run(monkeypatch, directory)
    np.testing.assert_allclose(swr_source.artifact_intervals(EPOCH_ID).to_numpy(), MASK)
    np.testing.assert_allclose(
        analyzed_time.excluded_intervals(EPOCH_ID).to_numpy(), EXCLUDED
    )


def _stop_tables() -> dict[str, pd.DataFrame]:
    trials = _trials()
    experiences = port_experiences_from_trials(trials, prepend_initial=True)
    return {
        "nonlocal": build_stop_table(_dot(), trials, experiences, _graph()),
        "contfrag": build_contfrag_stop_table(_dot(), trials, experiences, _graph()),
    }


class TestStopExposure:
    @pytest.mark.parametrize("table", ["nonlocal", "contfrag"])
    def test_falls_by_exactly_the_excluded_time_inside_the_stop(self, masked, table):
        stops = _stop_tables()[table]

        assert stops["stop_start"].tolist() == [0.0, 16.0]
        span = (stops["stop_end"] - stops["stop_start"]).to_numpy()
        # 3.6 s of [1.7, 5.3] in the first stop; [16.0, 16.35] of the line's
        # excluded time in the second (its part before 16.0 is not the stop's)
        np.testing.assert_allclose(
            span - stops["immobility_duration_s"].to_numpy(), [3.6, 0.35]
        )

    @pytest.mark.parametrize("table", ["nonlocal", "contfrag"])
    def test_only_exposure_and_location_change(self, tmp_path, monkeypatch, table):
        unmasked = _stop_tables()[table]
        directory = write_artifact_run(tmp_path, {EPOCH_ID: CORES})
        use_artifact_run(monkeypatch, directory)

        masked_stops = _stop_tables()[table]

        # The location summaries follow the stop's rows outside excluded time;
        # test_stop_location_excluded_time.py tests them.
        changed = ["immobility_duration_s", *LOCATION_COLUMNS]
        pd.testing.assert_frame_equal(
            masked_stops.drop(columns=changed, errors="ignore"),
            unmasked.drop(columns=changed, errors="ignore"),
        )

    @pytest.mark.parametrize("table", ["nonlocal", "contfrag"])
    def test_with_no_artifact_run_it_is_the_stops_span(self, table):
        stops = _stop_tables()[table]

        np.testing.assert_array_equal(
            stops["immobility_duration_s"].to_numpy(),
            (stops["stop_end"] - stops["stop_start"]).to_numpy(),
        )


def _design_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    stops = pd.DataFrame(
        {
            "animal": "m",
            "session_date": 20200101,
            "epoch": "02_r1",
            "stop_id": [11, 12],
            "stop_start": [0.0, 16.0],
            "stop_end": [10.0, 19.998],
            # Family A's exposure, the span less excluded time
            "immobility_duration_s": [6.4, 3.648],
        }
    )
    onsets = pd.DataFrame(
        {"stop_id": [11, 11, 11, 12], "t_since_stop_onset": [1.0, 5.6, 8.0, 2.0]}
    )
    return stops, onsets


def _mask_by_session() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "animal": "m",
            "session_date": 20200101,
            "epoch": "02_r1",
            "start_time": [s for s, _ in EXCLUDED],
            "end_time": [e for _, e in EXCLUDED],
        }
    )


class TestFamilyBDesign:
    def test_the_longest_history_window_comes_from_the_ladder(self):
        assert longest_history_window_s(MODEL_PER_BIN_RATE.values()) == pytest.approx(
            N_HISTORY_LAGS * RATE_BIN_S
        )
        assert longest_history_window_s(
            [*SWR_CONTRAST_RATE_G_SPECS.values(), *SWR_CONTRAST_RATE_S_SPECS.values()]
        ) == pytest.approx(N_HISTORY_LAGS * RATE_BIN_S)

    def test_bins_are_built_over_the_stops_whole_span(self):
        """The bins' clock is the bout's: Family A's shorter exposure must not
        cut the last bins off."""
        stops, onsets = _design_inputs()

        design = aggregate_to_timebins(
            stops, onsets, bin_s=0.5, tmax=20.0, arrival_clocks=False
        )

        assert design.loc[design["stop_id"] == 11, "binj"].max() == 19
        assert design["count"].sum() == 4

    def test_bins_in_the_mask_and_after_it_by_the_history_window_are_dropped(self):
        stops, onsets = _design_inputs()
        full = aggregate_to_timebins(
            stops,
            onsets,
            bin_s=0.5,
            tmax=20.0,
            excluded_intervals=NO_EXCLUDED_TIME,
            arrival_clocks=False,
        )

        design = aggregate_to_timebins(
            stops,
            onsets,
            bin_s=0.5,
            tmax=20.0,
            excluded_intervals=_mask_by_session(),
            history_window_s=1.0,
            arrival_clocks=False,
        )

        first = design.loc[design["stop_id"] == 11, "binj"].tolist()
        # [1.5, 2] touches excluded time's start, [5, 5.5] its end; [5.5, 6]
        # and [6, 6.5] look back into it within 1 s; [6.5, 7] does not
        assert first == [0, 1, 2, *range(13, 20)]
        # the second stop's first bin [16, 16.5] touches [15.55, 16.35], and the
        # next two look back at it; its history starts at the stop's start
        second = design.loc[design["stop_id"] == 12, "binj"].tolist()
        assert second == list(range(3, 8))
        # the onset at 5.6 s fell in a dropped bin; the others are counted
        assert design["count"].sum() == 3
        assert design.attrs["n_bins_excluded"] == 8 + 1
        assert design.attrs["n_bins_excluded_history"] == 2 + 2
        assert design.attrs["n_onsets_in_excluded_bins"] == 1
        # a kept row is the row of the design without the mask
        kept = full.merge(design[["stop_id", "binj"]], on=["stop_id", "binj"])
        pd.testing.assert_frame_equal(
            kept.reset_index(drop=True), design.reset_index(drop=True)
        )

    def test_history_is_computed_before_the_drop(self):
        """A kept bin's lags are its own past bins', never a dropped bin's
        shifted into place."""
        stops, onsets = _design_inputs()
        full = aggregate_to_timebins(
            stops,
            onsets,
            bin_s=0.5,
            tmax=20.0,
            self_history_lags=2,
            excluded_intervals=NO_EXCLUDED_TIME,
            arrival_clocks=False,
        )

        design = aggregate_to_timebins(
            stops,
            onsets,
            bin_s=0.5,
            tmax=20.0,
            self_history_lags=2,
            excluded_intervals=_mask_by_session(),
            history_window_s=1.0,
            arrival_clocks=False,
        )

        merged = design.merge(full, on=["stop_id", "binj"], suffixes=("", "_full"))
        np.testing.assert_array_equal(merged["onset_lag1"], merged["onset_lag1_full"])
        np.testing.assert_array_equal(merged["onset_lag2"], merged["onset_lag2_full"])

    def test_no_intervals_change_nothing(self):
        stops, onsets = _design_inputs()
        empty = _mask_by_session().iloc[0:0]

        full = aggregate_to_timebins(
            stops,
            onsets,
            bin_s=0.5,
            tmax=20.0,
            excluded_intervals=NO_EXCLUDED_TIME,
            arrival_clocks=False,
        )
        design = aggregate_to_timebins(
            stops,
            onsets,
            bin_s=0.5,
            tmax=20.0,
            excluded_intervals=empty,
            history_window_s=1.0,
            arrival_clocks=False,
        )

        pd.testing.assert_frame_equal(design, full)

    def test_by_default_excluded_time_is_read_from_the_source(self, masked):
        """A caller that names no mask gets its sessions' excluded time, not an
        unmasked design."""
        stops, onsets = _design_inputs()

        design = aggregate_to_timebins(
            stops,
            onsets,
            bin_s=0.5,
            tmax=20.0,
            history_window_s=1.0,
            arrival_clocks=False,
        )

        explicit = aggregate_to_timebins(
            stops,
            onsets,
            bin_s=0.5,
            tmax=20.0,
            excluded_intervals=_mask_by_session(),
            history_window_s=1.0,
            arrival_clocks=False,
        )
        pd.testing.assert_frame_equal(design, explicit, check_exact=True)
        assert design.attrs == explicit.attrs
        assert design.attrs["n_bins_excluded"] == 8 + 1

    def test_with_no_artifact_run_the_default_masks_nothing(self):
        stops, onsets = _design_inputs()

        design = aggregate_to_timebins(
            stops,
            onsets,
            bin_s=0.5,
            tmax=20.0,
            history_window_s=1.0,
            arrival_clocks=False,
        )

        full = aggregate_to_timebins(
            stops,
            onsets,
            bin_s=0.5,
            tmax=20.0,
            excluded_intervals=NO_EXCLUDED_TIME,
            arrival_clocks=False,
        )
        pd.testing.assert_frame_equal(design, full, check_exact=True)
        assert "n_bins_excluded" not in design.attrs

    def test_a_run_that_does_not_serve_the_session_raises(self, tmp_path, monkeypatch):
        """The default never falls back to an unmasked design; the no-mask
        sentinel still builds one."""
        directory = write_artifact_run(tmp_path, {"m20200101_04_r2_HPC": CORES})
        use_artifact_run(monkeypatch, directory)
        stops, onsets = _design_inputs()

        with pytest.raises(swr_source.SwrSourceError, match="not in the artifact run"):
            aggregate_to_timebins(
                stops, onsets, bin_s=0.5, tmax=20.0, arrival_clocks=False
            )
        design = aggregate_to_timebins(
            stops,
            onsets,
            bin_s=0.5,
            tmax=20.0,
            excluded_intervals=NO_EXCLUDED_TIME,
            arrival_clocks=False,
        )
        assert design["count"].sum() == 4


class TestPfcFitBins:
    def _position(self) -> pd.DataFrame:
        return pd.DataFrame({"head_speed": _speed()}, index=pd.Index(TIME, name="time"))

    def test_bins_whose_basis_window_reaches_excluded_time_are_not_fit(self):
        excluded = pd.DataFrame({"start_time": [2.0], "end_time": [5.0]})

        is_immobile, fit_mask, counts = build_session_fit_mask(
            self._position(),
            stop_cohort="all_stops",
            excluded_intervals=excluded,
        )

        reach = (TIME >= 1.75) & (TIME <= 5.25)
        np.testing.assert_array_equal(
            fit_mask.to_numpy(), is_immobile.to_numpy() & ~reach
        )
        # immobility itself is not redefined by the mask
        np.testing.assert_array_equal(is_immobile.to_numpy(), _speed() < 4.0)
        assert counts["n_bins_fit"] == int(fit_mask.sum())
        assert counts["n_bins_excluded"] == int(reach.sum())

    def test_the_excluded_count_is_of_bins_that_would_be_fit(self):
        """A running bin near excluded time was never going to be fit, so the
        count (and the run's log line) leaves it out."""
        excluded = pd.DataFrame({"start_time": [2.0, 12.0], "end_time": [5.0, 12.5]})

        is_immobile, fit_mask, counts = build_session_fit_mask(
            self._position(),
            stop_cohort="all_stops",
            excluded_intervals=excluded,
        )

        reach = ((TIME >= 1.75) & (TIME <= 5.25)) | ((TIME >= 11.75) & (TIME <= 12.75))
        assert counts["n_bins_excluded"] == int((is_immobile.to_numpy() & reach).sum())
        assert 0 < counts["n_bins_excluded"] < int(reach.sum())
        assert counts["n_bins_excluded"] == int(is_immobile.sum() - fit_mask.sum())

    def test_no_intervals_change_nothing(self):
        before = build_session_fit_mask(self._position(), stop_cohort="all_stops")
        after = build_session_fit_mask(
            self._position(),
            stop_cohort="all_stops",
            excluded_intervals=pd.DataFrame(
                {"start_time": [], "end_time": []}, dtype=float
            ),
            basis_half_window_s=0.25,
        )

        pd.testing.assert_series_equal(after[1], before[1])
        assert after[2] == before[2]
        assert after[2]["n_bins_excluded"] == 0


def test_every_rate_fit_drops_the_excluded_bins():
    """Each runner, or the package function it calls, that builds a per-bin
    design gives it its ladder's history window, as a keyword or through a
    ``**`` dict literal assigned in the same function, and leaves excluded time
    to the design, which reads it from the source."""
    import ast
    from pathlib import Path

    wanted = {"history_window_s"}
    scripts = Path(__file__).resolve().parents[3] / "scripts"
    rate = scripts / "replay_models" / "rate"
    offenders, n_calls = [], 0
    package = scripts.parent / "src" / "continuum_swr_replay" / "analysis"
    runners = [
        *rate.glob("fit_*.py"),
        rate / "plot_model_checking.py",
        # the per-session fits the runners call
        package / "replay_models" / "rate" / "per_bin_rate_ladder.py",
        package / "replay_models" / "rate" / "swr_contrast_rate.py",
        package / "replay_models" / "rate" / "uncertainty_rate.py",
        package / "replay_models" / "rate" / "cross_history.py",
    ]
    for path in sorted(runners):
        tree = ast.parse(path.read_text())
        for function in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
            literals = {
                target.id: {
                    key.value
                    for key in node.value.keys
                    if isinstance(key, ast.Constant)
                }
                for node in ast.walk(function)
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict)
                for target in node.targets
                if isinstance(target, ast.Name)
            }
            for node in ast.walk(function):
                if not (
                    isinstance(node, ast.Call)
                    and getattr(node.func, "id", None) == "aggregate_to_timebins"
                ):
                    continue
                n_calls += 1
                given = {k.arg for k in node.keywords if k.arg is not None}
                for spread in (k for k in node.keywords if k.arg is None):
                    if isinstance(spread.value, ast.Name):
                        given |= literals.get(spread.value.id, set())
                if not wanted <= given or "excluded_intervals" in given:
                    offenders.append(f"{path.name}:{node.lineno}")
    assert n_calls >= 5
    assert offenders == []


class TestPeriEventWindows:
    """A triggered window that touches the mask is dropped whole."""

    def _windows(self):
        from continuum_swr_replay.analysis.event_properties.triggered_analysis import (
            get_signal_windows,
        )

        data = _dot().assign(signal=1.0)
        events = pd.DataFrame(
            {
                "animal": "m",
                "day": 20200101,
                "epoch": "02_r1",
                # windows of +-0.5 s: [0.5, 1.5] clear, [1.4, 2.4] touches [2, 5],
                # [5.4, 6.4] clear, [15.3, 16.3] reaches [15.85, 16.05]
                "start_time": [1.0, 1.9, 5.9, 15.8],
                "event_number": [1, 2, 3, 4],
            }
        )
        return get_signal_windows(
            data, events, ["signal"], window_sec=1.0, event_id_col="event_number"
        )

    def test_a_window_touching_the_mask_is_dropped(self, masked):
        windows = self._windows()

        numbers = windows.indexes["event"].get_level_values("event_number")
        assert numbers.tolist() == [1, 3]

    def test_with_no_artifact_run_every_window_is_kept(self):
        windows = self._windows()

        assert windows.sizes["event"] == 4


class TestNoArtifactsChangeNothing:
    """With no artifact run configured, and with a run that has no cores for
    the epoch, every output of the chain is the same, value for value: the
    source's frame, the classified event table, the stop table, the onset and
    dense timebin tables, and Family B's design."""

    @staticmethod
    def _outputs(tmp_path) -> dict[str, pd.DataFrame]:
        from continuum_swr_replay.analysis.event_definition.classified_table import (
            build_classified_nonlocal_event_table,
        )
        from continuum_swr_replay.analysis.event_tables.timebin_tables import (
            build_event_onset_table,
        )

        dot = _dot()
        labels = np.zeros(TIME.size, dtype=np.int64)
        for number, (start, end) in enumerate(
            [(1.0, 1.1), (1.95, 2.05), (5.2, 5.3), (8.0, 8.1), (16.1, 16.2)], 1
        ):
            labels[(TIME >= start) & (TIME <= end)] = number
        dot["non_local_event"] = labels
        dot["MUA_event"] = np.select(
            [(TIME >= 4.0) & (TIME <= 4.05), (TIME >= 6.0) & (TIME <= 6.05)], [1, 2], 0
        )
        dot["Non-Local_Continuous_prob"] = np.where(labels > 0, 0.9, 0.0)
        read = analyzed_time.with_swr_source(dot)
        trials = _trials()
        experiences = port_experiences_from_trials(trials, prepend_initial=True)
        stops = build_stop_table(read, trials, experiences, _graph())
        onsets = build_event_onset_table(read, np.array([1.0]))
        return {
            "frame": read,
            "events": build_classified_nonlocal_event_table(read),
            "stops": stops,
            "onsets": onsets,
            "design": aggregate_to_timebins(
                stops,
                onsets[onsets.event_type == "non_local"].dropna(subset=["stop_id"]),
                bin_s=0.5,
                tmax=20.0,
                excluded_intervals=analyzed_time.excluded_intervals_by_session(stops),
                history_window_s=1.0,
            ),
        }

    def test_every_output_is_the_same(self, tmp_path, monkeypatch):
        from tests.support._swr_inventory_fixtures import (
            use_swr_source,
            write_swr_inventory,
        )

        inventory = write_swr_inventory(
            tmp_path, {EPOCH_ID: [(1.0, 1.05), (3.0, 3.05), (7.0, 7.05)]}
        )
        use_swr_source(monkeypatch, inventory)
        unconfigured = self._outputs(tmp_path)
        empty = write_artifact_run(tmp_path, {EPOCH_ID: []})
        use_swr_source(monkeypatch, inventory, empty)

        configured = self._outputs(tmp_path)

        assert len(unconfigured["events"]) == 5
        for name, frame in unconfigured.items():
            pd.testing.assert_frame_equal(configured[name], frame, check_exact=True)

    def test_an_event_the_mask_reaches_leaves_every_cohort(self, tmp_path, monkeypatch):
        """Non-local 2 overlaps [2, 5], non-local 3 starts 0.2 s after it and
        non-local 5 0.05 s after [15.85, 16.05]: the decoder's reach. SWR 2 and
        MUA 1 overlap [2, 5]; MUA 2, a second after it, stays (a detector's event
        has no reach)."""
        from tests.support._swr_inventory_fixtures import (
            use_swr_source,
            write_swr_inventory,
        )

        inventory = write_swr_inventory(
            tmp_path, {EPOCH_ID: [(1.0, 1.05), (3.0, 3.05), (7.0, 7.05)]}
        )
        use_swr_source(monkeypatch, inventory)
        before = self._outputs(tmp_path)
        artifacts = write_artifact_run(tmp_path, {EPOCH_ID: CORES})
        use_swr_source(monkeypatch, inventory, artifacts)

        after = self._outputs(tmp_path)

        assert before["events"]["non_local_event"].tolist() == [1, 2, 3, 4, 5]
        assert after["events"]["non_local_event"].tolist() == [1, 4]
        # the kept events are the same events, with the same event_id
        kept = before["events"].set_index("non_local_event").loc[[1, 4]]
        np.testing.assert_array_equal(
            after["events"]["event_id"], kept["event_id"].to_numpy()
        )
        assert sorted(set(after["frame"]["SWR_event"]) - {0}) == [1, 3]
        assert sorted(set(after["frame"]["MUA_event"]) - {0}) == [2]
        # non-local 1 and 4, SWR 1 and 3 and MUA 2 all lie in the first stop
        assert after["stops"]["n_selected_events"].tolist() == [2, 0]
        assert before["stops"]["n_selected_events"].tolist() == [4, 1]
        assert after["stops"]["n_swr_events"].tolist() == [2, 0]
        assert before["stops"]["n_swr_events"].tolist() == [3, 0]
        assert after["stops"]["n_mua_events"].tolist() == [1, 0]
        assert before["stops"]["n_mua_events"].tolist() == [2, 0]
        counts = after["onsets"].groupby("event_type").size().to_dict()
        assert counts == {"MUA": 1, "SWR": 2, "non_local": 2}
        assert after["design"]["count"].sum() == 2


def test_an_empty_mask_needs_no_stop_bounds():
    """A runner passes the source's mask whatever it holds; with no rows it
    masks nothing, so a stop table without ``stop_start`` still builds."""
    stops, onsets = _design_inputs()
    stops = stops.drop(columns=["stop_start", "stop_end"])

    design = aggregate_to_timebins(
        stops,
        onsets,
        bin_s=0.5,
        tmax=20.0,
        excluded_intervals=_mask_by_session().iloc[0:0],
        history_window_s=1.0,
        arrival_clocks=False,
    )

    pd.testing.assert_frame_equal(
        design,
        aggregate_to_timebins(
            stops,
            onsets,
            bin_s=0.5,
            tmax=20.0,
            excluded_intervals=NO_EXCLUDED_TIME,
            arrival_clocks=False,
        ),
    )
    with pytest.raises(ValueError, match="stop_start"):
        aggregate_to_timebins(
            stops,
            onsets,
            bin_s=0.5,
            tmax=20.0,
            excluded_intervals=_mask_by_session(),
            arrival_clocks=False,
        )


def test_every_per_bin_runner_checks_its_tables_source():
    """A runner drops bins by the configured source's excluded time, so it
    refuses stop and onset tables stamped with another source (another artifact
    run, say), and its provenance names the source."""
    from pathlib import Path

    rate = Path(__file__).resolve().parents[3] / "scripts" / "replay_models" / "rate"
    runners = [
        "fit_per_bin_rate_models.py",
        "fit_swr_contrast_rate.py",
        "fit_uncertainty_rate.py",
        "fit_cross_history_rate.py",
        "plot_model_checking.py",
    ]
    for name in runners:
        text = (rate / name).read_text()
        assert "require_swr_source(stop," in text, name
        assert "require_swr_source(onset_table," in text, name
    for name in runners[:3]:
        assert '"swr_source": swr_source_name()' in (rate / name).read_text(), name


# --------------------------------------------------------------------------
# Static: every script checks the source of the SWR tables it reads
# --------------------------------------------------------------------------

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
# The content contrast scripts read their tables through this module's
# ``assemble_arm`` / ``assemble_patch``, so the checks are made there.
CONTRAST_ASSEMBLY = (
    SCRIPTS.parent
    / "src"
    / "continuum_swr_replay"
    / "analysis"
    / "replay_models"
    / "contrast_aggregation.py"
)

#: The SWR-derived columns a fit or analysis reads from a stored table.
SWR_COLUMNS = re.compile(
    r"\b(is_swr|is_swr_stop|n_swr_events|n_nonlocal_swr_events"
    r"|n_nonlocal_non_swr_events|swr_x_\w+)\b"
)


def _supported_scripts() -> list[Path]:
    """Every script but the unsupported one-offs."""
    skipped = SCRIPTS / "exploratory"
    return sorted(
        path for path in SCRIPTS.rglob("*.py") if not path.is_relative_to(skipped)
    )


def _fit_runners() -> list[Path]:
    """The rate and content runners: ``replay_models/rate/fit_*``,
    ``replay_models/content/*fit*`` and ``replay_models/content/analyze_*content*``."""
    return sorted(
        {
            *SCRIPTS.glob("replay_models/rate/fit_*.py"),
            *SCRIPTS.glob("replay_models/content/*fit*.py"),
            *SCRIPTS.glob("replay_models/content/analyze_*content*.py"),
        }
    )


def _called(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _calls_by_function(tree: ast.Module) -> dict[ast.AST, list[ast.Call]]:
    """Every call, keyed by the function (or module) it is made in."""
    calls: dict[ast.AST, list[ast.Call]] = {}

    def visit(node: ast.AST, scope: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                visit(child, child)
                continue
            if isinstance(child, ast.Call):
                calls.setdefault(scope, []).append(child)
            visit(child, scope)

    visit(tree, tree)
    return calls


def _swr_tables_read_unchecked(path: Path) -> list[str]:
    """Each frame a script reads as an SWR-source table without checking its source.

    A frame is read as one where ``require_covariate_schema`` names a table of
    ``tables.io.SWR_SOURCE_TABLES``; its source is checked where
    ``require_swr_source`` is called on the same expression in the same function.
    """
    unchecked = []
    for calls in _calls_by_function(ast.parse(path.read_text())).values():
        read = {
            ast.unparse(call.args[0])
            for call in calls
            if _called(call) == "require_covariate_schema"
            and call.args
            and {
                node.value
                for keyword in call.keywords
                if keyword.arg == "table"
                for node in ast.walk(keyword.value)
                if isinstance(node, ast.Constant) and isinstance(node.value, str)
            }
            & SWR_SOURCE_TABLES
        }
        checked = {
            ast.unparse(call.args[0])
            for call in calls
            if _called(call) == "require_swr_source" and call.args
        }
        unchecked += sorted(read - checked)
    return unchecked


def test_every_script_checks_the_source_of_the_swr_tables_it_reads():
    """A table built from another SWR source (the stamp the regeneration has not
    replaced yet, another artifact run) is refused before anything is fit on it."""
    unchecked = {
        str(path.relative_to(SCRIPTS)): names
        for path in _supported_scripts()
        if (names := _swr_tables_read_unchecked(path))
    }

    assert unchecked == {}


def test_every_fit_runner_reading_swr_columns_checks_its_tables_source():
    """Family A, Family B and the content ladders: a runner that names an
    SWR-derived column reads it from a schema-checked SWR table, whose source the
    test above requires it to check."""
    runners = _fit_runners()
    assert {path.name for path in runners} >= {
        "fit_per_stop_rate_models.py",
        "fit_per_bin_rate_models.py",
        "fit_replay_content_models.py",
        "fit_swr_contrast_content.py",
        "analyze_uncertainty_content.py",
    }
    for path in runners:
        text = path.read_text()
        if not SWR_COLUMNS.search(text):
            continue
        readers = [path, *([CONTRAST_ASSEMBLY] if "assemble_arm(" in text else [])]
        read_text = "".join(reader.read_text() for reader in readers)
        assert "require_covariate_schema(" in read_text, path.name
        assert "require_swr_source(" in read_text, path.name
        for reader in readers:
            assert _swr_tables_read_unchecked(reader) == [], (path.name, reader.name)
