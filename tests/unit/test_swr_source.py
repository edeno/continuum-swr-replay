"""The one SWR source: the direct ripple_detection 2.0 Kay inventory.

What these pin: which inventory is read (one configured run id), which epochs
it serves, which events count (the canonical whole-event immobility rule and
the artifact hook, every drop with its reason), and that the per-bin labels are
exactly the convention ``SWR_event`` already had, so the builders downstream
see the same kind of label from a different detector run.

Offline: inventories are written under ``tmp_path`` by
``tests/unit/_swr_inventory_fixtures.py``; nothing reads Spyglass or
``Processed-Data``.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.hippocampus import swr_source
from continuum_swr_replay.analysis.hippocampus.event_immobility import (
    EventDefinitionError,
    build_speed_eligible_event_cohort,
    filter_event_labels_by_speed,
    intervals_from_labels,
)
from continuum_swr_replay.analysis.hippocampus.swr_source import (
    DropReason,
    SwrSourceError,
    read_dot_with_swr_source,
    read_swr_inventory,
    swr_event_labels,
    swr_events,
    swr_observed_mask,
    with_swr_source,
)
from tests.unit._swr_inventory_fixtures import RUN_ID, write_swr_inventory

REPO = Path(__file__).resolve().parents[2]
EPOCH = "j1620210706_02_r1_HPC"
TIME = np.round(100.0 + np.arange(5000) / 500.0, 6)
"""A 500 Hz decode grid, 100.000-109.998 s."""


def _speed() -> np.ndarray:
    speed = np.zeros(TIME.size)
    speed[(TIME >= 102.0) & (TIME <= 102.5)] = 10.0  # moving
    speed[(TIME >= 103.0) & (TIME <= 103.1)] = np.nan  # untracked
    return speed


EVENTS = [
    (100.500, 100.560),  # 1 kept
    (101.000, 101.020),  # 2 kept; ends on a grid sample
    (102.400, 102.600),  # 3 moving
    (103.050, 103.080),  # 4 speed unknown
    (104.000, 104.030),  # 5 inside an artifact
    (105.000, 105.040),  # 6 touches an artifact at 105.040
    (106.0011, 106.0019),  # 7 between two grid samples: none in it
    (107.000, 107.050),  # 8 kept
    (111.000, 111.050),  # 9 past the grid
]
ARTIFACTS = pd.DataFrame(
    {"start_time": [104.010, 105.040], "end_time": [104.020, 105.200]}
)


@pytest.fixture
def inventory(tmp_path):
    return write_swr_inventory(tmp_path / "swr", {EPOCH: EVENTS})


class TestTheConfiguredInventory:
    def test_the_run_id_and_detector_are_the_r3_kay_inventory(self):
        assert swr_source.SWR_INVENTORY_RUN_ID == "50a98592acee"
        assert swr_source.SWR_DETECTOR == "kay"
        assert swr_source.swr_inventory_dir() == (
            swr_source.SWR_INVENTORY_ROOT / swr_source.SWR_INVENTORY_RUN_ID
        )
        assert swr_source.swr_source_name() == "kay@50a98592acee"

    def test_the_folder_follows_the_constant_when_it_changes(self, monkeypatch):
        monkeypatch.setattr(swr_source, "SWR_INVENTORY_RUN_ID", "rerun_masked")

        assert swr_source.swr_inventory_dir().name == "rerun_masked"
        assert swr_source.swr_source_name() == "kay@rerun_masked"

    def test_the_run_id_is_written_in_one_module(self):
        # A literal anywhere else would survive the switch to R3's rerun.
        holders = sorted(
            str(path.relative_to(REPO))
            for folder in ("src", "scripts")
            for path in (REPO / folder).rglob("*.py")
            if swr_source.SWR_INVENTORY_RUN_ID in path.read_text()
        )

        assert holders == [
            "src/continuum_swr_replay/analysis/hippocampus/swr_source.py"
        ]

    def test_the_swr_runner_writes_where_the_source_reads(self):
        import scripts.decoding.run_swr_detection as runner

        assert runner.OUTPUT_ROOT == swr_source.SWR_INVENTORY_ROOT

    def test_the_default_folder_is_read_when_called(self, inventory, monkeypatch):
        monkeypatch.setattr(swr_source, "SWR_INVENTORY_ROOT", inventory.parent)
        monkeypatch.setattr(swr_source, "SWR_INVENTORY_RUN_ID", RUN_ID)

        events = read_swr_inventory(EPOCH)

        assert list(events["SWR_event"]) == list(range(1, len(EVENTS) + 1))

    def test_the_epoch_id_is_the_hpc_one(self):
        assert swr_source.swr_epoch_id("j1620210706_.nwb", "02_r1") == EPOCH


class TestWhichEpochsAreServed:
    @pytest.mark.parametrize(
        "record",
        [
            {"status": "failed", "error": "ValueError('Found  2 potential parents')"},
            {"status": "skipped", "reason": "no LFPBandV1 ripple band"},
            {
                "status": "partial",
                "detectors": {"kay": "ValueError('boom')", "yu": "ok"},
            },
        ],
        ids=["failed", "skipped", "partial-without-kay"],
    )
    def test_an_epoch_without_a_kay_inventory_raises(self, tmp_path, record):
        directory = write_swr_inventory(
            tmp_path / "swr", {}, records={"j1620210710_02_r1_HPC": record}
        )

        with pytest.raises(SwrSourceError, match="j1620210710_02_r1_HPC"):
            read_swr_inventory("j1620210710_02_r1_HPC", inventory_dir=directory)

    def test_a_partial_epoch_with_its_kay_inventory_is_served(self, tmp_path):
        # The Kay-input fetch failing (j16's DB inconsistency) leaves the
        # inventories intact; only the version check is missing.
        directory = write_swr_inventory(tmp_path / "swr", {EPOCH: EVENTS[:2]})
        manifest = directory / "manifest.json"
        manifest.write_text(manifest.read_text().replace('"ok"', '"partial"', 1))

        assert len(read_swr_inventory(EPOCH, inventory_dir=directory)) == 2

    def test_an_epoch_the_run_never_saw_raises(self, inventory):
        with pytest.raises(SwrSourceError, match="not in the SWR inventory"):
            read_swr_inventory("chimi20200212_02_r1_HPC", inventory_dir=inventory)

    def test_a_file_changed_after_the_run_raises(self, inventory):
        swr_source_events = inventory / f"{EPOCH}_kay_events.parquet"
        pd.read_parquet(swr_source_events).iloc[:2].to_parquet(swr_source_events)

        with pytest.raises(SwrSourceError, match="sha256"):
            read_swr_inventory(EPOCH, inventory_dir=inventory)

    def test_a_folder_holding_another_runs_manifest_raises(self, inventory, tmp_path):
        moved = inventory.rename(tmp_path / "swr" / "another_run")

        with pytest.raises(SwrSourceError, match="run 'testrun0001'"):
            read_swr_inventory(EPOCH, inventory_dir=moved)


class TestWhichEventsCount:
    def test_every_event_is_returned_and_each_drop_has_its_reason(self, inventory):
        events = swr_events(
            EPOCH,
            speed=pd.Series(_speed(), index=TIME),
            artifact_intervals=ARTIFACTS,
            inventory_dir=inventory,
        )

        assert list(events["SWR_event"]) == list(range(1, 10))
        assert list(events["drop_reason"]) == [
            None,
            None,
            DropReason.MOVING,
            DropReason.SPEED_UNKNOWN,
            DropReason.ARTIFACT,
            DropReason.ARTIFACT,
            DropReason.NO_SPEED_SAMPLE,
            None,
            DropReason.NO_SPEED_SAMPLE,
        ]
        np.testing.assert_allclose(
            events[["start_time", "end_time"]].to_numpy(), np.array(EVENTS)
        )

    def test_without_artifacts_only_the_immobility_rule_drops(self, inventory):
        events = swr_events(
            EPOCH, speed=pd.Series(_speed(), index=TIME), inventory_dir=inventory
        )

        assert events["drop_reason"].isna().tolist() == [
            True,
            True,
            False,
            False,
            True,
            True,
            False,
            True,
            False,
        ]

    def test_an_artifact_outranks_the_speed_rule(self, inventory):
        artifacts = pd.DataFrame({"start_time": [102.45], "end_time": [102.46]})

        events = swr_events(
            EPOCH,
            speed=pd.Series(_speed(), index=TIME),
            artifact_intervals=artifacts,
            inventory_dir=inventory,
        )

        assert events["drop_reason"].iloc[2] == DropReason.ARTIFACT

    def test_a_malformed_artifact_table_is_refused(self, inventory):
        with pytest.raises(KeyError, match="end_time"):
            swr_events(
                EPOCH,
                speed=pd.Series(_speed(), index=TIME),
                artifact_intervals=pd.DataFrame({"start_time": [1.0]}),
                inventory_dir=inventory,
            )


class TestTheLabels:
    def test_they_are_the_builders_labels_on_the_same_intervals(self, inventory):
        """``get_data_over_time`` writes ``SWR_event`` with this call; fed the same
        intervals, the source writes the same labels, bin for bin."""
        intervals = pd.DataFrame(EVENTS, columns=["start_time", "end_time"])
        _, builder = build_speed_eligible_event_cohort(
            intervals, speed=_speed(), time=pd.Index(TIME), event_col="SWR_event"
        )

        labels = swr_event_labels(TIME, EPOCH, speed=_speed(), inventory_dir=inventory)

        np.testing.assert_array_equal(labels, builder.to_numpy())

    def test_the_builders_intervals_follow(self, inventory):
        """The event table's SWR intervals (gate, then sampled bounds) agree."""
        intervals = pd.DataFrame(EVENTS, columns=["start_time", "end_time"])
        _, builder = build_speed_eligible_event_cohort(
            intervals, speed=_speed(), time=pd.Index(TIME), event_col="SWR_event"
        )
        labels = swr_event_labels(TIME, EPOCH, speed=_speed(), inventory_dir=inventory)

        def table_intervals(values):
            dot = pd.DataFrame(
                {
                    "SWR_event": values,
                    "head_speed": _speed(),
                    "animal": "j16",
                    "day": 20210706,
                    "epoch": "02_r1",
                },
                index=TIME,
            )
            return intervals_from_labels(
                filter_event_labels_by_speed(dot, ["SWR_event"]), "SWR_event"
            )

        pd.testing.assert_frame_equal(
            table_intervals(labels), table_intervals(builder.to_numpy())
        )

    def test_the_convention_by_hand(self, inventory):
        """Closed intervals, the inventory's own numbers, gaps for drops."""
        labels = swr_event_labels(TIME, EPOCH, speed=_speed(), inventory_dir=inventory)

        in_two = (TIME >= 101.0) & (TIME <= 101.02)
        assert in_two.sum() == 11  # both endpoints are grid samples
        assert (labels[in_two] == 2).all()
        assert sorted(set(labels) - {0}) == [1, 2, 5, 6, 8]
        assert (labels[(TIME >= 102.4) & (TIME <= 102.6)] == 0).all()

    def test_an_artifact_zeroes_its_event_and_nothing_else(self, inventory):
        plain = swr_event_labels(TIME, EPOCH, speed=_speed(), inventory_dir=inventory)
        masked = swr_event_labels(
            TIME,
            EPOCH,
            speed=_speed(),
            artifact_intervals=pd.DataFrame(
                {"start_time": [100.55], "end_time": [100.55]}
            ),
            inventory_dir=inventory,
        )

        np.testing.assert_array_equal(masked, np.where(plain == 1, 0, plain))

    def test_events_sharing_a_bin_are_refused_as_the_builders_refuse_them(
        self, tmp_path
    ):
        directory = write_swr_inventory(
            tmp_path / "swr", {EPOCH: [(100.500, 100.530), (100.520, 100.540)]}
        )

        with pytest.raises(EventDefinitionError, match="same bin"):
            swr_event_labels(
                TIME, EPOCH, speed=np.zeros(TIME.size), inventory_dir=directory
            )


class TestCoverage:
    def test_the_mask_is_the_stored_bands_closed_stretches(self, tmp_path):
        directory = write_swr_inventory(
            tmp_path / "swr",
            {EPOCH: EVENTS[:1]},
            observed={EPOCH: [(100.0, 101.0), (102.0, 103.0)]},
        )

        observed = swr_observed_mask(TIME, EPOCH, inventory_dir=directory)

        expected = ((TIME >= 100.0) & (TIME <= 101.0)) | (
            (TIME >= 102.0) & (TIME <= 103.0)
        )
        np.testing.assert_array_equal(observed, expected)


def _dot(stored_labels=None, **columns) -> pd.DataFrame:
    stored = np.zeros(TIME.size, dtype=int) if stored_labels is None else stored_labels
    return pd.DataFrame(
        {
            "SWR_event": stored,
            "head_speed": _speed(),
            "n_SWR": 99,
            "n_non_local_SWR": 7,
            "animal": "j16",
            "day": 20210706,
            "epoch": "02_r1",
            **columns,
        },
        index=pd.Index(TIME, name="time"),
    )


class TestAnAnalysisFrame:
    def test_its_labels_are_the_sources_not_the_decodes(self, inventory):
        stale = np.zeros(TIME.size, dtype=int)
        stale[(TIME >= 108.0) & (TIME <= 108.1)] = 1  # a stored ripple
        dot = _dot(stale)

        out = with_swr_source(dot, inventory_dir=inventory)

        np.testing.assert_array_equal(
            out["SWR_event"].to_numpy(),
            swr_event_labels(TIME, EPOCH, speed=_speed(), inventory_dir=inventory),
        )
        assert out["swr_observed"].all()
        assert not {"n_SWR", "n_non_local_SWR"} & set(out.columns)
        # the decode's frame is left as it was
        np.testing.assert_array_equal(dot["SWR_event"].to_numpy(), stale)
        assert dot["n_SWR"].eq(99).all()

    def test_a_time_column_is_the_axis_when_there_is_one(self, inventory):
        dot = _dot().reset_index()

        out = with_swr_source(dot, inventory_dir=inventory)

        np.testing.assert_array_equal(
            out["SWR_event"].to_numpy(),
            swr_event_labels(TIME, EPOCH, speed=_speed(), inventory_dir=inventory),
        )

    def test_one_epoch_at_a_time(self, inventory):
        dot = pd.concat([_dot(), _dot(epoch="04_r2")])

        with pytest.raises(ValueError, match="one epoch at a time"):
            with_swr_source(dot, inventory_dir=inventory)

    def test_a_decode_pickle_is_read_through_the_source(self, inventory, tmp_path):
        stale = np.ones(TIME.size, dtype=int)
        path = tmp_path / f"{EPOCH}_80_data_over_time.pkl"
        _dot(stale).to_pickle(path)

        out = read_dot_with_swr_source(path, inventory_dir=inventory)

        assert sorted(set(out["SWR_event"]) - {0}) == [1, 2, 5, 6, 8]

    def test_an_epoch_the_source_cannot_serve_stops_the_read(self, tmp_path):
        directory = write_swr_inventory(
            tmp_path / "swr",
            {},
            records={EPOCH: {"status": "failed", "error": "boom"}},
        )

        with pytest.raises(SwrSourceError, match="'failed'"):
            with_swr_source(_dot(), inventory_dir=directory)


def test_the_module_names_no_stored_ripple_source():
    text = (
        REPO / "src/continuum_swr_replay/analysis/hippocampus/swr_source.py"
    ).read_text()
    code = re.sub(r'""".*?"""', "", text, flags=re.S)

    assert "RippleTimesV1" not in code
    assert "ripple_times" not in code
