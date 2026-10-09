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
from continuum_swr_replay.analysis.hippocampus.artifacts.io import (
    cores_path,
    read_manifest,
    write_manifest,
)
from continuum_swr_replay.analysis.hippocampus.event_immobility import (
    EventDefinitionError,
    build_speed_eligible_event_cohort,
    filter_event_labels_by_speed,
    intervals_from_labels,
)
from continuum_swr_replay.analysis.hippocampus.swr_source import (
    DropReason,
    SwrSourceError,
    artifact_intervals,
    read_dot_with_swr_source,
    read_swr_inventory,
    require_swr_source,
    stamp_swr_source,
    swr_event_labels,
    swr_events,
    swr_intervals,
    swr_observed_mask,
    swr_observed_stretches,
    with_swr_source,
)
from continuum_swr_replay.table_io import sha256_file
from tests.unit._swr_inventory_fixtures import (
    ARTIFACT_RUN_ID,
    RUN_ID,
    use_swr_source,
    write_artifact_run,
    write_swr_inventory,
)

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
ARTIFACTS = [("spk", 104.11, 104.11), ("spk", 105.14, 105.14)]
"""Two spike-only cores: padded by 0.1 s (the analysis table), they cover
[104.01, 104.21] and [105.04, 105.24]."""
ARTIFACT_INTERVALS = [(104.01, 104.21), (105.04, 105.24)]


@pytest.fixture
def inventory(tmp_path, monkeypatch):
    """The source's configured inventory is a synthetic run folder."""
    directory = write_swr_inventory(tmp_path / "swr", {EPOCH: EVENTS})
    use_swr_source(monkeypatch, directory)
    return directory


def _artifacts(tmp_path, monkeypatch, cores, span=(0.0, 1e6)) -> None:
    """Configure an artifact run whose cores for EPOCH are `cores`
    (``(signatures, core_start, core_end)``)."""
    directory = write_artifact_run(tmp_path / "artifacts", {EPOCH: cores}, span=span)
    monkeypatch.setattr(swr_source, "ARTIFACT_INTERVALS_ROOT", directory.parent)
    monkeypatch.setattr(swr_source, "ARTIFACT_INTERVALS_RUN_ID", directory.name)


class TestTheConfiguredInventory:
    def test_the_run_id_and_detector_are_the_r3_kay_inventory(self):
        assert swr_source.SWR_INVENTORY_RUN_ID == "50a98592acee"
        assert swr_source.SWR_DETECTOR == "kay"
        assert swr_source.swr_inventory_dir() == (
            swr_source.SWR_INVENTORY_ROOT / swr_source.SWR_INVENTORY_RUN_ID
        )
        assert swr_source.ARTIFACT_INTERVALS_RUN_ID is None
        assert swr_source.artifact_intervals_dir() is None
        assert swr_source.swr_source_name() == "kay@50a98592acee+artifacts@none"

    def test_the_folders_and_name_follow_the_constants_when_they_change(
        self, monkeypatch
    ):
        monkeypatch.setattr(swr_source, "SWR_INVENTORY_RUN_ID", "rerun_masked")
        monkeypatch.setattr(swr_source, "ARTIFACT_INTERVALS_RUN_ID", "task16")

        assert swr_source.swr_inventory_dir().name == "rerun_masked"
        assert swr_source.artifact_intervals_dir() == (
            swr_source.ARTIFACT_INTERVALS_ROOT / "task16"
        )
        assert swr_source.swr_source_name() == "kay@rerun_masked+artifacts@task16"

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

    def test_the_configured_folder_is_read_when_called(self, inventory):
        events = read_swr_inventory(EPOCH)

        assert list(events["SWR_event"]) == list(range(1, len(EVENTS) + 1))

    def test_no_function_takes_a_location_or_artifacts_from_its_caller(self):
        import inspect

        for function in (
            swr_events,
            swr_event_labels,
            with_swr_source,
            read_dot_with_swr_source,
            read_swr_inventory,
            swr_observed_mask,
        ):
            parameters = set(inspect.signature(function).parameters)
            assert not {"artifact_intervals", "inventory_dir"} & parameters

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
    def test_an_epoch_without_a_kay_inventory_raises(
        self, tmp_path, monkeypatch, record
    ):
        directory = write_swr_inventory(
            tmp_path / "swr", {}, records={"j1620210710_02_r1_HPC": record}
        )
        use_swr_source(monkeypatch, directory)

        with pytest.raises(SwrSourceError, match="j1620210710_02_r1_HPC"):
            read_swr_inventory("j1620210710_02_r1_HPC")

    def test_a_partial_epoch_with_its_kay_inventory_is_served(
        self, tmp_path, monkeypatch
    ):
        # The Kay-input fetch failing (j16's DB inconsistency) leaves the
        # inventories intact; only the version check is missing.
        directory = write_swr_inventory(tmp_path / "swr", {EPOCH: EVENTS[:2]})
        manifest = directory / "manifest.json"
        manifest.write_text(manifest.read_text().replace('"ok"', '"partial"', 1))
        use_swr_source(monkeypatch, directory)

        assert len(read_swr_inventory(EPOCH)) == 2

    def test_an_epoch_the_run_never_saw_raises(self, inventory):
        with pytest.raises(SwrSourceError, match="not in the SWR inventory"):
            read_swr_inventory("chimi20200212_02_r1_HPC")

    def test_a_file_changed_after_the_run_raises(self, inventory):
        swr_source_events = inventory / f"{EPOCH}_kay_events.parquet"
        pd.read_parquet(swr_source_events).iloc[:2].to_parquet(swr_source_events)

        with pytest.raises(SwrSourceError, match="sha256"):
            read_swr_inventory(EPOCH)

    def test_a_folder_holding_another_runs_manifest_raises(
        self, inventory, tmp_path, monkeypatch
    ):
        moved = inventory.rename(tmp_path / "swr" / "another_run")
        use_swr_source(monkeypatch, moved)

        with pytest.raises(SwrSourceError, match="run 'testrun0001'"):
            read_swr_inventory(EPOCH)


class TestWhichEventsCount:
    def test_every_event_is_returned_and_each_drop_has_its_reason(
        self, inventory, tmp_path, monkeypatch
    ):
        _artifacts(tmp_path, monkeypatch, ARTIFACTS)

        events = swr_events(EPOCH, speed=pd.Series(_speed(), index=TIME))

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
        events = swr_events(EPOCH, speed=pd.Series(_speed(), index=TIME))

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

    def test_an_artifact_outranks_the_speed_rule(
        self, inventory, tmp_path, monkeypatch
    ):
        _artifacts(tmp_path, monkeypatch, [("spk", 102.55, 102.55)])

        events = swr_events(EPOCH, speed=pd.Series(_speed(), index=TIME))

        assert events["drop_reason"].iloc[2] == DropReason.ARTIFACT


class TestTheArtifactIntervals:
    def test_unset_means_none_and_the_name_says_so(self, inventory):
        intervals = artifact_intervals(EPOCH)

        assert list(intervals.columns) == ["start_time", "end_time"]
        assert intervals.empty
        assert swr_source.swr_source_name().endswith("+artifacts@none")

    def test_a_configured_run_is_read_by_epoch_and_named(
        self, inventory, tmp_path, monkeypatch
    ):
        _artifacts(tmp_path, monkeypatch, ARTIFACTS)

        np.testing.assert_allclose(
            artifact_intervals(EPOCH).to_numpy(), np.array(ARTIFACT_INTERVALS)
        )
        assert swr_source.swr_source_name() == (
            f"kay@{RUN_ID}+artifacts@{ARTIFACT_RUN_ID}"
        )

    def test_the_analysis_table_pads_each_kind(self, inventory, tmp_path, monkeypatch):
        """S by 1 s on each side; an LFP step by 0.3 s before and 1.3 s after
        (the post-step second); a spike-only line by 0.1 s; merged and clipped
        to the epoch's span."""
        _artifacts(
            tmp_path,
            monkeypatch,
            [
                ("S", 100.0, 102.0),
                ("ca1+ref", 104.5, 104.5),
                ("spk", 107.5, 107.5),
                ("ref", 109.5, 109.5),
            ],
            span=(99.5, 110.0),
        )

        np.testing.assert_allclose(
            artifact_intervals(EPOCH).to_numpy(),
            [(99.5, 103.0), (104.2, 105.8), (107.4, 107.6), (109.2, 110.0)],
        )

    def test_each_consumer_pads_the_same_cores_from_its_own_row(
        self, inventory, tmp_path, monkeypatch
    ):
        """The run's one set of cores gives every consumer's mask: the default
        is the analysis one, and the swr and decoder ones are asked for by
        name."""
        _artifacts(
            tmp_path,
            monkeypatch,
            [("S", 100.0, 102.0), ("ca1+ref", 104.5, 104.5), ("spk", 107.5, 107.5)],
            span=(99.5, 110.0),
        )

        by_consumer = {
            consumer: artifact_intervals(EPOCH, consumer).to_numpy()
            for consumer in ("swr", "decoder", "analysis")
        }

        np.testing.assert_allclose(
            by_consumer["swr"], [(100.0, 102.0), (104.2, 104.8), (107.4, 107.6)]
        )
        np.testing.assert_allclose(
            by_consumer["decoder"], [(99.5, 103.0), (104.2, 104.8), (107.4, 107.6)]
        )
        np.testing.assert_allclose(
            by_consumer["analysis"], [(99.5, 103.0), (104.2, 105.8), (107.4, 107.6)]
        )
        np.testing.assert_array_equal(
            artifact_intervals(EPOCH).to_numpy(), by_consumer["analysis"]
        )

    def test_a_configured_run_without_the_epochs_file_is_refused(
        self, inventory, tmp_path, monkeypatch
    ):
        # A missing file must not read as an epoch without artifacts.
        _artifacts(tmp_path, monkeypatch, ARTIFACTS)
        cores_path(swr_source.artifact_intervals_dir(), EPOCH).unlink()

        with pytest.raises(SwrSourceError, match="no artifact intervals"):
            swr_events(EPOCH, speed=pd.Series(_speed(), index=TIME))

    def test_an_epoch_the_artifact_run_never_saw_is_refused(
        self, inventory, tmp_path, monkeypatch
    ):
        directory = write_artifact_run(
            tmp_path / "artifacts", {"chimi20200225_02_r1_HPC": ARTIFACTS}
        )
        use_swr_source(monkeypatch, inventory, directory)

        with pytest.raises(SwrSourceError, match="not in the artifact run"):
            artifact_intervals(EPOCH)

    def test_an_artifact_file_changed_after_the_run_is_refused(
        self, inventory, tmp_path, monkeypatch
    ):
        _artifacts(tmp_path, monkeypatch, ARTIFACTS)
        path = cores_path(swr_source.artifact_intervals_dir(), EPOCH)
        pd.read_parquet(path).iloc[:1].to_parquet(path)

        with pytest.raises(SwrSourceError, match="sha256"):
            swr_events(EPOCH, speed=pd.Series(_speed(), index=TIME))

    def test_a_malformed_artifact_file_is_refused(
        self, inventory, tmp_path, monkeypatch
    ):
        _artifacts(tmp_path, monkeypatch, ARTIFACTS)
        directory = swr_source.artifact_intervals_dir()
        path = cores_path(directory, EPOCH)
        pd.DataFrame({"start_time": [1.0], "end_time": [2.0]}).to_parquet(path)
        manifest = read_manifest(directory)
        manifest["epochs"][EPOCH]["outputs"]["cores"]["sha256"] = sha256_file(path)
        write_manifest(directory, manifest)

        with pytest.raises(SwrSourceError, match="core_start"):
            swr_events(EPOCH, speed=pd.Series(_speed(), index=TIME))

    def test_every_entry_point_drops_the_same_artifact_events(
        self, inventory, tmp_path, monkeypatch
    ):
        _artifacts(tmp_path, monkeypatch, [("spk", 100.65, 100.65)])
        dot = _dot()
        path = tmp_path / f"{EPOCH}_80_data_over_time.pkl"
        dot.to_pickle(path)

        events = swr_events(EPOCH, speed=pd.Series(_speed(), index=TIME))
        labels = swr_event_labels(TIME, EPOCH, speed=_speed())

        assert events.loc[
            events["drop_reason"] == DropReason.ARTIFACT, "SWR_event"
        ].tolist() == [1]
        assert 1 not in set(labels)
        np.testing.assert_array_equal(with_swr_source(dot)["SWR_event"], labels)
        np.testing.assert_array_equal(
            read_dot_with_swr_source(path)["SWR_event"], labels
        )


class TestTheLabels:
    def test_they_are_the_builders_labels_on_the_same_intervals(self, inventory):
        """``get_data_over_time`` writes ``SWR_event`` with this call; fed the same
        intervals, the source writes the same labels, bin for bin."""
        intervals = pd.DataFrame(EVENTS, columns=["start_time", "end_time"])
        _, builder = build_speed_eligible_event_cohort(
            intervals, speed=_speed(), time=pd.Index(TIME), event_col="SWR_event"
        )

        labels = swr_event_labels(TIME, EPOCH, speed=_speed())

        np.testing.assert_array_equal(labels, builder.to_numpy())

    def test_the_builders_intervals_follow(self, inventory):
        """The event table's SWR intervals (gate, then sampled bounds) agree."""
        intervals = pd.DataFrame(EVENTS, columns=["start_time", "end_time"])
        _, builder = build_speed_eligible_event_cohort(
            intervals, speed=_speed(), time=pd.Index(TIME), event_col="SWR_event"
        )
        labels = swr_event_labels(TIME, EPOCH, speed=_speed())

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
        labels = swr_event_labels(TIME, EPOCH, speed=_speed())

        in_two = (TIME >= 101.0) & (TIME <= 101.02)
        assert in_two.sum() == 11  # both endpoints are grid samples
        assert (labels[in_two] == 2).all()
        assert sorted(set(labels) - {0}) == [1, 2, 5, 6, 8]
        assert (labels[(TIME >= 102.4) & (TIME <= 102.6)] == 0).all()

    def test_an_artifact_zeroes_its_event_and_nothing_else(
        self, inventory, tmp_path, monkeypatch
    ):
        plain = swr_event_labels(TIME, EPOCH, speed=_speed())
        _artifacts(tmp_path, monkeypatch, [("spk", 100.65, 100.65)])
        masked = swr_event_labels(TIME, EPOCH, speed=_speed())

        np.testing.assert_array_equal(masked, np.where(plain == 1, 0, plain))

    def test_events_sharing_a_bin_are_refused_as_the_builders_refuse_them(
        self, tmp_path, monkeypatch
    ):
        directory = write_swr_inventory(
            tmp_path / "swr", {EPOCH: [(100.500, 100.530), (100.520, 100.540)]}
        )
        use_swr_source(monkeypatch, directory)

        with pytest.raises(EventDefinitionError, match="same bin"):
            swr_event_labels(TIME, EPOCH, speed=np.zeros(TIME.size))


class TestCoverage:
    def test_the_mask_is_the_stored_bands_closed_stretches(self, tmp_path, monkeypatch):
        directory = write_swr_inventory(
            tmp_path / "swr",
            {EPOCH: EVENTS[:1]},
            observed={EPOCH: [(100.0, 101.0), (102.0, 103.0)]},
        )
        use_swr_source(monkeypatch, directory)

        observed = swr_observed_mask(TIME, EPOCH)

        expected = ((TIME >= 100.0) & (TIME <= 101.0)) | (
            (TIME >= 102.0) & (TIME <= 103.0)
        )
        np.testing.assert_array_equal(observed, expected)

    def test_artifact_time_is_not_observed(self, tmp_path, monkeypatch):
        """Artifact time is unobserved: an event there is neither SWR nor
        non-SWR. The analysis intervals leave the coverage, endpoints
        included, and the stretches say the same as the mask."""
        directory = write_swr_inventory(
            tmp_path / "swr",
            {EPOCH: EVENTS[:1]},
            observed={EPOCH: [(100.0, 101.0), (102.0, 109.0)]},
        )
        use_swr_source(monkeypatch, directory)
        _artifacts(tmp_path, monkeypatch, ARTIFACTS)

        observed = swr_observed_mask(TIME, EPOCH)
        stretches = swr_observed_stretches(EPOCH)

        in_band = ((TIME >= 100.0) & (TIME <= 101.0)) | (
            (TIME >= 102.0) & (TIME <= 109.0)
        )
        in_artifact = np.zeros(TIME.size, dtype=bool)
        for start, end in artifact_intervals(EPOCH).to_numpy():
            in_artifact |= (TIME >= start) & (TIME <= end)
        assert in_artifact.sum() == 2 * 101  # both endpoints are grid samples
        np.testing.assert_array_equal(observed, in_band & ~in_artifact)
        from_stretches = np.zeros(TIME.size, dtype=bool)
        for start, end in stretches:
            from_stretches |= (TIME >= start) & (TIME <= end)
        np.testing.assert_array_equal(from_stretches, observed)
        assert stretches.shape == (4, 2)

    def test_an_analysis_frame_is_unobserved_in_artifact_time(
        self, inventory, tmp_path, monkeypatch
    ):
        _artifacts(tmp_path, monkeypatch, ARTIFACTS)

        out = with_swr_source(_dot())

        np.testing.assert_array_equal(
            out["swr_observed"].to_numpy(), swr_observed_mask(TIME, EPOCH)
        )
        assert not out["swr_observed"].all()


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

        out = with_swr_source(dot)

        np.testing.assert_array_equal(
            out["SWR_event"].to_numpy(),
            swr_event_labels(TIME, EPOCH, speed=_speed()),
        )
        assert out["swr_observed"].all()
        assert not {"n_SWR", "n_non_local_SWR"} & set(out.columns)
        # the decode's frame is left as it was
        np.testing.assert_array_equal(dot["SWR_event"].to_numpy(), stale)
        assert dot["n_SWR"].eq(99).all()

    def test_a_time_column_is_the_axis_when_there_is_one(self, inventory):
        dot = _dot().reset_index()

        out = with_swr_source(dot)

        np.testing.assert_array_equal(
            out["SWR_event"].to_numpy(),
            swr_event_labels(TIME, EPOCH, speed=_speed()),
        )

    def test_one_epoch_at_a_time(self, inventory):
        dot = pd.concat([_dot(), _dot(epoch="04_r2")])

        with pytest.raises(ValueError, match="one epoch at a time"):
            with_swr_source(dot)

    def test_a_decode_pickle_is_read_through_the_source(self, inventory, tmp_path):
        stale = np.ones(TIME.size, dtype=int)
        path = tmp_path / f"{EPOCH}_80_data_over_time.pkl"
        _dot(stale).to_pickle(path)

        out = read_dot_with_swr_source(path)

        assert sorted(set(out["SWR_event"]) - {0}) == [1, 2, 5, 6, 8]

    def test_an_epoch_the_source_cannot_serve_stops_the_read(
        self, tmp_path, monkeypatch
    ):
        directory = write_swr_inventory(
            tmp_path / "swr",
            {},
            records={EPOCH: {"status": "failed", "error": "boom"}},
        )
        use_swr_source(monkeypatch, directory)

        with pytest.raises(SwrSourceError, match="'failed'"):
            with_swr_source(_dot())

    def test_the_frame_is_stamped_and_the_manifest_checked_once(
        self, inventory, monkeypatch
    ):
        calls = []
        original = swr_source.swr_source_files

        def counted(epoch_id):
            calls.append(epoch_id)
            return original(epoch_id)

        monkeypatch.setattr(swr_source, "swr_source_files", counted)

        out = with_swr_source(_dot())

        assert calls == [EPOCH]
        assert out.attrs["swr_source"] == swr_source.swr_source_name()

    def test_its_intervals_are_the_labels_sampled_bounds(self, inventory):
        out = with_swr_source(_dot())
        pooled = pd.concat([out, out.assign(epoch="04_r2")])

        intervals = swr_intervals(
            pooled, nwb_file_name="j1620210706_.nwb", epoch_name="02_r1"
        )

        pd.testing.assert_frame_equal(
            intervals, intervals_from_labels(out, "SWR_event")
        )
        assert intervals["SWR_event"].tolist() == [1, 2, 5, 6, 8]

    def test_intervals_need_a_frame_read_through_the_source(self, inventory):
        with pytest.raises(ValueError, match="swr_observed"):
            swr_intervals(_dot())


class TestTheStamp:
    def test_a_matching_stamp_passes(self, inventory):
        require_swr_source(stamp_swr_source(pd.DataFrame()), what="table")

    @pytest.mark.parametrize(
        "stamp",
        [None, "kay@an_earlier_run+artifacts@none", "kay@50a98592acee"],
        ids=["unstamped", "another-run", "retired-format"],
    )
    def test_another_stamp_is_refused(self, inventory, stamp):
        frame = pd.DataFrame()
        if stamp is not None:
            frame.attrs["swr_source"] = stamp

        with pytest.raises(SwrSourceError, match="regenerate"):
            require_swr_source(frame, what="the event table")

    def test_a_new_artifact_run_retires_every_stamp(self, inventory, monkeypatch):
        frame = stamp_swr_source(pd.DataFrame())
        monkeypatch.setattr(swr_source, "ARTIFACT_INTERVALS_RUN_ID", "task16")

        with pytest.raises(SwrSourceError):
            require_swr_source(frame, what="the event table")


def test_the_module_names_no_stored_ripple_source():
    text = (
        REPO / "src/continuum_swr_replay/analysis/hippocampus/swr_source.py"
    ).read_text()
    code = re.sub(r'""".*?"""', "", text, flags=re.S)

    assert "RippleTimesV1" not in code
    assert "ripple_times" not in code
