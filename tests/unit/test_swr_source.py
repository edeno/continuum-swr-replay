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

from continuum_swr_replay.analysis.events import analysed_time, swr_source
from continuum_swr_replay.analysis.events.analysed_time import (
    read_dot_with_swr_source,
    require_swr_source,
    stamp_swr_source,
    swr_event_labels,
    swr_intervals,
    swr_observed_mask,
    swr_observed_stretches,
    with_swr_source,
)
from continuum_swr_replay.analysis.events.artifact_intervals import (
    cores_path,
    read_manifest,
    write_manifest,
)
from continuum_swr_replay.analysis.events.eligibility import (
    DropReason,
    excluded_time,
)
from continuum_swr_replay.analysis.events.immobility import (
    EventDefinitionError,
    build_speed_eligible_event_cohort,
    filter_event_labels_by_speed,
    intervals_from_labels,
)
from continuum_swr_replay.analysis.events.swr_source import (
    SwrSourceError,
    artifact_intervals,
    read_swr_inventory,
    swr_events,
)
from continuum_swr_replay.runs.io import sha256_file
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
"""Two spike-only cores: padded by 0.1 s (``PADDING_S``), they cover
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
    monkeypatch.setattr(swr_source, "ARTIFACT_INTERVALS_DIR", directory.parent)
    monkeypatch.setattr(swr_source, "ARTIFACT_INTERVALS_RUN_ID", directory.name)


CONFIGURED_ARTIFACT_RUN_ID = swr_source.ARTIFACT_INTERVALS_RUN_ID
"""The committed value, read at import: the suite's autouse fixture sets the
module's to None for every test (tests/conftest.py)."""


class TestTheConfiguredInventory:
    def test_the_run_id_and_detector_are_the_r3_kay_inventory(self, monkeypatch):
        assert swr_source.SWR_INVENTORY_RUN_ID == "056eee0dcb5a"
        assert swr_source.SWR_DETECTOR == "kay"
        assert swr_source.swr_inventory_dir() == (
            swr_source.SWR_INVENTORY_ROOT / swr_source.SWR_INVENTORY_RUN_ID
        )
        assert CONFIGURED_ARTIFACT_RUN_ID == "fbbff144a123"
        monkeypatch.setattr(
            swr_source, "ARTIFACT_INTERVALS_RUN_ID", CONFIGURED_ARTIFACT_RUN_ID
        )
        assert swr_source.artifact_intervals_dir() == (
            swr_source.ARTIFACT_INTERVALS_DIR / "fbbff144a123"
        )
        assert swr_source.swr_source_name() == (
            "kay@056eee0dcb5a+artifacts@fbbff144a123"
        )

    def test_the_folders_and_name_follow_the_constants_when_they_change(
        self, monkeypatch
    ):
        monkeypatch.setattr(swr_source, "SWR_INVENTORY_RUN_ID", "rerun_masked")
        monkeypatch.setattr(swr_source, "ARTIFACT_INTERVALS_RUN_ID", "task16")

        assert swr_source.swr_inventory_dir().name == "rerun_masked"
        assert swr_source.artifact_intervals_dir() == (
            swr_source.ARTIFACT_INTERVALS_DIR / "task16"
        )
        assert swr_source.swr_source_name() == "kay@rerun_masked+artifacts@task16"

    def test_a_source_name_says_which_artifact_run_it_was_built_under(
        self, monkeypatch
    ):
        monkeypatch.setattr(swr_source, "ARTIFACT_INTERVALS_RUN_ID", "task16")
        assert swr_source.artifact_run_of(swr_source.swr_source_name()) == "task16"
        monkeypatch.setattr(swr_source, "ARTIFACT_INTERVALS_RUN_ID", None)
        assert swr_source.artifact_run_of(swr_source.swr_source_name()) is None
        # a name from before the artifact run was recorded in it: no run
        assert swr_source.artifact_run_of("Kay") is None

    def test_the_run_id_is_written_in_one_module(self):
        # A literal anywhere else would survive the switch to R3's rerun.
        holders = sorted(
            str(path.relative_to(REPO))
            for folder in ("src", "scripts")
            for path in (REPO / folder).rglob("*.py")
            if swr_source.SWR_INVENTORY_RUN_ID in path.read_text()
            or CONFIGURED_ARTIFACT_RUN_ID in path.read_text()
        )

        assert holders == ["src/continuum_swr_replay/analysis/events/swr_source.py"]

    def test_the_swr_runner_writes_where_the_source_reads(self):
        import scripts.detection.run_swr_detection as runner

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

    def test_the_one_table_pads_each_kind(self, inventory, tmp_path, monkeypatch):
        """S by 1 s on each side, an LFP step by 0.3 s on each side (no
        post-step second), a spike-only line by 0.1 s; merged and clipped to
        the epoch's span."""
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
            [(99.5, 103.0), (104.2, 104.8), (107.4, 107.6), (109.2, 109.8)],
        )

    def test_there_is_one_mask_and_no_consumer_to_choose(self):
        import inspect

        assert list(inspect.signature(artifact_intervals).parameters) == ["epoch_id"]

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

    def test_excluded_time_zeroes_its_events_and_nothing_else(
        self, inventory, tmp_path, monkeypatch
    ):
        """The mask [100.55, 100.75] overlaps SWR 1; SWR 2 (101.000-101.020)
        lies within its 0.3 s reach, in excluded time [100.25, 101.05]."""
        plain = swr_event_labels(TIME, EPOCH, speed=_speed())
        _artifacts(tmp_path, monkeypatch, [("spk", 100.65, 100.65)])
        masked = swr_event_labels(TIME, EPOCH, speed=_speed())

        np.testing.assert_array_equal(
            masked, np.where(np.isin(plain, [1, 2]), 0, plain)
        )

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

    def test_excluded_time_is_not_observed(self, tmp_path, monkeypatch):
        """Excluded time (the mask widened by 0.3 s) is not analysed: an event
        there is neither SWR nor non-SWR. Its intervals leave the coverage,
        endpoints included, and the stretches say the same as the mask."""
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
        in_excluded = np.zeros(TIME.size, dtype=bool)
        for start, end in analysed_time.excluded_intervals(EPOCH).to_numpy():
            in_excluded |= (TIME >= start) & (TIME <= end)
        # 0.8 s each; the padded bounds fall a hair off the grid
        assert in_excluded.sum() == 2 * 399
        np.testing.assert_array_equal(observed, in_band & ~in_excluded)
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

    def test_a_time_column_is_refused(self, inventory):
        """The index is the one time axis; a ``time`` column would be a second."""
        with pytest.raises(ValueError, match="'time' column"):
            with_swr_source(_dot().reset_index())

    def test_a_time_index_that_does_not_increase_is_refused(self, inventory):
        """Readers take an event's first and last row as its bounds."""
        with pytest.raises(ValueError, match="not strictly increasing"):
            with_swr_source(_dot().iloc[::-1])

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


def test_a_patch_on_swr_inventory_reaches_analysed_time(monkeypatch):
    """``analysed_time`` calls the source's functions through ``swr_source``,
    so one patch there reaches the stamp, its check and excluded time."""
    monkeypatch.setattr(swr_source, "swr_source_name", lambda: "patched@source")
    frame = stamp_swr_source(pd.DataFrame())
    assert frame.attrs["swr_source"] == "patched@source"
    require_swr_source(frame, what="the patched table")

    cores = pd.DataFrame({"start_time": [10.0], "end_time": [12.0]})
    monkeypatch.setattr(swr_source, "artifact_intervals", lambda epoch_id: cores)
    pd.testing.assert_frame_equal(
        analysed_time.excluded_intervals(EPOCH), excluded_time(cores)
    )


@pytest.mark.parametrize("module", ["swr_source", "analysed_time"])
def test_the_module_names_no_stored_ripple_source(module):
    path = REPO / f"src/continuum_swr_replay/analysis/events/{module}.py"
    code = re.sub(r'""".*?"""', "", path.read_text(), flags=re.S)

    assert "RippleTimesV1" not in code
    assert "ripple_times" not in code


def _labels(intervals: list[tuple[float, float]]) -> np.ndarray:
    labels = np.zeros(TIME.size, dtype=np.int64)
    for number, (start, end) in enumerate(intervals, start=1):
        labels[(TIME >= start) & (TIME <= end)] = number
    return labels


EXCLUDED_INTERVALS = [(103.71, 104.51), (104.74, 105.54)]
"""ARTIFACT_INTERVALS widened by 0.3 s on each side."""
NON_LOCAL = [
    (100.70, 100.80),  # 1 kept
    (103.80, 103.90),  # 2 near the mask: in excluded time [103.71, 104.51]
    (104.10, 104.15),  # 3 in the mask
    (105.60, 105.70),  # 4 starts 0.06 s after excluded time ends: kept
    (105.40, 105.50),  # 5 near the mask
    (103.40, 103.60),  # 6 outside excluded time, but overlaps MUA 4
]
MUA = [
    (103.80, 103.90),  # 1 near the mask: every cohort has the one reach
    (105.20, 105.30),  # 2 in the mask
    (108.00, 108.05),  # 3 kept
    (103.50, 103.75),  # 4 runs into excluded time: dropped, near the mask
]


class TestEveryCohortHonoursTheMask:
    """The source serves each analysis frame with every event cohort held to the
    one analysed-time rule: an event in excluded time (the mask widened by
    ``ARTIFACT_REACH_S``) is dropped whatever its kind, and a kept event that
    overlaps an SWR or MUA event dropped for excluded time is dropped too. A
    dropped event keeps its reason, and the others keep their numbers (so
    their ``event_id``)."""

    def _nonlocal_dot(self) -> pd.DataFrame:
        return _dot(
            non_local_event=_labels(NON_LOCAL),
            MUA_event=_labels(MUA),
            **{"Non-Local_Continuous_prob": 0.9, "Non-Local_Fragmented_prob": 0.0},
        )

    def _contfrag_dot(self) -> pd.DataFrame:
        return _dot(MUA_event=_labels(MUA), Continuous_prob=0.5, Fragmented_prob=0.5)

    def test_excluded_time_is_the_mask_widened_by_the_reach(
        self, inventory, tmp_path, monkeypatch
    ):
        _artifacts(tmp_path, monkeypatch, ARTIFACTS)

        np.testing.assert_allclose(
            analysed_time.excluded_intervals(EPOCH).to_numpy(), EXCLUDED_INTERVALS
        )
        np.testing.assert_allclose(
            analysed_time.excluded_intervals_of(_dot()).to_numpy(), EXCLUDED_INTERVALS
        )
        sessions = pd.DataFrame(
            {"animal": ["j16"], "session_date": [20210706], "epoch": ["02_r1"]}
        )
        by_session = analysed_time.excluded_intervals_by_session(sessions)
        assert list(by_session.columns) == [
            "animal",
            "session_date",
            "epoch",
            "start_time",
            "end_time",
        ]
        np.testing.assert_allclose(
            by_session[["start_time", "end_time"]].to_numpy(), EXCLUDED_INTERVALS
        )

    def test_every_cohort_leaves_excluded_time(self, inventory, tmp_path, monkeypatch):
        _artifacts(tmp_path, monkeypatch, ARTIFACTS)
        dot = self._nonlocal_dot()

        out = with_swr_source(dot)

        assert sorted(set(out["non_local_event"]) - {0}) == [1, 4]
        assert sorted(set(out["MUA_event"]) - {0}) == [3]
        assert sorted(set(out["SWR_event"]) - {0}) == [1, 2, 8]
        kept = np.isin(dot["non_local_event"], [1, 4])
        np.testing.assert_array_equal(
            out["non_local_event"].to_numpy()[kept],
            dot["non_local_event"].to_numpy()[kept],
        )
        # the decode's frame is left as it was
        assert sorted(set(dot["non_local_event"]) - {0}) == [1, 2, 3, 4, 5, 6]

    def test_each_dropped_event_keeps_its_reason(
        self, inventory, tmp_path, monkeypatch
    ):
        _artifacts(tmp_path, monkeypatch, ARTIFACTS)

        drops = analysed_time.artifact_event_drops(self._nonlocal_dot())

        def _reasons(column):
            rows = drops.loc[drops["event_col"] == column]
            return dict(zip(rows["label"], rows["drop_reason"], strict=True))

        assert _reasons("non_local_event") == {
            1: None,
            2: DropReason.NEAR_ARTIFACT,
            3: DropReason.ARTIFACT,
            4: None,
            5: DropReason.NEAR_ARTIFACT,
            6: DropReason.OVERLAPS_DROPPED_EVENT,
        }
        assert _reasons("MUA_event") == {
            1: DropReason.NEAR_ARTIFACT,
            2: DropReason.ARTIFACT,
            3: None,
            4: DropReason.NEAR_ARTIFACT,
        }
        swrs = _reasons("SWR_event")
        assert swrs[5] == swrs[6] == DropReason.ARTIFACT
        assert swrs[3] == DropReason.MOVING
        assert list(drops.columns) == analysed_time.DROP_COLUMNS
        dropped = analysed_time.dropped_for_excluded_time(drops)
        assert sorted(zip(dropped["event_col"], dropped["label"], strict=True)) == [
            ("MUA_event", 1),
            ("MUA_event", 2),
            ("MUA_event", 4),
            ("SWR_event", 5),
            ("SWR_event", 6),
        ]

    def test_the_drops_carry_each_events_sampled_bounds(
        self, inventory, tmp_path, monkeypatch
    ):
        """An SWR's sampled bounds are its first and last grid sample (NaN for
        one the speed rule rejects); a labelled event's are its bounds. The
        second pass reads the dropped events at them (``dropped_on_grid``)."""
        _artifacts(tmp_path, monkeypatch, ARTIFACTS)
        dot = self._nonlocal_dot()
        time = dot.index.to_numpy()

        drops = analysed_time.artifact_event_drops(dot)

        swr = drops["event_col"] == "SWR_event"
        rows = drops.loc[swr]
        first = np.searchsorted(time, rows["start_time"].to_numpy(), side="left")
        last = np.searchsorted(time, rows["end_time"].to_numpy(), side="right") - 1
        eligible = rows["speed_eligible"].to_numpy()
        assert not eligible.all()  # SWR 3 moves
        on_axis = np.c_[time[first.clip(0, time.size - 1)], time[last.clip(0)]]
        expected = np.where(eligible[:, None], on_axis, np.nan)
        np.testing.assert_array_equal(
            rows[["sampled_start_time", "sampled_end_time"]].to_numpy(), expected
        )
        np.testing.assert_array_equal(
            drops.loc[~swr, ["sampled_start_time", "sampled_end_time"]].to_numpy(),
            drops.loc[~swr, ["start_time", "end_time"]].to_numpy(),
        )
        on_grid = analysed_time.dropped_on_grid(drops)
        dropped = analysed_time.dropped_for_excluded_time(drops)
        assert on_grid.index.equals(dropped.index)
        np.testing.assert_array_equal(
            on_grid[["start_time", "end_time"]].to_numpy(),
            dropped[["sampled_start_time", "sampled_end_time"]].to_numpy(),
        )

    def test_a_frame_without_an_overlap_keeps_its_events(
        self, inventory, tmp_path, monkeypatch
    ):
        """Only an event dropped for excluded time drops the events that overlap
        it, and only once: without MUA 4, non-local 6 is kept."""
        _artifacts(tmp_path, monkeypatch, ARTIFACTS)
        dot = self._nonlocal_dot().assign(MUA_event=_labels(MUA[:3]))

        out = with_swr_source(dot)

        assert sorted(set(out["non_local_event"]) - {0}) == [1, 4, 6]

    def test_nonlocal_and_contfrag_frames_give_the_same_swr_and_mua_cohorts(
        self, inventory, tmp_path, monkeypatch
    ):
        _artifacts(tmp_path, monkeypatch, ARTIFACTS)

        nonlocal_frame = with_swr_source(self._nonlocal_dot())
        contfrag = with_swr_source(self._contfrag_dot())

        for column in ("SWR_event", "MUA_event"):
            np.testing.assert_array_equal(nonlocal_frame[column], contfrag[column])
        np.testing.assert_array_equal(
            nonlocal_frame["swr_observed"], contfrag["swr_observed"]
        )

    def test_the_decoders_keep_every_decoded_event(
        self, inventory, tmp_path, monkeypatch
    ):
        """A decode saves what it decoded; the read path drops, with reasons."""
        _artifacts(tmp_path, monkeypatch, ARTIFACTS)
        dot = self._nonlocal_dot()

        out = with_swr_source(dot, drop_masked_events=False)

        np.testing.assert_array_equal(out["non_local_event"], dot["non_local_event"])
        np.testing.assert_array_equal(out["MUA_event"], dot["MUA_event"])

    def test_with_no_artifact_run_every_label_is_the_decodes(self, inventory):
        dot = self._nonlocal_dot()

        out = with_swr_source(dot)

        for column in ("non_local_event", "MUA_event"):
            pd.testing.assert_series_equal(out[column], dot[column])
        reasons = analysed_time.artifact_event_drops(dot)["drop_reason"].dropna()
        assert set(reasons) <= {
            DropReason.MOVING,
            DropReason.SPEED_UNKNOWN,
            DropReason.NO_SPEED_SAMPLE,
        }


class TestTheLabelContract:
    """The read checks the decode's own labels once, for every consumer: each
    event is immobile as a whole, and each non-local event's bins are above the
    probability threshold the frame was decoded at. A frame that breaks either
    is stale and is refused, in both modes and in the drops; it is never
    clipped. An SWR that moves is the source's own, and is dropped (``moving``,
    :class:`TestWhichEventsCount`)."""

    def _frame(self, **columns) -> pd.DataFrame:
        frame = {
            "non_local_event": _labels(NON_LOCAL[:1]),
            "MUA_event": _labels(MUA[2:3]),
            "Non-Local_Continuous_prob": 0.9,
            "Non-Local_Fragmented_prob": 0.0,
            **columns,
        }
        return _dot(**frame)

    @pytest.mark.parametrize(
        ("lose_the_axis", "message"),
        [
            # reset_index(drop=True) / concat(ignore_index=True): row numbers
            (lambda f: f.reset_index(drop=True), "RangeIndex"),
            (lambda f: f.set_axis(np.arange(len(f)) * 2), "dtype int64"),
            (lambda f: f.reset_index(), "'time' column"),
            (lambda f: pd.concat([f.iloc[:10], f.iloc[9:]]), "strictly increasing"),
        ],
        ids=["range_index", "integer_index", "time_column", "repeated_time"],
    )
    def test_a_frame_without_its_time_axis_is_refused(
        self, inventory, lose_the_axis, message
    ):
        """Row numbers read as seconds would match no SWR and no excluded
        time, silently; the read refuses the frame and says what to do."""
        dot = lose_the_axis(self._frame())

        with pytest.raises(ValueError, match=message) as refused:
            with_swr_source(dot)
        assert "time" in str(refused.value)
        with pytest.raises(ValueError, match=message):
            analysed_time.artifact_event_drops(dot)

    def test_a_frame_that_keeps_the_contract_is_read(self, inventory):
        out = with_swr_source(self._frame())

        assert sorted(set(out["non_local_event"]) - {0}) == [1]
        assert sorted(set(out["MUA_event"]) - {0}) == [1]

    @pytest.mark.parametrize("column", ["non_local_event", "MUA_event"])
    @pytest.mark.parametrize("drop_masked_events", [True, False])
    def test_a_label_that_moves_is_refused(self, inventory, column, drop_masked_events):
        moving = _labels([(102.40, 102.60)])  # speed 10 cm/s from 102.0 to 102.5
        dot = self._frame(**{column: moving})

        with pytest.raises(EventDefinitionError, match="immobility contract"):
            with_swr_source(dot, drop_masked_events=drop_masked_events)
        with pytest.raises(EventDefinitionError, match="immobility contract"):
            analysed_time.artifact_event_drops(dot)

    def test_a_label_with_untracked_speed_is_refused(self, inventory):
        dot = self._frame(MUA_event=_labels([(103.05, 103.08)]))  # speed NaN

        with pytest.raises(EventDefinitionError, match="immobility contract"):
            with_swr_source(dot)

    @pytest.mark.parametrize("probability", [0.8, 0.5, np.nan, np.inf])
    def test_a_nonlocal_bin_at_or_below_the_threshold_is_refused(
        self, inventory, probability
    ):
        dot = self._frame()
        bin_in_event = np.flatnonzero(dot["non_local_event"].to_numpy() > 0)[3]
        dot.iloc[bin_in_event, dot.columns.get_loc("Non-Local_Continuous_prob")] = (
            probability
        )

        with pytest.raises(EventDefinitionError, match="probability-threshold"):
            with_swr_source(dot)

    def test_the_threshold_is_the_one_the_frame_was_decoded_at(
        self, inventory, tmp_path
    ):
        dot = self._frame(**{"Non-Local_Continuous_prob": 0.6})
        path = tmp_path / f"{EPOCH}_50_data_over_time.pkl"
        dot.to_pickle(path)

        with pytest.raises(EventDefinitionError, match="probability-threshold"):
            with_swr_source(dot)
        out = with_swr_source(dot, non_local_threshold=0.5)
        assert sorted(set(out["non_local_event"]) - {0}) == [1]
        read = read_dot_with_swr_source(path, non_local_threshold=0.5)
        pd.testing.assert_series_equal(read["non_local_event"], out["non_local_event"])

    def test_the_drops_take_the_frames_threshold_too(self, inventory, tmp_path):
        """A frame decoded at 0.5 reads at 0.5 in the drops as in the read."""
        dot = self._frame(**{"Non-Local_Continuous_prob": 0.6})
        path = tmp_path / f"{EPOCH}_50_data_over_time.pkl"
        dot.to_pickle(path)

        with pytest.raises(EventDefinitionError, match="probability-threshold"):
            analysed_time.artifact_event_drops(dot)
        drops = analysed_time.artifact_event_drops(dot, non_local_threshold=0.5)
        pd.testing.assert_frame_equal(
            analysed_time.read_artifact_event_drops(path, non_local_threshold=0.5),
            drops,
        )
        assert (drops["event_col"] == "non_local_event").sum() == 1

    def test_a_contfrag_frame_has_no_probability_contract(self, inventory):
        dot = _dot(
            MUA_event=_labels(MUA[2:3]), Continuous_prob=0.5, Fragmented_prob=0.5
        )

        out = with_swr_source(dot)

        assert sorted(set(out["MUA_event"]) - {0}) == [1]

    def test_every_label_it_hands_out_is_one_the_speed_rule_admits(self, inventory):
        out = with_swr_source(self._frame())

        for column in ("SWR_event", "non_local_event", "MUA_event"):
            pd.testing.assert_series_equal(
                filter_event_labels_by_speed(out, [column])[column], out[column]
            )


# --------------------------------------------------------------------------
# What the read adds and re-derives besides the labels (task 26a)
# --------------------------------------------------------------------------


def _closed_mask(intervals: pd.DataFrame, time: np.ndarray) -> np.ndarray:
    """By hand: True at each time in a closed ``[start_time, end_time]``."""
    inside = np.zeros(time.shape, dtype=bool)
    for start, end in intervals[["start_time", "end_time"]].to_numpy():
        inside |= (time >= start) & (time <= end)
    return inside


class TestTheExcludedTimeColumn:
    """``in_excluded_time`` marks the rows in the epoch's excluded time (the
    mask widened by 0.3 s), closed, so a reader can leave excluded time out
    without reading the intervals again."""

    def test_it_marks_the_closed_excluded_intervals(
        self, inventory, tmp_path, monkeypatch
    ):
        _artifacts(tmp_path, monkeypatch, ARTIFACTS)
        excluded = analysed_time.excluded_intervals(EPOCH)
        # the grid holds every interval's exact endpoints, so closedness shows
        time = np.unique(np.r_[TIME, excluded.to_numpy().ravel()])
        dot = pd.DataFrame(
            {
                "head_speed": np.zeros(time.size),
                "animal": "j16",
                "day": 20210706,
                "epoch": "02_r1",
            },
            index=pd.Index(time, name="time"),
        )

        out = with_swr_source(dot)

        column = out["in_excluded_time"].to_numpy()
        assert column.dtype == bool
        np.testing.assert_array_equal(column, _closed_mask(excluded, time))
        at_ends = np.isin(time, excluded.to_numpy().ravel())
        assert at_ends.sum() == 4
        assert column[at_ends].all()
        # it is not the coverage column: excluded time and nothing else
        assert column.sum() == 2 * 399 + 4

    def test_it_is_all_false_with_no_artifact_run(self, inventory):
        out = with_swr_source(_dot())

        assert out["in_excluded_time"].dtype == bool
        assert not out["in_excluded_time"].any()

    @pytest.mark.parametrize("drop_masked_events", [True, False])
    def test_both_modes_carry_it(
        self, inventory, tmp_path, monkeypatch, drop_masked_events
    ):
        _artifacts(tmp_path, monkeypatch, ARTIFACTS)

        out = with_swr_source(_dot(), drop_masked_events=drop_masked_events)

        np.testing.assert_array_equal(
            out["in_excluded_time"].to_numpy(),
            _closed_mask(analysed_time.excluded_intervals(EPOCH), TIME),
        )

    def test_a_contfrag_frame_read_from_its_pickle_carries_it(
        self, inventory, tmp_path, monkeypatch
    ):
        """The ContFrag prep reads its frames with ``read_dot_with_swr_source``."""
        _artifacts(tmp_path, monkeypatch, ARTIFACTS)
        path = tmp_path / f"{EPOCH}_contfrag_data_over_time.pkl"
        _dot(Continuous_prob=0.5, Fragmented_prob=0.5).to_pickle(path)

        out = read_dot_with_swr_source(path)

        np.testing.assert_array_equal(
            out["in_excluded_time"].to_numpy(),
            _closed_mask(analysed_time.excluded_intervals(EPOCH), TIME),
        )


class TestTheAnalysedBins:
    """``analysed_bins`` is the one check of the column every per-bin summary
    uses: the column must be there and boolean, with no coercion."""

    def test_it_is_the_rows_outside_excluded_time(self):
        frame = pd.DataFrame({"in_excluded_time": [False, True, True, False]})

        np.testing.assert_array_equal(
            analysed_time.analysed_bins(frame), [True, False, False, True]
        )
        assert "analysed_bins" in analysed_time.__all__

    def test_a_frame_without_the_column_is_refused(self):
        with pytest.raises(
            KeyError, match="in_excluded_time.*read_dot_with_swr_source"
        ):
            analysed_time.analysed_bins(pd.DataFrame({"head_speed": [0.0]}))

    @pytest.mark.parametrize(
        "values",
        [
            [False, True],  # boolean values, object dtype: no coercion
            [False, np.nan],  # a concat with a frame read another way
            [False, None],  # cast to bool, None would read as analysed
        ],
        ids=["object", "nan", "none"],
    )
    def test_a_column_that_is_not_boolean_is_refused(self, values):
        frame = pd.DataFrame({"in_excluded_time": pd.Series(values, dtype=object)})

        with pytest.raises(TypeError, match="in_excluded_time"):
            analysed_time.analysed_bins(frame)


def _raw_theta_power() -> np.ndarray:
    """Theta power with an artifact's large values in the mask and a few
    untracked (NaN) samples outside it."""
    rng = np.random.default_rng(26)
    power = rng.gamma(4.0, 2.0, TIME.size)
    in_mask = _closed_mask(pd.DataFrame(ARTIFACT_INTERVALS, columns=_BOUNDS), TIME)
    power[in_mask] *= 40.0
    power[(TIME >= 101.5) & (TIME <= 101.52)] = np.nan
    return power


_BOUNDS = ["start_time", "end_time"]


class TestTheNormalizers:
    """The two epoch normalizers are inputs, so they leave the artifact mask
    itself out, as ``SWR_zscore`` and ``MUA_zscore`` already do: the theta-power
    z-score is re-z-scored over the rows outside the mask (NaN inside it) and the
    run-speed baseline is the same percentile of head speed outside the mask."""

    def _theta_dot(self) -> pd.DataFrame:
        from continuum_swr_replay.analysis.hippocampus.data_aggregation import (
            theta_power_zscore,
        )

        # as the decoders stored it: z-scored over every bin, the mask included
        return _dot(theta_power_zscore=theta_power_zscore(_raw_theta_power()))

    def test_theta_has_mean_zero_and_sd_one_outside_the_mask(
        self, inventory, tmp_path, monkeypatch
    ):
        _artifacts(tmp_path, monkeypatch, ARTIFACTS)
        dot = self._theta_dot()
        outside = ~_closed_mask(artifact_intervals(EPOCH), TIME)
        stored = dot["theta_power_zscore"].to_numpy()
        # the decode-time z-score is not normalized outside the mask
        assert np.nanstd(stored[outside]) < 0.5

        theta = with_swr_source(dot)["theta_power_zscore"].to_numpy()

        np.testing.assert_allclose(np.nanmean(theta[outside]), 0.0, atol=1e-12)
        np.testing.assert_allclose(np.nanstd(theta[outside]), 1.0, atol=1e-12)

    def test_theta_is_nan_inside_the_mask(self, inventory, tmp_path, monkeypatch):
        _artifacts(tmp_path, monkeypatch, ARTIFACTS)
        in_mask = _closed_mask(artifact_intervals(EPOCH), TIME)
        assert in_mask.sum() == 2 * 101

        theta = with_swr_source(self._theta_dot())["theta_power_zscore"].to_numpy()

        assert np.isnan(theta[in_mask]).all()
        # the reach of the mask is outside it: finite there
        reach = _closed_mask(analysed_time.excluded_intervals(EPOCH), TIME) & ~in_mask
        assert np.isfinite(theta[reach]).all()

    def test_theta_is_the_raw_power_z_scored_outside_the_mask(
        self, inventory, tmp_path, monkeypatch
    ):
        """z is affine in raw power, so re-z-scoring the stored z over the rows
        outside the mask is z-scoring the raw power over those rows."""
        from continuum_swr_replay.analysis.hippocampus.data_aggregation import (
            theta_power_zscore,
        )

        _artifacts(tmp_path, monkeypatch, ARTIFACTS)
        outside = ~_closed_mask(artifact_intervals(EPOCH), TIME)

        theta = with_swr_source(self._theta_dot())["theta_power_zscore"].to_numpy()

        np.testing.assert_allclose(
            theta[outside],
            theta_power_zscore(_raw_theta_power()[outside]),
            rtol=0,
            atol=1e-12,
        )
        assert np.isnan(theta[outside]).sum() == 11  # the untracked samples

    @pytest.mark.parametrize("first_mode", [True, False], ids=["read", "decoder"])
    def test_theta_is_idempotent(self, inventory, tmp_path, monkeypatch, first_mode):
        """A frame already renormalized (by a read, or by a decoder saving its
        frame through the source) is not changed by the next read."""
        _artifacts(tmp_path, monkeypatch, ARTIFACTS)
        once = with_swr_source(self._theta_dot(), drop_masked_events=first_mode)

        twice = with_swr_source(once)

        np.testing.assert_allclose(
            twice["theta_power_zscore"].to_numpy(),
            once["theta_power_zscore"].to_numpy(),
            rtol=0,
            atol=1e-12,
        )
        np.testing.assert_array_equal(
            np.isnan(twice["theta_power_zscore"].to_numpy()),
            np.isnan(once["theta_power_zscore"].to_numpy()),
        )

    def test_the_speed_baseline_is_the_percentile_outside_the_mask(
        self, inventory, tmp_path, monkeypatch
    ):
        from continuum_swr_replay.analysis.events.epoch_normalizers import (
            RUN_SPEED_BASELINE_PERCENTILE,
        )

        _artifacts(tmp_path, monkeypatch, ARTIFACTS)
        in_mask = _closed_mask(artifact_intervals(EPOCH), TIME)
        speed = _speed() + np.random.default_rng(4).uniform(0.0, 3.9, TIME.size)
        speed[in_mask] = 100.0  # an artifact's tracking jumps
        dot = _dot(head_speed=speed, run_speed_baseline=np.nanpercentile(speed, 90))

        baseline = with_swr_source(dot)["run_speed_baseline"].to_numpy()

        # the decoders' percentile (data_aggregation, both decoders)
        assert RUN_SPEED_BASELINE_PERCENTILE == 90
        expected = np.nanpercentile(speed[~in_mask], 90)
        assert np.unique(baseline).tolist() == [expected]
        assert expected != pytest.approx(np.nanpercentile(speed, 90))

    def test_with_no_artifact_run_both_are_the_decodes(self, inventory):
        dot = self._theta_dot().assign(run_speed_baseline=123.0)

        out = with_swr_source(dot)

        pd.testing.assert_series_equal(
            out["theta_power_zscore"], dot["theta_power_zscore"]
        )
        pd.testing.assert_series_equal(
            out["run_speed_baseline"], dot["run_speed_baseline"]
        )

    def test_the_decodes_frame_is_left_as_it_was(
        self, inventory, tmp_path, monkeypatch
    ):
        _artifacts(tmp_path, monkeypatch, ARTIFACTS)
        dot = self._theta_dot().assign(run_speed_baseline=123.0)
        before = dot.copy(deep=True)

        with_swr_source(dot)

        pd.testing.assert_frame_equal(dot, before)

    def test_a_frame_without_them_is_read(self, inventory, tmp_path, monkeypatch):
        """A rule frame (the half-tetrode QA's) carries neither normalizer."""
        _artifacts(tmp_path, monkeypatch, ARTIFACTS)

        out = with_swr_source(_dot())

        assert not {"theta_power_zscore", "run_speed_baseline"} & set(out.columns)


@pytest.mark.parametrize("drop_masked_events", [True, False])
def test_the_decodes_event_counts_are_dropped(inventory, drop_masked_events):
    """Every decode-time count counts events the one rule drops; none reaches
    an analysis frame."""
    counts = {
        "n_non_local": 9,
        "n_MUA": 8,
        "n_non_local_MUA": 7,
        "n_non_local_no_MUA": 6,
    }
    assert set(counts) == set(analysed_time.DECODE_EVENT_COUNT_COLUMNS)

    out = with_swr_source(_dot(**counts), drop_masked_events=drop_masked_events)

    assert not set(counts) & set(out.columns)
    assert not set(analysed_time.DECODE_SWR_COUNT_COLUMNS) & set(out.columns)


def test_session_epoch_id_is_the_one_built_from_a_file_name():
    """The id is the one the frozen (ba2961a5) fake-file-name route built."""

    def frozen(animal, day, epoch):
        return swr_source.swr_epoch_id(f"{animal}{day}_.nwb", str(epoch))

    for animal, day, epoch in [
        ("chimi", 20200225, "02_r1"),
        ("j16", np.int64(20210706), np.str_("04_r2")),
        ("peanut", "20201125", "10_r5"),
        ("senor", np.int32(20201030), "02_r1"),
    ]:
        new = swr_source.swr_epoch_id_of_session(animal, day, epoch)
        assert new == frozen(animal, day, epoch)
    assert swr_source.swr_epoch_id_of_session("chimi", 20200225, "02_r1") == (
        "chimi20200225_02_r1_HPC"
    )
