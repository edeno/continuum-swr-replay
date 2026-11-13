"""Each detector's recorded outputs for one epoch, on the common schema.

One simulated epoch, written to temporary run folders as the runners write
them: the SWR run (Kay, Karlsson, Yu, Zugaro), a Carey run, a memoryless run,
an artifact run, and the decode frame with its MUA and non-local labels. The
artifact mask covers [104.01, 104.21] and [106.01, 106.21] (two spike-only
cores padded by 0.1 s); excluded time is [103.71, 104.51] and [105.71, 106.51].
The animal moves over 102.0-102.5 s.

Offline: nothing here touches Spyglass or ``Processed-Data``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.detector_comparison.adapters import (
    NATIVE_IS_ELIGIBLE,
    carey_inventory,
    memoryless_inventory,
    mua_inventory,
    state_space_inventory,
    swr_detector_inventory,
)
from continuum_swr_replay.analysis.detector_comparison.availability import (
    OBSERVATION_DTYPES,
)
from continuum_swr_replay.analysis.detector_comparison.overlap import (
    SET_ASIDE_DTYPES,
)
from continuum_swr_replay.analysis.detector_comparison.schema import (
    INVENTORY_DTYPES,
    from_classified_nonlocal_table,
    validate_inventory,
)
from continuum_swr_replay.analysis.event_definition.analysed_time import (
    artifact_event_drops,
    with_swr_source,
)
from continuum_swr_replay.analysis.event_definition.classified_table import (
    build_classified_nonlocal_event_table,
)
from continuum_swr_replay.runs.io import sha256_file
from tests.support._swr_inventory_fixtures import (
    RUN_ID,
    kay_events,
    use_swr_source,
    write_artifact_run,
    write_swr_inventory,
)

EPOCH = "j1620210706_02_r1_HPC"
PREFIX = "j1620210706_02_r1"
TIME = np.round(100.0 + np.arange(5000) / 500.0, 6)
"""A 500 Hz decode grid, 100.000-109.998 s."""
ARTIFACTS = [("spk", 104.11, 104.11), ("spk", 106.11, 106.11)]
STORED_BAND = [(100.0, 103.0), (103.2, 109.998)]

KAY = [
    (100.500, 100.560),  # 1 kept
    (102.400, 102.600),  # 2 moving: kept with its flag
    (103.600, 103.750),  # 3 reaches excluded time: near_artifact
    (104.000, 104.030),  # 4 in the mask: artifact
    (105.450, 105.520),  # 5 overlaps MUA 2, dropped for excluded time
    (107.000, 107.050),  # 6 kept
]
KARLSSON = [
    (100.505, 100.555),  # 1 kept
    (103.550, 103.650),  # 2 outside excluded time, overlaps Kay 3
    (104.100, 104.150),  # 3 artifact
    (107.010, 107.040),  # 4 kept
]
MUA = [
    (100.510, 100.550),  # 1 kept
    (105.500, 105.750),  # 2 reaches excluded time: near_artifact
    (108.000, 108.050),  # 3 kept
]
NON_LOCAL = [
    (100.520, 100.600),  # 1 kept
    (104.300, 104.400),  # 2 near_artifact
    (106.100, 106.150),  # 3 artifact
    (109.000, 109.100),  # 4 kept
]
CAREY = [(100.502, 100.580), (105.600, 105.720), (108.010, 108.040)]


def _labels(intervals) -> np.ndarray:
    labels = np.zeros(TIME.size, dtype=np.int64)
    for number, (start, end) in enumerate(intervals, start=1):
        labels[(TIME >= start) & (TIME <= end)] = number
    return labels


def raw_dot(mua=MUA, non_local=NON_LOCAL) -> pd.DataFrame:
    """The decode's frame, as its pickle holds it."""
    speed = np.zeros(TIME.size)
    speed[(TIME >= 102.0) & (TIME <= 102.5)] = 10.0
    continuous = np.full(TIME.size, 0.9)
    continuous[(TIME >= 108.5) & (TIME < 108.6)] = np.nan  # no posterior
    return pd.DataFrame(
        {
            "animal": "j16",
            "day": 20210706,
            "epoch": "02_r1",
            "head_speed": speed,
            "SWR_event": 0,
            "MUA_event": _labels(mua),
            "non_local_event": _labels(non_local),
            "Non-Local_Continuous_prob": continuous,
            "Non-Local_Fragmented_prob": 0.05,
        },
        index=pd.Index(TIME, name="time"),
    )


def _write_events(path: Path, intervals, **columns) -> str:
    table = kay_events(intervals)
    for name, values in columns.items():
        table[name] = values
    table.to_parquet(path)
    return sha256_file(path)


def add_detector(run_dir: Path, detector: str, intervals) -> None:
    """Add one SWR detector's events to the run folder and its manifest."""
    path = run_dir / f"{EPOCH}_{detector}_events.parquet"
    sha = _write_events(path, intervals)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["epochs"][EPOCH]["outputs"][detector] = {"path": str(path), "sha256": sha}
    manifest_path.write_text(json.dumps(manifest))


def write_carey_run(root: Path, events=CAREY, record=None) -> Path:
    directory = root / "careyrun01"
    directory.mkdir(parents=True)
    epochs = {}
    if record is not None:
        epochs[EPOCH] = record
    elif events is not None:
        events_path = directory / f"{EPOCH}_carey_events.parquet"
        events_sha = _write_events(
            events_path, events, n_active_units=np.full(len(events), 6)
        )
        observed_path = directory / f"{EPOCH}_carey_observed.parquet"
        pd.DataFrame(
            {
                "scope": ["carey", "another"],
                "start_s": [100.001, 0.0],
                "end_s": [109.997, 1e6],
            }
        ).to_parquet(observed_path)
        epochs[EPOCH] = {
            "status": "ok",
            "outputs": {
                "events": {"path": str(events_path), "sha256": events_sha},
                "observed": {
                    "path": str(observed_path),
                    "sha256": sha256_file(observed_path),
                },
            },
        }
    (directory / "manifest.json").write_text(
        json.dumps({"run_id": directory.name, "epochs": epochs})
    )
    return directory


def memoryless_table(swr_definition: str, n_shuffles: int = 100) -> pd.DataFrame:
    """Five candidates: published kept, published moving, published in the
    mask, replay criteria only, and a rejection."""
    starts = [100.502, 102.300, 104.050, 107.100, 108.200]
    ends = [100.598, 102.450, 104.100, 107.200, 108.300]
    return pd.DataFrame(
        {
            "animal": "j16",
            "day": 20210706,
            "epoch": "02_r1",
            "brain_area": "HPC",
            "run_id": "memorylessrun01",
            "ms_event_id": [11, 12, 13, 14, 15],
            "event_start_s": starts,
            "event_end_s": ends,
            "event_duration_s": np.subtract(ends, starts),
            "animal_max_speed_cm_s": [0.0, 10.0, 0.0, 0.0, 0.0],
            "animal_mean_speed_cm_s": [0.0, 5.0, 0.0, 0.0, 0.0],
            "whole_event_speed_eligible": [True, False, True, True, True],
            "is_replay": [True, True, True, True, False],
            "shuffle_pass": [True, True, True, False, False],
            "n_shuffles": [n_shuffles] * 4 + [0],
            "swr_definition": swr_definition,
            "artifact_drop_reason": [None, None, "artifact", None, None],
        }
    )


def write_memoryless_run(root: Path, table: pd.DataFrame) -> Path:
    directory = root / "memorylessrun01"
    directory.mkdir(parents=True)
    events_path = directory / f"{EPOCH}_memoryless_events.parquet"
    table.to_parquet(events_path)
    observed_path = directory / f"{EPOCH}_memoryless_observed.parquet"
    pd.DataFrame(
        {
            "kind": ["decoded_windows", "decoded_windows", "swr_labels"],
            "start_s": [100.04, 104.30, 100.0],
            "end_s": [103.96, 109.95, 109.998],
        }
    ).to_parquet(observed_path)
    record = {
        "status": "ok",
        "time_grid": {
            "start_s": float(TIME[0]),
            "end_s": float(TIME[-1]),
            "n_samples": int(TIME.size),
            "sampling_frequency": 500.0,
        },
        "n_candidates": len(table),
        "n_replay": int(table.is_replay.sum()),
        "n_shuffle_pass": int(table.shuffle_pass.sum()),
        "outputs": {"events": str(events_path), "observed": str(observed_path)},
    }
    (directory / "manifest.json").write_text(
        json.dumps({"run_id": directory.name, "epochs": {EPOCH: record}})
    )
    return directory


def edit_record(run_dir: Path, change) -> None:
    """Apply ``change(epochs)`` to the run folder's manifest epochs."""
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    change(manifest["epochs"])
    manifest_path.write_text(json.dumps(manifest))


def write_epoch(tmp_path, monkeypatch, kay=KAY, karlsson=KARLSSON) -> dict:
    """The SWR and artifact runs configured; the decode frame read once
    through the source, and its drops from the frame as saved."""
    swr_run = write_swr_inventory(tmp_path, {EPOCH: kay}, observed={EPOCH: STORED_BAND})
    add_detector(swr_run, "karlsson", karlsson)
    add_detector(swr_run, "yu", [])
    add_detector(swr_run, "zugaro", [(107.000, 107.030)])
    artifact_run = write_artifact_run(tmp_path, {EPOCH: ARTIFACTS})
    use_swr_source(monkeypatch, swr_run, artifact_run)
    raw = raw_dot()
    return {
        "swr_run": swr_run,
        "raw": raw,
        "dot": with_swr_source(raw),
        "drops": artifact_event_drops(raw),
    }


@pytest.fixture
def epoch(tmp_path, monkeypatch):
    return write_epoch(tmp_path, monkeypatch)


def _reasons(set_aside: pd.DataFrame) -> dict[str, str]:
    return dict(zip(set_aside.event_key, set_aside.reason, strict=True))


def _check_dtypes(tables) -> None:
    for table, dtypes in (
        (tables.inventory, INVENTORY_DTYPES),
        (tables.observations, OBSERVATION_DTYPES),
        (tables.set_aside, SET_ASIDE_DTYPES),
    ):
        assert list(table.columns) == list(dtypes)
        assert table.dtypes.to_dict() == {k: np.dtype(v) for k, v in dtypes.items()}
    validate_inventory(tables.inventory)


class TestTheSwrInventories:
    def test_every_event_is_in_the_inventory_or_set_aside_with_its_reason(self, epoch):
        tables = swr_detector_inventory(
            epoch["swr_run"], "kay", dot=epoch["dot"], drops=epoch["drops"]
        )

        _check_dtypes(tables)
        assert len(tables.inventory) + len(tables.set_aside) == len(KAY)
        assert tables.inventory.event_key.tolist() == [
            f"kay:{PREFIX}:1",
            f"kay:{PREFIX}:2",
            f"kay:{PREFIX}:6",
        ]
        # the moving event stays, with its flag
        assert tables.inventory.whole_event_speed_eligible.tolist() == [
            True,
            False,
            True,
        ]
        assert _reasons(tables.set_aside) == {
            f"kay:{PREFIX}:3": "near_artifact",
            f"kay:{PREFIX}:4": "artifact",
            f"kay:{PREFIX}:5": "overlaps_dropped_event",
        }
        aside = tables.set_aside.set_index("event_key")
        assert aside.loc[f"kay:{PREFIX}:3", "interval_start_s"] == 103.600
        assert aside.loc[f"kay:{PREFIX}:3", "interval_end_s"] == 103.750
        np.testing.assert_allclose(
            tables.inventory.native_duration_s, [0.06, 0.2, 0.05]
        )
        assert set(tables.inventory.run_id) == {RUN_ID}

    def test_kay_keeps_exactly_the_papers_cohort(self, epoch):
        tables = swr_detector_inventory(
            epoch["swr_run"], "kay", dot=epoch["dot"], drops=epoch["drops"]
        )

        kept = tables.inventory[tables.inventory.whole_event_speed_eligible]
        labels = sorted(set(epoch["dot"]["SWR_event"]) - {0})
        assert kept.event_key.tolist() == [f"kay:{PREFIX}:{n}" for n in labels]

    def test_an_event_overlapping_a_dropped_kay_swr_is_set_aside(self, epoch):
        tables = swr_detector_inventory(
            epoch["swr_run"], "karlsson", dot=epoch["dot"], drops=epoch["drops"]
        )

        _check_dtypes(tables)
        assert tables.inventory.event_key.tolist() == [
            f"karlsson:{PREFIX}:1",
            f"karlsson:{PREFIX}:4",
        ]
        assert _reasons(tables.set_aside) == {
            f"karlsson:{PREFIX}:2": "overlaps_dropped_event",
            f"karlsson:{PREFIX}:3": "artifact",
        }

    def test_the_second_pass_reads_bounds_sampled_on_the_grid(
        self, tmp_path, monkeypatch
    ):
        """Kay 5 and a new Karlsson event end 0.5 ms past the first sample of
        dropped MUA 2 (105.500): their last samples are 105.500, so neither
        overlaps it on the grid, and both are kept, as the source keeps Kay 5."""
        late = (105.300, 105.5005)
        kay = [*KAY[:4], late, KAY[5]]
        karlsson = [*KARLSSON[:3], late, KARLSSON[3]]
        epoch = write_epoch(tmp_path, monkeypatch, kay=kay, karlsson=karlsson)

        kay_tables = swr_detector_inventory(
            epoch["swr_run"], "kay", dot=epoch["dot"], drops=epoch["drops"]
        )
        karlsson_tables = swr_detector_inventory(
            epoch["swr_run"], "karlsson", dot=epoch["dot"], drops=epoch["drops"]
        )

        kept = kay_tables.inventory[kay_tables.inventory.whole_event_speed_eligible]
        labels = sorted(set(epoch["dot"]["SWR_event"]) - {0})
        assert labels == [1, 5, 6]
        assert kept.event_key.tolist() == [f"kay:{PREFIX}:{n}" for n in labels]
        assert f"karlsson:{PREFIX}:4" in set(karlsson_tables.inventory.event_key)
        assert set(karlsson_tables.set_aside.reason) == {
            "overlaps_dropped_event",
            "artifact",
        }

    def test_the_observed_time_is_the_detectors_scope(self, epoch):
        kay = swr_detector_inventory(
            epoch["swr_run"], "kay", dot=epoch["dot"], drops=epoch["drops"]
        ).observations
        zugaro = swr_detector_inventory(
            epoch["swr_run"], "zugaro", dot=epoch["dot"], drops=epoch["drops"]
        ).observations

        observed = kay[kay.available]
        np.testing.assert_allclose(
            observed[["start_s", "end_s"]].to_numpy(), STORED_BAND
        )
        assert set(kay.detector) == {"kay"}
        # the zugaro scope covers the whole epoch, cut to the decode grid
        assert zugaro.available.all()
        np.testing.assert_allclose(
            zugaro[["start_s", "end_s"]].to_numpy(), [(TIME[0], TIME[-1])]
        )

    def test_an_empty_inventory_keeps_the_schema(self, epoch):
        tables = swr_detector_inventory(
            epoch["swr_run"], "yu", dot=epoch["dot"], drops=epoch["drops"]
        )

        _check_dtypes(tables)
        assert tables.inventory.empty and tables.set_aside.empty
        assert tables.observations.available.any()

    def test_an_unknown_detector_is_refused(self, epoch):
        with pytest.raises(ValueError, match="detector"):
            swr_detector_inventory(
                epoch["swr_run"], "long", dot=epoch["dot"], drops=epoch["drops"]
            )


def _assert_unavailable(tables, detector: str, reason: str) -> None:
    _check_dtypes(tables)
    assert tables.inventory.empty and tables.set_aside.empty
    rows = tables.observations
    assert len(rows) == 1 and not rows.available.any()
    assert rows.detector.tolist() == [detector]
    assert (rows.start_s.iloc[0], rows.end_s.iloc[0]) == (TIME[0], TIME[-1])
    assert reason in rows.reason.iloc[0]


class TestAnUnusableRecordReadsUnavailable:
    def test_a_changed_file(self, epoch):
        path = epoch["swr_run"] / f"{EPOCH}_karlsson_events.parquet"
        kay_events(KARLSSON[:2]).to_parquet(path)

        tables = swr_detector_inventory(
            epoch["swr_run"], "karlsson", dot=epoch["dot"], drops=epoch["drops"]
        )

        _assert_unavailable(tables, "karlsson", "sha256")

    def test_an_output_the_manifest_records_no_sha256_for(self, epoch):
        """No output entry at all is a mismatch, not a pass."""
        edit_record(
            epoch["swr_run"],
            lambda epochs: epochs[EPOCH]["outputs"].pop("karlsson"),
        )

        tables = swr_detector_inventory(
            epoch["swr_run"], "karlsson", dot=epoch["dot"], drops=epoch["drops"]
        )

        _assert_unavailable(tables, "karlsson", "sha256")

    def test_a_record_without_its_detectors(self, epoch):
        edit_record(epoch["swr_run"], lambda epochs: epochs[EPOCH].pop("detectors"))

        tables = swr_detector_inventory(
            epoch["swr_run"], "zugaro", dot=epoch["dot"], drops=epoch["drops"]
        )

        _assert_unavailable(tables, "zugaro", "None")

    def test_events_not_numbered_one_to_n(self, epoch):
        path = epoch["swr_run"] / f"{EPOCH}_karlsson_events.parquet"
        kay_events(KARLSSON).reset_index(drop=True).to_parquet(path)
        sha = sha256_file(path)

        def rehash(epochs):
            epochs[EPOCH]["outputs"]["karlsson"]["sha256"] = sha

        edit_record(epoch["swr_run"], rehash)

        tables = swr_detector_inventory(
            epoch["swr_run"], "karlsson", dot=epoch["dot"], drops=epoch["drops"]
        )

        _assert_unavailable(tables, "karlsson", "event_number 1..n")

    def test_two_events_sharing_a_decode_grid_sample(self, tmp_path, monkeypatch):
        """Karlsson 4 and 5 touch at 107.020 s, a sample of the grid, so the
        speed rule cannot label both there: Karlsson reads unavailable, and
        Kay, whose events share no sample, is unaffected."""
        karlsson = [*KARLSSON[:3], (107.010, 107.020), (107.020, 107.030)]
        epoch = write_epoch(tmp_path, monkeypatch, karlsson=karlsson)

        tables = swr_detector_inventory(
            epoch["swr_run"], "karlsson", dot=epoch["dot"], drops=epoch["drops"]
        )
        kay = swr_detector_inventory(
            epoch["swr_run"], "kay", dot=epoch["dot"], drops=epoch["drops"]
        )

        _assert_unavailable(tables, "karlsson", "share a decode-grid sample")
        assert len(kay.inventory) == 3

    def test_a_missing_file(self, epoch):
        (epoch["swr_run"] / f"{EPOCH}_swr_observed.parquet").unlink()

        tables = swr_detector_inventory(
            epoch["swr_run"], "kay", dot=epoch["dot"], drops=epoch["drops"]
        )

        _assert_unavailable(tables, "kay", "missing")

    def test_a_missing_epoch(self, epoch, tmp_path):
        run = write_carey_run(tmp_path / "carey", events=None)

        tables = carey_inventory(run, dot=epoch["dot"], drops=epoch["drops"])

        _assert_unavailable(tables, "carey", "not in")

    def test_a_failed_record(self, epoch, tmp_path):
        run = write_carey_run(
            tmp_path / "carey", record={"status": "failed", "reason": "band hash"}
        )

        tables = carey_inventory(run, dot=epoch["dot"], drops=epoch["drops"])

        _assert_unavailable(tables, "carey", "failed")

    def test_a_failed_detector_in_a_partial_record(self, epoch):
        manifest_path = epoch["swr_run"] / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["epochs"][EPOCH]["status"] = "partial"
        manifest["epochs"][EPOCH]["detectors"]["yu"] = "failed"
        manifest_path.write_text(json.dumps(manifest))

        yu = swr_detector_inventory(
            epoch["swr_run"], "yu", dot=epoch["dot"], drops=epoch["drops"]
        )
        kay = swr_detector_inventory(
            epoch["swr_run"], "kay", dot=epoch["dot"], drops=epoch["drops"]
        )

        _assert_unavailable(yu, "yu", "failed")
        assert len(kay.inventory) == 3

    def test_a_run_without_a_manifest(self, epoch, tmp_path):
        tables = carey_inventory(
            tmp_path / "nothing", dot=epoch["dot"], drops=epoch["drops"]
        )

        _assert_unavailable(tables, "carey", "manifest")

    def test_a_folder_holding_another_runs_manifest_is_refused(self, epoch, tmp_path):
        run = write_carey_run(tmp_path / "carey")
        (run / "manifest.json").write_text(
            json.dumps({"run_id": "other", "epochs": {}})
        )

        with pytest.raises(ValueError, match="other"):
            carey_inventory(run, dot=epoch["dot"], drops=epoch["drops"])


class TestTheCareyInventory:
    def test_its_events_its_drops_and_its_observed_time(self, epoch, tmp_path):
        run = write_carey_run(tmp_path / "carey")

        tables = carey_inventory(run, dot=epoch["dot"], drops=epoch["drops"])

        _check_dtypes(tables)
        assert len(tables.inventory) + len(tables.set_aside) == len(CAREY)
        assert tables.inventory.event_key.tolist() == [
            f"carey:{PREFIX}:1",
            f"carey:{PREFIX}:3",
        ]
        assert _reasons(tables.set_aside) == {f"carey:{PREFIX}:2": "near_artifact"}
        observed = tables.observations[tables.observations.available]
        np.testing.assert_allclose(
            observed[["start_s", "end_s"]].to_numpy(), [(100.001, 109.997)]
        )
        assert set(tables.inventory.run_id) == {"careyrun01"}


class TestTheMuaInventory:
    def test_its_kept_events_and_the_ones_the_rule_drops(self, epoch):
        tables = mua_inventory(epoch["dot"], drops=epoch["drops"], run_id="decode01")

        _check_dtypes(tables)
        assert len(tables.inventory) + len(tables.set_aside) == len(MUA)
        assert tables.inventory.event_key.tolist() == [
            f"mua:{PREFIX}:1",
            f"mua:{PREFIX}:3",
        ]
        assert tables.inventory.whole_event_speed_eligible.all()
        assert _reasons(tables.set_aside) == {f"mua:{PREFIX}:2": "near_artifact"}
        aside = tables.set_aside.iloc[0]
        assert (aside.interval_start_s, aside.interval_end_s) == (105.5, 105.75)
        kept = sorted(set(epoch["dot"]["MUA_event"]) - {0})
        assert tables.inventory.event_key.tolist() == [
            f"mua:{PREFIX}:{n}" for n in kept
        ]

    def test_it_observes_the_grid_outside_the_mask(self, epoch):
        tables = mua_inventory(epoch["dot"], drops=epoch["drops"], run_id="decode01")

        observed = tables.observations[tables.observations.available]
        starts, ends = observed.start_s.to_numpy(), observed.end_s.to_numpy()
        assert starts[0] == TIME[0] and ends[-1] == TIME[-1]
        # each mask interval, closed, lies between two observed stretches
        np.testing.assert_allclose(ends[:-1], [104.008, 106.008])
        np.testing.assert_allclose(starts[1:], [104.212, 106.212])

    def test_an_epoch_without_events_keeps_the_schema(self, epoch):
        raw = raw_dot(mua=[], non_local=[])
        dot, drops = with_swr_source(raw), artifact_event_drops(raw)

        for tables in (
            mua_inventory(dot, drops=drops, run_id="decode01"),
            state_space_inventory(dot, drops=drops, run_id="decode01"),
        ):
            _check_dtypes(tables)
            assert tables.inventory.empty and tables.set_aside.empty


class TestTheStateSpaceInventory:
    def test_its_kept_events_and_the_ones_the_rule_drops(self, epoch):
        tables = state_space_inventory(
            epoch["dot"], drops=epoch["drops"], run_id="decode01"
        )

        _check_dtypes(tables)
        assert len(tables.inventory) + len(tables.set_aside) == len(NON_LOCAL)
        assert tables.inventory.whole_event_speed_eligible.all()
        np.testing.assert_allclose(
            tables.inventory.event_start_s, [NON_LOCAL[0][0], NON_LOCAL[3][0]]
        )
        assert sorted(tables.set_aside.reason) == ["artifact", "near_artifact"]
        # every key is the one the canonical table gives the event
        undropped = from_classified_nonlocal_table(
            build_classified_nonlocal_event_table(epoch["raw"]),
            brain_area="HPC",
            run_id="decode01",
        ).set_index("event_start_s")
        assert undropped.loc[NON_LOCAL[1][0], "event_key"] in set(
            tables.set_aside.event_key
        )
        assert undropped.loc[NON_LOCAL[2][0], "event_key"] in set(
            tables.set_aside.event_key
        )
        assert tables.inventory.event_key.tolist() == [
            undropped.loc[NON_LOCAL[0][0], "event_key"],
            undropped.loc[NON_LOCAL[3][0], "event_key"],
        ]

    def test_it_observes_where_the_posterior_is_finite_outside_the_mask(self, epoch):
        tables = state_space_inventory(
            epoch["dot"], drops=epoch["drops"], run_id="decode01"
        )

        observed = tables.observations[tables.observations.available]
        ends = observed.end_s.to_numpy()
        starts = observed.start_s.to_numpy()
        np.testing.assert_allclose(ends[:-1], [104.008, 106.008, 108.498])
        np.testing.assert_allclose(starts[1:], [104.212, 106.212, 108.6])

    def test_native_counts_are_eligible_counts_and_say_why(self):
        assert set(NATIVE_IS_ELIGIBLE) == {"mua", "state_space"}
        assert all("speed-eligible" in note for note in NATIVE_IS_ELIGIBLE.values())


class TestTheMemorylessInventory:
    def _run(self, tmp_path, n_shuffles=100):
        from continuum_swr_replay.analysis.event_definition import (
            swr_source,
        )

        table = memoryless_table(swr_source.swr_source_name(), n_shuffles)
        return write_memoryless_run(tmp_path / "memoryless", table)

    def test_each_cohort_under_the_callers_name(self, epoch, tmp_path):
        run = self._run(tmp_path)

        published = memoryless_inventory(
            run,
            cohort="published",
            detector="memoryless",
            dot=epoch["dot"],
            drops=epoch["drops"],
        )
        criteria = memoryless_inventory(
            run,
            cohort="replay_criteria",
            detector="memoryless_criteria",
            dot=epoch["dot"],
            drops=epoch["drops"],
        )

        for tables in (published, criteria):
            _check_dtypes(tables)
        assert published.inventory.event_key.tolist() == [
            f"memoryless:{PREFIX}:11",
            f"memoryless:{PREFIX}:12",
        ]
        assert published.inventory.whole_event_speed_eligible.tolist() == [True, False]
        assert criteria.inventory.event_key.tolist() == [
            f"memoryless_criteria:{PREFIX}:11",
            f"memoryless_criteria:{PREFIX}:12",
            f"memoryless_criteria:{PREFIX}:14",
        ]
        assert set(criteria.inventory.detector) == {"memoryless_criteria"}
        assert set(criteria.observations.detector) == {"memoryless_criteria"}
        assert _reasons(published.set_aside) == {f"memoryless:{PREFIX}:13": "artifact"}
        assert _reasons(criteria.set_aside) == {
            f"memoryless_criteria:{PREFIX}:13": "artifact"
        }
        # native: every candidate the cohort selects
        assert len(published.inventory) + len(published.set_aside) == 3

    def test_the_second_pass_reads_bounds_sampled_on_the_grid(self, epoch, tmp_path):
        """The stored ``overlaps_dropped_event`` is not read. Candidate 16 ends
        0.5 ms past the first sample of dropped MUA 2 (105.500), so its stored
        bounds overlap it, but its last sample is 105.500: on the grid it only
        touches MUA 2 and is kept. Candidate 17 overlaps dropped Kay 3 on the
        grid though its stored reason is None, and is set aside."""
        from continuum_swr_replay.analysis.event_definition import (
            swr_source,
        )

        table = memoryless_table(swr_source.swr_source_name())
        extra = table.iloc[[0, 0]].assign(
            ms_event_id=[16, 17],
            event_start_s=[105.300, 103.500],
            event_end_s=[105.5005, 103.650],
            artifact_drop_reason=["overlaps_dropped_event", None],
        )
        extra["event_duration_s"] = extra.event_end_s - extra.event_start_s
        run = write_memoryless_run(
            tmp_path / "memoryless", pd.concat([table, extra], ignore_index=True)
        )

        tables = memoryless_inventory(
            run,
            cohort="published",
            detector="memoryless",
            dot=epoch["dot"],
            drops=epoch["drops"],
        )

        _check_dtypes(tables)
        assert tables.inventory.event_key.tolist() == [
            f"memoryless:{PREFIX}:{n}" for n in (11, 12, 16)
        ]
        assert _reasons(tables.set_aside) == {
            f"memoryless:{PREFIX}:13": "artifact",
            f"memoryless:{PREFIX}:17": "overlaps_dropped_event",
        }

    def test_it_observes_the_decoded_windows(self, epoch, tmp_path):
        tables = memoryless_inventory(
            self._run(tmp_path),
            cohort="published",
            detector="memoryless",
            dot=epoch["dot"],
            drops=epoch["drops"],
        )

        observed = tables.observations[tables.observations.available]
        np.testing.assert_allclose(
            observed[["start_s", "end_s"]].to_numpy(),
            [(100.04, 103.96), (104.30, 109.95)],
        )

    @pytest.mark.parametrize(
        ("change", "reason"),
        [
            (lambda epochs: epochs.pop(EPOCH), "not in"),
            (
                lambda epochs: epochs.__setitem__(
                    EPOCH, {"status": "failed", "error": "boom"}
                ),
                "failed",
            ),
            (lambda epochs: epochs[EPOCH].__setitem__("n_replay", 3), "n_replay"),
            (
                lambda epochs: epochs[EPOCH]["time_grid"].__setitem__(
                    "n_samples", 4999
                ),
                "n_samples",
            ),
        ],
        ids=["missing epoch", "failed record", "other table", "other grid"],
    )
    def test_an_unusable_record_reads_unavailable(
        self, epoch, tmp_path, change, reason
    ):
        run = self._run(tmp_path)
        edit_record(run, change)

        tables = memoryless_inventory(
            run,
            cohort="published",
            detector="memoryless",
            dot=epoch["dot"],
            drops=epoch["drops"],
        )

        _assert_unavailable(tables, "memoryless", reason)

    def test_an_unshuffled_table_is_refused(self, epoch, tmp_path):
        with pytest.raises(ValueError, match="never shuffled"):
            memoryless_inventory(
                self._run(tmp_path, n_shuffles=0),
                cohort="published",
                detector="memoryless",
                dot=epoch["dot"],
                drops=epoch["drops"],
            )

    def test_an_empty_table_keeps_the_schema(self, epoch, tmp_path):
        from continuum_swr_replay.analysis.event_definition import (
            swr_source,
        )

        table = memoryless_table(swr_source.swr_source_name()).iloc[0:0]
        run = write_memoryless_run(tmp_path / "memoryless", table)

        tables = memoryless_inventory(
            run,
            cohort="published",
            detector="memoryless",
            dot=epoch["dot"],
            drops=epoch["drops"],
        )

        _check_dtypes(tables)
        assert tables.inventory.empty and tables.set_aside.empty


def test_a_frame_not_read_through_the_source_is_refused(epoch):
    with pytest.raises(ValueError, match="source"):
        mua_inventory(epoch["raw"], drops=epoch["drops"], run_id="decode01")
