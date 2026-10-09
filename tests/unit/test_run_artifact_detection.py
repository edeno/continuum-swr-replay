"""The artifact runner: its per-epoch file, its manifest, and its driver.

Nothing here reaches the Spyglass database. Every fetch is a fake built from
a synthetic epoch (``_artifact_synthetic``): the referenced CA1 LFP and the
reference electrode as ``get_ripple_band_inputs`` returns them, a position
span, and the file's HPC marks. The epoch register is stubbed. The parallel
driver runs on an in-process ``LocalCluster`` of threads; everything is
written under the test's ``tmp_path``.
"""

from __future__ import annotations

import json

import dask
import numpy as np
import pandas as pd
import pytest
from dask.distributed import LocalCluster

import scripts.decoding.run_artifact_detection as runner
from continuum_swr_replay.analysis.hippocampus import swr_source
from continuum_swr_replay.analysis.hippocampus.artifacts.io import (
    CORE_COLUMNS,
    cores_path,
    read_epoch_cores,
)
from continuum_swr_replay.analysis.hippocampus.artifacts.masks import (
    artifact_intervals,
)
from continuum_swr_replay.data_loaders.utils import parse_nwb_filename
from continuum_swr_replay.table_io import read_attrs, sha256_file
from tests.unit._artifact_synthetic import FS, NOISE, SyntheticEpoch

NWB_FILE_NAME = "sim20200101_.nwb"
EPOCH_NAME = "02_r1"
EPOCH_ID = "sim20200101_02_r1_HPC"
ELECTRODE_IDS = np.arange(20) * 4
REFERENCE_ID = 81
SPAN_TRIM = 500
"""Samples of LFP outside the position span at each end."""


def _artifact_epoch() -> SyntheticEpoch:
    """A reference step at 10.5 s and an S block over windows 20-22."""
    epoch = SyntheticEpoch()
    epoch.add_step(10.5, [], 60 * NOISE, on_reference=True)
    epoch.add_band_noise(20.0, 23.0, [0, 1, 2, 3], rms=5 * NOISE)
    return epoch


def _fetchers(epoch: SyntheticEpoch) -> dict:
    """``run_epoch``'s fetchers, reading ``epoch``."""

    def fetch_inputs(nwb_file_name, epoch_name):
        native = np.column_stack([epoch.ca1, epoch.reference]).astype(np.int64)
        return {
            "time": epoch.time.copy(),
            "ripple_band": epoch.ca1.copy(),
            "raw_referenced": native[:, :20].astype(np.float64),
            "raw_referenced_native": native,
            "raw_timestamps": epoch.time.copy(),
            "electrode_ids": ELECTRODE_IDS.copy(),
            "reference_ids": np.full(20, REFERENCE_ID),
            "lfp_band_elect_index": np.arange(20),
            "sampling_frequency": FS,
            "lfp_units": "volts",
            "lfp_conversion": 1.0,
        }

    def fetch_position(nwb_file_name, epoch_name):
        index = pd.Index(epoch.time[SPAN_TRIM:-SPAN_TRIM], name="time")
        return {"position_info": pd.DataFrame({"head_speed": 0.0}, index=index)}

    def fetch_marks(nwb_file_name):
        return list(epoch.spike_times), list(epoch.spike_marks)

    return {
        "fetch_inputs": fetch_inputs,
        "fetch_position": fetch_position,
        "fetch_marks": fetch_marks,
    }


def _run_epoch(tmp_path, epoch: SyntheticEpoch, **fetchers) -> dict:
    return runner.run_epoch(
        NWB_FILE_NAME,
        EPOCH_NAME,
        output_dir=tmp_path,
        **{**_fetchers(epoch), **fetchers},
    )


# ---------------------------------------------------------------------------
# One epoch
# ---------------------------------------------------------------------------


def test_an_epoch_writes_its_cores_and_record(tmp_path):
    epoch = _artifact_epoch()

    record = _run_epoch(tmp_path, epoch)

    assert record["status"] == "ok", record
    path = cores_path(tmp_path, EPOCH_ID)
    assert record["outputs"]["cores"] == {
        "path": str(path),
        "sha256": sha256_file(path),
    }
    cores = pd.read_parquet(path)
    assert list(cores.columns) == list(CORE_COLUMNS)
    assert cores["signatures"].tolist() == ["ref", "S"]
    # the span is the LFP's first and last sample inside the position span
    span = (epoch.time[SPAN_TRIM], epoch.time[-SPAN_TRIM - 1])
    assert record["span_s"] == list(span)
    assert record["position_span_s"] == list(span)
    assert record["n_samples"] == epoch.time.size - 2 * SPAN_TRIM
    assert record["n_channels"] == 20
    assert record["reference_electrode_ids"] == [REFERENCE_ID]
    assert record["n_tetrodes"] == 10
    assert record["n_spikes"] == sum(
        int(((t >= span[0]) & (t <= span[1])).sum()) for t in epoch.spike_times
    )
    assert record["n_cores"] == {"S": 1, "T": 1}
    assert record["n_invalid_samples"] == 0
    assert record["n_time_gaps"] == 0
    assert record["n_windows_without_power"] == 0
    assert record["t_ref_disabled"] is False
    assert record["n_cores_by_signatures"] == {"S": 1, "ref": 1}
    assert record["constants"] == runner.detector_constants()
    expected = artifact_intervals(cores, span)
    assert record["masked_s"] == pytest.approx(
        float((expected["end_time"] - expected["start_time"]).sum())
    )
    assert record["provenance"]["commit"]
    assert record["input_sha256"].keys() == {"time", "ca1_lfp", "reference_lfp"}
    assert record["wall_clock_s"] >= 0
    assert record["worker_peak_rss_gib"] > 0
    json.dumps(record, allow_nan=False)
    attrs = read_attrs(path)
    assert attrs["epoch_id"] == EPOCH_ID
    assert attrs["run_id"] == tmp_path.name
    assert attrs["span_s"] == list(span)


def test_an_epoch_without_artifacts_writes_an_empty_file(tmp_path):
    record = _run_epoch(tmp_path, SyntheticEpoch())

    assert record["status"] == "ok"
    cores = pd.read_parquet(cores_path(tmp_path, EPOCH_ID))
    assert cores.empty
    assert list(cores.columns) == list(CORE_COLUMNS)
    assert record["n_cores"] == {"S": 0, "T": 0}
    assert record["masked_s"] == 0.0


def test_an_epoch_whose_references_are_all_band_electrodes_runs_without_t_ref(
    tmp_path,
):
    """No reference electrode outside the band: T-ref is off, the epoch is
    ``ok``, and its record says so for R9 to flag."""
    epoch = _artifact_epoch()
    fetch = _fetchers(epoch)["fetch_inputs"]

    def fetch_inputs(nwb_file_name, epoch_name):
        inputs = fetch(nwb_file_name, epoch_name)
        inputs["reference_ids"] = np.full(20, ELECTRODE_IDS[0])
        inputs["raw_referenced_native"] = inputs["raw_referenced_native"][:, :20]
        return inputs

    record = _run_epoch(tmp_path, epoch, fetch_inputs=fetch_inputs)

    assert record["status"] == "ok", record
    assert record["n_references"] == 0
    assert record["reference_electrode_ids"] == []
    assert record["t_ref_disabled"] is True
    # the reference step goes unseen; the S block is still found
    assert record["n_cores_by_signatures"] == {"S": 1}


def test_timestamp_gaps_are_recorded(tmp_path):
    epoch = SyntheticEpoch()
    epoch.time[15000:] += 0.25

    record = _run_epoch(tmp_path, epoch)

    assert record["n_time_gaps"] == 1


def test_lfp_outside_the_position_span_is_not_read(tmp_path):
    """A step in the trimmed tail is outside the span, so it is not an artifact
    of this epoch."""
    epoch = SyntheticEpoch()
    epoch.add_step(epoch.time.size / FS - 0.2, [], 60 * NOISE, on_reference=True)

    _run_epoch(tmp_path, epoch)

    assert pd.read_parquet(cores_path(tmp_path, EPOCH_ID)).empty


def test_a_failed_epoch_is_recorded_and_keeps_no_file(tmp_path):
    _run_epoch(tmp_path, _artifact_epoch())
    assert cores_path(tmp_path, EPOCH_ID).exists()

    def fetch_marks(nwb_file_name):
        raise RuntimeError("simulated: no marks")

    record = _run_epoch(tmp_path, _artifact_epoch(), fetch_marks=fetch_marks)

    assert record["status"] == "failed"
    assert record["error"] == repr(RuntimeError("simulated: no marks"))
    assert "Traceback" in record["traceback"]
    assert not cores_path(tmp_path, EPOCH_ID).exists()


def test_lfp_columns_that_are_not_band_and_references_in_id_order_fail(tmp_path):
    epoch = _artifact_epoch()
    fetch = _fetchers(epoch)["fetch_inputs"]

    def fetch_inputs(nwb_file_name, epoch_name):
        inputs = fetch(nwb_file_name, epoch_name)
        inputs["lfp_band_elect_index"] = np.arange(1, 21)  # reference first
        return inputs

    record = _run_epoch(tmp_path, epoch, fetch_inputs=fetch_inputs)

    assert record["status"] == "failed"
    assert "reference" in record["error"]


def test_the_runner_writes_where_the_source_reads():
    assert runner.OUTPUT_ROOT == swr_source.ARTIFACT_INTERVALS_ROOT


def test_the_source_reads_a_run_the_runner_wrote(tmp_path, monkeypatch):
    """End to end: the runner's file and manifest are what ``swr_source`` reads,
    and it pads them into the one mask."""
    epoch = _artifact_epoch()
    output_dir = tmp_path / "artifacts" / "run0001"
    output_dir.mkdir(parents=True)
    record = _run_epoch(output_dir, epoch)
    runner.write_manifest(
        output_dir, {"run_id": "run0001", "epochs": {EPOCH_ID: record}}
    )
    monkeypatch.setattr(swr_source, "ARTIFACT_INTERVALS_ROOT", output_dir.parent)
    monkeypatch.setattr(swr_source, "ARTIFACT_INTERVALS_RUN_ID", "run0001")

    stored = read_epoch_cores(output_dir, EPOCH_ID)
    pd.testing.assert_frame_equal(
        swr_source.artifact_intervals(EPOCH_ID),
        artifact_intervals(stored.cores, stored.span),
    )


# ---------------------------------------------------------------------------
# The driver
# ---------------------------------------------------------------------------


def _epoch_info(*epochs: tuple[str, str]) -> pd.DataFrame:
    """The register's frame (``get_epoch_info``'s shape) for these epochs."""
    animal_date = [parse_nwb_filename(nwb_file_name) for nwb_file_name, _ in epochs]
    return pd.DataFrame(
        {
            "animal": [animal for animal, _ in animal_date],
            "date": [date for _, date in animal_date],
            "epoch": [int(name.split("_")[0]) for _, name in epochs],
        },
        index=pd.MultiIndex.from_tuples(
            epochs, names=["nwb_file_name", "interval_list_name"]
        ),
    )


@pytest.fixture
def in_tmp_path(tmp_path, monkeypatch):
    """The per-epoch logs, written under the working directory, land here."""
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def in_process_cluster(tmp_path, monkeypatch):
    """The runner's worker processes, replaced by two worker threads."""
    asked = []

    def cluster(**kwargs):
        asked.append(kwargs)
        return LocalCluster(
            n_workers=2,
            threads_per_worker=1,
            processes=False,
            dashboard_address=":0",
            local_directory=str(tmp_path / "dask-scratch"),
        )

    monkeypatch.setattr(runner, "LocalCluster", cluster)
    with dask.config.set({"temporary-directory": str(tmp_path / "dask-temporary")}):
        yield asked


@pytest.fixture
def register(monkeypatch):
    """The register serves two analysed epochs and one pending one; returns
    what each register function was asked."""
    asked = []

    def analysed_epochs(include_pending=False):
        asked.append(("analysed", include_pending))
        epochs = [(NWB_FILE_NAME, "02_r1"), (NWB_FILE_NAME, "04_r2")]
        if include_pending:
            epochs.append((NWB_FILE_NAME, "06_r3"))
        return _epoch_info(*epochs)

    def pending_epochs():
        asked.append(("pending", None))
        return _epoch_info((NWB_FILE_NAME, "06_r3"))

    monkeypatch.setattr(runner, "analysed_epochs", analysed_epochs)
    monkeypatch.setattr(runner, "pending_epochs", pending_epochs)
    return asked


REAL_RUN_EPOCH = runner.run_epoch


def _with_fakes(monkeypatch, failing_epoch: str | None = None) -> list:
    """The driver's ``run_epoch`` reads fakes; returns the epochs it ran."""
    real_run_epoch = REAL_RUN_EPOCH
    ran = []

    def run_epoch_on_fakes(nwb_file_name, epoch_name, **kwargs):
        ran.append(epoch_name)
        fetchers = _fetchers(_artifact_epoch())
        if epoch_name == failing_epoch:

            def fetch_inputs(nwb_file_name, epoch_name):
                raise RuntimeError("simulated: the LFP could not be read")

            fetchers["fetch_inputs"] = fetch_inputs
        return real_run_epoch(nwb_file_name, epoch_name, **kwargs, **fetchers)

    monkeypatch.setattr(runner, "run_epoch", run_epoch_on_fakes)
    return ran


def _cli(tmp_path, *extra: str) -> dict:
    output_root = tmp_path / "artifacts"
    runner.main(
        [
            "--run-id",
            "test-run",
            "--output-root",
            str(output_root),
            "--n-workers",
            "2",
            *extra,
        ]
    )
    return json.loads((output_root / "test-run" / "manifest.json").read_text())


@pytest.mark.parametrize(
    ("flags", "asked", "epochs"),
    [
        ([], [("analysed", False)], ["02_r1", "04_r2"]),
        (["--include-pending"], [("analysed", True)], ["02_r1", "04_r2", "06_r3"]),
        (["--pending-only"], [("pending", None)], ["06_r3"]),
    ],
    ids=["analysed", "include-pending", "pending-only"],
)
def test_the_epochs_come_from_the_register(
    tmp_path,
    monkeypatch,
    in_tmp_path,
    in_process_cluster,
    register,
    flags,
    asked,
    epochs,
):
    ran = _with_fakes(monkeypatch)

    manifest = _cli(tmp_path, *flags)

    assert register == asked
    assert sorted(ran) == epochs
    assert sorted(manifest["epochs"]) == [f"sim20200101_{name}_HPC" for name in epochs]
    assert in_process_cluster == [
        {"n_workers": 2, "processes": True, "threads_per_worker": 1}
    ]
    assert manifest["run_id"] == "test-run"
    assert manifest["settings"] == runner.run_settings()


def test_include_pending_and_pending_only_exclude_each_other(
    tmp_path, in_tmp_path, register
):
    with pytest.raises(SystemExit) as stopped:
        _cli(tmp_path, "--include-pending", "--pending-only")

    assert stopped.value.code == 2
    assert register == []


def test_the_flags_narrow_the_registers_epochs(
    tmp_path, monkeypatch, in_tmp_path, in_process_cluster, register
):
    ran = _with_fakes(monkeypatch)

    _cli(tmp_path, "--epoch", "04_r2")

    assert ran == ["04_r2"]


def test_a_failed_epoch_is_recorded_and_the_rest_run(
    tmp_path, monkeypatch, in_tmp_path, in_process_cluster, register
):
    _with_fakes(monkeypatch, failing_epoch="04_r2")

    with pytest.raises(SystemExit) as stopped:
        _cli(tmp_path)

    assert stopped.value.code not in (0, None)
    manifest = json.loads(
        (tmp_path / "artifacts" / "test-run" / "manifest.json").read_text()
    )
    good, bad = "sim20200101_02_r1_HPC", "sim20200101_04_r2_HPC"
    assert manifest["epochs"][good]["status"] == "ok"
    assert manifest["epochs"][bad]["status"] == "failed"
    assert manifest["epochs"][bad]["error"] == repr(
        RuntimeError("simulated: the LFP could not be read")
    )
    output_dir = tmp_path / "artifacts" / "test-run"
    assert cores_path(output_dir, good).exists()
    assert not cores_path(output_dir, bad).exists()
    for identifier in (good, bad):
        assert (tmp_path / "logs" / f"{identifier}_artifacts.log").exists()


def test_a_rerun_skips_finished_epochs_and_redoes_failed_ones(
    tmp_path, monkeypatch, capsys, in_tmp_path, in_process_cluster, register
):
    _with_fakes(monkeypatch, failing_epoch="04_r2")
    with pytest.raises(SystemExit):
        _cli(tmp_path)
    capsys.readouterr()
    ran = _with_fakes(monkeypatch)

    manifest = _cli(tmp_path)

    assert ran == ["04_r2"]
    assert "sim20200101_02_r1_HPC: already written" in capsys.readouterr().out
    assert {r["status"] for r in manifest["epochs"].values()} == {"ok"}

    ran = _with_fakes(monkeypatch)
    _cli(tmp_path, "--overwrite")

    assert sorted(ran) == ["02_r1", "04_r2"]


def test_a_rerun_with_other_settings_is_refused(
    tmp_path, monkeypatch, in_tmp_path, in_process_cluster, register
):
    ran = _with_fakes(monkeypatch)
    _cli(tmp_path)
    monkeypatch.setattr(runner.detect, "T_REF", 10.0)

    with pytest.raises(SystemExit, match="other settings"):
        _cli(tmp_path, "--overwrite")

    assert sorted(ran) == ["02_r1", "04_r2"]


def test_the_run_id_defaults_to_the_commit(
    tmp_path, monkeypatch, in_tmp_path, in_process_cluster, register
):
    _with_fakes(monkeypatch)
    monkeypatch.setattr(
        runner,
        "provenance_record",
        lambda repo: {"commit": "0123456789abcdef", "dirty": False},
    )

    runner.main(["--output-root", str(tmp_path / "artifacts"), "--n-workers", "2"])

    manifest_path = tmp_path / "artifacts" / "0123456789ab" / "manifest.json"
    assert json.loads(manifest_path.read_text())["run_id"] == "0123456789ab"


@pytest.mark.parametrize("outcome", ["ok", "raises"])
def test_each_epoch_task_closes_the_nwb_files_spyglass_opened(
    tmp_path, monkeypatch, outcome
):
    closed = []

    def release_nwb_files():
        closed.append(True)
        return 9

    def run_epoch(nwb_file_name, epoch_name, **kwargs):
        if outcome == "raises":
            raise RuntimeError("simulated: the task itself failed")
        return {"status": "ok"}

    monkeypatch.setattr(runner, "run_epoch", run_epoch)
    monkeypatch.setattr(runner, "release_nwb_files", release_nwb_files)

    def task():
        return runner._run_epoch_logged(
            NWB_FILE_NAME, EPOCH_NAME, {"output_dir": tmp_path}, tmp_path / "logs"
        )

    if outcome == "raises":
        with pytest.raises(RuntimeError, match="the task itself failed"):
            task()
    else:
        assert task()["status"] == "ok"

    assert closed == [True]
    log = tmp_path / "logs" / f"{EPOCH_ID}_artifacts.log"
    assert "closed 9 NWB files Spyglass held open" in log.read_text()


def test_main_raises_the_open_file_limit_before_its_cluster(
    tmp_path, monkeypatch, capsys, in_tmp_path, in_process_cluster, register
):
    clusters_at_raise = []

    def raise_open_file_limit():
        clusters_at_raise.append(len(in_process_cluster))
        return 1024, 65536

    monkeypatch.setattr(runner, "raise_open_file_limit", raise_open_file_limit)
    _with_fakes(monkeypatch)

    _cli(tmp_path)

    assert clusters_at_raise == [0]
    assert "open-file soft limit 1024 -> 65536" in capsys.readouterr().out
