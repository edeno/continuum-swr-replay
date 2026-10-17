"""The Carey runner: its outputs, its manifest record, its checks and its reruns.

Nothing here reaches the Spyglass database or ``Processed-Data``. Every fetch
is a fake: the epoch is ``ripple_detection``'s simulated session at 1000 Hz,
its band the package's 150-250 Hz filter of the simulated LFP, and its twenty
multiunit trains the simulated units, one per tetrode. The SWR run's record is
made from the same inputs, so its hashes match unless a test changes them.
The driver runs on an in-process ``LocalCluster`` of worker threads.
"""

from __future__ import annotations

import json
from functools import cache
from types import MappingProxyType, SimpleNamespace

import dask
import joblib
import numpy as np
import pandas as pd
import pytest
from dask.distributed import LocalCluster
from ripple_detection import filter_ripple_band
from ripple_detection.simulate import simulate_session, simulate_time

import scripts.decoding.run_carey_detection as runner
from continuum_swr_replay.analysis.hippocampus import swr_source
from continuum_swr_replay.analysis.hippocampus.swr_inventories import CAREY_SETTINGS
from continuum_swr_replay.data_loaders.utils import parse_nwb_filename
from continuum_swr_replay.table_io import read_table, sha256_file
from scripts.decoding.run_carey_detection import main, run_epoch
from scripts.decoding.run_swr_detection import array_sha256, epoch_id
from tests.unit._swr_inventory_fixtures import write_artifact_run

FS = 1000.0
BASE_TIME_S = 1.58e9
"""Unix-like timestamps, as Spyglass stores them."""

NWB_FILE_NAME = "sim20200101_.nwb"
EPOCH_NAME = "02_r1"
IDENTIFIER = epoch_id(NWB_FILE_NAME, EPOCH_NAME)

INJECTED_RIPPLE_TIMES_S = (5.0, 10.0, 15.0, 20.0, 25.0)
"""Seconds after the first sample: the five ripple-plus-burst events of 30 s."""

N_TETRODES = 20
TETRODE_RATE_HZ = (10.0, 40.0)
"""Each simulated tetrode's baseline multiunit rate, drawn in this range."""

BAND_ELECTRODE_IDS = np.array([0, 4, 8, 12])
KEPT_ELECTRODE_IDS = [0, 4, 8]
"""The SWR run's rule R-T dropped electrode 12."""

N_FEATURES = 4
SPEED_CM_PER_S = 2.0


@cache
def _simulated_epoch() -> SimpleNamespace:
    """A 30 s still session; cached, so copy an array before changing it."""
    time = BASE_TIME_S + simulate_time(30_000, FS)
    session = simulate_session(
        time,
        BASE_TIME_S + np.asarray(INJECTED_RIPPLE_TIMES_S),
        n_channels=BAND_ELECTRODE_IDS.size,
        n_units=N_TETRODES,
        baseline_rate=TETRODE_RATE_HZ,
        rng=0,
    )
    spike_times = [
        np.repeat(time, column.astype(int)) for column in session.multiunit.T
    ]
    rng = np.random.default_rng(1)
    features = [rng.normal(size=(times.size, N_FEATURES)) for times in spike_times]
    return SimpleNamespace(
        time=time,
        band=filter_ripple_band(session.lfps, sampling_frequency=FS),
        spike_times=spike_times,
        features=features,
        windows=session.ripple_windows,
    )


def _swr_inputs(epoch: SimpleNamespace) -> dict:
    """The SWR run's record of the epoch, made from the same inputs."""
    kept = np.isin(BAND_ELECTRODE_IDS, KEPT_ELECTRODE_IDS)
    return {
        "band_electrode_ids": BAND_ELECTRODE_IDS.tolist(),
        "kept_electrode_ids": KEPT_ELECTRODE_IDS,
        "sampling_frequency": FS,
        "position_span_s": (float(epoch.time[0]), float(epoch.time[-1])),
        "input_sha256": {
            "time": array_sha256(epoch.time),
            "ripple_band": array_sha256(epoch.band[:, kept]),
        },
    }


def _encoding_model(epoch: SimpleNamespace) -> dict:
    """Fitted on every other mark of each tetrode, as a training span would be."""
    return {"encoding_spike_waveform_features": [f[::2] for f in epoch.features]}


def _fetchers(
    epoch: SimpleNamespace,
    *,
    swr_inputs: dict | None = None,
    encoding_model: dict | None = None,
    n_tetrodes: int = N_TETRODES,
    decode_span_s: tuple[float, float] | None = None,
) -> dict:
    """``run_epoch``'s fetch arguments, every one a fake.

    The marks are loaded over ``decode_span_s`` (by default the LFP's span),
    and the encoding model is fitted on marks within it, as a decode's is.
    """
    position_time = np.linspace(epoch.time[0], epoch.time[-1], 15_001)
    position_info = pd.DataFrame(
        {"head_speed": np.full(position_time.size, SPEED_CM_PER_S)},
        index=pd.Index(position_time, name="time"),
    )
    inputs = {
        "time": epoch.time,
        "ripple_band": epoch.band,
        "electrode_ids": BAND_ELECTRODE_IDS,
        "sampling_frequency": FS,
    }
    if decode_span_s is None:
        decode_span_s = (float(epoch.time[0]), float(epoch.time[-1]))
    in_span = [
        (times >= decode_span_s[0]) & (times <= decode_span_s[1])
        for times in epoch.spike_times
    ]
    spike_times = [t[keep] for t, keep in zip(epoch.spike_times, in_span, strict=True)]
    features = [f[keep] for f, keep in zip(epoch.features, in_span, strict=True)]
    model = (
        {"encoding_spike_waveform_features": [f[::2] for f in features]}
        if encoding_model is None
        else encoding_model
    )
    marks = {
        "spike_times": spike_times[:n_tetrodes],
        "spike_waveform_features": features[:n_tetrodes],
        "encoding_model": {
            "encoding_spike_waveform_features": model[
                "encoding_spike_waveform_features"
            ][:n_tetrodes]
        },
        "record": {
            "rule": "simulated",
            "detector_sha256": "0" * 64,
            "decode_span_s": list(decode_span_s),
        },
    }
    return {
        "fetch_inputs": lambda nwb_file_name, epoch_name: inputs,
        "fetch_position": lambda nwb_file_name, epoch_name: {
            "position_info": position_info
        },
        "fetch_swr_inputs": lambda identifier: (
            _swr_inputs(epoch) if swr_inputs is None else swr_inputs
        ),
        "fetch_marks": lambda identifier: marks,
    }


def _outputs(output_dir) -> dict:
    return {
        "events": output_dir / f"{IDENTIFIER}_carey_events.parquet",
        "observed": output_dir / f"{IDENTIFIER}_carey_observed.parquet",
    }


def _overlapping(events: pd.DataFrame, lo: float, hi: float) -> pd.Series:
    return (events.start_time <= hi) & (events.end_time >= lo)


# ---------------------------------------------------------------------------
# One epoch
# ---------------------------------------------------------------------------


def test_an_epoch_writes_its_events_its_observed_stretches_and_its_record(tmp_path):
    epoch = _simulated_epoch()
    output_dir = tmp_path / "out"

    record = run_epoch(
        NWB_FILE_NAME, EPOCH_NAME, output_dir=output_dir, **_fetchers(epoch)
    )

    assert record["status"] == "ok", record
    paths = _outputs(output_dir)
    events = read_table(paths["events"])
    observed = read_table(paths["observed"])
    assert len(events) == record["n_events"]
    for start_s, end_s in epoch.windows:
        assert _overlapping(events, start_s, end_s).sum() == 1
    assert (events["n_active_units"] >= 5).all()
    for frame in (events, observed):
        assert frame.attrs["run_id"] == "out"
        assert frame.attrs["epoch_id"] == IDENTIFIER
        assert frame.attrs["produced_by"] == runner.PRODUCED_BY
        assert frame.attrs["commit"] == record["provenance"]["commit"]
    assert events.attrs["n_dropped_overlapping_unobserved"] == 0
    assert list(observed.columns) == ["scope", "start_s", "end_s"]
    assert (observed["scope"] == "carey").all()
    np.testing.assert_allclose(
        observed[["start_s", "end_s"]], [[epoch.time[0], epoch.time[-1]]]
    )

    assert record["electrode_ids"] == KEPT_ELECTRODE_IDS
    assert record["n_multiunit_tetrodes"] == N_TETRODES
    assert record["input_sha256"] == _swr_inputs(epoch)["input_sha256"]
    assert record["n_dropped_overlapping_unobserved"] == 0
    # a decode span covering the LFP sets nothing unobserved
    assert record["decode_span_s"] == [epoch.time[0], epoch.time[-1]]
    assert record["seconds_outside_decode_span"] == 0
    assert record["marks"]["rule"] == "simulated"
    assert len(record["marks"]["sha256"]) == 64
    assert record["wall_clock_s"] > 0
    assert record["worker_peak_rss_gib"] > 0
    assert {name: entry["path"] for name, entry in record["outputs"].items()} == {
        name: str(path) for name, path in paths.items()
    }
    for entry in record["outputs"].values():
        assert entry["sha256"] == sha256_file(entry["path"])
    json.dumps(record, allow_nan=False)


def test_the_artifact_mask_leaves_the_scaling_and_drops_what_it_overlaps(tmp_path):
    """The mask on the 15 s event: that event is dropped and counted, and its
    time is not observed."""
    epoch = _simulated_epoch()
    at = BASE_TIME_S + 15.0
    artifact_dir = write_artifact_run(
        tmp_path / "artifacts",
        {IDENTIFIER: [("ca1", at, at)]},
        span=(epoch.time[0], epoch.time[-1]),
    )

    unmasked = run_epoch(
        NWB_FILE_NAME, EPOCH_NAME, output_dir=tmp_path / "unmasked", **_fetchers(epoch)
    )

    record = run_epoch(
        NWB_FILE_NAME,
        EPOCH_NAME,
        output_dir=tmp_path / "out",
        artifact_dir=artifact_dir,
        **_fetchers(epoch),
    )

    assert record["status"] == "ok", record
    events = read_table(_outputs(tmp_path / "out")["events"])
    observed = read_table(_outputs(tmp_path / "out")["observed"])
    assert record["n_events"] == unmasked["n_events"] - 1
    assert record["n_dropped_overlapping_unobserved"] == 1
    assert record["artifacts"]["run_id"] == artifact_dir.name
    assert not _overlapping(events, at - 0.3, at + 0.3).any()
    in_mask = (observed.start_s <= at + 0.3) & (observed.end_s >= at - 0.3)
    assert not in_mask.any()
    assert len(observed) == 2


def test_lfp_outside_the_decode_span_is_unobserved(tmp_path):
    """No spike was loaded outside the decode span, so the LFP there is out of
    the scaling, the events and the observed stretches; the LFP is not cut."""
    epoch = _simulated_epoch()
    lo, hi = float(epoch.time[2_000]), float(epoch.time[27_000])

    record = run_epoch(
        NWB_FILE_NAME,
        EPOCH_NAME,
        output_dir=tmp_path / "out",
        **_fetchers(epoch, decode_span_s=(lo, hi)),
    )

    assert record["status"] == "ok", record
    assert record["input_sha256"] == _swr_inputs(epoch)["input_sha256"]
    assert record["decode_span_s"] == [lo, hi]
    assert record["position_span_s"] == [epoch.time[0], epoch.time[-1]]
    assert record["seconds_outside_decode_span"] == (2_000 + 2_999) / FS
    events = read_table(_outputs(tmp_path / "out")["events"])
    observed = read_table(_outputs(tmp_path / "out")["observed"])
    np.testing.assert_array_equal(observed[["start_s", "end_s"]], [[lo, hi]])
    assert ((events.start_time >= lo) & (events.end_time <= hi)).all()
    for start_s, end_s in epoch.windows:  # all five lie inside the decode span
        assert _overlapping(events, start_s, end_s).sum() == 1


@pytest.mark.parametrize("name", ["time", "ripple_band"])
def test_inputs_other_than_the_swr_runs_fail_the_epoch_and_write_nothing(
    tmp_path, name
):
    epoch = _simulated_epoch()
    swr_inputs = _swr_inputs(epoch)
    swr_inputs["input_sha256"][name] = "f" * 64
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    for path in _outputs(output_dir).values():  # left by an earlier run
        path.write_text("old")

    record = run_epoch(
        NWB_FILE_NAME,
        EPOCH_NAME,
        output_dir=output_dir,
        **_fetchers(epoch, swr_inputs=swr_inputs),
    )

    assert record["status"] == "failed"
    assert f"{name} hash" in record["reason"]
    assert swr_source.SWR_INVENTORY_RUN_ID in record["reason"]
    assert not any(path.exists() for path in _outputs(output_dir).values())


def test_a_band_without_the_swr_runs_electrodes_fails_the_epoch(tmp_path):
    epoch = _simulated_epoch()
    swr_inputs = _swr_inputs(epoch) | {"band_electrode_ids": [0, 4, 8, 16]}

    record = run_epoch(
        NWB_FILE_NAME,
        EPOCH_NAME,
        output_dir=tmp_path / "out",
        **_fetchers(epoch, swr_inputs=swr_inputs),
    )

    assert record["status"] == "failed"
    assert "electrodes" in record["reason"]
    assert not any(path.exists() for path in _outputs(tmp_path / "out").values())


def test_marks_that_do_not_match_the_encoding_model_fail_the_epoch(tmp_path):
    epoch = _simulated_epoch()
    model = _encoding_model(epoch)
    model["encoding_spike_waveform_features"][3] = (
        model["encoding_spike_waveform_features"][3] + 1.0
    )

    record = run_epoch(
        NWB_FILE_NAME,
        EPOCH_NAME,
        output_dir=tmp_path / "out",
        **_fetchers(epoch, encoding_model=model),
    )

    assert record["status"] == "failed"
    assert "Electrode 3" in record["error"]
    assert "encoding" in record["error"]
    assert not any(path.exists() for path in _outputs(tmp_path / "out").values())


def test_a_detector_error_is_recorded_as_failed_with_the_error(tmp_path):
    """Three tetrodes, fewer than the five active units Carey needs."""
    epoch = _simulated_epoch()

    record = run_epoch(
        NWB_FILE_NAME,
        EPOCH_NAME,
        output_dir=tmp_path / "out",
        **_fetchers(epoch, n_tetrodes=3),
    )

    assert record["status"] == "failed"
    assert record["error"].startswith("ValueError(")
    assert record["n_multiunit_tetrodes"] == 3
    assert not any(path.exists() for path in _outputs(tmp_path / "out").values())


def test_the_decodes_marks_are_read_over_its_grid_and_keyed_by_its_detector(
    tmp_path, monkeypatch
):
    """``load_marks`` gets the decode grid's span and its detector's sha256."""
    epoch = _simulated_epoch()
    identifier = "sim20200101_02_r1_HPC"
    detector = tmp_path / f"{identifier}_detector.pkl"
    joblib.dump({"stand-in": True}, detector)
    grid = epoch.time[::2]
    pd.DataFrame({"head_speed": 0.0}, index=pd.Index(grid, name="time")).to_pickle(
        tmp_path / f"{identifier}_80_data_over_time.pkl"
    )
    model = _encoding_model(epoch)
    monkeypatch.setattr(runner, "load_clusterless_encoding_model", lambda path: model)
    asked = []

    def marks_loader(key, cache_dir, span_s, detector_sha256):
        asked.append((key.epoch_id, cache_dir, span_s, detector_sha256))
        return epoch.spike_times, epoch.features

    marks = runner.decode_marks(
        identifier,
        root=tmp_path,
        cache_dir=tmp_path / "cache",
        marks_loader=marks_loader,
    )

    assert asked == [
        (
            identifier,
            tmp_path / "cache",
            (float(grid[0]), float(grid[-1])),
            sha256_file(detector),
        )
    ]
    assert marks["encoding_model"] is model
    assert marks["record"]["detector_sha256"] == sha256_file(detector)
    assert marks["record"]["decode_span_s"] == [float(grid[0]), float(grid[-1])]


# ---------------------------------------------------------------------------
# The driver
# ---------------------------------------------------------------------------


@pytest.fixture
def driver(tmp_path, monkeypatch):
    """``main`` over one simulated epoch, on two worker threads, with fakes.

    Returns the epochs ``run_epoch`` ran, in order.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(swr_source, "ARTIFACT_INTERVALS_RUN_ID", None)
    animal, date = parse_nwb_filename(NWB_FILE_NAME)
    monkeypatch.setattr(
        runner,
        "epochs_to_run",
        lambda: pd.DataFrame(
            {"animal": [animal], "date": [date], "epoch": [2]},
            index=pd.MultiIndex.from_tuples(
                [(NWB_FILE_NAME, EPOCH_NAME)],
                names=["nwb_file_name", "interval_list_name"],
            ),
        ),
    )
    monkeypatch.setattr(
        runner,
        "LocalCluster",
        lambda **kwargs: LocalCluster(
            n_workers=2,
            threads_per_worker=1,
            processes=False,
            dashboard_address=":0",
            local_directory=str(tmp_path / "dask-scratch"),
        ),
    )
    real_run_epoch, ran = runner.run_epoch, []

    def run_epoch_on_fakes(nwb_file_name, epoch_name, **kwargs):
        ran.append(epoch_name)
        return real_run_epoch(
            nwb_file_name, epoch_name, **kwargs, **_fetchers(_simulated_epoch())
        )

    monkeypatch.setattr(runner, "run_epoch", run_epoch_on_fakes)
    with dask.config.set({"temporary-directory": str(tmp_path / "dask-temporary")}):
        yield ran


def _cli(tmp_path, *extra: str) -> dict:
    output_root = tmp_path / "carey"
    main(
        [
            "--animal",
            "sim",
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


def test_the_manifest_records_the_settings_and_the_epoch(tmp_path, driver):
    manifest = _cli(tmp_path)

    settings = manifest["settings"]
    assert settings["carey_settings"] == dict(CAREY_SETTINGS)
    assert settings["swr_run_id"] == swr_source.SWR_INVENTORY_RUN_ID
    assert settings["artifacts"]["run_id"] is None
    assert len(settings["adaptations"]) == 4
    assert "tetrodes as units" in settings["configuration"]
    assert {"ripple_detection_commit", "marks_rule", "grid"} <= set(settings)
    assert manifest["epochs"][IDENTIFIER]["status"] == "ok"
    assert (tmp_path / "logs" / f"{IDENTIFIER}_carey.log").exists()


def test_a_rerun_skips_an_ok_epoch_unless_overwritten(tmp_path, driver, capsys):
    first = _cli(tmp_path)
    capsys.readouterr()

    again = _cli(tmp_path)

    assert "already written" in capsys.readouterr().out
    assert driver == [EPOCH_NAME]
    assert again == first

    _cli(tmp_path, "--overwrite")

    assert driver == [EPOCH_NAME, EPOCH_NAME]


def test_a_rerun_with_other_settings_is_refused(tmp_path, driver, monkeypatch):
    _cli(tmp_path)
    changed = dict(CAREY_SETTINGS) | {"high_threshold": 4.0}
    monkeypatch.setattr(runner, "CAREY_SETTINGS", MappingProxyType(changed))

    with pytest.raises(SystemExit, match="other settings"):
        _cli(tmp_path, "--overwrite")

    assert driver == [EPOCH_NAME]
