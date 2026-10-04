"""The state-evidence runner, one epoch at a time, on a synthetic decode.

A real ``NonLocalClusterlessDetector`` is fitted on simulated spikes over a Y
track and saved beside a ``results.nc`` holding the library's own non-local
log-likelihood and a ``data_over_time`` table, under a made-up epoch id. The
marks come from a stand-in for the database loader, so nothing here touches
Spyglass.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from non_local_detector import NonLocalClusterlessDetector
from non_local_detector.environment import Environment
from non_local_detector.likelihoods.clusterless_kde import (
    predict_clusterless_kde_log_likelihood,
)
from track_linearization import get_linearized_position

from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.likelihood import (
    CLUSTERLESS_LIKELIHOOD_KEYS,
)
from continuum_swr_replay.analysis.hippocampus.state_evidence import (
    ENCODING_KEY,
    STATES,
)
from continuum_swr_replay.table_io import read_table
from scripts.qa import run_state_evidence as runner
from tests.unit._memoryless_clusterless_epoch import _y_track

EPOCH = "aa20200101_02_r1_HPC"
FS = 500.0
EDGE_ORDER = [(0, 3), (3, 1), (3, 2)]


def simulate_decode(directory: Path) -> dict:
    """Write a decode's three files for EPOCH; return what the loader would give."""
    rng = np.random.default_rng(0)
    n_time = 6000
    time = 1000.0 + np.arange(n_time) / FS
    # the animal shuttles along the bottom of the Y and pauses at both ends
    phase = np.clip(1.6 * np.sin(2 * np.pi * np.arange(n_time) / FS / 3.0), -1, 1)
    position = np.c_[20.0 + 18.0 * phase, np.zeros(n_time)]
    speed = np.abs(np.gradient(position[:, 0], 1 / FS))
    graph = _y_track()
    linear = get_linearized_position(
        position, graph, edge_order=EDGE_ORDER, edge_spacing=8.0
    ).linear_position.to_numpy()

    spike_times, marks = [], []
    for k, center in enumerate([5.0, 15.0, 33.0, 43.0]):
        rate = 60.0 * np.exp(-0.5 * ((linear - center) / 5.0) ** 2) + 1.0
        sample = np.repeat(np.arange(n_time), rng.poisson(rate / FS))
        spike_times.append(np.sort(time[sample] + rng.uniform(0, 1 / FS, sample.size)))
        marks.append(rng.normal(100.0 + 30.0 * k, 10.0, (sample.size, 2)))
    is_training = np.ones(n_time, dtype=bool)
    is_training[1000:1100] = False  # a ripple

    detector = NonLocalClusterlessDetector(
        environments=Environment(
            place_bin_size=4.0,
            track_graph=graph,
            edge_order=EDGE_ORDER,
            edge_spacing=8.0,
        ),
        clusterless_algorithm="clusterless_kde",
        clusterless_algorithm_params={
            "position_std": 3.5,
            "waveform_std": 24.0,
            "block_size": 100,
            "disable_progress_bar": True,
        },
    )
    detector.initialize_environments(position)
    detector.fit_encoding_model(
        time, position, spike_times, marks, is_training=is_training
    )
    joblib.dump(detector, directory / f"{EPOCH}_detector.pkl")

    encoding = detector.encoding_model_[ENCODING_KEY]
    environment = encoding["environment"]
    interior = np.asarray(environment.is_track_interior_).ravel()
    non_local = np.asarray(
        predict_clusterless_kde_log_likelihood(
            time, time, position, spike_times, marks,
            **{key: encoding[key] for key in CLUSTERLESS_LIKELIHOOD_KEYS},
            is_local=False, disable_progress_bar=True,
        )
    )  # fmt: skip
    n_bins = interior.size
    state = (
        ["Local"] * n_bins
        + ["No-Spike"]
        + ["Non-Local Continuous"] * n_bins
        + ["Non-Local Fragmented"] * n_bins
    )
    log_likelihood = np.zeros((n_time, len(state)), dtype=np.float32)
    log_likelihood[:, n_bins + 1 : 2 * n_bins + 1][:, interior] = non_local
    xr.Dataset(
        {"log_likelihood": (("time", "state_bins"), log_likelihood)},
        coords={"time": time, "state": ("state_bins", state)},
    ).to_netcdf(directory / f"{EPOCH}_results.nc", engine="h5netcdf")

    raw = np.clip(rng.normal(0.0, 1.0, (n_time, len(STATES))), 0, None) ** 3 + 1e-3
    raw = np.apply_along_axis(np.convolve, 0, raw, np.ones(9) / 9, mode="same")
    probabilities = raw / raw.sum(1, keepdims=True)
    dot = pd.DataFrame(
        {
            "head_position_x": position[:, 0],
            "head_position_y": position[:, 1],
            "head_speed": speed,
            "linear_position": linear,
            "non_local_event": 0,
            **{
                column: probabilities[:, j]
                for j, column in enumerate(runner.PROBABILITY_COLUMNS)
            },
        },
        index=pd.Index(time, name="time"),
    )
    dot.to_pickle(directory / f"{EPOCH}_80_data_over_time.pkl")
    return {
        "time": time,
        "spike_times": spike_times,
        "marks": marks,
        "is_training": is_training,
    }


@pytest.fixture(autouse=True)
def short_blocks(monkeypatch):
    """2 s time-held-out blocks: the simulated epoch is 12 s long."""
    monkeypatch.setattr(runner, "CV_BLOCK_S", 2.0)


@pytest.fixture(scope="module")
def simulated(tmp_path_factory) -> tuple[Path, dict]:
    directory = tmp_path_factory.mktemp("decode")
    return directory, simulate_decode(directory)


def decode_copy(simulated: tuple[Path, dict], tmp_path: Path) -> Path:
    """This test's own copy of the simulated decode."""
    source, _ = simulated
    target = tmp_path / "decode"
    shutil.copytree(source, target)
    return target


def run(decode_dir: Path, output_dir: Path, inputs: dict) -> dict:
    return runner.run_epoch(
        EPOCH,
        decode_dir=decode_dir,
        output_dir=output_dir,
        load_inputs=lambda _epoch_id: inputs,
        log=lambda _message: None,
    )


def test_an_epoch_is_validated_and_written(simulated, tmp_path):
    decode_dir = decode_copy(simulated, tmp_path)
    output_dir = tmp_path / "out"

    record = run(decode_dir, output_dir, simulated[1])

    assert record["status"] == "ok", record.get("traceback")
    refit = record["canonical_refit"]
    assert refit["same_n_encoding_spikes"] and refit["max_abs_encoding_positions"] == 0
    assert record["stored_log_likelihood"]["max_rel"] < 1e-5
    for name in ("can", "run", "runcv0", "runcv1"):
        assert record["library_log_likelihood"][name]["max_rel"] < 1e-5
    running_only = record["running_only"]
    for name in ("run", "runcv0", "runcv1"):
        assert running_only[name]["event_bins_in_library_span"] == 0
        assert running_only[name]["training_bins"] > 0
    assert (
        running_only["runcv0"]["training_bins"]
        + running_only["runcv1"]["training_bins"]
        == running_only["run"]["training_bins"]
    )
    held_out = record["time_held_out"]
    assert held_out["block_s"] == 2.0 and held_out["windows"] > 0
    assert 0 < held_out["straddling_windows"] < held_out["windows"]
    paths = runner.output_paths(EPOCH, output_dir)
    assert json.loads(paths["validation"].read_text())["status"] == "ok"
    table = read_table(paths["windows"])
    assert (table.epoch_id == EPOCH).all() and (table.animal == "aa").all()
    # every encoding's evidence, and one copy of what they share
    assert {"runcv_loc_mark", "run_loc_mark", "can_loc_mark", "n_spikes"} <= set(table)
    assert not {"runcv_n_spikes", "run_n_spikes", "can_n_spikes"} & set(table)
    assert set(table.cv_fold) == {0, 1}
    assert (table.cv_insample_share > 0).any() and (table.cv_insample_share < 1).all()
    # the held-out scores differ from those of the model fit on every running bin
    assert not np.allclose(table.runcv_loc_mark, table.run_loc_mark, equal_nan=True)
    assert set(table.kind) == {"bout", "baseline", "calib"}
    assert record["windows"]["silent_ns_logratio_identity_max_abs"] < 1e-9

    runner.summarize(output_dir, [EPOCH])
    for name in ("bout_stats.csv", "trend_stats.csv", "silence_stats.csv",
                 "fig_bouts_vs_matched.png", "fig_calibration.png"):  # fmt: skip
        assert (output_dir / name).exists(), name
    # the window-level statistics only on request
    default = pd.read_csv(output_dir / "bout_stats.csv")
    assert not [c for c in default if c.endswith("_window_level")]
    runner.summarize(output_dir, [EPOCH], window_level=True)
    assert "p_signflip_window_level" in pd.read_csv(output_dir / "bout_stats.csv")


def test_the_window_level_flag_is_off_by_default():
    parser = runner.build_parser()

    assert not parser.parse_args([]).window_level_stats
    assert parser.parse_args(["--window-level-stats"]).window_level_stats


def test_a_decode_missing_a_file_is_skipped_with_its_reason(simulated, tmp_path):
    decode_dir = decode_copy(simulated, tmp_path)
    (decode_dir / f"{EPOCH}_results.nc").unlink()

    record = run(decode_dir, tmp_path / "out", simulated[1])

    assert record == {"status": "skipped", "reason": f"no results in {decode_dir}"}


def test_marks_that_are_not_the_decodes_fail_the_epoch(simulated, tmp_path):
    decode_dir = decode_copy(simulated, tmp_path)
    inputs = dict(simulated[1])
    inputs["spike_times"] = [st[1:] for st in inputs["spike_times"]]
    inputs["marks"] = [mk[1:] for mk in inputs["marks"]]

    record = run(decode_dir, tmp_path / "out", inputs)

    assert record["status"] == "failed"
    assert "does not reproduce the stored encoding model" in record["error"]
    paths = runner.output_paths(EPOCH, tmp_path / "out")
    assert not paths["windows"].exists() and not paths["validation"].exists()


def test_an_event_bin_at_running_speed_is_left_out_of_the_running_fits(
    simulated, tmp_path
):
    decode_dir = decode_copy(simulated, tmp_path)
    path = decode_dir / f"{EPOCH}_80_data_over_time.pkl"
    dot = pd.read_pickle(path)
    running = np.flatnonzero(dot.head_speed.to_numpy() >= 4.0)[100]
    dot.iloc[running, dot.columns.get_loc("non_local_event")] = 1
    dot.to_pickle(path)

    record = run(decode_dir, tmp_path / "out", simulated[1])

    assert record["status"] == "ok", record.get("traceback")
    running_only = record["running_only"]
    assert running_only["event_bins_at_running_speed"] == 1
    assert running_only["running_bins_left_out_near_events"] == 3
    for name in ("run", "runcv0", "runcv1"):
        assert running_only[name]["event_bins_in_training_bins"] == 0
        assert running_only[name]["event_bins_in_library_span"] == 0


def test_running_only_masks_refuse_an_event_spike_in_the_library_span():
    n = 200
    time = np.arange(n) * 0.002
    speed = np.full(n, 20.0)
    position = np.c_[np.linspace(0, 100, n), np.zeros(n)]
    folds = np.zeros(n, dtype=np.int8)
    folds[100:] = 1
    in_event = np.zeros(n, dtype=bool)
    in_event[50] = True

    masks, record = runner.running_only_masks(time, speed, position, in_event, folds)

    # the event bin and its neighbours are left out, and the padding stops short
    assert not masks["run"][49:52].any() and masks["run"][48] and masks["run"][52]
    assert record["run"]["event_bins_in_library_span"] == 0
    assert record["running_bins_left_out_near_events"] == 3
    # no position around the event: the library joins the runs across it
    position[49:52] = np.nan
    with pytest.raises(ValueError, match="1 non-local event bins are in the run"):
        runner.running_only_masks(time, speed, position, in_event, folds)
    # a fold with no running bin
    position[49:52] = 0.0
    with pytest.raises(ValueError, match="runcv1 encoding has no training bin"):
        runner.running_only_masks(time, speed, position, in_event, np.zeros(n, np.int8))


def test_every_hpc_decode_is_listed_even_an_incomplete_one(tmp_path):
    for name in [
        "aa20200101_02_r1_HPC_detector.pkl",
        "aa20200101_02_r1_HPC_results.nc",
        "aa20200101_02_r1_HPC_80_data_over_time.pkl",
        "bb20200101_04_r2_HPC_detector.pkl",  # no results or data_over_time
        "aa20200101_02_r1_HPC_contfrag_detector.pkl",  # another decoder's
        "aa20200101_02_r1_mPFC_detector.pkl",
    ]:
        (tmp_path / name).touch()

    assert runner.canonical_epochs(tmp_path) == [
        "aa20200101_02_r1_HPC",
        "bb20200101_04_r2_HPC",
    ]
