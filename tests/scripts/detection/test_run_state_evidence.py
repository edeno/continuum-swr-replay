"""The state-evidence runner, one epoch at a time, on a synthetic decode.

A real ``NonLocalClusterlessDetector`` is fitted on simulated spikes over a Y
track and saved beside a ``results.nc`` holding the library's own non-local
log-likelihood and a ``data_over_time`` table, under a made-up epoch id. It is
made as an HPC decode under the SWR source is: one inventory event and an
artifact mask leave encoding, and the mask's bins are missing (their stored
log-likelihood 0). The source is a synthetic run folder (``source``). The
marks come from a stand-in for the database loader, so nothing here touches
Spyglass.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import jax.numpy as jnp
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

from continuum_swr_replay.analysis.events import analysed_time
from continuum_swr_replay.analysis.events.artifact_masks import artifact_intervals
from continuum_swr_replay.analysis.events.immobility import (
    EventDefinitionError,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.likelihood import (
    CLUSTERLESS_LIKELIHOOD_KEYS,
)
from continuum_swr_replay.analysis.hippocampus.state_evidence import (
    ENCODING_KEY,
    STATES,
    UNDERFLOW_OCCUPANCY_FLOOR,
    interior_bin_mask,
    nearest_interior_bins,
)
from continuum_swr_replay.intervals import interval_mask
from continuum_swr_replay.runs import batch
from continuum_swr_replay.tables.io import read_table
from scripts.detection import run_state_evidence as runner
from tests.support._memoryless_clusterless_epoch import _y_track
from tests.support._swr_inventory_fixtures import (
    ARTIFACT_RUN_ID,
    RUN_ID,
    artifact_cores,
    use_swr_source,
    write_artifact_run,
    write_swr_inventory,
)

EPOCH = "aa20200101_02_r1_HPC"
FS = 500.0
EDGE_ORDER = [(0, 3), (3, 1), (3, 2)]
N_TIME = 6000
TIME = 1000.0 + np.arange(N_TIME) / FS
RIPPLE = (1000, 1100)
"""Bins (first, last + 1) of the Kay inventory's one event."""
ARTIFACT_CORES = [("spk", TIME[3000], TIME[3000]), ("spk", TIME[5000], TIME[5000])]
"""Spike-only lines at 6 s (while the animal runs) and at 10 s: the mask pads
each by 0.1 s (about bins 2950-3050 and 4950-5050, [9.9, 10.1] s into the
epoch) and excluded time by 0.3 s more."""


SAVED_DECODE_MASKS = {
    "swr_inventory_run_id": RUN_ID,
    "artifact_run_id": ARTIFACT_RUN_ID,
    "training_type": "no_ripple",
}
"""The run ids of the source the simulated decode was made under (``source``)."""


def simulated_is_missing() -> np.ndarray:
    """The decode's missing bins: the mask, as ``hpc_decode_masks`` builds it."""
    return interval_mask(
        TIME, artifact_intervals(artifact_cores(ARTIFACT_CORES), (0.0, 1e6))
    )


@pytest.fixture(autouse=True)
def source(tmp_path, monkeypatch):
    """The SWR source the simulated decode was made under."""
    inventory = write_swr_inventory(
        tmp_path,
        {EPOCH: [(TIME[RIPPLE[0]], TIME[RIPPLE[1] - 1])]},
        artifact_run_id=ARTIFACT_RUN_ID,
    )
    artifacts = write_artifact_run(tmp_path, {EPOCH: ARTIFACT_CORES})
    use_swr_source(monkeypatch, inventory, artifacts)


def simulate_decode(directory: Path) -> dict:
    """Write a decode's three files for EPOCH; return what the loader would give."""
    rng = np.random.default_rng(0)
    n_time = N_TIME
    time = TIME
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
    is_missing = simulated_is_missing()
    is_training = np.ones(n_time, dtype=bool)
    is_training[RIPPLE[0] : RIPPLE[1]] = False  # the inventory's event
    is_training &= ~is_missing

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
    log_likelihood[is_missing] = 0.0  # as the library's predict leaves them
    centers = np.arange(n_bins, dtype=float)
    # each state bin's place (NaN for No-Spike's one), as the decoders save it
    bin_position = np.concatenate([centers, [np.nan], centers, centers])
    xr.Dataset(
        {"log_likelihood": (("time", "state_bins"), log_likelihood)},
        coords={
            "time": time,
            "state": ("state_bins", state),
            "position": ("state_bins", bin_position),
        },
        # the masks' record an HPC decode stamps (hpc_decode_masks)
        attrs={"decode_masks": json.dumps(SAVED_DECODE_MASKS, sort_keys=True)},
    ).to_netcdf(directory / f"{EPOCH}_results.nc", engine="h5netcdf")

    raw = np.clip(rng.normal(0.0, 1.0, (n_time, len(STATES))), 0, None) ** 3 + 1e-3
    raw = np.apply_along_axis(np.convolve, 0, raw, np.ones(9) / 9, mode="same")
    probabilities = raw / raw.sum(1, keepdims=True)
    dot = pd.DataFrame(
        {
            "animal": "aa",
            "day": 20200101,
            "epoch": "02_r1",
            "MUA_event": 0,
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
        "is_missing": is_missing,
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
    # no bin underflows: the check was made, nothing is left out, no column is added
    assert record["underflowed_bins"] == {
        "floor": UNDERFLOW_OCCUPANCY_FLOOR, "n": 0, "bins": [],
    }  # fmt: skip
    assert not [c for c in table if "left_out" in c]
    assert "left_out_selection" not in record["windows"]
    # excluded time is no window's, and no masked bin is in a running-only fit
    excluded = interval_mask(TIME, analysed_time.excluded_intervals(EPOCH))
    assert excluded[2801:3199].all() and excluded[4801:5199].all()
    reach = np.r_[0, np.cumsum(excluded)]
    assert (reach[table.e.to_numpy() + 1] == reach[table.s.to_numpy()]).all()
    assert running_only["missing_bins_at_running_speed"] > 0
    for name in ("run", "runcv0", "runcv1"):
        assert running_only[name]["missing_bins_in_training_bins"] == 0

    runner.summarize(output_dir, [EPOCH])
    for name in ("bout_stats.csv", "trend_stats.csv", "silence_stats.csv",
                 "fig_bouts_vs_matched.png", "fig_calibration.png"):  # fmt: skip
        assert (output_dir / name).exists(), name
    # the window-level statistics only on request
    default = pd.read_csv(output_dir / "bout_stats.csv")
    assert not [c for c in default if c.endswith("_window_level")]
    runner.summarize(output_dir, [EPOCH], window_level=True)
    assert "p_signflip_window_level" in pd.read_csv(output_dir / "bout_stats.csv")


ZONE_SIZE = 3


def give_exact_zero_occupancy(
    decode_dir: Path, inputs: dict, real_bin: int, stored_offset: float = 0.0
) -> None:
    """Make one real bin one the decode saw with an occupancy of exactly 0.

    A bin the animal never reached and the KDE put no weight on. The decode (GPU
    or CPU) reads ``occupancy > 0`` as False there and takes the library's floor,
    so its stored likelihood and ground intensity are what a CPU rebuild gives
    (D1: they agree to 2e-14 per spike). The detector and ``results.nc`` are
    rewritten that way; ``stored_offset`` then moves the stored likelihood off it.
    """
    from non_local_detector.likelihoods.common import EPS

    path = decode_dir / f"{EPOCH}_detector.pkl"
    detector = joblib.load(path)
    encoding = detector.encoding_model_[ENCODING_KEY]
    occupancy = np.array(encoding["occupancy"], dtype=np.float32)
    occupancy[real_bin] = 0.0
    encoding["occupancy"] = occupancy
    ground = np.array(encoding["summed_ground_process_intensity"])
    ground[real_bin] = np.clip(
        np.asarray(encoding["mean_rates"]) * EPS, EPS, None
    ).sum()
    encoding["summed_ground_process_intensity"] = ground
    joblib.dump(detector, path)

    dot = pd.read_pickle(decode_dir / f"{EPOCH}_80_data_over_time.pkl")
    non_local = np.asarray(
        predict_clusterless_kde_log_likelihood(
            inputs["time"], inputs["time"],
            dot[["head_position_x", "head_position_y"]].to_numpy(),
            inputs["spike_times"], inputs["marks"],
            **{key: encoding[key] for key in CLUSTERLESS_LIKELIHOOD_KEYS},
            is_local=False, disable_progress_bar=True,
        )
    )  # fmt: skip
    results_path = decode_dir / f"{EPOCH}_results.nc"
    with xr.open_dataset(results_path, engine="h5netcdf") as results:
        results = results.load()
    first = int(np.flatnonzero(results["state"].values == "Non-Local Continuous")[0])
    interior = interior_bin_mask(encoding["environment"])
    column = first + int(np.flatnonzero(interior)[real_bin])
    values = results["log_likelihood"].values.copy()
    values[:, column] = non_local[:, real_bin] + stored_offset
    values[inputs["is_missing"], column] = 0.0
    results["log_likelihood"] = (("time", "state_bins"), values)
    results.to_netcdf(results_path, engine="h5netcdf")


def underflowing_copy(
    simulated, tmp_path: Path, *, zero_offset: float = 0.0
) -> tuple[Path, np.ndarray, int]:
    """A copy of the decode with an unreached arm tip: three subnormal bins, one zero.

    As in chimi20200221_08_r4 (D1): the animal never reached the third arm's tip.
    The stored occupancy of its three bins before the last is a float32 subnormal.
    ``results.nc`` stays as the library wrote it, with ordinary numbers there (the
    GPU decode kept the subnormals and so was close to exact arithmetic), while a
    CPU rebuild flushes the subnormal to zero and gets the library's floor. The
    last bin has an occupancy of exactly 0 and a stored likelihood that agrees
    with the CPU rebuild (``give_exact_zero_occupancy``); it comes after the
    others, so that in the CPU rebuild, where all four are alike, the first of
    the three is the one an unrestricted choice takes.

    Returns the decode folder, the subnormal real bins (indices into the real
    place bins) and the exact-zero real bin.
    """
    assert not bool(jnp.asarray(np.float32(1e-42)) > 0), (
        "this test needs a CPU that flushes float32 subnormals to zero, as XLA's "
        "does (D1); the premise of the rule is gone"
    )
    decode_dir = decode_copy(simulated, tmp_path)
    detector = joblib.load(decode_dir / f"{EPOCH}_detector.pkl")
    encoding = detector.encoding_model_[ENCODING_KEY]
    environment = encoding["environment"]
    environment._is_fitted = True
    n_real = int(interior_bin_mask(environment).sum())
    zero_bin = n_real - 1
    zone = np.arange(zero_bin - ZONE_SIZE, zero_bin)
    # the animal shuttles on the first two arms: the third arm's tip is never visited
    animal = nearest_interior_bins(
        environment, pd.read_pickle(decode_dir / f"{EPOCH}_80_data_over_time.pkl")
        .linear_position.to_numpy(),
    )  # fmt: skip
    assert not np.isin(np.r_[zone, zero_bin], animal).any()
    occupancy = np.array(encoding["occupancy"], dtype=np.float32)
    occupancy[zone] = np.float32(1e-42)
    assert (occupancy[zone] < np.finfo(np.float32).tiny).all()
    encoding["occupancy"] = occupancy
    joblib.dump(detector, decode_dir / f"{EPOCH}_detector.pkl")
    give_exact_zero_occupancy(decode_dir, simulated[1], zero_bin, zero_offset)
    return decode_dir, zone, zero_bin


def test_bins_with_a_subnormal_occupancy_are_left_out_of_the_checks_and_the_evidence(
    simulated, tmp_path
):
    decode_dir, zone, zero_bin = underflowing_copy(simulated, tmp_path)

    record = run(decode_dir, tmp_path / "out", simulated[1])

    assert record["status"] == "ok", record.get("traceback")
    # recorded: the floor, the count and the bins; the exact-zero bin is not one
    assert record["underflowed_bins"] == {
        "floor": UNDERFLOW_OCCUPANCY_FLOOR, "n": ZONE_SIZE, "bins": zone.tolist(),
    }  # fmt: skip
    # the stored-likelihood check and the library check pass on the other bins,
    # the exact-zero bin among them
    assert record["stored_log_likelihood"]["max_rel"] < 1e-5
    for name in ("can", "run", "runcv0", "runcv1"):
        assert record["library_log_likelihood"][name]["max_rel"] < 1e-5
    # no encoding chose a left-out bin as the selecting half's best bin
    table = read_table(runner.output_paths(EPOCH, tmp_path / "out")["windows"])
    for prefix in ("can", "run", "runcv"):
        for column in ("run0_mapS", "run0_mapSr"):
            assert not table[f"{prefix}_{column}"].isin(zone).any(), (prefix, column)
    # the exact-zero bin stays a candidate: a silent window still chooses it
    assert (table.can_run0_mapS == zero_bin).any()
    # and the windows that would have chosen a left-out bin are marked and counted,
    # not dropped
    marked = table.can_mapS_left_out_runs + table.can_mapSr_left_out_runs
    assert (marked > 0).any()
    counted = record["windows"]["left_out_selection"]
    assert counted["can"]["windows"] == int((marked > 0).sum())
    assert counted["can"]["bouts"] == int(((marked > 0) & (table.kind == "bout")).sum())
    assert set(counted) == {"runcv", "run", "can"}
    # every window is still scored
    assert len(table) == sum(record["windows"]["n"].values())


def test_the_stored_likelihood_check_fails_as_before_when_the_bins_are_not_left_out(
    simulated, tmp_path, monkeypatch
):
    decode_dir, _, _ = underflowing_copy(simulated, tmp_path)
    monkeypatch.setattr(runner, "UNDERFLOW_OCCUPANCY_FLOOR", 0.0)

    record = run(decode_dir, tmp_path / "out", simulated[1])

    assert record["status"] == "failed"
    assert "do not reproduce results.nc" in record["error"]


def test_a_bin_with_an_exact_zero_occupancy_is_still_checked_against_the_stored_one(
    simulated, tmp_path
):
    """Both devices agree at an exact zero, so a disagreement there is a real one."""
    decode_dir, _, _ = underflowing_copy(simulated, tmp_path, zero_offset=50.0)

    record = run(decode_dir, tmp_path / "out", simulated[1])

    assert record["status"] == "failed"
    assert "do not reproduce results.nc" in record["error"]


def test_a_running_nonlocal_label_is_refused_before_anything_is_fit(
    simulated, tmp_path
):
    """The review's probe, a running non-local event at 10.2-10.3 s, breaks the
    decoder's own definition (every sampled speed below 4 cm/s; every bin above
    the probability threshold). No decode makes one, so the read refuses the
    frame as stale (analysed_time.with_swr_source) and nothing is fit on it."""
    decode_dir = decode_copy(simulated, tmp_path)
    path = decode_dir / f"{EPOCH}_80_data_over_time.pkl"
    dot = pd.read_pickle(path)
    offset = TIME - TIME[0]
    event = (offset >= 10.2) & (offset <= 10.3)
    assert (dot.head_speed.to_numpy()[event] >= 4.0).all()  # running
    dot.iloc[np.flatnonzero(event), dot.columns.get_loc("non_local_event")] = 1
    dot.to_pickle(path)
    with pytest.raises(EventDefinitionError):
        analysed_time.read_dot_with_swr_source(path)

    record = run(decode_dir, tmp_path / "out", simulated[1])

    assert record["status"] == "failed"
    assert "non_local_event labels violate" in record["error"]


def test_a_decode_without_the_sources_masks_fails_before_the_marks_load(
    simulated, tmp_path
):
    """A decode made before R11 (no decode_masks) is refused before load_data:
    the message says to rerun on the R11 decodes."""
    decode_dir = decode_copy(simulated, tmp_path)
    path = decode_dir / f"{EPOCH}_results.nc"
    with xr.open_dataset(path, engine="h5netcdf") as results:
        unstamped = results.load()
    unstamped.attrs = {}
    unstamped.to_netcdf(path, engine="h5netcdf")

    def load_inputs(_epoch_id):
        raise AssertionError("the marks were loaded")

    record = runner.run_epoch(
        EPOCH,
        decode_dir=decode_dir,
        output_dir=tmp_path / "out",
        load_inputs=load_inputs,
        log=lambda _message: None,
    )

    assert record["status"] == "failed"
    assert "no decode_masks" in record["error"] and "R11" in record["error"]
    assert "--overwrite" in record["error"]  # ... does not help


def test_the_runner_holds_jax_to_the_cpu(monkeypatch):
    """Importing it sets the CPU whatever the environment said; run_epoch checks."""
    import importlib

    monkeypatch.setenv("JAX_PLATFORMS", "cuda")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    importlib.reload(runner)

    assert os.environ["JAX_PLATFORMS"] == "cpu"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    runner.require_cpu()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    with pytest.raises(RuntimeError, match="CPU only"):
        runner.require_cpu()
    with pytest.raises(RuntimeError, match="CPU only"):
        runner.run_epoch(EPOCH, decode_dir=Path("."), output_dir=Path("."))


def test_points_without_an_interval_get_no_bar():
    estimate = np.array([1.0, 2.0, 3.0])
    has, (below, above) = runner._error_bars(
        estimate, np.array([0.5, np.nan, 2.0]), np.array([1.5, np.nan, 4.0])
    )

    assert has.tolist() == [True, False, True]
    assert below.tolist() == [0.5, 1.0] and above.tolist() == [0.5, 1.0]


def test_the_window_level_flag_is_off_by_default():
    parser = runner.build_parser()

    assert not parser.parse_args([]).window_level_stats
    assert parser.parse_args(["--window-level-stats"]).window_level_stats


def test_a_decode_missing_a_file_is_skipped_with_its_reason(simulated, tmp_path):
    decode_dir = decode_copy(simulated, tmp_path)
    (decode_dir / f"{EPOCH}_results.nc").unlink()

    record = run(decode_dir, tmp_path / "out", simulated[1])

    assert record["status"] == "skipped"
    assert record["reason"] == f"no results in {decode_dir}"
    # the files it found, and the code that looked
    assert set(record["decode_files"]) == {"data_over_time", "detector"}
    assert set(record["code"]) == {"commit", "dirty"}


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


def test_the_loader_builds_the_decodes_masks_from_the_swr_source(
    simulated, monkeypatch
):
    """As the decoder built them (hpc_decode_masks): the inventory's event and
    the artifact mask leave encoding, the mask is missing, and the stored
    ripples are not read."""
    import continuum_swr_replay.data_loaders as data_loaders

    inputs = simulated[1]
    stored_ripple = pd.DataFrame({"start_time": [TIME[4000]], "end_time": [TIME[4050]]})

    def load_data(nwb_file_name, epoch_name, include=None):
        assert (nwb_file_name, epoch_name) == ("aa20200101_.nwb", "02_r1")
        return {
            "position_info": pd.DataFrame(index=pd.Index(TIME, name="time")),
            "spike_times": {"HPC": inputs["spike_times"]},
            "spike_waveform_features": {"HPC": inputs["marks"]},
            "ripple_times": stored_ripple,
        }

    # Set in the package's namespace directly: ``monkeypatch.setattr`` reads the
    # old value first, and the lazy ``data_loaders.load_data`` would import
    # ``bandit_task`` for nothing.
    monkeypatch.setitem(vars(data_loaders), "load_data", load_data)

    loaded = runner.load_marks_and_training_mask(EPOCH)

    np.testing.assert_array_equal(loaded["time"], TIME)
    np.testing.assert_array_equal(loaded["is_training"], inputs["is_training"])
    np.testing.assert_array_equal(loaded["is_missing"], inputs["is_missing"])
    assert loaded["is_missing"].any() and loaded["is_training"][4000:4051].all()


def test_running_only_masks_refuse_an_event_spike_in_the_library_span():
    n = 200
    time = np.arange(n) * 0.002
    speed = np.full(n, 20.0)
    position = np.c_[np.linspace(0, 100, n), np.zeros(n)]
    folds = np.zeros(n, dtype=np.int8)
    folds[100:] = 1
    in_event = np.zeros(n, dtype=bool)
    in_event[50] = True

    none = np.zeros(n, dtype=bool)
    masks, record = runner.running_only_masks(
        time, speed, position, in_event, folds, is_missing=none
    )

    # the event bin and two bins either side are left out, and the padding
    # stops short of it
    assert not masks["run"][48:53].any() and masks["run"][47] and masks["run"][53]
    assert record["run"]["event_bins_in_library_span"] == 0
    assert record["running_bins_left_out_near_events"] == 5
    # no position around the event: the library joins the runs across it
    position[48:53] = np.nan
    with pytest.raises(ValueError, match="1 non-local event bins are in the run"):
        runner.running_only_masks(
            time, speed, position, in_event, folds, is_missing=none
        )
    # a fold with no running bin
    position[48:53] = 0.0
    with pytest.raises(ValueError, match="runcv1 encoding has no training bin"):
        runner.running_only_masks(
            time, speed, position, in_event, np.zeros(n, np.int8), is_missing=none
        )
    # the decode's missing bins are in no running-only fit
    missing = none.copy()
    missing[150:160] = True
    masks, record = runner.running_only_masks(
        time, speed, position, in_event, folds, is_missing=missing
    )
    assert not masks["run"][150:160].any() and masks["run"][149] and masks["run"][160]
    assert record["missing_bins_at_running_speed"] == 10


def test_float_rounding_of_the_bin_width_cannot_carry_the_span_into_an_event():
    """At epoch timestamps a bin can be an ulp narrower than the first one.

    The library pads a run by the first two bins' spacing, so with one bin of
    margin its span can reach a sliver into the event bin (one j16 pilot epoch
    did); the runner's margin of two bins cannot.
    """
    from continuum_swr_replay.analysis.hippocampus.state_evidence import (
        training_spike_bins,
    )

    n = 20_000
    time = 1625682923.263429 + np.arange(n) * 0.002
    speed = np.full(n, 20.0)
    position = np.c_[np.linspace(0, 100, n), np.zeros(n)]
    in_event = np.zeros(n, dtype=bool)
    in_event[np.arange(100, n - 100, 97)] = True
    one_bin = ~(in_event | np.r_[in_event[1:], False] | np.r_[False, in_event[:-1]])
    assert (in_event & training_spike_bins(time, position, one_bin)).any()

    _, record = runner.running_only_masks(
        time,
        speed,
        position,
        in_event,
        np.arange(n) // 5000 % 2,
        is_missing=np.zeros(n, dtype=bool),
    )

    for name in ("run", "runcv0", "runcv1"):
        assert record[name]["event_bins_in_library_span"] == 0


def test_every_analysed_hpc_decode_is_listed_even_an_incomplete_one(
    tmp_path, monkeypatch
):
    """Only epochs the register analyses: a pending epoch's decode in the folder
    is not one."""
    from continuum_swr_replay.data_loaders import epoch_register

    register = tmp_path / "epoch_register.csv"
    pd.DataFrame(
        [
            ["aa20200101_.nwb", "02_r1", "analysed", "", "", "False"],
            ["bb20200101_.nwb", "04_r2", "analysed", "", "", "False"],
            [
                "bb20200101_.nwb",
                "06_r3",
                "pending",
                "PENDING_POPULATE",
                "no PIM",
                "False",
            ],
        ],
        columns=list(epoch_register.COLUMNS),
    ).to_csv(register, index=False)
    monkeypatch.setattr(epoch_register, "REGISTER_PATH", register)
    decodes = tmp_path / "decodes"
    decodes.mkdir()
    for name in [
        "aa20200101_02_r1_HPC_detector.pkl",
        "aa20200101_02_r1_HPC_results.nc",
        "aa20200101_02_r1_HPC_80_data_over_time.pkl",
        "bb20200101_04_r2_HPC_detector.pkl",  # no results or data_over_time
        "bb20200101_06_r3_HPC_detector.pkl",  # pending
        "aa20200101_02_r1_HPC_contfrag_detector.pkl",  # another decoder's
        "aa20200101_02_r1_mPFC_detector.pkl",
    ]:
        (decodes / name).touch()

    assert runner.canonical_epochs(decodes) == [
        "aa20200101_02_r1_HPC",
        "bb20200101_04_r2_HPC",
    ]


# --- main(): reuse, provenance and outputs -----------------------------------------
def serial_epochs(inputs: dict):
    """``run_epochs`` in this process, with the stand-in loader.

    Dask's worker processes would not see a stand-in loader, so ``main`` runs
    its epochs here instead.
    """

    def run_epochs(epochs, kwargs, n_workers):
        for epoch_id in epochs:
            record = runner.run_epoch(
                epoch_id, load_inputs=lambda _e: inputs, log=lambda _m: None, **kwargs
            )
            yield epoch_id, record

    return run_epochs


def provenance(commit: str, dirty: bool = False):
    record = {
        "commit": commit,
        "dirty": dirty,
        "uv_lock_sha256": "0" * 64,
        "ripple_detection_commit": None,
        "non_local_detector_version": "test",
        "cuda_plugin_version": None,
    }
    return lambda _repo: record


def snapshot(directory: Path) -> dict:
    """Every file's size and mtime_ns, by name."""
    return {
        path.name: (path.stat().st_size, path.stat().st_mtime_ns)
        for path in sorted(directory.iterdir())
    }


def test_reuse_reason_names_what_changed():
    settings, code = {"seed": 1}, {"commit": "a" * 40, "dirty": False}
    stats = {"detector": {"path": "/decode/x.pkl", "size": 10, "mtime_ns": 20}}
    previous = {
        "status": "ok",
        "settings": settings,
        "code": code,
        "decode_files": stats,
    }

    def reason(earlier=previous, **change):
        current = {"settings": settings, "code": code, "decode_stats": stats,
                   "windows_written": True} | change  # fmt: skip
        return runner.reuse_reason(earlier, **current)

    assert reason() is None
    assert reason(None) == "not written before"
    assert reason({**previous, "status": "failed"}) == "no complete earlier output"
    assert reason(windows_written=False) == "no complete earlier output"
    assert reason(settings={"seed": 2}) == "the run settings differ"
    assert reason(code={"commit": "b" * 40, "dirty": False}) == (
        "code commit aaaaaaaaa -> bbbbbbbbb"
    )
    assert reason(code={**code, "dirty": True}).startswith("uncommitted changes")
    assert reason({**previous, "code": {**code, "dirty": True}}).startswith(
        "uncommitted changes"
    )
    for field, value in [("size", 11), ("mtime_ns", 21), ("path", "/elsewhere/x.pkl")]:
        changed = {"detector": {**stats["detector"], field: value}}
        assert reason(decode_stats=changed) == (
            "a decode file's path, size or mtime_ns differs"
        )


def _register_analysing(tmp_path: Path, monkeypatch, *keys: tuple[str, str]) -> None:
    """``REGISTER_PATH`` at a register that analyses these (nwb, epoch) keys."""
    from continuum_swr_replay.data_loaders import epoch_register

    path = tmp_path / "epoch_register.csv"
    pd.DataFrame(
        [[name, epoch, "analysed", "", "", "False"] for name, epoch in keys],
        columns=list(epoch_register.COLUMNS),
    ).to_csv(path, index=False)
    monkeypatch.setattr(epoch_register, "REGISTER_PATH", path)


def test_main_reuses_an_epoch_only_while_nothing_changed(
    simulated, tmp_path, monkeypatch
):
    _register_analysing(tmp_path, monkeypatch, ("aa20200101_.nwb", "02_r1"))
    decode_dir = decode_copy(simulated, tmp_path)
    output_dir = tmp_path / "out"
    monkeypatch.setattr(runner, "run_epochs", serial_epochs(simulated[1]))
    monkeypatch.setattr(runner, "provenance_record", provenance("a" * 40))
    argv = ["--decode-dir", str(decode_dir), "--output-dir", str(output_dir)]
    before = snapshot(decode_dir)

    runner.main(argv)

    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["recomputed"] == {EPOCH: "not written before"}
    assert manifest["reused"] == []
    record = manifest["epochs"][EPOCH]
    assert record["status"] == "ok"
    assert record["code"] == {"commit": "a" * 40, "dirty": False}
    detector = decode_dir / f"{EPOCH}_detector.pkl"
    assert record["decode_files"]["detector"] == {
        "path": str(detector.resolve()),
        "size": detector.stat().st_size,
        "mtime_ns": detector.stat().st_mtime_ns,
    }
    discovery = manifest["discovery"]
    assert discovery["n_epochs"] == 1 and discovery["n_with_all_files"] == 1
    assert "the audit's job" in discovery["coverage"]
    readme = (output_dir / "README.md").read_text()
    assert (
        "every epoch the epoch register analyses with a canonical HPC decode file"
    ) in readme
    assert "1 listed, 1 with all three decode files" in readme
    assert "Computed in this run: 1; reused" in readme
    assert (output_dir / "bout_stats.csv").exists()
    # the No-Spike cost at the decoder's own rate
    rate, bin_s = record["decode"]["no_spike_rate"], record["decode"]["bin_s"]
    assert (
        f"One spike costs {-np.log(rate * bin_s):.1f} nats (log({rate:g} Hz" in readme
    )
    assert "1e-10 Hz x 2 ms" not in "\n".join(runner.CAVEATS)

    # the same commit, settings and decode files: reused, not rewritten
    windows = runner.output_paths(EPOCH, output_dir)["windows"]
    written = windows.stat().st_mtime_ns
    runner.main(argv)
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["reused"] == [EPOCH] and manifest["recomputed"] == {}
    assert manifest["epochs"][EPOCH] == record
    assert windows.stat().st_mtime_ns == written
    assert (
        "reused from an earlier run with the same commit"
        in (output_dir / "README.md").read_text()
    )
    # two runs read the decode folder and changed nothing in it
    assert snapshot(decode_dir) == before

    # a decode file that changed: recomputed, and the reason logged
    stat = detector.stat()
    os.utime(detector, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
    runner.main(argv)
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["recomputed"] == {
        EPOCH: "a decode file's path, size or mtime_ns differs"
    }
    assert manifest["epochs"][EPOCH]["decode_files"]["detector"]["mtime_ns"] == (
        stat.st_mtime_ns + 10**9
    )
    assert windows.stat().st_mtime_ns != written


def test_main_exits_with_an_error_when_an_epoch_fails(simulated, tmp_path, monkeypatch):
    decode_dir = decode_copy(simulated, tmp_path)
    output_dir = tmp_path / "out"
    inputs = dict(simulated[1])
    inputs["spike_times"] = [st[1:] for st in inputs["spike_times"]]
    inputs["marks"] = [mk[1:] for mk in inputs["marks"]]
    monkeypatch.setattr(runner, "run_epochs", serial_epochs(inputs))
    monkeypatch.setattr(runner, "provenance_record", provenance("a" * 40))

    with pytest.raises(SystemExit) as exit_:
        runner.main(["--epochs", EPOCH, "--decode-dir", str(decode_dir),
                     "--output-dir", str(output_dir)])  # fmt: skip

    assert exit_.value.code == f"1 epochs failed: {EPOCH}"
    manifest = json.loads((output_dir / "manifest.json").read_text())
    record = manifest["epochs"][EPOCH]
    assert record["status"] == "failed" and record["code"]["commit"] == "a" * 40
    assert manifest["discovery"]["rule"] == "the epochs given with --epochs"
    readme = (output_dir / "README.md").read_text()
    assert f"| {EPOCH} | failed |" in readme
    assert "does not reproduce the stored encoding model" in readme


# --- open files: Spyglass's NWB cache and the open-file limit ------------------------
@pytest.mark.parametrize("outcome", ["ok", "raises"])
def test_each_epoch_task_closes_the_nwb_files_spyglass_opened(
    tmp_path, monkeypatch, outcome
):
    """A worker runs one epoch at a time, so when its epoch ends, pass or fail,
    it closes the NWB files ``load_data`` opened through Spyglass and logs that
    it did."""
    closed = []

    def release_nwb_files():
        closed.append(True)
        return True

    def run_epoch(epoch_id, **kwargs):
        if outcome == "raises":
            raise RuntimeError("simulated: the task itself failed")
        return {"status": "ok"}

    monkeypatch.setattr(runner, "run_epoch", run_epoch)
    monkeypatch.setattr(batch, "release_nwb_files", release_nwb_files)

    def task():
        return runner._run_epoch_logged(EPOCH, {}, tmp_path / "logs")

    if outcome == "raises":
        with pytest.raises(RuntimeError, match="the task itself failed"):
            task()
    else:
        assert task() == {"status": "ok"}

    assert closed == [True]
    log = (tmp_path / "logs" / f"{EPOCH}.log").read_text().splitlines()
    assert any(
        line.endswith("NWB files Spyglass held open closed: True") for line in log
    )


class _WorkersStarted(Exception):
    """Raised where the run would start its workers."""


def test_main_raises_the_open_file_limit_before_its_workers_start(
    tmp_path, monkeypatch, capsys
):
    """The worker processes inherit the limit only if it is raised before they
    start; the run says what it was raised from and to."""
    events = []

    def raise_open_file_limit():
        events.append("limit")
        return 1024, 65536

    def run_epochs(epochs, kwargs, n_workers):
        events.append("workers")
        raise _WorkersStarted

    monkeypatch.setattr(runner, "raise_open_file_limit", raise_open_file_limit)
    monkeypatch.setattr(runner, "run_epochs", run_epochs)
    monkeypatch.setattr(runner, "provenance_record", provenance("a" * 40))
    (tmp_path / "decode").mkdir()

    with pytest.raises(_WorkersStarted):
        runner.main(["--epochs", EPOCH, "--decode-dir", str(tmp_path / "decode"),
                     "--output-dir", str(tmp_path / "out")])  # fmt: skip

    assert events == ["limit", "workers"]
    assert "open-file soft limit 1024 -> 65536" in capsys.readouterr().out
