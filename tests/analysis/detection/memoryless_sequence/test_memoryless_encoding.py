"""The memoryless detector's encoding rule: the paper's running-only refit, or
the decoder's saved model.

A real ``NonLocalClusterlessDetector`` is fitted on a synthetic epoch in which
the animal runs over 0-60 cm for two seconds, then sits still at 90 cm, so a
spike's encoding position says whether a slow sample entered the fit.
"""

import joblib
import numpy as np
import pandas as pd
import pytest
from non_local_detector import NonLocalClusterlessDetector
from non_local_detector.environment import Environment

from continuum_swr_replay.analysis.detection.memoryless_sequence import encoding
from continuum_swr_replay.analysis.detection.memoryless_sequence.encoding import (
    DECODER,
    PAPER,
    POSITION_STD_CM,
    RUNNING_SPEED_CM_PER_S,
    encoding_settings,
    epoch_encoding_model,
    run_encoding,
    running_encoding_model,
    running_training_mask,
)
from continuum_swr_replay.decoding.encoding_model import load_clusterless_encoding_model
from continuum_swr_replay.decoding.identifiers import parse_epoch_identifier
from continuum_swr_replay.decoding.results import ENCODING_KEY

SAMPLING_FREQUENCY = 500.0
N_TIME = 2000
RUN_END_S = 2.0
STILL_AT_CM = 90.0
DECODER_POSITION_STD_CM = 3.5
PADDING_CM = 60.0 / (RUN_END_S * SAMPLING_FREQUENCY)
"""How far past the run one padded bin reaches: the library keeps each training
run's spikes from one bin before it to one bin after."""


def _epoch():
    time = np.arange(N_TIME) / SAMPLING_FREQUENCY
    running = time < RUN_END_S
    position = np.where(running, 60.0 * time / RUN_END_S, STILL_AT_CM)
    speed = np.where(running, 30.0, 0.0)
    rng = np.random.default_rng(0)
    spike_times = np.sort(rng.uniform(0.0, time[-1], 3000))
    marks = rng.normal(100.0, 10.0, size=(spike_times.size, 2))
    return time, position, speed, [spike_times], [marks]


@pytest.fixture(scope="module")
def fitted():
    """The decoder: every sample trains, at its own kernel."""
    time, position, speed, spike_times, marks = _epoch()
    detector = NonLocalClusterlessDetector(
        environments=[Environment(place_bin_size=4.0)],
        clusterless_algorithm="clusterless_kde",
        clusterless_algorithm_params={
            "position_std": DECODER_POSITION_STD_CM,
            "waveform_std": 24.0,
            "block_size": 256,
        },
    )
    detector.initialize_environments(position[:, None])
    detector.fit_encoding_model(
        time, position[:, None], spike_times, marks, is_training=np.ones(N_TIME, bool)
    )
    return detector, time, position, speed, spike_times, marks


def test_the_running_mask_keeps_only_fast_training_samples():
    is_training = np.array([True, True, True, True, False, True])
    speed = np.array([0.0, 10.0, 10.5, 30.0, 30.0, np.nan])

    mask = running_training_mask(is_training, speed)

    # 10 cm/s is not running (strictly above); a NaN speed is not running;
    # a fast bin the decoder left out stays out
    assert mask.tolist() == [False, False, True, True, False, False]


def test_the_running_mask_refuses_misaligned_inputs():
    with pytest.raises(ValueError, match="shape"):
        running_training_mask(np.ones(3, bool), np.ones(4))


def test_the_refit_is_trained_on_running_samples_only(fitted):
    detector, time, position, speed, spike_times, marks = fitted
    decoder_positions = np.asarray(
        detector.encoding_model_[ENCODING_KEY]["encoding_positions"][0]
    )

    model, record = running_encoding_model(
        detector,
        time,
        position[:, None],
        speed,
        spike_times,
        marks,
        np.ones(N_TIME, bool),
    )

    positions = np.asarray(model["encoding_positions"][0])
    # the decoder's encoding holds the still animal; the refit's does not
    assert np.any(decoder_positions == STILL_AT_CM)
    assert positions.size > 0
    assert positions.max() <= 60.0 + PADDING_CM
    assert record["n_training_bins"] == int(np.count_nonzero(time < RUN_END_S))
    assert record["n_decoder_training_bins"] == N_TIME
    assert record["training_s"] == pytest.approx(RUN_END_S)


def test_the_refit_reaches_the_paper_kernel_and_leaves_the_detector(fitted):
    detector, time, position, speed, spike_times, marks = fitted
    decoder_model = detector.encoding_model_[ENCODING_KEY]

    model, _ = running_encoding_model(
        detector,
        time,
        position[:, None],
        speed,
        spike_times,
        marks,
        np.ones(N_TIME, bool),
    )

    assert np.asarray(model["position_std"]).ravel().tolist() == [POSITION_STD_CM]
    assert model["waveform_std"] == 24.0
    # the saved detector is untouched: its model, settings and environment
    assert detector.encoding_model_[ENCODING_KEY] is decoder_model
    assert detector.clusterless_algorithm_params["position_std"] == (
        DECODER_POSITION_STD_CM
    )
    assert model["environment"] is decoder_model["environment"]


def test_the_refit_refuses_an_epoch_with_no_running_training_bin(fitted):
    detector, time, position, _, spike_times, marks = fitted

    with pytest.raises(ValueError, match="No training bin"):
        running_encoding_model(
            detector,
            time,
            position[:, None],
            np.zeros(N_TIME),
            spike_times,
            marks,
            np.ones(N_TIME, bool),
        )


@pytest.fixture
def saved(fitted, tmp_path):
    """The fitted detector saved as a decode saves it, and its frame."""
    detector, time, position, speed, spike_times, marks = fitted
    path = tmp_path / "epoch_detector.pkl"
    joblib.dump(detector, path)
    frame = pd.DataFrame(
        {
            "linear_position": position,
            "head_speed": speed,
            "head_position_x": position,
            "head_position_y": np.zeros(N_TIME),
        },
        index=pd.Index(time, name="time"),
    )
    return path, frame, spike_times, marks


KEY = parse_epoch_identifier("sim20200101_02_r1_HPC")
NO_ARTIFACTS = pd.DataFrame({"start_time": [], "end_time": []})


def test_the_decoder_rule_returns_the_saved_model_unchanged(saved, monkeypatch):
    path, frame, spike_times, marks = saved

    def no_masks(*args, **kwargs):
        raise AssertionError("the decoder's rule builds no training mask")

    monkeypatch.setattr(encoding, "hpc_decode_masks", no_masks)

    model, record = epoch_encoding_model(
        DECODER, path, frame, spike_times, marks, KEY, NO_ARTIFACTS
    )

    stored = load_clusterless_encoding_model(path)
    assert set(model) == set(stored)
    for name in ("encoding_positions", "mean_rates", "summed_ground_process_intensity"):
        assert all(
            np.array_equal(a, b)
            for a, b in zip(
                np.atleast_1d(model[name]), np.atleast_1d(stored[name]), strict=True
            )
        )
    assert model["position_std"] == stored["position_std"]
    assert record["rule"] == DECODER
    assert record["n_encoding_spikes"] == [spike_times[0].size]


def test_the_paper_rule_narrows_the_decoders_own_mask(saved, monkeypatch):
    """The decoder's training mask, rebuilt for the epoch's training type, is
    narrowed to running; a fast bin it left out stays out."""
    path, frame, spike_times, marks = saved
    decoder_mask = np.ones(N_TIME, bool)
    decoder_mask[:100] = False  # an SWR the decoder left out, while running
    asked = {}

    class Masks:
        is_training = decoder_mask

    def masks(nwb_file_name, epoch_name, position_info, artifact_mask, training_type):
        asked.update(nwb=nwb_file_name, epoch=epoch_name, training_type=training_type)
        assert position_info.index.equals(frame.index)
        return Masks()

    monkeypatch.setattr(encoding, "hpc_decode_masks", masks)

    model, record = epoch_encoding_model(
        PAPER, path, frame, spike_times, marks, KEY, NO_ARTIFACTS
    )

    assert asked == {
        "nwb": KEY.nwb_file_name,
        "epoch": KEY.epoch_name,
        "training_type": KEY.training_type,
    }
    assert record["n_training_bins"] == int(
        np.count_nonzero(decoder_mask & (frame.index < RUN_END_S))
    )
    assert record["rule"] == PAPER
    assert np.asarray(model["encoding_positions"][0]).max() <= 60.0 + PADDING_CM
    assert record["n_encoding_spikes"] == [len(model["encoding_positions"][0])]


def test_marks_that_are_not_the_decoders_are_refused(saved):
    path, frame, spike_times, marks = saved

    with pytest.raises(ValueError, match="not among the reloaded marks"):
        epoch_encoding_model(
            DECODER, path, frame, spike_times, [marks[0] + 1.0], KEY, NO_ARTIFACTS
        )


def test_an_unknown_rule_is_refused():
    with pytest.raises(ValueError, match="encoding must be one of"):
        encoding_settings("running")


def test_the_papers_settings_are_recorded():
    settings = encoding_settings(PAPER)

    assert settings["running_speed_cm_per_s"] == RUNNING_SPEED_CM_PER_S == 10.0
    assert settings["position_std_cm"] == POSITION_STD_CM == 8.0


def test_a_run_records_its_rule_and_an_older_run_read_the_decoders():
    assert run_encoding({"encoding": encoding_settings(PAPER)}) == PAPER
    assert run_encoding({"encoding": encoding_settings(DECODER)}) == DECODER
    assert run_encoding({}) == DECODER
    with pytest.raises(ValueError, match="not one of"):
        run_encoding({"encoding": {"rule": "other"}})
