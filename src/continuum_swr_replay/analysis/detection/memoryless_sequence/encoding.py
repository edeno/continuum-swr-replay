"""The memoryless detector's encoding model: the paper's rule, or the decoder's.

Widloski & Foster (2025) fit their rate maps on running periods only (speed
above 10 cm/s) and smooth them with an 8 cm Gaussian (Methods; the
detector's design note, "The original method", step 1). The non-local
decoder's encoding model is trained on all time outside SWR events and the
artifact mask, at any speed, with a 3.5 cm position kernel. By default the
detector uses the paper's rule (:data:`PAPER`): the decoder's saved detector
refitted, with the same spikes and marks, on the decoder's own training bins
that are also running, and with the paper's kernel (:func:`running_encoding_model`).
The decoder's own model stays selectable (:data:`DECODER`), the shared-input
control in which both procedures read one encoding model.

What remains different from the paper: the marks are clusterless (waveform
features), not sorted cells; training still leaves out SWR events and the
artifact mask, as the decoder's does; and the kernel smooths in the
encoding's own position coordinate (the linearized track), not the paper's
open-field 2D.
"""

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from continuum_swr_replay.analysis.detection.memoryless_sequence.likelihood import (
    check_marks_match_encoding,
    clusterless_encoding_model,
)
from continuum_swr_replay.analysis.event_definition.decode_masks import (
    hpc_decode_masks,
)
from continuum_swr_replay.decoding.identifiers import EpochKey
from continuum_swr_replay.decoding.results import load_saved_detector

__all__ = [
    "DECODER",
    "ENCODINGS",
    "PAPER",
    "POSITION_STD_CM",
    "RUNNING_SPEED_CM_PER_S",
    "encoding_settings",
    "epoch_encoding_model",
    "run_encoding",
    "running_encoding_model",
    "running_training_mask",
]

PAPER = "paper"
"""Encoding rule: the paper's, running-only with an 8 cm kernel. The default."""
DECODER = "decoder"
"""Encoding rule: the non-local decoder's saved model, unchanged."""
ENCODINGS = (PAPER, DECODER)

RUNNING_SPEED_CM_PER_S = 10.0
"""Training bins must be faster than this, cm/s: the paper's running periods
(Widloski & Foster 2025, Methods, "> 10 cm/s")."""
POSITION_STD_CM = 8.0
"""The position kernel's SD, cm: the paper's 8 cm Gaussian smoothing (Widloski &
Foster 2025, Methods; the vendored Mallory code gives 4 cm)."""

POSITION_COLUMNS = ["head_position_x", "head_position_y"]
"""The position the decode's encoding was fitted on, from ``data_over_time``.
Fitting on ``linear_position`` instead runs, but gives a near-zero ground
term: the detector's environment linearizes 2D position itself."""


def encoding_settings(encoding: str) -> dict[str, Any]:
    """What a run records of its encoding rule.

    Raises
    ------
    ValueError
        `encoding` is not one of :data:`ENCODINGS`.
    """
    if encoding == PAPER:
        return {
            "rule": PAPER,
            "training": "the decoder's training bins (outside SWR events and the "
            "artifact mask) with head speed above running_speed_cm_per_s",
            "running_speed_cm_per_s": RUNNING_SPEED_CM_PER_S,
            "position_std_cm": POSITION_STD_CM,
            "source": "Widloski & Foster (2025), Methods",
        }
    if encoding == DECODER:
        return {
            "rule": DECODER,
            "training": "the non-local decoder's saved encoding model, unchanged",
        }
    raise ValueError(f"encoding must be one of {ENCODINGS}, got {encoding!r}.")


def run_encoding(settings: dict) -> str:
    """The encoding rule a memoryless run used, from its manifest's settings.

    A run made before the rule was recorded (before 2026-11-13) has no
    ``encoding`` entry; every such run read the decoder's model.

    Raises
    ------
    ValueError
        The recorded rule is not one of :data:`ENCODINGS`.
    """
    rule = settings.get("encoding", {"rule": DECODER})["rule"]
    if rule not in ENCODINGS:
        raise ValueError(f"The run's encoding rule {rule!r} is not one of {ENCODINGS}.")
    return str(rule)


def running_training_mask(
    is_training: NDArray,
    speed: NDArray,
    running_speed_cm_per_s: float = RUNNING_SPEED_CM_PER_S,
) -> NDArray[np.bool_]:
    """The decoder's training bins at which the animal runs.

    Parameters
    ----------
    is_training : ndarray of bool, shape (n_time,)
        The decoder's training mask on the decode grid.
    speed : ndarray, shape (n_time,)
        Head speed, cm/s. A NaN speed is not running.
    running_speed_cm_per_s : float, optional

    Returns
    -------
    ndarray of bool, shape (n_time,)
    """
    is_training = np.asarray(is_training, dtype=bool)
    speed = np.asarray(speed, dtype=np.float64)
    if is_training.shape != speed.shape:
        raise ValueError(
            f"is_training has shape {is_training.shape}, speed {speed.shape}."
        )
    with np.errstate(invalid="ignore"):
        return is_training & (speed > running_speed_cm_per_s)


def running_encoding_model(
    detector: Any,
    time: NDArray,
    position: NDArray,
    speed: NDArray,
    spike_times: list[NDArray],
    spike_waveform_features: list[NDArray],
    is_training: NDArray,
    *,
    running_speed_cm_per_s: float = RUNNING_SPEED_CM_PER_S,
    position_std_cm: float = POSITION_STD_CM,
) -> tuple[dict, dict[str, Any]]:
    """The decoder's detector refitted on its running training bins.

    The refit changes only the training bins and the position kernel: the
    environment, the waveform kernel and the library's other settings are the
    saved detector's. `detector` itself is not changed.

    Parameters
    ----------
    detector
        The decode's saved clusterless detector (``load_saved_detector``).
    time : ndarray, shape (n_time,)
        The decode grid, s.
    position : ndarray, shape (n_time, n_position_dims)
        The position the decode was fitted on (:data:`POSITION_COLUMNS`).
    speed : ndarray, shape (n_time,)
        Head speed, cm/s.
    spike_times, spike_waveform_features : list of ndarray
        The decode's marks, per electrode.
    is_training : ndarray of bool, shape (n_time,)
        The decoder's training mask (``hpc_decode_masks(...).is_training``).
    running_speed_cm_per_s, position_std_cm : float, optional

    Returns
    -------
    encoding_model : dict
        As :func:`~.likelihood.clusterless_encoding_model` returns it.
    record : dict
        The training seconds and bins, before and after the speed gate.

    Raises
    ------
    ValueError
        No training bin is running.
    """
    time = np.asarray(time, dtype=np.float64)
    is_running = running_training_mask(is_training, speed, running_speed_cm_per_s)
    if not is_running.any():
        raise ValueError(
            f"No training bin has head speed above {running_speed_cm_per_s} cm/s."
        )
    refit = copy.copy(detector)  # fit_encoding_model replaces encoding_model_
    refit.clusterless_algorithm_params = {
        **(detector.clusterless_algorithm_params or {}),
        "position_std": position_std_cm,
    }
    refit.fit_encoding_model(
        time,
        np.asarray(position, dtype=np.float64),
        spike_times,
        spike_waveform_features,
        is_training=is_running,
    )
    bin_s = float(np.median(np.diff(time)))
    record = {
        "n_training_bins": int(is_running.sum()),
        "training_s": float(is_running.sum() * bin_s),
        "n_decoder_training_bins": int(np.count_nonzero(is_training)),
        "decoder_training_s": float(np.count_nonzero(is_training) * bin_s),
    }
    return clusterless_encoding_model(refit), record


def epoch_encoding_model(
    encoding: str,
    detector_path: str | Path,
    data_over_time: pd.DataFrame,
    spike_times: list[NDArray],
    spike_waveform_features: list[NDArray],
    key: EpochKey,
    artifact_mask: pd.DataFrame,
) -> tuple[dict, dict[str, Any]]:
    """One epoch's encoding model for the memoryless detector, by rule.

    The marks are first checked against the decoder's model
    (``check_marks_match_encoding``), whichever rule is asked for: they must be
    the decode's. For :data:`PAPER` the decoder's training mask is rebuilt as
    the decoder built it (``hpc_decode_masks``, the epoch's training type) and
    narrowed to running (:func:`running_encoding_model`).

    Parameters
    ----------
    encoding : str
        One of :data:`ENCODINGS`.
    detector_path : str or Path
        The decode's ``{epoch_id}_detector.pkl``.
    data_over_time : DataFrame
        The decode's frame, indexed by the decode grid; ``linear_position``,
        ``head_speed`` and :data:`POSITION_COLUMNS` are read.
    spike_times, spike_waveform_features : list of ndarray
    key : EpochKey
        The decode's identifier, parsed.
    artifact_mask : DataFrame
        The epoch's artifact intervals (``swr_source.artifact_intervals``).

    Returns
    -------
    encoding_model : dict
    record : dict
        The rule, the encoding spikes per electrode and, for :data:`PAPER`,
        :func:`running_encoding_model`'s record. The rule's settings are the
        run's (:func:`encoding_settings`).
    """
    encoding_settings(encoding)  # refuses an unknown rule before any loading
    detector = load_saved_detector(detector_path)
    decoder_model = clusterless_encoding_model(detector)
    check_marks_match_encoding(decoder_model, spike_times, spike_waveform_features)
    if encoding == DECODER:
        model, record = decoder_model, {"rule": DECODER}
    else:
        masks = hpc_decode_masks(
            key.nwb_file_name,
            key.epoch_name,
            data_over_time[["linear_position"]],
            artifact_mask,
            key.training_type,
        )
        model, fit_record = running_encoding_model(
            detector,
            data_over_time.index.to_numpy(dtype=np.float64),
            data_over_time[POSITION_COLUMNS].to_numpy(dtype=np.float64),
            data_over_time["head_speed"].to_numpy(dtype=np.float64),
            spike_times,
            spike_waveform_features,
            masks.is_training,
        )
        record = {"rule": PAPER} | fit_record
    record["n_encoding_spikes"] = [
        int(np.asarray(p).shape[0]) for p in model["encoding_positions"]
    ]
    return model, record
