"""A saved decode's fitted clusterless encoding model, read back for reuse.

Every step that scores spikes against the decoder's place fields again (the
memoryless detector's control, the Carey detector, the event features, the
state evidence) starts from the encoding model the decode fitted and saved in
``{epoch}_detector.pkl``. This module reads it
(:func:`load_clusterless_encoding_model`), names the entries the library's
likelihood takes (:data:`CLUSTERLESS_LIKELIHOOD_KEYS`), refuses reloaded marks
that are not the ones it was fitted on (:func:`check_marks_match_encoding`),
and gives the track-graph distance between its place bins
(:func:`interior_bin_distances_cm`).
"""

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from continuum_swr_replay.decoding.results import ENCODING_KEY, load_saved_detector

__all__ = [
    "CLUSTERLESS_LIKELIHOOD_KEYS",
    "check_marks_match_encoding",
    "clusterless_encoding_model",
    "interior_bin_distances_cm",
    "load_clusterless_encoding_model",
]

CLUSTERLESS_LIKELIHOOD_KEYS = (
    "occupancy",
    "occupancy_model",
    "gpi_models",
    "encoding_spike_waveform_features",
    "encoding_positions",
    "environment",
    "mean_rates",
    "summed_ground_process_intensity",
    "position_std",
    "waveform_std",
)
"""The entries of a saved clusterless encoding model that its likelihood takes."""


def load_clusterless_encoding_model(detector_path: str | Path) -> dict:
    """Read the fitted encoding model out of a saved detector.

    Parameters
    ----------
    detector_path : str or Path
        The decode's ``{epoch}_detector.pkl``.

    Returns
    -------
    encoding_model : dict
        As :func:`clusterless_encoding_model`.

    Raises
    ------
    FileNotFoundError, KeyError
        As ``decoding.results.load_saved_detector``.
    ValueError
        The environment has no place grid
        (``decoding.results.load_saved_detector``), or as
        :func:`clusterless_encoding_model`.

    """
    return clusterless_encoding_model(load_saved_detector(detector_path))


def clusterless_encoding_model(detector: Any) -> dict:
    """A clusterless detector's fitted encoding model, as the likelihood takes it.

    Parameters
    ----------
    detector
        A fitted clusterless detector, such as ``load_saved_detector`` returns.

    Returns
    -------
    encoding_model : dict
        The entries in :data:`CLUSTERLESS_LIKELIHOOD_KEYS`, plus whatever else
        the library stored. ``block_size`` is passed through to the decode
        and the shuffle's re-decode; it sets how many spikes are evaluated at
        once and does not change the result.

    Raises
    ------
    ValueError
        ``waveform_std`` differs between features, which the per-electrode
        expansion below could not honor.

    """
    encoding_model = dict(detector.encoding_model_[ENCODING_KEY])

    # electrodes can differ in how many waveform features they have, so a
    # per-feature vector cannot be shared between them; the library expands a
    # scalar to each electrode's own feature count. That is only the same
    # model when every feature had the same bandwidth to begin with.
    waveform_std = np.asarray(encoding_model["waveform_std"], dtype=np.float64).ravel()
    if not np.allclose(waveform_std, waveform_std[0]):
        raise ValueError(
            f"waveform_std differs between features ({waveform_std.tolist()}); "
            "a single bandwidth cannot stand in for them."
        )
    encoding_model["waveform_std"] = float(waveform_std[0])
    return encoding_model


def check_marks_match_encoding(
    encoding_model: dict,
    spike_times: list[NDArray],
    spike_waveform_features: list[NDArray],
) -> None:
    """Refuse marks that are not the ones the encoding model was fitted on.

    Every encoding mark must be one of the reloaded marks of the same
    electrode. The encoding set is the training span's spikes, so it is a
    subset rather than the whole; the reloaded marks are cast to the model's
    dtype first, as the fit cast them.

    Raises
    ------
    ValueError
        A different number of electrodes, of features on an electrode, or of
        spike times and marks on one; or an encoding mark missing from its
        electrode's reloaded marks, which means the spike loader changed since
        the decode or the electrodes arrive in another order.
    """
    encoded_marks = encoding_model["encoding_spike_waveform_features"]
    if not len(spike_times) == len(spike_waveform_features) == len(encoded_marks):
        raise ValueError(
            f"{len(spike_times)} spike trains and {len(spike_waveform_features)} "
            f"mark arrays for {len(encoded_marks)} encoding electrodes."
        )
    for electrode, (encoded, marks, times) in enumerate(
        zip(encoded_marks, spike_waveform_features, spike_times, strict=True)
    ):
        encoded = np.asarray(encoded)
        marks = np.asarray(marks)
        if marks.ndim != 2 or marks.shape[1] != encoded.shape[1]:
            raise ValueError(
                f"Electrode {electrode} has marks of shape {marks.shape}; the "
                f"encoding model has {encoded.shape[1]} features there."
            )
        if len(times) != marks.shape[0]:
            raise ValueError(
                f"Electrode {electrode} has {len(times)} spike times for "
                f"{marks.shape[0]} marks."
            )
        row = np.dtype((np.void, encoded.dtype.itemsize * encoded.shape[1]))
        reloaded = np.ascontiguousarray(marks.astype(encoded.dtype)).view(row).ravel()
        fitted = np.ascontiguousarray(encoded).view(row).ravel()
        missing = int(np.count_nonzero(~np.isin(fitted, reloaded)))
        if missing:
            raise ValueError(
                f"Electrode {electrode}: {missing} of its {fitted.size} encoding "
                "marks are not among the reloaded marks. The spike loader has "
                "changed since the decode, or the electrodes are in another order."
            )


def interior_bin_distances_cm(environment: Any) -> NDArray:
    """Track-graph distance between every pair of real place bins.

    Two bins adjacent in linearized coordinate can sit on different arms, so
    the detector measures jumps and dispersion on the graph.

    Parameters
    ----------
    environment : non_local_detector.environment.Environment
        Fitted, with its place grid built.

    Returns
    -------
    distance_cm : ndarray, shape (n_interior_bins, n_interior_bins)

    """
    interior = np.asarray(environment.is_track_interior_).ravel()
    centers = np.asarray(environment.place_bin_centers_).ravel()[interior]
    distance: NDArray = np.asarray(
        environment.get_distances_to_interior_bins(centers[:, None]),
        dtype=np.float64,
    )
    return distance
