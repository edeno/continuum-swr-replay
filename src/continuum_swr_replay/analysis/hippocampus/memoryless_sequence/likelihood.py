"""Prior-excluded log-likelihood on the native decoding grid.

The detector decodes each window on its own under a uniform spatial prior, so
what it needs from the encoding model is the spike log-likelihood per time
sample and place bin, with no transition or state prior mixed in. That is what
``predict_clusterless_kde_log_likelihood`` returns for ``is_local=False``, and
it is additive over time: a window's log-likelihood is the sum of its rows.
:func:`~...statistics.aggregate_windows` relies on that, so the additivity is
tested against a direct marked-process sum rather than assumed.

**The final row.** ``get_spike_time_bin_ind`` assigns spikes with
``np.digitize(spike_times, time[1:-1])``, whose largest index is ``n_time - 2``.
The last row therefore carries a ground-process contribution that no spike can
ever offset. Summing a window that contains it would charge 80 ms of exposure
against fewer than 80 ms of spikes, so :func:`clusterless_epoch_log_likelihood`
and :func:`sorted_spikes_epoch_log_likelihood` both drop it.
What remains is exact: row ``k`` holds the spikes in ``[time[k], time[k + 1])``,
and the last surviving row also absorbs a spike landing exactly on the epoch's
final timestamp.

**Junction bins.** The library returns only ``is_track_interior_`` columns, and
the ``edge_id == -1`` placeholders are built from the non-interior bins, so they
are excluded here by construction. Reading a saved decode's
``log_likelihood`` instead would carry every ``state_bins`` column and would
need the interior mask applied explicitly; that path is deliberately not
implemented until it can be checked against a real file.

The design, including every departure from the original, is in
``docs/design/replay_detection_comparison/02-memoryless-sequence.md``.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
from numpy.typing import NDArray

from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.config import (
    MemorylessSequenceConfig,
)

__all__ = [
    "CLUSTERLESS_LIKELIHOOD_KEYS",
    "ENCODING_KEY",
    "SORTED_SPIKES_LIKELIHOOD_KEYS",
    "EpochLikelihood",
    "check_marks_match_encoding",
    "clusterless_epoch_log_likelihood",
    "interior_bin_distances_cm",
    "load_clusterless_encoding_model",
    "sorted_spikes_epoch_log_likelihood",
    "validate_time_grid",
    "window_midpoints_s",
]

ENCODING_KEY = ("", 0)
"""The encoding group a single-environment decode is saved under."""

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

SORTED_SPIKES_LIKELIHOOD_KEYS = (
    "environment",
    "marginal_models",
    "occupancy_model",
    "occupancy",
    "mean_rates",
    "place_fields",
    "no_spike_part_log_likelihood",
    "is_track_interior",
)
"""The entries of a sorted-spikes encoding model that its likelihood takes."""

_SAMPLE_TOLERANCE = 1e-3
"""How far, in samples, a timestamp may sit from its place on a uniform grid.

Checked in samples rather than as a relative spread of the spacing, because a
recording's clock counts seconds since 1970: near 1.6e9 s a float64 step is
2.4e-7 s, so the 2 ms spacings come out as 1.99986 or 2.00009 ms, a spread
no relative tolerance on the spacing could accept without also accepting a
real error. The same rounding moves a timestamp by 1.2e-4 of a sample at most.
"""


@dataclass(frozen=True)
class EpochLikelihood:
    """One epoch's spike log-likelihood over real place bins.

    Attributes
    ----------
    time : ndarray, shape (n_time,)
        Sample times, already trimmed to the rows a spike can reach.
    log_likelihood : ndarray, shape (n_time, n_interior_bins)
    place_bin_centers_cm : ndarray, shape (n_interior_bins,)
        Linearized center of each column's bin.

    """

    time: NDArray
    log_likelihood: NDArray
    place_bin_centers_cm: NDArray

    def __post_init__(self) -> None:
        time = np.asarray(self.time, dtype=np.float64)
        log_likelihood = np.asarray(self.log_likelihood)
        centers = np.asarray(self.place_bin_centers_cm, dtype=np.float64)
        if time.ndim != 1 or log_likelihood.ndim != 2 or centers.ndim != 1:
            raise ValueError(
                f"time {time.shape}, log_likelihood {log_likelihood.shape} and "
                f"place_bin_centers_cm {centers.shape} must be 1-D, 2-D and 1-D."
            )
        if log_likelihood.shape != (time.size, centers.size):
            raise ValueError(
                f"log_likelihood is {log_likelihood.shape} but there are "
                f"{time.size} times and {centers.size} bins."
            )
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "log_likelihood", log_likelihood)
        object.__setattr__(self, "place_bin_centers_cm", centers)


def load_clusterless_encoding_model(detector_path: str | Path) -> dict:
    """Read the fitted encoding model out of a saved detector.

    Parameters
    ----------
    detector_path : str or Path
        The decode's ``{epoch}_detector.pkl``.

    Returns
    -------
    encoding_model : dict
        The entries in :data:`CLUSTERLESS_LIKELIHOOD_KEYS`, plus whatever else
        the library stored. ``block_size`` is passed through to the decode
        and the shuffle's re-decode; it sets how many spikes are evaluated at
        once and does not change the result.

    Raises
    ------
    FileNotFoundError
        There is no detector at that path. Nothing is substituted for it.
    KeyError
        The detector holds no :data:`ENCODING_KEY` group.
    ValueError
        The environment has no place grid, or ``waveform_std`` differs between
        features, which the per-electrode expansion below could not honor.

    """
    detector_path = Path(detector_path)
    if not detector_path.exists():
        raise FileNotFoundError(f"No detector at {detector_path}.")

    detector = joblib.load(detector_path)
    if ENCODING_KEY not in detector.encoding_model_:
        raise KeyError(
            f"The detector at {detector_path} has no {ENCODING_KEY!r} encoding "
            f"group; it has {list(detector.encoding_model_)}."
        )
    encoding_model = dict(detector.encoding_model_[ENCODING_KEY])

    # a detector saved by an older library version carries no fitted flag,
    # though its grid is intact; the track-distance methods refuse to run
    # without one. Only a grid that is actually there is vouched for.
    environment = encoding_model["environment"]
    if (
        getattr(environment, "place_bin_centers_", None) is None
        or getattr(environment, "is_track_interior_", None) is None
    ):
        raise ValueError(
            f"The environment in {detector_path} has no fitted place grid."
        )
    environment._is_fitted = True

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


def validate_time_grid(time: NDArray, config: MemorylessSequenceConfig) -> None:
    """Check that windows may be summed over this grid at all.

    Sliding a window over the rows assumes every row covers the same span, and
    the window and stride sample counts come from the configured rate, so a
    grid at another rate would make windows of the wrong duration without
    changing any array shape.

    Parameters
    ----------
    time : ndarray, shape (n_time,)
    config : MemorylessSequenceConfig

    Raises
    ------
    ValueError
        The grid is not uniform, is not at the configured rate, or is too short
        to hold one window once its final row is dropped.

    """
    time = np.asarray(time, dtype=np.float64)
    if time.size < config.window_n_samples + 1:
        raise ValueError(
            f"The grid has {time.size} samples; one window needs "
            f"{config.window_n_samples} after the final row, which no spike can "
            "reach, is dropped."
        )
    step = float(np.median(np.diff(time)))
    if abs(step * config.sampling_frequency - 1.0) > _SAMPLE_TOLERANCE:
        raise ValueError(
            f"The grid is sampled every {step} s, not the "
            f"{1.0 / config.sampling_frequency} s of the configured "
            f"{config.sampling_frequency} Hz."
        )
    offset = (time - time[0]) * config.sampling_frequency - np.arange(time.size)
    worst = int(np.argmax(np.abs(offset)))
    if abs(offset[worst]) > _SAMPLE_TOLERANCE:
        raise ValueError(
            f"The decoding grid is not uniform: sample {worst} sits "
            f"{offset[worst]:.3g} samples from where a uniform grid puts it. "
            "Windows summed over it would cover unequal spans."
        )


def _validated_time_and_position(
    time: NDArray, linear_position: NDArray, config: MemorylessSequenceConfig
) -> tuple[NDArray, NDArray]:
    """The checked time grid, and the position as the column the library wants."""
    time = np.asarray(time, dtype=np.float64)
    validate_time_grid(time, config)
    position = np.asarray(linear_position, dtype=np.float64)
    if position.ndim == 1:
        position = position[:, None]
    return time, position


def clusterless_epoch_log_likelihood(
    time: NDArray,
    linear_position: NDArray,
    spike_times: list[NDArray],
    spike_waveform_features: list[NDArray],
    encoding_model: dict,
    config: MemorylessSequenceConfig | None = None,
) -> EpochLikelihood:
    """Decode one epoch on its native grid, with no prior.

    Parameters
    ----------
    time : ndarray, shape (n_time,)
        The decode's uniform sample times.
    linear_position : ndarray, shape (n_time,) or (n_time, 1)
        Linearized position at each sample.
    spike_times : list of ndarray
        Spike times per electrode, in seconds on the same clock as ``time``.
    spike_waveform_features : list of ndarray
        Waveform features per electrode, aligned with ``spike_times``.
    encoding_model : dict
        From :func:`load_clusterless_encoding_model`.
    config : MemorylessSequenceConfig, optional

    Returns
    -------
    likelihood : EpochLikelihood

    """
    from non_local_detector.likelihoods.clusterless_kde import (
        predict_clusterless_kde_log_likelihood,
    )

    time, position = _validated_time_and_position(
        time, linear_position, config or MemorylessSequenceConfig()
    )

    log_likelihood = np.asarray(
        predict_clusterless_kde_log_likelihood(
            cast(Any, time),
            cast(Any, time),
            cast(Any, position),
            cast(Any, spike_times),
            cast(Any, spike_waveform_features),
            **{key: encoding_model[key] for key in CLUSTERLESS_LIKELIHOOD_KEYS},
            is_local=False,
            block_size=encoding_model.get("block_size", 100),
            disable_progress_bar=True,
        )
    )

    return _drop_the_unreachable_row(log_likelihood, time, encoding_model)


def window_midpoints_s(time: NDArray, config: MemorylessSequenceConfig) -> NDArray:
    """Timestamp of each window: the grid sample half a window into its support.

    A window supporting rows ``[k, k + 40)`` is timestamped at row ``k + 20``,
    0.040 s after it begins, the time whose speed decides the window's
    eligibility and the time that bounds the events assembled from it. It is
    the sample itself, not ``time[k] + 0.040``: on a recording's clock the
    sum misses the sample by up to 2.4e-7 s, and an event bound must sit on
    the samples that speed and SWR labels are recorded at. The decoding
    support is a separate quantity and the window width is never added to an
    event's duration.

    Parameters
    ----------
    time : ndarray, shape (n_time,)
    config : MemorylessSequenceConfig

    Returns
    -------
    midpoint_s : ndarray, shape (n_windows,)
        One per window whose full support is present, matching
        :func:`~...statistics.aggregate_windows`.

    """
    time = np.asarray(time, dtype=np.float64)
    n_windows = max(
        0, 1 + (time.size - config.window_n_samples) // config.stride_n_samples
    )
    starts = np.arange(n_windows) * config.stride_n_samples
    midpoint_s: NDArray = time[starts + config.window_n_samples // 2]
    return midpoint_s


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


def _drop_the_unreachable_row(
    log_likelihood: NDArray, time: NDArray, encoding_model: dict
) -> EpochLikelihood:
    """Discard the final row and label the columns with their bin centers.

    No spike can be assigned to the final row, so it carries exposure that
    nothing offsets; see the module docstring.

    Raises
    ------
    ValueError
        A surviving row holds a NaN or infinite value. The library clips its
        place fields away from zero, so this means the encoding model or the
        spikes are broken, and summing windows over it would turn every
        window the row touches into a silent rejection.
    """
    environment = encoding_model["environment"]
    interior = np.asarray(environment.is_track_interior_).ravel()
    values = np.asarray(log_likelihood)[:-1]
    bad_rows = np.flatnonzero(~np.isfinite(values).all(axis=1))
    if bad_rows.size:
        raise ValueError(
            f"{bad_rows.size} rows of the log-likelihood hold non-finite values "
            f"(first at sample {bad_rows[0]}, t = {np.asarray(time)[bad_rows[0]]} s)."
        )
    return EpochLikelihood(
        time=np.asarray(time, dtype=np.float64)[:-1],
        log_likelihood=values,
        place_bin_centers_cm=np.asarray(environment.place_bin_centers_).ravel()[
            interior
        ],
    )


def sorted_spikes_epoch_log_likelihood(
    time: NDArray,
    linear_position: NDArray,
    spike_times: list[NDArray],
    encoding_model: dict,
    config: MemorylessSequenceConfig | None = None,
) -> EpochLikelihood:
    """Decode one epoch from sorted units, on its native grid, with no prior.

    The sorted-spikes likelihood adds each neuron's spike count times the log
    of its place field, and subtracts a per-sample term for the spikes that did
    not arrive. Both parts are per row, so the array is additive over time in
    the same way the clusterless one is, and it assigns spikes to rows with the
    same convention.

    Parameters
    ----------
    time : ndarray, shape (n_time,)
        The decode's uniform sample times.
    linear_position : ndarray, shape (n_time,) or (n_time, 1)
        Linearized position at each sample. The non-local likelihood does not
        read it; it is passed through for the library's signature.
    spike_times : list of ndarray
        Spike times per neuron, in seconds on the same clock as ``time``.
    encoding_model : dict
        Carrying :data:`SORTED_SPIKES_LIKELIHOOD_KEYS`.
    config : MemorylessSequenceConfig, optional

    Returns
    -------
    likelihood : EpochLikelihood

    """
    from non_local_detector.likelihoods.sorted_spikes_kde import (
        predict_sorted_spikes_kde_log_likelihood,
    )

    time, position = _validated_time_and_position(
        time, linear_position, config or MemorylessSequenceConfig()
    )

    log_likelihood = np.asarray(
        predict_sorted_spikes_kde_log_likelihood(
            cast(Any, time),
            cast(Any, time),
            cast(Any, position),
            cast(Any, spike_times),
            **{key: encoding_model[key] for key in SORTED_SPIKES_LIKELIHOOD_KEYS},
            is_local=False,
            disable_progress_bar=True,
        )
    )
    return _drop_the_unreachable_row(log_likelihood, time, encoding_model)
