"""Prior-excluded log-likelihood on the detector's own time grid.

The detector decodes each window on its own under a uniform spatial prior, so
what it needs from the encoding model is the spike log-likelihood per time
sample and place bin, with no transition or state prior mixed in. That is what
``predict_clusterless_kde_log_likelihood`` returns for ``is_local=False``, and
it is additive over time: a window's log-likelihood is the sum of its rows.
:func:`~...window_statistics.aggregate_windows` relies on that, so the additivity is
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

**The detector's grid.** The reference counts spikes in 5 ms bins and steps its
80 ms window by one of them; the decode's grid is 2 ms, on which 5 ms is 2.5
rows. So the likelihood is computed on a grid of its own,
:func:`likelihood_grid`, at ``config.sampling_frequency`` over the decode's
span, and a window is a whole number of its rows. Every function here that
takes a ``time`` takes the *decode's* grid and derives the detector's from it,
so a caller cannot pair the two wrongly. When the two rates agree the
detector's grid is the decode's, unchanged. The encoding model was fitted on
the decode's grid, and the library subtracts its ground-process term (the
spikes that did not arrive) once per row, per *encoding* bin
(``mean_rates`` are spikes per fitted bin). On a coarser grid that term is
therefore multiplied by :func:`encoding_bins_per_row`, 2.5 at 5 ms against
2 ms. The per-spike terms need nothing: a rate rescaled by a constant adds the
same amount to every place bin, which a window's normalization removes.

**Junction bins.** The library returns only ``is_track_interior_`` columns, and
the ``edge_id == -1`` placeholders are built from the non-interior bins, so they
are excluded here by construction. Reading a saved decode's
``log_likelihood`` instead would carry every ``state_bins`` column and would
need the interior mask applied explicitly; that path is deliberately not
implemented until it can be checked against a real file.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from continuum_swr_replay.analysis.detection.memoryless_sequence.config import (
    MemorylessSequenceConfig,
)
from continuum_swr_replay.analysis.detection.memoryless_sequence.window_statistics import (
    windows_holding,
)
from continuum_swr_replay.decoding.results import ENCODING_KEY, load_saved_detector

__all__ = [
    "CLUSTERLESS_LIKELIHOOD_KEYS",
    "REFERENCE_RATE_FLOOR_HZ",
    "SORTED_SPIKES_LIKELIHOOD_KEYS",
    "EpochLikelihood",
    "check_marks_match_encoding",
    "clusterless_encoding_model",
    "clusterless_epoch_log_likelihood",
    "decoded_windows",
    "encoding_bins_per_row",
    "interior_bin_distances_cm",
    "likelihood_grid",
    "load_clusterless_encoding_model",
    "missing_likelihood_rows",
    "sorted_spikes_epoch_log_likelihood",
    "validate_time_grid",
    "window_midpoints_s",
    "with_reference_rate_floor",
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

REFERENCE_RATE_FLOOR_HZ = 1e-3
"""What the reference adds to every rate map before decoding.

``rateMap = rateMap + 1e-3`` in ``compute_BayesianDecoding_fast.m``. It bounds
how strongly a spike from a cell that is nearly silent at a place can rule that
place out."""

_SAMPLE_TOLERANCE = 1e-3
"""How far, in samples, a timestamp may sit from its place on a uniform grid.

Checked in samples rather than as a relative spread of the spacing, because a
recording's clock counts seconds since 1970: near 1.6e9 s a float64 step is
2.4e-7 s, so the 2 ms spacings come out as 1.99986 or 2.00009 ms, a spread
no relative tolerance on the spacing could accept without also accepting a
real error. The same rounding moves a timestamp by 1.2e-4 of a sample at most.
It is measured about the grid's own rate, not the configured one: a clock a
few ppm fast is uniform yet drifts from an exact 500 Hz grid by most of a
sample over an epoch.
"""

_RATE_TOLERANCE = 1e-4
"""How far, as a fraction, the grid's own rate may sit from the configured one.

The decodes' clocks run 0.4-1.7 ppm fast of 500 Hz. At 1e-4 an 80 ms window is
off by at most 8 us, 0.004 of a sample, while a grid at another rate is refused.
"""

_WHOLE_SAMPLE_TOLERANCE = 1e-9
"""How far, in decode samples, a grid position may sit from a whole sample and
still be that sample."""


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


def _check_uniform(time: NDArray, sampling_frequency: float) -> None:
    """Refuse a grid that is not uniform or not at ``sampling_frequency``.

    Uniformity is judged about the grid's own mean rate, which must itself sit
    within ``_RATE_TOLERANCE`` of ``sampling_frequency``; a recording clock a
    few ppm fast passes.
    """
    step = (time[-1] - time[0]) / (time.size - 1)
    offset = (time - time[0]) / step - np.arange(time.size)
    worst = int(np.argmax(np.abs(offset)))
    if abs(offset[worst]) > _SAMPLE_TOLERANCE:
        raise ValueError(
            f"The decoding grid is not uniform: sample {worst} sits "
            f"{offset[worst]:.3g} samples from where a uniform grid puts it. "
            "Windows summed over it would cover unequal spans."
        )
    if abs(step * sampling_frequency - 1.0) > _RATE_TOLERANCE:
        raise ValueError(
            f"The grid is sampled every {step} s on average, not the "
            f"{1.0 / sampling_frequency} s of the configured "
            f"{sampling_frequency} Hz."
        )


def validate_time_grid(time: NDArray, config: MemorylessSequenceConfig) -> None:
    """Check that windows may be summed over this grid at all.

    Sliding a window over the rows assumes every row covers the same span, and
    the window and stride sample counts come from the configured rate, so a
    grid at another rate would make windows of the wrong duration without
    changing any array shape. Uniformity is judged about the grid's own mean
    rate, which must itself sit within ``_RATE_TOLERANCE`` of the configured
    one; a recording clock a few ppm fast passes.

    Parameters
    ----------
    time : ndarray, shape (n_time,)
        The detector's grid, at ``config.sampling_frequency``
        (:func:`likelihood_grid`).
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
    _check_uniform(time, config.sampling_frequency)


def encoding_bins_per_row(config: MemorylessSequenceConfig) -> float:
    """How many of the encoding model's time bins one row of the detector's grid spans.

    The encoding model was fitted on the decode's grid, at
    ``config.decode_sampling_frequency``, so its ground-process term is per
    bin of it (2 ms); a row of the detector's grid lasts
    ``1 / config.sampling_frequency``. 2.5 at 200 Hz, and exactly 1.0 when
    the two rates agree, where multiplying by it changes no value.

    Parameters
    ----------
    config : MemorylessSequenceConfig

    Returns
    -------
    bins_per_row : float

    """
    return config.decode_sampling_frequency / config.sampling_frequency


def _grid_positions(n_decode: int, config: MemorylessSequenceConfig) -> NDArray:
    """Each detector-grid sample's position on the decode grid, in decode samples.

    Shape (n_grid,): ``0, r, 2r, ...`` up to the decode's last sample, with
    ``r = encoding_bins_per_row(config)``.
    """
    step = encoding_bins_per_row(config)
    positions: NDArray = np.arange(0.0, n_decode - 1 + _WHOLE_SAMPLE_TOLERANCE, step)
    return positions


def likelihood_grid(decode_time: NDArray, config: MemorylessSequenceConfig) -> NDArray:
    """The detector's time grid over the decode's span.

    Samples every ``encoding_bins_per_row(config)`` decode samples, from the
    decode's first sample to its last at most, placed by linear interpolation
    between decode samples, so a sample that falls on a decode sample is that
    sample exactly (at 200 Hz, every second one). When ``config`` is at the
    decode's rate the decode grid is returned unchanged.

    Parameters
    ----------
    decode_time : ndarray, shape (n_decode,)
        The decode's uniform grid, at ``config.decode_sampling_frequency``.
    config : MemorylessSequenceConfig

    Returns
    -------
    grid : ndarray, shape (n_grid,)
        At ``config.sampling_frequency``; ``n_grid == n_decode`` when the two
        rates agree.

    Raises
    ------
    ValueError
        The decode grid is not uniform at the decode's rate, or the derived
        grid is too short to hold a window (:func:`validate_time_grid`).

    """
    decode_time = np.asarray(decode_time, dtype=np.float64)
    _check_uniform(decode_time, config.decode_sampling_frequency)
    if config.sampling_frequency == config.decode_sampling_frequency:
        grid = decode_time
    else:
        grid = np.interp(
            _grid_positions(decode_time.size, config),
            np.arange(decode_time.size),
            decode_time,
        )
    validate_time_grid(grid, config)
    return grid


def missing_likelihood_rows(
    decode_time: NDArray, is_missing: NDArray, config: MemorylessSequenceConfig
) -> NDArray[np.bool_]:
    """Which rows of the likelihood span a missing decode sample.

    Row ``k`` of the detector's grid covers ``[grid[k], grid[k + 1])``. It is
    missing when any decode row it overlaps, ``[time[j], time[j + 1])``, is
    missing: conservative, so a row is observed only when every spike it
    holds is. When the two rates agree that is decode row ``k`` itself.

    Parameters
    ----------
    decode_time : ndarray, shape (n_decode,)
    is_missing : ndarray of bool, shape (n_decode,)
        Decode samples with no observation, as the decoders' ``is_missing``.
    config : MemorylessSequenceConfig

    Returns
    -------
    missing_rows : ndarray of bool, shape (n_grid - 1,)
        One per row the likelihood keeps (its final row is dropped).

    """
    decode_time = np.asarray(decode_time, dtype=np.float64)
    is_missing = np.asarray(is_missing, dtype=bool)
    if is_missing.shape != decode_time.shape:
        raise ValueError(
            f"is_missing {is_missing.shape} must be on the decode grid "
            f"{decode_time.shape}."
        )
    # refuses a decode grid the likelihood would refuse
    likelihood_grid(decode_time, config)
    positions = _grid_positions(decode_time.size, config)
    first = np.floor(positions[:-1] + _WHOLE_SAMPLE_TOLERANCE).astype(int)
    stop = np.ceil(positions[1:] - _WHOLE_SAMPLE_TOLERANCE).astype(int)
    before = np.concatenate([[0], np.cumsum(is_missing, dtype=np.int64)])
    missing_rows: NDArray[np.bool_] = (before[stop] - before[first]) > 0
    return missing_rows


def decoded_windows(
    decode_time: NDArray, is_missing: NDArray, config: MemorylessSequenceConfig
) -> tuple[NDArray, NDArray[np.bool_]]:
    """Every window's midpoint, and which windows hold a missing row.

    What :func:`~...epoch.run_epoch` decodes, rebuilt from the decode grid
    alone, for anything that reads a run's windows back.

    Parameters
    ----------
    decode_time : ndarray, shape (n_decode,)
    is_missing : ndarray of bool, shape (n_decode,)
    config : MemorylessSequenceConfig

    Returns
    -------
    midpoint_s : ndarray, shape (n_windows,)
    missing_windows : ndarray of bool, shape (n_windows,)
        The windows left undecoded.

    """
    rows = likelihood_grid(decode_time, config)[:-1]
    missing_windows = windows_holding(
        missing_likelihood_rows(decode_time, is_missing, config),
        config.window_n_samples,
        config.stride_n_samples,
    )
    return window_midpoints_s(rows, config), missing_windows


def _validated_time_and_position(
    time: NDArray, linear_position: NDArray, config: MemorylessSequenceConfig
) -> tuple[NDArray, NDArray]:
    """The detector's grid, and the position on it as the column the library wants.

    Position is interpolated onto the grid only when the grid is not the
    decode's; the non-local likelihood does not read it.
    """
    decode_time = np.asarray(time, dtype=np.float64)
    grid = likelihood_grid(decode_time, config)
    position = np.asarray(linear_position, dtype=np.float64)
    if config.sampling_frequency != config.decode_sampling_frequency:
        position = np.interp(grid, decode_time, position.reshape(decode_time.size))
    if position.ndim == 1:
        position = position[:, None]
    return grid, position


def clusterless_epoch_log_likelihood(
    time: NDArray,
    linear_position: NDArray,
    spike_times: list[NDArray],
    spike_waveform_features: list[NDArray],
    encoding_model: dict,
    config: MemorylessSequenceConfig | None = None,
) -> EpochLikelihood:
    """Decode one epoch on the detector's grid, with no prior.

    The grid is :func:`likelihood_grid`, and the ground-process term is
    scaled to its rows (:func:`encoding_bins_per_row`).

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
        On the detector's grid, ``n_grid - 1`` rows.

    """
    from non_local_detector.likelihoods.clusterless_kde import (
        predict_clusterless_kde_log_likelihood,
    )

    config = config or MemorylessSequenceConfig()
    time, position = _validated_time_and_position(time, linear_position, config)
    model = {key: encoding_model[key] for key in CLUSTERLESS_LIKELIHOOD_KEYS}
    model["summed_ground_process_intensity"] = model[
        "summed_ground_process_intensity"
    ] * encoding_bins_per_row(config)

    log_likelihood = np.asarray(
        predict_clusterless_kde_log_likelihood(
            cast(Any, time),
            cast(Any, time),
            cast(Any, position),
            cast(Any, spike_times),
            cast(Any, spike_waveform_features),
            **model,
            is_local=False,
            block_size=encoding_model.get("block_size", 100),
            disable_progress_bar=True,
        )
    )

    return _drop_the_unreachable_row(log_likelihood, time, encoding_model)


def window_midpoints_s(time: NDArray, config: MemorylessSequenceConfig) -> NDArray:
    """Timestamp of each window: the grid sample half a window into its support.

    A window supporting rows ``[k, k + W)`` is timestamped at row ``k + W / 2``
    (``W`` is 16 rows at 200 Hz, 40 at 500 Hz), 0.040 s after it begins, the time whose speed decides the window's
    eligibility and the time that bounds the events assembled from it. It is
    the sample itself, not ``time[k] + 0.040``: on a recording's clock the
    sum misses the sample by up to 2.4e-7 s, and an event bound must sit on
    the samples that speed and SWR labels are recorded at. The decoding
    support is a separate quantity and the window width is never added to an
    event's duration.

    Parameters
    ----------
    time : ndarray, shape (n_time,)
        The likelihood's rows, ``EpochLikelihood.time``: the detector's grid
        without its final sample.
    config : MemorylessSequenceConfig

    Returns
    -------
    midpoint_s : ndarray, shape (n_windows,)
        One per window whose full support is present, matching
        :func:`~...window_statistics.aggregate_windows`.

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


def with_reference_rate_floor(
    encoding_model: dict, config: MemorylessSequenceConfig
) -> dict:
    """A sorted-spikes encoding model with the reference's floor under every field.

    The library clips its fields only at a numerical epsilon, so a spike from a
    cell that is nearly silent somewhere rules that place out almost entirely;
    the reference's :data:`REFERENCE_RATE_FLOOR_HZ` bounds that. The fields are
    per bin of the grid they were fitted on, the decode's, so the floor is
    divided by its rate (``config.decode_sampling_frequency``), and the
    term for the spikes that did not arrive, the fields summed over cells, is
    recomputed to match. There is no equivalent for a clusterless model.

    Parameters
    ----------
    encoding_model : dict
        Carrying :data:`SORTED_SPIKES_LIKELIHOOD_KEYS`. Not modified.
    config : MemorylessSequenceConfig
        Its ``decode_sampling_frequency`` is the rate the fields are per
        sample of.

    Returns
    -------
    floored : dict
        A shallow copy with ``place_fields`` and
        ``no_spike_part_log_likelihood`` replaced.

    """
    fields = (
        np.asarray(encoding_model["place_fields"], dtype=np.float64)
        + REFERENCE_RATE_FLOOR_HZ / config.decode_sampling_frequency
    )
    return encoding_model | {
        "place_fields": fields,
        "no_spike_part_log_likelihood": fields.sum(axis=0),
    }


def sorted_spikes_epoch_log_likelihood(
    time: NDArray,
    linear_position: NDArray,
    spike_times: list[NDArray],
    encoding_model: dict,
    config: MemorylessSequenceConfig | None = None,
) -> EpochLikelihood:
    """Decode one epoch from sorted units, on the detector's grid, with no prior.

    The sorted-spikes likelihood adds each neuron's spike count times the log
    of its place field, and subtracts a per-row term for the spikes that did
    not arrive, scaled here to the detector's rows as the clusterless ground
    term is (:func:`encoding_bins_per_row`). Both parts are per row, so the
    array is additive over time in the same way the clusterless one is, and it
    assigns spikes to rows with the same convention.

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
        On the detector's grid, ``n_grid - 1`` rows.

    """
    from non_local_detector.likelihoods.sorted_spikes_kde import (
        predict_sorted_spikes_kde_log_likelihood,
    )

    config = config or MemorylessSequenceConfig()
    time, position = _validated_time_and_position(time, linear_position, config)
    model = {key: encoding_model[key] for key in SORTED_SPIKES_LIKELIHOOD_KEYS}
    model["no_spike_part_log_likelihood"] = model[
        "no_spike_part_log_likelihood"
    ] * encoding_bins_per_row(config)

    log_likelihood = np.asarray(
        predict_sorted_spikes_kde_log_likelihood(
            cast(Any, time),
            cast(Any, time),
            cast(Any, position),
            cast(Any, spike_times),
            **model,
            is_local=False,
            disable_progress_bar=True,
        )
    )
    return _drop_the_unreachable_row(log_likelihood, time, encoding_model)
