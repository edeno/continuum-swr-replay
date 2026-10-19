"""Per-tetrode likelihood terms, and the encodings and decode files they come from.

The loaders are shared with ``scripts/qa/analyze_event_spike_evidence.py``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
from numpy.typing import NDArray

from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.likelihood import (
    CLUSTERLESS_LIKELIHOOD_KEYS,
    ENCODING_KEY,
)
from continuum_swr_replay.decode_files import decode_paths
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR


def load_detector(epoch_id: str, decode_dir: Path = PROCESSED_DATA_DIR) -> Any:
    """The decode's saved detector, its encoding environment marked fitted.

    Parameters
    ----------
    epoch_id : str
        e.g. ``"chimi20200213_02_r1_HPC"``.
    decode_dir : Path, optional
        Where ``{epoch_id}_detector.pkl`` is.

    Raises
    ------
    KeyError
        The detector holds no :data:`ENCODING_KEY` encoding model.
    """
    detector = joblib.load(decode_paths(epoch_id, Path(decode_dir)).detector)
    if ENCODING_KEY not in detector.encoding_model_:
        raise KeyError(
            f"ENCODING_KEY {ENCODING_KEY!r} not in encoding_model_; "
            f"available: {list(detector.encoding_model_.keys())}"
        )
    # The fit flag is not persisted through save/load, but the grid data is
    # intact; set it so native track-distance methods work.
    detector.encoding_model_[ENCODING_KEY]["environment"]._is_fitted = True
    return detector


def encoding_inputs(encoding_model: dict) -> dict:
    """The arrays a likelihood call and the reality metrics read from an encoding.

    Returns a dict with ``env`` (fitted Environment), ``encargs`` (kwargs for
    the likelihood call, with a scalar ``waveform_std``), ``pos`` (interior bin
    positions, cm), and ``gpi`` (summed ground-process intensity per bin).
    """
    env = encoding_model["environment"]
    env._is_fitted = True
    interior = np.asarray(env.is_track_interior_).reshape(-1)
    pos = np.asarray(env.place_bin_centers_).reshape(-1)[interior]
    gpi = np.asarray(encoding_model["summed_ground_process_intensity"]).reshape(-1)
    encargs = {key: encoding_model[key] for key in CLUSTERLESS_LIKELIHOOD_KEYS}
    # Pass waveform_std as a scalar so predict expands it per-electrode; some
    # tetrodes have 3 (not 4) waveform features and a non-scalar std crashes
    # the per-feature kde_distance zip.
    encargs["waveform_std"] = float(
        np.asarray(encoding_model["waveform_std"]).ravel()[0]
    )
    return {"env": env, "encargs": encargs, "pos": pos, "gpi": gpi}


def load_encoding(epoch_id: str, decode_dir: Path = PROCESSED_DATA_DIR) -> dict:
    """Load the fitted encoding model and derived position arrays.

    :func:`encoding_inputs` of the decode's own encoding model.
    """
    detector = load_detector(epoch_id, decode_dir)
    return encoding_inputs(detector.encoding_model_[ENCODING_KEY])


def encoding_comparison(refit: dict, stored: dict) -> dict:
    """How a refit encoding model differs from the stored one."""

    def max_abs(a: list, b: list) -> float:
        return max(
            (
                float(np.max(np.abs(np.asarray(x, float) - np.asarray(y, float))))
                for x, y in zip(a, b, strict=True)
                if np.size(x)
            ),
            default=0.0,
        )

    n_refit = [len(x) for x in refit["encoding_positions"]]
    n_stored = [len(x) for x in stored["encoding_positions"]]
    same = n_refit == n_stored
    return {
        "same_n_encoding_spikes": same,
        "n_encoding_spikes": n_stored,
        "max_abs_encoding_positions": (
            max_abs(refit["encoding_positions"], stored["encoding_positions"])
            if same
            else None
        ),
        "max_abs_encoding_marks": (
            max_abs(
                refit["encoding_spike_waveform_features"],
                stored["encoding_spike_waveform_features"],
            )
            if same
            else None
        ),
        "max_abs_mean_rates": float(
            np.max(
                np.abs(
                    np.asarray(refit["mean_rates"]) - np.asarray(stored["mean_rates"])
                )
            )
        ),
    }


def event_likelihood(
    s: int,
    e: int,
    t: NDArray,
    lin: NDArray,
    spike_times: list,
    waveform_features: list,
    encargs: dict,
) -> tuple[NDArray, int, int]:
    """Prior-excluded spike log-likelihood over interior bins for one window.

    The library's own ``predict_clusterless_kde_log_likelihood`` on the window's
    samples ``t[s : e + 1]``. Spikes are binned on the window's axis, so the
    final row holds no spike: it differs from a whole-epoch decode there.

    Returns ``(L, n_spikes, n_tetrodes)`` where ``L`` has shape
    ``(n_time, n_interior_bins)``.
    """
    from non_local_detector.likelihoods.clusterless_kde import (
        predict_clusterless_kde_log_likelihood,
    )

    time = t[s : e + 1]
    in_win = [el[(el >= time[0]) & (el <= time[-1])] for el in spike_times]
    wf_win = [
        w[(el >= time[0]) & (el <= time[-1])]
        for el, w in zip(spike_times, waveform_features, strict=True)
    ]
    L = np.asarray(
        predict_clusterless_kde_log_likelihood(
            cast(Any, time),
            cast(Any, time),
            cast(Any, lin[s : e + 1][:, None]),
            in_win,
            wf_win,
            **encargs,
            is_local=False,
            disable_progress_bar=True,
        )
    )
    n_spikes = sum(len(x) for x in in_win)
    n_tetrodes = sum(len(x) > 0 for x in in_win)
    return L, n_spikes, n_tetrodes


def nearest_interior_bins(
    environment: Any, positions: NDArray, *, chunk: int = 100_000
) -> NDArray[np.int64]:
    """Track-distance nearest interior bin index for each position (cm).

    Parameters
    ----------
    environment : non_local_detector.environment.Environment
        Fitted.
    positions : ndarray, shape (n_positions,)
        Linearized positions; a junction gap's maps to a real bin.
    chunk : int, optional
        Positions per distance lookup, bounding memory.

    Returns
    -------
    column : ndarray of int, shape (n_positions,)
        Index into the interior bins (``is_track_interior_`` order).

    Raises
    ------
    ValueError
        A position is not finite; the grid lookup would silently bin it.
    """
    positions = np.asarray(positions, dtype=float).reshape(-1)
    if not np.isfinite(positions).all():
        raise ValueError("positions must be finite to be binned")
    out = np.empty(positions.size, dtype=np.int64)
    for i in range(0, positions.size, chunk):
        distance = environment.get_distances_to_interior_bins(
            positions[i : i + chunk, None]
        )
        out[i : i + chunk] = np.asarray(distance).argmin(1)
    return out


def interior_bin_mask(environment: Any) -> NDArray[np.bool_]:
    """The real place bins: interior, and on a track graph never a junction bin.

    Parameters
    ----------
    environment : non_local_detector.environment.Environment
        Fitted.

    Returns
    -------
    is_real : ndarray of bool, shape (n_place_bins,)
        ``is_track_interior_``, the bins the library's likelihood covers.

    Raises
    ------
    ValueError
        On a track graph, the interior bins are not exactly the bins with
        ``edge_id != -1`` (the junction placeholders).
    """
    interior = np.asarray(environment.is_track_interior_, dtype=bool).ravel()
    nodes = getattr(environment, "place_bin_centers_nodes_df_", None)
    if nodes is not None:
        real = nodes["edge_id"].to_numpy() != -1
        if not np.array_equal(real, interior):
            raise ValueError(
                "The track-interior bins are not exactly the bins with "
                "edge_id != -1; a junction placeholder would enter the evidence."
            )
    return interior


UNDERFLOW_OCCUPANCY_FLOOR = 1e-20
"""Real place bins whose stored occupancy is above 0 and below this are left out.

Where the animal never went, the KDE occupancy is exactly 0 or underflows
float32. The two cases differ.

- **Exactly 0**: the GPU decode and the XLA CPU both read ``occupancy > 0`` as
  False and take the library's floor (EPS), so they agree. D1 measured 2e-14 per
  spike at bins 0-70 of ``chimi20200221_08_r4``, and its whole-epoch CPU rebuild
  matched the stored decode to 0.005 outside bins 70-79. These bins are not
  flagged (and their evidence, that silent windows prefer them, is a property of
  the model, not of the arithmetic).
- **Above 0 and tiny**: a GPU keeps float32 subnormals (below 1.18e-38); XLA on the
  CPU flushes them to zero. A CPU rebuild of the likelihood there differs from the
  stored decode by up to 20 nats per spike, and the stored decode is the closer to
  exact arithmetic (D1; ``chimi20200221_08_r4`` is the only analysed epoch with
  such bins: the animal never entered patch 1). These are flagged.

D1 measured the per-spike difference between the CPU rebuild and the decode, by
the stored occupancy of the bin: 20 nats from 5.6e-40 to 2.5e-34, falling to 16,
11 and 5.6 nats at 1e-31, 3e-29 and 8e-27, 0.35 at 1.4e-24, and 9e-4 (float
noise, as at every other bin) from 1.9e-22 up. The floor sits 7,000 times above
the last bin that disagreed and 4e14 times below the lowest occupancy of any
other analysed epoch's decode (4e-6), so it takes in no ordinary bin.
"""


def subnormal_occupancy_bin_mask(
    occupancy: NDArray,
    environment: Any,
    floor: float = UNDERFLOW_OCCUPANCY_FLOOR,
) -> NDArray[np.bool_]:
    """The real place bins whose stored occupancy is above 0 and below ``floor``.

    The one definition the state-evidence runner and the spike-evidence engine
    share (see :data:`UNDERFLOW_OCCUPANCY_FLOOR`): the bins where a GPU decode,
    which keeps float32 subnormals, and a CPU rebuild, which flushes them to
    zero, disagree. A bin with an occupancy of exactly 0 is not one: both devices
    read ``occupancy > 0`` as False there and use the same floor. Compare the
    occupancy as stored (float32 on a GPU decode), not a float64 recomputation:
    the stored array is what the decode used.

    Parameters
    ----------
    occupancy : ndarray, shape (n_real_bins,)
        The encoding model's stored ``occupancy``, on the real place bins.
    environment : non_local_detector.environment.Environment
        Fitted; :func:`interior_bin_mask` gives the real bins.
    floor : float, optional

    Returns
    -------
    left_out : ndarray of bool, shape (n_real_bins,)
        True where ``0 < occupancy < floor``. Indices are into the real place
        bins, ``is_track_interior_`` order, the order of :class:`TetrodeTerms`
        columns and of ``event_likelihood``.

    Raises
    ------
    ValueError
        The occupancy does not cover exactly the real bins.
    """
    n_real = int(interior_bin_mask(environment).sum())
    values = np.asarray(occupancy, dtype=np.float64).reshape(-1)
    if values.size != n_real:
        raise ValueError(
            f"{n_real} real place bins; the occupancy covers {values.size}"
        )
    left_out: NDArray[np.bool_] = (values > 0.0) & (values < floor)
    return left_out


@dataclass(frozen=True)
class TetrodeTerms:
    """Per-tetrode pieces of the decoder's non-local clusterless log-likelihood.

    For decode bin t and real place bin x,
    ``logL_t(x) = -sum_e ground[e, x] + sum_{spikes i in bin t} log_intensity[i, x]``.

    Attributes
    ----------
    ground : ndarray, shape (n_tetrodes, n_place_bins)
        g_e(x): the spikes tetrode e is expected to fire in one bin at x.
    log_intensity : ndarray, shape (n_spikes, n_place_bins)
        log lambda_e(x, mark) of each spike, at every real bin.
    time_bin : ndarray of int, shape (n_spikes,)
        Each spike's decode bin.
    tetrode : ndarray of int, shape (n_spikes,)
        Each spike's tetrode.
    bin_s : float
        The decode bin's duration, s; ``ground`` is per bin of this length.
    """

    ground: NDArray
    log_intensity: NDArray
    time_bin: NDArray
    tetrode: NDArray
    bin_s: float

    def __post_init__(self) -> None:
        ground = np.asarray(self.ground, dtype=np.float64)
        log_intensity = np.asarray(self.log_intensity)
        time_bin = np.asarray(self.time_bin, dtype=np.int64)
        tetrode = np.asarray(self.tetrode, dtype=np.int64)
        if ground.ndim != 2 or log_intensity.ndim != 2:
            raise ValueError("ground and log_intensity must be 2-D")
        if log_intensity.shape[1] != ground.shape[1]:
            raise ValueError(
                f"log_intensity has {log_intensity.shape[1]} place bins, ground "
                f"{ground.shape[1]}"
            )
        if not (time_bin.shape == tetrode.shape == (log_intensity.shape[0],)):
            raise ValueError("one time bin and one tetrode per spike")
        if tetrode.size and (tetrode.min() < 0 or tetrode.max() >= ground.shape[0]):
            raise ValueError("a spike's tetrode has no ground intensity")
        if (time_bin < 0).any():
            raise ValueError("time bins must be non-negative")
        if not (np.isfinite(ground).all() and np.isfinite(log_intensity).all()):
            raise ValueError("a ground or log intensity is not finite")
        object.__setattr__(self, "ground", ground)
        object.__setattr__(self, "log_intensity", log_intensity)
        object.__setattr__(self, "time_bin", time_bin)
        object.__setattr__(self, "tetrode", tetrode)
        object.__setattr__(self, "bin_s", float(self.bin_s))

    @property
    def n_tetrodes(self) -> int:
        return int(self.ground.shape[0])

    @property
    def n_place_bins(self) -> int:
        return int(self.ground.shape[1])

    def log_likelihood(self, time_bins: NDArray) -> NDArray:
        """The all-tetrode log-likelihood on some decode bins.

        Parameters
        ----------
        time_bins : ndarray of int, shape (n_rows,)
            Distinct decode bins.

        Returns
        -------
        log_likelihood : ndarray, shape (n_rows, n_place_bins)
        """
        time_bins = np.asarray(time_bins, dtype=np.int64)
        if time_bins.size == 0:
            return np.zeros((0, self.n_place_bins))
        order = np.argsort(time_bins)
        position = np.searchsorted(time_bins[order], self.time_bin)
        position = np.minimum(position, time_bins.size - 1)
        found = time_bins[order][position] == self.time_bin
        out = np.zeros((time_bins.size, self.n_place_bins))
        np.add.at(out, order[position[found]], self.log_intensity[found])
        log_likelihood: NDArray = out - self.ground.sum(0)
        return log_likelihood


def per_tetrode_terms(
    encoding: dict,
    spike_times: Sequence[NDArray],
    marks: Sequence[NDArray],
    time: NDArray,
    interior_bins: NDArray,
) -> TetrodeTerms:
    """Per-spike log joint mark intensities and per-tetrode ground intensities.

    The library's own kernels (``kde_distance``,
    ``block_estimate_log_joint_mark_intensity``), as
    ``predict_clusterless_kde_log_likelihood`` applies them, kept one spike at
    a time so that any half of the tetrodes can be summed later. Spikes are
    binned as the decoder bins them, ``np.digitize(spike_times, time[1:-1])``,
    and those outside ``[time[0], time[-1]]`` are dropped.

    Parameters
    ----------
    encoding : dict
        A fitted clusterless KDE encoding model (``detector.encoding_model_``
        entry).
    spike_times : sequence of ndarray, shape (n_tetrodes,) of (n_spikes_e,)
        Seconds, on the decode's clock.
    marks : sequence of ndarray, shape (n_tetrodes,) of (n_spikes_e, n_features)
        Waveform features, aligned with ``spike_times``.
    time : ndarray, shape (n_time,)
        The decode's time axis.
    interior_bins : ndarray, shape (n_place_bins, n_position_dims)
        Centers of the real place bins (:func:`interior_bin_mask`), the bins
        the encoding's occupancy is evaluated on.

    Returns
    -------
    TetrodeTerms
    """
    import jax.numpy as jnp
    from non_local_detector.likelihoods.clusterless_kde import (
        block_estimate_log_joint_mark_intensity,
        kde_distance,
    )
    from non_local_detector.likelihoods.common import EPS

    n_tetrodes = len(encoding["mean_rates"])
    if not len(spike_times) == len(marks) == n_tetrodes:
        raise ValueError(
            f"{len(spike_times)} spike trains and {len(marks)} mark arrays for "
            f"{n_tetrodes} encoded tetrodes"
        )
    centers = jnp.asarray(interior_bins)
    if centers.ndim == 1:
        centers = centers[:, None]
    occupancy = jnp.asarray(encoding["occupancy"])
    if centers.shape[0] != occupancy.shape[0]:
        raise ValueError(
            f"{centers.shape[0]} place bins given; the encoding's occupancy "
            f"covers {occupancy.shape[0]}"
        )
    time = np.asarray(time, dtype=np.float64)
    waveform_std = encoding["waveform_std"]
    block_size = int(encoding.get("block_size", 1024))
    ground = np.zeros((n_tetrodes, centers.shape[0]))
    intensities, time_bins, tetrodes = [], [], []
    for e in range(n_tetrodes):
        density = encoding["gpi_models"][e].predict(centers)
        ground[e] = np.asarray(
            jnp.clip(
                encoding["mean_rates"][e]
                * jnp.where(
                    occupancy > 0.0,
                    density / jnp.where(occupancy > 0.0, occupancy, 1.0),
                    EPS,
                ),
                min=EPS,
            )
        )
        st = np.asarray(spike_times[e])
        in_bounds = (st >= time[0]) & (st <= time[-1])
        if not in_bounds.any():
            continue
        encoded_marks = np.asarray(encoding["encoding_spike_waveform_features"][e])
        # the library expands a scalar bandwidth to each electrode's features
        electrode_std = (
            jnp.full(encoded_marks.shape[1], waveform_std)
            if np.ndim(waveform_std) == 0
            else jnp.asarray(waveform_std)
        )
        position_distance = kde_distance(
            centers,
            jnp.asarray(encoding["encoding_positions"][e]),
            std=encoding["position_std"],
        )
        intensities.append(
            np.asarray(
                block_estimate_log_joint_mark_intensity(
                    jnp.asarray(np.asarray(marks[e])[in_bounds]),
                    jnp.asarray(encoded_marks),
                    electrode_std,
                    occupancy,
                    encoding["mean_rates"][e],
                    position_distance,
                    block_size,
                ),
                dtype=np.float32,
            )
        )
        time_bins.append(np.digitize(st[in_bounds], time[1:-1]))
        tetrodes.append(np.full(int(in_bounds.sum()), e))
    n_place_bins = centers.shape[0]
    return TetrodeTerms(
        ground=ground,
        log_intensity=(
            np.concatenate(intensities)
            if intensities
            else np.zeros((0, n_place_bins), dtype=np.float32)
        ),
        time_bin=np.concatenate(time_bins) if time_bins else np.zeros(0, np.int64),
        tetrode=np.concatenate(tetrodes) if tetrodes else np.zeros(0, np.int64),
        bin_s=float(np.median(np.diff(time))),
    )


CV_BLOCK_S = 30.0
"""Length of the alternating time blocks of the 2-fold time-held-out encodings,
the prototype's."""


def time_folds(time: NDArray, block_s: float = CV_BLOCK_S) -> NDArray[np.int8]:
    """Each decode bin's fold, 0 or 1: alternating ``block_s`` blocks from the first.

    The prototype's ``cv_terms.fold_of``. The time-held-out encoding of fold f
    is fit on the running bins of fold f only, and scores the windows of the
    other fold (:func:`~.windows.window_folds`).

    Parameters
    ----------
    time : ndarray, shape (n_time,)
        The decode's time axis, s.
    block_s : float, optional

    Returns
    -------
    ndarray of int8, shape (n_time,)
    """
    time = np.asarray(time, dtype=float)
    folds: NDArray[np.int8] = (
        np.floor((time - time[0]) / block_s).astype(np.int64) % 2
    ).astype(np.int8)
    return folds


def training_spike_bins(
    time: NDArray, position: NDArray, is_training: NDArray
) -> NDArray[np.bool_]:
    """The decode bins whose spikes ``fit_encoding_model`` trains on.

    The library's clusterless fit (``ClusterlessDetector._get_group_spike_data``
    in ``non_local_detector.models.base``) drops the bins whose position has a
    NaN, labels the runs of consecutive training bins among the rest, and keeps
    every spike from one bin before a run's first bin time to one bin after its
    last (``start_time -= time_delta``; ``stop_time += time_delta``). So a run
    pads its neighbours, the bin before entirely and the bin after at its start,
    and a run across a NaN-position bin covers that bin too. A bin counts here
    when the decoder (``np.digitize(spike_times, time[1:-1])``) would put a
    kept spike in it; spikes outside ``time[0]``-``time[-1]`` are not scored.

    Parameters
    ----------
    time : ndarray, shape (n_time,)
        The decode's time axis, s.
    position : ndarray, shape (n_time,) or (n_time, n_dims)
        The position the encoding is fit with.
    is_training : ndarray of bool, shape (n_time,)

    Returns
    -------
    ndarray of bool, shape (n_time,)
    """
    time = np.asarray(time, dtype=float)
    position = np.asarray(position, dtype=float)
    if position.ndim == 1:
        position = position[:, np.newaxis]
    valid = ~np.isnan(position).any(axis=1)
    valid_time = time[valid]
    training = np.asarray(is_training, dtype=bool)[valid]
    covered = np.zeros(time.size, dtype=bool)
    if not training.any():
        return covered
    rows = np.flatnonzero(training)
    new_run = np.r_[True, np.diff(rows) > 1]  # consecutive among the valid bins
    first, last = rows[new_run], rows[np.r_[new_run[1:], True]]
    delta = valid_time[1] - valid_time[0]  # as the library takes it
    low = np.maximum(valid_time[first] - delta, time[0])
    high = np.minimum(valid_time[last] + delta, time[-1])
    edges = time[1:-1]
    change = np.zeros(time.size + 1, dtype=np.int64)
    np.add.at(change, np.digitize(low, edges), 1)
    np.add.at(change, np.digitize(high, edges) + 1, -1)
    covered[:] = np.cumsum(change[:-1]) > 0
    return covered
