"""Held-out spike evidence for each decoder state.

The decoder's four states each make a claim about the spikes:

- **Local**: the content is the animal's own position;
- **Non-Local Continuous / Fragmented**: the content is a remote position;
- **No-Spike**: the population is silent.

Each claim is judged by spikes that did not choose the location. The epoch's
tetrodes are split into two halves; the selecting half S chooses (its maximum
a posteriori position, MAP_S) and the scoring half C scores; then the roles
swap. Ten seeded partitions in both directions give 20 runs, and every value is
the mean over the runs of a window total, in nats.

Units (per epoch; regimes per 2 ms bin, :func:`regimes`)
--------------------------------------------------------
- **Bouts** (:func:`state_bouts`): maximal runs of P(state) > 0.8 with every bin
  in one regime, at least 5 bins; at most 1500 per state x regime (a seeded
  sample). A bout longer than 50 bins is scored on one random 50-bin
  sub-window, seeded by the bout's bounds.
- **Matched windows** (:func:`matched_windows`): one random window per scored
  bout window, same regime in every bin and same length, from a seeded
  sequence per (regime, length).
- **Calibration windows** (:func:`calibration_windows`): 2000 random 10-bin
  windows per regime, independent of the decode; each carries the mean
  posterior of every state.

Evidence (:func:`window_evidence`; ``_mark`` = spike-mark part, the primary;
``_full`` = the full clusterless log-likelihood, the secondary)
------------------------------------------------------------------------------
The spike-mark part of logL_C(x) is the sum over C's spikes of
log lambda_e(x, mark); the full part also subtracts C's ground intensity
sum_e g_e(x) in every bin. ``x`` runs over the real place bins only: the
junction placeholders (``edge_id == -1``) are exactly the bins outside the
track interior and never enter (:func:`interior_bin_mask`).

- ``loc_*``: logL_C(animal_t) - mean over 20 partners of
  logL_C(partner's animal position). Partners are other windows of the same
  pool whose centre-bin animal position is at least 20 cm away by track
  distance; never the window itself (:func:`null_partners`).
- ``nl_anim_*``: logL_C(MAP_S) - logL_C(animal_t), with MAP_S the argmax of S's
  window-summed log-likelihood.
- ``nl_null_*``: logL_C(MAP_S) - mean over 20 partners of logL_C(partner's
  MAP_S), partners being any other windows of the same pool.
- ``nlr_anim_*``: logL_C(S's best bin >= 20 cm from the animal) -
  logL_C(animal_t). Its negative is the only Local test in which S chooses.
- ``ns_logratio``: log P(C's counts | silence) - log P(C's counts | local
  coding), Poisson per tetrode-bin; silence is the decoder's ``no_spike_rate``
  x dt, local coding g_e(animal_t). With C silent it is E_C minus a negligible
  term. ``ns_count`` is C's spike count, ``ns_expected`` E_C (the spikes local
  coding expects of C), ``ns_silent_frac`` the share of runs with C silent and
  ``ns_pred_silent`` exp(-E_C).
- ``cond_*`` (summed over runs, not averaged): runs with S silent, runs with
  both halves silent, and exp(-E_C) summed over the S-silent runs; the
  decoder-free test of coordinated silence, P(C silent | S silent) against
  local coding's exp(-E_C) (:func:`coordinated_silence`).
- ``*_ps``: per held-out spike, runs with C silent excluded.
- ``mapS_dist_cm`` / ``mapSr_dist_cm``: track distance from the animal's
  window-centre position to MAP_S / S's best remote bin.

What is and is not held out (properties, not defects)
-----------------------------------------------------
- Only the remote location is held out: S alone chooses MAP_S. The state
  labels come from the all-tetrode decode, so C's spikes helped make them.
- Run-averaged Local and No-Spike values equal half the all-tetrode
  (in-sample) values: no half chooses the animal's position, and every
  tetrode scores in exactly half of the runs.
- One spike costs about 29 nats under No-Spike (log(1e-10 Hz x 2 ms)), so a
  confident No-Spike label forces both halves silent; only the size of E_C is
  informative there.
- With few spikes, S's silence term pulls MAP_S toward low-rate bins, so
  MAP_S's distance from the animal is not a Local test.
"""

from __future__ import annotations

import warnings
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from numpy.typing import NDArray

from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.likelihood import (
    CLUSTERLESS_LIKELIHOOD_KEYS,
    ENCODING_KEY,
)
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR

__all__ = [
    "CALIBRATION_METRICS",
    "ENCODING_KEY",
    "ENCODING_LABELS",
    "MIN_BIN_WINDOWS",
    "PRIMARY_METRICS",
    "PROBABILITY_BIN_LABELS",
    "REGIMES",
    "SECONDARY_METRICS",
    "STATES",
    "TetrodeTerms",
    "bout_statistics",
    "build_windows",
    "calibration_statistics",
    "calibration_windows",
    "coordinated_silence",
    "encoding_inputs",
    "event_likelihood",
    "interior_bin_mask",
    "keyed_rng",
    "load_detector",
    "load_encoding",
    "matched_windows",
    "nearest_interior_bins",
    "null_partners",
    "ols_slope_ci",
    "parse_epoch_identifier",
    "per_tetrode_terms",
    "rate_calibration",
    "regimes",
    "sign_flip_p",
    "state_bouts",
    "stratified_bootstrap_ci",
    "summarize_state_evidence",
    "tetrode_halves",
    "window_evidence",
]

STATES = ("Local", "No-Spike", "Non-Local Continuous", "Non-Local Fragmented")
"""The decoder's states, in its own order."""

REGIMES = ("immobile", "running")
IMMOBILE_BELOW_CM_S = 4.0
RUNNING_ABOVE_CM_S = 10.0

PRIMARY_METRICS: Mapping[str, tuple[str, ...]] = {
    "Local": ("run_loc_mark",),
    "No-Spike": ("run_ns_logratio", "can_ns_logratio"),
    "Non-Local Continuous": ("run_nl_null_mark", "run_nl_anim_mark"),
    "Non-Local Fragmented": ("run_nl_null_mark", "run_nl_anim_mark"),
}
"""The evidence each state's claim is judged by (``run_`` = running-only
encoding, ``can_`` = the decode's own)."""

SECONDARY_METRICS = (
    "run_loc_vs_Sremote_mark",
    "run_nlr_anim_mark",
    "can_loc_mark",
    "run_loc_full",
    "can_nl_null_mark",
    "can_nl_anim_mark",
    "run_nl_anim_full",
    "run_ns_count",
    "run_ns_silent_frac",
    "run_ns_expected",
    "can_ns_expected",
    "run_mapS_dist_cm",
    "run_mapSr_dist_cm",
)
"""Reported for the bouts of every state."""

CALIBRATION_METRICS: Mapping[str, tuple[str, ...]] = {
    "Local": ("run_loc_mark", "run_loc_vs_Sremote_mark", "can_loc_mark"),
    "No-Spike": (
        "run_ns_logratio",
        "can_ns_logratio",
        "run_ns_count",
        "run_ns_silent_frac",
    ),
    "Non-Local Continuous": (
        "run_nl_null_mark",
        "run_nl_anim_mark",
        "can_nl_null_mark",
        "can_nl_anim_mark",
    ),
    "Non-Local Fragmented": (
        "run_nl_null_mark",
        "run_nl_anim_mark",
        "can_nl_null_mark",
        "can_nl_anim_mark",
    ),
}
"""The calibration curves: each metric against its state's probability."""

PROBABILITY_BINS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0000001)
PROBABILITY_BIN_LABELS = ("0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0")
MIN_BIN_WINDOWS = 10
"""A calibration bin with fewer windows is not compared."""

SUMMARY_SEED = 20261004

ENCODING_LABELS = {"run": "running-only", "can": "canonical"}
"""The evidence columns' prefixes: the running-only refit and the decode's own."""

_PER_RUN_SUMMED = ("cond_S_silent", "cond_both_silent", "cond_pred_C_silent")
_PER_SPIKE = ("loc_mark", "loc_full", "nl_anim_mark", "nl_null_mark", "nlr_anim_mark")
_RESAMPLE_CELLS = 1 << 24
"""Random draws per chunk of a resampling test, so memory stays bounded."""

Seed = int | Sequence[int | str]
"""A random stream's name: a base seed, optionally followed by keys."""


# --- loading (shared with scripts/qa/analyze_event_spike_evidence.py) -------------
def parse_epoch_identifier(epoch_id: str) -> tuple[str, str, str]:
    """Split ``{animal}{date}_{epoch}_{brain_area}`` into load_data arguments.

    Parameters
    ----------
    epoch_id : str
        e.g. ``"chimi20200213_02_r1_HPC"``.

    Returns
    -------
    nwb_file_name, epoch_name, brain_area : str
        e.g. ``("chimi20200213_.nwb", "02_r1", "HPC")``.
    """
    parts = epoch_id.split("_")
    nwb_file_name = f"{parts[0]}_.nwb"
    epoch_name = f"{parts[1]}_{parts[2]}"
    brain_area = parts[3]
    return nwb_file_name, epoch_name, brain_area


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
    detector = joblib.load(Path(decode_dir) / f"{epoch_id}_detector.pkl")
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


# --- per-tetrode terms ---------------------------------------------------------------
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


# --- windows -----------------------------------------------------------------------------
def keyed_rng(seed: Seed, *keys: int | str) -> np.random.Generator:
    """A reproducible random stream named by ``seed`` and ``keys``.

    ``seed`` is a base seed (int) or a sequence starting with one; every later
    entry, then every key, is joined with ``"|"`` and hashed (CRC-32):
    ``default_rng([base, crc32("|".join(names))])``. Equal names give equal
    streams and each name its own, so one draw never shifts another: a new
    epoch or state leaves every other window as it was.
    """
    base, *names = (seed,) if isinstance(seed, int | np.integer) else tuple(seed)
    names = [*names, *keys]
    if not names:
        return np.random.default_rng(int(base))
    return np.random.default_rng(
        [int(base), zlib.crc32("|".join(map(str, names)).encode())]
    )


def _with_keys(seed: Seed, *keys: int | str) -> tuple[int | str, ...]:
    """``seed`` as a tuple, extended by ``keys``."""
    return (*((seed,) if isinstance(seed, int | np.integer) else tuple(seed)), *keys)


def regimes(head_speed: NDArray) -> NDArray:
    """Each bin's speed regime: ``"immobile"`` < 4 cm/s, ``"running"`` > 10.

    Parameters
    ----------
    head_speed : ndarray, shape (n_time,)
        cm/s.

    Returns
    -------
    regime : ndarray of str, shape (n_time,)
        ``""`` between the two, and where the speed is not finite.
    """
    speed = np.asarray(head_speed, dtype=float)
    out = np.full(speed.shape, "", dtype="<U8")
    out[speed < IMMOBILE_BELOW_CM_S] = "immobile"
    out[speed > RUNNING_ABOVE_CM_S] = "running"
    return out


def _runs(mask: NDArray) -> NDArray[np.int64]:
    """``(n_runs, 2)``: first and last index (inclusive) of each run of True."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    breaks = np.flatnonzero(np.diff(idx) > 1)
    runs: NDArray[np.int64] = np.c_[
        np.r_[idx[0], idx[breaks + 1]], np.r_[idx[breaks], idx[-1]]
    ].astype(np.int64)
    return runs


def _valid_starts(mask: NDArray, length: int) -> NDArray[np.int64]:
    """Every s with ``mask[s : s + length]`` all True."""
    if length > mask.size:
        return np.zeros(0, dtype=np.int64)
    cumulative = np.r_[0, np.cumsum(mask, dtype=np.int64)]
    return np.flatnonzero(cumulative[length:] - cumulative[:-length] == length)


def state_bouts(
    probability: NDArray,
    regime: NDArray,
    *,
    threshold: float = 0.8,
    min_bins: int = 5,
    max_scored_bins: int = 50,
    cap: int = 1500,
    seed: Seed,
) -> pd.DataFrame:
    """Confident bouts of one state, each with the sub-window that is scored.

    A bout is a maximal run of bins with ``probability > threshold`` and every
    bin in one regime, at least ``min_bins`` long. At most ``cap`` are kept per
    regime, a seeded sample. A bout longer than ``max_scored_bins`` is scored on
    one random sub-window of that length, seeded by its bounds, so the same
    bout gets the same sub-window whatever else is sampled.

    Parameters
    ----------
    probability : ndarray, shape (n_time,)
        The state's posterior probability.
    regime : ndarray of str, shape (n_time,)
        From :func:`regimes`.
    threshold, min_bins, max_scored_bins, cap : optional
    seed : int or sequence
        Names the stream (:func:`keyed_rng`); the runner passes
        ``(base, epoch_id, state)``.

    Returns
    -------
    pd.DataFrame
        One row per kept bout, by regime then start: ``regime``, ``bout_s``,
        ``bout_e`` (inclusive), ``bout_len``, ``full_len`` (the enclosing
        confident run, regardless of regime), ``clipped`` (the regime cut it),
        ``n_found`` (bouts before the cap), and ``s``, ``e``, the scored window.
    """
    probability = np.asarray(probability, dtype=float)
    regime = np.asarray(regime)
    if probability.shape != regime.shape:
        raise ValueError("probability and regime must be the same length")
    confident = probability > threshold
    full = _runs(confident)
    frames = []
    for name in REGIMES:
        runs = _runs(confident & (regime == name))
        runs = runs[runs[:, 1] - runs[:, 0] + 1 >= min_bins]
        n_found = len(runs)
        if n_found > cap:
            keep = keyed_rng(seed, name, "cap").choice(n_found, cap, replace=False)
            runs = runs[np.sort(keep)]
        enclosing = np.searchsorted(full[:, 0], runs[:, 0], side="right") - 1
        full_len = (full[enclosing, 1] - full[enclosing, 0] + 1).astype(np.int64)
        bout_len = runs[:, 1] - runs[:, 0] + 1
        s = runs[:, 0].copy()
        for i in np.flatnonzero(bout_len > max_scored_bins):
            stream = keyed_rng(seed, name, int(runs[i, 0]), int(runs[i, 1]), "sub")
            s[i] += stream.integers(0, bout_len[i] - max_scored_bins + 1)
        frames.append(
            pd.DataFrame(
                {
                    "regime": name,
                    "bout_s": runs[:, 0],
                    "bout_e": runs[:, 1],
                    "bout_len": bout_len,
                    "full_len": full_len,
                    "clipped": full_len > bout_len,
                    "n_found": n_found,
                    "s": s,
                    "e": s + np.minimum(bout_len, max_scored_bins) - 1,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def matched_windows(
    bouts: pd.DataFrame, regime: NDArray, *, seed: Seed
) -> pd.DataFrame:
    """One random window per bout: the same regime in every bin, the same length.

    Each (regime, length) has one seeded random order of all its valid
    windows; within a regime, bouts take them in that order, by start. The same
    order is shared by every state, so two states' bouts of one length draw
    the same windows.

    Parameters
    ----------
    bouts : pd.DataFrame
        From :func:`state_bouts` (``regime``, ``bout_s``, ``s``, ``e``).
    regime : ndarray of str, shape (n_time,)
    seed : int or sequence
        The runner passes ``(base, epoch_id)``.

    Returns
    -------
    pd.DataFrame
        Aligned with ``bouts``: ``regime``, ``s``, ``e`` and ``matched_bout_s``.
    """
    regime = np.asarray(regime)
    lengths = (bouts["e"] - bouts["s"] + 1).to_numpy(dtype=np.int64)
    bout_regime = bouts["regime"].to_numpy()
    bout_s = bouts["bout_s"].to_numpy()
    starts = np.zeros(len(bouts), dtype=np.int64)
    for name in pd.unique(bout_regime):
        mask = regime == name
        rows = np.flatnonzero(bout_regime == name)
        rows = rows[np.argsort(bout_s[rows], kind="stable")]
        orders: dict[int, NDArray] = {}
        used: dict[int, int] = {}
        for row in rows:
            length = int(lengths[row])
            if length not in orders:
                stream = keyed_rng(seed, name, length, "baseline")
                orders[length] = stream.permutation(_valid_starts(mask, length))
            k = used.get(length, 0)
            if k >= orders[length].size:
                raise ValueError(f"not enough {name} windows of {length} bins")
            starts[row] = orders[length][k]
            used[length] = k + 1
    return pd.DataFrame(
        {
            "regime": bout_regime,
            "s": starts,
            "e": starts + lengths - 1,
            "matched_bout_s": bout_s,
        },
        index=bouts.index,
    )


def calibration_windows(
    regime: NDArray, *, n_per_regime: int = 2000, n_bins: int = 10, seed: Seed
) -> pd.DataFrame:
    """Random windows with every bin in one regime, chosen without the decode.

    Parameters
    ----------
    regime : ndarray of str, shape (n_time,)
    n_per_regime : int, optional
        All valid windows when there are fewer.
    n_bins : int, optional
    seed : int or sequence
        The runner passes ``(base, epoch_id)``.

    Returns
    -------
    pd.DataFrame
        ``regime``, ``s``, ``e``; by regime, then start.
    """
    regime = np.asarray(regime)
    frames = []
    for name in REGIMES:
        starts = _valid_starts(regime == name, n_bins)
        n = min(n_per_regime, starts.size)
        chosen = (
            np.sort(keyed_rng(seed, name, "calib").choice(starts, n, replace=False))
            if n
            else starts[:0]
        )
        frames.append(
            pd.DataFrame({"regime": name, "s": chosen, "e": chosen + n_bins - 1})
        )
    return pd.concat(frames, ignore_index=True)


def build_windows(
    probabilities: NDArray,
    regime: NDArray,
    *,
    threshold: float = 0.8,
    min_bins: int = 5,
    max_scored_bins: int = 50,
    cap: int = 1500,
    n_per_regime: int = 2000,
    calibration_bins: int = 10,
    seed: Seed,
) -> pd.DataFrame:
    """Every window of one epoch: bouts, their matched windows, calibration windows.

    Parameters
    ----------
    probabilities : ndarray, shape (n_time, 4)
        The decode's state probabilities, in :data:`STATES` order.
    regime : ndarray of str, shape (n_time,)
    threshold, min_bins, max_scored_bins, cap : optional
        :func:`state_bouts`'s.
    n_per_regime, calibration_bins : optional
        :func:`calibration_windows`'s ``n_per_regime`` and ``n_bins``.
    seed : int or sequence
        The epoch's stream, ``(base, epoch_id)`` in the runner; each state's
        bouts extend it with the state's name.

    Returns
    -------
    pd.DataFrame
        One row per window: ``kind`` (``"bout"``, ``"baseline"``,
        ``"calib"``), ``state`` (``""`` for calibration windows), ``regime``,
        ``pool`` (``kind|state|regime``: the windows partners come from),
        ``s``, ``e``, ``n_bins``, the bout and matching columns, and
        ``p_{state}``, the window's mean posterior of every state.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    regime = np.asarray(regime)
    if probabilities.shape != (regime.size, len(STATES)):
        raise ValueError(
            f"probabilities are {probabilities.shape}; expected "
            f"({regime.size}, {len(STATES)})"
        )
    if not np.isfinite(probabilities).all():
        raise ValueError("state probabilities must be finite")
    bouts, matched = [], []
    for j, state in enumerate(STATES):
        found = state_bouts(
            probabilities[:, j],
            regime,
            threshold=threshold,
            min_bins=min_bins,
            max_scored_bins=max_scored_bins,
            cap=cap,
            seed=_with_keys(seed, state),
        )
        bouts.append(found.assign(kind="bout", state=state))
        matched.append(
            matched_windows(found, regime, seed=seed).assign(
                kind="baseline", state=state
            )
        )
    calibration = calibration_windows(
        regime, n_per_regime=n_per_regime, n_bins=calibration_bins, seed=seed
    ).assign(kind="calib", state="")
    windows = pd.concat([*bouts, *matched, calibration], ignore_index=True)
    windows["clipped"] = windows["clipped"].astype("boolean")  # missing off bouts
    windows["n_bins"] = windows.e - windows.s + 1
    windows["pool"] = windows.kind + "|" + windows.state + "|" + windows.regime
    cumulative = np.vstack(
        [np.zeros((1, len(STATES))), np.cumsum(probabilities, axis=0)]
    )
    s, e = windows.s.to_numpy(), windows.e.to_numpy()
    means = (cumulative[e + 1] - cumulative[s]) / windows.n_bins.to_numpy()[:, None]
    for j, state in enumerate(STATES):
        windows[f"p_{state}"] = means[:, j]
    first = ["kind", "state", "regime", "pool", "s", "e", "n_bins"]
    return windows[first + [c for c in windows.columns if c not in first]]


def tetrode_halves(
    n_tetrodes: int, *, n_partitions: int = 10, seed: Seed = 0
) -> list[tuple[NDArray, NDArray]]:
    """Seeded random partitions of the tetrodes into two halves.

    Parameters
    ----------
    n_tetrodes : int
    n_partitions : int, optional
    seed : int or sequence, optional

    Returns
    -------
    list of (ndarray, ndarray)
        Per partition, the sorted indices of half A (``n_tetrodes // 2``) and
        of half B (the rest). Each half selects once and scores once.
    """
    if n_tetrodes < 2:
        raise ValueError("a split needs at least two tetrodes")
    stream = keyed_rng(seed)
    halves = []
    for _ in range(n_partitions):
        order = stream.permutation(n_tetrodes)
        half = n_tetrodes // 2
        halves.append((np.sort(order[:half]), np.sort(order[half:])))
    return halves


# --- evidence ----------------------------------------------------------------------------
def null_partners(
    pool: NDArray,
    centre_bins: NDArray,
    track_distance: NDArray,
    *,
    n_partners: int = 20,
    min_cm: float = 20.0,
    seed: Seed,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Each window's null partners, drawn from its own pool.

    Parameters
    ----------
    pool : ndarray, shape (n_windows,)
        The pool label of each window.
    centre_bins : ndarray of int, shape (n_windows,)
        The animal's real place bin at each window's centre bin.
    track_distance : ndarray, shape (n_place_bins, n_place_bins)
        cm, between real place bins.
    n_partners : int, optional
    min_cm : float, optional
    seed : int or sequence
        Each pool draws from ``keyed_rng(seed, pool, "null")``.

    Returns
    -------
    remote : ndarray of int, shape (n_windows, n_partners)
        Other windows of the pool, uniformly with replacement (the MAP_S null).
    local : ndarray of int, shape (n_windows, n_partners)
        Other windows of the pool whose centre bin is at least ``min_cm`` from
        this window's (the Local null). -1 throughout when there is none, and in
        both arrays for a pool of one window.
    """
    pool = np.asarray(pool)
    centre = np.asarray(centre_bins, dtype=np.int64)
    distance = np.asarray(track_distance, dtype=float)
    remote = np.full((pool.size, n_partners), -1, dtype=np.int64)
    local = np.full((pool.size, n_partners), -1, dtype=np.int64)
    codes, names = pd.factorize(pool)
    for k, name in enumerate(names):
        idx = np.flatnonzero(codes == k)
        n = idx.size
        if n < 2:
            continue
        stream = keyed_rng(seed, name, "null")
        draw = stream.integers(0, n - 1, size=(n, n_partners))
        remote[idx] = idx[draw + (draw >= np.arange(n)[:, None])]
        eligible = distance[np.ix_(centre[idx], centre[idx])] >= min_cm
        np.fill_diagonal(eligible, False)
        k_eligible = eligible.sum(1)
        columns = np.argsort(~eligible, axis=1, kind="stable")  # eligible first
        pick = np.minimum(
            (stream.random((n, n_partners)) * k_eligible[:, None]).astype(np.int64),
            np.maximum(k_eligible - 1, 0)[:, None],
        )
        chosen = idx[np.take_along_axis(columns, pick, 1)]
        chosen[k_eligible == 0] = -1
        local[idx] = chosen
    return remote, local


def window_evidence(
    terms: TetrodeTerms,
    windows: pd.DataFrame,
    animal_bins: NDArray,
    track_distance: NDArray,
    halves: Sequence[tuple[NDArray, NDArray]],
    *,
    partner_min_cm: float = 20.0,
    n_partners: int = 20,
    no_spike_rate: float,
    seed: Seed,
) -> pd.DataFrame:
    """Held-out evidence of every window under one encoding (see the module notes).

    Parameters
    ----------
    terms : TetrodeTerms
        One encoding's terms, on the decode's time axis.
    windows : pd.DataFrame
        ``s``, ``e`` (inclusive decode bins) and ``pool``.
    animal_bins : ndarray of int, shape (n_time,)
        The animal's real place bin (an index into ``terms``' place bins) in
        every decode bin; any value outside the windows is ignored.
    track_distance : ndarray, shape (n_place_bins, n_place_bins)
        cm, between real place bins.
    halves : sequence of (ndarray, ndarray)
        From :func:`tetrode_halves`; each half selects once and scores once.
    partner_min_cm, n_partners : optional
        :func:`null_partners`'s.
    no_spike_rate : float
        The decoder's No-Spike rate, Hz.
    seed : int or sequence
        For the partners.

    Returns
    -------
    pd.DataFrame
        Indexed like ``windows``: the evidence columns of the module notes,
        ``loc_mark_sd_runs`` and ``nl_null_mark_sd_runs`` (their spread over
        runs), ``run0_mapS`` / ``run0_mapSr`` (run 0's choices, for spot
        checks), and ``n_spikes``, ``n_tetrodes`` (all tetrodes),
        ``animal_center_bin`` and ``has_loc_partner``.
    """
    s = windows["s"].to_numpy(dtype=np.int64)
    e = windows["e"].to_numpy(dtype=np.int64)
    animal_bins = np.asarray(animal_bins, dtype=np.int64)
    distance = np.asarray(track_distance, dtype=float)
    n_win, n_time, n_pos = s.size, animal_bins.size, terms.n_place_bins
    if n_win == 0:
        raise ValueError("no windows to score")
    if not halves:
        raise ValueError("no tetrode halves to split by")
    if (e < s).any() or (s < 0).any() or (e >= n_time).any():
        raise ValueError("every window must lie inside the time axis, s <= e")
    if distance.shape != (n_pos, n_pos):
        raise ValueError(f"track_distance is {distance.shape}; terms have {n_pos} bins")
    if terms.time_bin.size and terms.time_bin.max() >= n_time:
        raise ValueError("a spike's time bin is beyond animal_bins")
    lens = e - s + 1
    starts = np.r_[0, np.cumsum(lens)[:-1]].astype(np.int64)
    win_of_row = np.repeat(np.arange(n_win), lens)
    concat_bins = np.repeat(s, lens) + (np.arange(lens.sum()) - np.repeat(starts, lens))
    in_union = np.zeros(n_time, dtype=bool)
    in_union[concat_bins] = True
    union_bins = np.flatnonzero(in_union)
    n_rows = union_bins.size
    bin_rows = np.full(n_time, -1, dtype=np.int64)
    bin_rows[union_bins] = np.arange(n_rows)
    rows = bin_rows[concat_bins]
    animal_row = animal_bins[union_bins]
    if (animal_row < 0).any() or (animal_row >= n_pos).any():
        raise ValueError("a window bin has no animal place bin")
    animal_concat = animal_bins[concat_bins]
    centre = animal_bins[s + lens // 2]
    near_animal = distance[centre] < partner_min_cm
    window_sum = sp.csr_matrix(
        (np.ones(rows.size), (win_of_row, rows)), shape=(n_win, n_rows)
    )
    remote_partner, local_partner = null_partners(
        windows["pool"].to_numpy(),
        centre,
        distance,
        n_partners=n_partners,
        min_cm=partner_min_cm,
        seed=seed,
    )
    has_remote = remote_partner[:, 0] >= 0
    has_local = local_partner[:, 0] >= 0

    spike_row = bin_rows[terms.time_bin]
    keep = spike_row >= 0
    intensity = terms.log_intensity[keep].astype(np.float64)
    spike_row, spike_tetrode = spike_row[keep], terms.tetrode[keep]
    n_tet = terms.n_tetrodes
    counts = np.zeros((n_tet, n_rows))
    np.add.at(counts, (spike_tetrode, spike_row), 1.0)
    rho = no_spike_rate * terms.bin_s
    local_rate = terms.ground[:, animal_row]  # (n_tet, n_rows)
    silence_minus_local = counts * (np.log(rho) - np.log(local_rate)) + local_rate - rho

    def window_total(values: NDArray) -> NDArray:
        return np.add.reduceat(values, starts)

    def half_terms(mask: NDArray) -> tuple[NDArray, NDArray]:
        """The half's spike-mark sum per union row and its summed ground."""
        mine = mask[spike_tetrode]
        assign = sp.csr_matrix(
            (np.ones(mine.sum()), (spike_row[mine], np.arange(mine.sum()))),
            shape=(n_rows, mine.sum()),
        )
        return np.asarray(assign @ intensity[mine]), terms.ground[mask].sum(0)

    names = [
        "loc_mark", "loc_full", "nl_anim_mark", "nl_anim_full", "nl_null_mark",
        "nl_null_full", "nlr_anim_mark", "nlr_anim_full", "mapSr_dist_cm",
        "ns_logratio", "ns_count", "ns_expected", "ns_silent_frac", "ns_pred_silent",
        "sel_count", "mapS_dist_cm", *_PER_RUN_SUMMED,
        *(f"{name}_ps" for name in _PER_SPIKE),
    ]  # fmt: skip
    values = {name: np.full((2 * len(halves), n_win), np.nan) for name in names}
    first_choice: dict[str, NDArray] = {}
    tetrode_ids = np.arange(n_tet)
    base = rows * n_pos  # each concatenated row's offset into a flattened half
    for partition, pair in enumerate(halves):
        masks = [np.isin(tetrode_ids, half) for half in pair]
        sums = [half_terms(mask) for mask in masks]
        # run 2p: the first half selects and the second scores; run 2p + 1 swaps
        for direction, (si, ci) in enumerate(((0, 1), (1, 0))):
            run = 2 * partition + direction
            (sel_mark, sel_ground), (sco_mark, sco_ground) = sums[si], sums[ci]
            pooled = (
                np.asarray(window_sum @ sel_mark) - lens[:, None] * sel_ground[None, :]
            )
            choice = pooled.argmax(1)
            remote_choice = np.where(near_animal, -np.inf, pooled).argmax(1)
            if run == 0:
                first_choice = {"run0_mapS": choice, "run0_mapSr": remote_choice}
            flat = sco_mark.ravel()
            chosen, chosen_remote = choice[win_of_row], remote_choice[win_of_row]
            at_map = flat[base + chosen]
            at_animal = flat[base + animal_concat]
            at_remote = flat[base + chosen_remote]
            map_g = sco_ground[chosen]
            animal_g = sco_ground[animal_concat]
            remote_g = sco_ground[chosen_remote]
            null_m = np.zeros(rows.size)
            null_g = np.zeros(rows.size)
            loc_m = np.zeros(rows.size)
            loc_g = np.zeros(rows.size)
            for j in range(n_partners):
                partner_map = choice[np.maximum(remote_partner[:, j], 0)][win_of_row]
                null_m += flat[base + partner_map]
                null_g += sco_ground[partner_map]
                partner_animal = centre[np.maximum(local_partner[:, j], 0)][win_of_row]
                loc_m += flat[base + partner_animal]
                loc_g += sco_ground[partner_animal]
            null_m, null_g = null_m / n_partners, null_g / n_partners
            loc_m, loc_g = loc_m / n_partners, loc_g / n_partners
            n_sco = window_total(counts[masks[ci]].sum(0)[rows])
            n_sel = window_total(counts[masks[si]].sum(0)[rows])
            expected = window_total(sco_ground[animal_concat])
            s_silent = (n_sel == 0).astype(float)
            run_values = {
                "nl_anim_mark": window_total(at_map - at_animal),
                "nl_anim_full": window_total((at_map - map_g) - (at_animal - animal_g)),
                "nlr_anim_mark": window_total(at_remote - at_animal),
                "nlr_anim_full": window_total(
                    (at_remote - remote_g) - (at_animal - animal_g)
                ),
                "mapSr_dist_cm": distance[remote_choice, centre],
                "nl_null_mark": np.where(
                    has_remote, window_total(at_map - null_m), np.nan
                ),
                "nl_null_full": np.where(
                    has_remote,
                    window_total((at_map - map_g) - (null_m - null_g)),
                    np.nan,
                ),
                "loc_mark": np.where(
                    has_local, window_total(at_animal - loc_m), np.nan
                ),
                "loc_full": np.where(
                    has_local,
                    window_total((at_animal - animal_g) - (loc_m - loc_g)),
                    np.nan,
                ),
                "ns_logratio": window_total(
                    silence_minus_local[masks[ci]].sum(0)[rows]
                ),
                "ns_count": n_sco,
                "ns_expected": expected,
                "ns_silent_frac": (n_sco == 0).astype(float),
                "ns_pred_silent": np.exp(-expected),
                "sel_count": n_sel,
                "mapS_dist_cm": distance[choice, centre],
                "cond_S_silent": s_silent,
                "cond_both_silent": s_silent * (n_sco == 0),
                "cond_pred_C_silent": s_silent * np.exp(-expected),
            }
            with np.errstate(invalid="ignore", divide="ignore"):
                for name in _PER_SPIKE:
                    run_values[f"{name}_ps"] = np.where(
                        n_sco > 0, run_values[name] / n_sco, np.nan
                    )
            for name, value in run_values.items():
                values[name][run] = value
    out: dict[str, NDArray] = {}
    with warnings.catch_warnings():  # a window with no partner is NaN in every run
        warnings.simplefilter("ignore", RuntimeWarning)
        for name, value in values.items():
            summed = name in _PER_RUN_SUMMED
            out[name] = np.nansum(value, 0) if summed else np.nanmean(value, 0)
        out["loc_mark_sd_runs"] = np.nanstd(values["loc_mark"], 0)
        out["nl_null_mark_sd_runs"] = np.nanstd(values["nl_null_mark"], 0)
    out.update(first_choice)
    per_tetrode = np.add.reduceat(counts[:, rows], starts, axis=1)  # (n_tet, n_win)
    out["n_spikes"] = per_tetrode.sum(0).astype(np.int64)
    out["n_tetrodes"] = (per_tetrode > 0).sum(0)
    out["animal_center_bin"] = centre
    out["has_loc_partner"] = has_local
    return pd.DataFrame(out, index=windows.index)


# --- statistics ----------------------------------------------------------------------------
def _chunk(n_values: int, n_draws: int) -> int:
    return int(max(1, min(n_draws, _RESAMPLE_CELLS // max(n_values, 1))))


def sign_flip_p(x: NDArray, *, n_flips: int = 10_000, seed: Seed) -> float:
    """Two-sided sign-flip p of mean(x) = 0, ``(hits + 1) / (n_flips + 1)``.

    Non-finite values are dropped; NaN when none is left.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return float("nan")
    stream = keyed_rng(seed, "flip")
    observed = abs(x.mean())
    hits = 0
    chunk = _chunk(n, n_flips)
    for k in range(0, n_flips, chunk):
        b = min(chunk, n_flips - k)
        signs = stream.integers(0, 2, size=(b, n), dtype=np.int8) * 2 - 1
        hits += int((np.abs(signs @ x) / n >= observed * (1 - 1e-12)).sum())
    return (hits + 1) / (n_flips + 1)


def stratified_bootstrap_ci(
    x: NDArray, strata: NDArray, *, n_boot: int = 10_000, seed: Seed
) -> tuple[float, float]:
    """95 % percentile bootstrap CI of the pooled mean, resampling within strata.

    Non-finite values are dropped; NaNs when fewer than two are left.
    """
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    x, strata = x[finite], np.asarray(strata)[finite]
    if x.size < 2:
        return (float("nan"), float("nan"))
    stream = keyed_rng(seed, "boot")
    parts = [x[strata == g] for g in pd.unique(strata)]
    means = np.zeros(n_boot)
    chunk = _chunk(x.size, n_boot)
    for k in range(0, n_boot, chunk):
        b = min(chunk, n_boot - k)
        total = np.zeros(b)
        for part in parts:
            total += part[stream.integers(0, part.size, size=(b, part.size))].sum(1)
        means[k : k + b] = total / x.size
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def _slope(p: NDArray, x: NDArray) -> NDArray:
    centred = p - p.mean(-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        result: NDArray = (centred * (x - x.mean(-1, keepdims=True))).sum(-1) / (
            centred**2
        ).sum(-1)
    return result


def ols_slope_ci(
    p: NDArray, x: NDArray, strata: NDArray, *, n_boot: int = 2000, seed: Seed
) -> tuple[float, float, float]:
    """Window-level OLS slope of ``x`` on ``p``, with a 95 % bootstrap CI.

    Windows are resampled within strata. Pairs with a non-finite value are
    dropped.

    Returns
    -------
    slope, ci_lo, ci_hi : float
    """
    p, x = np.asarray(p, dtype=float), np.asarray(x, dtype=float)
    finite = np.isfinite(p) & np.isfinite(x)
    p, x, strata = p[finite], x[finite], np.asarray(strata)[finite]
    if p.size < 3:
        return (float("nan"),) * 3
    estimate = float(_slope(p, x))
    stream = keyed_rng(seed, "slope")
    by_stratum = [np.flatnonzero(strata == g) for g in pd.unique(strata)]
    slopes = np.zeros(n_boot)
    chunk = _chunk(p.size, n_boot)
    for k in range(0, n_boot, chunk):
        b = min(chunk, n_boot - k)
        idx = np.concatenate(
            [g[stream.integers(0, g.size, size=(b, g.size))] for g in by_stratum],
            axis=1,
        )
        slopes[k : k + b] = _slope(p[idx], x[idx])
    lo, hi = np.nanpercentile(slopes, [2.5, 97.5])
    return estimate, float(lo), float(hi)


def _ratio_ci(
    num: NDArray, den: NDArray, strata: NDArray, *, n_boot: int = 2000, seed: Seed
) -> tuple[float, float]:
    """95 % bootstrap CI of sum(num) / sum(den), resampling within strata."""
    stream = keyed_rng(seed, "ratio")
    parts = [(num[strata == g], den[strata == g]) for g in pd.unique(strata)]
    out = np.zeros(n_boot)
    chunk = _chunk(num.size, n_boot)
    for k in range(0, n_boot, chunk):
        b = min(chunk, n_boot - k)
        top, bottom = np.zeros(b), np.zeros(b)
        for nu, de in parts:
            idx = stream.integers(0, nu.size, size=(b, nu.size))
            top += nu[idx].sum(1)
            bottom += de[idx].sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[k : k + b] = np.where(bottom > 0, top / bottom, np.nan)
    lo, hi = np.nanpercentile(out, [2.5, 97.5])
    return float(lo), float(hi)


def _describe(values: NDArray, prefix: str = "") -> dict[str, float]:
    """Median, quartiles, share > 0 and mean of the finite values (NaN if none)."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        names = ("median", "q25", "q75", "frac_pos", "mean")
        return {f"{prefix}{name}": float("nan") for name in names}
    q25, median, q75 = np.percentile(finite, [25, 50, 75])
    return {
        f"{prefix}median": float(median),
        f"{prefix}q25": float(q25),
        f"{prefix}q75": float(q75),
        f"{prefix}frac_pos": float(np.mean(finite > 0)),
        f"{prefix}mean": float(finite.mean()),
    }


def _consistency(
    values: pd.Series, baseline: pd.Series, groups: pd.Series
) -> tuple[str, str]:
    """How many groups have a positive mean, and a mean above their baseline's."""
    mean = values.groupby(groups).mean()
    base = baseline.groupby(groups).mean()
    return (
        f"{int((mean > 0).sum())}/{len(mean)}",
        f"{int((mean > base).sum())}/{len(mean)}",
    )


def bout_statistics(
    windows: pd.DataFrame,
    *,
    metrics: Mapping[str, Sequence[str]] | None = None,
    n_flips: int = 10_000,
    n_boot: int = 10_000,
    seed: Seed = SUMMARY_SEED,
) -> pd.DataFrame:
    """Bouts against their matched windows, per state x regime x metric.

    Bouts are pooled over epochs. Each metric gets its median [IQR], share
    > 0, mean with a two-sided sign-flip p and a 95 % bootstrap CI (resampled
    within epochs), evidence per held-out spike (``{metric}_ps`` when present),
    the paired bout - baseline difference with the same tests, and the count of
    epochs and of animals whose mean is > 0 and above their baseline's.

    Parameters
    ----------
    windows : pd.DataFrame
        The per-window table: ``epoch_id``, ``animal``, ``kind``, ``state``,
        ``regime``, ``bout_s``, ``matched_bout_s``, ``n_found`` and the metrics.
    metrics : mapping, optional
        State -> metric columns; default each state's :data:`PRIMARY_METRICS`
        and every :data:`SECONDARY_METRICS`.

    Raises
    ------
    ValueError
        A bout has no matched window.
    """
    if metrics is None:
        metrics = {
            state: (*PRIMARY_METRICS[state], *SECONDARY_METRICS) for state in STATES
        }
    keys = ["epoch_id", "state", "regime"]
    bouts = windows[windows.kind == "bout"]
    paired = bouts.merge(
        windows[windows.kind == "baseline"],
        left_on=[*keys, "bout_s"],
        right_on=[*keys, "matched_bout_s"],
        suffixes=("", "_base"),
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if (paired["_merge"] != "both").any():
        raise ValueError("a bout lacks its matched window")
    rows = []
    for state, state_metrics in metrics.items():
        for regime in REGIMES:
            cell = paired[(paired.state == state) & (paired.regime == regime)]
            found = bouts[(bouts.state == state) & (bouts.regime == regime)]
            n_found = int(found.groupby("epoch_id").n_found.first().sum())
            for metric in state_metrics:
                key = (state, regime, metric)
                row: dict[str, Any] = {
                    "state": state,
                    "regime": regime,
                    "metric": metric,
                    "primary": metric in PRIMARY_METRICS.get(state, ()),
                    "n": len(cell),
                    "n_found": n_found,
                    "n_epochs": int(cell.epoch_id.nunique()),
                }
                if len(cell) == 0:
                    rows.append(row)
                    continue
                x = cell[metric].to_numpy(dtype=float)
                xb = cell[f"{metric}_base"].to_numpy(dtype=float)
                diff = x - xb
                epochs = cell.epoch_id.to_numpy()
                row.update(_describe(x))
                row["p_signflip"] = sign_flip_p(
                    x, n_flips=n_flips, seed=_with_keys(seed, *key)
                )
                row["ci_lo"], row["ci_hi"] = stratified_bootstrap_ci(
                    x, epochs, n_boot=n_boot, seed=_with_keys(seed, *key)
                )
                row.update(_describe(xb, prefix="base_"))
                row["diff_mean"] = _describe(diff)["mean"]
                row["p_diff"] = sign_flip_p(
                    diff, n_flips=n_flips, seed=_with_keys(seed, *key, "diff")
                )
                row["diff_ci_lo"], row["diff_ci_hi"] = stratified_bootstrap_ci(
                    diff, epochs, n_boot=n_boot, seed=_with_keys(seed, *key, "diff")
                )
                if f"{metric}_ps" in cell:
                    ps = _describe(cell[f"{metric}_ps"].to_numpy(dtype=float))
                    base_ps = _describe(cell[f"{metric}_ps_base"].to_numpy(dtype=float))
                    row.update(
                        ps_median=ps["median"],
                        ps_mean=ps["mean"],
                        base_ps_median=base_ps["median"],
                    )
                row["epochs_pos"], row["epochs_above_base"] = _consistency(
                    cell[metric], cell[f"{metric}_base"], cell.epoch_id
                )
                row["animals_pos"], row["animals_above_base"] = _consistency(
                    cell[metric], cell[f"{metric}_base"], cell.animal
                )
                for column, name in [
                    ("n_spikes", "median_n_spikes"),
                    ("n_bins", "median_bins"),
                    ("bout_len", "median_bout_bins"),
                ]:
                    if column in cell:
                        row[name] = float(cell[column].median())
                rows.append(row)
    return pd.DataFrame(rows)


def calibration_statistics(
    windows: pd.DataFrame,
    *,
    metrics: Mapping[str, Sequence[str]] | None = None,
    n_boot: int = 2000,
    seed: Seed = SUMMARY_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calibration curves: evidence against each state's probability.

    Calibration windows are binned by their mean posterior of the state (0-0.2,
    ..., 0.8-1.0); each bin has n, mean with a 95 % bootstrap CI (resampled
    within epochs), median and share > 0. A curve's trend is the window-level
    OLS slope with a bootstrap CI. It **rises** when the slope's CI is above 0
    and the top bin's CI is above the bottom bin's; its **top bin is > 0** when
    that CI is above 0. Bins with fewer than :data:`MIN_BIN_WINDOWS` windows
    are not compared.

    Parameters
    ----------
    windows : pd.DataFrame
        Needs ``kind``, ``regime``, ``epoch_id``, ``p_{state}`` and the metrics.
    metrics : mapping, optional
        State -> metric columns; default :data:`CALIBRATION_METRICS`.

    Returns
    -------
    bins, trends : pd.DataFrame
    """
    if metrics is None:
        metrics = CALIBRATION_METRICS
    calibration = windows[windows.kind == "calib"]
    bins, trends = [], []
    for regime in REGIMES:
        cell = calibration[calibration.regime == regime]
        epochs = cell.epoch_id.to_numpy()
        for state, state_metrics in metrics.items():
            p = cell[f"p_{state}"].to_numpy(dtype=float)
            label = pd.cut(
                cell[f"p_{state}"],
                PROBABILITY_BINS,
                right=False,
                labels=PROBABILITY_BIN_LABELS,
            ).to_numpy()
            for metric in state_metrics:
                x = cell[metric].to_numpy(dtype=float)
                curve = []
                for name in PROBABILITY_BIN_LABELS:
                    m = (label == name) & np.isfinite(x)
                    row: dict[str, Any] = {
                        "regime": regime,
                        "state": state,
                        "metric": metric,
                        "pbin": name,
                        "n": int(m.sum()),
                    }
                    if m.sum() >= 2:
                        row.update(
                            mean=float(x[m].mean()),
                            median=float(np.median(x[m])),
                            frac_pos=float((x[m] > 0).mean()),
                        )
                        row["ci_lo"], row["ci_hi"] = stratified_bootstrap_ci(
                            x[m],
                            epochs[m],
                            n_boot=n_boot,
                            seed=_with_keys(seed, regime, state, metric, name),
                        )
                    curve.append(row)
                bins += curve
                slope, lo, hi = ols_slope_ci(
                    p,
                    x,
                    epochs,
                    n_boot=n_boot,
                    seed=_with_keys(seed, regime, state, metric),
                )
                trend: dict[str, Any] = {
                    "regime": regime,
                    "state": state,
                    "metric": metric,
                    "n": int(np.isfinite(x).sum()),
                    "slope": slope,
                    "slope_ci_lo": lo,
                    "slope_ci_hi": hi,
                }
                bottom, top = curve[0], curve[-1]
                compared = (
                    bottom["n"] >= MIN_BIN_WINDOWS and top["n"] >= MIN_BIN_WINDOWS
                )
                separated = compared and top["ci_lo"] > bottom["ci_hi"]
                trend.update(
                    n_top=top["n"],
                    top_mean=top.get("mean", np.nan),
                    top_minus_bottom=(
                        top["mean"] - bottom["mean"] if compared else np.nan
                    ),
                    ci_separated=bool(separated),
                    rises=bool(separated and lo > 0),
                    top_positive=bool(top["n"] >= MIN_BIN_WINDOWS and top["ci_lo"] > 0),
                )
                trends.append(trend)
    return pd.DataFrame(bins), pd.DataFrame(trends)


def rate_calibration(windows: pd.DataFrame) -> pd.DataFrame:
    """Observed / expected held-out spikes in the calibration windows.

    Per regime and encoding, pooled (``animal == "all"``) and per animal:
    C's spike count summed over windows against E_C summed over windows,
    with the observed and predicted shares of silent C.
    """
    calibration = windows[windows.kind == "calib"]
    rows = []
    for regime in REGIMES:
        cell = calibration[calibration.regime == regime]
        for prefix, label in ENCODING_LABELS.items():
            groups = [("all", cell)] + [
                (str(animal), g) for animal, g in cell.groupby("animal", sort=True)
            ]
            for animal, g in groups:
                observed = float(g.run_ns_count.sum())
                expected = float(g[f"{prefix}_ns_expected"].sum())
                rows.append(
                    {
                        "regime": regime,
                        "encoding": label,
                        "animal": animal,
                        "n_windows": len(g),
                        "observed_spikes": observed,
                        "expected_spikes": expected,
                        "observed_over_expected": (
                            observed / expected if expected > 0 else np.nan
                        ),
                        "obs_silent_share": float(g.run_ns_silent_frac.mean()),
                        "pred_silent_share": float(
                            g[f"{prefix}_ns_pred_silent"].mean()
                        ),
                    }
                )
    return pd.DataFrame(rows)


def coordinated_silence(
    windows: pd.DataFrame, *, n_boot: int = 2000, seed: Seed = SUMMARY_SEED
) -> pd.DataFrame:
    """Held-out coordinated silence: P(C silent | S silent) against local coding.

    Pooled over windows and splits (a ratio of sums), for each state's bouts
    and matched windows, and for the calibration windows by P(No-Spike) bin and
    all together. Local coding makes the halves independent given position, so
    its prediction for P(C silent | S silent) is exp(-E_C) averaged over the
    S-silent runs, under either encoding.
    """
    groups: list[tuple[tuple[str, str, str, str], pd.DataFrame]] = []
    for regime in REGIMES:
        for kind in ("bout", "baseline"):
            for state in STATES:
                d = windows[
                    (windows.kind == kind)
                    & (windows.state == state)
                    & (windows.regime == regime)
                ]
                groups.append(((regime, kind, state, "all"), d))
        cell = windows[(windows.kind == "calib") & (windows.regime == regime)]
        label = pd.cut(
            cell["p_No-Spike"],
            PROBABILITY_BINS,
            right=False,
            labels=PROBABILITY_BIN_LABELS,
        ).to_numpy()
        for name in PROBABILITY_BIN_LABELS:
            groups.append(
                ((regime, "calib", "P(No-Spike) bin", name), cell[label == name])
            )
        groups.append(((regime, "calib", "all windows", "all"), cell))
    rows = []
    for (regime, kind, state, name), d in groups:
        if len(d) == 0:
            continue
        silent_s = d.run_cond_S_silent.to_numpy(dtype=float)
        both = d.run_cond_both_silent.to_numpy(dtype=float)
        row: dict[str, Any] = {
            "regime": regime,
            "kind": kind,
            "state": state,
            "pbin": name,
            "n_windows": len(d),
            "runs_S_silent": int(silent_s.sum()),
            "obs_C_silent": float(d.run_ns_silent_frac.mean()),
            "pred_C_silent_run": float(d.run_ns_pred_silent.mean()),
            "pred_C_silent_can": float(d.can_ns_pred_silent.mean()),
        }
        if silent_s.sum() > 0:
            row["obs_C_silent_given_S_silent"] = float(both.sum() / silent_s.sum())
            for prefix in ENCODING_LABELS:
                row[f"pred_given_S_{prefix}"] = float(
                    d[f"{prefix}_cond_pred_C_silent"].sum() / silent_s.sum()
                )
            if (silent_s > 0).sum() >= 5:
                row["obs_given_ci_lo"], row["obs_given_ci_hi"] = _ratio_ci(
                    both,
                    silent_s,
                    d.epoch_id.to_numpy(),
                    n_boot=n_boot,
                    seed=_with_keys(seed, regime, kind, state, name),
                )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_state_evidence(
    windows: pd.DataFrame, *, seed: Seed = SUMMARY_SEED
) -> dict[str, pd.DataFrame]:
    """Every summary of the per-window table, pooled over its epochs.

    ``run_loc_vs_Sremote_mark`` (and ``_ps``) are added as the negatives of
    ``run_nlr_anim_mark``, so that positive supports the Local claim.

    Returns
    -------
    dict
        ``bouts`` (:func:`bout_statistics`), ``calibration`` and ``trends``
        (:func:`calibration_statistics`), ``rate_calibration`` and
        ``silence`` (:func:`coordinated_silence`).
    """
    windows = windows.assign(
        run_loc_vs_Sremote_mark=-windows["run_nlr_anim_mark"],
        run_loc_vs_Sremote_mark_ps=-windows["run_nlr_anim_mark_ps"],
    )
    bins, trends = calibration_statistics(windows, seed=seed)
    return {
        "bouts": bout_statistics(windows, seed=seed),
        "calibration": bins,
        "trends": trends,
        "rate_calibration": rate_calibration(windows),
        "silence": coordinated_silence(windows, seed=seed),
    }
