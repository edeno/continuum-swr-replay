"""Speed regimes and the windows scored: bouts, matched windows, calibration windows.

Also the tetrode halves and the null partners, and the keyed random streams
every draw comes from.
"""

from __future__ import annotations

import zlib
from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

STATES = ("Local", "No-Spike", "Non-Local Continuous", "Non-Local Fragmented")
"""The decoder's states, in its own order."""

REGIMES = ("immobile", "running")
IMMOBILE_BELOW_CM_S = 4.0
RUNNING_ABOVE_CM_S = 10.0

Seed = int | Sequence[int | str]
"""A random stream's name: a base seed, optionally followed by keys."""

HALVES_SEED = 20261003
"""The tetrode partitions' seed (:func:`tetrode_halves`), the prototype's: its
pilot splits are these. The half decodes use the first two partitions."""


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


def window_folds(windows: pd.DataFrame, folds: NDArray) -> pd.DataFrame:
    """Each window's own fold and how much of it its scoring model was fit on.

    A window belongs to the fold of its centre bin, ``s + n_bins // 2`` (the
    prototype's rule), and is scored by the time-held-out encoding fit on the
    other fold (:func:`~.evidence.held_out_evidence`). A window that straddles
    a block edge has some bins in that model's blocks.

    Parameters
    ----------
    windows : pd.DataFrame
        ``s``, ``e`` (inclusive decode bins).
    folds : ndarray of int, shape (n_time,)
        From :func:`~.terms.time_folds`.

    Returns
    -------
    pd.DataFrame
        Indexed like ``windows``: ``cv_fold`` (int8) and ``cv_insample_share``,
        the share of the window's bins in the scoring model's fold.
    """
    s = windows["s"].to_numpy(dtype=np.int64)
    e = windows["e"].to_numpy(dtype=np.int64)
    folds = np.asarray(folds, dtype=np.int8)
    n_bins = e - s + 1
    own = folds[s + n_bins // 2]
    in_fold_1 = np.r_[0, np.cumsum(folds == 1)]
    n_fold_1 = in_fold_1[e + 1] - in_fold_1[s]
    other = np.where(own == 0, n_fold_1, n_bins - n_fold_1)
    return pd.DataFrame(
        {"cv_fold": own, "cv_insample_share": other / n_bins}, index=windows.index
    )
