"""Held-out evidence of every window, by tetrode half (see the package notes)."""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp
from numpy.typing import NDArray

from continuum_swr_replay.analysis.hippocampus.state_evidence.terms import TetrodeTerms
from continuum_swr_replay.analysis.hippocampus.state_evidence.windows import (
    Seed,
    null_partners,
)

_PER_RUN_SUMMED = ("cond_S_silent", "cond_both_silent", "cond_pred_C_silent")
_LEFT_OUT_COLUMNS = ("mapS_left_out_runs", "mapSr_left_out_runs")
"""Present only when place bins are left out; summed over runs like ``cond_*``."""


def held_out_evidence(
    per_fold: Sequence[pd.DataFrame], window_fold: NDArray
) -> pd.DataFrame:
    """Each window's evidence from the time-held-out model of the other fold.

    The prototype's ``runcv`` columns: every window is scored by both folds'
    models (:func:`window_evidence`, with the same windows, partners and
    halves), and keeps the score of the model fit without its own block.

    Parameters
    ----------
    per_fold : sequence of two pd.DataFrame
        ``per_fold[f]``: every window's evidence under the model fit on fold f;
        indexed and columned alike.
    window_fold : ndarray of int, shape (n_windows,)
        Each window's own fold, 0 or 1 (:func:`~.windows.window_folds`).

    Returns
    -------
    pd.DataFrame
        Row i from ``per_fold[1 - window_fold[i]]``.
    """
    first, second = per_fold
    window_fold = np.asarray(window_fold)
    if not np.isin(window_fold, (0, 1)).all():
        raise ValueError("every window's fold must be 0 or 1")
    if not (first.index.equals(second.index) and first.columns.equals(second.columns)):
        raise ValueError(
            "the two folds' evidence must cover the same windows and columns"
        )
    held_out = second.copy()  # fold-0 windows take fold 1's model
    in_fold_1 = window_fold == 1
    held_out.loc[in_fold_1] = first.loc[in_fold_1]
    return held_out


_PER_SPIKE = ("loc_mark", "loc_full", "nl_anim_mark", "nl_null_mark", "nlr_anim_mark")


def window_evidence(
    terms: TetrodeTerms,
    windows: pd.DataFrame,
    animal_bins: NDArray,
    track_distance: NDArray,
    halves: Sequence[tuple[NDArray, NDArray]],
    *,
    partner_min_cm: float = 20.0,
    n_partners: int = 20,
    both_directions: bool = True,
    left_out_bins: NDArray | None = None,
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
    both_directions : bool, optional
        True (default): each pair is two runs, each half selecting once. False:
        one run per pair, its first half selecting and its second scoring, for
        labels decoded from the first half alone (Task 13's half decodes), which
        must not score them.
    left_out_bins : ndarray of bool, shape (n_place_bins,), optional
        Place bins that are no candidate for MAP_S or the best remote bin,
        :func:`~.terms.subnormal_occupancy_bin_mask`'s: bins where the CPU
        rebuild of the likelihood is not the decode's. The choice is the best
        remaining bin; the result is that of a model without those bins. The
        animal must not be in one. None, or all False, changes nothing.
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
        ``animal_center_bin`` and ``has_loc_partner``. With bins left out, also
        ``mapS_left_out_runs`` and ``mapSr_left_out_runs``: the runs in which
        MAP_S, or the best remote bin, would have been a left-out bin without
        the rule (the window is scored all the same, on the remaining bins).

    Raises
    ------
    ValueError
        An input does not fit the others, ``left_out_bins`` is not one flag per
        place bin or leaves out every bin, or a window's animal is in a left-out
        bin.
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
    left_out: NDArray | None = None
    if left_out_bins is not None:
        flags = np.asarray(left_out_bins, dtype=bool)
        if flags.shape != (n_pos,):
            raise ValueError(
                f"left_out_bins needs one flag per place bin ({n_pos}); "
                f"got shape {flags.shape}"
            )
        if flags.all():
            raise ValueError("left_out_bins leaves out every place bin")
        left_out = flags if flags.any() else None
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
    if left_out is not None and left_out[animal_row].any():
        raise ValueError(
            "a window's animal is in a left-out place bin, so the evidence there "
            "cannot be scored"
        )
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
        *(_LEFT_OUT_COLUMNS if left_out is not None else ()),
    ]  # fmt: skip
    directions = ((0, 1), (1, 0)) if both_directions else ((0, 1),)
    values = {
        name: np.full((len(directions) * len(halves), n_win), np.nan) for name in names
    }
    first_choice: dict[str, NDArray] = {}
    tetrode_ids = np.arange(n_tet)
    base = rows * n_pos  # each concatenated row's offset into a flattened half
    for partition, pair in enumerate(halves):
        masks = [np.isin(tetrode_ids, half) for half in pair]
        sums = [half_terms(mask) for mask in masks]
        # run 2p: the first half selects and the second scores; run 2p + 1 swaps
        # (one direction: run p, the first half selecting)
        for direction, (si, ci) in enumerate(directions):
            run = len(directions) * partition + direction
            (sel_mark, sel_ground), (sco_mark, sco_ground) = sums[si], sums[ci]
            pooled = (
                np.asarray(window_sum @ sel_mark) - lens[:, None] * sel_ground[None, :]
            )
            remote_pooled = np.where(near_animal, -np.inf, pooled)
            choice, remote_choice = pooled.argmax(1), remote_pooled.argmax(1)
            if left_out is not None:
                # marked where the rule moved the choice; then the best other bin
                would_be_left_out = (left_out[choice], left_out[remote_choice])
                choice = np.where(left_out, -np.inf, pooled).argmax(1)
                remote_choice = np.where(left_out, -np.inf, remote_pooled).argmax(1)
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
            if left_out is not None:
                run_values["mapS_left_out_runs"] = would_be_left_out[0].astype(float)
                run_values["mapSr_left_out_runs"] = would_be_left_out[1].astype(float)
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
            summed = name in _PER_RUN_SUMMED or name in _LEFT_OUT_COLUMNS
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
