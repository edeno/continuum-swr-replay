"""Held-out spike evidence per decoder state (``analysis/event_evidence/state_evidence``).

Every input is synthetic. The per-tetrode terms are checked against the
library's own clusterless likelihood on a fitted KDE encoding model; the window
evidence is checked against an independent per-window loop written here, and
on terms planted so that the answer is known.
"""

from __future__ import annotations

import warnings
from itertools import chain, product
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from non_local_detector.likelihoods.clusterless_kde import (
    predict_clusterless_kde_log_likelihood,
)
from scipy.stats import t as student_t
from scipy.stats import ttest_1samp

from continuum_swr_replay.analysis.event_evidence.state_evidence import (
    CALIBRATION_METRICS,
    ENCODING_LABELS,
    PRIMARY_METRICS,
    SECONDARY_METRICS,
    STATES,
    UNDERFLOW_OCCUPANCY_FLOOR,
    TetrodeTerms,
    animal_t_test,
    bout_animal_summaries,
    bout_statistics,
    build_windows,
    calibration_statistics,
    calibration_windows,
    coordinated_silence,
    epoch_summaries,
    epoch_summary_kind,
    held_out_evidence,
    interior_bin_mask,
    keyed_rng,
    matched_windows,
    nearest_interior_bins,
    nested_mean_intervals,
    null_partners,
    ols_slope_ci,
    per_tetrode_terms,
    rate_calibration,
    regimes,
    sign_flip_p,
    state_bouts,
    stratified_bootstrap_ci,
    subnormal_occupancy_bin_mask,
    summarize_state_evidence,
    tetrode_halves,
    time_folds,
    training_spike_bins,
    window_evidence,
    window_folds,
)
from continuum_swr_replay.analysis.shared_statistics import (
    NestedDraw,
    nested_bootstrap_draws,
    nested_codes,
)
from continuum_swr_replay.decoding.encoding_model import CLUSTERLESS_LIKELIHOOD_KEYS
from tests.support._memoryless_clusterless_epoch import build_simulated_epoch

BIN_CM = 2.0
NO_SPIKE_RATE = 1e-10


# --- synthetic inputs ----------------------------------------------------------
def linear_distance(n_pos: int) -> np.ndarray:
    """Track distance between bins of a straight track, ``BIN_CM`` apart."""
    centers = np.arange(n_pos) * BIN_CM
    return np.abs(centers[:, None] - centers[None, :])


def random_terms(
    n_time: int, n_pos: int, n_tetrodes: int, rate: float, seed: int
) -> TetrodeTerms:
    """Poisson spikes on every tetrode, random mark terms and ground."""
    rng = np.random.default_rng(seed)
    counts = rng.poisson(rate, size=(n_tetrodes, n_time))
    time_bin = np.concatenate([np.repeat(np.arange(n_time), c) for c in counts])
    tetrode = np.repeat(np.arange(n_tetrodes), counts.sum(1))
    return TetrodeTerms(
        ground=rng.uniform(0.001, 0.05, size=(n_tetrodes, n_pos)),
        log_intensity=rng.normal(-6.0, 2.0, size=(time_bin.size, n_pos)).astype(
            np.float32
        ),
        time_bin=time_bin,
        tetrode=tetrode,
        bin_s=0.002,
    )


def without_spikes(terms: TetrodeTerms, first: int, last: int) -> TetrodeTerms:
    """The same terms with every spike in bins ``first`` to ``last`` removed."""
    keep = (terms.time_bin < first) | (terms.time_bin > last)
    return TetrodeTerms(
        ground=terms.ground,
        log_intensity=terms.log_intensity[keep],
        time_bin=terms.time_bin[keep],
        tetrode=terms.tetrode[keep],
        bin_s=terms.bin_s,
    )


SILENT = (100, 159)
"""Bins with no spike in the loop-equivalence test."""


def overlapping_windows(n_time: int, seed: int) -> pd.DataFrame:
    """Windows of three pools, of mixed lengths, some overlapping each other."""
    rng = np.random.default_rng(seed)
    rows = []
    for pool, n in [("bout|A|immobile", 9), ("baseline|A|immobile", 7), ("calib", 6)]:
        for _ in range(n):
            length = int(rng.integers(3, 15))
            s = int(rng.integers(0, n_time - length))
            rows.append({"pool": pool, "s": s, "e": s + length - 1})
    # inside and across the silent stretch, so both halves are silent at times
    for s, e in [(105, 112), (120, 129), (150, 165), (95, 104)]:
        rows.append({"pool": "bout|A|immobile", "s": s, "e": e})
    # a pool of one window has no partner at all
    rows.append({"pool": "lonely", "s": 10, "e": 14})
    return pd.DataFrame(rows)


def reference_evidence(
    terms: TetrodeTerms,
    windows: pd.DataFrame,
    animal_bins: np.ndarray,
    distance: np.ndarray,
    halves: list,
    nl_partner: np.ndarray,
    loc_partner: np.ndarray,
    *,
    min_cm: float,
    rho: float,
) -> pd.DataFrame:
    """The evidence, one window and one run at a time, straight from the method.

    Written without the vectorized code's union index, sparse sums or
    reductions: each half's log-likelihood is rebuilt per window from its spikes.
    """
    li = terms.log_intensity.astype(np.float64)
    n_pos = li.shape[1]
    s_all, e_all = windows.s.to_numpy(), windows.e.to_numpy()
    center = animal_bins[s_all + (e_all - s_all + 1) // 2]

    def mark_part(half: np.ndarray, bins: np.ndarray) -> np.ndarray:
        out = np.zeros((bins.size, n_pos))
        for i, b in enumerate(bins):
            spikes = (terms.time_bin == b) & np.isin(terms.tetrode, half)
            out[i] = li[spikes].sum(0)
        return out

    def counts(half: np.ndarray, bins: np.ndarray) -> np.ndarray:
        return np.array(
            [
                [np.sum((terms.time_bin == b) & (terms.tetrode == t)) for b in bins]
                for t in half
            ]
        ).reshape(len(half), bins.size)

    runs = []
    for a, b in halves:  # half a selects and b scores, then the reverse
        runs += [(a, b), (b, a)]
    keys = [
        "loc_mark", "loc_full", "nl_anim_mark", "nl_anim_full", "nl_null_mark",
        "nl_null_full", "nlr_anim_mark", "nlr_anim_full", "ns_logratio",
        "ns_count", "ns_expected", "ns_silent_frac", "ns_pred_silent", "sel_count",
        "mapS_dist_cm", "mapSr_dist_cm", "cond_S_silent", "cond_both_silent",
        "cond_pred_C_silent", "loc_mark_ps", "loc_full_ps", "nl_anim_mark_ps",
        "nl_null_mark_ps", "nlr_anim_mark_ps",
    ]  # fmt: skip
    values = {k: np.full((len(runs), len(windows)), np.nan) for k in keys}
    for r, (sel, sco) in enumerate(runs):
        g_sel, g_sco = terms.ground[sel].sum(0), terms.ground[sco].sum(0)
        maps, maps_remote = [], []
        for w in range(len(windows)):
            bins = np.arange(s_all[w], e_all[w] + 1)
            pooled = mark_part(sel, bins).sum(0) - bins.size * g_sel
            maps.append(int(np.argmax(pooled)))
            remote = np.where(distance[center[w]] < min_cm, -np.inf, pooled)
            maps_remote.append(int(np.argmax(remote)))
        for w in range(len(windows)):
            bins = np.arange(s_all[w], e_all[w] + 1)
            mc = mark_part(sco, bins)
            rows = np.arange(bins.size)
            ab = animal_bins[bins]
            m, mr = maps[w], maps_remote[w]
            own, anim, rem = mc[:, m], mc[rows, ab], mc[:, mr]
            v = values
            v["nl_anim_mark"][r, w] = np.sum(own - anim)
            v["nl_anim_full"][r, w] = np.sum((own - g_sco[m]) - (anim - g_sco[ab]))
            v["nlr_anim_mark"][r, w] = np.sum(rem - anim)
            v["nlr_anim_full"][r, w] = np.sum((rem - g_sco[mr]) - (anim - g_sco[ab]))
            v["mapS_dist_cm"][r, w] = distance[m, center[w]]
            v["mapSr_dist_cm"][r, w] = distance[mr, center[w]]
            if nl_partner[w, 0] >= 0:
                pm = [maps[j] for j in nl_partner[w]]
                null_m = np.mean([mc[:, j] for j in pm], axis=0)
                null_g = np.mean([g_sco[j] for j in pm])
                v["nl_null_mark"][r, w] = np.sum(own - null_m)
                v["nl_null_full"][r, w] = np.sum((own - g_sco[m]) - (null_m - null_g))
            if loc_partner[w, 0] >= 0:
                lp = [center[j] for j in loc_partner[w]]
                null_m = np.mean([mc[:, j] for j in lp], axis=0)
                null_g = np.mean([g_sco[j] for j in lp])
                v["loc_mark"][r, w] = np.sum(anim - null_m)
                v["loc_full"][r, w] = np.sum((anim - g_sco[ab]) - (null_m - null_g))
            n_sco, n_sel = counts(sco, bins), counts(sel, bins)
            local = terms.ground[sco][:, ab]  # (|C|, n_bins)
            v["ns_logratio"][r, w] = np.sum(
                n_sco * (np.log(rho) - np.log(local)) + local - rho
            )
            v["ns_count"][r, w] = n_sco.sum()
            v["sel_count"][r, w] = n_sel.sum()
            v["ns_expected"][r, w] = local.sum()
            v["ns_silent_frac"][r, w] = float(n_sco.sum() == 0)
            v["ns_pred_silent"][r, w] = np.exp(-local.sum())
            s_silent = float(n_sel.sum() == 0)
            v["cond_S_silent"][r, w] = s_silent
            v["cond_both_silent"][r, w] = s_silent * (n_sco.sum() == 0)
            v["cond_pred_C_silent"][r, w] = s_silent * np.exp(-local.sum())
            if n_sco.sum() > 0:
                for key in ("loc_mark", "loc_full", "nl_anim_mark", "nl_null_mark"):
                    v[f"{key}_ps"][r, w] = v[key][r, w] / n_sco.sum()
                v["nlr_anim_mark_ps"][r, w] = v["nlr_anim_mark"][r, w] / n_sco.sum()
    out = {}
    with warnings.catch_warnings():  # a window with no partner is all NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        for key, a in values.items():
            summed = key.startswith("cond_")
            out[key] = np.nansum(a, 0) if summed else np.nanmean(a, 0)
        out["loc_mark_sd_runs"] = np.nanstd(values["loc_mark"], 0)
        out["nl_null_mark_sd_runs"] = np.nanstd(values["nl_null_mark"], 0)
    return pd.DataFrame(out, index=windows.index)


# --- per-tetrode terms -----------------------------------------------------------
def test_summed_per_tetrode_terms_reproduce_the_decoder_log_likelihood(tmp_path):
    epoch = build_simulated_epoch(tmp_path)
    environment = epoch.environment
    mask = interior_bin_mask(environment)
    terms = per_tetrode_terms(
        epoch.encoding_model,
        epoch.spike_times,
        epoch.spike_waveform_features,
        epoch.time,
        environment.place_bin_centers_[mask],
    )
    expected = np.asarray(
        predict_clusterless_kde_log_likelihood(
            epoch.time,
            epoch.time,
            epoch.linear_position[:, None],
            epoch.spike_times,
            epoch.spike_waveform_features,
            **{key: epoch.encoding_model[key] for key in CLUSTERLESS_LIKELIHOOD_KEYS},
            is_local=False,
            disable_progress_bar=True,
        )
    )
    mine = terms.log_likelihood(np.arange(epoch.time.size))

    assert mine.shape == expected.shape == (epoch.time.size, mask.sum())
    np.testing.assert_allclose(mine, expected, rtol=1e-5, atol=1e-3)
    # every spike inside the axis is binned the decoder's way, the epoch's
    # final timestamp included (np.digitize on time[1:-1] puts it in row n - 2)
    n_spikes = [
        np.sum((st >= epoch.time[0]) & (st <= epoch.time[-1]))
        for st in epoch.spike_times
    ]
    assert np.bincount(terms.tetrode, minlength=3).tolist() == n_spikes
    assert terms.time_bin.max() == epoch.time.size - 2
    np.testing.assert_allclose(
        terms.ground.sum(0),
        np.asarray(epoch.encoding_model["summed_ground_process_intensity"]),
        rtol=1e-5,
    )


def test_interior_bins_are_exactly_the_non_junction_bins(tmp_path):
    environment = build_simulated_epoch(tmp_path).environment
    mask = interior_bin_mask(environment)
    edge_id = environment.place_bin_centers_nodes_df_["edge_id"].to_numpy()

    assert (edge_id == -1).any(), "the fixture must have junction placeholder bins"
    np.testing.assert_array_equal(mask, edge_id != -1)


def test_a_junction_bin_is_never_chosen(tmp_path):
    environment = build_simulated_epoch(tmp_path).environment
    mask = interior_bin_mask(environment)
    edge_id = environment.place_bin_centers_nodes_df_["edge_id"].to_numpy()
    place_bin_of_column = np.flatnonzero(mask)
    n_pos = int(mask.sum())

    # positions over the whole linearized range, the junction gaps included,
    # map to real bins
    edges = np.asarray(environment.edges_[0])
    positions = np.linspace(edges[0], edges[-1], 400)
    columns = nearest_interior_bins(environment, positions)
    assert (edge_id[place_bin_of_column[columns]] != -1).all()

    # content planted on the real bins that border each junction gap: the
    # selecting half's choice is that bin, and a real one
    is_gap = edge_id == -1
    borders = np.flatnonzero(
        ~is_gap & (np.r_[is_gap[1:], False] | np.r_[False, is_gap[:-1]])
    )
    planted = np.searchsorted(place_bin_of_column, borders)
    n_time = 20 * planted.size
    time_bin = np.repeat(np.arange(n_time), 4)
    tetrode = np.tile(np.arange(4), n_time)
    li = np.full((time_bin.size, n_pos), -12.0, dtype=np.float32)
    li[np.arange(time_bin.size), np.repeat(planted, 80)] = 0.0
    terms = TetrodeTerms(
        ground=np.full((4, n_pos), 0.01),
        log_intensity=li,
        time_bin=time_bin,
        tetrode=tetrode,
        bin_s=0.002,
    )
    windows = pd.DataFrame({"s": np.arange(planted.size) * 20, "pool": "bout"}).assign(
        e=lambda d: d.s + 19
    )
    distance = np.asarray(
        environment.get_distances_to_interior_bins(environment.place_bin_centers_[mask])
    )
    animal = np.zeros(n_time, dtype=np.int64)
    out = window_evidence(
        terms, windows, animal, distance, tetrode_halves(4, n_partitions=2),
        no_spike_rate=NO_SPIKE_RATE, seed=1,
    )  # fmt: skip

    np.testing.assert_array_equal(out["run0_mapS"].to_numpy(), planted)
    assert (edge_id[place_bin_of_column[out["run0_mapS"].to_numpy()]] != -1).all()


# --- regimes and windows ----------------------------------------------------------
def test_time_folds_alternate_30_s_blocks_from_the_first_bin():
    time = 100.0 + np.arange(0.0, 95.0, 0.5)

    folds = time_folds(time)

    assert folds.dtype == np.int8
    np.testing.assert_array_equal(
        folds, np.floor((time - time[0]) / 30.0).astype(int) % 2
    )
    assert folds[time.searchsorted(129.5)] == 0 and folds[time.searchsorted(130.0)] == 1
    assert folds[time.searchsorted(160.0)] == 0


def test_training_spike_bins_are_the_bins_the_library_trains_on():
    """The library keeps spikes from one bin before to one bin after each run."""
    from non_local_detector.models.base import ClusterlessDetector

    def library_bins(time, position, is_training, spikes):
        valid = ~np.isnan(position).any(axis=1)
        kept, _ = ClusterlessDetector._get_group_spike_data(
            None, [spikes], [np.zeros((spikes.size, 1))], is_training[valid],
            time[valid],
        )  # fmt: skip
        return np.unique(np.digitize(kept[0], time[1:-1]))

    n = 400
    position = np.random.default_rng(17).normal(size=(n, 2))
    position[[50, 51, 200, 250], 0] = np.nan  # NaN positions, some inside runs
    is_training = np.zeros(n, dtype=bool)
    is_training[[*range(40, 60), *range(195, 205), 249, 251, 300, *range(350, n)]] = (
        True
    )
    # bins of 1/16 s keep the library's arithmetic exact; a spike at the start
    # and the middle of every bin
    time = 10.0 + np.arange(n) / 16
    spikes = np.sort(np.r_[time, time + 1 / 32])

    covered = training_spike_bins(time, position, is_training)

    np.testing.assert_array_equal(
        np.flatnonzero(covered), library_bins(time, position, is_training, spikes)
    )
    # one bin before a run, and the start of the one after it
    assert covered[39] and covered[60] and not covered[38] and not covered[61]
    # bins 249 and 251 are one run once the NaN-position bin 250 is dropped
    assert covered[250]
    # on the decoder's 2 ms grid, rounding can add a sliver of the bin before;
    # the bins stay a superset of the library's, by at most that bin
    time = 10.0 + np.arange(n) * 0.002
    spikes = np.sort(np.r_[time, time + 0.001])
    covered = np.flatnonzero(training_spike_bins(time, position, is_training))
    library = library_bins(time, position, is_training, spikes)
    extra = np.setdiff1d(covered, library)
    assert np.isin(library, covered).all() and np.isin(extra + 1, library).all()


def test_each_window_is_scored_by_the_fold_without_its_center_bin():
    folds = np.repeat([0, 1, 0, 1], 10).astype(np.int8)  # blocks of 10 bins
    windows = pd.DataFrame({"s": [0, 8, 15, 25], "e": [4, 12, 19, 34]})

    cv = window_folds(windows, folds)

    # center bins 2, 10, 17 and 30
    assert cv.cv_fold.tolist() == [0, 1, 1, 1]
    # the share of a window's bins in the blocks its scoring model was fit on
    assert cv.cv_insample_share.tolist() == pytest.approx([0.0, 2 / 5, 0.0, 5 / 10])
    first = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "k": [1, 1, 1, 1]})
    second = pd.DataFrame({"x": [10.0, 20.0, 30.0, 40.0], "k": [2, 2, 2, 2]})
    held_out = held_out_evidence([first, second], cv.cv_fold.to_numpy())
    # a window in fold 0 takes the model fit on fold 1, and the other way round
    assert held_out.x.tolist() == [10.0, 2.0, 3.0, 4.0]
    assert held_out.k.tolist() == [2, 1, 1, 1]
    with pytest.raises(ValueError, match="fold"):
        held_out_evidence([first, second], np.array([0, 1, 2, 0]))


def test_regimes_split_speed_at_four_and_ten_cm_per_s():
    speed = np.array([np.nan, 0.0, 3.99, 4.0, 7.0, 10.0, 10.01, 30.0])
    np.testing.assert_array_equal(
        regimes(speed),
        ["", "immobile", "immobile", "", "", "", "running", "running"],
    )


def test_bouts_respect_the_threshold_the_regime_and_the_minimum_length():
    p = np.zeros(60)
    p[2:9] = 0.9  # 7 bins, immobile
    p[12:16] = 0.95  # 4 bins: too short
    p[20:28] = 0.8  # exactly at the threshold: not a bout
    p[30:45] = 0.85  # 15 bins straddling a regime change
    p[50:56] = 0.99  # 6 bins in no regime
    regime = np.full(60, "immobile", dtype=object)
    regime[38:48] = "running"
    regime[48:] = ""

    bouts = state_bouts(p, regime, min_bins=5, seed=0)

    assert bouts[["regime", "bout_s", "bout_e"]].values.tolist() == [
        ["immobile", 2, 8],
        ["immobile", 30, 37],
        ["running", 38, 44],
    ]
    assert bouts.bout_len.tolist() == [7, 8, 7]
    assert bouts.full_len.tolist() == [7, 15, 15]
    assert bouts.clipped.tolist() == [False, True, True]
    assert bouts.n_found.tolist() == [2, 2, 1]
    for row in bouts.itertuples():
        assert (p[row.s : row.e + 1] > 0.8).all()
        assert (regime[row.s : row.e + 1] == row.regime).all()


def test_bouts_respect_the_cap_the_scored_length_and_the_seed():
    n_bouts, length = 40, 70
    p = np.zeros(n_bouts * (length + 5))
    starts = np.arange(n_bouts) * (length + 5)
    for s in starts:
        p[s : s + length] = 0.9
    regime = np.full(p.size, "immobile", dtype=object)

    a = state_bouts(p, regime, cap=15, max_scored_bins=50, seed=(3, "x"))
    b = state_bouts(p, regime, cap=15, max_scored_bins=50, seed=(3, "x"))
    c = state_bouts(p, regime, cap=15, max_scored_bins=50, seed=(4, "x"))
    uncapped = state_bouts(p, regime, cap=n_bouts, max_scored_bins=50, seed=(3, "x"))

    pd.testing.assert_frame_equal(a, b)
    assert len(a) == 15 and (a.n_found == n_bouts).all()
    assert set(a.bout_s) <= set(starts) and a.bout_s.is_monotonic_increasing
    assert set(a.bout_s) != set(c.bout_s)
    assert ((a.e - a.s + 1) == 50).all()
    assert ((a.s >= a.bout_s) & (a.e <= a.bout_e)).all()
    assert len(set(a.s - a.bout_s)) > 1  # the sub-window is drawn, not fixed
    # the scored sub-window is seeded by the bout's bounds, not by the sample
    assert len(uncapped) == n_bouts
    shared = a.merge(uncapped, on=["bout_s", "bout_e"], suffixes=("", "_all"))
    assert len(shared) == 15
    np.testing.assert_array_equal(shared.s, shared.s_all)


def test_matched_windows_share_the_regime_and_length_of_their_bout():
    rng = np.random.default_rng(5)
    regime = np.where(rng.random(3000) < 0.6, "immobile", "running").astype(object)
    regime[1000:1400] = "immobile"
    regime[2000:2400] = "running"
    p = np.zeros(3000)
    p[1010:1040] = p[1100:1108] = p[2050:2120] = 0.9
    bouts = state_bouts(p, regime, seed=1)

    base = matched_windows(bouts, regime, seed=(1, "e"))
    again = matched_windows(bouts, regime, seed=(1, "e"))

    pd.testing.assert_frame_equal(base, again)
    assert base.matched_bout_s.tolist() == bouts.bout_s.tolist()
    np.testing.assert_array_equal(base.e - base.s, bouts.e - bouts.s)
    for row in base.itertuples():
        assert (regime[row.s : row.e + 1] == row.regime).all()


def test_calibration_windows_are_seeded_and_inside_their_regime():
    rng = np.random.default_rng(2)
    regime = np.where(rng.random(5000) < 0.5, "immobile", "running").astype(object)
    regime[:2000] = "immobile"
    regime[3000:4000] = "running"

    cal = calibration_windows(regime, n_per_regime=50, n_bins=10, seed=(7, "e"))

    pd.testing.assert_frame_equal(
        cal, calibration_windows(regime, n_per_regime=50, n_bins=10, seed=(7, "e"))
    )
    assert cal.groupby("regime").size().to_dict() == {"immobile": 50, "running": 50}
    assert ((cal.e - cal.s + 1) == 10).all()
    for row in cal.itertuples():
        assert (regime[row.s : row.e + 1] == row.regime).all()
    # fewer valid windows than asked for: every one of them
    short = np.full(30, "running", dtype=object)
    few = calibration_windows(short, n_per_regime=50, n_bins=10, seed=0)
    assert few.regime.tolist() == ["running"] * 21


def test_tetrode_halves_partition_the_tetrodes_reproducibly():
    halves = tetrode_halves(7, n_partitions=10, seed=3)

    assert len(halves) == 10
    for a, b in halves:
        assert len(a) == 3 and len(b) == 4
        np.testing.assert_array_equal(np.sort(np.r_[a, b]), np.arange(7))
    again = tetrode_halves(7, n_partitions=10, seed=3)
    assert all(
        np.array_equal(x[0], y[0]) and np.array_equal(x[1], y[1])
        for x, y in zip(halves, again, strict=True)
    )
    # the prototype's draw: seed 20261003 gave peanut's 20 tetrodes these halves
    first = tetrode_halves(20, seed=20261003)[0][0]
    assert first.tolist() == [0, 2, 3, 7, 10, 12, 13, 15, 16, 19]


# --- partners and evidence --------------------------------------------------------
def test_a_partner_is_never_the_window_itself_and_is_far_enough():
    rng = np.random.default_rng(0)
    n = 300
    pool = rng.choice(["a", "b", "c"], n).astype(object)
    pool[0] = "alone"
    center = rng.integers(0, 40, n)
    distance = linear_distance(40)

    nl, loc = null_partners(pool, center, distance, n_partners=20, min_cm=20.0, seed=4)

    assert (nl[0] == -1).all() and (loc[0] == -1).all()
    for w in range(1, n):
        assert (nl[w] != w).all() and (pool[nl[w]] == pool[w]).all()
        if loc[w, 0] >= 0:
            assert (loc[w] != w).all() and (pool[loc[w]] == pool[w]).all()
            assert (distance[center[w], center[loc[w]]] >= 20.0).all()
        else:  # no window of its pool is 20 cm away
            others = (pool == pool[w]) & (np.arange(n) != w)
            assert (distance[center[w], center[others]] < 20.0).all()


def test_the_vectorized_evidence_matches_a_per_window_loop():
    n_time, n_pos, n_tet = 260, 25, 5
    terms = without_spikes(
        random_terms(n_time, n_pos, n_tet, rate=0.5, seed=11), *SILENT
    )
    windows = overlapping_windows(n_time, seed=12)
    animal = np.random.default_rng(13).integers(0, n_pos, n_time)
    distance = linear_distance(n_pos)
    halves = tetrode_halves(n_tet, n_partitions=3, seed=14)

    out = window_evidence(
        terms, windows, animal, distance, halves,
        n_partners=6, no_spike_rate=NO_SPIKE_RATE, seed=15,
    )  # fmt: skip
    center = animal[windows.s + (windows.e - windows.s + 1) // 2]
    nl, loc = null_partners(windows["pool"], center, distance, n_partners=6, seed=15)
    expected = reference_evidence(
        terms, windows, animal, distance, halves, nl, loc,
        min_cm=20.0, rho=NO_SPIKE_RATE * 0.002,
    )  # fmt: skip

    for column in expected.columns:
        np.testing.assert_allclose(
            out[column], expected[column], rtol=1e-9, atol=1e-6, err_msg=column
        )
    assert out.loc[windows.pool == "lonely", "nl_null_mark"].isna().all()
    # the silent windows exercise the No-Spike and coordinated-silence branches
    assert (expected.cond_both_silent > 0).sum() >= 2
    assert (expected.ns_count == 0).sum() >= 2
    np.testing.assert_array_equal(out.has_loc_partner, loc[:, 0] >= 0)
    counts = [
        np.sum((terms.time_bin >= s) & (terms.time_bin <= e))
        for s, e in zip(windows.s, windows.e, strict=True)
    ]
    np.testing.assert_array_equal(out.n_spikes, counts)


def test_run_averaged_local_and_no_spike_values_are_half_the_all_tetrode_values():
    """Not held out: every tetrode scores in exactly half of the runs."""
    n_time, n_pos, n_tet = 200, 20, 5
    terms = random_terms(n_time, n_pos, n_tet, rate=0.4, seed=21)
    windows = overlapping_windows(n_time, seed=22)
    animal = np.random.default_rng(23).integers(0, n_pos, n_time)
    distance = linear_distance(n_pos)

    out = window_evidence(
        terms, windows, animal, distance, tetrode_halves(n_tet, seed=24),
        n_partners=5, no_spike_rate=NO_SPIKE_RATE, seed=25,
    )  # fmt: skip
    center = animal[windows.s + (windows.e - windows.s + 1) // 2]
    _, loc = null_partners(windows["pool"], center, distance, n_partners=5, seed=25)
    rho = NO_SPIKE_RATE * 0.002
    everyone = np.arange(n_tet)
    li = terms.log_intensity.astype(np.float64)
    g_all = terms.ground.sum(0)
    for w, (s, e) in enumerate(zip(windows.s, windows.e, strict=True)):
        bins = np.arange(s, e + 1)
        in_window = (terms.time_bin >= s) & (terms.time_bin <= e)
        ab_spike = animal[terms.time_bin[in_window]]
        local = terms.ground[everyone][:, animal[bins]]
        counts = np.array(
            [[np.sum((terms.time_bin == b) & (terms.tetrode == t)) for b in bins]
             for t in everyone]
        )  # fmt: skip
        all_ns = np.sum(counts * (np.log(rho) - np.log(local)) + local - rho)
        assert out.ns_logratio[w] == pytest.approx(all_ns / 2, rel=1e-9)
        assert out.ns_count[w] == pytest.approx(counts.sum() / 2)
        assert out.ns_expected[w] == pytest.approx(local.sum() / 2, rel=1e-9)
        if loc[w, 0] < 0:
            continue
        partners = center[loc[w]]
        rows = np.arange(in_window.sum())
        mark = li[in_window][rows, ab_spike] - li[in_window][:, partners].mean(1)
        assert out.loc_mark[w] == pytest.approx(mark.sum() / 2, rel=1e-9)
        full = mark.sum() - np.sum(g_all[animal[bins]] - g_all[partners].mean())
        assert out.loc_full[w] == pytest.approx(full / 2, rel=1e-9)


def planted_terms(
    animal: np.ndarray, content: np.ndarray, n_pos: int, n_tet: int
) -> TetrodeTerms:
    """Two spikes per tetrode per bin, every one pointing at ``content[t]``."""
    n_time = animal.size
    time_bin = np.repeat(np.arange(n_time), 2 * n_tet)
    tetrode = np.tile(np.repeat(np.arange(n_tet), 2), n_time)
    li = np.full((time_bin.size, n_pos), -10.0, dtype=np.float32)
    li[np.arange(time_bin.size), content[time_bin]] = 0.0
    return TetrodeTerms(
        ground=np.full((n_tet, n_pos), 0.02),
        log_intensity=li,
        time_bin=time_bin,
        tetrode=tetrode,
        bin_s=0.002,
    )


def test_planted_content_gives_evidence_for_the_planted_claim():
    n_pos, n_tet, length = 40, 4, 10
    animal_at = np.array([2, 12, 22, 32, 5, 5, 5, 5])
    remote_at = np.array([2, 12, 22, 32, 30, 35, 39, 25])  # first four are local
    animal = np.repeat(animal_at, length)
    content = np.repeat(remote_at, length)
    terms = planted_terms(animal, content, n_pos, n_tet)
    windows = pd.DataFrame(
        {
            "s": np.arange(8) * length,
            "e": np.arange(8) * length + length - 1,
            "pool": ["local"] * 4 + ["remote"] * 4,
        }
    )

    out = window_evidence(
        terms, windows, animal, linear_distance(n_pos), tetrode_halves(n_tet, seed=2),
        n_partners=3, no_spike_rate=NO_SPIKE_RATE, seed=3,
    )  # fmt: skip
    local, remote = out[windows.pool == "local"], out[windows.pool == "remote"]

    assert (local.loc_mark > 0).all()
    assert np.allclose(local.nl_anim_mark, 0.0)
    assert (remote.nl_anim_mark > 0).all()
    assert (remote.nl_null_mark > 0).all()
    # each held-out spike is worth the planted 10 nats
    np.testing.assert_allclose(remote.nl_anim_mark_ps, 10.0)
    np.testing.assert_allclose(remote.mapS_dist_cm, BIN_CM * (remote_at[4:] - 5))


# --- the runner's chain, reproducibility and the summary ------------------------------
def synthetic_epoch_table(seed: int, epoch_id: str) -> pd.DataFrame:
    """One epoch's windows and evidence under three encodings, as the runner builds them."""
    rng = np.random.default_rng(99)
    n_time, n_pos, n_tet = 4000, 30, 6
    # 100-bin stretches of immobility, running and the speeds between them
    speed = np.repeat(rng.choice([1.0, 7.0, 20.0], n_time // 100), 100)
    raw = np.clip(rng.normal(0.0, 1.0, (n_time, len(STATES))), 0, None) ** 3
    raw = np.apply_along_axis(np.convolve, 0, raw, np.ones(5) / 5, mode="same")
    probabilities = raw / raw.sum(1, keepdims=True)
    windows = build_windows(
        probabilities, regimes(speed), cap=6, n_per_regime=40, seed=(seed, epoch_id)
    )
    animal = rng.integers(0, n_pos, n_time)
    halves = tetrode_halves(n_tet, n_partitions=4, seed=seed)
    columns = [windows]
    for prefix, terms_seed in [("runcv", 9), ("run", 7), ("can", 8)]:
        evidence = window_evidence(
            random_terms(n_time, n_pos, n_tet, rate=0.3, seed=terms_seed),
            windows, animal, linear_distance(n_pos), halves,
            n_partners=5, no_spike_rate=NO_SPIKE_RATE, seed=(seed, epoch_id),
        )  # fmt: skip
        shared = ["n_spikes", "n_tetrodes", "animal_center_bin", "has_loc_partner"]
        if prefix == "runcv":
            columns.append(evidence[shared])
        columns.append(evidence.drop(columns=shared).add_prefix(f"{prefix}_"))
    return pd.concat(columns, axis=1).assign(epoch_id=epoch_id, animal=epoch_id[:2])


def test_build_windows_labels_every_state_regime_and_kind():
    table = synthetic_epoch_table(seed=1, epoch_id="aa20200101_02_r1_HPC")

    bouts = table[table.kind == "bout"]
    base = table[table.kind == "baseline"]
    cal = table[table.kind == "calib"]
    assert set(bouts.state) <= set(STATES) and len(bouts) > 0
    assert (bouts.pool == "bout|" + bouts.state + "|" + bouts.regime).all()
    assert len(base) == len(bouts)
    np.testing.assert_array_equal(base.n_bins.to_numpy(), bouts.n_bins.to_numpy())
    assert cal.groupby("regime").size().tolist() == [40, 40]
    assert (cal.state == "").all() and (cal.pool == "calib||" + cal.regime).all()
    # each window carries every state's mean posterior over its bins, and a
    # bout's own state is above the threshold in every one of them
    p = table[[f"p_{state}" for state in STATES]].to_numpy()
    np.testing.assert_allclose(p.sum(1), 1.0)
    own = [row[f"p_{row['state']}"] for _, row in bouts.iterrows()]
    assert min(own) > 0.8


def test_fixed_seeds_give_identical_reruns():
    first = synthetic_epoch_table(seed=1, epoch_id="aa20200101_02_r1_HPC")
    again = synthetic_epoch_table(seed=1, epoch_id="aa20200101_02_r1_HPC")

    pd.testing.assert_frame_equal(first, again)
    other = synthetic_epoch_table(seed=2, epoch_id="aa20200101_02_r1_HPC")
    assert not other[["s", "e"]].equals(first[["s", "e"]])


def test_summarize_state_evidence_covers_every_state_and_table():
    table = pd.concat(
        [
            synthetic_epoch_table(seed=1, epoch_id="aa20200101_02_r1_HPC"),
            synthetic_epoch_table(seed=1, epoch_id="bb20200101_02_r1_HPC"),
        ],
        ignore_index=True,
    )

    summary = summarize_state_evidence(table)

    assert set(summary) == {
        "bouts",
        "calibration",
        "trends",
        "rate_calibration",
        "silence",
    }
    bouts = summary["bouts"]
    for row in bouts[bouts.primary].itertuples():
        cell = table[
            (table.kind == "bout")
            & (table.state == row.state)
            & (table.regime == row.regime)
        ]
        assert row.n == len(cell)
    assert {
        "epoch_summary", "animal_estimate", "animal_ci_lo", "animal_ci_hi",
        "p_animal_t", "animals_pos", "epochs_pos", "epoch_estimate",
        "epoch_ci_lo", "epoch_ci_hi", "p_epoch_signflip", "diff_animal_estimate",
        "diff_animal_ci_lo", "diff_p_animal_t", "diff_animals_pos",
        "other_animal_estimate", "other_p_animal_t", "other_diff_animal_ci_hi",
    } <= set(bouts.columns)  # fmt: skip
    for row in bouts.itertuples():
        assert row.epoch_summary == epoch_summary_kind(row.metric)
    # the primary metrics are scored by the time-held-out running-only encoding
    assert set(bouts[bouts.primary].metric) == {
        "runcv_loc_mark", "runcv_ns_logratio", "runcv_nl_null_mark", "runcv_nl_anim_mark",
    }  # fmt: skip
    assert {"zero_variance", "diff_zero_variance"} <= set(bouts.columns)
    assert set(bouts.epoch_summary) == {"median", "mean"}
    assert set(summary["trends"].state) == set(STATES)
    # the window-level statistics are opt-in
    for name in ("bouts", "calibration", "trends", "silence"):
        assert not [c for c in summary[name] if c.endswith("_window_level")], name
    full = summarize_state_evidence(table, window_level=True)
    for name, column in [
        ("bouts", "p_signflip_window_level"),
        ("calibration", "ci_lo_window_level"),
        ("trends", "slope_window_level"),
    ]:
        assert column in full[name], name
    pd.testing.assert_series_equal(full["bouts"].animal_ci_lo, bouts.animal_ci_lo)


# --- statistics ---------------------------------------------------------------------
def test_sign_flip_p_and_stratified_bootstrap_ci():
    rng = np.random.default_rng(0)
    shifted = rng.normal(1.0, 1.0, 200)
    half = rng.normal(0.0, 1.0, 100)
    centered = np.r_[half, -half]  # mean exactly 0: every flip is as extreme
    strata = np.repeat(["a", "b"], 100)

    assert sign_flip_p(shifted, seed=(1,)) == pytest.approx(1 / 10_001)
    assert sign_flip_p(centered, seed=(1,)) == pytest.approx(1.0)
    assert sign_flip_p(shifted, seed=(1,)) == sign_flip_p(shifted, seed=(1,))
    assert np.isnan(sign_flip_p(np.array([np.nan]), seed=(1,)))
    lo, hi = stratified_bootstrap_ci(shifted, strata, seed=(2,))
    assert lo < shifted.mean() < hi
    assert hi - lo == pytest.approx(2 * 1.96 * shifted.std() / np.sqrt(200), rel=0.15)


def test_ols_slope_ci_recovers_a_known_slope():
    rng = np.random.default_rng(1)
    p = rng.random(2000)
    x = 3.0 * p + rng.normal(0.0, 0.5, 2000)
    strata = np.repeat(np.arange(4), 500)

    slope, lo, hi = ols_slope_ci(p, x, strata, seed=(3,))

    assert slope == pytest.approx(3.0, abs=0.1)
    assert lo < 3.0 < hi


def test_sign_flip_p_is_exact_for_few_values():
    """With no more sign patterns than flips, each pattern counts once."""
    assert sign_flip_p(np.array([1.0, 2.0, 3.0]), seed=(1,)) == pytest.approx(2 / 8)
    assert sign_flip_p(np.array([1.0, -1.0]), seed=(1,)) == pytest.approx(1.0)
    assert sign_flip_p(np.array([2.0]), seed=(1,)) == pytest.approx(1.0)
    # the smallest p five epochs allow
    five = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
    assert sign_flip_p(five, seed=(1,)) == pytest.approx(2 / 32)


def test_the_nested_intervals_are_the_corrected_t_rule_on_nested_replicates():
    # Animal a has epochs 1 and 3 (mean 2), b one epoch 0.5, c epochs 2, 2.5
    # and 3.5 (mean 8/3); a NaN epoch is dropped.
    summaries = np.array([1.0, 3.0, 0.5, 2.0, 2.5, 3.5, np.nan])
    animals = np.array(["a", "a", "b", "c", "c", "c", "b"])
    result = nested_mean_intervals(summaries, animals, n_boot=300, seed=(4,))

    assert result["animal_estimate"] == pytest.approx((2.0 + 0.5 + 8 / 3) / 3)
    assert result["epoch_estimate"] == pytest.approx(12.5 / 6)
    # The replicates by hand, one loop per replicate: the animals drawn, each
    # drawn animal's epochs drawn as often as it was (pooled).
    values, owner = summaries[:6], animals[:6]
    animal_codes, epoch_codes = nested_codes(owner, np.arange(6))
    replicates = nested_bootstrap_draws(
        animal_codes, epoch_codes, keyed_rng((4,), "hierarchical"), 300
    )
    animal_draws, epoch_draws = [], []
    for draw in zip(*replicates, strict=True):
        draw = NestedDraw(*draw)
        own = [
            np.average(values[animal_codes == a], weights=draw.rows[animal_codes == a])
            for a in np.flatnonzero(draw.animals)
        ]
        animal_draws.append(np.average(own, weights=draw.animals[draw.animals > 0]))
        epoch_draws.append(np.average(values, weights=draw.rows))
    for name, estimate, draws in (
        ("animal", result["animal_estimate"], animal_draws),
        ("epoch", result["epoch_estimate"], epoch_draws),
    ):
        # three animals: t(0.975, 2) = 4.3026527, the correction sqrt(3/2)
        half = 4.30265273 * np.sqrt(3 / 2) * np.std(draws, ddof=1)
        assert result[f"{name}_ci_lo"] == pytest.approx(estimate - half), name
        assert result[f"{name}_ci_hi"] == pytest.approx(estimate + half), name


def test_the_nested_intervals_resample_animals_then_epochs():
    rng = np.random.default_rng(8)
    animal = np.repeat(["a", "b", "c", "d"], 20)
    summaries = np.repeat([-1.0, 0.0, 1.0, 2.0], 20) + rng.normal(0, 0.1, 80)

    result = nested_mean_intervals(summaries, animal, seed=(5,))
    lo, hi = result["epoch_ci_lo"], result["epoch_ci_hi"]
    flat_lo, flat_hi = stratified_bootstrap_ci(summaries, np.zeros(80), seed=(5,))

    assert lo < summaries.mean() < hi
    # four animals, not eighty epochs, set the uncertainty
    assert hi - lo > 3 * (flat_hi - flat_lo)
    assert result == nested_mean_intervals(summaries, animal, seed=(5,))
    # one epoch per animal: the t-interval over the animals, up to the draws
    one_each = summaries[::20]
    single = nested_mean_intervals(one_each, np.arange(4), seed=(6,))
    t_lo, t_hi = student_t.interval(
        0.95, 3, loc=one_each.mean(), scale=one_each.std(ddof=1) / 2
    )
    assert single["animal_ci_hi"] - single["animal_ci_lo"] == pytest.approx(
        t_hi - t_lo, rel=0.03
    )
    # nothing varies: nothing is uncertain
    same = nested_mean_intervals(np.ones(6), np.repeat([0, 1, 2], 2), seed=(7,))
    assert (same["animal_ci_lo"], same["animal_ci_hi"]) == (1.0, 1.0)
    # one animal: no t quantile, no interval; no epoch: no estimate
    alone = nested_mean_intervals(np.array([1.0, 2.0]), np.zeros(2), seed=(7,))
    assert alone["animal_estimate"] == 1.5
    assert np.isnan([alone["animal_ci_lo"], alone["epoch_ci_hi"]]).all()
    assert np.isnan(
        nested_mean_intervals(np.array([np.nan]), np.zeros(1), seed=(7,))[
            "animal_estimate"
        ]
    )


def test_the_epoch_summary_follows_the_kind_of_metric():
    """Log-likelihood ratios take each epoch's median; counts and shares its mean."""
    for metric in [
        "run_loc_mark", "can_loc_full", "run_nl_anim_mark", "can_nl_null_mark",
        "run_nlr_anim_mark", "run_loc_vs_Sremote_mark", "can_ns_logratio",
        "run_loc_mark_ps", "run_nlr_anim_mark_ps",
    ]:  # fmt: skip
        assert epoch_summary_kind(metric) == "median", metric
    for metric in [
        "run_ns_expected", "can_ns_expected", "run_ns_count", "run_ns_silent_frac",
        "can_ns_pred_silent", "run_cond_S_silent", "can_cond_pred_C_silent",
        "run_mapS_dist_cm", "run_sel_count",
    ]:  # fmt: skip
        assert epoch_summary_kind(metric) == "mean", metric
    # every metric the summaries use has its rule
    used = {
        *SECONDARY_METRICS,
        *chain.from_iterable(PRIMARY_METRICS.values()),
        *chain.from_iterable(CALIBRATION_METRICS.values()),
    }
    assert {epoch_summary_kind(metric) for metric in used} == {"median", "mean"}
    # and nothing else is summarized
    for metric in ["run_loc_mark_sd_runs", "run_run0_mapS", "n_spikes"]:
        with pytest.raises(ValueError, match="no epoch summary"):
            epoch_summary_kind(metric)


def test_the_animal_t_test_takes_each_animal_once():
    summaries = np.array([1.0, 3.0, 0.5, 2.0, 2.5, 3.5, np.nan])
    animals = np.array(["a", "a", "b", "c", "c", "c", "b"])  # b: one finite epoch

    result = animal_t_test(summaries, animals)

    by_animal = np.array([2.0, 0.5, 8.0 / 3.0])  # each animal's mean epoch summary
    sem = by_animal.std(ddof=1) / np.sqrt(3)
    assert result.estimate == pytest.approx(by_animal.mean())
    assert result.p == pytest.approx(ttest_1samp(by_animal, 0.0).pvalue)
    t = by_animal.mean() / sem
    assert result.p == pytest.approx(2 * student_t.sf(t, df=2))  # A - 1 df
    # 40 more epochs of animal a at its own mean change nothing: a counts once
    more = animal_t_test(np.r_[summaries, [2.0] * 40], np.r_[animals, ["a"] * 40])
    assert tuple(more) == pytest.approx(tuple(result))
    # one animal: an estimate, but no test
    single = animal_t_test(np.array([1.0, 2.0]), np.array(["a", "a"]))
    assert single.estimate == 1.5
    assert np.isnan(single.p)
    assert not result.zero_variance and not single.zero_variance


def test_identical_animal_summaries_are_flagged_and_untested():
    """Every animal the same: there is no p.

    0.1 and 0.7 are not exact in binary, so three of them have an SD of about
    1e-17 rather than 0; they are still identical.
    """
    for value in (2.0, 0.0, 0.1, 0.7):
        result = animal_t_test(np.full(3, value), np.array(["a", "b", "c"]))

        assert result.zero_variance, value
        assert np.isnan(result.p), value
        assert result.estimate == pytest.approx(value)


def bouts_of_six_epochs() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Local bouts and their matched windows: six epochs of three animals.

    Every epoch's bout and baseline medians (and means) are set exactly.
    """
    rng = np.random.default_rng(10)
    bout_means = np.array([1.0, 2.0, -0.5, 3.0, 4.0, 0.5])
    base_means = np.array([0.0, 0.5, 0.0, 0.0, 5.0, 0.0])
    frames = []
    for k, (mean, base) in enumerate(zip(bout_means, base_means, strict=True)):
        n = 30
        half = rng.normal(0, 1, (2, n // 2))
        noise = np.c_[half, -half]  # symmetric: median 0 and mean 0
        bout_s = np.arange(n) * 100
        animal = ["aa", "bb", "cc"][k // 2]
        epoch = f"{animal}2020010{k}_02_r1_HPC"
        frames += [
            pd.DataFrame(
                {
                    "kind": "bout", "state": "Local", "regime": "immobile",
                    "bout_s": bout_s, "matched_bout_s": np.nan, "n_found": n,
                    "run_loc_mark": mean + noise[0], "epoch_id": epoch,
                    "animal": animal,
                }
            ),
            pd.DataFrame(
                {
                    "kind": "baseline", "state": "Local", "regime": "immobile",
                    "bout_s": np.nan, "matched_bout_s": bout_s, "n_found": np.nan,
                    "run_loc_mark": base + noise[1], "epoch_id": epoch,
                    "animal": animal,
                }
            ),
        ]  # fmt: skip
    return pd.concat(frames, ignore_index=True), bout_means, base_means


def test_bout_statistics_infer_at_the_epoch_level():
    table, means, base = bouts_of_six_epochs()

    summary = bout_statistics(
        table, metrics={"Local": ["run_loc_mark"]}, seed=(1,), window_level=True
    )
    row = summary.iloc[0]

    patterns = np.array(list(product([1.0, -1.0], repeat=6)))
    exact = np.mean(np.abs(patterns @ means) / 6 >= abs(means.mean()) * (1 - 1e-12))
    assert row["n_epochs"] == 6 and row["n_animals"] == 3
    assert row["epoch_summary"] == "median"
    # each epoch's median (primary) and its mean (secondary) are equal here
    for role in ("", "other_"):
        assert row[f"{role}epoch_estimate"] == pytest.approx(means.mean())
        assert row[f"{role}epoch_ci_lo"] < means.mean() < row[f"{role}epoch_ci_hi"]
        assert row[f"{role}p_epoch_signflip"] == pytest.approx(exact)
    # each animal once, as the mean of its epochs: 1.5, 1.25 and 2.25
    by_animal = np.array([1.5, 1.25, 2.25])
    for role in ("", "other_"):
        assert row[f"{role}animal_estimate"] == pytest.approx(by_animal.mean())
    # The intervals are nested_mean_intervals of the epoch values; the mean
    # summary keeps the bout's own seed, the median one adds its kind.
    epoch_animals = np.repeat(["aa", "bb", "cc"], 2)
    seed = (1, "Local", "immobile", "run_loc_mark", "epoch")
    for role, role_seed in (("other_", seed), ("", (*seed, "median"))):
        expected = nested_mean_intervals(means, epoch_animals, seed=role_seed)
        for name in ("animal_ci_lo", "animal_ci_hi", "epoch_ci_lo", "epoch_ci_hi"):
            assert row[f"{role}{name}"] == pytest.approx(expected[name]), role + name
    assert row["p_animal_t"] == pytest.approx(ttest_1samp(by_animal, 0.0).pvalue)
    assert row["epochs_pos"] == "5/6" and row["animals_pos"] == "3/3"
    differences = means - base  # 1, 1.5, -0.5, 3, -1, 0.5
    exact = np.mean(
        np.abs(patterns @ differences) / 6 >= abs(differences.mean()) * (1 - 1e-12)
    )
    assert row["diff_epoch_estimate"] == pytest.approx(differences.mean())
    assert row["diff_p_epoch_signflip"] == pytest.approx(exact)
    by_animal = np.array([1.25, 1.25, -0.25])
    assert row["diff_animal_estimate"] == pytest.approx(by_animal.mean())
    low, high = row["diff_animal_ci_lo"], row["diff_animal_ci_hi"]
    assert (low + high) / 2 == pytest.approx(by_animal.mean())
    assert row["diff_p_animal_t"] == pytest.approx(ttest_1samp(by_animal, 0.0).pvalue)
    assert row["diff_epochs_pos"] == "4/6" and row["diff_animals_pos"] == "2/3"
    # the window-level statistics, opt-in, as they were
    x = table.loc[table.kind == "bout", "run_loc_mark"].to_numpy()
    assert row["p_signflip_window_level"] == sign_flip_p(
        x, seed=(1, "Local", "immobile", "run_loc_mark")
    )


def test_bout_animal_summaries_are_what_the_animal_level_inference_averages():
    table, _, _ = bouts_of_six_epochs()
    metrics = {"Local": ["run_loc_mark"]}

    summaries = bout_animal_summaries(table, metrics=metrics)

    assert summaries.animal.tolist() == ["aa", "bb", "cc"]
    # each animal's mean of its two epochs' medians, and of their differences
    assert summaries.value.to_numpy() == pytest.approx([1.5, 1.25, 2.25])
    assert summaries["diff"].to_numpy() == pytest.approx([1.25, 1.25, -0.25])
    row = bout_statistics(table, metrics=metrics, seed=(1,)).iloc[0]
    assert row["animal_estimate"] == pytest.approx(summaries.value.mean())
    assert row["diff_animal_estimate"] == pytest.approx(summaries["diff"].mean())


def test_epoch_slopes_are_not_fooled_by_differences_between_epochs():
    """Within each epoch the evidence ignores the probability; across them it does not."""
    rng = np.random.default_rng(11)
    frames = [
        pd.DataFrame(
            {
                "kind": "calib",
                "state": "",
                "regime": "immobile",
                "p_Local": 0.15 * k + 0.2 * rng.random(300),
                "run_loc_mark": float(k),  # constant within the epoch
                "epoch_id": f"e{k}",
                "animal": "abc"[k // 2],
            }
        )  # fmt: skip
        for k in range(6)
    ]
    table = pd.concat(frames, ignore_index=True)

    _, trends = calibration_statistics(
        table, metrics={"Local": ["run_loc_mark"]}, window_level=True
    )
    trend = trends.iloc[0]

    assert trend["slope_window_level"] > 3.0 and trend["rises_window_level"] == "yes"
    # epochs 0 and 4 hold windows in one probability bin only: no slope
    assert trend["n_epochs"] == 4 and trend["slope_epochs_pos"] == "0/4"
    assert trend["slope_animal_estimate"] == 0.0 and trend["rises"] == "no"


def test_epoch_summaries_drop_what_is_not_finite():
    summaries = epoch_summaries(
        np.array([1.0, 3.0, 8.0, np.nan, 5.0, np.nan]),
        np.array(["e1", "e1", "e1", "e1", "e2", "e3"]),
        np.array(["a", "a", "a", "a", "b", "b"]),
    )

    assert summaries.index.tolist() == ["e1", "e2"]
    assert summaries["median"].tolist() == [3.0, 5.0]
    assert summaries["mean"].tolist() == [4.0, 5.0]
    assert summaries["animal"].tolist() == ["a", "b"]
    assert summaries["n"].tolist() == [3, 1]


def test_epoch_medians_resist_an_artifact_burst():
    """One burst of spikes in a No-Spike bout drags its epoch's mean, not its median."""
    rng = np.random.default_rng(12)
    frames = []
    for k, animal in enumerate(["aa", "bb", "cc"]):
        n = 40
        values = 5.0 + rng.normal(0, 1, n)
        if k == 2:
            values[0] = -1641.0  # a burst on every tetrode
        bout_s = np.arange(n) * 100
        common = {"state": "No-Spike", "regime": "immobile", "animal": animal,
                  "epoch_id": f"{animal}2020010{k}_02_r1_HPC"}  # fmt: skip
        frames += [
            pd.DataFrame({"kind": "bout", "bout_s": bout_s, "matched_bout_s": np.nan,
                          "n_found": n, "can_ns_logratio": values, **common}),
            pd.DataFrame({"kind": "baseline", "bout_s": np.nan, "matched_bout_s": bout_s,
                          "n_found": np.nan, "can_ns_logratio": -100.0, **common}),
        ]  # fmt: skip
    table = pd.concat(frames, ignore_index=True)

    row = bout_statistics(table, metrics={"No-Spike": ["can_ns_logratio"]}).iloc[0]

    assert row["epoch_summary"] == "median"
    assert row["epochs_pos"] == "3/3" and row["animal_estimate"] > 4.0
    assert row["other_epochs_pos"] == "2/3" and row["other_animal_estimate"] < 0.0


def test_counts_and_shares_take_each_epochs_mean():
    """A count that is usually 0: its epoch medians are 0, its epoch means are not."""
    rng = np.random.default_rng(14)
    frames = []
    for k, animal in enumerate(["aa", "bb", "cc"]):
        n = 40
        bout_s = np.arange(n) * 100
        common = {"regime": "immobile", "animal": animal,
                  "epoch_id": f"{animal}2020010{k}_02_r1_HPC"}  # fmt: skip
        frames += [
            pd.DataFrame({"kind": "bout", "state": "No-Spike", "bout_s": bout_s,
                          "matched_bout_s": np.nan, "n_found": n,
                          "run_ns_count": (rng.random(n) < 0.3) * 2.0, **common}),
            pd.DataFrame({"kind": "baseline", "state": "No-Spike", "bout_s": np.nan,
                          "matched_bout_s": bout_s, "n_found": np.nan,
                          "run_ns_count": 0.0, **common}),
            pd.DataFrame({"kind": "calib", "state": "", "bout_s": np.nan,
                          "matched_bout_s": np.nan, "n_found": np.nan,
                          "p_No-Spike": rng.random(600),
                          "run_ns_count": (rng.random(600) < 0.3) * 1.0, **common}),
        ]  # fmt: skip
    table = pd.concat(frames, ignore_index=True)
    bouts = table[table.kind == "bout"].groupby("epoch_id").run_ns_count
    assert (bouts.median() == 0).all()

    row = bout_statistics(table, metrics={"No-Spike": ["run_ns_count"]}).iloc[0]

    assert row["epoch_summary"] == "mean"
    assert row["animal_estimate"] == pytest.approx(bouts.mean().mean())
    assert row["epochs_pos"] == "3/3" and row["animals_pos"] == "3/3"
    assert row["diff_animal_estimate"] == pytest.approx(bouts.mean().mean())
    assert row["other_animal_estimate"] == 0.0 and row["other_epochs_pos"] == "0/3"
    # the top calibration bin is judged by its epoch means too
    _, trends = calibration_statistics(table, metrics={"No-Spike": ["run_ns_count"]})
    trend = trends.iloc[0]
    assert trend["epoch_summary"] == "mean" and trend["top_positive"] == "yes"
    assert trend["top_other_animal_estimate"] == 0.0


def test_calibration_verdicts_need_enough_windows():
    """No verdict without 10 windows in the top and the bottom bins, pooled."""
    rng = np.random.default_rng(13)

    def calibration(spans: list[tuple[float, float]], slope: float) -> pd.DataFrame:
        frames = []
        for k, (low, high) in enumerate(spans):
            p = low + (high - low) * rng.random(600)
            frames.append(
                pd.DataFrame(
                    {"kind": "calib", "state": "", "regime": "immobile", "p_Local": p,
                     "run_loc_mark": slope * p + rng.normal(0, 0.05, 600),
                     "epoch_id": f"e{k}", "animal": "abc"[k]}
                )
            )  # fmt: skip
        return pd.concat(frames, ignore_index=True)

    def trend(table: pd.DataFrame) -> pd.Series:
        _, trends = calibration_statistics(
            table, metrics={"Local": ["run_loc_mark"]}, window_level=True
        )
        return trends.iloc[0]

    # no window reaches the top bin: a steep rise, but no verdict
    no_top = trend(calibration([(0.0, 0.6)] * 3, slope=5.0))
    assert no_top["n_top"] == 0 and no_top["slope_animal_ci_lo"] > 4.0
    for column in (
        "rises",
        "top_positive",
        "rises_window_level",
        "top_positive_window_level",
    ):
        assert no_top[column] == "too few windows", column
    # a bottom bin with fewer than 10 windows, pooled
    thin = calibration([(0.199, 1.0)] * 3, slope=5.0)
    assert ((thin.p_Local < 0.2).sum() < 10) and ((thin.p_Local >= 0.8).sum() >= 10)
    thin_trend = trend(thin)
    assert thin_trend["rises"] == "too few windows"
    assert thin_trend["top_positive"] == "too few windows"
    # enough windows: a verdict
    assert trend(calibration([(0.0, 1.0)] * 3, slope=5.0))["rises"] == "yes"
    # enough windows, but one animal gives no animal-level interval
    single = trend(calibration([(0.0, 1.0)], slope=5.0))
    assert single["rises"] == "too few animals"
    assert single["top_positive"] == "too few animals"
    assert single["rises_window_level"] == "yes"
    # an epoch in one probability bin enters no slope, however steep its own
    with_one_bin = pd.concat(
        [
            calibration([(0.0, 1.0)] * 3, slope=2.0),
            calibration([(0.45, 0.55)], slope=-50.0).assign(epoch_id="e9", animal="d"),
        ],
        ignore_index=True,
    )
    one_bin = trend(with_one_bin)
    assert one_bin["n_epochs"] == 3
    assert one_bin["slope_animal_estimate"] == pytest.approx(2.0, abs=0.05)


def test_the_verdicts_read_the_small_sample_interval():
    """Five animals' slopes, all positive but one far larger than the rest.

    No resample of them averages below 0.4, so the replicates' percentiles
    would lie above 0; the interval with the small-sample t correction does
    not, and "rises" follows it.
    """
    rng = np.random.default_rng(16)
    frames = []
    for k, slope in enumerate([0.5, 0.6, 0.4, 0.55, 5.0]):
        p = rng.random(600)
        frames.append(
            pd.DataFrame(
                {"kind": "calib", "state": "", "regime": "immobile", "p_Local": p,
                 "run_loc_mark": slope * p + rng.normal(0, 0.05, 600),
                 "epoch_id": f"e{k}", "animal": "abcde"[k]}
            )
        )  # fmt: skip
    table = pd.concat(frames, ignore_index=True)

    _, trends = calibration_statistics(table, metrics={"Local": ["run_loc_mark"]})
    trend = trends.iloc[0]

    for level in ("animal", "epoch"):  # one epoch per animal: the two agree
        assert trend[f"slope_{level}_ci_lo"] < 0.0 < trend[f"slope_{level}_ci_hi"]
    assert trend["slope_p_animal_t"] > 0.05
    assert trend["rises"] == "no"


def evidence_table() -> pd.DataFrame:
    """Local bouts, their matched windows and calibration windows of two epochs.

    Bouts carry local evidence about 2 and their baselines about 0; calibration
    evidence rises with P(Local) as 4 p - 1.
    """
    rng = np.random.default_rng(4)
    frames = []
    for epoch, animal in [
        ("aa20200101_02_r1_HPC", "aa"),
        ("bb20200101_02_r1_HPC", "bb"),
    ]:
        n = 60
        bout_s = np.arange(n) * 100
        bouts = pd.DataFrame(
            {
                "kind": "bout", "state": "Local", "regime": "immobile",
                "bout_s": bout_s, "matched_bout_s": np.nan, "n_found": n,
                "run_loc_mark": 2.0 + rng.normal(0, 0.5, n),
            }
        )  # fmt: skip
        base = pd.DataFrame(
            {
                "kind": "baseline", "state": "Local", "regime": "immobile",
                "bout_s": np.nan, "matched_bout_s": bout_s, "n_found": np.nan,
                "run_loc_mark": rng.normal(0, 0.5, n),
            }
        )  # fmt: skip
        p = rng.random(400)
        cal = pd.DataFrame(
            {
                "kind": "calib", "state": "", "regime": "immobile",
                "bout_s": np.nan, "matched_bout_s": np.nan, "n_found": np.nan,
                "p_Local": p, "run_loc_mark": 4 * p - 1 + rng.normal(0, 0.3, 400),
            }
        )  # fmt: skip
        frames.append(
            pd.concat([bouts, base, cal]).assign(epoch_id=epoch, animal=animal)
        )
    return pd.concat(frames, ignore_index=True)


def test_bout_statistics_pair_each_bout_with_its_matched_window():
    table = evidence_table()
    bouts = table[table.kind == "bout"]
    base = table[table.kind == "baseline"]

    summary = bout_statistics(
        table, metrics={"Local": ["run_loc_mark"]}, window_level=True
    )
    row = summary.set_index(["state", "regime", "metric"]).loc[
        ("Local", "immobile", "run_loc_mark")
    ]

    assert row["n"] == 120 and row["n_found"] == 120 and row["n_epochs"] == 2
    assert row["median"] == pytest.approx(bouts.run_loc_mark.median())
    assert row["q25"] == pytest.approx(bouts.run_loc_mark.quantile(0.25))
    assert row["frac_pos"] == pytest.approx((bouts.run_loc_mark > 0).mean())
    assert row["mean"] == pytest.approx(bouts.run_loc_mark.mean())
    assert row["base_median"] == pytest.approx(base.run_loc_mark.median())
    assert row["diff_mean"] == pytest.approx(
        bouts.run_loc_mark.mean() - base.run_loc_mark.mean()
    )
    assert row["epochs_pos"] == "2/2" and row["diff_epochs_pos"] == "2/2"
    assert row["animals_pos"] == "2/2" and row["diff_animals_pos"] == "2/2"
    # one epoch per animal: the t-test has 1 degree of freedom
    medians = bouts.groupby("animal").run_loc_mark.median()
    assert row["p_animal_t"] == pytest.approx(ttest_1samp(medians, 0.0).pvalue)
    assert row["p_epoch_signflip"] == pytest.approx(0.5)  # the least two epochs allow
    # the window-level tests, opt-in, for comparison only
    assert row["ci_lo_window_level"] < row["mean"] < row["ci_hi_window_level"]
    assert row["p_signflip_window_level"] == pytest.approx(1 / 10_001)
    assert row["diff_ci_lo_window_level"] < row["diff_mean"]
    assert row["diff_mean"] < row["diff_ci_hi_window_level"]
    # a bout whose matched window is missing is an error, not a silent drop
    with pytest.raises(ValueError, match="matched"):
        bout_statistics(
            table.drop(index=base.index[0]), metrics={"Local": ["run_loc_mark"]}
        )


def test_calibration_statistics_bin_by_probability_and_call_the_trend():
    table = evidence_table()
    cal = table[table.kind == "calib"]

    bins, trends = calibration_statistics(
        table, metrics={"Local": ["run_loc_mark"]}, window_level=True
    )

    top = bins[(bins.metric == "run_loc_mark") & (bins.pbin == "0.8-1.0")].iloc[0]
    in_top = cal[cal.p_Local >= 0.8]
    assert top["n"] == len(in_top)
    assert top["mean"] == pytest.approx(in_top.run_loc_mark.mean())
    assert top["ci_lo_window_level"] < top["mean"] < top["ci_hi_window_level"]
    # two animals of one epoch each: the interval is the t-interval with 1
    # degree of freedom, up to the bootstrap draws
    per_epoch = in_top.groupby("epoch_id").run_loc_mark.median()
    sem = per_epoch.std(ddof=1) / np.sqrt(2)
    assert top["epoch_summary"] == "median"
    assert top["animal_estimate"] == pytest.approx(per_epoch.mean())
    assert (top["animal_ci_lo"], top["animal_ci_hi"]) == pytest.approx(
        student_t.interval(0.95, 1, loc=per_epoch.mean(), scale=sem), rel=0.01
    )
    # one epoch per animal: the epoch-weighted mean and its interval are the
    # animal-level ones
    assert top["epoch_estimate"] == pytest.approx(per_epoch.mean())
    assert top["epoch_ci_lo"] == pytest.approx(top["animal_ci_lo"])
    assert top["epoch_ci_hi"] == pytest.approx(top["animal_ci_hi"])
    means = in_top.groupby("epoch_id").run_loc_mark.mean()
    assert top["other_animal_estimate"] == pytest.approx(means.mean())
    curve = trends.set_index(["state", "regime", "metric"]).loc[
        ("Local", "immobile", "run_loc_mark")
    ]
    assert curve["slope_animal_estimate"] == pytest.approx(4.0, abs=0.2)
    assert curve["slope_window_level"] == pytest.approx(4.0, abs=0.2)
    assert curve["slope_epochs_pos"] == "2/2" and curve["slope_animals_pos"] == "2/2"
    assert curve["rises"] == "yes" and curve["top_positive"] == "yes"
    flat = table.assign(run_loc_mark=np.where(table.kind == "calib", -1.0, 0.0))
    _, flat_trends = calibration_statistics(flat, metrics={"Local": ["run_loc_mark"]})
    assert (
        flat_trends.rises.iloc[0] == "no" and flat_trends.top_positive.iloc[0] == "no"
    )


def test_rate_calibration_and_coordinated_silence():
    """Observed spikes and silences against each encoding's prediction."""
    rng = np.random.default_rng(6)
    n = 300
    table = pd.DataFrame(
        {
            "epoch_id": np.repeat(["aa20200101_02_r1_HPC", "bb20200101_02_r1_HPC"], n // 2),
            "animal": np.repeat(["aa", "bb"], n // 2),
            "kind": "calib", "state": "", "regime": "immobile",
            "p_No-Spike": rng.random(n),
            "runcv_ns_count": rng.poisson(1.0, n).astype(float),
            "runcv_ns_silent_frac": rng.random(n),
            "runcv_cond_S_silent": rng.integers(0, 20, n).astype(float),
        }
    )  # fmt: skip
    table["runcv_cond_both_silent"] = np.floor(table.runcv_cond_S_silent * 0.5)
    for prefix, share in [("runcv", 0.1), ("run", 0.2), ("can", 0.3)]:
        table[f"{prefix}_ns_expected"] = rng.uniform(0.5, 2.0, n)
        table[f"{prefix}_ns_pred_silent"] = rng.random(n)
        table[f"{prefix}_cond_pred_C_silent"] = table.runcv_cond_S_silent * share

    rate = rate_calibration(table).set_index(["regime", "encoding", "animal"])
    assert set(rate.index.get_level_values("encoding")) == set(ENCODING_LABELS.values())
    pooled = rate.loc[("immobile", ENCODING_LABELS["runcv"], "all")]
    assert pooled["observed_over_expected"] == pytest.approx(
        table.runcv_ns_count.sum() / table.runcv_ns_expected.sum()
    )
    bb = table[table.animal == "bb"]
    assert rate.loc[("immobile", ENCODING_LABELS["can"], "bb")][
        "observed_over_expected"
    ] == pytest.approx(bb.runcv_ns_count.sum() / bb.can_ns_expected.sum())

    default = coordinated_silence(table)
    assert not [c for c in default if c.endswith("_window_level")]
    silence = coordinated_silence(table, window_level=True)
    pd.testing.assert_frame_equal(silence[default.columns], default)
    every = silence[(silence.kind == "calib") & (silence.state == "all windows")].iloc[
        0
    ]
    assert every["obs_C_silent_given_S_silent"] == pytest.approx(
        table.runcv_cond_both_silent.sum() / table.runcv_cond_S_silent.sum()
    )
    for prefix, share in [("runcv", 0.1), ("run", 0.2), ("can", 0.3)]:
        assert every[f"pred_given_S_{prefix}"] == pytest.approx(share)
        assert every[f"pred_{prefix}_animal_estimate"] == pytest.approx(share)
    assert every["obs_given_ci_lo_window_level"] < every["obs_C_silent_given_S_silent"]
    assert every["obs_C_silent_given_S_silent"] < every["obs_given_ci_hi_window_level"]
    sums = table.groupby("epoch_id")[
        ["runcv_cond_both_silent", "runcv_cond_S_silent"]
    ].sum()
    ratios = sums.runcv_cond_both_silent / sums.runcv_cond_S_silent
    assert every["obs_animal_estimate"] == pytest.approx(ratios.mean())
    assert every["obs_epoch_estimate"] == pytest.approx(ratios.mean())
    assert every["excess_can_animal_estimate"] == pytest.approx(ratios.mean() - 0.3)
    excess = ratios - 0.3
    # one epoch per animal: the t-interval, up to the bootstrap draws
    assert (every["excess_can_animal_ci_lo"], every["excess_can_animal_ci_hi"]) == (
        pytest.approx(
            student_t.interval(
                0.95, 1, loc=excess.mean(), scale=excess.std(ddof=1) / np.sqrt(2)
            ),
            rel=0.01,
        )
    )
    assert every["excess_can_epochs_pos"] == f"{int((ratios > 0.3).sum())}/2"
    # one epoch per animal here: the t-test is over the two epochs' excesses
    assert every["excess_can_p_animal_t"] == pytest.approx(
        ttest_1samp(ratios - 0.3, 0.0).pvalue
    )
    assert np.isfinite(every["excess_runcv_p_epoch_signflip"])
    by_bin = silence[silence.state == "P(No-Spike) bin"]
    assert by_bin.n_windows.sum() == n


def test_tetrode_terms_refuse_what_would_corrupt_the_evidence():
    good = random_terms(20, 6, 3, rate=1.0, seed=0)
    fields = {
        "ground": good.ground,
        "log_intensity": good.log_intensity,
        "time_bin": good.time_bin,
        "tetrode": good.tetrode,
        "bin_s": 0.002,
    }
    nan_ground = good.ground.copy()
    nan_ground[1, 2] = np.nan
    bad_intensity = good.log_intensity.copy()
    bad_intensity[0, 0] = -np.inf
    for change, message in [
        ({"ground": nan_ground}, "not finite"),
        ({"log_intensity": bad_intensity}, "not finite"),
        ({"log_intensity": good.log_intensity[:, :5]}, "place bins"),
        ({"tetrode": good.tetrode + 3}, "no ground intensity"),
        ({"time_bin": good.time_bin[1:]}, "one time bin"),
    ]:
        with pytest.raises(ValueError, match=message):
            TetrodeTerms(**(fields | change))


# --- place bins whose occupancy underflows --------------------------------------------
def environment_with_real_bins(n_real: int, n_junction: int = 2) -> SimpleNamespace:
    """The one attribute ``interior_bin_mask`` needs: ``n_real`` real bins and some
    junction placeholders around them."""
    interior = np.r_[
        np.zeros(1, bool), np.ones(n_real, bool), np.zeros(n_junction, bool)
    ]
    return SimpleNamespace(is_track_interior_=interior)


def test_flagged_bins_are_the_real_bins_with_occupancy_above_zero_and_below_the_floor():
    # the decode's occupancy of chimi20200221_08_r4 (D1) at bins 69-70 (exact
    # zeros: both devices agree there), 71 and 72 (float32 subnormals), 78 (1.4e-24,
    # the highest at which the CPU and the GPU disagreed), 79 (1.9e-22, where the CPU
    # matched) and two ordinary values
    occupancy = np.array(
        [0.0, 0.0, 5.4e-43, 5.6e-40, 1.4e-24, 1.9e-22, 4e-6, 5.8e-5], dtype=np.float32
    )
    assert occupancy[2] < np.finfo(np.float32).tiny  # really subnormal
    environment = environment_with_real_bins(occupancy.size)

    left_out = subnormal_occupancy_bin_mask(occupancy, environment)

    assert left_out.dtype == bool and left_out.shape == (occupancy.size,)
    np.testing.assert_array_equal(
        left_out, [False, False, True, True, True, True, False, False]
    )
    # the floor is a parameter, and the comparison is strict
    wide = occupancy.astype(float)
    np.testing.assert_array_equal(
        subnormal_occupancy_bin_mask(wide, environment, floor=4e-6),
        (wide > 0) & (wide < 4e-6),
    )


def test_an_exact_zero_occupancy_is_not_flagged_but_a_subnormal_one_is():
    """Both devices test ``occupancy > 0``: False at an exact zero, so both use the
    same floor and agree (D1: 2e-14 per spike at bins 0-70); a subnormal is True on
    the GPU and False on the XLA CPU, so they disagree."""
    occupancy = np.array([0.0, np.float32(5.4e-43), 0.0, 3e-2], dtype=np.float32)

    left_out = subnormal_occupancy_bin_mask(occupancy, environment_with_real_bins(4))

    np.testing.assert_array_equal(left_out, [False, True, False, False])


def test_a_normal_occupancy_has_no_flagged_bin(tmp_path):
    epoch = build_simulated_epoch(tmp_path)

    left_out = subnormal_occupancy_bin_mask(
        epoch.encoding_model["occupancy"], epoch.environment
    )

    assert left_out.shape == (int(interior_bin_mask(epoch.environment).sum()),)
    assert not left_out.any()
    assert np.asarray(epoch.encoding_model["occupancy"]).min() > 1e-8


def test_the_floor_sits_between_the_float32_trouble_and_every_real_occupancy():
    # D1: the CPU rebuild and the decode disagreed up to an occupancy of 1.4e-24;
    # the lowest occupancy of any other analyzed epoch's decode is 4e-6
    assert UNDERFLOW_OCCUPANCY_FLOOR >= 1e3 * 1.4e-24
    assert UNDERFLOW_OCCUPANCY_FLOOR <= 1e-6 * 4e-6
    assert UNDERFLOW_OCCUPANCY_FLOOR > 1e6 * np.finfo(np.float32).tiny


def test_the_occupancy_must_cover_exactly_the_real_bins():
    environment = environment_with_real_bins(6)

    with pytest.raises(ValueError, match="6 real place bins"):
        subnormal_occupancy_bin_mask(np.ones(7), environment)


def drop_bins(terms: TetrodeTerms, left_out: np.ndarray) -> TetrodeTerms:
    """The same terms with some place bins removed."""
    keep = ~left_out
    return TetrodeTerms(
        ground=terms.ground[:, keep],
        log_intensity=terms.log_intensity[:, keep],
        time_bin=terms.time_bin,
        tetrode=terms.tetrode,
        bin_s=terms.bin_s,
    )


def underflow_case(n_pos: int = 30, n_time: int = 600, seed: int = 4):
    """Random terms, windows and halves, and bins 0-5 and 22 left out."""
    rng = np.random.default_rng(seed)
    terms = random_terms(n_time, n_pos, 4, rate=0.3, seed=seed)
    left_out = np.zeros(n_pos, dtype=bool)
    left_out[[0, 1, 2, 3, 4, 5, 22]] = True
    # strongly favor the left-out bins, as the CPU does for a zone it flushed
    li = terms.log_intensity.copy()
    li[:, left_out] += 6.0
    terms = TetrodeTerms(
        ground=terms.ground,
        log_intensity=li,
        time_bin=terms.time_bin,
        tetrode=terms.tetrode,
        bin_s=terms.bin_s,
    )
    animal = rng.integers(6, n_pos, n_time)
    animal[animal == 22] = 23  # the animal is never where the model was never fit
    windows = overlapping_windows(n_time, seed)
    return terms, windows, animal, left_out


def test_left_out_bins_are_no_choice_and_the_evidence_is_that_of_the_remaining_bins():
    terms, windows, animal, left_out = underflow_case()
    n_pos = terms.n_place_bins
    distance = linear_distance(n_pos)
    halves = tetrode_halves(terms.n_tetrodes, n_partitions=3, seed=5)
    options = {"n_partners": 4, "no_spike_rate": NO_SPIKE_RATE, "seed": 6}

    out = window_evidence(
        terms, windows, animal, distance, halves, left_out_bins=left_out, **options
    )
    # the same evidence as if the bins were not in the model at all
    renumber = np.cumsum(~left_out) - 1
    keep = ~left_out
    absent = window_evidence(
        drop_bins(terms, left_out), windows, renumber[animal],
        distance[np.ix_(keep, keep)], halves, **options,
    )  # fmt: skip

    index_columns = ["run0_mapS", "run0_mapSr", "animal_center_bin"]
    marks = [c for c in absent.columns if c not in index_columns]
    pd.testing.assert_frame_equal(out[marks], absent[marks])
    kept_bins = np.flatnonzero(keep)
    for column in ("run0_mapS", "run0_mapSr"):
        np.testing.assert_array_equal(out[column], kept_bins[absent[column]])
        assert not left_out[out[column].to_numpy()].any()
    # without the rule the planted favor picks a left-out bin, which is marked
    free = window_evidence(terms, windows, animal, distance, halves, **options)
    assert left_out[free.run0_mapS.to_numpy()].any()
    assert (out.mapS_left_out_runs > 0).any() and (out.mapSr_left_out_runs > 0).any()
    n_runs = 2 * len(halves)
    assert out.mapS_left_out_runs.between(0, n_runs).all()
    assert out.mapSr_left_out_runs.between(0, n_runs).all()


def test_nothing_left_out_is_the_evidence_as_before():
    terms, windows, animal, left_out = underflow_case()
    distance = linear_distance(terms.n_place_bins)
    halves = tetrode_halves(terms.n_tetrodes, n_partitions=2, seed=5)
    options = {"n_partners": 4, "no_spike_rate": NO_SPIKE_RATE, "seed": 6}
    before = window_evidence(terms, windows, animal, distance, halves, **options)

    none = window_evidence(
        terms, windows, animal, distance, halves, left_out_bins=None, **options
    )
    empty = window_evidence(
        terms, windows, animal, distance, halves,
        left_out_bins=np.zeros_like(left_out), **options,
    )  # fmt: skip

    pd.testing.assert_frame_equal(none, before)
    pd.testing.assert_frame_equal(empty, before)
    assert not [c for c in before if "left_out" in c]


def test_an_animal_in_a_left_out_bin_is_refused():
    terms, windows, animal, left_out = underflow_case()
    animal = animal.copy()
    animal[windows.s.iloc[0] + 1] = int(np.flatnonzero(left_out)[0])

    with pytest.raises(ValueError, match="animal"):
        window_evidence(
            terms, windows, animal, linear_distance(terms.n_place_bins),
            tetrode_halves(terms.n_tetrodes, n_partitions=2, seed=5),
            left_out_bins=left_out, n_partners=4, no_spike_rate=NO_SPIKE_RATE, seed=6,
        )  # fmt: skip


def test_left_out_bins_must_be_one_flag_per_place_bin_and_leave_some():
    terms, windows, animal, left_out = underflow_case()
    kwargs = {"n_partners": 4, "no_spike_rate": NO_SPIKE_RATE, "seed": 6}
    distance = linear_distance(terms.n_place_bins)
    halves = tetrode_halves(terms.n_tetrodes, n_partitions=2, seed=5)

    with pytest.raises(ValueError, match="one flag per place bin"):
        window_evidence(
            terms, windows, animal, distance, halves,
            left_out_bins=left_out[:-1], **kwargs,
        )  # fmt: skip
    with pytest.raises(ValueError, match="every place bin"):
        window_evidence(
            terms, windows, animal, distance, halves,
            left_out_bins=np.ones_like(left_out), **kwargs,
        )  # fmt: skip
