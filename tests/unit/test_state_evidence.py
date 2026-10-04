"""Held-out spike evidence per decoder state (``analysis/hippocampus/state_evidence``).

Every input is synthetic. The per-tetrode terms are checked against the
library's own clusterless likelihood on a fitted KDE encoding model; the window
evidence is checked against an independent per-window loop written here, and
on terms planted so that the answer is known.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from non_local_detector.likelihoods.clusterless_kde import (
    predict_clusterless_kde_log_likelihood,
)

from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.likelihood import (
    CLUSTERLESS_LIKELIHOOD_KEYS,
)
from continuum_swr_replay.analysis.hippocampus.state_evidence import (
    STATES,
    TetrodeTerms,
    bout_statistics,
    build_windows,
    calibration_statistics,
    calibration_windows,
    coordinated_silence,
    interior_bin_mask,
    matched_windows,
    nearest_interior_bins,
    null_partners,
    ols_slope_ci,
    per_tetrode_terms,
    rate_calibration,
    regimes,
    sign_flip_p,
    state_bouts,
    stratified_bootstrap_ci,
    summarize_state_evidence,
    tetrode_halves,
    window_evidence,
)
from tests.unit._memoryless_clusterless_epoch import build_simulated_epoch

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
    centre = animal_bins[s_all + (e_all - s_all + 1) // 2]

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
            remote = np.where(distance[centre[w]] < min_cm, -np.inf, pooled)
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
            v["mapS_dist_cm"][r, w] = distance[m, centre[w]]
            v["mapSr_dist_cm"][r, w] = distance[mr, centre[w]]
            if nl_partner[w, 0] >= 0:
                pm = [maps[j] for j in nl_partner[w]]
                null_m = np.mean([mc[:, j] for j in pm], axis=0)
                null_g = np.mean([g_sco[j] for j in pm])
                v["nl_null_mark"][r, w] = np.sum(own - null_m)
                v["nl_null_full"][r, w] = np.sum((own - g_sco[m]) - (null_m - null_g))
            if loc_partner[w, 0] >= 0:
                lp = [centre[j] for j in loc_partner[w]]
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
    centre = rng.integers(0, 40, n)
    distance = linear_distance(40)

    nl, loc = null_partners(pool, centre, distance, n_partners=20, min_cm=20.0, seed=4)

    assert (nl[0] == -1).all() and (loc[0] == -1).all()
    for w in range(1, n):
        assert (nl[w] != w).all() and (pool[nl[w]] == pool[w]).all()
        if loc[w, 0] >= 0:
            assert (loc[w] != w).all() and (pool[loc[w]] == pool[w]).all()
            assert (distance[centre[w], centre[loc[w]]] >= 20.0).all()
        else:  # no window of its pool is 20 cm away
            others = (pool == pool[w]) & (np.arange(n) != w)
            assert (distance[centre[w], centre[others]] < 20.0).all()


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
    centre = animal[windows.s + (windows.e - windows.s + 1) // 2]
    nl, loc = null_partners(windows["pool"], centre, distance, n_partners=6, seed=15)
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
    centre = animal[windows.s + (windows.e - windows.s + 1) // 2]
    _, loc = null_partners(windows["pool"], centre, distance, n_partners=5, seed=25)
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
        partners = centre[loc[w]]
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
    """One epoch's windows and evidence under two encodings, as the runner builds them."""
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
    for prefix, terms_seed in [("run", 7), ("can", 8)]:
        evidence = window_evidence(
            random_terms(n_time, n_pos, n_tet, rate=0.3, seed=terms_seed),
            windows, animal, linear_distance(n_pos), halves,
            n_partners=5, no_spike_rate=NO_SPIKE_RATE, seed=(seed, epoch_id),
        )  # fmt: skip
        shared = ["n_spikes", "n_tetrodes", "animal_center_bin", "has_loc_partner"]
        if prefix == "run":
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
    assert set(summary["trends"].state) == set(STATES)


# --- statistics ---------------------------------------------------------------------
def test_sign_flip_p_and_stratified_bootstrap_ci():
    rng = np.random.default_rng(0)
    shifted = rng.normal(1.0, 1.0, 200)
    half = rng.normal(0.0, 1.0, 100)
    centred = np.r_[half, -half]  # mean exactly 0: every flip is as extreme
    strata = np.repeat(["a", "b"], 100)

    assert sign_flip_p(shifted, seed=(1,)) == pytest.approx(1 / 10_001)
    assert sign_flip_p(centred, seed=(1,)) == pytest.approx(1.0)
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

    stats = bout_statistics(table, metrics={"Local": ["run_loc_mark"]})
    row = stats.set_index(["state", "regime", "metric"]).loc[
        ("Local", "immobile", "run_loc_mark")
    ]

    assert row["n"] == 120 and row["n_found"] == 120 and row["n_epochs"] == 2
    assert row["median"] == pytest.approx(bouts.run_loc_mark.median())
    assert row["q25"] == pytest.approx(bouts.run_loc_mark.quantile(0.25))
    assert row["frac_pos"] == pytest.approx((bouts.run_loc_mark > 0).mean())
    assert row["mean"] == pytest.approx(bouts.run_loc_mark.mean())
    assert row["ci_lo"] < row["mean"] < row["ci_hi"]
    assert row["p_signflip"] == pytest.approx(1 / 10_001)
    assert row["base_median"] == pytest.approx(base.run_loc_mark.median())
    assert row["diff_mean"] == pytest.approx(
        bouts.run_loc_mark.mean() - base.run_loc_mark.mean()
    )
    assert row["diff_ci_lo"] < row["diff_mean"] < row["diff_ci_hi"]
    assert row["epochs_pos"] == "2/2" and row["epochs_above_base"] == "2/2"
    assert row["animals_pos"] == "2/2" and row["animals_above_base"] == "2/2"
    # a bout whose matched window is missing is an error, not a silent drop
    with pytest.raises(ValueError, match="matched"):
        bout_statistics(
            table.drop(index=base.index[0]), metrics={"Local": ["run_loc_mark"]}
        )


def test_calibration_statistics_bin_by_probability_and_call_the_trend():
    table = evidence_table()
    cal = table[table.kind == "calib"]

    bins, trends = calibration_statistics(table, metrics={"Local": ["run_loc_mark"]})

    top = bins[(bins.metric == "run_loc_mark") & (bins.pbin == "0.8-1.0")].iloc[0]
    in_top = cal[cal.p_Local >= 0.8]
    assert top["n"] == len(in_top)
    assert top["mean"] == pytest.approx(in_top.run_loc_mark.mean())
    assert top["ci_lo"] < top["mean"] < top["ci_hi"]
    curve = trends.set_index(["state", "regime", "metric"]).loc[
        ("Local", "immobile", "run_loc_mark")
    ]
    assert curve["slope"] == pytest.approx(4.0, abs=0.2)
    assert curve["rises"] and curve["top_positive"]
    flat = table.assign(run_loc_mark=np.where(table.kind == "calib", -1.0, 0.0))
    flat.loc[flat.kind == "calib", "run_loc_mark"] += np.random.default_rng(0).normal(
        0, 0.1, (flat.kind == "calib").sum()
    )
    _, flat_trends = calibration_statistics(flat, metrics={"Local": ["run_loc_mark"]})
    assert not flat_trends.rises.iloc[0] and not flat_trends.top_positive.iloc[0]


def test_rate_calibration_and_coordinated_silence():
    rng = np.random.default_rng(6)
    n = 300
    table = pd.DataFrame(
        {
            "epoch_id": np.repeat(["aa20200101_02_r1_HPC", "bb20200101_02_r1_HPC"], n // 2),
            "animal": np.repeat(["aa", "bb"], n // 2),
            "kind": "calib", "state": "", "regime": "immobile",
            "p_No-Spike": rng.random(n),
            "run_ns_count": rng.poisson(1.0, n).astype(float),
            "run_ns_expected": rng.uniform(0.5, 2.0, n),
            "can_ns_expected": rng.uniform(0.5, 1.5, n),
            "run_ns_silent_frac": rng.random(n),
            "run_ns_pred_silent": rng.random(n),
            "can_ns_pred_silent": rng.random(n),
            "run_cond_S_silent": rng.integers(0, 20, n).astype(float),
        }
    )  # fmt: skip
    table["run_cond_both_silent"] = np.floor(table.run_cond_S_silent * 0.5)
    table["run_cond_pred_C_silent"] = table.run_cond_S_silent * 0.2
    table["can_cond_pred_C_silent"] = table.run_cond_S_silent * 0.3

    rate = rate_calibration(table).set_index(["regime", "encoding", "animal"])
    pooled = rate.loc[("immobile", "running-only", "all")]
    assert pooled["observed_over_expected"] == pytest.approx(
        table.run_ns_count.sum() / table.run_ns_expected.sum()
    )
    bb = table[table.animal == "bb"]
    assert rate.loc[("immobile", "canonical", "bb")][
        "observed_over_expected"
    ] == pytest.approx(bb.run_ns_count.sum() / bb.can_ns_expected.sum())

    silence = coordinated_silence(table)
    every = silence[(silence.kind == "calib") & (silence.state == "all windows")].iloc[
        0
    ]
    assert every["obs_C_silent_given_S_silent"] == pytest.approx(
        table.run_cond_both_silent.sum() / table.run_cond_S_silent.sum()
    )
    assert every["pred_given_S_can"] == pytest.approx(0.3)
    assert every["pred_given_S_run"] == pytest.approx(0.2)
    assert every["obs_given_ci_lo"] < every["obs_C_silent_given_S_silent"]
    assert every["obs_C_silent_given_S_silent"] < every["obs_given_ci_hi"]
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
