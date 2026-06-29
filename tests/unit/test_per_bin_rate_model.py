"""Tests for the per-bin conditional-intensity Poisson model (Family B).

Covers the ladder formula strings, parameter recovery on simulated designs with
a planted reward effect / time course, the perfect-separation no-crash
guarantee, stop-based cross-validation, and the point-process goodness-of-fit
wrapper (which must reuse ``non_local_detector.model_checking.sorted_spikes``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.replay import (
    MODEL_PER_BIN_RATE,
    PerBinRateModelSpec,
    aggregate_to_timebins,
    cross_validate_per_bin_rate_model,
    fit_per_bin_rate_model,
    per_bin_rate_gof,
)


def make_per_bin_rate_design(
    *,
    n_stops: int = 400,
    n_bins: int = 50,
    bin_s: float = 0.015,
    base_rate: float = 0.05,
    reward_effect: float = 1.0,
    time_profile=None,
    seed: int = 0,
) -> pd.DataFrame:
    """Simulate an ``aggregate_to_timebins``-shaped design with known truth.

    Half the stops are rewarded (``reward_lag1 == 1``); rewarded bins draw their
    onset count from a Poisson with ``base_rate * reward_effect`` (times the
    optional ``time_profile(tc)``), so the true per-bin reward rate ratio is
    ``reward_effect`` and the true time course is ``time_profile``.
    """
    rng = np.random.default_rng(seed)
    recs = []
    for s in range(n_stops):
        reward = float(s % 2)
        for b in range(n_bins):
            tc = (b + 0.5) * bin_s
            tp = time_profile(tc) if time_profile is not None else 1.0
            mu = base_rate * tp * (reward_effect if reward else 1.0)
            recs.append(
                ("sim", 1, "02_r1", s, b, tc, bin_s, reward, int(rng.poisson(mu)))
            )
    return pd.DataFrame(
        recs,
        columns=[
            "animal", "day", "epoch", "stop_id", "binj", "tc",
            "exposure", "reward_lag1", "count",
        ],
    )


def test_formula_strings():
    m = MODEL_PER_BIN_RATE
    # Fixed-knot spline: deterministic basis (reconstructable / cross-session
    # comparable), so the term carries explicit knots + bounds, not df.
    spline = "bs(tc, knots=(1.0, 2.0, 3.0), lower_bound=0.0, upper_bound=4.0)"
    assert m["M0_const"].formula() == "count ~ 1"
    assert m["M5_time"].formula() == f"count ~ {spline}"

    rewhist = m["M2_rewhist"].formula()
    assert "bs(tc" not in rewhist
    for j in range(1, 11):
        assert f"reward_lag{j}" in rewhist

    full = m["M7_time_hist_rew"].formula()
    assert spline in full
    for j in range(1, 11):
        assert f"reward_lag{j}" in full
        assert f"onset_lag{j}" in full

    assert f"reward_lag1:{spline}" in m["M9_rew_x_time"].formula()

    # Raised-cosine self-history variants use hist_rcos_*, not onset_lag dummies.
    m3r = m["M3r_histcos"].formula()
    assert "hist_rcos_0" in m3r and "onset_lag1" not in m3r
    m7r = m["M7r_time_histcos_rew"].formula()
    assert "hist_rcos_0" in m7r and "onset_lag1" not in m7r
    assert spline in m7r and "reward_lag1" in m7r
    # The dummy models still use onset_lag (kernel-shape readout preserved).
    assert "onset_lag1" in m["M7_time_hist_rew"].formula()


def test_extra_covariates_formula():
    # extra_covariates append as plain linear terms to formula() and
    # _design_columns; default () leaves the shipped ladder unchanged.
    from continuum_swr_replay.analysis.replay.per_bin_rate_model import (
        _design_columns,
    )

    base = MODEL_PER_BIN_RATE["M7r_time_histcos_rew"]
    spec = PerBinRateModelSpec(
        base.name,
        base.time_spline,
        base.reward_lags,
        base.reward_rate,
        base.self_history_lags,
        base.switch,
        base.reward_time_interaction,
        base.description,
        base.parent,
        self_history_basis=base.self_history_basis,
        extra_covariates=("a", "b"),
    )
    f = spec.formula()
    assert f.endswith("+ a + b")
    cols = _design_columns(spec)
    assert "a" in cols and "b" in cols

    # Default () => shipped specs are byte-identical (no trailing terms added).
    for shipped in MODEL_PER_BIN_RATE.values():
        assert shipped.extra_covariates == ()
        assert not shipped.formula().rstrip().endswith("+ ")
    # The empty-extra spec matches the shipped formula exactly.
    same = PerBinRateModelSpec(
        base.name,
        base.time_spline,
        base.reward_lags,
        base.reward_rate,
        base.self_history_lags,
        base.switch,
        base.reward_time_interaction,
        base.description,
        base.parent,
        self_history_basis=base.self_history_basis,
    )
    assert same.formula() == base.formula()


def test_recovers_known_reward_effect():
    design = make_per_bin_rate_design(reward_effect=2.0, base_rate=0.08, seed=1)
    fit = fit_per_bin_rate_model(design, MODEL_PER_BIN_RATE["M1_rew1"])
    assert fit is not None
    assert fit["rr_reward_lag1"] == pytest.approx(2.0, rel=0.10)


def test_spline_captures_time_peak():
    # A peaked time course at tc ~ 0.4 s, no reward effect.
    def bump(tc):
        return 1.0 + 3.0 * np.exp(-((tc - 0.4) ** 2) / (2 * 0.1**2))

    design = make_per_bin_rate_design(
        reward_effect=1.0, base_rate=0.06, time_profile=bump, n_bins=60, seed=2
    )
    spec = PerBinRateModelSpec(
        "time_rew1", True, 1, False, 0, False, False, "test", None
    )
    fit = fit_per_bin_rate_model(design, spec)
    assert fit is not None
    # No planted reward effect.
    assert fit["rr_reward_lag1"] == pytest.approx(1.0, rel=0.12)
    # Spline captures the peak: predicted near the peak tc exceeds the edges.
    pred = fit["predicted"]
    binj = design.loc[pred.index, "binj"].to_numpy()
    by_bin = pd.Series(pred.to_numpy(), index=binj).groupby(level=0).mean()
    peak_bin = int(0.4 / 0.015)
    assert by_bin.loc[peak_bin] > by_bin.iloc[0]
    assert by_bin.loc[peak_bin] > by_bin.iloc[-1]


def test_separation_no_crash():
    # onset_lag1 > 0 perfectly predicts count == 0 (hard refractoriness).
    # Poisson separation only warns (does not raise), so a finite dict with an
    # extreme coefficient is the expected output -- the point is no crash.
    rng = np.random.default_rng(3)
    recs = []
    for s in range(300):
        for b in range(40):
            onset_lag1 = float(b % 5 == 0)  # structured "previous-bin had event"
            count = 0 if onset_lag1 else int(rng.poisson(0.1))
            recs.append(("sim", 1, "02_r1", s, b, (b + 0.5) * 0.015, 0.015,
                         float(s % 2), onset_lag1, count))
    design = pd.DataFrame(recs, columns=[
        "animal", "day", "epoch", "stop_id", "binj", "tc", "exposure",
        "reward_lag1", "onset_lag1", "count"])
    spec = PerBinRateModelSpec(
        "hist1", False, 0, False, 1, False, False, "test", None
    )
    # Must not raise; returns None or a finite fit dict.
    result = fit_per_bin_rate_model(design, spec)
    assert result is None or np.isfinite(result["llf"])


def test_cv_folds_score_and_beat_constant():
    design = make_per_bin_rate_design(reward_effect=2.5, base_rate=0.08, seed=4)
    cv = cross_validate_per_bin_rate_model(
        design, MODEL_PER_BIN_RATE["M1_rew1"], n_folds=10
    )
    assert cv is not None
    assert cv["cv_n_folds"] == 10
    assert np.isfinite(cv["cv_ll_per_bin_mean"])
    # A real held-out reward effect must beat the constant-rate baseline.
    assert cv["cv_bits_per_event"] > 0


def test_cv_folds_split_on_stop():
    """Folds partition whole stops -- a stop's bins never straddle train/test.

    Replicates the splitting the CV performs (KFold over the unique stop_ids,
    then ``isin`` to assign bins) and asserts every fold's train and test stop
    sets are disjoint and every stop is held out exactly once. This locks the
    "split on stops, not bins" contract the leakage-free CV depends on.
    """
    from sklearn.model_selection import KFold

    design = make_per_bin_rate_design(n_stops=120, seed=8)
    stops = design["stop_id"].unique()
    kf = KFold(n_splits=10, shuffle=True, random_state=0)
    held_out: list[int] = []
    for train_idx, test_idx in kf.split(stops):
        train_stops, test_stops = set(stops[train_idx]), set(stops[test_idx])
        assert train_stops.isdisjoint(test_stops)
        # Bins assigned by the same isin rule never share a stop across sides.
        train_bins = design[design.stop_id.isin(train_stops)]
        test_bins = design[design.stop_id.isin(test_stops)]
        assert set(train_bins.stop_id).isdisjoint(set(test_bins.stop_id))
        held_out.extend(test_stops)
    assert sorted(held_out) == sorted(stops)  # each stop held out exactly once


def test_gof_reuses_package():
    # Import path is the package's sorted-spikes model-checking module.
    import non_local_detector.model_checking.sorted_spikes as ss

    assert hasattr(ss, "TimeRescaling")

    # A truly homogeneous-Poisson design fitted by M0_const lands in the KS band.
    design = make_per_bin_rate_design(
        reward_effect=1.0, base_rate=0.02, n_stops=500, n_bins=60, seed=5
    )
    fit = fit_per_bin_rate_model(design, MODEL_PER_BIN_RATE["M0_const"])
    assert fit is not None
    gof = per_bin_rate_gof(design, fit)
    assert {"ks_statistic", "ks_within_band", "dispersion"} <= set(gof)
    assert gof["ks_within_band"] is True


def test_cross_history_recovers_planted_excitation():
    """A planted cross-stream excitation (response onsets elevated 1-2 bins after
    a cross onset) is recovered as positive cross_lag1/cross_lag2 coefficients.

    This is the bivariate / Hawkes-style coupling: the response population's
    conditional intensity depends on the recent history of a *different* onset
    stream (e.g. non-SWR rate driven by recent with-SWR events).
    """
    rng = np.random.default_rng(0)
    n_stops, nbin, bin_s = 300, 20, 0.1
    stop_ids = np.arange(n_stops)
    stops = pd.DataFrame(
        {
            "animal": "x", "day": 1, "epoch": "02_r1",
            "stop_id": stop_ids, "immobility_time": nbin * bin_s,
        }
    )
    cross_rows, resp_rows = [], []
    for sid in stop_ids:
        cross_bins = np.flatnonzero(rng.random(nbin) < 0.12)
        cross_rows += [(sid, (b + 0.3) * bin_s) for b in cross_bins]
        recent = np.zeros(nbin, dtype=bool)
        for b in cross_bins:  # excitation in the 1-2 bins AFTER a cross onset
            for lag in (1, 2):
                if b + lag < nbin:
                    recent[b + lag] = True
        counts = rng.poisson(np.where(recent, 0.5, 0.05))  # 10x elevation
        for b in range(nbin):
            resp_rows += [(sid, (b + 0.3) * bin_s)] * int(counts[b])
    cross = pd.DataFrame(cross_rows, columns=["stop_id", "t_since_stop_onset"])
    resp = pd.DataFrame(resp_rows, columns=["stop_id", "t_since_stop_onset"])

    design = aggregate_to_timebins(
        stops, resp, bin_s=bin_s, tmax=nbin * bin_s,
        cross_onset_times=cross, cross_history_lags=4,
    )
    spec = PerBinRateModelSpec(
        "Mx", False, 0, False, 0, False, False, "", None, cross_history_lags=4,
    )
    fit = fit_per_bin_rate_model(design, spec, min_bins=50, min_events=20)
    assert fit is not None
    # planted excitation at lags 1-2 -> positive coefficients there
    assert fit["coef_cross_lag1"] > 0
    assert fit["coef_cross_lag2"] > 0
    # and stronger than the un-planted later lag
    assert fit["coef_cross_lag1"] > fit["coef_cross_lag4"]
