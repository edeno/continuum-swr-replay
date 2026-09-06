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
    REWARD_HISTORY_N_BASIS,
    PerBinRateModelSpec,
    aggregate_to_timebins,
    cross_validate_per_bin_rate_model,
    fit_per_bin_rate_model,
    freeze_per_bin_cohort,
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
    arrival_clocks: bool = False,
    arrival_lead_max: float = 3.0,
    arrival_time_profile=None,
    seed: int = 0,
) -> pd.DataFrame:
    """Simulate an ``aggregate_to_timebins``-shaped design with known truth.

    Half the complete stops have the immediate reward-history component at 1
    (``reward_history_rcos_0 == 1``); their bins draw their onset count from a
    Poisson with ``base_rate * reward_effect`` (times the optional
    ``time_profile(tc)``), so the true per-bin rate ratio on
    ``reward_history_rcos_0`` is ``reward_effect`` and the true time course is
    ``time_profile``. The slower components carry independent noise and every
    tenth stop is structurally incomplete (features zero-filled, indicator 1,
    baseline rate), mirroring the persisted stop-table contract.

    With ``arrival_clocks``, each stop also carries the absolute clocks
    (``stop_start``, ``last_well_arrival_time_s``, ``outcome_known_time_s``) and
    the two per-bin clocks ``aggregate_to_timebins`` derives from them. Each
    stop's arrival is drawn uniformly ``[0, arrival_lead_max)`` seconds before
    its bout starts, so the two clocks disagree by a per-stop offset, and
    ``arrival_time_profile`` (a callable on the arrival clock, in seconds since
    arrival) multiplies the rate the same way ``time_profile`` does on the stop
    clock. The boundary-shifted reward history is not simulated here (it is
    covered on real stop-table inputs in ``test_per_bin_rate_helpers``), so the
    lag columns it reads are deliberately absent rather than inconsistent with
    the planted ``reward_history_rcos_*``. Off by default, so the shipped
    ladder's designs are exactly what they were.
    """
    rng = np.random.default_rng(seed)
    leads = (
        rng.uniform(0.0, arrival_lead_max, n_stops)
        if arrival_clocks
        else np.zeros(n_stops)
    )
    recs = []
    for s in range(n_stops):
        incomplete = float(s % 10 == 0)
        reward = 0.0 if incomplete else float(s % 2)
        rcos1 = 0.0 if incomplete else float(rng.uniform(0, 1))
        rcos2 = 0.0 if incomplete else float(rng.uniform(0, 1))
        for b in range(n_bins):
            tc = (b + 0.5) * bin_s
            tp = time_profile(tc) if time_profile is not None else 1.0
            ap = (
                arrival_time_profile(tc + leads[s])
                if arrival_time_profile is not None
                else 1.0
            )
            mu = base_rate * tp * ap * (reward_effect if reward else 1.0)
            recs.append(
                ("sim", 1, "02_r1", s, b, tc, bin_s, reward, rcos1, rcos2,
                 incomplete, int(rng.poisson(mu)))
            )
    design = pd.DataFrame(
        recs,
        columns=[
            "animal", "session_date", "epoch", "stop_id", "binj", "tc",
            "exposure", "reward_history_rcos_0", "reward_history_rcos_1",
            "reward_history_rcos_2", "reward_history_incomplete", "count",
        ],
    )
    if not arrival_clocks:
        return design

    # Stops start 10 s apart; the animal arrived ``leads[stop]`` seconds before
    # each bout and the outcome became observable 0.2 s into it, so the arrival
    # clock is positive throughout and the outcome clock changes sign inside
    # every stop.
    stop_start = 1000.0 + 10.0 * design["stop_id"].to_numpy(dtype=float)
    design["stop_start"] = stop_start
    design["last_well_arrival_time_s"] = (
        stop_start - leads[design["stop_id"].to_numpy()]
    )
    design["outcome_known_time_s"] = stop_start + 0.2
    center_abs = stop_start + design["tc"].to_numpy(dtype=float)
    arrival = design["last_well_arrival_time_s"].to_numpy(dtype=float)
    outcome = design["outcome_known_time_s"].to_numpy(dtype=float)
    design["time_since_last_well_arrival_s"] = center_abs - arrival
    design["time_since_outcome_known_s"] = center_abs - outcome
    return design


def test_formula_strings():
    m = MODEL_PER_BIN_RATE
    # Fixed-knot spline: deterministic basis (reconstructable / cross-session
    # comparable), so the term carries explicit knots + bounds, not df.
    spline = "bs(tc, knots=(1.0, 2.0, 3.0), lower_bound=0.0, upper_bound=4.0)"
    assert m["M0_const"].formula() == "count ~ 1"
    assert m["M5_time"].formula() == f"count ~ {spline}"

    rewhist = m["M1_rew"].formula()
    assert "bs(tc" not in rewhist
    for k in range(3):
        assert f"reward_history_rcos_{k}" in rewhist
    assert "reward_history_incomplete" in rewhist

    full = m["M7_time_hist_rew"].formula()
    assert spline in full
    for k in range(3):
        assert f"reward_history_rcos_{k}" in full
    for j in range(1, 11):
        assert f"onset_lag{j}" in full

    assert f"reward_history_rcos_0:{spline}" in m["M9_rew_x_time"].formula()

    # Raised-cosine self-history variants use hist_rcos_*, not onset_lag dummies.
    m3r = m["M3r_histcos"].formula()
    assert "hist_rcos_0" in m3r and "onset_lag1" not in m3r
    m7r = m["M7r_time_histcos_rew"].formula()
    assert "hist_rcos_0" in m7r and "onset_lag1" not in m7r
    assert spline in m7r and "reward_history_rcos_0" in m7r
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
        base.reward_basis,
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
        base.reward_basis,
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
    fit = fit_per_bin_rate_model(design, MODEL_PER_BIN_RATE["M1_rew"])
    assert fit is not None
    assert fit["rr_reward_history_rcos_0"] == pytest.approx(2.0, rel=0.10)


def test_spline_captures_time_peak():
    # A peaked time course at tc ~ 0.4 s, no reward effect.
    def bump(tc):
        return 1.0 + 3.0 * np.exp(-((tc - 0.4) ** 2) / (2 * 0.1**2))

    design = make_per_bin_rate_design(
        reward_effect=1.0, base_rate=0.06, time_profile=bump, n_bins=60, seed=2
    )
    spec = PerBinRateModelSpec(
        "time_rew", True, REWARD_HISTORY_N_BASIS, False, 0, False, False, "test", None
    )
    fit = fit_per_bin_rate_model(design, spec)
    assert fit is not None
    # No planted reward effect.
    assert fit["rr_reward_history_rcos_0"] == pytest.approx(1.0, rel=0.12)
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
        "animal", "session_date", "epoch", "stop_id", "binj", "tc", "exposure",
        "reward_history_rcos_0", "onset_lag1", "count"])
    spec = PerBinRateModelSpec(
        "hist1", False, 0, False, 1, False, False, "test", None
    )
    # Must not raise; returns None or a finite fit dict.
    result = fit_per_bin_rate_model(design, spec)
    assert result is None or np.isfinite(result["llf"])


def test_cv_folds_score_and_beat_constant():
    design = make_per_bin_rate_design(reward_effect=2.5, base_rate=0.08, seed=4)
    cv = cross_validate_per_bin_rate_model(
        design, MODEL_PER_BIN_RATE["M1_rew"], n_folds=10
    )
    assert cv is not None
    assert cv["cv_n_folds"] == 10
    assert np.isfinite(cv["cv_ll_per_bin_mean"])
    # A real held-out reward effect must beat the constant-rate baseline.
    assert cv["cv_bits_per_event"] > 0
    # A clean run reports zero bad folds.
    assert cv["cv_n_nonconverged_folds"] == 0
    assert cv["cv_n_separated_folds"] == 0


def test_cv_one_bad_fold_invalidates_metric(monkeypatch):
    """A single nonconverged fold (model OR baseline) invalidates the whole CV
    metric to nan without dropping the fold; counts are reported alongside."""
    import continuum_swr_replay.analysis.replay.per_bin_rate_model as pbm

    calls = {"n": 0}
    real = pbm.fit_status

    def fake_fit_status(result, kind, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:  # first fold's model fit only
            return {"converged": False, "separated": False}
        return real(result, kind, **kwargs)

    monkeypatch.setattr(pbm, "fit_status", fake_fit_status)
    design = make_per_bin_rate_design(reward_effect=2.5, base_rate=0.08, seed=4)
    cv = cross_validate_per_bin_rate_model(
        design, MODEL_PER_BIN_RATE["M1_rew"], n_folds=5
    )
    assert cv is not None  # fold NOT dropped
    assert cv["cv_n_folds"] == 5
    assert cv["cv_n_nonconverged_folds"] == 1
    assert cv["cv_n_separated_folds"] == 0
    assert np.isnan(cv["cv_ll_per_bin_mean"])
    assert np.isnan(cv["cv_bits_per_event"])


def test_cv_folds_split_on_stop(monkeypatch):
    """Folds partition whole stops -- a stop's bins never straddle train/test.

    Exercises the real ``cross_validate_per_bin_rate_model`` split by recording
    the ``stop_id`` sets it actually hands statsmodels: ``smf.glm(data=tr)``
    supplies the train stops, and the fitted result's ``predict(te, ...)``
    supplies the held-out stops. A bin-level split (or an ``isin`` on the wrong
    side) would show up here as overlapping train/test stop sets, which the
    old KFold-replicating test could not see.
    """
    import continuum_swr_replay.analysis.replay.per_bin_rate_model as pbm

    real_glm = pbm.smf.glm
    recorded: list[tuple[frozenset[int], frozenset[int]]] = []

    class _RecordingResult:
        """Real fit result; records the held-out stops each ``predict`` scores."""

        def __init__(self, result, train_stops: frozenset[int]) -> None:
            self._result = result
            self._train_stops = train_stops

        def predict(self, exog=None, *args, **kwargs):
            recorded.append((self._train_stops, frozenset(exog["stop_id"])))
            return self._result.predict(exog, *args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._result, name)

    class _RecordingModel:
        def __init__(self, model, train_stops: frozenset[int]) -> None:
            self._model = model
            self._train_stops = train_stops

        def fit(self, *args, **kwargs):
            return _RecordingResult(self._model.fit(*args, **kwargs), self._train_stops)

        def __getattr__(self, name):
            return getattr(self._model, name)

    def recording_glm(formula, data=None, **kwargs):
        return _RecordingModel(
            real_glm(formula, data=data, **kwargs), frozenset(data["stop_id"])
        )

    monkeypatch.setattr(pbm.smf, "glm", recording_glm)

    design = make_per_bin_rate_design(n_stops=120, seed=8)
    frozen, report = freeze_per_bin_cohort(design, [MODEL_PER_BIN_RATE["M1_rew"]])
    assert report["n_stops_after"] == 120
    all_stops = frozenset(frozen["stop_id"])

    cv = cross_validate_per_bin_rate_model(
        frozen, MODEL_PER_BIN_RATE["M1_rew"], n_folds=10
    )
    assert cv is not None
    assert cv["cv_n_folds"] == 10
    # The recorder keeps real predict semantics, so the CV still scores.
    assert np.isfinite(cv["cv_ll_per_bin_mean"])

    # Model and M0_const baseline are fit on the same train fold and scored on
    # the same held-out fold, so 10 folds produce 20 records over 10 distinct
    # (train, test) stop-set pairs.
    assert len(recorded) == 20
    folds = sorted(set(recorded), key=lambda pair: sorted(pair[1]))
    assert len(folds) == 10
    for train_stops, test_stops in folds:
        assert train_stops.isdisjoint(test_stops)
        assert len(test_stops) == 12
        assert train_stops | test_stops == all_stops
    assert frozenset().union(*(test for _, test in folds)) == all_stops
    # Each stop is held out exactly once.
    assert sum(len(test) for _, test in folds) == 120


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
            "animal": "x", "session_date": 1, "epoch": "02_r1",
            "stop_id": stop_ids, "immobility_duration_s": nbin * bin_s,
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


def test_freeze_per_bin_cohort_makes_rungs_share_rows():
    """Freezing puts every rung of a ladder on the same rows; unfrozen raises.

    A covariate that only ONE rung uses is missing on a tenth of the stops. Per
    rung NaN-dropping would fit that rung on 90% of the bins and the others on
    100%, so their AIC / deviance would not be comparable. The frozen cohort
    removes those stops once, for the whole ladder.
    """
    from continuum_swr_replay.analysis.replay import freeze_per_bin_cohort

    design = make_per_bin_rate_design(n_stops=300, n_bins=40, seed=7)
    rng = np.random.default_rng(7)
    design["extra_cov"] = rng.normal(size=len(design))
    design.loc[design["stop_id"] % 10 == 0, "extra_cov"] = np.nan

    with_extra = PerBinRateModelSpec(
        "M_extra", True, REWARD_HISTORY_N_BASIS, False, 0, False, False,
        "test", None, extra_covariates=("extra_cov",),
    )
    ladder = [
        MODEL_PER_BIN_RATE["M0_const"],
        MODEL_PER_BIN_RATE["M1_rew"],
        with_extra,
    ]

    frozen, report = freeze_per_bin_cohort(design, ladder)
    assert frozen.attrs["per_bin_cohort_frozen"] is True
    assert report["n_stops_before"] == 300
    assert report["n_stops_after"] == 270  # the 30 stops with a NaN covariate
    assert report["n_bins_before"] == 300 * 40
    assert report["n_bins_after"] == 270 * 40
    assert report["n_event_bins_after"] == int((frozen["count"] > 0).sum())
    assert report["n_event_bins_after"] < report["n_event_bins_before"]
    assert "extra_cov" in report["required_columns"]
    assert len(design) == 300 * 40  # the input frame is not mutated

    fits = {spec.name: fit_per_bin_rate_model(frozen, spec) for spec in ladder}
    assert all(fit is not None for fit in fits.values())
    assert {fit["n_bins"] for fit in fits.values()} == {270 * 40}
    assert {fit["n_stops"] for fit in fits.values()} == {270}

    # The unfrozen design is refused rather than silently subset.
    with pytest.raises(ValueError, match="extra_cov"):
        fit_per_bin_rate_model(design, with_extra)
    with pytest.raises(ValueError, match="extra_cov"):
        cross_validate_per_bin_rate_model(design, with_extra, n_folds=5)

    # A zero-exposure bin is refused too (log(0) offset), and the freeze removes it.
    bad = frozen.copy()
    bad.loc[0, "exposure"] = 0.0
    with pytest.raises(ValueError, match="exposure <= 0"):
        fit_per_bin_rate_model(bad, MODEL_PER_BIN_RATE["M0_const"])
    assert freeze_per_bin_cohort(bad, ladder)[1]["n_bins_after"] == 270 * 40 - 1


def test_freeze_per_bin_cohort_requires_its_columns():
    """A design that cannot carry the ladder is a configuration error."""
    from continuum_swr_replay.analysis.replay import freeze_per_bin_cohort

    design = make_per_bin_rate_design(n_stops=20, n_bins=5, seed=0)
    with pytest.raises(KeyError, match="onset_lag1"):
        freeze_per_bin_cohort(design, [MODEL_PER_BIN_RATE["M3_hist"]])


def _design_with_separated_indicator(seed: int = 11) -> pd.DataFrame:
    """A design whose ``reward_history_incomplete`` stops carry no events at all.

    That is the real corpus shape: the indicator marks a handful of stops with
    fewer than two known outcomes, and in most sessions none of them contains an
    event, so the indicator's coefficient runs to -inf with an exploded SE while
    every other term is ordinary.
    """
    rng = np.random.default_rng(seed)
    recs = []
    for s in range(300):
        incomplete = float(s % 15 == 0)
        reward = 0.0 if incomplete else float(s % 2)
        for b in range(40):
            mu = 0.0 if incomplete else 0.08 * (2.5 if reward else 1.0)
            recs.append(
                ("sim", 1, "02_r1", s, b, (b + 0.5) * 0.015, 0.015,
                 reward, 0.0, 0.0, incomplete, int(rng.poisson(mu)))
            )
    return pd.DataFrame(recs, columns=[
        "animal", "session_date", "epoch", "stop_id", "binj", "tc", "exposure",
        "reward_history_rcos_0", "reward_history_rcos_1", "reward_history_rcos_2",
        "reward_history_incomplete", "count",
    ])


def test_fit_reports_separated_terms_and_predictor_sds():
    """Every fit carries the named separated terms and each term's design SD."""
    design = _design_with_separated_indicator()
    fit = fit_per_bin_rate_model(design, MODEL_PER_BIN_RATE["M1_rew"])

    assert fit is not None
    # The nuisance indicator IS degenerate here but must not be named.
    assert fit["separated_terms"] == ""
    assert fit["separated"] is False
    assert fit["sd_Intercept"] == 0.0
    assert fit["sd_reward_history_rcos_0"] == pytest.approx(
        design["reward_history_rcos_0"].std(ddof=0), rel=1e-6
    )


def test_cv_fold_separated_only_on_the_indicator_keeps_the_metric():
    """A fold whose only separated term is the nuisance indicator stays scored."""
    design = _design_with_separated_indicator()
    cv = cross_validate_per_bin_rate_model(
        design, MODEL_PER_BIN_RATE["M1_rew"], n_folds=5
    )

    assert cv is not None
    assert cv["cv_n_separated_folds"] == 0
    assert np.isfinite(cv["cv_ll_per_bin_mean"])
    assert np.isfinite(cv["cv_bits_per_event"])
