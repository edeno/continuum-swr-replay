"""Cohort-audit records for the rate runners (Family A/B + swr-contrast).

C3 regression guard for C1: the per-bin / contrast audits must attribute
`immobility<=0` and NaN stop-level-predictor drops to `n_drop_own_predictor_missing`
(model-specific), not silently fold them into `n_modeled`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.replay.cohort_audit import DROP_FIELDS

SC = ["animal", "day", "epoch"]


def _reconciles(rec: dict) -> bool:
    return rec["n_in"] - sum(rec[f] for f in DROP_FIELDS) == rec["n_modeled"]


def _stop_table(n_per_session=20, n_immobility_bad=2, n_rewardlag_nan=3, seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    for animal, day in [("a", 1), ("b", 2)]:
        n = n_per_session
        df = pd.DataFrame({
            "animal": animal, "day": day, "epoch": "e",
            "immobility_time": rng.uniform(5, 30, n),
            **{f"reward_lag{j}": rng.integers(0, 2, n).astype(float) for j in range(1, 11)},
        })
        frames.append(df)
    stop = pd.concat(frames, ignore_index=True)
    stop.loc[: n_immobility_bad - 1, "immobility_time"] = 0.0  # immobility<=0 own-drop
    stop.loc[
        n_immobility_bad : n_immobility_bad + n_rewardlag_nan - 1, "reward_lag1"
    ] = np.nan  # reward_lag1 NaN own-drop (only reward-using models)
    return stop


def test_per_bin_records_attribute_own_drops():
    from scripts.rate import fit_per_bin_rate_models as mod

    stop = _stop_table()
    total = len(stop)
    # Synthetic models: M0_const and M1_rew1 fit on 'non_local' for both sessions.
    models = pd.DataFrame([
        {"model": m, "outcome": "non_local", "animal": a, "day": d, "epoch": "e",
         "n_stops": 15, "converged": True, "separated": False}
        for m in ("M0_const", "M1_rew1") for a, d in [("a", 1), ("b", 2)]
    ])
    recs = mod._cohort_records(models, stop)
    assert all(_reconciles(r) for r in recs)

    by = {(r["model"], r["population"]): r for r in recs}
    m0 = by[("M0_const", "non_local")]
    m1 = by[("M1_rew1", "non_local")]
    # Const rung uses no reward lag -> own drops are only the immobility<=0 stops.
    assert m0["n_drop_own_predictor_missing"] == 2
    # Reward rung additionally drops the reward_lag1 NaN stops.
    assert m1["n_drop_own_predictor_missing"] == 5
    # n_modeled is the fitter's stop count (2 sessions x 15), not "all stops of session".
    assert m0["n_modeled"] == 30
    assert m0["n_in"] == total


def test_contrast_records_attribute_qvals_and_own():
    from scripts.rate import fit_swr_contrast_rate as mod

    stop = _stop_table()
    # Post-join stop table carries hmm_* (non-NaN after the inner join).
    stop["hmm_relative_value"] = 0.1
    stop["hmm_rpe"] = 0.0
    models = pd.DataFrame([
        {"model": "G0_base", "population": "non_local", "animal": a, "day": d,
         "epoch": "e", "n_stops": 12, "converged": True, "separated": False}
        for a, d in [("a", 1), ("b", 2)]
    ])
    recs = mod._cohort_records(models, stop, n_qvals_dropped=7)
    assert all(_reconciles(r) for r in recs)

    g0 = {(r["model"], r["population"]): r for r in recs}[("G0_base", "all")]
    assert g0["n_drop_qvals_gap"] == 7  # the inner-join drop (mechanism B)
    # G0_base's need includes reward_lag1 -> immobility(2) + reward_lag1 NaN(3) = 5 own.
    assert g0["n_drop_own_predictor_missing"] == 5
    assert g0["n_modeled"] == 24  # 2 sessions x 12, the fitter's stop count
    assert g0["n_in"] == len(stop) + 7


def test_uncertainty_records_reconcile_and_sessions_in_is_prefirstdrop():
    """The uncertainty audit must report sessions_in for the post-join (pre-first-drop)
    population so it matches n_in, and attribute the qvals-join drop to qvals_gap."""
    from scripts.rate import fit_uncertainty_rate as mod

    # Fitted frame = 2 sessions (one animal's first session already dropped upstream).
    rng = np.random.default_rng(0)
    stop_fitted = pd.concat([
        pd.DataFrame({
            "animal": a, "day": d, "epoch": "e",
            "immobility_time": rng.uniform(5, 30, 15),
            **{f"reward_lag{j}": rng.integers(0, 2, 15).astype(float) for j in range(1, 11)},
        })
        for a, d in [("a", 2), ("b", 3)]
    ], ignore_index=True)
    stop_fitted.loc[0:1, "immobility_time"] = 0.0  # 2 own-drops

    models = pd.DataFrame([
        {"model": mod.MU_UNCERTAINTY.name, "outcome": o, "animal": a, "day": d,
         "epoch": "e", "n_stops": 10, "converged": True, "separated": False}
        for o in mod.POPULATIONS_TO_FIT for a, d in [("a", 2), ("b", 3)]
    ])
    recs = mod._cohort_records(
        models, stop_fitted, n_postjoin=100, n_sessions_postjoin=3, qvals_gap=8,
    )
    assert recs and all(_reconciles(r) for r in recs)
    r0 = recs[0]
    # sessions_in is the pre-first-drop count (3), NOT the fitted frame's 2.
    assert r0["n_sessions_in"] == 3
    assert r0["n_drop_qvals_gap"] == 8
    assert r0["n_drop_own_predictor_missing"] == 2  # immobility<=0
    assert r0["n_in"] == 108  # n_postjoin 100 + qvals_gap 8


def test_swr_power_records_split_qvals_and_own():
    from scripts.rate import fit_swr_power_model as mod

    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        "animal": np.repeat(["a", "b"], n // 2), "day": 1, "epoch": "e",
        **{c: rng.normal(size=n) for c in mod.PREDICTORS},
        mod.POWER_COL: rng.normal(size=n),
    })
    df.loc[0:9, "q_current"] = np.nan       # 10 qvals-derived NaN -> qvals_gap
    df.loc[100:104, "reward_lag1"] = np.nan  # 5 own-predictor NaN -> own
    coefs = pd.DataFrame([
        {"animal": "a", "day": 1, "epoch": "e", "n_events": 85},
        {"animal": "b", "day": 1, "epoch": "e", "n_events": 90},
    ])
    recs = mod._cohort_records(df, coefs)
    assert all(_reconciles(r) for r in recs)
    r = recs[0]
    assert r["n_drop_qvals_gap"] == 10
    assert r["n_drop_own_predictor_missing"] == 5
    assert r["n_in"] == n


def test_per_stop_hierarchy_records_reconcile():
    from scripts.rate import fit_per_stop_rate_models as mod

    rng = np.random.default_rng(1)
    frames = []
    for a, d in [("a", 1), ("b", 2)]:
        n = 60
        frames.append(pd.DataFrame({
            "animal": a, "day": d, "epoch": "e",
            "n_non_local": rng.poisson(2, n).astype(float),
            "immobility_time": rng.uniform(5, 30, n),
            **{f"reward_lag{j}": rng.integers(0, 2, n).astype(float) for j in range(1, 11)},
            "is_after_omission": rng.integers(0, 2, n).astype(float),
            "is_after_switch": rng.integers(0, 2, n).astype(float),
            "recent_trial_reward_rate": rng.uniform(0, 1, n),
        }))
    stop = pd.concat(frames, ignore_index=True)
    stop.loc[0:1, "immobility_time"] = 0.0
    _, recs = mod.fit_per_stop_rate_hierarchy(stop, run_cv=False)
    assert recs and all(_reconciles(r) for r in recs)
    # immobility<=0 stops are attributed as own drops.
    assert any(r["n_drop_own_predictor_missing"] >= 2 for r in recs)
