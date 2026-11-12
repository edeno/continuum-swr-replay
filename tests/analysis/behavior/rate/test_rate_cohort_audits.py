"""Cohort-audit records for the rate runners (Family A/B + swr-contrast).

C3 regression guard for C1: the per-bin / contrast audits must attribute
`immobility<=0` and NaN stop-level-predictor drops to `n_drop_own_predictor_missing`
(model-specific), not silently fold them into `n_modeled`.
"""

from __future__ import annotations

import functools

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.behavior.cohort_audit import DROP_FIELDS
from tests.support._event_end_fixtures import with_event_ends

SC = ["animal", "session_date", "epoch"]


def _reconciles(rec: dict) -> bool:
    return rec["n_in"] - sum(rec[f] for f in DROP_FIELDS) == rec["n_modeled"]


def _stop_table(n_per_session=20, n_immobility_bad=2, n_rewardbasis_nan=3, seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    for animal, session_date in [("a", 1), ("b", 2)]:
        n = n_per_session
        df = pd.DataFrame(
            {
                "animal": animal,
                "session_date": session_date,
                "epoch": "e",
                "immobility_duration_s": rng.uniform(5, 30, n),
                **{f"reward_history_rcos_{k}": rng.uniform(0, 1, n) for k in range(3)},
                "reward_history_incomplete": 0.0,
            }
        )
        frames.append(df)
    stop = pd.concat(frames, ignore_index=True)
    stop.loc[: n_immobility_bad - 1, "immobility_duration_s"] = (
        0.0  # immobility<=0 own-drop
    )
    stop.loc[
        n_immobility_bad : n_immobility_bad + n_rewardbasis_nan - 1,
        "reward_history_rcos_0",
    ] = np.nan  # basis-feature NaN own-drop (only reward-using models)
    return stop


def test_per_bin_records_attribute_own_drops():
    from continuum_swr_replay.analysis.behavior.rate import (
        per_bin_rate_ladder as mod,
    )

    stop = _stop_table()
    total = len(stop)
    # Synthetic models: M0_const and M1_rew fit on 'non_local' for both sessions.
    models = pd.DataFrame(
        [
            {
                "model": m,
                "outcome": "non_local",
                "animal": a,
                "session_date": d,
                "epoch": "e",
                "n_stops": 15,
                "converged": True,
                "separated": False,
            }
            for m in ("M0_const", "M1_rew")
            for a, d in [("a", 1), ("b", 2)]
        ]
    )
    recs = mod.per_bin_cohort_records(models, stop)
    assert all(_reconciles(r) for r in recs)

    by = {(r["model"], r["population"]): r for r in recs}
    m0 = by[("M0_const", "non_local")]
    m1 = by[("M1_rew", "non_local")]
    # Const rung uses no reward history -> own drops are only the immobility<=0 stops.
    assert m0["n_drop_own_predictor_missing"] == 2
    # Reward rung additionally drops the NaN basis-feature stops.
    assert m1["n_drop_own_predictor_missing"] == 5
    # n_modeled is the fitter's stop count (2 sessions x 15), not "all stops of session".
    assert m0["n_modeled"] == 30
    assert m0["n_in"] == total


def test_contrast_records_attribute_qvals_and_own():
    from scripts.behavior.rate import fit_swr_contrast_rate as mod

    stop = _stop_table()
    # Post-join stop table carries the beta value scalars (non-NaN after the inner join).
    stop["relative_value_switch_minus_stay_preoutcome"] = 0.1
    stop["rpe_postoutcome"] = 0.0
    models = pd.DataFrame(
        [
            {
                "model": "G0_base",
                "population": "non_local",
                "cohort": "primary",
                "animal": a,
                "session_date": d,
                "epoch": "e",
                "n_stops": 12,
                "converged": True,
                "separated": False,
            }
            for a, d in [("a", 1), ("b", 2)]
        ]
    )
    recs = mod._cohort_records(models, stop, n_qvals_dropped=7)
    assert all(_reconciles(r) for r in recs)

    by = {(r["model"], r["population"], r["cohort_mode"]): r for r in recs}
    g0 = by[("G0_base", "all", "common")]
    assert g0["n_drop_qvals_gap"] == 7  # the inner-join drop (mechanism B)
    # G0_base's need includes reward_history_rcos_0 -> immobility(2) + NaN feature(3) = 5 own.
    assert g0["n_drop_own_predictor_missing"] == 5
    assert g0["n_modeled"] == 24  # 2 sessions x 12, the fitter's stop count
    assert g0["n_in"] == len(stop) + 7
    # The same rung refit on the post-outcome cohort is a separate record.
    assert by[("G0_base", "all", "common_post_outcome")]["n_modeled"] == 0


def test_uncertainty_records_reconcile_and_sessions_in_is_pre_exposure_drop():
    """The uncertainty audit must report sessions_in for the post-join (pre-exposure-drop)
    population so it matches n_in, and attribute the qvals-join drop to qvals_gap."""
    from scripts.behavior.rate import fit_uncertainty_rate as mod

    # Fitted frame = 2 sessions (one first-exposure session already dropped upstream).
    rng = np.random.default_rng(0)
    stop_fitted = pd.concat(
        [
            pd.DataFrame(
                {
                    "animal": a,
                    "session_date": d,
                    "epoch": "e",
                    "immobility_duration_s": rng.uniform(5, 30, 15),
                    **{
                        f"reward_history_rcos_{k}": rng.uniform(0, 1, 15)
                        for k in range(3)
                    },
                    "reward_history_incomplete": 0.0,
                }
            )
            for a, d in [("a", 2), ("b", 3)]
        ],
        ignore_index=True,
    )
    stop_fitted.loc[0:1, "immobility_duration_s"] = 0.0  # 2 own-drops

    models = pd.DataFrame(
        [
            {
                "model": mod.MU_UNCERTAINTY.name,
                "outcome": o,
                "animal": a,
                "session_date": d,
                "epoch": "e",
                "n_stops": 10,
                "converged": True,
                "separated": False,
            }
            for o in mod.POPULATIONS_TO_FIT
            for a, d in [("a", 2), ("b", 3)]
        ]
    )
    recs = mod._cohort_records(
        models,
        stop_fitted,
        n_postjoin=100,
        n_sessions_postjoin=3,
        qvals_gap=8,
    )
    assert recs and all(_reconciles(r) for r in recs)
    r0 = recs[0]
    # sessions_in is the pre-exposure-drop count (3), NOT the fitted frame's 2.
    assert r0["n_sessions_in"] == 3
    assert r0["n_drop_qvals_gap"] == 8
    assert r0["n_drop_own_predictor_missing"] == 2  # immobility<=0
    assert r0["n_in"] == 108  # n_postjoin 100 + qvals_gap 8


def test_per_stop_hierarchy_records_reconcile():
    from scripts.behavior.rate import fit_per_stop_rate_models as mod

    rng = np.random.default_rng(1)
    frames = []
    for si, (a, d) in enumerate([("a", 1), ("b", 2)]):
        n = 60
        frames.append(
            pd.DataFrame(
                {
                    "animal": a,
                    "session_date": d,
                    "epoch": "e",
                    "stop_id": np.arange(si * n, (si + 1) * n),
                    "n_selected_events": rng.poisson(2, n).astype(float),
                    "immobility_duration_s": rng.uniform(5, 30, n),
                    **{
                        f"reward_history_rcos_{k}": rng.uniform(0, 1, n)
                        for k in range(3)
                    },
                    "reward_history_incomplete": 0.0,
                    "is_after_omission": rng.integers(0, 2, n).astype(float),
                    "did_last_completed_trip_switch_patch": rng.integers(
                        0, 2, n
                    ).astype(float),
                    "recent_trial_reward_rate": rng.uniform(0, 1, n),
                }
            )
        )
    stop = pd.concat(frames, ignore_index=True)
    stop.loc[0:1, "immobility_duration_s"] = 0.0
    _, recs = mod.fit_per_stop_rate_hierarchy(stop, run_cv=False)
    assert recs and all(_reconciles(r) for r in recs)
    # immobility<=0 stops are attributed as own drops.
    assert any(r["n_drop_own_predictor_missing"] >= 2 for r in recs)


def test_contrast_ladder_rungs_share_one_cohort():
    """Each is_swr-contrast cohort is frozen once per session.

    A post-outcome covariate is NaN on a few stops. Per rung NaN-dropping would
    fit G0_base (which does not use it) on more bins than G5_rpe, so their AIC
    would not be comparable. The nested ladder keeps every stop -- the
    post-outcome columns are not in its union -- and G5_rpe is fit on a second
    cohort alongside a G0_base refit, which shares its rows. The audit books
    the missing stops as ``n_drop_own_predictor_missing`` for G5_rpe and as
    ``n_drop_common_cohort_other`` for the null refit beside it.
    """
    from threadpoolctl import threadpool_limits

    from scripts.behavior.rate import fit_swr_contrast_rate as mod

    rng = np.random.default_rng(0)
    stop_rows, onset_rows = [], []
    for sid in range(40):
        immobility = float(rng.uniform(1.0, 1.6))
        row = {
            "animal": "a",
            "session_date": 1,
            "epoch": "02_r1",
            "stop_id": sid,
            "immobility_duration_s": immobility,
            "reward_history_rcos_0": float(sid % 2),
            "reward_history_rcos_1": float(rng.uniform(0, 1)),
            "reward_history_rcos_2": float(rng.uniform(0, 1)),
            "reward_history_incomplete": 0.0,
            **{c: float(rng.normal()) for c in mod.STOP_COVARS},
        }
        if sid % 10 == 0:  # 4 stops with no fitted value state
            row["rpe_postoutcome"] = np.nan
        stop_rows.append(row)
        for _ in range(rng.poisson(12 * immobility)):
            onset_rows.append(
                {
                    "animal": "a",
                    "session_date": 1,
                    "epoch": "02_r1",
                    "stop_id": sid,
                    "event_type": "non_local",
                    "t_since_stop_onset": float(rng.uniform(0, immobility)),
                    "is_swr": bool(rng.integers(0, 2)),
                }
            )
    stop = pd.DataFrame(stop_rows)

    # The synthetic stops carry neither the arrival clocks nor the outcome
    # lags, so the runner's designs opt out of both by name.
    design = functools.partial(
        mod.aggregate_to_timebins, arrival_clocks=False, outcome_shift=False
    )
    with threadpool_limits(limits=1), pytest.MonkeyPatch.context() as patch:
        patch.setattr(mod, "aggregate_to_timebins", design)
        _, rows, skips, drops = mod._fit_one_session(
            ("a", 1, "02_r1"), stop, with_event_ends(pd.DataFrame(onset_rows))
        )
    fits = pd.DataFrame(rows)

    assert not skips  # every rung fit
    assert drops["non_local"] == 0  # the primary cohort keeps all 40 stops
    assert drops[("non_local", "post_outcome")] == 4  # the four NaN stops
    assert drops["contrast"] == 0  # the strata were frozen before stacking
    assert drops[("contrast", "post_outcome")] == 4
    for (population, cohort), cell in fits.groupby(["population", "cohort"]):
        assert cell["n_bins"].nunique() == 1, (population, cohort)
        assert cell["n_stops"].nunique() == 1, (population, cohort)
        assert cell["n_stops"].iloc[0] == (40 if cohort == "primary" else 36)
    # The stacked contrast design keeps its within-bin pairing: two rows per bin.
    for cohort in ("primary", "post_outcome"):
        cell = fits[fits.cohort == cohort]
        assert (
            cell.loc[cell.population == "contrast", "n_bins"].iloc[0]
            == 2 * cell.loc[cell.population == "non_local_swr", "n_bins"].iloc[0]
        )

    records = mod._cohort_records(fits, stop, 7, drops)
    assert all(_reconciles(r) for r in records)
    by = {(r["model"], r["population"], r["cohort_mode"]): r for r in records}
    g0 = by[("G0_base", "all", "common")]
    g0_post = by[("G0_base", "all", "common_post_outcome")]
    g5 = by[("G5_rpe", "all", "common_post_outcome")]
    assert g0["n_drop_own_predictor_missing"] == 0
    assert g0["n_drop_common_cohort_other"] == 0
    assert g0["n_modeled"] == 40
    assert g5["n_drop_own_predictor_missing"] == 4
    assert g5["n_drop_common_cohort_other"] == 0
    assert g0_post["n_drop_own_predictor_missing"] == 0
    assert g0_post["n_drop_common_cohort_other"] == 4
    assert g0_post["n_modeled"] == g5["n_modeled"] == 36
