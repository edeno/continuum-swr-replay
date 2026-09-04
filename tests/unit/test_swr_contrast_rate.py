"""Cohort gating and separation flagging in the is_swr-contrast rate runner.

Two guards:

* ``_mask_post_outcome_covariates`` -- a stop that starts *before* the trial's
  outcome is known cannot carry a post-outcome covariate, so those values are
  NaN'd (and later removed by the ladder's cohort freeze) rather than being fit
  as if the animal already knew the outcome.
* ``_fit`` -- the separation flag is judged on the effect scale, so a
  small-SD design column with an ordinary effect is not mistaken for a
  quasi-separated fit just because its per-unit coefficient is large.
* ``_fit_one_session`` -- the gated stops cost only the rungs that need a
  post-outcome covariate: those are fit on a second cohort together with their
  ladder's null, while the nested ladder keeps every stop.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.replay import (
    BIN_S,
    N_HISTORY_BASIS,
    N_HISTORY_LAGS,
    POPULATIONS,
    TMAX,
    aggregate_to_timebins,
)
from scripts.rate import fit_swr_contrast_rate as mod


@pytest.fixture
def stop_frame() -> pd.DataFrame:
    """Three stops: one starts before its outcome is known, two do not."""
    return pd.DataFrame(
        {
            "stop_id": [1, 2, 3],
            "stop_start": [10.0, 20.0, 30.0],
            # stop 2 starts 1 s before the outcome is known; stop 3 starts
            # exactly at it (not before -> keeps its post-outcome values).
            "outcome_known_time": [9.5, 21.0, 30.0],
            "occupancy_state": ["at_well", "at_well", "in_transit"],
            "rpe_postoutcome": [0.4, -0.6, 0.2],
            "reward_belief_mean_update_l1_all_ports_postoutcome": [0.1, 0.2, 0.3],
            "relative_value_switch_minus_stay_preoutcome": [1.0, 2.0, 3.0],
        }
    )


def test_post_outcome_gate_nans_only_pre_outcome_stops(stop_frame):
    gated, counts = mod._mask_post_outcome_covariates(stop_frame)

    assert gated["rpe_postoutcome"].tolist()[0] == 0.4
    assert np.isnan(gated["rpe_postoutcome"].iloc[1])
    assert gated["rpe_postoutcome"].iloc[2] == 0.2
    update = gated["reward_belief_mean_update_l1_all_ports_postoutcome"]
    assert update.iloc[0] == 0.1
    assert np.isnan(update.iloc[1])
    assert update.iloc[2] == 0.3
    # Pre-outcome covariates are untouched on the same rows.
    assert gated["relative_value_switch_minus_stay_preoutcome"].tolist() == [
        1.0,
        2.0,
        3.0,
    ]
    assert counts["at_well"] == 1
    assert counts.get("in_transit", 0) == 0


def test_post_outcome_gate_does_not_mutate_the_input(stop_frame):
    mod._mask_post_outcome_covariates(stop_frame)
    assert stop_frame["rpe_postoutcome"].tolist() == [0.4, -0.6, 0.2]


def test_post_outcome_gate_requires_the_outcome_clock(stop_frame):
    with pytest.raises(ValueError, match="outcome_known_time"):
        mod._mask_post_outcome_covariates(stop_frame.drop(columns="outcome_known_time"))


def test_small_sd_predictor_with_ordinary_effect_is_not_flagged_separated():
    """|coef| = 12 on a column with SD 0.03 is a 0.36-log-unit effect, not separation.

    The legacy ``|coef| >= 10`` rule flagged exactly this design; the effect-scale
    rule (``bse * SD``) does not.
    """
    rng = np.random.default_rng(0)
    n = 4000
    x = rng.normal(0.0, 0.03, n)
    exposure = np.full(n, 0.015)
    rate = np.exp(np.log(2.0) + 12.0 * x)
    design = pd.DataFrame(
        {
            "stop_id": np.repeat(np.arange(n // 10), 10),
            "exposure": exposure,
            "x": x,
            "count": rng.poisson(rate * exposure),
        }
    ).astype({"count": float})

    row, reason = mod._fit(design, "x", ["x"])

    assert reason is None
    assert row["coef_x"] == pytest.approx(12.0, abs=3.0)
    # The design column's SD is small, so the per-unit coefficient is large ...
    assert abs(row["coef_x"]) >= 10.0
    # ... but the effect-scale standard error is far below the separation bound.
    assert row["bse_x"] * x.std() < 1.0
    assert row["separated"] is False
    assert row["converged"] is True


# ── Two cohorts per ladder ──


def _gated_session(session_date: int, n_stops: int = 24, seed: int = 0):
    """One synthetic session where every third stop is gated as pre-outcome.

    Returns ``(stop_frame, onset_frame, gated_stop_ids)``. The gate is applied
    through :func:`_mask_post_outcome_covariates`, exactly as the runner does.
    """
    rng = np.random.default_rng(seed)
    stop_rows, onset_rows = [], []
    for i in range(n_stops):
        stop_id = session_date * 1000 + i
        immobility = 1.2
        pre_outcome = i % 3 == 0  # a third of the stops start before the outcome
        stop_rows.append(
            {
                "animal": "a",
                "session_date": session_date,
                "epoch": "02_r1",
                "stop_id": stop_id,
                "immobility_duration_s": immobility,
                "stop_start": 100.0 * i,
                "outcome_known_time": 100.0 * i + (1.0 if pre_outcome else -1.0),
                "occupancy_state": "at_well",
                "reward_history_rcos_0": float(i % 2),
                "reward_history_rcos_1": float(rng.uniform(0, 1)),
                "reward_history_rcos_2": float(rng.uniform(0, 1)),
                "reward_history_incomplete": 0.0,
                **{c: float(rng.normal()) for c in mod.STOP_COVARS},
            }
        )
        for _ in range(rng.poisson(14 * immobility)):
            onset_rows.append(
                {
                    "animal": "a",
                    "session_date": session_date,
                    "epoch": "02_r1",
                    "stop_id": stop_id,
                    "event_type": "non_local",
                    "t_since_stop_onset": float(rng.uniform(0, immobility)),
                    "is_swr": bool(rng.integers(0, 2)),
                }
            )
    stops = pd.DataFrame(stop_rows)
    gated = [r["stop_id"] for i, r in enumerate(stop_rows) if i % 3 == 0]
    return stops, pd.DataFrame(onset_rows), gated


@pytest.fixture(scope="module")
def two_session_fits():
    """Both ladders fit on two synthetic sessions with gated post-outcome stops."""
    from threadpoolctl import threadpool_limits

    frames, rows, drops, gated_ids = [], [], {}, []
    for session_date in (1, 2):
        stops, onsets, gated = _gated_session(session_date, seed=session_date)
        stops, _ = mod._mask_post_outcome_covariates(stops)
        frames.append(stops)
        gated_ids.extend(gated)
        with threadpool_limits(limits=1):
            _, session_rows, skips, session_drops = mod._fit_one_session(
                ("a", session_date, "02_r1"), stops, onsets
            )
        assert not skips, skips
        rows.extend(session_rows)
        for key, value in session_drops.items():
            drops[key] = drops.get(key, 0) + value
    return pd.DataFrame(rows), pd.concat(frames, ignore_index=True), drops, gated_ids


def test_primary_rungs_share_one_cohort_unaffected_by_the_gate(two_session_fits):
    """The gate costs the nested ladder nothing: every primary rung, all stops."""
    fits, stops, _drops, _gated = two_session_fits
    primary = fits[fits.cohort == "primary"]
    assert set(primary.model) == {
        m
        for specs in (mod.G_SPECS, mod.S_SPECS)
        for m in specs
        if m not in mod.POST_OUTCOME_RATE_RUNGS
    }
    for (session_date, population), cell in primary.groupby(
        ["session_date", "population"]
    ):
        assert cell["n_bins"].nunique() == 1, (session_date, population)
        assert cell["n_stops"].nunique() == 1, (session_date, population)
        assert cell["n_stops"].iloc[0] == 24  # every stop of the session


def test_post_outcome_rung_and_its_null_share_the_gated_cohort(two_session_fits):
    """G5_rpe and the G0_base refit beside it lose exactly the gated stops' bins."""
    fits, stops, _drops, gated = two_session_fits
    design = aggregate_to_timebins(
        stops[stops.session_date == 1],
        POPULATIONS["non_local"](_gated_session(1, seed=1)[1].assign(is_swr=False)),
        bin_s=BIN_S,
        tmax=TMAX,
        reward_basis=True,
        self_history_lags=N_HISTORY_LAGS,
        self_history_basis=N_HISTORY_BASIS,
        stop_covariates=mod.STOP_COVARS,
    )
    gated_bins = int(design["stop_id"].isin(gated).sum())
    assert gated_bins > 0

    cell = fits[(fits.session_date == 1) & (fits.population == "non_local")]
    primary_bins = cell.loc[cell.cohort == "primary", "n_bins"].iloc[0]
    post = cell[cell.cohort == "post_outcome"]
    assert set(post.model) == {"G5_rpe", "G7_update", "G0_base"}
    assert post["n_bins"].nunique() == 1
    assert post["n_stops"].iloc[0] == 16  # 24 stops - 8 gated
    assert int(primary_bins) - int(post["n_bins"].iloc[0]) == gated_bins


def test_contrast_ladder_keeps_its_pairing_in_both_cohorts(two_session_fits):
    """The stacked design carries two rows per bin in the post-outcome cohort too."""
    fits, _stops, _drops, _gated = two_session_fits
    cell = fits[fits.session_date == 1]
    contrast = cell[cell.population == "contrast"]
    stratum = cell[cell.population == "non_local_swr"]
    for cohort in ("primary", "post_outcome"):
        assert (
            contrast.loc[contrast.cohort == cohort, "n_bins"].iloc[0]
            == 2 * stratum.loc[stratum.cohort == cohort, "n_bins"].iloc[0]
        )
    assert set(contrast.loc[contrast.cohort == "post_outcome", "model"]) == {
        "S7_rpe",
        "S9_update",
        "S0_rate",
    }


def test_fitted_rows_are_unique_per_session_population_model_cohort(two_session_fits):
    fits, _stops, _drops, _gated = two_session_fits
    keys = ["animal", "session_date", "epoch", "population", "model", "cohort"]
    assert not fits.duplicated(subset=keys).any()
    # The ladder null appears once per cohort, not twice in one.
    nulls = fits[(fits.model == "G0_base") & (fits.population == "non_local")]
    assert sorted(nulls.cohort.unique()) == ["post_outcome", "primary"]
    assert len(nulls) == 4  # 2 sessions x 2 cohorts


def test_cohort_audit_reconciles_across_both_cohorts(two_session_fits):
    fits, stops, drops, gated = two_session_fits
    records = mod._cohort_records(fits, stops, 0, drops)
    for record in records:
        recon = record["n_in"] - sum(
            record[f]
            for f in (
                "n_drop_qvals_gap",
                "n_drop_own_predictor_missing",
                "n_drop_nonfinite",
                "n_drop_zero_response",
                "n_drop_common_cohort_other",
                "n_excluded_session",
            )
        )
        assert recon == record["n_modeled"], record

    by = {(r["model"], r["population"], r["cohort_mode"]): r for r in records}
    n_gated = len(gated)
    assert n_gated == 16  # 8 per session
    # The rung that needs the gated covariate books them as its own missing.
    g5 = by[("G5_rpe", "all", "common_post_outcome")]
    assert g5["n_drop_own_predictor_missing"] == n_gated
    assert g5["n_drop_common_cohort_other"] == 0
    assert g5["n_modeled"] == 32  # 2 sessions x 16 stops
    # The null refit beside it loses the same stops, but to another rung's column.
    null_post = by[("G0_base", "all", "common_post_outcome")]
    assert null_post["n_drop_own_predictor_missing"] == 0
    assert null_post["n_drop_common_cohort_other"] == n_gated
    assert null_post["n_modeled"] == 32
    # On the primary cohort the ladder null keeps every stop.
    null_primary = by[("G0_base", "all", "common")]
    assert null_primary["n_drop_own_predictor_missing"] == 0
    assert null_primary["n_drop_common_cohort_other"] == 0
    assert null_primary["n_modeled"] == 48  # 2 sessions x 24 stops
