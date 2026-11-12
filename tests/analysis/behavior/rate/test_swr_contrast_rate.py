"""The is_swr-contrast rate ladders' two cohorts per session and their audit.

The gated stops cost only the rungs that need a post-outcome covariate: those
are fit on a second cohort together with their ladder's null, while the nested
ladder keeps every stop.
"""

from __future__ import annotations

import functools

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.behavior import (
    BIN_S,
    N_HISTORY_BASIS,
    N_HISTORY_LAGS,
    POPULATIONS,
    TMAX,
    aggregate_to_timebins,
)
from continuum_swr_replay.analysis.behavior.rate import swr_contrast_rate as mod
from continuum_swr_replay.analysis.behavior.rate.duration_aware_history import (
    at_risk_events,
)
from continuum_swr_replay.analysis.behavior.rate.post_outcome import (
    mask_post_outcome_covariates,
)
from tests.support._event_end_fixtures import with_event_ends

# ── Two cohorts per ladder ──


def _gated_session(session_date: int, n_stops: int = 24, seed: int = 0):
    """One synthetic session where every third stop is gated as pre-outcome.

    Returns ``(stop_frame, onset_frame, gated_stop_ids)``. The gate is applied
    through :func:`mask_post_outcome_covariates`, exactly as the runner does.
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
    return stops, with_event_ends(pd.DataFrame(onset_rows)), gated


_without_clocks_or_shift = functools.partial(
    aggregate_to_timebins, arrival_clocks=False, outcome_shift=False
)


def fit_two_sessions():
    """Both ladders fit on two synthetic sessions with gated post-outcome stops.

    Returns ``(fits, stops, cohort_drops, gated_stop_ids)``.
    """
    from threadpoolctl import threadpool_limits

    frames, rows, drops, gated_ids = [], [], {}, []
    for session_date in (1, 2):
        stops, onsets, gated = _gated_session(session_date, seed=session_date)
        stops, _ = mask_post_outcome_covariates(stops)
        frames.append(stops)
        gated_ids.extend(gated)
        # The synthetic stops carry neither the arrival clocks nor the outcome
        # lags, so the runner's designs opt out of both by name.
        with threadpool_limits(limits=1), pytest.MonkeyPatch.context() as patch:
            patch.setattr(mod, "aggregate_to_timebins", _without_clocks_or_shift)
            _, session_rows, skips, session_drops = mod.fit_contrast_session(
                ("a", session_date, "02_r1"), stops, onsets
            )
        assert not skips, skips
        rows.extend(session_rows)
        for key, value in session_drops.items():
            drops[key] = drops.get(key, 0) + value
    return pd.DataFrame(rows), pd.concat(frames, ignore_index=True), drops, gated_ids


@pytest.fixture(scope="module")
def two_session_fits():
    return fit_two_sessions()


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
    onsets = _gated_session(1, seed=1)[1].assign(is_swr=False)
    design = aggregate_to_timebins(
        stops[stops.session_date == 1],
        POPULATIONS["non_local"](onsets),
        bin_s=BIN_S,
        tmax=TMAX,
        reward_basis=True,
        self_history_lags=N_HISTORY_LAGS,
        self_history_basis=N_HISTORY_BASIS,
        stop_covariates=mod.STOP_COVARS,
        arrival_clocks=False,
        outcome_shift=False,
        at_risk_events=at_risk_events(onsets, "non_local"),
        offset_history=True,
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
    records = mod.contrast_cohort_records(fits, stops, 0, drops)
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
