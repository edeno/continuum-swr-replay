"""Tests for the per-bin rate-model orchestrator's tidy output.

Runs the orchestrator's in-memory entry point on a tiny synthetic two-session
table (no disk I/O) and checks the tidy schema: one row per
(outcome, model, animal, day, epoch), with the coefficient / CV / GOF columns,
and ``outcome`` / ``model`` values drawn from the canonical key sets.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from continuum_swr_replay.analysis.replay import (
    MODEL_PER_BIN_RATE,
    POPULATIONS,
)

_SPEC = importlib.util.spec_from_file_location(
    "fit_per_bin_rate_models",
    Path(__file__).resolve().parents[2] / "scripts" / "fit_per_bin_rate_models.py",
)
fprm = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fprm)


def _toy_tables(n_stops_per_session: int = 24, seed: int = 0):
    """Two sessions of stop + onset rows with a planted per-stop reward effect.

    Deliberately small (short stops -> few 15 ms bins) so the full ladder + CV
    runs in seconds while still exercising every schema column.
    """
    rng = np.random.default_rng(seed)
    stop_rows, onset_rows = [], []
    sid = 0
    for animal, day, epoch in [("a", 1, "02_r1"), ("b", 1, "02_r1")]:
        for _ in range(n_stops_per_session):
            immob = float(rng.uniform(0.3, 0.7))
            reward = float(sid % 2)
            stop_rows.append({
                "animal": animal, "day": day, "epoch": epoch, "stop_id": sid,
                "immobility_time": immob,
                **{f"reward_lag{j}": float(rng.integers(0, 2)) for j in range(1, 11)},
                "reward_lag1": reward,
                "recent_trial_reward_rate": float(rng.uniform(0, 1)),
                "is_after_switch": float(rng.integers(0, 2)),
            })
            # onset times within the stop; rewarded stops get more events.
            rate = 12.0 if reward else 6.0
            n_ev = rng.poisson(rate * min(immob, 4.0))
            for et in ["SWR", "non_local"]:
                for _ in range(n_ev):
                    t = float(rng.uniform(0, min(immob, 4.0)))
                    is_swr = bool(rng.integers(0, 2)) if et == "non_local" else np.nan
                    onset_rows.append({
                        "animal": animal, "day": day, "epoch": epoch,
                        "stop_id": sid, "event_type": et,
                        "t_since_stop_onset": t, "onset_time": t, "is_swr": is_swr,
                    })
            sid += 1
    return pd.DataFrame(stop_rows), pd.DataFrame(onset_rows)


def test_tidy_schema():
    stop, onsets = _toy_tables()
    # onsets already carry is_swr (as attach_is_swr_to_onsets would set it).
    # Pin BLAS: tiny-matrix GLM fits are faster single-threaded.
    with threadpool_limits(limits=1):
        out = fprm.fit_per_bin_rate_models(stop, onsets, run_cv=True, run_gof=True)

    assert len(out) > 0
    expected_cols = {
        "outcome", "model", "animal", "day", "epoch",
        "rr_reward_lag1", "n_bins", "n_events", "aic", "deviance", "dispersion",
        "ks_statistic", "ks_within_band",
        "cv_ll_per_bin_mean", "cv_bits_per_event", "cv_n_folds",
    }
    assert expected_cols <= set(out.columns)

    # All four populations must produce rows (not just a subset) so a silently
    # dropped outcome can't make the schema check vacuous.
    assert set(out["outcome"]) == set(POPULATIONS)
    assert set(out["model"]) <= set(MODEL_PER_BIN_RATE)
    # The core (always-identifiable) models must be present for every outcome.
    for outcome in POPULATIONS:
        sub_models = set(out[out.outcome == outcome]["model"])
        assert {"M0_const", "M1_rew1", "M5_time"} <= sub_models

    # 'predicted' (the per-bin Series) must not leak into the tidy table.
    assert "predicted" not in out.columns

    # One row per (outcome, model, session) — no duplicates.
    keys = ["outcome", "model", "animal", "day", "epoch"]
    assert not out.duplicated(keys).any()


def test_baseline_present_across_outcomes():
    stop, onsets = _toy_tables()
    with threadpool_limits(limits=1):
        out = fprm.fit_per_bin_rate_models(stop, onsets, run_cv=False, run_gof=False)
    # M0_const must fit for every outcome that produced any rows.
    for outcome in out["outcome"].unique():
        sub = out[out.outcome == outcome]
        assert "M0_const" in set(sub["model"])
