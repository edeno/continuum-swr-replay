"""Regression test: CV must drop zero-count events before fold building.

``fit_conditional_poisson`` already filters zero-count events through
``_drop_zero_count_events``, but the public CV helper used to count
them in ``n_test_events`` (the per-event LL denominator). This made
the per-event LL drift while the per-timebin LL stayed identical:
with one zero-count event, ``cv_ll_per_event_mean`` shifted from
-2.7465 to -3.2958 after explicit pre-filtering, exposing the
inconsistency.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.replay import model_checking
from continuum_swr_replay.analysis.replay.model_checking import (
    cross_validate_conditional_poisson,
)


def _toy_long_df(n_events: int, n_arms: int = 9) -> pd.DataFrame:
    rng = __import__("numpy").random.default_rng(0)
    rows = []
    for ei in range(n_events):
        # Each event has a random non-zero count distribution across arms.
        counts = rng.integers(low=0, high=3, size=n_arms).astype(float)
        if counts.sum() == 0:
            counts[0] = 1.0
        for arm in range(n_arms):
            rows.append({
                "event_id": ei,
                "arm": arm,
                "arm_count_hard": counts[arm],
            })
    return pd.DataFrame(rows)


class TestCvZeroCountFilter:
    def test_zero_count_event_dropped_in_null_branch_per_event_mean(self):
        """Use the null branch (no predictors) where the held-out LL has
        a closed form ``-Σ_e L_e · log(n_arms)``. A zero-count event
        contributes 0 to the numerator but used to be counted in the
        denominator (n_test_events), which dragged the per-event mean
        toward zero. After the fix, the metric matches the clean
        baseline exactly."""
        df_a = _toy_long_df(n_events=12)
        # Add a zero-count event.
        df_b = df_a.copy()
        zero_event_id = df_a["event_id"].max() + 1
        zero_rows = pd.DataFrame(
            [
                {"event_id": zero_event_id, "arm": arm, "arm_count_hard": 0.0}
                for arm in range(9)
            ]
        )
        df_b = pd.concat([df_b, zero_rows], ignore_index=True)

        out_a = cross_validate_conditional_poisson(
            df_a,
            predictors=[],  # null branch
            event_col="event_id",
            response_col="arm_count_hard",
            n_folds=3,
        )
        out_b = cross_validate_conditional_poisson(
            df_b,
            predictors=[],
            event_col="event_id",
            response_col="arm_count_hard",
            n_folds=3,
        )

        assert out_a is not None and out_b is not None
        assert out_a["cv_ll_per_event_mean"] == out_b["cv_ll_per_event_mean"]
        assert out_a["cv_ll_per_timebin_mean"] == out_b["cv_ll_per_timebin_mean"]


def _toy_long_df_with_predictor(n_events: int, n_arms: int = 9) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    rows = []
    for ei in range(n_events):
        x = rng.normal(size=n_arms)
        chosen = int(np.argmax(x + rng.normal(size=n_arms)))
        for arm in range(n_arms):
            rows.append({
                "event_id": ei,
                "arm": arm,
                "arm_count_hard": float(arm == chosen),
                "pred": float(x[arm]),
            })
    return pd.DataFrame(rows)


class TestCvConditionalPoissonInvalidation:
    def test_bad_fold_invalidates_metric_without_dropping(self, monkeypatch):
        """A nonconverged/separated fold must invalidate the content CV metric to
        nan (matching per_bin/per_stop) rather than being silently dropped and the
        mean reported over surviving folds."""
        calls = {"n": 0}
        real = model_checking.fit_status

        def fake_fit_status(result, kind, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return {"converged": True, "separated": True}
            return real(result, kind, **kwargs)

        monkeypatch.setattr(model_checking, "fit_status", fake_fit_status)
        df = _toy_long_df_with_predictor(n_events=90)
        out = cross_validate_conditional_poisson(
            df, predictors=["pred"], event_col="event_id",
            response_col="arm_count_hard", n_folds=5,
        )
        assert out is not None  # fold NOT dropped
        assert out["cv_n_separated_folds"] == 1
        assert out["cv_n_nonconverged_folds"] == 0
        assert np.isnan(out["cv_ll_per_event_mean"])
        assert np.isnan(out["cv_ll_per_timebin_mean"])
