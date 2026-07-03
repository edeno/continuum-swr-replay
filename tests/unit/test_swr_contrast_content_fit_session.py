"""Tests for the SWR-contrast content runner wrapper."""

from __future__ import annotations

import pandas as pd

from scripts.content import fit_swr_contrast_content as mod


def test_fit_session_drops_nan_events_not_rows(monkeypatch):
    """Missing arm covariates must drop an event, not shrink its arm set."""
    df = pd.DataFrame(
        {
            "event_id": [1, 1, 1, 2, 2, 2],
            "arm_count_hard": [1, 0, 0, 0, 1, 0],
            "reward_rate_arm_k10": [float("nan"), 0.25, 0.50, 0.0, 0.5, 1.0],
        }
    )
    captured = {}

    def fake_fit(long_df, **kwargs):
        captured["long_df"] = long_df.copy()
        captured["kwargs"] = kwargs
        return {"n_events": long_df["event_id"].nunique()}

    monkeypatch.setattr(mod, "MIN_EVENTS", 1)
    monkeypatch.setattr(mod, "fit_conditional_poisson", fake_fit)

    out = mod._fit_session(
        df,
        ["reward_rate_arm_k10"],
        response_col="arm_count_hard",
        n_categories=3,
    )

    assert out == {"n_events": 1}
    assert captured["kwargs"]["n_categories"] == 3
    assert captured["long_df"]["event_id"].unique().tolist() == [2]
    assert len(captured["long_df"]) == 3
