"""F4: a content session missing a required predictor is a clean, auditable skip.

Previously ``_fit_session`` forwarded a design with absent columns straight to
``fit_conditional_poisson`` (which logs an error and returns None) — a hidden path
that diverged from the rate runner's ``missing:<cols>`` skip. It must now short-circuit
to an ``n_excluded_session`` skip WITHOUT calling the fitter.
"""

from __future__ import annotations

import pandas as pd

from continuum_swr_replay.analysis.behavior import contrast_aggregation
from scripts.behavior.content import fit_swr_contrast_content as mod


def test_missing_predictor_is_clean_skip(monkeypatch):
    df = pd.DataFrame(
        {
            "event_id": [1, 1, 1, 2, 2, 2],
            "arm_count_hard": [1, 0, 0, 0, 1, 0],
            # NOTE: the required predictor column is intentionally absent.
        }
    )
    called = {"fit": False}

    def fake_fit(*args, **kwargs):
        called["fit"] = True
        return {"n_events": 0}

    monkeypatch.setattr(mod, "fit_conditional_poisson", fake_fit)

    out, audit = mod._fit_session(
        df,
        ["reward_belief_mean_candidate_arm_preoutcome"],
        response_col="arm_count_hard",
        n_categories=9,
    )

    assert out is None
    assert called["fit"] is False  # fitter NOT called with absent columns
    assert audit["reason"].startswith("missing:")
    assert "reward_belief_mean_candidate_arm_preoutcome" in audit["reason"]
    # The whole session's events land in n_excluded_session and the record reconciles.
    assert audit["n_in"] == 2
    assert audit["n_excluded_session"] == 2
    assert audit["n_modeled"] == 0


def test_qvals_gap_drop_is_bucketed_separately(monkeypatch):
    """An event whose q_arm (a qvals-tail covariate) is NaN drops into
    n_drop_qvals_gap, distinct from mechanism-A own-predictor missingness."""
    df = pd.DataFrame(
        {
            "event_id": [1, 1, 1, 2, 2, 2],
            "has_qvals_row": [False, False, False, True, True, True],
            "arm_count_hard": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            "reward_belief_mean_candidate_arm_preoutcome": [
                float("nan"),
                0.2,
                0.3,
                0.1,
                0.4,
                0.5,
            ],
        }
    )
    monkeypatch.setattr(contrast_aggregation, "MIN_EVENTS", 1)
    monkeypatch.setattr(mod, "fit_conditional_poisson", lambda *a, **k: {"n_events": 1})

    out, audit = mod._fit_session(
        df,
        ["reward_belief_mean_candidate_arm_preoutcome"],
        response_col="arm_count_hard",
        n_categories=9,
    )
    assert out is not None
    assert audit["n_drop_qvals_gap"] == 1  # event 1
    assert audit["n_drop_own_predictor_missing"] == 0
    assert audit["n_modeled"] == 1
