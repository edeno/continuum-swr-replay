"""Tests for analysis/replay/choice_prediction.py.

Stage 5A asks whether replay content predicts the next choice. The
data pipeline (``build_choice_dataset`` →
``build_residualized_choice_dataset``) and the conditional-logistic
fit (``fit_choice_model``) all live in this module. Pin the
invariants:

- ``build_choice_dataset``: 8-arm choice sets, exactly one chosen
  arm per stop, current_arm always excluded, NaN-current_arm filter
- ``predict_arm_fractions``: predictions sum to 1 per event, residual
  has the right sign, numerically-stable softmax handles large logits
- ``build_residualized_choice_dataset``: residual column merged on
  (stop_id, arm), missing residuals filled with 0, stop_id ordering
  preserved
- ``fit_choice_model``: rejects under-determined input, recovers a
  known coefficient
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.replay.choice_prediction import (
    build_choice_dataset,
    build_residualized_choice_dataset,
    fit_choice_model,
    predict_arm_fractions,
)

# ── Shared fixture builder ──


def _make_minimal_tables(
    stops: list[dict],
    *,
    arm_fractions: dict[int, list[float]] | None = None,
):
    """Construct minimal event_table / event_arm_table / stop_table
    for build_choice_dataset.

    Parameters
    ----------
    stops : list[dict]
        Each dict needs at least ``stop_id``, ``current_arm``,
        ``next_arm``.
    arm_fractions : dict[stop_id, list[float]], optional
        Per-stop arm_fraction vector (length 9). Defaults to uniform
        1/9 on every arm.
    """
    stop_table = pd.DataFrame(
        [
            {
                "animal": "test",
                "day": 1,
                "epoch": "02_r1",
                "has_replay": True,
                "n_events": 1,
                "reward_outcome": 0,
                "is_after_reward": False,
                "is_after_omission": False,
                **s,
            }
            for s in stops
        ]
    )
    event_table = pd.DataFrame(
        {
            "event_id": range(1, len(stops) + 1),
            "stop_id": [s["stop_id"] for s in stops],
        }
    )
    arms = list(range(9))
    event_arm_rows = []
    for s in stops:
        fracs = (
            arm_fractions[s["stop_id"]]
            if arm_fractions and s["stop_id"] in arm_fractions
            else [1.0 / 9] * 9
        )
        for arm in arms:
            event_arm_rows.append(
                {
                    "event_id": s["stop_id"],
                    "stop_id": s["stop_id"],
                    "arm": arm,
                    "arm_fraction": fracs[arm],
                    "is_prev": 0.0,
                    "is_next": 0.0,
                    "visit_recency": 0.0,
                    "cumulative_reward": 0.0,
                    "recent_reward_rate": 0.0,
                    "dist_to_arm": 1.0,
                    "is_same_patch": 0,
                }
            )
    event_arm_table = pd.DataFrame(event_arm_rows)
    return event_table, event_arm_table, stop_table


# ── build_choice_dataset ──


class TestBuildChoiceDataset:
    def test_drops_stops_with_nan_current_arm(self):
        """arm != NaN is True for every arm in IEEE 754, so the
        ``choice_df[choice_df['arm'] != choice_df['current_arm']]``
        filter silently keeps all 9 arms when current_arm is NaN.
        Such stops must be filtered at the stop level."""
        et, eat, st = _make_minimal_tables(
            [
                {"stop_id": 1, "current_arm": 0, "next_arm": 3},
                {"stop_id": 2, "current_arm": np.nan, "next_arm": 2},
                {"stop_id": 3, "current_arm": 5, "next_arm": 2},
            ]
        )
        ds = build_choice_dataset(et, eat, st)
        sizes = ds.groupby("stop_id").size()
        assert (sizes == 8).all()
        assert 2 not in sizes.index
        assert set(sizes.index) == {1, 3}

    def test_exactly_one_chosen_arm_per_stop(self):
        et, eat, st = _make_minimal_tables(
            [
                {"stop_id": 1, "current_arm": 0, "next_arm": 3},
                {"stop_id": 2, "current_arm": 5, "next_arm": 2},
            ]
        )
        ds = build_choice_dataset(et, eat, st)
        chosen_per_stop = ds.groupby("stop_id")["chosen"].sum()
        assert (chosen_per_stop == 1).all()

    def test_chosen_arm_matches_next_arm(self):
        """The single ``chosen=1`` row in each stop must be the arm
        equal to that stop's ``next_arm``."""
        et, eat, st = _make_minimal_tables(
            [{"stop_id": 1, "current_arm": 0, "next_arm": 3}]
        )
        ds = build_choice_dataset(et, eat, st)
        chosen_row = ds.loc[ds["chosen"] == 1].iloc[0]
        assert chosen_row["arm"] == 3

    def test_current_arm_never_in_choice_set(self):
        """current_arm is the arm the animal is sitting on — by
        definition it cannot be the next destination, so it never
        appears as a candidate."""
        et, eat, st = _make_minimal_tables(
            [
                {"stop_id": 1, "current_arm": 0, "next_arm": 3},
                {"stop_id": 2, "current_arm": 5, "next_arm": 2},
            ]
        )
        ds = build_choice_dataset(et, eat, st)
        for stop_id, current_arm in [(1, 0), (2, 5)]:
            stop_arms = set(ds.loc[ds["stop_id"] == stop_id, "arm"])
            assert current_arm not in stop_arms

    def test_drops_stops_with_nan_next_arm(self):
        """Without a known next destination, ``chosen`` is undefined;
        such stops must be excluded."""
        et, eat, st = _make_minimal_tables(
            [
                {"stop_id": 1, "current_arm": 0, "next_arm": 3},
                {"stop_id": 2, "current_arm": 5, "next_arm": np.nan},
            ]
        )
        ds = build_choice_dataset(et, eat, st)
        assert set(ds["stop_id"].unique()) == {1}

    def test_drops_stops_without_replay(self):
        """``has_replay = False`` means no events to compute
        replay_frac from — exclude the stop."""
        et, eat, st = _make_minimal_tables(
            [
                {"stop_id": 1, "current_arm": 0, "next_arm": 3},
                {"stop_id": 2, "current_arm": 5, "next_arm": 2},
            ]
        )
        st.loc[st["stop_id"] == 2, "has_replay"] = False
        ds = build_choice_dataset(et, eat, st)
        assert set(ds["stop_id"].unique()) == {1}

    def test_replay_frac_averaged_across_events_in_stop(self):
        """Two events in the same stop with different replay_frac
        vectors should yield a per-arm average."""
        et, eat, st = _make_minimal_tables(
            [{"stop_id": 1, "current_arm": 0, "next_arm": 3}]
        )
        # Add a second event for stop 1 with all mass on arm 3.
        new_event_rows = []
        for arm in range(9):
            new_event_rows.append({
                "event_id": 2,
                "stop_id": 1,
                "arm": arm,
                "arm_fraction": 1.0 if arm == 3 else 0.0,
                "is_prev": 0.0,
                "is_next": 0.0,
                "visit_recency": 0.0,
                "cumulative_reward": 0.0,
                "recent_reward_rate": 0.0,
                "dist_to_arm": 1.0,
                "is_same_patch": 0,
            })
        eat = pd.concat([eat, pd.DataFrame(new_event_rows)], ignore_index=True)
        et = pd.concat(
            [et, pd.DataFrame({"event_id": [2], "stop_id": [1]})],
            ignore_index=True,
        )

        ds = build_choice_dataset(et, eat, st)
        # arm 3: mean of (1/9, 1.0) = 5/9
        arm3 = ds.loc[ds["arm"] == 3, "replay_frac"].iloc[0]
        assert arm3 == pytest.approx((1.0 / 9 + 1.0) / 2)
        # arm 1: mean of (1/9, 0) = 1/18
        arm1 = ds.loc[ds["arm"] == 1, "replay_frac"].iloc[0]
        assert arm1 == pytest.approx((1.0 / 9) / 2)

# ── predict_arm_fractions ──


class TestPredictArmFractions:
    def _eat_for_softmax(self, n_events=2, predictor_value_fn=None):
        rows = []
        for ev in range(1, n_events + 1):
            for arm in range(9):
                rows.append({
                    "event_id": ev,
                    "arm": arm,
                    "is_prev": predictor_value_fn(ev, arm) if predictor_value_fn else 0.0,
                    "arm_fraction": 1.0 / 9,
                })
        return pd.DataFrame(rows)

    def test_predicted_frac_sums_to_one_per_event(self):
        """Softmax is normalized within each event, so per-event
        ``sum(predicted_frac) == 1.0`` must hold exactly."""
        eat = self._eat_for_softmax(
            n_events=3,
            predictor_value_fn=lambda ev, arm: 1.0 if arm == ev else 0.0,
        )
        fit_result = {"coef_is_prev": 1.5}
        out = predict_arm_fractions(eat, fit_result, predictors=["is_prev"])
        per_event_sum = out.groupby("event_id")["predicted_frac"].sum()
        np.testing.assert_allclose(per_event_sum.to_numpy(), 1.0, atol=1e-12)

    def test_zero_coefficients_yield_uniform_predictions(self):
        """With every coefficient at zero, softmax(0) over 9 arms is
        1/9 on every arm."""
        eat = self._eat_for_softmax(
            n_events=2,
            predictor_value_fn=lambda ev, arm: 1.0 if arm == 0 else 0.0,
        )
        fit_result = {"coef_is_prev": 0.0}
        out = predict_arm_fractions(eat, fit_result, predictors=["is_prev"])
        np.testing.assert_allclose(out["predicted_frac"].to_numpy(), 1.0 / 9)

    def test_residual_is_observed_minus_predicted(self):
        eat = self._eat_for_softmax(n_events=1)
        fit_result = {"coef_is_prev": 0.0}
        out = predict_arm_fractions(eat, fit_result, predictors=["is_prev"])
        np.testing.assert_allclose(
            out["residual_frac"].to_numpy(),
            (out["arm_fraction"] - out["predicted_frac"]).to_numpy(),
            atol=1e-12,
        )

    def test_residual_has_correct_sign(self):
        """When the model overpredicts an arm, residual = observed -
        predicted should be negative there. Construct an event where
        observed mass is uniform but model puts most mass on arm 0:
        predicted[arm=0] >> 1/9, so residual[arm=0] < 0; predicted on
        other arms < 1/9, so residual > 0."""
        rows = [
            {"event_id": 1, "arm": arm, "is_prev": 1.0 if arm == 0 else 0.0,
             "arm_fraction": 1.0 / 9}
            for arm in range(9)
        ]
        eat = pd.DataFrame(rows)
        fit_result = {"coef_is_prev": 5.0}  # huge positive: arm 0 dominates
        out = predict_arm_fractions(eat, fit_result, predictors=["is_prev"])
        residual_arm_0 = out.loc[out["arm"] == 0, "residual_frac"].iloc[0]
        residual_arm_1 = out.loc[out["arm"] == 1, "residual_frac"].iloc[0]
        assert residual_arm_0 < 0, "model overpredicted arm 0; residual should be negative"
        assert residual_arm_1 > 0, "model underpredicted arm 1; residual should be positive"

    def test_softmax_numerically_stable_under_large_logits(self):
        """The implementation subtracts the per-event max before
        exponentiating. Verify this doesn't blow up for huge β·X."""
        rows = [
            {"event_id": 1, "arm": arm, "is_prev": 1000.0 if arm == 0 else 0.0,
             "arm_fraction": 1.0 / 9}
            for arm in range(9)
        ]
        eat = pd.DataFrame(rows)
        fit_result = {"coef_is_prev": 1.0}
        out = predict_arm_fractions(eat, fit_result, predictors=["is_prev"])
        assert np.isfinite(out["predicted_frac"]).all()
        # arm 0 has logit 1000; everything else has logit 0. After
        # max-shift, arm 0's exp(0) = 1; others exp(-1000) ≈ 0. So
        # arm 0 fraction ≈ 1.
        assert out.loc[out["arm"] == 0, "predicted_frac"].iloc[0] == pytest.approx(1.0)


# ── build_residualized_choice_dataset ──


class TestBuildResidualizedChoiceDataset:
    def test_residual_column_added_and_finite(self):
        et, eat, st = _make_minimal_tables(
            [
                {"stop_id": 1, "current_arm": 0, "next_arm": 3},
                {"stop_id": 2, "current_arm": 5, "next_arm": 2},
            ]
        )
        fit_result = {"coef_is_prev": 0.0}
        ds = build_residualized_choice_dataset(
            et, eat, st, fit_result, content_predictors=["is_prev"]
        )
        assert "replay_frac_residual" in ds.columns
        assert np.isfinite(ds["replay_frac_residual"]).all()

    def test_zero_coefficients_residual_equals_replay_frac_minus_uniform(self):
        """With β = 0, predicted_frac is 1/9 everywhere, so
        residual = replay_frac - 1/9. After aggregating to stop×arm,
        this should hold for every row."""
        et, eat, st = _make_minimal_tables(
            [{"stop_id": 1, "current_arm": 0, "next_arm": 3}]
        )
        fit_result = {"coef_is_prev": 0.0}
        ds = build_residualized_choice_dataset(
            et, eat, st, fit_result, content_predictors=["is_prev"]
        )
        np.testing.assert_allclose(
            ds["replay_frac_residual"].to_numpy(),
            (ds["replay_frac"] - 1.0 / 9).to_numpy(),
            atol=1e-12,
        )

    def test_returns_empty_when_no_valid_stops(self):
        """When every stop fails the has_replay/next_arm/current_arm
        filter, the residualized dataset is empty (not a crash)."""
        et, eat, st = _make_minimal_tables(
            [{"stop_id": 1, "current_arm": np.nan, "next_arm": np.nan}]
        )
        fit_result = {"coef_is_prev": 0.0}
        ds = build_residualized_choice_dataset(
            et, eat, st, fit_result, content_predictors=["is_prev"]
        )
        assert len(ds) == 0


# ── fit_choice_model ──


class TestFitChoiceModel:
    def _choice_df_for_fit(self, n_stops=20, *, beta_replay=2.0, seed=0):
        """Generate a choice dataset where replay_frac is correlated
        with the chosen arm. Each stop has 8 candidate arms; the
        chosen arm is sampled from a softmax over replay_frac with
        coefficient ``beta_replay``."""
        rng = np.random.default_rng(seed)
        rows = []
        for stop_id in range(1, n_stops + 1):
            arms = rng.choice(9, size=8, replace=False)
            replay_frac = rng.uniform(0, 1, size=8)
            replay_frac /= replay_frac.sum()  # normalize
            logits = beta_replay * replay_frac
            probs = np.exp(logits - logits.max())
            probs /= probs.sum()
            chosen_idx = rng.choice(8, p=probs)
            for i, arm in enumerate(arms):
                rows.append({
                    "stop_id": stop_id,
                    "arm": int(arm),
                    "replay_frac": replay_frac[i],
                    "chosen": 1 if i == chosen_idx else 0,
                })
        return pd.DataFrame(rows)

    def test_returns_none_for_too_few_stops(self):
        df = self._choice_df_for_fit(n_stops=1)
        # 1 stop, 1 predictor → n_stops < 2 * n_params → None
        result = fit_choice_model(df, predictors=["replay_frac"])
        assert result is None

    def test_recovers_known_coefficient(self):
        """Generate with β_replay = 2.0; recovered coefficient should
        be within 3 SE of the truth."""
        df = self._choice_df_for_fit(n_stops=400, beta_replay=2.0, seed=11)
        result = fit_choice_model(df, predictors=["replay_frac"])
        assert result is not None
        z = (result["coef_replay_frac"] - 2.0) / result["bse_replay_frac"]
        assert abs(z) < 3, (
            f"recovered β = {result['coef_replay_frac']:.3f} (truth 2.0, z={z:.2f})"
        )

    def test_returns_none_when_predictor_missing(self):
        df = self._choice_df_for_fit(n_stops=20)
        result = fit_choice_model(df, predictors=["nonexistent"])
        assert result is None
