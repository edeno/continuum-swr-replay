"""Tests for analysis/behavior/choice/choice_prediction.py.

The choice models ask whether replay content predicts the next choice. The
data pipeline (``build_choice_dataset`` →
``build_residualized_choice_dataset``) and the conditional-logistic
fit (``fit_choice_model``) all live in this module. Pin the
invariants:

- ``build_choice_dataset``: well-arms-only choice sets (5 arms after
  removing occupancy_well_arm; trunk arms 6/7/8 excluded because they can
  never be next_arm), exactly one chosen arm per stop, occupancy_well_arm
  always excluded, NaN-occupancy_well_arm filter
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

from continuum_swr_replay.analysis.behavior.choice.choice_prediction import (
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
        Each dict needs at least ``stop_id``, ``occupancy_well_arm``,
        ``next_arm``.
    arm_fractions : dict[stop_id, list[float]], optional
        Per-stop arm_fraction vector (length 9). Defaults to uniform
        1/9 on every arm.
    """
    stop_table = pd.DataFrame(
        [
            {
                "animal": "test",
                "session_date": 1,
                "epoch": "02_r1",
                "has_replay": True,
                "n_selected_events": 1,
                "occupancy_state": "at_well"
                if pd.notna(s["occupancy_well_arm"])
                else "unknown",
                "at_well": pd.notna(s["occupancy_well_arm"]),
                "is_rewarded_outcome": 0,
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
                    "candidate_arm": arm,
                    "arm_fraction": fracs[arm],
                    "is_prev_arm": 0.0,
                    "is_next_arm": 0.0,
                    "visit_recency_trials": 0.0,
                    "is_candidate_arm_never_visited": float(arm in {2, 4}),
                    "reward_count_candidate_arm_last10_known_trials": 0.0,
                    "reward_fraction_candidate_arm_last10_visits": 0.0,
                    "reward_belief_mean_candidate_arm_preoutcome": 0.5,
                    "task_reward_probability_candidate_arm": 0.5,
                    "distance_to_candidate_arm_cm": 1.0,
                    "candidate_arm_distance_rank": 1.0,
                    "is_same_patch_as_current": 0,
                }
            )
    event_arm_table = pd.DataFrame(event_arm_rows)
    return event_table, event_arm_table, stop_table


# ── build_choice_dataset ──


class TestBuildChoiceDataset:
    def test_filters_explicit_non_well_stop_even_with_leaf_occupancy_well_arm(self):
        """Location state, not a coincidental arm value, defines the cohort."""
        et, eat, st = _make_minimal_tables(
            [
                {"stop_id": 1, "occupancy_well_arm": 0, "next_arm": 3},
                {
                    "stop_id": 2,
                    "occupancy_well_arm": 5,
                    "next_arm": 2,
                    "occupancy_state": "in_transit",
                    "at_well": False,
                },
            ]
        )

        ds = build_choice_dataset(et, eat, st)

        assert ds["stop_id"].unique().tolist() == [1]
        assert ds["at_well"].eq(True).all()

    def test_filters_stop_whose_at_well_is_missing(self):
        """A location-unknown stop is excluded, not admitted by default.

        ``at_well`` is NA before the first trip, where ``.eq(True)``
        yields NA; only the explicit fill keeps it out of the cohort.
        """
        et, eat, st = _make_minimal_tables(
            [
                {"stop_id": 1, "occupancy_well_arm": 0, "next_arm": 3},
                {
                    "stop_id": 2,
                    "occupancy_well_arm": 5,
                    "next_arm": 2,
                    "occupancy_state": "unknown",
                    "at_well": pd.NA,
                },
            ]
        )

        ds = build_choice_dataset(et, eat, st)

        assert ds["stop_id"].unique().tolist() == [1]

    def test_drops_stops_with_nan_occupancy_well_arm(self):
        """arm != NaN is True for every arm in IEEE 754, so the
        ``choice_df[choice_df['candidate_arm'] != choice_df['occupancy_well_arm']]``
        filter silently keeps all 9 arms when occupancy_well_arm is NaN.
        Such stops must be filtered at the stop level. The choice
        set is now WELL_ARMS \\ {occupancy_well_arm} = 5 arms (well arms
        0-5, minus the well the animal is sitting on)."""
        et, eat, st = _make_minimal_tables(
            [
                {"stop_id": 1, "occupancy_well_arm": 0, "next_arm": 3},
                {"stop_id": 2, "occupancy_well_arm": np.nan, "next_arm": 2},
                {"stop_id": 3, "occupancy_well_arm": 5, "next_arm": 2},
            ]
        )
        ds = build_choice_dataset(et, eat, st)
        sizes = ds.groupby("stop_id").size()
        assert (sizes == 5).all()
        assert 2 not in sizes.index
        assert set(sizes.index) == {1, 3}

    def test_exactly_one_chosen_arm_per_stop(self):
        et, eat, st = _make_minimal_tables(
            [
                {"stop_id": 1, "occupancy_well_arm": 0, "next_arm": 3},
                {"stop_id": 2, "occupancy_well_arm": 5, "next_arm": 2},
            ]
        )
        ds = build_choice_dataset(et, eat, st)
        chosen_per_stop = ds.groupby("stop_id")["chosen"].sum()
        assert (chosen_per_stop == 1).all()

    def test_chosen_arm_matches_next_arm(self):
        """The single ``chosen=1`` row in each stop must be the arm
        equal to that stop's ``next_arm``."""
        et, eat, st = _make_minimal_tables(
            [{"stop_id": 1, "occupancy_well_arm": 0, "next_arm": 3}]
        )
        ds = build_choice_dataset(et, eat, st)
        chosen_row = ds.loc[ds["chosen"] == 1].iloc[0]
        assert chosen_row["candidate_arm"] == 3

    def test_choice_set_excludes_trunk_arms(self):
        """Regression: the choice set must be the WELL ARMS (leaf
        arms 0-5, one per well) minus ``occupancy_well_arm`` — not
        ``ALL_ARMS`` minus ``occupancy_well_arm``. Trunk arms 6/7/8 connect
        patch junctions and can never be ``next_arm``; including
        them as unchosen alternatives in the conditional logit
        denominator changes the likelihood and coefficient
        interpretation. The same eligible-set is used by
        ``StopSummaryConfig.well_arms`` for "best arm" lookups, and
        ``arm_path_well_to_well`` documents that trunk arms cannot be
        stop endpoints.
        """
        et, eat, st = _make_minimal_tables(
            [
                {"stop_id": 1, "occupancy_well_arm": 0, "next_arm": 3},
                {"stop_id": 2, "occupancy_well_arm": 5, "next_arm": 2},
            ]
        )
        ds = build_choice_dataset(et, eat, st)
        # Trunk arms 6, 7, 8 must never appear as candidates.
        candidates = set(ds["candidate_arm"].unique())
        assert candidates.isdisjoint({6, 7, 8}), (
            f"trunk arms leaked into choice set: {candidates & {6, 7, 8}}; "
            "the conditional-logit denominator must be the well arms, "
            "not all 9 track segments."
        )
        # Per-stop choice set is exactly WELL_ARMS \\ {occupancy_well_arm}.
        for stop_id, occupancy_well_arm in [(1, 0), (2, 5)]:
            stop_arms = set(ds.loc[ds["stop_id"] == stop_id, "candidate_arm"])
            assert stop_arms == {0, 1, 2, 3, 4, 5} - {occupancy_well_arm}

    def test_occupancy_well_arm_never_in_choice_set(self):
        """occupancy_well_arm is the arm the animal is sitting on — by
        definition it cannot be the next destination, so it never
        appears as a candidate."""
        et, eat, st = _make_minimal_tables(
            [
                {"stop_id": 1, "occupancy_well_arm": 0, "next_arm": 3},
                {"stop_id": 2, "occupancy_well_arm": 5, "next_arm": 2},
            ]
        )
        ds = build_choice_dataset(et, eat, st)
        for stop_id, occupancy_well_arm in [(1, 0), (2, 5)]:
            stop_arms = set(ds.loc[ds["stop_id"] == stop_id, "candidate_arm"])
            assert occupancy_well_arm not in stop_arms

    def test_carries_learned_and_nominal_value_as_distinct_columns(self):
        et, eat, st = _make_minimal_tables(
            [{"stop_id": 1, "occupancy_well_arm": 0, "next_arm": 3}]
        )
        eat["reward_belief_mean_candidate_arm_preoutcome"] = eat["candidate_arm"] / 10.0
        eat["task_reward_probability_candidate_arm"] = 1.0 - eat["candidate_arm"] / 20.0

        ds = build_choice_dataset(et, eat, st).set_index("candidate_arm")

        np.testing.assert_allclose(
            ds["reward_belief_mean_candidate_arm_preoutcome"],
            ds.index.to_numpy() / 10.0,
        )
        np.testing.assert_allclose(
            ds["task_reward_probability_candidate_arm"],
            1.0 - ds.index.to_numpy() / 20.0,
        )

    def test_drops_stops_with_nan_next_arm(self):
        """Without a known next destination, ``chosen`` is undefined;
        such stops must be excluded."""
        et, eat, st = _make_minimal_tables(
            [
                {"stop_id": 1, "occupancy_well_arm": 0, "next_arm": 3},
                {"stop_id": 2, "occupancy_well_arm": 5, "next_arm": np.nan},
            ]
        )
        ds = build_choice_dataset(et, eat, st)
        assert set(ds["stop_id"].unique()) == {1}

    def test_drops_stops_without_replay(self):
        """``has_replay = False`` means no events to compute
        replay_fraction from — exclude the stop."""
        et, eat, st = _make_minimal_tables(
            [
                {"stop_id": 1, "occupancy_well_arm": 0, "next_arm": 3},
                {"stop_id": 2, "occupancy_well_arm": 5, "next_arm": 2},
            ]
        )
        st.loc[st["stop_id"] == 2, "has_replay"] = False
        ds = build_choice_dataset(et, eat, st)
        assert set(ds["stop_id"].unique()) == {1}

    def test_replay_fraction_averaged_across_events_in_stop(self):
        """Two events in the same stop with different replay_fraction
        vectors should yield a per-arm average."""
        et, eat, st = _make_minimal_tables(
            [{"stop_id": 1, "occupancy_well_arm": 0, "next_arm": 3}]
        )
        # Add a second event for stop 1 with all mass on arm 3.
        new_event_rows = []
        for arm in range(9):
            new_event_rows.append(
                {
                    "event_id": 2,
                    "stop_id": 1,
                    "candidate_arm": arm,
                    "arm_fraction": 1.0 if arm == 3 else 0.0,
                    "is_prev_arm": 0.0,
                    "is_next_arm": 0.0,
                    "visit_recency_trials": 0.0,
                    "is_candidate_arm_never_visited": float(arm in {2, 4}),
                    "reward_count_candidate_arm_last10_known_trials": 0.0,
                    "reward_fraction_candidate_arm_last10_visits": 0.0,
                    "reward_belief_mean_candidate_arm_preoutcome": 0.5,
                    "task_reward_probability_candidate_arm": 0.5,
                    "distance_to_candidate_arm_cm": 1.0,
                    "candidate_arm_distance_rank": 1.0,
                    "is_same_patch_as_current": 0,
                }
            )
        eat = pd.concat([eat, pd.DataFrame(new_event_rows)], ignore_index=True)
        et = pd.concat(
            [et, pd.DataFrame({"event_id": [2], "stop_id": [1]})],
            ignore_index=True,
        )

        ds = build_choice_dataset(et, eat, st)
        # arm 3: mean of (1/9, 1.0) = 5/9
        arm3 = ds.loc[ds["candidate_arm"] == 3, "replay_fraction"].iloc[0]
        assert arm3 == pytest.approx((1.0 / 9 + 1.0) / 2)
        # arm 1: mean of (1/9, 0) = 1/18
        arm1 = ds.loc[ds["candidate_arm"] == 1, "replay_fraction"].iloc[0]
        assert arm1 == pytest.approx((1.0 / 9) / 2)

    def test_partial_nan_event_propagates_nan_replay_frac(self):
        """Regression: build_choice_dataset's per-(stop, arm) mean
        must be strict NaN-propagating.

        Pre-fix used pandas' default ``"mean"`` aggregator
        (``skipna=True``), so a stop with two events — one with
        finite ``arm_fraction`` and one with NaN — silently emitted
        ``replay_fraction = <surviving event's value>`` while
        ``n_selected_events == 2``. The downstream chosen-arm sanity check
        only inspects ``chosen.sum() == 1`` per stop, so it does
        not detect the NaN that has already been collapsed away.
        That biases Stage-5A choice prediction toward the subset of
        events that happened to have a finite ``arm_fraction``.

        Post-fix uses ``mean(skipna=False)`` so any NaN event
        propagates a NaN ``replay_fraction`` for that (stop, arm), and
        ``fit_choice_model``'s drop-NaN-required-predictors step
        removes the contaminated stop.
        """
        et, eat, st = _make_minimal_tables(
            [{"stop_id": 1, "occupancy_well_arm": 0, "next_arm": 3}]
        )
        # Add a second event for the same stop where arm 3's
        # arm_fraction is NaN (other arms keep finite values).
        new_event_rows = []
        for arm in range(9):
            new_event_rows.append(
                {
                    "event_id": 2,
                    "stop_id": 1,
                    "candidate_arm": arm,
                    "arm_fraction": np.nan if arm == 3 else 1.0 / 8,
                    "is_prev_arm": 0.0,
                    "is_next_arm": 0.0,
                    "visit_recency_trials": 0.0,
                    "is_candidate_arm_never_visited": float(arm in {2, 4}),
                    "reward_count_candidate_arm_last10_known_trials": 0.0,
                    "reward_fraction_candidate_arm_last10_visits": 0.0,
                    "reward_belief_mean_candidate_arm_preoutcome": 0.5,
                    "task_reward_probability_candidate_arm": 0.5,
                    "distance_to_candidate_arm_cm": 1.0,
                    "candidate_arm_distance_rank": 1.0,
                    "is_same_patch_as_current": 0,
                }
            )
        eat = pd.concat([eat, pd.DataFrame(new_event_rows)], ignore_index=True)
        et = pd.concat(
            [et, pd.DataFrame({"event_id": [2], "stop_id": [1]})],
            ignore_index=True,
        )

        ds = build_choice_dataset(et, eat, st)
        # arm 3 had NaN in event 2; replay_fraction for arm 3 must be
        # NaN, not the surviving event 1's value of 1/9.
        arm3 = ds.loc[ds["candidate_arm"] == 3, "replay_fraction"].iloc[0]
        assert np.isnan(arm3), (
            f"replay_fraction for arm 3 = {arm3} — strict aggregator "
            "should propagate NaN when any event has NaN "
            "arm_fraction; pre-fix silently emitted the surviving "
            "event's value (~0.111)."
        )
        # Other arms (no NaN) should still be finite.
        arm1 = ds.loc[ds["candidate_arm"] == 1, "replay_fraction"].iloc[0]
        assert not np.isnan(arm1)

    def test_never_visited_indicator_is_carried_into_choice_rows(self):
        et, eat, st = _make_minimal_tables(
            [{"stop_id": 1, "occupancy_well_arm": 0, "next_arm": 3}]
        )
        ds = build_choice_dataset(et, eat, st).set_index("candidate_arm")
        np.testing.assert_array_equal(
            ds.loc[[1, 2, 3, 4, 5], "is_candidate_arm_never_visited"].to_numpy(),
            [0.0, 1.0, 0.0, 1.0, 0.0],
        )


# ── predict_arm_fractions ──


class TestPredictArmFractions:
    def _eat_for_softmax(self, n_events=2, predictor_value_fn=None):
        rows = []
        for ev in range(1, n_events + 1):
            for arm in range(9):
                rows.append(
                    {
                        "event_id": ev,
                        "candidate_arm": arm,
                        "is_prev_arm": predictor_value_fn(ev, arm)
                        if predictor_value_fn
                        else 0.0,
                        "arm_fraction": 1.0 / 9,
                    }
                )
        return pd.DataFrame(rows)

    def test_predicted_frac_sums_to_one_per_event(self):
        """Softmax is normalized within each event, so per-event
        ``sum(predicted_frac) == 1.0`` must hold exactly."""
        eat = self._eat_for_softmax(
            n_events=3,
            predictor_value_fn=lambda ev, arm: 1.0 if arm == ev else 0.0,
        )
        fit_result = {"coef_is_prev_arm": 1.5}
        out = predict_arm_fractions(eat, fit_result, predictors=["is_prev_arm"])
        per_event_sum = out.groupby("event_id")["predicted_frac"].sum()
        np.testing.assert_allclose(per_event_sum.to_numpy(), 1.0, atol=1e-12)

    def test_zero_coefficients_yield_uniform_predictions(self):
        """With every coefficient at zero, softmax(0) over 9 arms is
        1/9 on every arm."""
        eat = self._eat_for_softmax(
            n_events=2,
            predictor_value_fn=lambda ev, arm: 1.0 if arm == 0 else 0.0,
        )
        fit_result = {"coef_is_prev_arm": 0.0}
        out = predict_arm_fractions(eat, fit_result, predictors=["is_prev_arm"])
        np.testing.assert_allclose(out["predicted_frac"].to_numpy(), 1.0 / 9)

    def test_residual_is_observed_minus_predicted(self):
        eat = self._eat_for_softmax(n_events=1)
        fit_result = {"coef_is_prev_arm": 0.0}
        out = predict_arm_fractions(eat, fit_result, predictors=["is_prev_arm"])
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
            {
                "event_id": 1,
                "candidate_arm": arm,
                "is_prev_arm": 1.0 if arm == 0 else 0.0,
                "arm_fraction": 1.0 / 9,
            }
            for arm in range(9)
        ]
        eat = pd.DataFrame(rows)
        fit_result = {"coef_is_prev_arm": 5.0}  # huge positive: arm 0 dominates
        out = predict_arm_fractions(eat, fit_result, predictors=["is_prev_arm"])
        residual_arm_0 = out.loc[out["candidate_arm"] == 0, "residual_frac"].iloc[0]
        residual_arm_1 = out.loc[out["candidate_arm"] == 1, "residual_frac"].iloc[0]
        assert residual_arm_0 < 0, (
            "model overpredicted arm 0; residual should be negative"
        )
        assert residual_arm_1 > 0, (
            "model underpredicted arm 1; residual should be positive"
        )

    def test_softmax_numerically_stable_under_large_logits(self):
        """The implementation subtracts the per-event max before
        exponentiating. Verify this doesn't blow up for huge β·X."""
        rows = [
            {
                "event_id": 1,
                "candidate_arm": arm,
                "is_prev_arm": 1000.0 if arm == 0 else 0.0,
                "arm_fraction": 1.0 / 9,
            }
            for arm in range(9)
        ]
        eat = pd.DataFrame(rows)
        fit_result = {"coef_is_prev_arm": 1.0}
        out = predict_arm_fractions(eat, fit_result, predictors=["is_prev_arm"])
        assert np.isfinite(out["predicted_frac"]).all()
        # arm 0 has logit 1000; everything else has logit 0. After
        # max-shift, arm 0's exp(0) = 1; others exp(-1000) ≈ 0. So
        # arm 0 fraction ≈ 1.
        assert out.loc[out["candidate_arm"] == 0, "predicted_frac"].iloc[
            0
        ] == pytest.approx(1.0)

    def test_nonfinite_predictor_drops_whole_event(self):
        eat = self._eat_for_softmax(n_events=2)
        eat.loc[(eat["event_id"] == 1) & (eat["candidate_arm"] == 0), "is_prev_arm"] = (
            np.inf
        )

        out = predict_arm_fractions(
            eat, {"coef_is_prev_arm": 1.0}, predictors=["is_prev_arm"]
        )

        assert out.loc[out["event_id"] == 1, "predicted_frac"].isna().all()
        assert np.isfinite(out.loc[out["event_id"] == 2, "predicted_frac"]).all()


def test_predictions_take_a_fits_unidentified_terms_to_their_limits():
    """A content fit whose session never reached arm 3 reports that arm's dummy
    as not estimated (its coefficient runs to -inf). Prediction must use that
    limit, not fail: arm 3 gets fraction 0, the column constant within every
    event contributes nothing, and the other arms share the mass by the
    identified terms."""
    from continuum_swr_replay.analysis.behavior.content.simulation import (
        simulate_with_event_covariates,
    )
    from continuum_swr_replay.analysis.behavior.fitting import fit_conditional_poisson

    eat = simulate_with_event_covariates(
        {"is_current_arm": 1.2, "is_prev_arm": 0.6}, n_events=150, L_e=10, seed=4
    )
    eat.loc[eat["arm_id"] == 3, "arm_count_hard"] = 0
    eat = eat[eat.groupby("event_id")["arm_count_hard"].transform("sum") > 0]
    eat["arm_fraction"] = eat["arm_count_hard"] / eat.groupby("event_id")[
        "arm_count_hard"
    ].transform("sum")
    eat["ev_level"] = eat["event_id"].astype(float)
    identified = ["is_current_arm", "is_prev_arm"]
    predictors = [*identified, "is_candidate_arm_3", "ev_level"]
    fit = fit_conditional_poisson(eat, predictors)
    assert set(fit["unidentified_terms"].split("|")) == {
        "is_candidate_arm_3:separated_to_minus_inf",
        "ev_level:no_within_event_variation",
    }

    out = predict_arm_fractions(eat, fit, predictors)

    arm_3 = out["arm_id"] == 3
    assert (out.loc[arm_3, "predicted_frac"] == 0).all()
    np.testing.assert_allclose(
        out.groupby("event_id")["predicted_frac"].sum().to_numpy(), 1.0, atol=1e-12
    )
    rest = out[~arm_3].copy()
    beta = np.array([fit[f"coef_{t}"] for t in identified])
    rest["w"] = np.exp(rest[identified].to_numpy(dtype=float) @ beta)
    expected = rest["w"] / rest.groupby("event_id")["w"].transform("sum")
    np.testing.assert_allclose(rest["predicted_frac"], expected, rtol=1e-12)


# ── build_residualized_choice_dataset ──


class TestBuildResidualizedChoiceDataset:
    def test_residual_column_added_and_finite(self):
        et, eat, st = _make_minimal_tables(
            [
                {"stop_id": 1, "occupancy_well_arm": 0, "next_arm": 3},
                {"stop_id": 2, "occupancy_well_arm": 5, "next_arm": 2},
            ]
        )
        fit_result = {"coef_is_prev_arm": 0.0}
        ds = build_residualized_choice_dataset(
            et, eat, st, fit_result, content_predictors=["is_prev_arm"]
        )
        assert "replay_fraction_residual" in ds.columns
        assert np.isfinite(ds["replay_fraction_residual"]).all()

    def test_zero_coefficients_residual_equals_replay_frac_minus_uniform(self):
        """With β = 0, predicted_frac is 1/9 everywhere, so
        residual = replay_fraction - 1/9. After aggregating to stop×arm,
        this should hold for every row."""
        et, eat, st = _make_minimal_tables(
            [{"stop_id": 1, "occupancy_well_arm": 0, "next_arm": 3}]
        )
        fit_result = {"coef_is_prev_arm": 0.0}
        ds = build_residualized_choice_dataset(
            et, eat, st, fit_result, content_predictors=["is_prev_arm"]
        )
        np.testing.assert_allclose(
            ds["replay_fraction_residual"].to_numpy(),
            (ds["replay_fraction"] - 1.0 / 9).to_numpy(),
            atol=1e-12,
        )

    def test_returns_empty_when_no_valid_stops(self):
        """When every stop fails the has_replay/next_arm/occupancy_well_arm
        filter, the residualized dataset is empty (not a crash)."""
        et, eat, st = _make_minimal_tables(
            [{"stop_id": 1, "occupancy_well_arm": np.nan, "next_arm": np.nan}]
        )
        fit_result = {"coef_is_prev_arm": 0.0}
        ds = build_residualized_choice_dataset(
            et, eat, st, fit_result, content_predictors=["is_prev_arm"]
        )
        assert len(ds) == 0


# ── fit_choice_model ──


class TestFitChoiceModel:
    def _choice_df_for_fit(self, n_stops=20, *, beta_replay=2.0, seed=0):
        """Generate a choice dataset where replay_fraction is correlated
        with the chosen arm. Each stop has 5 candidate arms (matches
        production: well arms 0-5 minus occupancy_well_arm). The chosen arm
        is sampled from a softmax over replay_fraction with coefficient
        ``beta_replay``."""
        rng = np.random.default_rng(seed)
        well_arms = (0, 1, 2, 3, 4, 5)
        n_alts = len(well_arms) - 1  # 5 — well arms minus occupancy_well_arm
        rows = []
        for stop_id in range(1, n_stops + 1):
            arms = rng.choice(well_arms, size=n_alts, replace=False)
            replay_fraction = rng.uniform(0, 1, size=n_alts)
            replay_fraction /= replay_fraction.sum()  # normalize
            logits = beta_replay * replay_fraction
            probs = np.exp(logits - logits.max())
            probs /= probs.sum()
            chosen_idx = rng.choice(n_alts, p=probs)
            for i, arm in enumerate(arms):
                rows.append(
                    {
                        "stop_id": stop_id,
                        "candidate_arm": int(arm),
                        "replay_fraction": replay_fraction[i],
                        "chosen": 1 if i == chosen_idx else 0,
                    }
                )
        return pd.DataFrame(rows)

    def test_returns_none_for_too_few_stops(self):
        df = self._choice_df_for_fit(n_stops=1)
        # 1 stop, 1 predictor → n_stops < 2 * n_params → None
        result = fit_choice_model(df, predictors=["replay_fraction"])
        assert result is None

    def test_recovers_known_coefficient(self):
        """Generate with β_replay = 2.0; recovered coefficient should
        be within 3 SE of the truth."""
        df = self._choice_df_for_fit(n_stops=400, beta_replay=2.0, seed=11)
        result = fit_choice_model(df, predictors=["replay_fraction"])
        assert result is not None
        z = (result["coef_replay_fraction"] - 2.0) / result["bse_replay_fraction"]
        assert abs(z) < 3, (
            f"recovered β = {result['coef_replay_fraction']:.3f} (truth 2.0, z={z:.2f})"
        )

    def test_returns_none_when_predictor_missing(self):
        df = self._choice_df_for_fit(n_stops=20)
        result = fit_choice_model(df, predictors=["nonexistent"])
        assert result is None

    def test_nonfinite_predictor_drops_whole_stop(self):
        df = self._choice_df_for_fit(n_stops=20)
        df.loc[df["stop_id"] == 1, "replay_fraction"] = np.inf

        result = fit_choice_model(df, predictors=["replay_fraction"])

        assert result is not None
        assert result["n_stops"] == 19
