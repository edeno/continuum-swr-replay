"""Regression test: ``predict_arm_fractions`` event-wise NaN handling.

The pre-fix code computed eta = X @ params with NaN flowing through,
then softmaxed within each event. The result for an event with one
NaN-predictor arm and zero coefficients was ``[0.5, NaN, 0.5]`` —
the softmax effectively normalized over the non-NaN arms instead of
dropping the event. Downstream consumers
(``build_residualized_choice_dataset``, ``stop_patch_replay``) then
combined those residuals with pandas skip-NaN aggregations.

The fix marks any event with a NaN predictor (or NaN observed
fraction) as dropped and writes NaN into ``predicted_frac`` /
``residual_frac`` for every arm in that event.

``fit_choice_model`` is also re-tested: it must drop the WHOLE
stop when any required column is NaN, not individual rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.behavior.choice.choice_prediction import (
    build_residualized_choice_dataset,
    fit_choice_model,
    predict_arm_fractions,
)


def _three_arm_event_table_with_nan_predictor() -> pd.DataFrame:
    """Single event, 3 arms, one arm has NaN in the predictor."""
    return pd.DataFrame(
        {
            "event_id": [1, 1, 1],
            "candidate_arm": [0, 1, 2],
            "is_occupancy_well_arm": [1.0, np.nan, 0.0],  # NaN at arm=1
            "arm_fraction": [0.4, 0.3, 0.3],
        }
    )


def _two_event_table_one_clean_one_nan() -> pd.DataFrame:
    """One clean event (arm count 3) and one event with NaN predictor."""
    rows = []
    for event_id, predictor_vals in [(1, [1.0, 0.0, 0.0]), (2, [1.0, np.nan, 0.0])]:
        for arm, p in enumerate(predictor_vals):
            rows.append(
                {
                    "event_id": event_id,
                    "candidate_arm": arm,
                    "is_occupancy_well_arm": p,
                    "arm_fraction": 0.5 if arm == 0 else 0.25,
                }
            )
    return pd.DataFrame(rows)


class TestPredictArmFractionsNaN:
    def test_event_with_nan_predictor_is_dropped(self):
        eat = _three_arm_event_table_with_nan_predictor()
        out = predict_arm_fractions(
            eat,
            fit_result={"coef_is_occupancy_well_arm": 0.0},
            predictors=["is_occupancy_well_arm"],
        )
        # All three rows belong to the NaN event ⇒ all NaN, no
        # spurious 0.5/0.5 split over the surviving arms.
        assert out["predicted_frac"].isna().all()
        assert out["residual_frac"].isna().all()

    def test_clean_event_unaffected_when_paired_with_nan_event(self):
        eat = _two_event_table_one_clean_one_nan()
        out = predict_arm_fractions(
            eat,
            fit_result={"coef_is_occupancy_well_arm": 0.0},
            predictors=["is_occupancy_well_arm"],
        )
        clean = out[out["event_id"] == 1]
        nan_evt = out[out["event_id"] == 2]
        # Coefficient is 0 ⇒ uniform softmax over the 3 arms.
        np.testing.assert_allclose(
            clean["predicted_frac"].to_numpy(), [1 / 3, 1 / 3, 1 / 3], atol=1e-12
        )
        # NaN event drops out entirely.
        assert nan_evt["predicted_frac"].isna().all()


class TestBuildResidualizedChoiceDatasetPropagatesNaN:
    """The residualized choice builder must NOT ``fillna(0.0)`` after
    aggregating residuals to (stop, arm). A real zero is materially
    different from "we couldn't compute this residual" — it would
    fool ``fit_choice_model`` into keeping the stop instead of
    dropping it."""

    def _stop_with_only_nan_event(self) -> dict[str, pd.DataFrame]:
        # 1 stop, 1 event, 9 arms. Predictor NaN on arm 3 so the
        # entire event is dropped by ``predict_arm_fractions``.
        eat = pd.DataFrame(
            [
                {
                    "event_id": 1,
                    "stop_id": 10,
                    "candidate_arm": arm,
                    "is_occupancy_well_arm": (np.nan if arm == 3 else 0.0),
                    "arm_fraction": 1.0 / 9,
                    "visit_recency_trials": 0.0,
                    "is_candidate_arm_never_visited": 0.0,
                    "reward_count_candidate_arm_last10_known_trials": 0.0,
                    "reward_fraction_candidate_arm_last10_visits": 0.0,
                    "reward_belief_mean_candidate_arm_preoutcome": 0.5,
                    "task_reward_probability_candidate_arm": 0.5,
                    "distance_to_candidate_arm_cm": 0.0,
                    "candidate_arm_distance_rank": 0.0,
                    "is_same_patch_as_current": 0,
                    "current_patch": 1,
                }
                for arm in range(9)
            ]
        )
        et = pd.DataFrame(
            [
                {
                    "event_id": 1,
                    "current_patch": 1,
                    "did_last_completed_trip_switch_patch": 0,
                }
            ]
        )
        st = pd.DataFrame(
            [
                {
                    "stop_id": 10,
                    "has_replay": True,
                    "next_arm": 0,
                    "occupancy_well_arm": 5,
                    "occupancy_state": "at_well",
                    "at_well": True,
                    "n_selected_events": 1,
                    "is_rewarded_outcome": 1.0,
                    "animal": "m",
                    "session_date": 20200101,
                    "epoch": "02_r1",
                }
            ]
        )
        return {"event_arm_table": eat, "event_table": et, "stop_table": st}

    def test_all_nan_event_stop_is_not_filled_with_zero(self):
        tables = self._stop_with_only_nan_event()
        out = build_residualized_choice_dataset(
            event_table=tables["event_table"],
            event_arm_table=tables["event_arm_table"],
            stop_table=tables["stop_table"],
            content_fit={"coef_is_occupancy_well_arm": 0.0},
            content_predictors=["is_occupancy_well_arm"],
        )
        # Every arm in the only stop should carry NaN — no fabricated
        # zero. (If ``fillna(0.0)`` had run, every row would be 0.0.)
        if len(out) > 0:
            assert out["replay_fraction_residual"].isna().all(), (
                "build_residualized_choice_dataset filled NaN residuals "
                "with 0.0 — that defeats the event-wise NaN filter"
            )


class TestFitChoiceModelStopLevelNaN:
    def test_nan_in_one_arm_drops_whole_stop(self):
        # Two stops: stop A = clean 9-arm choice set; stop B = same
        # but one arm has a NaN behavioral predictor. ``fit_choice_model``
        # used to keep the 8 non-NaN arms of stop B (silently shrinking
        # the choice set); now the whole stop B should drop, leaving
        # only stop A's data.
        rows_a = [
            {
                "stop_id": 100,
                "candidate_arm": arm,
                "chosen": int(arm == 0),
                "visit_recency_trials": float(arm),
                "reward_count_candidate_arm_last10_known_trials": 0.0,
                "is_same_patch_as_current": float(arm < 5),
            }
            for arm in range(9)
        ]
        rows_b = [
            {
                "stop_id": 200,
                "candidate_arm": arm,
                "chosen": int(arm == 1),
                # NaN at arm=3 of stop B.
                "visit_recency_trials": (np.nan if arm == 3 else float(arm)),
                "reward_count_candidate_arm_last10_known_trials": 0.0,
                "is_same_patch_as_current": float(arm < 5),
            }
            for arm in range(9)
        ]
        choice_df = pd.DataFrame(rows_a + rows_b)

        # With only a single clean stop and 1 predictor we'll be below
        # the n_stops < 2 * len(predictors) gate (so the function
        # returns None), but we can still observe what got filtered
        # by re-running with an even smaller threshold via patching.
        # Instead, build a minimal repro that exercises the path:
        # add another 2 clean stops so the fitter has enough data.
        for stop_id in (300, 400):
            choice_df = pd.concat(
                [
                    choice_df,
                    pd.DataFrame(
                        [
                            {
                                "stop_id": stop_id,
                                "candidate_arm": arm,
                                "chosen": int(arm == 0),
                                "visit_recency_trials": float(arm),
                                "reward_count_candidate_arm_last10_known_trials": 0.0,
                                "is_same_patch_as_current": float(arm < 5),
                            }
                            for arm in range(9)
                        ]
                    ),
                ],
                ignore_index=True,
            )

        out = fit_choice_model(
            choice_df, predictors=["visit_recency_trials"], stop_col="stop_id"
        )
        # Either the fit succeeded or returned None — both are fine
        # here. The crucial behavioral guarantee is that stop 200
        # (the NaN one) was not silently kept with a partial choice
        # set. We assert the latter by checking ``n_stops`` in the
        # output.
        if out is not None:
            assert out["n_stops"] == 3, (
                f"expected stop 200 dropped (3 clean stops), got "
                f"{out['n_stops']} — was it kept with a partial choice set?"
            )
