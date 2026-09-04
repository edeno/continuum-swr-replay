"""Tests for the behavioral-uncertainty/replay-content estimands."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.replay.uncertainty_content import (
    assign_within_arm_quantiles,
    build_trial_leaf_table,
    hierarchical_quartile_profile,
    paired_quartile_contrasts,
    summarize_horserace,
    summarize_mu_coefficients,
    within_arm_horserace,
)
from scripts.figures import generate_uncertainty_content_figure as figure_script


def _horserace_frame(uncertainty_slope: float, *, animal: str = "rat") -> pd.DataFrame:
    """Event x arm frame where non-current-leaf content tracks uncertainty within arm.

    Content on the five non-current leaf arms is ``0.2 + slope * variance_z``;
    the current arm (0) and trunk arms (6-8) carry unrelated content that the
    estimator must exclude. Three control predictors (value / distance / recency)
    vary independently of uncertainty, so a correct within-arm fit recovers ``slope``
    on uncertainty and ~0 on the controls.
    """
    rows = []
    for event in range(20):
        uncertainty = float(event)
        value = float(event % 5)
        distance = float((event * 7) % 11)
        recency = float((event * 3) % 13)
        for arm in range(9):
            is_current = int(arm == 0)
            if arm > 5:  # trunk — must be excluded
                fraction = 0.9
            elif is_current:  # current leaf — must be excluded
                fraction = 0.9
            else:
                fraction = 0.2 + uncertainty_slope * uncertainty
            rows.append(
                {
                    "animal": animal,
                    "session_date": 1,
                    "epoch": "e1",
                    "event_id": 100 + event,
                    "candidate_arm": arm,
                    "is_current_arm": is_current,
                    "is_swr": False,
                    "arm_fraction": fraction,
                    "reward_belief_variance_candidate_arm_preoutcome_z": uncertainty,
                    "reward_belief_mean_candidate_arm_preoutcome": value,
                    "distance_to_candidate_arm_cm": distance,
                    "visit_recency_trials": recency,
                }
            )
    return pd.DataFrame(rows)


def test_within_arm_horserace_recovers_uncertainty_and_excludes_current_trunk():
    coefs = within_arm_horserace(_horserace_frame(0.1), min_rows=10)

    # only the 'all' and 'non_SWR' populations fit (is_swr is all False)
    fit_all = coefs[coefs["population"] == "all"]
    assert len(fit_all) == 1
    # 20 events x 5 non-current leaf arms; current (arm 0) and trunk (6-8) dropped
    assert int(fit_all["n_rows"].iloc[0]) == 100
    row = fit_all.iloc[0]
    assert row[
        "coef_reward_belief_variance_candidate_arm_preoutcome_z"
    ] == pytest.approx(0.1, abs=1e-6)
    assert row[
        "coef_reward_belief_mean_candidate_arm_preoutcome_z"
    ] == pytest.approx(0.0, abs=1e-6)
    assert row["coef_distance_to_candidate_arm_cm_z"] == pytest.approx(
        0.0, abs=1e-6
    )
    assert row["coef_visit_recency_trials_z"] == pytest.approx(0.0, abs=1e-6)
    assert np.isfinite(row["cond_number"])


def test_within_arm_horserace_skips_rank_deficient_session_without_crashing():
    # Make two predictors perfectly collinear (distance == value) so the within-arm
    # design is rank-deficient; the session must be dropped, not crash or emit garbage.
    frame = _horserace_frame(0.1)
    frame["distance_to_candidate_arm_cm"] = frame["reward_belief_mean_candidate_arm_preoutcome"]
    coefs = within_arm_horserace(frame, min_rows=10)
    assert coefs.empty  # every (session, population) design is rank-deficient -> skipped


def test_within_arm_horserace_recovers_per_population_slopes_on_mixed_swr():
    # SWR and non-SWR events carry DIFFERENT within-arm uncertainty slopes; each
    # population must recover its own, and row counts must partition (SWR+nonSWR=all).
    rows = []
    for event in range(40):
        is_swr = event >= 20
        slope = 0.2 if is_swr else 0.05
        uncertainty = float(event)
        for arm in range(9):
            is_current = int(arm == 0)
            fraction = 0.9 if (arm > 5 or is_current) else 0.2 + slope * uncertainty
            rows.append(
                {
                    "animal": "rat", "session_date": 1, "epoch": "e1", "event_id": 100 + event,
                    "candidate_arm": arm, "is_current_arm": is_current, "is_swr": bool(is_swr),
                    "arm_fraction": fraction, "reward_belief_variance_candidate_arm_preoutcome_z": uncertainty,
                    "reward_belief_mean_candidate_arm_preoutcome": float(event % 5), "distance_to_candidate_arm_cm": float((event * 7) % 11),
                    "visit_recency_trials": float((event * 3) % 13),
                }
            )
    coefs = within_arm_horserace(pd.DataFrame(rows), min_rows=10)

    def _coef(pop):
        return coefs[coefs["population"] == pop][
            "coef_reward_belief_variance_candidate_arm_preoutcome_z"
        ].iloc[0]

    assert _coef("SWR") == pytest.approx(0.2, abs=1e-6)
    assert _coef("non_SWR") == pytest.approx(0.05, abs=1e-6)
    # 20 events x 5 non-current leaf arms per population; all = SWR + non_SWR
    n = {p: int(coefs[coefs.population == p]["n_rows"].iloc[0]) for p in ("all", "SWR", "non_SWR")}
    assert n["SWR"] == 100 and n["non_SWR"] == 100 and n["all"] == 200


def test_summarize_horserace_is_animal_balanced_and_signs_uncertainty_positive():
    coefs = pd.concat(
        [
            within_arm_horserace(_horserace_frame(0.1, animal="a"), min_rows=10),
            within_arm_horserace(_horserace_frame(0.08, animal="b"), min_rows=10),
        ],
        ignore_index=True,
    )
    summary = summarize_horserace(coefs)
    unc = summary[
        (summary["population"] == "all")
        & (summary["predictor"] == "reward_belief_variance_candidate_arm_preoutcome_z")
    ].iloc[0]
    assert unc["median"] > 0
    assert unc["n_animals"] == 2
    assert unc["n_animals_consistent"] == 2
    assert unc["median_cond_number"] > 0


def _synthetic_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create complete stop, belief, and event-arm tables without mocks."""
    stop_rows = []
    qval_rows = []
    stop_specs = [
        # stop_id, trial, exposure, chosen arm
        (10, 0, 1.0, 0),
        (11, 0, 2.0, 0),  # a second stop carrying the same trial belief
        (12, 1, 1.5, 1),  # zero-event trial
        (13, 2, 2.0, 2),
        (14, 3, 1.0, 3),
    ]
    for stop_idx, (stop_id, trial, exposure, chosen_arm) in enumerate(stop_specs):
        stop_rows.append(
            {
                "animal": "rat",
                "session_date": 1,
                "epoch": "e1",
                "stop_idx": stop_idx,
                "stop_id": stop_id,
                "immobility_duration_s": exposure,
            }
        )
        qval_row = {
            "stop_id": stop_id,
            "epoch": "e1",
            "trial_number_by_epoch": trial,
            "chosen_arm": chosen_arm,
            "trial_reward": int(trial % 2 == 0),
            "in_poke": bool(stop_idx % 2),
            "pump_on_at_bout_mid": False,
        }
        # Each physical well has four ordered, trial-level uncertainty states.
        for arm in range(6):
            qval_row[f"reward_belief_variance_port_{arm}"] = 0.01 * (trial + 1) + arm * 1e-4
        qval_rows.append(qval_row)

    # Trial 0 has one event at each of its two stops. Trial 2 and 3 have one.
    event_specs = [
        (100, 10, [1.0, 3.0, 2.0, 0.5, 0.5, 1.0]),
        (101, 11, [2.0, 1.0, 1.0, 1.0, 0.5, 0.5]),
        (102, 13, [0.5, 0.5, 3.0, 1.0, 1.0, 1.0]),
        (103, 14, [0.5, 0.5, 1.0, 3.0, 1.0, 1.0]),
    ]
    event_arm_rows = []
    for event_id, stop_id, leaf_mass in event_specs:
        for arm in range(9):
            mass = leaf_mass[arm] if arm < 6 else 0.0
            event_arm_rows.append(
                {
                    "animal": "rat",
                    "session_date": 1,
                    "epoch": "e1",
                    "event_id": event_id,
                    "stop_id": stop_id,
                    "candidate_arm": arm,
                    "arm_count_hard": float(round(mass)),
                    "arm_count_soft": mass,
                    "arm_posterior_mass": mass,
                }
            )

    return (
        pd.DataFrame(event_arm_rows),
        pd.DataFrame(stop_rows),
        pd.DataFrame(qval_rows),
    )


def test_build_trial_leaf_table_aggregates_repeated_stops_once_per_trial():
    event_arm, stops, qvals = _synthetic_inputs()

    trial_leaf, audit = build_trial_leaf_table(event_arm, stops, qvals)

    assert len(trial_leaf) == 4 * 6
    assert trial_leaf.groupby("trial_number_by_epoch")["candidate_arm"].nunique().eq(6).all()
    assert audit["n_mapped_trials"] == 4

    trial0 = trial_leaf[trial_leaf["trial_number_by_epoch"] == 0]
    assert trial0["n_stops"].eq(2).all()
    assert trial0["immobility_duration_s"].eq(3.0).all()
    assert trial0["n_events"].eq(2).all()
    assert trial0["posterior_leaf_share"].sum() == pytest.approx(1.0)
    assert trial0.loc[trial0["candidate_arm"] == 0, "posterior_mass"].item() == 3.0
    assert trial0.loc[trial0["candidate_arm"] == 0, "posterior_mass_rate"].item() == 1.0


def test_build_trial_leaf_table_keeps_zero_event_trials_as_zero_rate():
    event_arm, stops, qvals = _synthetic_inputs()

    trial_leaf, _ = build_trial_leaf_table(event_arm, stops, qvals)
    zero = trial_leaf[trial_leaf["trial_number_by_epoch"] == 1]

    assert not zero["has_replay"].any()
    assert zero["posterior_mass"].eq(0.0).all()
    assert zero["posterior_mass_rate"].eq(0.0).all()
    assert zero["posterior_leaf_share"].isna().all()
    assert zero["posterior_event_share"].isna().all()


def test_event_share_includes_trunk_content_in_denominator():
    event_arm, stops, qvals = _synthetic_inputs()
    trunk = (event_arm["event_id"] == 100) & (event_arm["candidate_arm"] == 6)
    event_arm.loc[trunk, ["arm_count_hard", "arm_count_soft", "arm_posterior_mass"]] = (
        4.0
    )

    trial_leaf, _ = build_trial_leaf_table(event_arm, stops, qvals)
    trial0 = trial_leaf[trial_leaf["trial_number_by_epoch"] == 0]

    assert trial0["posterior_leaf_share"].sum() == pytest.approx(1.0)
    assert trial0["posterior_event_share"].sum() == pytest.approx(14.0 / 18.0)
    assert trial0["posterior_leaf_fraction"].nunique() == 1
    assert trial0["posterior_leaf_fraction"].iloc[0] == pytest.approx(14.0 / 18.0)


def test_build_trial_leaf_table_maps_leaf_uncertainty_and_chosen_arm_exactly():
    event_arm, stops, qvals = _synthetic_inputs()

    trial_leaf, _ = build_trial_leaf_table(event_arm, stops, qvals)
    row = trial_leaf.query(
        "trial_number_by_epoch == 2 and candidate_arm == 4"
    ).iloc[0]

    assert row["reward_belief_variance_candidate_arm_preoutcome"] == pytest.approx(0.0304)
    assert not bool(row["is_chosen"])
    chosen = trial_leaf.query(
        "trial_number_by_epoch == 2 and candidate_arm == 2"
    ).iloc[0]
    assert bool(chosen["is_chosen"])


def test_assign_within_arm_quantiles_preserves_ties_and_marks_constants():
    frame = pd.DataFrame(
        {
            "session_id": ["s"] * 10,
            "candidate_arm": [0] * 5 + [1] * 5,
            "reward_belief_variance_candidate_arm_preoutcome": [1.0, 1.0, 2.0, 3.0, 4.0] + [2.0] * 5,
        }
    )

    got = assign_within_arm_quantiles(frame, n_quantiles=4)

    assert got.loc[:1, "uncertainty_quartile"].nunique() == 1
    assert got.loc[:4, "uncertainty_quartile"].min() == 1
    assert got.loc[:4, "uncertainty_quartile"].max() == 4
    assert got.loc[5:, "uncertainty_quartile"].isna().all()


def test_hierarchical_profile_balances_animals_not_sessions():
    rows = []
    # Animal a contributes two sessions with a high-low contrast of 2.
    for session in ["a1", "a2"]:
        for quartile, value in [(1, 0.0), (4, 2.0)]:
            rows.append(
                {
                    "animal": "a",
                    "session_id": session,
                    "candidate_arm": 0,
                    "uncertainty_quartile": quartile,
                    "reward_belief_variance_candidate_arm_preoutcome": float(quartile),
                    "value": value,
                    "is_chosen": False,
                }
            )
    # Animal b contributes one session with a contrast of 6.
    for quartile, value in [(1, 0.0), (4, 6.0)]:
        rows.append(
            {
                "animal": "b",
                "session_id": "b1",
                "candidate_arm": 0,
                "uncertainty_quartile": quartile,
                "reward_belief_variance_candidate_arm_preoutcome": float(quartile),
                "value": value,
                "is_chosen": False,
            }
        )
    frame = pd.DataFrame(rows)

    profile = hierarchical_quartile_profile(frame, numerator_col="value")
    overall = profile["overall"].set_index("uncertainty_quartile")
    contrasts = paired_quartile_contrasts(profile["candidate_arm"])

    assert overall.loc[4, "value"] == pytest.approx(4.0)
    assert contrasts["animal"].set_index("animal").loc["a", "contrast"] == 2.0
    assert contrasts["animal"].set_index("animal").loc["b", "contrast"] == 6.0
    assert contrasts["overall"]["contrast"].item() == pytest.approx(4.0)


def test_rate_profile_uses_ratio_of_sums_not_mean_of_trial_rates():
    frame = pd.DataFrame(
        {
            "animal": ["a", "a"],
            "session_id": ["s", "s"],
            "candidate_arm": [0, 0],
            "uncertainty_quartile": [1, 1],
            "reward_belief_variance_candidate_arm_preoutcome": [0.1, 0.1],
            "posterior_mass": [10.0, 0.0],
            "immobility_duration_s": [1.0, 9.0],
            "is_chosen": [False, False],
        }
    )

    profile = hierarchical_quartile_profile(
        frame,
        numerator_col="posterior_mass",
        denominator_col="immobility_duration_s",
    )

    assert profile["candidate_arm"]["value"].item() == pytest.approx(1.0)


def test_builder_rejects_duplicate_qvalue_stop_ids():
    event_arm, stops, qvals = _synthetic_inputs()
    qvals = pd.concat([qvals, qvals.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="qvals.*stop_id"):
        build_trial_leaf_table(event_arm, stops, qvals)


def test_builder_rejects_inconsistent_trial_uncertainty_across_stops():
    event_arm, stops, qvals = _synthetic_inputs()
    qvals.loc[qvals["stop_id"] == 11, "reward_belief_variance_port_0"] += 0.1

    with pytest.raises(ValueError, match="constant within session.*trial.*arm"):
        build_trial_leaf_table(event_arm, stops, qvals)


def test_builder_uses_float_responses_without_integer_coercion():
    event_arm, stops, qvals = _synthetic_inputs()
    event_arm.loc[
        (event_arm["event_id"] == 100) & (event_arm["candidate_arm"] == 0),
        "arm_posterior_mass",
    ] = 1.25

    trial_leaf, _ = build_trial_leaf_table(event_arm, stops, qvals)

    got = trial_leaf.query("trial_number_by_epoch == 0 and candidate_arm == 0")[
        "posterior_mass"
    ].item()
    assert np.issubdtype(trial_leaf["posterior_mass"].dtype, np.floating)
    assert got == pytest.approx(3.25)


def test_builder_drops_zero_exposure_stops_and_reports_them():
    event_arm, stops, qvals = _synthetic_inputs()
    stops.loc[stops["stop_id"] == 12, "immobility_duration_s"] = 0.0

    trial_leaf, audit = build_trial_leaf_table(event_arm, stops, qvals)

    assert audit["n_zero_exposure_mapped_stops"] == 1
    assert audit["n_positive_exposure_mapped_stops"] == len(stops) - 1
    assert 1 not in set(trial_leaf["trial_number_by_epoch"])


def test_mu_summary_marks_fit_flags_and_balances_animal_medians():
    mu = pd.DataFrame(
        {
            "animal": ["a", "a", "b", "b"],
            "session_date": [1, 2, 1, 2],
            "epoch": ["e", "e", "e", "e"],
            "coef_reward_belief_variance_candidate_arm_preoutcome_z": [
                0.0,
                2.0,
                4.0,
                100.0,
            ],
            "bse_reward_belief_variance_candidate_arm_preoutcome_z": [0.1] * 4,
            "converged": [True, True, True, True],
            "separated": [False, False, False, True],
        }
    )

    summary = summarize_mu_coefficients(mu)
    sessions = summary["session"].set_index(["animal", "session_date"])
    clean_animals = summary["animal"].query("cohort == 'clean'").set_index("animal")
    overall = summary["overall"].query("cohort == 'clean'").iloc[0]

    assert sessions.loc[("b", 2), "fit_quality"] == "flagged"
    assert clean_animals.loc["a", "coefficient"] == 1.0
    assert clean_animals.loc["b", "coefficient"] == 4.0
    assert overall["coefficient"] == 2.5
    assert overall["relative_weight"] == pytest.approx(np.exp(2.5))


def test_figure_loader_rejects_pre_leaf_support_result_bundle(tmp_path):
    """A stale result CSV bundle cannot outlive the corrected model/table guard."""
    pd.DataFrame({"placeholder": []}).to_pickle(tmp_path / "trial_leaf_table.pkl")
    for name in (
        "quartile_profiles",
        "quartile_contrasts",
        "mu_session_coefficients",
        "mu_animal_coefficients",
        "mu_overall_coefficients",
        "horserace_summary",
    ):
        pd.DataFrame({"placeholder": []}).to_csv(tmp_path / f"{name}.csv", index=False)
    (tmp_path / "cohort_audit.json").write_text(json.dumps({"n_events": 10}))

    with pytest.raises(ValueError, match="predate leaf-support"):
        figure_script._load_results(tmp_path)
