"""Tests for the behavioral-uncertainty/replay-content estimands."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.replay.uncertainty_content import (
    assign_within_arm_quantiles,
    build_trial_leaf_table,
    hierarchical_quartile_profile,
    paired_quartile_contrasts,
    summarize_mu_coefficients,
)


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
                "day": 1,
                "epoch": "e1",
                "stop_idx": stop_idx,
                "stop_id": stop_id,
                "immobility_time": exposure,
            }
        )
        qval_row = {
            "stop_id": stop_id,
            "epoch": "e1",
            "trial_number_by_epoch": trial,
            "arm": chosen_arm,
            "trial_reward": int(trial % 2 == 0),
            "in_poke": bool(stop_idx % 2),
            "is_rewarding": False,
        }
        # Each physical well has four ordered, trial-level uncertainty states.
        for arm in range(6):
            qval_row[f"betadist_var{arm + 1}"] = 0.01 * (trial + 1) + arm * 1e-4
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
                    "day": 1,
                    "epoch": "e1",
                    "event_id": event_id,
                    "stop_id": stop_id,
                    "arm": arm,
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
    assert trial_leaf.groupby("trial_number_by_epoch")["arm"].nunique().eq(6).all()
    assert audit["n_mapped_trials"] == 4

    trial0 = trial_leaf[trial_leaf["trial_number_by_epoch"] == 0]
    assert trial0["n_stops"].eq(2).all()
    assert trial0["immobility_time"].eq(3.0).all()
    assert trial0["n_events"].eq(2).all()
    assert trial0["posterior_leaf_share"].sum() == pytest.approx(1.0)
    assert trial0.loc[trial0["arm"] == 0, "posterior_mass"].item() == 3.0
    assert trial0.loc[trial0["arm"] == 0, "posterior_mass_rate"].item() == 1.0


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
    trunk = (event_arm["event_id"] == 100) & (event_arm["arm"] == 6)
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
    row = trial_leaf.query("trial_number_by_epoch == 2 and arm == 4").iloc[0]

    assert row["arm_uncertainty"] == pytest.approx(0.0304)
    assert not bool(row["is_chosen"])
    chosen = trial_leaf.query("trial_number_by_epoch == 2 and arm == 2").iloc[0]
    assert bool(chosen["is_chosen"])


def test_assign_within_arm_quantiles_preserves_ties_and_marks_constants():
    frame = pd.DataFrame(
        {
            "session_id": ["s"] * 10,
            "arm": [0] * 5 + [1] * 5,
            "arm_uncertainty": [1.0, 1.0, 2.0, 3.0, 4.0] + [2.0] * 5,
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
                    "arm": 0,
                    "uncertainty_quartile": quartile,
                    "arm_uncertainty": float(quartile),
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
                "arm": 0,
                "uncertainty_quartile": quartile,
                "arm_uncertainty": float(quartile),
                "value": value,
                "is_chosen": False,
            }
        )
    frame = pd.DataFrame(rows)

    profile = hierarchical_quartile_profile(frame, numerator_col="value")
    overall = profile["overall"].set_index("uncertainty_quartile")
    contrasts = paired_quartile_contrasts(profile["arm"])

    assert overall.loc[4, "value"] == pytest.approx(4.0)
    assert contrasts["animal"].set_index("animal").loc["a", "contrast"] == 2.0
    assert contrasts["animal"].set_index("animal").loc["b", "contrast"] == 6.0
    assert contrasts["overall"]["contrast"].item() == pytest.approx(4.0)


def test_rate_profile_uses_ratio_of_sums_not_mean_of_trial_rates():
    frame = pd.DataFrame(
        {
            "animal": ["a", "a"],
            "session_id": ["s", "s"],
            "arm": [0, 0],
            "uncertainty_quartile": [1, 1],
            "arm_uncertainty": [0.1, 0.1],
            "posterior_mass": [10.0, 0.0],
            "immobility_time": [1.0, 9.0],
            "is_chosen": [False, False],
        }
    )

    profile = hierarchical_quartile_profile(
        frame,
        numerator_col="posterior_mass",
        denominator_col="immobility_time",
    )

    assert profile["arm"]["value"].item() == pytest.approx(1.0)


def test_builder_rejects_duplicate_qvalue_stop_ids():
    event_arm, stops, qvals = _synthetic_inputs()
    qvals = pd.concat([qvals, qvals.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="qvals.*stop_id"):
        build_trial_leaf_table(event_arm, stops, qvals)


def test_builder_rejects_inconsistent_trial_uncertainty_across_stops():
    event_arm, stops, qvals = _synthetic_inputs()
    qvals.loc[qvals["stop_id"] == 11, "betadist_var1"] += 0.1

    with pytest.raises(ValueError, match="constant within session.*trial.*arm"):
        build_trial_leaf_table(event_arm, stops, qvals)


def test_builder_uses_float_responses_without_integer_coercion():
    event_arm, stops, qvals = _synthetic_inputs()
    event_arm.loc[
        (event_arm["event_id"] == 100) & (event_arm["arm"] == 0),
        "arm_posterior_mass",
    ] = 1.25

    trial_leaf, _ = build_trial_leaf_table(event_arm, stops, qvals)

    got = trial_leaf.query("trial_number_by_epoch == 0 and arm == 0")[
        "posterior_mass"
    ].item()
    assert np.issubdtype(trial_leaf["posterior_mass"].dtype, np.floating)
    assert got == pytest.approx(3.25)


def test_builder_drops_zero_exposure_stops_and_reports_them():
    event_arm, stops, qvals = _synthetic_inputs()
    stops.loc[stops["stop_id"] == 12, "immobility_time"] = 0.0

    trial_leaf, audit = build_trial_leaf_table(event_arm, stops, qvals)

    assert audit["n_zero_exposure_mapped_stops"] == 1
    assert audit["n_positive_exposure_mapped_stops"] == len(stops) - 1
    assert 1 not in set(trial_leaf["trial_number_by_epoch"])


def test_mu_summary_marks_fit_flags_and_balances_animal_medians():
    mu = pd.DataFrame(
        {
            "animal": ["a", "a", "b", "b"],
            "day": [1, 2, 1, 2],
            "epoch": ["e", "e", "e", "e"],
            "coef_arm_uncertainty_z": [0.0, 2.0, 4.0, 100.0],
            "bse_arm_uncertainty_z": [0.1] * 4,
            "converged": [True, True, True, True],
            "separated": [False, False, False, True],
        }
    )

    summary = summarize_mu_coefficients(mu)
    sessions = summary["session"].set_index(["animal", "day"])
    clean_animals = summary["animal"].query("cohort == 'clean'").set_index("animal")
    overall = summary["overall"].query("cohort == 'clean'").iloc[0]

    assert sessions.loc[("b", 2), "fit_quality"] == "flagged"
    assert clean_animals.loc["a", "coefficient"] == 1.0
    assert clean_animals.loc["b", "coefficient"] == 4.0
    assert overall["coefficient"] == 2.5
    assert overall["relative_weight"] == pytest.approx(np.exp(2.5))
