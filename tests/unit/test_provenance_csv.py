"""The covariate-provenance audit CSV is the load-bearing timing reference.

`.claude/docs/reviews/06_covariate_provenance.csv` is treated as the authoritative
per-covariate timing source that the covariate-timing enforcement reads. Two staleness
items must be kept explicit in it:

  (a) candidate-arm belief mean / variance / task-probability rows must record that they share
      *identical* missingness (8064 event-arm rows / 896 events, verified), so no reader
      one value column as a fuller-cohort proxy;
  (b) each derived canonical product / ``swr_x_*`` interaction row must carry a
      ``time_availability`` consistent with the timing class of its *source* covariate,
      so a post-outcome or future-referencing operand cannot hide behind the interaction
      name (for example the current-arm belief-update product inherits its
      post-outcome source's timing).
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

CSV_PATH = (
    Path(__file__).resolve().parents[2]
    / ".claude"
    / "docs"
    / "reviews"
    / "06_covariate_provenance.csv"
)

# Value covariates that share one identical missingness mask (verified in-table).
_IDENTICAL_MISSINGNESS_ROWS = (
    "reward_belief_mean_candidate_arm_preoutcome",
    "reward_belief_variance_candidate_arm_preoutcome",
    "task_reward_probability_candidate_arm",
)

# Derived interactions whose source covariate is post-outcome (valid only once the
# current trip's reward is known).
_POST_OUTCOME_DERIVED = (
    "rpe_postoutcome_z_x_is_current_arm",
    "reward_belief_mean_update_l1_all_ports_postoutcome_z_x_is_current_arm",
)

# Derived interactions whose source references a not-yet-occurred switch/choice — no
# within-stop temporal gate makes them predictive-safe.
_FUTURE_DERIVED = (
    "periswitch_index_trials_z_x_is_current_arm",
    "is_periswitch_post_x_is_current_arm",
    "periswitch_index_trials_z_x_is_periswitch_post_x_is_current_arm",
    "swr_x_is_next_arm",
)

# Derived interactions built from backward/pre-outcome-safe source covariates.
_SAFE_DERIVED = (
    "swr_x_is_current_arm",
    "swr_x_is_prev_arm",
    "swr_x_is_last_rewarded_arm",
    "swr_x_reward_fraction_candidate_arm_last10_visits",
    "swr_x_visit_recency_trials_event_leaf_centered",
    "swr_x_reward_belief_mean_candidate_arm_preoutcome",
    "relative_value_switch_minus_stay_preoutcome_z_x_is_current_arm",
    "trials_since_block_change_z_x_is_current_arm",
    "patch_switch_fraction_last5_completed_trips_z_x_is_current_arm",
    "log1p_time_since_stop_start_s_z_x_is_current_arm",
)


@pytest.fixture(scope="module")
def rows() -> dict[str, dict[str, str]]:
    with CSV_PATH.open() as f:
        reader = csv.DictReader(f)
        return {r["covariate"]: r for r in reader}


def test_csv_well_formed(rows):
    assert rows, "provenance CSV parsed no rows"
    for name, r in rows.items():
        assert r["time_availability"] is not None, name


def test_identical_missingness_noted(rows):
    for name in _IDENTICAL_MISSINGNESS_ROWS:
        assert name in rows, f"{name} row missing from provenance CSV"
        note = rows[name]["notes"].lower()
        assert "identical missingness" in note, (
            f"{name} row must record identical missingness with the other value "
            f"covariates; got notes={rows[name]['notes']!r}"
        )


def test_derived_interaction_timing(rows):
    for name in _POST_OUTCOME_DERIVED:
        assert name in rows, f"{name} row missing"
        ta = rows[name]["time_availability"]
        assert "POST-outcome" in ta, f"{name} must inherit its source's post-outcome flag; got {ta!r}"

    for name in _FUTURE_DERIVED:
        assert name in rows, f"{name} row missing"
        ta = rows[name]["time_availability"]
        assert "FUTURE-referencing" in ta, f"{name} must inherit future-referencing; got {ta!r}"

    for name in _SAFE_DERIVED:
        assert name in rows, f"{name} row missing"
        ta = rows[name]["time_availability"]
        assert "POST-outcome" not in ta and "FUTURE-referencing" not in ta, (
            f"{name} is built from a pre-outcome-safe source and must not be flagged "
            f"post-outcome/future-referencing; got {ta!r}"
        )
