"""The covariate-provenance audit CSV is the load-bearing timing reference.

`.claude/docs/reviews/06_covariate_provenance.csv` is treated as the authoritative
per-covariate timing source that the covariate-timing enforcement reads. Two staleness
items must be corrected in it:

  (a) the ``q_arm`` / ``q_arm_hmm`` / ``p_reward_arm`` rows must record that they share
      *identical* missingness (8064 event-arm rows / 896 events, verified), so no reader
      treats beta ``q_arm`` as a fuller-cohort proxy (the F3 issue);
  (b) each derived ``*_x_current`` / ``swr_x_*`` interaction row must carry a
      ``time_availability`` consistent with the timing class of its *source* covariate,
      so a post-outcome or future-referencing operand cannot hide behind the interaction
      name (e.g. ``update_x_current`` inherits ``hmm_value_update``'s post-outcome flag).
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
_IDENTICAL_MISSINGNESS_ROWS = ("q_arm", "q_arm_hmm", "p_reward_arm")

# Derived interactions whose source covariate is post-outcome (valid only once the
# current trip's reward is known).
_POST_OUTCOME_DERIVED = ("rpe_x_current", "update_x_current")

# Derived interactions whose source references a not-yet-occurred switch/choice — no
# within-stop temporal gate makes them predictive-safe.
_FUTURE_DERIVED = (
    "tss_x_current",
    "postsw_x_current",
    "tsspostsw_x_current",
    "swr_x_next",
)

# Derived interactions built from backward/pre-outcome-safe source covariates.
_SAFE_DERIVED = (
    "swr_x_current",
    "swr_x_prev",
    "swr_x_last_rewarded",
    "swr_x_reward_rate_arm_k10",
    "swr_x_visit_recency_demeaned",
    "swr_x_q_arm",
    "swr_x_q_arm_hmm",
    "relval_x_current",
    "block_x_current",
    "switchrate_x_current",
    "tsc_x_current",
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
