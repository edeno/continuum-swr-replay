"""The behavior summary: how the events relate to the task, SWR against non-SWR.

Every value is read from an output a fit already wrote; nothing is refitted
here. This module names and describes the values. The statistics are computed
in ``analysis.replay`` (``coefficient_statistics``, ``rate_fit_statistics``,
``history_kernel_statistics``, ``stop_statistics``, ``content_statistics`` and
``learning``); here only the size of an input is counted and a statistic an
input already holds is copied. A value read from a fit made on other tables
than the current ones is pending (:func:`fit_staleness`). The inputs
(:class:`BehaviorInputs`), with their runs in ``scripts/qa/run_refits.py``:

- the per-bin rate ladder, Family B (``models_per_bin_rate/all_models.pkl``,
  ``rate.per_bin.at_well``; ``all_models_allstops.pkl``,
  ``rate.per_bin.all_stops``, the sensitivity cohort);
- the per-stop rate ladder, Family A (``models_per_stop_rate/all_models.pkl``,
  ``rate.per_stop.at_well``);
- the content ``is_swr``-contrast ladders' audited aggregates
  (``models_swr_contrast/swr_contrast_content_audited*.pkl``,
  ``content.swr_contrast.*`` and ``content.breadth_contrast.*``);
- the three-estimator comparison (``content_estimator_comparison/``,
  ``content.estimator_comparison``);
- the leaf-only arm ladder (``models_leaf_M0-M0a-M1-M2-M3-M4-Mv-Mu-Mvu/``,
  ``content.arm_ladder.leaf_arms``);
- the event-source suite's choice block
  (``event_source_comparison/non_local_HPC_comparison_fits.pkl``,
  ``behavior.event_source_suite.non_local``);
- the stay/leave fits (``stage6_is_stay_*.pkl``, ``behavior.stay_leave.*``);
- the SWR-power model (``models_per_event_swr_power/swr_power_model.pkl``,
  ``coupling.ripple_power``);
- the duration-aware history refit (``models_per_bin_rate/duration_aware_history_
  pooled.pkl``, ``rate.duration_aware_history.pooled_at_well``),
  the source of the self-history values, and its ridge cross-validated
  cross-history comparison
  (``models_per_bin_rate/cross_history_ridge_cv/cross_history_ridge_cv.pkl``,
  ``rate.cross_history_ridge_cv.at_well``), the source of the cross values;
- the within-arm horse race (``uncertainty_content/horserace_summary.csv``,
  ``content.uncertainty``);
- the canonical event and stop tables, for event lengths and cohort sizes.

Each aggregate is computed with the rule the producing script or its audit
states, named in the value's description. The content-ladder, leaf-ladder and
choice coefficients aggregated for this summary are read through the shipped
per-term identifiability gate
(:func:`~continuum_swr_replay.analysis.replay.contrast_aggregation.gate_term_sessions`).
The per-bin rate values pass the same per-term gate: a reward kernel is read
over the sessions whose three reward-basis terms all pass it, and a single
coefficient over the sessions where that term does; the counts left out, and
the kept fits that did not converge, are reported beside them. The reward
effect is the reconstructed kernel; a single basis coefficient is reported only
as itself (:data:`BASIS_COEFFICIENT_NOTE`). The time courses drop fits flagged
separated, as ``compare_fits`` does. The
self-history kernel is many sessions' separated term, so its values carry a
pending note and a gated reading beside them. "Animals positive" counts the
animals whose own median is above zero.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.events.cohort_selection import (
    select_event_cohort,
    select_stop_cohort,
)
from continuum_swr_replay.analysis.replay import (
    content_statistics,
    history_kernel_statistics,
    rate_fit_statistics,
    stop_statistics,
)
from continuum_swr_replay.analysis.replay.coefficient_statistics import (
    aic_difference_statistics,
    audited_drop_statistics,
    coefficient_statistics,
    fit_rows,
    fits_carrying,
    gated_coefficient_difference_statistics,
    gated_coefficient_statistics,
    gated_exp_median_ratio,
    group_range,
    known_fraction,
    rung_separation_range,
    separated_fraction,
    term_separation_statistics,
)
from continuum_swr_replay.analysis.replay.contrast_aggregation import (
    MAX_UNIDENTIFIED_SHARE,
    MIN_SESSIONS_PER_TERM,
)
from continuum_swr_replay.analysis.replay.fit_provenance import (
    check_fit_provenance,
    fit_provenance_path,
)
from continuum_swr_replay.analysis.replay.learning import (
    cross_session_role_summary,
    fit_status_counts,
    pooled_gee_role_summary,
    smallest_adjusted_p,
)
from continuum_swr_replay.analysis.replay.model_specs import (
    BIN_S,
    PER_BIN_DESIGN,
    PER_BIN_HISTORY_COLUMNS,
    REWARD_HISTORY_COLUMNS,
    REWARD_HISTORY_N_LAGS,
    TMAX,
    TMAX_ARRIVAL,
)
from continuum_swr_replay.paths import (
    CANONICAL_TABLE_BASENAMES,
    ROOT_DIR,
    model_dir,
    replay_content_dir,
    table_path,
)
from continuum_swr_replay.reported.values import (
    Summary,
    Tags,
    Value,
    file_record,
    pend,
    rel,
    value_factory,
)
from continuum_swr_replay.tables.session_keys import n_sessions

TAGS = {
    "rate": Tags(("3.3",)),
    "anchoring": Tags(("3.1",)),
    "estimators": Tags(("3.1", "3.6", "3.7"), ("content_estimators",)),
    "estimators_anchoring": Tags(("3.1",), ("content_estimators",)),
    "leaf_ladder": Tags(("3.7",), ("uncertainty_content",)),
    "choice": Tags(("3.4",)),
    "stay_leave": Tags(("3.5",)),
    "stay_leave_rate": Tags(("3.5", "3.3")),
    "ripple_power": Tags(("3.8",)),
    "content_reward": Tags(("3.2",)),
    "content_value": Tags(("3.6",)),
    "content_uncertainty": Tags(("3.7",)),
    "content_reward_value": Tags(("3.2", "3.6")),
    "content_reward_uncertainty": Tags(("3.2", "3.7")),
}
"""Each group of values: the questions it answers and the figures that show it."""

#: Rate populations: the word their keys use, the fits' ``outcome``, the label.
RATE_POPULATIONS = {
    "swr": ("swr", "detected SWR"),
    "non_local": ("non_local", "non-local (all)"),
    "non_local_swr": ("non_local_swr", "non-local with SWR"),
    "non_local_no_swr": ("non_local_no_swr", "non-local without SWR"),
}
FULL_RATE_RUNG = "M7r_time_histcos_rew"
IMMEDIATE_COEF = f"coef_{REWARD_HISTORY_COLUMNS[0]}"
#: What every value read off one reward basis coefficient says about itself.
BASIS_COEFFICIENT_NOTE = (
    "a basis coefficient, not the effect of a reward at any lag: the three "
    "coefficients are strongly collinear and a 0 to 1 change in one feature with "
    "the others held fixed is no reward history that occurs; the effect of a reward "
    "at lag 1 is the reconstructed kernel (rate_kernel_lag_one_*)"
)
STOP_TIME_RUNG = "M5_time"
ARRIVAL_TIME_RUNG = "M5a_time_arrival"

#: Content contrast specs: key word -> (audited file suffix, model, population, term).
#: The suffix picks the bundle: "" fraction / all stratum, "_rewarded" and
#: "_omitted" the outcome strata, "_dm" the Dirichlet-multinomial, "_dm_len" the
#: Dirichlet-multinomial with the length-dependent concentration.
_POPS = {"all": "all", "swr": "SWR", "non_swr": "nonSWR"}
_SPLIT_RUNGS = {
    "base_current": ("C2_base", "is_current_arm"),
    "base_current_no_controls": ("C1_current", "is_current_arm"),
    "armreward": ("C3_armreward", "reward_fraction_candidate_arm_last10_visits"),
    "armvalue": ("C4_armvalue", "reward_belief_mean_candidate_arm_preoutcome"),
    "rpe_current": ("C6_rpe", "rpe_postoutcome_z_x_is_current_arm"),
    "relative_value_current": (
        "C7_value",
        "relative_value_switch_minus_stay_preoutcome_z_x_is_current_arm",
    ),
    "entropy_current": (
        "C8_uncert",
        "reward_belief_differential_entropy_leaf_mean_preoutcome_z_x_is_current_arm",
    ),
    "update_current": (
        "C9_update",
        "reward_belief_mean_update_l1_all_ports_postoutcome_z_x_is_current_arm",
    ),
    "periswitch_current": (
        "C10_periswitch",
        "periswitch_index_trials_z_x_is_current_arm",
    ),
    "block_current": ("C11_block", "trials_since_block_change_z_x_is_current_arm"),
    "switchrate_current": (
        "C12_switchrate",
        "patch_switch_fraction_last5_completed_trips_z_x_is_current_arm",
    ),
    "armuncert": ("C13_armuncert", "reward_belief_variance_candidate_arm_preoutcome_z"),
}
_POOLED_RUNGS = {
    "swr_current_direct": ("CS1_current", "swr_x_is_current_arm"),
    "swr_current_total": ("CS1nt_current", "swr_x_is_current_arm"),
    "swr_prev_direct": ("CS2_prev", "swr_x_is_prev_arm"),
    "swr_armreward_direct": (
        "CS4_armreward",
        "swr_x_reward_fraction_candidate_arm_last10_visits",
    ),
    "swr_armreward_total": (
        "CS4nt_armreward",
        "swr_x_reward_fraction_candidate_arm_last10_visits",
    ),
    "swr_armvalue_direct": (
        "CS5_armvalue",
        "swr_x_reward_belief_mean_candidate_arm_preoutcome",
    ),
    "swr_armvalue_total": (
        "CS5nt_armvalue",
        "swr_x_reward_belief_mean_candidate_arm_preoutcome",
    ),
    "swr_prevgoal": (
        "CS5b_valuevsprevgoal",
        "swr_x_reward_belief_mean_candidate_arm_preoutcome",
    ),
    "swr_armuncert_direct": (
        "CS6_armuncert",
        "swr_x_reward_belief_variance_candidate_arm_preoutcome_z",
    ),
    "swr_armuncert_total": (
        "CS6nt_armuncert",
        "swr_x_reward_belief_variance_candidate_arm_preoutcome_z",
    ),
    "patch_current": ("CP0_base", "is_current_patch"),
    "swr_patch_current_direct": ("CPS1_current", "swr_x_is_current_patch"),
    "swr_patch_current_total": ("CPS1nt_current", "swr_x_is_current_patch"),
}


def _content_specs() -> dict[str, tuple[str, str, str, str]]:
    specs: dict[str, tuple[str, str, str, str]] = {}
    for word, (model, term) in _SPLIT_RUNGS.items():
        for pop_word, pop in _POPS.items():
            specs[f"{word}_{pop_word}"] = ("", model, pop, term)
    for word, (model, term) in _POOLED_RUNGS.items():
        specs[word] = ("", model, "all", term)
    for word in ("swr_current_direct", "swr_armvalue_direct", "swr_armvalue_total"):
        model, term = _POOLED_RUNGS[word]
        specs[f"{word}_rewarded"] = ("_rewarded", model, "all", term)
    model, term = _POOLED_RUNGS["swr_current_direct"]
    specs["swr_current_direct_omitted"] = ("_omitted", model, "all", term)
    for word in (
        "swr_armvalue_direct",
        "swr_armvalue_total",
        "swr_prevgoal",
        "swr_armuncert_direct",
        "swr_armuncert_total",
        "swr_armreward_direct",
    ):
        model, term = _POOLED_RUNGS[word]
        specs[f"dm_{word}"] = ("_dm", model, "all", term)
    specs["dm_breadth"] = ("_dm", "C2_base", "all", "log_phi_is_swr")
    specs["dm_breadth_length_adjusted"] = (
        "_dm_len",
        "C2_base",
        "all",
        "log_phi_is_swr",
    )
    specs["dm_breadth_length_term"] = (
        "_dm_len",
        "C2_base",
        "all",
        "log_phi_log_n_nonlocal_bins_z",
    )
    return specs


CONTENT_SPECS = _content_specs()
"""Every content-contrast aggregate the behavior summary reports."""

#: Estimator-comparison terms: key word -> (rung, term).
ESTIMATOR_TERMS = {
    "current": ("C2_base", "is_current_arm"),
    "swr_current": ("CS1_current", "swr_x_is_current_arm"),
    "swr_armreward": (
        "CS4_armreward",
        "swr_x_reward_fraction_candidate_arm_last10_visits",
    ),
    "swr_armvalue": (
        "CS5_armvalue",
        "swr_x_reward_belief_mean_candidate_arm_preoutcome",
    ),
    "armvalue": ("CS5_armvalue", "reward_belief_mean_candidate_arm_preoutcome"),
}
ESTIMATORS = {
    "count": "conditional_poisson_count",
    "fraction": "conditional_poisson_fraction",
    "dm": "dirichlet_multinomial",
}

#: Leaf-ladder steps: key word -> (parent, child).
LEAF_STEPS = {"uncertainty": ("M4", "Mu"), "value": ("M4", "Mv"), "both": ("Mu", "Mvu")}

#: Stay/leave content strata, by the file suffix their bundles carry.
STAY_STRATA = {"all": "", "swr": "_swr", "non_swr": "_non_swr", "mua": "_mua"}

#: The SWR-power model's covariates the behavior summary reports: key word -> column.
POWER_COVARIATES = {
    "reward": "known_reward_outcome_lag_1",
    "rpe": "rpe_postoutcome",
    "exploration": "patch_switch_fraction_last5_completed_trips",
}


@dataclass(frozen=True)
class BehaviorInputs:
    """The files the behavior summary reads."""

    per_bin: Path
    per_bin_all_stops: Path
    per_bin_in_transit: Path
    rate_contrast: Path
    per_bin_audit: Path
    per_stop_audit: Path
    per_stop: Path
    content_contrast_dir: Path
    estimator_summary: Path
    estimator_bootstrap: Path
    leaf_dir: Path
    choice_fits: Path
    stay_leave_dir: Path
    swr_power: Path
    event_table: Path
    stop_table: Path
    onset_table: Path
    event_arm_table: Path
    duration_aware_pooled: Path
    cross_ridge_cv: Path
    horserace_summary: Path


def default_inputs() -> BehaviorInputs:
    rc = replay_content_dir()
    estimators = rc / "content_estimator_comparison"
    return BehaviorInputs(
        per_bin=model_dir("per_bin_rate") / "all_models.pkl",
        per_bin_all_stops=model_dir("per_bin_rate") / "all_models_allstops.pkl",
        per_bin_in_transit=model_dir("per_bin_rate") / "all_models_intransit.pkl",
        rate_contrast=model_dir("per_bin_rate") / "swr_contrast_rate.pkl",
        per_bin_audit=model_dir("per_bin_rate") / "per_bin_rate_cohort_audit.csv",
        per_stop_audit=model_dir("per_stop_rate") / "all_models_cohort_audit.csv",
        per_stop=model_dir("per_stop_rate") / "all_models.pkl",
        content_contrast_dir=rc / "models_swr_contrast",
        estimator_summary=estimators / "content_estimator_comparison_summary.csv",
        estimator_bootstrap=estimators / "content_estimator_bootstrap.csv",
        leaf_dir=rc / "models_leaf_M0-M0a-M1-M2-M3-M4-Mv-Mu-Mvu",
        choice_fits=rc
        / "event_source_comparison"
        / "non_local_HPC_comparison_fits.pkl",
        stay_leave_dir=rc,
        swr_power=model_dir("per_event_swr_power") / "swr_power_model.pkl",
        event_table=table_path("event_table", rc),
        stop_table=table_path("stop_table", rc),
        onset_table=table_path("event_onset_table", rc),
        event_arm_table=table_path("event_arm_table", rc),
        duration_aware_pooled=model_dir("per_bin_rate")
        / "duration_aware_history_pooled.pkl",
        cross_ridge_cv=model_dir("per_bin_rate")
        / "cross_history_ridge_cv"
        / "cross_history_ridge_cv.pkl",
        horserace_summary=rc / "uncertainty_content" / "horserace_summary.csv",
    )


def _gate_note(statistics: Mapping[str, Any], term: str) -> str | None:
    """Why the values read from a gated statistic are pending, or None.

    The per-term session gate refuses a term when it keeps too few sessions or
    leaves out too large a share of them
    (:attr:`~continuum_swr_replay.analysis.replay.contrast_aggregation.TermSessionGate.refusal`),
    and the audited aggregates publish no row for such a term. The values read
    here over the kept sessions are then kept and marked pending.
    """
    if not statistics["gate_refuses"]:
        return None
    return (
        f"the per-term session gate refuses {term}: it keeps fewer than "
        f"{MIN_SESSIONS_PER_TERM} sessions, or leaves out more than "
        f"{100 * MAX_UNIDENTIFIED_SHARE:.0f} % of the sessions with a fit"
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


# --- rate, per time bin (Family B) --------------------------------------------


def per_bin_values(
    fits: pd.DataFrame, all_stops: pd.DataFrame, source: str
) -> list[Value]:
    """Question 3.3: the per-bin conditional-intensity ladder (Family B).

    Read on the well-anchored cohort; ``allstops`` keys read the all-stops
    sensitivity fit with the same rules.
    """
    make = value_factory(per_bin_values, source, TAGS["rate"])
    values: list[Value] = []
    cohort = rate_fit_statistics.frozen_cohort_statistics(
        fit_rows(fits, model="M0_const")
    )
    ladder = rate_fit_statistics.ladder_fit_statistics(fits)
    late_lags = f"{rate_fit_statistics.FIRST_LATE_REWARD_LAG}-{REWARD_HISTORY_N_LAGS}"
    values += [
        make(
            "rate_sessions",
            cohort["n_sessions"],
            "count",
            "count",
            "sessions in the per-bin rate fit",
        ),
        make(
            "rate_stops",
            cohort["n_stops"],
            "count",
            "count",
            "stops in the frozen common cohort (M0_const, every population)",
        ),
        make(
            "rate_bins",
            cohort["n_bins_non_local"],
            "count",
            "count",
            f"{1000 * BIN_S:g} ms bins at risk in the frozen common cohort, non-local "
            "populations (the bins inside a non-local event are not at risk)",
        ),
        make(
            "rate_bins_swr",
            cohort["n_bins_swr"],
            "count",
            "count",
            f"{1000 * BIN_S:g} ms bins at risk in the frozen common cohort, detected "
            "SWR (the bins inside a Kay SWR are not at risk)",
        ),
    ]
    for word, (outcome, label) in RATE_POPULATIONS.items():
        null = fit_rows(fits, model="M0_const", outcome=outcome)
        full = fit_rows(fits, model=FULL_RATE_RUNG, outcome=outcome)
        baseline = rate_fit_statistics.baseline_rate_statistics(null)
        kernel = rate_fit_statistics.reward_kernel_statistics(full)
        immediate = gated_coefficient_statistics(full, REWARD_HISTORY_COLUMNS[0])
        immediate_note = _gate_note(immediate, REWARD_HISTORY_COLUMNS[0])
        for stats in (kernel, immediate):
            _require(stats["n_animals"] == 5, f"{stats['n_animals']} animals, not 5")
        marginal = aic_difference_statistics(
            fit_rows(fits, model="M1_rew", outcome=outcome), null
        )
        conditional = aic_difference_statistics(
            full, fit_rows(fits, model="M6r_time_histcos", outcome=outcome)
        )
        values += [
            make(
                f"rate_baseline_{word}",
                baseline["median"],
                "quantity",
                "dec2",
                f"{label}: median over sessions of exp(intercept) of M0_const, events per at-risk second (no event of its detector ongoing)",
                "events/s",
            ),
            make(
                f"rate_full_sessions_{word}",
                ladder.loc[(FULL_RATE_RUNG, outcome), "n_fitted"],
                "count",
                "count",
                f"{label}: sessions with a fit of {FULL_RATE_RUNG}",
            ),
            make(
                f"rate_time_sessions_{word}",
                ladder.loc[(STOP_TIME_RUNG, outcome), "n_fitted"],
                "count",
                "count",
                f"{label}: sessions with a fit of {STOP_TIME_RUNG}",
            ),
            make(
                f"rate_kernel_lag_one_{word}",
                kernel["rate_ratio_lag_1"],
                "quantity",
                "dec2",
                f"{label}: reward kernel at lag 1, reconstructed per session from the three basis "
                f"coefficients of {FULL_RATE_RUNG}, median over the sessions whose three "
                "reward terms pass the per-term separation gate (session-pooled)",
                "rate ratio",
            ),
            make(
                f"rate_kernel_lag_one_animal_{word}",
                kernel["lag_one_animal_rate_ratio"],
                "quantity",
                "dec2",
                f"{label}: the same lag-1 kernel, median over animals of each animal's median "
                "(animal-balanced)",
                "rate ratio",
            ),
            make(
                f"rate_kernel_lag_one_animals_positive_{word}",
                kernel["n_animals_lag_one_positive"],
                "count",
                "count",
                f"{label}: animals whose median lag-1 kernel is above 1",
            ),
            make(
                f"rate_kernel_lag_one_positive_fraction_{word}",
                kernel["lag_one_positive_fraction"],
                "fraction",
                "pct",
                f"{label}: share of sessions whose lag-1 kernel is above 1",
            ),
            make(
                f"rate_kernel_lag_one_p_{word}",
                kernel["lag_one_signed_rank_p"],
                "quantity",
                "pval",
                f"{label}: signed-rank p of the log lag-1 kernel against 0 across sessions, "
                "treating sessions as independent",
            ),
            make(
                f"rate_kernel_sessions_{word}",
                kernel["n_sessions"],
                "count",
                "count",
                f"{label}: sessions whose {FULL_RATE_RUNG} reward terms all pass the "
                "per-term separation gate (the kernel values' sessions)",
            ),
            make(
                f"rate_kernel_dropped_{word}",
                kernel["n_dropped"],
                "count",
                "count",
                f"{label}: fitted sessions the gate left out because a reward term was "
                "separated or ran away",
            ),
            make(
                f"rate_kernel_nonconverged_{word}",
                kernel["n_nonconverged"],
                "count",
                "count",
                f"{label}: of the kernel values' sessions, fits the optimizer flagged "
                "as not converged (kept: convergence is not part of the gate)",
            ),
            make(
                f"rate_kernel_late_min_{word}",
                kernel["late_rate_ratio_min"],
                "quantity",
                "dec2",
                f"{label}: lowest median reconstructed kernel over lags {late_lags}",
                "rate ratio",
            ),
            make(
                f"rate_kernel_late_max_{word}",
                kernel["late_rate_ratio_max"],
                "quantity",
                "dec2",
                f"{label}: highest median reconstructed kernel over lags {late_lags}",
                "rate ratio",
            ),
            make(
                f"rate_kernel_lag_ten_{word}",
                kernel[f"rate_ratio_lag_{REWARD_HISTORY_N_LAGS}"],
                "quantity",
                "dec2",
                f"{label}: median reconstructed kernel at lag {REWARD_HISTORY_N_LAGS}",
                "rate ratio",
            ),
            make(
                f"rate_immediate_session_{word}",
                immediate["exp_median"],
                "quantity",
                "sig2",
                f"{label}: exp of the across-session median of the first reward basis "
                f"coefficient {IMMEDIATE_COEF} in {FULL_RATE_RUNG}; {BASIS_COEFFICIENT_NOTE}",
                "rate ratio",
                pending=immediate_note,
            ),
            make(
                f"rate_immediate_session_positive_fraction_{word}",
                immediate["positive_fraction"],
                "fraction",
                "pct",
                f"{label}: share of sessions whose first reward basis coefficient is "
                "positive (a basis coefficient, not an effect size)",
                pending=immediate_note,
            ),
            make(
                f"rate_immediate_session_animals_{word}",
                immediate["n_animals_same_sign"],
                "count",
                "count",
                f"{label}: animals whose median first reward basis coefficient has the "
                "sign of the across-session median",
                pending=immediate_note,
            ),
            make(
                f"rate_immediate_session_p_{word}",
                immediate["signed_rank_p"],
                "quantity",
                "pval",
                f"{label}: signed-rank p of the first reward basis coefficient against 0 "
                "across sessions (a basis coefficient, not an effect size)",
                pending=immediate_note,
            ),
            make(
                f"rate_immediate_animal_{word}",
                immediate["exp_animal_median"],
                "quantity",
                "sig2",
                f"{label}: exp of the animal-balanced median first reward basis "
                f"coefficient (rr_immediate); {BASIS_COEFFICIENT_NOTE}",
                "rate ratio",
                pending=immediate_note,
            ),
            make(
                f"rate_immediate_animals_positive_{word}",
                immediate["n_animals_positive"],
                "count",
                "count",
                f"{label}: animals whose median first reward basis coefficient is positive",
                pending=immediate_note,
            ),
            make(
                f"rate_marginal_reward_delta_aic_{word}",
                marginal["median"],
                "quantity",
                "sig2s",
                f"{label}: median per-session AIC(M1_rew) - AIC(M0_const); negative = child better",
                "AIC",
            ),
            make(
                f"rate_conditional_reward_delta_aic_{word}",
                conditional["median"],
                "quantity",
                "sig2s",
                f"{label}: median per-session AIC({FULL_RATE_RUNG}) - AIC(M6r_time_histcos); "
                "negative = child better",
                "AIC",
            ),
        ]
        window = rate_fit_statistics.arrival_window_statistics(
            fits.attrs["arrival_clock_cohort"][outcome]
        )
        window_all = rate_fit_statistics.arrival_window_statistics(
            all_stops.attrs["arrival_clock_cohort"][outcome]
        )
        values += [
            make(
                f"rate_arrival_event_share_{word}",
                window["event_share"],
                "fraction",
                "dec2",
                f"{label}: share of its event bins the arrival window "
                f"[0, {TMAX_ARRIVAL:g}] s keeps",
            ),
            make(
                f"rate_allstops_arrival_event_share_{word}",
                window_all["event_share"],
                "fraction",
                "dec2",
                f"{label}: the same share on the all-stops fit",
            ),
            make(
                f"rate_arrival_separated_fraction_{word}",
                ladder.loc[(ARRIVAL_TIME_RUNG, outcome), "separated_fraction"],
                "fraction",
                "pct",
                f"{label}: share of {ARRIVAL_TIME_RUNG} fits flagged separated",
            ),
        ]
        for prefix, frame in (("rate", fits), ("rate_allstops", all_stops)):
            shape = rate_fit_statistics.clock_shape_statistics(
                fit_rows(frame, model=STOP_TIME_RUNG, outcome=outcome),
                fit_rows(frame, model=ARRIVAL_TIME_RUNG, outcome=outcome),
            )
            cohort_label = "" if prefix == "rate" else ", all-stops fit"
            for clock, rung in (
                ("time", STOP_TIME_RUNG),
                ("arrival", ARRIVAL_TIME_RUNG),
            ):
                edge = shape[f"{clock}_peak_edge"]
                interval = (
                    f"[{shape[f'{clock}_interval_start_s']:g}, "
                    f"{shape[f'{clock}_interval_end_s']:g}] s"
                )
                values += [
                    make(
                        f"{prefix}_{clock}_peak_s_{word}",
                        shape[f"{clock}_peak_s"],
                        "quantity",
                        "dec2",
                        f"{label}: argmax of the median fitted {rung} time course, read on "
                        f"{interval}{cohort_label}"
                        + (
                            "; it sits on the interval's edge, so it bounds the peak rather "
                            "than locating it"
                            if edge
                            else ""
                        ),
                        "s",
                    ),
                    make(
                        f"{prefix}_{clock}_peak_edge_{word}",
                        edge,
                        "count",
                        "count",
                        f"{label}: 1 if that argmax sits on the edge of {interval}{cohort_label}",
                    ),
                    make(
                        f"{prefix}_{clock}_peak_sessions_{word}",
                        shape[f"{clock}_peak_sessions"],
                        "count",
                        "count",
                        f"{label}: unflagged session curves behind that median{cohort_label}",
                    ),
                ]
            values.append(
                make(
                    f"{prefix}_clock_ks_{word}",
                    shape["clock_ks"],
                    "quantity",
                    "dec2",
                    f"{label}: KS distance between the two clocks' median time courses, each "
                    "normalized over its own supported interval and read on relative "
                    f"position{cohort_label}",
                )
            )
    full_swr = fit_rows(fits, model=FULL_RATE_RUNG, outcome="non_local_swr")
    full_no_swr = fit_rows(fits, model=FULL_RATE_RUNG, outcome="non_local_no_swr")
    no_swr_all = fit_rows(all_stops, model=FULL_RATE_RUNG, outcome="non_local_no_swr")
    paired = rate_fit_statistics.paired_lag_one_kernel_statistics(full_swr, full_no_swr)
    paired_all = rate_fit_statistics.paired_lag_one_kernel_statistics(
        fit_rows(all_stops, model=FULL_RATE_RUNG, outcome="non_local_swr"), no_swr_all
    )
    for stats in (paired, paired_all):
        _require(
            stats["n_animals"] == 5,
            f"paired test spans {stats['n_animals']} animals, not 5",
        )
    immediate_all = gated_coefficient_statistics(no_swr_all, REWARD_HISTORY_COLUMNS[0])
    immediate_all_note = _gate_note(immediate_all, REWARD_HISTORY_COLUMNS[0])
    kernel_all = rate_fit_statistics.reward_kernel_statistics(no_swr_all)
    _require(kernel_all["n_animals"] == 5, f"{kernel_all['n_animals']} animals, not 5")
    paired_what = (
        "the reconstructed lag-1 reward kernel, SWR-coupled minus non-SWR non-local "
        "events, within session, over the sessions in which both streams' reward "
        "terms pass the per-term separation gate"
    )
    values += [
        make(
            "rate_paired_sessions",
            paired["n_sessions"],
            "count",
            "count",
            f"sessions in the paired comparison of {paired_what}",
        ),
        make(
            "rate_paired_positive_fraction",
            paired["positive_fraction"],
            "fraction",
            "pct",
            f"share of paired sessions in which {paired_what} is positive",
        ),
        make(
            "rate_paired_animals_positive",
            paired["n_animals_positive"],
            "count",
            "count",
            "animals whose median paired difference is positive",
        ),
        make(
            "rate_paired_p",
            paired["signed_rank_p"],
            "quantity",
            "pval",
            "signed-rank p of the paired difference, treating sessions as independent",
        ),
        make(
            "rate_allstops_paired_sessions",
            paired_all["n_sessions"],
            "count",
            "count",
            "the same, all-stops fit: paired sessions",
        ),
        make(
            "rate_allstops_paired_positive_fraction",
            paired_all["positive_fraction"],
            "fraction",
            "pct",
            "the same, all-stops fit: share positive",
        ),
        make(
            "rate_allstops_paired_p",
            paired_all["signed_rank_p"],
            "quantity",
            "pval",
            "the same, all-stops fit: signed-rank p",
        ),
        make(
            "rate_immediate_ratio_swr_to_no_swr",
            gated_exp_median_ratio(full_swr, full_no_swr, REWARD_HISTORY_COLUMNS[0]),
            "quantity",
            "sig2",
            "rate_immediate_session_non_local_swr divided by "
            f"rate_immediate_session_non_local_no_swr; {BASIS_COEFFICIENT_NOTE}",
            "ratio",
        ),
        make(
            "rate_allstops_immediate_session_no_swr",
            immediate_all["exp_median"],
            "quantity",
            "sig2",
            "all-stops fit, non-local without SWR: exp of the across-session median first "
            f"reward basis coefficient; {BASIS_COEFFICIENT_NOTE}",
            "rate ratio",
            pending=immediate_all_note,
        ),
        make(
            "rate_allstops_immediate_session_p_no_swr",
            immediate_all["signed_rank_p"],
            "quantity",
            "pval",
            "its signed-rank p (a basis coefficient, not an effect size)",
            pending=immediate_all_note,
        ),
        make(
            "rate_allstops_immediate_session_animals_no_swr",
            immediate_all["n_animals_same_sign"],
            "count",
            "count",
            "its animals with the median's sign",
            pending=immediate_all_note,
        ),
        make(
            "rate_allstops_kernel_lag_one_no_swr",
            kernel_all["rate_ratio_lag_1"],
            "quantity",
            "dec2",
            "all-stops fit, non-local without SWR: median reconstructed kernel at lag 1 "
            "over the sessions passing the reward gate",
            "rate ratio",
        ),
        make(
            "rate_allstops_kernel_lag_one_p_no_swr",
            kernel_all["lag_one_signed_rank_p"],
            "quantity",
            "pval",
            "all-stops fit, non-local without SWR: signed-rank p of the log lag-1 kernel "
            "against 0 across sessions",
        ),
        make(
            "rate_allstops_kernel_lag_one_animals_positive_no_swr",
            kernel_all["n_animals_lag_one_positive"],
            "count",
            "count",
            "all-stops fit, non-local without SWR: animals whose median lag-1 kernel is "
            "above 1",
        ),
    ]
    for word, (outcome, label) in RATE_POPULATIONS.items():
        stats = gated_coefficient_statistics(
            fit_rows(all_stops, model=FULL_RATE_RUNG, outcome=outcome),
            REWARD_HISTORY_COLUMNS[0],
        )
        _require(stats["n_animals"] == 5, f"{stats['n_animals']} animals, not 5")
        values += pend(
            [
                make(
                    f"rate_allstops_immediate_animal_{word}",
                    stats["exp_animal_median"],
                    "quantity",
                    "sig2",
                    f"all-stops fit, {label}: rr_immediate (animal-balanced); "
                    f"{BASIS_COEFFICIENT_NOTE}",
                    "rate ratio",
                ),
                make(
                    f"rate_allstops_immediate_animals_positive_{word}",
                    stats["n_animals_positive"],
                    "count",
                    "count",
                    f"all-stops fit, {label}: animals positive",
                ),
            ],
            _gate_note(stats, REWARD_HISTORY_COLUMNS[0]),
        )
    for word, model in (
        ("const", "M0_const"),
        ("time", STOP_TIME_RUNG),
        ("full", FULL_RATE_RUNG),
        ("reward_by_time", "M9r_rew_x_time"),
    ):
        values += [
            make(
                f"rate_ladder_aic_{word}",
                ladder.loc[(model, "non_local"), "aic_median"],
                "quantity",
                "int",
                f"non-local (all): median AIC of {model}",
                "AIC",
            ),
            make(
                f"rate_ladder_cv_bits_{word}",
                ladder.loc[(model, "non_local"), "cv_bits_median"],
                "quantity",
                "dec2",
                f"non-local (all): median held-out bits per event of {model}, over sessions with a finite score",
                "bits/event",
            ),
            make(
                f"rate_ladder_cv_sessions_{word}",
                ladder.loc[(model, "non_local"), "n_cv_finite"],
                "count",
                "count",
                f"non-local (all): sessions with a finite held-out score for {model}",
            ),
            make(
                f"rate_ladder_ks_band_{word}",
                ladder.loc[(model, "non_local"), "ks_band_fraction"],
                "fraction",
                "dec2",
                f"non-local (all): share of {model} fits inside the KS band",
            ),
        ]
    return values + _population_ranges(values, make)


_LAG_WORDS = (
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
)
#: The stop-clock ladder rungs, by the word their keys use.
LADDER = {
    "const": "M0_const",
    "rew": "M1_rew",
    "time": STOP_TIME_RUNG,
    "histcos": "M3r_histcos",
    "time_histcos": "M6r_time_histcos",
    "full": FULL_RATE_RUNG,
    "reward_by_time": "M9r_rew_x_time",
}


def per_bin_detail_values(
    fits: pd.DataFrame, all_stops: pd.DataFrame, in_transit: pd.DataFrame, source: str
) -> list[Value]:
    """Question 3.3: the rest of the per-bin ladder's readout.

    The quartiles of the baseline rate, the single reward-basis coefficients
    over the sessions the per-term gate keeps, the reconstructed kernel at the
    lags between the first and the last, the self-history block's separation,
    the arrival-clock rungs' fit counts, the within-session baseline rate
    ratios, each stop-clock rung's fit quality, and what the arrival clock
    keeps of the all-stops and in-transit fits.
    """
    make = value_factory(per_bin_detail_values, source, TAGS["rate"])
    values: list[Value] = []
    ladder = rate_fit_statistics.ladder_fit_statistics(fits)
    transit_shares: dict[str, float] = {}
    for word, (outcome, label) in RATE_POPULATIONS.items():
        null = fit_rows(fits, model="M0_const", outcome=outcome)
        reward_only = fit_rows(fits, model="M1_rew", outcome=outcome)
        full = fit_rows(fits, model=FULL_RATE_RUNG, outcome=outcome)
        baseline = rate_fit_statistics.baseline_rate_statistics(null)
        raw = gated_coefficient_statistics(reward_only, REWARD_HISTORY_COLUMNS[0])
        raw_note = _gate_note(raw, REWARD_HISTORY_COLUMNS[0])
        kernel = rate_fit_statistics.reward_kernel_statistics(full)
        history = term_separation_statistics(
            fits_carrying(full, PER_BIN_HISTORY_COLUMNS), PER_BIN_HISTORY_COLUMNS
        )
        marginal = aic_difference_statistics(reward_only, null)
        values += [
            make(
                f"rate_baseline_lower_quartile_{word}",
                baseline["lower_quartile"],
                "quantity",
                "dec2",
                f"{label}: 25th percentile over sessions of the M0_const rate",
                "events/s",
            ),
            make(
                f"rate_baseline_upper_quartile_{word}",
                baseline["upper_quartile"],
                "quantity",
                "dec2",
                f"{label}: 75th percentile over sessions of the M0_const rate",
                "events/s",
            ),
            make(
                f"rate_bin_count_{word}",
                baseline["events_per_bin"],
                "quantity",
                "sig2",
                f"{label}: median M0_const rate times the {1000 * BIN_S:g} ms bin, the "
                "mean per-bin count",
                "events/bin",
            ),
            make(
                f"rate_raw_immediate_session_{word}",
                raw["exp_median"],
                "quantity",
                "sig2",
                f"{label}: exp of the across-session median {IMMEDIATE_COEF} of M1_rew "
                f"(uncontrolled); {BASIS_COEFFICIENT_NOTE}",
                "rate ratio",
                pending=raw_note,
            ),
            make(
                f"rate_marginal_reward_child_better_fraction_{word}",
                marginal["negative_fraction"],
                "fraction",
                "pct",
                f"{label}: share of sessions in which M1_rew has the lower AIC than M0_const",
            ),
            make(
                f"rate_hist_sessions_{word}",
                history["n_fits"],
                "count",
                "count",
                f"{label}: sessions whose {FULL_RATE_RUNG} fit carries the self-history block",
            ),
            make(
                f"rate_hist_separated_fraction_{word}",
                history["flagged_fraction"],
                "fraction",
                "pct",
                f"{label}: share of those fits flagging a self-history term "
                "(counted from event ends) separated",
            ),
        ]
        for k, bump in ((1, "one"), (2, "two")):
            agg = gated_coefficient_statistics(full, REWARD_HISTORY_COLUMNS[k])
            values += pend(
                [
                    make(
                        f"rate_bump_{bump}_session_{word}",
                        agg["exp_median"],
                        "quantity",
                        "sig2",
                        f"{label}: exp of the across-session median coef_{REWARD_HISTORY_COLUMNS[k]} "
                        f"of {FULL_RATE_RUNG}; {BASIS_COEFFICIENT_NOTE}",
                        "rate ratio",
                    ),
                    make(
                        f"rate_bump_{bump}_positive_fraction_{word}",
                        agg["positive_fraction"],
                        "fraction",
                        "pct",
                        f"{label}: share of sessions with that coefficient positive",
                    ),
                    make(
                        f"rate_bump_{bump}_animals_{word}",
                        agg["n_animals_same_sign"],
                        "count",
                        "count",
                        f"{label}: animals whose median has the across-session median's sign",
                    ),
                ],
                _gate_note(agg, REWARD_HISTORY_COLUMNS[k]),
            )
        for i, lag in enumerate(_LAG_WORDS[1:-1], start=1):  # lag ten is above
            values.append(
                make(
                    f"rate_kernel_lag_{lag}_{word}",
                    kernel[f"rate_ratio_lag_{i + 1}"],
                    "quantity",
                    "dec2",
                    f"{label}: median reconstructed reward kernel at lag {i + 1}",
                    "rate ratio",
                )
            )
        for rung_word, rung in (
            ("const", "M0a_const_arrival"),
            ("time", ARRIVAL_TIME_RUNG),
            ("full", "M7ra_time_arrival_histcos_rew"),
        ):
            values.append(
                make(
                    f"rate_arrival_{rung_word}_sessions_{word}",
                    ladder.loc[(rung, outcome), "n_finite_aic"],
                    "count",
                    "count",
                    f"{label}: {rung} fits with a finite AIC",
                )
            )
        transit = in_transit.attrs["arrival_clock_cohort"].get(outcome)
        if transit and transit["n_event_bins_frozen"]:
            transit_shares[word] = rate_fit_statistics.arrival_window_statistics(
                transit
            )["event_share"]
            values.append(
                make(
                    f"rate_intransit_arrival_event_share_{word}",
                    transit_shares[word],
                    "fraction",
                    "dec2",
                    f"{label}: share of its event bins the arrival window keeps on the in-transit fit",
                )
            )
    pairs = (
        ("swr_coupled_to_no_swr", "non_local_swr", "non_local_no_swr"),
        ("swr_to_non_local", "swr", "non_local"),
        ("swr_coupled_to_swr", "non_local_swr", "swr"),
        ("no_swr_to_swr", "non_local_no_swr", "swr"),
        ("swr_coupled_to_non_local", "non_local_swr", "non_local"),
    )
    for word, a, b in pairs:
        ratio = rate_fit_statistics.baseline_rate_ratio_statistics(
            fit_rows(fits, model="M0_const", outcome=a),
            fit_rows(fits, model="M0_const", outcome=b),
        )
        values += [
            make(
                f"rate_baseline_ratio_{word}",
                ratio["median_ratio"],
                "quantity",
                "dec2",
                f"median within-session M0_const rate ratio, {a} over {b}",
                "ratio",
            ),
            make(
                f"rate_baseline_ratio_{word}_p",
                ratio["signed_rank_p"],
                "quantity",
                "pval",
                "its paired signed-rank p",
            ),
        ]
    for word, model in LADDER.items():
        values.append(
            make(
                f"rate_ladder_sessions_{word}",
                ladder.loc[(model, "non_local"), "n_finite_aic"],
                "count",
                "count",
                f"non-local (all): sessions with a finite-AIC fit of {model}",
            )
        )
        if word in ("rew", "histcos", "time_histcos"):
            values += [
                make(
                    f"rate_ladder_aic_{word}",
                    ladder.loc[(model, "non_local"), "aic_median"],
                    "quantity",
                    "int",
                    f"non-local (all): median AIC of {model}",
                    "AIC",
                ),
                make(
                    f"rate_ladder_cv_bits_{word}",
                    ladder.loc[(model, "non_local"), "cv_bits_median"],
                    "quantity",
                    "dec2",
                    f"non-local (all): median held-out bits per event of {model}, over sessions with a finite score",
                    "bits/event",
                ),
                make(
                    f"rate_ladder_cv_sessions_{word}",
                    ladder.loc[(model, "non_local"), "n_cv_finite"],
                    "count",
                    "count",
                    f"non-local (all): sessions with a finite held-out score for {model}",
                ),
                make(
                    f"rate_ladder_ks_band_{word}",
                    ladder.loc[(model, "non_local"), "ks_band_fraction"],
                    "fraction",
                    "dec2",
                    f"non-local (all): share of {model} fits inside the KS band",
                ),
            ]
    for word, model in (
        ("time", ARRIVAL_TIME_RUNG),
        ("full", "M7ra_time_arrival_histcos_rew"),
    ):
        for pop in ("non_local", "non_local_swr"):
            values.append(
                make(
                    f"rate_arrival_{word}_cv_nan_fraction_{pop}",
                    ladder.loc[(model, pop), "cv_not_finite_fraction"],
                    "fraction",
                    "pct",
                    f"{RATE_POPULATIONS[pop][1]}: share of {model} fits whose held-out score is not finite",
                )
            )
    marginal_all = aic_difference_statistics(
        fit_rows(all_stops, model="M1_rew", outcome="non_local_no_swr"),
        fit_rows(all_stops, model="M0_const", outcome="non_local_no_swr"),
    )
    cohort = all_stops.attrs["arrival_clock_cohort"]["non_local"]
    window = rate_fit_statistics.arrival_window_statistics(cohort)
    transit_range = group_range(transit_shares)
    values += [
        make(
            "rate_allstops_marginal_reward_delta_aic_non_local_no_swr",
            marginal_all["median"],
            "quantity",
            "sig2s",
            "all-stops fit, non-local without SWR: median per-session AIC(M1_rew) - AIC(M0_const)",
            "AIC",
        ),
        make(
            "rate_allstops_marginal_reward_child_better_fraction_non_local_no_swr",
            marginal_all["negative_fraction"],
            "fraction",
            "pct",
            "all-stops fit, non-local without SWR: share of sessions in which M1_rew has the lower AIC",
        ),
        make(
            "rate_allstops_arrival_bins_dropped",
            cohort["n_bins_dropped_arrival_clock"],
            "count",
            "count",
            "all-stops fit: bins the arrival clock removed from the common cohort",
        ),
        make(
            "rate_allstops_arrival_bins_before",
            window["n_bins_before"],
            "count",
            "count",
            "all-stops fit: bins before that removal",
        ),
        make(
            "rate_allstops_arrival_bins_dropped_fraction",
            window["dropped_bin_fraction"],
            "fraction",
            "pct",
            "all-stops fit: share of bins the arrival clock removed",
        ),
        make(
            "rate_allstops_arrival_stops_dropped",
            cohort["n_stops_dropped_arrival_clock"],
            "count",
            "count",
            "all-stops fit: stops the arrival clock removed",
        ),
        make(
            "rate_allstops_arrival_window_bins",
            cohort["n_bins_arrival_window"],
            "count",
            "count",
            "all-stops fit: frozen-cohort bins inside the arrival window",
        ),
        make(
            "rate_allstops_frozen_bins",
            cohort["n_bins_frozen"],
            "count",
            "count",
            "all-stops fit: bins in the frozen common cohort",
        ),
        make(
            "rate_allstops_arrival_window_fraction",
            window["bin_share"],
            "fraction",
            "pct",
            "all-stops fit: share of frozen-cohort bins inside the arrival window",
        ),
        make(
            "rate_intransit_arrival_event_share_low",
            transit_range["low"],
            "fraction",
            "dec2",
            "lowest over populations of the in-transit fit's arrival-window event share",
        ),
        make(
            "rate_intransit_arrival_event_share_high",
            transit_range["high"],
            "fraction",
            "dec2",
            "highest over populations of the in-transit fit's arrival-window event share",
        ),
    ]
    return values


def stop_table_values(
    stops: pd.DataFrame, onsets: pd.DataFrame, source: str
) -> list[Value]:
    """Stop- and onset-table facts the rate docs quote: dwell by outcome on the
    well-anchored cohort (the analysed immobility: the stop's span less the
    excluded time inside it), survival past 2 s and 4 s (from the stop's
    start, excluded time not taken out), the outcome-boundary shift
    (``compare_fits.outcome_boundary_shift_cohort``), and the share of onsets
    inside the ``TMAX`` truncation."""
    make = value_factory(stop_table_values, source, TAGS["rate"])
    anchored, _ = select_stop_cohort(stops, cohort="at_well")
    dwell = stop_statistics.dwell_statistics(anchored)
    boundary = stop_statistics.outcome_boundary_statistics(stops)
    onset_times = stop_statistics.onset_time_statistics(onsets, TMAX)
    incomplete = stop_statistics.reward_history_incomplete_statistics(
        stops, onsets, "non_local"
    )
    analysed = "analysed immobility (the stop's span less the excluded time inside it)"
    values = [
        make(
            "stop_duration_mean_rewarded",
            dwell["mean_rewarded_s"],
            "quantity",
            "sig2",
            f"well-anchored stops with a rewarded outcome: mean {analysed}",
            "s",
        ),
        make(
            "stop_duration_mean_unrewarded",
            dwell["mean_omitted_s"],
            "quantity",
            "sig2",
            f"well-anchored stops with an omitted outcome: mean {analysed}",
            "s",
        ),
        make(
            "stop_duration_mean_ratio",
            dwell["mean_ratio"],
            "quantity",
            "sig2",
            "ratio of those means",
            "ratio",
        ),
        make(
            "stop_duration_median_rewarded",
            dwell["median_rewarded_s"],
            "quantity",
            "sig2",
            f"well-anchored stops with a rewarded outcome: median {analysed}",
            "s",
        ),
        make(
            "stop_duration_median_unrewarded",
            dwell["median_omitted_s"],
            "quantity",
            "sig2",
            f"well-anchored stops with an omitted outcome: median {analysed}",
            "s",
        ),
        make(
            "stop_duration_median_ratio",
            dwell["median_ratio"],
            "quantity",
            "sig2",
            "ratio of those medians",
            "ratio",
        ),
    ]
    values += [
        make(
            f"stop_survival_{word}_s",
            stop_statistics.still_immobile_fraction(anchored, at_s),
            "fraction",
            "pct",
            f"share of well-anchored stops still immobile {at_s} s after they began "
            f"(stop_end - stop_start at least {at_s} s; excluded time inside the "
            "stop is not taken out)",
        )
        for word, at_s in (("two", 2), ("four", 4))
    ]
    values += [
        make(
            "stops_in_transit",
            stop_statistics.stop_cohort_counts(stops)["n_in_transit"],
            "count",
            "count",
            "in-transit stops in the canonical stop table",
        ),
        make(
            "stops_crossing_outcome",
            boundary["n_crossing"],
            "count",
            "count",
            "stops with stop_start < outcome_known_time_s <= stop_end (all stops)",
        ),
        make(
            "stops_crossing_outcome_fraction",
            boundary["crossing_fraction"],
            "fraction",
            "pct",
            "their share of all stops",
        ),
        make(
            "stops_outcome_shifted",
            boundary["n_shifted"],
            "count",
            "count",
            "crossing stops whose per-bin reward history can be shifted: no later "
            "arrival inside, and a bin at or past the boundary in the stop-level "
            "design (before a fitted population's bins inside its ongoing events "
            "are dropped, so an upper bound for each fitted population)",
        ),
        make(
            "stops_outcome_shifted_fraction",
            boundary["shifted_fraction"],
            "fraction",
            "pct",
            "their share of all stops",
        ),
        make(
            "stops_spanning_arrival",
            boundary["n_spanning_arrival"],
            "count",
            "count",
            "stops containing a later well arrival",
        ),
        make(
            "stops_spanning_arrival_crossing",
            boundary["n_spanning_arrival_crossing"],
            "count",
            "count",
            "of those, stops that also cross their outcome boundary",
        ),
    ]
    for word, event_type in (
        ("non_local", "non_local"),
        ("swr", "SWR"),
        ("mua", "MUA"),
    ):
        values.append(
            make(
                f"onsets_within_tmax_fraction_{word}",
                onset_times.loc[event_type, "within_window_fraction"],
                "fraction",
                "pct",
                f"share of {event_type} onsets within the first {TMAX:g} s of their stop (event_onset_table)",
            )
        )
    values += [
        make(
            "stops_reward_history_incomplete_fraction",
            incomplete["stop_fraction"],
            "fraction",
            "sig2",
            "share of stops flagged reward_history_incomplete (fewer than two known outcomes)",
        ),
        make(
            "onsets_reward_history_incomplete_fraction_non_local",
            incomplete["onset_fraction"],
            "fraction",
            "sig2",
            "share of non-local onsets in those stops",
        ),
    ]
    values.append(
        make(
            "onset_median_s_non_local",
            onset_times.loc["non_local", "median_s"],
            "quantity",
            "sig2",
            "median time of a non-local onset after its stop's speed-threshold crossing",
            "s",
        )
    )
    return values


def trunk_values(arm: pd.DataFrame, source: str) -> list[Value]:
    """Question 3.1: SWR content shifts onto the trunk arms, and which trunk.

    Each animal's mean share over its events, SWR and non-SWR; the value is the
    median over animals, with the animals where the SWR share is higher.
    """
    make = value_factory(trunk_values, source, TAGS["anchoring"])
    events = content_statistics.trunk_shares(arm)
    shares = content_statistics.trunk_share_statistics(events)
    current_patch = content_statistics.current_patch_trunk_fraction(events)
    values: list[Value] = []
    labels = {
        "trunk": "all three trunks",
        "current_patch_trunk": "the trunk of the current arm's patch",
        "other_trunks": "the other two trunks",
    }
    for column, label in labels.items():
        word = {"trunk": "", "current_patch_trunk": "_current_patch"}.get(
            column, "_other_patches"
        )
        values += [
            make(
                f"trunk_share{word}_swr",
                shares.loc[column, "swr_median"],
                "fraction",
                "pct",
                f"SWR events: mean share of an event's content on {label}, median "
                f"over the {shares.loc[column, 'n_animals']} animals (at-well events)",
            ),
            make(
                f"trunk_share{word}_non_swr",
                shares.loc[column, "non_swr_median"],
                "fraction",
                "pct",
                f"non-SWR events: mean share of an event's content on {label}, "
                "median over animals",
            ),
            make(
                f"trunk_share{word}_animals_swr_higher",
                shares.loc[column, "n_animals_swr_higher"],
                "count",
                "count",
                f"animals whose SWR share on {label} is above their non-SWR share",
            ),
        ]
    for flag, word in ((True, "swr"), (False, "non_swr")):
        values.append(
            make(
                f"trunk_current_patch_fraction_{word}",
                current_patch[word],
                "fraction",
                "pct",
                f"{'SWR' if flag else 'non-SWR'} events: the current patch's trunk "
                "as a share of all trunk content (1/3 if the three trunks shared it "
                "evenly), median over animals",
            )
        )
    return values


def current_arm_paired_values(fits: pd.DataFrame, source: str) -> list[Value]:
    """Question 3.1: the within-session SWR - non-SWR difference of the separately
    fitted current-arm preference (``C2_base`` ``is_current_arm``).

    A session enters when both of its split fits keep the term
    (:func:`gate_term_sessions`); each animal is the median of its sessions'
    differences, and the value is the median over animals.
    """
    make = value_factory(current_arm_paired_values, source, TAGS["anchoring"])
    stats = gated_coefficient_difference_statistics(
        fit_rows(fits, model="C2_base", population="SWR"),
        fit_rows(fits, model="C2_base", population="nonSWR"),
        "is_current_arm",
    )
    _require(
        stats["n_sessions"] > 0, "no session keeps is_current_arm in both split fits"
    )
    return pend(
        [
            make(
                "current_arm_paired_difference",
                stats["animal_median"],
                "quantity",
                "sig2s",
                "SWR minus non-SWR is_current_arm (C2_base, fitted separately), the "
                "difference taken within each session, median over the session "
                f"differences per animal, then over the {stats['n_animals']} animals "
                f"({stats['n_sessions']} sessions with both fits kept)",
                "log weight",
            ),
            make(
                "current_arm_paired_difference_animals_positive",
                stats["n_animals_positive"],
                "count",
                "count",
                "animals whose median within-session difference is positive",
            ),
        ],
        _gate_note(stats, "is_current_arm"),
    )


#: Per-population rate values whose range over the four populations is quoted.
_RANGED = (
    "rate_kernel_late_min",
    "rate_time_peak_s",
    "rate_arrival_peak_s",
    "rate_clock_ks",
    "rate_allstops_time_peak_s",
    "rate_allstops_arrival_peak_s",
    "rate_allstops_clock_ks",
    "rate_arrival_separated_fraction",
)


def _population_ranges(values: list[Value], make: Callable[..., Value]) -> list[Value]:
    """The lowest and highest of each :data:`_RANGED` value over the populations.

    A time-course peak on the edge of its supported interval bounds the peak
    rather than locating it, so the peak ranges are over the located peaks only,
    with the number of edge peaks beside them.
    """
    by_key = {v.key: v for v in values}
    ranges = []
    for stem in _RANGED:
        members = {word: by_key[f"{stem}_{word}"] for word in RATE_POPULATIONS}
        on_edge = None
        if stem.endswith("_peak_s"):
            edge_stem = stem.replace("_peak_s", "_peak_edge")
            on_edge = {
                word: by_key[f"{edge_stem}_{word}"].value for word in RATE_POPULATIONS
            }
        span = group_range(
            {word: member.value for word, member in members.items()}, on_edge
        )
        if on_edge is not None:
            ranges.append(
                make(
                    f"{edge_stem}_count",
                    span["n_excluded"],
                    "count",
                    "count",
                    "populations whose "
                    + next(iter(members.values()))
                    .description.split(": ", 1)[1]
                    .split(";")[0]
                    + " sits on the interval's edge",
                )
            )
        for end in ("low", "high"):
            chosen = members[span[f"{end}_group"]]
            ranges.append(
                make(
                    f"{stem}_{end}",
                    chosen.value,
                    chosen.kind,
                    chosen.style,
                    f"{'lowest' if end == 'low' else 'highest'} over the populations"
                    + (
                        " whose peak is located (not on the edge)"
                        if stem.endswith("_peak_s")
                        else ""
                    )
                    + ": "
                    + chosen.description.split(": ", 1)[1].split(";")[0],
                    chosen.unit,
                    pending=chosen.pending,
                )
            )
    return ranges


# --- rate, per stop (Family A) -----------------------------------------------


def rate_contrast_separation_values(fits: pd.DataFrame, source: str) -> list[Value]:
    """How heavily the per-bin ``is_swr``-contrast rate ladder is separated: the
    share of its ``contrast`` fits flagged separated, per cohort, and the share of
    fits carrying ``is_swr:reward_history_rcos_1`` that flag that term."""
    make = value_factory(rate_contrast_separation_values, source, TAGS["rate"])
    contrast = fit_rows(fits, population="contrast")
    term = "is_swr:reward_history_rcos_1"
    carrying = term_separation_statistics(fits_carrying(fits, [term]), [term])
    values = [
        make(
            f"rate_contrast_separated_fraction_{cohort}",
            separated_fraction(fit_rows(contrast, cohort=cohort)),
            "fraction",
            "pct",
            f"share of the is_swr-contrast rate ladder's {cohort} contrast fits flagged "
            "separated",
        )
        for cohort in ("primary", "post_outcome")
    ]
    expanded = rung_separation_range(contrast, term)
    plain = term_separation_statistics(
        fit_rows(contrast, model="S1_rew"), ["reward_history_rcos_1"]
    )
    for column, what in (
        ("fits", "fits"),
        ("separated", "fits flagged separated"),
        ("flagged", f"fits flagging {term}"),
    ):
        for end, pick in (("low", "min"), ("high", "max")):
            values.append(
                make(
                    f"rate_contrast_expanded_{column}_{end}",
                    expanded[f"n_{column}_{pick}"],
                    "count",
                    "count",
                    f"{'fewest' if end == 'low' else 'most'} {what} over the "
                    f"{expanded['n_rungs']} contrast rungs with a fit carrying {term}",
                )
            )
    values += [
        make(
            "rate_contrast_plain_fits",
            plain["n_fits"],
            "count",
            "count",
            "S1_rew contrast fits",
        ),
        make(
            "rate_contrast_plain_flagged",
            plain["n_flagged"],
            "count",
            "count",
            "S1_rew contrast fits flagging reward_history_rcos_1 separated",
        ),
    ]
    values.append(
        make(
            "rate_contrast_term_separated_fraction",
            carrying["flagged_fraction"],
            "fraction",
            "pct",
            f"share of the fits carrying {term} that flag it separated",
        )
    )
    return values


def rate_contrast_audit_values(audited: pd.DataFrame, source: str) -> list[Value]:
    """The is_swr-contrast rate ladder's audited aggregate: how many rows it
    publishes, and how many sessions its key contrast terms lose to the gate."""
    make = value_factory(rate_contrast_audit_values, source, TAGS["rate"])
    drops = audited_drop_statistics(fit_rows(audited, population="contrast"))
    worst = f"{drops['most_dropped_model']} {drops['most_dropped_term']}"
    return [
        make(
            "rate_contrast_audited_rows",
            len(audited),
            "count",
            "count",
            "rows the audited is_swr-contrast rate aggregate publishes",
        ),
        make(
            "rate_contrast_terms_losing_sessions",
            drops["n_terms_losing_sessions"],
            "count",
            "count",
            "key is_swr contrast terms that lose any session to the per-term gate",
        ),
        make(
            "rate_contrast_most_dropped",
            drops["most_dropped"],
            "count",
            "count",
            f"most sessions a key contrast term loses to the gate ({worst})",
        ),
        make(
            "rate_contrast_most_dropped_of",
            drops["most_dropped_of"],
            "count",
            "count",
            "sessions with a coefficient for that term, kept and dropped together "
            f"({drops['most_dropped_model']})",
        ),
    ]


def cohort_audit_values(inputs: BehaviorInputs) -> list[Value]:
    """What the rate families' common-cohort freeze cost, as their runners'
    cohort audits record it (population non-local; every population shares it)."""
    values = []
    for prefix, path, rows in (
        ("rate_audit", inputs.per_bin_audit, {"modeled": "M0_const"}),
        ("per_stop_audit", inputs.per_stop_audit, {"modeled": "R0", "full": "R1"}),
    ):
        audit = pd.read_csv(path)
        audit = audit[audit["population"] == "non_local"].set_index("model")
        make = value_factory(
            cohort_audit_values, f"{rel(path)} (cohort audit)", TAGS["rate"]
        )
        first = audit.loc[rows["modeled"]]
        values += [
            make(
                f"{prefix}_stops_in",
                first["n_in"],
                "count",
                "count",
                f"{path.name}: stops entering the cohort",
            ),
            make(
                f"{prefix}_stops_own_drop",
                first["n_drop_own_predictor_missing"],
                "count",
                "count",
                f"{path.name}: stops dropped for a missing own predictor",
            ),
            make(
                f"{prefix}_stops_common_drop",
                first["n_drop_common_cohort_other"],
                "count",
                "count",
                f"{path.name}: stops dropped by the common-cohort freeze (another rung's covariate, here the arrival clock)",
            ),
        ]
        for word, model in rows.items():
            row = audit.loc[model]
            values += [
                make(
                    f"{prefix}_stops_{word}",
                    row["n_modeled"],
                    "count",
                    "count",
                    f"{path.name}: stops modelled on {model}",
                ),
                make(
                    f"{prefix}_stops_{word}_session_excluded",
                    row["n_excluded_session"],
                    "count",
                    "count",
                    f"{path.name}: stops in sessions {model} gates out",
                ),
            ]
    return values


def per_stop_values(fits: pd.DataFrame, source: str) -> list[Value]:
    """Question 3.3: the per-stop reward step ``R1 - R0`` on the well-anchored
    cohort (``compare_fits.cohort_per_stop_delta_aic_table``): a session counts
    only when both rungs saw the same stops; negative = child better."""
    make = value_factory(per_stop_values, source, TAGS["rate"])
    _require(
        set(fits["stop_cohort"]) == {"at_well"}, "per-stop fit is not the at_well one"
    )
    values = [
        make(
            "per_stop_cv_finite_fraction_full",
            known_fraction(fit_rows(fits, model="R1"), "cv_ll_per_stop_mean"),
            "fraction",
            "pct",
            "share of R1 session-fits (all populations) whose cv_ll_per_stop_mean is "
            "not missing",
        ),
    ]
    for word, (outcome, label) in RATE_POPULATIONS.items():
        step = aic_difference_statistics(
            fit_rows(fits, model="R1", outcome=outcome),
            fit_rows(fits, model="R0", outcome=outcome),
            same=("n_stops",),
        )
        _require(step["n_animals"] == 5, f"{outcome}: {step['n_animals']} animals")
        values += [
            make(
                f"per_stop_delta_aic_animal_{word}",
                step["animal_median"],
                "quantity",
                "sig2s",
                f"{label}: median over animals of the per-animal median AIC(R1) - AIC(R0)",
                "AIC",
            ),
            make(
                f"per_stop_delta_aic_session_{word}",
                step["median"],
                "quantity",
                "sig2s",
                f"{label}: median over sessions of AIC(R1) - AIC(R0)",
                "AIC",
            ),
            make(
                f"per_stop_sessions_{word}",
                step["n_sessions"],
                "count",
                "count",
                f"{label}: sessions with both rungs fit on the same stops",
            ),
            make(
                f"per_stop_child_better_fraction_{word}",
                step["negative_fraction"],
                "fraction",
                "pct",
                f"{label}: share of sessions in which R1 has the lower AIC",
            ),
            make(
                f"per_stop_animals_child_better_{word}",
                step["n_animals_negative"],
                "count",
                "count",
                f"{label}: animals whose median step is negative",
            ),
        ]
    return values


# --- content -------------------------------------------------------------------


#: The tags of each content key word, by its leading words.
_CONTENT_TAGS = (
    ("swr_armreward", TAGS["content_reward_value"]),
    ("swr_armvalue", TAGS["content_value"]),
    ("swr_prevgoal", TAGS["content_value"]),
    ("swr_armuncert", TAGS["content_uncertainty"]),
    ("armreward", TAGS["content_reward"]),
    ("armvalue", TAGS["content_reward_uncertainty"]),
    ("armuncert", TAGS["content_uncertainty"]),
    ("switchrate", TAGS["content_value"]),
)


def _content_tags(word: str) -> Tags:
    """The tags of a content key: by its leading words, else content anchoring."""
    return next(
        (tags for lead, tags in _CONTENT_TAGS if word.startswith(lead)),
        TAGS["anchoring"],
    )


def content_contrast_values(
    audited: dict[str, pd.DataFrame], sources: dict[str, str]
) -> list[Value]:
    """Questions 3.1, 3.2, 3.6 and 3.7: the content ``is_swr``-contrast ladders.

    Read from each bundle's audited aggregate as the fitting script wrote it:
    the animal-balanced median, the sessions kept by the per-term gate, the
    animals with a positive median, and the session signed-rank BH-FDR q.
    """
    values = []
    for word, (suffix, model, pop, term) in CONTENT_SPECS.items():
        frame = audited[suffix]
        rows = frame[
            (frame["model"] == model)
            & (frame["population"] == pop)
            & (frame["term"] == term)
        ]
        _require(
            len(rows) == 1, f"{word}: {len(rows)} audited rows for {model}/{pop}/{term}"
        )
        row = rows.iloc[0]
        make = value_factory(
            content_contrast_values,
            sources[suffix],
            _content_tags(word.removeprefix("dm_")),
        )
        estimator = "Dirichlet-multinomial" if suffix.startswith("_dm") else "fraction"
        stratum = suffix.removeprefix("_dm").removeprefix("_len").strip("_") or "all"
        what = (
            f"{model} {term} ({estimator}, population {pop}, outcome stratum {stratum})"
        )
        if not suffix == "_omitted":
            _require(
                int(row["n_animals"]) == 5, f"{word}: {row['n_animals']} animals, not 5"
            )
        values += [
            make(
                word,
                row["coef_animal_median"],
                "quantity",
                "sig2s",
                f"{what}: animal-balanced median coefficient",
                "log weight",
            ),
            make(
                f"{word}_sessions",
                row["n_sessions"],
                "count",
                "count",
                f"{what}: sessions the per-term gate keeps",
            ),
            make(
                f"{word}_animals_positive",
                row["n_animals_pos"],
                "count",
                "count",
                f"{what}: animals with a positive median",
            ),
        ]
        if word.startswith("dm_breadth"):
            values.append(
                make(
                    f"{word}_positive_fraction",
                    row["pct_positive_session"],
                    "fraction",
                    "pct",
                    f"{what}: share of sessions with a positive coefficient",
                )
            )
        if word.endswith("prevgoal"):
            values.append(
                make(
                    f"{word}_separated",
                    row["n_sessions_flagged_separated"],
                    "count",
                    "count",
                    f"{what}: session fits flagged separated",
                )
            )
        if suffix == "_omitted":
            values.append(
                make(
                    f"{word}_animals",
                    row["n_animals"],
                    "count",
                    "count",
                    f"{what}: animals contributing",
                )
            )
        if np.isfinite(row["session_signrank_fdr_q"]):
            values.append(
                make(
                    f"{word}_q",
                    row["session_signrank_fdr_q"],
                    "quantity",
                    "pval",
                    f"{what}: session signed-rank BH-FDR q (across the bundle's rungs per population)",
                )
            )
    return values


def breadth_derived_values(
    values: dict[str, Value], events: pd.DataFrame, source: str
) -> list[Value]:
    """Question 3.1: what the concentration contrast means, and event length."""
    make = value_factory(breadth_derived_values, source, TAGS["anchoring"])
    contrast = content_statistics.concentration_contrast_statistics(
        values["dm_breadth"].value, values["dm_breadth_length_adjusted"].value
    )
    length = content_statistics.event_length_statistics(events)
    return [
        make(
            "dm_breadth_ratio",
            contrast["ratio"],
            "quantity",
            "sig2",
            "exp of the concentration contrast log_phi_is_swr on C2_base: an SWR event's Dirichlet concentration over a non-SWR event's",
            "ratio",
        ),
        make(
            "dm_breadth_length_adjusted_ratio",
            contrast["length_adjusted_ratio"],
            "quantity",
            "sig2",
            "the same with the length-dependent concentration",
            "ratio",
        ),
        make(
            "dm_breadth_length_share",
            contrast["length_share"],
            "fraction",
            "pct",
            "share of the concentration contrast that the length term absorbs: 1 - adjusted / total",
        ),
        make(
            "event_bins_median_swr",
            length["median_bins_swr"],
            "quantity",
            "int",
            "median non-local bins of SWR-overlapping events (event table)",
            "bins",
        ),
        make(
            "event_bins_median_non_swr",
            length["median_bins_non_swr"],
            "quantity",
            "int",
            "median non-local bins of non-SWR events (event table)",
            "bins",
        ),
        make(
            "event_bins_median",
            length["median_bins"],
            "quantity",
            "int",
            "median non-local bins per event, all events (event table)",
            "bins",
        ),
        make(
            "event_bins_ratio",
            length["swr_to_non_swr_ratio"],
            "quantity",
            "sig2",
            "ratio of those medians",
            "ratio",
        ),
    ]


def estimator_values(
    summary: pd.DataFrame, bootstrap: pd.DataFrame, source: str
) -> list[Value]:
    """Questions 3.1, 3.6 and 3.7: the three-estimator comparison on one shared event
    set. Coefficients are animal-balanced medians as the comparison writes them;
    intervals are its animal bootstrap's 2.5 and 97.5 percentiles."""
    make = value_factory(estimator_values, source, TAGS["estimators"])
    values = []
    for word, (rung, term) in ESTIMATOR_TERMS.items():
        (row,) = summary[
            (summary["model"] == rung) & (summary["term"] == term)
        ].itertuples()
        for est_word, estimator in ESTIMATORS.items():
            what = f"{rung} {term}, {estimator}"
            _require(
                getattr(row, f"n_animals_{est_word}") == 5, f"{what}: not 5 animals"
            )
            values += [
                make(
                    f"est_{word}_{est_word}",
                    getattr(row, f"median_coef_{est_word}"),
                    "quantity",
                    "sig2s",
                    f"{what}: animal-balanced median",
                    "log weight",
                ),
                make(
                    f"est_{word}_{est_word}_animals_positive",
                    getattr(row, f"n_animals_pos_{est_word}"),
                    "count",
                    "count",
                    f"{what}: animals positive",
                ),
                make(
                    f"est_{word}_{est_word}_sessions",
                    getattr(row, f"n_sessions_{est_word}"),
                    "count",
                    "count",
                    f"{what}: sessions",
                ),
            ]
            boot = bootstrap[
                (bootstrap["model"] == rung)
                & (bootstrap["term"] == term)
                & (bootstrap["estimator"] == estimator)
            ]
            _require(len(boot) == 1, f"{what}: {len(boot)} bootstrap rows")
            values += [
                make(
                    f"est_{word}_{est_word}_low",
                    boot["p2_5"].iloc[0],
                    "quantity",
                    "sig2s",
                    f"{what}: animal bootstrap 2.5th percentile",
                    "log weight",
                ),
                make(
                    f"est_{word}_{est_word}_high",
                    boot["p97_5"].iloc[0],
                    "quantity",
                    "sig2s",
                    f"{what}: animal bootstrap 97.5th percentile",
                    "log weight",
                ),
            ]
        if word == "swr_armvalue":
            for est_word in ESTIMATORS:
                values.append(
                    make(
                        f"est_armvalue_rung_delta_aic_{est_word}",
                        getattr(row, f"median_delta_aic_{est_word}"),
                        "quantity",
                        "sig2s",
                        f"CS5_armvalue against C2_base_swr_leaf (the same base and SWR arm baselines, without the two value terms), {ESTIMATORS[est_word]}: animal-balanced median per-session AIC(rung) - AIC(parent); negative = rung better",
                        "AIC",
                    )
                )
    boot = bootstrap[
        (bootstrap["model"] == "C2_base")
        & (bootstrap["term"] == "log_phi_is_swr")
        & (bootstrap["estimator"] == ESTIMATORS["dm"])
    ]
    _require(len(boot) == 1, f"C2_base log_phi_is_swr: {len(boot)} bootstrap rows")
    for end, column in (("low", "p2_5"), ("high", "p97_5")):
        values.append(
            make(
                f"est_breadth_dm_{end}",
                boot[column].iloc[0],
                "quantity",
                "sig2s",
                f"C2_base log_phi_is_swr, {ESTIMATORS['dm']}: animal bootstrap "
                f"{'2.5th' if end == 'low' else '97.5th'} percentile",
                "log concentration",
                tags=TAGS["estimators_anchoring"],
            )
        )
    return values


def leaf_values(leaf_dir: Path, source: str) -> list[Value]:
    """Question 3.7: the leaf-only arm ladder's uncertainty and value rungs."""
    make = value_factory(leaf_values, source, TAGS["leaf_ladder"])
    values = []
    for word, model, term in (
        ("leaf_uncertainty", "Mu", "reward_belief_variance_candidate_arm_preoutcome_z"),
        ("leaf_value", "Mv", "reward_belief_mean_candidate_arm_preoutcome"),
    ):
        stats = gated_coefficient_statistics(
            pd.read_pickle(leaf_dir / f"{model}.pkl"), term
        )
        _require(stats["n_animals"] == 5, f"{term}: {stats['n_animals']} animals")
        what = f"leaf ladder {model} {term}"
        values += pend(
            [
                make(
                    word,
                    stats["median"],
                    "quantity",
                    "sig2s",
                    f"{what}: session median coefficient",
                    "log weight",
                ),
                make(
                    f"{word}_sessions",
                    stats["n_sessions"],
                    "count",
                    "count",
                    f"{what}: sessions kept",
                ),
                make(
                    f"{word}_animals_positive",
                    stats["n_animals_positive"],
                    "count",
                    "count",
                    f"{what}: animals positive",
                ),
                make(
                    f"{word}_se",
                    stats["session_se_median"],
                    "quantity",
                    "sig2",
                    f"{what}: median event-clustered session SE",
                    "log weight",
                ),
            ],
            _gate_note(stats, term),
        )
    comparison = pd.read_pickle(leaf_dir / "model_comparison.pkl")
    for word, (parent, child) in LEAF_STEPS.items():
        (row,) = comparison[
            (comparison["parent"] == parent) & (comparison["child"] == child)
        ].itertuples()
        what = f"leaf ladder {parent} to {child}"
        _require(row.n_animals == 5, f"{what}: not 5 animals")
        values += [
            make(
                f"leaf_step_{word}",
                row.delta_aic_median,
                "quantity",
                "sig2s",
                f"{what}: median per-session AIC(parent) - AIC(child); positive = child better",
                "AIC",
            ),
            make(
                f"leaf_step_{word}_clear",
                row.child_better_aic,
                "count",
                "count",
                f"{what}: sessions in which the child's AIC is lower by more than 2",
            ),
            make(
                f"leaf_step_{word}_sessions",
                row.n_sessions,
                "count",
                "count",
                f"{what}: sessions compared",
            ),
            make(
                f"leaf_step_{word}_animals",
                row.n_animals_child_better,
                "count",
                "count",
                f"{what}: animals whose median step favours the child",
            ),
        ]
    return values


# --- choice and stay/leave --------------------------------------------------------


def choice_values(fits: dict, source: str) -> list[Value]:
    """Question 3.4: replay fraction in the conditional-logit choice block."""
    make = value_factory(choice_values, source, TAGS["choice"])
    values = []
    for word, block, model, term in (
        (
            "choice_non_swr",
            "choice_swr_split",
            "C_full_swr_split",
            "replay_fraction_non_swr",
        ),
        ("choice_swr", "choice_swr_split", "C_full_swr_split", "replay_fraction_swr"),
        ("choice_full", "choice", "C_full", "replay_fraction"),
        ("choice_raw", "choice", "C_replay_only", "replay_fraction"),
    ):
        stats = gated_coefficient_statistics(fits[block][model], term)
        _require(stats["n_animals"] == 5, f"{term}: {stats['n_animals']} animals")
        what = f"{model} {term}"
        values += pend(
            [
                make(
                    word,
                    stats["median"],
                    "quantity",
                    "sig2s",
                    f"{what}: session median coefficient (gated)",
                    "log odds",
                ),
                make(
                    f"{word}_p",
                    stats["signed_rank_p"],
                    "quantity",
                    "pval",
                    f"{what}: signed-rank p across sessions",
                ),
                make(
                    f"{word}_sessions",
                    stats["n_sessions"],
                    "count",
                    "count",
                    f"{what}: sessions kept",
                ),
                make(
                    f"{word}_positive_sessions",
                    stats["n_positive"],
                    "count",
                    "count",
                    f"{what}: sessions with a positive coefficient",
                ),
                make(
                    f"{word}_negative_sessions",
                    stats["n_negative"],
                    "count",
                    "count",
                    f"{what}: sessions with a negative coefficient",
                ),
                make(
                    f"{word}_animals_positive",
                    stats["n_animals_positive"],
                    "count",
                    "count",
                    f"{what}: animals with a positive median",
                ),
                make(
                    f"{word}_significant_fraction",
                    stats["significant_fraction"],
                    "fraction",
                    "pct",
                    f"{what}: share of sessions individually significant",
                ),
            ],
            _gate_note(stats, term),
        )
    return values


def stay_leave_values(
    bundles: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    stops: pd.DataFrame,
    source: str,
) -> list[Value]:
    """Question 3.5: stay/leave against content, per stratum.

    The per-session Wilcoxon is ``learning.cross_session_role_summary`` (the
    runner's own); the pooled GEE's q is BH across converged roles within a
    stratum, as ``run_stay_leave.py`` logs it.
    """
    make = value_factory(stay_leave_values, source, TAGS["stay_leave"])
    values = []
    for stratum, (logistic, _) in bundles.items():
        fits = fit_status_counts(logistic)
        values += [
            make(
                f"stay_converged_{stratum}",
                fits["n_converged"],
                "count",
                "count",
                f"stratum {stratum}: converged session-by-role logistic fits",
            ),
            make(
                f"stay_fits_{stratum}",
                fits["n_fits"],
                "count",
                "count",
                f"stratum {stratum}: session-by-role rows of the logistic fits, the "
                "sessions skipped before fitting (too few stops, a constant stay "
                "or outcome) included",
            ),
        ]
    signrank = smallest_adjusted_p(
        {
            stratum: cross_session_role_summary(logistic)
            for stratum, (logistic, _) in bundles.items()
        }
    )
    gee = smallest_adjusted_p(
        {
            stratum: pooled_gee_role_summary(pooled)
            for stratum, (_, pooled) in bundles.items()
        }
    )
    cohort = stop_statistics.stop_cohort_counts(stops)
    values += [
        make(
            "stay_min_q_signrank",
            signrank["p_fdr"],
            "quantity",
            "pval",
            f"smallest FDR-adjusted per-session Wilcoxon p over roles and strata (role {signrank['role']}, stratum {signrank['stratum']})",
        ),
        make(
            "stay_min_q_gee",
            gee["p_fdr"],
            "quantity",
            "pval",
            f"smallest FDR-adjusted pooled-GEE p over roles and strata (role {gee['role']}, stratum {gee['stratum']})",
        ),
        make(
            "stops_total",
            cohort["n_stops"],
            "count",
            "count",
            "stops in the canonical stop table",
            tags=TAGS["stay_leave_rate"],
        ),
        make(
            "stops_well_anchored",
            cohort["n_well_anchored"],
            "count",
            "count",
            "stops with occupancy_state at_well or mixed (the shared cohort)",
            tags=TAGS["stay_leave_rate"],
        ),
        make(
            "stops_dropped",
            cohort["n_not_anchored"],
            "count",
            "count",
            "stops the well-anchored cohort leaves out",
        ),
        make(
            "stops_dropped_with_stay",
            cohort["n_not_anchored_with_stay"],
            "count",
            "count",
            "of those, stops carrying an is_stay",
        ),
        make(
            "stops_mixed",
            cohort["n_mixed"],
            "count",
            "count",
            f"mixed stops ({cohort['n_mixed_with_stay']} of them with an is_stay)",
        ),
        make(
            "stops_mixed_with_stay",
            cohort["n_mixed_with_stay"],
            "count",
            "count",
            "mixed stops carrying an is_stay",
        ),
        make(
            "stops_at_well_with_stay",
            cohort["n_at_well_with_stay"],
            "count",
            "count",
            "at_well stops carrying an is_stay (Stage 6's population)",
        ),
    ]
    return values


# --- the SWR-power mark -----------------------------------------------------------


def power_values(coefs: pd.DataFrame, source: str) -> list[Value]:
    """The SWR-coupling mark: per-session OLS of event ripple power on
    standardized covariates, summarized as the fitting script prints it (median,
    share positive, animals whose median is positive, signed-rank p)."""
    make = value_factory(power_values, source, TAGS["ripple_power"])
    values = [
        make(
            "power_sessions",
            len(coefs),
            "count",
            "count",
            "sessions in the SWR-power model",
        )
    ]
    for word, column in POWER_COVARIATES.items():
        stats = coefficient_statistics(coefs, column)
        values += [
            make(
                f"power_{word}",
                stats["median"],
                "quantity",
                "sig2s",
                f"{column}: median per-session coefficient on ripple_power_z",
                "SD of power per SD",
            ),
            make(
                f"power_{word}_animals_positive",
                stats["n_animals_positive"],
                "count",
                "count",
                f"{column}: animals with a positive median",
            ),
            make(
                f"power_{word}_p",
                stats["signed_rank_p"],
                "quantity",
                "pval",
                f"{column}: signed-rank p",
            ),
        ]
    return values


# --- event history on the duration-aware design -------------------------------

#: The caveat every pooled history value carries.
UPPER_BOUND = (
    "pooled within animal with a session fixed effect; an upper bound, since rate "
    "variation between stops that the model does not capture inflates history terms"
)
#: Cross directions: key word -> (response population, label).
CROSS_DIRECTIONS = {
    "non_swr_from_swr": ("non_local_no_swr", "non-SWR rate after SWR-coupled events"),
    "swr_from_non_swr": ("non_local_swr", "SWR-coupled rate after non-SWR events"),
}
CROSS_LAG_MS = 45
#: The ridge cross-validated rungs (``analysis/replay/ridge_history_cv.py``).
RIDGE_SELF_RUNG = "Mself_long_dur"
RIDGE_CROSS_RUNG = "Mselfcross_long_dur"


def duration_aware_history_values(
    ridge: dict, pooled: dict, source: str
) -> list[Value]:
    """Question 3.3: refractoriness, burst and cross-coupling once duration is handled.

    ``pooled`` is ``duration_aware_history_pooled.pkl``: per animal and
    population, the rungs fit to the animal's stacked sessions. The
    self-history kernels are the both-anchor rung's (``Mself_dur``), read at
    measurable lags only (expected onsets >= ``HISTORY_MIN_SUPPORT``, Wald
    SE <= ``HISTORY_MAX_SE``) of converged fits; an animal whose lag is not
    measurable is left out of that lag's values, and the description says how
    many were kept. ``ridge`` is ``cross_history_ridge_cv.pkl``: per animal
    and non-local stream, the self-only and self+cross rungs with the cross
    block reaching 1 s, fit with a ridge prior on the history terms; the cross
    values are the self+cross rung's (:data:`RIDGE_CROSS_RUNG`) end-locked
    rate ratio at :data:`CROSS_LAG_MS`, read where that lag has at least
    ``HISTORY_MIN_SUPPORT`` expected onsets, whatever its posterior SD, and
    the rung's held-out gain (outer five-fold cross-validation over whole
    stops). Every ridge fit must have converged.
    """
    make = value_factory(duration_aware_history_values, source, TAGS["rate"])
    by_population = history_kernel_statistics.self_history_statistics(
        pooled["results"], "Mself_dur"
    )
    outcomes = {outcome for outcome, _ in RATE_POPULATIONS.values()}
    _require(
        set(by_population.index) == outcomes,
        f"pooled populations {sorted(by_population.index)} != {sorted(outcomes)}",
    )
    overall = history_kernel_statistics.self_history_range_statistics(by_population)
    n_fits, n_measured = overall["n_fits"], overall["n_first_lag_measurable"]
    end_peaks = (
        f"{overall['n_end_peaks']} of the {n_fits} animal x population fits have a "
        "measurable end-locked lag"
    )
    values: list[Value] = []
    first_ms = history_kernel_statistics.lag_ms(0)
    for word, (outcome, label) in RATE_POPULATIONS.items():
        with_peak = (
            f"the {by_population.loc[outcome, 'n_onset_peaks']} of "
            f"{by_population.loc[outcome, 'n_fits']} animals with a measurable "
            "onset-locked lag"
        )
        values += [
            make(
                f"history_onset_peak_{word}",
                by_population.loc[outcome, "onset_peak_rate_ratio_median"],
                "quantity",
                "dec2",
                f"{label}: onset-locked part of the self-history (both-anchor rung "
                f"Mself_dur, at-risk design): median over {with_peak} of each "
                f"animal's peak rate ratio over its measurable lags; {UPPER_BOUND}",
                "rate ratio",
            ),
            make(
                f"history_onset_peak_lag_ms_{word}",
                by_population.loc[outcome, "onset_peak_lag_ms_median"],
                "quantity",
                "int",
                f"{label}: median over {with_peak} of that onset-locked peak's lag",
                "ms",
            ),
            make(
                f"history_onset_animals_{word}",
                by_population.loc[outcome, "n_onset_peaks_above_one"],
                "count",
                "count",
                f"{label}: animals, of {with_peak}, whose onset-locked peak has its "
                "95 % CI above 1, the interval widened for the measurable lags "
                "searched (Bonferroni)",
            ),
        ]
    values += [
        make(
            "history_end_first_lag_rr_low",
            overall["first_lag_rate_ratio_low"],
            "quantity",
            "dec2",
            "lowest over the four populations of the median (over animals) rate ratio "
            f"{first_ms} ms after the previous event's end (end-locked part of "
            f"Mself_dur, at-risk design), over the {n_measured} of {n_fits} animal x "
            f"population fits whose {first_ms} ms lag is measurable; {UPPER_BOUND}",
            "rate ratio",
        ),
        make(
            "history_end_first_lag_rr_high",
            overall["first_lag_rate_ratio_high"],
            "quantity",
            "dec2",
            "highest over the four populations of that median rate ratio "
            f"{first_ms} ms after the previous event's end ({n_measured} of {n_fits} "
            "fits measurable)",
            "rate ratio",
        ),
        make(
            "history_end_first_lag_suppressed_fits",
            overall["n_first_lag_suppressed"],
            "count",
            "count",
            f"animal x population fits whose rate {first_ms} ms after the previous "
            "event's end is below 1 with its 95 % CI (suppression after the end), of "
            "the history_fits fits where that lag is measurable",
        ),
        make(
            "history_end_first_lag_raised_fits",
            overall["n_first_lag_raised"],
            "count",
            "count",
            f"animal x population fits whose rate {first_ms} ms after the previous "
            "event's end is above 1 with its 95 % CI, of the history_fits fits where "
            "that lag is measurable",
        ),
        make(
            "history_fits",
            n_measured,
            "count",
            "count",
            f"animal x population pooled fits ({n_fits} converged) whose {first_ms} ms "
            "end-locked lag is measurable: the denominator of the raised and "
            "suppressed counts",
        ),
        make(
            "history_end_peak_low",
            overall["end_peak_rate_ratio_low"],
            "quantity",
            "dec2",
            "lowest over the four populations of the median (over the animals with "
            "a measurable end-locked lag) peak rate ratio of the end-locked "
            "self-history (Mself_dur, at-risk design, measurable lags); "
            f"{end_peaks}; {UPPER_BOUND}",
            "rate ratio",
        ),
        make(
            "history_end_peak_high",
            overall["end_peak_rate_ratio_high"],
            "quantity",
            "dec2",
            f"highest over the four populations of that end-locked peak ({end_peaks})",
            "rate ratio",
        ),
        make(
            "history_end_peak_lag_ms_low",
            overall["end_peak_lag_ms_low"],
            "quantity",
            "int",
            "shortest over the populations of the median end-locked peak lag "
            f"({end_peaks})",
            "ms",
        ),
        make(
            "history_end_peak_lag_ms_high",
            overall["end_peak_lag_ms_high"],
            "quantity",
            "int",
            "longest over the populations of the median end-locked peak lag "
            f"({end_peaks})",
            "ms",
        ),
        make(
            "history_end_peak_animals_min",
            overall["n_end_peaks_above_one_min"],
            "count",
            "count",
            "fewest, over the populations, animals whose end-locked peak has its 95 % "
            "CI above 1, the interval widened for the measurable lags searched "
            f"(Bonferroni); {end_peaks}",
        ),
    ]
    for word, (outcome, label) in CROSS_DIRECTIONS.items():
        cross = history_kernel_statistics.cross_history_statistics(
            ridge["results"], outcome, RIDGE_CROSS_RUNG, CROSS_LAG_MS
        )
        n_animals = cross["n_animals"]
        kept = (
            f"over the {cross['n_measurable']} of {n_animals} animals whose "
            f"{CROSS_LAG_MS} ms lag has at least "
            f"{history_kernel_statistics.HISTORY_MIN_SUPPORT:g} expected onsets"
        )
        values += [
            make(
                f"history_cross_end_rr_{word}",
                cross["rate_ratio_median"],
                "quantity",
                "dec2",
                f"{label}: rate ratio {CROSS_LAG_MS} ms after the other stream's event "
                f"ended (end-locked cross block of {RIDGE_CROSS_RUNG}, 1 s cross "
                "window, at-risk design, ridge prior on the history terms with its "
                f"strength chosen by cross-validation), median {kept}; the prior "
                f"shrinks it toward 1; {UPPER_BOUND}",
                "rate ratio",
            ),
            make(
                f"history_cross_end_animals_{word}",
                cross["n_animals_above_one"],
                "count",
                "count",
                f"{label}: animals whose rate ratio at {CROSS_LAG_MS} ms has its 95 % "
                f"interval (posterior mode +- 1.96 posterior SD) above 1, {kept}",
            ),
            make(
                f"history_cross_heldout_gain_{word}",
                cross["heldout_gain_median"],
                "quantity",
                "sig2s",
                f"{label}: median over the {n_animals} animals of the held-out "
                f"log-likelihood gain per held-out event of {RIDGE_CROSS_RUNG} over "
                f"{RIDGE_SELF_RUNG} (five folds of whole stops, each rung's ridge "
                "strength chosen by inner cross-validation); positive = the cross "
                "block predicts new data",
                "nats per event",
            ),
            make(
                f"history_cross_heldout_animals_positive_{word}",
                cross["n_animals_gain_positive"],
                "count",
                "count",
                f"{label}: animals, of {n_animals}, whose held-out cross gain is "
                "positive",
            ),
        ]
    return values


# --- the within-arm horse race ------------------------------------------------

#: Horse-race predictors: key word -> (summary column, label).
HORSERACE_PREDICTORS = {
    "uncertainty": (
        "reward_belief_variance_candidate_arm_preoutcome_z",
        "reward-belief variance (uncertainty)",
    ),
    "value": (
        "reward_belief_mean_candidate_arm_preoutcome_z",
        "reward-belief mean (value)",
    ),
    "distance": ("distance_to_candidate_arm_cm_z", "distance to the arm"),
    "recency": ("visit_recency_trials_z", "visit recency"),
}
HORSERACE_POPULATIONS = {"all": "all", "swr": "SWR", "non_swr": "non_SWR"}


def horserace_values(summary: pd.DataFrame, source: str) -> list[Value]:
    """Question 3.7: are uncertainty, value and visit history separated?

    ``summary`` is ``horserace_summary.csv`` (``uncertainty_content.
    summarize_horserace``): per population and predictor, the median over
    sessions of the within-arm OLS coefficient (all four predictors entered
    together, arm-demeaned within session; non-current leaf arms), the animals
    whose own median has that sign, and the median design condition number.
    """
    make = value_factory(horserace_values, source, TAGS["content_uncertainty"])
    values: list[Value] = []
    for pop_word, pop in HORSERACE_POPULATIONS.items():
        rows = summary[summary["population"] == pop].set_index("predictor")
        _require(len(rows) == len(HORSERACE_PREDICTORS), f"horse race: {pop} rows")
        for word, (column, label) in HORSERACE_PREDICTORS.items():
            row = rows.loc[column]
            values += [
                make(
                    f"horserace_{word}_{pop_word}",
                    row["median"],
                    "quantity",
                    "sig2s",
                    f"{pop} events: within-arm coefficient of {label}, the four "
                    "predictors entered together; median over sessions",
                    "arm fraction per SD",
                ),
                make(
                    f"horserace_{word}_{pop_word}_animals",
                    int(row["n_animals_consistent"]),
                    "count",
                    "count",
                    f"{pop} events: animals whose own median {label} coefficient has "
                    "the sign of the overall median",
                ),
            ]
        values.append(
            make(
                f"horserace_condition_number_{pop_word}",
                float(rows["median_cond_number"].iloc[0]),
                "quantity",
                "dec2",
                f"{pop} events: median per-session condition number of the "
                "four-predictor design (collinearity check; below 10 = separable)",
            )
        )
    return values


def fit_staleness(*artifacts: Path) -> str | None:
    """Why the values read from fitted ``artifacts`` are pending, or None.

    A fit's provenance sidecar records the SHA-256 of each table and upstream
    fit it read. A fit with no sidecar, one that records no input, or one that
    records other inputs than the current ones, cannot be tied to the current
    data.
    """
    notes = []
    for artifact in artifacts:
        sidecar = fit_provenance_path(artifact)
        if not sidecar.exists():
            notes.append(
                f"{rel(artifact)} has no provenance record; refit before citing"
            )
            continue
        recorded = json.loads(sidecar.read_text())["inputs"]
        if not recorded:
            notes.append(
                f"{rel(artifact)} records no input it was fit on; refit before citing"
            )
            continue
        inputs = {
            name: (
                table_path(name, replay_content_dir())
                if name in CANONICAL_TABLE_BASENAMES
                else artifact.with_name(f"{name}.pkl")  # an upstream fit beside it
            )
            for name in recorded
        }
        stale = check_fit_provenance(artifact, inputs)
        if stale:
            notes.append(
                f"{rel(artifact)} was fit on other inputs than the current ones "
                f"({', '.join(stale)}); refit before citing"
            )
    return "; ".join(notes) or None


def _read_per_bin_fits(path: Path) -> pd.DataFrame:
    """A per-bin rate fit table, refused unless every row is on the at-risk design.

    Fits from the old all-bins, onset-history design carry no
    ``per_bin_design`` column; their ``hist_rcos`` terms are not the
    ``PER_BIN_HISTORY_COLUMNS`` this module reads.
    """
    fits = pd.read_pickle(path)
    if "per_bin_design" not in fits or (fits["per_bin_design"] != PER_BIN_DESIGN).any():
        raise ValueError(
            f"{path}: not fit on the {PER_BIN_DESIGN!r} per-bin design; refit it"
        )
    return fits


def summarize(inputs: BehaviorInputs, produced_by: str, commit: str) -> Summary:
    """The behavior summary from ``inputs``."""
    summary = Summary(name="behavior", produced_by=produced_by, commit=commit)

    def record(*paths: Path) -> None:
        summary.inputs.extend(file_record(p, ROOT_DIR) for p in paths)

    rate_fits = (
        inputs.per_bin,
        inputs.per_bin_all_stops,
        inputs.per_bin_in_transit,
        inputs.rate_contrast,
        inputs.per_stop,
    )
    contrast_audited = inputs.rate_contrast.with_name("swr_contrast_rate_audited.pkl")
    summary.add(
        *pend(
            rate_contrast_audit_values(
                pd.read_pickle(contrast_audited),
                f"{rel(contrast_audited)} (rate.swr_contrast.at_well)",
            ),
            fit_staleness(contrast_audited),
        )
    )
    record(contrast_audited)
    summary.add(
        *pend(
            rate_contrast_separation_values(
                _read_per_bin_fits(inputs.rate_contrast),
                f"{rel(inputs.rate_contrast)} (rate.swr_contrast.at_well)",
            ),
            fit_staleness(inputs.rate_contrast),
        )
    )
    in_transit = _read_per_bin_fits(inputs.per_bin_in_transit)
    make_transit = value_factory(
        summarize,
        f"{rel(inputs.per_bin_in_transit)} (rate.per_bin.in_transit)",
        TAGS["rate"],
        fit_staleness(inputs.per_bin_in_transit),
    )
    summary.add(
        make_transit(
            "rate_intransit_sessions",
            n_sessions(in_transit),
            "count",
            "count",
            "sessions with any per-bin fit on the in-transit stops (the sample gates "
            "leave the rest out)",
        ),
        *(
            make_transit(
                f"rate_intransit_sessions_{word}",
                n_sessions(fit_rows(in_transit, outcome=outcome)),
                "count",
                "count",
                f"{label}: sessions with a per-bin fit on the in-transit stops",
            )
            for word, (outcome, label) in RATE_POPULATIONS.items()
        ),
    )
    summary.add(*cohort_audit_values(inputs))
    record(inputs.per_bin_audit, inputs.per_stop_audit)
    per_bin = _read_per_bin_fits(inputs.per_bin)
    per_bin_all = _read_per_bin_fits(inputs.per_bin_all_stops)
    summary.add(
        *pend(
            per_bin_values(
                per_bin,
                per_bin_all,
                f"{rel(inputs.per_bin)} (rate.per_bin.at_well) and {rel(inputs.per_bin_all_stops)} (rate.per_bin.all_stops)",
            ),
            fit_staleness(inputs.per_bin, inputs.per_bin_all_stops),
        )
    )
    summary.add(
        *pend(
            per_bin_detail_values(
                per_bin,
                per_bin_all,
                in_transit,
                f"{rel(inputs.per_bin)}, {rel(inputs.per_bin_all_stops)} and "
                f"{rel(inputs.per_bin_in_transit)} (rate.per_bin.*)",
            ),
            fit_staleness(
                inputs.per_bin, inputs.per_bin_all_stops, inputs.per_bin_in_transit
            ),
        )
    )
    summary.add(
        *pend(
            per_stop_values(
                pd.read_pickle(inputs.per_stop),
                f"{rel(inputs.per_stop)} (rate.per_stop.at_well)",
            ),
            fit_staleness(inputs.per_stop),
        )
    )
    record(*rate_fits)

    summary.add(
        *pend(
            duration_aware_history_values(
                pd.read_pickle(inputs.cross_ridge_cv),
                pd.read_pickle(inputs.duration_aware_pooled),
                f"{rel(inputs.duration_aware_pooled)} "
                "(rate.duration_aware_history.pooled_at_well) and "
                f"{rel(inputs.cross_ridge_cv)} (rate.cross_history_ridge_cv.at_well)",
            ),
            fit_staleness(inputs.duration_aware_pooled, inputs.cross_ridge_cv),
        )
    )
    record(inputs.duration_aware_pooled, inputs.cross_ridge_cv)

    audited, sources = {}, {}
    for suffix in sorted({spec[0] for spec in CONTENT_SPECS.values()}):
        path = inputs.content_contrast_dir / f"swr_contrast_content_audited{suffix}.pkl"
        audited[suffix] = pd.read_pickle(path)
        sources[suffix] = f"{rel(path)} (audited aggregate)"
        record(path)
    content_stale = fit_staleness(
        *(
            inputs.content_contrast_dir / f"swr_contrast_content_audited{suffix}.pkl"
            for suffix in audited
        )
    )
    content = content_contrast_values(audited, sources)
    summary.add(*pend(content, content_stale))
    events = pd.read_parquet(
        inputs.event_table, columns=["animal", "is_swr", "n_nonlocal_bins"]
    )
    summary.add(
        *pend(
            breadth_derived_values(
                {v.key: v for v in content},
                events,
                f"{sources['_dm']}, {sources['_dm_len']} and {rel(inputs.event_table)}",
            ),
            content_stale,
        )
    )
    record(inputs.event_table)

    summary.add(
        *pend(
            estimator_values(
                pd.read_csv(inputs.estimator_summary),
                pd.read_csv(inputs.estimator_bootstrap),
                f"{rel(inputs.estimator_summary.parent)} (content.estimator_comparison)",
            ),
            fit_staleness(
                inputs.estimator_summary.with_name(
                    "content_estimator_comparison.parquet"
                )
            ),
        )
    )
    record(inputs.estimator_summary, inputs.estimator_bootstrap)

    leaf_fits = [inputs.leaf_dir / f"{n}.pkl" for n in ("Mu", "Mv", "model_comparison")]
    summary.add(
        *pend(
            leaf_values(
                inputs.leaf_dir,
                f"{rel(inputs.leaf_dir)} (content.arm_ladder.leaf_arms)",
            ),
            fit_staleness(*leaf_fits),
        )
    )
    record(*leaf_fits)

    summary.add(
        *pend(
            horserace_values(
                pd.read_csv(inputs.horserace_summary),
                f"{rel(inputs.horserace_summary)} (content.uncertainty)",
            ),
            fit_staleness(inputs.horserace_summary),
        )
    )
    record(inputs.horserace_summary)

    summary.add(
        *pend(
            choice_values(
                pd.read_pickle(inputs.choice_fits),
                f"{rel(inputs.choice_fits)} (behavior.event_source_suite.non_local)",
            ),
            fit_staleness(inputs.choice_fits),
        )
    )
    record(inputs.choice_fits)

    bundles = {}
    stay_leave_fits = []
    for stratum, sfx in STAY_STRATA.items():
        logistic = inputs.stay_leave_dir / f"stage6_is_stay_logistic{sfx}.pkl"
        pooled = inputs.stay_leave_dir / f"stage6_is_stay_pooled_gee{sfx}.pkl"
        stay_leave_fits += [logistic, pooled]
        bundles[stratum] = (pd.read_pickle(logistic), pd.read_pickle(pooled))
        record(logistic, pooled)
    stops = pd.read_parquet(inputs.stop_table)
    summary.add(
        *pend(
            stay_leave_values(
                bundles,
                stops,
                f"{rel(inputs.stay_leave_dir)}/stage6_is_stay_*.pkl (behavior.stay_leave.at_well*) and {rel(inputs.stop_table)}",
            ),
            fit_staleness(*stay_leave_fits),
        )
    )
    record(inputs.stop_table)
    onsets = pd.read_parquet(
        inputs.onset_table, columns=["event_type", "t_since_stop_onset", "stop_id"]
    )
    summary.add(
        *stop_table_values(
            stops, onsets, f"{rel(inputs.stop_table)} and {rel(inputs.onset_table)}"
        )
    )
    record(inputs.onset_table)
    arm, _ = select_event_cohort(
        pd.read_parquet(
            inputs.event_arm_table,
            columns=[
                *content_statistics.TRUNK_COLUMNS,
                content_statistics.TRUNK_RESPONSE,
                "occupancy_state",
            ],
        ),
        cohort="at_well",
    )
    summary.add(*trunk_values(arm, f"{rel(inputs.event_arm_table)} (at-well events)"))
    record(inputs.event_arm_table)
    split_fits = inputs.content_contrast_dir / "swr_contrast_content_arm.pkl"
    summary.add(
        *pend(
            current_arm_paired_values(
                pd.read_pickle(split_fits), f"{rel(split_fits)} (per-session fits)"
            ),
            fit_staleness(split_fits),
        )
    )
    record(split_fits)

    summary.add(
        *pend(
            power_values(
                pd.read_pickle(inputs.swr_power),
                f"{rel(inputs.swr_power)} (coupling.ripple_power)",
            ),
            fit_staleness(inputs.swr_power),
        )
    )
    record(inputs.swr_power)
    return summary
