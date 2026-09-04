"""Covariate naming contract — the single source of truth for column names.

Generated from the specification in
``docs/design/2026-08-21-covariate-naming-scheme.md``. Each :class:`Entry`
records the *canonical* column name together with the legacy spelling(s) it
replaced and the relationship between them (a plain rename, an alias/numerical
collapse, a derived complement, a retired predictor, or a dead output). The
rename map is **generated** from this registry (:func:`rename_map`), never
hand-authored, so there is one explicit decision per column.

The repository-owned table builders now emit canonical names directly and
validate their output with :func:`stamp_covariate_schema`; nothing in the
pipeline routes data through the legacy translation. Two roles keep this module
live and permanent:

- **Validation.** :func:`assert_canonical_vocabulary` uses the recorded legacy
  spellings as a deny-list, while :func:`assert_unique_scientific_rows` checks
  the registered row identity. The persisted stamp certifies exactly those two
  layers—not table-specific candidate support, dtypes, ranges, or fit eligibility.
- **Vendored ingestion.** The behavioral-model adapter in
  ``analysis.behavior.qvals_table`` translates the external model's native
  vocabulary (``Q``/``betadist_mu``/``Qstem`` ...) into the canonical
  reward-belief schema, reusing :func:`canonicalize_numerical_paths` /
  :func:`apply_rename` here as the one ingestion boundary.

Nothing in this module reads or writes a persisted table; it is a pure naming
contract. :func:`finalize_covariate_schema` (the full legacy->canonical
transform) is retained only for the migration-parity checking tools and is not
part of the production pipeline.
"""

from __future__ import annotations

import keyword
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Migration actions
# ---------------------------------------------------------------------------
# rename       : one current column -> one canonical name, value unchanged
#                (migration expects bit-for-bit equality).
# alias        : several byte-identical current columns collapse to one
#                canonical name; migration keeps one, drops the rest, and
#                asserts bit-for-bit equality across them.
# canonicalize : several current columns are the SAME quantity by separate
#                numerical paths (NOT proven byte-identical); collapse to one
#                canonical family under a declared float tolerance.
# retire       : a disendorsed column removed entirely, not carried forward
#                under any name (e.g. an interaction on a retired predictor).
# drop         : a dead model output (constant because its model is disabled).
# derive       : an exact complement of a stored primitive; do not store it,
#                derive it in a design matrix (``derive_expr``).
# audit        : a selection invariant / diagnostic; move to ``destination``
#                (metadata or an event-keyed audit), do not delete.
ACTIONS = frozenset(
    {"rename", "alias", "canonicalize", "retire", "drop", "derive", "audit"}
)
_COLLAPSE = frozenset({"alias", "canonicalize"})  # many current -> one canonical
_RENAMING = frozenset({"rename", "alias", "canonicalize"})  # produce a canonical

# Letter-then-digit is disallowed in a canonical name (`port0`) except inside a
# window/norm/percentile idiom where the number reads as part of the word.
_DIGIT_IDIOM_STEMS = ("last", "k", "l", "p", "log")  # last10, k10, l1, p90, log1p

# The version value is centralized in analysis/replay/versions.py (with the
# related cache/manifest literals, reviewed together); re-exported here so the
# many importers of covariate_registry.COVARIATE_SCHEMA_VERSION are unaffected.
from continuum_swr_replay.analysis.replay.versions import (  # noqa: E402
    COVARIATE_SCHEMA_VERSION,
)

COVARIATE_SCHEMA_SCOPE = "canonical_names_and_unique_row_keys_v1"

_ROW_KEYS: dict[str, tuple[str, ...]] = {
    "stop": ("animal", "session_date", "epoch", "stop_id"),
    "event": ("animal", "session_date", "epoch", "event_id"),
    "event_arm": (
        "animal",
        "session_date",
        "epoch",
        "event_id",
        "candidate_arm",
    ),
    "event_patch": (
        "animal",
        "session_date",
        "epoch",
        "event_id",
        "patch",
    ),
    "qvals": ("stop_id",),
    "event_sequence": (
        "animal",
        "session_date",
        "epoch",
        "event_id",
        "t_in_event",
    ),
    "event_onset": (
        "animal",
        "session_date",
        "epoch",
        "stop_id",
        "event_type",
        "onset_time",
    ),
    "stop_timebin": (
        "animal",
        "session_date",
        "epoch",
        "stop_id",
        "bin_in_stop",
    ),
}


def row_keys(*, table: str) -> tuple[str, ...]:
    """Return the canonical scientific row key for one registered table."""
    _entries_for(table)
    try:
        return _ROW_KEYS[table]
    except KeyError as exc:
        raise RegistryError(f"no row key is registered for table {table!r}") from exc


@dataclass(frozen=True)
class Entry:
    """One migration decision.

    ``current`` holds every legacy name that maps here (>= 2 for alias /
    canonicalize; exactly 1 otherwise). Exactly one action-appropriate target
    is set: ``canonical`` (rename/alias/canonicalize), ``primitive`` +
    ``derive_expr`` (derive), ``destination`` (audit); ``retire`` and ``drop``
    set none.
    """

    current: tuple[str, ...]
    action: str
    table: str
    canonical: str | None = None
    primitive: str | None = None
    derive_expr: str | None = None
    destination: str | None = None
    role: str = ""
    notes: str = ""


def _rename(current: str, canonical: str, table: str, **kw: str) -> Entry:
    return Entry((current,), "rename", table, canonical=canonical, **kw)


def _alias(current: tuple[str, ...], canonical: str, table: str, **kw: str) -> Entry:
    return Entry(current, "alias", table, canonical=canonical, **kw)


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------
REGISTRY: tuple[Entry, ...] = (
    # -- Cleanup 1: exact aliases -> collapse to one canonical --------------
    *[
        _alias(
            ("current_arm", "location_arm"),
            "occupancy_well_arm",
            table,
            notes="byte-identical incl. NaN; task state, NaN off-well, never trunk",
        )
        for table in ("stop", "event", "event_arm", "event_patch")
    ],
    *[
        _alias(
            ("reward_outcome", "is_after_reward"),
            "is_rewarded_outcome",
            table,
            role="outcome",
            notes="byte-identical; one nullable 0/1 primitive",
        )
        for table in ("stop", "event", "event_arm")
    ],
    _alias(("label_event_start", "event_start"), "event_start_s", "event"),
    _alias(("label_event_end", "event_end"), "event_end_s", "event"),
    _alias(("label_event_duration", "event_duration"), "event_duration_s", "event"),
    _alias(
        ("label_event_midpoint_time", "event_midpoint_time"),
        "event_midpoint_s",
        "event",
    ),
    _alias(
        ("posterior_mass_total", "nonlocal_weight_sum"),
        "nonlocal_probability_sum",
        "event",
        notes="byte-identical Sigma w_t",
    ),
    *[
        _alias(
            ("time_since_port_entry", "time_since_last_port_entry"),
            "time_since_last_well_arrival_s",
            table,
            notes="same variable assigned twice; elapsed since goal arrival",
        )
        for table in ("event", "event_arm")
    ],
    _rename("event_start", "event_start_s", "event_arm"),
    _rename("event_end", "event_end_s", "event_arm"),
    _rename("event_duration", "event_duration_s", "event_arm"),
    # -- Cleanup 2: numerical-path duplicates -> canonicalize (tolerance) ---
    Entry(
        ("prev_global_variance", "global_leaf_var_sum_preoutcome_sess"),
        "canonicalize",
        "qvals",
        canonical="reward_belief_variance_leaf_sum_preoutcome",
        notes="allclose, corr 1.0; declared tolerance atol=1e-15 rtol=1e-12",
    ),
    # (q_port / betadist_mu numerical-path family is in the value section.)
    # -- Cleanup 3: dead model outputs -> drop ------------------------------
    *[
        Entry(
            (f"depletion_leaf_{i}",),
            "drop",
            "qvals",
            notes="constant 1.0; withleaf fit has no depletion",
        )
        for i in range(1, 7)
    ],
    *[
        Entry(
            (f"depletion_mean_stem_{i}",),
            "drop",
            "qvals",
            notes="constant 1.0; withleaf fit has no depletion",
        )
        for i in range(1, 4)
    ],
    # -- Cleanup 4: selection invariants / diagnostics -> audit -------------
    Entry(
        ("whole_event_speed_eligible",),
        "audit",
        "event",
        destination="artifact/cohort metadata",
        notes="certifies the immobility/speed filter the cohort passed",
    ),
    Entry(
        ("endpoints_speed_below_thresh",),
        "audit",
        "event",
        destination="artifact/cohort metadata",
        notes="legacy endpoint speed certificate",
    ),
    Entry(
        ("nonlocal_fraction",),
        "audit",
        "event",
        destination="event_table / event-keyed audit",
        notes="all 1.0 here; eligibility fraction defining each event",
    ),
    Entry(
        ("nospike_fraction",),
        "audit",
        "event",
        destination="event_table / event-keyed audit",
        notes="all 0.0 here; eligibility fraction defining each event",
    ),
    Entry(
        ("n_junction_bins",),
        "rename",
        "event",
        canonical="n_undecoded_bins",
        role="diagnostic",
        notes="counts undecoded bins, not junctions; retain at event grain "
        "(sum(arm_count_hard) = n_nonlocal_bins - n_undecoded_bins)",
    ),
    # -- Cleanup 5 / Switch family: complements + switch renames ------------
    *[
        Entry(
            ("is_after_switch",),
            "rename",
            table,
            canonical="did_last_completed_trip_switch_patch",
            role="provenance",
            notes="completed-action flag; retired as a fitted predictor",
        )
        for table in ("stop", "event", "event_arm")
    ],
    *[
        Entry(
            ("is_after_stay",),
            "derive",
            table,
            primitive="did_last_completed_trip_switch_patch",
            derive_expr="did_last_completed_trip_stay_in_patch = 1 - switch",
            notes="verified exact complement; do not persist",
        )
        for table in ("stop", "event", "event_arm")
    ],
    *[
        Entry(
            ("is_after_omission",),
            "derive",
            table,
            primitive="is_rewarded_outcome",
            derive_expr="is_omitted_outcome = 1 - reward",
            notes="derive with the primitive's missingness; do not persist",
        )
        for table in ("stop", "event", "event_arm")
    ],
    _rename(
        "is_post_switch",
        "is_periswitch_post",
        "qvals",
        notes="peri-switch alignment flag; hinge partner",
    ),
    _rename(
        "trials_since_switch",
        "periswitch_index_trials",
        "qvals",
        notes="signed, two-sided; drops the false 'since'",
    ),
    _rename(
        "recent_switch_rate",
        "patch_switch_fraction_last5_completed_trips",
        "stop",
        role="predictor",
        notes="fraction over a fixed 5-completed-trip window, in [0,1]",
    ),
    Entry(
        ("switch_x_current",),
        "retire",
        "event_arm",
        role="interaction",
        notes="interaction on the retired immediate-switch predictor; use the "
        "graded patch_switch_fraction_last5_completed_trips instead",
    ),
    Entry(
        ("switch_x_prev",),
        "retire",
        "event_arm",
        role="interaction",
        notes="sibling of switch_x_current; same retired-predictor interaction",
    ),
    # -- Arm occupancy ------------------------------------------------------
    # (location_arm / current_arm collapse is in Cleanup 1 above.)
    *[
        _rename(
            "physical_arm",
            "modal_observed_track_segment_id",
            table,
            notes="modal observed track segment, trunks included, never NaN",
        )
        for table in ("stop", "event", "event_arm", "event_patch")
    ],
    _rename(
        "arm",
        "candidate_arm",
        "event_arm",
        role="predictor",
        notes="the row's arm in the choice set",
    ),
    *[
        _rename("location_well", "occupancy_well_id", table)
        for table in ("stop", "event", "event_arm", "event_patch")
    ],
    *[
        _rename("location_state", "occupancy_state", table)
        for table in ("stop", "event", "event_arm", "event_patch")
    ],
    # -- Relational arm indicators ------------------------------------------
    _rename("is_current", "is_current_arm", "event_arm", role="predictor"),
    _rename("is_prev", "is_prev_arm", "event_arm", role="predictor"),
    _rename("is_next", "is_next_arm", "event_arm", role="outcome"),
    _rename("is_last_rewarded", "is_last_rewarded_arm", "event_arm"),
    _rename(
        "is_last_rewarded_elsewhere",
        "is_last_rewarded_noncurrent_arm",
        "event_arm",
        notes="omission-only support is structural; keep out of the primary hierarchy",
    ),
    _rename("arm_never_visited", "is_candidate_arm_never_visited", "event_arm"),
    _rename(
        "arm_never_visited_patch",
        "is_patch_never_visited",
        "event_patch",
        notes="patch-grain column in event_patch_table (tables.py:1685); "
        "used by BASEc_PATCH (model_specs.py:1233)",
    ),
    _rename("is_same_patch", "is_same_patch_as_current", "event_arm"),
    _rename("dist_to_arm", "distance_to_candidate_arm_cm", "event_arm"),
    _rename("arm_dist_rank", "candidate_arm_distance_rank", "event_arm"),
    _rename("visit_recency", "visit_recency_trials", "event_arm"),
    # -- Behavioral-model value and belief ----------------------------------
    _rename(
        "q_arm",
        "reward_belief_mean_candidate_arm_preoutcome",
        "event_arm",
        role="predictor",
        notes="Beta belief mean; reward_belief_ separates it from the "
        "spatial decoder posterior",
    ),
    _rename(
        "p_reward_arm",
        "task_reward_probability_candidate_arm",
        "event_arm",
        notes="programmed contingency, not a belief",
    ),
    _rename(
        "arm_uncertainty",
        "reward_belief_variance_candidate_arm_preoutcome",
        "event_arm",
        role="predictor",
    ),
    # Q{i+1}, q_port{i}, and betadist_mu{i+1} are the SAME belief mean by
    # separate paths. q_port is assigned directly from Q; betadist_mu is the
    # model-native posterior mean and is tolerance-equal on real artifacts.
    *[
        Entry(
            (f"Q{i + 1}", f"q_port{i}", f"betadist_mu{i + 1}"),
            "canonicalize",
            "qvals",
            canonical=f"reward_belief_mean_port_{i}",
            notes="old Beta index i maps to port i-1; tol atol=1e-15 rtol=1e-12",
        )
        for i in range(6)
    ],
    *[
        _rename(f"betadist_var{i + 1}", f"reward_belief_variance_port_{i}", "qvals")
        for i in range(6)
    ],
    *[
        _rename(f"p_reward_port{i}", f"task_reward_probability_port_{i}", "qvals")
        for i in range(6)
    ],
    *[
        _rename(
            f"Qstem{i}",
            f"decision_value_patch_{i}",
            "qvals",
            notes="1-based task patches; not a probability",
        )
        for i in range(1, 4)
    ],
    _rename(
        "switch_advantage_mu_preoutcome_sess",
        "relative_value_switch_minus_stay_preoutcome",
        "qvals",
        role="predictor",
        notes="manuscript relative value; sign convention",
    ),
    _rename(
        "switch_advantage_preoutcome_sess_Qstem",
        "relative_decision_value_switch_minus_stay_preoutcome",
        "qvals",
        notes="Qstem-scale contrast, distinct from the reward-belief one",
    ),
    _rename(
        "stay_sibling_mu_preoutcome_sess",
        "stay_value_preoutcome",
        "qvals",
        role="predictor",
        notes="manuscript term; sibling-leaf belief mean",
    ),
    _rename(
        "best_other_patch_mean_mu_preoutcome_sess",
        "switch_value_preoutcome",
        "qvals",
        notes="manuscript term; max over two alternative patches",
    ),
    _rename(
        "global_mu_update_abs_postoutcome_sess",
        "reward_belief_mean_update_l1_all_ports_postoutcome",
        "qvals",
        notes="L1 aggregate over six ports",
    ),
    _rename("rpe", "rpe_postoutcome", "qvals", role="predictor"),
    _rename("rpe_abs", "rpe_postoutcome_abs", "qvals"),
    _rename("q_chosen", "reward_belief_mean_chosen_arm_preoutcome", "qvals"),
    _rename(
        "arm",
        "chosen_arm",
        "qvals",
        notes="realized trial destination; not a choice-set candidate row",
    ),
    _rename(
        "beta_reward_entropy",
        "reward_belief_differential_entropy_leaf_mean_preoutcome",
        "qvals",
        notes="Beta differential entropy; can be negative, not Shannon",
    ),
    _rename(
        "is_rewarding",
        "pump_on_at_bout_mid",
        "qvals",
        role="diagnostic",
        notes="mid-bout pump flag (4.8%), not a trial outcome",
    ),
    _rename(
        "leaf_choice_p", "leaf_choice_logp", "qvals", notes="a log probability (<=0)"
    ),
    _rename(
        "p_reward_current",
        "task_reward_probability_chosen_arm",
        "qvals",
        role="predictor",
        notes="scheduled reward probability selected by the realized destination arm",
    ),
    _rename(
        "p_reward_best",
        "task_reward_probability_best_arm",
        "qvals",
        role="diagnostic",
        notes="maximum scheduled reward probability over six arms",
    ),
    _rename(
        "value_diff_patch",
        "task_reward_probability_patch_mean_gap_best_vs_second",
        "qvals",
        role="predictor",
        notes="best minus second-best mean scheduled reward probability over patches",
    ),
    _rename(
        "chosen_leaf_mu_update_abs_postoutcome_sess",
        "reward_belief_mean_update_chosen_arm_postoutcome_abs",
        "qvals",
        role="diagnostic",
        notes="absolute scalar update at the chosen arm",
    ),
    _rename(
        "chosen_patch_mu_update_abs_postoutcome_sess",
        "reward_belief_mean_update_l1_chosen_patch_postoutcome",
        "qvals",
        role="diagnostic",
        notes="L1 update over the two ports in the chosen patch",
    ),
    _rename(
        "unchosen_patches_mu_update_abs_postoutcome_sess",
        "reward_belief_mean_update_l1_alternative_patches_postoutcome",
        "qvals",
        role="predictor",
        notes="L1 update over the four ports in alternative patches",
    ),
    _rename(
        "global_mu_update_signed_postoutcome_sess",
        "reward_belief_mean_update_sum_all_ports_postoutcome",
        "qvals",
        role="diagnostic",
        notes="signed sum of reward-belief mean updates over six ports",
    ),
    _rename(
        "chosen_leaf_var_preoutcome_sess",
        "reward_belief_variance_chosen_arm_preoutcome",
        "qvals",
        role="diagnostic",
        notes="pre-outcome Beta variance at the realized chosen arm",
    ),
    # -- Reward history and time --------------------------------------------
    _rename(
        "cumulative_reward",
        "reward_count_candidate_arm_last10_known_trials",
        "event_arm",
        role="predictor",
        notes="rolling 10-known-trial count in [0,5], not cumulative",
    ),
    _rename(
        "reward_rate_arm_k10",
        "reward_fraction_candidate_arm_last10_visits",
        "event_arm",
        role="predictor",
        notes="last 10 visits to that arm; a fraction",
    ),
    _rename(
        "reward_rate_trial_k10",
        "reward_fraction_last10_known_trials",
        "stop",
        notes="arm-agnostic; last 10 outcome-known trials",
    ),
    *[
        _rename(
            f"reward_lag{j}",
            f"known_reward_outcome_lag_{j}",
            "stop",
            notes="j-th most recently KNOWN outcome, not a fixed lag",
        )
        for j in range(1, 11)
    ],
    _alias(
        ("n_events", "n_non_local"),
        "n_selected_events",
        "stop",
        role="outcome",
        notes="event-source count per stop; exact duplicates in Replay inputs",
    ),
    _rename("n_SWR", "n_swr_events", "stop", role="outcome"),
    _rename("n_MUA", "n_mua_events", "stop", role="outcome"),
    _rename("n_non_local_SWR", "n_nonlocal_swr_events", "stop", role="outcome"),
    _rename("n_non_local_no_SWR", "n_nonlocal_non_swr_events", "stop", role="outcome"),
    _rename("n_SWR", "n_swr_events", "stop_timebin", role="outcome"),
    _rename("n_MUA", "n_mua_events", "stop_timebin", role="outcome"),
    _rename("n_non_local", "n_nonlocal_events", "stop_timebin", role="outcome"),
    *[
        _rename("time_since_stop", "time_since_stop_start_s", table)
        for table in ("event", "event_arm")
    ],
    *[
        _rename("time_since_reward", "time_since_last_reward_s", table)
        for table in ("event", "event_arm")
    ],
    # (time_since_port_entry alias pair is in Cleanup 1 above.)
    _rename(
        "immobility_time",
        "immobility_duration_s",
        "stop",
        notes="stop_end - stop_start; a duration",
    ),
    # -- Event-table geometry: the explicitly-named exceptions --------------
    *[
        _rename(
            f"posterior_hpd_width_{c}",
            f"posterior_hpd_arm_count_mean_coverage_{c}pct",
            "event",
            role="diagnostic",
            notes="prob-weighted mean of per-bin HPD arm count; fractional",
        )
        for c in (50, 80, 95)
    ],
    *[
        Entry(
            ("day",),
            "rename",
            table,
            canonical="session_date",
            role="provenance",
            notes="YYYYMMDD date, not a day index",
        )
        for table in (
            "stop",
            "event",
            "event_arm",
            "event_patch",
            "event_sequence",
            "event_onset",
            "stop_timebin",
        )
    ],
    # -- Transforms and interactions ----------------------------------------
    _rename(
        "visit_recency_demeaned",
        "visit_recency_trials_event_leaf_centered",
        "event_arm",
        notes="centered across visited leaf candidates within each event",
    ),
    _rename(
        "swr_x_visit_recency_demeaned",
        "swr_x_visit_recency_trials_event_leaf_centered",
        "event_arm",
        role="interaction",
    ),
    _rename(
        "visit_recency_spline1_demeaned",
        "visit_recency_trials_event_leaf_centered_spline_1",
        "event_arm",
    ),
    _rename(
        "visit_recency_spline2_demeaned",
        "visit_recency_trials_event_leaf_centered_spline_2",
        "event_arm",
    ),
    _rename(
        "visit_recency_spline3_demeaned",
        "visit_recency_trials_event_leaf_centered_spline_3",
        "event_arm",
    ),
    _rename(
        "swr_x_visit_recency_spline1_demeaned",
        "swr_x_visit_recency_trials_event_leaf_centered_spline_1",
        "event_arm",
        role="interaction",
    ),
    _rename(
        "swr_x_visit_recency_spline2_demeaned",
        "swr_x_visit_recency_trials_event_leaf_centered_spline_2",
        "event_arm",
        role="interaction",
    ),
    _rename(
        "swr_x_visit_recency_spline3_demeaned",
        "swr_x_visit_recency_trials_event_leaf_centered_spline_3",
        "event_arm",
        role="interaction",
    ),
    # Interaction names inherit the canonical spelling of every base term.
    # Keeping only the ``*_x_*`` product grammar stable prevents a renamed
    # main effect from acquiring a second, legacy identity in model output.
    _rename(
        "current_x_last_rewarded",
        "is_current_arm_x_is_last_rewarded_arm",
        "event_arm",
        role="interaction",
    ),
    _rename(
        "time_x_current",
        "time_since_stop_start_s_x_is_current_arm",
        "event_arm",
        role="interaction",
    ),
    _rename(
        "time_x_prev",
        "time_since_stop_start_s_x_is_prev_arm",
        "event_arm",
        role="interaction",
    ),
    _rename(
        "time_x_next",
        "time_since_stop_start_s_x_is_next_arm",
        "event_arm",
        role="interaction",
    ),
    _rename("swr_x_current", "swr_x_is_current_arm", "event_arm", role="interaction"),
    _rename("swr_x_prev", "swr_x_is_prev_arm", "event_arm", role="interaction"),
    _rename("swr_x_next", "swr_x_is_next_arm", "event_arm", role="interaction"),
    _rename(
        "swr_x_last_rewarded",
        "swr_x_is_last_rewarded_arm",
        "event_arm",
        role="interaction",
    ),
    _rename(
        "swr_x_reward_rate_arm_k10",
        "swr_x_reward_fraction_candidate_arm_last10_visits",
        "event_arm",
        role="interaction",
    ),
    _rename(
        "swr_x_arm_never_visited",
        "swr_x_is_candidate_arm_never_visited",
        "event_arm",
        role="interaction",
    ),
    _rename(
        "swr_x_is_last_rewarded_elsewhere",
        "swr_x_is_last_rewarded_noncurrent_arm",
        "event_arm",
        role="interaction",
    ),
    _rename(
        "swr_x_arm_uncertainty",
        "swr_x_reward_belief_variance_candidate_arm_preoutcome",
        "event_arm",
        role="interaction",
    ),
    _rename("mua_x_current", "mua_x_is_current_arm", "event_arm", role="interaction"),
    _rename("mua_x_prev", "mua_x_is_prev_arm", "event_arm", role="interaction"),
    _rename("mua_x_next", "mua_x_is_next_arm", "event_arm", role="interaction"),
    _rename(
        "mua_x_last_rewarded",
        "mua_x_is_last_rewarded_arm",
        "event_arm",
        role="interaction",
    ),
    _rename(
        "continuous_x_current",
        "continuous_replay_score_mean_x_is_current_arm",
        "event_arm",
        role="interaction",
    ),
    _rename(
        "continuous_x_prev",
        "continuous_replay_score_mean_x_is_prev_arm",
        "event_arm",
        role="interaction",
    ),
    _rename(
        "continuous_x_next",
        "continuous_replay_score_mean_x_is_next_arm",
        "event_arm",
        role="interaction",
    ),
    _rename(
        "omission_x_last_rewarded",
        "is_omitted_outcome_x_is_last_rewarded_arm",
        "event_arm",
        role="interaction",
    ),
    # Patch-grain aggregates preserve the arm-level scientific definition and
    # append ``_patch`` to state that the two leaf-arm values were aggregated.
    _rename(
        "cumulative_reward_patch",
        "reward_count_candidate_arm_last10_known_trials_patch",
        "event_patch",
        role="predictor",
    ),
    _rename(
        "reward_rate_arm_k10_patch",
        "reward_fraction_candidate_arm_last10_visits_patch",
        "event_patch",
        role="predictor",
    ),
    _rename(
        "p_reward_arm_patch",
        "task_reward_probability_candidate_arm_patch",
        "event_patch",
        role="predictor",
    ),
    _rename(
        "q_arm_patch",
        "reward_belief_mean_candidate_arm_preoutcome_patch",
        "event_patch",
        role="predictor",
    ),
    _rename(
        "visit_recency_patch",
        "visit_recency_trials_patch",
        "event_patch",
        role="predictor",
    ),
    _rename(
        "is_last_rewarded_patch_elsewhere",
        "is_last_rewarded_noncurrent_patch",
        "event_patch",
    ),
    Entry(
        ("switch_x_current_patch",),
        "retire",
        "event_patch",
        role="interaction",
        notes="interaction with retired immediate-switch predictor",
    ),
    _rename(
        "swr_x_current_patch",
        "swr_x_is_current_patch",
        "event_patch",
        role="interaction",
    ),
    _rename(
        "swr_x_prev_patch", "swr_x_is_prev_patch", "event_patch", role="interaction"
    ),
    _rename(
        "swr_x_last_rewarded_patch_elsewhere",
        "swr_x_is_last_rewarded_noncurrent_patch",
        "event_patch",
        role="interaction",
    ),
    # Choice-grain fractions are built after the persisted event-arm schema.
    # Register them so model/result scans treat the old abbreviation as legacy.
    _rename("replay_frac", "replay_fraction", "choice", role="predictor"),
    _rename(
        "replay_frac_residual", "replay_fraction_residual", "choice", role="predictor"
    ),
    _rename("replay_frac_swr", "replay_fraction_swr", "choice", role="predictor"),
    _rename(
        "replay_frac_non_swr", "replay_fraction_non_swr", "choice", role="predictor"
    ),
    _rename(
        "replay_frac_current_patch",
        "replay_fraction_current_patch",
        "choice",
        role="diagnostic",
    ),
    _rename(
        "replay_frac_noncurrent",
        "replay_fraction_noncurrent",
        "choice",
        role="diagnostic",
    ),
    _rename(
        "replay_frac_residual_noncurrent",
        "replay_fraction_residual_noncurrent",
        "choice",
        role="diagnostic",
    ),
    # Fit-time products spell out every canonical factor. These are generated
    # directly (not persisted source-table columns), but the registry keeps the
    # legacy abbreviations visible to repository-wide guards.
    _rename(
        "rpe_x_current",
        "rpe_postoutcome_z_x_is_current_arm",
        "fit_time",
        role="interaction",
    ),
    _rename(
        "relval_x_current",
        "relative_value_switch_minus_stay_preoutcome_z_x_is_current_arm",
        "fit_time",
        role="interaction",
    ),
    _rename(
        "entropy_x_current",
        "reward_belief_differential_entropy_leaf_mean_preoutcome_z_x_is_current_arm",
        "fit_time",
        role="interaction",
    ),
    _rename(
        "update_x_current",
        "reward_belief_mean_update_l1_all_ports_postoutcome_z_x_is_current_arm",
        "fit_time",
        role="interaction",
    ),
    _rename(
        "tss_x_current",
        "periswitch_index_trials_z_x_is_current_arm",
        "fit_time",
        role="interaction",
    ),
    _rename(
        "postsw_x_current",
        "is_periswitch_post_x_is_current_arm",
        "fit_time",
        role="interaction",
    ),
    _rename(
        "tsspostsw_x_current",
        "periswitch_index_trials_z_x_is_periswitch_post_x_is_current_arm",
        "fit_time",
        role="interaction",
    ),
    _rename(
        "block_x_current",
        "trials_since_block_change_z_x_is_current_arm",
        "fit_time",
        role="interaction",
    ),
    _rename(
        "switchrate_x_current",
        "patch_switch_fraction_last5_completed_trips_z_x_is_current_arm",
        "fit_time",
        role="interaction",
    ),
    _rename(
        "tsc_x_current",
        "log1p_time_since_stop_start_s_z_x_is_current_arm",
        "fit_time",
        role="interaction",
    ),
    _rename(
        "currentpatch_x_tss",
        "periswitch_index_trials_z_x_is_current_patch",
        "fit_time",
        role="interaction",
    ),
    _rename(
        "currentpatch_x_tsc",
        "log1p_time_since_stop_start_s_z_x_is_current_patch",
        "fit_time",
        role="interaction",
    ),
    _rename(
        "currentpatch_x_switchrate",
        "patch_switch_fraction_last5_completed_trips_z_x_is_current_patch",
        "fit_time",
        role="interaction",
    ),
    # Fit-time arm fixed effects: arm 8 is the dropped reference (see spec).
    *[
        _rename(
            f"arm_fe_{k}",
            f"is_candidate_arm_{k}",
            "fit_time",
            role="predictor",
            notes="arm_fe_k = 1 iff arm==k; reference_arm=8",
        )
        for k in range(8)
    ],
    # -- Event-table geometry: units (append _cm / _cm_s / _fraction) -------
    # Replay MAP geometry (trajectory.py), unambiguously replay: append _cm.
    _rename("net_displacement", "net_displacement_cm", "event", role="diagnostic"),
    _rename("path_length", "path_length_cm", "event", role="diagnostic"),
    _rename("spatial_extent", "spatial_extent_cm", "event", role="diagnostic"),
    _rename(
        "step_distance_median", "step_distance_median_cm", "event", role="diagnostic"
    ),
    _rename("step_distance_p90", "step_distance_p90_cm", "event", role="diagnostic"),
    _rename("step_distance_max", "step_distance_max_cm", "event", role="diagnostic"),
    _rename("max_ahead", "max_ahead_cm", "event", role="diagnostic"),
    _rename("max_behind", "max_behind_cm", "event", role="diagnostic"),
    _rename(
        "max_abs_ahead_behind", "max_abs_ahead_behind_cm", "event", role="diagnostic"
    ),
    _rename(
        "mean_abs_ahead_behind", "mean_abs_ahead_behind_cm", "event", role="diagnostic"
    ),
    # Position collides with the animal's position -> subject prefix + unit.
    _rename(
        "start_position",
        "replay_start_position_cm",
        "event",
        role="diagnostic",
        notes="replay MAP position; animal has its own",
    ),
    _rename("end_position", "replay_end_position_cm", "event", role="diagnostic"),
    # Animal head speed (event_table.py): prefix animal_, append _cm_s.
    _rename(
        "max_speed",
        "animal_max_speed_cm_s",
        "event",
        role="diagnostic",
        notes="animal head speed, not replay speed",
    ),
    _rename("mean_speed", "animal_mean_speed_cm_s", "event", role="diagnostic"),
    _rename("start_speed", "animal_start_speed_cm_s", "event", role="diagnostic"),
    _rename("end_speed", "animal_end_speed_cm_s", "event", role="diagnostic"),
    # Replay MAP speed (already replay_-prefixed): append _cm_s.
    _rename(
        "replay_speed_median", "replay_speed_median_cm_s", "event", role="diagnostic"
    ),
    _rename("replay_speed_p90", "replay_speed_p90_cm_s", "event", role="diagnostic"),
    _rename(
        "replay_speed_continuous_median",
        "replay_speed_continuous_median_cm_s",
        "event",
        role="diagnostic",
    ),
    _rename(
        "ab_slope",
        "ab_slope_cm_s",
        "event",
        role="diagnostic",
        notes="slope of ahead/behind vs time; a replay speed",
    ),
    # Fraction: keep the full word 'fraction'; only path_continuity lacks it.
    _rename("path_continuity", "path_continuity_fraction", "event", role="diagnostic"),
)


# ---------------------------------------------------------------------------
# Derived views
# ---------------------------------------------------------------------------
def _entries_for(table: str) -> tuple[Entry, ...]:
    """Return registry entries for one persisted table/grain.

    Column spellings are not globally meaningful: for example ``arm`` is the
    choice-set row in ``event_arm`` but the realized destination in ``qvals``.
    Requiring a table makes it impossible to apply one grain's semantic rename
    to another merely because the legacy spelling happens to match.
    """
    known = frozenset(e.table for e in REGISTRY)
    if table not in known:
        raise RegistryError(
            f"unknown covariate-registry table {table!r}; expected {sorted(known)}"
        )
    return tuple(e for e in REGISTRY if e.table == table)


def rename_map(*, table: str) -> dict[str, str]:
    """Map current columns to canonical names for exactly one table/grain."""
    out: dict[str, str] = {}
    for e in _entries_for(table):
        if e.action in _RENAMING:
            assert e.canonical is not None
            for name in e.current:
                out[name] = e.canonical
    return out


def retired_columns(*, table: str | None = None) -> frozenset[str]:
    entries = REGISTRY if table is None else _entries_for(table)
    return frozenset(n for e in entries if e.action == "retire" for n in e.current)


def dropped_columns(*, table: str | None = None) -> frozenset[str]:
    entries = REGISTRY if table is None else _entries_for(table)
    return frozenset(n for e in entries if e.action == "drop" for n in e.current)


def derived_columns(*, table: str | None = None) -> tuple[Entry, ...]:
    entries = REGISTRY if table is None else _entries_for(table)
    return tuple(e for e in entries if e.action == "derive")


def audit_columns(*, table: str | None = None) -> tuple[Entry, ...]:
    entries = REGISTRY if table is None else _entries_for(table)
    return tuple(e for e in entries if e.action == "audit")


def all_current_columns(*, table: str | None = None) -> frozenset[str]:
    """Every legacy column named anywhere in the registry."""
    entries = REGISTRY if table is None else _entries_for(table)
    return frozenset(n for e in entries for n in e.current)


def canonical_names(*, table: str | None = None) -> frozenset[str]:
    """Every canonical name, globally or for one explicit table/grain."""
    entries = REGISTRY if table is None else _entries_for(table)
    return frozenset(e.canonical for e in entries if e.canonical is not None)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
_IDENT = re.compile(r"^[a-z][a-z0-9_]*$")
_LETTER_DIGIT = re.compile(r"([a-z]+)\d")


class RegistryError(ValueError):
    """A structural inconsistency in the naming registry."""


def _is_snake_identifier(name: str) -> bool:
    return bool(_IDENT.match(name)) and not keyword.iskeyword(name)


def _digit_boundary_ok(name: str) -> bool:
    """Enumerated indices take a leading ``_``; only idioms may glue a digit."""
    for stem in _LETTER_DIGIT.findall(name):
        if stem not in _DIGIT_IDIOM_STEMS:
            return False
    return True


def validate() -> None:
    """Raise :class:`RegistryError` on any structural inconsistency.

    Checks: known action; action-appropriate target present and correct arity;
    each current name claimed once; each canonical produced by one entry;
    canonical names are snake_case identifiers obeying the digit-boundary rule;
    no canonical collides with a surviving legacy name; every derived complement
    references a stored primitive.
    """
    seen_current: dict[tuple[str, str], Entry] = {}
    seen_canonical: dict[tuple[str, str], Entry] = {}
    for e in REGISTRY:
        if e.action not in ACTIONS:
            raise RegistryError(f"unknown action {e.action!r} for {e.current}")
        if not e.current:
            raise RegistryError(f"entry with no current column: {e}")
        for name in e.current:
            key = (e.table, name)
            if key in seen_current:
                raise RegistryError(
                    f"current column {name!r} claimed twice for table {e.table!r}"
                )
            seen_current[key] = e

        if e.action in _RENAMING:
            if not e.canonical:
                raise RegistryError(f"{e.action} without canonical: {e.current}")
            if e.action in _COLLAPSE and len(e.current) < 2:
                raise RegistryError(
                    f"{e.action} must collapse >=2 columns: {e.current}"
                )
            if e.action == "rename" and len(e.current) != 1:
                raise RegistryError(f"rename must be 1:1: {e.current}")
            canonical_key = (e.table, e.canonical)
            if canonical_key in seen_canonical:
                raise RegistryError(
                    f"canonical {e.canonical!r} produced by two entries for "
                    f"table {e.table!r}"
                )
            seen_canonical[canonical_key] = e
            if not _is_snake_identifier(e.canonical):
                raise RegistryError(f"canonical {e.canonical!r} is not snake_case")
            if not _digit_boundary_ok(e.canonical):
                raise RegistryError(
                    f"canonical {e.canonical!r} glues a digit outside an idiom"
                )
        elif e.action == "derive":
            if not e.primitive or not e.derive_expr:
                raise RegistryError(f"derive without primitive/expr: {e.current}")
        elif e.action == "audit":
            if not e.destination:
                raise RegistryError(f"audit without destination: {e.current}")
        else:  # retire, drop
            if e.canonical:
                raise RegistryError(f"{e.action} must not set canonical: {e.current}")

    # A canonical name must not collide with any legacy column name — otherwise
    # the new name shadows a column still present (or renamed elsewhere) at
    # migration time.
    collisions = set(seen_canonical) & set(seen_current)
    if collisions:
        raise RegistryError(
            "canonical names collide with legacy column names in one table: "
            f"{sorted(collisions)}"
        )

    # Every derived complement must resolve to a stored primitive.
    stored_by_table = {
        table: canonical_names(table=table) for table in {e.table for e in REGISTRY}
    }
    for e in derived_columns():
        if e.primitive not in stored_by_table[e.table]:
            raise RegistryError(
                f"derived {e.current} references unknown primitive {e.primitive!r} "
                f"for table {e.table!r}"
            )


@dataclass(frozen=True)
class RenamePlan:
    """A concrete plan to migrate one frame's columns.

    ``rename`` maps a surviving legacy column to its canonical name.
    ``drop`` lists columns removed from this frame: redundant alias twins
    (the survivor is renamed instead), retired and dead columns, and derived
    complements (recomputed in a design matrix, not stored). ``audit`` columns
    are intentionally left in place here; relocating them to metadata / an
    audit artifact is a separate, destination-aware step.
    """

    rename: dict[str, str]
    drop: tuple[str, ...]
    audit_present: tuple[str, ...]


def plan_rename(columns: Iterable[str], *, table: str) -> RenamePlan:
    """Compute the migration plan for a frame with these columns.

    Pure and pandas-free: it decides names, not values. For an alias /
    canonicalize group it keeps the first present member and drops the rest.
    Raises :class:`RegistryError` if applying the plan would collide two
    columns onto one name.
    """
    present = list(columns)
    pset = set(present)
    rename: dict[str, str] = {}
    drop: list[str] = []
    audit_present: list[str] = []
    for e in _entries_for(table):
        here = [c for c in e.current if c in pset]
        if not here:
            continue
        if e.action in _RENAMING:
            assert e.canonical is not None
            rename[here[0]] = e.canonical
            drop.extend(here[1:])  # collapse redundant alias twins
        elif e.action in ("retire", "drop", "derive"):
            drop.extend(here)
        elif e.action == "audit":
            audit_present.extend(here)

    dropset = set(drop)
    outputs = [rename.get(c, c) for c in present if c not in dropset]
    if len(outputs) != len(set(outputs)):
        dupes = sorted({o for o in outputs if outputs.count(o) > 1})
        raise RegistryError(f"rename plan collides columns onto: {dupes}")
    return RenamePlan(rename, tuple(drop), tuple(audit_present))


def export_json() -> str:
    """Serialize the registry as JSON for non-Python migration tooling."""
    import json

    return json.dumps(
        [
            {k: v for k, v in vars(e).items() if v not in ((), "", None)}
            for e in REGISTRY
        ],
        indent=2,
    )


# ---------------------------------------------------------------------------
# Frame transforms (in-memory; no disk I/O)
# ---------------------------------------------------------------------------
# These apply the registry to a pandas frame. They are separated by migration
# step so each carries the correct comparison policy: the mechanical rename is
# bit-for-bit invariant, the numerical-path canonicalization uses a declared
# tolerance, and schema pruning is a distinct step.
def apply_rename(df, *, table: str):  # type: ignore[no-untyped-def]
    """Apply the mechanical rename to a frame: 1:1 renames and byte-identical
    alias collapses only.

    Numerically invariant. Each alias group is checked with ``Series.equals``
    -- value-equal with a matching NaN mask and dtype -- before its redundant
    twins are dropped and the survivor is renamed, so every output column holds
    exactly the input values. Does NOT
    touch numerical-path ``canonicalize`` families, ``retire`` / ``drop`` /
    ``derive`` columns, or ``audit`` relocations -- those are separate steps
    (:func:`canonicalize_numerical_paths`, :func:`plan_rename`'s ``drop`` /
    ``audit_present``). Raises :class:`RegistryError` on an alias mismatch or a
    resulting name collision.
    """
    validate()
    present = set(df.columns)
    rename: dict[str, str] = {}
    drop: list[str] = []
    for e in _entries_for(table):
        here = [c for c in e.current if c in present]
        if not here:
            continue
        if e.action == "rename":
            rename[here[0]] = e.canonical  # type: ignore[assignment]
        elif e.action == "alias":
            # Some producer families naturally materialize only one spelling
            # (for example ContFrag events have no separate classifier-label
            # timestamp).  A lone member is therefore a plain rename.  When
            # two or more aliases are present, equality is mandatory before
            # collapsing them.
            survivor = here[0]
            for twin in here[1:]:
                if not df[survivor].equals(df[twin]):
                    raise RegistryError(
                        f"alias twins are not bit-for-bit equal: {survivor!r} "
                        f"vs {twin!r} (canonical {e.canonical!r})"
                    )
            rename[survivor] = e.canonical  # type: ignore[assignment]
            drop.extend(here[1:])
    out = df.drop(columns=drop).rename(columns=rename)
    if out.columns.duplicated().any():
        dupes = sorted(out.columns[out.columns.duplicated()])
        raise RegistryError(f"rename collides output columns onto: {dupes}")
    return out


def canonicalize_numerical_paths(  # type: ignore[no-untyped-def]
    df, *, table: str, atol=1e-15, rtol=1e-12
):
    """Collapse ``canonicalize`` duplicate families under a declared tolerance.

    A distinct migration step from :func:`apply_rename`: these families are the
    same quantity by two numerical paths, agreeing to floating point but not
    bit-for-bit. Each group is checked -- identical NaN masks, and finite values
    within ``(atol, rtol)`` -- before the twins are dropped and the survivor is
    renamed to the canonical family name. The tolerance must be declared before
    the migration result is inspected. Raises :class:`RegistryError` on a
    mismatch beyond tolerance or a differing missingness pattern.
    """
    import numpy as np

    validate()
    present = set(df.columns)
    rename: dict[str, str] = {}
    drop: list[str] = []
    for e in _entries_for(table):
        if e.action != "canonicalize":
            continue
        here = [c for c in e.current if c in present]
        if not here:
            continue
        if len(here) != len(e.current):
            missing = sorted(set(e.current) - set(here))
            raise RegistryError(
                f"canonicalize family for {e.canonical!r} is incomplete in "
                f"{table!r}; missing {missing}"
            )
        survivor = here[0]
        a = df[survivor].to_numpy(dtype=float)
        for twin in here[1:]:
            b = df[twin].to_numpy(dtype=float)
            if not np.array_equal(np.isnan(a), np.isnan(b)):
                raise RegistryError(
                    f"canonicalize twins differ in missingness: {survivor!r} "
                    f"vs {twin!r} (canonical {e.canonical!r})"
                )
            finite = np.isfinite(a) & np.isfinite(b)
            if not np.array_equal(np.isfinite(a), np.isfinite(b)):
                raise RegistryError(
                    f"canonicalize twins differ in finiteness: {survivor!r} "
                    f"vs {twin!r} (canonical {e.canonical!r})"
                )
            if np.isinf(a).any() or np.isinf(b).any():
                raise RegistryError(
                    f"canonicalize family {e.canonical!r} contains infinity"
                )
            if len(df) and not finite.any():
                # No finite values to compare -> agreement is unverifiable, not
                # confirmed. Refuse rather than collapse two quantities vacuously.
                raise RegistryError(
                    f"canonicalize agreement unverifiable: {survivor!r} vs "
                    f"{twin!r} are all-NaN in this frame (canonical {e.canonical!r})"
                )
            if not np.allclose(a[finite], b[finite], atol=atol, rtol=rtol):
                worst = (
                    float(np.max(np.abs(a[finite] - b[finite])))
                    if finite.any()
                    else 0.0
                )
                raise RegistryError(
                    f"canonicalize twins exceed tolerance: {survivor!r} vs "
                    f"{twin!r} (max abs diff {worst:.2e}, canonical {e.canonical!r})"
                )
        rename[survivor] = e.canonical  # type: ignore[assignment]
        drop.extend(here[1:])
    out = df.drop(columns=drop).rename(columns=rename)
    if out.columns.duplicated().any():
        dupes = sorted(out.columns[out.columns.duplicated()])
        raise RegistryError(f"canonicalize collides output columns onto: {dupes}")
    return out


def canonical_output_name(name: str, *, table: str) -> str:
    """Return the canonical spelling for a raw or derived result column.

    Besides direct registry entries, fitted artifacts deterministically prefix
    predictors with ``coef_`` / ``bse_`` and standardized tables suffix them
    with ``_z``. Keeping this rule here prevents a second hand-maintained rename
    map for model outputs.
    """
    direct: dict[str, str] = {}
    for entry in _entries_for(table):
        if entry.canonical is None:
            continue
        direct.update(dict.fromkeys(entry.current, entry.canonical))
    if name in direct:
        return direct[name]
    for prefix in ("coef_", "bse_"):
        if name.startswith(prefix):
            base = name[len(prefix) :]
            canonical = canonical_output_name(base, table=table)
            return f"{prefix}{canonical}" if canonical != base else name
    if name.endswith("_z"):
        base = name[:-2]
        canonical = canonical_output_name(base, table=table)
        return f"{canonical}_z" if canonical != base else name
    return name


def rename_dynamic_columns(df, *, table: str):  # type: ignore[no-untyped-def]
    """Rename deterministic standardized/coefficient derivatives."""
    rename = {
        column: canonical_output_name(column, table=table)
        for column in df.columns
        if canonical_output_name(column, table=table) != column
    }
    outputs = [rename.get(column, column) for column in df.columns]
    if len(outputs) != len(set(outputs)):
        duplicates = sorted({column for column in outputs if outputs.count(column) > 1})
        raise RegistryError(
            f"dynamic canonicalization collides columns in {table!r}: {duplicates}"
        )
    return df.rename(columns=rename)


def assert_canonical_vocabulary(df, *, table: str) -> None:  # type: ignore[no-untyped-def]
    """Validate column vocabulary, not scientific values or candidate support.

    This is deliberately a naming-layer check: it rejects registered legacy
    spellings, derived-name aliases, duplicate columns, and missing row-key
    *columns*.  Scientific row identity is validated separately by
    :func:`assert_unique_scientific_rows`; table-specific value/candidate-set
    contracts remain the responsibility of their producer and fitter.
    """
    validate()
    columns = list(df.columns)
    if len(columns) != len(set(columns)):
        duplicates = sorted({c for c in columns if columns.count(c) > 1})
        raise RegistryError(f"{table} has duplicate columns {duplicates}")

    forbidden: set[str] = set()
    for entry in _entries_for(table):
        if entry.action != "audit":
            forbidden.update(entry.current)
    surviving = sorted(forbidden & set(columns))
    if surviving:
        raise RegistryError(
            f"{table} retains legacy/retired covariate columns {surviving}"
        )

    dynamic_survivors = sorted(
        column
        for column in columns
        if canonical_output_name(column, table=table) != column
    )
    if dynamic_survivors:
        raise RegistryError(
            f"{table} retains legacy derived result columns {dynamic_survivors}"
        )

    row_keys = _ROW_KEYS.get(table, ())
    missing_keys = sorted(set(row_keys) - set(columns))
    if missing_keys:
        raise RegistryError(f"{table} is missing required row keys {missing_keys}")


def assert_unique_scientific_rows(df, *, table: str) -> None:  # type: ignore[no-untyped-def]
    """Require nonmissing, unique registered row keys for a persisted table.

    This does not claim that the table contains every scientifically required
    column or every candidate category.  Those table-specific invariants are
    intentionally checked by the relevant builder/fitter after this generic
    identity layer.
    """
    keys = list(row_keys(table=table))
    if df[keys].isna().any(axis=None):
        missing_rows = int(df[keys].isna().any(axis=1).sum())
        raise RegistryError(
            f"{table} has {missing_rows} row(s) with missing scientific key values "
            f"in {keys}"
        )
    duplicated = df.duplicated(keys, keep=False)
    if duplicated.any():
        raise RegistryError(
            f"{table} has {int(duplicated.sum())} row(s) with duplicated "
            f"scientific keys {keys}"
        )


def stamp_covariate_schema(df, *, table: str):  # type: ignore[no-untyped-def]
    """Stamp canonical names plus unique scientific row identity.

    The stamp is intentionally narrower than a full scientific-table contract:
    it certifies canonical vocabulary and nonmissing/unique registered row keys.
    Producers and fitters must still validate required columns, candidate sets,
    dtypes, value ranges, and missing-data semantics for their own estimand.
    """
    assert_canonical_vocabulary(df, table=table)
    assert_unique_scientific_rows(df, table=table)
    df.attrs["covariate_schema_version"] = COVARIATE_SCHEMA_VERSION
    df.attrs["covariate_schema_table"] = table
    df.attrs["covariate_schema_scope"] = COVARIATE_SCHEMA_SCOPE
    return df


def require_covariate_schema(df, *, table: str) -> None:  # type: ignore[no-untyped-def]
    """Reject a legacy/mislabeled artifact before scientific computation."""
    version = df.attrs.get("covariate_schema_version")
    labeled_table = df.attrs.get("covariate_schema_table")
    scope = df.attrs.get("covariate_schema_scope")
    if (
        version != COVARIATE_SCHEMA_VERSION
        or labeled_table != table
        or scope != COVARIATE_SCHEMA_SCOPE
    ):
        raise ValueError(
            f"{table} uses a legacy or mismatched covariate schema "
            f"(version={version!r}, table={labeled_table!r}, scope={scope!r}); regenerate "
            "canonical tables before fitting or plotting"
        )
    assert_canonical_vocabulary(df, table=table)
    assert_unique_scientific_rows(df, table=table)


def finalize_covariate_schema(df, *, table: str):  # type: ignore[no-untyped-def]
    """Apply the full migration contract to one producer-owned dataframe."""
    attrs = dict(df.attrs)
    out = canonicalize_numerical_paths(df, table=table, atol=1e-15, rtol=1e-12)
    out = apply_rename(out, table=table)
    out = rename_dynamic_columns(out, table=table)
    plan = plan_rename(out.columns, table=table)
    if plan.drop:
        out = out.drop(columns=list(plan.drop))
    out.attrs.update(attrs)
    return stamp_covariate_schema(out, table=table)
