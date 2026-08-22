"""Covariate naming registry — the single source for the old->new migration.

Generated from the specification in
``docs/design/2026-08-21-covariate-naming-scheme.md``. Each :class:`Entry`
records one or more *current* column names, the migration *action*, and the
*canonical* name. The rename map is **generated** from this registry
(:func:`rename_map`), never hand-authored, so a migration tool has one
explicit action per column and no unresolved "retire one" choices.

Scope: this first pass covers every column the specification gives an explicit
canonical name -- the rename tables, the cleanup families, and the event-table
geometry family (``_cm`` / ``_cm_s`` / ``_fraction`` with ``animal_`` /
``replay_`` subjects), each enumerated against the live tables.

Nothing in this module reads or writes a data table; it is a pure naming
contract. Applying it is the mechanical-rename step of the migration contract.
"""

from __future__ import annotations

import keyword
import re
from collections.abc import Iterable
from dataclasses import dataclass

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
_DIGIT_IDIOM_STEMS = ("last", "k", "l", "p")  # last10, k10, l1, p90


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
    _alias(("current_arm", "location_arm"), "occupancy_well_arm", "stop",
           notes="byte-identical incl. NaN; task state, NaN off-well, never trunk"),
    _alias(("reward_outcome", "is_after_reward"), "is_rewarded_outcome", "stop",
           role="outcome", notes="byte-identical; one nullable 0/1 primitive"),
    _alias(("label_event_start", "event_start"), "event_start_s", "event"),
    _alias(("label_event_end", "event_end"), "event_end_s", "event"),
    _alias(("label_event_duration", "event_duration"), "event_duration_s", "event"),
    _alias(("label_event_midpoint_time", "event_midpoint_time"),
           "event_midpoint_s", "event"),
    _alias(("posterior_mass_total", "nonlocal_weight_sum"),
           "nonlocal_probability_sum", "event", notes="byte-identical Sigma w_t"),
    _alias(("time_since_port_entry", "time_since_last_port_entry"),
           "time_since_last_well_arrival_s", "event",
           notes="same variable assigned twice; elapsed since goal arrival"),
    # -- Cleanup 2: numerical-path duplicates -> canonicalize (tolerance) ---
    Entry(("prev_global_variance", "global_leaf_var_sum_preoutcome_sess"),
          "canonicalize", "qvals",
          canonical="reward_belief_variance_leaf_sum_preoutcome",
          notes="allclose, corr 1.0; declared tolerance atol=1e-15 rtol=1e-12"),
    # (q_port / betadist_mu numerical-path family is in the value section.)
    # -- Cleanup 3: dead model outputs -> drop ------------------------------
    *[Entry((f"depletion_leaf_{i}",), "drop", "qvals",
            notes="constant 1.0; withleaf fit has no depletion")
      for i in range(1, 7)],
    *[Entry((f"depletion_mean_stem_{i}",), "drop", "qvals",
            notes="constant 1.0; withleaf fit has no depletion")
      for i in range(1, 4)],
    # -- Cleanup 4: selection invariants / diagnostics -> audit -------------
    Entry(("whole_event_speed_eligible",), "audit", "event",
          destination="artifact/cohort metadata",
          notes="certifies the immobility/speed filter the cohort passed"),
    Entry(("endpoints_speed_below_thresh",), "audit", "event",
          destination="artifact/cohort metadata",
          notes="legacy endpoint speed certificate"),
    Entry(("nonlocal_fraction",), "audit", "event",
          destination="event_table / event-keyed audit",
          notes="all 1.0 here; eligibility fraction defining each event"),
    Entry(("nospike_fraction",), "audit", "event",
          destination="event_table / event-keyed audit",
          notes="all 0.0 here; eligibility fraction defining each event"),
    Entry(("n_junction_bins",), "rename", "event",
          canonical="n_undecoded_bins", role="diagnostic",
          notes="counts undecoded bins, not junctions; retain at event grain "
                "(sum(arm_count_hard) = n_nonlocal_bins - n_undecoded_bins)"),
    # -- Cleanup 5 / Switch family: complements + switch renames ------------
    Entry(("is_after_switch",), "rename", "stop",
          canonical="did_last_completed_trip_switch_patch", role="provenance",
          notes="completed-action flag; retired as a fitted predictor (rank "
                "deficiency), kept only to define a dedicated switch cohort"),
    Entry(("is_after_stay",), "derive", "stop",
          primitive="did_last_completed_trip_switch_patch",
          derive_expr="did_last_completed_trip_stay_in_patch = 1 - switch",
          notes="verified 1 - is_after_switch exactly"),
    Entry(("is_after_omission",), "derive", "stop",
          primitive="is_rewarded_outcome",
          derive_expr="is_omitted_outcome = 1 - reward",
          notes="derive with the primitive's missingness"),
    _rename("is_post_switch", "is_periswitch_post", "qvals",
            notes="peri-switch alignment flag; hinge partner"),
    _rename("trials_since_switch", "periswitch_index_trials", "qvals",
            notes="signed, two-sided; drops the false 'since'"),
    _rename("recent_switch_rate", "patch_switch_fraction_last5_completed_trips",
            "stop", role="predictor",
            notes="fraction over a fixed 5-completed-trip window, in [0,1]"),
    Entry(("switch_x_current",), "retire", "event_arm", role="interaction",
          notes="interaction on the retired immediate-switch predictor; use the "
                "graded patch_switch_fraction_last5_completed_trips instead"),
    Entry(("switch_x_prev",), "retire", "event_arm", role="interaction",
          notes="sibling of switch_x_current; same retired-predictor interaction"),
    # -- Arm occupancy ------------------------------------------------------
    # (location_arm / current_arm collapse is in Cleanup 1 above.)
    _rename("physical_arm", "modal_observed_track_segment_id", "stop",
            notes="modal observed track segment, trunks included, never NaN"),
    _rename("arm", "candidate_arm", "event_arm",
            role="predictor", notes="the row's arm in the choice set"),
    _rename("location_well", "occupancy_well_id", "stop"),
    _rename("location_state", "occupancy_state", "stop"),
    # -- Relational arm indicators ------------------------------------------
    _rename("is_current", "is_current_arm", "event_arm", role="predictor"),
    _rename("is_prev", "is_prev_arm", "event_arm", role="predictor"),
    _rename("is_next", "is_next_arm", "event_arm", role="outcome"),
    _rename("is_last_rewarded", "is_last_rewarded_arm", "event_arm"),
    _rename("is_last_rewarded_elsewhere", "is_last_rewarded_noncurrent_arm",
            "event_arm",
            notes="omission-only support is structural; keep out of the "
                  "primary hierarchy"),
    _rename("arm_never_visited", "is_candidate_arm_never_visited", "event_arm"),
    _rename("arm_never_visited_patch", "is_patch_never_visited", "event_patch",
            notes="patch-grain column in event_patch_table (tables.py:1685); "
                  "used by BASEc_PATCH (model_specs.py:1233)"),
    _rename("is_same_patch", "is_same_patch_as_current", "event_arm"),
    _rename("dist_to_arm", "distance_to_candidate_arm_cm", "event_arm"),
    _rename("arm_dist_rank", "candidate_arm_distance_rank", "event_arm"),
    _rename("visit_recency", "visit_recency_trials", "event_arm"),
    # -- Behavioral-model value and belief ----------------------------------
    _rename("q_arm", "reward_belief_mean_candidate_arm_preoutcome",
            "event_arm", role="predictor",
            notes="Beta belief mean; reward_belief_ separates it from the "
                  "spatial decoder posterior"),
    _rename("p_reward_arm", "task_reward_probability_candidate_arm",
            "event_arm", notes="programmed contingency, not a belief"),
    _rename("arm_uncertainty", "reward_belief_variance_candidate_arm_preoutcome",
            "event_arm", role="predictor"),
    # q_port{i} and betadist_mu{i+1} are the SAME belief mean by two paths.
    *[Entry((f"q_port{i}", f"betadist_mu{i + 1}"), "canonicalize", "qvals",
            canonical=f"reward_belief_mean_port_{i}",
            notes="old Beta index i maps to port i-1; tol atol=1e-15 rtol=1e-12")
      for i in range(6)],
    *[_rename(f"betadist_var{i + 1}", f"reward_belief_variance_port_{i}", "qvals")
      for i in range(6)],
    *[_rename(f"p_reward_port{i}", f"task_reward_probability_port_{i}", "qvals")
      for i in range(6)],
    *[_rename(f"Qstem{i}", f"decision_value_patch_{i}", "qvals",
              notes="1-based task patches; not a probability")
      for i in range(1, 4)],
    _rename("switch_advantage_mu_preoutcome_sess",
            "relative_value_switch_minus_stay_preoutcome", "qvals",
            role="predictor", notes="manuscript relative value; sign convention"),
    _rename("switch_advantage_preoutcome_sess_Qstem",
            "relative_decision_value_switch_minus_stay_preoutcome", "qvals",
            notes="Qstem-scale contrast, distinct from the reward-belief one"),
    _rename("stay_sibling_mu_preoutcome_sess", "stay_value_preoutcome", "qvals",
            role="predictor", notes="manuscript term; sibling-leaf belief mean"),
    _rename("best_other_patch_mean_mu_preoutcome_sess", "switch_value_preoutcome",
            "qvals", notes="manuscript term; max over two alternative patches"),
    _rename("global_mu_update_abs_postoutcome_sess",
            "reward_belief_mean_update_l1_all_ports_postoutcome", "qvals",
            notes="L1 aggregate over six ports"),
    _rename("rpe", "rpe_postoutcome", "qvals", role="predictor"),
    _rename("rpe_abs", "rpe_postoutcome_abs", "qvals"),
    _rename("q_chosen", "reward_belief_mean_chosen_arm_preoutcome", "qvals"),
    _rename("beta_reward_entropy",
            "reward_belief_differential_entropy_leaf_mean_preoutcome", "qvals",
            notes="Beta differential entropy; can be negative, not Shannon"),
    _rename("is_rewarding", "pump_on_at_bout_mid", "qvals", role="diagnostic",
            notes="mid-bout pump flag (4.8%), not a trial outcome"),
    _rename("leaf_choice_p", "leaf_choice_logp", "qvals",
            notes="a log probability (<=0)"),
    # -- Reward history and time --------------------------------------------
    _rename("cumulative_reward", "reward_count_candidate_arm_last10_known_trials",
            "event_arm", role="predictor",
            notes="rolling 10-known-trial count in [0,5], not cumulative"),
    _rename("reward_rate_arm_k10", "reward_fraction_candidate_arm_last10_visits",
            "event_arm", role="predictor",
            notes="last 10 visits to that arm; a fraction"),
    _rename("reward_rate_trial_k10", "reward_fraction_last10_known_trials",
            "stop", notes="arm-agnostic; last 10 outcome-known trials"),
    *[_rename(f"reward_lag{j}", f"known_reward_outcome_lag_{j}", "stop",
              notes="j-th most recently KNOWN outcome, not a fixed lag")
      for j in range(1, 11)],
    _rename("time_since_stop", "time_since_stop_start_s", "event"),
    _rename("time_since_reward", "time_since_last_reward_s", "event"),
    # (time_since_port_entry alias pair is in Cleanup 1 above.)
    _rename("immobility_time", "immobility_duration_s", "stop",
            notes="stop_end - stop_start; a duration"),
    # -- Event-table geometry: the explicitly-named exceptions --------------
    *[_rename(f"posterior_hpd_width_{c}",
              f"posterior_hpd_arm_count_mean_coverage_{c}pct", "event",
              role="diagnostic",
              notes="prob-weighted mean of per-bin HPD arm count; fractional")
      for c in (50, 80, 95)],
    Entry(("day",), "rename", "event", canonical="session_date",
          role="provenance", notes="YYYYMMDD date, not a day index"),
    # -- Transforms and interactions ----------------------------------------
    _rename("visit_recency_spline1_demeaned",
            "visit_recency_trials_event_leaf_centered_spline_1", "event_arm"),
    _rename("visit_recency_spline2_demeaned",
            "visit_recency_trials_event_leaf_centered_spline_2", "event_arm"),
    _rename("visit_recency_spline3_demeaned",
            "visit_recency_trials_event_leaf_centered_spline_3", "event_arm"),
    _rename("swr_x_visit_recency_spline1_demeaned",
            "swr_x_visit_recency_trials_event_leaf_centered_spline_1",
            "event_arm", role="interaction"),
    _rename("swr_x_visit_recency_spline2_demeaned",
            "swr_x_visit_recency_trials_event_leaf_centered_spline_2",
            "event_arm", role="interaction"),
    _rename("swr_x_visit_recency_spline3_demeaned",
            "swr_x_visit_recency_trials_event_leaf_centered_spline_3",
            "event_arm", role="interaction"),
    # Fit-time arm fixed effects: arm 8 is the dropped reference (see spec).
    *[_rename(f"arm_fe_{k}", f"is_candidate_arm_{k}", "fit_time",
              role="predictor", notes="arm_fe_k = 1 iff arm==k; reference_arm=8")
      for k in range(8)],
    # -- Event-table geometry: units (append _cm / _cm_s / _fraction) -------
    # Replay MAP geometry (trajectory.py), unambiguously replay: append _cm.
    _rename("net_displacement", "net_displacement_cm", "event", role="diagnostic"),
    _rename("path_length", "path_length_cm", "event", role="diagnostic"),
    _rename("spatial_extent", "spatial_extent_cm", "event", role="diagnostic"),
    _rename("step_distance_median", "step_distance_median_cm", "event",
            role="diagnostic"),
    _rename("step_distance_p90", "step_distance_p90_cm", "event", role="diagnostic"),
    _rename("step_distance_max", "step_distance_max_cm", "event", role="diagnostic"),
    _rename("max_ahead", "max_ahead_cm", "event", role="diagnostic"),
    _rename("max_behind", "max_behind_cm", "event", role="diagnostic"),
    _rename("max_abs_ahead_behind", "max_abs_ahead_behind_cm", "event",
            role="diagnostic"),
    _rename("mean_abs_ahead_behind", "mean_abs_ahead_behind_cm", "event",
            role="diagnostic"),
    # Position collides with the animal's position -> subject prefix + unit.
    _rename("start_position", "replay_start_position_cm", "event",
            role="diagnostic", notes="replay MAP position; animal has its own"),
    _rename("end_position", "replay_end_position_cm", "event", role="diagnostic"),
    # Animal head speed (event_table.py): prefix animal_, append _cm_s.
    _rename("max_speed", "animal_max_speed_cm_s", "event", role="diagnostic",
            notes="animal head speed, not replay speed"),
    _rename("mean_speed", "animal_mean_speed_cm_s", "event", role="diagnostic"),
    _rename("start_speed", "animal_start_speed_cm_s", "event", role="diagnostic"),
    _rename("end_speed", "animal_end_speed_cm_s", "event", role="diagnostic"),
    # Replay MAP speed (already replay_-prefixed): append _cm_s.
    _rename("replay_speed_median", "replay_speed_median_cm_s", "event",
            role="diagnostic"),
    _rename("replay_speed_p90", "replay_speed_p90_cm_s", "event", role="diagnostic"),
    _rename("replay_speed_continuous_median",
            "replay_speed_continuous_median_cm_s", "event", role="diagnostic"),
    _rename("ab_slope", "ab_slope_cm_s", "event", role="diagnostic",
            notes="slope of ahead/behind vs time; a replay speed"),
    # Fraction: keep the full word 'fraction'; only path_continuity lacks it.
    _rename("path_continuity", "path_continuity_fraction", "event",
            role="diagnostic"),
)


# ---------------------------------------------------------------------------
# Derived views
# ---------------------------------------------------------------------------
def rename_map() -> dict[str, str]:
    """Map every current column to its canonical name (rename/alias/canonicalize)."""
    out: dict[str, str] = {}
    for e in REGISTRY:
        if e.action in _RENAMING:
            assert e.canonical is not None
            for name in e.current:
                out[name] = e.canonical
    return out


def retired_columns() -> frozenset[str]:
    return frozenset(n for e in REGISTRY if e.action == "retire" for n in e.current)


def dropped_columns() -> frozenset[str]:
    return frozenset(n for e in REGISTRY if e.action == "drop" for n in e.current)


def derived_columns() -> tuple[Entry, ...]:
    return tuple(e for e in REGISTRY if e.action == "derive")


def audit_columns() -> tuple[Entry, ...]:
    return tuple(e for e in REGISTRY if e.action == "audit")


def all_current_columns() -> frozenset[str]:
    """Every legacy column named anywhere in the registry."""
    return frozenset(n for e in REGISTRY for n in e.current)


def canonical_names() -> frozenset[str]:
    """Every canonical name a current column resolves to."""
    return frozenset(rename_map().values())


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
    seen_current: dict[str, Entry] = {}
    seen_canonical: dict[str, Entry] = {}
    for e in REGISTRY:
        if e.action not in ACTIONS:
            raise RegistryError(f"unknown action {e.action!r} for {e.current}")
        if not e.current:
            raise RegistryError(f"entry with no current column: {e}")
        for name in e.current:
            if name in seen_current:
                raise RegistryError(f"current column {name!r} claimed twice")
            seen_current[name] = e

        if e.action in _RENAMING:
            if not e.canonical:
                raise RegistryError(f"{e.action} without canonical: {e.current}")
            if e.action in _COLLAPSE and len(e.current) < 2:
                raise RegistryError(
                    f"{e.action} must collapse >=2 columns: {e.current}"
                )
            if e.action == "rename" and len(e.current) != 1:
                raise RegistryError(f"rename must be 1:1: {e.current}")
            if e.canonical in seen_canonical:
                raise RegistryError(
                    f"canonical {e.canonical!r} produced by two entries"
                )
            seen_canonical[e.canonical] = e
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
            f"canonical names collide with legacy column names: {sorted(collisions)}"
        )

    # Every derived complement must resolve to a stored primitive.
    stored = canonical_names()
    for e in derived_columns():
        if e.primitive not in stored:
            raise RegistryError(
                f"derived {e.current} references unknown primitive {e.primitive!r}"
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


def plan_rename(columns: Iterable[str]) -> RenamePlan:
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
    for e in REGISTRY:
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
def apply_rename(df):  # type: ignore[no-untyped-def]
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
    for e in REGISTRY:
        here = [c for c in e.current if c in present]
        if not here:
            continue
        if e.action == "rename":
            rename[here[0]] = e.canonical  # type: ignore[assignment]
        elif e.action == "alias":
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


def canonicalize_numerical_paths(df, *, atol=1e-15, rtol=1e-12):  # type: ignore[no-untyped-def]
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
    for e in REGISTRY:
        if e.action != "canonicalize":
            continue
        here = [c for c in e.current if c in present]
        if not here:
            continue
        survivor = here[0]
        a = df[survivor].to_numpy(dtype=float)
        for twin in here[1:]:
            b = df[twin].to_numpy(dtype=float)
            if not np.array_equal(np.isnan(a), np.isnan(b)):
                raise RegistryError(
                    f"canonicalize twins differ in missingness: {survivor!r} "
                    f"vs {twin!r} (canonical {e.canonical!r})"
                )
            finite = ~np.isnan(a)
            if not finite.any():
                # No finite values to compare -> agreement is unverifiable, not
                # confirmed. Refuse rather than collapse two quantities vacuously.
                raise RegistryError(
                    f"canonicalize agreement unverifiable: {survivor!r} vs "
                    f"{twin!r} are all-NaN in this frame (canonical {e.canonical!r})"
                )
            if not np.allclose(a[finite], b[finite], atol=atol, rtol=rtol):
                worst = float(np.max(np.abs(a[finite] - b[finite]))) if finite.any() else 0.0
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
