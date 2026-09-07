"""Covariate naming contract -- the runtime half of the 2026-08-21 rename.

The full rename ledger (every legacy spelling, its canonical replacement, and
the decision that connects them) is the appendix of
``docs/design/2026-08-21-covariate-naming-scheme.md``. It is a historical
record: the repository-owned builders emit canonical names directly and nothing
in the pipeline routes data through the legacy translation, so the ledger is
read by people, not by code. Two roles keep *this* module live and permanent:

- **Validation.** :func:`assert_canonical_vocabulary` rejects the ledger's
  legacy spellings (frozen here as :data:`LEGACY_COLUMN_NAMES`), while
  :func:`assert_unique_scientific_rows` checks the registered row identity. The
  persisted stamp certifies exactly those two layers -- not table-specific
  candidate support, dtypes, ranges, or fit eligibility.
- **Vendored ingestion.** The behavioral-model adapter in
  ``analysis.behavior.qvals_table`` translates the external model's native
  vocabulary (``Q``/``betadist_mu``/``Qstem`` ...) into the canonical
  reward-belief schema. That translation still runs on every build, so the
  ``qvals`` rows of the ledger -- and only those -- stay here as executable
  :class:`Entry` records feeding :func:`canonicalize_numerical_paths` /
  :func:`apply_rename` / :func:`rename_dynamic_columns` / :func:`plan_rename`,
  the one ingestion boundary.

Nothing in this module reads or writes a persisted table; it is a pure naming
contract. :func:`finalize_covariate_schema` (the full transform) is retained
only for the migration-parity checking tools and is not part of the production
pipeline; with the non-``qvals`` rows now in the design doc it renames nothing
outside ``qvals`` and reduces to the stamp there.
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
_DIGIT_IDIOM_STEMS = ("last", "k", "l", "p", "log")  # last10, k10, l1, p90, log1p

# The version value is centralized in analysis/replay/versions.py (with the
# related cache/manifest literals, reviewed together); re-exported here so the
# many importers of covariate_registry.COVARIATE_SCHEMA_VERSION are unaffected.
from continuum_swr_replay.analysis.replay.versions import (  # noqa: E402
    COVARIATE_SCHEMA_VERSION,
    assert_versions_compatible,
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


# ---------------------------------------------------------------------------
# The vendored-ingestion rows
# ---------------------------------------------------------------------------
# The ``qvals`` slice of the ledger, kept executable because the external
# behavioral model still emits its native vocabulary on every build. Every
# other grain's rows live in the design doc named in the module docstring.
REGISTRY: tuple[Entry, ...] = (
    Entry(
        ("prev_global_variance", "global_leaf_var_sum_preoutcome_sess"),
        "canonicalize",
        "qvals",
        canonical="reward_belief_variance_leaf_sum_preoutcome",
        notes="allclose, corr 1.0; declared tolerance atol=1e-15 rtol=1e-12",
    ),
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
)

# ---------------------------------------------------------------------------
# The frozen deny-list
# ---------------------------------------------------------------------------
# Every legacy spelling the 2026-08-21 ledger retired, across all table grains,
# excluding the ``audit`` relocations (those columns are allowed to stay put).
# Extracted once from the ledger at the commit that moved it into
# ``docs/design/2026-08-21-covariate-naming-scheme.md``:
#
#     {n for e in REGISTRY if e.action != "audit" for n in e.current}
#
# The ledger's grains are collapsed into one set on purpose: no canonical name
# of any table is a legacy name of another (the ledger's own
# no-collision invariant, checked table-by-table, holds globally), so a single
# set rejects exactly the same columns a per-grain deny-list would.
LEGACY_COLUMN_NAMES: frozenset[str] = frozenset(
    """
    Q1 Q2 Q3 Q4 Q5 Q6 Qstem1 Qstem2 Qstem3 ab_slope arm arm_dist_rank arm_fe_0
    arm_fe_1 arm_fe_2 arm_fe_3 arm_fe_4 arm_fe_5 arm_fe_6 arm_fe_7
    arm_never_visited arm_never_visited_patch arm_uncertainty
    best_other_patch_mean_mu_preoutcome_sess beta_reward_entropy betadist_mu1
    betadist_mu2 betadist_mu3 betadist_mu4 betadist_mu5 betadist_mu6 betadist_var1
    betadist_var2 betadist_var3 betadist_var4 betadist_var5 betadist_var6
    block_x_current chosen_leaf_mu_update_abs_postoutcome_sess
    chosen_leaf_var_preoutcome_sess chosen_patch_mu_update_abs_postoutcome_sess
    continuous_x_current continuous_x_next continuous_x_prev cumulative_reward
    cumulative_reward_patch current_arm current_x_last_rewarded
    currentpatch_x_switchrate currentpatch_x_tsc currentpatch_x_tss day
    depletion_leaf_1 depletion_leaf_2 depletion_leaf_3 depletion_leaf_4
    depletion_leaf_5 depletion_leaf_6 depletion_mean_stem_1 depletion_mean_stem_2
    depletion_mean_stem_3 dist_to_arm end_position end_speed entropy_x_current
    event_duration event_end event_midpoint_time event_start
    global_leaf_var_sum_preoutcome_sess global_mu_update_abs_postoutcome_sess
    global_mu_update_signed_postoutcome_sess immobility_time is_after_omission
    is_after_reward is_after_stay is_after_switch is_current is_last_rewarded
    is_last_rewarded_elsewhere is_last_rewarded_patch_elsewhere is_next
    is_post_switch is_prev is_rewarding is_same_patch label_event_duration
    label_event_end label_event_midpoint_time label_event_start leaf_choice_p
    location_arm location_state location_well max_abs_ahead_behind max_ahead
    max_behind max_speed mean_abs_ahead_behind mean_speed mua_x_current
    mua_x_last_rewarded mua_x_next mua_x_prev n_MUA n_SWR n_events n_junction_bins
    n_non_local n_non_local_SWR n_non_local_no_SWR net_displacement
    nonlocal_weight_sum omission_x_last_rewarded p_reward_arm p_reward_arm_patch
    p_reward_best p_reward_current p_reward_port0 p_reward_port1 p_reward_port2
    p_reward_port3 p_reward_port4 p_reward_port5 path_continuity path_length
    physical_arm posterior_hpd_width_50 posterior_hpd_width_80
    posterior_hpd_width_95 posterior_mass_total postsw_x_current
    prev_global_variance q_arm q_arm_patch q_chosen q_port0 q_port1 q_port2 q_port3
    q_port4 q_port5 recent_switch_rate relval_x_current replay_frac
    replay_frac_current_patch replay_frac_non_swr replay_frac_noncurrent
    replay_frac_residual replay_frac_residual_noncurrent replay_frac_swr
    replay_speed_continuous_median replay_speed_median replay_speed_p90 reward_lag1
    reward_lag10 reward_lag2 reward_lag3 reward_lag4 reward_lag5 reward_lag6
    reward_lag7 reward_lag8 reward_lag9 reward_outcome reward_rate_arm_k10
    reward_rate_arm_k10_patch reward_rate_trial_k10 rpe rpe_abs rpe_x_current
    spatial_extent start_position start_speed stay_sibling_mu_preoutcome_sess
    step_distance_max step_distance_median step_distance_p90
    switch_advantage_mu_preoutcome_sess switch_advantage_preoutcome_sess_Qstem
    switch_x_current switch_x_current_patch switch_x_prev switchrate_x_current
    swr_x_arm_never_visited swr_x_arm_uncertainty swr_x_current swr_x_current_patch
    swr_x_is_last_rewarded_elsewhere swr_x_last_rewarded
    swr_x_last_rewarded_patch_elsewhere swr_x_next swr_x_prev swr_x_prev_patch
    swr_x_reward_rate_arm_k10 swr_x_visit_recency_demeaned
    swr_x_visit_recency_spline1_demeaned swr_x_visit_recency_spline2_demeaned
    swr_x_visit_recency_spline3_demeaned time_since_last_port_entry
    time_since_port_entry time_since_reward time_since_stop time_x_current
    time_x_next time_x_prev trials_since_switch tsc_x_current tss_x_current
    tsspostsw_x_current unchosen_patches_mu_update_abs_postoutcome_sess
    update_x_current value_diff_patch visit_recency visit_recency_demeaned
    visit_recency_patch visit_recency_spline1_demeaned
    visit_recency_spline2_demeaned visit_recency_spline3_demeaned
    """.split()
)

# The subset of the deny-list that was never given a canonical replacement
# (retired predictors, dead model outputs, derived complements). A decorated
# spelling of one of these -- ``coef_switch_x_current`` -- has no canonical form
# to rewrite it to, so the decorated-name check below skips them.
_LEGACY_WITHOUT_CANONICAL: frozenset[str] = frozenset(
    """
    depletion_leaf_1 depletion_leaf_2 depletion_leaf_3 depletion_leaf_4
    depletion_leaf_5 depletion_leaf_6 depletion_mean_stem_1 depletion_mean_stem_2
    depletion_mean_stem_3 is_after_omission is_after_stay switch_x_current
    switch_x_current_patch switch_x_prev
    """.split()
)
_LEGACY_RENAMED: frozenset[str] = LEGACY_COLUMN_NAMES - _LEGACY_WITHOUT_CANONICAL

# ---------------------------------------------------------------------------
# Derived views
# ---------------------------------------------------------------------------


def _entries_for(table: str) -> tuple[Entry, ...]:
    """Return ledger entries for one persisted table/grain.

    Column spellings are not globally meaningful: for example ``arm`` is the
    choice-set row in ``event_arm`` but the realized destination in ``qvals``.
    Requiring a table makes it impossible to apply one grain's semantic rename
    to another merely because the legacy spelling happens to match. Only
    ``qvals`` still has rows; the other registered grains resolve to an empty
    tuple, which is what "the builders already emit canonical names" means.
    """
    if table not in _ROW_KEYS:
        raise RegistryError(
            f"unknown covariate-registry table {table!r}; expected {sorted(_ROW_KEYS)}"
        )
    return tuple(e for e in REGISTRY if e.table == table)


def derived_columns(*, table: str | None = None) -> tuple[Entry, ...]:
    entries = REGISTRY if table is None else _entries_for(table)
    return tuple(e for e in entries if e.action == "derive")


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


# ---------------------------------------------------------------------------
# Frame transforms (in-memory; no disk I/O)
# ---------------------------------------------------------------------------
# These apply the ledger to a pandas frame. They are separated by migration
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


def _undecorate(name: str) -> frozenset[str]:
    """Every stem a deterministic result decoration could have been built from.

    Fitted artifacts prefix predictors with ``coef_`` / ``bse_`` and
    standardized tables suffix them with ``_z``; a legacy predictor therefore
    survives into a result table under a decorated spelling. Stripping the
    decorations back off is what lets one deny-list cover both.
    """
    stems = {name}
    for prefix in ("coef_", "bse_"):
        if name.startswith(prefix):
            stems |= _undecorate(name[len(prefix) :])
    if name.endswith("_z"):
        stems |= _undecorate(name[:-2])
    return frozenset(stems)


def assert_canonical_vocabulary(df, *, table: str) -> None:  # type: ignore[no-untyped-def]
    """Validate column vocabulary, not scientific values or candidate support.

    This is deliberately a naming-layer check: it rejects registered legacy
    spellings, derived-name aliases, duplicate columns, and missing row-key
    *columns*.  Scientific row identity is validated separately by
    :func:`assert_unique_scientific_rows`; table-specific value/candidate-set
    contracts remain the responsibility of their producer and fitter.
    """
    validate()
    _entries_for(table)
    columns = list(df.columns)
    if len(columns) != len(set(columns)):
        duplicates = sorted({c for c in columns if columns.count(c) > 1})
        raise RegistryError(f"{table} has duplicate columns {duplicates}")

    surviving = sorted(LEGACY_COLUMN_NAMES & set(columns))
    if surviving:
        raise RegistryError(
            f"{table} retains legacy/retired covariate columns {surviving}"
        )

    dynamic_survivors = sorted(
        column for column in columns if _undecorate(column) & _LEGACY_RENAMED
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
    # An unstamped frame carries no version to compare, so it cannot reach the
    # centralized check; it is legacy by construction.
    if version is None or labeled_table != table or scope != COVARIATE_SCHEMA_SCOPE:
        raise ValueError(
            f"{table} uses a legacy or mismatched covariate schema "
            f"(version={version!r}, table={labeled_table!r}, scope={scope!r}); regenerate "
            "canonical tables before fitting or plotting"
        )
    assert_versions_compatible(covariate_schema_version=version)
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
