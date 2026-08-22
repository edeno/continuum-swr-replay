# Covariate naming specification

**Status:** approved specification; implementation pending. **Version:** 1.1.
**Date:** 2026-08-21.

The canonical naming specification for columns across `event_arm_table`,
`stop_table`, `qvals_table`, and `event_table`. Every definition below was
checked against the producing code and recomputed from the materialized
tables by four independent verification passes; every quantitative claim was
measured, not inferred. Subsequent reviews corrected the arm fixed-effect
renumbering, distinguished exact aliases from numerical-path duplicates,
separated selection invariants from dead outputs, made aggregation and support
explicit, and tightened the migration contract — those corrections are folded
in here.

The pass surfaced a problem underneath the naming: the tables carry
**duplicate, dead, and invariant columns** that a rename cannot fix. Those,
and the migration discipline, are as important as the names.

These names have not yet been applied to generated schemas. Implementation
must follow the [migration contract](#migration-contract); until that
migration is complete, current artifacts retain their legacy column names.

---

## Conventions

Names should expose the **scientifically load-bearing** distinctions; a
machine-readable registry (below) holds everything else. Forcing every
attribute into the string produces names like
`patch_switch_fraction_last5_completed_trips` that are long *and* still
incomplete. A loose grammar orders the components that do belong in the name:

```
<quantity>_<subject/scope>_<window>_<availability>_<transform>_<unit>
```

Not every field appears in every name. Examples:

- `reward_belief_mean_candidate_arm_preoutcome`
- `reward_fraction_candidate_arm_last10_visits`
- `visit_recency_trials_event_leaf_centered`
- `animal_speed_mean_cm_s`
- `replay_path_length_cm`

The supporting rules:

1. **Expose load-bearing distinctions, not every grain.** Mark the unit a
   column varies over *when confusing it with another grain would mislead*
   (`is_current` → `is_current_arm`, because a patch twin exists). Let the
   registry carry the rest.
2. **Self-describing beats provenance.** Name the quantity, not the file it
   came from. A blanket `beta_` prefix was rejected: `beta_*` already means
   the regression coefficient ~130× in `src/`, and β is the softmax inverse
   temperature inside `Qstem = β·Q + b`. `Qstem1` → `decision_value_patch_1`.
3. **Timing is named where the outcome clock moves the value.** `_preoutcome`
   / `_postoutcome`. Drop `_sess` — it marks within-session computation scope,
   which every exported column already shares (it is **not** session-scoped
   normalization). **Field order:** an *aggregation* that defines the quantity
   (`sum`, `l1`, `mean`, `max` over a set) is part of the quantity and precedes
   availability; a *post-hoc value transform* on a single scalar (`abs`,
   `signed`, `log`, `_z`) follows it. So `rpe_postoutcome_abs` (abs transforms
   the scalar RPE) but `reward_belief_mean_update_l1_all_ports_postoutcome`
   (the L1 is the aggregation) — both correct, one rule.
4. **Units and window parameters live in the name.** Seconds, centimetres,
   trials, visit count, log-scale. The failure mode is guessable-the-wrong-way:
   `reward_rate_arm_k10` counts the last 10 *visits*, not trials. **Digit
   boundary:** a numeric field takes a leading `_` when it identifies a member
   of an enumerated family (`port_0`, `patch_1`, `spline_1`, `lag_1`,
   `coverage_50`, `arm_0`); it stays glued only inside a window idiom where the
   number reads as part of the word (`last10`, `k10`).
5. **"Rate" means per unit time; a rolling mean of a binary outcome is a
   "fraction."** These are dimensionless proportions. `reward_rate_*` →
   `reward_fraction_*`, `switch_rate_*` → `switch_fraction_*`.
6. **One quantity, one column; one primitive per complement.** Retire exact
   aliases. For exact complements (switch/stay, reward/omission), store one
   primitive and derive the other only where its approved role requires it.
   Storage does not imply predictor eligibility: the immediate completed-trip
   switch flag is behavioral metadata and an optional cohort selector, not a
   fitted covariate.
7. **Booleans read as predicates; interactions read as products.** Use
   `is_`/`has_`/`did_` as the quantity requires, and
   `{modifier}_x_{base}` for interactions. `did_` is important for a completed
   action such as `did_last_completed_trip_switch_patch`; forcing that fact
   into `is_after_*` changes its apparent meaning.
8. **Use an established scientific term instead of spelling out an equation.**
   A primary covariate such as `stay_value_preoutcome` should match the
   manuscript, while the registry records its exact sibling-arm formula.
   Formula-like names are appropriate for generated transforms and design
   columns when their construction changes interpretation, not as a substitute
   for a registry. There is no hard character limit; every token must resolve
   a plausible scientific ambiguity. **When a manuscript-term name denotes the
   same quantity as a systematically-named column** — `stay_value_preoutcome`
   *is* a `reward_belief_mean` of the sibling leaf — the registry must record
   that equivalence (shared construct and units), so a reader comparing
   coefficients on the two can see they are the same kind of quantity despite
   the different vocabularies.

### Machine-readable column registry

The name carries the load-bearing distinctions; a per-column registry carries
the rest, and is what the rename map is *generated from* (not hand-authored).
Each exported column gets:

- row grain
- entity / support
- availability clock
- timebase / timestamp origin (for example, Unix time, session-relative time,
  or event-relative time)
- aggregation
- units
- window
- transform / standardization reference
- valid range
- missing-value meaning
- dtype
- structural / off-support encoding (for example, the deliberate trunk zero
  in a leaf-only predictor, distinct from an observed zero and from missing)
- role: predictor / outcome / interaction / diagnostic / provenance
- canonical producer / computation path
- for indicator families: the **reference level** (see `arm_fe`)
- for indexed families: the **0-based-vs-1-based** indexing decision

This is more reliable than forcing all of it into the column name, and it
makes the ID and reference-level conventions explicit rather than implied by a
suffix.

---

## Cleanup — five kinds, not one

The verification found columns that a rename cannot fix. They divide five
ways; conflating them (as an earlier draft did, calling everything constant
"dead") is itself an error — a constant field can be a *selection invariant*
that certifies how the table was filtered.

### 1. Exact aliases — retire the old names

Byte-identical values under two names; a later edit makes them silently
diverge. The canonical winner is explicit so the generated migration map has
no unresolved "retire one" choices.

| Old columns | Evidence | Canonical column |
|---|---|---|
| `current_arm`, `location_arm` | byte-identical incl. NaN, all rows (stop, event, event_arm) | `occupancy_well_arm` |
| `reward_outcome`, `is_after_reward` | byte-identical, same NaN mask (stop, event_arm) | nullable 0/1 indicator `is_rewarded_outcome` |
| `label_event_start/end/duration/midpoint_time`, `event_start/end/duration/midpoint_time` | byte-identical, 4 columns (event) | `event_start_s`, `event_end_s`, `event_duration_s`, `event_midpoint_s` |
| `posterior_mass_total`, `nonlocal_weight_sum` | byte-identical Σwₜ (event) | `nonlocal_probability_sum` |
| `time_since_port_entry`, `time_since_last_port_entry` | same variable assigned twice (event_arm, event) | `time_since_last_well_arrival_s` |

### 2. Semantic duplicates from separate numerical paths — canonicalize deliberately

These pairs represent the same scientific quantity but are **not** proven
byte-identical. Consolidating them is a small computational-path change, not a
pure rename. It therefore gets its own commit and a predeclared floating-point
comparison tolerance.

| Old columns | Evidence | Canonical column |
|---|---|---|
| `prev_global_variance`, `global_leaf_var_sum_preoutcome_sess` | `allclose` True, corr 1.0 | `reward_belief_variance_leaf_sum_preoutcome` |
| `q_port0..5`, `betadist_mu1..6` | max difference 2e-16; byte-exact 62%; two code paths and an off-by-one index | one 0-based family `reward_belief_mean_port_0..5`; old `betadist_mu{i}` maps to new `port_{i-1}` |

### 3. Disabled / dead model outputs — remove

Constant because the model that would vary them is off.

| Columns | Status |
|---|---|
| `depletion_leaf_1..6`, `depletion_mean_stem_1..3` | constant 1.0 (9 cols) — the `withleaf` fit has no depletion |

### 4. Selection invariants & diagnostics — preserve at the correct grain, do **not** delete

Constant *because they certify how the cohort was filtered*, or diagnostics
that document a build. Artifact-wide selection invariants belong in artifact
metadata or a cohort-audit table. Event-varying diagnostics remain keyed
one-to-one to the event, even if their redundant broadcast across candidate
rows is removed from a modeling table. They carry real provenance and must not
simply vanish.

| Columns | Destination and reason |
|---|---|
| `whole_event_speed_eligible`, `endpoints_speed_below_thresh` | artifact/cohort metadata: certify the immobility/speed filter the cohort passed |
| `nonlocal_fraction` (all 1.0 here), `nospike_fraction` (all 0.0 here) | retain in `event_table` or an event-keyed audit: eligibility fractions defining each event; omit a candidate-row broadcast only when the event-key join is exact |
| `n_junction_bins` → `n_undecoded_bins` | retain at event grain: this is an exact response-accounting term in `sum(arm_count_hard) = n_nonlocal_bins - n_undecoded_bins`; it counts undecoded bins, not junctions. It may be omitted from candidate rows only when preserved by an exact event-key join |

### 5. Exact complements — store one primitive, derive the other

The derived value preserves the primitive's missingness: before the relevant
trip or outcome is known, both are missing rather than being treated as zero.

| Old pair | Canonical primitive | Derived counterpart and disposition |
|---|---|---|
| `is_after_switch` / `is_after_stay` | `did_last_completed_trip_switch_patch` | `did_last_completed_trip_stay_in_patch = 1 − switch` |
| `is_after_reward` / `is_after_omission` | nullable 0/1 `is_rewarded_outcome` | `is_omitted_outcome = 1 − reward` |

`did_last_completed_trip_switch_patch` is retained to describe behavioral
context and, when explicitly requested, to select a switch-restricted cohort.
It is **retired as a fitted regression covariate**. Its stay complement may be
derived for metadata, display, or cohort selection, but neither complement is
part of a primary design matrix.

---

## Switch family — correctness hazard

Two near-identical names for unrelated quantities. `is_after_switch` is ours
(`_helpers.py:525`): the immediately preceding completed trip changed patch.
`is_post_switch` (`qvals_table`) is a *peri-switch alignment* flag — 1 when
the *nearest* switch lies behind, 0 when ahead. Base rates **8.6%** vs
**68.9%**, correlation **0.19**, join-independent. Selecting the wrong one
yields a plausible fit and no error.

`trials_since_switch` is worse than ambiguous: a *signed* index clipped to
[−20, +20], **31.1%** negative — meaning trials *until the next* switch.
"Since" is wrong for a third of the data. It equals `is_post_switch ≥ 0` on
**100.00%** of rows: the two are the sign and magnitude of one quantity, kept
as a pair so `model_specs.py:1025` can fit a hinge with separate pre- and
post-switch slopes.

| Current | Canonical | Why |
|---|---|---|
| `is_after_switch` | `did_last_completed_trip_switch_patch` | The immediately preceding completed trip changed patch. Retain as behavioral metadata or an explicit switch-cohort selector; retire from fitted predictor sets. This is a completed-action predicate, not an elapsed "after switch" state. |
| `is_post_switch` | `is_periswitch_post` | Names the alignment it encodes; the hinge partner, not an event flag. |
| `trials_since_switch` | `periswitch_index_trials` | Signed and two-sided; drops the false "since". |
| `is_after_stay` | *derive only for metadata/selection* | Verified `1 − is_after_switch` exactly. Do not materialize it as a second predictor. |
| `recent_switch_rate` | `patch_switch_fraction_last5_completed_trips` | Verified: switch count over a fixed five-completed-trip window, in [0,1] — a fraction, not a rate. The window unit is load-bearing and stays in the name. |
| `switch_x_current` / `switch_x_prev` | *retire* | Interactions with the retired immediate-switch predictor. Use the graded `patch_switch_fraction_last5_completed_trips` construct for an approved switch-history model instead. |

### Immediate switch is not a nuisance adjustment

The immediate switch flag was previously carried through the per-stop reward
models as a generic adjustment. That changes the estimability of an otherwise
coherent reward-history question: after the ten-known-outcome complete-case
filter, one observed session retained 310 usable stops and variation in all ten
reward lags, but the switch flag was constant. The intercept plus ten reward
lags were estimable; adding the constant switch column reduced the design to
rank 11 of 12 and caused the entire session to be rejected. Sparse switch
support can likewise disappear inside a cross-validation fold.

Accordingly, the migration must remove the immediate switch flag from R1, R2,
Rlag, and other general-purpose fitted predictor sets. It must also retire the
formal per-bin switch-main-effect extension rather than preserving it under a
new name. A dedicated analysis may still use the metadata flag to define a
prespecified switch-only cohort; that is a cohort definition, not a nuisance
coefficient. The canonical graded switch-history predictor, when such a
question is scientifically motivated, is
`patch_switch_fraction_last5_completed_trips`.

---

## Arm occupancy — correctness hazard

Verified in code: `location_arm` is a *task state* — the well being visited,
whole-window rollup, `NaN` off-well (34.9% of stops), never a trunk.
`current_arm` is a byte-identical *alias* of it. `physical_arm` is the
different construct — the modal observed track segment, trunks included,
**never NaN**. So the column named `current_arm` is NaN a third of the time
and cannot be a trunk, while the one that actually holds the animal's location
is `physical_arm`. (See also
[`known_issues/current_arm_trunk_fallback.md`](../known_issues/current_arm_trunk_fallback.md).)

| Current | Canonical | Why |
|---|---|---|
| `location_arm` | `occupancy_well_arm` | The task state (well occupancy); `NaN` off-well now reads as intended. |
| `current_arm` | *retire* | Byte-identical alias. |
| `physical_arm` | `modal_observed_track_segment_id` | Observational, includes trunks (0–8), never NaN — a *position*, not a state. |
| `arm` | `candidate_arm` | In `event_arm_table` this is the row's arm in the choice set — the response dimension. |
| `location_well` / `location_state` | `occupancy_well_id` / `occupancy_state` | Split the overloaded `location_` stem by what each holds. |

---

## Relational arm indicators — legibility

Arm grain reads as the unmarked default, which is why
`arm_never_visited_patch` parses as "arm…patch". Marking both grains fixes the
family.

| Current | Canonical | Why |
|---|---|---|
| `is_current` | `is_current_arm` | Its patch twin already names the grain. |
| `is_prev` / `is_next` | `is_prev_arm` / `is_next_arm` | Same asymmetry; `is_next_arm` is response-only. |
| `is_last_rewarded` | `is_last_rewarded_arm` | Family consistency. |
| `is_last_rewarded_elsewhere` | `is_last_rewarded_noncurrent_arm` | Names the *definition* (`is_last_rewarded ∧ ¬is_current`), not its support. Its omission-only support is **structural** (a reward makes last-rewarded == current, so the residual is identically zero on rewarded stops — `tables.py:1402`), so any coefficient is within-omission. **Keep it out of the primary hierarchy** — it is thin, feeds only pooled contrast rungs, and the last-rewarded family is currently refused as unidentified. |
| `arm_never_visited` | `is_candidate_arm_never_visited` | Boolean without an `is_`. |
| `arm_never_visited_patch` | `is_patch_never_visited` | Removes the "arm…patch" collision. |
| `is_same_patch` | `is_same_patch_as_current` | "Same as what" is load-bearing and unstated. |
| `dist_to_arm` / `arm_dist_rank` | `distance_to_candidate_arm_cm` / `candidate_arm_distance_rank` | Verified cm; `candidate` distinguishes the response dimension from occupancy and observed track segment. |
| `visit_recency` | `visit_recency_trials` | Verified integer trials-ago, not seconds. |

---

## Behavioral-model value and belief — correctness hazard

**`q_arm` and `p_reward_arm` are not two flavours of value — one is belief,
the other is ground truth.** Verified: `q_arm` sources from `q_port*`, the
Beta posterior *mean* (the animal's estimate, 96,431 distinct values).
`p_reward_arm` sources from `p_reward_port*`, the *programmed contingency* —
the source ports take only {0.2, 0.5, 0.8, 1.0}. A coefficient on one says
*replay tracks belief*; on the other, *replay tracks the true contingency*.

Caveat from the check: in the shipped `event_arm_table`, trunk arms are filled
`0.0`, so `p_reward_arm` holds five values {0, 0.2, 0.5, 0.8, 1.0} and a naive
`==0` filter is contaminated by structural trunk zeros. Same for `q_arm` and
`arm_uncertainty`.

Two more misleading names in this block: `is_rewarding` reads as "the trial
was rewarded" but is a mid-bout pump-active flag — **4.8%** True against a
52.5% reward rate; the code comment forbids gating on it. And `leaf_choice_p`
is a *natural-log* probability (range [−1.79, 0]) sitting in a `_p` family
where `stem_choice_p`, `stem_stay_p` are genuine [0,1] probabilities.

The reward-belief names need an explicit namespace because this repository
also contains a spatial decoder posterior. Bare `posterior_*` names make those
two unrelated distributions easy to confuse. The manuscript calls these
quantities the animal's beliefs about port reward probabilities, so the
canonical namespace is `reward_belief_*`.

The indexed families also need an explicit mapping. Canonical arm and port IDs
are zero-based: arms are `arm_0..8` and reward ports are `port_0..5`.
`q_port0..5`, `betadist_mu1..6`, and `betadist_var1..6` therefore map to
`reward_belief_{mean,variance}_port_0..5`, with old Beta index `i` mapping to
new port index `i - 1`. Task patches remain the established one-based IDs
`patch_1..3`, so `Qstem1..3` maps to `decision_value_patch_1..3`. The registry
must record these index bases; a migration generator must encode the mapping
explicitly rather than infer it from a suffix.

| Current | Canonical | Why |
|---|---|---|
| `q_arm` | `reward_belief_mean_candidate_arm_preoutcome` | Mean of the Beta reward-probability belief for the candidate arm. `reward_belief` distinguishes it from the spatial decoder posterior. |
| `p_reward_arm` | `task_reward_probability_candidate_arm` | The programmed contingency. `task_` keeps it from reading as a belief. |
| `arm_uncertainty` | `reward_belief_variance_candidate_arm_preoutcome` | Variance of the same Beta reward-probability belief. |
| `q_port0..5` / `betadist_mu1..6` | `reward_belief_mean_port_0..5` | One canonical 0-based family; old Beta index `i` maps to port `i - 1`. |
| `betadist_var1..6` | `reward_belief_variance_port_0..5` | The matching 0-based variance family; old Beta index `i` maps to port `i - 1`. |
| `p_reward_port0..5` | `task_reward_probability_port_0..5` | The per-port programmed contingency, the ground-truth counterpart of `reward_belief_mean_port_0..5`; `task_` keeps it from reading as a belief. Same 0-based indexing as the belief family. |
| `Qstem{i}` | `decision_value_patch_{i}` | Verified range −0.98 to 19.33 — says the scale out loud, cannot read as a probability. |
| `switch_advantage_mu_preoutcome_sess` | `relative_value_switch_minus_stay_preoutcome` | The manuscript's relative value: Switch value minus Stay value. The registry defines Switch as the larger mean belief of the two alternative patches and Stay as the sibling/upcoming-port belief; those asymmetric formulas do not belong in the column name. |
| `switch_advantage_preoutcome_sess_Qstem` | `relative_decision_value_switch_minus_stay_preoutcome` | The analogous contrast on the Qstem decision-value scale (range −11.78 to 0.76), explicitly distinct from the manuscript's reward-belief relative value. |
| `stay_sibling_mu_preoutcome_sess` | `stay_value_preoutcome` | The manuscript term. The registry records that Stay is the sibling/upcoming leaf's reward-belief mean, not a patch mean. |
| `best_other_patch_mean_mu_preoutcome_sess` | `switch_value_preoutcome` | The manuscript term. The registry records that Switch is the maximum of the two alternative-patch mean reward beliefs. |
| `global_leaf_var_sum_preoutcome_sess` / `prev_global_variance` | `reward_belief_variance_leaf_sum_preoutcome` | Sum of the six leaf-port reward-belief variances. |
| `global_mu_update_abs_postoutcome_sess` | `reward_belief_mean_update_l1_all_ports_postoutcome` | Sum of absolute reward-belief mean changes over all six ports: an L1 aggregate, not a generic update. |
| `rpe` / `rpe_abs` | `rpe_postoutcome` / `rpe_postoutcome_abs` | `rpe` is standard; availability precedes the absolute-value transform under the naming grammar. |
| `q_chosen` | `reward_belief_mean_chosen_arm_preoutcome` | Consistent with the candidate-arm reward-belief column. |
| `beta_reward_entropy` | `reward_belief_differential_entropy_leaf_mean_preoutcome` | Mean of six per-leaf Beta differential entropies. It can be negative and is not categorical Shannon entropy; the registry records the Beta family. |
| `is_rewarding` | `pump_on_at_bout_mid` | A mid-bout pump flag (4.8%), not a trial outcome. |
| `leaf_choice_p` | `leaf_choice_logp` | A log probability (≤0), not a probability. |

---

## Reward history and time — legibility

**`cumulative_reward` is not cumulative.** The code computes a *rolling
10-trial window* count (`cumsum[t] − cumsum[t−10]`), confirmed by the data:
integer, range [0, 5]. The current name incorrectly implies an unbounded
cumulative count.

| Current | Canonical | Why |
|---|---|---|
| `cumulative_reward` | `reward_count_candidate_arm_last10_known_trials` | Not cumulative — a rolling count over the ten most recent outcome-known trials. "Known" matches the outcome clock and the sibling fraction name. |
| `reward_rate_arm_k10` | `reward_fraction_candidate_arm_last10_visits` | Verified: last 10 *visits to that arm*; a fraction, not a rate. |
| `reward_rate_trial_k10` | `reward_fraction_last10_known_trials` | Arm-agnostic; denominator is the last 10 outcome-known trials. |
| `reward_lag{j}` | `known_reward_outcome_lag_{j}` | The j-th most recently known outcome at query time, not a fixed trial-relative lag (`lag1` may be the just-arrived trip after feedback). The timing distinction is load-bearing enough to name. |
| `reward_outcome` / `is_after_reward` | `is_rewarded_outcome` | One nullable 0/1 primitive, finite only after feedback is observable; derive omission with the same missingness. |
| `time_since_stop`, `time_since_reward` | `time_since_stop_start_s`, `time_since_last_reward_s` | Names the reference time and unit rather than only appending a suffix. |
| `time_since_port_entry` / `time_since_last_port_entry` | `time_since_last_well_arrival_s` | Byte-identical aliases of elapsed time since `_goal_arrival_time`; "well arrival" matches the producing clock. |
| `immobility_time` | `immobility_duration_s` | Verified `stop_end − stop_start` — a duration, not a timestamp. |

---

## Event-table geometry: units are missing wholesale — legibility

The per-event diagnostic columns systematically omit units, and the
*animal's* speed/position share vocabulary with the *replay's* — different
subjects, same words.

| Pattern | Columns | Fix |
|---|---|---|
| cm not in name | `net_displacement`, `path_length`, `spatial_extent`, `step_distance_*`, `max_ahead`, `max_behind`, `*_ahead_behind`, `start_position`, `end_position` | append `_cm` |
| cm/s not in name | `replay_speed_*`, `ab_slope`, `max_speed`, `mean_speed`, `start_speed`, `end_speed` | append `_cm_s` |
| animal vs replay subject collision | `*_speed` (animal) vs `replay_speed_*` / `start_position` (replay) | prefix `animal_` / `replay_` |
| fraction naming | `pause_fraction`, `multimodal_fraction`, `path_continuity`, `chosen_arm_fraction*`, `chosen_patch_fraction` | keep the names already ending in `_fraction`; rename only `path_continuity` → `path_continuity_fraction`. Use the full word `fraction`, not `_frac`. |
| name asserts wrong unit and type | `posterior_hpd_width_50/80/95` | not cm, and **not an integer count**: a non-local-probability-weighted *mean* of the per-bin HPD arm count (`uncertainty.py:137`), fractional on 36–66% of rows (range 1–9) → `posterior_hpd_arm_count_mean_coverage_50pct` (and `_80pct`/`_95pct` analogs). `coverage` prevents reading 50 as a percentile. |
| name asserts wrong thing | `n_junction_bins` | counts *undecoded* bins, all zero here → `n_undecoded_bins`; retain at event grain as specified in cleanup §4 |
| date, not index | `day` | YYYYMMDD date → `session_date` |
| post-hoc / future quantities | `next_arm`, `next_patch`, `is_stay`, `chosen_*`, `ends_at_next*`, `contains_path_to_next` | retain the established future-bearing names; classify every one as `role = outcome` and `availability = post_hoc` in the registry, and reject them from predictor specifications through the timing guard |

---

## Transforms and interactions — legibility

**Make `_z` honest.** Verified against `model_specs.py`: `arm_uncertainty_z`
enters the ladders standardized while `q_arm`, `p_reward_arm`, and
`reward_rate_arm_k10` enter **raw**. Canonical source-table columns retain
their raw scientific names and values. Any generated standardized design
feature receives the `_z` suffix and records its centering population, scale,
support, and training/fitting scope in the registry. A standardized feature
must never silently overwrite or reuse the raw column name. Changing which
predictors are standardized is a separate model-definition change, not part
of the naming migration.

| Current | Canonical | Why |
|---|---|---|
| `visit_recency_spline1_demeaned` | `visit_recency_trials_event_leaf_centered_spline_1` | "demeaned" is too vague: the transform is centering across visited leaf candidates *within an event* (`_center_visit_recency_within_event`), not a generic demean. Suffix order: base, transform, basis index. |
| `swr_x_visit_recency_spline1_demeaned` | `swr_x_visit_recency_trials_event_leaf_centered_spline_1` | Interaction inherits the renamed base. |
| `arm_fe_0 … arm_fe_7` | `is_candidate_arm_0 … is_candidate_arm_7`, with `reference_arm = 8` recorded in the registry | **Correctness fix.** `arm_fe_k` = 1 iff `arm == k` for k in 0–7, with **arm 8 the dropped reference** (`model_specs.py:153`, tuple at `:159`); they are generated at fit time, not stored. Renumbering to 1–8 would mislabel every indicator or silently move the reference. Retain 0-based arm identity and record the reference level explicitly rather than in a suffix. |
| `swr_x_*`, `mua_x_*`, `time_x_*` | *unchanged* | Consistent, greppable, reads as a product. |

---

## What the verification confirmed

The pass was not all corrections — most of the scheme held, and one
cross-table risk came back clean.

- **Classification flags agree across tables.** `is_swr`, `is_mua`, `is_stay`,
  `is_after_*`, `reward_outcome` are computed once in `event_table` and
  broadcast unchanged into `event_arm_table` — same name, same meaning. The
  only deliberate difference, `stop_table`'s `is_swr_stop`/`is_mua_stop`, is
  correctly given a distinct name (see
  [`known_issues/swr_mua_count_conventions.md`](../known_issues/swr_mua_count_conventions.md)).
- **The hazard constructs are real.** `current_arm` alias, `physical_arm`
  trunks, the belief-vs-contingency split, peri-switch alignment, the ΔV
  identity, and `q_arm` = posterior mean all verified exactly.
- **The good patterns to copy.** The window encoded by
  `reward_rate_arm_k10` is retained, with its inaccurate `rate` and `k10`
  vocabulary corrected to `reward_fraction_candidate_arm_last10_visits`;
  `swr_x_*` reads as a product; `_cm` and `_z` are honest where present.

---

## Migration contract

A pure rename must be **provably numerically invariant**. In a paper repo,
conflating it with scientific recomputation makes it impossible to attribute a
later coefficient or cohort change to the migration versus the science.
Exact renames, schema pruning, and consolidation of two floating-point
computation paths are three different operations and use three different
comparison policies.

The steps:

1. **Correct the mappings** — the `arm_fe` reference level and the HPD
   quantity are corrected above; do not carry the earlier errors forward.
2. **Use the canonical vocabulary** — occupancy, reward-fraction,
   reward-belief, and manuscript value terms defined above.
3. **Add the column registry** with explicit ID, dtype, support, structural
   encoding, producer, timebase/timestamp origin, and reference-level
   conventions (0-based arm and port identity; one-based task patches;
   `reference_arm = 8`).
4. **Archive a reproducible verification manifest** containing the source git
   commit, table build IDs and SHA-256 hashes, row/key counts, verification
   command or script, and the measured assertions quoted in this document.
   Percentages without a source artifact identity become stale after the next
   regeneration.
5. **Generate the rename map from the registry**, not by hand. Every old
   column maps to one explicit keep/rename/derive/audit/drop action; no
   "retire one" decisions remain for the implementation.
6. **Test the mechanical rename with exact equality.** Align by canonical row
   keys and require identical key sets, missing masks, dtypes, categorical
   levels, and values. Integer counts and labels are exact; a float column that
   is only renamed must also be bit-for-bit identical. The selected event,
   stop, and session cohorts must not change.
7. **Test complement derivation and schema pruning separately.** Verify the
   exact nullable identities (`stay = 1 − switch`, `omission = 1 − reward`),
   and verify that every removed diagnostic survives in the named audit or
   artifact metadata destination.
8. **Canonicalize the two numerical-path duplicate families in their own
   commit.** Compare `q_port`/`betadist_mu` and the global-variance pair under
   tolerances declared *before* the migration result is inspected (for the
   current float64 probability-scale path, `atol=1e-15`, `rtol=1e-12`), while
   requiring identical keys, missing masks, support, and downstream design
   rows. This is not labeled a pure rename.
9. **Verify consumer and model parity.** After applying the old→new name map,
   require identical design-matrix rows, values, reference levels, cohort
   hashes, and requested model sets. Figure-data tables should be numerically
   identical; a visual comparison is a final smoke check, not the primary
   parity test.
10. **Bump the table schema version; make old schemas fail loudly.**
11. **Keep all migration commits separate from scientific recomputation**
   (outcome-clock, standardization, or model-definition changes), even when
   they are scheduled inside the same regeneration window.

The rename may be scheduled alongside a regeneration, but it must remain a
separate, numerically invariant commit so that any coefficient or cohort
change is attributable to the scientific recomputation rather than the schema
migration.

### Also worth fixing regardless of any rename

- `covariates.py:550` docstring says `reward_lag1` is "the immediate previous
  trial"; the data contradicts it (lag1 is on the outcome clock).
- `uncertainty.py:152` docstring calls the HPD quantity a width "near
  `n_arms`", reinforcing the integer-count misreading.

---

*Measured against the shipped columns of `event_arm_table` (91), `stop_table`
(54), `qvals_table` (88), and `event_table` (113) by four independent passes,
each reading the producing code and recomputing from the materialized tables,
then reviewed. Before implementation, migration step 4 makes those checks
reproducible by binding them to exact source artifacts. Structural/ID columns
are named for reference but are out of rename scope.*
