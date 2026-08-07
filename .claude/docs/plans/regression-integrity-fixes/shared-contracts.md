# Shared contracts

[← back to PLAN.md](PLAN.md)

- [Session input-bundle contract](#session-input-contract)
- [Run artifact and promotion contract](#run-artifact-contract)
- [Cohort audit schema](#cohort-audit-schema)
- [Comparison identity contract](#comparison-identity)
- [Confirmatory hypothesis registry](#hypothesis-registry)
- [Fold assignment contract](#fold-assignment)
- [Trial identity and covariate construction](#covariate-construction)
- [Covariate timing contract](#covariate-timing)
- [Decision-table contract](#decision-table)
- [Fit-validity and diagnostics contract](#fit-validity)
- [Term diagnostics](#term-diagnostics)
- [Animal-balanced summary contract](#animal-summary)

## Session input-bundle contract {#session-input-contract}

Every decoded session consumed by a table builder has a `session_input_manifest.json`. Its identity is
content-based, not merely the session name:

```json
{
  "schema_version": 1,
  "session_id": "animal-day-epoch",
  "nwb_file_name": "...",
  "epoch_number": 0,
  "epoch_name": "...",
  "session_epoch_crosswalk_sha256": "...",
  "position": {
    "source": "DLCPosV1",
    "merge_id": "...",
    "interval_list_name": "...",
    "valid_times_sha256": "...",
    "raw_index_sha256": "...",
    "analysis_index_sha256": "...",
    "max_raw_gap_s": 0.0
  },
  "track_graph": {
    "source": "IntervalLinearizedPosition",
    "track_graph_name": "...",
    "graph_sha256": "...",
    "linear_edge_order_sha256": "...",
    "linear_edge_spacing_sha256": "..."
  },
  "dio": {"query_sha256": "...", "beam_sha256": "...", "pump_sha256": "..."},
  "trials": {"builder_schema_version": 1, "sha256": "..."},
  "decoder": {
    "results_sha256": "...",
    "results_time_sha256": "...",
    "detector_sha256": "...",
    "ripple_query_sha256": "...",
    "training_type": "...",
    "environment_sha256": "...",
    "parameters_sha256": "...",
    "software_sha256": "..."
  },
  "data_over_time": {"sha256": "...", "index_sha256": "..."},
  "git_sha": "...",
  "dirty_diff_sha256": null
}
```

Promotion requires all of the following:

- the position and track-graph queries each resolve to exactly one declared source row. Fallback to
  legacy position or the declared animal graph is allowed only when the preferred restriction has
  zero rows, never on a broad `DataJointError`; the fallback relation must itself have exactly one row;
- the analysis index lies inside the exact union of `valid_times`, never merely between the first
  start and last end;
- raw-position gaps greater than 0.100 seconds *within any one declared valid interval* fail source
  validation instead of being silently bridged by linear interpolation; a declared gap between valid
  intervals is not compared with this threshold;
- `results.time`, the position analysis index, and `data_over_time.index` have the same length/order,
  a maximum absolute difference of at most \(10^{-6}\) seconds, and matching SHA-256 hashes after
  canonical float64 rounding to six decimal places;
- every DIO timestamp used for a trial is inside the exact valid-time union; every pump maps to exactly one
  visit; a zero-pump session requires a pre-run, reviewed family-scoped exclusion;
- position at a poke/run endpoint is at most 8 cm from its assigned well; out-of-range timestamps are
  rejected rather than clamped to the first/last position sample;
- the trial cache key includes position, track-graph, DIO, and trial-builder hashes. A changed source,
  query, graph, builder, or schema is a cache miss; presence alone is never sufficient.

Decoder reuse additionally requires reconstructable evidence for the exact source/query, detector and
ripple query, training type, parameters, software, result bytes, environment, and time-axis hashes.
Equal time axes without the remaining identities do not authorize reuse. Unattested legacy results are
audit inputs only and must be regenerated into a new candidate root.

The exact task/brain-area source query is snapshotted as `expected_sessions.parquet` before compute.
The source audit emits one row per expected session, including failures and measured out-of-bounds or
otherwise invalid exposure. An expected family/session is promotion-complete only when its audit is
`ok` or it matches a pre-run, reviewed entry in `config/regression_session_exclusions.yaml` with
analysis scope, typed reason code, reviewer, approval date, and evidence hash. A new source, query, QC,
cache, or fit failure is not an exclusion and cannot be waived in Phase 13. Thresholds above are locked
for production; an empirical failure returns to Phase 7 for reviewed revision.

## Run artifact and promotion contract {#run-artifact-contract}

Every canonical runner writes into `<regression_root>/runs/<analysis>/<run_id>.staging/`, validates
the run, and atomically renames it to `<regression_root>/runs/<analysis>/<run_id>/`. Finalizing an
individual family never makes it canonical. Phase 13 binds all required immutable runs into an
explicit `<regression_root>/release_candidates/<release_id>.json` and freezes it without making it
current. PFC and paper-output generation receive that exact candidate path explicitly; default
consumers may not discover a candidate.

Generated report, figure, and proposed documentation files are written beneath
`<regression_root>/release_outputs/<release_id>.staging/` using repository-relative destination
paths. They are produced through a strict `resolve_release_candidate(candidate_path)` API, never
through `CURRENT.json`. After PFC and every output gate passes, Phase 13 atomically finalizes that
directory, writes `<regression_root>/releases/<release_id>.json`, validates the complete cross-family
release, and finally updates the single `<regression_root>/CURRENT.json`. A post-promotion
`publish_release_outputs(release_id)` copies only hash-matching bytes from the immutable release-output
directory to their repository destinations and verifies the copies. Thus an interrupted build cannot
require or expose a not-yet-current release, and paper-facing files cannot become the authority for
analysis identity.

`manifest.json` is written last and contains:

```json
{
  "schema_version": 1,
  "run_id": "<UTC>-<git8>-<spec8>-<input8>",
  "analysis": "per_bin_rate",
  "completed": true,
  "started_utc": "...",
  "completed_utc": "...",
  "git_sha": "...",
  "dirty": false,
  "dirty_diff_sha256": null,
  "argv": ["..."],
  "software_versions": {"python": "...", "statsmodels": "..."},
  "model_spec_sha256": "...",
  "model_specs": {},
  "inputs": [{"path": "...", "size": 0, "sha256": "...", "session_input_manifest_sha256": "..."}],
  "comparison_sets": {},
  "expected_cells": {},
  "failed_cells": [],
  "outputs": [{"path": "...", "size": 0, "sha256": "...", "columns": []}]
}
```

The strict loader requires `schema_version == 1`, `completed is true`, a matching run ID in every
output, exact output hashes/schemas, valid session-input manifests for every contributing session,
all expected cells or explicit failure records, and the audit, identity, and fold tables declared by
the manifest. There is no legacy fallback.

The release-candidate descriptor and completed release manifest contain `schema_version`,
`release_id`, `git_sha`, `dirty_diff_sha256`, `canonical_table_generation_sha256`, the exact
required-analysis list, and an `analyses` mapping from analysis name to
`{run_id, manifest_sha256}`. The completed release additionally has `completed=true`,
`release_outputs_sha256`, exact per-output path/size/hash records, and optional components such as
PFC represented by a validated immutable manifest path/hash or an explicit unavailable reason.
Release validation binds already-finalized family runs to the new release ID; family manifests do
not invent or predict that ID. Every required regression family must have the release's exact
git/diff and canonical-table-generation identities, appear exactly once, use no staging path, and
match its declared manifest hash. The candidate descriptor is immutable after PFC starts; any change
abandons the candidate and its output directory. The global pointer changes only after the completed
manifest, finalized release-output tree, and every declared hash validate.

For a production release, `git_sha` is the clean, frozen analysis-code/config commit used by every
table/model/PFC run and `dirty_diff_sha256` is null. All code-bearing regression and PFC PRs land
before that freeze. Production execution writes only generated data and the release-output staging
tree. Repository-facing publication happens after promotion from those already-hashed bytes and is
not recomputed in a dirty analysis checkout, so later paper edits cannot masquerade as a different
analysis-code identity.

For schema v1, `required_analyses` is exactly:
`canonical_tables`, `content_base`, `content_event_source`, `content_swr_contrast`,
`content_uncertainty`, `rate_per_stop`, `rate_per_bin`, `rate_swr_contrast`,
`rate_uncertainty`, `rate_cross_history`, `decision_choice`, `decision_stage6`, and `swr_power`.
Each has one finalized manifest even when some declared cells failed; those cells remain explicit and
their claims unavailable. PFC is the only optional component and is represented by the immutable
completion-manifest path/hash or an unavailable reason, never by a second current pointer.

## Cohort-audit table schema {#cohort-audit-schema}

The landed row-drop fields remain, with exact identity added. One record is emitted per
`(run_id, fit_purpose, comparison_set, model, population)`:

| Column | Meaning |
| --- | --- |
| `schema_version`, `run_id` | artifact contract identity |
| `fit_purpose` | `estimation` or `selection` |
| `comparison_set` | declared set ID; empty only for standalone estimation |
| `model`, `population`, `grain` | fit cell and counting unit |
| `n_in`, mutually exclusive `n_drop_*`, `n_modeled` | existing reconciled row/group accounting |
| `n_rows_modeled`, `n_groups_modeled` | exact likelihood rows and conditioning/block groups |
| `row_key_columns`, `group_key_columns` | typed keys used to construct hashes |
| `cohort_hash`, `group_hash` | exact canonical typed hashes |
| `fold_hash` | selection CV assignment hash; null only when CV is not run |
| `n_sessions_*`, `n_animals_modeled`, `drop_notes` | existing coverage/profile fields |

The invariant remains `n_in - sum(mutually exclusive drop buckets) == n_modeled`. Schema-versioned
writes reject absent identity fields rather than filling them with NaN.

## Comparison identity contract {#comparison-identity}

Selection results may be compared only when all of these match: `run_id`, `fit_purpose=selection`,
`comparison_set`, response, estimator/family, link, offset/exposure semantics, grain, category-set
semantics, `cohort_hash`, and `group_hash`. CV also requires equal `fold_hash`.

Declared sets:

| Analysis | Row key | Group/block | Selection sets |
| --- | --- | --- | --- |
| Per-stop rate | `stop_id` | chronological stop/decision | `R0-R1-R2`, `R0-Rlag` |
| Per-bin rate | `(stop_id, binj)` | whole stop | the `PB_EDGE_*` pairs and `PB_PAPER_RANKING` locked below |
| Arm content | `(event_id, arm)` | whole stop | `ARM_BASE`, `ARM_SPARSE_MEMORY`, `ARM_SWR_SPATIAL`, `ARM_SWR_MEMORY`, `ARM_VALUE`, `ARM_UNCERTAINTY`, `ARM_VALUE_GIVEN_UNCERTAINTY` locked below |
| Patch content | `(event_id, patch)` | whole stop | `PATCH_BASE`, `PATCH_VALUE` locked below |
| SWR contrasts | native likelihood key | whole stop | exact rate/content interaction edges and estimation-only timing sets locked below |
| Decision choice | `(decision_id, arm)` | whole decision | behavior/value baseline versus +replay; split replay separately |
| SWR power | `event_id` | whole stop | `POWER_REWARD_Q`, `POWER_RPE_SENS` |

The per-bin registry must expose these exact parent edges. `PB_EDGE_<parent>__<child>` is the
comparison-set ID for the named two-model pair:

| Parent | Child | Role |
| --- | --- | --- |
| `M0_const` | `M1_rew1` | previous-reward increment |
| `M1_rew1` | `M2_rewhist` | longer reward-history increment |
| `M0_const` | `M2r_rewrate` | continuous recent-reward-rate alternative |
| `M0_const` | `M3_hist` | raw lag-dummy history sensitivity |
| `M3_hist` | `M4_hist_rew` | reward history beyond raw history |
| `M0_const` | `M5_time` | within-stop time course |
| `M5_time` | `M6_time_hist` | raw history beyond time sensitivity |
| `M6_time_hist` | `M7_time_hist_rew` | reward history beyond raw time/history |
| `M7_time_hist_rew` | `M8_full_switch` | completed-switch context on the raw branch |
| `M7_time_hist_rew` | `M9_rew_x_time` | reward-by-time interaction on the raw branch |
| `M0_const` | `M3r_histcos` | raised-cosine history |
| `M5_time` | `M6r_time_histcos` | raised-cosine history beyond time |
| `M6r_time_histcos` | `M7r_time_histcos_rew` | reward history beyond raised-cosine time/history |
| `M7r_time_histcos_rew` | `M8r_full_switch` | completed-switch context on the raised-cosine branch |
| `M7r_time_histcos_rew` | `M9r_rew_x_time` | reward-by-time interaction on the raised-cosine branch |

The existing formulas remain the source of truth. Before a selection run, the registry adds the
mechanical raised-cosine counterparts `M6r_time_histcos` (`M5_time` plus the existing raised-cosine
self-history block), `M8r_full_switch` (`M7r_time_histcos_rew` plus
`previous_completed_trial_is_switch`), and `M9r_rew_x_time` (`M7r_time_histcos_rew` plus the existing
reward-lag-1 by time-spline block), and changes the existing `M7r_time_histcos_rew` parent from
`M5_time` to `M6r_time_histcos`. No other formula or parent is inferred from the suffix.

`PB_PAPER_RANKING` is fixed to
`M0_const`, `M1_rew1`, `M2_rewhist`, `M2r_rewrate`, `M3r_histcos`, `M5_time`,
`M6r_time_histcos`, `M7r_time_histcos_rew`, `M8r_full_switch`, and
`M9r_rew_x_time`. It is the only per-bin set from which a cross-model AIC/BIC or blocked-CV ranking
may be reported. Raw lag-dummy models remain named estimability sensitivities and cannot be inserted
into that ranking after results are seen. LRTs use only the parent edges above; membership in the
ranking set does not create additional nesting.

The `PB_PAPER_RANKING` winner is deterministic and predictive-score based. Retain only sessions where
all ten models have valid scores on the exact shared rows/folds and require at least one such session
from each of the five animals; otherwise no paper winner exists. For each model, take the unweighted
mean `cv_bits_per_event` across retained sessions within animal, then the unweighted mean across the
five animal values. Choose the largest value. Scores within `1e-6` bits/event of the maximum are tied;
break a tie by fewer registered design columns, then lexicographic model ID. Report the full score
table, animal values, retained-session IDs/hash, tolerance, and tie-break result. AIC/BIC remain
paired descriptive diagnostics and never override this rule.

Arm/patch content sets are also exact:

| Set | Exact models and declared parents | Role |
| --- | --- | --- |
| `ARM_BASE` | `M0 → M0a → M1 → M2 → M3 → M4` | stable spatial/reward-history ladder |
| `ARM_SPARSE_MEMORY` | `M4 → M5` | last-rewarded-elsewhere sensitivity |
| `ARM_SWR_SPATIAL` | `M5 → M6` | SWR-by-spatial interaction block |
| `ARM_SWR_MEMORY` | `M6 → M7` | SWR-by-reward/memory interaction block |
| `ARM_VALUE` | `M5 → Mv` | nominal plus Beta-value sensitivity |
| `ARM_UNCERTAINTY` | `M5 → Mu` | Beta-uncertainty sensitivity |
| `ARM_VALUE_GIVEN_UNCERTAINTY` | `Mu → Mvu` | Beta value increment given uncertainty; sensitivity only |
| `PATCH_BASE` | `P0 → P1 → P2 → P3 → P4` | patch spatial/reward/SWR ladder |
| `PATCH_VALUE` | `P3 → Pv` | patch nominal plus Beta-value sensitivity |

These IDs replace shorthand such as “M0–M4.” `M0a` may not be omitted from `ARM_BASE`; `Mv` is
compared only with its actual `M5` parent, never `M4` or `Mvu`; and no direct `Mv`-versus-`Mvu`
comparison exists. `Mvu` is excluded from headline/confirmatory reporting even when its sensitivity
set is fit. Each set freezes its own union-of-required-columns cohort; no result is copied across set
IDs.

For each `(session_id, population, comparison_set)`, freeze the per-bin selection cohort once from
canonical 15-ms rows truncated at 4 seconds. A row is eligible only when its exposure is finite and
strictly positive, its response is finite, it passes the timing gate below, and every predictor needed
by every member of that set is finite. An edge set uses the union for its two named models;
`PB_PAPER_RANKING` uses the union for all ten named models. Refit a model separately when it appears
in more than one set; never copy likelihoods, AIC, or fold scores across set IDs. Every member inherits
the identical sorted row keys, stop keys, offset values, and fold assignment for that set. A
maximal-cohort `fit_purpose=estimation` fit is allowed but has an empty `comparison_set` and contributes
no selection metric.

SWR-power selection is equally explicit:

| Set | Exact models and parents | Frozen cohort |
| --- | --- | --- |
| `POWER_REWARD_Q` | `P0` → `P1` → `P3` | Post-outcome events complete for power, duration, time since stop, `trial_reward`, `q_current`, `p_reward_current`, and `value_diff_patch` |
| `POWER_RPE_SENS` | `P0` → `P2` | Post-outcome events complete for power, duration, time since stop, and `rpe` |

Within each power set, `P0` is refit on that set's rows and all models share the exact `event_id`,
`stop_id`, and fold hashes. `P1` versus `P2` is not a declared comparison: no AIC/BIC, LRT, or CV-score
delta may be computed across the two sets. A standalone P0 on all response-valid events may be emitted
only as `fit_purpose=estimation`; it is not a substitute parent for either selection set.

SWR/non-SWR comparisons use pooled interactions, never a comparison of two separately fitted
stratum p-values. Each formal set below contains exactly one generated parent and the named child;
the parent uses the child's formula/cohort but drops only the tested `is_swr` main effect or
interaction block. Generated parents are serialized registry entries and receive their own fit rows,
not formulas invented by the comparison helper.

| Timing family | Exact formal rate edge sets |
| --- | --- |
| Pre-outcome-safe | `SR_LEVEL={SR0_main,S0_rate}`; `SR_REWARD1={S0_rate,S1_rew1}`; `SR_BASE={SR_S2_main,S2_base}`; `SR_BLOCK={SR_S4_main,S4_block}`; `SR_SWITCHRATE={SR_S5_main,S5_switchrate}`; `SR_VALUE={SR_S6_main,S6_value}`; `SR_UNCERT={SR_S8_main,S8_uncert}`; `SR_REWXTIME={SR_S10_main,S10_rewxtime}` |
| Post-outcome | `SR_RPE_POST={SR_S7_main,S7_rpe}`; `SR_UPDATE_POST={SR_S9_main,S9_update}` |
| Future-referencing descriptive | `SR_PERISWITCH_DESC={SR_S3_main,S3_periswitch}` |

For `SR0_main`, drop `is_swr` from `S0_rate`. For `SR_S2_main`, drop all
`is_swr × BASE` interactions from `S2_base`. For every `SR_Sk_main` with `k=3..10`,
retain the `S2_base` interactions and the added block's main effects, but drop only that added block's
`is_swr` interactions. The post-outcome sets use the physical gate; the descriptive set remains
future-referencing even after correct timestamps.

| Timing family | Exact formal content edge sets |
| --- | --- |
| Pre-outcome-safe arm | `SC_CURRENT_TIME={SC_CS1_main,CS1_current}`; `SC_CURRENT_UNCONTROLLED={SC_CS1nt_main,CS1nt_current}`; `SC_PREVIOUS={SC_CS2_main,CS2_prev}`; `SC_LAST_REWARDED={SC_CS3_main,CS3_lastrew}`; `SC_ARM_REWARD_RATE={SC_CS4_main,CS4_armreward}`; `SC_ARM_VALUE={SC_CS5_main,CS5_armvalue}`; `SC_VALUE_VS_PREVIOUS_GOAL={SC_CS5b_main,CS5b_valuevsprevgoal}`; `SC_ARM_UNCERTAINTY={SC_CS6_main,CS6_armuncert}` |
| Pre-outcome-safe patch | `SC_PATCH_CURRENT_TIME={SC_CPS1_main,CPS1_current}`; `SC_PATCH_CURRENT_UNCONTROLLED={SC_CPS1nt_main,CPS1nt_current}`; `SC_PATCH_LAST_REWARDED={SC_CPS2_main,CPS2_lastrew}` |
| Future-referencing descriptive patch | `SC_PATCH_OLD_DECAY_DESC={SC_CPS3_main,CPS3_olddecay}` |

Each `SC_*_main` drops only its named `swr_x_*` target term from the child, retaining all controls;
`SC_CS5b_main` specifically retains the previous-goal main effect and SWR interaction while dropping
only `swr_x_q_arm`. Split `G*`, `C*`, and `CP*` estimates never establish an SWR difference.
The exact estimation-only timing groups are
`SR_POSTOUTCOME_EST={G5_rpe,G7_update}`,
`SC_POSTOUTCOME_EST={C6_rpe,C9_update}`,
`SR_FUTURE_DESC={G1_periswitch}`, and
`SC_FUTURE_DESC={C10_periswitch,CP2_olddecay,CP3_newgrowth}`. Each model is still fitted on its own
declared maximal estimation cohort and is not ranked against another member of these labels.

AIC/BIC require the same likelihood and rows. LRT additionally requires a declared strict parent and
positive parameter difference. Let
`delta_ll = ll_child - ll_parent` and
`ll_tol = 1e-8 * max(1, abs(ll_parent), abs(ll_child))`. If `delta_ll < -ll_tol`, raise
`ComparisonContractError`. If `abs(delta_ll) <= ll_tol`, emit `LR=0`, `p=1`, and
`lr_within_tolerance=true`; otherwise require `delta_ll > 0` and compute the ordinary statistic.
No other negative-LR clipping or plausible-looking fallback is allowed.

## Confirmatory hypothesis registry {#hypothesis-registry}

The checked-in source of truth is
`src/continuum_swr_replay/analysis/replay/confirmatory_hypotheses.json`. Phase 10b owns its schema
loader, reference validation, joint-test construction, and multiplicity correction. The analysis
authors own the scientific entries and lock them in a reviewed commit before the production fits are
run; Phase 13 may validate and consume the registry but must not create or edit hypotheses after seeing
results.

The file contains `schema_version`, `registry_status` (`draft` or `locked`), one `family_id`, and an
`entries` array. Each entry contains a unique `hypothesis_id`, `analysis`, `comparison_set`, `model`,
`population`, `test_kind` (`term` or `joint`), a non-empty ordered `terms` list, and `alternative`
(`two-sided`, `greater`, or `less`). A locked entry must resolve exactly to a registered selection
model, set, population, and fitted term names; descriptive or estimation-only fits are rejected.
Joint entries require `alternative="two-sided"`.
This contract defines no scientific entry or direction. A locked empty registry is permitted only as
an explicit declaration that the release makes no confirmatory claims.

For hashing, parse the file and serialize it as UTF-8 with sorted object keys, compact separators, no
ASCII escaping, and one trailing newline. Those canonical bytes are SHA-256 hashed into generated
analysis metadata, every affected run manifest, and the release manifest. Any edit, including term
order or direction, requires new run and release IDs. Phase 13 refuses a confirmatory label when the
registry is `draft`, its hash differs, an entry is unresolved, or the entry was not included in the
single declared multiplicity family.

## Fold assignment contract {#fold-assignment}

`folds.parquet` contains one row per block group:

| Column | Meaning |
| --- | --- |
| `run_id`, `comparison_set`, `session_id` | owner |
| `group_id` | stop or decision ID; all descendant rows inherit its fold |
| `order_time` | canonical start time used for ordering |
| `fold_id` | contiguous balanced test block `0..K-1` |

Assignments are deterministic and invariant to input row order: sort by `(order_time, group_id)`,
split unique groups into contiguous balanced blocks, and score every group exactly once. The typed
hash of `(session_id, group_id, fold_id)` is `fold_hash`. Missing order time, duplicated group IDs, or
an unassigned group raises. Forward-chaining, if used, is a separately named scheme and is never
silently substituted for blocked interpolation.

## Trial identity and covariate construction {#covariate-construction}

`session_epoch_crosswalk.parquet` is built before trial covariates. It is bijective between source
`(nwb_file_name, epoch_number)` and canonical `(session_id, animal, day, epoch)` keys and is hashed
into every behavior/table manifest. IDs and joins use this table; filename synthesis and parsing an
integer from the canonical epoch label are forbidden.

`behavior_trial_table.parquet` contains every source trial before a behavior-model join, keyed by
`(session_id, trial_number_by_epoch)`. `behavior_model_trial_table.parquet` uses the same canonical
key and is left-joined one-to-one. Internal model gaps or key disagreement fail; a contiguous
post-fit terminal tail is named `q_terminal_tail`, remains in non-Q models, and is excluded only by
declared Q-dependent specifications.

At a query time, let the completed prefix contain trials whose outcome/arrival cutoff is
`<= query_time`. Arm and patch histories use only that prefix:

| Field family | Exact window |
| --- | --- |
| `reward_count_{arm,patch}_global_k10`, `visit_count_{arm,patch}_global_k10` | rewarded visits and all visits to the target among the last ten completed trials globally |
| `reward_count_{arm,patch}_visits_k10`, `visit_count_{arm,patch}_visits_k10` | rewarded visits and denominator among the last up-to-ten completed visits to that target |
| `reward_rate_{arm,patch}_visits_k10` | the visit-window numerator divided by its matching visit-window denominator |
| `has_{arm,patch}_history` | visit-window denominator is positive |
| `visit_recency_trials` | completed-trial distance to last visit; `n_completed + 1` when never visited |

An unvisited reward target has counts/denominators 0, rate 0, and history 0; a visited target with no
reward has positive visit exposure, rate 0, and history 1. Before the first completed trial, leaf/patch
history fields are `NaN`. Trunks use `is_reward_site=0` and structural zeros. Patch histories are
computed from patch visits, not by averaging leaf-arm rates. Persist raw `visit_recency_trials` only;
fitters apply fixed `log1p`, then learn mean/SD from the exact estimation cohort or current CV training
fold and record that transform hash.

## Covariate pre/post-outcome timing contract {#covariate-timing}

The existing `PRE_OUTCOME_SAFE`, `POST_OUTCOME`, and `FUTURE_REFERENCING` registry remains the
authoritative model-build gate. `outcome_known_time` is reward onset for a rewarded trial and poke-in
for an omission. The exact row/query times and boundary rules are:

| Grain/use | Time used | Inclusion rule |
| --- | --- | --- |
| Per-stop response | `stop_start` | post-outcome only when `stop_start >= outcome_known_time` |
| Per-bin response | `bin_start = stop_start + binj * BIN_S` | the represented interval `[bin_start, bin_start + exposure)` is post-outcome only when `bin_start >= outcome_known_time` |
| Event content or power | full `event_start` | post-outcome only when `event_start >= outcome_known_time` |
| Past-trial switch/history query | the applicable stop, bin-start, or event-start time above | `searchsorted(..., side="right")`; a trial completed exactly at query time is available |
| Decision replay window | `[outcome_known_time[t-1], poke_out_ts[t-1])` | a positive-duration event is included only when `event_start >= window_start` and `event_end <= window_end`; exact left-start/right-end equality is retained, and either-boundary straddlers drop |

For modeled rows, `response_time` is materialized as the applicable stop start, bin start, or event
start in this table; a missing/non-finite required timestamp makes a post-outcome row ineligible.
Equality at the outcome boundary is included. For per-bin models, `tc = (binj + 0.5) * BIN_S` remains
the spline coordinate but is not the information/timing gate; a bin that begins before the outcome is
excluded even when its center is after it. Final partial-bin exposure does not change this rule. This
row-start rule is normative and supersedes bin-center shorthand elsewhere in the plan.

Phase 8 changes three covariate details:

- Replace `recent_switch_rate` with `recent_completed_switch_rate_k5`, computed at response/query time
  from completed trials.
- Predictive switch-context terms use `previous_completed_trial_is_switch`. The current trip's
  eventual patch-change label is named `current_trip_is_switch` and is descriptive only unless a
  separately recorded choice-known timestamp is introduced.
- A `POST_OUTCOME` predictor is accepted only when the fitted row carries
  the grain-specific time above and passes its boundary rule; setting `descriptive=True` without a
  physical gate is not a substitute for a post-outcome analysis.

Future-referencing peri-switch predictors remain descriptive; no within-stop timing gate makes them
predictive-safe.

## Decision-table contract {#decision-table}

`decision_arm_table` has exactly six rows per eligible decision, keyed by `(decision_id, arm)` with
arms `0..5` and exactly one `chosen == 1`. `decision_id` is the stable session/trial identifier for the
target trial. Decisions with the same chosen and current arm and decisions with no replay are retained.

Required decision-level fields include target/current trial indices and arms, previous outcome,
`window_start`, `window_end`, duration, session/animal keys, and behavior-model coverage. Arm-varying
fields include pre-outcome Q/value, nominal probability, `visit_recency_trials`,
`reward_rate_arm_visits_k10`, `visit_count_arm_visits_k10`, `has_arm_history`, distance rank, and
hard/soft replay mass for all/SWR/non-SWR events. Replay mass is zero—not NaN—when a decision window
contains no event of that type.

`decision_summary_table` contains one row per decision for Stage 6, derived from the same arm table and
window. No stop-grain response is allowed in either canonical model.

## Fit-validity and diagnostics contract {#fit-validity}

Every attempted cell produces a row, including failures. Required fields:

| Field | Rule |
| --- | --- |
| `fit_status` | `ok`, `rank_deficient`, `nonconverged`, `separated`, `insufficient_data`, or `fit_error` |
| `failure_reason` | non-empty unless `ok` |
| `design_rank`, `n_design_columns` | computed on the actual fitted design |
| `condition_number_scaled`, singular-value fields | computed after centering appropriate to estimator and unit-RMS scaling |
| `vif_centered_*` | conditional estimators use within-group-centered design |
| `n_clusters`, `covariance_type` | actual uncertainty unit/method |
| `converged`, `separated` | explicit booleans |

Rank-deficient, nonconverged, separated, or fit-error cells invalidate every likelihood comparison and
all term-level headline use from that cell; raw diagnostics remain visible. A cell-level
ill-conditioning warning is not itself a failure. For an otherwise `ok` cell, a failed robust
covariance calculation invalidates inference only for the affected term while leaving likelihood/CV
selection eligible. Headline aggregation therefore requires both `fit_status=ok` and an eligible
term-level status, in addition to the Phase-10b dependence and animal checks. A penalized sensitivity
is labeled as a different estimator and is never used in an ordinary LRT/AIC comparison. Phase 10a
locks the numerical thresholds and status precedence.

## Term-level diagnostics table {#term-diagnostics}

The existing table is extended with `fit_status`, scaled term coefficient, estimability, robust-SE
availability, and exclusion reason. The old universal raw `abs(coef) < 10` rule is removed; scale-free
fit validity and explicit warnings own exclusion.

## Animal-balanced summary contract {#animal-summary}

For each estimand/term, first aggregate valid session estimates within animal, then report:

- all five per-animal effects and contributing session counts;
- median of animal effects;
- `n_animals_positive / n_animals_used`;
- the locked 10,000-resample percentile animal-bootstrap interval and
  leave-one-animal-out range from [the dependence design](designs.md#dependence);
- an incomplete-animal warning when `n_animals_used < 5`.

Session signed-rank p-values may be retained only under names ending `_descriptive`. Neuron-level PFC
effects first aggregate to session and then animal under the dedicated PFC plan. Confirmatory term
and joint p-values use the exact five-animal sign-flip procedures in the dependence design; no
session-level p-value enters BH-FDR.
