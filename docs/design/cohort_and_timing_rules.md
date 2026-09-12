# Cohort and covariate-timing rules

Three cross-cutting rules that every fit runner enforces, moved out of the
model catalog so that file stays a catalog:

1. **Shared stop cohort** — which stops (and therefore which events) the rate
   and content families are both about.
2. **Covariate causal hygiene** — which covariates may be a forward predictor,
   and which rungs are retained only under an explicitly *descriptive* claim.
3. **Cohort-audit tables** — the `<run>_cohort_audit.csv` schema, its
   mutually-exclusive drop buckets, and its reconciliation invariant.

The rungs themselves, in R notation, are in
[`../models_table.md`](../models_table.md); the covariate definitions are in
[`../covariate_dictionary.md`](../covariate_dictionary.md).

## Shared stop cohort

**The rule.** Every rate and content runner selects its population through one
function, `analysis.replay.cohort_selection.select_stop_cohort` (and its
event-grain sibling `select_event_cohort`), on the canonical `occupancy_state`
column of the stop / event table:

| cohort | rule | artifacts |
| --- | --- | --- |
| `at_well` (canonical, default) | `occupancy_state == "at_well"` | the unsuffixed names |
| `in_transit` | `occupancy_state == "in_transit"` | `_intransit` |
| `all_stops` | no occupancy restriction | `_allstops` |

The four rate runners (`fit_per_bin_rate_models.py`,
`fit_per_stop_rate_models.py`, `fit_swr_contrast_rate.py`,
`fit_uncertainty_rate.py`) expose it as `--stop-cohort`, applied to the stop
table **before** the `--max-per-animal` smoke subset and **before** the
common-cohort freeze, logged with its counts, written onto every fitted row as
`stop_cohort`, and recorded in the provenance sidecar's `extra`. The two
content runners (`fit_replay_content_models.py`,
`fit_swr_contrast_content.py`) apply `at_well` unconditionally — the content
ladders are only defined there, since every location-derived predictor
(`is_current_arm` and everything built from it) is undefined off a well.
`model_comparison.METADATA_COLUMNS` carries `stop_cohort`, so an at-well rung
compared against an all-stops rung is NaN'd with a reason rather than
differenced.

**The counts** (canonical tables, schema `v5_dio_poke_trials`): the stop table
holds **107,400** immobility stops — 69,224 at-well, 22,850 in-transit, 14,845
mixed and 481 unknown — so the at-well cohort keeps 64.5% of stops. The arm
event table holds **65,913** non-local events, of which **59,699** (90.6%) are
at-well.

**Why.** The rate ladders used to fit every immobility stop while the content
ladders fit only at-well events, so "non-SWR replay is not reward-gated" (a
rate claim) and every content claim about non-SWR replay were statements about
different populations. In-transit stops carry a disproportionate share of
non-SWR non-local onsets, so pooling them into the rate family and not the
content family is not a neutral difference in scope: it moves exactly the
contrast the two families are compared on. At-well stops are therefore the
shared canonical cohort for both. The all-stops fits remain available as a
sensitivity bundle under `_allstops`, and in-transit stops may be fit as their
own bundle under `_intransit`, but an in-transit population is never pooled
with an at-well one.

The restriction is not balanced across the SWR contrast (3.3% of SWR events are
off-well against 13.0% of non-SWR events) — expected, since SWR events
concentrate at wells — so the cohort belongs in Methods.

## Covariate causal hygiene

Every ladder runner enforces covariate timing at model-build via
`analysis.replay.covariate_timing.assert_predictor_timing`; a rung whose
predictor is post-outcome or future-referencing must be **declared
descriptive**, else the run fails.

- **Per-stop `R1`** uses the reward-history basis over the outcomes physically
  known at bout start, instead of a current-outcome flag. The newest known
  outcome is not a fixed current/previous-trial relabeling (see
  [`../rate_models.md`](../rate_models.md)).
- **Descriptive rungs** are retained under an explicit *association* /
  *peri-switch alignment* claim, **not** a forward prediction. The allowlists
  are `DESCRIPTIVE_CONTENT_RUNGS` and `DESCRIPTIVE_RATE_RUNGS` in
  `model_specs.py`, and the catalog enumerates them under "Descriptive rungs".
  A new rung that introduces a leaky covariate but is not on an allowlist fails
  the timing assertion rather than being silently promoted to a predictive
  claim. Two kinds of leak are covered:
  - **Future-referencing.** `periswitch_index_trials` and `is_periswitch_post`
    sign toward the *next* switch, so they can only ever be descriptive.
  - **Post-outcome.** `rpe_postoutcome` and
    `reward_belief_mean_update_l1_all_ports_postoutcome` embed the current
    trial's reward. They were relabeled descriptive here rather than gated.
- **Never-visited encoding.** An unvisited leaf's
  `reward_fraction_candidate_arm_last10_visits` is `0.0` (not NaN), while
  `is_candidate_arm_never_visited` explicitly identifies novelty. The two enter
  together wherever recency/reward history is modeled.
- The stay/leave model `is_stay ~ outcome * content`
  (`scripts/pfc/run_stage6_learning.py`) legitimately uses the post-outcome
  `is_omitted_outcome`: its modeled unit (the stay/leave decision) is
  post-outcome by construction, so the timing gate does not apply.

## Cohort-audit tables

The **primary** fit runners emit a `<run>_cohort_audit.csv` next to their
outputs: **one row per (model, population)** stating who entered the model and
who was dropped, so complete-case filtering is visible rather than implied.
Covered runners: `fit_replay_content_models.py` (content base ladder, common
cohort), `fit_swr_contrast_content.py` and `fit_swr_contrast_rate.py` (the
`is_swr`-contrast ladders), `fit_per_stop_rate_models.py` (Family A) and
`fit_per_bin_rate_models.py` (Family B). The emitter is
`analysis.replay.cohort_audit`. Counts are in the row's `grain` (`event` for
content, `stop` for the rate families). Drops are **mutually exclusive**,
assigned by priority:

`n_drop_qvals_gap` (mechanism B — a value covariate is missing on the unmapped
stop tail) → `n_drop_own_predictor_missing` (this model's own predictor NaN,
e.g. an early-session reward-history gap) → `n_drop_nonfinite` (a required
value is present but invalid, such as ±Inf) → `n_drop_zero_response`
(Σcount == 0) → `n_drop_common_cohort_other` (union complete-case: dropped only
for *another* rung's predictor) → `n_excluded_session` (data-level
`min_*`/rank gates).

**The rate families populate `n_drop_common_cohort_other`.** Both rate runners
freeze one cohort per fitted unit before any rung is fit —
`freeze_per_bin_cohort` over the union of every `MODEL_PER_BIN_RATE` design
column (per session × population, `exposure > 0`) and `freeze_common_cohort`
over the union of every `MODEL_PER_STOP_RATE` predictor (per session, positive
`immobility_duration_s` offset) — and the fitters raise on a residual NaN
instead of dropping rows, so a union drop is a recorded number rather than an
invisible per-rung `dropna`. Read the column as "complete for this rung, dropped
for another rung's predictor". For the **shipped** Family-A and Family-B
ladders no rung needs a covariate another rung lacks, so the frozen cohort is
simply every stop with a positive immobility offset and the column is zero on
every row (`models_per_stop_rate/all_models_cohort_audit.csv`,
`models_per_bin_rate/per_bin_rate_cohort_audit.csv`). A non-zero value appears
where rungs genuinely differ in what they need — the `is_swr`-contrast rate
ladder (`swr_contrast_rate_cohort_audit.csv`), whose value-covariate rungs also
charge their missing tail to `n_drop_qvals_gap` before the union is taken, as
does `fit_uncertainty_rate.py`.

**Reconciliation invariant (asserted, fail-loud):**
`n_in − Σ n_drop_* == n_modeled`. Convergence/separation counts
(`n_sessions_flagged_*`) and the `|coef| >= 10` quasi-separation counts (the
separate `<run>_term_diagnostics.csv`) are **session/term-level** and excluded
from the reconciliation. `scripts/qa/summarize_cohort_audits.py` stacks every
audit CSV and flags low-retention (`n_modeled/n_in < 0.9`) or biased
(mechanism-A early / mechanism-B qvals-tail) drops.

Also covered: `compare_event_source_models.py` maps its native
`_freeze_common_cohort` provenance (the content / patch / rate / **choice**
blocks) onto the standardized schema via
`records_from_common_cohort_provenance`, writing
`*_comparison_fits_cohort_audit.csv`. **This audit is block-grained, not
per-rung:** every model in a block shares ONE frozen common cohort, so it emits
one row per block with `model` = the block/analysis name (`choice`,
`choice_swr_split`, `patch_hierarchy`, …) rather than duplicating identical
counts per rung. The block's up-front selection — e.g. choice's
valid-replay-stop filter — is *definitional*, so `n_in` is the post-selection
candidate count. The qvals-merging auxiliary runners `fit_uncertainty_rate.py`
(inner join → `n_drop_qvals_gap`) and `fit_swr_power_model.py` (left join +
per-session dropna on qvals-derived predictors → `n_drop_qvals_gap`) each emit
their own cohort audit. `fit_cross_history_rate.py` (a bivariate
self-/cross-history variant with no new qvals dependence) is **not** audited —
documented, not silently dropped.
