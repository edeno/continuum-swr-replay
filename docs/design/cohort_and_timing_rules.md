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
function, `analysis.replay.cohort_selection.select_stop_cohort` (stop grain) or
its sibling `select_event_cohort` (event grain), on the canonical
`occupancy_state` column:

| cohort | stop-grain predicate | event-grain predicate | artifacts |
| --- | --- | --- | --- |
| `at_well` (canonical, default) | `occupancy_state in {"at_well", "mixed"}` — *well-anchored* | `occupancy_state == "at_well"` | the unsuffixed names |
| `in_transit` | `occupancy_state == "in_transit"` | `occupancy_state == "in_transit"` | `_intransit` |
| `all_stops` | no occupancy restriction | no occupancy restriction | `_allstops` |

**The two grains need different predicates to mean the same thing.**
`occupancy_state` is not the same statement about a stop as about an event. A
stop is labelled `mixed` whenever its sampled states differ; an event is
`at_well` only when its whole window is at one well. 14,845 of the 107,400
canonical stops are `mixed`, and they hold **18,274 of the 59,699 content
events (30.6%)** — so a naive stop-grain `occupancy_state == "at_well"` filter
would make the rate and content cohorts **cross** rather than nest.

Measured, those `mixed` stops are at-well bouts. Of the 14,743 (99.3%) that
have a last-well arrival at all, **every one** begins *after* it — median
0.73 s after, p10 0.22 s (the other 102 begin before the session's first
arrival, so they have no clock, not a late one). Their nearest-well distance
over the bout has median 0 cm at both ends and mean 0.28 cm, and is exactly 0
throughout for 81.6% of them, against a 5.2 cm median for `in_transit` stops.
And **96.8%** of their events (18,274 of 18,883) are themselves event-grain
`at_well`. Their first settling samples simply read as in-transit. The
stop-grain predicate is therefore *well-anchored* — a bout that begins after
arriving at a well and sits at it.

**The counts** (canonical tables, schema `v5_dio_poke_trials`):

- stop table: **107,400** stops → **84,069 well-anchored** (69,224 `at_well` +
  14,845 `mixed`), 22,850 `in_transit`, 481 `unknown`.
- arm event table: **65,913** events → **59,699 at-well** (90.6%).

What reaches the shipped rate fits from those 84,069: Family A models **84,064**
stops on `R0` (5 dropped for a missing own predictor), Family B
**83,549** over **6,777,953** bins (the same 5, plus 515 with no arrival clock,
which the arrival rungs put into the frozen union). Counts from
`models_per_stop_rate/all_models_cohort_audit.csv` and
`models_per_bin_rate/per_bin_rate_cohort_audit.csv`; the three cohorts are placed
side by side in
[`../audits/2026-09-12-rate-cohort-atwell.md`](../audits/2026-09-12-rate-cohort-atwell.md).

**Nesting, and its residual.** Every event in the content cohort lies in a stop
in the rate cohort. The converse does not hold exactly: 609 events sit in
well-anchored stops without being event-grain `at_well` (446 `in_transit`, 144
`mixed`, 19 `unknown`) — **1.0%** of the 60,308 events those stops hold. So
`content ⊂ rate` holds up to that residual, which is in the rate cohort and not
the content one by design.

**How it is applied.** The four rate runners (`fit_per_bin_rate_models.py`,
`fit_per_stop_rate_models.py`, `fit_swr_contrast_rate.py`,
`fit_uncertainty_rate.py`) expose it as `--stop-cohort`, applied to the stop
table **before** the `--max-per-animal` smoke subset and **before** the
common-cohort freeze, logged with its counts, written onto every fitted row as
`stop_cohort`, and recorded in the provenance sidecar's `extra`. The two
content runners (`fit_replay_content_models.py`,
`fit_swr_contrast_content.py`) apply the event-grain `at_well` rule
unconditionally — the content ladders are only defined there, since every
location-derived predictor (`is_current_arm` and everything built from it) is
undefined off a well.

`model_comparison.METADATA_COLUMNS` carries `stop_cohort`, so an at-well rung
compared against an all-stops rung is NaN'd with a reason rather than
differenced. That guard is soft on legacy artifacts: fits written before the
flag existed carry no `stop_cohort` at all, and `compare_nested_models`
deliberately treats a missing value as *unknown*, not *different*, so the
`n_stops` / `n_bins` row-count guard is the backstop for those until the
refits land.

**The PFC analyses read the same rule, at three grains.** The mPFC unit GLM
(`scripts/pfc/run_pfc_analysis_bandit.py`) fits 2 ms bins, so it applies the
stop-grain cohort **to the bins**: the fit mask is `is_immobile &
stop_interval_mask(position_time, kept_stops)`, i.e. an immobile sample that
falls inside a `[stop_start, stop_end]` interval of a kept stop. The events its
design matrices are keyed on are the **canonical** ones, read rather than
re-detected (`analysis/pfc/canonical_events.py` is the only place the PFC path
loads either): the non-local events are the session's rows of the shipped event
table (`Processed-Data/replay_content/event_table.parquet` — `event_id`,
`event_start_s`/`event_end_s`, `is_swr`, `occupancy_state`), and the SWR
intervals are the `SWR_event` labels of the cached
`{epoch_identifier_HPC}_80_data_over_time.pkl` frame, speed-gated and reduced to
their sampled bounds by the same two helpers
(`filter_event_labels_by_speed`, `intervals_from_labels`) the hippocampal event
table calls. The SWR split is that table's own `is_swr` column — strict
positive-duration overlap with those same labels — so a Figure 4 event and a
content- or rate-family event with the same `event_id` are one event, classified
one way; `data["ripple_times"]` no longer enters any fit path. Events are
restricted **by onset**: an event enters when its onset lies inside a kept stop,
applied *before* the split is read so `non_local_SWR` and `non_local_no_SWR`
partition exactly the kept events. Any canonical event or SWR interval falling
outside the aligned position frame is counted (`n_events_outside_frame`,
`n_swr_intervals_outside_frame`) and dropped rather than silently clipped.
The regressor basis still spans ±250 ms around each kept event, so an
event dropped just outside a stop's leading edge could have covered bins that
*are* fit; those bins are fit with its contribution zero, and the count of such
events is written to every provenance sidecar
(`n_{nonlocal,ripple}_dropped_overlapping_fit`) rather than assumed away.
Corpus-wide that is **6 non-local events and 4 SWRs** out of the 15,143 and
11,997 that are kept. Across the 71 mPFC sessions the bin mask keeps
**92.3%** of the immobile bins (19,732,623 of 21,384,252), and the onset rule
keeps **91.9%** of the non-local events (15,143 of 16,477) and 93.6% of the SWRs
(11,997 of 12,821); the per-session table is in
[`../audits/2026-09-14-pfc-atwell-cohort.md`](../audits/2026-09-14-pfc-atwell-cohort.md);
those counts were measured on bundles cut with the earlier run-time event
construction, which on this population labels the same events the same way —
identical event count and identical SWR-coupled count in **71 of 71** sessions,
median per-session SWR fraction **0.447** either way, and an identical SWR
population wherever the two constructions were run side by side.
Reading the canonical table therefore buys traceability, not a different split;
`scripts/qa/diagnose_pfc_event_definition.py` recomputes the comparison for any
session.
The unit GLM also draws a **matched non-event control** from that same fit
mask, one window per kept non-local event: the control onset is drawn uniformly
over the sample times of the *same* contiguous run of fitted bins the event's
onset falls in, and the draw is rejected unless the window lies wholly inside
that run, overlaps no real event of any class (`SWR` included) and overlaps no
control already placed. It preserves its source event's duration, so the
comparison `matched_control + non_local_no_SWR vs. matched_control` asks whether
non-SWR replay modulates firing beyond a duration-matched non-event baseline
sampled from the same bins — the question a replay regressor on its own cannot
separate from "the animal is in the stretch of immobility replay happens in".
An event whose bout holds no free window of its duration is dropped rather than
placed elsewhere, so the control count is `<=` the event count; the draw's seed
(a hash of the session key) and both counts are written to every provenance
sidecar (`matched_control_seed`, `n_matched_control`,
`n_matched_control_unplaced`).

Three properties of that control decide what the comparison can be read as.

- **It is matched one-per-non-local event, over both classes**, not against the
  non-SWR class alone, so the control regressor generally carries *more* windows
  than the class it is compared with (one smoked session: 275 controls against
  187 non-SWR events, from 277 kept events). The two regressors are therefore
  not equal-n.
- **The preserved duration governs only where a control may be placed** — it
  sets the rejection window that must fit inside the bout and clear every real
  and already-placed window. The regressor itself is built from onsets alone,
  for every event class: each class's indicator is a spike-count-per-bin of its
  `start_time`s convolved with the lagged spline basis, so a window's length
  never enters the design.
- **Control onsets are uniform over the fitted bout, while real replay clusters
  after arrival**, so the control matches bout membership and duration but not
  time-since-arrival; `matched_control vs. constant` is the diagnostic for that
  residual — it asks whether the baseline itself modulates firing. And because
  a draw is rejected where real events already sit, controls are pushed into the
  event-free stretches of a bout, which in a short event-dense bout is
  *anti-conservative* for the engagement test. The pair is also not
  SWR-controlled: the adjudicating rung
  `SWR + matched_control + non_local_no_SWR vs. SWR + matched_control` is not
  fit yet.

Stage 6
(`run_stage6_learning.py`) applies `select_stop_cohort` at the stop grain before
its per-stop summary, and Stage 5B (`run_stage5b_retrospective.py`) takes its
at-well membership from `select_event_cohort` at the event grain (keeping its
finite-`occupancy_well_arm` conjunct). All three carry `--stop-cohort`; only the
unit GLM and Stage 6 offer `all_stops`.

**The PFC path aligns the decode to the position frame first.** It is the only
consumer that reloads `load_data` alongside a saved decode, and decodes written
before `get_position_info` was trimmed carry more time bins than `load_data` now
returns for the same epoch. `analysis.hippocampus.data_aggregation.align_results_to_position`
intersects the two axes on exact timestamps and accepts the result only when the
mismatch is a head/tail trim within budget (600 s on the results axis, 0.1 s on
the position axis); interleaved mismatches and drift inside the shared run still
raise, and the restricted pair goes to the same strict 1-to-1 validator. On
matched axes it is the identity. What it trimmed is recorded per session under
`cohort_counts.alignment` in the sidecar.

**Why at-well is the shared cohort.** The rate ladders used to fit every
immobility stop while the content ladders fit only at-well events, so "non-SWR
replay is not reward-gated" (a rate claim) and every content claim about
non-SWR replay were statements about different populations. And the difference
is not neutral: in-transit stops hold **7.0%** of the corpus's immobility time
(9,093 s of 129,997 s) but **11.9%** of the non-SWR non-local onsets (4,690 of
39,510) and only **3.3%** of the SWR-coupled ones (859 of 26,403). Per second
of immobility that is **1.79x** the well-anchored non-SWR onset rate (0.516 vs
0.288 Hz) and **0.45x** the well-anchored SWR-coupled rate (0.094 vs 0.212 Hz).
Pooling those stops into the rate family and not the content family therefore
moves exactly the SWR-vs-non-SWR contrast the two families are compared on.

The all-stops fits remain available as a sensitivity bundle under `_allstops`,
and in-transit stops may be fit as their own bundle under `_intransit`, but an
in-transit population is never pooled with a well-anchored one.

The restriction is not balanced across the SWR contrast (3.3% of SWR events are
off-well against 13.0% of non-SWR events) — expected, since SWR events
concentrate at wells — so the cohort belongs in Methods.

**Which estimator answered for a unit is device-dependent.** The mPFC unit GLMs
are fit by a batched Poisson IRLS in JAX, one call per model for every unit of a
session, and it declines units whose fits leave the range it can vouch for —
`needs_reference_fit` flags a unit when its fitted means spread past a
threshold, or when it did not converge — refitting those with statsmodels
instead. That criterion reads floating-point quantities, so a GPU run and a CPU
run of the same session can flag slightly different units. Both fitters agree to
~1e-10 on every unit that stays inside the flag (that is the equivalence gate),
and a flagged unit's shipped numbers are statsmodels' own either way, so the
estimates do not depend on the device — only the route to them does. The backend
is recorded in each bundle's provenance sidecar (`fitter`, `jax_backend`,
`jax_version`, `gpu_index`) alongside `n_neurons_reference_fit`, the count of
units that took the statsmodels route.

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
