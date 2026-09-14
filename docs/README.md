# Documentation index

Every documentation artifact has **one home by type**. This index states the scheme so
it stops drifting; add new docs to the matching home.

## Where each doc TYPE lives

| type | home | what it is |
|---|---|---|
| catalog | `docs/models_table.md` | every regression model in R notation (mirrors `model_specs.py`) |
| definitions | `docs/covariate_dictionary.md`, `docs/behavior_model_covariates.md` | canonical covariate definitions; the `qvals_table` data dictionary |
| reference (models) | `docs/behavior_models.md` | the SpatialBanditTask generative models (Beta-Bernoulli, HMM, Q-learners) that produce the behavioral covariates |
| methods | `docs/rate_models.md` | how the model families work (not results) |
| results | `docs/per_bin_rate_results.md` (auto-generated), `docs/replay_rate_content_swr_synthesis.md` (living note) | fitted effects + integrative findings |
| reference | `docs/task_description.md`, `docs/cached_trials_for_testing.md`, `docs/known_issues/` | task/data background |
| **design spec** (scientific/methodological) | **`docs/design/`** | analysis designs — human-discoverable for collaborators/reviewers |
| implementation plan (agent process) | `.claude/docs/plans/` | phased execution plans |
| review (code/methods output) | `.claude/docs/reviews/` | point-in-time review reports |
| exploratory notes + figures | `.claude/research/` | scratch analyses; `*_outputs/` figures |
| code map (generated) | `docs/regression_pipeline_map.md` | every pipeline module and the public symbols it defines, by stage |
| model spec (code) | `src/continuum_swr_replay/analysis/replay/model_specs.py` | single source of truth for all model ladders |

Rule of thumb: **published/scientific → `docs/`; agent-process/scratch → `.claude/`.**

## `docs/` contents

- [models_table.md](models_table.md) — *catalog*: every model in R formula notation.
- [covariate_dictionary.md](covariate_dictionary.md) — *definitions*: covariate semantics across families.
- [behavior_model_covariates.md](behavior_model_covariates.md) — *definitions*: `qvals_table.parquet` data dictionary.
- [behavior_models.md](behavior_models.md) — *reference*: the generative behavioral models (Beta-Bernoulli, HMM, Q-learners) whose latent variables the above columns hold.
- [rate_models.md](rate_models.md) — *methods*: the two event-rate families (A per-stop, B per-bin).
- [per_bin_rate_results.md](per_bin_rate_results.md) — *results*: fitted Family-B effects (auto-generated).
- [replay_rate_content_swr_synthesis.md](replay_rate_content_swr_synthesis.md) — *results*: integrative rate × content × SWR synthesis (living note).
- [task_description.md](task_description.md) — *reference*: behavioral task / recording setup.
- [cached_trials_for_testing.md](cached_trials_for_testing.md) — *reference*: test-fixture trial cache.
- [regression_pipeline_map.md](regression_pipeline_map.md) — *code map*: every module
  from decoded session to fitted regression, grouped by stage, with the public symbols
  each defines. Generated — regenerate with `scripts/qa/write_pipeline_map.py`.
- [design/](design/) — *design specs*: per-analysis design documents.
  - [design/model_history.md](design/model_history.md) — *design journal*: why each
    replay-content model rung carries the predictors it does (reversed reasoning
    and retracted numbers included). The machine-readable catalog is
    `analysis/replay/model_specs.py`; this is the record of how it got that way.
  - [design/cohort_and_timing_rules.md](design/cohort_and_timing_rules.md) —
    *design spec*: the covariate causal-hygiene rules every fit runner enforces
    and the `<run>_cohort_audit.csv` schema (drop buckets + reconciliation
    invariant).
  - [design/builder_test_floor.md](design/builder_test_floor.md) — *reference*: the
    measured value-level test coverage of the covariate builders and the target,
    kept honest by `scripts/qa/mutation_check.py`.
  - [design/2026-07-13-content-rate-uncertainty-model-ladders.md](design/2026-07-13-content-rate-uncertainty-model-ladders.md)
    — proposed Aim-1 content-allocation, replay-occurrence, event-size, and
    marginal-content ladders in R formula notation.
  - [design/2026-08-21-covariate-naming-scheme.md](design/2026-08-21-covariate-naming-scheme.md)
    — proposed predictor-column names across the four canonical tables,
    code-verified; also flags the duplicate and dead columns the pass found.
  - [design/content_estimator_comparison.md](design/content_estimator_comparison.md)
    — the retired count multinomial vs the shipped per-event fractions vs a
    Dirichlet-multinomial, on one 218-session cohort: what each assumes,
    simulation recovery, which conclusions are estimator-dependent, and the
    `log_phi_is_swr` concentration result. Not yet reflected in the claims table.
- [known_issues/](known_issues/) — *reference*: documented data/pipeline issues.
  - [three_sessions_blocked_by_nwb_checksum.md](known_issues/three_sessions_blocked_by_nwb_checksum.md)
    — the canonical cohort is 215 of 218; three sessions fail an upstream NWB
    checksum we don't control. Proceed with 215; don't gate on recovery.
  - [swr_mua_count_conventions.md](known_issues/swr_mua_count_conventions.md) — don't
    cross-substitute the stop-level vs bin-level `n_swr_events`/`n_mua_events` count.
  - [current_arm_trunk_fallback.md](known_issues/current_arm_trunk_fallback.md) —
    historical diagnosis of the destination fallback; fixed by explicit
    behavioral-location labels and past-only context (regeneration pending).
  - [contains_path_length_confound.md](known_issues/contains_path_length_confound.md) —
    `contains_path_to_next` is 2.8× harder for cross-patch than within-patch trips
    by construction; don't pool across trip types.

## Figure outputs

- **`paper/figures/<topic>/`** — publication figures only, in topic subdirs (main-text
  `figure0N/`, `supp_*`, and analysis topics like `rate/`, `reality/`, `quality_metrics/`).
  No loose files at the top level. Generated by `scripts/figures/*` and the `scripts/*/plot_*.py`.
- **`.claude/research/*_outputs/`** — exploratory / scratch figures (from `scripts/exploratory/_*.py`),
  not results-of-record.

## Where results live

Three tiers:

1. **Result DATA** (`.parquet`/`.pkl`/`.nc`, gitignored) under `Processed-Data/replay_content/`:

   The canonical tables (events, event-arms, event-patches, sequences, stops) are
   built by `scripts/decoding/prepare_replay_content_data.py`, and `qvals_table` by
   `scripts/behavior/build_qvals_table.py`. They
   are **parquet**; read and write them through `continuum_swr_replay.table_io`
   (`read_table` / `write_table`), never `pd.read_pickle`. Fitted-model artifacts
   (`all_models.pkl` and friends) hold live statsmodels objects and stay pickled.
   Superseded table pickles are kept under `_pickle_archive/`; nothing reads them.

   | dir | producer | holds |
   |---|---|---|
   | `models_per_bin_rate/` | `scripts/rate/fit_per_bin_rate_models.py` | `all_models.pkl` (Family-B ladder); also holds `qvals_table.parquet`, `swr_contrast_rate.pkl`, `uncertainty_rate.pkl`, `swr_power_model.pkl` |
   | `models_per_stop_rate/` | `scripts/rate/fit_per_stop_rate_models.py` | Family-A per-stop ladder |
   | `models/`, `models_patch/` | `scripts/content/fit_replay_content_models.py` (arm / patch) | the default content ladders — arm `M0`, `M0a`, `M1`–`M6`; patch `P0`–`P3` — over all events, plus the opt-in value/uncertainty rungs (`Mv`, `Mu`, `Mvu`, `Pv`) when named with `--models`; also holds `swr_contrast_content_{arm,patch}.pkl` |
   | `models_swr_{T,F}/`, `models_patch_swr_{T,F}/` | same, `--swr-subset T\|F` | content split by SWR (T) / non-SWR (F) |
   | `models_dm/` | same, `--estimator dirichlet_multinomial` | the M0–M6 arm ladder refit as a Dirichlet-multinomial (count response + fitted within-event concentration); the matching contrast bundles `swr_contrast_content_{arm,patch,audited}_dm{,_rewarded,_omitted,_len}.pkl` sit in `models/` beside their conditional twins |
   | `content_estimator_comparison/` | `scripts/content/compare_content_estimators.py` | the three-estimator comparison on one shared cohort — `content_estimator_comparison.parquet`, `_summary.csv`, `content_estimator_bootstrap.csv`, `_skips.csv`. Read with [design/content_estimator_comparison.md](design/content_estimator_comparison.md) |

   Both rate families are fit on the **well-anchored stop cohort** — the same
   population the content family is fit on at the event grain — and an
   unsuffixed rate artifact is that cohort. The sensitivity bundles carry the
   cohort in the filename: `*_allstops` (every immobility stop) and
   `*_intransit` (in-transit stops only, thin — 50 of 218 sessions). Every
   fitted row and provenance sidecar records its `stop_cohort`. See
   [design/cohort_and_timing_rules.md](design/cohort_and_timing_rules.md) and
   [audits/2026-09-12-rate-cohort-atwell.md](audits/2026-09-12-rate-cohort-atwell.md).

   (These dir names are terse — a descriptive rename and un-mix of the content directories
   has been considered and deferred; the names above are the ones the runners write.)

   Each **primary** fit runner (content base/contrast, rate contrast, Family A/B) also
   writes a **`<run>_cohort_audit.csv`** (one row per model × population; retained-N and
   mutually-exclusive drop buckets, reconciling `n_in − Σdrops == n_modeled`; coverage
   boundary in [models_table.md](models_table.md#cohort-audit-tables)) and the contrast
   runners a `<run>_term_diagnostics.csv`
   (per-term quasi-separation counts). `scripts/qa/summarize_cohort_audits.py` stacks and
   flags them; the schema is documented in [models_table.md](models_table.md#cohort-audit-tables).

   The **PFC** analyses write outside that table — the unit bundles sit at the
   `Processed-Data/` root, the two staged analyses in `replay_content/` beside the
   hippocampal fits:

   | artifact | producer | holds |
   |---|---|---|
   | `Processed-Data/{epoch_identifier}_mPFC_modulation.pkl` + five `*_modulation.npy` | `scripts/pfc/run_pfc_analysis_bandit.py` | per-neuron GLM statistics and the five peri-event modulation time courses (`SWR`, `no_SWR`, `non_local_SWR`, `non_local_no_SWR`, and `matched_control` — the duration-matched non-event baseline); the Figure 4 population, read by `analysis/pfc/load_population.py`, which still loads only the first four curves — the matched-control curve is written per session but not yet read into the population, which waits on the loader update |
   | `replay_content/stop_replay_summary.pkl`, `stage6_is_stay_{logistic,pooled_gee}{,_swr,_non_swr,_mua}.pkl` | `scripts/pfc/run_stage6_learning.py` | the per-stop replay summary and the stay/leave models (claim 19e) |
   | `replay_content/stage5b_retrospective{,_swr_only,_non_swr_only}.pkl` | `scripts/pfc/run_stage5b_retrospective.py` | the retrospective/prospective event-grain metrics against their visit-context shuffle |

   All three are fit on the **well-anchored stop cohort**, the same population as the
   rate and content ladders, and an unsuffixed PFC artifact is that cohort; `_allstops`
   is the unrestricted sensitivity bundle (unit bundles and Stage 6 only — Stage 5B has
   one rule at the event grain). **Only Stage 6 has shipped an `_allstops` bundle**;
   the unit-GLM one has never been run — produce it with
   `run_pfc_analysis_bandit.py --stop-cohort all_stops` (a full 71-session refit), which
   is the only way to compare the unit fits across populations on artifacts rather than
   on the unit test that pins the unrestricted path. Every PFC artifact carries a `*.provenance.json`
   sidecar recording `stop_cohort`, the stop-table SHA-256, and the bins and events
   before and after the restriction. The unit bundles' sidecar also states **which
   events the fit describes**: `is_swr_definition`, the SHA-256 of the canonical event
   table and of the session's cached decode frame
   (`canonical_event_table_sha256`, `data_over_time_sha256`), the cohort-restricted
   `n_events_total` / `n_events_swr` / `n_events_non_swr`, and the matched control's
   `matched_control_seed` / `n_matched_control` / `n_matched_control_unplaced`;
   `run_pfc_analysis_bandit.py` refuses to skip a bundle that does not carry the
   current values. **The 71 shipped at-well unit bundles (fit 2026-09-13/14) predate
   the event definition and the matched control**, so the next batch refits them —
   until it does, the Figure 4 population and claim 20 rest on bundles that label the
   same events the same way but carry neither the canonical `event_id`s nor a
   control. See
   [design/cohort_and_timing_rules.md](design/cohort_and_timing_rules.md),
   [audits/2026-09-14-pfc-atwell-cohort.md](audits/2026-09-14-pfc-atwell-cohort.md) and
   [audits/2026-09-13-stage5b-stage6-atwell.md](audits/2026-09-13-stage5b-stage6-atwell.md).

2. **Result DOCS**: `per_bin_rate_results.md` (auto-generated by `scripts/rate/summarize_per_bin_rate_models.py`)
   vs `replay_rate_content_swr_synthesis.md` (hand-maintained living note).

3. **Figures**: `paper/figures/<topic>/` (publication) vs `.claude/research/*_outputs/` (exploratory) — see above.
