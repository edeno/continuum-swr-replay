# Phase 2a execution plan (task-structured view of phase-2a-content-fractions.md)

Spec authority: `.claude/docs/plans/regression-pipeline-fixes/phase-2a-content-fractions.md`,
`shared-contracts.md` (Fraction response; Content estimator result schema; Standardization entry
point; Fit provenance sidecar), `designs.md` §"Event-clustered sandwich covariance", `overview.md`
(decisions: hard-MAP `arm_fraction` is the shipped response; posterior fraction is a phase-2b
sensitivity), and the review `.claude/docs/reviews/2026-09-03-regression-pipeline-review.md`
(findings A1, A5, A6, A7, B1, B5, B6). Where this file and those differ, they win.

## Global Constraints

- Run Python with `uv run …` from the repo root. Never install into the base environment.
- `uv run ruff check` and `uv run ruff format` on every file touched (do not reformat unrelated
  code in files that are not already format-clean; check added lines with `ruff format --diff`).
- Docstrings, test names, module names, and commit messages must not mention this plan, phase
  numbers, task numbers, or `.claude/docs/plans/`.
- Tests are value-level (numbers and behaviour, not presence). No test-only methods in production
  code. Shared setup in fixtures.
- Fail loud: contracts are enforced with `raise`, never warn-and-continue.
- Canonical tables are parquet under `Processed-Data/replay_content/`; read through
  `continuum_swr_replay.table_io.read_table`. Never modify a canonical artifact without first
  copying it into a dated `_pre_fix_backup_<YYYYMMDD>/` directory (today: 20260905). Never delete
  data. Nothing under `Processed-Data/` or untracked `paper/figures/` is committed.
- Every persisted canonical column has a catalog entry in
  `analysis/replay/covariates/definitions.py`; new model-facing product columns are registered in
  `covariate_timing.DERIVED_COVARIATE_SOURCES`.
- Commit after each coherent unit with a message ending in
  `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01JGZPceMjeWcjYZQ26byzzz`.
- Long runs (> 5 min): smoke first (`--max-per-animal 1`), record wall time and peak RSS with
  `/usr/bin/time -v`, extrapolate to 218 sessions, then run the full job inside `tmux` and report
  the session name. Poll with bounded waits.
- Do not run `git checkout` / `git stash` / `git clean`; the working tree holds the user's
  untracked files. Only edit files your task names plus their tests.
- State inherited from phase 1 (merged to master at 6244433): every content fitter in `fitting.py`
  already persists `sd_<term>` and `separated_terms`; `contrast_aggregation.aggregate_contrast`
  drops a session for term T only when T separated or `|coef_T| × sd_T ≥ 10` (effect-scale
  backstop), and prefers `bse_cluster_<term>` for `typical_session_se` once that column exists
  (Task 2 adds it); `_resolve_run_config` keys `--models` subsets to `<base>_<models joined by '-'>`
  and refuses canonical bundles (`canonical_output_subdir`, `canonical_output_subdirs` at
  `scripts/content/fit_replay_content_models.py:464-520`); rate fits carry provenance sidecars via
  `analysis/replay/fit_provenance.py` (`record_inputs` right after loading, `write_fit_provenance`
  at the end, `require_fresh_inputs` in readers with `--allow-stale`).
- The fraction-response contract (verbatim from shared-contracts.md): `arm_fraction[e, a] =
  arm_count_hard[e, a] / (n_nonlocal_bins[e] − n_undecoded_bins[e])` (already persisted,
  `tables_arm_patch.py:176-188`); `patch_fraction[e, p] = patch_count_hard[e, p] / (n_nonlocal_bins[e]
  − n_undecoded_bins[e])` (new). Both sum to exactly 1.0 over the candidate set of every event with
  at least one real-arm bin and are NaN otherwise (dropped by `freeze_common_cohort(...,
  drop_zero_response=True)`). `fit_conditional_poisson` accepts them unchanged; its analytic null
  becomes `-n_events × log(n_categories)`. `arm_count_hard` / `patch_count_hard` stay persisted and
  remain the response for the Dirichlet-multinomial fitter (phase 2b) and the retired-count
  comparison. Runners select the response with `--response {fraction,hard_count}`, default
  `fraction`.
- The estimator result schema (verbatim): every content fitter returns `coef_/bse_/pval_<term>`,
  `bse_cluster_<term>` (conditional Poisson on fractions only), `n_events`, `n_rows`, `n_params`,
  `llf`, `llf_null`, `aic`, `bic`, `pseudo_r2`, `llf_includes_constants` (False for statsmodels
  conditional fits), `converged`, `separated`, `separated_terms`, `n_convergence_warnings`,
  `estimator` ∈ {"conditional_poisson_fraction", "conditional_poisson_count",
  "conditional_logit_dominant"}.
- Standardization (verbatim): `standardization.attach_standardized(df, cols, reference=None, *,
  support_mask=None, support_label="all_rows")` is the only z-scoring code path; fit-time z-scoring
  in runners uses `support_label="fit_cohort_all_rows"` and persists the returned reference next to
  the fit output as `<artifact-stem>.z_reference.json`. No runner defines its own `_z`.

### Task 1: Persist `patch_fraction` and regenerate the patch table

Files: `src/continuum_swr_replay/analysis/replay/tables_arm_patch.py:221-300` (`build_event_patch_table`;
`patch_count_hard` aggregation at 277; `arm_fraction_from_event_total` already imported),
`src/continuum_swr_replay/analysis/replay/table_contracts.py` (`validate_event_patch_table`, ~278),
`src/continuum_swr_replay/analysis/replay/covariates/definitions.py` (catalog), the patch table's
persisted column order (find where `build_event_patch_table` orders/validates columns),
`scripts/qa/audit_stop_table_reward_history.py` (pattern for a table-regeneration parity audit;
generalize or write a sibling `audit_event_patch_fraction.py`), `src/continuum_swr_replay/table_parity.py`
(`diff_table(..., expected_added_columns=...)`).

- After the `patch_count_hard` aggregation add `patch_fraction = arm_fraction_from_event_total(patch_count_hard,
  n_nonlocal_bins − n_undecoded_bins)` (the two event-level counts are broadcast onto patch rows
  already or can be carried from the arm table — confirm and carry them if not). Add the column
  to the patch schema/order and to `validate_event_patch_table` (finite in [0, 1] where
  `patch_count_hard` is finite; sums to 1 per event with real-arm bins). Catalog entry: grain
  "event × candidate_patch", units "fraction", clock EVENT_RESPONSE, PRE_OUTCOME, producer
  `replay.tables_arm_patch.build_event_patch_table`, missingness "event has no real-arm bins".
- Tests in `tests/unit/test_replay_tables.py`: `patch_fraction` sums to 1.0 per event with real-arm
  bins and equals `patch_count_hard / (n_nonlocal_bins − n_undecoded_bins)`; NaN for an all-junction
  event; the arm and patch fractions of one event are consistent (patch = sum of its two leaf arms
  plus its trunk).
- Regenerate: back up `event_patch_table.parquet` to `_pre_fix_backup_20260905/`; rebuild with
  `prepare_data --from-cache --output-dir <staging>` (the path phase 1 used; ~1 h 40 for 218
  sessions, 12.5 GB; smoke `--max-per-animal 1` first) in `tmux` `patch_table_regen`; promote only
  `event_patch_table.parquet` after parity: every pre-existing patch column byte-identical (use
  `diff_table` with `expected_added_columns=["patch_fraction"]`), and every other rebuilt table
  byte-identical (SHA-256) to the live one. Write `docs/audits/2026-09-05-event-patch-fraction-regen.md`
  (row counts, SHA-256 old/new, per-column parity, per-event sum check, regeneration command and
  timing, promotion command). Note the schema stamp stays `covariate_contract_v5_dio_poke_trials`
  and that the golden-baseline gate must be invoked with `--expected-added-columns` for the patch
  table as well (extend the existing audit paragraph or the gate's documentation).

### Task 2: Event-clustered covariance and the estimator result schema

Read first: `designs.md` §"Event-clustered sandwich covariance" (complete code).

Files: `src/continuum_swr_replay/analysis/replay/fitting.py` (`fit_conditional_poisson` 84-260;
output dict ~230-250; `fit_null_model_conditional` 263; `fit_null_model_conditional_logit` 319;
`fit_conditional_logit` 377-528), `src/continuum_swr_replay/analysis/replay/contrast_aggregation.py:240-247`
(`typical_session_se`), `src/continuum_swr_replay/analysis/replay/simulation.py:307, 471`.

- Add `event_clustered_covariance(X, y, groups, params)` exactly as designed; in
  `fit_conditional_poisson` emit `bse_cluster_<term> = sqrt(diag(V))`. Add `estimator` and
  `llf_includes_constants=False` to every conditional fitter's output (`"conditional_poisson_count"`
  when the response column is `arm_count_hard`/`patch_count_hard`, `"conditional_poisson_fraction"`
  when it is `arm_fraction`/`patch_fraction`, `"conditional_logit_dominant"` for the logit; nulls
  inherit the caller's estimator string via a parameter, default count).
- `aggregate_contrast`: `typical_session_se` (and `std_typical_session_se`) use `bse_cluster_<term>`
  when that column exists in the frame and is finite for the session, else `bse_<term>`; record
  which was used in a new column `se_source` ∈ {"cluster", "model"}.
- Tests (`tests/unit/test_fitting.py`): on `simulate_with_event_covariates(L_e=1)` data,
  `bse_cluster` within 5 % of `bse` for every term; on `simulate_autocorrelated_counts(rho=0.95)`
  converted to fractions over 200 replicates (mark `slow` if > 20 s), the clustered SE is within
  20 % of the empirical SD of the coefficient while the model SE understates it by ≥ 2×; the
  fraction response's `llf_null == -n_events × log(n_categories)`; `estimator` and
  `llf_includes_constants` present with the right values for count, fraction, and logit fits.
  `tests/unit/test_contrast_aggregation.py`: `typical_session_se` prefers the cluster column and
  reports `se_source`.

### Task 3: Fraction as the shipped response, event-scale reporting, provenance for content fits

Files: `scripts/content/fit_replay_content_models.py` (`prepare_canonical_fit_request` 523-624,
response resolution at 373 and 621; `fit_model_per_session` 625; `_load_and_filter_cohort` 1298;
`_fit_content_ladder` 1472; argparse ~1050-1180; `_write_complete_output_bundle`),
`scripts/content/fit_swr_contrast_content.py` (`main` 584; `_run` calls at 627/637 pass
`response_col="arm_count_hard"` / `"patch_count_hard"`), `src/continuum_swr_replay/analysis/replay/model_comparison.py:119-180`
(`summarize_comparison`; `child_better_aic` 157, `lrt_significant` 160), the `print_model_results`
function in the content runner, `src/continuum_swr_replay/analysis/replay/fit_provenance.py`.

- Add `--response {fraction,hard_count}` (default `fraction`) to both runners. In
  `prepare_canonical_fit_request` resolve to `arm_fraction`/`patch_fraction` or
  `arm_count_hard`/`patch_count_hard`; validate the fraction column exists and sums to 1 per event
  (within 1e-9) on the loaded table; thread through `_ContentCohort.response_col` and the
  `_run(..., response_col=...)` calls. `freeze_common_cohort(..., drop_zero_response=True)` already
  removes NaN/zero-fraction events; assert the count of such events is reported in the cohort audit
  as `n_drop_zero_response`.
- Event-scale reporting: `summarize_comparison` keeps `delta_aic_*` and the animal medians and adds
  `estimator`; `print_model_results` and the model-comparison CSV/markdown emitted by the runner stop
  printing `lrt_significant` and `child_better_aic` in the headline block for count-response runs
  (they remain in the per-session frame); for fraction/logit runs they are printed with the label
  "event-scale". Record `estimator` and `response_col` in every results frame and in
  `model_comparison.pkl`.
- Provenance: both content runners call `record_inputs` right after loading (event-arm/patch table,
  stop table, qvals table where used) and `write_fit_provenance` for every pickle they write
  (`all_models.pkl`, per-model pickles if any, `model_comparison.pkl`, `swr_contrast_content_*.pkl`,
  audited variants); readers `scripts/content/generate_replay_content_figures.py`,
  `scripts/content/analyze_uncertainty_content.py`, `scripts/figures/plot_swr_contrast_coefs.py`
  (content half) gain `require_fresh_inputs` + `--allow-stale`; extend the fitter/reader mapping
  drift tests from phase 1 (`tests/unit/test_fit_provenance.py`) to these runners.
- Tests: `tests/unit/test_fit_replay_content_models.py` — `--response fraction` resolves to
  `arm_fraction` for arm and `patch_fraction` for patch; `--response hard_count` to the counts; a
  table whose fraction does not sum to 1 raises; `estimator` recorded; `tests/unit/test_replay_content_model_selection.py`
  — a planted allocation effect is recovered on a fraction response within 3 clustered SE.

### Task 4: Event-length control, shared standardization, and outcome strata in the SWR-content runner

Files: `scripts/content/fit_swr_contrast_content.py` (`_ZSCALARS` 120, `_z` 131, `_assemble_arm`
136-231 with `_TSC` built at 222, `_assemble_patch` 233, `_run` 366, `main` 584 with the timing gate
at ~595-600, `_z_reference`/`_fit_z_reference` 511-540), `src/continuum_swr_replay/analysis/replay/model_specs.py`
(`BASEc` 1144-1150, `_TSC` 1151, arm CS ladder below, `BASEc_PATCH` 1296, `SWR_CONTRAST_CONTENT_*_KEY_TERM`),
`src/continuum_swr_replay/analysis/replay/covariate_timing.py:64-166`, `covariates/definitions.py`,
`src/continuum_swr_replay/analysis/replay/standardization.py`.

- Replace the runner-local `_z` and its uses with `attach_standardized(ev, cols, support_label="fit_cohort_all_rows")`;
  persist the returned reference as `<artifact-stem>.z_reference.json` (arm and patch), replacing
  the ad-hoc `_z_reference` if it duplicates that role.
- Event-length control: build `log_n_nonlocal_bins` = `np.log(n_nonlocal_bins)` (n ≥ 1 on every
  event; assert), standardize it with `attach_standardized`, and create
  `log_n_nonlocal_bins_z_x_is_current_arm` and `log_n_nonlocal_bins_z_x_is_prev_arm`; patch
  analogues `log_n_nonlocal_bins_z_x_is_current_patch`, `..._x_is_prev_patch`. In `model_specs.py`
  add both arm products and `_TSC` to `BASEc` so every C/CS rung carries them (this makes CS2/CS4/
  CS5/CS5b/CS6 consistent with CS1 and C6–C12; `CS1nt_current`/`CPS1nt_current`, the deliberately
  uncontrolled twins, keep BASEc without `_TSC` — define `BASEc_NT` for them); add the patch
  products plus `log1p_time_since_stop_start_s_z_x_is_current_patch` to `BASEc_PATCH` (same twin
  treatment). Register every new product in `DERIVED_COVARIATE_SOURCES` with sources
  `["n_nonlocal_bins", "is_current_arm"]` etc., and add `n_nonlocal_bins` to `definitions.py`
  (grain event, units "decoder bins", clock EVENT_RESPONSE, PRE_OUTCOME, producer
  `replay.tables_event.build_event_table`).
- Outcome strata: `--outcome-stratum {all,rewarded,omitted}` (default `all`); `rewarded` keeps
  `is_rewarded_outcome == 1`, `omitted` keeps `== 0`; events with NaN outcome are excluded from both
  strata and their count by `is_swr` is logged; output artifact names gain `_rewarded`/`_omitted`
  suffixes (and the provenance/z-reference sidecars follow the artifact stem). The stratum is a
  cohort selection; the timing gate is unchanged.
- Tests: `tests/unit/test_swr_contrast_content.py` (create if absent; import the runner's pure
  helpers, never execute `main`): the products equal `z(log n) × indicator` on a synthetic frame;
  the arm/patch reference JSON reproduces the `_z` columns exactly; `rewarded` and `omitted` strata
  partition the finite-outcome events and exclude NaN; `tests/unit/test_covariate_timing.py`: the
  new products resolve to PRE_OUTCOME atoms; `tests/unit/test_model_specs.py`: every C/CS rung except
  the `nt` twins contains `_TSC` and both length products; `BASEc_NT` equals `BASEc` minus those three.

### Task 5: Leaf-only value chain

Files: `scripts/content/fit_replay_content_models.py` (`_load_and_filter_cohort` at-well filter
~1393; `_fit_content_ladder` `n_categories` selection; `_resolve_run_config` 1186 and
`canonical_output_subdir` 464), `src/continuum_swr_replay/analysis/replay/model_specs.py:1388-1403`
(`LEAF_ONLY_N_CATEGORIES`, `leaf_only_predictors`, `LEAF_ONLY_CONTRAST_RUNGS`, the comment block
above them about Mv/Mu/Mvu nesting).

- Add `--leaf-only`: after the at-well filter keep `candidate_arm < LEAF_ONLY_N_CATEGORIES`; set
  `n_categories = 6`; map every requested rung's predictors through `leaf_only_predictors`; require
  `--response fraction|hard_count` as usual; output directory gains the suffix `_leaf` (via
  `canonical_output_subdir`, so the canonical-bundle guard covers the leaf variants too); refuse
  `--leaf-only` with `--hierarchy patch`. Update the `model_specs.py` comment block to say
  Mv/Mu/Mvu are compared to leaf-only M0–M4 fitted in the same run
  (`--leaf-only --models M0,M0a,M1,M2,M3,M4,Mv,Mu,Mvu`).
- Tests (`tests/unit/test_fit_replay_content_models.py`): `--leaf-only` yields `n_categories == 6`,
  arm FE 6/7 absent from every rung's predictors, and directory suffix `_leaf`; patch + leaf-only
  raises; a leaf-only subset run cannot target a canonical bundle.

### Task 6: Refit all content ladders, compare, and document

Files: `Processed-Data/replay_content/models*/` (content dirs: `models`, `models_patch`,
`models_swr_{T,F}`, `models_patch_swr_{T,F}`, `models_event_grain`, `models/swr_contrast_content_*`),
`scripts/qa/compare_fits.py` (extend for content frames), `docs/audits/`, `docs/models_table.md:9-46,
155-196`, `docs/design/content_model_design.md`, `docs/covariate_dictionary.md`.

- Back up every content fit directory (move, preserving relative paths, with any sidecars) into
  `Processed-Data/replay_content/_pre_fix_backup_20260905/`. Smoke each runner configuration with
  `--max-per-animal 1` under `/usr/bin/time -v`, extrapolate, then in `tmux` session `content_refit`
  run: default arm ladder (restores `models/`), default patch ladder, `--swr-subset T` and `F` for
  both hierarchies, the leaf-only belief run (`--leaf-only --models M0,M0a,M1,M2,M3,M4,Mv,Mu,Mvu`),
  the event-grain run if the repo still ships it, and `fit_swr_contrast_content.py` for
  `--outcome-stratum all`, `rewarded`, `omitted`. Every artifact must carry a provenance sidecar.
- Extend `scripts/qa/compare_fits.py` with a content mode: join backup (count response) and new
  (fraction response) `all_models.pkl` on (session, model); per (model, term) report median
  coefficient old/new, median `bse` old vs `bse_cluster` new, `n_animals_pos` old/new, median
  per-session ΔAIC vs parent old/new (labelled as different scales), sessions fitted old/new; for
  the SWR-content contrast report the key-term rows old/new and the outcome-stratum rows side by
  side. Save `docs/audits/2026-09-05-content-refit-comparison.md` (generated; commit).
- Docs: `docs/models_table.md` conventions rows for content (response = per-event fraction; SEs =
  event-clustered; the count response retained for the DM comparison), content-arm block gains
  `Mu`/`Mvu` and the leaf-only chain, the C/CS ladder table gains the length products and `_TSC` in
  BASEc with the `nt` twins noted; `docs/design/content_model_design.md` gains a section on why the
  response changed (cite the review's measured numbers: 66 % single-arm events, ~6× SE, ΔAIC
  collapse) and on the event-length control; `docs/covariate_dictionary.md` entries for
  `patch_fraction` and the length products. `paper/claims_table.md` is NOT edited (phase 4).
