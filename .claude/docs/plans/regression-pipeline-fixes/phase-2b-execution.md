# Phase 2b execution plan (task-structured view of phase-2b-dirichlet-multinomial.md)

Spec authority: `.claude/docs/plans/regression-pipeline-fixes/phase-2b-dirichlet-multinomial.md`,
`designs.md` §"Dirichlet-multinomial content fitter" (model, likelihood, fitter code, simulator
signature), `shared-contracts.md` (Content estimator result schema; Fraction response; Fit
provenance sidecar), `overview.md` (decisions: fractions stay the shipped response; DM is the
comparison estimator and the direct "SWR broadens" test; inference is 5-animal sign consistency +
animal bootstrap, no hierarchical model), and the review
`.claude/docs/reviews/2026-09-03-regression-pipeline-review.md` (findings A1, A8, B1). Where this
file and those differ, they win.

## Global Constraints

- Run Python with `uv run …` from the repo root. Never install into the base environment. No new
  dependencies: the fitter uses `jax` (CPU build already installed, 0.8.0) and `scipy.optimize`.
- `uv run ruff check` and `uv run ruff format` on every file touched (check added lines with
  `ruff format --diff`; do not reformat unrelated code).
- Docstrings, test names, module names, and commit messages must not mention this plan, phase
  numbers, task numbers, or `.claude/docs/plans/`.
- Tests are value-level (numbers and behaviour, not presence). No test-only methods in production
  code. Shared setup in fixtures. Simulation tests pin seeds. Tests over 20 s wall are marked `slow`.
- Fail loud: contracts are enforced with `raise`, never warn-and-continue.
- Canonical tables are parquet under `Processed-Data/replay_content/`; read through
  `continuum_swr_replay.table_io.read_table`. Nothing under `Processed-Data/` is committed;
  `paper/figures/**/*.png` is git-ignored (write figures there anyway; captions/markdown next to
  them are tracked, as in `paper/figures/uncertainty_content/`). Never delete or overwrite an
  existing fitted artifact: DM outputs go to new `_dm`-suffixed directories/files only.
- Commit after each coherent unit with a message ending in
  `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01JGZPceMjeWcjYZQ26byzzz`.
- Long runs (> 5 min): smoke first (`--max-per-animal 1`), record wall time and peak RSS with
  `/usr/bin/time -v`, extrapolate to ~215 sessions, then run the full job inside `tmux` and report
  the session name. Poll with bounded waits.
- Do not run `git checkout` / `git stash` / `git clean`; the working tree holds the user's
  untracked files. Only edit files your task names plus their tests.
- No old code path is removed in this phase (the count and fraction conditional-Poisson paths, the
  conditional-logit event grain, and every runner flag stay as they are).
- State inherited from phases 1 and 2a (merged to master): content fitters in `fitting.py` return
  the estimator result schema below and persist `sd_<term>`, `separated_terms`, `bse_cluster_<term>`
  (fraction fits); `fit_replay_content_models.py` selects `--response {fraction,hard_count}` and
  `--leaf-only`, keys output dirs with `canonical_output_subdir(event_source, brain_area,
  hierarchy_kind, swr_subset, grain, response, leaf_only)` (~line 507) and refuses canonical bundles
  for `--models` subsets; `fit_swr_contrast_content.py` assembles the at-well arm/patch cohorts with
  `_assemble_arm` / `_assemble_patch` / `prepare_cohorts(ev, pt, *, outcome_stratum,
  max_per_animal)` (~355 / 478 / 307), fits leaf-only rungs listed in
  `model_specs.LEAF_ONLY_CONTRAST_RUNGS` by renormalizing the fraction response over the six
  leaves (`content_response.renormalize_fraction_over_candidates`), and writes provenance sidecars
  (`fit_provenance.write_fit_provenance(artifact_path, inputs, estimator, cli, extra=...)`);
  `contrast_aggregation.aggregate_contrast` uses `std_factor` (`BINARY_TERMS` per-category,
  `PER_UNIT_TERMS` per unit, else per SD), prefers `bse_cluster_<term>`, and drops a session for
  term T when T separated or `|coef_T| × sd_T ≥ 10`; `model_comparison.METADATA_COLUMNS =
  ("estimator", "response_col")` and `compare_nested_models` NaNs comparisons across mismatched
  metadata; `fit_replay_content_models.scale_label(estimator)` labels likelihood-based numbers.
- The estimator result schema (verbatim from shared-contracts.md, binding for the DM fitter): every
  content fitter returns `coef_/bse_/pval_<term>`, `bse_cluster_<term>` (conditional Poisson on
  fractions only — ABSENT for DM, whose model-based SE is already event-level), `n_events`,
  `n_rows`, `n_params`, `llf`, `llf_null`, `aic`, `bic`, `pseudo_r2`, `llf_includes_constants`
  (`False` for statsmodels conditional fits, `True` for DM), `converged`, `separated`,
  `separated_terms`, `n_convergence_warnings`, `coef_log_phi_<term>` / `bse_log_phi_<term>` (DM
  concentration coefficients: `log_phi_intercept` and `log_phi_is_swr`), `estimator` ∈
  {"conditional_poisson_fraction", "conditional_poisson_count", "conditional_logit_dominant",
  "dirichlet_multinomial"}. Cross-estimator AIC must equalize likelihood constants first.
- The DM model (verbatim from designs.md): per event `e` with candidate set `A_e`,
  `p_e = softmax_a(x_eaᵀβ)`, `log φ_e = γ₀ + γᵀ z_e` with `z_e = (is_swr,)` by default,
  `c_e ~ DirichletMultinomial(L_e, φ_e · p_e)`, `log P(c_e) = lgamma(L_e+1) − Σ_a lgamma(c_ea+1) +
  lgamma(φ_e) − lgamma(L_e+φ_e) + Σ_a [lgamma(c_ea + φ_e p_ea) − lgamma(φ_e p_ea)]`. The response is
  the COUNT (`arm_count_hard` / `patch_count_hard`), never the fraction. φ → ∞ recovers the count
  multinomial.

### Task 1: Dirichlet-multinomial fitter, simulator, and cross-validation

Read first: `designs.md` §"Dirichlet-multinomial content fitter" (complete fitter code — start from
it verbatim, then add what the schema requires).

Files: new `src/continuum_swr_replay/analysis/replay/dirichlet_multinomial.py`;
`src/continuum_swr_replay/analysis/replay/simulation.py` (`simulate_with_event_covariates` ~307,
`sample_arm_design` ~145, `_softmax` ~106); `src/continuum_swr_replay/analysis/replay/fitting.py`
(`within_group_full_rank` ~131 — reuse for the event-constant check; result-dict conventions in
`fit_conditional_poisson` ~161-350); `src/continuum_swr_replay/analysis/replay/_fit_diagnostics.py`
(`fit_status` ~101: the effect-scale separation rule — factor the rule into a helper both call if
`fit_status` cannot take the DM result directly; do not duplicate the threshold);
`src/continuum_swr_replay/analysis/replay/model_checking.py` (`cross_validate_conditional_poisson`
~21-170, the GroupKFold pattern to mirror); `src/continuum_swr_replay/analysis/replay/__init__.py`
(export). Tests: new `tests/unit/test_dirichlet_multinomial.py`; extend
`tests/unit/test_replay_content_simulation.py` (or the module that tests `simulation.py`).

- `fit_dirichlet_multinomial(long_df, predictors, *, response_col="arm_count_hard",
  event_col="event_id", concentration_cols=("is_swr",), n_categories=9, fixed_log_phi=None) ->
  dict | None` as designed, plus: `pval_<term>` (Wald z on `coef/bse`, for β and γ terms alike);
  `sd_<term>` for every β term (design-column SD over rows, same convention as `fitting.py`);
  `separated_terms` / `separated` by the same effect-scale rule as `fit_status`
  (`bse × sd ≥ 1` or non-finite SE); `llf_null` = the DM fit with β = 0 and γ free (fit it; it is a
  1-2 parameter problem), `pseudo_r2 = 1 − llf/llf_null`; `n_convergence_warnings` (0/1 from
  `res.success`); `llf_includes_constants=True`; `estimator="dirichlet_multinomial"`;
  `n_categories` taken from the packed shape and checked against the argument. Raise `ValueError`
  when a β predictor is constant within every event (use `within_group_full_rank`), when events
  carry unequal candidate counts, when any count is negative or non-integer, or when the response
  column looks like a fraction (all event totals ≤ 1). Events with `L_e == 0` are dropped before
  packing with the count recorded as `n_drop_zero_response` in the result. `fixed_log_phi` pins
  γ₀ (excluded from the free parameters and from `n_params`); `coef_log_phi_intercept` still
  reports the pinned value and `bse_log_phi_intercept` is NaN in that case.
- `multinomial_log_constant(y, groups) -> float` (Σ_e [lgamma(L_e+1) − Σ_a lgamma(c_ea+1)]) and
  `event_loglik(theta, X, Y, Z) -> np.ndarray` (per-event log P) as public functions.
- `cross_validate_dirichlet_multinomial(long_df, predictors, *, response_col, event_col,
  concentration_cols, n_folds=5, seed=0) -> dict | None`: GroupKFold over events; refit on train
  folds, evaluate held-out per-event `event_loglik`; return `cv_ll_per_event_mean`,
  `cv_ll_per_event_null_mean` (DM null: β = 0, γ refit per fold), `cv_bits_per_event_vs_null`
  (difference / log 2), `n_folds`, `n_events_cv`.
- `simulate_dirichlet_multinomial_counts(beta_true, *, log_phi_intercept, log_phi_is_swr=0.0,
  n_events=300, L_e=32, seed=None) -> pd.DataFrame` in `simulation.py`: draw the design with the
  same sampler `simulate_with_event_covariates` uses (so `is_swr` and every arm-varying predictor
  keep the real vocabulary, and `event_id` / `candidate_arm` columns match the canonical long
  format), then `p_e ~ Dirichlet(φ_e · softmax(X_e β))`, `c_e ~ Multinomial(L_e, p_e)`; write the
  counts to `arm_count_hard` and also the implied `arm_fraction`. Return the true `p_e` in
  `df.attrs["p_true"]` for tests.
- JIT compile cost: shapes change per session, so the fitter recompiles per call (~1 s). Accept it;
  do not pad shapes. `jax.config.update("jax_enable_x64", True)` at module import (check for an
  existing repo-wide setting first and reuse it if present).
- Tests (`tests/unit/test_dirichlet_multinomial.py`, all seeds pinned):
  `test_recovers_beta_and_phi` — on simulated data with φ ∈ {1, 3, 10, 100} (`n_events=300`,
  `L_e=32`, β for `is_current_arm=1.0`, `is_prev_arm=−0.5`), over 40 replicates each β is within
  3 SE in ≥ 95 % of replicates and `coef_log_phi_intercept` within 0.3 of `log φ` in median (mark
  `slow` if > 20 s; a 10-replicate fast variant must remain unmarked);
  `test_large_phi_matches_conditional_poisson` — φ = 1e4 data: DM β within 0.02 of
  `fit_conditional_poisson` β and `llf` within 1.0 after adding `multinomial_log_constant` to the
  statsmodels `llf`;
  `test_count_model_se_too_small_on_dm_data` (`slow`) — φ = 2, 100 replicates: count-model
  `bse / empirical SD < 0.5` while DM `bse / empirical SD ∈ [0.8, 1.25]` for `is_current_arm`;
  `test_is_swr_concentration_recovered` — planted `log_phi_is_swr = −1.0` recovered within 0.3
  (median over 20 replicates);
  `test_event_constant_predictor_rejected` — a column constant within events raises `ValueError`;
  `test_fraction_response_rejected` — passing `arm_fraction` raises;
  `test_fixed_log_phi_pins_intercept` — `n_params` drops by one, the intercept equals the pin, β
  still recovered;
  `test_cv_folds_over_events` — folds partition events (no event in two test folds),
  `cv_bits_per_event_vs_null > 0` on planted data;
  `test_result_matches_estimator_schema` — every key in the schema list above is present with the
  documented type/value (`llf_includes_constants is True`, `estimator ==
  "dirichlet_multinomial"`, no `bse_cluster_` key).

### Task 2: Runner integration and aggregation support

Files: `scripts/content/fit_replay_content_models.py` (`fit_model_per_session` ~860-960 — add the
DM branch; `canonical_output_subdir` ~507 and `canonical_output_subdirs` ~576 — `_dm` suffix;
CLI ~1440-1620 — `--estimator {conditional_poisson,dirichlet_multinomial}`, default
`conditional_poisson`; `scale_label` / `_SCALE_LABELS` ~1240-1270 — DM label "event-scale (full
likelihood)"; provenance estimator string), `scripts/content/fit_swr_contrast_content.py`
(`_fit_session` ~552, `_run` ~648 — estimator switch; `main` ~889 — the same flag; output file
suffix), `src/continuum_swr_replay/analysis/replay/contrast_aggregation.py` (`BINARY_TERMS` ~57,
`specs_from_ladder` ~318), `src/continuum_swr_replay/analysis/replay/model_comparison.py`
(`compare_nested_models` ~26 — refuse to compare rows whose `llf_includes_constants` differ),
`src/continuum_swr_replay/analysis/replay/covariate_timing.py` only if a new model-facing column is
introduced (none expected). Tests: `tests/unit/test_fit_replay_content_models.py`,
`tests/unit/test_swr_contrast_content.py`, `tests/unit/test_contrast_aggregation.py`,
`tests/unit/test_replay_content_model_selection.py`.

- Content runner: `--estimator dirichlet_multinomial` forces the count response (`--response
  fraction` together with it is a CLI error, message names both flags), calls
  `fit_dirichlet_multinomial` per session with `concentration_cols=("is_swr",)` when the cohort
  carries `is_swr` with both values, else `()`; with `--leaf-only` the counts are restricted to the
  six leaves and events with zero leaf mass are dropped (booked as `n_drop_zero_response` in the
  cohort audit, same bucket the fraction path uses). Output dir = `canonical_output_subdir(...)`
  + `_dm` (add an `estimator` parameter, default `"conditional_poisson"`, and include it in the
  `canonical_output_subdirs` product). Provenance `estimator = "dirichlet_multinomial"`.
- Small-session fallback (both runners): a session with fewer than 40 modelled events is fit with
  `fixed_log_phi` = the animal-pooled intercept, defined as the median `coef_log_phi_intercept`
  over that animal's sessions with ≥ 40 events from a first pass (same rung, same cohort); the row
  records `phi_source ∈ {"session", "animal_pooled"}`. If an animal has no session with ≥ 40
  events, the fallback is the median over all animals' qualifying sessions and `phi_source =
  "pooled_all_animals"`. `--min-events-session-phi 40` exposes the threshold. Log how many
  sessions fell back, per animal.
- SWR-contrast runner: `--estimator dirichlet_multinomial` switches `_run` to the DM fitter on
  the count response with `concentration_cols=("is_swr",)` for every rung (so
  `coef_log_phi_is_swr` is the concentration contrast on every row); leaf-only rungs use
  leaf-restricted counts as above. Artifacts get suffix `_dm` before the stratum suffix
  (`swr_contrast_content_arm_dm.pkl`, `..._audited_dm.pkl`, provenance and z-reference sidecars
  likewise). `aggregate_contrast` is run additionally for the key term `log_phi_is_swr` on every
  rung so the audited table carries the concentration contrast next to the β contrast.
- `contrast_aggregation.BINARY_TERMS` gains `log_phi_intercept` and `log_phi_is_swr` (per-category
  scale). `typical_session_se` falls back to `bse_` for DM rows (no cluster column) and reports
  `se_source = "model"`.
- `model_comparison.compare_nested_models`: NaN the ΔAIC/LRT when parent and child rows differ in
  `llf_includes_constants`, and expose `llf_includes_constants` through `METADATA_COLUMNS`.
- Tests: CLI rejects `--estimator dirichlet_multinomial --response fraction`; `canonical_output_subdir`
  appends `_dm` last and `canonical_output_subdirs` contains both estimators' names; a two-session
  synthetic cohort (one session with 20 events, one with 60) runs `fit_model_per_session(...,
  estimator="dirichlet_multinomial")` and the 20-event row has `phi_source == "animal_pooled"` with
  `coef_log_phi_intercept` equal to the 60-event session's; `_run` with the DM estimator emits
  `coef_log_phi_is_swr` on every rung row; `std_factor("log_phi_is_swr", sd)` returns factor 1 with
  the binary label; `compare_nested_models` NaNs a mixed-constants pair.

### Task 3: Three-estimator comparison script

Files: new `scripts/content/compare_content_estimators.py`; new
`tests/unit/test_compare_content_estimators.py`; `scripts/README.md` (one line under `content/`).

- Cohort: import and reuse the SWR-contrast runner's assembly (`_assemble_arm`, `prepare_cohorts`
  with `outcome_stratum="all"`), so the comparison runs on exactly the at-well cohort the shipped
  contrast fits use. `--max-per-animal` for smoke. Do not re-derive the cohort.
- Per session and rung ∈ {M2, M3, M4 from `MODEL_HIERARCHY`; CS1_current, CS4_armreward,
  CS5_armvalue from the contrast catalog}, fit three estimators on the same events: (i)
  `fit_conditional_poisson` on `arm_count_hard` (retired count model), (ii) `fit_conditional_poisson`
  on `arm_fraction` with `bse_cluster_` (shipped), (iii) `fit_dirichlet_multinomial` on
  `arm_count_hard` with `concentration_cols=("is_swr",)` and the Task 2 small-session fallback.
  Rungs in `LEAF_ONLY_CONTRAST_RUNGS` (CS5_armvalue) are fit leaf-only for all three estimators
  (fraction renormalized over leaves; counts restricted to leaves; events with zero leaf mass
  dropped for all three so the event sets match — assert equal `n_events` across estimators per
  (session, rung) and record the drop). Before any AIC comparison add `multinomial_log_constant`
  to the two statsmodels `llf`s (fraction fit: use the count matrix for the constant) and flag the
  fraction AIC as quasi.
- Outputs under `Processed-Data/replay_content/content_estimator_comparison/`:
  `content_estimator_comparison.parquet` — one row per (session, rung, term, estimator) with
  `coef`, `se` (cluster SE for the fraction estimator, model SE otherwise), `se_source`,
  `n_events`, `phi_source`, `llf_equalized`, `aic_equalized`;
  `content_estimator_comparison_summary.csv` — per (rung, term): median coef by estimator, median
  `se_estimator / se_count` ratio, `n_animals_pos` by estimator, median per-session ΔAIC vs parent
  by estimator (parent map: M2→M1 is not fit, so use M3→M2, M4→M3, CS*→C2_base fit alongside;
  fit the parents too), and for DM the animal-median `coef_log_phi_is_swr` with the sign count;
  `content_estimator_bootstrap.csv` — animal bootstrap: resample the 5 animals with replacement,
  2000 draws, seed 0; statistic = median over the resampled animals of each animal's per-session
  median coefficient; report 2.5/50/97.5 percentiles per (rung, term, estimator);
  provenance sidecar (`write_fit_provenance`) naming the input tables and `estimator =
  "three_estimator_comparison"`.
- Figures under `paper/figures/content_estimators/`: `coef_agreement.png` — per key term, scatter of
  per-session coefficients count vs fraction and count vs DM with the identity line and animal
  colour; `log_phi_by_session.png` — `coef_log_phi_intercept` per session (x = session order within
  animal) with `coef_log_phi_is_swr` as a second panel, both with the bootstrap interval as a band.
  A `content_estimators_caption.md` next to them (tracked) states what each panel shows and the
  seed.
- Test: on a synthetic two-session cohort built from `simulate_dirichlet_multinomial_counts` (with
  the columns the rungs need, `is_swr` both values, ≥ 40 events per session), the script's
  library entry point (`run_comparison(cohort, rungs, ...)`) produces the parquet with exactly one
  row per (session, rung, term, estimator), equal `n_events` across estimators per (session,
  rung), the summary with one row per (rung, term), and the bootstrap with the requested
  percentiles; with φ = 3 planted, DM SE ratio vs count > 1 for `is_current_arm`. Mark `slow` if
  over 20 s; keep a one-rung fast variant.

### Task 4: Real-data runs

Files: none edited beyond logs and the report. Runs write only into new `_dm` /
`content_estimator_comparison` locations.

- Smoke: `compare_content_estimators.py --max-per-animal 1` and
  `fit_swr_contrast_content.py --estimator dirichlet_multinomial --max-per-animal 1` under
  `/usr/bin/time -v`, outputs to a scratch `--output-dir`; record wall and RSS; extrapolate ×~43.
- Full runs in `tmux` session `content_estimators` (logs under `logs/content_estimators/`):
  the comparison script on the full cohort; `fit_swr_contrast_content.py --estimator
  dirichlet_multinomial` for `--outcome-stratum all` (rewarded/omitted strata only if the smoke
  extrapolation is under 2 h each — otherwise report them as not run). The DM arm ladder via
  `fit_replay_content_models.py --estimator dirichlet_multinomial` is run for the default arm
  hierarchy only (writes `models_dm/`); the patch/subset DM ladders are not run in this phase.
- Verify each artifact: loads, ~215 sessions, every rung present, `estimator` column all
  `dirichlet_multinomial`, provenance sidecar present, `phi_source` counts logged. Record the
  animal-level `coef_log_phi_is_swr` medians and sign count, and whether the CS5_armvalue β sign
  agrees across the three estimators, in the report.

### Task 5: Result doc and catalogue entries

Files: new `docs/design/content_estimator_comparison.md`; `docs/models_table.md` (Conventions
section ~9-46: an estimator row for DM; the content block: `_dm` bundles); `docs/covariate_dictionary.md`
(entries for `log_phi_intercept`, `log_phi_is_swr`, `phi_source`); `docs/README.md` ("Where results
live": the `_dm` bundles and the comparison directory).

- `content_estimator_comparison.md`: what each estimator assumes (count multinomial = independent
  bins; fraction + clustered SE = event-level quasi-likelihood; DM = explicit within-event
  concentration with an SWR modifier); the simulation recovery table (φ ∈ {1, 3, 10, 100}: β
  coverage, `log φ` error, count-model SE understatement — numbers from the Task 1 tests' fixtures
  re-run with the same seeds, cite the test names); the real-data summary table (from
  `content_estimator_comparison_summary.csv`, plus the bootstrap intervals for the key terms
  `is_current_arm`, `swr_x_is_current_arm`, `reward_fraction_candidate_arm_last10_visits`,
  `reward_belief_mean_candidate_arm_preoutcome`, and `log_phi_is_swr`); which conclusions are
  estimator-dependent (sign or animal count changes); the recommendation (fractions remain the
  shipped response; DM `log_phi_is_swr` is the direct "SWR broadens" test; state its sign, animal
  count, and bootstrap interval). Numbers live only here until phase 4. Date the doc, state cohort
  size, and link the figures and the parquet path. `paper/claims_table.md` is NOT edited.

## Review

Before the final review, confirm: every task implemented as specified; "Deliberately not in this
phase" honored (DM not made primary, no rate-family change, no hierarchical model, no code path
removed); validation-slice tests pass with `slow` marks where needed; tests are value-level; no
plan/phase references in shipped artifacts; docs updated in this phase; the comparison summary
and both figures are referenced from the result doc with their paths; every real-data artifact
has a provenance sidecar; the per-run smoke timings and full wall times are in the report.
