# Scripts index

Analysis scripts grouped by domain. Run from the repo root, e.g.
`uv run python scripts/rate/fit_per_bin_rate_models.py`. Scripts import the package
(`continuum_swr_replay`) and read/write `Processed-Data/`; they do not import each other.

## `decoding/` — produce the decoded data + event/stop tables
- `run_decoding_bandit.py` — continuum decoding for the bandit task (HPC/mPFC/OFC).
- `run_decoding_contfrag.py` — decoding for the ContFrag (SWR/MUA) event sources.
- `prepare_replay_content_data.py` — build the canonical replay-content tables
  (event / event-arm / event-patch / event-sequence / stop) across all sessions.
- `regenerate_stop_table.py` — rebuild the per-stop immobility table.
- `build_timebin_tables.py` — build the per-(stop×bin) design tables.
- `prepare_contfrag_replay_content_data.py` — assemble ContFrag content tables.

## `rate/` — event-rate models + their plots
- `fit_per_bin_rate_models.py`, `fit_per_stop_rate_models.py` — Family B / A rate ladders.
- `fit_swr_contrast_rate.py` — the `is_swr`-contrast per-bin ladder (G0–G8 / S0–S10).
- `fit_uncertainty_rate.py`, `fit_cross_history_rate.py`, `fit_swr_power_model.py` — MU / cross-history / SWR-power models.
- `plot_per_bin_rate_models.py`, `plot_per_bin_rate_summary.py`, `plot_model_checking.py`, `plot_uncertainty_rate.py` — rate figures.
- `summarize_per_bin_rate_models.py` — generates `docs/per_bin_rate_results.md`.

## `content/` — replay content / choice models
- `fit_replay_content_models.py` — arm/patch content hierarchy (M0–M7 / P0–P4), per `is_swr` split.
- `fit_swr_contrast_content.py` — the HMM-harmonized content `is_swr`-contrast ladder (C / CP).
- `compare_event_source_models.py` — choice (clogit) model comparison.
- `compare_content_estimators.py` — conditional Poisson (counts), conditional Poisson
  (fractions, clustered SE) and Dirichlet-multinomial fit on one shared event set,
  with equalized AIC, an animal bootstrap, and `paper/figures/content_estimators/`.
- `analyze_content_occupancy.py`, `quantify_swr_overlap_geometry.py`, `generate_replay_content_figures.py`.
- `analyze_uncertainty_content.py` — trial×well reward-belief uncertainty profiles,
  absolute content rates, response sensitivities, and cohort audit tables.

## `behavior/`
- `fetch_spatial_bandit_behavior.sh` — pull the canonical SpatialBanditTask behavior CSVs.
- `build_qvals_table.py` — build the per-stop behavioral-model covariate table
  (`qvals_table`) from the shared DataJoint RL-model and trial-timing schemas.

## `figures/` — publication figures
- `generate_figure01.py … generate_figure04.py`, `generate_all_figures.py` (orchestrator).
  `generate_figure04.py` takes `--stop-cohort` and describes the bundle of that name:
  `at_well` (default) is the canonical population, `all_stops` reads the `_allstops`
  sensitivity bundle and refits the example neuron unrestricted.
- `generate_uncertainty_content_figure.py` — ground-up uncertainty/content story
  and diagnostic figures from the supported content-analysis artifacts.
- `generate_supp_*.py`, `generate_event_gallery.py`, `run_event_gallery.sh`.
- `plot_swr_contrast_coefs.py` — per-animal coefficient forest plots for the is_swr
  rate + content contrast ladders (standardized effect sizes, all sessions + animal 95% CI) → `paper/figures/swr_contrast/`.
- `archive/` — retired figure scripts (not linted; kept for provenance).

## `qa/` — diagnostics / validation
- `check_data_availability.py`, `diagnose_trial_detection.py`, `compute_hpc_raw_stats.py`,
  `run_quality_metrics.py`, `analyze_event_spike_evidence.py` (spike-evidence reality check).
- `compare_fits.py` — compares old vs new fitted rate artifacts (Family B, Family A
  and the `is_swr`-contrast ladder) on the quantities that survive a respecification
  — joint reward step, reward-free goodness of fit, frozen-cohort size, per-rung
  session counts, per-term separation shares — and writes the markdown comparison
  under `docs/audits/`.
- `check_fit_provenance.py` — sweeps the whole `models*/` tree and reports which
  fitted artifacts are stale for the tables they read, from the SHA-256 sidecars
  each runner writes.
- `audit_event_patch_fraction.py`, `audit_stop_table_reward_history.py` — keyed-parity
  audits of a candidate event-patch / stop table against its pre-change reference:
  every pre-existing column must be unchanged and only the declared new columns may
  appear.
- `freeze_regression_baseline.py`, `compare_to_golden_baseline.py` — freeze the
  canonical tables and model outputs as a read-only golden baseline (with a sha256
  manifest), and diff a regenerated root against it.
- `mutation_check.py` — AST mutation check on the covariate builders; the measured
  floor is recorded in `docs/design/builder_test_floor.md`.
- `summarize_cohort_audits.py` — rolls the per-run `*_cohort_audit.csv` files into
  one table.
- `compare_pfc_cohorts.py` — writes the mPFC unit-GLM before/after audit: per-session
  bins and events on either side of the stop cohort (from the bundles' own provenance
  sidecars), the decode/position alignment trim, the leading-edge coverage diagnostic,
  and the Figure 4 population loaded from both bundle trees.
- `diagnose_pfc_event_definition.py` — reports, per session, what the mPFC unit GLM's
  retired run-time event construction yielded against the canonical event table it now
  reads: the two event counts, how many events change SWR class once the lists are
  matched by overlap, the start-time shifts, and how many reclassifications are
  partial-overlap cases the two split rules may legitimately disagree on.
- `summarize_stage6_fdr.py` — prints Stage 6's per-role cross-session FDR table for
  any saved `stage6_is_stay_logistic*.pkl`, using the runner's own statistic, so a
  before/after comparison across populations reads one implementation.

## `pfc/` — cross-area / staged analyses
- `run_pfc_analysis_bandit.py`, `run_stage5b_retrospective.py`, `run_stage6_learning.py`.
  All three take `--stop-cohort`: the well-anchored stop cohort the hippocampal rate
  and content ladders are read on is the default (`at_well`, canonical artifact
  names), `all_stops` publishes an unrestricted sensitivity bundle under `_allstops`
  names, and Stage 5B accepts only `at_well` because its metrics need a current arm.

## `exploratory/` — unsupported one-offs
Scratch scripts (`_*.py`): diagnostics, ad-hoc plots, verification checks. Not part of the
audited pipeline; outputs go to `.claude/research/*_outputs/`. May be stale.
