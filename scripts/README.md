# Scripts index

Analysis scripts grouped by domain. Run from the repo root, e.g.
`uv run python scripts/rate/fit_per_bin_rate_models.py`. Scripts import the package
(`continuum_swr_replay`) and read/write `Processed-Data/`; they do not import each other.

## `decoding/` — produce the decoded data + event/stop tables
- `run_decoding_bandit.py` — continuum decoding for the bandit task (HPC/mPFC/OFC).
- `run_decoding_contfrag.py` — decoding for the ContFrag (SWR/MUA) event sources.
- `run_memoryless_detection.py` — the Widloski & Foster sliding-window detector
  on each HPC epoch's saved decode, with its encoding-pair shuffle; one row per
  candidate sequence (spec 02). Loads the marks from the Spyglass database
  unless `--cache-dir` already holds them; a failing epoch is recorded and the
  rest run; `run.log` and `manifest.json` sit beside the outputs.
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
- `generate_memoryless_comparison.py` — supplemental figure: the ported Widloski & Foster
  sliding-window detector against the state-space detector on the Figure 1 simulation, with
  the sweep summary below; every verdict in its titles and caption is read from the data
  → `paper/figures/memoryless_comparison/`.
- `archive/` — retired figure scripts (not linted; kept for provenance).

## `qa/` — diagnostics / validation
- `check_data_availability.py`, `diagnose_trial_detection.py`, `compute_hpc_raw_stats.py`,
  `run_quality_metrics.py`, `analyze_event_spike_evidence.py` (spike-evidence reality check).
- `sweep_memoryless_detection.py` — injects replays of chosen path, duration, rate and
  distance into a simulated recording and scores both detectors by one rule; writes the raw
  table to `Processed-Data/` and the aggregates to
  `docs/design/replay_detection_comparison/detection_sweep_summary.csv` (about six minutes).
- `compare_memoryless_to_reference.py` — runs the memoryless port beside
  `memoryless_reference.py`, an independent re-implementation of the Widloski & Foster
  reference's sorted-cell arithmetic, on the Figure 1 simulation step by step and on
  every sweep cell; writes
  `docs/design/replay_detection_comparison/reference_check_summary.csv` (about ten
  minutes; `--n-seeds 1` for a one-minute check).
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
- `write_pipeline_map.py` — regenerates `docs/regression_pipeline_map.md`, the
  symbol-anchored map of the modules carrying a decoded session to a fitted
  regression, read out of the source with `ast`; `--check` fails on drift, as
  `tests/unit/test_pipeline_map_is_current.py` does.
- `compare_pfc_cohorts.py` — writes the mPFC unit-GLM before/after audit: per-session
  bins and events on either side of the stop cohort (from the bundles' own provenance
  sidecars), the decode/position alignment trim, the leading-edge coverage diagnostic,
  and the Figure 4 population loaded from both bundle trees. When the earlier tree
  also carries sidecars it audits a **rerun** as well: whether the two trees describe
  the same events, the unit-by-unit differences against the estimator's gate
  tolerances (per quantity and per session), the matched non-event control's tests,
  the estimator/device/fit counts per card and per session, and — given
  `--worker-time <batch>/worker*.time` — the wall clock each worker of the batch
  cost. When the after tree's sidecars name the penalised estimator it writes a
  different report instead (`--comparison` forces either): what the prior did to each
  unit's peak modulation against the same block fit unpenalised, split by whether that
  fit ran away; how many units carried a basis column with no spikes under it; what
  the two inferences call, per class and animal; how steady each nested pair's
  held-out difference is between folds; and the peaks' sensitivity to the prior,
  against whatever `_sigma<value>` bundles are on disk (the report names how many
  sessions those are). `--geometry <report.md>` folds in the null-geometry
  measurements beside the p-values they qualify.
- `measure_pfc_null_geometry.py` — measures what the mPFC held-out null's windows
  look like beside the real events, and changes nothing: the inter-onset spacing of
  the shipped matched control and of the null's own draws (rebuilt draw by draw from
  their recorded seeds, for the sessions named on the command line) against the real
  classes'; how much of the engagement null's draws land on the shipped control they
  are entered beside; how many units called at raw p < .05 have no positive held-out
  improvement of their own; and how many events of each class had no legal within-bout
  control and so entered neither side of its test.
- `diagnose_pfc_event_definition.py` — reports, per session, what the mPFC unit GLM's
  retired run-time event construction yielded against the canonical event table it now
  reads: the two event counts, how many events change SWR class once the lists are
  matched by overlap, the start-time shifts, and how many reclassifications are
  partial-overlap cases the two split rules may legitimately disagree on.
- `summarize_stage6_fdr.py` — prints Stage 6's per-role cross-session FDR table for
  any saved `stage6_is_stay_logistic*.pkl`, using the runner's own statistic, so a
  before/after comparison across populations reads one implementation.
- `check_pfc_ridge_lambda0.py` — writes the audit that the penalised solver with
  its penalty set to zero is still the maximum-likelihood one: it rebuilds each
  gate session's design *as it stood when the reference was fit* (the basis that
  dropped only patsy's first boundary function) from that session's own fit mask,
  event sets and seeded control draw, fits it unpenalised, and compares against
  the stored statsmodels diagnostics. Cells whose maximum-likelihood estimate
  does not exist are reported rather than gated — they are the ones the prior was
  introduced for.
- `compare_pfc_fitters.py` — **history.** Wrote the equivalence audit of the batched
  JAX Poisson fitter against the per-neuron statsmodels IRLS, from two runs of the
  same sessions (`--fitter jax` / `--fitter statsmodels`, both with
  `--fit-diagnostics`): every numeric column of the shipped statistics, the five
  modulation time courses, the best-model and NaN patterns that must be identical, the
  per-model parameters and fit criteria, and the wall clock of both runs. The unit
  runner no longer offers the unpenalised per-neuron backend, so this compares two
  archived trees rather than two runs anyone can make now.

## `pfc/` — cross-area / staged analyses
- `run_pfc_analysis_bandit.py`, `run_stage5b_retrospective.py`, `run_stage6_learning.py`.
  All three take `--stop-cohort`: the well-anchored stop cohort the hippocampal rate
  and content ladders are read on is the default (`at_well`, canonical artifact
  names), `all_stops` publishes an unrestricted sensitivity bundle under `_allstops`
  names, and Stage 5B accepts only `at_well` because its metrics need a current arm.
  `run_pfc_analysis_bandit.py` fits the unit GLMs under a Gaussian prior on the
  event-basis coefficients and scores them out of fold: `--penalty-sigma` (the prior
  standard deviation on the log-rate scale; another value writes a `_sigma<value>`
  sensitivity bundle and draws no null), `--n-folds` (blocked over whole immobility
  runs, with runs whose ±250 ms kernel supports overlap joined into one block before
  assignment), `--n-control-draws` with `--fold-seed` / `--control-draw-seed` (derived
  from the session key unless given, so a lone refit reproduces the batch's split and
  null). The canonical batch is **σ = 3**, **K = 5** and **R = 1,000** — a plus-one
  p-value floor of 1/1,001 — and every requested draw must be usable for a unit's
  p-value to be reported; the 20-draw floor is for deliberately shortened
  diagnostics. It takes `--gpu N` / `--allow-cpu` to say which card it may take on a shared
  host. A batch runs as one process per card — `--n_workers 1 --gpu N --sessions
  <partition file>` — because one card handed to several workers is refused, and the
  same card handed twice to one process is ignored.

## `exploratory/` — unsupported one-offs
Scratch scripts (`_*.py`): diagnostics, ad-hoc plots, verification checks. Not part of the
audited pipeline; outputs go to `.claude/research/*_outputs/`. May be stale.
