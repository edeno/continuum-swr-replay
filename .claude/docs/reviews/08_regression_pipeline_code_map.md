# Content and Rate Regression Pipeline Code Map

**Date:** 2026-07-29

**Status:** Current-working-tree execution map; no statistical-correctness verdict

**Scope:** Data loading, decoding, table construction, covariate attachment, and the
production content and rate regression entry points

## Purpose and audit boundary

This document identifies the executable code that carries the data from Spyglass/NWB-backed
session loading to the replay-content and replay-rate regression models. It is intended to define
the code-review boundary for a later assessment of whether each loading, filtering, alignment,
transformation, and modeling step is appropriate.

This is a map, not a new statistical audit. Existing statistical findings remain in
[`07_comprehensive_regression_model_statistical_audit.md`](./07_comprehensive_regression_model_statistical_audit.md).
Line numbers and model membership below reflect the working tree reviewed on the date above.
For the core ladders, the Python model specifications are the source of truth when prose tables or
design notes disagree with the executable code. Some secondary models define their predictors
inside their runners instead.

The primary production flow is:

```text
Spyglass / NWB
    |
    v
bandit-task session loader
    |
    v
non-local decoder + aligned data_over_time
    |
    v
canonical event, arm, patch, stop, and onset tables
    |
    v
behavioral-value and replay covariate joins
    |
    +----------------------------+
    |                            |
    v                            v
content allocation models       rate occurrence models
```

## 1. Raw and session-level data loading

The main loader is
[`load_data()`](../../../src/continuum_swr_replay/data_loaders/bandit_task.py#L180). It delegates
the retrieval and alignment of position, spikes, LFP, digital inputs, trials, and task variables
for a session. This is the first relevant review boundary because all downstream event detection
and behavioral covariates inherit its timestamps, sampling, and trial labels.

| Role | Executable code | Downstream relevance |
|---|---|---|
| Load a bandit-task session | [`bandit_task.py:180`](../../../src/continuum_swr_replay/data_loaders/bandit_task.py#L180) | Supplies neural, position, DIO, and trial data to decoding |
| Construct well visits and trials | [`trials.py:261`](../../../src/continuum_swr_replay/data_loaders/trials.py#L261) and [`trials.py:355`](../../../src/continuum_swr_replay/data_loaders/trials.py#L355) | Converts beam breaks, pumps, position, and graph metadata into behavioral visits and trial summaries |
| Run the primary hippocampal decoder | [`run_decoding_bandit.py:289`](../../../scripts/decoding/run_decoding_bandit.py#L289) | Writes decoder results, environment metadata, and the aligned `*_HPC_80_data_over_time.pkl` artifact |
| Assemble aligned time series | [`data_aggregation.py:283`](../../../src/continuum_swr_replay/analysis/hippocampus/data_aggregation.py#L283) | Produces `data_over_time`, including decoded probabilities and event indicators |

`load_data()` delegates its concrete retrieval and filtering work to:

- position:
  [`get_position_info()`](../../../src/continuum_swr_replay/data_loaders/position.py#L133);
- spike data:
  [`get_spike_data()`](../../../src/continuum_swr_replay/data_loaders/spikes.py#L451);
- theta, ripple, and multiunit signals:
  [`lfp.py`](../../../src/continuum_swr_replay/data_loaders/lfp.py);
- digital inputs:
  [`load_dios()`](../../../src/continuum_swr_replay/data_loaders/events.py#L183); and
- task variables:
  [`make_task_variables()`](../../../src/continuum_swr_replay/data_loaders/task_variables.py#L480).

Two additional transformation modules sit between decoder output and the canonical tables:

- [`event_detection.py`](../../../src/continuum_swr_replay/analysis/hippocampus/event_detection.py)
  defines confident event intervals and overlap classification; and
- [`posterior_analysis.py`](../../../src/continuum_swr_replay/analysis/hippocampus/posterior_analysis.py)
  computes conditional non-local posteriors, decoded step distances/speeds, and posterior summary
  quantities.

The primary decoded input to the regression-preparation code is therefore not the raw NWB data
directly. It is the persisted, time-aligned decoder output produced by the steps above.
`prepare_data._process_session()` reads that pickle directly at
[`prepare_data.py:222`](../../../src/continuum_swr_replay/analysis/replay/prepare_data.py#L222);
the convenience `load_data_over_time()` function is not on this canonical preparation path.

## 2. Canonical replay-table construction

The production preparation entry point is
[`prepare_data.py`](../../../src/continuum_swr_replay/analysis/replay/prepare_data.py). Its
session worker begins at
[`_process_session()`](../../../src/continuum_swr_replay/analysis/replay/prepare_data.py#L199), and
the command-line orchestration begins at
[`main()`](../../../src/continuum_swr_replay/analysis/replay/prepare_data.py#L307).

The core table builders are in
[`tables.py`](../../../src/continuum_swr_replay/analysis/replay/tables.py):

| Canonical table | Builder | Intended row unit | Main model use |
|---|---|---|---|
| `event_table` | [`build_event_table()`](../../../src/continuum_swr_replay/analysis/replay/tables.py#L215) | One row per detected non-local event | Event metadata and joins |
| `event_arm_table` | [`build_event_arm_table()`](../../../src/continuum_swr_replay/analysis/replay/tables.py#L1076) | Nine candidate-arm rows per event | Arm-level content models |
| `event_patch_table` | [`build_event_patch_table()`](../../../src/continuum_swr_replay/analysis/replay/tables.py#L1373) | Three candidate-patch rows per event | Patch-level content models |
| `stop_table` | [`build_stop_table()`](../../../src/continuum_swr_replay/analysis/replay/tables.py#L774) | One row per immobility stop, including zero-event stops | Per-stop rates and the base table for per-bin rates |
| `event_sequence_table` | [`build_event_sequence_table()`](../../../src/continuum_swr_replay/analysis/replay/tables.py#L672) | One row per retained in-event time bin | Sequence/fragmentation analyses, not the core arm/patch response |

This distinction in row units is central to interpreting the regressions:

- Content models compare alternatives *within an observed event*. They do not model whether an
  event occurred.
- Rate models compare event counts or event onsets across stops and time bins. They do model event
  occurrence.
- Zero-event stops must remain in `stop_table` for rate models, but there is no corresponding
  content choice set for a stop without an event.

The common low-speed bout definition is implemented by
[`immobility_bout_intervals()`](../../../src/continuum_swr_replay/analysis/replay/_helpers.py#L152).
The final SWR/non-SWR per-stop count split is added by
[`add_nonlocal_swr_split()`](../../../src/continuum_swr_replay/analysis/replay/tables.py#L1009).

### Per-bin onset tables

[`build_timebin_tables.py`](../../../scripts/decoding/build_timebin_tables.py) calls:

- [`build_event_onset_table()`](../../../src/continuum_swr_replay/analysis/replay/timebin_tables.py#L93)
  to create the sparse onset table used by production per-bin fits; and
- [`build_stop_timebin_table()`](../../../src/continuum_swr_replay/analysis/replay/timebin_tables.py#L140)
  to create a dense, default-4-ms descriptive table.

The production per-bin regressions do **not** fit the persisted
`stop_timebin_table.parquet`. They dynamically aggregate 15-ms model bins from
`stop_table.pkl` plus the sparse `event_onset_table.pkl`; `event_table.pkl` supplies the SWR label
matched onto non-local onsets. The dense table is a descriptive and diagnostic artifact and
should not be mistaken for the model design matrix.

Onset extraction, event-type definitions, reward-bout classification, and session keys are implemented
in [`timebin_tables.py:33-139`](../../../src/continuum_swr_replay/analysis/replay/timebin_tables.py#L33).
Reward bouts are flagged with `is_reward_stop`; they are not removed by this step.

## 3. Covariates, timing, scaling, and behavioral values

Replay covariates are constructed in
[`covariates.py`](../../../src/continuum_swr_replay/analysis/replay/covariates.py), beginning with
the shared covariate utilities near
[`covariates.py:47`](../../../src/continuum_swr_replay/analysis/replay/covariates.py#L47).

Four supporting modules are part of the model input contract:

| Concern | Code | Review question |
|---|---|---|
| Covariate construction and joins | [`covariates.py`](../../../src/continuum_swr_replay/analysis/replay/covariates.py) | Is each value computed on the correct behavioral unit and joined to the intended event/stop? |
| Temporal availability | [`covariate_timing.py:128`](../../../src/continuum_swr_replay/analysis/replay/covariate_timing.py#L128) | Was the covariate knowable at the modeled time, with no future leakage? |
| Standardization | [`standardization.py:28`](../../../src/continuum_swr_replay/analysis/replay/standardization.py#L28) | Is scaling fitted on the intended cohort and reused consistently? |
| Behavioral Q/value alignment | [`qvals_table.py:271`](../../../src/continuum_swr_replay/analysis/behavior/qvals_table.py#L271) | Are trial-level model values assigned to the correct stops and events? |

The broader behavioral-value path includes the DataJoint result fetch and merge near
[`qvals_table.py:296`](../../../src/continuum_swr_replay/analysis/behavior/qvals_table.py#L296),
per-animal construction near
[`qvals_table.py:343`](../../../src/continuum_swr_replay/analysis/behavior/qvals_table.py#L343),
and the build/write entry point at
[`qvals_table.py:381`](../../../src/continuum_swr_replay/analysis/behavior/qvals_table.py#L381).

Not every scale transformation goes through `standardization.py`. The SWR-content runner defines
runner-local z-scoring at
[`fit_swr_contrast_content.py:117`](../../../scripts/content/fit_swr_contrast_content.py#L117),
and the SWR-rate runner computes separate reporting-scale references near
[`fit_swr_contrast_rate.py:449`](../../../scripts/rate/fit_swr_contrast_rate.py#L449).

### Two-pass behavioral-value dependency

The value-model table and replay tables form an intentional materialization loop:

```text
prepare_data
    |
    v
initial stop_table
    |
    v
qvals_table --write
    |
    v
prepare_data --from-cache
    |
    v
replay tables with behavioral-value columns attached
```

`qvals_table` starts from the canonical `stop_table`, while `prepare_data` can attach an already
materialized `qvals_table`. A clean build therefore requires the two passes shown above. Reviewing
only the final model runner would miss this upstream dependency and the possibility of stale or
misaligned value artifacts.

## 4. Replay-content models

### 4.1 Base arm and patch hierarchy

The authoritative arm and patch formulas are in
[`model_specs.py`](../../../src/continuum_swr_replay/analysis/replay/model_specs.py):

- the arm hierarchy's rationale and predictor blocks begin near
  [`model_specs.py:153`](../../../src/continuum_swr_replay/analysis/replay/model_specs.py#L153),
  with the executable `MODEL_HIERARCHY` dictionary at
  [`model_specs.py:240`](../../../src/continuum_swr_replay/analysis/replay/model_specs.py#L240);
- the executable patch hierarchy is
  [`MODEL_PATCH`](../../../src/continuum_swr_replay/analysis/replay/model_specs.py#L365).

The production runner is
[`fit_replay_content_models.py`](../../../scripts/content/fit_replay_content_models.py):

- session-level fitting:
  [`fit_model_per_session()`](../../../scripts/content/fit_replay_content_models.py#L135);
- command-line setup and input selection:
  [`main()`](../../../scripts/content/fit_replay_content_models.py#L274);
- canonical-table loading and the legacy fallback begin near
  [`fit_replay_content_models.py:406`](../../../scripts/content/fit_replay_content_models.py#L406);
- common complete-case event-cohort construction begins near
  [`fit_replay_content_models.py:468`](../../../scripts/content/fit_replay_content_models.py#L468);
- the per-session hierarchy is fit beginning near
  [`fit_replay_content_models.py:549`](../../../scripts/content/fit_replay_content_models.py#L549),
  followed by result, cohort-audit, and comparison outputs.

The eight arm fixed-effect columns used by the richer arm models are generated at fit time by
[`add_arm_fixed_effects()`](../../../src/continuum_swr_replay/analysis/replay/_helpers.py#L395);
they are not persisted in the canonical event-arm table.

The primary hard-count response is `arm_count_hard`: the number of qualifying decoded time bins
allocated to each of nine arms within an event. Patch counts are obtained by summing the relevant
arm counts to three patches.

The estimator is the conditional Poisson implementation in
[`fit_conditional_poisson()`](../../../src/continuum_swr_replay/analysis/replay/fitting.py#L67).
Conditioning on the event total makes this an allocation model: coefficients describe how decoded
content is distributed among arms or patches, conditional on the event and its total qualifying
decoded content. It is not a model of event incidence or total event content.

The same module also contains
[`fit_conditional_logit()`](../../../src/continuum_swr_replay/analysis/replay/fitting.py#L285), which
is used by the base runner's explicit `--logit` dominant-arm robustness path near
[`fit_replay_content_models.py:680`](../../../scripts/content/fit_replay_content_models.py#L680).
It is not the estimator for the primary hard-count content hierarchy.

Model diagnostics and comparisons are implemented in:

- [`model_checking.py`](../../../src/continuum_swr_replay/analysis/replay/model_checking.py); and
- [`model_comparison.py`](../../../src/continuum_swr_replay/analysis/replay/model_comparison.py).

### 4.2 SWR-versus-non-SWR content contrasts

The SWR content specifications begin near
[`model_specs.py:905`](../../../src/continuum_swr_replay/analysis/replay/model_specs.py#L905).
They define the arm `C`/`CS` and patch `CP`/`CPS` ladders and their key terms.

The runner is
[`fit_swr_contrast_content.py`](../../../scripts/content/fit_swr_contrast_content.py):

- arm assembly and interaction construction:
  [`_assemble_arm()`](../../../scripts/content/fit_swr_contrast_content.py#L122);
- patch aggregation:
  [`_assemble_patch()`](../../../scripts/content/fit_swr_contrast_content.py#L174);
- per-session fitting:
  [`_fit_session()`](../../../scripts/content/fit_swr_contrast_content.py#L207);
- orchestration, output, and provenance:
  [`main()`](../../../scripts/content/fit_swr_contrast_content.py#L484).

These models use the same conditional-Poisson content estimator, with interactions used to test
whether content allocation differs by SWR membership. Cross-level aggregation helpers are in
[`contrast_aggregation.py:57`](../../../src/continuum_swr_replay/analysis/replay/contrast_aggregation.py#L57).

### 4.3 Alternate ContFrag event source

The base content runner also supports a ContFrag-derived event source through its
`--event-source` and `--brain-area` path resolution:

- decoder: [`run_decoding_contfrag.py:249`](../../../scripts/decoding/run_decoding_contfrag.py#L249);
- content-table preparation:
  [`prepare_contfrag_replay_content_data.py:171`](../../../scripts/decoding/prepare_contfrag_replay_content_data.py#L171);
- table logic:
  [`analysis/contfrag/tables.py`](../../../src/continuum_swr_replay/analysis/contfrag/tables.py);
- path selection:
  [`resolve_contfrag_paths()`](../../../scripts/content/fit_replay_content_models.py#L100).

This is an alternate upstream event/table source feeding the same base content fitting machinery,
not a separate estimator.

## 5. Replay-rate models

### 5.1 Family A: per-stop event counts

The per-stop model rationale begins near
[`model_specs.py:481`](../../../src/continuum_swr_replay/analysis/replay/model_specs.py#L481), and
the executable `R0`-`R2`/`Rlag` catalog begins at
[`MODEL_PER_STOP_RATE`](../../../src/continuum_swr_replay/analysis/replay/model_specs.py#L514).

The runner is
[`fit_per_stop_rate_models.py`](../../../scripts/rate/fit_per_stop_rate_models.py). Its explicit
response mapping is defined at
[`fit_per_stop_rate_models.py:60`](../../../scripts/rate/fit_per_stop_rate_models.py#L60):

| Outcome label | `stop_table` response |
|---|---|
| `swr` | `n_SWR` |
| `non_local` | `n_non_local` |
| `non_local_swr` | `n_non_local_SWR` |
| `non_local_no_swr` | `n_non_local_no_SWR` |

The hierarchy is run by
[`fit_per_stop_rate_hierarchy()`](../../../scripts/rate/fit_per_stop_rate_models.py#L68).
Each session/model/outcome fit is delegated to
[`fit_per_stop_rate_model()`](../../../src/continuum_swr_replay/analysis/replay/per_stop_rate_model.py#L53).
It is a Poisson GLM for event counts with `log(immobility_time)` as an exposure offset.
Its per-model complete-case and positive-exposure filter begins near
[`per_stop_rate_model.py:86`](../../../src/continuum_swr_replay/analysis/replay/per_stop_rate_model.py#L86).
Cross-validation is implemented in
[`cross_validate_per_stop_rate_model()`](../../../src/continuum_swr_replay/analysis/replay/per_stop_rate_model.py#L175).

### 5.2 Family B: per-bin event intensity

The executable per-bin configuration begins at
[`model_specs.py:559`](../../../src/continuum_swr_replay/analysis/replay/model_specs.py#L559), its
specification class is
[`PerBinRateModelSpec`](../../../src/continuum_swr_replay/analysis/replay/model_specs.py#L611), and
the ladder is
[`MODEL_PER_BIN_RATE`](../../../src/continuum_swr_replay/analysis/replay/model_specs.py#L705):

- `BIN_S = 0.015`, so all production ladder models use 15-ms bins;
- `TMAX = 4.0`, so only the first four seconds of a stop are modeled;
- fixed spline knots are at 1, 2, and 3 seconds;
- ten onset-history lags cover 150 ms;
- the `M` model ladder begins in this section.

The design is assembled dynamically by helpers in
[`analysis/replay/_helpers.py`](../../../src/continuum_swr_replay/analysis/replay/_helpers.py):

- assign SWR membership to non-local onsets by within-stop rank:
  [`_helpers.py:490`](../../../src/continuum_swr_replay/analysis/replay/_helpers.py#L490);
- aggregate stops and sparse onsets into model bins:
  [`_helpers.py:581`](../../../src/continuum_swr_replay/analysis/replay/_helpers.py#L581);
- define/select the modeled event populations:
  [`_helpers.py:735`](../../../src/continuum_swr_replay/analysis/replay/_helpers.py#L735).

The production runner is
[`fit_per_bin_rate_models.py`](../../../scripts/rate/fit_per_bin_rate_models.py):

- one-session assembly and population loop:
  [`_fit_one_session()`](../../../scripts/rate/fit_per_bin_rate_models.py#L94);
- hierarchy orchestration:
  [`fit_per_bin_rate_models()`](../../../scripts/rate/fit_per_bin_rate_models.py#L157);
- command-line loading and output:
  [`main()`](../../../scripts/rate/fit_per_bin_rate_models.py#L271).

The runner reads `stop_table.pkl`, `event_onset_table.pkl`, and `event_table.pkl` near
[`fit_per_bin_rate_models.py:289`](../../../scripts/rate/fit_per_bin_rate_models.py#L289).
Its work-item construction requires a session to have an onset-table group
([`fit_per_bin_rate_models.py:171`](../../../scripts/rate/fit_per_bin_rate_models.py#L171)), an
inclusion rule that belongs in the later cohort audit.

The estimator is
[`fit_per_bin_rate_model()`](../../../src/continuum_swr_replay/analysis/replay/per_bin_rate_model.py#L84).
It performs per-model listwise filtering and applies minimum-bin/minimum-event gates before fitting
a Poisson event-intensity model with `log(exposure)` offset. Bins are nominally 15 ms, but
[`aggregate_to_timebins()`](../../../src/continuum_swr_replay/analysis/replay/_helpers.py#L581)
retains a shorter terminal remainder bin and records its actual exposure. Cross-validation groups
by whole stops, and goodness-of-fit is implemented in
[`cross_validate_per_bin_rate_model()`](../../../src/continuum_swr_replay/analysis/replay/per_bin_rate_model.py#L195)
and
[`per_bin_rate_gof()`](../../../src/continuum_swr_replay/analysis/replay/per_bin_rate_model.py#L306).

### 5.3 SWR-versus-non-SWR rate contrasts

The `G` and `S` contrast specifications are generated near
[`model_specs.py:829-903`](../../../src/continuum_swr_replay/analysis/replay/model_specs.py#L829).

The runner is
[`fit_swr_contrast_rate.py`](../../../scripts/rate/fit_swr_contrast_rate.py):

- its local Statsmodels Poisson fitter:
  [`_fit()`](../../../scripts/rate/fit_swr_contrast_rate.py#L145);
- per-session/population design and fitting:
  [`_fit_one_session()`](../../../scripts/rate/fit_swr_contrast_rate.py#L193);
- `qvals_table`, `stop_table`, and sparse onset loading:
  [`_load()`](../../../scripts/rate/fit_swr_contrast_rate.py#L237);
- orchestration and output:
  [`main()`](../../../scripts/rate/fit_swr_contrast_rate.py#L342).

It reuses the Family-B constants, event-population selectors, and 15-ms
`aggregate_to_timebins()` design builder, but it does not call `fit_per_bin_rate_model()`.
The `G` rungs are fit separately to the three non-local strata; the `S` rungs use a stacked
SWR/non-SWR contrast design.

## 6. Secondary and adjacent regressions

The following scripts use related inputs or model components but are not the two primary
content/rate hierarchies:

| Script | Role |
|---|---|
| [`fit_uncertainty_rate.py`](../../../scripts/rate/fit_uncertainty_rate.py) | Uncertainty/value extensions of the per-bin rate framework |
| [`fit_cross_history_rate.py`](../../../scripts/rate/fit_cross_history_rate.py) | Cross-history rate models |
| [`fit_swr_power_model.py`](../../../scripts/rate/fit_swr_power_model.py) | Event-level OLS model of SWR power; a mark/amplitude model, not an occurrence-rate model |
| [`compare_event_source_models.py`](../../../scripts/content/compare_event_source_models.py) | Cross-source patch-content, patch-switch, per-stop-rate, conditional-choice, and switch-aligned analyses |
| [`analyze_uncertainty_content.py`](../../../scripts/content/analyze_uncertainty_content.py) | Content-adjacent uncertainty analysis |

These should be audited after the shared data contracts and core content/rate paths because they
inherit many of the same joins, covariates, timing assumptions, and cohort decisions.

## 7. Tests that define or protect the pipeline

The most relevant tests are:

| Layer | Tests |
|---|---|
| Canonical replay tables | [`test_replay_tables.py:341`](../../../tests/unit/test_replay_tables.py#L341) |
| Conditional estimators | [`test_fitting.py:50`](../../../tests/unit/test_fitting.py#L50) |
| Simulated content recovery/model selection | [`test_replay_content_model_selection.py:36`](../../../tests/unit/test_replay_content_model_selection.py#L36) |
| Per-stop rate fitting | [`test_per_stop_rate_model.py:68`](../../../tests/unit/test_per_stop_rate_model.py#L68) |
| Per-bin rate fitting | [`test_per_bin_rate_model.py:62`](../../../tests/unit/test_per_bin_rate_model.py#L62) |
| Per-bin assembly helpers | [`test_per_bin_rate_helpers.py:61`](../../../tests/unit/test_per_bin_rate_helpers.py#L61) |
| Onset and dense time-bin tables | [`test_timebin_tables.py:76`](../../../tests/unit/test_timebin_tables.py#L76) |
| Covariates | [`test_covariates.py:91`](../../../tests/unit/test_covariates.py#L91) |
| Behavioral-value alignment | [`test_qvals_table.py:49`](../../../tests/unit/test_qvals_table.py#L49) |
| Scaling | [`test_standardization.py:17`](../../../tests/unit/test_standardization.py#L17) |
| Temporal-availability registry | [`test_covariate_timing.py:16`](../../../tests/unit/test_covariate_timing.py#L16) |
| Rate cohort audits | [`test_rate_cohort_audits.py:41`](../../../tests/unit/test_rate_cohort_audits.py#L41) |
| Per-bin runner orchestration | [`test_fit_per_bin_rate_models.py`](../../../tests/unit/test_fit_per_bin_rate_models.py) |
| Preparation cache behavior | [`test_prepare_data_cache.py:57`](../../../tests/unit/test_prepare_data_cache.py#L57) |
| Content audit emission | [`test_content_emits_audit.py:54`](../../../tests/integration/test_content_emits_audit.py#L54) |
| Standardization/reference builder parity | [`test_builder_parity.py:58`](../../../tests/integration/test_builder_parity.py#L58) |
| Cache/rebuild parity | [`test_from_cache_parity.py:29`](../../../tests/integration/test_from_cache_parity.py#L29) |
| Aligned decoder-frame construction | [`test_data_aggregation.py:209`](../../../tests/analysis/hippocampus/test_data_aggregation.py#L209) |
| Aggregation drift detection | [`test_aggregation_drift_check.py:28`](../../../tests/unit/test_aggregation_drift_check.py#L28) |
| Well-visit and trial construction | [`test_well_visits_and_trials.py:91`](../../../tests/unit/test_well_visits_and_trials.py#L91) |
| Task-variable construction | [`test_task_variables.py:20`](../../../tests/unit/test_task_variables.py#L20) |
| Pre-trial event leakage | [`test_event_table_pre_trial_no_leak.py:110`](../../../tests/unit/test_event_table_pre_trial_no_leak.py#L110) |

Passing tests establish only the contracts that are asserted. They do not by themselves establish
that an estimand is scientifically appropriate, a covariate is causally or temporally valid, or
the production artifacts were generated from the current code.

## 8. Source-of-truth and scope cautions

1. [`model_specs.py`](../../../src/continuum_swr_replay/analysis/replay/model_specs.py) is the
   authoritative executable catalog for the core content, per-stop rate, per-bin rate, and
   SWR-contrast ladders. Runner-local specifications still exist in `fit_uncertainty_rate.py`,
   `fit_cross_history_rate.py`, and `fit_swr_power_model.py`. Documentation tables can lag both.
2. [`2026-07-13-content-rate-uncertainty-model-ladders.md`](../../../docs/design/2026-07-13-content-rate-uncertainty-model-ladders.md)
   is explicitly a proposed design. It should not be treated as evidence that a model is present
   in production.
3. Notebooks and old output artifacts are consumers or snapshots, not authoritative definitions
   of the current pipeline.
4. A legacy `long_count` compatibility path may appear in content-related code and artifacts. The
   current hard-count content response is `arm_count_hard`; any fallback must be identified when
   reviewing a particular run.
5. The base content, SWR-content contrast, per-stop rate, per-bin rate, and SWR-rate contrast
   runners make their own cohort and missing-data decisions. Shared upstream tables do not imply
   identical retained observations across models.

## 9. Recommended review order

For the subsequent appropriateness review, inspect the code in dependency order:

1. Verify timestamps, trial identities, and sampling conventions in the session loader and decoder.
2. Verify event definitions and the row/inclusion invariants of every canonical table.
3. Trace every modeled covariate to its source observation, lag, join key, and availability time.
4. Reproduce the two-pass behavioral-value build and check cache/provenance consistency.
5. Verify each runner's exact cohort, complete-case filtering, standardization reference, and
   response construction.
6. Verify estimator likelihoods, conditioning, offsets/exposure, dependence assumptions,
   cross-validation grouping, and comparison eligibility.
7. Match persisted results to code, data-build, formula, cohort, and fold hashes before using them
   for scientific conclusions.
