# Analysis map: each question, the analyses that answer it, and where they live

Every analysis in this repository answers one of five questions. This page lists, for each question, what is
estimated, the script, the run name in `scripts/qa/run_refits.py` (for the CPU fits it reruns), the figures, and
the claims rows in [../paper/claims_table.md](../paper/claims_table.md) it supports. Current numbers live in the
results docs: [replay_rate_content_swr_synthesis.md](replay_rate_content_swr_synthesis.md),
[per_bin_rate_results.md](per_bin_rate_results.md) and the claims table. Model formulas live in
[models_table.md](models_table.md).

The claims table groups claims by goal. Goals 1, 2 and 4 are Q1; Goal 3a is Q3a; Goal 3b is Q2, Q3b and Q4;
Goal 3c is Q5.

Run names read `<question>.<analysis>[.<variant>]`. `--dry-run` lists them by question. The `at_well` cohort
(well-anchored stops) is the shipped one; `all_stops` and `in_transit` are sensitivity cohorts.

## The flow

1. **Decode** every epoch: `scripts/decoding/run_decoding_bandit.py` (non-local detector) and
   `run_decoding_contfrag.py` (ContFrag). Before that, `run_artifact_detection.py` masks artifacts, and
   `run_swr_detection.py` builds the SWR inventories the decoders read.
2. **Tables**: `prepare_replay_content_data.py`, `build_qvals_table.py`, `build_timebin_tables.py` and the
   ContFrag table builder. They give one row per event, event × arm, stop and time bin.
3. **Q1 checks** on the decodes (below).
4. **Fits** for Q2–Q4 and the figures that read them: `scripts/qa/run_refits.py`.
5. **PFC unit GLM** (GPU) for Q5: `scripts/pfc/run_pfc_analysis_bandit.py`, then Figure 4.

## Q1. Are the decoded non-local events real, including those outside SWRs?

| What it tests | Script | Output, figure | Claims |
|---|---|---|---|
| How common non-SWR events are; SWR detection and non-local detection as separate axes | `scripts/figures/generate_figure02.py`, `generate_figure03.py` | Figures 2–3 | 1, 2, 4, 5 |
| The decoder on simulated data | `scripts/figures/generate_figure01.py` | Figure 1 | — |
| Decoder validation (held-out likelihood, label stability) | `scripts/figures/generate_supp_decoder_validation.py`, `scripts/qa/run_quality_metrics.py` | supplement; `paper/figures/quality_metrics/` | 10, 11 |
| Each event's spikes support its decoded location better than a matched baseline | `scripts/qa/analyze_event_spike_evidence.py` | `paper/figures/reality/` | 6, 7, 7b, 7c, 9 |
| Held-out tetrodes support each decoder state (Local, No-Spike, Continuous, Fragmented) | `scripts/qa/run_state_evidence.py` | `Processed-Data/_refresh_<date>/state_evidence/` | 7, 9 |
| How SWR and non-local events overlap | `scripts/content/quantify_swr_overlap_geometry.py` | printed tables | 4, 22 |
| Four SWR detectors from shared inputs | `scripts/decoding/run_swr_detection.py` | SWR inventories | 3, 23, 24 |
| A second sequence detector (Widloski & Foster) on the same decodes | `scripts/decoding/run_memoryless_detection.py`, `scripts/figures/generate_memoryless_comparison.py` | `Processed-Data/replay_detection_comparison/memoryless/`; supplement | 21, 22, 25 |

Claims 8 (re-decode with a wider Local kernel) and 12 (false-positive rate) have no script yet. Goal 4's
full detector comparison is planned and not built:
[.claude/docs/plans/replay_detection_comparison/PLAN.md](../.claude/docs/plans/replay_detection_comparison/PLAN.md),
Phase 3.

## Q2. How often do events occur, and what drives the rate, for SWR against non-SWR events?

Two model families: per time bin (Family B, the main one; it has a spline time course, a reward kernel and self-history) and
per stop (Family A, comparable with the literature). See [rate_models.md](rate_models.md).

| What it estimates | Run | Script | Figures | Claims |
|---|---|---|---|---|
| Rate per time bin: time course, reward kernel, history | `rate.per_bin.{at_well,all_stops,in_transit}` | `scripts/rate/fit_per_bin_rate_models.py` | `rate.fig.per_bin_ladder`, `rate.fig.per_bin_summary`, `rate.fig.model_checking`; `rate.doc.per_bin_results` writes [per_bin_rate_results.md](per_bin_rate_results.md) | 19c |
| Rate per stop | `rate.per_stop.at_well` | `scripts/rate/fit_per_stop_rate_models.py` | — | 19c |
| How SWR and non-SWR rates differ, term by term, paired within session | `rate.swr_contrast.{at_well,all_stops}` | `scripts/rate/fit_swr_contrast_rate.py` | `content.fig.swr_contrast_coefs` | 19c, 19f |
| Rate against arm uncertainty and value | `rate.uncertainty.at_well` | `scripts/rate/fit_uncertainty_rate.py` | `rate.fig.uncertainty` | 19g |
| Ripple power as a continuous mark of SWR coupling | `rate.ripple_power` | `scripts/rate/fit_swr_power_model.py` | — | — |
| Whether one stream's history drives the other's rate | `rate.cross_history` | `scripts/rate/fit_cross_history_rate.py` | — (no result written yet) | — |

## Q3a. What are the events like: location, speed, extent, certainty, duration?

| What it measures | Script | Figure | Claims |
|---|---|---|---|
| Distance from the animal (close against remote) | `scripts/figures/generate_figure03.py` | Figure 3 | 13 |
| Where content sits relative to the animal and to reward wells | `scripts/content/analyze_content_occupancy.py` | `paper/figures/content_occupancy/` | 14 |
| Speed, extent, stationarity, certainty and duration, SWR against non-SWR, duration-controlled | `scripts/figures/generate_figure03.py` (trajectory metrics from the event table) | Figure 3, supplement | 15–18 |

## Q3b. What do the events represent (arm or patch), what shapes that, and does it differ for SWR events?

All content models are conditional on the event: they ask which arm (or patch) an event's posterior mass falls
on.

| What it estimates | Run | Script | Figures | Claims |
|---|---|---|---|---|
| Which task variables shape content (M0–M6 ladder, all nine arms) | `content.arm_ladder.all_arms` | `scripts/content/fit_replay_content_models.py` | `content.fig.ladder` | 19a, 19b |
| The same on the six leaf arms, with value and uncertainty | `content.arm_ladder.leaf_arms` | same, `--leaf-only` | — | 19g |
| How SWR content differs, term by term, paired within session (arm and patch) | `content.swr_contrast.{all,rewarded,omitted}` | `scripts/content/fit_swr_contrast_content.py` | `content.fig.swr_contrast_coefs`, `content.fig.ladder` | 19a, 19b, 19f |
| How widely an event spreads over arms (Dirichlet-multinomial concentration), SWR against non-SWR | `content.breadth_contrast.{all,all_length_adjusted,rewarded,omitted}` | same, `--estimator dirichlet_multinomial` | `content.fig.swr_contrast_coefs` | 19a, 19f |
| Whether conclusions depend on the estimator (fraction, count, Dirichlet-multinomial) | `content.estimator_comparison` | `scripts/content/compare_content_estimators.py` | `paper/figures/content_estimators/` | 19f |
| Content against reward-belief uncertainty | `content.uncertainty` | `scripts/content/analyze_uncertainty_content.py` | `content.fig.uncertainty` | 19g |
| Content of events found by SWR- or MUA-anchored detection (ContFrag) | `content.event_source.{swr,mua}_{arm,patch}` | `scripts/content/fit_replay_content_models.py --event-source` | — | — |

## Q4. Does content relate to what the animal does next?

| What it estimates | Run | Script | Claims |
|---|---|---|---|
| Content predicts the next choice (conditional logit over a behavioural baseline); the patch, per-stop rate and patch-switch ladders and switch-aligned content, per event source, on one common cohort | `behavior.event_source_suite.{non_local,swr,mua}` | `scripts/content/compare_event_source_models.py` | 19d |
| Content during a stop matches the arm visited next, against a shuffle (formerly "Stage 5B") | `behavior.retrospective.{all,swr,non_swr}` | `scripts/behavior/run_retrospective_alignment.py` | 19e |
| Content × reward outcome → stay or leave (formerly "Stage 6") | `behavior.stay_leave.*` | `scripts/behavior/run_stay_leave.py` | 19e |

A decision-grain rebuild of the choice models is planned:
[.claude/docs/plans/regression-integrity-fixes/phase-11-decision-models.md](../.claude/docs/plans/regression-integrity-fixes/phase-11-decision-models.md).

## Q5. Does PFC respond to the events, SWR or not?

| What it estimates | Script | Output, figure | Claims |
|---|---|---|---|
| Per-unit modulation by SWR and non-SWR non-local events (penalised GLM, held-out deviance) | `scripts/pfc/run_pfc_analysis_bandit.py` (GPU; about 14–17 h per card) | `Processed-Data/{epoch}_mPFC_modulation.pkl`; Figure 4 (`pfc.fig.figure04`) | 20 |

## Removed as redundant (2026-10-17)

| Was | Why |
|---|---|
| The content ladder refit on SWR-only and on non-SWR-only events (`--swr-subset`; `models_swr_{T,F}/`, `models_patch_swr_{T,F}/`) | The content SWR contrast answers the same question with paired within-session terms, and nothing read these bundles |
| The base patch ladder on non-local events (`models_patch/`) | The event-source suite fits the same patch models on the same events, and the content SWR contrast fits the patch contrast |
| The Dirichlet-multinomial content ladder (`models_dm/`) | Nothing read it; the breadth contrast and the estimator comparison carry the Dirichlet-multinomial results |
| The rate SWR-contrast `--aggregate-only` reruns | The full fit already writes the same aggregation; the flag stays for re-auditing persisted fits |

## Old run ids

The 2026-10 refresh (its R6 and R7 stages) named runs by code. They map as follows:

| Old | Run | Old | Run |
|---|---|---|---|
| R01–R03 | `rate.per_bin.{at_well,all_stops,in_transit}` | K01–K03 | `content.swr_contrast.{all,rewarded,omitted}` |
| R04 | `rate.per_stop.at_well` | K04–K07 | `content.breadth_contrast.{all,all_length_adjusted,rewarded,omitted}` |
| R05, R06 | `rate.swr_contrast.{at_well,all_stops}` | X01 | `content.estimator_comparison` |
| R07, R08 | removed | X02 | `content.uncertainty` |
| R09 | `rate.uncertainty.at_well` | X03–X05 | `behavior.event_source_suite.{non_local,swr,mua}` |
| R10 | `rate.ripple_power` | F01–F04 | `content.event_source.{swr_arm,swr_patch,mua_arm,mua_patch}` |
| R11 | `rate.cross_history` | P01–P03 | `behavior.retrospective.{all,swr,non_swr}` |
| C01 | `content.arm_ladder.all_arms` | S01–S08 | `behavior.stay_leave.*` |
| C02 | `content.arm_ladder.leaf_arms` | C03–C08 | removed |
