# Analysis map: the paper's four parts, the analyses under each, and where they live

The paper has four parts:

1. **The model:** the non-local detector, how it works, and what it captures.
2. **The events:** what the non-local detector's events are like, and how they compare with the events people
   already detect (SWRs, and the Widloski & Foster sequence detector): where they agree and where they differ.
3. **Behavior:** how the events relate to the task, and how SWR and non-SWR events differ there.
4. **Other brain areas:** how other areas respond to the events, and how SWR and non-SWR events differ there.

Parts 3 and 4 compare SWR with non-SWR events only, with the Kay detector's SWRs. The comparison with every
other detector stays in Part 2.

For each analysis, this page lists:

- what is estimated;
- the script;
- the run name in `scripts/qa/run_refits.py`, for the CPU fits it reruns;
- the figures;
- the rows of [../paper/claims_table.md](../paper/claims_table.md) it supports. The claims table is grouped into
  the same four parts.

Reported numbers are read from one summary file per part, `paper/reported/part{n}.json`
(`scripts/reported/`; [design](design/2026-10-26-reported-values-design.md)). Other current numbers live in the results docs ([replay_rate_content_swr_synthesis.md](replay_rate_content_swr_synthesis.md),
[per_bin_rate_results.md](per_bin_rate_results.md)) and in the claims table. Model formulas live in
[models_table.md](models_table.md).

A claim marked "no script" has no tracked analysis yet.

Run names read `<group>.<analysis>[.<variant>]`; `--dry-run` lists them under their group. Every group
(`rate`, `coupling`, `content`, `behavior`) is in Part 3. The `at_well` cohort (well-anchored stops) is the
shipped one. `all_stops` and `in_transit` are sensitivity cohorts. The uncited sensitivity runs
(`rate.per_bin.in_transit` and `behavior.stay_leave.all_stops*`) are opt-in: a run skips them unless they are
named in full or `--include-optional` is given.

## The flow

All scripts are in `scripts/`. Steps 1–7 are the full rebuild. After a change to the table code, start at step 4;
after a change to fitting code only, start at step 6.

1. **Cohort.** The analysed epochs are the register's (`data_loaders/epoch_register.csv`;
   [epoch_register.md](epoch_register.md)).
2. **Artifacts and SWRs.**
   - Run `detection/run_artifact_detection.py` (the artifact mask), then `detection/run_swr_detection.py` (the
     four SWR inventories).
   - Then set the two run ids the decoders read: `ARTIFACT_INTERVALS_RUN_ID` and `SWR_INVENTORY_RUN_ID` in
     `src/continuum_swr_replay/analysis/events/swr_source.py`. A decode refuses to start without them.
3. **Decodes.** `decoding/run_decoding_bandit.py` (non-local detector) and `decoding/run_decoding_contfrag.py`
   (ContFrag), on GPUs.
4. **Tables**, in this order:
   1. `behavior/fetch_spatial_bandit_behavior.sh`;
   2. `tables/prepare_replay_content_data.py --stop-only` (the stop table);
   3. `behavior/build_qvals_table.py --write` (needs the stop table);
   4. `tables/prepare_replay_content_data.py` (the event, event × arm, event × patch, sequence and stop
      tables; needs `qvals_table`);
   5. `tables/build_timebin_tables.py`;
   6. `tables/prepare_contfrag_replay_content_data.py --event-source SWR` and `--event-source MUA`.
5. **Parts 1 and 2** on the decodes: the checks, the detector comparison and the decode-only figures (Figures 1–3,
   the supplements).
6. **Part 3 fits and the figures that read them:** `qa/run_refits.py`.
7. **Part 4:** `pfc/run_pfc_analysis_bandit.py` on GPUs, then `figures/generate_figure04.py`.

`figures/generate_all_figures.py` draws Figures 1–4 in one go. Run it only after step 7.

## Part 1. The model: how it works and what it captures

| What it tests | Script | Output, figure | Claims |
|---|---|---|---|
| The decoder on simulated data | `figures/generate_figure01.py` | Figure 1 | — |
| The 0.8 threshold, state fractions, calibration, and theta and MUA triggered averages (all in sample) | `figures/generate_supp_decoder_validation.py` | supplement a–c, e | 10, 5 |
| Decoding error in the Local state | `figures/generate_figure02.py` | Figure 2e | 10 |
| Decoding quality per session | `qa/run_quality_metrics.py`, after `qa/compute_hpc_raw_stats.py` | `paper/figures/quality_metrics/` | 10 |
| Held-out tetrodes support each decoder state (Local, No-Spike, Continuous, Fragmented): half the tetrodes choose the location, the other half score it. Ran on all <!--v:part1.state_evidence_epochs-->242<!--/v--> epochs, <!--v:part1.state_evidence_epochs_ok-->242<!--/v--> ok (chimi20200221_08_r4, which failed its stored-likelihood check, was rerun with the occupancy-floor fix on 2026-10-17 and passes). At rest, Local (<!--v:part1.heldout_local_immobile_animals_positive-->5<!--/v-->/<!--v:part1.state_evidence_animals-->5<!--/v--> animals), No-Spike (<!--v:part1.heldout_no_spike_immobile_animals_positive-->5<!--/v-->/<!--v:part1.state_evidence_animals-->5<!--/v-->) and Non-Local Continuous (<!--v:part1.heldout_continuous_immobile_animals_positive-->5<!--/v-->/<!--v:part1.state_evidence_animals-->5<!--/v-->) are supported; Non-Local Continuous while running is not (<!--v:part1.heldout_continuous_running_animals_positive-->0<!--/v-->/<!--v:part1.state_evidence_animals-->5<!--/v-->) | `detection/run_state_evidence.py` | `Processed-Data/_refresh_20261016/state_evidence/` | 7, 9 |
| What our detector and the Widloski detector report on simulated replays of known shape, and where there is no replay; the port checked against an independent re-implementation of the reference code | `detection/sweep_memoryless_detection.py`, `detection/sweep_memoryless_controls.py`, `detection/sweep_memoryless_heldout.py`, `qa/compare_memoryless_to_reference.py` (with `qa/_memoryless_reference.py`), `figures/generate_memoryless_comparison.py` | `docs/design/replay_detection_comparison/*.csv`; supplement | — (validates the port; the simulated side of 12) |

**No script:**
- 3 (the threshold sweep);
- 8 (a re-decode with a wider Local kernel);
- 11 (held-out model selection);
- 12 (the false-positive rate on real data).

## Part 2. The events: what they are, and how they compare with SWRs and Widloski events

### 2a. How common they are, and how they sit against SWRs

| What it tests | Script | Output, figure | Claims |
|---|---|---|---|
| Example events per overlap class; how many non-local events have no SWR; NL against SWR start and end offsets | `figures/generate_figure02.py` | Figure 2 (a, b, d) | 1, 2, 22 |
| Non-local probability and SWR z-score around non-local event starts, by SWR overlap | `figures/generate_figure02.py` | Figure 2c | — |
| SWR power at Local against Non-Local times | `figures/generate_figure03.py` | Figure 3b | 24 |
| The sub-cases of SWR and non-local overlap (for Methods numbers) | `content/quantify_swr_overlap_geometry.py` | printed tables | 4 (its non-local side only) |
| The other side of the overlap: how many of each detector's events (five SWR definitions, MUA, Widloski) overlap no non-local event, and how those divide by state; and, in the ones that do overlap, each state's share of their time bins (Local, No-Spike, non-local above 0.8, or uncertain). Kay for the main figure, the rest supplement | `detection/add_state_shares.py` (the shares), then `reported/summarize_part.py --part 2` (`continuum_swr_replay.reported.part2`) | `<event features run>/*_state_shares.parquet`; `paper/reported/part2.json` | 4, 23 |

### 2b. Spike support: are the events backed by spikes?

| What it tests | Script | Output, figure | Claims |
|---|---|---|---|
| Each event's spikes favor its decoded location over the animal's, against matched quiet windows. Not held out: the same spikes choose the location and score it. Ran on all 242 epochs, at most 30 events per group per epoch | `detection/analyze_event_spike_evidence.py --output-dir paper/figures/reality_r11` | `paper/figures/reality_r11/` (the default `paper/figures/reality/` holds the June prototype, 5 epochs) | 6, 7, 7b, 7c, 9 |
| The held-out test above (Part 1), split by SWR: both kinds are supported, the non-SWR events less per spike | `.claude/research/refresh_followup/f1_scripts/state_evidence_by_swr.py` (untracked; report `.claude/research/refresh_followup/f1_evidence_report.md`) | `.claude/research/refresh_followup/f1_outputs/` | 7 |

### 2c. Their properties

| What it measures | Script | Figure | Claims |
|---|---|---|---|
| Replay speed by SWR status | `figures/generate_figure03.py` | Figure 3c | 15 |
| Where content sits relative to the animal and to reward wells | `content/analyze_content_occupancy.py` | `paper/figures/content_occupancy/` | 13, 14 |
| Replayed position along the track, by SWR status | `figures/generate_figure03.py` | Figure 3d | — |
| Event duration of non-local, SWR and MUA events | `figures/generate_supp_decoder_validation.py` | supplement d | — (18, non-SWR against SWR, has no panel) |

Claims 15–18 and 16b (speed, extent, stillness, certainty, duration) are plain medians by SWR status, with the
animals in the claimed direction, from the event table's per-event columns (built in
`src/continuum_swr_replay/analysis/replay/trajectory.py`): `reported/summarize_part.py --part 2`. They are not
duration-controlled; no tracked analysis does that.

### 2d. Against the other detectors

| What it tests | Script | Output, figure | Claims |
|---|---|---|---|
| Four SWR detectors from shared inputs | `detection/run_swr_detection.py` | SWR inventories | — (the inputs to 23, 24) |
| Carey's joint ripple-power and multiunit detector, a fifth SWR definition, on the same CA1 inputs (ran on all 242 epochs, run `fa6537e48bcf`) | `detection/run_carey_detection.py` | `Processed-Data/replay_detection_comparison/carey/` | — (an input to the detector comparison) |
| A second sequence detector (Widloski & Foster) on the same decodes | `detection/run_memoryless_detection.py` | `Processed-Data/replay_detection_comparison/memoryless/` | — (a comparator for 21, 22, 25) |
| Nine inventories of each epoch (the five SWR definitions, MUA, the state-space detector, and the Widloski detector under two cohorts) on one substrate: overlap pairs, per-detector denominators, cells, the SWR label matrix. Ran on all 242 analysed epochs (run `afeb897d850f`) | `detection/run_inventory_comparison.py` | `Processed-Data/replay_detection_comparison/inventory/` | 23 |
| That comparison's counts per session and per animal | `detection/report_inventory_comparison.py` | `<inventory run>/report/`; `paper/figures/inventory_comparison/` | 23 |
| Sampled examples from each cell of that comparison, on one time axis | `figures/generate_inventory_examples.py` | `paper/figures/inventory_comparison/<run id>/examples/` | — |
| Event features on every compared event of the nine inventories (no baseline windows: dropped by the lead, 2026-10-18): timing, engagement, ripple and MUA on one trace, spatial features, context, and the common spike-evidence score; the populations (shared, unmatched, unknown); the memoryless rejection stage of each ours-only event. Ran on all 242 epochs (run `30cf1cc432e9`, all `ok`) | `detection/run_event_features.py` | `Processed-Data/replay_detection_comparison/event_features/` | 23 |
| That run's features per session and per animal (counts and quantiles, no verdicts) | `detection/report_event_features.py` | `<event features run>/report/`; `paper/figures/event_features/` | 23 |
| Sampled comparator-only and ours-only events, with their spike evidence and spikes per row | `figures/generate_event_feature_examples.py` | `paper/figures/event_features/<run id>/examples/` | — |
| Single decoded events, for checking categories by eye | `figures/generate_event_gallery.py`, `figures/run_event_gallery.sh` | `event_gallery/` | — |

Every Widloski column and every `ms_*` feature is pending the Phase 2 pilot checks
([.claude/docs/plans/replay_detection_comparison/PLAN.md](../.claude/docs/plans/replay_detection_comparison/PLAN.md)).

## Part 3. Behavior: how the events relate to the task, SWR against non-SWR

### 3a. How often events occur, and what drives the rate

There are two model families; see [rate_models.md](rate_models.md):

- **Per time bin (Family B), the main one:** a spline time course, a reward kernel and self-history.
- **Per stop (Family A):** comparable with the literature.

| What it estimates | Run | Script | Figures | Claims |
|---|---|---|---|---|
| Rate per time bin: time course, reward kernel, history | `rate.per_bin.{at_well,all_stops,in_transit}` | `rate/fit_per_bin_rate_models.py` | `rate.fig.per_bin_ladder`, `rate.fig.per_bin_summary`, `rate.fig.model_checking`; `rate.doc.per_bin_results` writes [per_bin_rate_results.md](per_bin_rate_results.md) | 19c |
| Rate per stop | `rate.per_stop.{at_well,all_stops}` | `rate/fit_per_stop_rate_models.py` | — | 19c |
| How SWR and non-SWR rates differ, term by term, paired within session | `rate.swr_contrast.{at_well,all_stops}` | `rate/fit_swr_contrast_rate.py` | `content.fig.swr_contrast_coefs` | 19c, 19f |
| Rate against arm uncertainty and value | `rate.uncertainty.at_well` | `rate/fit_uncertainty_rate.py` | `rate.fig.uncertainty` | — |
| Whether one stream's history drives the other's rate | `rate.cross_history` | `rate/fit_cross_history_rate.py` | — (the fit is `models_per_bin_rate/cross_history_models.pkl`; no figure or write-up) | — |
| SWR-coupling strength (ripple power as a continuous mark) against the behavioral covariates | `coupling.ripple_power` | `coupling/fit_swr_power_model.py` | — | — |

### 3b. What the events represent (arm or patch), and what shapes it

Every content model is conditional on the event: it asks which arm (or patch) an event's posterior mass falls on.

| What it estimates | Run | Script | Figures | Claims |
|---|---|---|---|---|
| Which task variables shape content (the M0–M6 ladder, all nine arms) | `content.arm_ladder.all_arms` | `content/fit_replay_content_models.py` | `content.fig.ladder`, `content.fig.pipeline_data_flow` | — |
| The same on the six leaf arms, with value and uncertainty | `content.arm_ladder.leaf_arms` | same, `--leaf-only` | — | 19g |
| The arm ladder (M0–M4) on SWR events alone and on non-SWR events alone: which terms help in each population | `content.arm_ladder.{swr_only,non_swr_only}` | same, `--swr-subset {T,F}` | — | — |
| The arm ladder (M0–M6) under the Dirichlet-multinomial estimator: does each step still help once breadth is modelled | `content.arm_ladder.dirichlet_multinomial` | same, `--estimator dirichlet_multinomial` | — | — |
| Which task variables shape patch content (P0–P3), on all events and on each population alone | `content.patch_ladder.{all_events,swr_only,non_swr_only}` | same, `--hierarchy patch` | — | — |
| How SWR content differs, term by term, paired within session (arm and patch) | `content.swr_contrast.{all,rewarded,omitted}` | `content/fit_swr_contrast_content.py` | `content.fig.swr_contrast_coefs` | 19a, 19b, 19f, 19g |
| How widely an event spreads over arms (Dirichlet-multinomial concentration), SWR against non-SWR | `content.breadth_contrast.{all,all_length_adjusted,rewarded,omitted}` | same, `--estimator dirichlet_multinomial` | — | 19a, 19f |
| Whether the conclusions depend on the estimator (fraction, count, Dirichlet-multinomial) | `content.estimator_comparison` | `content/compare_content_estimators.py` | `paper/figures/content_estimators/` | 19f |
| Content against reward-belief uncertainty | `content.uncertainty` | `content/analyze_uncertainty_content.py` | `content.fig.uncertainty` | 19g |
| Content of events found by SWR- or MUA-anchored detection (ContFrag) | `content.event_source.{swr,mua}_{arm,patch}` | `content/fit_replay_content_models.py --event-source` | — | — |

### 3c. Whether content relates to what the animal does next

| What it estimates | Run | Script | Claims |
|---|---|---|---|
| Content predicts the next choice (conditional logit over a behavioural baseline). Each event source's suite also fits patch, per-stop rate, patch-switch and switch-aligned blocks on one common cohort; only the choice block is cited | `behavior.event_source_suite.{non_local,swr,mua}` | `content/compare_event_source_models.py` | 19d |
| Content during a stop matches the arm visited next, against a shuffle (was "Stage 5B") | `behavior.retrospective.{all,swr,non_swr}` | `behavior/run_retrospective_alignment.py` | — |
| Content × reward outcome → stay or leave (was "Stage 6"); FDR summary across roles | `behavior.stay_leave.*` | `behavior/run_stay_leave.py`; `qa/summarize_stage6_fdr.py` | 19e |

A decision-grain rebuild of the choice models is planned:
[.claude/docs/plans/regression-integrity-fixes/phase-11-decision-models.md](../.claude/docs/plans/regression-integrity-fixes/phase-11-decision-models.md).

## Part 4. Other brain areas: how they respond to the events, SWR against non-SWR

| What it estimates | Script | Output, figure | Claims |
|---|---|---|---|
| Per-unit modulation by SWR and non-SWR non-local events (penalised GLM, held-out deviance) | `pfc/run_pfc_analysis_bandit.py` (GPU; about 14–17 h per card) | `Processed-Data/{epoch}_mPFC_modulation.pkl` | 20 |
| Figure 4, drawn from those bundles | `figures/generate_figure04.py` | Figure 4 | 20 |
| Claim 20's numbers, read from those bundles by Figure 4's loader and rules | `reported/summarize_part.py --part 4` (`continuum_swr_replay.reported.part4`) | `paper/reported/part4.json` | 20 |

Only mPFC is fitted. The runner takes `--brain_area OFC`, but no OFC fits exist.

## Supporting scripts (no part of their own)

| Script | What it does |
|---|---|
| `qa/check_data_availability.py`, `qa/diagnose_trial_detection.py` | Input checks per session |
| `qa/check_fit_provenance.py` | Is each fit current for the tables it read (`run_refits.py` runs it) |
| `qa/compare_fits.py`, `pfc/compare_pfc_cohorts.py` | Old against new fits, when code or tables change |
| `figures/generate_supp_pipeline_schematic.py`, `figures/generate_supp_pipeline_data_flow.py` | The pipeline supplement figures (the second reads `models/M3.pkl`, so the driver redraws it) |
| `figures/generate_all_figures.py` | Figures 1–4 in one go |

## Removed as redundant (2026-10-17)

| Was | Why |
|---|---|
| The rate SWR-contrast `--aggregate-only` reruns | The full fit already writes the same aggregation, so the rerun rewrote identical files. The flag stays for re-auditing saved fits |

Three content ladders removed the same day (the SWR-only and non-SWR-only ladders, the patch ladder and the
Dirichlet-multinomial arm ladder) were restored: they overlap other analyses but are not duplicates.

## Old run ids

The 2026-10 refresh (its R6 and R7 stages) named runs by code:

| Old | Run | Old | Run |
|---|---|---|---|
| R01–R03 | `rate.per_bin.{at_well,all_stops,in_transit}` | K01–K03 | `content.swr_contrast.{all,rewarded,omitted}` |
| R04 | `rate.per_stop.at_well` (and new: `rate.per_stop.all_stops`) | K04–K07 | `content.breadth_contrast.{all,all_length_adjusted,rewarded,omitted}` |
| R05, R06 | `rate.swr_contrast.{at_well,all_stops}` | X01 | `content.estimator_comparison` |
| R07, R08 | removed | X02 | `content.uncertainty` |
| R09 | `rate.uncertainty.at_well` | X03–X05 | `behavior.event_source_suite.{non_local,swr,mua}` |
| R10 | `coupling.ripple_power` | F01–F04 | `content.event_source.{swr_arm,swr_patch,mua_arm,mua_patch}` |
| R11 | `rate.cross_history` | P01–P03 | `behavior.retrospective.{all,swr,non_swr}` |
| C01 | `content.arm_ladder.all_arms` | S01–S08 | `behavior.stay_leave.*` |
| C02 | `content.arm_ladder.leaf_arms` | C03 | `content.patch_ladder.all_events` |
| C04, C05 | `content.arm_ladder.{swr_only,non_swr_only}` | C06, C07 | `content.patch_ladder.{swr_only,non_swr_only}` |
| C08 | `content.arm_ladder.dirichlet_multinomial` | | |

## Old question labels

Until 2026-10-26 this page was grouped by question:

| Old | Now |
|---|---|
| Q1 (are the events real) | Part 1 (the model, the held-out state evidence) and Part 2 (overlap with SWRs, spike support, the detector comparison) |
| Q2 (rate) | Part 3a |
| Q3a (what the events are like) | Part 2c; the ripple-power fit is in Part 3a |
| Q3b (content) | Part 3b |
| Q4 (next choice) | Part 3c |
| Q5 (PFC) | Part 4 |
