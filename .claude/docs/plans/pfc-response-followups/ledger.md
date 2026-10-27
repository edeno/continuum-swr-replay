# Ledger — mPFC response follow-ups

[← back to PLAN.md](PLAN.md)

What ran, where, and what came out. Results are cited as `part4.json` keys (or summary-table keys), never as typed numbers.

Worktree `/cumulus/edeno/continuum-swr-replay-pfc-followups`, branch `pfc-response-followups` (stacked on `reported-values-2026-10-26` at `d6b8e0fd`). Host breeze.

## Log

| date | phase | what | commit / run | where | result |
|---|---|---|---|---|---|
| 2026-10-26 | plan | plan written | `ece3eb97` | `.claude/docs/plans/pfc-response-followups/` | — |
| 2026-10-26 | 1 (A) | `analysis/pfc/timing.py` (peak, half-max onset, population timing), calibration script | `b809c9b9` | — | tests green |
| 2026-10-26 | 1 (A) | calibration run, CPU, 16 scenarios × 200 units, 2 min 15 s | run at `5fbb626e` (tree dirty only from the phase 0 port in progress) | `Processed-Data/pfc_response_followups/timing_calibration.json` | keys `timing_calibration_*` |
| 2026-10-26 | 1 (A) | `timing_values` in Part 4; summary + emit | `5fbb626e`, `985abafc` | `paper/reported/part4.json`, `overview.md` (**When?** bullet) | 245 values (was 97); no existing value changed (`changes.md` lists only `?` → new markers) |
| 2026-10-26 | 1 (A) | diagnostics and spot checks | — | `.claude/research/pfc_response_followups_outputs/timing_diagnostics.png`, `timing_spotcheck_nonswr_suppressed.png` | see Findings |

| 2026-10-26 | 1 (A) | independent review (opus) of the phase diff | — | — | no code bug; four reading problems fixed in the next commit: session bootstrap → day bootstrap and per-animal rise onsets; calibration now measures the quoted estimators (unit half-max, population half-max and rise onset, population peak) against the truth's own half-way point; matched-control sentence replaced by the calibration floor; confounds named in the bullet (event clustering, non-local start after SWR start, sign selection) |
| 2026-10-26 | 1 (A) | calibration rerun, summary + emit | run at `985abafc` (dirty: phase 0 port in progress) | `timing_calibration.json`, `part4.json`, `overview.md` | rise onset before event onset: non-SWR replay in `timing_population_rise_onset_animals_negative_non_swr_replay_enhanced` animals; SWR-coupled and all SWRs only in `..._swr_replay_enhanced` / `..._swr_enhanced` animals, with day intervals reaching zero |

| 2026-10-26 | 0 | marked family ported (opus worker): raw copy, layout + canonical assembly, docs | `17874e46`, `62b2f2d7`, `96c60a83` | — | not ported: `66471fd3` (Figure 4 gate), `setup_logger` "a" mode; `measure_pfc_mark_context.py` now in `scripts/pfc/`; runners select from `analysed_epochs()`; skip contracts gain the SWR-source keys |
| 2026-10-26 | 0 | real-data equivalence (`-m real_data`, CPU): marked context vs `run_session` stopped before fitting, j1620210710 08_r4 | test `test_marked_context_matches_canonical_assembly` | — | fit mask, fit-bin counts, lags and both replay classes' onsets equal exactly (this session has no excluded time, so that path is not exercised) |
| 2026-10-26 | 0 | independent review (opus) | — | — | no bug; risk: a default-directory rerun of the family runner deletes the 2026-09-19 corpus (stale under the new contract) — written into phase 3b; two stale paths fixed (`fda7a938`+1) |
| 2026-10-26 | 0/1 | tests | — | — | `tests/analysis/pfc` + `tests/unit`: 5580 passed, 11 skipped, 1 failed — `test_session_keys.py::test_no_module_level_session_constant_defined_elsewhere` flags `scripts/detection/measure_event_gamma.py` from base commit `23e86d21`, not this branch |
| 2026-10-26 | 1 | calibration and Part 4 summary rerun on a clean tree | at `96c60a83`, committed `fda7a938` | — | no value changed |

| 2026-10-26 | lead | answers (~20:47): measure decreases properly before Phase B; 66471fd3 out; Phase B definition confirmed, no power matching, ripple power as a graded mark on every class; null-only fallback allowed above ~16 h; Phase C defaults accepted | — | overview.md Open Questions | — |
| 2026-10-26 | 1 (A) decreases | reference-window baseline, `response_size`, `steepest_approach_lag`; calibration with decreases (−0.7, −0.3), two windows (±0.2 s, ±1.0 s) and a stop-level rate confound | `c7236a50` | — | tests green |
| 2026-10-26 | 1 (A) decreases | calibration, CPU, 40 scenarios | run at `f173d89b` (clean) | `timing_calibration.json` | rise onset late by `timing_calibration_bias_ms_onset_rise_*` for both signs; stop-level confound: `timing_calibration_stop_offset_size` from zero vs `timing_calibration_stop_offset_rise_size` from −200 ms; wide window changes nothing (`timing_calibration_wide_*`) → no wide-window refit |
| 2026-10-26 | 1 (A) decreases | Part 4 summary + emit | `718cb7d2` | `part4.json` (396 values), `overview.md` **When, and how much?** | decreases: `timing_population_rise_onset_ms_*_suppressed`, `timing_population_rise_onset_animals_negative_*_suppressed`, `timing_population_rise_size_*`; recommended rule per sign stated in the overview |
| 2026-10-26 | 2 (B) | extended event set, models, null, curves, runner flags, comparison script | `44b20a72` | — | 604 passed (pfc/reported selection) |
| 2026-10-26 | 2 (B) | worktree venv lacked CUDA jax (first smoke launch fell back to CPU and the runner refused it); installed with `uv sync --frozen --inexact --extra cuda` (no lockfile change) | — | `logs/pfc_response_followups/swr_without_replay_smoke/failed_cpu_jax/` | — |
| 2026-10-26 | 2 (B) | smoke, 4 sessions (j1620210710 12_r6, senor20201101 06_r3, wilbur20210331 16_r8, peanut20201128 16_r8), GPUs 1–4, R = 1,000, started 21:11 | `44b20a72` (+ later commits that do not touch the fit path) | tmux `pfc_b_smoke`; logs `logs/pfc_response_followups/swr_without_replay_smoke/`; output `Processed-Data/pfc_response_followups/swr_without_replay_smoke/` | SWRs without replay per session (sidecar `n_swr_no_non_local`): 38, 2, 34, 9 |

| 2026-10-26 | 2 (B) | smoke finished; wall times 48 min (j16, 74 units), 66 min (senor, 277), 70 min (peanut, 296), 102 min (wilbur, 511) — 1.09–1.26× the canonical batch's minutes for the same sessions; peak GPU memory about 17 GB per worker over an 8 GB baseline; host RSS ≤ 9.4 GB | `44b20a72` | smoke tree | `compare_pfc_event_set_bundles.py`: every canonical column and all five canonical curves **identical** (max difference 0) in all 4 sessions; 26 new columns per bundle |
| 2026-10-26 | 2 (B) | smoke finding for the lead: in sessions with very few SWRs without replay the new class's empirical p is NaN for many units (the canonical rule needs all 1,000 null draws finite; a draw whose fit fails on a low-rate unit leaves the unit untested). Finite p: 72/74 (38 events), 296/296 (9), 83/277 (2), 504/511 (34) | — | — | reporting rule needed in phase 4 (e.g. a minimum number of events per session, as the marked family's 10) |
| 2026-10-26 | 2 (B) | extrapolation: canonical batch 80.9 GPU-h × ~1.2 ≈ 97 GPU-h; LPT partition over 9 cards ≈ 10.6–11.2 h per card (< 16 h) → full refit, all classes' nulls | — | `logs/pfc_response_followups/swr_without_replay/gpu<N>.sessions` | — |
| 2026-10-26 | 2 (B) | **full batch launched 22:55**, 75 sessions, GPUs 1–9, R = 1,000, ETA about 10:00 on 2026-10-27 | `ccb5b8d0` | tmux `pfc_b_swr_without_replay` (window `gpumem` samples memory each minute); logs `logs/pfc_response_followups/swr_without_replay/`; output `Processed-Data/pfc_response_followups/swr_without_replay/` | running |

| 2026-10-27 | lead | ~01:45: start Phase C; rare-class rule NOT decided — report the new class both ways (all sessions, ≥10-event sessions) | — | — | — |
| 2026-10-27 | 4 (D, part) | `swr_without_replay_values` in Part 4 (both subsets, pending note while the batch is incomplete), tested on the smoke tree | `0695b326` | — | — |
| 2026-10-27 | 3a (C code) | opus worker: baseline columns (contract 2), forward patch-switch marks, extended event set with ripple power for SWRs without replay (real-data check: event table's `ripple_power_z` reproduced exactly for 136 events of j1620210710 08_r4), follow-up specs + pooled ripple block, marginal tier, shuffled-mark reference (stop-level marks permuted as whole stops), runner flags, summary.parquet, docs, slow planted tests | `ed5356ad`..`b000330c` | — | 150 passed, 3 skipped; CPU smoke of one model stopped after 60 min unfinished (no runtime figure) |
| 2026-10-27 | 3a | `trips_to_patch_switch` takes the cap when ≥3 staying trips are known and no switch follows | `3bd90f08` | — | — |
| 2026-10-27 | 3a | cost warning: `--tier all --slope-permutations 30` ≈ 1,140 full-data refits per session; must be measured on the GPU smoke before a corpus launch | — | — | open |

| 2026-10-27 | 3a | review fixes by the implementer: animal weighting in the summary, residualized (Freedman–Lane-type) shuffled-mark reference, max-statistic call with plus-one p, per-class eligibility (MIN_LEVEL_COUNT 10), reference only after the convergence gate, batched warm-started refits, per-model seeds; ridge-leakage study (bias small, carried by the reference; false-outside rate 4.2 % with the plus-one rule) | `a8928340`..`d8b0c02e` | — | 210 passed, 3 skipped |
| 2026-10-27 | 2 (B) | **batch finished**: 75/75 sessions, every worker EXIT 0, 9.6–10.9 h per card, no Traceback; `compare_pfc_event_set_bundles.py` over all 75: every canonical column and curve identical to the canonical bundles | started `ccb5b8d0` | `Processed-Data/pfc_response_followups/swr_without_replay/` | — |
| 2026-10-27 | 4 (D) | Phase B values into Part 4 + **SWRs without replay?** overview bullet | (this commit) | `part4.json` (478 values) | `swr_no_replay_called_fraction_all` / `_min_events` against `swr_replay_called_fraction`; `swr_no_replay_overlap_odds_ratio_all`; `swr_no_replay_rank_correlation_all`; per animal `swr_no_replay_called_fraction_all_<animal>` |
| 2026-10-27 | 3b (C) | GPU smoke started on cards freed by Phase B: j1620210710 12_r6 (GPU 4, 08:30), senor20201101 06_r3 (GPU 2, 08:40), wilbur20210331 16_r8 (GPU 5, 08:40); `--model-set followup --tier all --event-set with_swr_without_replay --slope-permutations 30` | `d8b0c02e` | tmux `pfc_c_smoke`; logs `logs/pfc_response_followups/marked_family_smoke/`; output `Processed-Data/pfc_response_followups/marked_family_smoke/` | j16 `replay_content`: observed fit ≈ 9 min, reference 1,351 s for 180 refits (batch 8) |

| 2026-10-27 | 3b (C) | smoke timings (`runner_timing_seconds`): j16 12_r6 (74 units) whole `--tier all` 1 h 52 min (EXIT 0); `replay_content` observed fit / reference: j16 526 s / 1,351 s, senor 06_r3 (277 units) 1,147 s / 6,888 s, wilbur 16_r8 (511 units) 1,818 s / 10,994 s → about 21 s per unit per 6-block model of reference | `d8b0c02e` | smoke tree | extrapolation for `--tier all`, R = 30: ≈ 180 s per unit × 18,552 units ≈ 900+ GPU-h (≈ 4 days on 9 cards); without the reference ≈ 200 GPU-h; **stopped for the lead** |
| 2026-10-27 | 3b (C) | smoke finding: `stop_context` and `marginal_rewarded_context` ineligible on j16 (unrewarded-stop replay too rare: 9/3 events), `marginal_next_trip_switches_patch` no variation; corpus check from the event table (well-anchored events, 75 PFC sessions): ≥10 events at each reward level in 23 (non-SWR) / 7 (SWR-coupled) sessions; ≥10 at each next-trip-switch level in 15 / 8 | — | — | Q8 is mostly not estimable within a session; lead decision |
| 2026-10-27 | 3b (C) | summary pipeline run on the j16 smoke bundles (scratch copy): report, figures and `summary.parquet` (1,020 rows) written | — | scratchpad `csmoke/` | works end to end |

## Runs

| run | tmux session | cards | log dir | output dir | started | finished | runtime / memory |
|---|---|---|---|---|---|---|---|
| timing calibration | — (foreground, CPU) | none | — | `Processed-Data/pfc_response_followups/` | 2026-10-26 | 2026-10-26 | 2 min 15 s CPU |
| timing calibration with decreases | — (background, CPU) | none | — | `Processed-Data/pfc_response_followups/timing_calibration.json` | 2026-10-26 | 2026-10-26 | about 40 min CPU (shared with tests) |
| Phase B smoke | `pfc_b_smoke` (closed) | 1–4 | `logs/pfc_response_followups/swr_without_replay_smoke/` | `Processed-Data/pfc_response_followups/swr_without_replay_smoke/` | 2026-10-26 21:11 | 2026-10-26 22:53 | 48–102 min per session |
| Phase B batch | `pfc_b_swr_without_replay` | 1–9 | `logs/pfc_response_followups/swr_without_replay/` | `Processed-Data/pfc_response_followups/swr_without_replay/` | 2026-10-26 22:55 | 2026-10-27 09:48 | 9.6–10.9 h per card |
| Phase C smoke | `pfc_c_smoke` | 4, 2, 5 | `logs/pfc_response_followups/marked_family_smoke/` | `Processed-Data/pfc_response_followups/marked_family_smoke/` | 2026-10-27 08:30 | | |

## Findings (phase 1)

- Calibration: for a response starting exactly at onset, the half-max onset read off the fitted curve is *later* than the truth (`timing_calibration_onset_bias_ms_*`), so smoothing does not make onsets early. Per unit, a sizeable share of such units still shows an onset before zero by noise (`timing_calibration_onset_before_zero_fraction_*`): the per-unit "before onset" shares must be read against it.
- Peak lags cluster at a few lags spaced roughly by the 50 ms knot spacing (diagnostic plot); per-unit peaks also land on the window edge (`timing_peak_at_edge_fraction_*`).
- Suppressed units: their sign-aligned mean curves sit far from zero across the window (`timing_population_pre_event_level_fraction_*_suppressed`), so the half-max-from-zero onset runs to the edge (`timing_population_onset_ms_*_suppressed`). Added the rise onset (half way from the −200 ms level to the peak) as a second rule; both are in `part4.json`. This is overview Open Question 2, now with evidence.
- Is the suppressed units' pre-event level a sign-alignment artefact? Diagnostic (not reported): the same units' matched-control curves, signed by the *event* curve's peak, sit much closer to zero at −200 ms than the event curves do (printed on 2026-10-26, not stored; rerun the snippet in the session log if needed). So the offset is specific to the event regressors: these units fire less for a stretch longer than the ±200 ms window around events, either a slow pre-event decrease or a rate difference between the stretches where events occur and the rest of the stop. The window cannot tell the two apart; a wider window or the per-stop baseline terms of phase 3a can.
- Spot check of 12 called, suppressed non-SWR units: curves are oscillatory, several are offset below zero over the whole window, and some peak at ±200 ms. Single-unit timing is noisy; the population curves are the readable quantity.
- Enhanced units (after the review): population peaks `timing_population_peak_ms_*_enhanced`. With day-level resampling only the non-SWR replay rise onset has an interval below zero (`timing_population_rise_onset_ms_ci_*`); it is negative in most animals, while the SWR-coupled and all-SWR rise onsets are negative only in senor and wilbur (`timing_population_rise_onset_ms_*_enhanced_<animal>`). Paired units: `timing_peak_lag_difference_median_ms`. (The first commit's epoch-level intervals were too narrow; superseded.)

## Open questions

See [overview.md#open-questions](overview.md#open-questions). Answers from the lead go here with the date.
