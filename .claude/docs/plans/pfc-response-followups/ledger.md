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

## Runs

| run | tmux session | cards | log dir | output dir | started | finished | runtime / memory |
|---|---|---|---|---|---|---|---|
| timing calibration | — (foreground, CPU) | none | — | `Processed-Data/pfc_response_followups/` | 2026-10-26 | 2026-10-26 | 2 min 15 s CPU |

## Findings (phase 1)

- Calibration: for a response starting exactly at onset, the half-max onset read off the fitted curve is *later* than the truth (`timing_calibration_onset_bias_ms_*`), so smoothing does not make onsets early. Per unit, a sizeable share of such units still shows an onset before zero by noise (`timing_calibration_onset_before_zero_fraction_*`): the per-unit "before onset" shares must be read against it.
- Peak lags cluster at a few lags spaced roughly by the 50 ms knot spacing (diagnostic plot); per-unit peaks also land on the window edge (`timing_peak_at_edge_fraction_*`).
- Suppressed units: their sign-aligned mean curves sit far from zero across the window (`timing_population_pre_event_level_fraction_*_suppressed`), so the half-max-from-zero onset runs to the edge (`timing_population_onset_ms_*_suppressed`). Added the rise onset (half way from the −200 ms level to the peak) as a second rule; both are in `part4.json`. This is overview Open Question 2, now with evidence.
- Is the suppressed units' pre-event level a sign-alignment artefact? Diagnostic (not reported): the same units' matched-control curves, signed by the *event* curve's peak, sit much closer to zero at −200 ms than the event curves do (printed on 2026-10-26, not stored; rerun the snippet in the session log if needed). So the offset is specific to the event regressors: these units fire less for a stretch longer than the ±200 ms window around events, either a slow pre-event decrease or a rate difference between the stretches where events occur and the rest of the stop. The window cannot tell the two apart; a wider window or the per-stop baseline terms of phase 3a can.
- Spot check of 12 called, suppressed non-SWR units: curves are oscillatory, several are offset below zero over the whole window, and some peak at ±200 ms. Single-unit timing is noisy; the population curves are the readable quantity.
- Enhanced units (after the review): population peaks `timing_population_peak_ms_*_enhanced`. With day-level resampling only the non-SWR replay rise onset has an interval below zero (`timing_population_rise_onset_ms_ci_*`); it is negative in most animals, while the SWR-coupled and all-SWR rise onsets are negative only in senor and wilbur (`timing_population_rise_onset_ms_*_enhanced_<animal>`). Paired units: `timing_peak_lag_difference_median_ms`. (The first commit's epoch-level intervals were too narrow; superseded.)

## Open questions

See [overview.md#open-questions](overview.md#open-questions). Answers from the lead go here with the date.
