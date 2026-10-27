# Overview — Scope, dependencies, integration, risks

[← back to PLAN.md](PLAN.md)

Line numbers verified on branch `pfc-response-followups` at `d6b8e0fd` (2026-10-26). Re-grep the named function if the branch has moved.

## Current codebase integration points

- `src/continuum_swr_replay/analysis/pfc/event_sets.py:255` — `build_session_event_sets`: builds the four classes from canonical events and Kay SWR intervals; `:418` `EVENT_CLASSES`. Phase 2 adds an opt-in fifth class; the default output is unchanged.
- `src/continuum_swr_replay/analysis/pfc/design.py:300` — `_make_design_matrices`: the nine-model ladder. Phase 2 adds models only when the new class is present.
- `src/continuum_swr_replay/analysis/pfc/modulation.py:240` — `CURVE_SOURCES` (model and basis block of each shipped curve); `:271` `NESTED_MODEL_PAIRS`. Phase 2 extends both behind the same switch.
- `src/continuum_swr_replay/analysis/pfc/session.py:583` `_MAX_MOD_STAT`, `:592` `_modulation_curves`, `:811` `run_session`, `:1162` `_load_aligned_session`, `:1220` `_session_fit_population`, `:1283` `_session_events`, `:1589` `_stack_session_outputs` (the curves are trimmed here to ±200 ms). Phase 0 reuses the three private assembly helpers for the marked family; phase 2 threads the extra class and curve through.
- `src/continuum_swr_replay/analysis/pfc/heldout.py:111` `ENGAGEMENT_PAIR`, `:116` `BEST_MODEL_CANDIDATES`, `:608` `control_draw_null` (one matched null per class, already generic over `classes`).
- `src/continuum_swr_replay/analysis/pfc/matched_control.py:171` `make_matched_control_times`: avoids every `EVENT_CLASSES` window. The new class is a subset of `SWR`, so the shipped control and the existing classes' nulls are unchanged by it.
- `scripts/pfc/run_pfc_analysis_bandit.py:218` `modulation_path`, `:234` `results_exist`, `:390` `score_session`, `:650` `maximum_likelihood_comparison`, `:800` `run_analysis`, `:1276` CLI. Phase 2 adds `--event-set`.
- `src/continuum_swr_replay/analysis/pfc/load_population.py:53` `HELDOUT_P_COLUMNS`, `:275` `load_population_data` (returns curves `swr`, `no_swr`, `non_local_swr`, `non_local_no_swr`, `matched_control`, each `(201, n_units)` on ±200 ms). Phase 1 reads it unchanged; phase 2 adds an optional extra curve.
- `src/continuum_swr_replay/reported/part4.py:809` `overlap_and_effect_values` (the pattern every new value function copies), `:1054` `summarize`. Phases 1 and 4 add value functions.
- `paper/reported/overview.md:140` — the Part 4 section; new bullets carry markers only.
- Branch `feature/pfc-marked-event-kernels` (tip `656fac63`, merge base `1e434151`, 694 commits behind): `marked_glm.py`, `marked_model_family.py`, `runner_support.py`, three runners and their tests. Phase 0 ports them; commit `66471fd3` is not ported (see Open Questions 1).
- Canonical bundles: `Processed-Data/{epoch}_mPFC_modulation.*` (75 sessions, commit `110c9ae`, 2026-10-11/12; logs `logs/pfc_refresh2_20261011/breeze/`). Read only, never written.

## Scope and dependency policy

### Goals

- (4) Timing of the response peak and its onset relative to event onset, per unit and for the population, for SWR-coupled replay, non-SWR replay and all SWRs, with a reference for how much the spline basis and prior smear a response that starts at onset.
- (2) The response to Kay SWRs that overlap no non-local event, against its own duration-matched null and against SWR-coupled replay in the same model.
- (3, 5, 8) Graded marks on events (content, strength, context), estimated against per-stop baselines, each reported jointly and alone, each read against a shuffled-mark reference.
- Every number reported through `part4.json`; the canonical Figure 4 bundles stay untouched.

### Non-Goals

- No change to the canonical Figure 4 model, its bundles, its 1 Hz gate, or claim 20.
- No OFC.
- No per-event response estimates; strength is a graded mark (lead's decision).
- From the marked-family v2 plan: no stop-rate link (its phase 4), no reactivation analysis (its phase 5), no interaction models; none answers the five questions.
- No push, no merge into another branch.

## Metrics

- Phase 0: the ported tests pass; the layering test passes; on one real session the marked context's fit mask and event onsets equal `run_session`'s exactly.
- Phase 1: timing values in `part4.json`; the calibration recovers a planted response onset within the stated error on simulated data.
- Phase 2: with the new class present, every canonical-class column of a smoke bundle equals the canonical bundle's within the fitter tolerance; 75 sessions, one contract, no failed sessions.
- Phase 3: as in the v2 plan (baseline columns unpenalised; no-baseline run reproduces contract 1 bit for bit); every target has joint and marginal estimates and a null band.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| Spline basis (50 ms knots) and the σ = 3 prior spread a post-onset response into negative lags, so "starts before onset" is partly an artefact. | Phase 1 measures the smear on simulated responses fit by the same basis and prior, and reports onsets beside that reference. |
| Peak-sign alignment of noisy curves makes a population curve that peaks wherever noise peaks. | Population timing uses only units called for that class; the matched control's peak-lag distribution is reported as the floor. |
| SWRs without replay are few per session (~17 % of Kay SWRs) and weaker than SWR-coupled ones; a smaller response could reflect strength, not content. | Their own count-matched null; counts reported per session; strength confound listed for the lead (Open Questions 4). |
| The extended refit drifts from the canonical fit (code moved since `110c9ae`). | The smoke compares the canonical-class columns of the new bundles with the canonical bundles before the batch is launched. |
| Context marks again carry stationary rate differences between stops (the v1 failure). | Every stop-level context mark has a matching unpenalised per-stop baseline column (phase 3a). |
| Marked-family code ported from a base 694 commits old silently uses the old session assembly (no excluded-time rule). | Phase 0 rewires it to the current helpers and checks equality with `run_session` on a real session. |
| GPU contention with other jobs on breeze. | `nvidia-smi` before every launch; GPU 0 never used; one tmux worker per card. |

## Rollout Strategy

New bundle trees only (see [shared-contracts.md#bundle-trees](shared-contracts.md#bundle-trees)). The canonical runner's default path, artifact names and outputs are unchanged; the new class and models exist only under `--event-set with_swr_without_replay`. The marked family writes under its own directory with a new contract version; the 2026-09-19 contract-1 corpus is renamed, not deleted.

## Open Questions

Questions for the lead. Each has the default the plan proceeds with.

1. **Commit `66471fd3` ("Make the PFC population fully descriptive")** on the marked branch drops Figure 4's 1 Hz firing-rate gate and the "has a held-out fit" screen. It changes the canonical Figure 4 population, which this plan must not touch. Default: not ported.
2. **Onset definition (phase 1).** Default: the half-maximum onset, the earliest lag of the contiguous run around the peak where the curve keeps the peak's sign and at least half its size. Reported with the smoothing calibration. Alternatives: a fixed threshold in log-rate units, or the lag where the sign-aligned population curve leaves the matched-control band.
3. **Unit set for timing.** Default: units called for that class (raw p < 0.05 against its own matched null); all units reported beside as a descriptive check.
4. **SWRs without replay are weaker events.** Their ripple power is not in the event table (they are not non-local events). Default: report their count and duration beside the response; no strength matching. Option: read per-SWR power from the Kay inventory and compare power-matched subsets.
5. **"Spike count" for strength (phase 3).** The event table has `mua_rate_z` (event-mean z-scored hippocampal multiunit rate), not a count. A count is rate × duration and would be nearly collinear with duration in a joint strength model. Default: `mua_rate_z`; a log spike count is listed as the alternative.
6. **"Recently rewarded arm" (phase 3).** Default: the existing `content_weighted_recent_reward` (content-weighted reward fraction over each arm's last ten visits). Alternative: the content fraction on the last rewarded arm, which equals the current arm after a rewarded stop and so overlaps the current-arm mark.
7. **"Trials leading up to a patch switch" (phase 3).** The existing `switch_patch_context` looks back (the last completed trip switched patch). Default: a new forward mark, `trips_to_patch_switch`: completed trips from this stop until the next trip that leaves the patch, capped at 3, used as an ordinal 0–3; the binary "the next trip leaves the patch" is its marginal companion. Undefined after the session's last switch.
8. **Strength marks on the new class.** Default: strength marks only on the two replay classes; SWRs without replay are not marked.

## Estimated Effort

- Phase 0: ~10.9 k lines brought over, a few hundred lines of import and assembly edits.
- Phase 1: ~200 LOC library, ~150 LOC calibration script, ~150 LOC part4 values, ~200 LOC tests; CPU only.
- Phase 2: ~250 LOC across event sets, design, session and runner, ~250 LOC tests; batch of order 80–100 GPU-hours (measure on the smoke).
- Phase 3a: ~600 LOC source, ~500 LOC tests. Phase 3b: launcher, audit; batch cost measured on the smoke.
- Phase 4: ~300 LOC part4 values, overview bullets.
