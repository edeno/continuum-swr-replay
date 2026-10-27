# Reported values and follow-ups: status

Branch `reported-values-2026-10-26` (stacked on master `6e33c499`; not merged, not pushed: merging needs
the lead's OK). Design: `docs/design/2026-10-26-reported-values-design.md`. Every reported number lives in
`paper/reported/part{1,2,3,4}.json`; `paper/reported/overview.md` reads them question by question.

## In flight (2026-10-26 evening)

| What | Where | State | When it finishes |
|---|---|---|---|
| Uncapped spike evidence (claims 6, 7, 7c): every event of every epoch, no 30-event cap | tmux `reality_all`; `logs/reality_all_20261026/progress.txt`; outputs `paper/figures/reality_all/per_epoch/` | 104/242 epochs at 19:40, none failed | about midnight |
| Slow and fast gamma during events (claim 5) | tmux `event_gamma`; `logs/event_gamma_20261026/run.log`; outputs `Processed-Data/event_gamma/` | done 19:44: 242/242 ok | done |

## To do when both runs are done

1. `uv run python scripts/reported/summarize_part.py --part 2` (it refuses until the uncapped run has all
   242 epochs and the gamma reference exists).
2. Claims table: claim 6's placement still says "at most 30 events per group per epoch" — reword for the
   uncapped run; check claim 7's wording against the new values.
3. Add markers for the new Part 2 values to the claims table (claims 2, 5, 21, 24) and to `overview.md`:
   - left out without the non-local detector: `nonlocal_events_outside_{kay,trigger,widloski}*`,
     `nonlocal_time_outside_{kay,swr,trigger,widloski}_*`, `rest_hours`, `kay_time_s`;
   - theta / ripple / MUA / slow and fast gamma during events against running and quiet rest:
     `{theta,ripple,mua,slow_gamma,fast_gamma}_{median_*,running_reference,rest_reference,animals_*}`,
     `theta_like_fraction_*`.
   - Gamma caveat to state with the values: spiking and the ripple leak broadband power into both gamma
     bands (fast more than slow); the SWR rise, and part of the non-SWR rise, may be leakage, not oscillation.
   - Open: compare `theta_like_fraction_non_swr` (13 % in the preview) with how often quiet rest reaches
     running theta, so 13 % has a reference.
4. Emit; run the unit suite; update `removed_part2.md` if wording drops anything.
5. Report to the lead; ask about merging into master.

## Open items the lead has not ruled on

- Claim 19a: separately fitted SWR content leans more to the current arm (+0.88 vs +0.53) but the pooled
  SWR x current-arm contrast is negative (-0.41). The models disagree in sign; check before citing 19a.
- `uncertainty_content/horserace_summary.csv` exists (242 sessions) but the docs still call the horse-race
  pending; not migrated.
- Retraction notes still quote the numbers they retract; the Part 3 agent treated them as dated records.
- Claim 10's per-session quality output (`quality_summary.csv`, April, 218 epochs) predates the re-decode.
- The three part worktrees (`continuum-swr-replay-reported-part{1,3,4}`) are merged and can be removed.

## Duration-aware event history (2026-10-26, simulation stage)

The per-bin rate design treats events as instantaneous onsets: bins inside an ongoing event count as at risk,
and history counts past onsets only. With events lasting ~50-90 ms, the self-history "burst" (Part 3
`rate_burst_*`, already pending) and the cross-history coupling (`models_per_bin_rate/cross_history_models.pkl`,
2026-10-11, no provenance sidecar, NOT in part3.json) may be duration artifacts. An agent is testing an at-risk
mask plus onset- and offset-locked history on simulated data (own worktree; report
`.claude/docs/reviews/2026-10-26-duration-aware-history-simulation.md`). Hold the cross-history values out of
Part 3 until then; then decide on refitting the real self- and cross-history models.

## Trends across sessions (2026-10-26, exploratory)

An agent (own worktree, read-only on real data) is checking whether event properties, behavioral correlates,
the decoder's transition matrix and mPFC response shares change across days and within days, with confound
checks. Output: `scripts/exploratory/_learning_trends.py` and `.claude/docs/reviews/2026-10-26-learning-trends.md`
on its branch. Memory: project-learning-evolution-followup.
Done 2026-10-26 20:40: branch `worktree-agent-a52c665f3acd09fe4` commit `8dc03829` (not merged). No change
generalises across animals (smallest q 0.12 on slopes over animals); across-day content trends track each
animal's HPC spike-rate drift and the per-epoch fitted transitions. Next checks: re-decode with a common
transition matrix; match epochs on spike count; within-day reward vs reward delivered so far.
Follow-up (commit `e14301d5`): Kay SWRs and SWR-coupled events are elevated early (mostly the first-exposure
epoch: novelty and p(R)=1.0 confounded); SWR-coupled keeps 1.32x without it; non-SWR shows no early peak.
Next: split the first-exposure epoch into its p(R) 1.0 and 0.5 halves.

The mPFC agent finished Phases A (timing) and 0 (port) on `pfc-response-followups`; six questions for the lead
before Phase B (in its `overview.md` Open Questions).

## Next project: mPFC response (approved 2026-10-26)

Items 2 (SWRs without replay), 3 (content), 4 (timing), 5 (event strength), 8 (behavioral context).
Plan: `.claude/docs/plans/pfc-response-followups/`. Order: Phase A (timing, no refit) and Phase 0 (port
the marked-family branch) first; then Phase B (new event class, GPU refit) and Phase C (marked family v2:
content, strength, context marks; GPU refit); results into `part4.json`.
