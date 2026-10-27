# Reported values and follow-ups: status

Branch `reported-values-2026-10-26` (stacked on master `6e33c499`; not merged, not pushed: merging needs
the lead's OK). Design: `docs/design/2026-10-26-reported-values-design.md`. Every reported number lives in
`paper/reported/part{1,2,3,4}.json`; `paper/reported/overview.md` reads them question by question.

## In flight (2026-10-26 evening)

| What | Where | State | When it finishes |
|---|---|---|---|
| Uncapped spike evidence (claims 6, 7, 7c): every event of every epoch, no 30-event cap | tmux `reality_all`; `logs/reality_all_20261026/progress.txt`; outputs `paper/figures/reality_all/per_epoch/` | 104/242 epochs at 19:40, none failed | about midnight |
| Slow and fast gamma during events (claim 5) | tmux `event_gamma`; `logs/event_gamma_20261026/run.log`; outputs `Processed-Data/event_gamma/` | 160/242 at 19:40, none failed | about 20:15 |

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

## Next project: mPFC response (approved 2026-10-26)

Items 2 (SWRs without replay), 3 (content), 4 (timing), 5 (event strength), 8 (behavioral context).
Plan: `.claude/docs/plans/pfc-response-followups/`. Order: Phase A (timing, no refit) and Phase 0 (port
the marked-family branch) first; then Phase B (new event class, GPU refit) and Phase C (marked family v2:
content, strength, context marks; GPU refit); results into `part4.json`.
