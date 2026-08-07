# Phase 2 — SWR/non-SWR continuum verdict + peri-onset cross-correlogram

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Settle whether SWR-coupled and non-SWR non-local events are **one graded process
or two** — the "continuum vs two mechanisms" hypothesis the synthesis lists as
suggestive-but-untested. Two deliverables: (a) consume and interpret the
cross-history GLM output promoted by the canonical regression Phase-13 refit; (b)
build the definitive **peri-onset** test that Layer 2 flags "not yet run." This
phase does not own a second cross-history fit path.

**Inputs to read first:**
- [scripts/rate/fit_cross_history_rate.py:129,144,172,195](../../../../scripts/rate/fit_cross_history_rate.py) — current `fit_cross_history` implementation (self vs self+cross, both directions) and historical fixed-path writer; Phase 13 migrates/refits it, so this phase reads only the promoted result.
- [docs/replay_rate_content_swr_synthesis.md](../../../../docs/replay_rate_content_swr_synthesis.md) — **Layer 2** (`ripple_power_z` unimodal → "cut on a continuum"; the peri-onset **peak** is "the definitive test and is not yet run") and the **Hypotheses** block ("Continuum, not two mechanisms").
- [docs/per_bin_rate_results.md](../../../../docs/per_bin_rate_results.md) — §4 (the cross-history note this phase writes the verdict into).
- Phase-13 promoted `event_onset_table` + `event_table` outputs — per-event onset
  times, `is_swr`, session keys, and source manifests; fixed-path processed-data
  pickles are not accepted.

## Tasks
- **Consume the promoted cross-history GLM and document the verdict.** Require the
  completed cross-history cell from
  [`regression-integrity-fixes` Phase 13](../regression-integrity-fixes/phase-13-refit-reporting.md);
  do not invoke a legacy fixed-path `--write` run. Interpret its held-out score
  delta (per animal) and the
  cross-history coefficient sign/timescale in **both** directions (does a recent
  with-SWR event raise/lower the non-SWR rate, and vice-versa?). Write a short
  **continuum verdict** into `docs/per_bin_rate_results.md` §4 and a one-paragraph
  cross-link in the synthesis Layer 2. Bidirectional coupling supports a shared or
  interacting process; a null is inconclusive and must not be presented as evidence
  for either a continuum or two distinct mechanisms.
- **Build the peri-onset cross-correlogram (the definitive test).** New script
  `scripts/rate/nonlocal_onset_crosscorr.py`: per session, take non-local **onset**
  times split by `is_swr`, compute the SWR-onset ↔ non-SWR-onset cross-correlogram
  over immobility (e.g. ±0.5 s, small bins), normalize against a jitter/shuffle null,
  aggregate across sessions/animals. A short-lag peak supports temporal co-recruitment
  and the graded-process hypothesis; a flat CCG means no detected short-lag coupling,
  not proof of distinct mechanisms. Emit a figure to
  `paper/figures/<topic>/` + a caption. (Optionally add the ripple-power **peri-onset
  peak** distribution — the per-event peak ripple power at onset, split by is_swr — to
  test whether the coupling mark is unimodal at the detector's actual criterion, which
  the event-mean in Layer 2 could not.)
- **Update the synthesis "Hypotheses" block** to promote or retract "Continuum, not
  two mechanisms" based on the two results.

## Deliberately not in this phase
- Choice modeling; the historical Phase 1 is superseded by regression-integrity Phase 11.
- The joint replay-onset/departure process — that is the separate
  `joint-replay-departure` plan. Neither that plan nor this phase owns a replay-content
  mark model.
- Promoting the graded ripple-power content rung (kept exploratory by decision).

## Validation slice
| Test | Asserts |
| --- | --- |
| cross-correlogram on synthetic onsets | two independent Poisson onset streams → flat (within-null) CCG; a coupled stream (non-SWR onsets seeded near SWR onsets) → a peak at the seeded lag above the shuffle null. |
| jitter null | the shuffle/jitter null band is computed from the same onsets (per session), not a global constant. |
| cross-history input gate | the Phase-13 run manifest contains the expected cross-history cells, shared folds, fit-validity fields, and animal-balanced score/effect summaries; no fixed-path pickle is read. |
| doc verdict | `per_bin_rate_results.md` §4 states coupled / no-detected-coupling / inconclusive from the manifest-declared held-out score and effect intervals, without treating a null as proof. |

Mark the all-session CCG slow; the cross-history production run remains Phase-13 work.

## Fixtures
- Synthetic onset streams (independent + coupled) in the test module for the CCG.
- One session's `event_onset_table` slice for a CCG smoke run.

## Review
Before opening the PR, dispatch `scientific-code-reviewer` against the diff. Confirm:
- The CCG null is a per-session shuffle/jitter, not a hand-set band; the peak claim is relative to it.
- The cross-history verdict uses the promoted held-out score/effect intervals with
  per-animal summaries, and a null is reported as inconclusive rather than chased or
  counted as support for either mechanism.
- Onset times are the non-local **onset** (not full-window) and immobility-restricted, consistent with the rate ladder's event definition.
