# Phase 2 — SWR/non-SWR continuum verdict + peri-onset cross-correlogram

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Settle whether SWR-coupled and non-SWR non-local events are **one graded process
or two** — the "continuum vs two mechanisms" hypothesis the synthesis lists as
suggestive-but-untested. Two deliverables: (a) run the already-built cross-history
GLM and write its verdict; (b) build the definitive **peri-onset** test that Layer 2
flags "not yet run."

**Inputs to read first:**
- [scripts/rate/fit_cross_history_rate.py:129,144,172,195](../../../../scripts/rate/fit_cross_history_rate.py) — `fit_cross_history` (self vs self+cross, both directions), `_summary` (per-animal `delta_cv_bits` at `:123`), `main`/`--write` → `models_per_bin_rate/cross_history_models.pkl`. Built; never run/documented.
- [docs/replay_rate_content_swr_synthesis.md](../../../../docs/replay_rate_content_swr_synthesis.md) — **Layer 2** (`ripple_power_z` unimodal → "cut on a continuum"; the peri-onset **peak** is "the definitive test and is not yet run") and the **Hypotheses** block ("Continuum, not two mechanisms").
- [docs/per_bin_rate_results.md](../../../../docs/per_bin_rate_results.md) — §4 (the cross-history note this phase writes the verdict into).
- [Processed-Data/replay_content/event_onset_table.pkl](../../../../Processed-Data/replay_content/event_onset_table.pkl) + `event_table.pkl` — per-event onset times, `is_swr`, session keys; the cross-correlogram substrate.

## Tasks
- **Run the cross-history GLM and document the verdict.** `uv run python
  scripts/rate/fit_cross_history_rate.py --write` (smoke first with
  `--max-per-animal 1`; it fits per-bin clogit both directions, ~minutes/session —
  extrapolate before the full run). Interpret `delta_cv_bits` (per-animal) and the
  cross-history coefficient sign/timescale in **both** directions (does a recent
  with-SWR event raise/lower the non-SWR rate, and vice-versa?). Write a short
  **continuum verdict** into `docs/per_bin_rate_results.md` §4 and a one-paragraph
  cross-link in the synthesis Layer 2 (coupling ⇒ one modulated process, vs
  independence ⇒ two). A null cross-history effect is a valid, continuum-consistent
  result — state it plainly.
- **Build the peri-onset cross-correlogram (the definitive test).** New script
  `scripts/rate/nonlocal_onset_crosscorr.py`: per session, take non-local **onset**
  times split by `is_swr`, compute the SWR-onset ↔ non-SWR-onset cross-correlogram
  over immobility (e.g. ±0.5 s, small bins), normalize against a jitter/shuffle null,
  aggregate across sessions/animals. A peak at short lag ⇒ the two streams co-recruit
  (one graded process); a flat CCG ⇒ independent. Emit a figure to
  `paper/figures/<topic>/` + a caption. (Optionally add the ripple-power **peri-onset
  peak** distribution — the per-event peak ripple power at onset, split by is_swr — to
  test whether the coupling mark is unimodal at the detector's actual criterion, which
  the event-mean in Layer 2 could not.)
- **Update the synthesis "Hypotheses" block** to promote or retract "Continuum, not
  two mechanisms" based on the two results.

## Deliberately not in this phase
- The choice-mediation work (Phase 1) — unrelated.
- The full marked point process (joint intensity+coupling+content) — that is the
  separate `joint-replay-departure` plan; this phase only tests the continuum, not the
  joint model.
- Promoting the graded ripple-power content rung (kept exploratory by decision).

## Validation slice
| Test | Asserts |
| --- | --- |
| cross-correlogram on synthetic onsets | two independent Poisson onset streams → flat (within-null) CCG; a coupled stream (non-SWR onsets seeded near SWR onsets) → a peak at the seeded lag above the shuffle null. |
| jitter null | the shuffle/jitter null band is computed from the same onsets (per session), not a global constant. |
| cross-history run | `cross_history_models.pkl` written; `delta_cv_bits` present and finite for ≥ (stated fraction) of sessions; `_summary` prints per-animal aggregation. |
| doc verdict | `per_bin_rate_results.md` §4 states a directional verdict (coupled / independent / null) tied to the `delta_cv_bits` sign, not left as "pending". |

Mark the full cross-history run + all-session CCG slow.

## Fixtures
- Synthetic onset streams (independent + coupled) in the test module for the CCG.
- One session's `event_onset_table` slice for a CCG smoke run.

## Review
Before opening the PR, dispatch `scientific-code-reviewer` against the diff. Confirm:
- The CCG null is a per-session shuffle/jitter, not a hand-set band; the peak claim is relative to it.
- The cross-history verdict is stated from `delta_cv_bits` (per-animal), and a null is reported as continuum-consistent rather than chased.
- Onset times are the non-local **onset** (not full-window) and immobility-restricted, consistent with the rate ladder's event definition.
