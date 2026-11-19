# Phase 3 — Canonical `is_swr` sweep across ancillary analyses

> **Scope reconciliation (2026-07-29):** only the non-PFC ancillary
> reality/occupancy adapter and divergence-test slice remains. PFC event plumbing is
> owned by [`pfc-fig4-recut`](../pfc-fig4-recut/PLAN.md), and this residual phase
> must consume the run promoted by
> [`regression-integrity-fixes` Phase 13](../regression-integrity-fixes/phase-13-refit-reporting.md).
> Do not execute the PFC tasks in this historical file.

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

> **STATUS: NARROWED / NOT started.** Run only after regression-integrity Phase 13
> promotes the canonical event inputs.

Every paper claim should rest on **one** event definition. The canonical non-local
event set and its SWR-overlap label come from
`build_classified_nonlocal_event_table` — events over the full non-local label
window, with `is_swr` = **strict positive-duration overlap** with the `SWR_event`
label intervals (`is_swr = overlap_class != "non_swr"`). But two non-PFC ancillary
analyses **re-derive** events / `is_swr` differently from raw per-bin `SWR_event`
flags, so they run on subtly different event sets and non-SWR denominators than the
rate/content claims. This phase points those two consumers at the canonical table
and adds tests that fail on divergence.

This residual phase changes no regression covariate or model, but it depends on the
source-valid canonical event table promoted by regression-integrity Phase 13.

**Inputs to read first:**
- `analysis/hippocampus/event_table.py:206` `build_classified_nonlocal_event_table` (canonical timing + `is_swr`); column list `:121-140`; `is_swr` rule `:324`. CLAUDE.md "Canonical per-event classification table" — `is_swr` is strict positive-duration overlap, *not* the inclusive `any SWR_event > 0`, and threshold-crossing `ripple_times` are **not** used for classification.
- Reality: `scripts/analyze_event_spike_evidence.py:139-168` (`build_session_events` reconstructs events from thresholded prob + `MIN_EVENT_BINS=3` at `:63`), `:149` `swr = dot["SWR_event"] > 0`, `:159` `is_swr = bool(swr[sl].any())` (inclusive).
- Occupancy: `scripts/analyze_content_occupancy.py:138` `is_swr = dot["SWR_event"] > 0` (raw timebin), `:179`/`:203` the split.
- The audit findings this residual phase closes: [02 (#2, Next Checks 1; Test Gaps "is_swr agrees with canonical")](../../reviews/02_spike_evidence_reality_review.md) and [04 (#3, Next Checks 4)](../../reviews/04_content_behavior_covariates_review.md).

## Tasks

- **Add one canonical adapter** `events_for_time_index(classified_events, time_index)` in `analysis/hippocampus/event_table.py` (next to the builder): given the per-session classified event table and a decode time index (the `data_over_time`/`position_info` index the consumer bins against), return one row per canonical event with `(start_bin, end_bin, is_swr, overlap_class, event_start, event_end)` where `start_bin`/`end_bin` are the half-open bin indices covering `[event_start, event_end]` under that index's left-edge convention. This is the single seam both consumers call — no consumer re-thresholds or re-overlaps.
- **Reality** (`analyze_event_spike_evidence.py`): replace the event-reconstruction half of `build_session_events` (`:139-168`) — the prob-threshold + `MIN_EVENT_BINS` + `is_swr = swr.any()` path — with `events_for_time_index(...)`; the event windows and `is_swr` now come from canonical. Keep the matched **non-event immobile baseline** machinery untouched (it is the null, not an event set). Re-run the existing 5-session prototype; report whether the headline `dll_animal`/`dll_per_spike` reproduces within noise or shifts (boundary events reclassify). `MIN_EVENT_BINS` and the inclusive `swr.any()` become dead — delete them.
- **Occupancy** (`analyze_content_occupancy.py`): add a canonical **event-level** `is_swr` split alongside the existing raw-timebin split (`:138`) — join non-local timebins to canonical event windows via the adapter and label each bin by its event's `is_swr`. Report **both** the timebin-weighted and the event-level-canonical split (do not silently replace; #14's existing number is timebin-weighted). The raw-timebin `SWR_event` split stays as the labeled-as-such comparison.
- **Divergence tests** (the durable guard): a test per residual consumer that, on a fixture session, asserts the consumer's event set / `is_swr` **equals** the canonical table's (same event count, same `is_swr` per event, same start/end within bin tolerance) — fails if anyone re-introduces a private event/`is_swr` construction.
- **Docs:** `paper/claims_table.md` #14 — state the percentage is timebin-weighted content-mass occupancy and add the event-level-canonical sensitivity. Note in both scripts' module docstrings that event/`is_swr` come from the canonical table.

## Deliberately not in this phase
- **No reality full-dataset / all-events run** and **no matched-baseline strengthening** — gated on the re-decode (the W3 reality workstream); this phase only swaps the *event/`is_swr` source*, not the sampling or the baseline.
- **No PFC code, tests, batch rerun, controls, inference, or Figure 4 work.** All PFC
  event plumbing and analysis are owned by
  [`pfc-fig4-recut`](../pfc-fig4-recut/PLAN.md).
- No covariate changes (phases 0–2); no new science.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_event_split::test_adapter_bins_cover_window` | `events_for_time_index` start/end bins half-open-cover `[event_start, event_end]`; `is_swr`/`overlap_class` pass through unchanged from the canonical table. |
| `test_event_split::test_reality_events_canonical` | reality event set on a fixture session == canonical table (count, `is_swr` per event, start/end within one bin). |
| `test_event_split::test_occupancy_event_split_canonical` | occupancy event-level `is_swr` labels match canonical; both denominators emitted. |
| reality repro (slow) | 5-session prototype re-run: headline `dll` reproduces within noise or the shift is reported. |

## Fixtures
One real session's classified event table + its `data_over_time.pkl` for the adapter/divergence tests (the reality repro reuses the existing 5-session set).

## Review
Dispatch `code-reviewer`. Confirm: both consumers call the single
`events_for_time_index` adapter and neither re-derives events/`is_swr`; the inclusive
`swr.any()` and `MIN_EVENT_BINS` are deleted from reality (no dead fallback);
occupancy reports **both** denominators (timebin number not silently dropped); the
divergence tests fail when a private construction is reintroduced (not tautological);
claim #14 wording is updated; and no PFC code or run is pulled into this residual phase.
