# Phase 3 — Canonical `is_swr` sweep across ancillary analyses

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

> **STATUS: NOT started.** Independent of phases 0–2/1.5 (only swaps the event/`is_swr`
> source in ancillary analyses); can proceed in parallel.

Every paper claim should rest on **one** event definition. The canonical non-local
event set and its SWR-overlap label come from
`build_classified_nonlocal_event_table` — events over the full non-local label
window, with `is_swr` = **strict positive-duration overlap** with the `SWR_event`
label intervals (`is_swr = overlap_class != "non_swr"`). But three ancillary
analyses **re-derive** events / `is_swr` differently — from raw per-bin `SWR_event`
flags or from threshold-crossing `ripple_times` — so they run on subtly different
event sets and non-SWR denominators than the rate/content claims. This phase points
them all at the canonical table and adds tests that fail on divergence.

This phase is **independent of phases 0–2** (it consumes the canonical event table,
which already exists; it touches no covariate). It is sequenced after the rebuild
only to avoid re-touching the reality consumers twice (the phase-1 rebuild already
quiesces them).

**Inputs to read first:**
- `analysis/hippocampus/event_table.py:206` `build_classified_nonlocal_event_table` (canonical timing + `is_swr`); column list `:121-140`; `is_swr` rule `:324`. CLAUDE.md "Canonical per-event classification table" — `is_swr` is strict positive-duration overlap, *not* the inclusive `any SWR_event > 0`, and threshold-crossing `ripple_times` are **not** used for classification.
- Reality: `scripts/analyze_event_spike_evidence.py:139-168` (`build_session_events` reconstructs events from thresholded prob + `MIN_EVENT_BINS=3` at `:63`), `:149` `swr = dot["SWR_event"] > 0`, `:159` `is_swr = bool(swr[sl].any())` (inclusive).
- Occupancy: `scripts/analyze_content_occupancy.py:138` `is_swr = dot["SWR_event"] > 0` (raw timebin), `:179`/`:203` the split.
- PFC: `analysis/pfc/glm.py:1003` `get_confident_times` (event reconstruction), `:1014` `max_speed < immobility`, `:1017` `get_overlap_times(non_local_times, data["ripple_times"])` (the non-canonical split), `:1042` `is_immobile = head_speed < immobility`; `scripts/generate_figure04.py:344` same `ripple_times` split, `:357` `head_speed <= immobility` (the `<=`/`<` mismatch vs the batch).
- The audit findings this phase closes: [02 (#2, Next Checks 1; Test Gaps "is_swr agrees with canonical")](../../reviews/02_spike_evidence_reality_review.md), [04 (#3, Next Checks 4)](../../reviews/04_content_behavior_covariates_review.md), [05 (#2, Definition-Drift, Next Checks 1/2/5)](../../reviews/05_pfc_cross_area_review.md).

## Tasks

- **Add one canonical adapter** `events_for_time_index(classified_events, time_index)` in `analysis/hippocampus/event_table.py` (next to the builder): given the per-session classified event table and a decode time index (the `data_over_time`/`position_info` index the consumer bins against), return one row per canonical event with `(start_bin, end_bin, is_swr, overlap_class, event_start, event_end)` where `start_bin`/`end_bin` are the half-open bin indices covering `[event_start, event_end]` under that index's left-edge convention. This is the single seam all three consumers call — no consumer re-thresholds or re-overlaps.
- **Reality** (`analyze_event_spike_evidence.py`): replace the event-reconstruction half of `build_session_events` (`:139-168`) — the prob-threshold + `MIN_EVENT_BINS` + `is_swr = swr.any()` path — with `events_for_time_index(...)`; the event windows and `is_swr` now come from canonical. Keep the matched **non-event immobile baseline** machinery untouched (it is the null, not an event set). Re-run the existing 5-session prototype; report whether the headline `dll_animal`/`dll_per_spike` reproduces within noise or shifts (boundary events reclassify). `MIN_EVENT_BINS` and the inclusive `swr.any()` become dead — delete them.
- **Occupancy** (`analyze_content_occupancy.py`): add a canonical **event-level** `is_swr` split alongside the existing raw-timebin split (`:138`) — join non-local timebins to canonical event windows via the adapter and label each bin by its event's `is_swr`. Report **both** the timebin-weighted and the event-level-canonical split (do not silently replace; #14's existing number is timebin-weighted). The raw-timebin `SWR_event` split stays as the labelled-as-such comparison.
- **PFC** (`pfc/glm.py`): replace the `get_overlap_times(non_local_times, data["ripple_times"])` split (`:1017`) with a canonical `is_swr` assignment from the adapter (the non-local events keep coming from the detector, but their SWR label is the canonical label-derived one, not the `ripple_times` overlap). `data["ripple_times"]` stays only as the standalone `"SWR"` ripple **regressor** (`:1041` `"SWR": data["ripple_times"]`), not as the classifier. **Unify the immobility inequality** to `<` (the batch convention, `glm.py:1042`): fix `generate_figure04.py:357` `<=`→`<` so the example-neuron path matches the batch.
- **Divergence tests** (the durable guard): a test per consumer that, on a fixture session, asserts the consumer's event set / `is_swr` **equals** the canonical table's (same event count, same `is_swr` per event, same start/end within bin tolerance) — fails if anyone re-introduces a private event/`is_swr` construction. Plus a unit test that `generate_figure04`'s immobility inequality matches `glm.py`'s.
- **Docs:** `paper/claims_table.md` #14 — state the percentage is timebin-weighted content-mass occupancy and add the event-level-canonical sensitivity; #20 — note the PFC split is now the canonical label-derived `is_swr` (not `ripple_times` overlap) and that Fig. 4 numbers refresh on the next PFC batch run. Note in each script's module docstring that event/`is_swr` come from the canonical table.

## Deliberately not in this phase
- **No reality full-dataset / all-events run** and **no matched-baseline strengthening** — gated on the re-decode (the W3 reality workstream); this phase only swaps the *event/`is_swr` source*, not the sampling or the baseline.
- **No PFC batch rerun, matched non-event control, or animal/session-level inference** — the `run_pfc_analysis_bandit.py` Spyglass-backed rerun + the matched-control + hierarchical-inference work is the separate deferred PFC workstream ([overview](overview.md#folded-in-audit-findings-claudereview)). This phase ships the PFC *code* change + per-session verification + the divergence test; the refreshed Fig. 4 numbers land when that workstream runs.
- No covariate changes (phases 0–2); no new science.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_event_split::test_adapter_bins_cover_window` | `events_for_time_index` start/end bins half-open-cover `[event_start, event_end]`; `is_swr`/`overlap_class` pass through unchanged from the canonical table. |
| `test_event_split::test_reality_events_canonical` | reality event set on a fixture session == canonical table (count, `is_swr` per event, start/end within one bin). |
| `test_event_split::test_occupancy_event_split_canonical` | occupancy event-level `is_swr` labels match canonical; both denominators emitted. |
| `test_event_split::test_pfc_split_canonical` | PFC non-local SWR/non-SWR partition == canonical `is_swr` (not `ripple_times` overlap). |
| `test_event_split::test_immobility_inequality_unified` | `generate_figure04` and `glm.py` use the same immobility comparison (`<`). |
| reality repro (slow) | 5-session prototype re-run: headline `dll` reproduces within noise or the shift is reported. |

## Fixtures
One real session's classified event table + its `data_over_time.pkl` for the adapter/divergence tests (the reality repro reuses the existing 5-session set).

## Review
Dispatch `code-reviewer`. Confirm: all three consumers call the single `events_for_time_index` adapter and none re-derive events/`is_swr`; the inclusive `swr.any()` and `MIN_EVENT_BINS` are deleted from reality (no dead fallback); occupancy reports **both** denominators (timebin number not silently dropped); PFC uses canonical `is_swr` for the split while `ripple_times` remains only the standalone ripple regressor; the immobility inequality is unified; the divergence tests actually fail when a private construction is reintroduced (not tautological); claims #14/#20 wording updated; the deferred reality/PFC heavy runs are not silently pulled in.
