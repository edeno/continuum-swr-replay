# Phase 4 — Does mPFC firing scale with the number of replays in a stop?

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Why: the rate family says reward raises SWR-coupled replay rate and the rate rises over the first two seconds of a stop; the marked family says the per-event mPFC response barely changes across the stop. The link between them, mPFC firing against replay count per stop, has never been fit. This phase is independent of phases 1–3 and runs on CPU.

**Inputs to read first:**

- [src/continuum_swr_replay/analysis/pfc/marked_glm.py:307-437](../../../../src/continuum_swr_replay/analysis/pfc/marked_glm.py#L307) — `prepare_marked_session_context`: gives `fit_mask`, `counts` `(n_fit_bins, n_units)`, and the two replay event frames with `start_time`; everything this phase needs, loaded once.
- [src/continuum_swr_replay/analysis/pfc/cohort.py:32](../../../../src/continuum_swr_replay/analysis/pfc/cohort.py#L32) — `load_session_stop_cohort`: stops with `stop_id`, `stop_start`, `stop_end`, `immobility_duration_s`, `occupancy_well_id`, `is_rewarded_outcome`.
- [src/continuum_swr_replay/analysis/pfc/canonical_events.py:179](../../../../src/continuum_swr_replay/analysis/pfc/canonical_events.py#L179) — `load_session_swr_intervals`, for the exclusion windows of the outside-window variant.
- [src/continuum_swr_replay/analysis/replay/per_stop_rate_model.py:65](../../../../src/continuum_swr_replay/analysis/replay/per_stop_rate_model.py#L65) — `fit_per_stop_rate_model`: the rate family's per-stop Poisson with a `log(immobility)` offset; mirror its offset convention, do not reuse its design.
- `docs/rate_models.md` — which rate numbers this phase is meant to connect to.
- [scripts/pfc/summarize_pfc_marked_family.py:129-180](../../../../scripts/pfc/summarize_pfc_marked_family.py#L129) — the unit → session → animal aggregation to mirror.

**Contracts referenced:**

- [Bundle directory suffixes](shared-contracts.md#bundle-directory-suffixes) — `Processed-Data/pfc_stop_replay_rate/`.

## Tasks

- **Per-stop table builder** in a new module `src/continuum_swr_replay/analysis/pfc/stop_replay_rate.py`: `build_stop_unit_table(context, cohort_stops, swr_intervals, *, window_s=0.25)` returning one row per stop × unit with:
  - `n_bins`: fitted bins inside the stop; `spikes`: summed counts;
  - `n_nonswr`, `n_swr_coupled`: onsets of each replay class inside the stop;
  - `spikes_outside`, `n_bins_outside`: the same after excluding bins within `±window_s` of any replay onset and inside any SWR interval (so the tonic and the event-locked contributions can be separated);
  - `well`, `rewarded_outcome`, `outcome_unknown`, `log_dwell` (`log(immobility_duration_s)`), `stop_id`.
  Stops with no fitted bins are dropped and counted.

- **Per-unit models** in the same module, `fit_stop_replay_rate(table)` using `statsmodels` Poisson with offset `log(n_bins / 500)`:
  - M0: well indicators + rewarded_outcome + outcome_unknown + log_dwell;
  - M1: M0 + n_nonswr + n_swr_coupled;
  - M2: M1 on `spikes_outside` with offset `log(n_bins_outside / 500)` (tonic only).
  Return per unit the two replay coefficients from M1 and M2 as rate ratios per event, their standard errors, M1 − M0 deviance improvement, convergence flags. Guard: a unit with fewer than 20 spikes over the session's fitted bins is fit but flagged `low_count`; nothing is dropped.

- **Reference.** Within session, permute the pair `(n_nonswr, n_swr_coupled)` across stops 200 times, refit M1 per unit, and record the per-unit tail fraction of the observed coefficient (plus-one rule, `empirical_pvalue` from `heldout.py`). Descriptive only, as elsewhere.

- **Script** `scripts/pfc/fit_pfc_stop_replay_rate.py`: one parquet per session under `Processed-Data/pfc_stop_replay_rate/` (`<epoch_identifier>_stop_replay_rate.parquet`) with a provenance sidecar (input hashes via `record_inputs`, window, seed, permutations), the usual session selection flags, CPU only. Aggregation script `scripts/pfc/summarize_pfc_stop_replay_rate.py`: unit → session → animal medians of the two rate ratios for M1 and M2, sign consistency across sessions and animals, fraction of units with tail fraction below 0.05 by animal, and one figure (rate ratio per event by animal, M1 versus M2). Audit `docs/audits/<date>-pfc-stop-replay-rate.md`.

- **Docs.** `docs/README.md` results row; `scripts/README.md` entries; one paragraph in `docs/replay_rate_content_swr_synthesis.md` under the mPFC layer once numbers exist.

## Deliberately not in this phase

- No time-resolved (within-stop) mPFC rate model; the stop is the unit.
- No SWR-only events as a predictor (the family's `SWR` class); the question is about replay.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_stop_unit_table_counts_bins_spikes_and_onsets` | synthetic three-stop session: `n_bins`, `spikes`, `n_nonswr`, `n_swr_coupled` per stop match hand counts; an onset exactly at `stop_end` belongs to that stop. |
| `test_outside_window_exclusion_removes_event_bins_and_swr_intervals` | with one onset and one SWR interval planted, `n_bins_outside` equals `n_bins` minus the union of the excluded bins (overlap counted once). |
| `test_per_stop_model_recovers_a_planted_rate_ratio` | simulate 300 stops where rate = base × 1.3^n_nonswr with reward and well offsets: M1's non-SWR rate ratio is 1.3 ± 0.05, the SWR-coupled one 1.0 ± 0.05, well and reward coefficients recovered. |
| `test_tonic_and_event_locked_split` | plant the extra spikes only inside ±250 ms of onsets: M1 shows the ratio, M2 does not; plant them uniformly over the stop: both show it. |
| `test_permutation_reference_preserves_stop_marginals` | the shuffled count columns are a permutation of the originals per session; same seed same draws. |
| `test_aggregation_is_animal_balanced` | reuse the summary module's rule on a two-animal fixture. |
| real-data smoke (`slow`) | one session end to end; rate ratios finite for every unit; runtime under 5 min. |

## Fixtures

Synthetic stops and spike counts built in `tests/analysis/pfc/test_stop_replay_rate.py`; real-data smoke on j16 20210710 02_r1 (74 units).

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind).
- User-facing documentation listed as tasks is updated, not deferred.
