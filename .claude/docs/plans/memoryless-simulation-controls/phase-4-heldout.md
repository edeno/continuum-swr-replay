# Phase 4 — Held-out evidence and figure row 6

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#held-out-rule)

**Inputs to read first:**

- The design's "Held-out evidence" section.
- `docs/design/replay_detection_comparison/05-validation.md` — the parked study this
  tries first; its pilot issues list.
- `scripts/qa/sweep_memoryless_detection.py:159` — `build_recording`.
- `.venv/.../non_local_detector/likelihoods/sorted_spikes_kde.py:65` —
  `fit_sorted_spikes_kde_encoding_model` signature.

**Contracts referenced:** [Reported events](shared-contracts.md#reported-events),
[Truth labels](shared-contracts.md#truth-labels),
[Summary CSV layouts](shared-contracts.md#summary-csv-layouts).

**Designs referenced:** [Held-out rule](designs.md#held-out-rule), [Folds](designs.md#folds).

## Tasks

- **`src/continuum_swr_replay/analysis/inventory_comparison/heldout.py`**:
  `interleaved_folds`, `poor_coverage_fold`, `expected_log_likelihood`,
  `heldout_score` returning a frozen `HeldoutScore` (spike term, silence term, gain vs
  animal, shuffle p, coverage, supported, informative), `nearest_window_content` (maps
  a window posterior to the 2 ms grid), and `event_classes` (shared / ours-only /
  comparator-only by strictly positive overlap).
- **`build_recording(cells=...)`** in the sweep: fit the detector on a subset of cells;
  default all cells, so existing callers are unchanged.
- **`scripts/qa/sweep_memoryless_heldout.py`**: for folds A, B, C: build the discovery
  recording, fit the evaluation fields, run both detectors from discovery cells on every
  injected-replay cell and every controls trial, class each event, label its truth, and
  score it under both content estimators with 100 seeded evaluation-field permutations.
  Summaries per fold x estimator x class x truth: support rate, false-support rate,
  median gain, coverage, under-crediting, estimator agreement. Raw table to
  `Processed-Data`, summary to `detection_heldout_summary.csv` with provenance.
- **Smoke run** on one seed of fold A, timed; extrapolate; full run on a clean tree.
- **Figure row 6**: (n) gain by class split by truth, per estimator, fold C hatched;
  (o) false-support and support rates. Caption paragraph with checks; state that
  held-out support tests spatial information, not sequence, and is read with the
  simulation's error rates.
- **Spec**: spec 02 and spec 05 each gain a short pointer to the results; spec 05 notes
  the rule it tried and that it is ready for the pilot.

## Deliberately not in this phase

- Running the rule on recorded tetrodes (pilot).
- Choosing between the two content estimators: both are reported (design C8).

## Validation slice

| Test | Asserts |
| --- | --- |
| Folds | Disjoint; A and B each cover the track; C's evaluation cells cover only the lower half |
| `expected_log_likelihood` | Matches a hand computation on a 3-bin, 2-cell case; spike and silence terms separate |
| Content at the true place | Evaluation spikes drawn from the content's place give positive gain vs the animal |
| Content elsewhere | Negative gain |
| Shuffled fields | Gain near zero on average over permutations |
| Silence | Silence where fields predict firing lowers the silence term |
| `nearest_window_content` | Each 2 ms bin takes the nearest window midpoint's posterior |
| `event_classes` | Touching-but-not-overlapping events are not shared |
| Smoke run (slow, manual) | Completes; runtime recorded |

## Fixtures

Hand-built fields and counts for unit tests; the simulated recording for the sweep.

## Review

Before merging, dispatch `code-reviewer` (read-only) against the diff. Confirm the same
rule is applied to every event regardless of which detector found it, both estimators
are scored, and the floor on evaluation fields is documented.
