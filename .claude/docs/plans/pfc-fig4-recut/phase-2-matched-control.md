# Phase 2 — Matched non-event control (time-shifted onsets)

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [shared-contracts](shared-contracts.md)

Add the matched immobile non-event control promised by claim #20. Time-shifted
onsets, drawn within the same immobility bouts, matched on count and duration,
never overlapping a real event. Wire a `matched_control` event class through the
design matrices and add the key engagement LR test. Validate on synthetic +
small real data; **no full rerun** (Phase 3).

**Inputs to read first:**

- [src/continuum_swr_replay/analysis/pfc/glm.py:347-441](../../../../src/continuum_swr_replay/analysis/pfc/glm.py#L347) — `_make_design_matrices`; the model dict is built at lines 435-439. A new `matched_control` design matrix parallels `non_local_no_SWR`.
- [src/continuum_swr_replay/analysis/pfc/glm.py:538-789](../../../../src/continuum_swr_replay/analysis/pfc/glm.py#L538) — `process_neuron`; the LR-test pair list is at lines 733-743. Add one pair here.
- [src/continuum_swr_replay/analysis/pfc/glm.py:1034-1090](../../../../src/continuum_swr_replay/analysis/pfc/glm.py#L1034) — `run_session` `event_times` dict + fit/return; where control onsets are generated and threaded.

**Contracts referenced:**

- [Matched-control interface](shared-contracts.md#matched-control-interface) — the frozen `make_matched_control_times` signature, output columns, and invariants. **Do not weaken.**

## Tasks

- **Implement `make_matched_control_times`** in `glm.py` per [the contract](shared-contracts.md#matched-control-interface). Algorithm (time-shifted onsets, seeded):
  ```python
  def make_matched_control_times(event_times, is_immobile, rng, max_attempts=50):
      time = is_immobile.index.to_numpy()
      immobile = is_immobile.to_numpy()
      # contiguous immobility bouts as (start_idx, end_idx) index ranges
      bouts = _contiguous_true_runs(immobile)
      real_windows = event_times[["start_time", "end_time"]].to_numpy()
      out = []
      for start_t, end_t in real_windows:
          dur = end_t - start_t
          bout = _bout_containing(bouts, time, start_t)   # the immobility bout of this event
          if bout is None:
              continue
          placed = False
          for _ in range(max_attempts):
              cand_start = rng.uniform(time[bout[0]], time[bout[1]] - dur)
              cand_end = cand_start + dur
              if not _within_immobile(cand_start, cand_end, time, immobile):
                  continue
              if _overlaps_any(cand_start, cand_end, real_windows):
                  continue
              out.append((cand_start, cand_end)); placed = True; break
          if not placed:
              logger.debug("matched control: could not place event [%f,%f]", start_t, end_t)
      return pd.DataFrame(out, columns=["start_time", "end_time"])
  ```
  Write the small helpers (`_contiguous_true_runs`, `_bout_containing`, `_within_immobile`, `_overlaps_any`) as pure, offline-importable functions next to the existing pure helpers so they stay unit-testable without Spyglass.
- **Thread control onsets into `run_session`.** After building the canonical `non_local_times` (Phase 1), call `make_matched_control_times(non_local_times, is_immobile, rng=np.random.default_rng(seed))` with a seed derived deterministically from `(nwb_file_name, epoch_name)` — document the derivation. Add `"matched_control": matched_control_times` to the `event_times` dict at [glm.py:1034](../../../../src/continuum_swr_replay/analysis/pfc/glm.py#L1034).
- **Add the `matched_control` design matrix** in `_make_design_matrices` (mirror the `non_local_no_SWR` construction) and register it in the model dict at [glm.py:435-439](../../../../src/continuum_swr_replay/analysis/pfc/glm.py#L435).
- **Add the engagement LR test** `("non_local_no_SWR", "matched_control")` to the pair list at [glm.py:733-743](../../../../src/continuum_swr_replay/analysis/pfc/glm.py#L733), plus its modulation extraction. This is the test the claim actually needs: non-SWR replay modulation *above a matched non-event baseline*.
- **Extend `run_session`'s return + the runner** to save a `matched_control` modulation `.npy` and set `n_matched_control` in the provenance sidecar. Update `results_exist` ([run_pfc_analysis_bandit.py:86-95](../../../../scripts/pfc/run_pfc_analysis_bandit.py#L86)) and the save block ([lines 168-183](../../../../scripts/pfc/run_pfc_analysis_bandit.py#L168)) to include the new file.
- **Docstring:** update `run_session` and `process_neuron` docstrings to describe the matched-control model and the new test; note the seed derivation for reproducibility.

## Deliberately not in this phase

- The full batch rerun (Phase 3).
- `load_population` changes and the figure panel showing the control (Phase 4) —
  this phase only produces the per-session artifact.
- Any change to the canonical event definition (Phase 1, already landed).

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_matched_control_within_immobility` | Every returned window lies entirely in immobile bins (synthetic bouts). |
| `test_matched_control_no_event_overlap` | Zero overlap between any control window and any real event window. |
| `test_matched_control_count_and_duration` | Control count ≤ real count; each control duration equals its source event's duration. |
| `test_matched_control_deterministic` | Same seed → identical output; different seed → different placement. |
| `test_matched_control_unplaceable_dropped` | An event whose bout is too short to fit a non-overlapping control is dropped, not force-placed. |
| `test_process_neuron_has_matched_control_test` | `process_neuron` output includes `"non_local_no_SWR vs. matched_control: pval"` and a finite modulation array on a fixture where the neuron responds to non-SWR events. |
| `test_run_session_saves_matched_control` (slow) | Small real session: the matched-control `.npy` is written and the sidecar `n_matched_control > 0`. |

## Fixtures

- Synthetic immobility bouts of varying length (including one too short to place a
  control) + real event windows, in `tests/analysis/pfc/conftest.py`.
- The Phase 1 synthetic `results`/`data` fixture, extended with a neuron whose
  firing tracks non-SWR events for the `process_neuron` test.
- One real session for the slow save test.

## Review

Before opening the PR, dispatch `scientific-code-reviewer` against the diff. Confirm:
- `make_matched_control_times` matches the frozen contract signature and all four invariants have failing-if-violated tests.
- Randomness is seeded and the seed derivation is documented (no bare `default_rng()`).
- The new model/test slots in without altering existing test-pair semantics (the Phase 1 event classes are untouched).
- Validation slice passes; slow tests marked.
- Docstrings/tests don't reference this plan or "Phase 2".
