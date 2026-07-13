# Phase 1 — Canonical event definition + provenance

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [shared-contracts](shared-contracts.md)

Replace the PFC pipeline's home-grown `is_swr` split with the paper-wide
canonical event table, settle the immobility inequality, emit a provenance
sidecar, and ship a per-session old-vs-new diagnostic so the size of the change
is measured before any full rerun. **No batch rerun in this phase** — validate on
2–3 sessions.

**Inputs to read first:**

- [src/continuum_swr_replay/analysis/pfc/glm.py:930-1090](../../../../src/continuum_swr_replay/analysis/pfc/glm.py#L930) — `run_session`; the event-construction block to replace is lines 1002-1046.
- [src/continuum_swr_replay/analysis/hippocampus/event_table.py:206-345](../../../../src/continuum_swr_replay/analysis/hippocampus/event_table.py#L206) — `build_classified_nonlocal_event_table`: returns `start_time`/`end_time`/`is_swr` (line 324), one row per non-local event over the full label window, already immobility-aware. Read its docstring for the arguments it needs (results, position/speed, thresholds).
- [scripts/figures/generate_figure04.py:320-360](../../../../scripts/figures/generate_figure04.py#L320) — the duplicate non-canonical construction (lines 330-352) and the `<=` immobility filter (line 357) that must match `glm.py`.

**Contracts referenced:**

- [Provenance sidecar schema](shared-contracts.md#provenance-sidecar-schema) — this phase writes the first sidecar (`n_matched_control = 0`).

## Tasks

- **Replace the event-construction block in `run_session`** ([glm.py:1002-1046](../../../../src/continuum_swr_replay/analysis/pfc/glm.py#L1002-L1046)). Delete the `get_confident_times` → `get_overlap_times(..., data["ripple_times"])` path. Instead call `build_classified_nonlocal_event_table` for this session, restrict to immobile events, and split on the canonical `is_swr` column:
  ```python
  event_table = build_classified_nonlocal_event_table(
      results, data, non_local_threshold=non_local_threshold,
      immobility_threshold=immobility_threshold,
  )  # verify exact kwarg names against event_table.py:206 before writing
  immobile = event_table[event_table["endpoints_speed_below_thresh"]]  # or the table's immobility column
  non_local_times = immobile[["start_time", "end_time"]]
  non_local_swr_times = immobile.loc[immobile["is_swr"], ["start_time", "end_time"]]
  non_local_no_swr_times = immobile.loc[~immobile["is_swr"], ["start_time", "end_time"]]
  ```
  Keep the `event_times` dict keys (`non_local_no_SWR`, `non_local_SWR`, `non_local`, `SWR`) unchanged so `_make_design_matrices` and every downstream test stay bit-identical in structure. `SWR` still comes from `data["ripple_times"]` (the ripple *regressor* control is unchanged — only the non-local *classification* moves to canonical).
- **Settle the immobility inequality.** Adopt `<` (the batch convention at [glm.py:1042](../../../../src/continuum_swr_replay/analysis/pfc/glm.py#L1042)) everywhere. Change [generate_figure04.py:357](../../../../scripts/figures/generate_figure04.py#L357) from `<=` to `<`. (Resolves Open Question 1.)
- **Write the provenance sidecar** in `run_session` next to the modulation artifacts, per [the schema](shared-contracts.md#provenance-sidecar-schema). `run_session` currently only returns arrays; thread the counts + `git_hash` through so the runner can write the JSON. Simplest: have `run_session` return an extra `provenance: dict` element (extends the tuple — update the runner unpack at [run_pfc_analysis_bandit.py:150-155](../../../../scripts/pfc/run_pfc_analysis_bandit.py#L150) and the `results_exist` set at lines 86-95 to include the sidecar). `is_swr_definition="canonical_event_table"`, `immobility_inequality="<"`, `n_matched_control=0`.
- **Add a per-session old-vs-new diagnostic script** `scripts/qa/diagnose_pfc_event_definition.py`: for a given session, build events both ways (old threshold/ripple-times split — inline the deleted logic here, it lives *only* in this diagnostic now — and new canonical), and report: total event count old vs new, n reclassified SWR↔non-SWR, event-start shift distribution, and n boundary/partial-overlap cases. Print a table; save a CSV under `.claude/research/pfc_event_diagnostic_outputs/`. This is the review's Next Check #2.
- **Update the module docstring** in [glm.py `run_session`](../../../../src/continuum_swr_replay/analysis/pfc/glm.py#L978) to state events use the canonical `is_swr` (strict positive-duration `SWR_event` overlap), not `ripple_times` threshold crossings.

## Deliberately not in this phase

- The matched non-event control (Phase 2) — this phase keeps the existing model set.
- The full batch rerun (Phase 3) — validate on a handful of sessions only.
- Per-animal inference and the figure rewrite (Phase 4).

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_run_session_uses_canonical_is_swr` (integration, `pytest.mark.slow`) | On one real session, `run_session`'s SWR/non-SWR event counts equal the canonical `build_classified_nonlocal_event_table` counts restricted to immobility — not the `get_overlap_times`/`ripple_times` split. |
| `test_event_construction_divergence_guard` | A synthetic results/data fixture where the canonical `is_swr` and the old ripple-times split disagree on ≥1 boundary event → the PFC path follows the canonical column (guards Next Check #5). |
| `test_immobility_inequality_consistent` | Parses the `<`/`<=` usage: `generate_figure04.py` and `glm.py` use the same operator (fails if either drifts). |
| `test_provenance_sidecar_written` (slow) | After `run_session` on a small session, the `.provenance.json` exists, validates against the schema keys, and has `is_swr_definition == "canonical_event_table"`. |
| `diagnose_pfc_event_definition.py` smoke | Runs on one session, emits the CSV with the expected columns; manual read-through of the reclassification counts. |

Mark the real-data tests `pytest.mark.slow`.

## Fixtures

- Synthetic `results` (acausal state probabilities with a non-local bump) + `data`
  (position_info with an immobility bout, `ripple_times` offset so the two splits
  disagree on a boundary event) in `tests/analysis/pfc/conftest.py`. Reuse the
  existing PFC fixtures if present; extend rather than duplicate.
- One real session (e.g. `wilbur20210329_.nwb`, `02_r1`) with existing HPC decoding
  results for the slow tests + diagnostic smoke.

## Review

Before opening the PR for this phase, dispatch `scientific-code-reviewer` against the diff. Confirm:
- The old `get_confident_times`/`get_overlap_times`-on-`ripple_times` non-local split is **deleted** from `run_session` (not left as a dead branch); it survives only inside the diagnostic script.
- Event-dict keys and `_make_design_matrices` inputs are structurally unchanged — the only change is which events populate the SWR/non-SWR classes.
- The diagnostic's old-path reconstruction matches what `run_session` did before this PR (so the comparison is honest).
- Validation slice passes; slow tests marked.
- Docstrings/tests don't reference this plan or "Phase 1".
