# Phase 2 (B) — SWRs without replay: a new event class and a refit

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [contracts](shared-contracts.md)

Why: question 2. About a sixth of Kay SWRs overlap no non-local event, and their time bins are mostly Local (`part2.kay_no_nonlocal_fraction`, `part2.kay_no_nonlocal_mostly_local_fraction`). If mPFC responds to them as it does to SWR-coupled replay, it follows the ripple; if not, it follows replay. The canonical ladder has no such class: its `SWR` regressor pools them with the SWRs that carry replay.

**Inputs to read first:**

- [src/continuum_swr_replay/analysis/pfc/event_sets.py:255-418](../../../../src/continuum_swr_replay/analysis/pfc/event_sets.py#L255) — `build_session_event_sets` (cohort restriction by onset, then the `is_swr` split) and `EVENT_CLASSES`.
- [src/continuum_swr_replay/analysis/pfc/session.py:1283](../../../../src/continuum_swr_replay/analysis/pfc/session.py#L1283) `_session_events` (where the artifact-dropped windows become known) and `:811` `run_session`; `:583-657` `_MAX_MOD_STAT` / `_modulation_curves`; `:1589` `_stack_session_outputs`.
- [src/continuum_swr_replay/analysis/pfc/design.py:300](../../../../src/continuum_swr_replay/analysis/pfc/design.py#L300) `_make_design_matrices`; [modulation.py:240](../../../../src/continuum_swr_replay/analysis/pfc/modulation.py#L240) `CURVE_SOURCES`, `:271` `NESTED_MODEL_PAIRS`.
- [src/continuum_swr_replay/analysis/pfc/heldout.py:608](../../../../src/continuum_swr_replay/analysis/pfc/heldout.py#L608) `control_draw_null` (already per class over `classes`).
- [scripts/pfc/run_pfc_analysis_bandit.py:234](../../../../scripts/pfc/run_pfc_analysis_bandit.py#L234) `results_exist` (required files, sidecar keys), `:390` `score_session` (the null loop over `EVENT_CLASSES` at the `n_control_draws > 0` branch), `:800` `run_analysis` (saving the curves and the sidecar).
- [src/continuum_swr_replay/analysis/pfc/load_population.py:275](../../../../src/continuum_swr_replay/analysis/pfc/load_population.py#L275).
- `logs/pfc_refresh2_20261011/breeze/*.log` — the canonical batch: per session 30–100 min, almost all in the null.

**Contracts referenced:**

- [Event classes and curves](shared-contracts.md#event-classes-and-curves) — the `SWR_no_non_local` definition; do not weaken.
- [Bundle trees](shared-contracts.md#bundle-trees), [Commits and GPU runs](shared-contracts.md#commits-and-gpu-runs).

## Tasks

- **Event set.** In `event_sets.py` add `SWR_NO_NON_LOCAL = "SWR_no_non_local"`, `EXTENDED_EVENT_CLASSES = (*EVENT_CLASSES, SWR_NO_NON_LOCAL)` and

  ```python
  def swr_without_non_local(
      swr_intervals: pd.DataFrame,      # kept, cohort-restricted SWRs
      non_local_events: pd.DataFrame,   # every canonical non-local event of the session in the frame
      unknown_windows: pd.DataFrame,    # artifact-dropped windows
  ) -> tuple[pd.DataFrame, dict[str, int]]:
      starts = swr_intervals.start_time.to_numpy(float); ends = swr_intervals.end_time.to_numpy(float)
      nl = merge_intervals(non_local_events.start_time.to_numpy(float), non_local_events.end_time.to_numpy(float))
      unk = merge_intervals(unknown_windows.start_time.to_numpy(float), unknown_windows.end_time.to_numpy(float))
      overlaps_replay = overlaps_open(starts, ends, nl[:, 0], nl[:, 1])
      unknown = ~overlaps_replay & overlaps_open(starts, ends, unk[:, 0], unk[:, 1])
      keep = ~overlaps_replay & ~unknown
      return swr_intervals.loc[keep].reset_index(drop=True), {
          "n_swr_with_non_local": int(overlaps_replay.sum()),
          "n_swr_no_non_local": int(keep.sum()),
          "n_swr_no_non_local_unknown": int(unknown.sum()),
      }
  ```

  (check `overlaps_open`'s exact signature in `intervals.py` and handle empty inputs). `build_session_event_sets` keeps the pre-cohort canonical frame (already restricted to the position frame) so the overlap test sees replay outside the cohort too, and returns it under a private key, or `_session_events` reloads it; `_session_events` calls `swr_without_non_local` after `artifact_dropped` exists, only when `event_set == "with_swr_without_replay"`, and adds the counts to `event_counts`.
- **Models and curves (extended only).** `_make_design_matrices(..., event_set="canonical")`: under the extended set add `"SWR_no_non_local"` and `"non_local_no_SWR + non_local_SWR + SWR_no_non_local"` (blocks in that order). `CURVE_SOURCES` gains, through a second mapping used only by the extended set, `"SWR_no_non_local": (<three-class model>, 2)` and `"non_local_SWR_3class": (<three-class model>, 1)`; `NESTED_MODEL_PAIRS` gains `("SWR_no_non_local", "constant")`, `(<three-class>, "non_local_SWR + non_local_no_SWR")` and `(<three-class>, "SWR_no_non_local")`. `BEST_MODEL_CANDIDATES` gains `"min_heldout_model_three_class": ("constant", "non_local", "non_local_SWR + non_local_no_SWR", <three-class>)`. A canonical-set run produces exactly today's models, columns and files (test below).
- **Session outputs.** `run_session` / `process_neuron` / `_stack_session_outputs` return curves as a dict keyed by curve name instead of the fixed 5-tuple, so a sixth and seventh curve do not need new positional slots; the runner saves `<artifact>.<curve>_modulation.npy` for every key (the five canonical names are unchanged on disk). Update every caller (grep `run_session(` and `process_neuron(`, including `scripts/figures/generate_figure04.py` and the tests).
- **Null.** `score_session` loops over the run's class tuple (`EXTENDED_EVENT_CLASSES` when extended), so the new class gets its own fixed matchable subset and R draws matched to its durations; `n_usable_null_draws` and `p_heldout` columns appear for it.
- **Runner.** `--event-set {canonical,with_swr_without_replay}` (default `canonical`). The extended set refuses `--output-dir` equal to `processed_data_dir()` (so no file can land beside the canonical bundles), records `event_set` in the sidecar and in the skip check, and requires the two new `.npy` files in `results_exist`. `load_population_data(..., extra_curves=("SWR_no_non_local", "non_local_SWR_3class"))` reads them when asked; `HELDOUT_P_COLUMNS` gains the new class only through an extended mapping, so canonical reads are unchanged.
- **Smoke** (GPU, measure first). Three sessions spanning the canonical unit counts (smallest, median, largest by `cohort_counts`/unit count from the canonical sidecars), one card each, `--n_workers 1 --gpu <N> --sessions <file> --event-set with_swr_without_replay --output-dir Processed-Data/pfc_response_followups/swr_without_replay_smoke`, full R = 1,000. Record wall time and peak GPU memory (`nvidia-smi --query-gpu=memory.used` sampled every 30 s into the log directory). Checks, scripted in `scripts/qa/compare_pfc_event_set_bundles.py` (new): for every canonical column and the five canonical curves, the smoke bundle equals the canonical bundle within the fitter tolerance (`atol 1e-6` on curves, p-values identical, since folds, seeds and the matched control are unchanged); the new class's event counts, called fraction and null usable draws are sane (all R usable or the shortfall explained). Stop and fix if the canonical columns differ.
- **Extrapolate** from the smoke (sum of per-session minutes scaled by unit count against the canonical batch's own per-session minutes) and write the estimate in the ledger. If the full batch would exceed about 16 h on the free cards, restrict the null to the new class (`--null-classes SWR_no_non_local`, recorded in the sidecar) and read the other classes' p-values from the canonical bundles in phase 4; otherwise refit everything.
- **Batch.** Partition the 75 sessions across the free cards (never 0) balanced by the smoke's minutes-per-unit; tmux session `pfc_b_swr_without_replay`, one window per card, logs `logs/pfc_response_followups/swr_without_replay/`; output `Processed-Data/pfc_response_followups/swr_without_replay/`. Monitor for `Traceback`, `MemoryError`, `EXIT`. Afterward: 75 bundles, one `event_set`, the comparison script over all 75 sessions.
- **Docs.** `scripts/README.md`: `--event-set`; one paragraph in the module docstring of `event_sets.py` defining the class.

## Deliberately not in this phase

- No reported values (phase 4 reads this tree).
- No strength matching of SWRs without replay (overview Open Questions 4).
- No change to the canonical default path, its artifact names or Figure 4.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_swr_without_non_local_partitions_swrs` | synthetic SWRs: one overlapping replay, one touching a replay only at an endpoint (kept), one overlapping a dropped window (unknown), one clear (kept); counts sum to the input |
| `test_swr_without_non_local_sees_replay_outside_the_cohort` | an SWR in the cohort overlapping a non-local event whose onset is outside the cohort is not "without replay" |
| `test_canonical_event_set_is_unchanged` | `_make_design_matrices` and `NESTED_MODEL_PAIRS`/`CURVE_SOURCES` under `canonical` equal today's (names, widths, order) |
| `test_extended_design_block_order` | the three-class model's blocks are `non_local_no_SWR`, `non_local_SWR`, `SWR_no_non_local`, and the two new curves read blocks 2 and 1 |
| `test_matched_control_unchanged_by_the_extra_class` | same seed: the control drawn with and without the class is identical |
| `test_control_null_covers_the_extra_class` | `control_draw_null(classes=EXTENDED_EVENT_CLASSES, ...)` on a small synthetic session returns improvements for the new class with its own `n_tested` |
| `test_runner_extended_set_refuses_the_canonical_directory` | CLI error |
| `test_results_exist_requires_event_set_and_new_curves` | a canonical sidecar never satisfies an extended request and vice versa |
| smoke (`slow`, GPU, recorded in the ledger) | canonical columns equal; runtime and memory recorded |

## Fixtures

Synthetic interval frames and a small synthetic session (existing `tests/analysis/pfc/test_heldout.py` / `test_matched_control.py` helpers). Real data: the three smoke sessions.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind): the fixed 5-tuple curve returns are gone.
- User-facing documentation listed as tasks is updated, not deferred.
