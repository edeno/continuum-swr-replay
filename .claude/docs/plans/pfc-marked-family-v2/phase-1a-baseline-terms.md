# Phase 1a — Per-stop baseline terms in every marked design

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Why: the contract-1 corpus showed reward, switch, novelty and previous-arm "effects" that are as large 200 ms before a replay as after it. A design with one intercept per session cannot tell a context's baseline rate from a replay response. This phase adds unpenalised, session-wide well and reward-outcome columns so the event kernels are estimated against the right baseline. Code only; no refit here (that is phase 1c).

**Inputs to read first:**

- [src/continuum_swr_replay/analysis/pfc/marked_glm.py:145-190](../../../../src/continuum_swr_replay/analysis/pfc/marked_glm.py#L145) — `KernelDesign` and its `__post_init__`; the contiguity check assumes the first block starts at column 1.
- [src/continuum_swr_replay/analysis/pfc/marked_glm.py:273-437](../../../../src/continuum_swr_replay/analysis/pfc/marked_glm.py#L273) — `MarkedSessionContext` and `prepare_marked_session_context`; the fit mask is built at `:381` from `cohort_stops`, spikes are binned at `:413`.
- [src/continuum_swr_replay/analysis/pfc/marked_glm.py:917-1003](../../../../src/continuum_swr_replay/analysis/pfc/marked_glm.py#L917) — `build_kernel_design` (intercept column at `:954`) and `_without_block` (`:983`, slices restart at 1).
- [src/continuum_swr_replay/analysis/pfc/marked_glm.py:1006-1130](../../../../src/continuum_swr_replay/analysis/pfc/marked_glm.py#L1006) — `build_marked_model_ladder` and the single-mark `fit_marked_kernel_ladder`, which call `make_ridge_penalty(width, sigma)` with the default one unpenalised column.
- [src/continuum_swr_replay/analysis/pfc/marked_model_family.py:1063](../../../../src/continuum_swr_replay/analysis/pfc/marked_model_family.py#L1063) — the family's `build_kernel_design` call; [`:1135-1160`](../../../../src/continuum_swr_replay/analysis/pfc/marked_model_family.py#L1135) `family_ridge_penalty`.
- [src/continuum_swr_replay/analysis/pfc/batched_glm.py:605](../../../../src/continuum_swr_replay/analysis/pfc/batched_glm.py#L605) — `make_ridge_penalty(n_features, sigma, *, n_unpenalized=1)`; the `n_unpenalized` keyword already exists.
- [src/continuum_swr_replay/analysis/pfc/cohort.py:32](../../../../src/continuum_swr_replay/analysis/pfc/cohort.py#L32) — `load_session_stop_cohort`: the stops carry `stop_start`, `stop_end` (seconds on the position clock), `occupancy_well_id` (0–5 or NaN), `is_rewarded_outcome` (0/1/NaN).
- [scripts/pfc/run_pfc_marked_model_family.py:109-260](../../../../scripts/pfc/run_pfc_marked_model_family.py#L109) — `BUNDLE_CONTRACT_VERSION` and `build_fit_contract`; [`:594-625`](../../../../scripts/pfc/run_pfc_marked_model_family.py#L594) where the context is prepared and checked.

**Contracts referenced:**

- [Baseline columns](shared-contracts.md#baseline-columns) — names, order, reference well, constant-column rule; do not weaken.
- [Fit contract version 2](shared-contracts.md#fit-contract-version-2) — the keys this phase adds.

## Tasks

- **Baseline builder in `marked_glm.py`.** Add `build_baseline_columns(fit_mask, cohort_stops, *, reference_well=0)` returning `(baseline, names, report)` with `baseline` shape `(n_fit_bins, n_baseline)`:

  ```python
  def build_baseline_columns(
      fit_mask: pd.Series,
      cohort_stops: pd.DataFrame,
      *,
      reference_well: int = BASELINE_REFERENCE_WELL,
  ) -> tuple[NDArray, tuple[str, ...], dict[str, Any]]:
      time = fit_mask.index.to_numpy(dtype=float)
      rows = fit_mask.to_numpy(dtype=bool)
      order = np.argsort(cohort_stops["stop_start"].to_numpy(dtype=float))
      starts = cohort_stops["stop_start"].to_numpy(dtype=float)[order]
      ends = cohort_stops["stop_end"].to_numpy(dtype=float)[order]
      wells = cohort_stops["occupancy_well_id"].to_numpy(dtype=float)[order]
      rewarded = cohort_stops["is_rewarded_outcome"].to_numpy(dtype=float)[order]
      stop_of_bin = np.searchsorted(starts, time[rows], side="right") - 1
      inside = (stop_of_bin >= 0) & (time[rows] <= ends[np.clip(stop_of_bin, 0, None)])
      if not inside.all():
          raise ValueError(f"{int((~inside).sum())} fitted bins lie outside every cohort stop")
      well_of_bin = wells[stop_of_bin]
      reward_of_bin = rewarded[stop_of_bin]
      kept = sorted({int(w) for w in well_of_bin[np.isfinite(well_of_bin)]})
      reference = reference_well if reference_well in kept else (kept[0] if kept else reference_well)
      columns, names = [], []
      for well in kept:
          if well == reference:
              continue
          columns.append((well_of_bin == well).astype(float)); names.append(f"well_{well}")
      columns.append(np.where(np.isfinite(reward_of_bin), reward_of_bin, 0.0)); names.append("rewarded_outcome")
      columns.append((~np.isfinite(reward_of_bin)).astype(float)); names.append("outcome_unknown")
      baseline = np.column_stack(columns)
      constant = np.all(baseline == baseline[:1], axis=0)   # all-zero or all-one on fitted rows
      report = {
          "baseline_reference_well": int(reference),
          "baseline_dropped_constant": [n for n, c in zip(names, constant, strict=True) if c],
          "n_bins_without_well": int((~np.isfinite(well_of_bin)).sum()),
      }
      keep = ~constant
      return baseline[:, keep], tuple(n for n, k in zip(names, keep, strict=True) if k), report
  ```

  Every fitted bin must fall inside a cohort stop (the fit mask is built from those stops at `marked_glm.py:381`); a violation raises rather than silently zero-fills. NumPy docstring with shapes.

- **`KernelDesign` gains `n_baseline: int = 0`.** `__post_init__` (`marked_glm.py:166-189`) requires the first block to start at `1 + n_baseline` and the docstring says columns `1..n_baseline` are the unpenalised baseline. `build_kernel_design(..., baseline: NDArray | None = None)` concatenates `[ones, baseline, blocks...]` and sets `n_baseline`; it raises if `baseline.shape[0] != rows.sum()`. `_without_block` starts its recomputed slices at `1 + design.n_baseline` and carries `n_baseline` through.

- **`MarkedSessionContext` gains `baseline` and `baseline_names` and `baseline_report`.** `prepare_marked_session_context` calls the builder after the fit mask at `:381` and stores all three; a new keyword `baseline_terms: str = "well_and_reward_outcome"` accepts `"none"` and then stores an empty `(n_fit_bins, 0)` array. `build_marked_model_ladder` (`:1006`) and `build_marked_family_ladder` (`marked_model_family.py:980`) take `baseline: NDArray | None` and pass it to `build_kernel_design`.

- **Penalties follow the baseline width.** `family_ridge_penalty` (`marked_model_family.py:1135`) and the single-mark `_score_designs` / `fit_marked_kernel_ladder` (`marked_glm.py:1120-1132` and `:1423`) pass `n_unpenalized=1 + design.n_baseline`. `permuted_mark_blocks` and `_mark_block_geometry` are unaffected (they address blocks by slice).

- **Runners.** Both `scripts/pfc/run_pfc_marked_model_family.py` and `scripts/pfc/run_pfc_marked_kernels.py`: `BUNDLE_CONTRACT_VERSION = 2`; `--no-baseline` flag mapping to `baseline_terms="none"`; `build_fit_contract` gains `baseline_terms`; the sidecar `extra` gains `baseline_column_names`, `baseline_reference_well`, `baseline_dropped_constant`, `n_bins_without_well` (from the context's report). The design-width and rank diagnostics already recorded will change by `n_baseline`; nothing else in the sidecar moves.

- **Docs.** `docs/design/pfc_marked_model_family.md` "Shared temporal estimand": one paragraph stating the baseline columns, the reference well, and that the typical-event kernel is now the response at the reference well after an unrewarded known outcome. `docs/design/pfc_marked_event_kernels.md` gets the same sentence for the single-mark model. `scripts/README.md`: the `--no-baseline` flag on both runners.

## Deliberately not in this phase

- No refit of any bundle; phase 1c owns the batch and the renaming of the contract-1 tree.
- No new marks and no spec changes (phase 1b), so this PR's diff against the contract-1 numbers is attributable to the baseline alone.
- No per-stop random intercepts or a within-stop time term; see overview open question 1.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_baseline_columns_map_fitted_bins_to_their_stop` | three stops at wells 0, 2, 2 with outcomes 1, 0, NaN; the well_2 column is 1 exactly on the second and third stops' fitted bins, `rewarded_outcome` is 1 only on the first, `outcome_unknown` only on the third; no `well_0` column (reference). |
| `test_baseline_reference_falls_back_when_well_0_is_absent` | stops at wells 3 and 4 only: names are `("well_4", ...)` and the report says reference 3. |
| `test_baseline_drops_constant_columns_and_reports_them` | all outcomes known and rewarded: `rewarded_outcome` (all one) and `outcome_unknown` (all zero) are dropped and named in `baseline_dropped_constant`. |
| `test_baseline_refuses_a_fitted_bin_outside_every_stop` | a fit mask with one bin past the last `stop_end` raises `ValueError` naming the count. |
| `test_kernel_design_places_first_block_after_the_baseline` | `build_kernel_design` with a 3-column baseline: `block_slices` start at 4, `n_baseline == 3`, and `__post_init__` rejects a hand-built design whose first block starts at 1 with `n_baseline == 3`. |
| `test_without_block_keeps_the_baseline_columns` | the reduced design's first `1 + n_baseline` columns equal the full design's, and its slices restart at `1 + n_baseline`. |
| `test_penalty_leaves_baseline_columns_free` | `family_ridge_penalty` and the single-mark penalty are exactly zero on columns `0..n_baseline` and `1/σ²` elsewhere. |
| `test_empty_baseline_reproduces_the_contract_one_fit` | with `baseline_terms="none"` the existing single-target equivalence test (`tests/analysis/pfc/test_marked_model_family.py`, "numerically identical to single mark estimator") still passes bit for bit. |
| `test_planted_kernel_is_recovered_with_a_context_offset_present` (`slow`) | extend the two-target recovery test: add a per-stop rate offset of ×1.5 on rewarded stops to the generator; with the baseline present the recovered planted peak is still 0.9 ± 0.35 **and** the reward-context mark's pre-onset mean is within ±0.01 of zero; without the baseline it is not. |
| `test_runner_contract_records_baseline_names_and_version_two` | sidecar round trip: `bundle_contract_version == 2`, `baseline_terms`, `baseline_column_names` present; a contract-1 sidecar is stale. |

## Fixtures

Synthetic: a 600-bin fit mask over three or four stops with known wells and outcomes, built in the test module (extend the `_eligibility_inputs` helper in `tests/analysis/pfc/test_marked_model_family.py:554`). The recovery test reuses its existing generator and adds the stop offset. No real data in this phase.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind).
- User-facing documentation listed as tasks is updated, not deferred.
