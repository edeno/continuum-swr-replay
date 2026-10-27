# Phase 3 — Shuffled-mark reference for the population slope kernels

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Why: the quantity the summary quotes is the population mean one-SD slope kernel, and its strongest support so far is "same sign in 5 of 5 animals". A 1 % slope needs a reference: what the same estimator gives when the mark carries no information. The reference is cheap if it refits only the full-data fit (no held-out scorer): about 25 s per draw per model on a large session.

**Inputs to read first:**

- [src/continuum_swr_replay/analysis/pfc/marked_model_family.py:1229-1420](../../../../src/continuum_swr_replay/analysis/pfc/marked_model_family.py#L1229) — `fit_marked_family_ladder`: the full-data `fit_poisson_glms` call and how `mark_curves[class][target]` are read from `fitted.params` by block slice.
- [src/continuum_swr_replay/analysis/pfc/marked_glm.py:1137-1272](../../../../src/continuum_swr_replay/analysis/pfc/marked_glm.py#L1137) — `permuted_mark_blocks`, `_mark_block_geometry`, `within_available_mark_permutations`: the batched block rebuild used by the single-mark label shuffle; typed for `MarkedModelLadder` but only reads `events[class]`, `fit_mask`, `spline_basis`, `lags` and a z column.
- [src/continuum_swr_replay/analysis/pfc/batched_glm.py:1227](../../../../src/continuum_swr_replay/analysis/pfc/batched_glm.py#L1227) — `fit_poisson_glms(design, counts, penalty, compute_covariance)`: one design, all units.
- [scripts/pfc/summarize_pfc_marked_family.py:181-245](../../../../scripts/pfc/summarize_pfc_marked_family.py#L181) — `population_curve_summary`.
- `.claude/docs/reviews/2026-09-18-pfc-marked-family-smoke.md` — the σ_mark scan; the reference must use the same prior as the observed fit.

**Contracts referenced:**

- [Fit contract version 2](shared-contracts.md#fit-contract-version-2) — `slope_permutations`, `slope_permutation_seed`.
- [Bundle directory suffixes](shared-contracts.md#bundle-directory-suffixes) — `.slope_null.npy` companions.

## Tasks

- **Generalise the block rebuild.** Move the geometry/permutation helpers' dependence on `MarkedModelLadder` to explicit arguments: `permuted_mark_blocks(events, fit_mask, spline_basis, lags, mark_z_column, permutations, *, geometry=None)`. Keep the single-mark call sites working (they pass `ladder.events[c]`, etc.). Existing tests for these helpers keep passing unchanged in behaviour.

- **Draws in `fit_marked_family_ladder`.** New keywords `slope_permutations: int = 0`, `slope_permutation_seed: int = 0`, `slope_permutation_targets: tuple[str, ...] | None = None` (default: every target). After the observed full-data fit, for each class and target in scope:

  ```python
  rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(n_class_target)[k])
  available = events[f"{target}_available"].to_numpy(bool)
  null = np.full((R, n_lags, n_units), np.nan, dtype=np.float32)
  for start in range(0, R, batch):
      perms = within_available_mark_permutations(rng, available, min(batch, R - start))
      blocks = permuted_mark_blocks(events, fit_mask, basis, lags, f"{target}_z", perms)
      for i, block in enumerate(blocks):
          design = full.matrix.copy(); design[:, slice_of(class, target)] = block
          fit = fit_poisson_glms(design, counts, penalty=full_penalty, compute_covariance=False)
          params = np.asarray(fit.params)[:, slice_of(class, target)]
          params[~np.asarray(fit.converged)] = np.nan
          null[start + i] = (basis @ params.T)[report]   # report window only
  ```

  Only the permuted class's block changes; the other class's marks stay observed. Results go in `MarkedFamilyResults.slope_null[class][target]` with shape `(R, n_report_lags, n_units)` and the count of converged draws per unit in diagnostics.

- **Runner.** `--slope-permutations R` (default 0), `--slope-permutation-seed` (default: `session_seed("marked_family_slope_null", ...)`), `--slope-permutation-models` (default `spatial_role belief_content trajectory_geometry`). Each null array is written as `<artifact>.<class>.<target>.slope_null.npy` (float32) and listed in `family_companion_paths` when R > 0, so the skip check requires it. Contract keys as in shared-contracts.

- **Summary.** `load_bundles` reads the null arrays when present. For each (model, class, target): per draw index r, the population null curve is the mean over sessions of the session mean over units of `null[r]`, giving `(R, n_lags)`; report the 2.5th and 97.5th per-lag percentiles as a band, whether the observed population curve's peak lies outside the band at its peak lag, and the same per animal. Per unit, the fraction whose observed |peak| exceeds the 95th percentile of its own null |peak| (descriptive). `plot_model` draws the band in grey behind the observed curve.

- **Run** with R = 30 on the three default models over the corpus (about 15 GPU-hours on three cards), then regenerate the summary and add a "Reference" column to the corpus-v2 audit's reading section: for each event-locked effect named there, inside or outside the band.

- **Docs.** `docs/design/pfc_marked_model_family.md`: a "Slope reference" subsection stating what is permuted (the target within class among events where it is defined), what is refit (full data only), what the band is, and that it is a reference for the population curve, not a per-unit test.

## Deliberately not in this phase

- No held-out refit per draw (a null for the delta); that is the single-mark runner's job.
- No permutation for the marginal tier in the first run; the machinery works on any spec and can be pointed at `--tier marginal` later.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_permuted_block_rebuild_is_ladder_agnostic` | the generalised `permuted_mark_blocks` reproduces the single-mark test's batched-vs-scalar equality on the boundary fixture. |
| `test_slope_null_is_seeded_and_only_permutes_the_named_target` | two runs with the same seed give identical null arrays; the other class's block and every adjustment column in the permuted design equal the observed design. |
| `test_slope_null_recovers_zero_for_a_planted_kernel` (`slow`) | on the two-target recovery fixture with R = 8: the planted target's null curves have mean within ±0.15 of zero at the planted peak lag while the observed curve is 0.9 ± 0.35; the unplanted target's observed curve lies inside its null band. |
| `test_runner_writes_null_companions_and_requires_them` | with R = 2, the `.slope_null.npy` files exist for the named models only, are float32 with shape `(2, 201, n_units)`, and deleting one makes the bundle stale. |
| `test_summary_band_and_peak_call` | on synthetic nulls with a known spread the band percentiles and the inside/outside call are right. |
| corpus run (`slow`, GPU) | 71 sessions × 3 models × R = 30, one contract. |

## Fixtures

The two-target recovery generator in `tests/analysis/pfc/test_marked_model_family.py` (reused), synthetic null arrays in `tests/unit/test_summarize_pfc_marked_family.py`.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind).
- User-facing documentation listed as tasks is updated, not deferred.
