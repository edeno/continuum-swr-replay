# Phase 8 — The mPFC unit models on the convention; Figure 5 redrawn

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#jax-batch-twin)

Branch: `mpfc-convention` from master after Phases 2 and 3. The one GPU family. Independent of the
Figure 4 chain (5a to 7, 9, 10).

**Inputs to read first:**

- The design, sections 4.3 to 4.5, 5 ("the mPFC unit models are the GPU case"), 6 (check 6), and section
  7's row for the mPFC models (what changes: the penalty by REML shared over the units of a session, trial
  folds, the shared convergence and support rules, population statements reduced session to day to animal;
  what stays: the peri-event design, the matched-control draws and the per-unit test).
- `src/continuum_swr_replay/analysis/pfc/batched_glm.py:129` (`_in_float64`), `:156-185` (constants,
  `DEFAULT_PENALTY_SIGMA` `:167`), `:460` (`_sweep`), `:677` (`make_ridge_penalty`), `:1190`
  (`_full_fit_batch`), `:1422` (`fit_poisson_designs`).
- `src/continuum_swr_replay/analysis/pfc/fit.py:72` (`fit_models_batched`), `:112` (`model_penalties`),
  `:207` (`_fit_immobility_models`); `design.py:159` (`event_basis_functions`), `:273`
  (`_make_conv_spline_basis`); `session.py:79` (`process_neuron`), `:821` (`run_session`), `:1449`
  (`_session_events`), `:1632` (`_fit_session_neurons`); `cohort.py:65` (`load_session_stop_cohort`);
  `canonical_events.py:95` (`load_session_canonical_events`).
- Folds and the null: `heldout.py:222` (`assign_blocked_folds`, its `support_radius_bins`), `:625`
  (`control_draw_null`); callers `scoring.py:190`, `scripts/pfc/run_pfc_marked_kernels.py:560`,
  `scripts/pfc/run_pfc_marked_model_family.py:894`.
- Marked kernels and stop context: `marked_glm.py:1409-1422` (ridge penalties), `marked_model_family.py`,
  `stop_context.py:977` (`fit_stop_context_glm`), `:1193` (`summarize_stop_context`).
- Population: `population_statistics.py` (`called_fraction_by_animal` `:305`, `bh_called_fraction` `:373`,
  `session_value_statistics` `:519`); `reported/mpfc/` (`inputs.py`, `population.py`, `timing.py`);
  `scripts/figures/generate_figure05.py:533` (`generate_figure`), `:662` (`load_population_data`).
- `scripts/pfc/run_pfc_analysis_bandit.py` (usage block: `--gpu`, `--sessions`, `--n_workers`,
  `--penalty-sigma`, `--max-sessions`, `--output-dir`).
- Branch `swr-trial-folds` (on master by then): `fold_training_masks` and `margin_in_bins` in
  `analysis/event_evidence/tetrode_holdout.py` (a margin that keeps training bins away from a held-out
  piece).
- References: [designs.md#jax-batch-twin](designs.md#jax-batch-twin) and the appendix.

**Contracts referenced:** [C4](shared-contracts.md#c4-the-selection-function-is-the-only-filter) (Figure 5:
in-visit bins in epochs with mPFC units), [C5](shared-contracts.md#c5-the-convention),
[C7](shared-contracts.md#c7-the-fit-row-record), [C11](shared-contracts.md#c11-gpus-long-runs-and-the-other-session).

## Tasks

- **Baseline**: save `paper/reported/mpfc.json`, Figure 5's files, and three sessions' current bundles
  (one per animal with mPFC units) fit with the current code into a scratch output directory
  (`--max-sessions`, `--output-dir`), with their wall time per GPU.
- **JAX batch twin** (`analysis/regression/jax_poisson.py`, new): Poisson penalized Newton vmapped over
  units, the pooled REML over a session's units with one strength per smooth term (the sum of the units'
  scores, as `_glm_numpy.py:408` sums), exact gradient and Hessian as the NumPy reference. float64 on CPU
  for tests; float32 on GPU with `jax.default_matmul_precision("highest")` for check 6. Chunk units as
  `batched_glm.py:425` (`_chunk_rows`) does to bound memory.
- **Check 6**: on three real sessions, GPU float32 against CPU float64 (the NumPy reference): selected log
  penalties within 1e-3, coefficients within 1e-4 relative, deviance within 1e-6 relative; a test version on
  synthetic units runs on CPU in both precisions.
- **Penalty by REML**: replace the fixed `sigma` (`model_penalties`, `fit.py:112`; `DEFAULT_PENALTY_SIGMA`)
  with the REML-selected penalty per design and session, shared over the session's units. The penalty on the
  convolved B-spline kernel weights becomes the convention's second-order difference penalty (call: the
  design says the same penalty rule as every family; a ridge on spline weights is not a smoothness penalty);
  the intercept stays free. Keep `--penalty-sigma` only as a fixed-penalty sensitivity path. Apply the same
  to the marked kernels (`marked_glm.py:1409-1422`).
  WAITS FOR THE LEAD (smooth-term rule): here REML pools over the units of one session, not over sessions,
  so the kernels' penalty is still chosen from the data within a session. Whether the per-session
  smooth-term rule (no penalty chosen from the data for smooth terms with few events) applies to these
  kernels, or REML over a session's many units stays, is part of the decision.
- **Fit mask on the common subset**: replace the stop-cohort mask (`cohort.py:65`) with in-visit immobile
  bins (the visit intervals of Phase 3, selected by the one function, C4) and events selected by
  `select_visit_context` in `canonical_events.py:95` (the default set: events that start inside a visit,
  `spans_visit_end` included, C3); record counts per label in each bundle's sidecar. The mPFC models use the
  same unit, event set, clocks and fitter as every other family (C14).
- **Trial folds**: replace `assign_blocked_folds` in `heldout.py`, `scoring.py:190` and the two runners
  with Phase 2's fold function on the bins' times, plus a training margin of the kernel half-width (250 ms)
  around each held-out piece (as `fold_training_masks` on master does), so a symmetric kernel cannot see
  held-out bins. The matched-control draws and the per-unit empirical p keep their definitions; only the
  folds they use change.
- **Shared convergence and support**: the Newton decrement rule replaces the absolute-change rule of the
  batched IRLS; a unit with too few spikes on each side of an event kind (support rule) is recorded, not
  fit for that term.
- **Population statements**: each population number (called fractions, best-model fractions, peak
  modulation) becomes a per-session value reduced session to day to animal with `estimand_summary`
  (`population_statistics.py` functions take the reduction; the BH reference stays as a reference).
- **Stop context** (`stop_context.py`): onto the in-visit cohort and the shared fitter; its day-clustered
  sandwich is replaced by the session to day to animal reduction (call; it is the design's rule for every
  family).
- **Measure, then run** (C11): three sessions on one free GPU (read `nvidia-smi` on zephyr and breeze first;
  message the other session); record GPU-hours per session; extrapolate to every epoch with mPFC units (count them from the epoch register; the last
  full fit took about 14 GPU-hours with the fixed penalty; REML adds outer iterations: expect a multiple,
  measure it). Present the estimate and the free cards to the lead before the full run; run it partitioned
  over the free cards (`--gpu`, `--sessions`) in `tmux` (`csr-8-mpfc`).
- **Rebuild and redraw**: `summarize.py --summary mpfc`, emit, `generate_figure05.py`; commit the figure
  files.
- **Docs**: `docs/design/pfc_unit_glm_estimator.md`, `pfc_marked_event_kernels.md` and `pfc_marked_model_family.md`
  rewritten for the new estimator (the fixed sigma and blocked folds they describe are removed): REML over units, trial folds,
  the in-visit cohort; remove statements about the fixed sigma 3.0 except as the sensitivity path.

- **Retire what this phase replaced** (C13; the items are the REMOVE rows of
  [legacy-inventory.md#phase-8](legacy-inventory.md#phase-8)). In this PR, or in a second PR of this phase
  that merges right after it:
  1. Write the old-against-new comparison for the lead: each old value the section lists, read from the
     copy of the summary saved before the rebuild, beside its replacement from the rebuilt summary, with the
     reason for each change. Put it in the PR description.
  2. List the macros `paper/*.tex` cites from those old values (none of `behavior.json` or `mpfc.json` was
     cited on 2026-11-20; check again). Ask the lead about any that are; remove a cited value only after
     the yes.
  3. First commit fixtures of the batched ridge fitter's coefficients (`batched_glm.fit_poisson_glms` at sigma 3 on the synthetic units of `tests/analysis/pfc/test_batched_glm.py`) for any test of the new fitter that compares with it. Remove every REMOVE item of the section: modules, specs, runners, `RUNS` entries, plotters,
     value groups and their summary wiring, question-list entries (mark the replaced 3.x questions
     `dropped`, reason "superseded by" the new id), tests and doc sections.
  4. Check: `git grep -n -w <name> -- src scripts tests docs` gives no match for any removed symbol, file or
     column (dated records exempt, C13); `scripts/checks/write_pipeline_map.py` regenerated; one row per
     removal in `docs/history/2026-module-moves.md`; the full suite passes.
  The ASK rows of the section are not removed here; list them in the PR for the lead.

## Deliberately not in this phase

- Changing the peri-event basis, the matched-control draw rule or the per-unit test (they keep their
  definitions).
- Any hippocampal model.
- A pooled model across sessions (stage 3).

## Validation slice

| Test | Asserts |
| --- | --- |
| `tests/analysis/regression/test_jax_poisson.py::test_matches_numpy_reference` | float64 JAX equals the NumPy fitter and its REML selection on synthetic units to 1e-8 relative |
| `test_jax_poisson.py::test_float32_close_to_float64` | check 6 on synthetic units: float32 within the stated tolerances of float64 (CPU) |
| `test_jax_poisson.py::test_pooled_reml_over_units_invariant_to_duplication` | duplicating units leaves the penalty unchanged |
| `tests/analysis/pfc/test_heldout.py` (updated) | trial folds with the margin: no training bin within 250 ms of a held-out bin; every bin in one fold |
| `tests/analysis/pfc/test_glm_pipeline.py` (updated) | a planted peri-event kernel is recovered with the REML penalty; a unit with no modulation is called at the nominal rate against matched controls |
| `tests/analysis/pfc/test_population_statistics.py` (updated) | the reduced statistics on a hand table |
| check 6 on real sessions (script, GPU) | the tolerances above on three sessions |

## Fixtures

Synthetic units and events (the existing `tests/analysis/pfc/` helpers, for example those of
`test_marked_planted.py`); three real sessions for check 6 and the smoke run.

## Verification and runtime

- Simulated first (tests), then three real sessions (baseline against new: per-unit called status, curves,
  deviance), then all sessions.
- Compare: the fraction of units called per class per animal before and after; the REML penalty against the
  old `sigma = 3.0` (state the implied prior SD, `batched_glm.py:740`, `implied_prior_sd`); explain each large
  change.

## Rebuilt

The mPFC bundles (new directory or a new suffix; the old ones stay on disk, unread); `paper/reported/mpfc.json`
and the emitted files; `paper/figures/figure05/`.

## Exit criteria

Check 6 passes on three sessions; the full run is done on free cards with the other session told; the
mPFC summary and Figure 5 are rebuilt on the in-visit cohort with REML penalties and trial folds.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind).
- User-facing documentation listed as tasks is updated, not deferred.
