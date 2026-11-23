# Phase 2c — One JAX fitter for every model family

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [JAX fitter design](jax-fitter-design.md)

Branches: one per sub-phase below (`regression-jax-0` to `regression-jax-6`), each from master after the
one before. Phase 2b first (the author's call; the lead did not object). The port runs in parallel with
[Phase 5a](phase-5a-rate-models.md), on its own branches, and does not block Figure 4 (design,
twenty-third round). At its end the port replaces the solver, and every fit is rerun once to confirm
equality.

**Status:** not started. The full port design is [jax-fitter-design.md](jax-fitter-design.md) (a copy of
`.claude/research/2026-11-20-regression-prototypes/jax_fitter/DESIGN.md`; read all of it). This file lists
the work in the plan's form; the design holds the reasons, the measurements and the code references.

**Decided by the lead:**

- "I would prefer a jax fitter for all models" (2026-11-23, design, twentieth round). One fitter, in JAX,
  for the rate, content, stay or leave, ripple power and mPFC models. The JAX twin for the mPFC models
  only is dropped. By the lead's rule on legacy code (design, ninth round; C13), the NumPy solver is
  removed when the JAX one replaces it.
- Precision (2026-11-23, 12:32, "sure"): float64 on the CPU and on the GPU, turned on for each public
  call of the fitter (a scoped `jax.enable_x64(True)`), never for the whole process. No single-precision
  path. The GPU check is GPU float64 against CPU float64.
- The acceptance rule (2026-11-23, 12:32, "ok"): the JAX fitter matches the stored NumPy outputs within
  1e-9 relative, with every status, column left out and count exactly equal. Bit-for-bit equality is not
  required (XLA's products cannot give it). The tolerances per output are those of the port design's
  section 9. (**call**, twenty-third round) The rule has an absolute floor: a value `b` matches when the
  difference is at most `1e-11 + 1e-9 |b|` (the port design's `beta` tolerance, `rtol=1e-9, atol=1e-11`),
  with the statuses and counts exactly equal. The prototype had one coefficient near zero (about 4e-4) at
  a relative difference of 5.4e-9 ([jax-fitter-timing.md](jax-fitter-timing.md)). The author told the
  lead, and the lead did not object.

**Entry condition: met.** The lead, 12:34: "I think I would in general prefer one fitter, but if there's a
good reason not to, then that's fine". The whole workloads were timed with the present fitter and a JAX
prototype, under load ([jax-fitter-timing.md](jax-fitter-timing.md)): per whole rate fit 0.349 s
(NumPy), 0.727 s (JAX on CPU), 0.102 s (one A100). The lead asked "what do you recommend?"; the author
recommended that the port go ahead without blocking Figure 4, in the order of the overview's Status; the
lead, 13:21: "ok" (design, twenty-third round). The open question on the port's CPU speed is closed.

**Reference outputs are captured after the weak-penalty change.** The NumPy fixtures of sub-phase 0
(task 2) are written from a fitter that has the weak penalty on simple terms and the support rule on the
design (branch `weak-penalty`, aa8f8425), so that the port is checked against the rules it will run.

**Inputs to read first:**

- [jax-fitter-design.md](jax-fitter-design.md), all of it; sections 0, 2 to 5, 8 to 10, 12 and 13 are the
  work.
- `src/continuum_swr_replay/analysis/regression/` (`likelihoods.py`, `newton.py`, `reml.py`,
  `sessions.py`, `calibration.py`, `collinearity.py`, `constants.py`) and `docs/design/regression_fitter.md`
  ("Cost").
- `src/continuum_swr_replay/analysis/pfc/batched_glm.py:129` (`_in_float64`, the scoped float64 pattern),
  `:425` (`_chunk_rows`), `:460` (`_sweep`).
- `src/continuum_swr_replay/runs/device.py` (`default_jax_platforms`, `configure_jax_gpu`,
  `REQUIRED_XLA_FLAGS`) and `runs/threads.py` (`pin_blas_threads`).
- The `jax` skill, and the installed JAX version's own documentation (JAX 0.8.0 in the project venv on
  2026-11-23); do not rely on memory.

**Contracts referenced:** [C1](shared-contracts.md#c1-running-code-and-tests), [C11](shared-contracts.md#c11-gpus-long-runs-and-the-other-session),
[C12](shared-contracts.md#c12-layering-of-the-new-package), [C13](shared-contracts.md#c13-no-legacy-left-behind),
[C14](shared-contracts.md#c14-one-of-everything).

## Tasks

The sub-phases are the port design's Phases 0 and 2 to 6. Its Phase 1 (the simulation engine) is
[Phase 2b](phase-2b-simulation-engine.md). Each sub-phase is one PR, merged and tested on its own.

- **Sub-phase 0: Measure whole workloads, then capture behavior** (the entry condition, then the port design's
   Phase 0; no change to results).
   1. Done (2026-11-23): the whole workloads were timed with the present fitter and a JAX prototype, and
      the lead answered (above; [jax-fitter-timing.md](jax-fitter-timing.md)). Copy the table into the
      sub-phase's PR.
   2. Write `scripts/checks/write_regression_reference.py` and commit the fixtures of the port design's
      section 9 (`tests/analysis/regression/data/numpy_reference/*.json`, each with the commit and the
      command), with the fixed-seed fixture of Phase 2b. Capture them after the weak-penalty change is
      merged (above), and include a design with a penalized slope, a penalized slope by a category and a
      penalized category.
   3. Add `tests/analysis/regression/test_reference_fixtures.py`, which compares the fitter with the
      fixtures at the section 9 tolerances. It passes on NumPy.
   4. Write `scripts/checks/time_regression_fitter.py` (the port design's section 10) and record its
      numbers in `docs/design/regression_fitter.md`, "Cost".
   5. Run the real slice (three sessions, one each of chimi, j16 and wilbur, rate and content at their
      declared penalties) and keep its output in a scratch folder.
- **Sub-phase 2: JAX plumbing** (no change to results): the `in_float64` decorator, the trace-time dtype guard, the
   device guard, row and group buckets with masks, the XLA thread flags in `runs/threads.py`; every
   fitting runner calls `default_jax_platforms("cpu")` unless it asks for a GPU, and uses `spawn` pools.
- **Sub-phase 3: The fit at fixed penalties in JAX**: the likelihood kernels of the four families with their
   derivatives and the softmax segment sums; the batch-first Newton control (host loop) with the
   iteration and "evaluate at beta" kernels; the fixed-penalty summary (score, `A^-1`, edf, `log|A|`), the
   profiled Gaussian, and `collinearity`'s public weights (`lik.weighted_design`). Remove the NumPy
   formulas these replace (the likelihood bodies, `_newton_system`, `_objective`, `_covariance_and_edf`,
   the score in `_reml_in_basis`). Run the padding tests (`tests/analysis/regression/test_padding.py`)
   again against these kernels, the guard for a softmax group with no rows included; every quantity
   that depends on the number of rows uses `n_real`, not the padded count.
- **Sub-phase 4: The REML derivatives in JAX**: one jitted kernel per family for the gradient and Hessian in the log
   penalties; the `cho_solve` loops of `_reml_in_basis` and `_gaussian_profiled` removed. The outer
   search stays in Python.
- **Sub-phase 5: The GPU check**: `scripts/checks/check_regression_gpu.py` (new) on one free A100, with the
   deterministic XLA flags: GPU float64 against CPU float64 within the section 9 tolerances, every
   discrete output exact, two GPU runs bitwise equal. Write the device policy into
   `docs/design/regression_fitter.md`.
- **Sub-phase 6: Remove the rest of the NumPy solver**: every remaining NumPy numerical path in `likelihoods.py`,
   `newton.py` and `reml.py`; a repository test that no `scipy.linalg`, `np.linalg.cholesky`,
   `np.linalg.solve` or `np.add.reduceat` is left in those three modules; "Cost", the precision section
   and the module table of `docs/design/regression_fitter.md` rewritten; one row per removal in
   `docs/history/2026-module-moves.md`. Whether `write_regression_reference.py` is kept (frozen, with its
   commit) or removed with a README note is a question for the lead at this sub-phase (the author's
   default: keep it).

The public interface does not change (the port design's section 2): the signatures of `fit_sessions`,
`log_penalty_for_edf`, `refit_session`, `select_penalties`, `fit_at_penalties`, `pooled_reml`,
`reml_and_derivatives` and `penalized_newton`, and every returned array a NumPy float64 array.

The mPFC entry point (the units of one session batched on a shared design, and a pooled REML over those
units if [Phase 8](phase-8-mpfc.md) keeps it) is built in Phase 8, on this fitter's kernels.

## Tests that gate it

| Sub-phase | Gate |
| --- | --- |
| 0 | the full regression suite; `test_reference_fixtures.py` passes on NumPy, on fixtures captured after the weak-penalty change |
| 2 | the float64 scope tests (the flag unchanged after a fit in a thread; arrays still float32 in the caller; a raw kernel with the flag off raises; the same coefficients with the caller's flag on or off), the device guard test, the padding tests (padded equals unpadded to 1e-12 for each family, a softmax with an empty dump group included), a repository test that no script calls `get_context("fork")` |
| 3 | `test_reference_fixtures.py` with exact discrete fields; all of `tests/analysis/regression`; `test_likelihoods.py::test_hand_derivatives_equal_autodiff` (1e-12 relative); a compile-count test (10 row counts give at most the number of buckets); a test that a patched rule constant (`MAX_STEP_HALVINGS`) takes effect after the first compile; the Phase 2b fixed-seed fixture; the real slice in the PR; benchmarks 1 to 3 and 5 |
| 4 | the mgcv and neurospatial fixtures; the finite-difference tests; the REML fixtures; `test_simulation.py` (slow); benchmark 4 on CPU |
| 5 | GPU against CPU within the section 9 tolerances; two GPU runs bitwise equal; benchmark 4 on the GPU |
| 6 | the full suite; the `git grep` gate (C13); the fixture tests; the repository test of the removed calls |

## Deliberately not in this phase

- Any change to a rule, a status or a result of the fitter. A discrete output that differs is examined and
  gets a written exception in its fixture; the tolerance is not widened.
- The mPFC batched entry point and its pooled REML over units (Phase 8).
- The simulation engine (Phase 2b).
- A `lax.while_loop` form of the whole fit. It is built only if the mPFC benchmark shows the per-step
  sync costing more than 10 % on a GPU (port design, section 5).
- A float32 path (dropped by the lead, 2026-11-23).

## Verification and runtime

Each sub-phase records `time_regression_fitter.py`'s numbers in its PR. No speed claim is made without
them (the port design's section 10). On 2026-11-23, under load: one whole rate fit took 0.349 s in NumPy,
0.727 s in JAX on one CPU core and 0.102 s on one A100; compile 15.5 s per process
([jax-fitter-timing.md](jax-fitter-timing.md)).

## Rebuilt

The fixtures under `tests/analysis/regression/data/numpy_reference/`; "Cost" in
`docs/design/regression_fitter.md`. No stored fit and nothing in `paper/reported/`.

## Exit criteria

Every sub-phase's gate passes; no NumPy numerical path is
left in the three solver modules; the GPU check passes on one card; the public interface is unchanged.

## Review

Before opening the PR for each sub-phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind).
- User-facing documentation listed as tasks is updated, not deferred.
