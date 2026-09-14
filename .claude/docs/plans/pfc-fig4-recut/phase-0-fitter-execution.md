# Recut phase 0 (fitter): batched JAX Poisson GLMs, equivalence-gated, then the batch rerun

The user's instruction (2026-09-14): "start the batch re-run, switching the fitter to nemos: we can
probably just look at the nemos source and implement what we need in jax". Spec authority: this
file; `phase-0-glm-mechanics.md` §"PFC fit validity" only where the new fitter must report
validity (the spike-bin edge rewrite, design preflight and theta rework in that file stay out —
they change numbers and belong to their own PR); `phase-1-2-execution.md` for the event
definition and matched control the rerun fits under.

**Why.** The mPFC batch takes ~17 h because 17,328 unit-sessions × 9 Poisson GLMs are fit one
unit at a time with statsmodels IRLS on 270k–700k-row designs (~17 s per unit serial), and the
parallelism is per session, so a 511-unit session is a 2.4 h straggler. Every unit in a session
shares the same design matrix; only the response column differs. A batched Newton/IRLS in JAX
fits all units of a session per model in one call.

## Decisions (binding)

1. **Own implementation, no new dependency.** `nemos` is the reference for conventions (read its
   Poisson observation model and pseudo-R² source if useful) but the fitter is written here in JAX:
   `src/continuum_swr_replay/analysis/pfc/batched_glm.py`. Poisson, log link, float64
   (`jax.config.update("jax_enable_x64", True)` inside the module's entry, as
   `analysis/replay/dirichlet_multinomial.py` does), Newton = IRLS: per iteration
   `mu = exp(X β)`, gradient `Xᵀ(y − mu)`, Hessian `Xᵀ diag(mu) X` per unit
   (`einsum("tn,tp,tq->npq")`, chunked over time so no `(n_time, p, n_units)` intermediate is
   materialized), solve with a pseudo-inverse-tolerant solver so a rank-deficient design behaves
   like statsmodels' pinv-based WLS rather than raising. **Same convergence rule as the installed
   statsmodels 0.14.5** (`GLM.fit(method="IRLS", maxiter=100, tol=1e-8, atol=1e-8, rtol=0)`: stop
   when `|deviance − deviance_prev| < 1e-8`; read `_fit_irls`/`_check_convergence` and mirror the
   starting values `family.starting_mu` = `(y + mean(y)) / 2`), so both fitters converge to the
   same point to numerical precision.
2. **Results contract.** `fit_poisson_glms(design (n_time, p), counts (n_time, n_units)) ->
   BatchedGLMResults` with per-unit `params (n_units, p)`, `llf` (full Poisson log-likelihood
   including the `−log(y!)` term, as statsmodels' `llf`), `deviance`, `aic = 2p − 2 llf`,
   `converged`, `n_iter`, `rank`; and `.for_unit(i)` returning an object exposing `.params`,
   `.llf`, `.deviance`, `.aic`, `.converged` so `process_neuron` and everything downstream keep
   reading the same attributes. Runs on GPU when JAX sees one, CPU otherwise; the module never
   imports Spyglass.
3. **Integration.** `run_session` fits every unit at once per model (the nine immobility designs
   on the fit mask, and the three-column theta design on the running mask), then calls
   `process_neuron` per unit with the prefitted results; `_fit_immobility_models` and
   `test_theta_modulation` keep their names and semantics but take prefitted results (or a fitter
   callable) instead of fitting. A `--fitter {jax,statsmodels}` flag on
   `scripts/pfc/run_pfc_analysis_bandit.py` (default `jax`) keeps the statsmodels path runnable for
   the gate; the sidecar records `fitter`, `fitter_tol`, `jax_backend` (gpu/cpu), `jax_version`.
4. **Equivalence gate before any batch.** On the three smoke sessions (`j1620210710_02_r1`,
   `chimi20200223_06_r3`, `wilbur20210329_02_r1`) fit both ways to scratch dirs and compare per
   unit and model: `max|Δparams|`, `max|Δdeviance|`, `max|Δaic|`, `max|Δp|` over every LR pair,
   best-AIC model identity, NaN/convergence flags identity. Tolerances (fail the gate otherwise):
   deviance and llf `1e-6` absolute, params `1e-5` absolute, LR p-values `1e-6` absolute,
   best-AIC model and flags identical. Generated audit
   `docs/audits/2026-09-14-pfc-jax-fitter-equivalence.md` (`scripts/qa/compare_pfc_fitters.py`,
   `Command:` line) with wall-clock per session for both fitters and the JAX backend used.
5. **Then the batch rerun** (recut Phase 3's work, brought forward on the user's instruction):
   move the 71 canonical `Processed-Data/*_mPFC_modulation*` files (+ sidecars) to
   `Processed-Data/_pre_canonical_events_backup_20260914/` (never delete); smoke one session with
   `/usr/bin/time -v`, extrapolate; run in tmux `pfc_canonical` with logs under
   `logs/pfc_canonical/`; sessions may run in parallel across the four A100s (one Dask worker per
   GPU with `CUDA_VISIBLE_DEVICES`, or sequentially on one GPU if the per-session time is
   dominated by `load_data`) — measure on the smoke and choose; record the choice. Verify 71
   bundles + sidecars (`is_swr_definition == canonical_event_table`, `fitter == jax`,
   `n_matched_control > 0`). Regenerate Figure 4. Audit vs the backup with
   `scripts/qa/compare_pfc_cohorts.py --before <backup> --after Processed-Data` (extend it to
   report the matched-control tests and the fitter): the pre-existing model coefficients should
   agree with the backup to the gate tolerance on every unit (same events, same cohort, same
   design — a real-data end-to-end check of the swap); the two new tests' significant fractions
   per animal are the new result. Docs and claim 20 requoted with a dated note.
6. **GPU install.** `uv sync --extra cuda` into the project `.venv` is allowed (it is the
   project's own extra); verify `jax.devices()` shows GPUs and that the existing suite still
   passes; if the CUDA plugin fails against the pinned versions, stay on CPU (still ≫ faster than
   the per-unit loop) and say so.

## Global Constraints

As in `phase-1-2-execution.md` (uv; ruff on added lines; no plan/phase/task references; nothing
under `Processed-Data/`, `logs/`, `paper/figures/**/*.png` committed; trailers; no
checkout/stash/clean; stage by name; `glm.py` importable without Spyglass; layering test green;
value-level tests; `slow` over 20 s). Never overwrite a canonical artifact without moving it first.
Commit before launching the batch and do not commit while it runs (sidecars hash HEAD at write
time).

### Task A: the batched fitter, integration, equivalence gate

Files: new `analysis/pfc/batched_glm.py` (Decisions 1–2); `analysis/pfc/glm.py`
(`_fit_immobility_models`, `test_theta_modulation`, `process_neuron` signature: prefitted results
in; `run_session`: batch-fit per model; keep the rank-deficiency gate and NaN handling semantics);
`scripts/pfc/run_pfc_analysis_bandit.py` (`--fitter`, sidecar keys); `scripts/figures/generate_figure04.py`
(same fitter path for panel a); new `scripts/qa/compare_pfc_fitters.py` (Decision 4). Tests:
`tests/analysis/pfc/test_batched_glm.py` — on synthetic Poisson data with a planted β, the batched
fit equals `statsmodels.GLM(...).fit()` per unit to 1e-8 on params/deviance/llf/aic (statsmodels is
a test dependency already); a rank-deficient design gives the same params as statsmodels' pinv
solution and `rank < p`; a unit with all-zero counts; convergence flags; CPU-only run;
`test_glm_pipeline.py` — the existing pipeline tests pass under both fitters and the `all_stops`
bit-for-bit test holds with `fitter="statsmodels"`; a test that `process_neuron`'s outputs are
identical (to 1e-8) between the two fitters on the synthetic fixture. Gate on the three sessions
(scratch dirs); audit committed. Report wall-clock per session for both fitters.

### Task B: the batch rerun, Figure 4, audit, docs

Decision 5 end to end; `scripts/qa/compare_pfc_cohorts.py` extended (matched-control tests,
fitter column, tolerance check of shared coefficients vs the backup); audit
`docs/audits/2026-09-14-pfc-canonical-rerun.md`; `docs/README.md`, `docs/design/cohort_and_timing_rules.md`
(fitter), `scripts/README.md`, `paper/claims_table.md` row 20 (numbers from the audit, dated
note; 5 cells; RETRACTED kept), pipeline map; `PLAN.md` status.

## Review

Final review (Fable): the fitter reproduces statsmodels to the gate tolerances on real data (audit);
no downstream statistic changed meaning; the batch ran under the canonical definition with the
control; shared coefficients agree with the backup; every requoted number traces; no plan
references; suite green.
