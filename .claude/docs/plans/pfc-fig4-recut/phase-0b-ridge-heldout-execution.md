# Recut phase 0b: informative ridge + held-out deviance for the mPFC unit GLMs

User decision (2026-09-15): "Informative ridge with held out deviance." Replaces the
statsmodels hand-back of separated fits (phase-0-fitter) and the asymptotic likelihood-ratio /
AIC machinery for the per-unit statistics. Spec authority: this file; `phase-2b-dependence-inference.md`
only where it names the blocked held-out contract; `shared-contracts.md` §"Matched-control
interface" (unchanged). Everything else in the recut plan that this contradicts is superseded here.

**Implementation status (reviewed 2026-09-17).** Task A and the Task B integration are implemented
and passed the corrected three-session GPU gate at `00765b2` plus the full R = 1,000 endurance
gate; Task C (the production batch, final audit, Figure 4 and claim update) has not run. The
original smoke selected σ = 3, K = 5 and R = 40.
A follow-up prediction-only, microbatched JAX scorer reduced the j16 null phase from 8:26 to 1:49
and made **R = 1,000** the production setting; R = 40 is retained only as the historical
equivalence gate. The implementation review supersedes the draft values
below where they conflict: empirical p-values use `(b + 1)/(R + 1)`, not `b/R`; incompletely
placed draws are excluded and every requested production draw must be usable (20 remains only the
diagnostic floor); runs whose ±250 ms kernel
supports overlap are one fold block; and raw-p unit fractions are descriptive previews, not
population inference. In the corrected gate every class produced 40/40 exact draws on its fixed
matchable event subset, the combined population loader and estimator audit passed, and the largest
session completed in 27:25 at 8.89 GiB peak RSS. The optimized j16 gate completed in 5:23 at
4.66 GiB peak RSS with an equivalent 74 x 131 result table and byte-identical modulation arrays.
Task C may launch on idle cards.

**The problem.** 23 % of mPFC units have at least one model whose unpenalised Poisson fit
separates (a spline column with no spikes under its support drives a coefficient to −∞); statsmodels
runs those to its iteration cap and ships meaningless LR statistics (≈16 % of FDR-significant calls
across the 13 tests on the gate sessions, 25 % on the replay-split test). The batched JAX fitter
reproduced that behaviour by handing those units back to statsmodels, which is 77 % of a session's
wall clock.

## Decisions (binding)

1. **Estimator: penalised Poisson GLM with a loose Gaussian prior on the event-basis
   coefficients.** The coefficients live on the log-rate scale (the cubic B-spline basis with 50 ms
   knots is a partition of unity, so each coefficient is the log-rate modulation near its knot). The
   prior's only job is to exclude implausible rates, not to shape plausible ones: `σ = 2` per
   coefficient (`λ = 1/σ² = 0.25`), 95 % of prior mass on rate multipliers between about ×0.02 and
   ×55; rat mPFC units are reported to move by at most ~10× above baseline and to near silence below,
   and 99 % of this corpus's well-posed SWR modulations are under ×20, so the prior binds only where
   the unpenalised fit diverges (a basis column with no spikes under it, whose coefficient the prior
   settles near −2 to −3, i.e. "rate at most a small fraction of baseline"). The same prior for every
   unit, model and event class; the intercept is unpenalised. Sensitivity bundles at `σ = 1` and
   `σ = 3`, plus an empirical-Bayes estimate of σ from the converged units' coefficient spread with
   estimation noise deconvolved, reported next to the chosen value. The audit reports for how many
   units the prior binds (penalised vs unpenalised peak modulation differing by more than the
   estimation error). The choice is documented as a stated scientific assumption.
2. **One fitter.** The batched JAX Newton solver gains the penalty (`H + Λ`, `Λ = λ` on the
   basis columns in the ORIGINAL coordinates — with the orthonormal-basis trick the penalty must be
   transformed accordingly, and the intercept direction must stay unpenalised: orthogonalise the
   basis columns against the constant first). The penalised objective is strictly concave, so every
   unit converges; the statsmodels hand-back, `needs_reference_fit`, `--reference-fit-workers` and
   `--fitter statsmodels` are deleted from the runner (the gate script and audit stay as history).
   Convergence rule unchanged (|Δ penalised deviance| < 1e-8, maxiter 100); a unit that still fails
   is non-estimable (NaN everywhere, counted).
3. **Inference by blocked held-out deviance.** `K` folds (5 or 10 — decided on the three smoke
   sessions by the fold-to-fold spread of each pair's held-out difference relative to its mean, and
   by events-per-fold for the rarest class; recorded in the sidecar) over contiguous runs of the fit mask
   (whole runs assigned to folds, balanced by bin count, assignment seeded from the session name and
   recorded). For each model and unit: fit on K−1 folds, evaluate the Poisson deviance on the held-out
   fold, sum over folds → `heldout_deviance[model]`. Per nested pair `(full, reduced)`:
   `heldout_delta_dev = D_ho(reduced) − D_ho(full)` (positive = the added term predicts held-out
   spikes better), stored per fold too. Best model per unit = lowest held-out deviance (replaces
   best-AIC). Deviance fractions use held-out deviances. The columns `… : lr`, `… : pval`, `… : aic`
   are no longer produced; the stats table gains `… : heldout_delta_dev`, `… : heldout_delta_dev_folds`
   (K values), `heldout_deviance[model]`, `separated_columns[model]` (count of basis columns with no
   spikes under their support in the training data — reported, not acted on).
4. **Per-unit null from repeated matched-control draws.** The matched control already defines "a
   duration-matched non-event regressor"; drawing it `R` times (different seeds, same rules) gives the
   null distribution of `heldout_delta_dev` for a regressor with no event content. For each unit:
   `p_heldout[class]` = fraction of the R control draws whose `D_ho(constant) − D_ho(control_r)` is
   ≥ the observed `D_ho(constant) − D_ho(class)` for `class ∈ {SWR, non_local, non_local_no_SWR,
   non_local_SWR}`; and the engagement statistic `D_ho(MC) − D_ho(MC + non_local_no_SWR)` with its
   own null from re-drawn controls. `R` is set from a smoke (target 50; floor 20); each draw is one
   batched fit per fold across all units, so cost scales as `R × K` batched fits per session — measure.
   **Amended after the 2026-09-16 exact-null gate:** if a real event has no legal non-event window
   in its own fitted bout, it cannot enter a matched randomisation test. Select one deterministic,
   jointly matchable event subset per class using timing/duration only, and use that identical subset
   for the observed statistic and all null draws. Keep the original all-event `heldout_delta_dev` as a
   separate descriptive result; store the matched-test observed statistic separately and record total,
   tested and excluded event counts plus tested source indices in the sidecar. Never compare an
   all-event observed statistic with a thinner null.
5. **Modulation curves** come from the full-data penalised fit at `σ = 2`.
6. **Basis fixes folded in (user instruction 2026-09-15).** (a) Boundary symmetry: the patsy
   `bs(...) - 1` basis drops one boundary function, pinning the kernel to zero at −250 ms but
   leaving +250 ms free. Drop BOTH boundary functions so the kernel returns to baseline at both
   ends of the fit window (state the parameter count per class before/after; the report window
   ±200 ms is unchanged); verify the basis sums to one on the interior and to zero at both ends.
   (b) Build the convolution columns by direct convolution of the onset indicator with the basis
   kernels (one `convolve` per basis function per class) instead of materialising 251 lagged
   copies (≈1.4 GB per class on the longest sessions); assert equality with the lagged-matrix
   construction on a synthetic indicator to 1e-12 before deleting the old path. (c) NOT in this
   phase: an event-duration regressor ("event in progress" box, or an offset-locked kernel) — the
   onset-locked kernel ignores duration (at-well events median 62 ms, SWR-coupled 88 ms, 11 % of
   SWR-coupled events > 200 ms); scheduled as its own reviewed change with an explicit comparison
   against the onset-only kernel.
7. **Population inference stays Phase 4** (neuron → session → animal; the audit here reports the
   fraction of units with `p_heldout < 0.05` per class and animal as a preview, clearly labelled).
8. **Validation before the batch.** (a) `σ → ∞` (λ = 0): the penalised solver equals the
   unpenalised JAX fit and statsmodels on the gate sessions' well-posed units to the existing
   tolerances (reuse the gate script); (b) synthetic recovery: planted modulation curves at several
   amplitudes and spike rates — ridge recovers them within the expected shrinkage, and a separated
   synthetic unit gets a finite, small curve; (c) held-out deviance prefers the generating model on
   synthetic data and is indifferent under the null; (d) fold assignment never splits a run and is
   reproducible; (e) the control-draw null is uniform-ish on synthetic null units (KS against uniform,
   report).
9. **Batch.** Back up the hand-back batch (`Processed-Data/*_mPFC_modulation*` → `Processed-Data/_pre_ridge_backup_<date>/`),
   refit 71 sessions on free GPUs (launch script pattern from phase-0-fitter, one worker per free card,
   never 0/1/5/8 or whatever is busy at launch), verify, Figure 4 regenerated with the new per-unit
   statistics (panel a from the ridge fit; population panels from `load_population_data`, which must
   read the new columns — that loader change is in scope here since the old columns disappear),
   audit `docs/audits/<date>-pfc-ridge-heldout.md` (`scripts/qa/compare_pfc_cohorts.py` extended:
   coefficient shrinkage vs the MLE bundles; old FDR fractions vs new held-out fractions per test and
   animal; separated-column counts; σ sensitivity), docs (`docs/design/cohort_and_timing_rules.md`
   estimator + inference section; `docs/README.md`; claims row 20 requoted with a dated note; scripts
   README; pipeline map).

## Global Constraints

As in `phase-0-fitter-execution.md`; plus: every estimator/inference choice above appears in the
sidecar (`penalty_sigma_log_rate`, `n_folds`, `fold_seed`, `n_control_draws`, `control_draw_seed`);
no plan/phase/task references in shipped code/tests/docs; commit before launching, not during.

### Task A: penalised batched solver + blocked folds + control-draw null (library + tests + validation)
### Task B: runner/Figure 4/loader integration, smoke on three sessions, audit script extension
### Task C: batch, audit, docs, claims

(Task briefs are the sections above; the implementer of each task reads Decisions 1–8.)

## Review

Final review (Fable): the penalty is applied in the original coordinates with the intercept free;
λ → 0 reproduces the MLE; folds respect runs; the null is built from the same control rules; no
LR/AIC column survives; every sidecar records the estimator choices; the audit traces; claim 20
names the prior and the held-out criterion; no plan references; suite green.
