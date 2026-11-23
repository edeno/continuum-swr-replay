<!-- Copy of the port design, made 2026-11-23 from the main checkout's
.claude/research/2026-11-20-regression-prototypes/jax_fitter/DESIGN.md. -->

> **Note on this copy (plan folder, 2026-11-23).** The text below is the port design as written, not
> shortened. One line is changed: in section 12, "After the series", the marker of the old smooth-term
> question is replaced by a pointer to the open question that remains for the mPFC models. Since it was
> written, the lead decided (design, twenty-first round, 2026-11-23, 12:32): float64 on the CPU and on the
> GPU ("sure"; section 8 and question 2 of section 13), and the acceptance rule of section 9 ("ok";
> question 3 of section 13, except whether the fixture script is kept). Question 1 (a slower CPU fit) is
> open: the lead, "depends on how much time things take". Question 4 (the order) is the author's call;
> the lead did not object. The plan's phases are [phase-2b](phase-2b-simulation-engine.md) (this
> design's Phase 1) and [phase-2c](phase-2c-jax-fitter.md) (its Phases 0, 2 to 6).

# One JAX fitter for every model family: design and phased tasks

Date: 2026-11-23. Status: design only. Nothing is implemented.

The decision: the lead wants one fitter, in JAX, for every family (rate, content, stay/leave,
ripple power, mPFC units). Today the shared NumPy fitter does the first four, and the plan
had a separate JAX twin for mPFC only (`.claude/docs/plans/regression-convention/designs.md`,
"JAX batch twin"). This design replaces that twin with a port of the shared fitter itself.

Code read: the worktree `/cumulus/edeno/continuum-swr-replay-convention-tools`, branch
`regression-declared-penalty` at `86ad3fc5`. Every `file:line` below is in that worktree
unless another path is given. JAX is 0.8.0, jaxlib 0.8.0, NumPy 2.3.4 (the project venv).
Every timing here was **taken under load**: one core (`taskset -c 0`), the machine at a load
average of 30 to 40 on 96 cores. The experiments are in the appendix.

## 0. The recommendations in short

- **Float64**: on by a scoped `jax.enable_x64(True)` at every public entry of the fitter,
  never process-wide. Every JAX kernel refuses at trace time to run on anything but float64.
- **Shapes**: one session per fit. Rows are padded with zero-weight rows to a few bucket
  sizes. Columns are never padded (the inactive columns are cut out in NumPy, as now). The
  mPFC models batch the units of one session (shared design) with `vmap`. Sessions are not
  vmapped.
- **Newton loop**: a host loop in Python and NumPy that holds every rule and status, around
  one jitted kernel per Newton iteration (the system and the step-halving search). The
  control is written batch-first (a leading axis of fits, one for a session, many for mPFC
  units), so the same control code serves every family.
- **Derivatives**: keep the hand-written information and its directional derivatives.
  Use automatic differentiation only as a test oracle.
- **REML**: port its numerical kernels now (fixed-penalty score, covariance, edf, and the
  gradient and Hessian in the log penalties). Keep the outer search in Python. Build the
  batched-over-units version in the mPFC phase.
- **Removal**: each phase removes the NumPy code it replaces. The last phase removes what is
  left and adds a repository test that keeps it out.

## 1. Map of the current core

"Fit" means numerical work that moves to JAX. "Bookkeeping" stays in NumPy.

### `likelihoods.py`

| Function (line) | What it computes | Fate |
|---|---|---|
| `_weighted_gram` (95) | `X' diag(w) X`. The docs' "Cost" section puts 82 % of an evaluation's time here | fit |
| `Poisson.loglik`, `score_eta`, `_weights` (162–173) | `y'(eta+o) - sum exp(eta+o)`, `y - mu`, `(mu, mu, mu)` | fit |
| `Bernoulli.*` (194–205) | logit loglik, score, `w, w(1-2mu), w(1-6w)` | fit |
| `Gaussian.*` (224–238) | loglik with its constant, score, `1/phi, 0, 0` | fit |
| `_DiagonalFamily.information`, `dinformation`, `d2information` (113–132) | `X' diag(w) X`, `X' diag(w' d) X`, `X' diag(w'' d1 d2) X` | fit |
| `GroupedSoftmax._parts`, `loglik`, `score_eta`, `_sandwich`, `information`, `dinformation`, `d2information`, `_g_times` (309–395) | per-group softmax with `reduceat` | fit (segment sums) |
| `*.check` (144, 185, 217, 281) | input validation | bookkeeping |
| `group_starts` (241), `GroupedSoftmax.identifying_design` (300) | group starts; X less group means | bookkeeping (design check) |

### `newton.py`

| Function (line) | What it computes | Fate |
|---|---|---|
| `_check_inputs` (183) | shapes, finiteness, `lik.check` | bookkeeping |
| `_free_columns` (159), `_unidentified` (165) | the design check: `eigh` of the unit-scaled Gram matrix of the free coordinates | bookkeeping (it is a design check; see section 5) |
| `_objective` (348) | `F = -l + b'Sb/2` | fit |
| `_newton_system` (361) | `g`, `A = H + S`, `cho_factor`, the step, the decrement | fit |
| `_eta_step` (209) | `max |X step|` | fit |
| `penalized_newton` loop (243–345), `_last_step` (427), `_runs_away_at_the_cap` (409), `_no_finite_maximum` (386) | the control rules and statuses | host control (Python and NumPy) |
| `running_away_columns` (214) | columns by their part of the step, with the 9-decimal tie rule | bookkeeping (kept in NumPy so that the order rule stays the same on every machine) |

### `reml.py`

| Function (line) | What it computes | Fate |
|---|---|---|
| `PenaltyBlock.__post_init__` (188) | penalty rank, log pseudo-determinant, eigenbasis (`eigh`) | bookkeeping (once per block) |
| `assemble_penalty`, `_embedded`, `_strength_blocks`, `_summed`, `free_basis`, `_log_pdet`, `_eigenbasis`, `_rotate`, `_rotate_both`, `_null_dimension` (540–637) | p×p bookkeeping | bookkeeping |
| the rotation `X[:, cols] @ V` in `reml_and_derivatives` (671) | design in the penalty eigenbasis, n×k×k per evaluation (18 ms per session, measured 2026-11-21) | bookkeeping at first; move to JAX if the benchmark says so |
| `_half_log_det` (640), `_coefficient_derivatives` (645), `_covariance_and_edf` (654) | `log|A|/2`, `-A^-1 S_j b`, `A^-1`, edf | fit |
| `_reml_in_basis` (781) | the score; its gradient and Hessian in rho (with `dinformation`, `d2information` and `cho_solve`) | fit |
| `_gaussian_profiled` (848) | `X'X + S`, profiled scale, score and derivatives | fit |
| `_session_score` (954) | active-column cut and NaN embedding | bookkeeping |
| `pooled_reml` (1056), `_Evaluator` (1126), `_grid_start` (1172), `_Curvature` (1235, an m×m `eigh`), `_try_step` (1293), `_search` (1321), `_screen` (1430), `_named_coordinates` (1479), `refit_session` (1568), `select_penalties` (1813), `fit_at_penalties` (1871) | the pooled score, the search and the refit rules | host control |

### Other modules

| Module | Numerical work | Fate |
|---|---|---|
| `sessions.py` | none of its own. It masks rows, applies the support rule, calls `refit_session` (`_Session.fit`, 310) and `term_collinearity` (`_record`, 438) | bookkeeping |
| `calibration.py` | bisection on the log penalty. It calls `fit_sessions` about 18 times (`log_penalty_for_edf`, 89) | unchanged |
| `collinearity.py` | `weighted_design` (74) reads the private `lik._parts` (97) and `lik._weights` (103). `concurvity` (107) uses QR, SVD and `eigh`, once per kept fit | the weights come from a public likelihood method; the measures stay NumPy |
| `terms.py`, `pspline.py`, `support.py`, `folds.py`, `record.py`, `contrasts.py`, `bands.py`, `constants.py` | design assembly, the B-spline basis, counts, folds, records, `c'Vc`, band draws | unchanged |

Callers outside the package: `scripts/checks/compare_regression_to_mgcv.py`
(`select_penalties`), `compare_regression_to_current_fitters.py` (`penalized_newton`),
`simulate_declared_penalty.py` (`fit_sessions`, `log_penalty_for_edf`), and the family runners
that the plan's phases 5a, 6a and 7 will write on `fit_sessions`.

## 2. The public interface

The callers can stay as they are. Only the solver underneath changes.

These signatures do not change: `sessions.fit_sessions`, `calibration.log_penalty_for_edf`,
`reml.refit_session`, `select_penalties`, `refine_penalties`, `fit_at_penalties`,
`pooled_reml`, `reml_and_derivatives`, `newton.penalized_newton`, `newton.running_away_columns`,
`record.*`, `contrasts.*`, `bands.*` and `collinearity.term_collinearity`.

The data classes (`NewtonFit`, `RemlScore`, `PooledScore`, `PooledFit`, `SessionRefit`,
`SetAsideSession`, `PenaltySearch`, `FitRecord`) keep their fields. Every array in them stays a
NumPy float64 array. No JAX array leaves the package.

What changes inside:

1. The likelihood classes keep their names and their public methods (`check`,
   `identifying_design`, `loglik`, `score_eta`, `information`, `dinformation`,
   `d2information`), with NumPy in and NumPy out. Each method runs a jitted kernel under the
   float64 scope. The tests in `test_newton.py` and `test_likelihoods.py` call these methods
   directly, so they keep working.
2. Each family gains pure JAX kernels that the jitted Newton and REML kernels call. They are
   private to the package.
3. `collinearity.weighted_design` stops reading `lik._parts` and `lik._weights`. It calls one
   public method, `lik.weighted_design(X, y, aux, eta)`, which returns NumPy. This is the only
   change a caller sees, and it is inside the package.
4. No signature gains a device argument. The device is chosen by the runner
   (`runs/device.py`), as it is now for the decoders.

One signature is added later, in the mPFC phase: a pooled REML over the units of one session
that share one design (for example `pooled_reml_shared_design(lik, X, Y, aux, blocks, rho)`,
with `Y` of shape `(n_rows, n_units)`). It is a new entry point. No existing one changes.

Test seams that must survive, because tests patch them:
- `reml.penalized_newton`: `test_reml.py:258`, `:676`, `:700`.
- `reml._labeled_score`: `test_reml.py:1176`.
- `sessions.refit_session`: `test_sessions.py:740`.
- the module constants `REML_MAX_STEPS`, `MAX_STEP_HALVINGS` and `REML_DECREMENT_TOL` in
  `reml`: `test_reml.py:594`, `:600`, `:952`, `:1442`.

So the host loop keeps those names. A rule constant is passed into a kernel as an argument and
is never closed over when the kernel is traced. Otherwise a patched value would be ignored by
code that was already compiled.

## 3. Float64

### What the flag does (measured, appendix A)

- With the flag off, `jnp.asarray(np.float64 array)` silently gives float32. There is no
  warning: `1 + 1e-12` became exactly `1.0`. A warning appears only when float64 is asked for
  explicitly (`dtype=jnp.float64`). A jitted function given a float64 NumPy array with the
  flag off computes in float32.
- Inside `with jax.enable_x64(True):` the same calls give float64. An array made inside the
  context stays float64 after it. A jitted function called inside and then outside the
  context has two compiled entries (`_cache_size() == 2`). This matches the JAX 0.8.0 source:
  `jax_enable_x64` is declared with `include_in_jit_key=True` and
  `include_in_trace_context=True` (`jax/_src/config.py:1719–1724`), so the jit caches are keyed
  on it.
- `jax.enable_x64` is exported in 0.8.0 (`jax/__init__.py:50`). The project already uses it
  as a context that is local to the call and the thread: `analysis/pfc/batched_glm.py:129`
  (`_in_float64`), `replay_models/content/dirichlet_multinomial.py:73`, `:191`, `:418`, `:456`,
  and `replay_models/rate/ridge_history_cv.py:423`. Two tests check that the process-wide
  value is left as it was (`tests/analysis/pfc/test_batched_glm.py:833–845`,
  `tests/analysis/replay_models/content/test_dirichlet_multinomial.py:291–306`).

### Effect on the decoder in the same process

`non_local_detector` never sets the flag. A grep of the installed package finds no
`enable_x64` and no `matmul_precision`. It runs in JAX's default, float32. If the fitter set
the flag process-wide (as neurospatial does at import, `encoding/_backend.py:95`), every
decoder array built from float64 NumPy inputs would become float64. That doubles memory,
slows the GPU and changes results that the memoryless detector's checks expect to repeat
bit for bit (`runs/device.py`, `DETERMINISTIC_XLA_FLAG`).

### Policy

1. There is one decorator in the package, `in_float64`, which is the `batched_glm._in_float64`
   pattern moved into `analysis/regression/`. It enters a fresh `jax.enable_x64(True)` on every
   call. Every public function that makes a JAX array or runs a kernel carries it. The flag
   is never set process-wide, and nothing sets it at import.
2. Every kernel checks at trace time that its float inputs are `float64` and raises
   otherwise. A kernel reached without the scope then fails loudly instead of fitting in
   float32.
3. Inputs are converted with `jnp.asarray(x, dtype=jnp.float64)` inside the scope. Outputs are
   converted with `np.asarray` before they leave.
4. Tests:
   - A fit run in a new thread leaves `jax.config.read("jax_enable_x64")` as it was (the
     `test_batched_glm.py:833` pattern).
   - After a fit, `jnp.asarray(np.float64 array).dtype` is still float32 in the caller.
   - Calling a raw kernel with the flag off raises the dtype error.
   - A fit returns the same coefficients whether the caller has the flag on or off.

## 4. Shapes and compilation

### What was measured (appendix B and D, under load)

- A whole Poisson fit (37,000 × 80, `lax.while_loop` with step halving) took **0.61 s** to
  compile and 0.41 s to run (8 steps). A new row count (n − 1) cost 1.03 s with compilation.
- `jit` compiles again for each new shape. Row counts 100, 101, 100 and 128 gave 3 cache
  entries.
- `segment_sum` needs `num_segments` known at trace time. A traced value raises
  `ConcretizationTypeError`. So the number of events (groups) is part of the shape.
- Padding rows with an offset of `-inf` gives `F = nan` (`0 * -inf`). Padding needs a row
  mask, not an offset trick.
- `vmap` over 4 copies of a session's Gram matrix took 0.143 s against 0.039 s for one. On one
  CPU core vmapping gives nothing. It helps only on a GPU, or where the design is shared.

### The options

| Option | For | Against |
|---|---|---|
| Compile per exact shape | simplest; exact | about 242 × 2 kinds × the column counts after refits, at 0.6 to 1 s each: minutes of compiling per pass, against about 0.08 s per rate fit (C11) and 0.03 s per content fit |
| Pad rows to buckets, mask with zero weight | few compiles; the result is exact up to the summation order | up to one bucket step of extra rows |
| vmap over sessions | one program for the cohort | sessions differ in active columns, and padding columns changes the Cholesky factor and the traces. 242 × 37,000 × 80 × 8 B = 5.7 GB of designs. No gain on CPU (measured) |

### Recommendation

- **Rate** (and stay/leave, ripple power): one session per fit. Rows are padded to the next
  multiple of 4096, so at most 11 % extra on a 37,000-row session. A padded row has `X = 0`,
  `y = 0`, offset 0 and weight 0. Every likelihood sum is weighted, so a padded row adds
  exactly 0. With `X = 0` it adds 0 to `X'WX` and to `max |X step|` anyway. Columns are not
  padded. `p` takes the few values the refit rounds leave.
- **Content**: one session per fit. Rows are padded to a multiple of 1024 and groups to a
  multiple of 64. Padded rows go to one extra dump group with weight 0 and `n_g = 0`. An empty
  group's `log(sum exp)` is `-inf`, and `0 * -inf = nan`, so every per-group term is guarded
  with `where(n_g > 0, ...)`. A test checks this.
- **mPFC**: the units of one session share one design. The kernels are vmapped over units, or
  written as one `einsum` over units for `X' diag(w_u) X`. Units are taken in chunks of a fixed
  size (padded units masked), and rows in chunks through `lax.scan`. This is the pattern of
  `batched_glm.py:425` (`_chunk_rows`) and `:460` (`_sweep`). The shape is then fixed per
  session and per chunk.
- **The JIT boundary** is one Newton iteration (section 5), one "evaluate at beta" (for the
  last step), one fixed-penalty summary (`A^-1`, edf, `log|A|`), and one REML-derivative
  kernel. Each is jitted per family. The design check, the support rule, the refit rounds and
  the REML search stay in Python.
- **Expected compile cost.** The estimate: about 10 to 15 row buckets × 2 to 4 column counts ×
  4 kernels per family, so on the order of 100 compiles at about 0.5 s each. That is about 1
  to 2 minutes the first time, then nothing, because the suite already keeps a persistent
  compilation cache (`tests/conftest.py:30–55`). Runners should set the same cache directory.
  This is an estimate. Phase 0 measures it.

## 5. The Newton loop and the no-finite-maximum rule

### Choice: a host loop around a jitted iteration kernel

| | Host loop around a jitted step | `lax.while_loop` for the whole fit |
|---|---|---|
| The rules as written now (`newton.py:311–345`) | carried over almost line for line | rewritten as a state machine with ring buffers |
| Statuses and reasons | Python strings, as now | integer codes, mapped back afterwards |
| Patched constants (section 2) | read on the host | must be arguments, or patches are ignored |
| Cost | one dispatch and one host read per step: 25–35 µs measured (appendix E), about 0.1 % of a rate step (about 50 ms) | none per step |
| GPU | one sync per step | none |

The recommendation is the host loop. The control code is written **batch-first**: every state
array has a leading axis of fits (one for a session; one per unit for mPFC), and each rule is
a vectorized NumPy expression over that axis. That keeps one copy of the control code for
every family, which the lead's "one fitter" rule needs. A `lax.while_loop` version is kept in
reserve. It is built only if the mPFC benchmark shows the per-step sync costing more than
10 % on a GPU.

### The iteration kernel

The kernel is jitted and vmapped over the batch. Given `beta`, `F` and the cap, it returns:

- `A = H + S`, the Cholesky factor, and `chol_ok = isfinite(L).all()`;
- the step `A^-1 g`, the decrement `g' step / 2`, and `eta_step = max |X step|`;
- `scale0 = min(1, cap / max |step|)` and `capped = scale0 < 1`;
- the halving search as an inner `lax.while_loop`. It tries `scale0 * 2^-k` for
  `k = 0..MAX_STEP_HALVINGS-1` (30 tries, as now) and stops at the first finite `F_new <= F`.
  It returns `candidate`, `F_new`, `loglik_new` and `halving_ok`.

The kernel always runs the halving search. The host ignores it when the decrement is already
below the tolerance. That costs one extra objective evaluation at the end of a fit and saves
a second dispatch on every step.

### The host rules, the same as `newton.py:311–345`, per member of the batch

1. `not chol_ok` gives "information not positive definite". JAX's Cholesky returns NaN where
   SciPy raises (appendix A: an indefinite `A` and a NaN `A` both give a NaN factor). So the
   status comes from `chol_ok`, not from an exception. On CPU both use the same LAPACK `potrf`:
   jaxlib loads its LAPACK kernels from `scipy.linalg.cython_lapack`
   (`jaxlib/cpu/_lapack.so`, symbol `GetLapackKernelsFromScipy`). The near-singular factor in
   appendix A agreed to every printed digit. Use `jax.scipy.linalg.cho_factor`, which reads
   one triangle as SciPy does (`symmetrize_input=False`, `jax/_src/scipy/linalg.py:47`), not
   `jnp.linalg.cholesky`, which averages `A` with its transpose by default.
2. `decrement < NEWTON_DECREMENT_TOL`:
   - for a family that is not quadratic with `eta_step > NO_MAXIMUM_ETA_STEP`, the status is
     "no finite maximum". The member keeps `step`, and `running_away_columns(X, step)` is
     computed in NumPy;
   - otherwise the last-step rule of `_last_step` (427). It uses the "evaluate at beta" kernel
     at `beta + step`, accepts when `F_new <= F + 8 eps max(1, |F|)` and the new decrement is
     below the tolerance, and adds 1 to `n_iter`.
3. Otherwise the host appends `eta_step` to the member's history. At `n_iter == max_iter` it
   stops. Then `_runs_away_at_the_cap` decides between "no finite maximum" and "step limit
   reached". This runs on host arrays of shape `(B, NO_MAXIMUM_CAPPED_STEPS + 1)`.
4. `not halving_ok` gives "step halving failed". Otherwise the member accepts the candidate
   and records `capped and F_new < F`.
5. The Gaussian (quadratic) keeps `cap = inf` and never takes the no-maximum branch, as now.

### What stays exactly as it is

- The design check (`_unidentified`, 165) runs in NumPy before the loop. It is an
  eigen-decomposition with a threshold (`IDENTIFIABILITY_TOL = 1e-12`). Doing it in NumPy keeps
  its set of named coordinates the same.
- `refit_session` (`reml.py:1568`) and the session loop (`sessions.py:417`) do not change. They
  read `NewtonFit.status`, `running_away` and `RemlScore.running_away_directions`, which are
  NumPy as now. The rule that a session whose simple terms have no finite maximum keeps its
  other terms is therefore untouched code. It is pinned by
  `test_sessions.py::test_a_separated_session_is_kept_with_its_terms_missing` (91), `:205`,
  `:566`, `:600`, `:626` and `:722`, and by `test_reml.py:991–1300`.
- `max_iter` keeps its meaning: the system is evaluated at `n_iter == max_iter` to test the
  decrement, then the loop stops (`newton.py:311`, `for n_iter in range(max_iter + 1)`).

## 6. Derivatives

Keep the hand-written forms: the score on the eta scale, `X' diag(w) X`, its directional
derivatives, and the grouped-softmax sandwich forms. The reasons:

- They are already checked against finite differences
  (`test_likelihoods.py::test_derivatives_match_finite_differences`, 66;
  `test_reml.py::test_gradient_and_hessian_finite_differences`, 142; `:1457` at strong
  penalties).
- They are faster. Measured (appendix C): the Poisson information by hand took 39 ms, and
  `jax.hessian` of the log-likelihood 62 ms. The softmax information by hand took 4.5 ms, and
  `jax.hessian` 23.5 ms.
- REML needs `dinformation` and `d2information`, which are third and fourth derivatives of
  the log-likelihood along directions. Through autodiff these would be nested `jvp`s of the
  Hessian, and their cost has not been measured.

Use automatic differentiation as the oracle in a new test, `test_likelihoods.py::test_hand_derivatives_equal_autodiff`.
For every family at random `eta`:

- the score equals `jax.grad` of `loglik`;
- the information equals `-jax.hessian(loglik)`;
- `dinformation(d)` equals `jax.jvp` of the information along `X d`;
- `d2information(d1, d2)` equals the nested `jvp`.

The tolerance is 1e-12 relative. Measured: the Poisson information differed from autodiff by
5.8e-19 relative, and the softmax information by 1.1e-15.

## 7. The grouped softmax

- The rows of a group are contiguous (sorted by unit, `sessions.py:124`). So the per-group sums
  are `jax.ops.segment_sum(..., num_segments=G, indices_are_sorted=True)`, with the group id
  of each row as the segment id, and `segment_max` for the stabilizing shift.
  `num_segments` is static (section 4).
- `p = exp(eta - max_g) / sum_g`, and the log-likelihood is `y'(eta - max_g) - n_g' log sum_g`.
  These are the NumPy formulas of `likelihoods.py:310–328`.
- The information is `X' diag(n_g p) X - Xp' diag(n_g) Xp`, with `Xp = segment_sum(p X)`
  (G × p). `dinformation` and `d2information` follow the `_sandwich` pairs of
  `likelihoods.py:335–388`: a diagonal part plus a list of `(u, v)` pairs, each giving
  `segment_sum(u X)' diag(n_g) segment_sum(v X)`.
- A dense reshape `(G, K, p)` is possible only when every event has the same number of rows.
  The row rule drops a removed level's rows (`sessions.py:167`), so groups can be shorter than
  nine. Segment sums handle that. Measured, both forms agree to 1.6e-16, and the segment form
  was the faster (4.5 ms against 5.9 ms).
- Measured against the repository's `GroupedSoftmax.information`: the JAX segment-sum form
  agreed to 9.3e-16 relative (4,000 events × 9 arms × 20 columns).

## 8. Precision and devices

**Float64 everywhere, on CPU and on GPU.** That is the convention. Float32 is not offered.
The reasons:

- The stopping rule is a decrement below 1e-8 with `F` of order 1e4 to 1e5. Float32's
  rounding of `F` is about 1e-2 there.
- The last-step rule uses `8 eps max(1, |F|)`.
- The no-maximum and tie rules are written for double precision
  (`constants.py`, `RUNNING_AWAY_TIE_DECIMALS = 9`).

The repository's own float64 comments say the same (`batched_glm.py:131`,
`dirichlet_multinomial.py:67`). So the plan's float32 GPU path and "check 6" (float32 within
1e-3 or 1e-4 of float64; `phase-8-mpfc.md`) is dropped. It is replaced by a GPU float64
against CPU float64 check.

**GPU float64 is usable here.** Zephyr has 10 A100 80GB PCIe cards (read from `nvidia-smi` on
2026-11-23, all idle), and breeze has 10 A100 80GB (project memory). The A100's float64 rate
is half of its plain float32 rate (NVIDIA's specifications, not measured here).

Known pitfalls:

- **Matmul precision.** In JAX 0.8.0, `jax_default_matmul_precision` is documented as
  controlling "the default matmul and conv precision for 32bit inputs"
  (`jax/_src/config.py:1820`). The TF32 shortcut on the A100 therefore does not apply to
  float64 products. Nothing needs setting. The dtype guard of section 3 makes sure the inputs
  are float64.
- **Nondeterministic reductions on GPU.** XLA's scatter-adds (`segment_sum`) can sum in a
  different order from run to run. A last-digit change can flip a decision near a threshold,
  or the order of two running-away columns that are equal but for rounding. So fits on a GPU
  run with `runs/device.py`'s `REQUIRED_XLA_FLAGS` (deterministic ops, and Triton GEMM off).
- **Different linear algebra on GPU.** The GPU uses cuSOLVER and cuBLAS, so the rounding
  differs from CPU. This is caught by the check below with the same tolerances as section 9.
- **Grabbing cards.** On zephyr's GPU environment a process without `JAX_PLATFORMS` takes most
  of every card the first time it computes (project memory "JAX_PLATFORMS=cpu in GPU
  worktrees"; `runs/device.py:1–40`). The fitter therefore refuses to start its first kernel
  unless `JAX_PLATFORMS` names only `cpu`, or `CUDA_VISIBLE_DEVICES` names exactly one card.
  The error message names the remedy, `runs.device.default_jax_platforms("cpu")` or
  `configure_jax_gpu()`. The check reads the environment and the JAX config only, so it does
  not start a backend. The guard lives inside `analysis/regression/`, because C12 forbids
  importing `runs/` there. Every runner that fits calls `default_jax_platforms("cpu")` in
  `main` unless it asks for a GPU. `simulate_declared_penalty.py` already does.
- **CPU threads.** XLA's CPU dot products use their own thread pool. LAPACK calls go through
  SciPy's OpenBLAS, so `OPENBLAS_NUM_THREADS` applies to them too. `runs/threads.pin_blas_threads`
  should also set `XLA_FLAGS=--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1`
  for process pools. XLA accepted these flags (no error). Their effect on the thread count was
  not measured.
- **Fork after JAX starts.** `simulate_declared_penalty.py:481` uses a `fork` pool. A fork after
  JAX's backend has started can deadlock. Pools that fit should use `spawn`, as
  `fit_cross_history_ridge_cv.py:185` does.

How the tests catch a drift:

- A manual script, `scripts/checks/check_regression_gpu.py` (new), runs the Phase 0 fixture
  cases and one real session on one free card, after reading `nvidia-smi`. It compares them
  with CPU using the section 9 tolerances, with every discrete output exact. It runs twice on
  the GPU and requires bitwise equality.
- In CI, the fixture tests and the dtype guard run on CPU.

## 9. Behavior capture

### Fixtures to store before the change

These come from the NumPy fitter at a named commit. They are written by
`scripts/checks/write_regression_reference.py` (new) to
`tests/analysis/regression/data/numpy_reference/*.json`, each with the commit and the command,
as the mgcv and neurospatial fixtures do (`data/mgcv/README.md`).

| Case | Outputs stored |
|---|---|
| Every `penalized_newton` call in `test_newton.py`: hand cases (34, 59), softmax (109), halving (154), statuses (181, 204, 329), last step (220, 381), separation cases (279–329), collinear seeds (363), far maximum (463), softmax not identified (480) | `beta`, `A`, `status`, `n_iter`, `loglik`, whether `decrement < tol`, `step`, `running_away`, `unidentified` |
| `reml_and_derivatives` for each family × 1 and 2 smooths (the `_session` builder of `test_reml.py`) at log penalties −2, 6, 14 and 20 | `score`, `gradient`, `hessian`, `beta`, `covariance`, `edf`, `edf_total`, `scale` |
| `select_penalties` on the four mgcv data sets, and the set-aside and refit cases of `test_reml.py:991–1300` | `rho`, `score`, per session `beta`, `covariance`, `edf`; `in_choice`, `set_aside` (index, refits' statuses, `left_out`, `reason`), `at_boundary`, `grid_failures`, `n_outer_steps` |
| `refit_session` cases (`test_reml.py:1065`, `:1116`, `:1140`, `:1185`, `:1219`) | the refits' statuses, `left_out`, `rounds`, `ended`, `reason` |
| `fit_sessions` on `conftest.rate_rows` and the content rows, and the cases of `test_sessions.py:91–760` | every `FitRecord` field |
| `log_penalty_for_edf` on the rate sessions (`test_calibration.py:62`) | `log_penalty`, `median_edf`, `edf` per session |
| The slow simulations: `test_simulation.py::test_coverage_and_size` and `test_sessions.py::test_the_declared_penalty_recovers_a_planted_curve` (781), on their seeds | their summary numbers |

### Tolerances, and why

- Exact: `status`, `no_fit`, `left_out`, `left_out_round`, `running_away` and its order, `n_iter`,
  `in_choice`, `set_aside`, `at_boundary`, `grid_failures`, `support`, and the cohort counts.
- `beta`: `rtol=1e-9, atol=1e-11`. The measured difference on a 37,000 × 80 Poisson fit was
  1.6e-15 relative (appendix B). The margin covers ill-conditioned fixtures, and a wrong result
  (a sign, a dropped term) is far outside it.
- `vcov` and `covariance`: `rtol=1e-8`, `atol=1e-14 × max|V|`. `edf`: `abs=1e-9`.
- `loglik` and the REML score: `abs = 1e-9 × max(1, |score|)`. Gradient and Hessian:
  `rtol=1e-7, atol=1e-10`.
- Selected `rho`: `abs=1e-7`, except where the score is flat (`test_reml.py:921`, `:943`). There
  `rho` is weakly determined, so the score and the edf are compared instead.
- Decrement: both below the tolerance. A decrement at the stop (about 1e-19) is rounding and is
  not compared.

If a discrete output differs (for example `n_iter` by one at a decrement near 1e-8), that case
is examined and its fixture gets a written exception. The tolerance is not widened.

### Tests that already pin behavior independently of the NumPy solver (keep as they are)

- mgcv: `test_reml.py::test_matches_mgcv_fixtures` (474),
  `test_collinearity.py::test_matches_mgcv_concurvity` (124).
- neurospatial: `test_reml.py::test_one_penalty_matches_neurospatial_fixture` (496).
- finite differences: `test_likelihoods.py:66`, `test_reml.py:142`, `:1457`.
- closed forms and hand cases: `test_newton.py:34`, `:59`;
  `test_reml.py::test_gaussian_scale_matches_closed_form` (344).
- invariances: duplication (`test_reml.py:322`), outcome scale (`:539`), softmax constant
  within groups (`test_likelihoods.py:156`).
- simulations: `test_simulation.py:106`, `test_bands.py:64`, `test_sessions.py:781`.

### Tests that compare with code that will disappear

- `test_no_penalty_equals_current.py` compares with statsmodels' GLM and with
  `fit_conditional_poisson`. Those are legacy fitters that the plan removes in its own phases
  (`legacy-inventory.md`, row 7-4), not the NumPy solver. They keep passing through the port.
  Their C13 fixture is already planned there.
- No test in `tests/analysis/regression/` imports a NumPy-only function of the solver by name.
  Some build references inline with `np.linalg.solve` and `np.linalg.inv` (`test_newton.py:150`,
  `test_reml.py:199`). These are independent oracles and stay.
- Outside the package, `tests/analysis/pfc/test_batched_glm.py` compares `batched_glm` with
  statsmodels. It leaves in the mPFC phase with `batched_glm` itself, after its fixture is
  committed (`phase-8-mpfc.md`, retirement step 3).

### A real-data slice (not committed)

Before and after, run three sessions (one each of chimi, j16 and wilbur) through the rate and
content models at their declared penalties, with `scripts/checks/compare_regression_reference.py --real`.
Its table goes in the PR. It uses the same tolerances.

## 10. Performance baseline

What `docs/design/regression_fitter.md` ("Cost", line 643) already records:

- 2026-11-20, synthetic sessions: 242 sessions × 40,000 bins, a Poisson model with two smooths
  and 56 coefficients.
  - One pooled evaluation: 8 s (score only, warm start) to 32 s (with gradient and Hessian,
    cold start).
  - One penalty selection (25 grid points and 4 Newton steps): 6.3 min, with about 5 cores busy
    and 4.5 GB resident.
  - 82 % of an evaluation's time is `X'WX`.
- 2026-11-21, after the move to the eigenbasis, one session (40,000 × 53, one thread, warm,
  median of 5): 54–72 ms for the score, 165–191 ms with derivatives.
- C11 (`shared-contracts.md`): about 0.08 s per two-clock rate fit and 0.03 s per content fit.

What was measured here (under load, one core, appendix B):

- Gram matrix alone, 37,000 × 80: NumPy (OpenBLAS) 21 ms; XLA float64 39 ms.
- Whole fit: NumPy 0.26 s (7 steps and the last step); JAX 0.41 s (8 steps), plus 0.61 s of
  compilation.

So on one CPU thread **the JAX fitter is likely slower**, by up to about 1.5 to 2 times in the
dominant product. On one thread the port is not a speed-up. Multi-threaded XLA and the GPU are
unmeasured.

What to measure before and after, with the same cases, by `scripts/checks/time_regression_fitter.py`
(new). It reports wall time and peak memory (`/usr/bin/time -v`), the median of 5, and the
compile time separately (cold) from the run (warm). It is run on an idle machine, or labeled as
run under load.

1. One session fit at fixed penalties, for each family: rate (37,000 × 80), content (a
   typical session), Bernoulli, Gaussian. One thread, then all threads.
2. `fit_sessions` over the 30 synthetic sessions of `test_sessions.py:781`.
3. `log_penalty_for_edf` over the same 30 sessions (about 18 passes).
4. `select_penalties` on the "Cost" case (242 × 40,000, two smooths): CPU on one thread, CPU on
   all threads, and one free A100.
5. `uv run --frozen pytest tests/analysis/regression`, with a cold and a warm compilation cache.
6. For mPFC, the three baseline sessions of `phase-8-mpfc.md`.

No speed claim is made without these numbers.

## 11. Risks

| Risk | How likely | How it would show | Guard |
|---|---|---|---|
| A kernel runs in float32 because a call missed the float64 scope | likely without the guard | decrements that never fall below 1e-8, "step limit reached", coefficients off by 1e-7 | trace-time dtype check in every kernel; the scope tests of section 3 |
| The float64 flag leaks to the decoder | low with the scoped policy | decoder memory doubles; memoryless repeats differ | test: the flag is unchanged after a fit in a thread, and arrays are still float32 in the caller |
| A CPU runner without `JAX_PLATFORMS` takes every card | medium (the GPU environment on zephyr has the CUDA plugin) | other people's jobs run out of memory | the device guard (section 8); a test with the variables removed expects the error |
| A status flips near a threshold (decrement 1e-8, eta step 0.1, identifiability 1e-12, running-away ties) through new rounding | low to medium | a fixture's exact field fails | exact fixture fields; the tie rule; a written exception for each case examined |
| Cholesky failure is a NaN in JAX, not an exception | certain | a non-positive-definite `A` read as a step | `chol_ok` from `isfinite`; `test_newton.py:204`, `:329` |
| Too many compiles (many shapes) | likely without buckets | slow suite and slow calibration | buckets; a test that fits 10 row counts and checks the number of compiled entries is at most the number of buckets |
| Padding changes results (NaN from an empty group, an `-inf` offset, a shifted log-likelihood) | medium (NaN from an `-inf` offset measured) | NaN scores, or `F` off by a constant that changes the last-step rule | weighted sums; a test that padded equals unpadded to 1e-12 for each family, including a softmax with an empty dump group |
| CPU slower than NumPy | likely on one thread (measured 1.85× in the Gram matrix) | longer calibration and simulation runs | the benchmark of section 10; mitigations are all threads, a spawn pool, or one free GPU for sweeps |
| A fork after JAX has started deadlocks | medium (`simulate_declared_penalty.py:481` forks) | a hung pool | spawn pools; a repository test that no script calls `get_context("fork")` |
| A patched constant is ignored by a compiled kernel | medium | `test_reml.py:594`, `:600` fail, or pass for the wrong reason | constants passed as arguments; a test that changes `MAX_STEP_HALVINGS` after the first compile and sees its effect |
| GPU results do not repeat | medium on GPU | two GPU runs differ in the last digit | deterministic XLA flags; the bitwise-repeat check of section 8 |
| A JAX upgrade changes the numbers | low (pinned in `uv.lock`) | fixture tests fail after `uv lock` | the fixture tests |
| Removed code leaves orphans | low | stale imports or docs | the `git grep` gate and the repository test of Phase 6 |

## 12. Phased task list

Each phase is one PR, merged and tested on its own, in this order. The lead's rule holds in
every phase: what a phase replaces, it removes, and what tests need of it is kept as fixtures.
The NumPy solver therefore goes piece by piece. The last phase removes what is left and proves
that nothing is.

**Phase 0: capture behavior and cost (no change to results).**
- Write `write_regression_reference.py` and commit the fixtures of section 9.
- Add `tests/analysis/regression/test_reference_fixtures.py`, which compares the fitter with the
  fixtures using the section 9 tolerances. It passes on NumPy.
- Write `time_regression_fitter.py` and record its numbers in "Cost".
- Run the real slice and keep its output in a scratch folder.
- Gate: the full regression suite and the new fixture test.

**Phase 1: the simulation engine** (`analysis/regression/simulation.py`, if it is not already
merged by then).
- It draws outcomes from each likelihood with a NumPy `Generator`, from a given truth on a
  given design, and runs `fit_sessions`. The draws do not depend on the fitter.
- The same seeds then give the same data before and after the port, which turns the engine
  into a whole-pipeline regression check of bias and coverage. It also makes the rule hold
  that simulations and real fits run the same code.
- Its outputs on fixed seeds join the Phase 0 fixtures.
- Gate: planted effects recovered; its fixture.

**Phase 2: JAX plumbing (no change to results).**
- The `in_float64` decorator, the trace-time dtype guard, the device guard, row and group
  buckets with masks, and the XLA thread flags in `runs/threads.py`.
- Every fitting runner calls `default_jax_platforms("cpu")` and uses spawn pools.
- Gate: the float64, device and padding tests of sections 3, 8 and 11, and the fork test.

**Phase 3: the fit at fixed penalties in JAX.**
- The likelihood kernels, all four families, with their derivatives and the softmax segment
  sums.
- The batch-first Newton control with the iteration and "evaluate at beta" kernels.
- The fixed-penalty REML summary (score, `A^-1`, edf, `log|A|`), the profiled Gaussian, and
  `collinearity`'s public weights.
- Remove the NumPy formulas these replace: the likelihood bodies, `_newton_system`,
  `_objective`, `_covariance_and_edf` and the score in `_reml_in_basis`.
- The REML gradient and Hessian keep their present assembly, calling the JAX derivative
  kernels through the likelihood methods.
- Gate: the fixture test with exact discrete fields; the whole `tests/analysis/regression`;
  the autodiff test; the compile-count test; the Phase 1 simulation fixture; the real slice
  (in the PR); benchmarks 1 to 3 and 5 recorded.

**Phase 4: the REML derivatives in JAX.**
- One jitted kernel per family for the gradient and Hessian in rho. The `cho_solve` loops of
  `_reml_in_basis` and `_gaussian_profiled` are removed.
- The outer search (`_search`, `_grid_start`, `_screen`, refits) stays in Python.
- Gate: the mgcv and neurospatial fixtures, finite differences, the REML fixtures,
  `test_simulation.py` (slow), and benchmark 4 on CPU.

**Phase 5: the GPU check.**
- `check_regression_gpu.py` on one free A100 (read `nvidia-smi` on zephyr and breeze first,
  and tell the other session), with deterministic flags. GPU against CPU within the section 9
  tolerances; two GPU runs bitwise equal; benchmark 4 on the GPU.
- Write the device policy into `docs/design/regression_fitter.md`.
- The code is the same. Only the measurement and the document are new.

**Phase 6: remove the rest of the NumPy solver** (the explicit last phase of this series).
- Remove every remaining NumPy numerical path in `likelihoods.py`, `newton.py` and `reml.py`
  (`scipy.linalg` imports, NumPy formulas left over), and `compare_regression_to_current_fitters.py`'s
  reliance on them if the legacy fitters are gone by then.
- Keep `write_regression_reference.py` only as the record of how the fixtures were made, or
  remove it with a README note (lead's choice, Q3).
- Add a repository test: no `scipy.linalg`, `np.linalg.cholesky`, `np.linalg.solve` or
  `np.add.reduceat` in those three modules.
- Rewrite "Cost", the precision section and the module table of
  `docs/design/regression_fitter.md`. Add one row per removal in
  `docs/history/2026-module-moves.md`.
- Gate: the full suite, the `git grep` gate (C13), and the fixture tests.

**After the series: the mPFC models, in the plan's Phase 8.**
- The batched entry over the units of one session with a shared design: vmapped kernels or an
  `einsum` over units, chunked units and rows, and the pooled REML over units on the Phase 4
  kernels.
- `batched_glm`'s IRLS is replaced and removed. Check 6 becomes GPU float64 against CPU float64.
- This waits for the open question on the mPFC penalty (`phase-8-mpfc.md`, "OPEN (lead): mPFC and history penalties").

**Why REML is ported now (Phases 3–4) rather than in the mPFC phase:**
- The per-session path already goes through REML's fixed-penalty pieces (`refit_session` →
  `_session_score` → `reml_and_derivatives`, `reml.py:1568`, `:954`, `:671`).
- The NumPy likelihood derivatives that the search reads are removed in Phase 3.
- The mgcv agreement and the coverage simulation run through `select_penalties`, and they are
  the strongest independent checks the port has.

What waits for the mPFC phase is only the batched-over-units variant. Its shapes, its memory,
and whether REML over units is used at all depend on the lead's mPFC decision.

## 13. Questions only the lead can answer

1. **A CPU slowdown.** On one thread, XLA's float64 `X'WX` was 1.85 times slower than OpenBLAS
   (measured under load). Is a slower CPU fit acceptable for one fitter?
   Recommended default: yes, as long as a whole model run stays within C11's 30-minute rule;
   otherwise calibration and simulation sweeps use one free GPU after the usual check.
2. **Float32 on the GPU.** Should the plan's float32 GPU path for mPFC, and its check 6, be
   dropped, with float64 on the GPU too? Recommended default: drop them.
3. **The acceptance rule.** Is "equal to the NumPy fixtures within 1e-9 relative, with every
   status, column left out and count exactly equal" the acceptance of the refactor, rather than
   bitwise equality, which the XLA products cannot give? Also, should the fixture-writing script
   be kept after Phase 6? Recommended default: yes to the rule; keep the script, frozen, with
   its commit.
4. **The order.** Should the simulation engine land before the port (Phase 1), so that it
   checks the port? Recommended default: yes.

## Appendix: experiments

All were run with `JAX_PLATFORMS=cpu`, `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`,
`XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"`, `taskset -c 0`,
and `/cumulus/edeno/continuum-swr-replay/.venv/bin/python`. They were taken under load (load
average 30 to 40 on 96 cores). The scripts were scratch files and are not kept. The code below
is their essential part.

**A. The float64 flag and Cholesky.**
```python
a = np.array([1.0 + 1e-12, 2.0])
jnp.asarray(a)                      # float32, a[0] - 1 == 0.0, no warning
jnp.asarray(a, dtype=jnp.float64)   # float32, warning "Explicitly requested dtype float64 ... truncated"
f = jax.jit(lambda x: x * 2); f(a)  # float32
with jax.enable_x64(True): f(a)     # float64; f._cache_size() == 2 afterwards
jnp.linalg.cholesky([[1, 2], [2, 1]]) under x64  # [[nan, 0], [nan, nan]]: NaN, no exception
jax.scipy.linalg.cho_factor(same)                # NaN likewise; a NaN input also gives NaN
# near-singular [[1, 1], [1, 1 + 1e-15]]: JAX L[1,1] = 3.33200094e-08, SciPy 3.33200094e-08
```

**B. One Poisson fit, 37,000 × 80, penalty on 40 columns.** It used `lax.while_loop` for the
Newton steps and an inner `lax.while_loop` for halving, mirroring `newton.py`'s rules except
the last step and the statuses.
```
compile 0.61 s, jax run 0.444 s / 0.412 s (8 steps), numpy 0.261 s (7 steps)
max relative difference of beta: 1.6e-15
new row count (n - 1): 1.03 s including compile
rows padded with offset -inf: F = nan
```

**C. Products and derivatives.**
```
X'WX 37,000 x 80: numpy 21.3 ms, jax 39.3 ms
Poisson information: by hand 39.2 ms, -jax.hessian(loglik) 62.0 ms, relative difference 5.8e-19
softmax information, 4,000 events x 9 arms x 20 columns:
  segment sums by hand 4.5 ms, autodiff 23.5 ms, dense einsum 5.9 ms
  hand vs autodiff 1.1e-15, hand vs dense 1.6e-16, JAX hand vs the repository's NumPy 9.3e-16
vmap of X'WX over 4 sessions: 142.7 ms (one: 39.3 ms)
```

**D. Batching and shapes.**
```
vmap of a while_loop with limits [3, 7, 1]: per-member counts [3, 7, 1]
segment_sum with a traced num_segments: ConcretizationTypeError; static: fine
padded rows sent to a dump segment: [2, 2, 1]
jit entries after row counts 100, 101, 100, 128: 3
```

**E. Dispatch overhead.**
```
a jitted call and a host read, tiny arrays: 25 us; with NumPy inputs: 35 us;
with an enable_x64 context entered per call: 29 us
```

**Sources read.** JAX 0.8.0: `jax/_src/config.py` (`enable_x64` at 1719–1744,
`default_matmul_precision` at 1807–1830, the trace context at 225–250), `jax/__init__.py:50`,
`jax/_src/numpy/linalg.py:82` (`cholesky`, `symmetrize_input=True`),
`jax/_src/scipy/linalg.py:47`, and the `jaxlib/cpu/_lapack.so` strings (`scipy.linalg.cython_lapack`).
Repository: the modules and tests cited above; `runs/device.py`; `runs/threads.py`;
`analysis/pfc/batched_glm.py:100–200` and `:425–640`; neurospatial
`encoding/_glm_jax.py:1–300` and `_backend.py:95–206`.
