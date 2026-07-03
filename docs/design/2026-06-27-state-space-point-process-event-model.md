# Spec: State-space point-process model for non-local / SWR event trains

Status: design (not yet implemented; extends Family B — see
`.claude/docs/plans/per-bin-rate-models/`)
Date: 2026-06-27
Reference: Czanner, Eden, Wirth, Yanike, Suzuki & Brown (2008), "Analysis of
between-trial and within-trial neural spiking dynamics," J Neurophysiol
99:2672–2693 (doi:10.1152/jn.00343.2007). In the lab corpus
(`gs://franklab_pdf/pdf_flat/Czanner et al_2008.pdf`).

## Goal

Model the **event onset train** (detected SWR / non-local-no-SWR / non-local-w-SWR
onsets, treated as a point process) with a single state-space generalized linear
model (SS-GLM) that unifies the three structures we have characterized piecemeal:

1. **Within-stop time course** — the quiescence/consummatory profile (rate rises
   ~2–4 s into immobility / after reward), currently a fixed spline `bs(tc)`.
2. **Self-history** — refractoriness (<60 ms) and burstiness (~100 ms),
   currently fixed `onset_lag1..L`.
3. **Between-stop / across-session dynamics** — slow drift of the rate and its
   time course across trials (experience, satiety, learning, value), which we
   have only approximated with explicit `reward_lag` predictors and
   `recent_trial_reward_rate`. **This is the piece the SS-GLM adds**: the
   within-stop coefficients evolve as a latent random walk across stops, so the
   slow dynamics are estimated data-driven rather than assumed.

It also gives principled goodness-of-fit (KS / time-rescaling) — something the
per-bin Poisson GLMs lack.

## The Czanner SS-GLM (what we adapt)

Conditional intensity for trial `k`, within-trial bin `l` (width Δ):

```
λ_k(lΔ | θ_k, γ, H) = exp{ Σ_r θ_{k,r} g_r(lΔ) } · exp{ Σ_j γ_j n_{k,l-j} }
```
- **Stimulus / within-trial:** basis `g_r(lΔ)` (splines, or unit pulses → the
  PSTH is the special case), coefficients `θ_k` *per trial*.
- **Spike history:** `Σ_j γ_j n_{k,l-j}` over the last `J` bins; `γ` fixed across
  trials, captures refractoriness/bursting.
- **Between-trial state:** `θ_k = θ_{k-1} + ε_k`, `ε_k ~ N(0, Σ)` — a random walk
  that lets the within-trial response drift smoothly across trials.
- **Fit:** approximate EM. E-step = point-process filter + fixed-interval
  smoother for the latent `θ_k` trajectory; M-step = ML for `γ`, `Σ`, `θ_0`.
  GOF via KS plots (time-rescaling theorem); AIC for model order.

## Adaptation to our data

| Czanner | Ours |
|---|---|
| neuron's spike train | event-onset train of one **population** (SWR / non-local-no-SWR / …) |
| "trial" `k` | an **immobility stop** (or a **reward delivery**, alignment TBD — see Open Q1) |
| within-trial time `lΔ` | `time_since_stop_onset` (or `time_since_pump`), at Δ = 15 ms |
| stimulus basis `g_r` | spline over the within-stop clock — the **quiescence/consummatory time course** |
| `θ_k` per trial | the time-course coefficients for stop `k` (amplitude + shape) |
| spike history `γ_j n_{k,l-j}` | **self-history** `onset_lag_j` — refractoriness/burstiness (already validated: refractory <60 ms, bursty ~100 ms) |
| between-trial walk `θ_k=θ_{k-1}+ε_k` | **across-stop drift** of the rate/time-course (experience, satiety, value) |

Extensions beyond vanilla Czanner, optional:
- **External covariates in the intensity** (the GLM term Czanner allows): add the
  per-stop `reward_lag1..k` and behavioral state as fixed-effect terms, so the
  *latent* state captures only the unexplained slow drift on top of the explicit
  reward kernel. This directly tests "is there across-session drift *beyond*
  reward history?" (e.g., satiety, experience).
- **Reward as the trial-aligning event** (align `lΔ` to `time_since_pump`) turns
  the within-trial component into the consummatory PSTH and the between-trial
  walk into the across-reward learning curve.

## Relationship to Family B (what's new vs reused)

- **Reused:** the onset extraction + `is_swr` split + 15 ms within-stop bins +
  self-history lags are exactly the `M7` design
  (`aggregate_to_timebins`, [Family B plan](../../../.claude/docs/plans/per-bin-rate-models/shared-contracts.md)).
  The SS-GLM consumes the same per-(stop, bin) design.
- **`M7` = the SS-GLM with `Σ = 0`** (θ fixed across stops). So Family B is a
  nested special case — fit `M7` first; the SS-GLM asks whether relaxing the
  fixed-θ constraint (letting it drift) improves AIC/KS.
- **The EM estimator is mostly already built** — in `/cumulus/edeno/state_space_practice`,
  a production, JAX, pytest-covered state-space library (importable). The
  point-process Laplace-EKF filter + RTS smoother (the E-step, = Eden & Brown
  2004) and the random-walk EM M-step exist and are tested; there is even a
  pedagogical `notebooks/2008_czanner_brown.ipynb` of exactly this model. So the
  SS-GLM is **not a from-scratch fitter** — it is an *adapter* feeding our event
  bins into that API, plus two small custom pieces (self-history columns + KS
  GOF). See the reuse table.

## Reuse vs build (anchored in `state_space_practice`)

| Need | Reuse / Build | Where |
|---|---|---|
| Point-process Laplace-EKF filter (E-step) | **reuse** | `state_space_practice.point_process_kalman.stochastic_point_process_filter` |
| RTS smoother on filter output (E-step) | **reuse** | `point_process_kalman.stochastic_point_process_smoother` |
| EM M-step for the random-walk dynamics (A≈I, Q) | **reuse** | `state_space_practice.kalman.kalman_maximization_step` (observation-agnostic — works on the Laplace posteriors) |
| within-stop spline basis `g_r` | **reuse** | `patsy.dmatrix("bs(tc, df=4) - 1")` (the library already depends on patsy) |
| EM convergence / numerics | **reuse** | `state_space_practice.utils.check_converged`, `psd_solve`, `symmetrize` |
| optional M-step for baseline/history weights | **adapt** | `switching_point_process.update_spike_glm_params` (Newton + line search) |
| **self-history term** | **build (trivial)** | the library intensity is linear `log λ = Z @ x`; append lagged onset-count columns to `Z` (state `x = [θ_spline; γ_history]`) — no new term needed, history rides in the design matrix |
| **time-rescaling / KS goodness-of-fit** | **build** | `scipy.stats.ks_1samp` on rescaled inter-event intervals from the cumulative intensity (library doesn't provide it) |
| **per-stop "trial" structure** | **build** | reset `init_mean`/`init_cov` at each stop boundary (the library assumes a continuous series); stops are the trial unit |
| **event-data adapter** | **build** | our onset bins → `spike_indicator (n_time,)`; the `aggregate_to_timebins` design (Family B) → `design_matrix`; same 15 ms grid |

Dependency note: add `state_space_practice` as a dependency and set
`jax.config.update("jax_enable_x64", True)` (required for Laplace-EKF stability).
GPU optional (`state_space_practice[gpu]`), not needed at this data scale.

So the real work is the **adapter + KS GOF + per-stop reset + validation against
`M7`** — substantially less than implementing the filter/smoother. The spec's
heavy lifting is reuse, not rebuild.

## Raster plots — the visualization layer

The SS-GLM is a model *of* the raster, so the basic rasters come first (and are
worth having regardless of the model). Each is: rows = trials, x = time, one tick
per event **onset**, faceted/colored by population.

1. **Stop-onset-aligned raster** — rows = stops, x = `time_since_stop_onset`
   (0 → 4 s truncation), ticks = onsets. The fundamental view; the within-stop
   spline is the column-wise rate of this.
2. **Reward-aligned raster** — rows = reward deliveries, x = `time_since_pump`,
   ticks = onsets. The consummatory PSTH's raster (the `rate_timecourse_vs_pump`
   figure is its column average).
3. **Row ordering reveals the structure** (same data, different sorts):
   - by **stop duration** → shows the heavy-tailed/bimodal duration structure +
     right-censoring (short stops = short rows).
   - by **reward outcome** (rewarded block above omitted) → the reward effect by eye.
   - by **trial number / session time** → the **between-trial axis**: any
     across-session drift (experience/satiety) shows as a gradient down the rows —
     this is precisely what the SS-GLM's `θ_k` trajectory quantifies.
4. **Per-population facets** — SWR / non-local-no-SWR / non-local-w-SWR side by
   side (or color-overlaid on one axis) → the dissociation as a raster: SWR rows
   dense ~3 s post-reward, non-SWR rows flatter/sustained.
5. **Single-stop population raster** — one long stop, all event types on one time
   axis → within-stop co-occurrence / coupling (and the self-history clustering).
6. **Whole-session strip** — x = session time, ticks = events → the macro train
   with the immobility/movement gaps (reminds that history/within-trial structure
   lives inside stops, not across the gaps).

Companion summaries to each raster: the **PSTH** (column-mean rate, = the
within-stop spline) and the **ISI / autocorrelogram** (= the self-history kernel).
These three (raster + PSTH + autocorrelogram) are the standard point-process
triptych and are the natural figure for the model.

## Scientific questions it answers (beyond Family B)

- **Does the event rate / time course drift across the session, beyond reward
  history?** The latent `θ_k` trajectory (with reward as a fixed covariate)
  isolates experience/satiety/learning drift — the "duration is a behavioral
  signal" layer, formalized. (Connects to Jackson 2006 / Csicsvari 2007 experience
  effects and Kaya 2024 satiety.)
- **Do SWR vs non-SWR events have different across-session dynamics?** Fit per
  population; compare the `θ_k` trajectories / `Σ`. A new axis of the dissociation.
- **Principled GOF** via KS/time-rescaling — tells us whether the spline+history+
  drift model actually captures the point process, or whether something is missing.

## Open questions / pitfalls

1. **Trial unit = stop vs reward delivery.** Stops vary in duration (Czanner
   trials are fixed length) → the within-stop window is truncated to a common
   span (e.g., 0–4 s) and short stops are right-censored (fewer bins). Reward-
   aligned framing gives fixed-ish windows but only covers rewarded stops. Decide
   per question (consummatory dynamics → reward-aligned; general occurrence →
   stop-aligned). Likely fit both.
2. **State dimension.** `θ_k` is R-dimensional (one per spline basis). A full
   R-dim random walk may be over-parameterized; start with a 1-D state (a scalar
   gain on a fixed time-course shape) before a full drifting shape.
3. **Sparsity + separation.** Events are sparse and self-history has perfect
   refractoriness at short lags (RR≈0) — the EM/filter must be numerically robust
   to this (the `M7` GLM already hits separation there; carry the same care).
4. **Cost.** EM over ~100–200 trials/session × per-bin point process × per
   population × 5 animals. Smoke-test one session first; the filter/smoother is
   O(trials × state² ) per EM iteration.
5. **Does the latent state beat explicit predictors?** The honest test is AIC/KS:
   `M7` (fixed θ, with reward_lag) vs SS-GLM (drifting θ). If the random walk
   doesn't improve fit, the explicit reward kernel already captures the slow
   dynamics and we report that (a clean negative is informative).

## Scope / sequencing

This is the **tier above Family B**: implement and validate Family B
(the `M0…M9` ladder) first — `M7` is the `Σ=0` baseline the SS-GLM must beat.
The rasters (above) are independent and worth building now as the descriptive
layer. With the EM machinery reused from `state_space_practice`, the SS-GLM
shrinks to an adapter + KS GOF + per-stop reset + AIC/KS validation vs `M7` —
still its own plan, but a small one. Sequencing: (1) rasters now; (2) Family B
`M7`; (3) only if AIC/KS on `M7` shows unexplained between-stop structure,
the SS-GLM adapter over `state_space_practice`.
