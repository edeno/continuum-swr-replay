# SpatialBanditTask behavioral models — generative-model reference

The latent variables we use as replay covariates (value, uncertainty, RPE,
value-updates, entropy) are produced by behavioral models fit to the spatial
bandit task in the **LorenFrankLab/SpatialBanditTask** repository (Comrie et al.).
This document describes those **models** — their structure, assumptions, and free
parameters — so the covariates can be interpreted correctly.

- **This doc = the models.** For the **column-by-column data dictionary** of the
  exported latent variables, see
  [`behavior_model_covariates.md`](behavior_model_covariates.md) (the
  `qvals_table.pkl` reference). The two are complementary: read this for "what does
  `betadist_var` *mean*," read that for "which column holds it."
- **Paper:** Comrie et al., *Hippocampal representations of alternative
  possibilities are flexibly generated to meet cognitive demands* (bioRxiv
  2024.09.23.613567; `references/nihpp-2024.09.23.613567v1.pdf`).

## Provenance & verification status

- **Source:** <https://github.com/LorenFrankLab/SpatialBanditTask> (`main`,
  read 2026-07-07). Model code is **Julia** (`code/*.jl`); the fitting harness is
  the `EM/` hierarchical package; analysis lives in `notebooks_metalearning/`.
- **How this doc was written:** by reading the repository source
  (`code/beta_bernoulli.jl`, `code/hmm.jl`, directory listings, the `EM/`
  README) **and the paper Methods directly** (Comrie et al. pp. 24–28,
  "Behavioral Hidden Markov Model" through "Behavioral model variables";
  `references/nihpp-2024.09.23.613567v1.pdf`). Details marked _[source]_ were
  read from the `.jl` files (via web fetch of the raw files — a summarizer read,
  so confirm an exact constant against the file before quoting it verbatim in a
  manuscript); _[paper]_ items are from the PDF Methods. The five items originally
  flagged for verification are now **resolved** — see
  [Resolution of open items](#resolution-of-open-items).
- **Which fit we actually consume** (from
  [`behavior_model_covariates.md`](behavior_model_covariates.md)): **all** behavioral
  covariates — value / uncertainty / RPE / value-update **and** the belief entropy
  (`beta_reward_entropy`, derived in-repo from the Beta posterior `α/β`) — come from
  the single **Beta-Bernoulli** fit `beta_stable_withleaf`, stored in DataJoint
  (`alison_rlmodel.BehaviorModelResults`), not re-fit in this repo. (The HMM
  `default_hmm` was **retired 2026-07-13**; its value/RPE/entropy covariates were
  replaced with beta equivalents — see §Model 2.)

## The shared decision architecture (belief core is the *only* difference)

The Beta-Bernoulli and HMM models are **not** two different theories of choice —
they share the entire value→action policy and differ **only in how per-leaf reward
probability is inferred**. _[source]_ Everything in this section is common to both.

Per port/leaf the model holds an estimated reward probability `Q` (the belief
core supplies it — §Beta / §HMM below). Choice is a softmax over these values with:

- **Inverse temperatures** `βgo`, `βstay`, `βleaf` — separate temperatures for the
  go/stay stem decision and the leaf decision.
- **Depletion (satiety)** `Qdep = Q · depletion`, where `depletion` *decreases* with
  repeated choices of the same port (a within-session satiety/adaptation term),
  governed by `depletion_factor`.
- **Within-stem leaf weighting** `γ2`: the value of staying in a patch is
  `(1−γ2)·Qdep[other leaf] + γ2·Qdep[current leaf]`.
- **Choice biases**: `stay_bias`, `turn_bias`, `leaf_turn_bias`,
  `spatial_bias[3]`, `leaf_spatial_bias[3]` (directional/handedness offsets).
- **Leaf decision only on switch trials**; on stay trials the leaf is fixed by
  the stay policy. _[source]_

The paper's published choice model (Comrie et al. p26) is: a **softmax over the
3 patch values** — `V_current = βstay·Qpatch + b_stay`, alternatives
`V = βgo·Qpatch (+ b_turn)` — followed on switch trials by a **per-port softmax**
with temperature `βport` and per-patch `b_portbias`. Note the repo code's shared
layer is a **superset**: it additionally carries the `depletion` (satiety) and
`γ2` terms above (and they surface as the `depletion_*` columns in
`qvals_table.pkl`), which the paper's Methods choice-model description does not
list. In code the paper's `βport` is named `βleaf`. _[paper p26 + source]_

**Consequence:** any divergence between the two models in the exported quantities
is attributable **purely to the belief representation** (independent Beta trackers
vs. a joint latent-state filter), not to a different choice model. This is what
lets us treat "Beta value vs. HMM value" as a clean construct contrast.

## Model 1 — Beta-Bernoulli (`beta_stable_withleaf`): value & uncertainty source

Independent conjugate belief per port×leaf. **This is the fit that supplies the
`Q`/`betadist_mu` value columns and the `betadist_var` uncertainty columns.**

**Assumptions** _[source]_:

1. **Binary reward likelihood** — each poke outcome is Bernoulli(`p`); no reward
   magnitude.
2. **Independence across ports** — each port's belief updates *only* from its own
   outcomes. The model has **no representation of the task's patch/block structure**;
   learning about one port does not move belief about any other. (This is the
   defining contrast with the HMM.)
3. **Conjugate Beta belief** — belief over each port's `p` is `Beta(α, β)`,
   initialized `Beta(a_baseline, b_baseline)` = **`Beta(1,1)`** (uniform) by default.
   `p` is treated as **continuous** on [0,1].
4. **Update rule** — reward → `α += 1`; no-reward →
   `β += d·(1 − target)` where `d = depletion[port]`,
   `target = μ(1−d)/(1−d·μ)`, `μ = α/(α+β)` — a depletion/expected-value-scaled
   increment, **not** a plain `+1`. _[source, verbatim]_
5. **Forgetting** — between trials `α, β` decay toward baseline by
   `(1 − beta_decay·0.5)` (this is the "stable" in the fit name); at session
   boundaries belief partially resets:
   `(1 − retain_belief)·baseline + retain_belief·previous`.
6. **Value = posterior mean** `Q = α/(α+β)` (myopic expected reward prob, then
   passed through the shared depletion/policy layer).
7. **Uncertainty = posterior variance** `betadist_var = αβ / ((α+β)²(α+β+1))`.

**Belief-specific free parameters:** `beta_decay`, `a_baseline`, `b_baseline`,
`retain_belief` (plus the shared policy parameters above).

**Caveat that matters for our uncertainty covariate.** Because of forgetting
(assumption 5), an **unvisited** port's `α, β` decay back toward `Beta(1,1)`, whose
variance (1/12) is the **maximum**. So `betadist_var` rises toward its ceiling
*precisely because a port has not been visited recently* — the recency dependence
is **built into `beta_decay`**, not incidental. Empirically
`corr(betadist_var, visit_recency) ≈ +0.54`. Any use of `betadist_var` as an
"uncertainty" covariate must disentangle this from visit-recency (see
[`replay_rate_content_swr_synthesis.md`](replay_rate_content_swr_synthesis.md) /
the content-model uncertainty work).

## Model 2 — HMM (`default_hmm`)

> **RETIRED from the analysis (2026-07-13).** The paper standardized on the
> Beta-Bernoulli model (§Model 1); the HMM value/RPE/update/entropy covariates
> (`q_arm_hmm`, `hmm_*`, `reward_entropy`, `state_entropy`) were dropped from every
> model ladder and replaced with beta equivalents (belief entropy → the derived
> `beta_reward_entropy`). This section is retained as **background** — it documents
> a model that still lives in Alison's `hmm.jl`, not a covariate source the current
> analysis uses.

Joint latent-state belief over reward contingencies. (Historically supplied
`reward_entropy` and `state_entropy`.)

**Assumptions** _[source]_:

1. **Discrete latent contingency states** — each state is an assignment of a reward
   probability from the **three discrete levels {0.2, 0.5, 0.8}** to every one of
   the 6 ports (e.g. `ϕᵢ = [0.2, 0.2, 0.5, 0.2, 0.8, 0.5]`). The generic grid is
   `3⁶ = 729`, but the HMM uses **`n = 72`** — the *72 most-frequently-occurring*
   contingencies in the actual experimental design (`nstates = size(ϕ,1)`); Comrie
   et al. report results are "relatively insensitive to the exact choice of
   contingencies." _[paper p25 + source]_
2. **Markov switching** — a single free **hazard rate `h`** (the code's
   `volatility`): transition matrix `T` has diagonal `1 − h` and uniform
   off-diagonal `h/(n−1)`, so belief diffuses toward baseline each step
   (`α ← T α`). One rate for all transitions ⇒ **geometric dwell times / constant
   hazard**. _[paper p25–26 + source]_
3. **Bernoulli emissions** — reward | (state, port) is Bernoulli with the state's
   port probability (`ϕ`), fixed by the 72 contingency configurations.
4. **Exact ideal-observer inference** — after each outcome every state is reweighted
   by the outcome likelihood and renormalized (forward filtering): reward at port
   *j* → `αᵢ ← αᵢ · ϕᵢⱼ`; omission → `αᵢ ← αᵢ · (1 − ϕᵢⱼ)`, for all contingencies
   *i*. Row *t* holds the **pre-outcome** posterior; trial *t*'s outcome enters on
   row *t+1*. _[paper p26]_
5. **Value = state-marginalized reward prob** `Q = Σ_i α_i · ϕ_i` (the MSE-minimizing
   estimate), then through the shared policy layer. Patch value: the current patch
   takes its **upcoming port's** `Q`; the other two patches take the **mean of
   their two ports** `(Q_a + Q_b)/2`. _[paper p25]_
6. **Coupled ports** — because ports co-vary through the shared state, an inferred
   switch updates belief about **unvisited** ports too (the structural opposite of
   Beta's independence).

**Belief-specific free parameter:** the hazard rate `h`/`volatility` (plus the
shared policy parameters).

**Regime-level uncertainty is what we export:** `state_entropy = H[α]` over the 72
contingency states, and per-port `reward_entropy = −Σ_k p_k log p_k` over the port's
marginal reward distribution `[p(0.2), p(0.5), p(0.8)]` (paper p28, Fig 5D). **But
`hmm.jl` also computes a per-leaf posterior variance** —
`leaf_var = Σ_i α_i (ϕ_{i,leaf} − Q_leaf)²` (the variables `leaf_1_var`,
`leaf_2_var`), i.e. exactly the construct-consistent per-arm uncertainty. It is
simply **not pulled into `qvals_table.pkl`**, which takes only the two entropies
from the HMM fit. Unlike `betadist_var`, this per-leaf HMM variance updates for
**unvisited** ports through the shared state (assumption 6), so it is **not**
mechanically tied to visit-recency — which is why it, not `betadist_var`, is the
right per-arm uncertainty covariate to pair with HMM value. Recoverable from the
`hmm.jl` output, or by recomputing from the fitted posterior `α` and `ϕ`.
_[source: `hmm.jl` computes it; export status per behavior_model_covariates.md]_

## Model 3 — Q-learners (`qlearner.jl`, `q2learner.jl`): RL baselines

Model-free delta-rule reinforcement learning — single learning rate (`qlearner`)
and dual learning rate for reward/omission (`q2learner`). These are **baseline
comparators** in the model set (they carry the same shared policy layer, with a
learned `Q` updated by `Q ← Q + η·(r − Q)` instead of a belief posterior). We do
**not** consume their outputs as covariates; documented here for completeness.
_[source: file names/directory; internal update rule not line-verified — verify]_

## HMM variants in `code/`

The repo carries several HMM structural variants, used for model comparison in the
paper (they differ in how leaves/stems are allowed to co-vary):

| File | Structure |
|---|---|
| `hmm.jl` | **full joint** contingency-state HMM = the paper's *Global* model = the consumed `default_hmm` (§Model 2) |
| `hmm_independent.jl` | the paper's *Local* model: **six independent per-port HMMs** — only the visited port's belief updates each trial (paper p27, Fig 5A,B) |
| `hmm_baseline.jl` | baseline variant |
| `hmm_independentsamedist.jl` | independent + identically distributed ports |
| `hmm_independentsamedist_avgedturn.jl` | i.i.d. ports, turn parameter averaged |
| `hmm_partial_independence.jl`, `hmm_partial_independence2.jl` | partial cross-port dependence |

We consume **`default_hmm` = the Global joint HMM (`hmm.jl`)**. The paper's central
model comparison is **Global (joint) vs. Local (independent, `hmm_independent.jl`)**:
because the two differ *only* in whether unvisited ports update, it isolates the
question of **non-local value updating** — and the Global model wins (paper Fig 5B).
The remaining variants are additional comparators.

## Fitting harness — `EM/`

Hierarchical expectation–maximization for RL model fitting, after Huys et al. (PLoS
Comput Biol 2012). Julia (≥1.5). **The fits we consume are hierarchical and fit
separately per rat**: each free parameter varies **per day**, with the day-level
values drawn from a **per-rat population-level Gaussian**; the model is estimated by
EM with a Laplace approximation to the day-level marginal likelihood in the M-step
(paper p28). So the parameters behind our consumed fits are **per-day within
per-rat** — neither pooled across rats nor a single value per rat. Model comparison
(Global vs. Local, full vs. `βgo`/`βstay`-ablated) uses leave-one-session-out CV on
the population-level prior with a Laplace approximation for the per-day parameters.
_[paper p27–28]_

## Shared assumptions (both belief models)

- Reward is **binary** (Bernoulli), no magnitude.
- Value is **myopic** — expected *immediate* reward probability, **not** a
  temporally-discounted return over the track, despite the "Q" naming.
- Both are fit to **behavior** (choices + rewards) via the shared softmax policy;
  we then treat their internal beliefs as **what the hippocampus represents in
  replay** — a substantive assumption inherent in using them as neural covariates
  at all.
- **The models differ only in the belief core** (§shared architecture), so the
  Beta-vs-HMM value divergence we observe on non-current arms is a clean
  belief-representation contrast, not a policy artifact.

## Resolution of open items

All five items originally flagged were resolved 2026-07-07 from the paper Methods
(Comrie et al. pp. 24–28) and the raw source:

1. **State count — 72, *and* 729.** Both are correct at different levels: the generic
   configuration grid is `3⁶ = 729`, but the HMM's state set `ϕ` is the **`n = 72`
   most-frequently-occurring contingencies** in the actual design (`nstates =
   size(ϕ,1) = 72`); "relatively insensitive to the exact choice" (paper p25). The
   72 in [`behavior_model_covariates.md`](behavior_model_covariates.md) is right.
2. **Beta no-reward update — resolved verbatim.** `β += d·(1 − target)`,
   `target = μ(1−d)/(1−d·μ)`, `μ = α/(α+β)` (§Model 1).
3. **Per-port HMM uncertainty — computed, not exported.** `hmm.jl` computes the
   per-leaf posterior variance (`leaf_1_var`/`leaf_2_var`); it is simply not pulled
   into `qvals_table.pkl`. Recoverable from the fit or recomputable from `α`, `ϕ`
   (§Model 2).
4. **Fit granularity — hierarchical, per-day within per-rat**, fit separately per
   rat; EM + Laplace (§EM harness, paper p28).
5. **`default_hmm` ↔ variant — the Global joint HMM (`hmm.jl`)**; the paper's Local
   model is `hmm_independent.jl` (§HMM variants).

**One residual naming inference:** that the DataJoint fit label `default_hmm` binds
to `hmm.jl` (Global) specifically is inferred from the model description matching,
not from a code-level name map — solid but not code-verified. The paper-vs-code
choice-model delta (repo adds `depletion`/`γ2`) is documented in §shared
architecture, not an open question.
