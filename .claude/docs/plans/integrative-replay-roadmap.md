# Integrative roadmap: how replay rate, content, duration, and behavior intersect

**Status:** Roadmap (not a build). Each rung below becomes its own spec → plan
when picked up. The concrete next step already written is the
[joint-replay-departure](joint-replay-departure/PLAN.md) plan (the rate×duration
rung).

This maps the larger object behind the per-bin rate work: a stop is not just a
replay-rate process but the intersection of **four axes**, and the scientific
payoff is in their couplings. The governing principle (same as the joint-
departure plan): **build margins → pairwise links → capstone, gated** — never fit
the high-dimensional integrative model until the pairwise rungs show which
couplings are real.

## The four axes (and what exists today)

| axis | what it is | current status / infra |
| --- | --- | --- |
| **Rate** | how often replay occurs | Family B per-bin conditional intensity (`MODEL_PER_BIN_RATE`, `fit_per_bin_rate_model`) ✅ |
| **Content** | what replay represents (which arms/paths) | arm-level content hierarchy (`MODEL_HIERARCHY`, `summaries.posterior_to_arm_marginal`, decoder posteriors) ✅ separate |
| **Duration** | dwell / when the animal leaves | joint-departure plan (departure hazard) 📋 |
| **Behavior** | choices, value, learning, exploration | bandit value/RPE from the StateScript contingency (`contingency_table`), `MODEL_CHOICE`, the uncertainty-rate covariates ◐ partial |

## Cross-cutting method principles (apply to every rung)

- **Shared substrate:** the `(stop × 15 ms bin)` design (`aggregate_to_timebins`)
  is the common scaffold; per-stop covariates (value/RPE/uncertainty from
  `contingency_table`, content summaries per event/stop) attach to it; the
  per-stop **frailty/latent** in the joint-departure plan is the seed of the
  capstone latent state; time-rescaling GOF (`non_local_detector`) checks any
  point-process piece.
- **Reflect vs drive (the recurring causal trap):** replay correlating with
  value/behavior ≠ replay *driving* it. Use mediation framing (does replay add
  predictive power beyond the value state) and watch colliders (conditioning on
  duration or on the realized choice).
- **Circularity guard:** content is derived from the same decoder that defines
  the events — reuse the claims-table circularity audit (spike-supported reality,
  distance controls) before any content claim.
- **SWR vs non-SWR as the through-line:** every rung is read split by SWR
  overlap; the working hypothesis is SWR-coupled = consummatory/value read-out,
  non-SWR = deliberative/alternative-sampling.

## The six pairwise rungs (method sketches)

### 1. Rate × Behavior — *largely done*
- **Q:** does value / RPE / uncertainty modulate replay rate, and differ by SWR?
- **Model:** per-bin Poisson conditional intensity, per session × population:
  `log λ_R(t) = log(exposure_t) + bs(tc) + Σ β·[value, RPE, uncertainty] +
  history`. Effect = `exp(β)` rate ratio; the SWR dissociation = β differing
  across populations.
- **Method:** per-bin rate ~ value/RPE/exploration covariates, per population
  (the `nonswr-uncertainty-rate` plan). **Result:** SWR-coupled rate is
  value/RPE-gated (RR 5–7×), non-SWR is value-independent; non-SWR rises with
  exploration. ✅
- **Next:** the block-anchored / RL-model-fit refinements already noted there.

### 2. Rate × Duration — *planned*
- **Q:** is the within-stop time course real vs duration composition; do replay
  and dwell co-drive?
- **Model:** two coupled point processes per stop — replay Poisson
  `log λ_R(t) = log(exposure_t) + bs(tc) + βx` and departure discrete hazard
  `cloglog h_L(t) = bs(tc) + γx`, optionally coupled by a per-stop frailty `u`
  (`λ_R·u`, `h_L·u^ξ`, `u ~ Gamma(1/θ,1/θ)`; `θ=0` = independent). See the
  [joint-departure shared-contracts](joint-replay-departure/shared-contracts.md).
- **Method:** the [joint-replay-departure](joint-replay-departure/PLAN.md) plan
  (departure hazard + optional shared frailty). 📋

### 3. Content × Behavior — *the mediation rung*
- **Q:** does replay **content** predict the next **choice** beyond a value
  baseline (replay as a decision variable), and does it differ by SWR?
- **Model:** multinomial / conditional logit over the next-arm choice:
  `P(choice = a) ∝ exp( α·V(a) + β·Content_a + δ·Spatial_a )`, where `V(a)` =
  nominal/experienced value of arm `a`, `Content_a` = replay mass on `a` this
  stop (split `Content_a^{SWR}` vs `^{non-SWR}`), `Spatial_a` = distance/
  availability baseline. **Mediation test** = held-out log-likelihood gain of
  `+Content` over the value-only model (replay as decision variable beyond value).
- **Method:** conditional logit on the next-arm choice (`MODEL_CHOICE`
  framework): `P(next_arm) ~ value(arm) + content_replayed(arm) + spatial
  baseline`. Mediation: does `content_replayed` add predictive power beyond
  nominal/experienced value? Per SWR split. A within-stop temporal version: does
  *late*-stop content (pre-departure) predict the choice better than early
  content (the planning window, Tirole/Pfeiffer).
- **Infra:** `MODEL_CHOICE` exists; needs per-stop replay-content-by-arm summary
  joined to the upcoming choice + `contingency_table` value.
- **Gate:** does content beat the value-only choice model out-of-sample?
- **Hazards:** circularity (content↔decoder), distance/occupancy confound, and
  the strongest collider risk (choice is downstream) — use held-out prediction,
  not in-sample fit.

### 4. Content × Duration
- **Q:** does *what* is replayed relate to *how long* the animal stays — e.g.
  current/rewarded-arm (consummatory) content → longer dwell; alternative-arm
  content → shorter (about to leave)?
- **Model:** (a) departure hazard with content covariates:
  `cloglog h_L(t) = bs(tc) + γ_cur·CurrentArmContent(t) + γ_alt·AltContent(t) +
  reward` (does alternative content raise the leave hazard?); (b) within-stop
  content trajectory: per-bin multinomial over the represented arm
  `P(repr = a | tc) = softmax_a( s_a(tc) )` — test a current→prospective shift in
  `s_a(tc)` as `tc` grows (the pre-departure planning shift).
- **Method:** (a) departure hazard ~ within-stop content summary (fraction
  current-arm vs alternative, posterior entropy) — does alternative content raise
  the leave hazard? (b) content *trajectory within the stop*: does the
  represented arm drift from current → prospective as `tc` grows (a pre-departure
  planning shift)? — a per-bin multinomial/Dirichlet content model over `tc`.
- **Infra:** content summaries per bin/stop + the Phase-2 departure hazard.
- **Gate:** does content add to the hazard beyond reward/time?
- **Hazards:** content is sparse per bin (few events) — aggregate carefully;
  collider if conditioning on duration.

### 5. Content × Rate
- **Q:** do high-rate stops carry different content (more current-arm? more
  diverse/alternative?) — is the rate dissociation (SWR value-gated vs non-SWR
  exploratory) mirrored in content?
- **Model:** marked point process — onsets at intensity `λ_R`, each with a
  content mark `m ~ Categorical(π)`; test (i) whether `π` differs by SWR and
  (ii) whether `λ_R` and `π` covary via a shared per-stop term (e.g. high-rate
  SWR stops over-represent the current/rewarded arm; high-rate non-SWR stops
  over-represent alternatives). Estimate rate and mark on **split-half / disjoint
  event subsets** to avoid shared-noise correlation.
- **Method:** per-stop, relate replay rate (Family B intensity or count) to a
  content-composition summary (current-arm fraction, arm entropy, value of
  represented arms). Test whether SWR-coupled high-rate stops over-represent the
  current/rewarded arm while non-SWR high-rate stops over-represent alternatives.
- **Infra:** per-stop rate (have) + content summary (have).
- **Gate:** is there a rate↔content-type association beyond shared reward/value?
- **Hazards:** both derive from the same events (shared-noise correlation) —
  use cross-validated / split-half or independent event subsets.

### 6. Duration × Behavior
- **Q:** does dwell relate to the upcoming **choice / value** — do longer stays
  precede staying (exploit) vs switching (explore); does the departure hazard
  carry choice information?
- **Model:** two complementary fits — departure hazard with the upcoming choice:
  `cloglog h_L(t) = bs(tc) + γ·V + η·will_stay`; and a stay/switch logit on dwell
  beyond value: `P(stay) = σ( α·V + β·DurationResid )` where `DurationResid` is
  the dwell residualized on value. β ≠ 0 = dwell carries choice info beyond value.
- **Method:** departure hazard ~ value + `is_stay`/next-choice indicators; and
  next-choice ~ duration + value (does dwell predict the decision beyond value).
- **Infra:** the departure hazard (Phase 2) + `MODEL_CHOICE` + value.
- **Gate:** does duration carry choice info beyond value?
- **Hazards:** dwell, value, and choice are mutually entangled — partial out
  value; mediation framing.

## The capstone — latent-state integrative model

- **Q:** is there a shared latent state (value/engagement/learning phase) that
  jointly drives replay **rate**, replay **content bias**, and **departure
  hazard**, and **predicts the next choice** — with replay as a candidate
  **mediator** from value to behavior, and SWR/non-SWR as distinct emission
  channels?
- **Model (generative state-space):** a low-dim latent state `z_s` per stop `s`
  evolving slowly (gap-aware in wall-clock): `z_s = A z_{s-1} + ε_s`. It **emits**
  the four axes:
  - rate: `log λ_R(t,s) = bs(tc) + b_R·z_s + βx`
  - content: `P(repr arm = a | event in s) = softmax_a( c_a + d_a·z_s )`
  - duration: `cloglog h_L(t,s) = bs(tc) + b_L·z_s + γx`
  - behavior: `P(choice_s = a) = softmax_a( α·V(a) + λ·Replay_a(s) + e_a·z_s )`

  with **SWR / non-SWR as separate emission channels** (own `λ_R`, `c_a`). Replay
  is a **mediator** of value→behavior iff `λ ≠ 0` beyond `z_s` (replay carries
  choice information the latent value state does not). `b_R = b_L = d_a = λ = 0`
  collapses to the independent per-axis models — the nested null. Estimate by
  EM / variational / particle methods (reuse the `state_space_practice`
  point-process EM library).
- **Method:** a **state-space / latent-variable model** generalizing the joint-
  departure per-stop frailty from a scalar to a (low-dim) latent `z_t` that
  evolves slowly (across stops / wall-clock, gap-aware) and **emits**: the replay
  intensity (Poisson), the content bias (multinomial over arms), and the
  departure hazard; and **links to** the choice model. Replay enters the choice
  model as a mediator to test drive-vs-reflect. Estimation: EM / variational /
  particle filtering — reuse the `state_space_practice` point-process EM library
  (noted in memory). The within-session drift (the preliminary non-stationarity)
  is naturally `z_t`'s slow component.
- **Gate (strict):** build only after the pairwise rungs identify which couplings
  are real — otherwise it is a 20+-parameter latent model fit "for nothing."
- **Hazards:** identifiability (many latents/emissions), the full mediation/
  causal burden, and compute. This is the integrative *endpoint*, not a near-term
  build.

## Sequencing

1. Finish **rate × behavior** refinements (uncertainty plan).
2. **rate × duration** — joint-departure plan (already written).
3. **content × behavior** (mediation) — highest scientific value of the
   remaining rungs; extends `MODEL_CHOICE`.
4. **content × duration** and **content × rate** — content-side couplings.
5. **duration × behavior**.
6. **Capstone state-space** — only if the rungs warrant it.

Each step is its own spec → plan; this file is the map, not a build.
