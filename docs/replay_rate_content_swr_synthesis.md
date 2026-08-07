# Replay rate × content × SWR-coupling: integrative synthesis

**Status:** living results note. Confirmed findings, open hypotheses, and unknowns
for how the **rate**, **content**, and **SWR-coupling** of non-local replay events
relate. Companion to the per-layer references: rate ladder
[per_bin_rate_results.md](per_bin_rate_results.md) / [rate_models.md](rate_models.md);
content claims registry [../paper/claims_table.md](../paper/claims_table.md).

## Provenance & methods (read first)

- **Data:** event tables (`event_table.pkl`, `event_arm_table.pkl`, `stop_table.pkl`)
  regenerated 2026-06-24. Nominal value from the canonical
  LorenFrankLab/SpatialBanditTask parsed CSVs (commit `96dfed1`), mapped to
  `stop_id` (`scripts/_bandit_csv_contingency.py`; 96% agreement with the retired
  StateScript parser). Content models re-fit 2026-06-28 on the current tables
  (the prior fits were stale, from 2026-06-08).
- **Inference convention (all layers):** fit **per session**, then test the
  per-session coefficients **across sessions** (Wilcoxon signed-rank; the session
  is the exchangeable unit), with **per-animal sign consistency (n/5)**. This is
  robust to within-session clustering; it is *not* an event-level test.
- **Everything here is correlational.** Covariate↔replay associations, not
  established causation. Two confounds are explicitly controlled where noted:
  **occupancy** (via the within-well outcome contrast) and **within-stop time /
  dwell** (via a `time_since_stop` control); **collinearity** was audited (VIF).
- **One process, three layers.** A non-local event is a *point* (occurrence)
  carrying *marks*: its **SWR-coupling** (ripple power) and its **content**
  (represented arm). "Rate" = occurrence intensity; "coupling" and "content" =
  marks. The layers are separable factors of one marked-point-process likelihood
  (see *Open* — we have not fit the full joint model).

## Layer 1 — Rate (occurrence)

Per-bin conditional-intensity ladder (Family B; `MODEL_PER_BIN_RATE`). Baseline
rates (M0): non-local 0.405 > detected-SWR 0.285 > NL-no-SWR 0.234 > NL-w/SWR
0.181 events/s. Dominant structure (predictive, all populations): within-stop
**time course** (rise→plateau over ~1.5–2 s), **reward-history suppression**
(lags 2–10, ~0.85×), **self-history** clustering. Best model M7r (AIC 2105,
CV +0.199).

Value/uncertainty covariates (canonical-contingency MU model), **with-SWR vs
non-SWR**:

| covariate | NL with-SWR | NL non-SWR | note |
| --- | --- | --- | --- |
| reward lag-1 (M7) | **3.93×** | 1.11× | |
| `p_reward_current` | **~1.0× (null, all p>0.47)** | ~1.0× (null) | no sign-consistency, ~zero rate information; *historical* de-inflation 6.6→3.1→2.0 (collinearity fix + canonical contingency) ended at null in the current MU |
| `rpe` | **1.69×** (RR1.693, p 5e-4) | 1.10 (n.s.) | |
| `recent_switch_rate` | **0.57×** (G3 stratified, p 5e-10) | 1.16 (n.s.) | exploration suppression — 0.55× (S5 contrast); near-null in the joint MU model (covariate competition) |
| `value_diff_patch` | 0.52 (p 1e-3) | 0.77 (p 0.04) | **4/5 animals only** (j16 reverses), inference-only, **opposite sign in the power model** |

**Inference ≠ prediction (important):** value/RPE are real *population-level*
effects (consistent sign across animals) but **do not improve per-session
prediction** — the **entire value/switch block fails held-out CV in every
population and animal (0/5)** (ΔCV −0.075 non-SWR, −0.163 with-SWR, −0.043 SWR);
per-session ΔAIC sits at the null. Only time + reward kernel +
self-history forecast the rate. A fitted within-block Q-value (`q_chosen`) does
**not** predict better than nominal value (falsified). So the value-gating is a
modest, detectable-by-pooling effect, not a strong driver.

### Update 2026-07-02 — `is_swr`-contrast ladder (per-bin, LR-tested)

Re-fit of the occurrence-rate dissociation with the **beta-bernoulli covariates**
(`switch_advantage_mu_preoutcome_sess`/`rpe`/`beta_reward_entropy`/`global_mu_update_abs_postoutcome_sess`;
refit 2026-07-13, HMM retired) on the per-bin **`is_swr`-contrast (Design B)** ladder —
stratified `G0–G8` (×{all, SWR, non-SWR}) + stacked-contrast `S0–S10`.
Runner `scripts/rate/fit_swr_contrast_rate.py`; 218 sessions →
`models_per_bin_rate/swr_contrast_rate.pkl`. Inference: per-session fit + **per-session
likelihood-ratio tests** aggregated as *% sessions significant / binomial p* (the
power-robust read; pooled χ² is over-powered by the per-bin n).

- **Reward-gating dissociation confirmed and strongest by far.** `is_swr:reward_lag1`
  significant in **53% of sessions (binom p=7e-78)**; direction **91–100% positive in
  all 5 animals**. The ~50% per-session significance is a **power gradient**
  (43/53/60% by event-count tertile), **not animal clustering** — the effect is
  near-universal in sign; low-event sessions just can't clear p<0.05 individually.
- **Headline magnitude is model-dependent:** the `is_swr:reward_lag1` effect is 5.1×
  when the reward-history kernel is shared (`S1_rew1`) but **3.1× when `is_swr` is
  crossed with the full kernel** (`S2_base`, history-controlled) — both p<1e-23.
  The 3.1× matches the prior SWR reward-RR (3.10). *(Reporting choice deferred.)*
- **Dissociation hierarchy** (`is_swr:X`, % sessions sig / binom p):
  reward **53% / 7e-78** ≫ periswitch **21% / 1.5e-8** > rpe / switchrate / value
  (~10–12% / 5e-3–2e-2) > block (marginal) > **update / uncert (null)**.
- **Stratified pattern (G-ladder):** every task dimension modulates rate far more
  consistently in **non-SWR** than SWR events (periswitch 25% vs 9%, rpe 25% vs 8%,
  value 15% vs 4%; SWR at chance) → **SWR events narrowly reward-gated, non-SWR events
  broadly task-modulated** (value / RPE / context).
- **Dwell-shape is NULL.** `G8 rewxtime` (`reward_lag1 : bs(tc)` — does reward reshape
  the within-stop rate *profile*?) is not consistently significant (8% / p=0.088),
  **numerically unstable** (205/218 fit, mean CV −9.6e+302) and **CV-harmful**
  (M9−M7 −0.22, 5/5 neg). **Reward shifts the rate LEVEL, not the profile SHAPE** — the reward-gating headline
  is not a dwell-profile artifact (the per-bin offset already removes the pure
  duration confound; rewarded stops are ~3.6× longer, 1.86 s vs 0.51 s).
- **RPE caveat:** coefficient signed-rank leans SWR-specific, but the per-session LR
  is *more consistent in non-SWR* (25% vs 8% of sessions) — direction-consistent vs
  per-session-needed diverge; do **not** call RPE cleanly "SWR-specific". Also,
  SWR RPE is **reward-collinear** (`rpe = reward − q_chosen`, kernel retained) —
  largely reward-gating re-expressed, not independent RPE coding.

*(The `is_swr`-contrast ladder has no auto-generated results doc — `per_bin_rate_results.md`
covers the separate M0–M9 ladder. These findings live here.)*

## Layer 2 — SWR-coupling (ripple power as a continuous mark)

`ripple_power_z` (event-mean consensus ripple power — the signal the Kay detector
thresholds, so `is_swr` is its thresholded version). Per-event regression
(`scripts/rate/fit_swr_power_model.py`):

- **Distribution is ~unimodal** (Sarle bimodality 0.524; no gap at the detection
  threshold) → the SWR/non-SWR split looks like a **cut on a continuum**, not two
  modes. *Caveat:* event-mean smears any gap; the **peri-onset peak** (the
  detector's actual criterion) is the definitive test and is **not yet run**.
- Not duration-confounded (`corr(power, duration)` = 0.04).
- **Reward (+0.141/SD, 5/5, p 5.5e-17)** and **RPE (5/5)** raise SWR power;
  **exploration lowers it (5/5)**. Combined with Layer 1 (occurrence reward
  effect only ~1.5×), reward acts on **coupling** more than on bare occurrence —
  i.e. the binary "with-SWR rate" dissociation is largely reward pushing events'
  ripple power up across the threshold. *Caveat:* the power-model reward covariate
  is the **current-well** reward (corr 0.799 with current `reward_outcome`,
  dwell-confounded), so "reward raises coupling" may be partly a **consummatory
  LFP artifact**, not a clean pre-event mark.
- **Scope:** this is the *mark factor only*, by per-event OLS — **not** a full
  marked point process (no joint intensity, no test of power↔history independence,
  no shared latent).

## Layer 3 — Content (represented arm)

Arm-hierarchy conditional Poisson (`MODEL_HIERARCHY`), re-fit on the 2026-06-24
tables. Content predictors are *arm-relative* (a constant-across-arms covariate
cancels in the per-event normalization), so scalar behavioral covariates enter
only as **interactions with `is_current`** (the arm the animal is physically on).

**Re-verified hierarchy (reproduces the documented findings, several near-exactly):**

| finding | non-SWR (M5) | with-SWR (M5) |
| --- | --- | --- |
| current-arm bias (`is_current`) | **+0.881** (RR2.41, 5/5, p 4e-28) | +0.509 (RR1.66, weaker → broadened) |
| avoids last-rewarded-elsewhere | **−1.485** (5/5, p 2e-11) | strongly neg. (−11.73 = sparse-subset **separation artifact**, 89/169 separated; direction robust, magnitude not) |
| raw arm reward-rate (`reward_rate_arm_k10`) | **+0.00 (p 0.31, null)** | **−0.065 (p 0.066, marginal)** — no consistent value main effect |
| SWR broadening (arm-all M6) | — | `swr_x_current` −0.84, `swr_x_prev` −1.06 (5/5) |

So **non-SWR content is value-neutral; with-SWR content is value-avoidant**
(broadened off current/prev, avoids recently-rewarded — the last-rewarded-elsewhere
row, still strongly negative; the raw arm reward-rate main effect is only marginal).
Patch-level mirrors arm. *(These older pooled-M5 rows are corrected in place but are
superseded by the C-split in the 2026-07-02 update below; the SWR×value result
(content *avoids* high beta value, `swr_x_q_arm −2.45`) is in that block.)*

**New — behavioral modulation of current-arm content** (`is_current`
interactions, **duration-controlled** with `time_since_stop × is_current`;
`scripts/exploratory/_content_reward_interactions.py`):

| interaction | non-SWR | with-SWR | reading |
| --- | --- | --- | --- |
| within-stop time (`tsc_x_current`) | **+0.23** (5/5, p 3e-10) | **+0.40** (5/5) | content **deepens onto** current arm over the stop |
| reward (`rew_x_current`) | **−0.76** (5/5, p 6e-5) | −0.13 (n.s.) | reward pushes non-SWR content **off** current |
| current value (`value_x_current`) | **−0.53** (5/5, p 4e-3) | −0.25 (n.s.) | high-value current → off current |
| exploration (`switch_x_current`) | **+0.53** (4/5, p 0.02) | **+1.21** (5/5, p 1e-4) | exploration **locks** content onto current |

Note: the reward effect **strengthened** under the duration control (−0.32→−0.76),
because the within-stop drift runs *toward* current (`tsc_x_current` > 0) and was
masking it — so it is not a dwell artifact. The reward/value content effects are
**non-SWR-specific**; exploration and within-stop deepening hold for both types.

### Update 2026-07-02 — content `is_swr`-contrast ladder (arm + patch, `is_swr` split)

Content analog of the rate `is_swr`-contrast ladder, with the beta-bernoulli
covariates entered as `:is_current` interactions (event-scalars) or per-arm main
effects. Runner `scripts/content/fit_swr_contrast_content.py`; 218 sessions →
`models/swr_contrast_content_{arm,patch}.pkl`. Conditional Poisson over arms(9)/
patches(3), per-session fit + across-session median / signed-rank / per-animal (n/5).
**Value construct: beta-bernoulli `q_arm` (the single behavioral model since the
2026-07-13 HMM retirement).** The value–content result is that **content *avoids*
high beta-value arms** — `swr_x_q_arm −2.45, 5/5, p=2e-36` (refit 2026-07-13). The
earlier "construct-dependence" caveat is **moot**: there is one construct now (the
retired HMM `q_arm_hmm` had shown the opposite `+1.06` sign, but it is gone). The
beta value-avoidance reinforces the exploration reading — uncertainty↑ and value↓
both point to under-explored arms. Audited per-fit SE/p and family FDR are in place;
within-group-centered VIF and a common complete-case cohort remain deferred (see the
runner provenance).

- **Anchors reproduce at full power:** `is_current` +0.69 (5/5); SWR broadening
  `swr_x_current` −0.81 (p=7e-35), `swr_x_prev` −0.93; SWR reward/value-avoidance
  `swr_x_reward_rate_arm` −1.72 / `swr_x_q_arm` −2.45 / `swr_x_last_rewarded` −1.62
  (all 0/5, p<1e-11). Net of autocorrelation controls (`C2_base`), the SWR current-
  bias **vanishes** (−0.04 n.s.) while non-SWR persists (+0.67, p=3e-11).
- **Beta state modulation of the current-bias** (refit 2026-07-13, beta covariates):
  relative value ΔV **sharpens onto current** (`relval_x_current` +0.16, 4/5, p=9e-15;
  SWR +0.16 5/5); **RPE pushes off current** (`rpe_x_current` −0.24, 5/5, p=2e-11;
  non-SWR −0.20) — value and RPE oppose; value-update **sharpens onto current**
  (`update_x_current` +0.28, 5/5, p=3e-20; SWR +0.34). The scalar **belief-entropy
  interaction is null** (`entropy_x_current` −0.00 n.s.) — the beta-entropy scalar
  does not modulate current-arm focus (the per-arm uncertainty *main effect* is the
  content-uncertainty result, C13 below; do not conflate).
- **New — context/switch strongly modulate content:** **exploration locks content
  onto current** (`switchrate_x_current` +0.23, 5/5, p=1e-17; SWR +0.34 — reproduces
  the prior +0.53); later-in-block moves **off** current (`block_x_current` −0.19,
  p=6e-12); near a switch → **broader/off-current** (`tss_x_current` +0.33, 5/5).
- **Patch:** current-patch bias +1.05 (5/5); SWR broadens off-patch (`swr_x_current_patch`
  −0.38, p=3e-18); avoids the abandoned patch (`swr_x_last_rewarded_patch` −1.49,
  p=9e-5); exploration → current patch (+0.16). The switch-timing rungs (old-patch
  decay `CP2`, new-patch growth `CP3`, `CPS3`) are **sparse/unstable** (n=43–62;
  `CP2` magnitude unreliable) — the design-flagged rungs.
- **Rate↔content asymmetry:** on *rate*, non-SWR events are the broadly task-modulated
  ones; on *content*, several modulations (value, uncertainty, periswitch, switchrate)
  are **SWR-equal-or-stronger** — SWR content is *broader* at baseline yet *more
  retunable* by value/uncertainty/context.

### Update 2026-07-08 — per-arm uncertainty (exploration); realized-gain test; CS6 caveat; prediction-model status

New per-arm content covariate **`arm_uncertainty`** = the per-leaf beta belief
variance (`betadist_var`, the variance counterpart of beta value `q_arm` from the
SAME Beta posterior). Built into `event_arm_table` (leaf arm *j* → `betadist_var{j+1}`,
trunk arms → 0.0, standardized `arm_uncertainty_z`); models `Mu`/`Mvu` (hierarchy)
and `C13`/`CS6` (is_swr ladder). Commits `bfb0012` + `f3d1656`; generative-model
assumptions in [behavior_models.md](behavior_models.md); memory
`project_arm_uncertainty_content`.

- **Content over-represents high-uncertainty arms** — `M5→Mu` ΔLR 371, p=1.7e-36;
  `arm_uncertainty_z` **+0.83, 5/5, p=3e-32**. This is the exploration / gain axis
  (Mattar-Daw) the value/reward/space ladder was missing.
- **NOT a recency/reward confound.** VIF-decomposition: recency adds only **+1% R²**
  beyond arm structure; the recency-*orthogonalized* effect **+0.62 (5/5, retains
  74% of the raw +0.83)** survives.
  The high raw **VIF 18 is a structural artifact** of the trunk=0 encoding against the
  arm FE (arm FE alone explain **75%** of `arm_uncertainty`), not a scientific confound —
  the effect is identified from within-leaf-arm variation.
- **SWR separation — trust the split (`C13`), NOT the pooled (`CS6`).** Split: both
  populations over-represent uncertain arms, **SWR +0.99 > non-SWR +0.60** (5/5 each) —
  SWR weights uncertainty *more* (the SWR enhancement holds in **69% of sessions**;
  chimi ≈ null). The pooled `CS6` interaction (`swr_x_arm_uncertainty_z`
  **−0.41**, 0/5) is an **artifact** (its SWR *component* is still **+0.62**, positive —
  not below baseline): the shared arm FE (75%-collinear with uncertainty)
  differ by SWR population and the misfit leaks into the interaction. Neither
  `swr_x_current` nor `swr_x_recency` rescues it; the **saturated `is_swr × ALL BASEc`
  model recovers +0.39 = the split.** General lesson: **pooled `swr_x` contrasts are
  unreliable for any covariate tied to arm identity** — report the split.
- **Exploit-vs-explore: the `Mvu` arm-FE degeneracy (VIF 19) is a parameterization
  artifact, not a real collinearity.** Within-arm (arm-demeaned, non-current-leaf),
  beta value `q_arm` and beta uncertainty `arm_uncertainty` separate cleanly
  (condition number ≈ 3.4, corr ≈ 0.16) — so the beta-only exploit/explore model **is
  identifiable** in that frame (no HMM value needed; HMM is retired). The within-arm
  horse-race shows content tracks **uncertainty (+0.06, 5/5)** while beta value is
  ~null and distance flips negative — the explore axis, ~orthogonal to exploit. See
  the Phase 5 uncertainty_content extension.
- **Realized-gain (the PE face of Mattar-Daw) does NOT drive content.** The per-arm
  realized-gain test used the retired HMM per-leaf update `|Δ hmm_q_port|` and found
  `M5+gain` **−0.064, 5/5, p=2e-11** (content *avoids* recently-updated arms — the
  update sits at the just-visited/rewarded arm, a reward/recency-avoidance signal, not
  gain-seeking); with uncertainty co-fit, uncertainty stayed **+0.83** and gain → −0.03.
  **The gain axis of content is epistemic UNCERTAINTY, not realized value-update.**
  *(HMM-era number; the beta per-arm value-update re-run is a Phase-5 to-do — the beta
  scalar `global_mu_update_abs × current` is +0.28, a different estimand.)*

**Prediction models (Stage 5A — does content predict the next choice?).** The
behavioral-relevance layer, and the weakest link. `replay_frac` over a behavioral
baseline (conditional logit): non-SWR **+0.34 (p=0.019)**, SWR **null** (claims #19d),
but the model **drops `dist_to_arm`** (quasi-separation — the animal picks the closest
arm 95–99%), so it is **not distance-controlled** — the +0.34 may be current/nearest-arm
bias. Content does **not** predict stay/leave (#19e, null across strata). So: replay
content has, at best, a modest *uncontrolled* choice correlation and no stay/leave signal.
**The choice-prediction fits exist** (`event_source_comparison/*_comparison_fits.pkl`,
`choice` / `choice_swr_split`) but are **in-sample only — no held-out CV**. Raw `replay_frac`
is *negative* (−0.94, 5/5 — current-arm bias: the animal leaves the arm it most replays),
flipping to **+0.38 over the behavior baseline** (4/5 animals, but only 13% of sessions
individually significant; in-sample llf gain +0.32, 100% positive but that is guaranteed by
adding a parameter). The positive signal is carried by **non-SWR content**
(`replay_frac_non_swr` +0.335, 5/5) while **SWR content is null** (−0.14, n.s.); the split is
**underpowered (n 214→183)**. So there is a weak, non-SWR-carried, sign-flips-on-control,
**in-sample-only** choice correlation — not a cross-validated SWR prediction.

**Cross-validation for content models** — the content hierarchy had been fit with the
hard-count response only, **no CV** (unlike the rate ladder). A `--cv` run over
`M0a→M5,Mv,Mu,Mvu` (event-based per-session CV) is being added so the uncertainty rungs
carry held-out `cv_bits` alongside the LR/AIC comparison.

### Update 2026-07-08 (b) — spatial FORM of SWR vs non-SWR content: wells vs paths, and trajectory start/end

**Correction to the "non-SWR is current-focused" framing.** The arm-model `is_current`
coefficient is a *within-event relative* over-weighting (conditional Poisson conditions on
each event's total, so event-constants cancel and coefficients are compositional, not
absolute). The **content-occupancy** analysis (`scripts/content/analyze_content_occupancy.py`,
218 sessions) gives the absolute picture: **non-SWR content is reward-well-dominated
(~67% at wells: ~37% remote reward wells + ~30% own well; ~33% path); SWR content is
path-dominated (~78% path vs 22% at-well)** — SWR-vs-non-SWR path fraction 0.78 vs 0.32,
**p=1.6e-37, 5/5 animals**. So non-SWR represents **reward-well *locations*** (current *and
remote*), SWR represents **paths**.

- **How "at well" vs "path" is defined** (note the caveat): per non-local *immobile timebin*,
  take the **MAP** decode; `d_well` = track-graph distance to the nearest reward well (well =
  the animal's median immobile position on each leaf arm); **`d_well < 20 cm` = "at a well",
  `≥ 20 cm` = "path"** (crossed with `d_animal`, 20 cm, for remote vs own). So **"path" means
  ">20 cm from any reward port" — it includes mid-leaf-arm, NOT only trunk/inter-patch arms**,
  and it is MAP-based (not posterior mass). The clean, artifact-immune part is `remote_path`
  (far from animal *and* wells). The strict trunk-vs-leaf version is the separate
  `swr_x_trunk` 3.55× enrichment.
- **Retraction:** `arm_uncertainty` is **0 on all trunk arms** (a per-*leaf* quantity), so the
  `C13`/`Mu` uncertainty effects are estimated **only on well/leaf arms** — we have **no
  evidence** on whether SWR *path* content relates to uncertainty. The earlier "SWR extends
  uncertainty to the paths toward those wells" was unsupported; testing it needs a path/trunk
  uncertainty covariate (not built).

**Trajectory start/end IS characterized** (per-event rollups from `analysis/replay/trajectory.py`,
in `event_table`: `start_arm`/`end_arm`, `start_position`/`end_position`, `net_displacement`,
`spatial_extent`, `step_distance_*`, `max_ahead`/`max_behind`/`ab_slope`, `n_reversals`,
`ends_at_next`). A **quick SWR split (not yet a formal analysis)**:

| metric (mean) | non-SWR | SWR |
|---|---|---|
| spatial_extent (cm) | 26 | **85** |
| net_displacement (cm) | 17 | 48 |
| step_distance_max | 20 | 67 |
| n_reversals | 3.8 | 10.9 |
| max_ahead / max_behind | 18 / 114 | 14 / 145 |
| ends_at_next choice | 0.12 | 0.07 |

→ **SWR content = extended, multi-reversal trajectories; non-SWR = localized at wells.** Both
are **predominantly *behind* the animal** (max_behind ≫ max_ahead → retrospective), and content
**rarely ends at the next chosen arm** (7–12%) — consistent with not-choice-predictive. A
formal SWR-split trajectory characterization has **not** been run/reported yet.

### Update 2026-07-27 — distinction from Gillespie et al. 2021 (Neuron)

Gillespie et al. 2021 (same lab, same clusterless decoder, predecessor spatial task)
is the key replication risk for the content layer. Much of our **spatial** content
ladder *recovers* their results (a pipeline validation, to be framed as replicate +
extend, not as new): current-arm bias (their Fig 8), immediate-past suppression,
the **future/choice null** (#19d ↔ their Fig 5), and reward-history dependence
(their Fig 3/6).

**Open — previous-goal sign disagrees.** Gillespie report previous-goal
*enrichment* (their Fig 8); every estimate in this document goes the other way
(`is_last_rewarded_elsewhere` −1.485, 5/5, p 2e-11 at the arm grain; SWR-split
`swr_x_last_rewarded` −1.62; patch-grain −1.49). This is a **divergence to
explain, not a replication** — resolve before framing the spatial ladder as
"replicate + extend".

Two headline claims are **not** reducible to Gillespie:

- **Graded-value avoidance is not the previous-goal effect (their Fig 8).** Refitting
  `CS5` with the previous-goal analog added (`CS5b_valuevsprevgoal` =
  `+ is_last_rewarded_elsewhere + swr_x_is_last_rewarded_elsewhere`), `swr_x_q_arm`
  is **unmoved: −2.45 → −2.43, 5/5 animals, p=8e-28**; the added SWR-by-previous-goal
  interaction (`swr_x_is_last_rewarded_elsewhere`) is itself significant at −0.79,
  p=5e-8 — i.e. the two effects coexist, both as avoidance. Their binary-goal task
  structurally cannot yield a graded-value coefficient; this is genuinely new.
  *Provenance:* these values came from an exploratory refit that has since been
  removed; `CS5b_valuevsprevgoal` is now a permanent rung of
  `SWR_CONTRAST_CONTENT_ARM_LADDER`, so **re-derive them from a standard
  `fit_swr_contrast_content.py` run** before citing.
- **Uncertainty→content is not their visit-recency bias (their Fig 7).** In the
  within-arm frame (arm identity demeaned out), recency's coefficient is ≈0 while
  uncertainty's is +0.06 (5/5) — the tested `within_arm_horserace` estimand. The
  global `corr(uncertainty, recency)=0.54` is a **between-arm**
  artifact. Caveat: this is a *within-arm* estimand (cleaner but different from their
  *between-arm* binned recency) — the claim is "uncertainty dominates recency in the
  frame that isolates state from arm identity, which Fig 7 could not separate," not a
  refutation of Fig 7.

The remaining structurally-novel axes (no Gillespie analog): the **SWR/non-SWR
dissociation** and the **rate/intensity models** (they modeled content only).

## The intersection — what's new here

**The covariate atlas** (duration-controlled where applicable; non-SWR | with-SWR):

| driver | RATE / coupling | CONTENT (current-arm) |
| --- | --- | --- |
| reward | non-SWR small (~1.11×) but reliable / **with-SWR ↑ rate & coupling** | **non-SWR ↓ current** / with-SWR n.s. |
| current value | with-SWR ↑ rate (modest) | **non-SWR ↓ current** / with-SWR n.s. |
| exploration | **both ↓ SWR-coupled rate** | **both ↑ current** (with-SWR stronger) |
| within-stop time | rate rise→plateau | **both deepen onto current** |

**The headline: a double dissociation by SWR status.**

> Reward acts on with-SWR **occurrence** but on non-SWR **content**.

- **with-SWR:** reward/value/RPE gate **how much** (rate 3.9×, coupling); content
  is **broad and value-avoidant by default** and is **not** retuned by the
  immediate reward (`rew_x_current` n.s.).
- **non-SWR:** reward/value only weakly gate how much (rate ~1.11×, small but
  reliable 5/5 — a ratio dissociation, not flat-vs-gated); content is
  **current-locked but dynamically pushed off-current** by immediate reward/value
  (`rew_x_current` −0.76).
- **Shared:** exploration suppresses SWR-coupled rate yet **concentrates** content
  on current in both; content **deepens onto current** through the stop in both.

## Confirmed (robust: multi-control and/or 5/5 animals)

1. Rate baseline ordering and the dominant time-course/reward-history/self-history
   structure (all populations).
2. **Exploration suppression of SWR-coupled rate** (0.57× (G3), p 5e-10 —
   model-fragile magnitude; source-independent — does not depend on the
   contingency parse).
3. **Content current-arm bias** and **last-rewarded-elsewhere avoidance**
   (non-SWR), reproduced on current tables (+0.881 / −1.485).
4. **Non-SWR content is value-neutral; with-SWR content is value-avoidant.**
4b. **Content over-represents high-*uncertainty* arms** (`Mu` +0.83, 5/5, p=3e-32),
   robust to the recency confound (VIF-decomposition: recency +1% R²; recency-
   orthogonalized +0.62 = 74% retained, 5/5). Present in both populations, **stronger in SWR** (split
   `C13` +0.99 vs +0.60). The exploration/epistemic axis — *not* realized value-update
   (per-arm gain is negative). See the 2026-07-08 update.
5. **SWR broadening** of content off current/prev.
6. **Reward/value push non-SWR content off the current arm** (duration-controlled,
   5/5); **exploration locks content onto current** (both types).
7. **The double dissociation** (reward → with-SWR rate, non-SWR content).
8. **The reward-driven off-current content is spike-supported, not a decode/
   consummation artifact.** Reality engine (`analyze_event_spike_evidence.py`,
   5 sessions): post-reward off-current (nonSWR-remote) `dll_per_spike` = 0.63,
   **99% > 0** vs baseline ≈ 0; rewarded ≈ omitted (0.63 vs 0.65), so reward does
   not degrade spike support. The off-current (remote) fraction of non-SWR events
   rises 0.45 → 0.50 with reward (the `rew_x_current` shift, replicated).
9. **Off-current content's destination is the "novel" patch.** On patch-switch
   events the off-current content prefers the **third (other) patch over the
   previous patch** (non-SWR 0.24 vs 0.19; with-SWR 0.23 vs 0.16) — and the
   previous patch is avoided largely because it is often the **last-rewarded**
   one (30%/20% vs 1–2% for the third patch). So replay samples the
   non-recent, non-rewarded option, not where it came from.

## Hypotheses (suggestive; need a further test)

- **Continuum, not two mechanisms.** Unimodal coupling distribution + graded SWR
  broadening of content. *Needs:* peri-onset **peak** distribution; the joint MPP.
- **Reward shifts SWR-coupling continuously** (binary dissociation = thresholded
  shadow). *Needs:* the joint occurrence+mark model.
- **Value-gating of with-SWR rate is modest and not predictive** — report as a
  population inferential effect, not a forecaster.

## Unknown / pending — including the models we DON'T have

- **Clean exploit-vs-explore content model — identifiable within-arm, being built
  (Phase 5).** The `Mvu` arm-FE degeneracy (VIF 19) was a parameterization artifact:
  within-arm (arm-demeaned, non-current-leaf) beta value `q_arm` and beta uncertainty
  `arm_uncertainty` separate cleanly (cond ≈ 3.4, corr ≈ 0.16). No HMM value needed
  (HMM retired). The within-arm horse-race already shows the answer — content tracks
  **uncertainty** (+0.06, 5/5), value ~null — being folded into the
  `uncertainty_content` feature.
- **Full marked point process** (joint intensity + coupling + content; shared
  latent; power↔history independence). **Not built.** The three layers are treated as
  separable factors but never jointly fit.
- **Need / successor (prospective) content — structurally UN-buildable here.** The task
  cannot identify it: 84% alternation makes `is_next`/`is_prev` collinear (r=0.81). No
  covariate separates prospective from retrospective on this track.
- **Graded ripple-power content model — NOT a committed rung.** Graded
  `ripple_power × current` beats binary SWR (97% sessions, −291 AIC) but lives only in
  an exploratory script, not the ladder.
- **Content choice-mediation** (does rate × content predict the next choice). The
  prior choice correlation (#19d) is modest and **not distance-controlled** (`dist_to_arm`
  dropped for quasi-separation); stay/leave is null (#19e).
- **Per-arm realized-gain: tested (HMM-era), negative.** `|Δ hmm value|` per arm
  *avoids* content (−0.06, 5/5) — the gain axis of content is epistemic uncertainty,
  not realized value-update (see the 2026-07-08 update). Not a driver to add; the beta
  per-arm value-update re-run is a Phase-5 to-do.
- **CS6-class pooled `swr_x` contrasts are unreliable** for arm-identity-tied covariates
  (the `swr_x_arm_uncertainty` artifact) — a model to *distrust*, use the split.
- **Other content responses** (soft-count, posterior-mass) and **CV /
  model-comparison** for the content hierarchy — the base ladder used the hard-count
  response only; a `--cv` run (M0a→M5,Mv,Mu,Mvu) is in progress (2026-07-08).

## Caveats & confounds (and how addressed)

- **Occupancy** (immobility happens at wells → current arm = reward well): the
  reward content effect uses the **within-well outcome contrast** (rewarded vs
  omitted at the same well, identifiable via the probabilistic schedule), which
  isolates the reward *outcome* from the well's location.
- **Dwell / within-stop time** (reward → ~3.5× longer stops): controlled by
  `time_since_stop × is_current`; the reward content effect *strengthened* under
  it (drift is toward current), so it is not a dwell artifact. Control is linear
  in `z(log1p t)` — a spline would be stricter.
- **Collinearity:** `reward_rate_trial_k10` (= mean of the reward lags, VIF ∞)
  removed; value covariate set curated to VIF ≤ 1.07.
- **Separation:** with-SWR `is_last_rewarded_elsewhere` magnitude is a sparse-
  subset artifact (the per-session `ValueError`s) — direction only.
- **Circularity** (content from the same decoder that defines events): both the
  baseline current-arm claim *and* the new reward-driven off-current content
  clear the spike-evidence engine (post-reward off-current `dll_per_spike` 0.63,
  99% > 0 vs baseline ≈ 0; **5 sessions, in-principle validation**). The
  **full-dataset** reality run is deliberately **gated on the planned full
  re-decode**, not run now: (i) the re-decode persists `log_likelihood` to
  `results.nc` so reality is a cheap direct read rather than a ~20 min/session
  recompute; (ii) the current decodes are stale (the position-trim drift), so a
  full run now would validate the wrong decodes; (iii) the close-event W3
  (place-field-extent) worry needs the local-kernel re-decode regardless. See the
  reality design spec (TODO §2, "gated on a first re-decoded session").

## Reproducibility (scripts & key commits)

- Rate: `scripts/rate/fit_per_bin_rate_models.py`, `fit_uncertainty_rate.py` (canonical
  contingency, curated covariates).
- Coupling: `scripts/rate/fit_swr_power_model.py`.
- Content: `scripts/content/fit_replay_content_models.py` (re-fit 6 configs);
  `scripts/exploratory/_verify_content_models.py`; `scripts/exploratory/_content_reward_interactions.py`.
- Contingency: `scripts/behavior/fetch_spatial_bandit_behavior.sh`,
  `scripts/_bandit_csv_contingency.py`.
- Reality check on reward-modulated content: `scripts/qa/analyze_event_spike_evidence.py`
  (now emits per-event `start_time`) + `scripts/exploratory/_reality_reward_test.py`;
  content destination: `scripts/exploratory/_content_patch_destination.py`.
- Commits: `6d63566` (collinearity fix), `b246e66` (canonical contingency),
  `6d9983c` (SWR-power model), `0dc2dfa` (content re-verify), `45dc147`
  (duration-controlled content interactions).
