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
| `p_reward_current` | **2.01×** (p 4e-3) | 1.04 (n.s.) | de-inflated 6.6→3.1→2.0 after collinearity fix + canonical contingency |
| `rpe` | **1.87×** (p 2e-3) | 1.10 (n.s.) | |
| `recent_switch_rate` | **0.69×** (p 6e-7) | 1.16 (n.s.) | exploration suppression — robust, source-independent |
| `value_diff_patch` | 0.52 (p 1e-3) | 0.77 (p 0.04) | shared-ish |

**Inference ≠ prediction (important):** value/RPE are real *population-level*
effects (consistent sign across animals) but **do not improve per-session
prediction** — held-out `cv_bits` *drops* when the value block is added (−0.16,
non-SWR); per-session ΔAIC sits at the null. Only time + reward kernel +
self-history forecast the rate. A fitted within-block Q-value (`q_chosen`) does
**not** predict better than nominal value (falsified). So the value-gating is a
modest, detectable-by-pooling effect, not a strong driver.

### Update 2026-07-02 — HMM-harmonized `is_swr`-contrast ladder (per-bin, LR-tested)

Re-fit of the occurrence-rate dissociation with **paper-faithful HMM covariates**
(`hmm_relative_value`/`hmm_rpe`/`reward_entropy`/`hmm_value_update`; single
`default_hmm` model) on the per-bin **`is_swr`-contrast (Design B)** ladder —
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
  the within-stop rate *profile*?) is not consistently significant (8% / p=0.088).
  **Reward shifts the rate LEVEL, not the profile SHAPE** — the reward-gating headline
  is not a dwell-profile artifact (the per-bin offset already removes the pure
  duration confound; rewarded stops are ~3.6× longer, 1.86 s vs 0.51 s).
- **RPE caveat:** coefficient signed-rank leans SWR-specific, but the per-session LR
  is *more consistent in non-SWR* (25% vs 8% of sessions) — direction-consistent vs
  per-session-needed diverge; do **not** call RPE cleanly "SWR-specific".

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
- **Reward (+0.139/SD, 5/5, p 3e-17)** and **RPE (5/5)** raise SWR power;
  **exploration lowers it (5/5)**. Combined with Layer 1 (occurrence reward
  effect only ~1.5×), reward acts on **coupling** more than on bare occurrence —
  i.e. the binary "with-SWR rate" dissociation is largely reward pushing events'
  ripple power up across the threshold.
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
| current-arm bias (`is_current`) | **+0.98** (5/5, p 4e-28) | +0.70 (weaker → broadened) |
| avoids last-rewarded-elsewhere | **−1.66** (5/5, p 2e-11) | strongly neg. (−13.76 = sparse-subset **separation artifact**; direction robust, magnitude not) |
| drawn to high-value arms (`recent_reward_rate`) | **+0.00 (p 0.31, null)** | **−0.29 (p 2e-4)** — value-*suppressed* |
| SWR broadening (arm-all M6) | — | `swr_x_current` −0.84, `swr_x_prev` −1.06 (5/5) |

So **non-SWR content is value-neutral; with-SWR content is value-avoidant**
(broadened off current/prev, avoids recently-rewarded). Patch-level mirrors arm.

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

### Update 2026-07-02 — HMM-harmonized content ladder (arm + patch, `is_swr` split)

Content analog of the rate `is_swr`-contrast ladder, with paper-faithful HMM
covariates entered as `:is_current` interactions (event-scalars) or per-arm main
effects. Runner `scripts/content/fit_swr_contrast_content.py`; 218 sessions →
`models/swr_contrast_content_{arm,patch}.pkl`. Conditional Poisson over arms(9)/
patches(3), per-session fit + across-session median / signed-rank / per-animal (n/5).
**Exploratory** pending Phase-4 audited diagnostics (per-fit SE/p, family FDR,
within-group-centered VIF, common cohort); `q_arm` is still the beta per-arm value
(HMM harmonization deferred).

- **Anchors reproduce at full power:** `is_current` +0.69 (5/5); SWR broadening
  `swr_x_current` −0.81 (p=7e-35), `swr_x_prev` −0.93; SWR reward/value-avoidance
  `swr_x_reward_rate_arm` −1.72 / `swr_x_q_arm` −2.45 / `swr_x_last_rewarded` −1.62
  (all 0/5, p<1e-11). Net of autocorrelation controls (`C2_base`), the SWR current-
  bias **vanishes** (−0.04 n.s.) while non-SWR persists (+0.67, p=3e-11).
- **New — HMM state modulation of the current-bias:** relative value ΔV **sharpens
  onto current** (`relval_x_current` +0.15, 5/5, p=2e-13; SWR +0.18); **RPE pushes
  off current** (`rpe_x_current` −0.14, non-SWR, p=6e-8) — value and RPE oppose;
  uncertainty sharpens onto current **SWR-specifically** (`entropy_x_current` +0.13,
  p=4e-4).
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

## The intersection — what's new here

**The covariate atlas** (duration-controlled where applicable; non-SWR | with-SWR):

| driver | RATE / coupling | CONTENT (current-arm) |
| --- | --- | --- |
| reward | non-SWR flat / **with-SWR ↑ rate & coupling** | **non-SWR ↓ current** / with-SWR n.s. |
| current value | with-SWR ↑ rate (modest) | **non-SWR ↓ current** / with-SWR n.s. |
| exploration | **both ↓ SWR-coupled rate** | **both ↑ current** (with-SWR stronger) |
| within-stop time | rate rise→plateau | **both deepen onto current** |

**The headline: a double dissociation by SWR status.**

> Reward acts on with-SWR **occurrence** but on non-SWR **content**.

- **with-SWR:** reward/value/RPE gate **how much** (rate 3.9×, coupling); content
  is **broad and value-avoidant by default** and is **not** retuned by the
  immediate reward (`rew_x_current` n.s.).
- **non-SWR:** reward/value do **not** gate how much (rate flat); content is
  **current-locked but dynamically pushed off-current** by immediate reward/value
  (`rew_x_current` −0.76).
- **Shared:** exploration suppresses SWR-coupled rate yet **concentrates** content
  on current in both; content **deepens onto current** through the stop in both.

## Confirmed (robust: multi-control and/or 5/5 animals)

1. Rate baseline ordering and the dominant time-course/reward-history/self-history
   structure (all populations).
2. **Exploration suppression of SWR-coupled rate** (0.69×, p 6e-7; source-
   independent — does not depend on the contingency parse).
3. **Content current-arm bias** and **last-rewarded-elsewhere avoidance**
   (non-SWR), reproduced on current tables (+0.98 / −1.66).
4. **Non-SWR content is value-neutral; with-SWR content is value-avoidant.**
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

## Unknown / pending

- **Full marked point process** (joint intensity + coupling + content; shared
  latent; power↔history independence). **Not built.**
- **Content choice-mediation** (does rate × content predict the next choice). The
  prior choice correlation (#19d) is modest and **not distance-controlled**.
- **Other content responses** (soft-count, posterior-mass) and **CV /
  model-comparison** for the content hierarchy — re-fit used the hard-count
  response only.

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
