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

## Layer 2 — SWR-coupling (ripple power as a continuous mark)

`ripple_power_z` (event-mean consensus ripple power — the signal the Kay detector
thresholds, so `is_swr` is its thresholded version). Per-event regression
(`scripts/fit_swr_power_model.py`):

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
`scripts/_content_reward_interactions.py`):

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
- **Collinearity:** `recent_trial_reward_rate` (= mean of the reward lags, VIF ∞)
  removed; value covariate set curated to VIF ≤ 1.07.
- **Separation:** with-SWR `is_last_rewarded_elsewhere` magnitude is a sparse-
  subset artifact (the per-session `ValueError`s) — direction only.
- **Circularity** (content from the same decoder that defines events): both the
  baseline current-arm claim *and* the new reward-driven off-current content
  clear the spike-evidence engine (post-reward off-current `dll_per_spike` 0.63,
  99% > 0 vs baseline ≈ 0; 5 sessions). A full-dataset run is still the eventual
  goal.

## Reproducibility (scripts & key commits)

- Rate: `scripts/fit_per_bin_rate_models.py`, `fit_uncertainty_rate.py` (canonical
  contingency, curated covariates).
- Coupling: `scripts/fit_swr_power_model.py`.
- Content: `scripts/fit_replay_content_models.py` (re-fit 6 configs);
  `scripts/_verify_content_models.py`; `scripts/_content_reward_interactions.py`.
- Contingency: `scripts/fetch_spatial_bandit_behavior.sh`,
  `scripts/_bandit_csv_contingency.py`.
- Reality check on reward-modulated content: `scripts/analyze_event_spike_evidence.py`
  (now emits per-event `start_time`) + `scripts/_reality_reward_test.py`;
  content destination: `scripts/_content_patch_destination.py`.
- Commits: `6d63566` (collinearity fix), `b246e66` (canonical contingency),
  `6d9983c` (SWR-power model), `0dc2dfa` (content re-verify), `45dc147`
  (duration-controlled content interactions).
