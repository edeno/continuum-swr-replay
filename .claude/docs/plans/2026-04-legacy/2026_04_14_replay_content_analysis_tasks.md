
# Replay Function Analysis — Implementation Tasks

## Objective

Build a reproducible analysis pipeline to determine the function of hippocampal
replay in the Spatial Bandit task. The task is a probabilistic foraging task with
6 reward ports (2 per patch, 3 patches), probabilistic rewards p(R) ∈ {0.2, 0.5,
0.8}, and covert contingency blocks that change every 60–80 trials. Animals learn
to concentrate visits in the highest-value patch and adapt to uncued block changes.

The pipeline answers, in increasing order of complexity:

$$Q1:\ \text{Is replay spatially structured enough to analyze?}$$

$$Q2:\ \text{Does replay rate depend on behavioral context?}$$

$$Q3:\ \text{What does replay represent at the arm level?}$$

$$Q4:\ \text{Does replay predict upcoming choice?}$$

$$Q5:\ \text{Does replay predict learning or policy updates?}$$

$$Q6:\ \text{Is replay related to value?}$$

$$Q7:\ \text{Does replay contain trajectory structure beyond unordered arm occupancy?}$$

The implementation proceeds from descriptive summaries to simple models to richer
computational models. Do not start with a full HMM or RL sequence model.

---

# 1. Notation and variable definitions

## Naming conventions

All variable names used as DataFrame column names or function arguments follow
PEP 8 and the conventions below. Mathematical notation in equations (e.g., $w_t$,
$L_e$, $V_{ea}$) is separate from variable names used in code.

- **snake_case** throughout. No camelCase, no CapWords, no single-letter names
  outside of loop indices.
- **Descriptive over abbreviated.** Write `immobility_time` not `T_immobile`; write
  `latent_value` not `value_latent`; write `n_events` not `N_events`.
- **Boolean columns prefixed with `is_`** (e.g., `is_swr`, `is_after_reward`,
  `is_current`).
- **Counts prefixed with `n_`** (e.g., `n_events`, `n_nonlocal_bins`,
  `n_unique_arms`).
- **Rates and probabilities** use descriptive suffixes (`_rate`, `_prob`).
- **Ranks** suffixed with `_rank` (e.g., `arm_dist_rank`, `latent_value_rank`).
- **Interaction terms** use `_x_` as a separator to denote a product of two
  variables (e.g., `swr_x_current` = `is_swr` × `is_current`). This is a
  domain-standard convention for statistical interaction terms.
- **Adjective before noun**: `latent_value` not `value_latent`; `arm_patch_id`
  not `patch_id_a`.
- **No abbreviations** except established domain acronyms (`swr`, `mua`, `hpd`,
  `map`) and universally understood short forms (`prev`, `n_`).

## Inference hierarchy

The primary inferential unit is the **session**. Event-arm rows within a session
are not independent — they share the same decoder, animal state, and task context
— so pooling all rows across sessions and computing a single p-value constitutes
pseudoreplication.

Required approach:

1. **Fit within session**: estimate model coefficients separately per session.
2. **Summarize across sessions**: report per-session coefficient distributions
   (median, IQR, sign consistency). Use a one-sample signed-rank test on
   per-session estimates against zero as the primary inferential test.
3. **Cross-session pooling** (secondary): pooled model with session as fixed or
   random effect, or cluster-robust SEs clustered by session.

CV for held-out likelihood must be split by session (leave-one-session-out), not
by random event-arm row splits.

## 1.1 Basic indices

- $(e)$: replay event index
- $(a)$: arm index (port/well index; per codebase graph numbering; see CLAUDE.md)
- $(t)$: timebin index within replay event
- $(s)$: stopping period or decision-point index
- $(j)$: session index
- $(i)$: animal index

Arm indices follow the track-graph node numbering in CLAUDE.md. There are 6 reward
ports plus hub/junction positions. Confirm the exact mapping before computing any
spatial variable.

---

## 1.2 Core replay variables

### `nonlocal_prob`

Per-timebin variable; column in `event_sequence_table`. Smoothed non-local state
posterior for timebin $t$. Mathematical notation: $w_t$.

$$
w_t = P(\text{state}_t = \text{non-local} \mid \text{spikes}_{1:T})
$$

Use the forward-backward smoother for offline analysis. Do not use the filtered
(causal) posterior.

---

### Replay summary variables — ordered by analysis priority

Three versions of the replay summary, reflecting increasing use of decoder
information. Core analyses use `arm_count_hard` first. `arm_count_soft` and
`arm_posterior_mass` are used for robustness and to assess threshold sensitivity.

#### `arm_count_hard` — primary analysis variable

Hard MAP count with fixed non-local threshold $\theta$ (default: 0.8).

$$
c_{ea}^{\text{hard}} = \sum_{t \in e} \mathbf{1}[w_t > \theta] \cdot
\mathbf{1}[\text{MAP\_arm}_{et} = a]
$$

Simplest, most interpretable, closest to field convention. Each timebin either
counts or does not.

$$
\texttt{n\_nonlocal\_bins}_e = \sum_a c_{ea}^{\text{hard}}
= \sum_{t \in e} \mathbf{1}[w_t > \theta]
$$

Robustness: repeat at $\theta \in \{0.5,\ 0.7,\ 0.8,\ 0.9\}$.

#### `arm_count_soft` — second analysis variable

Soft-weighted MAP count using the continuous non-local posterior.

$$
c_{ea}^{\text{soft}} = \sum_{t \in e} w_t \cdot \mathbf{1}[\text{MAP\_arm}_{et} = a]
$$

Same MAP arm assignment as the hard version; each timebin contributes its full
non-local weight rather than a binary 0/1.

$$
\texttt{nonlocal\_weight\_sum}_e = \sum_{t \in e} w_t
$$

#### `arm_posterior_mass` — third analysis variable

Full conditional non-local posterior mass.

$$
m_{ea} = \sum_{t \in e} w_t \cdot P(\text{arm} = a \mid \text{spikes}_t,\
\text{state}_t = \text{non-local})
$$

Uses the conditional non-local posterior — not the full marginal
$P(\text{arm}=a \mid \text{spikes}_t)$, which would mix local and non-local
contributions.

$$
\texttt{posterior\_mass\_total}_e = \sum_{t \in e} w_t \leq \texttt{nonlocal\_weight\_sum}_e
$$

---

### `arm_fraction`

$$
f_{ea} = \frac{\texttt{arm\_count\_hard}_{ea}}{\texttt{n\_nonlocal\_bins}_e}
$$

Primary analysis uses `arm_count_hard`. Undefined when `n_nonlocal_bins` = 0;
such events are excluded from content analyses. Analogous fractions are computed
for `arm_count_soft` and `arm_posterior_mass` in robustness analyses.

### `decoded_arm_map`

Hard arm label from MAP decoding at each timebin:
$\text{arg max}_a P(\text{arm}=a \mid \text{spikes}_t, \text{state}_t = \text{non-local})$.

### `posterior_over_arms`

Vector of conditional non-local posterior probabilities over all arms for each
timebin.

---

### Decoder uncertainty variables

All three computed from the conditional non-local posterior, summarized over
event $e$ with $w_t$ weighting.

#### `posterior_peak_mean`

$$
\texttt{posterior\_peak\_mean}_e =
\frac{\sum_{t \in e} w_t \cdot \max_a P(\text{arm} = a \mid \text{spikes}_t,
\text{state} = \text{non-local})}{\texttt{nonlocal\_weight\_sum}_e}
$$

High values: confident arm assignment. Low values: diffuse posterior.

#### `posterior_entropy_mean`

$$
\texttt{posterior\_entropy\_mean}_e =
\frac{\sum_{t \in e} w_t \cdot H\bigl[P(\text{arm} \mid \text{spikes}_t,
\text{state} = \text{non-local})\bigr]}{\texttt{nonlocal\_weight\_sum}_e}
$$

#### `posterior_hpd_width`

Mean number of arms needed to capture a fixed fraction of the posterior (HPD
credible interval width). Store as three columns:
`posterior_hpd_width_50`, `posterior_hpd_width_80`, `posterior_hpd_width_95`.

$$
\texttt{posterior\_hpd\_width\_80}_e =
\frac{\sum_{t \in e} w_t \cdot |\mathcal{A}^{0.8}(t)|}{\texttt{nonlocal\_weight\_sum}_e}
$$

where $\mathcal{A}^{0.8}(t)$ is the smallest set of arms accounting for 80% of
the conditional non-local posterior at timebin $t$.

Interpretation: more interpretable than entropy for communicating decoder confidence.
An HPD width of 1 means the decoder is essentially certain; a width near the number
of arms means the posterior is nearly uniform. Recommended variable for figures.

---

## 1.3 Task-state variables

### `current_arm`

Port where the animal is located during the stopping period (beam-break location).
Consecutive pokes at the same port are never rewarded, so `next_arm ≠ current_arm`
always.

### `prev_arm`

Port most recently visited before `current_arm`.

### `next_arm`

Port visited next after the stopping period ends.

**Post-hoc**: observed after replay occurs. All results using `next_arm`, `is_next`,
or any `_next_` variable must be explicitly labeled as post-hoc associations.

### `last_rewarded_arm`

Most recently rewarded port prior to the replay event.

### `current_patch`, `prev_patch`, `next_patch`

Patch (0, 1, or 2) containing the respective arm. `next_patch` is post-hoc.

### `arm_patch_id`

Patch membership for arm $a$ (fixed maze property).

### `is_same_patch`

$$
\texttt{is\_same\_patch}_{ea} = \mathbf{1}[\texttt{arm\_patch\_id}(a) = \texttt{current\_patch}_e]
$$

Whether the replayed arm is in the same patch as the current arm. Important control
for spatial proximity confounds in the three-patch geometry.

### `is_stay`

Boolean. True if the upcoming choice is within the same patch (Stay). Post-hoc.
Use `~is_stay` for switch indicator. Unified from `is_stay_context` and
`stay_vs_switch` in earlier versions.

### Binary arm indicators

$$\texttt{is\_current}_{ea} = \mathbf{1}[a = \texttt{current\_arm}_e]$$

$$\texttt{is\_prev}_{ea} = \mathbf{1}[a = \texttt{prev\_arm}_e]$$

$$\texttt{is\_next}_{ea} = \mathbf{1}[a = \texttt{next\_arm}_e] \quad \text{(post-hoc)}$$

$$\texttt{is\_last\_rewarded}_{ea} = \mathbf{1}[a = \texttt{last\_rewarded\_arm}_e]$$

### Collinearity of task-state indicators

Common coincidences in the Spatial Bandit task:

- `is_prev` and `is_last_rewarded`: coincide when the animal just left a rewarded
  port (frequent given stay/switch structure).
- `is_current` and `is_last_rewarded`: coincide at rewarded Stay trials.
- `is_next` and `is_last_rewarded`: coincide when the animal returns to a recently
  rewarded port.
- `is_prev` and `is_next`: coincide at alternation trials.

**Required QC**: 4×4 co-occurrence matrix for all four indicators. Flag pairs with
>30% co-occurrence. Report predictor correlation matrix alongside all model outputs.

---

## 1.4 Spatial variables

### `dist_to_arm`

Shortest-path graph distance from the animal's current stopping position to arm $a$.

### `arm_dist_rank`

Rank of `dist_to_arm` within an event; 1 = closest arm.

### `n_unique_arms`

Number of distinct arms with `arm_count_hard` > 0 in event $e$.

### `path_continuity`

Fraction of consecutive decoded-arm pairs that are graph-adjacent:

$$
\texttt{path\_continuity}_e = \frac{1}{T_e - 1} \sum_{t=1}^{T_e - 1}
\mathbf{1}[\texttt{decoded\_arm\_map}_{t+1} \in \text{neighbors}(\texttt{decoded\_arm\_map}_t)]
$$

Null: circular time-shuffle within each event (permute timebin order, preserving
the marginal arm distribution).

---

## 1.5 Reward and observable value variables

### `reward_outcome`

Observed reward (binary) at the port visited immediately before the stopping period.

### `is_after_reward`

Indicator that replay follows a rewarded outcome.

### `is_after_omission`

Indicator that replay follows a nonrewarded outcome at a port that could have
delivered reward (genuine omission).

### `is_after_switch`

Indicator that replay follows a patch Switch trial.

### `is_after_stay`

Indicator that replay follows a within-patch Stay trial.

### `trials_since_block_change`

Trials elapsed since the most recent contingency block change.

### `block_number`

Index of the current contingency block within session (1-indexed).

### `visit_recency`

Trials since arm $a$ was last visited before event $e$. Set to session length if
arm has not been visited. Given the consecutive-poke constraint, `visit_recency`
for `current_arm` is always ≥ 1.

### `cumulative_reward`

Total rewards received at arm $a$ up to and including event $e$.

### `recent_reward_rate`

Empirical reward rate for arm $a$ over the last $k$ visits (default: $k = 5$).

### `nominal_reward_prob`

Programmed p(R) ∈ {0.2, 0.5, 0.8} for arm $a$ at the time of event $e$. Changes
at block boundaries. The best patch always contains ports at 0.5 and 0.8.

---

## 1.6 Latent value variables

These require a fitted behavior model. Do not compute until Stage 7B.
Specify the model (Q-learning, Bayesian learner, softmax RL) before implementing.

### `latent_value`

Estimated latent subjective value $V_{ea}$ for arm $a$ at event $e$.

### `latent_value_rank`

Rank of `latent_value` across arms within event $e$.

### `value_diff_from_current`

$$V_{ea} - V_{e,\text{current}}$$

### `is_best_alternative`

Indicator that arm $a$ is the highest-value arm excluding `current_arm`.

### `best_alternative_value`

$$\max_{a \neq \text{current}} V_{ea}$$

### `switch_advantage`

$$\texttt{switch\_advantage}_e = \max_{a \neq \text{current}} V_{ea} - V_{e,\text{current}}$$

### `policy_prob`

Behavior-model-derived choice probability for arm $a$.

### `value_uncertainty`

Posterior uncertainty of the latent value estimate for arm $a$, if the behavior
model provides it.

### `reward_prediction_error`

Reward prediction error at the relevant stop, if available from behavior model.

### `priority_gain_need`

Computational priority score for arm $a$ (e.g., gain × need), if implemented.

---

## 1.7 Temporal and contextual variables

### `time_since_stop`

Elapsed time from stop onset to replay event onset.

### `immobility_time`

Total immobility time during the stopping period. Defined by speed threshold,
computed independently of replay content. Mathematical notation: $T_{\text{immobile}}$.
Used as offset in the Poisson rate model: `log(immobility_time)`.

### `n_events`

Count of replay events during a stopping period. Mathematical notation: $N_s$.
Response variable for the Poisson rate model.

### `is_swr`

Indicator that event $e$ occurs during a detected sharp-wave ripple (LFP-based
detection).

### `is_mua`

Indicator that event $e$ occurs during a high multiunit activity (MUA) event,
detected from population spiking rate independently of LFP features.

An event can be both `is_swr` and `is_mua`. Report the 2×2 cross-tabulation
(`is_swr` × `is_mua`) in Stage 0 QC.

### Interaction terms

All interaction terms are products of an event-level indicator and an arm-level
indicator. The `_x_` separator denotes multiplication.

$$\texttt{time\_x\_current}_{ea} = \texttt{time\_since\_stop}_e \times \texttt{is\_current}_{ea}$$
$$\texttt{time\_x\_next}_{ea} = \texttt{time\_since\_stop}_e \times \texttt{is\_next}_{ea}$$
$$\texttt{time\_x\_prev}_{ea} = \texttt{time\_since\_stop}_e \times \texttt{is\_prev}_{ea}$$

$$\texttt{swr\_x\_current}_{ea} = \texttt{is\_swr}_e \times \texttt{is\_current}_{ea}$$
$$\texttt{swr\_x\_next}_{ea} = \texttt{is\_swr}_e \times \texttt{is\_next}_{ea}$$
$$\texttt{swr\_x\_prev}_{ea} = \texttt{is\_swr}_e \times \texttt{is\_prev}_{ea}$$
$$\texttt{swr\_x\_last\_rewarded}_{ea} = \texttt{is\_swr}_e \times \texttt{is\_last\_rewarded}_{ea}$$

$$\texttt{mua\_x\_current}_{ea} = \texttt{is\_mua}_e \times \texttt{is\_current}_{ea}$$
$$\texttt{mua\_x\_next}_{ea} = \texttt{is\_mua}_e \times \texttt{is\_next}_{ea}$$
$$\texttt{mua\_x\_prev}_{ea} = \texttt{is\_mua}_e \times \texttt{is\_prev}_{ea}$$

$$\texttt{switch\_x\_current}_{ea} = \texttt{is\_after\_switch}_e \times \texttt{is\_current}_{ea}$$
$$\texttt{switch\_x\_prev}_{ea} = \texttt{is\_after\_switch}_e \times \texttt{is\_prev}_{ea}$$
$$\texttt{omission\_x\_last\_rewarded}_{ea} = \texttt{is\_after\_omission}_e \times \texttt{is\_last\_rewarded}_{ea}$$

---

## 1.8 Choice and learning outcome variables

### `chosen_next_arm`

Observed next port visit following the stopping period.

### `is_stay`

Boolean. True if the next choice is within the same patch (Stay); False if
cross-patch (Switch). Post-hoc. Use `~is_stay` for switch indicator.

### `policy_update`

Optional continuous or binary measure of policy change after outcome. Define only
after behavior model is finalized.

---

## 1.9 Sequence variables

### `replay_start_arm`, `replay_end_arm`

MAP arm at the first and last timebin of event $e$ with `nonlocal_prob` > 0.5.
Named with `replay_` prefix to avoid collision with behavioral `current_arm` /
`next_arm`.

### `n_arm_transitions`

Number of consecutive arm-to-arm changes within event $e$.

### `contains_path_to_next`, `contains_path_to_prev`, `contains_path_to_best_alternative`

Indicators that replay contains a graph-adjacent transition from `current_arm`
toward the named destination. `contains_path_to_next` is post-hoc.

### `ends_at_next`, `ends_at_best_alternative`

Indicators that `replay_end_arm` equals the named arm. `ends_at_next` is post-hoc.

### `forward_score`

Graph-based measure of whether replay moves away from or toward the animal's
current position. For each consecutive timebin pair, compute the signed change in
graph distance from `current_arm`:

$$
\delta_t = \texttt{dist\_to\_arm}(\texttt{decoded\_arm\_map}_{t+1}, \texttt{current\_arm})
- \texttt{dist\_to\_arm}(\texttt{decoded\_arm\_map}_t, \texttt{current\_arm})
$$

$$
\texttt{forward\_score}_e = \frac{1}{T_e - 1} \sum_{t=1}^{T_e-1} \delta_t
$$

Positive: replay moves away from current position (outward). Negative: replay
returns toward current position (inward / reverse).

Do not use a Spearman correlation between arm label and timebin index — arm
numbering is arbitrary and not ordered along a single path, so the rank correlation
is not meaningful.

### `reverse_score`

$$\texttt{reverse\_score}_e = -\texttt{forward\_score}_e$$

---

# 2. Required output tables

## 2.1 `event_table`

One row per replay event.

Required columns:

- `animal`, `day`, `epoch`, `stop_id`, `event_id`
- `event_start`, `event_end`, `event_duration`
- `is_swr`, `is_mua`
- `time_since_stop`
- `is_after_reward`, `is_after_omission`, `is_after_switch`, `is_after_stay`
- `current_arm`, `prev_arm`, `next_arm` (post-hoc), `last_rewarded_arm`
- `current_patch`, `prev_patch`, `next_patch` (post-hoc)
- `block_number`, `trials_since_block_change`
- `reward_outcome`
- `n_nonlocal_bins` (hard, default $\theta = 0.8$)
- `nonlocal_weight_sum` (soft)
- `posterior_mass_total`
- `n_junction_bins`, `n_unique_arms`
- `path_continuity`
- `posterior_peak_mean`, `posterior_entropy_mean`
- `posterior_hpd_width_50`, `posterior_hpd_width_80`, `posterior_hpd_width_95`
- `immobility_time`

## 2.2 `event_arm_table`

One row per event-arm pair.

Required columns:

- all event keys from `event_table`
- `arm`, `arm_patch_id`
- `arm_count_hard`, `arm_count_soft`, `arm_posterior_mass`
- `arm_fraction` (from `arm_count_hard` by default)
- `n_nonlocal_bins`
- `is_current`, `is_prev`, `is_next` (post-hoc), `is_last_rewarded`
- `is_same_patch`
- `dist_to_arm`, `arm_dist_rank`
- `visit_recency`, `cumulative_reward`, `recent_reward_rate`, `nominal_reward_prob`
- all interaction terms from Section 1.7

## 2.3 `event_sequence_table`

One row per replay timebin.

Required columns:

- event keys
- `t_in_event`
- `nonlocal_prob`
- `decoded_arm_map`
- `posterior_over_arms`
- context variables carried from `event_table`

## 2.4 `stop_table`

One row per stopping period.

Required columns:

- behavior outcomes, value variables
- `n_events`
- `immobility_time`
- `is_swr_stop`, `is_mua_stop`
- replay summary variables aggregated from `event_arm_table`

---

# 3. Analysis stages and implementation tasks

## Stage 0. QC and sanity checks

### Goal

Verify that replay events are structured enough to analyze and that the pipeline
is functioning correctly.

### Tasks

1. Build QC metrics on `event_table`.
2. Circular time-shuffle baseline: permute timebin order within each event,
   preserving marginal arm distribution. Compare `path_continuity` real vs. shuffled.
3. Indicator collinearity check: 4×4 co-occurrence matrix for `is_current`,
   `is_prev`, `is_next`, `is_last_rewarded`. Flag pairs with >30% co-occurrence.
4. `is_swr` × `is_mua` cross-tabulation: report four-way event type breakdown.
5. Invariant checks:
   - `sum(arm_count_hard) == n_nonlocal_bins` for all events
   - `arm_fraction` sums to 1 when `n_nonlocal_bins` > 0
6. Decoder confidence summary: `posterior_hpd_width_80` distribution split by
   `is_swr` and `is_mua`.
7. Negative controls — implement all three and save outputs for downstream use:
   - **Within-event circular time shuffle**: permute timebin order within each
     event, preserving marginal arm distribution. Primary null for `path_continuity`
     and sequence metrics.
   - **Within-session stop-label shuffle**: randomly permute behavioral labels
     (`is_after_reward`, `reward_outcome`, `current_arm`, `prev_arm`, etc.) across
     stops within session, preserving events-per-stop counts. Null for replay-
     behavior alignment analyses (Stages 5–6).
   - **Within-session `next_arm` circular shift**: circularly shift `next_arm` and
     associated reward labels across stops within session. Null specifically for
     Stage 5B retrospective alignment, guarding against patch geometry or block-drift
     correlations.

### Required plots

- histograms: `event_duration`, `n_nonlocal_bins`, `n_unique_arms`,
  `posterior_peak_mean`, `posterior_entropy_mean`, `posterior_hpd_width_80`
- `path_continuity` real vs. shuffled (paired per session)
- indicator co-occurrence heatmap
- `is_swr` × `is_mua` event type bar chart
- `posterior_hpd_width_80` by event type

### Acceptance criteria

- Median `path_continuity` exceeds shuffled median by > 0.05 in ≥80% of sessions.
- ≥60% of detected events have `n_nonlocal_bins` > 2.
- Sum invariants hold for all events.

Do not proceed to modeling if these criteria fail.

---

## Stage 1. Descriptive replay content

### Goal

Simple, model-free summaries of what replay represents.

### Tasks

Compute per-event summaries from `event_arm_table`:

- `replay_frac_current`, `replay_frac_next` (post-hoc), `replay_frac_prev`
- `replay_frac_last_rewarded`
- `replay_frac_highest_prob_arm` (arm with highest `nominal_reward_prob`)
- `replay_frac_best_alternative` (best arm excluding current)

### Required plots

- replay fraction on current / next / prev / last-rewarded vs. uniform (1/N)
- replay fraction by `arm_dist_rank`
- replay fraction by `nominal_reward_prob` rank
- each split by: SWR / MUA / neither, after reward / after omission,
  Stay / Switch context (post-hoc), early / late in stop

### Acceptance criteria

Plots generated reproducibly, pooled and per session.

---

## Stage 2. Replay rate model

### Goal

Test whether replay rate depends on behavioral context, independently of content.

### Model family

Poisson GLM. Response: `n_events` per stop. Offset: `log(immobility_time)`.

$$
\log E[N_s] = \log(\texttt{immobility\_time}_s) + \gamma_0
+ \gamma_1 \cdot \texttt{is\_after\_reward}_s
+ \gamma_2 \cdot \texttt{is\_after\_omission}_s
+ \gamma_3 \cdot \texttt{is\_after\_switch}_s + \ldots
$$

$\exp(\gamma)$: multiplicative change in events per second of immobility.

### Models to fit

#### R0 — Intercept only

Baseline replay rate per unit immobility.

#### R1

Add `is_after_reward`, `is_after_omission`, `is_after_switch`.

#### R2

Add `trials_since_block_change`, `block_number`.

#### R3 — Event-type split

Fit R1 separately for SWR-only, MUA-only, and neither event types.

### Required outputs

- rate ratios $\exp(\hat\gamma)$ with confidence intervals
- per-session rate estimates
- observed vs. predicted counts

### Acceptance criteria

Rate model fits and outputs are saved. Coefficients are interpretable.

---

## Stage 3. Primary replay-content model

### Goal

Fit the main arm-level conditional Poisson model of replay content.

### Model family

Conditional Poisson on `arm_count_hard`, grouped by `event_id`, offset
`log(n_nonlocal_bins)`. Primary analysis. Robustness analyses repeat with
`arm_count_soft` and `arm_posterior_mass`.

$$
\log E[\texttt{arm\_count\_hard}_{ea}] = \log(\texttt{n\_nonlocal\_bins}_e)
+ \beta_1 \cdot \texttt{is\_current}_{ea}
+ \beta_2 \cdot \texttt{is\_prev}_{ea} + \ldots
$$

$\exp(\beta)$: multiplicative change in fraction of event content going to an arm
with that label, relative to an unlabeled arm.

### Models to fit

#### M0 — Uniform null (intercept only)

$$
\log E[\texttt{arm\_count\_hard}_{ea}] = \log(\texttt{n\_nonlocal\_bins}_e) + \beta_0
$$

All arms receive equal expected content. Baseline for all comparisons.

#### M1

Predictors: `is_current`

#### M2

Predictors: `is_current`, `is_prev`, `is_next` (post-hoc)

#### M3

Predictors: `is_current`, `is_prev`, `is_next`, `is_last_rewarded`,
`visit_recency`, `cumulative_reward`

#### M4

M3 plus `dist_to_arm`

#### M5

M4 plus `time_x_current`, `time_x_next`, `time_x_prev`

#### M6

M5 plus `swr_x_current`, `swr_x_next`, `swr_x_prev`, `swr_x_last_rewarded`,
`mua_x_current`, `mua_x_next`, `mua_x_prev`

#### M7

M6 plus `switch_x_current`, `switch_x_prev`, `omission_x_last_rewarded`

### Required outputs

For each model:

- coefficients, SE, z-statistics, p-values
- $\exp(\hat\beta)$ with 99% CIs
- log-likelihood, AIC, BIC
- held-out event log-likelihood (5-fold CV)
- observed vs. predicted arm fractions
- predictor correlation matrix

### Required plots

- coefficient forest plot (M0–M7 on same axes)
- model comparison plot (AIC / held-out LL vs. complexity)
- marginal effect plots for major interactions
- per-session coefficient distributions for `is_current`, `is_next`, `is_prev`

### Acceptance criteria

- All models fit or fall back to regularized fit with logged warning.
- Outputs saved in standard format.
- `is_next` coefficients labeled post-hoc in all outputs.

---

## Stage 4. Hurdle model

### Goal

Formally separate replay occurrence from replay content. Test whether behavioral
covariates affect rate, content, or both.

### Observation unit

`event_table` contains only detected events — zero-event stops are absent by
construction. Part 1 operates at the **stop level**. Ensure `stop_table` includes
all stopping periods including those with zero replay events.

### Model structure

**Part 1 — Occurrence** (logistic, stop level):

$$\text{logit}\ P(\texttt{has\_replay}_s) = \gamma_0 + \gamma_1 \cdot X_s$$

where `has_replay_s` = 1 if stop $s$ contains at least one event with
`n_nonlocal_bins` > 0. Stop-level predictors: `is_after_reward`,
`is_after_omission`, `is_after_switch`, `immobility_time`, `is_swr_stop`,
`is_mua_stop`.

**Part 2 — Content** (conditional Poisson, event-arm level):

$$
\log E[\texttt{arm\_count\_hard}_{ea} \mid \texttt{n\_nonlocal\_bins}_e > 0]
= \log(\texttt{n\_nonlocal\_bins}_e) + \beta_1 \cdot Z_{ea}
$$

Use M2 predictor set for Part 2 as the default.

### Key scientific use

Test whether `is_swr` vs. `is_mua` affects occurrence only, content only, or both.
Test whether low-power events have normal content given occurrence (Part 2 matches)
but lower occurrence (Part 1 intercept lower). The dissociation is itself a finding.

### Required outputs

- Part 1 and Part 2 coefficients separately
- Summary of which predictors appear in one part, both, or neither
- Sensitivity to `n_nonlocal_bins` threshold

### Acceptance criteria

Both parts fit and outputs saved. Occurrence definition (stop-level, detection-
dependent) reported explicitly.

---

## Stage 5. Replay predicting upcoming choice

### Goal

Test whether replay content predicts the next choice. Split into two sub-analyses:

- **5A — Predictive**: no `_next_` variables; can replay improve choice prediction?
- **5B — Retrospective alignment**: did replay content align with what the animal did?

These must not be mixed in the same model.

---

### Stage 5A. Predictive analysis

#### Regression families

**Model family 1 — arm-level choice model (primary)**

Response: `chosen_next_arm`. Conditional logistic or conditional Poisson at the
stop × arm level. Predictors must vary across arms within a stop. This is the
primary outcome because it preserves which alternative was chosen.

$$
\log P(\text{choose arm } a \mid \text{stop } s) \propto
\alpha_1 \cdot \texttt{arm\_fraction}_{sa} + \alpha_2 \cdot \texttt{nominal\_reward\_prob}_{sa}
+ \alpha_3 \cdot \texttt{visit\_recency}_{sa} + \ldots
$$

**Model family 2 — stop-level logistic (secondary)**

Response: `is_stay`. Stop-level predictors only.

#### Baseline predictors

*Arm-level*: `nominal_reward_prob`, `visit_recency`, `recent_reward_rate`,
`cumulative_reward`, `dist_to_arm`

*Stop-level*: `reward_outcome`, `recent_reward_rate` of current arm,
`trials_since_block_change`

#### Predictive replay predictors (no `_next_` variables)

*Arm-level*: `arm_fraction` per arm

*Stop-level*: `replay_frac_current`, `replay_frac_best_alternative`,
`replay_frac_last_rewarded`, `n_events`, `posterior_hpd_width_80`

#### Required models

- baseline only
- baseline + arm-level `arm_fraction`
- baseline + stop-level content predictors
- baseline + `n_events` only (isolates rate from content)
- SWR-only replay features
- MUA-only replay features

#### Required outputs

- held-out log-likelihood (leave-one-session-out CV)
- calibration plot
- per-session likelihood gain from adding replay
- SWR vs. MUA vs. all-events comparison

#### Acceptance criteria

Replay contribution quantified above a behavior-only baseline.

---

### Stage 5B. Retrospective alignment analysis

Did replay content align with what the animal subsequently did? All outputs labeled
retrospective. Do not include `_next_` variables in Stage 5A models.

#### Analysis

Compare mean `arm_fraction` for chosen arm vs. unchosen arms within each stop.
Test per-session with signed-rank test; aggregate across sessions.

#### Response variables

- `replay_frac_next` vs. `replay_frac` on unchosen arms
- `replay_ends_at_next`: does replay terminate at the chosen arm?
- `contains_path_to_next`: does replay trace a path toward the chosen arm?

#### Required plots

- `replay_frac_next` vs. unchosen arm fraction (paired per stop)
- `replay_ends_at_next` rate by context (SWR / MUA, after reward / omission)
- `contains_path_to_next` rate by context
- all repeated against within-session `next_arm` circular-shift null

---

## Stage 6. Replay predicting learning

### Goal

Test whether replay predicts trial-to-trial updating after outcomes.

### Outcomes

Select one or more:

- `is_stay` on the trial immediately following reward or omission
- change in visit rate toward best arm over next $k$ trials (default: 5)
- `policy_update` if behavior model available

### Predictors

- `reward_outcome`
- `replay_frac_current`, `replay_frac_best_alternative`, `replay_frac_last_rewarded`
- `contains_path_to_best_alternative`
- `trials_since_block_change`
- `is_swr`, `is_mua` as moderators

### Required models

- outcome-only
- outcome + replay fractions
- outcome + replay + event-type interactions

### Required plots

- stay/switch probability after omission vs. replay on alternatives
- stay probability after reward vs. replay on current arm
- learning update magnitude vs. replay features

### Acceptance criteria

Replay evaluated as incremental predictor of learning-related changes.

---

## Stage 7. Value branch

### Goal

Determine whether replay is related to value beyond simple spatial labels. Staged
after choice prediction because it requires a fitted behavior model.

### Stage 7A. Observable value analyses

Uses `recent_reward_rate`, `visit_recency`, `cumulative_reward`,
`nominal_reward_prob`. No behavior model required.

#### Required descriptive plots

- replay fraction by `nominal_reward_prob` rank
- replay fraction by `recent_reward_rate` rank
- replay fraction on highest-value arm
- replay fraction vs. `recent_reward_rate` minus current-arm rate

#### Required models

- observable value only
- spatial/task labels only (M2)
- spatial + observable value (combined)

### Stage 7B. Latent value analyses

Requires a fitted behavior model. Derive: `latent_value`, `latent_value_rank`,
`value_diff_from_current`, `best_alternative_value`, `switch_advantage`.

#### Required descriptive plots

- replay fraction by `latent_value_rank`
- replay fraction vs. `value_diff_from_current`
- replay fraction on `is_best_alternative` arms
- heatmap: replay fraction as function of current and best-alternative value

#### Required models

1. spatial/task-label model (M2)
2. observable value only
3. spatial + observable value
4. spatial + `latent_value`
5. spatial + `latent_value` + `dist_to_arm`

### Acceptance criteria

Value variables directly compared against `is_current`, `is_next`, `is_prev` in
the same model. Observable value analyses completed before latent.

---

## Stage 8. Simple sequence analysis

### Goal

Test whether replay contains trajectory structure beyond arm occupancy.

### Tasks

Compute from `event_sequence_table`:

- `replay_start_arm`, `replay_end_arm`
- `n_arm_transitions`
- `contains_path_to_next` (post-hoc), `contains_path_to_prev`
- `contains_path_to_best_alternative`
- `ends_at_next` (post-hoc), `ends_at_best_alternative`
- `forward_score`, `reverse_score`

### Required descriptive plots

- `replay_start_arm` and `replay_end_arm` distributions
- arm transition matrix
- probability `replay_end_arm` = `next_arm` (post-hoc)
- `forward_score` vs. `reverse_score` by context (SWR / MUA, after reward / omission)

### Required models

#### S0 — Content-only baseline

Conditional Poisson (M2 or M3) on same events.

#### S1 — First-order transition model

$$P(x_t \mid x_{t-1})$$

#### S2 — Input-driven transition model

$$P(x_t \mid x_{t-1},\ \texttt{current\_arm},\ \texttt{next\_arm},\
\texttt{latent\_value},\ \texttt{is\_swr},\ \texttt{is\_mua})$$

### Required metric

Held-out sequence log-likelihood.

### Acceptance criteria

Sequence models escalated only if they outperform unordered content summaries.

---

## Stage 9. Fine spatial analysis

### Goal

Refine arm-level findings at graph-segment or position-bin level.

### Tasks

Create `event_bin_table` with event-bin counts or mass, graph position features,
and well / path / junction labels.

### Required questions

- Is replay concentrated at wells, paths, or junctions?
- Is future-arm replay specifically about future wells?

### Required plots

- average replay density on graph
- density split by chosen next arm (post-hoc)
- density split by high vs. low value

### Acceptance criteria

Only implement if arm-level results leave ambiguity that finer resolution resolves.

---

## Stage 10. Computational predictors

### Goal

Test whether richer computational variables explain replay better than descriptive labels.

### Tasks

From behavior model, derive: `policy_prob`, `value_uncertainty`,
`reward_prediction_error`, `expected_future_occupancy`, `priority_gain_need`.

### Required analyses

Add these predictors to the replay-content model, choice prediction model, and
sequence model if justified.

### Required comparisons

- spatial-only
- spatial + observable value
- spatial + `latent_value` + `policy_prob`
- spatial + `latent_value` + `priority_gain_need`

### Acceptance criteria

Only continue to full sequence computational models if these predictors improve fit.

---

## Stage 11. Full sequence computational model

### Goal

Richer sequence model, only if earlier stages justify it.

### Preferred order

1. Input-driven Markov model
2. IO-HMM on arm sequences
3. IO-HMM on `posterior_over_arms` sequences
4. Full simulated-run RL model

### Acceptance criteria

Do not implement unless replay predicts behavior or learning, sequence order matters,
and computational predictors already help.

---

# 4. Robustness analyses

## Replay summary robustness

Repeat core analyses (Stages 3–5) with each replay summary in order:

1. `arm_count_hard` at $\theta \in \{0.5,\ 0.7,\ 0.8,\ 0.9\}$
2. `arm_count_soft` with offset `nonlocal_weight_sum`
3. `arm_posterior_mass` with offset `posterior_mass_total`

Report whether coefficient signs and magnitudes are consistent across all three.

## Inclusion robustness

Repeat with: `n_nonlocal_bins` > 5, SWR-only events, MUA-only events, excluding
high-junction-fraction events.

## Statistical robustness

Repeat: pooled fit, session-wise fit, bootstrap across sessions, bootstrap across
animals if sample size allows.

## Decoder robustness

Compare alternate smoothing settings if codebase supports it.

---

# 5. Statistical reporting requirements

For every major result, save:

- per-session coefficient estimates (primary inferential unit)
- sign consistency across sessions (% sessions with same sign as median)
- one-sample signed-rank test on per-session estimates against zero
- pooled estimate with cluster-robust SE clustered by session (secondary)
- held-out log-likelihood split by session (leave-one-session-out CV)
- predictor correlation matrix
- associated descriptive summary
- comparison against the relevant negative control shuffle

Do not report only pooled p-values. Per-session estimates are the primary result.

---

# 6. Required deliverables

## Code deliverables

- data-preparation functions
- model-fitting functions (conditional Poisson, Poisson rate, hurdle, logistic)
- plotting functions
- unit test suite
- notebook or CLI entry points

## Data deliverables

- `event_table`
- `event_arm_table`
- `event_sequence_table`
- `stop_table`
- optional `event_bin_table`

## Result deliverables

- model summary tables (all three replay summaries)
- held-out performance tables
- per-session summary tables
- figure-ready plot outputs

---

# 7. Implementation order

1. Table-building utilities: `event_table`, `event_arm_table`, `stop_table`
   — build all three replay summaries (`arm_count_hard`, `arm_count_soft`,
   `arm_posterior_mass`) and all three length measures from the start.
   Ensure `stop_table` includes all stops including zero-event stops.
2. QC metrics and QC plots (Stage 0) including all three negative control shuffles
   — must pass before proceeding
3. Descriptive replay-content summaries (Stage 1) using `arm_count_hard`
4. Poisson rate model (Stage 2): R0–R3
5. Conditional Poisson M0–M4 (Stage 3) using `arm_count_hard`
6. Hurdle model (Stage 4) with stop-level Part 1
7. Choice prediction (Stage 5A predictive, then 5B retrospective)
8. Learning prediction with replay summaries (Stage 6)
9. Conditional Poisson M5–M7 (Stage 3, interaction models)
   — after Stages 5–6 so interaction choices are motivated by results
10. Observable value branch (Stage 7A)
11. Simple sequence summaries and transition model (Stage 8)
12. Latent value branch (Stage 7B) — requires behavior model fit
13. Computational predictor branch (Stage 10)
14. Full sequence computational models only if justified (Stage 11)
15. Robustness pass: repeat Stages 3–5 with `arm_count_soft` and `arm_posterior_mass`

---

# 8. Testing requirements

## Table-building

- `sum(arm_count_hard) == n_nonlocal_bins` for all events (floating point tolerance)
- `sum(arm_count_soft) == nonlocal_weight_sum` for all events
- `arm_fraction` sums to 1.0 when `n_nonlocal_bins` > 0
- `is_current`, `is_prev`, `is_next`, `is_last_rewarded` consistent with source arms
- `nonlocal_prob` values in [0, 1]
- `nonlocal_weight_sum` equals `sum(nonlocal_prob)` over event timebins
- `is_mua` and `is_swr` are boolean and populated for all events

## Uncertainty variables

- `posterior_peak_mean` in [0, 1]
- `posterior_entropy_mean` ≥ 0
- `posterior_hpd_width_50` ≤ `posterior_hpd_width_80` ≤ `posterior_hpd_width_95`

## Spatial variables

- `dist_to_arm` ≥ 0; arm with `arm_dist_rank` = 1 has lowest `dist_to_arm`
- `path_continuity` in [0, 1]
- `path_continuity` on circularly-shuffled sequences near null expectation

## Value variables

- `latent_value_rank` ranks correctly within event
- `is_best_alternative` is False for `current_arm`
- `switch_advantage` equals `best_alternative_value` minus current arm value

## Models

- conditional Poisson on synthetic data with known arm preference recovers
  coefficients within 2 SE (all three replay summaries)
- M0 predicts uniform arm fractions equal to 1 / N_arms
- rate model on synthetic Poisson counts with known rate recovers true rate
- hurdle Part 1 logistic matches standard logistic on same data
- model comparison metrics reproducible across runs

---

# 9. Non-goals for first implementation

Do not implement initially:

- full mixed-effects conditional Poisson (deferred to robustness)
- full joint replay-behavior latent model
- full RL simulated-run generator
- fine spatial model unless arm-level results motivate it
- latent value branch before Stage 7 (requires behavior model)

First goal: a stable, interpretable, reproducible pipeline using `arm_count_hard`
as the primary variable.

---

# 10. Success criteria

The pipeline succeeds if it can cleanly determine:

1. Whether replay events are structured (continuity > null) and have meaningful
   `n_nonlocal_bins` in a sufficient fraction of events
2. Whether replay rate varies with behavioral context (reward, omission, switch,
   SWR vs. MUA)
3. Whether replay content is enriched on current, next, previous, or reward-relevant
   arms, and whether effects survive controlling for spatial proximity and event type
4. Whether replay improves prediction of next choice above a behavior-only baseline
5. Whether replay improves prediction of learning-related behavioral changes
6. Whether replay content is related to value beyond simple spatial labels
7. Whether sequence structure is present and informative beyond arm occupancy
