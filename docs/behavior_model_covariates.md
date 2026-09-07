# Behavioral-model covariates — `qvals_table.parquet` data dictionary

Per-stop table of the SpatialBanditTask behavioral-model latent variables (value,
decision variables, uncertainty, value updates, entropy), one row per stop,
keyed by `stop_id` (joins to `stop_table.parquet`).

- **Generator:** [`analysis/behavior/qvals_table.py`](../src/continuum_swr_replay/analysis/behavior/qvals_table.py)
  — run `uv run python scripts/behavior/build_qvals_table.py --write`.
- **Output:** `Processed-Data/replay_content/models_per_bin_rate/qvals_table.parquet`
  (gitignored data artifact; regenerate from the script).
- **Decision-variable functions:** vendored verbatim from Alison Comrie's
  `rl_helpers.py` →
  [`src/continuum_swr_replay/analysis/behavior/_spatial_bandit_rl.py`](../src/continuum_swr_replay/analysis/behavior/_spatial_bandit_rl.py).
- **Paper:** Comrie et al., *Hippocampal representations of alternative
  possibilities are flexibly generated to meet cognitive demands* (bioRxiv
  2024.09.23.613567). No copy of that preprint version is in `references/`; the
  manuscript PDF held there is `Spatial_bandit_manuscript_v3.2_biorxiv_figs.pdf`.

## Provenance & model

| Piece | Source (DataJoint, read-only) |
|---|---|
| Value / belief columns | `alison_rlmodel.BehaviorModelResults`, fit `beta_stable_withleaf` (Beta-Bernoulli, stable task, with-leaf) |
| Decision variables (ΔV, stay/switch value, value updates, Beta posterior means) | computed locally by the vendored `make_full_beta_bernoulli_dv_df` on the above |
| Belief entropy | `reward_belief_differential_entropy_leaf_mean_preoutcome` — computed locally from the Beta posterior by `qvals_table.add_beta_belief_entropy` |
| Poke / reward timestamps | `alison_behav.TrialsInfoByEpoch.ByTrial` |

**Model note.** The analysis uses the **Beta-Bernoulli** fit
(`beta_stable_withleaf`) as the single behavioral model. Candidate-arm value
(`reward_belief_mean_candidate_arm_preoutcome`), candidate-arm uncertainty
(`reward_belief_variance_candidate_arm_preoutcome`), ΔV, RPE, value updates,
and belief entropy all come from it. The HMM (`default_hmm`)
was **retired from the analysis 2026-07-13**; its `hmm_*`/`reward_entropy` columns
are no longer fitted (see [`behavior_models.md`](behavior_models.md) for the HMM's
description as a generative model).

**Timing convention** (from the vendored docstrings): row *t* holds the
**pre-outcome** belief state for trial *t*; trial *t*'s outcome is incorporated on
row *t+1*. "Value update due to trial *t*" is therefore the *t→t+1* delta aligned
back onto row *t*.

## Stop → trial alignment

Each stop is assigned to a trial by Alison's convention
(`alison_decoding.add_trial_info_to_clusterless_results`): within an epoch, trial
*t* spans `[poke_out_ts[t-1], poke_out_ts[t])`, i.e.
`trial = searchsorted(poke_out_ts, stop_start, side="right")`. Stop-level
covariates are frozen at bout start throughout the replay tables; using a
midpoint can silently jump to a later trial when a low-speed bout spans a
trial boundary. The trial's
pre-outcome state is attached to every stop whose **start** lies in that
interval and is then held fixed across that bout. Validated against
signals the alignment never uses: at-well (`in_poke`) chosen arm vs
`occupancy_well_arm`
**99.9–100%**, at-well reward vs DIO `reward_outcome` **99.8–100%**. Caveats:
- **Coverage is 97–99%**; the unmapped 1–3% are *trailing* stops after the epoch's
  last `poke_out` (post-final-trial; no trial defined — Alison's trailing drop).
- **`chosen_arm` is the trial's destination leaf**, so for transit stops
  (`in_poke=False`, ~20%) it differs from `occupancy_well_arm` *by design*; reward
  (a per-trial property) still agrees ~98% there.

## Column reference

Indexing: ports/leaves are 1–6, patches/stems 1–3 (A/B/C). `{arm}` = `leaf-1`.

### Keys / per-stop flags
| Column | Meaning |
|---|---|
| `stop_id` | join key to `stop_table` (uint64) |
| `epoch` | epoch label, e.g. `"02_r1"` |
| `trial_number_by_epoch` | assigned trial (0-based) within the epoch |
| `qvals_context_time` | `stop_start`, the timestamp used for trial assignment |
| `in_poke` | stop midpoint inside `[poke_in_ts, poke_out_ts]` (at-well vs transit) |
| `pump_on_at_bout_mid` | stop midpoint inside `[reward_on_ts, reward_off_ts]` (pump on) |
| `trial_reward` | the trial's reward outcome (0/1) |
| `chosen_arm` | trial's destination leaf − 1 (0–5) |

### Value (expected reward)
| Column | Meaning |
|---|---|
| `reward_belief_mean_port_0…5` | Beta posterior mean `α/(α+β)` per port, ∈[0,1] |
| `reward_belief_mean_chosen_arm_preoutcome` | pre-outcome belief mean at `chosen_arm` |
| `decision_value_patch_1…3` | per-patch decision value; not a probability |

### Decision variables (Comrie et al., Fig 1E/F)
| Column | Paper term | Meaning |
|---|---|---|
| `stay_value_preoutcome` | **Stay value** | value of the upcoming port in the current patch |
| `switch_value_preoutcome` | **Switch value** | best alternative patch value (max over the two other patches of their 2-port mean) |
| `relative_value_switch_minus_stay_preoutcome` | **Relative value ΔV** | Switch − Stay (the central decision variable) |
| `relative_decision_value_switch_minus_stay_preoutcome` | ΔV on the **decision-value** scale | **NOT the same construct.** `decision_value_patch_1..3` range −0.983 to 19.335, so they are decision values, not probabilities: beyond `β·Q + b` they also carry the model's γ2-weighting of the occupied stem, a depletion term, and stay/turn/spatial biases (see the `_spatial_bandit_rl.py` docstring) — an affine fit on patch-mean reward belief explains only ~23%. The two ΔV variants also differ structurally: the decision-value path takes Stay = the whole **start stem**, whereas the reward-belief path takes Stay = the **sibling leaf**. Net effect: the decision-value contrast is negative on **99.90%** of rows (99 of 102,975 positive, max +0.755) against 51% positive for the reward-belief contrast; they correlate only 0.750. The manuscript defines ΔV *"without accounting for the cost of Switching or other biases"*, so **`relative_value_switch_minus_stay_preoutcome` is the construct used by the ladders.** |

### Verified against the manuscript (2026-08-20)

Checked against `references/Spatial_bandit_manuscript_v3.2_biorxiv_figs.pdf`
(Methods, "Behavioral model" / "Behavioral model variables"):

- **Beta posterior is internally consistent.** The manuscript gives
  `μ = α/(α+β)` and `σ² = αβ/((α+β)²(α+β+1))`. Inverting the shipped
  `reward_belief_mean_port_*` / `reward_belief_variance_port_*` values recovers an
  implied `α+β` spanning **2.00 to 73.26**, with the
  minimum exactly at the `α = β = 1` initialisation the manuscript specifies.
- **ΔV reproduces the manuscript's definition.** Stay = the *upcoming* port
  within the patch; Switch = the **greater** of the two alternative patches' mean
  port values; ΔV = **Switch − Stay**. The shipped
  `relative_value_switch_minus_stay_preoutcome` equals
  `switch_value_preoutcome − stay_value_preoutcome`
  **exactly on all 102,975 rows** that carry a finite value (r = 1.000000, agreement
  to floating-point tolerance). Note 102,975 counts *stop rows*, not distinct trials.
  There is no residual to explain: the γ-weighting the manuscript applies to the
  occupied patch belongs to the `_Qstem` path, not to this one.
- **Sign convention.** Switch − Stay, so ΔV is ≈ −(the current option's value):
  r = −0.952 with `stay_value_preoutcome`. A *positive* coefficient on
  `relative_value_switch_minus_stay_preoutcome_z_x_is_current_arm` means content
  sharpens onto the current arm when that arm's
  value is **low**. Earlier labels called this "Stay-vs-Switch", which inverts the
  reading; corrected 2026-08-19 in `model_specs.py` and the docs, and the manuscript
  independently confirms the direction. **One stale label remains** in
  `scripts/rate/fit_swr_contrast_rate.py:23`.
- **Scale.** `reward_belief_mean_port_*` and
  `reward_belief_mean_chosen_arm_preoutcome` are probabilities in [0,1];
  `decision_value_patch_*` is the decision value `β·Q + b` and is **not**
  comparable to them.

**One place we extend beyond the manuscript.** Its uncertainty measure (Fig 5B) is
*"the variance over the reward belief state of the **upcoming port**"* — a
per-trial scalar for one port. Our
`reward_belief_variance_candidate_arm_preoutcome` is the per-arm Beta variance
across all six leaves, which is what makes a within-event arm contrast possible.
That is a deliberate generalisation, not the manuscript's quantity; any sentence
citing the paper for "uncertainty" should say which of the two it means.

### Value-update magnitudes (Fig 5E / S5; absolute, summed across ports)
| Column | Scope |
|---|---|
| `reward_belief_mean_update_l1_all_ports_postoutcome` | all 6 ports (global) |
| `reward_belief_mean_update_chosen_arm_postoutcome_abs` | realized chosen port |
| `reward_belief_mean_update_l1_chosen_patch_postoutcome` | chosen patch (2 ports) |
| `reward_belief_mean_update_l1_alternative_patches_postoutcome` | the two patches alternative to the chosen patch (4 ports) |
| `reward_belief_mean_update_sum_all_ports_postoutcome` | signed global update (not absolute) |

### Uncertainty
| Column | Meaning |
|---|---|
| `reward_belief_variance_port_0…5` | Beta posterior variance per port `αβ/((α+β)²(α+β+1))` |
| `reward_belief_variance_leaf_sum_preoutcome` | Σ of the 6 port variances |
| `reward_belief_variance_chosen_arm_preoutcome` | variance of the chosen port |
| `prev_local_variance` | upstream diagnostic retained under its native, insufficiently documented name; outside the v1 fitted vocabulary |
| `reward_belief_variance_leaf_sum_preoutcome` | sum of the six port variances; canonicalized from the numerically duplicate global-variance paths |
| `upcoming_leaf_variance`, `next_choice_variance` | uncertainty about the upcoming choice — **look-ahead**: both use trial `t+1` (sourced from the upstream DataJoint `BehaviorModelResults`, not recomputed here). They **are** exported (listed in `COVARIATES`, present in `qvals_table.parquet`) but are **never fitted** (not in `fit_uncertainty_rate.RL_COVARIATES`); do not use them as contemporaneous stop-time covariates for a replay event at trial `t`. The repo's own session-safe analogues are the `_sess` variants. |

### Belief entropy (beta — the outcome-uncertainty covariate)
| Column | Meaning |
|---|---|
| `reward_belief_differential_entropy_leaf_mean_preoutcome` | mean per-leaf **Beta** differential entropy; the beta analog of the retired HMM entropy |

### Reward-prediction error
| Column | Meaning |
|---|---|
| `rpe_postoutcome` | `trial_reward − reward_belief_mean_chosen_arm_preoutcome` (same-trial model RPE) |
| `rpe_postoutcome_abs` | `|rpe_postoutcome|` (unsigned surprise) |

### Depletion / choice
| Column | Meaning |
|---|---|
| `stem_choice_p`, `leaf_choice_logp` | model probability of the chosen stem / log probability of the chosen leaf |
| `stem_stay_p`, `stem_1/2/3_p` | model choice probabilities |

## Sanity / alignment evidence
Alignment-sensitive features (present when correctly aligned, collapse under a
within-epoch shuffle): the model predicts the chosen stem at **89.5%** (chance
0.33; shuffled 0.35); reward rate rises with
`reward_belief_mean_chosen_arm_preoutcome` (Δ **+0.31**, shuffled +0.00).
Internal: `reward_belief_variance_port_*` matches the Beta formula,
`reward_belief_mean_port_* ∈ [0,1]`, 0 duplicate
`stop_id`.

Additional exported (not all fitted) columns: `task_reward_probability_best_arm` — the maximum nominal
p(reward) across the 6 ports (best available arm value; the nominal counterpart to
`task_reward_probability_chosen_arm`); `stem_1_p` / `stem_2_p` / `stem_3_p` — the behavior model's
P(choose stem i) for the 3 stems.

See also [`covariate_dictionary.md`](covariate_dictionary.md) for the rate/content
model-family covariates (a separate, smaller set), and
[`behavior_models.md`](behavior_models.md) for the **generative models**
(Beta-Bernoulli, HMM, Q-learners) that produce the columns above — their
assumptions, update rules, and free parameters.
