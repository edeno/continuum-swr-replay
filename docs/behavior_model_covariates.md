# Behavioral-model covariates — `qvals_table.pkl` data dictionary

Per-stop table of the SpatialBanditTask behavioral-model latent variables (value,
decision variables, uncertainty, value updates, entropy), one row per stop,
keyed by `stop_id` (joins to `stop_table.pkl`).

- **Generator:** [`analysis/behavior/qvals_table.py`](../src/continuum_swr_replay/analysis/behavior/qvals_table.py)
  — run `python -m continuum_swr_replay.analysis.behavior.qvals_table --write`.
- **Output:** `Processed-Data/replay_content/models_per_bin_rate/qvals_table.pkl`
  (gitignored data artifact; regenerate from the script).
- **Decision-variable functions:** vendored verbatim from Alison Comrie's
  `rl_helpers.py` →
  [`src/continuum_swr_replay/analysis/behavior/_spatial_bandit_rl.py`](../src/continuum_swr_replay/analysis/behavior/_spatial_bandit_rl.py).
- **Paper:** Comrie et al., *Hippocampal representations of alternative
  possibilities are flexibly generated to meet cognitive demands* (bioRxiv
  2024.09.23.613567; `references/nihpp-2024.09.23.613567v1.pdf`).

## Provenance & model

| Piece | Source (DataJoint, read-only) |
|---|---|
| Value / belief columns | `alison_rlmodel.BehaviorModelResults`, fit `beta_stable_withleaf` (Beta-Bernoulli, stable task, with-leaf) |
| Decision variables (ΔV, V_stay/V_switch, value updates, betadist_mu) | computed locally by the vendored `make_full_beta_bernoulli_dv_df` on the above |
| Reward / state entropy | `alison_rlmodel.BehaviorModelResults`, fit `default_hmm` (the paper's HMM), merged on `(nwb_file_name, epoch, trial_number_by_epoch)` |
| Poke / reward timestamps | `alison_behav.TrialsInfoByEpoch.ByTrial` |

**Model note.** The *paper* uses the HMM (`default_hmm`): belief `α` over 72
discrete contingency states, `Q = Σ αᵢφᵢ`, reward entropy `−Σ pᵢ log pᵢ` (Comrie
et al. Methods, "Behavioral Hidden Markov Model"; the HMM fit lives in DataJoint,
not this repo). This
table uses the newer **Beta-Bernoulli** fit (`beta_stable_withleaf`) for the
value/uncertainty columns (a project decision), and pulls the **HMM** reward/state
entropy in separately so the entropy covariate matches the paper. The covariate
*structure* (Q, patch value, ΔV, value updates) is identical across the two models.

**Timing convention** (from the vendored docstrings): row *t* holds the
**pre-outcome** belief state for trial *t*; trial *t*'s outcome is incorporated on
row *t+1*. "Value update due to trial *t*" is therefore the *t→t+1* delta aligned
back onto row *t*.

## Stop → trial alignment

Each stop is assigned to a trial by Alison's convention
(`alison_decoding.add_trial_info_to_clusterless_results`): within an epoch, trial
*t* spans `[poke_out_ts[t-1], poke_out_ts[t])`, i.e.
`trial = searchsorted(poke_out_ts, stop_midpoint, side="right")`. The trial's
pre-outcome state is broadcast to every stop in that interval. Validated against
signals the alignment never uses: at-well (`in_poke`) arm vs position `current_arm`
**99.9–100%**, at-well reward vs DIO `reward_outcome` **99.8–100%**. Caveats:
- **Coverage is 97–99%**; the unmapped 1–3% are *trailing* stops after the epoch's
  last `poke_out` (post-final-trial; no trial defined — Alison's trailing drop).
- **`arm` is the trial's destination leaf**, so for transit stops (`in_poke=False`,
  ~20%) it differs from the animal's physical `current_arm` *by design*; reward
  (a per-trial property) still agrees ~98% there.

## Column reference

Indexing: ports/leaves are 1–6, patches/stems 1–3 (A/B/C). `{arm}` = `leaf-1`.

### Keys / per-stop flags
| Column | Meaning |
|---|---|
| `stop_id` | join key to `stop_table` (uint64) |
| `epoch` | epoch label, e.g. `"02_r1"` |
| `trial_number_by_epoch` | assigned trial (0-based) within the epoch |
| `in_poke` | stop midpoint inside `[poke_in_ts, poke_out_ts]` (at-well vs transit) |
| `is_rewarding` | stop midpoint inside `[reward_on_ts, reward_off_ts]` (pump on) |
| `trial_reward` | the trial's reward outcome (0/1) |
| `arm` | trial's destination leaf − 1 (0–5) |

### Value (expected reward)
| Column | Meaning |
|---|---|
| `Q1…Q6` | per-port expected value (≈ p(reward) estimate), ∈[0,1] |
| `q_chosen` | `Q` of the visited port (`Q{leaf}`) |
| `Qstem1…3` | per-patch (stem) value |
| `betadist_mu1…6` | Beta posterior mean `α/(α+β)` per port (the "value" used by the DVs) |

### Decision variables (Comrie et al., Fig 1E/F)
| Column | Paper term | Meaning |
|---|---|---|
| `stay_sibling_mu_preoutcome_sess` | **Stay value** | value of the upcoming port in the current patch |
| `best_other_patch_mean_mu_preoutcome_sess` | **Switch value** | best alternative patch value (max over the two other patches of their 2-port mean) |
| `switch_advantage_mu_preoutcome_sess` | **Relative value ΔV** | Switch − Stay (the central decision variable) |
| `switch_advantage_preoutcome_sess_Qstem` | ΔV (Qstem) | same, from `Qstem` patch values |

### Value-update magnitudes (Fig 5E / S5; absolute, summed across ports)
| Column | Scope |
|---|---|
| `global_mu_update_abs_postoutcome_sess` | all 6 ports (global) |
| `chosen_leaf_mu_update_abs_postoutcome_sess` | current port |
| `chosen_patch_mu_update_abs_postoutcome_sess` | current patch (2 ports) |
| `unchosen_patches_mu_update_abs_postoutcome_sess` | unoccupied alternative patches (4 ports) |
| `global_mu_update_signed_postoutcome_sess` | signed global update (not absolute) |

### Uncertainty
| Column | Meaning |
|---|---|
| `betadist_var1…6` | Beta posterior variance per port `αβ/((α+β)²(α+β+1))` |
| `global_leaf_var_sum_preoutcome_sess` | Σ of the 6 port variances |
| `chosen_leaf_var_preoutcome_sess` | variance of the chosen port |
| `prev_local_variance`, `prev_global_variance` | pre-outcome uncertainty (chosen-leaf / global) |
| `upcoming_leaf_variance`, `next_choice_variance` | uncertainty about the upcoming choice — **look-ahead**: both use trial `t+1` (sourced from the upstream DataJoint `BehaviorModelResults`, not recomputed here). They **are** exported (listed in `COVARIATES`, present in `qvals_table.pkl`) but are **never fitted** (not in `fit_uncertainty_rate.RL_COVARIATES`); do not use them as contemporaneous stop-time covariates for a replay event at trial `t`. The repo's own session-safe analogues are the `_sess` variants. |

### Reward-state entropy (HMM — the paper's outcome-uncertainty covariate)
| Column | Meaning |
|---|---|
| `reward_entropy` | `−Σ pᵢ log pᵢ` over reward states at the upcoming/chosen port (Fig 5D) |
| `state_entropy` | entropy over the HMM contingency-state belief |

### Reward-prediction error
| Column | Meaning |
|---|---|
| `rpe` | `trial_reward − q_chosen` (same-trial model RPE; both terms from the fitted beta model). Merged into `fit_uncertainty_rate.py` and fit there under the alias **`rpe_model`** — it substitutes for the contingency-table nominal `rpe` (r≈0.71), which is dropped from the MU panel. |
| `rpe_abs` | `|rpe|` (unsigned surprise) |

### Depletion / choice
| Column | Meaning |
|---|---|
| `depletion_leaf_1…6` | per-port depletion state |
| `depletion_mean_stem_1…3` | per-patch mean depletion |
| `stem_choice_p`, `leaf_choice_p` | model probability of the chosen stem / leaf |
| `stem_stay_p`, `stem_1/2/3_p` | model choice probabilities |

## Sanity / alignment evidence
Alignment-sensitive features (present when correctly aligned, collapse under a
within-epoch shuffle): the model predicts the chosen stem at **89.5%** (chance
0.33; shuffled 0.35); reward rate rises with `q_chosen` (Δ **+0.31**, shuffled
+0.00). Internal: `betadist_var` matches the Beta formula, `Q∈[0,1]`, 0 duplicate
`stop_id`.

Additional exported (not all fitted) columns: `p_reward_best` — the maximum nominal
p(reward) across the 6 ports (best available arm value; the nominal counterpart to
`p_reward_current`); `stem_1_p` / `stem_2_p` / `stem_3_p` — the behavior model's
P(choose stem i) for the 3 stems.

See also [`covariate_dictionary.md`](covariate_dictionary.md) for the rate/content
model-family covariates (a separate, smaller set).
