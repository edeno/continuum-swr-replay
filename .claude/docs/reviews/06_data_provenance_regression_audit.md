# Data-Provenance & Regression-Integrity Audit

**Date:** 2026-07-04
**Scope:** Full trace from the three root inputs → canonical tables → content / rate / prediction
regression coefficients. Answers: *for each fitted coefficient, can we name the upstream data
product, the grain, the join, and whether the covariate was available at the time the model
implies?*
**Method:** 5 parallel reviewers, each evidence-driven against the **materialized** tables in
`Processed-Data/replay_content/` (not code-trace alone). Empirical numbers below were computed by
loading the real tables.

---

## Verdict

**The provenance and relational backbone is sound and unusually leakage-aware.** Every canonical
table traces cleanly to one of the three roots; every stated grain and primary key was verified
unique against the real data; every child→parent join is clean (0 unmatched, 0 duplicate keys, 0
fan-out); and **no fitted model uses an outcome-derived variable as a predictor** — the future-choice
columns (`next_arm`, `is_next`, `ends_at_next`, `chosen_arm_fraction`) are correctly quarantined to
the response side.

The residual risks are **not** in the data plumbing. They are (1) an **inference-layer**
pseudoreplication gap that is fixed only in the `is_swr`-contrast runners, and (2) a specific,
enumerable set of **post-/future-outcome behavioral covariates** used as *predictors* in the value
and peri-switch contrast rungs, whose time-availability is violated for pre-reward events. Both are
partly documented in-source. There is also a **non-random complete-case shrinkage** (two disjoint
mechanisms) that should be reported as retained-N wherever it bites.

The documented state-space point-process (Czanner SS-GLM) model is **planned/design-only** — no
fitter exists. The actual modeling flow is canonical tables + per-session conditional-Poisson /
conditional-logit / Poisson-GLM regressions with behavior-HMM covariates merged on `stop_id`.

---

## 1. Data-flow map

```
THREE ROOT INPUTS
 R1 loaded session data      load_data(): data_over_time.pkl, trials, track_graph, DIO/pump/reward
 R2 non-local detector out   *_data_over_time.pkl (event/SWR/MUA bins, MAP, posterior), environment.pkl, results.nc
 R3 imported behavior/HMM    qvals_table.pkl  <- DataJoint alison_rlmodel.BehaviorModelResults (beta + default_hmm)
                                              <- alison_behav.TrialsInfoByEpoch.ByTrial (nominal p_reward)
                                              built ON the stop_table spine (shared stop_id namespace)
        │
        ▼  canonical builders  (event_table.py, replay/tables.py, replay/timebin_tables.py, behavior/qvals_table.py)
  event_table (65913×102)  ── stop_id ──►  stop_table (107400×41)  ◄── stop_id ──  qvals_table (105344×97)
        │  event_id                              │ stop_id                              (R3)
        ├──► event_arm_table  (593217 = 65913×9) ── q_arm/q_arm_hmm/p_reward_arm merged from qvals
        ├──► event_patch_table(197739 = 65913×3)
        ├──► event_onset_table(138142, sparse onsets)
        └──► stop_timebin_table.parquet (32,533,075 = one row per (stop,4ms bin))   [exploratory only]
        │
        ▼  per-session statsmodels regressions  (NO state space)
  CONTENT   ConditionalPoisson over event_arm/event_patch, groups=event_id      fitting.py:159
  RATE A    Poisson GLM, offset=log(immobility), 1 row/stop                      per_stop_rate_model.py:129
  RATE B    Poisson GLM, offset=log(exposure), 1 row/(stop,15ms bin), bs(tc)+history  per_bin_rate_model.py:148
  CHOICE    ConditionalLogit over stop×well-arm choice set, groups=stop_id       choice_prediction.py:491
        │
        ▼  aggregate across sessions → animals
  contrast_aggregation: per-animal median + n=5 sign consistency (is_swr ladders ONLY)
  base ladders: session-level Wilcoxon signed-rank (descriptive)
```

**Root-tracing result:** all loads map to exactly one root (or a deterministic projection of them).
The only cross-root coupling is `qvals_table`'s dependence on `stop_table.pkl` for its stop spine —
which is precisely what guarantees the shared `stop_id` join key. **No untraced source.**

> Note: `stop_timebin_table.parquet` (32.5M rows) is read only by `scripts/exploratory/_timeresolved_psth.py`.
> Rate Family B does **not** consume it — it rebuilds its (stop,bin) design on the fly via
> `_helpers.aggregate_to_timebins` from `stop_table` + `event_onset_table`. The parquet is a
> standalone descriptive artifact, not a wiring gap.

---

## 2. Table contracts (grain / key / source / merged-from-HMM) — all verified empirically

| table | rows | grain | primary key (verified unique) | HMM/behavior cols merged | key consumers |
|---|---|---|---|---|---|
| event_table | 65913 | 1 / non-local event | `event_id` | none | rate fitters, figures, regenerate_stop_table |
| stop_table | 107400 | 1 / immobility stop (incl. 0-event) | `stop_id` (==`(animal,day,epoch,stop_idx)`) | none | all rate fitters, qvals spine, choice |
| event_arm_table | 593217 | 1 / (event,arm), exactly 9/event | `(event_id, arm)` | `p_reward_arm`,`q_arm`,`q_arm_hmm` | content, choice, switch/retrospective |
| event_patch_table | 197739 | 1 / (event,patch), exactly 3/event | `(event_id, patch)` | `*_patch` leaf-means (no `q_arm_hmm_patch`) | content `--hierarchy patch` |
| event_onset_table | 138142 | 1 / event onset | `(event_type, stop_id, onset_time)` | none | rate Family B, PSTH |
| stop_timebin_table | 32,533,075 | 1 / (stop,4ms bin) | `(stop_id, bin_in_stop)` | none | exploratory PSTH only |
| qvals_table | 105344 | 1 / behavioral stop | `stop_id` (⊂ stop_table) | (is the R3 source) | value-block attach, contrast rate/content |

**Provenance-integrity flag (reproducibility):** the `_z` columns on `event_arm_table` and
`qvals_table`, and historically `q_arm_hmm` on `event_arm_table`, are stamped **in-place by
exploratory scripts** (`scripts/exploratory/_add_standardized_cols.py`, `_augment_qarm_hmm.py`), not
by the canonical builders. A clean `prepare_data.py` rebuild emits `q_arm_hmm` natively but **drops
the `_z` columns** until the standardization one-off is re-run. This is a silent-regeneration hazard.

---

## 3. Invariant-check report (empirical, PASS/FAIL)

| invariant | expected | observed | verdict |
|---|---|---|---|
| Σ stop.n_events == len(event_table) | 65913 | 65913 | PASS |
| n_non_local_SWR + n_non_local_no_SWR == n_non_local | per-stop 0 mismatch | 0 | PASS |
| Σ n_non_local_SWR == Σ event.is_swr | 26413 | 26413 | PASS |
| every event has exactly 9 arm rows | 0 off | 0 | PASS |
| every event has exactly 3 patch rows | 0 off | 0 | PASS |
| stop_id unique in stop_table / qvals | 0 dup | 0 / 0 | PASS |
| no dup (animal,day,epoch,stop_id,bin) in timebin | 0 of 32.5M | 0 | PASS |
| bins×4ms ≈ immobility_time | ≤4ms over | resid ∈ [−4,0] ms (ceil-binning) | PASS |
| is_swr ⟺ overlap_class ⟺ matched_swr_event | consistent | exact 3-way | PASS |
| all child→parent joins (event/arm/patch/onset/timebin → parent) | 0 unmatched | 0 | PASS |
| Σ timebin n_SWR == Σ stop_table n_SWR | equal | 38729 vs 38811 (Δ82, 0.21%) | **FAIL (convention)** |
| Σ timebin n_MUA == Σ stop_table n_MUA | equal | 33500 vs 33750 (Δ250, 0.75%) | **FAIL (convention)** |

- **SWR/MUA Δ is a dual-convention artifact, not corruption:** `stop_table` uses the inclusive
  per-timebin `is_swr_stop` rule over the full bout; timebin/onset count onsets inside the
  ceil-binned window. Boundary-straddling events are counted by one and not the other.
  **`n_non_local` matches exactly** across tables → it is the safe cross-table count. **Do not
  cross-substitute `n_SWR`/`n_MUA` between stop-level and bin-level tables in an exposure/offset.**

---

## 4. Complete-case / sample-size summary

| model | input N | drop mechanism | final N | flag |
|---|---|---|---|---|
| Content M0–M7 (arm) | 65913 events | union complete-case incl. `reward_rate_arm_k10` NaN | **~58.2k (−11.6%)** | **non-random** (see below) |
| Content Mv/Pv (value block) | 65913 | `q_arm_hmm`/`p_reward_arm` NaN (2056-stop gap) | ~65k (−1.4%) | excluded from default run by design |
| swr-contrast content C0–CS5 | 65913 | per-rung NaN drop; **no common cohort across rungs** | rung-dependent | ΔAIC not on identical events |
| Rate Family A R0–Rlag | 107400 stops | `dropna` + `immobility>0` | ~107k | no HMM dependency → no shrink |
| Rate Family B base M0–M9 | 107400 stops | per-bin `dropna` + `exposure>0` | per-session | no HMM dependency → no shrink |
| Rate Family B swr-contrast G/S | 107400 | **inner join stop×qvals** | **105344 (−1.9%, logged)** | non-random tail (see below) |
| Prediction (choice) | replay-stops w/ valid next_arm | stop-wise NaN drop, ≠1-chosen drop | replay-stops | — |

**Two disjoint non-random drops (empirically confirmed, `both = 0.0000`):**

1. **`reward_rate_arm_k10` NaN → 11.65% of events** dropped from the content M3+ ladder. The NaN is a
   **documented deliberate** value (`covariates.py:189`–`:198`): a leaf arm with no prior visit history
   has an *undefined* — not zero — reward rate, so it is left NaN by design (trunk arms are filled `0.0`).
   It is not a silent/accidental drop. The subsequent complete-case exclusion sits on **leaf arms**
   (5–6% each; trunk 0.5%) and is therefore biased toward **early-session / rarely-visited-arm states**.
   Because the ladder uses a *union* complete-case set (to keep ΔAIC on identical events),
   **M0–M2 are also refit on the reduced set.** This is the single largest selection effect in the
   content family and is *not* the qvals gap. (Resolved: mechanism-A **encodes** this state —
   `reward_rate_arm_k10=0` + `has_arm_history` indicator — retaining the events; see the phase-4 plan.)
2. **2056-stop qvals tail → 1.9% of stops / 1.36% of events (896 events).** A per-epoch **trailing
   tail** (211/211 epochs; `min(missing idx) > max(present idx)` in every epoch), enriched for
   **late-session unrewarded** stops (86.6% unrewarded vs 47.4% baseline). This is exactly the
   low-value regime that drives the `swr_x_value` construct-dependence result — so value-content and
   value-rate models should **report retained-N and confirm the value effect survives** the drop.

---

## 5. Ranked findings

### HIGH — inference validity

**H1. Pseudoreplication in the base content and both rate ladders.**
The per-animal aggregation that correctly treats animal (n=5) as the independent unit
(`contrast_aggregation.py:89`) is wired **only into the `is_swr`-contrast runners**. The base content
M0–M7 ladder (`fit_replay_content_models.py:241`) and the per-stop / per-bin rate ladders report
**session-level Wilcoxon + naive within-session model SEs**, treating many sessions/animal as
independent. Any headline coefficient from the base ladders inherits **understated uncertainty**.
→ *Fix:* route the base ladders through per-animal aggregation, or report cluster-robust / GLMM SEs
clustered on animal.

**H2. Post-/future-outcome behavioral covariates used as predictors (time-availability violation).**
For events occurring **before reward delivery on the current trip**, these predictors encode
information not yet available:
| covariate | mechanism | used in | severity |
|---|---|---|---|
| `is_after_reward`/`is_after_omission`/`reward_outcome` | indexed by trip-in-progress; = current (undelivered) reward. Disagrees with causal `reward_lag1` on **10.0%** of stops | Rate Family A **R1/R2** | confirmed-leak, **mitigated** — Family B uses `reward_lag1` (goal-arrival cutoff, past-only) instead |
| `hmm_rpe` / `rpe` | `reward − Q(t)`, needs current outcome | Rate **G5/S7**, Content **C6** | plausible |
| `hmm_value_update` | `Σ|Q(t+1)−Q(t)|`, references **next** trial | Rate **G7/S9**, Content **C9** | plausible |
| `trials_since_switch` / `is_post_switch` | signed distance; negative = trials until the **next** switch | Rate **G1/S3**, Content **C10** | plausible |
→ *Fix:* restrict value/RPE/update/peri-switch predictor fits to **post-consummatory bins**, or state
explicitly that these rungs are descriptive peri-event alignments, not causal predictors. The
confound-free Family B design already models this correctly with `reward_lag1`.

### MEDIUM — reporting / robustness

- **M1.** Non-random complete-case drops (§4): report retained-N for the 11.6% `reward_rate_arm_k10`
  drop and the 1.9%/1.4% qvals tail; confirm the value-content result survives the tail.
- **M2.** No common complete-case cohort across content contrast rungs → ΔAIC across C-rungs is not on
  identical events (source-documented deferral).
- **M3.** **No convergence check** in the primary GLM/conditional fitters (`fitting.py`,
  `per_bin_rate_model.py`, `per_stop_rate_model.py`, `choice_prediction.py` call `.fit()` without
  inspecting `mle_retvals["converged"]`; only `learning.py` does). Non-convergence returns
  finite-but-wrong coefficients, backstopped only by the `|coef|<10` filter at aggregation.
- **M4.** **Perfect separation** yields extreme finite coefficients (documented for short self-history
  lags, `per_bin_rate_model.py:118`); filtered only at contrast aggregation (`|coef|<10`) — the base
  ladder and choice conditional-logit can still surface inflated coefs.

### LOW — documented / benign

- **L1.** SWR/MUA count convention mismatch (§3) — document; don't cross-substitute in offsets.
- **L2.** Global (pooled) z-scoring of `_z` covariates — **currently safe**: the only cross-validated
  family (per-bin rate) fits on **raw** columns; `_z` columns appear only in non-CV in-sample
  inference, and an unsupervised rescale of X can't leak the response into a coefficient. Latent
  hazard if a `_z` column is ever added to a CV'd formula.
- **L3.** `KFold(shuffle=True)` over stops (`per_bin_rate_model.py:231`) — bins kept within stop
  (good); minor residual cross-stop autocorrelation across folds.
- **L4.** `replay_frac` in the choice model may partly launder `dist_to_arm` (dropped for
  quasi-separation, animal picks closest arm 95–99%). Documented; principled fix (fit on
  non-closest-choice stops) deferred.
- **L5.** Local-only BH-FDR (within predictor set / population), not family-wide across
  ladder×population×term. Softened by reliance on n=5 sign-consistency over p-values.
- **L6.** Post-hoc `_z`/`q_arm_hmm` schema divergence (§2) — reproducibility hazard on clean rebuild.

### CONFIRMED CLEAN (verified, no action)

- Every child→parent join clean; grains/keys unique; ×9 arm / ×3 patch cardinality exact.
- **No outcome-derived predictor leakage in any fitted model.** `next_arm`/`is_next`/`ends_at_next`/
  `chosen_arm_fraction` are response-only; `is_next` explicitly excluded from every predictor ladder.
- Content conditioning is correct: event-level scalars enter **only** as arm-interactions;
  `within_group_full_rank()` rejects any within-event-constant design column.
- Both rate families use **true `offset=`** (coefficient fixed at 1), not a free covariate.
- Collinearity handled: VIF + condition number computed per fit; within-group rank guards reject
  unidentified designs.
- Point-process time-rescaling KS GOF implemented and used (Family B rate).
- `reward_lag*`, `visit_recency`, `cumulative_reward`, `reward_rate_arm_k10` windows are all
  **trailing/backward** (goal-arrival cutoff), not look-ahead.

---

## 6. Answer to the driving question

For **every fitted coefficient**, provenance is fully recoverable:
- **upstream product** → §1 root map + §2 contracts (machine-readable in `06_covariate_provenance.csv`);
- **grain** → §2 (verified unique keys);
- **join** → §3 (all clean, spine = `stop_id` / `event_id`);
- **time-availability** → **mostly yes**, with the enumerated exceptions in **H2** (post-/future-outcome
  value, RPE, update, and peri-switch predictors) that are *not* available at the implied time for
  pre-reward events. These are the covariates to either time-gate or re-label as descriptive.

The state-space point-process model remains **planned** — treat any reference to it as a design target,
not an implemented result.
