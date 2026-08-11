# Replay-table subsystem audit — 2026-08-08

Twelve independent auditors over `analysis/replay/` + `analysis/hippocampus/event_table.py`
+ `analysis/contfrag/` (~4,500 lines), followed by eight adversarial verifiers instructed to
*refute* the findings that would change a published number. Findings that survived refutation
are below; findings that did not are recorded too, because knowing a scare is unfounded is
worth as much as the scare.

**No production code was changed.** Deliverable is this report plus
`tests/unit/test_replay_tables_audit_regressions.py`.

---

## 0. The meta-finding: the builder layer has no effective test coverage

A7 injected 14 silent defects into `tables.py` and ran the full 203-test subsystem suite.
**13 of 14 passed.** Survivors included:

| mutation | what it would do to the paper |
|---|---|
| `arm_axis_reverse` (arm 0↔8, 1↔7, …) | every content coefficient inverts in meaning |
| `classified_is_swr_flip` | the primary SWR contrast reverses sign |
| `posterior_time_roll` (1 bin) | every uncertainty covariate silently biased |
| `eq_indicator_zero` | all four arm indicators → 0; M1–M7 collapse to intercept |
| `bout_intervals_shift` (+1.0 s) | every Family-A `log(immobility)` offset wrong |
| `lagged_rewards_reverse` (lag1↔lag10) | the Rlag reward-kernel shape inverts |
| `arm_count_hard_constant`, `arm_marginal_roll`, `visit_recency_constant`, `dist_to_arm_constant`, `time_since_stop_zero`, `reward_rate_trial_k10=0.5`, `safe_mode_nan` | — |

Only `strict_first → safe_first` was caught (2 failures).

**Why.** The *primitive* layer is genuinely well tested with exact-value assertions —
`summaries.py`, `uncertainty.py`, `spatial_mapping.py`, `hippocampus/event_table.py` (~20
tests covering all six overlap geometries), and the covariate helpers in `covariates.py`.
The *builder* layer — what every published number is fit on — is tested almost entirely with
**per-event `groupby.sum()` invariants, which are invariant under any permutation of the nine
arm columns**, over a fixture whose `trials` frame is empty, leaving 30 columns all-NaN and
18 more constant. `is_swr`'s only assertion is `.dtype == bool`.

The two integration parity tests check the `_z` *formula* only; per their own docstrings the
raw-column rebuild "is NOT run here." **There is no end-to-end gate comparing a rebuilt table
against real data.** That is the root cause that let everything below persist.

No test tolerance was ever widened and there is no `xfail` anywhere in the subsystem — this
is a coverage gap, not erosion.

---

## 1. What is correct (verified, not assumed)

Worth stating plainly, because these were the highest-risk hypotheses going in:

- **Junction-bin exclusion (`edge_id == -1`) is correct on every MAP→arm and posterior→arm
  path in both pipelines.** Four agents confirmed independently. On real data the 8 junction
  columns are entirely NaN and are sliced out *before* any arithmetic; real-bin row sums are
  0.999996–1.000005; `arm_fraction` sums to exactly 1.0 per event in both shipped tables; the
  degenerate all-junction fallback fires on **0 of 65,811** in-event bins.
- **Referential integrity is perfect.** 0 orphan `event_id`/`stop_id` in either direction,
  exactly 9 arm rows and 3 patch rows per event (593,217 = 9 × 65,913; 197,739 = 3 ×), 0
  duplicate primary keys, all join keys `uint64` on both sides, no `_x`/`_y` survivors,
  Σ`n_non_local` = 65,913 = `len(event_table)` exactly.
- **No float-time `.loc` slicing** anywhere in the subsystem (the inclusive-both-endpoints
  double-count risk does not exist here); event→bout assignment is `searchsorted`-based.
- **Threshold inequalities are consistent** across `_in_event_mask`,
  `immobility_bout_intervals`, `build_stop_table`, `event_table.py`, and both contfrag
  builders. No `<` / `<=` divergence.
- **Every event lies inside its stop's window** (0 violations on all four boundary checks),
  and zero-event stops are genuinely retained (83,136/107,400 = 77.4%).
- The value block is **byte-identical across 141,021 rows** between the two pipelines;
  entropy guards `0·log0` correctly and is in nats throughout; **no circular quantity is
  linearly averaged** (direction comes from the sign of a slope on a signed graph geodesic —
  the correct construction, not the classic bug).
- The documented Δ82/Δ250 count-convention asymmetry still holds exactly, and no code was
  found cross-substituting counts between the stop-level and bin-level conventions.
- `next_*` is **response-only and clean**: `next_arm`/`next_patch`/`is_next`/`is_stay` appear
  in no spec as predictors.

---

## 2. Confirmed findings

### C1 — `is_after_switch` encodes the in-progress trip (CONFIRMED)

`_helpers.py:83,124-130`. Trial context is indexed by `searchsorted(trial_start_times, …)`,
and a trial spans `[start_time, goal_arrival_time)` = departure → arrival (`trials.py:369-387`).
During the run the destination is not yet reached, so `is_after_switch` describes a trip that
has not happened.

The repo's recorded defense (`.claude/docs/reviews/06_covariate_provenance.csv:34`) is
"destination chosen at departure." **The verifier falsified it empirically** by asking where
the animal physically is during transit stops — on this spider track the same-patch vs
different-patch choice is forced only at the *patch junction*:

| location during transit stop | senor | wilbur |
|---|---|---|
| **still on origin leaf arm — choice not yet made** | **80.1%** | **72.2%** |
| trunk arm (committed) | 13.3% | 17.2% |

Also, `to_well` is *measured* from position at the run's end, so the label is constructed
from a future observation.

**Scope, corrected from the auditor's framing.** Transit stops are ~24–30% of rows but only
~4–5% of exposure — so the row framing overstates leverage ~7×. The honest number is that
**~55% of every `is_after_switch == 1` row refers to an uncompleted trip**, and the
within-transit rate contrast differs sharply from the post-arrival one, so the rows push the
contrast directionally rather than merely attenuating it. Since `is_after_switch == 1` is
over-represented among the shortest stops *because of the timing error*, the repo's own
conclusion that "`is_after_switch` is a dwell artifact" (`claims_table.md:53`) is itself partly
downstream of the error.

`covariate_timing.py:41` lists it `PRE_OUTCOME_SAFE`, so `assert_predictor_timing` **passes** —
the guard gives false assurance. Affected: R1/R2/Rlag (`model_specs.py:549,557,584`), per-bin
`M8_full_switch` via `STOP_COVARIATES`, and — missed by the auditor — **`MODEL_PATCH_SWITCH`
PS0–PS3, whose entire cohort is defined by `is_after_switch == 1`**.

Not an oversight: `phase-4-covariate-causal-hygiene.md:67` says "Keep `is_after_switch`
(pre-outcome-safe)". The repo's own **unexecuted** `phase-8-covariate-timebase.md:53` already
calls for removing it. `covariate_dictionary.md:82` and `:207` contradict each other on
whether it is current- or previous-trip.

*Not affected:* content ladders use `recent_switch_rate` (backward-safe), so claim 19f's
`switch_x_current = +0.53` is clean.

### C2 — `arm_uncertainty_z` is standardized over a support that is 1/3 placeholder (CONFIRMED, larger than reported)

`standardization.py:69-71` computes the pooled SD over all nine arms; `covariates.py:817-822`
fills trunk arms (not reward sites) with a constant `0.0`. Pooled SD 0.034314 vs leaf-only
0.021175 → **ratio 1.6205**. Leaf `_z` has SD 0.617, not 1.

The verifier's decisive test: varying the arbitrary trunk placeholder and re-standardizing
swings the reported coefficient **12-fold (+0.27 → +3.20)** while the t-statistic is invariant
to six digits and the raw-unit slope is exactly invariant (15.498). So this is purely a
**reporting-unit** error — signs, t-stats, p-values, ΔLR and animal-consistency counts are all
untouched. Every consumer carries arm fixed effects, which absorb the constant entirely.

`standardization.py:8` documents the pooled convention, so the code matches its own spec —
but `model_specs.py:256-257` claims the `_z` form is "well-scaled (SD 1)", which is false on
the identifying support, and `uncertainty_content.py:660` calls the difference "slight."

**Restatement needed — divide by 1.6205:**

| quantity | published | corrected |
|---|---|---|
| Mu (clean) — the published **exp(β) = 2.18** | +0.7784 | **+0.4803 → exp(β) = 1.62** |
| Mu (all) | +0.7502 | +0.4629 |
| claims_table / memory "Mu +0.83" | +0.83 | **+0.51** |
| C13 SWR / non-SWR / all | +0.9869 / +0.5952 / +0.8164 | +0.609 / +0.367 / +0.504 |
| CS6 main | +1.0168 | +0.628 |

**Missed by the auditor:** diagnostics panel E (`_plot_horserace`) puts the pooled-standardized
`arm_uncertainty_z` on the same "per 1 SD" axis as predictors standardized over the
non-current-leaf subset. That panel needs **1.6805** (different comparator population). Its
qualitative conclusion survives (+0.038 vs −0.012 / −0.002 / −0.002, still ~3× the nearest
competitor) but it currently overstates the margin.

*Latent, not live:* `q_arm` (ratio 2.096) and `p_reward_arm` (1.170) have the same structure
but are entered **raw** in every ladder; their persisted `_z` columns are consumed by no model.

### C3 — rate-ladder rungs are fit on different cohorts (mechanism CONFIRMED, headline REFUTED)

`per_bin_rate_model.py:129-135` and `per_stop_rate_model.py:96-106` drop NaNs against each
rung's *own* predictor list. `freeze_common_cohort` exists in `cohort_selection.py` and is
called by `scripts/content/` but **by nothing in `scripts/rate/`**.

The ladder splits into **four** cohorts (the regression test found a fourth the verifier
missed): 7,387,816 / 7,348,182 / 6,849,272 / 6,466,795 bins, differing in 218/218 sessions.
Family A likewise: R0 107,380 / R1,R2 106,511 / Rlag 100,077 stops — and its CV means are
taken over **different session sets** (R0 218, R1/R2 204, Rlag 164).

**The headline did not survive.** Refitting M5 and M7r on a common cohort (60 sessions, all 5
animals, 2,288,111 bins) preserves the ranking — M7r better on **48/60**, not the predicted
coin flip — but shrinks the margin from 14,472 to **1,355 AIC (9.4% surviving)**. The
auditor's rescaling was unstable (pooled and per-session forms give opposite signs) and biased
against M7r, because the bins M7r drops are *easier* than average.

What is genuinely unsupported:
1. "One bin width ⇒ AIC/deviance/GOF comparable" (`docs/rate_models.md:68`,
   `per_bin_rate_model.py:21-22`). Sharing a bin *width* is not sharing bins. CV bits/event
   **is** comparable — it refits `count ~ 1` inside the same dropna'd frame.
2. `docs/rate_models.md:216` is correct as a ranking but inflated ~10× as evidence.
3. **Panel E of `paper/figures/rate/per_bin_rate_ladder_summary.png`** plots ΔAIC across all
   rungs = a direct cross-cohort comparison, with a ~204 AIC/session mechanical offset. Note
   `M2_rewhist`'s −164 is essentially *all* artifact (its per-bin llf beats M0's by 0.00025).

Within a cohort block the comparison is perfectly valid — M7r vs M7 vs M8 is legitimate.

### C4 — `visit_recency` fills never-visited arms with `len(trials)+1` (CONFIRMED, severity major not critical)

`covariates.py:64-66`; no production caller ever supplies `n_default`. The fill is the
session's **total** trial count — unknowable at event time, varying 127–280 across sessions.

The strongest defense fails decisively: **no event has the fill on all nine arms** (max 5 leaf
arms, always alongside a real-valued leaf), so arm fixed effects and conditional-Poisson
conditioning do *not* absorb it. Trunk fill contributes SS = 0.0 exactly; leaf fill contributes
**57.5%** of `visit_recency_demeaned`'s sum of squares on 4.78% of rows.

Two of the auditor's claims are wrong as worded: it is *not* out of range in the pooled
marginal (fill median 182 vs legitimate max 198) — the correct and stronger framing is
**within-event**, where fill ÷ largest legitimate trials-ago has median 33×. And it is *not*
undocumented — `covariate_dictionary.md:196-198` states the sentinel explicitly; only the
docstring omits it.

**Blast radius is bounded** (the verifier's contribution): β scales as ≈1/fill, so the fit is
nearly scale-equivariant and the co-fitted headline coefficients barely move (`q_arm` ≤0.23,
`arm_uncertainty_z` ≤0.22, `reward_rate_arm_k10` ≤0.13 at 4× fill, no sign flips). **Published
claim 19g survives**: over all 218 sessions the horse-race recency coefficient stays an order
of magnitude below `arm_uncertainty_z` in every variant — and since the fill is 54.5% of that
column's SS, this is a real robustness result, not a null test.

What *is* damaged: the reported `visit_recency_demeaned` / `swr_x_visit_recency_demeaned` /
raw `visit_recency` **coefficients themselves** (M4–M7, Mv/Mu/Mvu, C2–C13/CS, BASEc_PATCH,
MODEL_CHOICE) are a never-visited contrast in arbitrary units rather than a graded trials-ago
slope, and flip sign when the fill rows are removed (3/5 sessions).

### C5 — contfrag artifacts predate the encode change → 11% non-random event loss

Verified directly. contfrag tables are dated Jun 30 / Jul 1; the `reward_rate_arm_k10` encode
commits (`14300fc`, `649952b`) landed **2026-07-07**. Event-level drop (the fitter discards all
9 arm rows if any is NaN): **contfrag 4,232/38,558 = 10.98%** vs **replay 347/65,913 = 0.53%**.
The drop criterion is non-random (unvisited-leaf → early-session). A stale-artifact problem —
**fix by rebuild, not edit.**

### C6 — contfrag stop table cannot fit the rate ladder at all

`contfrag/tables.py:787` omits the `reward_lag{1..10}` block its replay twin emits
(`tables.py:860-862`). `_freeze_common_cohort` takes the *union* of predictors across the
block, so every rung including R0 returns `status="missing_columns"` and an empty cohort. The
entire rate hierarchy for both contfrag sources produces **zero fits**, recorded only as a
status string. Worse, the stored artifacts show `SWR_HPC_comparison_fits.pkl` has
`{R0,R1,R2,R3}` with `coef_is_after_omission` while `non_local_HPC` has `{R0,R1,R2,Rlag}` —
**two different models under the same names, related by a documented sign flip.**

### C7 — contfrag `is_swr` is a third convention (3 agents, independently)

`contfrag/tables.py:364` uses the inclusive `any(SWR_event > 0)` rule; the canonical
event-level rule is strict positive-duration overlap over the full label window. One-sided:
contfrag's flag is always ≥ canonical, so any cross-pipeline "% SWR-coupled" comparison is
biased high. `docs/known_issues/swr_mua_count_conventions.md` documents only the stop-vs-bin
pair. It also gates `fit_replay_content_models.py --swr-subset`.

Related: in the SWR-sourced table `is_swr` is constant `True`, making all seven `swr_x_*`
columns bit-identical to their main effects. **`model_specs.py:444-451` already drops P4/M6/M7
for contfrag sources**, so no fitted model is affected — but the degenerate columns ship in the
canonical table.

### C8 — value-block join drops a 1.9% session tail — DOWNGRADED on investigation: correct behavior, undocumented

**Follow-up investigation (2026-08-08) resolved this. It is not a data-integrity defect.** The
recording is fully valid in those stops — over 699 unmatched stops in 27 sessions, **0 NaN
bins** in `head_position_x/y`, `head_speed`, `linear_position`, `map_estimate`, or any state
probability; 0 frozen-position stops (the matched baseline has 9); non-local events fire at
0.480/s of immobility vs 0.507 matched. What has ended is the **trial sequence**, so there is
no trial *t* to attach a value to.

Two classes, from DIO pokes across all 218 sessions:

| | sessions | stops | events | pokes after last `poke_out` | median tail |
|---|---|---|---|---|---|
| A. rat quit early | 8 | 314 | 454 (51%) | **0 in every session** | 140 s |
| B. hit the trial cap | 210 | 1,742 | 442 (49%) | median 1, ~0.6 s total | 6.7 s |

Trial counts pile up at exactly 180/200/300 (38% are exactly 180) — a cap, not truncation.
95.4% of unmatched event-arm rows have NaN `next_arm`/`is_stay` (vs 2.7% matched): the animal
never went anywhere afterward.

**At stake: 896/65,913 events = 1.36%** (214 SWR, 682 non-SWR); 1,868 s of 130,000 s
immobility = 1.44% of exposure. **The base ladder (M0–M5, `BASEc`) loses nothing** — the stops
and events are present with valid non-value covariates; NaN is confined to
`q_arm`/`p_reward_arm`/`arm_uncertainty`, exactly and only in this set, disjoint from the
`reward_rate_arm_k10`/`visit_recency` gaps.

**Do not "fix" by carrying values forward** — that imports post-task rest into the value
coefficients. Instead: (1) document the exclusion as "stops past the end of the modeled trial
sequence" and log value-block coverage at build time; (2) sensitivity-check the 8 quit-early
sessions if per-session or per-animal estimates are quoted — three lose 21–33% of their events
(`wilbur20210329_16_r8` 33%, `j1620210707_16_r8` 24%, `j1620210706_10_r5` 21%), so base and
value rungs are fit on measurably different samples there.

Mechanism: `behavior/qvals_table.py:271-293` (`assign_stops_to_trials`) and `:343-378`; the
left-merge at `replay/prepare_data.py:336-341` is why stops survive and only value columns
go NaN.

### C8-orig — the join mechanics (unchanged, verified sound)

2,056/107,400 stops have no `qvals_table` row. Unmatched vs matched: mean within-session
position **0.984 vs 0.491**, mean `reward_outcome` **0.134 vs 0.526**. Cause is upstream
(`qvals_table.py:359-364`: stops past the last `poke_out` get `ti = -1`), so the loss is a
trailing per-epoch **block**. Every value coefficient is estimated with the low-reward
end-of-session regime absent, and **no runner logs value-block coverage**, so it leaves no
trace in the run log. The join *mechanics* are sound — key, dtype, cardinality all verified.

### C9 — the value block is stale by one outcome at the current arm (CONFIRMED, framing inverted)

`q_arm` and `arm_uncertainty` are joined from the beta-Bernoulli fit's row for the trial the
stop maps to, and that row is the **pre-outcome** belief: at trial 0 of every session
`betadist_α/β = 1`, `Q = 0.5` regardless of that trial's reward, and
`α_chosen(t+1) − α_chosen(t)` is +0.81 after reward vs −0.12 after omission — trial *t*'s
outcome lands on row *t+1*. The vendored pipeline itself reads post-outcome from *t+1*
(`_spatial_bandit_rl.py:346`).

**The auditor had the direction backwards.** Every modeled non-local event occurs *after*
reward delivery — on `event_onset_table.pkl`, the fraction with `t_since_pump < 0` is
**0.0000**, min +0.035 s, median +4.07 s. So the reward-history block (which includes the
just-completed trip) correctly encodes what the animal has experienced at event time; it is the
**value/uncertainty block that is stale**. Offset magnitude at the current arm: **0.39 SD of
`q_arm`**, 0.33 SD of `arm_uncertainty_z`; `corr(q_pre, reward) = 0.286` vs
`corr(q_post, reward) = 0.782`. Sibling columns carry explicit `_preoutcome_sess` /
`_postoutcome_sess` suffixes; `q_port{i} → q_arm` carries none.

Also note the auditor's stated evidence (`max|trial_reward − q_chosen − rpe| = 0.0`) is
**circular** — `qvals_table.py:226` *defines* `rpe = reward − q_chosen`. It should not be cited.

**Both named headlines survive.** Refit over 15 sessions, all 5 animals: `swr_x_q_arm` moves
3–6% with no sign change under either correction (−3.151 shipped, −3.354 history-corrected,
−3.053 value-corrected). The `within_arm_horserace` behind "uncertainty ≠ recency" restricts to
`is_current == 0`, and the offset is a current-arm phenomenon — out of scope by construction,
and empirically unmoved (0.0589 → 0.0588).

**One real sensitivity, not named by the auditor:** `C4_armvalue`'s SWR `q_arm` **flips sign**
(−0.267 → +0.059) when the value block is advanced to post-outcome. C4 is a descriptive split
rung, not a headline. The clean fix is to add a post-outcome variant (row *t+1*) and report
CS5/C13/Mu against both — *not* to change the history block, which is right.

### C10 — smaller confirmed items

| id | finding |
|---|---|
| C9a | `attach_is_swr_to_onsets` (`_helpers.py:559-592`) pairs onsets to events by **within-stop ordinal rank**; its guard detects only *un*matched rows, never *mis*-matched ones. A time-based reconciliation is available and unused. Currently correct (`onset_time == event_start` exactly for all 65,913). |
| C9b | `theta_z` 100% NaN for **26/218 contfrag sessions** — **root-caused, see below**. |
| C9c | `is_reward_stop` in the canonical onset/timebin tables is effectively constant `False` (235/138,142 = 0.17%): it tests whether a pump falls *inside* an immobility bout, but pumps fire ~0.79 s *before* bout start. |
| C9d | All interval durations are `(k−1)·dt` — one 2 ms bin short. Affects `immobility_time`, the `log()` exposure offset; duration-dependent so not absorbed by the intercept, but measured small (median stop −0.6%; refit moves `reward_lag1` RR +0.3%). 20 stops get exactly 0 s exposure and are silently dropped. |
| C9e | `nonlocal_fraction` ≡ 1.0 and `nospike_fraction` ≡ 0.0 for every event — structurally constant, because the rollups are computed over already-filtered rows. Consumed by no model. |
| C9f | `ab_slope` is fit by `np.polyfit` on raw Unix-epoch timestamps (~1.58e9) over a ~0.05 s window; for the 1.26% of events with exactly flat `ahead_behind`, 1e-12 float noise decides the binary `is_outbound` (0.84% flip vs a mean-centred fit). Consumed by no model. |
| C9g | Doc drift: `models_table.md:143-148` names `is_after_omission` for R1/R2 where the code fits `reward_lag1` (**opposite sign**); `models_table.md` omits four whole ladders (Mu/Mvu, PS0–PS3, C0–C13/CS, CP0–CP4) including the rungs carrying the cited uncertainty results; `build_event_table`'s docstring says `current_arm` is the modal **`decoded_arm`** when the code uses `track_segment_id` (the docstring describes a circular computation — if "fixed" to match, `is_current` would become tautological); `model_specs.py:39-40` still documents the retired NaN behavior. |
| C9h | Shipped `event_arm_table.pkl` carries three orphan columns (`q_arm_hmm`, `swr_x_q_arm_hmm`, `q_arm_hmm_z`) that no code emits or reads — the table cannot be reproduced by the current builder. `corr(q_arm, q_arm_hmm) = −0.196`, so confusing them gives a sign-inverted covariate. |

---

### C9b root cause (investigated 2026-08-08) — one NaN poisons the whole session

Not missing data. **The theta band is present in Spyglass for every affected session** —
missing `LFPBandV1` rows, wrong `filter_name`, interval-naming mismatch, bad electrodes, and
`.squeeze()` axis bugs are all ruled out.

Chain: `_nearest(...)` reindexes theta onto the decode axis with
`method="nearest", tolerance=0.004`, producing NaN wherever no theta sample falls within 4 ms.
Two sources of such gaps — **genuine LFP acquisition dropouts of ~180–190 ms** sitting inside a
single continuous valid interval (the same gap appears in the ripple-band LFP, so it is
LFP-wide, not theta-specific), and for j16 a **1.359 s tail overhang** where the decode axis
runs past the end of the theta LFP. The contfrag files were then built with a version calling
bare `scipy.stats.zscore(...)` (no `nan_policy`), so mean/std go NaN and **all** bins go NaN.

Reproduced bit-for-bit:

| session | LFP gaps | reindex NaN | stored `theta_phase` NaN | stored `theta_power_zscore` NaN |
|---|---|---|---|---|
| chimi20200213_02_r1 | one 181.2 ms | 87 | 87 | **1807725 / 1807725** |
| peanut20201202_02_r1 | 189.9, 189.2 ms | 182 | 182 | **540209 / 540209** |
| j1620210707_12_r6 | none; 1.359 s tail overhang | 678 | 678 | **822478 / 822478** |
| chimi20200213_04_r2 (control) | one 8.2 ms | 0 | 0 | **0** |

`theta_phase` is not z-scored, which is why it shows only the genuinely-bad bins — that
asymmetry is the fingerprint.

**Why the NonLocal `_80` files are clean: code vintage, not data source.** `tolerance=0.004`
landed 2026-04-07 (`71d19eb`); before it the reindex filled gaps with the nearest sample at any
distance → 0 NaN. The contfrag aggregator was added 2026-04-16 — after the tolerance, before
the fix. `nan_policy="omit"` landed 2026-05-07 (`d6defdc`) and **is in the current code, in both
aggregators**. The Jun-23 mtimes are a red herring: `_augment_data_over_time.py` only appends
trajectory columns and reuses the stored z-scores. `replay_content/event_table.pkl` has
`theta_z` NaN fraction 0.0000 across all 218 sessions.

**Repair (cheap, no re-decode, no `load_data`):** per affected session, `get_theta(nwb, epoch)`
→ reindex onto the stored index with the same tolerance → `theta_power_zscore` → overwrite the
column → rebuild event tables. Verified by reproduction. Yields finite `theta_z` for ~99.99% of
bins. Sidesteps the known `load_data` position-trim drift.

**Still open:** `contfrag/data_aggregation.py:167-169` has no finiteness guard, unlike
`run_speed_baseline` at `:120-124` which raises. Add one there and at
`hippocampus/data_aggregation.py:452` — it would have failed loudly in April instead of
silently shipping 26 dead sessions. The j16 tail overhang deserves separate attention (same
class as the Apr-24 `get_position_info` trim).

Affected (26/218): chimi 20200213_02_r1, 20200214_02_r1, 20200218_06_r3; j16 20210707_12_r6;
peanut 20201127_14_r7, 20201129_06_r3, 20201129_10_r5, 20201202_{02_r1,04_r2,06_r3,10_r5,16_r8};
senor 20201027_02_r1, 20201030_{10_r5,14_r7}, 20201101_{02_r1,04_r2,06_r3,10_r5,12_r6,16_r8},
20201102_04_r2, 20201104_16_r8, 20201105_11_r5, 20201106_{04_r2,10_r5}.

## 3. Refuted / downgraded — do not act on these

| finding | verdict |
|---|---|
| **Per-bin design predicted by undelivered rewards** (A4-3) | **REFUTED.** Mechanism stated backwards: immobility onset *follows* the pump by a median 0.48 s. Only **313 of 733,666 bins (0.043%)** precede reward delivery. `reward_lag1`'s identity with `reward_outcome` on post-arrival stops is the definition working, not failing. |
| **0.0 fill inflates apparent low-value avoidance** (A2-2) | **Direction inverted.** Refitting the production rung: the fill *attenuates* a negative effect toward null (shipped p=0.31 vs p=0.0013 neutral). Shipped estimate is the **conservative** one. Also a documented, reviewed decision — an indicator was built, measured, and deliberately dropped. |
| **Trajectory boundary step biases SWR contrasts** (A1-1) | **Downgraded to minor.** Already documented in the design spec's "Known caveats" *and* the code comment. 58.2% of boundary steps are exactly 0 cm (not a "teleport"). A drop-last control shows ~90% of the apparent effect is the fix's own order-statistic shrinkage. Consumed by no model or scripted figure. |
| **"M7r lowest AIC is a cohort artifact"** (A3-1 headline) | **REFUTED** by common-cohort refit — ranking holds 48/60. Only the magnitude (~10×) and the comparability claim are wrong. |
| **Straddling-event double-count** (B4-4) | **Latent, not firing.** Σ`n_non_local` = `len(event_table)` = 65,913 exactly, 0/218 sessions mismatched. |
| **`is_mua` uses a different window than `is_swr`** (B4-3) | **Real but inert.** Two agents measured identical `is_mua` under both windows for every event. |
| **`align_count_matrix` dtype-dependent fill** (A2-3) | **Unreachable today** — `n_junction_bins` is 0 for all 65,913 events. |
| **`eq_indicator` silent-False** | **Benign under this estimator** — an event-constant predictor is absorbed by conditional-Poisson conditioning; score-equivalent to exclusion. |

A structural note that resolves several concerns at once: the "filtered survivor window" and
the "full label window" are **byte-identical for all 65,913 events** (four agents confirmed
independently) — `_in_event_mask` is a no-op because the decoder's `non_local_event` label
already encodes both the immobility and probability conditions. So no window-mixing bug can
currently change a number, and `label_event_*` carries no information beyond `event_*`.

---

## 4. Regression tests added

`tests/unit/test_replay_tables_audit_regressions.py` — 4 pass, 4 `xfail(strict=True)`.
Strict xfail keeps the suite green while a defect stands and turns it red the moment a fix
lands, so no marker is silently left behind.

**Coverage (pass) — close the mutation-survival gap:**
- `test_arm_count_hard_is_pinned_per_arm_not_only_per_event_sum` — catches `arm_axis_reverse`
- `test_arm_fraction_is_pinned_per_arm` — catches `arm_marginal_roll`
- `test_is_swr_and_is_mua_values_are_pinned_not_only_dtype` — catches `classified_is_swr_flip`
- `test_swr_interactions_follow_the_is_swr_flag`

**Failing regressions (xfail-strict) — one per confirmed defect:**
- `test_visit_recency_does_not_depend_on_future_trials` (C4) — appending later trials must not
  change an earlier event's value. The cleanest statement of the defect.
- `test_trial_context_is_not_taken_from_the_in_progress_trip` (C1)
- `test_arm_uncertainty_z_is_unit_sd_on_the_support_that_identifies_it` (C2)
- `test_rate_ladder_rungs_share_a_cohort` (C3) — this is what found the fourth cohort block.

---

## 5. Suggested priority (no fixes applied — your call)

1. **C3 panel E** and the `docs/rate_models.md:68` comparability claim — a published figure
   currently plots ΔAIC across four cohorts. `freeze_common_cohort` already exists.
2. **C2 restatement** — `exp(β) = 2.18` should read 1.62. Cheap: divide, or persist an
   `arm_uncertainty_zleaf`. No refit needed; no p-value moves.
3. **C1** — `phase-8-covariate-timebase.md` already specifies the fix (split into
   `previous_completed_trial_is_switch` vs `current_trip_is_switch`); it is unexecuted.
4. **C5/C6** — rebuild contfrag artifacts; add the `reward_lag` block. Until then the
   cross-pipeline rate comparison is vacuous and the content comparison is on an 11%-smaller
   non-random cohort.
5. **C4** — replace the sentinel with an explicit never-visited indicator, or right-censor at
   a value fixed independently of session length.
6. **C9** — add a post-outcome `q_arm` / `arm_uncertainty` variant (row *t+1*, which the
   vendored pipeline already reads) and report CS5/C13/Mu against both. Cheap and additive;
   do **not** change the history block. Also give `q_arm` the `_preoutcome_sess` suffix its
   sibling columns carry.
7. **Section 0** — the builder-layer fixture needs a non-empty `trials` frame. That single
   change would make ~48 currently-uncheckable columns testable.

### Findings whose scope was *narrowed* by verification — do not over-correct

Three confirmed findings turned out to leave the headline numbers intact, and it is worth
recording that explicitly so a later reader does not re-escalate them:

- **C2** changes reporting units only — no sign, t-statistic, p-value, ΔLR or animal-consistency
  count moves.
- **C4** damages the recency coefficients themselves but not the co-fitted headlines; claim 19g
  ("uncertainty ≠ recency") survives over all 218 sessions.
- **C9** moves `swr_x_q_arm` by 3–6% with no sign change, and cannot touch the
  `within_arm_horserace` at all (it excludes the current arm, where the offset lives).
