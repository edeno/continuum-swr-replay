# Full pipeline refresh: final audit and report

Date: 2026-10-12. Runbook step R8 of
[`.claude/docs/plans/full-pipeline-refresh/PLAN.md`](../../.claude/docs/plans/full-pipeline-refresh/PLAN.md)
(section "R8: audit and report", plus the R13 additions). Written by the R8 agent from the brief
`.superpowers/sdd/PLAN/r8-brief.md`.

**Updated 2026-10-16** with the day's follow-up results (brief `.superpowers/sdd/PLAN/r8-update-brief.md`).
Section 10 has one subsection per follow-up. The update also changed §0, the content rows of §3, §4.6, §4.7,
§5, §6.1 and §6.6. Each change says where its numbers come from.

**This report changes no number in any other document.** Section 6 lists the hand-maintained
doc claims that need new numbers. The project lead reviews them before any doc is requoted
(plan R8 "STOP").

What "before" and "now" mean everywhere below (the R8a definitions, `r8a_report.md` §0):

- **Before:** the golden baseline `Processed-Data/replay_content/_golden_baseline_pre_redecode_20261003/`
  (tables on 218 epochs, fits written 2026-09-12), the April decodes, the September SWR labels, no
  artifact rule. PFC before: the 71 canonical-suffix mPFC bundles in
  `Processed-Data/_pre_r13_backup_20261011/mpfc/` (2026-09-17/18, commit `db01df0`).
- **Now:** the canonical R13c tables (built at `bb19449`), the R6b refits, the R11 decodes, SWR run
  `056eee0dcb5a`, artifact run `fbbff144a123`, Widloski run `d0964cc76ed2-seed20260922`, and the 75 PFC
  bundles at `110c9ae`. Cohort: 242 analysed epochs (255 less the 13 `SUSTAINED_NOISE` epochs).

**Audit: PASS.** `audit_refresh_consistency.py audit --check-marks` exited 0 on 2026-10-12 (19:58 PDT, main at `d29cc3c`): 17 of 17 checks ok, none failed, none skipped, no `--exceptions` file (section 1).

## 0. Summary

- **Audit:** see section 1.
- **What changed, and why** (R8a, section 3 here):
  - The code changed no fitted result. Current code rebuilds the golden tables and rate fits exactly
    from the same inputs. It changed two descriptives: the content-quadrant definition (2026-08-12)
    and the gallery's SWR-not-overlapping-NL count (−260).
  - The SWR reward rate ratio rose (3.44 → 6.39) through the SWR tetrode rule R-T with the mask
    normalisation (mostly j16) and the 35 added epochs.
  - The non-SWR non-local reward rate ratio fell (2.00 → 1.42) through R-T and the added epochs.
  - The SWR × current-arm content term (CS1), the current-arm main effect (C1) and the
    Dirichlet-multinomial `log_phi_is_swr` moved with the R11 decode, which loses short non-SWR
    own-arm events.
  - CS2, CS3, CS5 (`swr_x_reward_belief_mean_candidate_arm_preoutcome`), CS6 and the non-local-with-SWR rate moved within noise.
- **Pre/post-26a tables:** only `theta_z`, `compression_factor` (85 epochs with excluded time) and the
  stop-location columns (877 stops touching excluded time) changed; everything else is byte- or
  value-identical (R13c, section 2.1).
- **Not run in the refresh:** Task 12 on the R11 decodes (only the 5-epoch pilot on pre-R11 decodes
  exists), Task 13's half-tetrode decodes, and the R13 robustness and sensitivity reruns (section 4.7).
  *Update 2026-10-16:* Task 12 (240 of 242 epochs) and Task 13 have now run; R13 ran in part and is paused
  (sections 4.6, 4.7, 10).
- **Open items:** 44 — 1 High (the doc requotes), 10 Medium, 33 Low (section 5). *Update 2026-10-16:* the
  follow-ups close 10 items and add 10. Still open: 44 of 54, again 1 High, 10 Medium, 33 Low.
- **Hand-maintained doc claims needing new numbers:** 78 (section 6). Eight of them change a stated
  conclusion, not only a number (section 6.1): the NL w/ SWR vs no-SWR baseline rates no longer differ;
  NL w/ SWR is no longer the burstiest stream; the non-SWR time-course peak is back on the window edge;
  the non-SWR immediate reward response is weaker (kernel lag 1 = 1.01) while the paired dissociation is
  stronger; the CS5 three-estimator sign pattern reversed and no interval excludes zero; CS1's
  intervals include zero; the own-well share is 0.13, not 0.30 (a 2026-08-12 script change); and the
  arm-value main effect is 3/5 animals with an interval that includes zero. *Update 2026-10-16:* a ninth
  conclusion change (CS5b, from R6c), and corrections to V4 and V6 (from F2), are in section 6.1a. Two of
  the 78 claims (M2, M3) are now fixed on `full-pipeline-refresh-code` (section 6.6).

**Follow-ups, 2026-10-16** (section 10):

- **The content fits now reach their optimum.** F3 found the shipped BFGS fits short of the optimum (gap up
  to 4.81 log-likelihood) in a nuisance spline term. Task 28 fixed the fitter, and R6c refit the content
  family. The terms F3 checked move in the third or fourth decimal, with no sign change. Five rows move more,
  mostly by cohort, and must be re-read before quoting: CS5b, C3 SWR, C13 SWR, CS3 and the omitted stratum
  (section 3; R6c also marks C2_base SWR's animal count). The rewarded-stratum
  CS5b row is the CS5 model, not a previous-goal control.
- **The SWR reward rate ratio does not rest on j16** (6.89 without it). About a fifth of it rests on the 31
  added early peanut epochs (5.29 without them). The NL-no-SWR response is the fragile number (1.21, p 0.083,
  without wilbur). The paired dissociation holds in every leave-out (F2).
- **CS1 is negative because SWR content shifts onto the trunk arms**, and the "+0.88 vs +0.53" split compares
  unpaired animal medians (F2). See section 6.1a for the wording.
- **Spike evidence holds on the R11 decodes.** Held-out tetrodes support immobile Non-Local Continuous bouts:
  6.10 nats per bout, 5/5 animals, 240 epochs (F1). Non-SWR events carry less evidence per spike than SWR
  events. Half-tetrode labels support Local and the immobile non-local states, more weakly, but not No-Spike
  (Task 13, 5 epochs).
- **chimi20200221_08_r4:** its decode is correct. The check failed on float32 subnormal occupancy, which the
  CPU flushes to zero (D1). Task 29 fixes the checks. The artifact rule's V0 and V3 variants move no headline
  beyond its interval; V1, V2 and X_R10nr are paused (R13 interim).
- **The lead's decisions:**
  - the review of this report;
  - the merges `full-pipeline-refresh-code` → `memoryless-sequence-detection` → `master`;
  - resuming R13 (V1, V2, X_R10nr; about 4–8 h);
  - keeping chimi20200221_08_r4 (D1 option (a), the coordinator's ruling);
  - push access: the commits on `full-pipeline-refresh-code` from `a857191` onward (24 before this update)
    are not pushed (`origin/full-pipeline-refresh-code` is at `d29cc3c`), and the non_local_detector issue is not filed.

## 1. Audit

Command, run from the main tree at `d29cc3c` (tmux `r8_audit`, log `logs/r8_audit_20261012/audit.log`):

```
uv run --no-sync python scripts/qa/audit_refresh_consistency.py audit \
  --output Processed-Data/_refresh_20261012/audit.json --check-marks \
  --widloski-manifest Processed-Data/replay_detection_comparison/memoryless/d0964cc76ed2-seed20260922/manifest.json \
  --swr-manifest Processed-Data/replay_detection_comparison/swr/056eee0dcb5a/manifest.json
```

- Exit 0. `audit.json`: `ok: true`, `failed_checks: []`, `not_checked: []`, `repo_head` `d29cc3c`.
- The expected decoder settings came from the code (`expected_settings.source`: `expected_decoder_settings()`). No `--expected-settings` override and no `--exceptions` file were used.
- The log's one traceback is JAX's CUDA plugin probe under `CUDA_VISIBLE_DEVICES=''` (no device); the audit runs on CPU and is not affected.

| Check | Result |
|---|---|
| `one_commit` | ok |
| `decoder_settings_nonlocal` | ok |
| `decoder_settings_contfrag` | ok |
| `grids_match` | ok |
| `minimum_sample_counts` | ok |
| `widloski_hashes` | ok |
| `swr_spans` | ok |
| `spike_loader_unchanged` | ok |
| `tables_recorded` | ok |
| `pfc_sidecars` | ok |
| `fit_provenance` | ok |
| `decode_folder` | ok |
| `epochs_registered` | ok |
| `pending_epochs` | ok |
| `sustained_noise` | ok |
| `epoch_sets` | ok |
| `encoding_marks` | ok |


## 2. Drift and cohort comparisons

### 2.1 Tables before and after task 26a (the R13c drift gate)

Source: `.claude/research/refresh_r13/r13c_report.md` §4–§6 and the JSONs in
`.claude/research/refresh_r13/r13c/` (`drift_{stop,qvals,replay,contfrag,refs}.json`,
`drift_stop_changes.json`). R13c built every table at `bb19449` (task 26a plus review fixes) and
compared each staged table with the canonical one on its keys. Confirmed there:

- Row counts, key sets and row order are identical in every table. `event_id` and `stop_id` sets are
  unchanged in all 15 compared tables.
- **Byte-identical:** `qvals_table` and its standardization reference, `event_arm_table`,
  `event_patch_table`, `event_sequence_table`, `standardization_reference.json`, `event_onset_table`,
  `stop_timebin_table`, and the ContFrag SWR/MUA arm and patch tables.
- **`event_table` (and both ContFrag event tables):** only `theta_z` and `compression_factor` differ,
  and only in the 85 epochs with excluded time (NonLocal: 25,752 and 25,405 of 75,675 events). They are
  bit-identical in the other 157 epochs. `compression_factor` changes by one constant per epoch
  (ratio 0.99664–1.00144), as expected when only `run_speed_baseline` moves.
- **Stop tables:** only the ten location columns differ, and only on the 877 stops that touch excluded
  time (439 wholly inside it, 438 partly). The 439 wholly excluded stops now read `unknown`. Two partial
  stops (24 ms and 4.6 ms, 0 events) went `mixed` → `in_transit`. The largest distance change on a
  partial stop is 1.38 cm.
- **Well-anchored cohort `{at_well, mixed}`:** 93,996 → **93,560** stops (436 leave, none join).
- **ContFrag stop tables** gained the four `reward_history_*` columns (task 25); they equal the NonLocal
  values on all 120,320 stops.
- **Theta SD per epoch** in the 85 epochs with excluded time: median 0.530 → 0.681 (NonLocal). A gap
  remains against the 157 epochs without excluded time (0.68 vs 0.74, Mann–Whitney p = 0.003). It is
  not explained (open item, section 5).

R6b then refit every family on these tables (`.claude/research/refresh_r13/r6b_cpu_report.md` §4): of
369 fit artifacts, 334 are byte- or numerically identical, 2 are exact rescales (R09 z-reference, R05/R07
SD reference, ≤ 0.2 % in any SD) and 33 changed, all traced to the 436 leaving stops, the two short
partial stops, the `is_stay` change on 420 stops, or the newly fit X04/X05 rate block.

### 2.2 `scripts/qa/compare_fits.py`, rate families

Command (log `logs/r8_audit_20261012/compare_fits_rate.log`, exit 0); output
`Processed-Data/_refresh_20261012/compare_fits_rate.md`. Old = the golden baseline's
`models_per_bin_rate/{all_models,swr_contrast_rate,swr_contrast_rate_audited}.pkl` and
`models_per_stop_rate/all_models.pkl`; new = the canonical files.

**Tool caveat (confirmed).** The script's default mode was written for the 2026-09-04 reward-basis
change. Its "before" reward-step sections look for the retired rung names (`M2_rewhist`,
`M7_time_hist_rew`, Family A `R2`/`Rlag`), which the golden baseline does not have, so those "before"
cells are empty ("0 sessions"). Both sides use one specification here. The sections that do compare
like with like:

| quantity | before (218 epochs) | now (242 epochs) |
|---|---|---|
| `M5_time` median KS, swr / non_local / nl_swr / nl_no_swr | 0.081 / 0.066 / 0.089 / 0.079 | 0.083 / 0.062 / 0.074 / 0.074 |
| `M5_time` share of sessions inside the KS band, same order | 0.862 / 0.908 / 0.931 / 0.922 | 0.876 / 0.934 / 0.942 / 0.967 |
| frozen per-bin cohort, median `n_bins` per session | 29,314 | 29,232 (207 shared sessions; 57 changed) |
| Family A `R1 − R0` median ΔAIC, session-pooled, same order | −2.43 / −5.37 / −1.79 / −0.44 (217) | −4.26 / −4.20 / −3.14 / **+0.88** (241) |
| Family B flagged-separated share, `M7r_time_histcos_rew` | 38.3 % | 40.7 % |
| Family B CV-scored share, `M7r_time_histcos_rew` | 16.5 % | 12.2 % |
| is_swr-contrast flagged share, contrast/primary | 70.0 % | 70.5 % |
| audited is_swr-contrast aggregate rows emitted | 6 per population (primary) | 6 per population (primary) |

The new-only sections (reward kernel, paired test, marginal and conditional reward steps, arrival
clock, CV validity) are transcribed into section 6.3, since they are what
`docs/per_bin_rate_results.md` quotes.

### 2.3 `scripts/qa/compare_fits.py --content`

Command `--content --content-framing cohort --old-content-root <golden baseline>` (log
`logs/r8_audit_20261012/compare_fits_content.log`, exit 0); output
`Processed-Data/_refresh_20261012/compare_fits_content.md`. The script's prose describes the
2026-09-12 cohort change it was written for; its tables are the before/now comparison here. Its
audited `is_swr`-contrast rows reproduce R8a's values (CS1 −0.3730 → −0.0685, CS4 −1.1975 → −1.0032,
CS5 +0.2363 → −0.0119, CS5nt −0.6506 → −0.2594, CS6 +0.0039 → −0.0585, C1 0.8732 → 0.7813), so the two
accessors agree. Sessions fitted per bundle rose 217 → 241 (`models`), 215 → 237 (leaf ladder). Rows
not in R8a that docs quote are listed in section 6.

### 2.4 `scripts/qa/compare_pfc_cohorts.py`

**Could not run on this pair (confirmed, a tool limitation, not a data problem).** Both modes exit 1:

```
ValueError: Processed-Data/_pre_r13_backup_20261011/mpfc holds bundles scored out of fold,
which carry no likelihood-ratio statistics; read them with inference='heldout'.
```

The script always reads the "before" tree with `inference="legacy"`
(`compare_pfc_cohorts.py:2627-2629`); it was written for the ML → penalised change, and cannot compare
two held-out (ridge) trees. Logs: `logs/r8_audit_20261012/compare_pfc_cohorts.log` (auto → estimator
mode) and `compare_pfc_cohorts_cohortmode.log`. No code was changed. The PFC before/now comparison is
R8a's `s1_pfc.py` (`.claude/research/refresh_attribution/s1_pfc.txt`, R8a §6), which reads both trees
with the same held-out accessor:

- Figure 4 n: 10,907 → 11,604. The 69 shared sessions keep the same 10,587 units; −320 from the two
  removed senor S epochs; +1,017 from six added peanut20201128 sessions.
- Raw p < 0.05 SWR / replay / no-SWR / beyond control: 28.5 / 39.9 / 24.4 / 24.7 % → 30.8 / 40.2 /
  22.8 / 23.4 %. Best held-out model const / SWR / replay / SWR+no-SWR: 49.3 / 12.8 / 33.1 / 4.8 % →
  48.2 / 13.7 / 32.9 / 5.2 %.
- Shared units: SWR raw-p 28.6 → 30.5 % (j16 24.1 → 36.7 %, other animals within ±1 point); engagement
  24.7 → 22.8 %. Per-unit turnover of significant calls is about 25 % and is not attributed.
- PFC was not refit inside the R8a ladder, so the within-shared changes are attributed only by the j16
  pattern (hypothesis: R-T, as for the hippocampal SWR rate).

## 3. Headline numbers, before against now, with causes

**Source:** `.claude/research/refresh_attribution/r8a_report.md`, section 0 table and summary, copied
here without re-deriving. Its inputs: Stage 1 (`s1_headlines.{csv,pkl}`, `s1_per_session.txt`,
`s1_paired.txt`, `s1_events.txt`, `s1_decode_summary.txt`, `s1_pfc.txt`) and the Stage 2 sandbox ladder
on 205 shared epochs (`stage2/s2_ladder.txt`, `s2_summary.txt`, `s2_bootstrap.txt`). The parts of each
row sum to the total (`s2_summary.txt`, check sum 0). **Bold** marks a step whose 95 % paired session
bootstrap (2,000 resamples within animal) excludes zero.

| Step | What it changes |
|---|---|
| cohort (before side) | 218 → the 205 ladder epochs: the 11 S epochs removed, plus 2 j16 epochs that R3 lacks |
| code | 10-03 golden code → bb19449, with the same April decodes and the same September SWR labels |
| Kay 2.0 | September SWR labels → Kay 2.0 on all tetrodes (inventory R3) |
| R-T + mask | R3 → R10: tetrode rule R-T, mask-excluded normalisation and the artifact rule |
| decode | April → R11 decodes |
| ref | the fit-time reference cohort, 205 vs 242 epochs (content/DM only) |
| cohort (now side) | 205 → 242: the 35 added epochs plus the 2 j16 epochs |

| Headline | before → now (R8) → R6c, content rows only | cohort (before) | code | Kay 2.0 | R-T + mask | decode | ref | cohort (now) |
|---|---|---|---|---|---|---|---|---|
| per-bin M7r immediate RR, SWR | 3.44 → 6.39 | +0.04 | 0 | −0.33 | **+2.13** | 0 (exact) | – | +1.12 |
| same, NL with SWR | 7.18 → 9.91 | −0.15 | 0 | +0.78 | −0.45 | +1.87 | – | +0.67 |
| same, NL no-SWR | 2.00 → 1.42 | +0.01 | 0 | +0.02 | **−0.36** | +0.01 | – | −0.26 |
| rate contrast G0, nonSWR | 0.680 → 0.635 | 0 | 0 | 0 | −0.01 | +0.05 | – | −0.09 |
| rate contrast G0, SWR | 1.919 → 2.202 | −0.01 | 0 | +0.07 | −0.02 | +0.24 | – | 0 |
| CS1 swr × current | −0.373 → −0.069 → **−0.0709** (242, 1/5) | 0 | 0 | +0.05 | +0.06 | **+0.21** | 0 | −0.01 |
| CS2 swr × prev | −0.646 → −0.425 → −0.4186 (240, 1/5) | +0.01 | 0 | +0.06 | −0.07 | +0.23 | 0 | −0.01 |
| CS3 swr × last-rewarded (64–98 sessions identified; R6c 55) **re-read** | −1.156 → −0.758 → **−0.5525** (55, 0/5) | 0 | 0 | +0.20 | −0.31 | +0.73 | −0.15 | −0.06 |
| CS4 swr × reward fraction (raw; std −0.361 → −0.307 → −0.3089) | −1.198 → −1.003 → −1.0086 (242, 0/5) | 0 | 0 | 0 | +0.20 (borderline) | +0.01 | 0 | −0.02 |
| CS5 swr × arm value = `swr_x_reward_belief_mean_candidate_arm_preoutcome` (raw; std +0.028 → −0.001 → −0.0017) | +0.236 → −0.012 → −0.0142 (238, 2/5) | 0 | 0 | −0.19 | +0.09 | +0.04 | 0 | −0.19 |
| CS5nt (raw) | −0.651 → −0.259 → −0.2528 (239, 0/5) | 0 | 0 | +0.18 | +0.08 | +0.13 | 0 | 0 |
| CS6 swr × uncertainty | +0.004 → −0.059 → −0.0585 (242, 2/5) | +0.003 | 0 | −0.02 | −0.01 | −0.04 | 0 | +0.01 |
| C1 current arm (fraction) | 0.873 → 0.781 → 0.7814 (242, 5/5) | 0 | 0 | 0 | 0 | −0.07 (DM: **−0.09**) | 0 | −0.02 |
| DM `log_phi_is_swr` (C1 rung) | 1.817 → 1.724 (not refit in R6c) | +0.002 | 0 | −0.02 | +0.13 | **−0.21** | 0 | +0.01 |
| DM CS1 | −0.297 → −0.036 (not refit in R6c) | 0 | 0 | +0.01 | −0.03 | **+0.27** | 0 | +0.01 |
| immobile Local / No-Spike (descriptive) | 0.777/0.151 → 0.702/0.229 | +0.007/−0.001 | 0 | 0 | 0 | −0.061/+0.068 | – | −0.021/+0.012 |
| non-SWR content own well (descriptive) | 0.299 → 0.130 | +0.010 | −0.162 (script, 2026-08-12) | label (Kay + R-T together) +0.004 | | −0.028 | – | +0.007 |
| SWR not overlapping NL (count) | 12,245 → 7,061 | −391 (and −242 from the 2 j16 epochs R3 lacks) | −260 | +2,391 | −6,141 (j16 −6,447) | −1,122 | – | +581 |
| PFC (Figure 4 n) | 10,907 → 11,604 | −320 | | | | | | +1,017 |
| PFC raw-p SWR / engagement | 28.5/24.7 → 30.8/23.4 % | +0.1/0.0 | not refit: shared sessions +1.9/−1.9 pts | | | | | +0.3/+0.6 |

**R6c values (added 2026-10-16).** The content rows now end with R6c's value: the refit with the Task 28
fitter, as animal median (sessions, animals positive). Source: `r6c_report.md` §5.4, the re-aggregation
table (accessor `aggregate_contrast` with each bundle's own per-SD scale; CS3 by R8a's no-gate accessor).
- The step columns are still R8a's. They were fit with the old BFGS fitter. F3 infers that each step value
  can carry optimizer noise up to about 0.007, below their two-decimal precision (`f3_optimizer_report.md`
  §5; inferred, not re-derived).
- CS3's R6c value rests on 55 sessions, not 98: 74 uninformed coefficients are now NaN. Its step columns no
  longer describe the current value.
- Not refit in R6c, so unchanged: the DM rows, the rate rows, the PFC rows and the descriptive rows.

**Rows R6c says must be re-read before quoting** (`r6c_report.md` §5.4 and concern 3). R6c also marks
C2_base SWR.

| quantity | R8 (R6b fits) | R6c | change: fit / cohort | why |
|---|---|---|---|---|
| CS5b, stratum all | −0.4234 (155, 1/5, q 0.21) | −0.0608 (198, 2/5, q 0.16) | +0.0813 / +0.2812 | 74 sessions added, 29 removed. In many added sessions the previous-goal control `swr_x_is_last_rewarded_noncurrent_arm` is unidentified, so the per-session estimand is mixed |
| CS5b, rewarded stratum | withheld | −0.0526 (237, 2/5) | – | both previous-goal controls are unidentified in every session, so this row **is the CS5 rewarded model, not a previous-goal-controlled estimate** |
| C3 SWR | −0.1617 (222, 0/5) | −0.2570 (241, 0/5) | +0.0032 / −0.0984 | 19 added sessions, each with `is_candidate_arm_never_visited` unidentified |
| C13 SWR | +0.3398 (217, 5/5) | +0.3704 (239, 5/5) | 0 / +0.0306 | 22 added sessions, the same |
| CS3 (no-gate accessor) | −0.7583 (98, 1/5) | −0.5525 (55, 0/5) | 74 coefficients now NaN | still gated out of the audited table |
| omitted stratum, every rung | – | 16 of 44 audited rows move by more than 0.1 | e.g. C2_base nonSWR +1.301 (14) → +0.482 (18), −0.001 / −0.818 | 13–40 sessions per rung; full list in `r6c_runs/_decompose.csv` |
| C2_base SWR | +0.8839 (221, 4/5) | +0.8823 (241, 4/5) | +0.0001 / −0.0018 | F3 predicted 5/5 on the same 221 sessions. wilbur's median is ≈ 0, so neither 4/5 nor 5/5 carries weight |

The plan's headline list maps onto this table as follows: its arm-value term = CS5 (`swr_x_reward_belief_mean_candidate_arm_preoutcome`); the reward-gating rate
ratios = the three per-bin M7r rows and the two rate-contrast rows; CS4/CS5/CS6 and the DM
`log_phi_is_swr` as named; PFC engagement = the last row.

**R8a's summary, in short** (its §0 "Short form" and §10):

- **Code changes nothing.** With the same decodes and labels, current code rebuilds the golden tables
  exactly (every column except the cohort-standardised `*_z`) and refits the golden rate coefficients
  exactly (817/817 M7r rows). Content moves ≤ 0.02 except CS3.
- **The SWR reward-RR rise is R-T (in j16) plus the cohort.** Kay 2.0 itself lowers peanut and wilbur.
  The decode cannot change SWR onsets.
- **The NL no-SWR fall is R-T plus the cohort.** R-T moves reward-responsive j16 (and chimi/senor)
  events into the SWR class.
- **The NL-with-SWR rise is not separable from noise** within the shared epochs; the cohort adds
  +0.67.
- **CS1, C1, DM `log_phi` (and DM CS1/CS4nt) move with the decode.** The R11 decode loses 4,718 short
  (44 ms), mostly non-SWR (96 %), own-arm events.
- **CS4 moves at R-T** (borderline). **CS2, CS3, CS5, CS6 move within noise**; their five-animal medians
  switch animal.
- The own-well content change 0.30 → 0.13 is a **script definition change of 2026-08-12**, not the
  refresh.

**Caveats R8a states (§11–§12), carried here:**

1. Several "before" values that R7 quoted came from older figures or code, not from the 10-03 bundles:
   rate-contrast nonSWR 0.0 (bundle 0.68), quadrants 0.30/0.27 (June script), SWR-not-NL 12,245 (June
   catalog), PFC 17,280 / 23–20 % / 52–6 % (bundles 10,907 / 28.5–24.7 % / 49.3–4.8 %). Quoting R7's
   deltas would misstate the refresh.
2. CS3 is gated out of the audited table in both bundles (> 20 % of sessions unidentified); its figure
   value rests on 64–98 sessions and moves ±0.15 with the fit cohort alone.
3. Not split: R-T against the mask normalisation; which decoder change (spike fix, P30-B transitions,
   Gaussian Local, 7-sample rule, missing artifact bins) moves CS1/C1/`log_phi`; and the PFC changes
   (not refit).
4. The five-animal median is fragile (CS1's median animal was j16 before and senor now).
5. The ladder's L3 step (R11 + R10 without the artifact rule) was dropped by the agreed plan; by Stage 1
   counts the rule itself moves 21 NL events, 2 SWRs and 263 stops holding 4 events.

## 4. The plan's other report items

### 4.1 Task 1: the MUA version shift, and non-local events under both rules

Source: `.superpowers/sdd/PLAN/task-1-report.md` ("Step 5: version-shift report"; full tables in
`.claude/research/refresh_version_shift/outputs/version_shift_report.md`). Five pilot epochs, one per
animal, `ripple_detection` 1.7.0 against the pinned 2.0 (`68e1784`).

**MUA (HSE) events**, strict positive-duration overlap:

| epoch | 1.7.0 | 2.0 | matched | missed | added |
|---|---|---|---|---|---|
| chimi20200225_02_r1 | 161 | 178 | 161 | 0 | 17 |
| wilbur20210329_12_r6 | 118 | 118 | 118 | 0 | 0 |
| j1620210707_06_r3 | 195 | 214 | 195 | 0 | 19 |
| peanut20201202_10_r5 | 202 | 227 | 202 | 0 | 25 |
| senor20201101_06_r3 | 87 | 88 | 87 | 0 | 1 |
| **total** | 763 | 825 | 763 | 0 | 62 (+8 %) |

Every 1.7.0 event is kept with identical bounds. On peanut the 25 added events come entirely from the
7-sample minimum (at 0.0175 s, 9 samples, 2.0 gives the 202 old events exactly); the other 37 added
events were not checked one by one. The SWR consensus z-score differs only within 11–13 ms of the data
ends (interior max |Δz| 8.8e-4).

**Non-local events under both rules** (April probabilities, runs of P(non-local) > 0.8):

| epoch | 1.7.0 rule (last − first ≥ 0.015 s) | 2.0 rule (n ≥ `minimum_sample_count` = 7) | ratio |
|---|---|---|---|
| chimi20200225_02_r1 | 425 | 428 | 1.007 |
| wilbur20210329_12_r6 | 567 | 571 | 1.007 |
| j1620210707_06_r3 | 439 | 457 | 1.041 |
| peanut20201202_10_r5 | 362 | 367 | 1.014 |
| senor20201101_06_r3 | 572 | 582 | 1.017 |
| **total** | 2,365 | 2,405 | 1.017 |

An 8-sample rule would give 2,382. Each package's own `segment_boolean_series` count equals these.

### 4.2 Minimum sample counts (R1 and R11)

`minimum_sample_count(time, 0.015)` is **7 on every epoch**, not the 8 that the plan's text first
assumed: the timestamps are absolute Unix seconds, where a 2 ms step is 0.0019998550 or 0.0020000935 s,
and 0.015 / 0.0020000935 = 7.4996 rounds to 7 (task-1 report, "Finding").

- R1 decodes (2026-10-04, 218 epochs): `values: [7]` (`.claude/research/refresh_r1/gates_full.json`,
  `4_min_count`).
- R11 decodes (the canonical ones, 255 epochs): `min_sample_count_values: [7]` in
  `.claude/research/refresh_r11/gates/full_summary.json` (197 epochs) and `j16_summary.json` (58).
- The audit's `minimum_sample_counts` check on the 242 canonical epochs: section 1.

### 4.3 Both Kay version checks

The SWR runner writes, per epoch, two comparisons against the canonical Spyglass `SWR_event`
(`RippleTimesV1`) by strict positive-duration overlap (design §"Per epoch, the Kay version check"):
**version only** (Kay at `68e1784` on Spyglass's own inputs and stored parameters) and **version and
inputs** (the direct Kay inventory). No earlier report aggregates them over the canonical run
`056eee0dcb5a`, so R8 summed the per-epoch `kay_version_check.counts` of its manifest (read only;
script `.claude/research/refresh_r8/kay_checks.py`, output `kay_checks.txt`, per-epoch CSV
`kay_checks_per_epoch.csv`). Both checks ran in 255/255 epochs; `settings_mismatch` is false in all 255
(the R10 gate report says the same).

On the 242 analysed epochs, all events:

| comparison | canonical events | canonical matched | new events | new added | ratio new / canonical | per-epoch ratio median (IQR) | epochs outside [0.6, 1.3] |
|---|---|---|---|---|---|---|---|
| version only | 43,026 | 41,186 (95.7 %) | 44,783 | 3,629 | 1.041 | 1.034 (1.000–1.135) | 10 (8 low, 2 high) |
| version and inputs (R-T + mask) | 43,026 | 35,250 (81.9 %) | 41,221 | 6,207 | 0.958 | 1.017 (0.958–1.067) | 4 (1 low, 3 high) |

Speed-eligible events only give the same picture (version only 95.7 % matched, ratio 1.040; version and
inputs 82.3 %, 0.963). Per animal, all events (ratio new / canonical, share of canonical events matched):

| animal | version only | version and inputs |
|---|---|---|
| chimi | 0.883, 87.7 % | 0.959, 92.9 % |
| j16 | 1.200, 99.2 % | 0.822, **50.1 %** |
| peanut | 1.009, 98.8 % | 1.067, 97.9 % |
| senor | 1.009, 97.2 % | 1.051, 99.3 % |
| wilbur | 1.057, 99.0 % | 1.061, 99.0 % |

Reading: the package version alone keeps 95.7 % of the stored events and adds 4 %. The change in
inputs (rule R-T dropping tetrodes, and the mask) is what moves j16: half of j16's stored events have
no partner in the canonical inventory. This agrees with R13a §4.5 (R-T replaces j16 events) and with
R8a (R-T carries the j16 SWR changes). The R1-era design numbers
(`docs/design/2026-10-09-swr-source.md`, from `.claude/research/refresh_r1/kay_version_ratio.csv`, 216
epochs, before R-T and the mask) were 38,307 stored vs 38,879 direct, 92.8 % matched.

### 4.4 Probe marks removed per wilbur epoch

**Missing from every earlier report.** The design table (`docs/design/2026-10-03-full-pipeline-refresh-design.md:75`)
only says that wilbur20210331 loses probe sort groups 17–19 (32 channels each) under the tetrode-only
rule of `f172084`. R8 counted them (read-only Spyglass fetches; script
`.claude/research/refresh_r8/probe_marks_wilbur.py`, log `logs/r8_audit_20261012/probe_marks_wilbur.log`,
CSV `.claude/research/refresh_r8/probe_marks_wilbur.csv`). Count = raw marks of the non-tetrode sort
groups inside the epoch's decode span (the SWR run's `position_span_s`, which the audit's `swr_spans`
check holds equal to the decode span), before coincident-spike removal.

Result (exit 0): of the 45 analysed wilbur epochs, **only the 8 wilbur20210331 epochs have probe sort
groups**; every other wilbur day has 13 tetrode groups and nothing removed. On wilbur20210331 the
UnitMarks restriction holds 15 sort groups: 12 tetrodes and groups 17, 18 and 19 with 32 electrodes
each.

| epoch | probe marks removed | tetrode marks kept | probe share of all marks |
|---|---|---|---|
| wilbur20210331_02_r1 | 429,018 | 209,770 | 0.672 |
| wilbur20210331_04_r2 | 452,211 | 199,663 | 0.694 |
| wilbur20210331_06_r3 | 475,484 | 218,520 | 0.685 |
| wilbur20210331_08_r4 | 456,150 | 241,636 | 0.654 |
| wilbur20210331_10_r5 | 455,604 | 243,172 | 0.652 |
| wilbur20210331_12_r6 | 457,324 | 239,850 | 0.656 |
| wilbur20210331_14_r7 | 407,431 | 227,695 | 0.641 |
| wilbur20210331_16_r8 | 452,614 | 235,167 | 0.658 |
| **total (8 epochs)** | **3,585,836** | 1,815,473 | 0.664 |

Across all 45 wilbur epochs the tetrode marks in span total 14,857,460. Two thirds of
wilbur20210331's April marks therefore came from the probe. Context, not attribution: these epochs
had 821 table events in the golden baseline and have 740 now (77–120 → 66–122 per epoch); the
decode, SWR and rule changes also act here, and this report does not separate them.

### 4.5 Non-local events per epoch

Sources: the R11 gate tables (`.claude/research/refresh_r11/gates/{full,j16}_events_per_epoch.csv`,
column `events_new`: every non-local event the decode labels) and the canonical event table (kept
events after the analysed-time and whole-event immobility rules). R8 joined them for the 242 analysed
epochs (`.claude/research/refresh_r8/nonlocal_events_per_epoch.{py,csv,txt}`).

| animal | epochs | decode events (min / median / max per epoch) | table events (min / median / max) | table `is_swr` share |
|---|---|---|---|---|
| chimi | 46 | 24,600 (165 / 312 / 2,734) | 24,598 (165 / 312 / 2,734) | 0.383 |
| j16 | 58 | 14,746 (136 / 226 / 898) | 14,746 (136 / 226 / 898) | 0.535 |
| peanut | 44 | 15,668 (185 / 316 / 750) | 15,662 (185 / 316 / 749) | 0.346 |
| senor | 49 | 13,490 (177 / 277 / 485) | 13,489 (177 / 277 / 484) | 0.481 |
| wilbur | 45 | 7,180 (66 / 138 / 558) | 7,180 (66 / 138 / 558) | 0.489 |
| **all** | 242 | **75,684** (66 / 244 / 2,734) | **75,675** (66 / 244 / 2,734) | 0.432 |

The 9-event difference equals the 9 non-local events R13a reports the analysed-time rule dropping
(9 non-local, 5 SWR, 0 MUA); they were not matched one by one. The five epochs with the fewest events
are all wilbur20210331 (66–89); the most is chimi20200213_06_r3 (2,734). Against the R1 decodes, 154 of 160 comparable non-j16 epochs are within 5 % (median change 0);
the six outliers (−8 % to −57 %) are all senor epochs that the register now excludes for
`SUSTAINED_NOISE`, where R1 had trained through the S noise
(`.claude/research/refresh_r11/gates/r11_full_gates_report.md`, `full_summary.json`). j16 rose 2.8 % (training now
excludes R10 SWRs). Against the April decodes on the 207 shared epochs: 61,163 → 61,349 events, 86 % with
a one-to-one partner (R8a §4).

### 4.6 Held-out spike evidence per decoder state (Task 12)

*Rewritten 2026-10-16.* The R8 text reported only the 5-epoch pilot. F1 has now run Task 12 on the R11
decodes. Source: [`f1_evidence_report.md`](../../.claude/research/refresh_followup/f1_evidence_report.md)
§1 and §3 (main tree `d29cc3c`, CPU only). Pooled outputs: `.claude/research/refresh_followup/f1_outputs/b_pooled_240/`;
per-epoch tables: `Processed-Data/_refresh_20261016/state_evidence/per_epoch/`.

**Coverage: 240 of 242 epochs.**
- **j1620210709_10_r5 is still running** (788,696 spikes). It is 1 of j16's 58 epochs, and each animal's
  summary is the mean of its epochs, so it cannot change a verdict (F1 §5).
- **chimi20200221_08_r4 failed** the stored-likelihood gate (max abs 90.45 nats, max rel 1.21). F1 calls the
  cause unknown. D1 found it: the animal never entered patch 1, so the occupancy at bins 71–78 is a float32
  subnormal. The GPU decode keeps subnormals; JAX on the CPU flushes them to zero. The decode is correct; the
  CPU check is not ([`d1_decode_mismatch_report.md`](../../.claude/research/refresh_followup/d1_decode_mismatch_report.md)).
  Task 29 (`da34665`, `e224a09`) makes the checks leave out bins with 0 < occupancy < 1e-20. This epoch's
  rerun waits for the merge.
- Every other gate passes in 240/240: exact encoding refit; summed per-tetrode terms reproduce the stored
  likelihood to max rel ≤ 4.6e-5; A's marks equal B's.

**Results.** Primary encoding `runcv_` (time-held-out, running-only), spike-mark part, nats per window. Animal
mean with a t-interval on 4 df (F1 §3). The pilot column is Task 12's `pilot_fix1`, as F1 reports it; R8's
earlier text quoted `pilot_fix0d`.

| state, regime | evidence | estimate [95 % t] | p | animals, epochs > 0 | bout − matched [t]; p; animals | pilot estimate; p; animals |
|---|---|---|---|---|---|---|
| Local, immobile | animal vs mismatched | 2.76 [1.17, 4.36] | 0.009 | 5/5, 240/240 | 0.63 [0.24, 1.01]; 0.011; 5/5 | 2.16; 0.026; 5/5 |
| Local, running | animal vs mismatched | 15.2 [9.2, 21.3] | 0.002 | 5/5, 240/240 | 0.08 [−0.90, 1.06]; 0.83; 3/5 | 11.8; 0.014; 5/5 |
| No-Spike, immobile | silence vs local coding | 7.70 [5.70, 9.70] | 0.0004 | 5/5, 240/240 | 85.6 [70.2, 101]; 0.0001; 5/5 | 6.68; 0.002; 5/5 |
| No-Spike, running | silence vs local coding | 4.65 [3.80, 5.49] | 0.0001 | 5/5, 240/240 | 125 [98, 151]; 0.0002; 5/5 | 4.04; 0.0009; 5/5 |
| NL-Continuous, immobile | MAP_S vs animal | 6.10 [4.25, 7.94] | 0.0008 | 5/5, 237/240 | 7.82 [5.74, 9.90]; 0.0005; 5/5 | 5.03; 0.042; 5/5 |
| NL-Continuous, immobile | MAP_S vs mismatched MAP | 6.41 [2.94, 9.88] | 0.007 | 5/5, 240/240 | 5.24 [2.51, 7.98]; 0.006; 5/5 | 4.78; 0.003; 5/5 |
| NL-Continuous, running | MAP_S vs animal | −1.19 [−1.88, −0.50] | 0.009 | 0/5, 23/240 | 2.21 [1.60, 2.81]; 0.0005; 5/5 | −1.54; 0.12; 1/5 |
| NL-Fragmented, immobile | MAP_S vs animal | 5.22 [3.07, 7.37] | 0.003 | 5/5, 218/240 | 6.46 [4.30, 8.61]; 0.001; 5/5 | 6.09; 0.15; 4/5 |
| NL-Fragmented, immobile | MAP_S vs mismatched | 0.13 [−0.56, 0.81] | 0.64 | 3/5, 116/240 | −0.45 [−0.96, 0.06]; 0.07; 0/5 | 1.29; 0.16; 3/5 |

- Every pilot verdict holds at 240 epochs, and two firm up: Local immobile vs matched windows (p 0.08 → 0.011)
  and NL-Fragmented immobile vs animal (p 0.15 → 0.003).
- Not supported: running NL-Continuous against the animal's position (0/5; all its bouts are non-SWR), and
  immobile NL-Fragmented against a mismatched location (3/5).
- Coordinated silence, immobile: excess over local coding 0.449 [0.203, 0.694], p 0.007, 5/5 (pilot 0.420).
- Split by SWR (F1's supplement): immobile NL-Continuous per held-out spike, against the animal's position,
  0.60 [0.45, 0.74] with SWR and 0.20 [0.16, 0.25] without, 5/5 each.

**Caveats (F1 §6).** B's labels come from the all-tetrode decode, so only the remote location is truly held
out. The NL-Continuous immobile top calibration bin does not separate from 0 against the animal (0.75, lower
bound −0.08). Task 13's half-tetrode labels now exist; section 10.6 compares them.

### 4.7 Not run in the refresh

- **Task 13 (half-tetrode held-out labels).** The code is done and reviewed (`a06892a..3f50a90`); the
  20 GPU decodes were never run. One CPU end-to-end on peanut exists (half decodes found 153–182 events
  against 245; `.superpowers/sdd/PLAN/progress.md:307`), which is not evidence.
  **Run 2026-10-16:** 20 half decodes plus 2 S = all checks, all ok, on the 5 pilot epochs (section 10.6).
- **Task 12 full run on the R11 decodes** (4.6). **Run 2026-10-16** by F1 on 240 of 242 epochs (section 4.6).
- **The R13 additions** (plan "R13: tables, refits, PFC, figures, audit"): the sensitivity analysis
  without the epochs over 2 % masked (to be restated on the 242 epochs), the masked-time table with the
  events and bins the mask and the 0.3-s reach remove, and the robustness reruns of the headline
  results (without the mask; every artifact threshold 25 % lower and higher; 1 s masked after each
  step). No ledger entry or output exists for any of them.
  **Run in part 2026-10-16** (R13 interim, section 10.10): the >2 % sensitivity, the masked-time table, V0
  (no mask) and V3 (+1 s after steps) are done. V1 (thresholds ×0.75), V2 (×1.25) and X_R10nr (R10 labels,
  no rule) are paused by the lead. V1's and V2's detector reruns are done and kept; X_R10nr's root and inputs
  exist, its tables are not built.
- **Plan R8 "Docs" step** (a `docs/known_issues/` note on the re-decode; the
  `three_sessions_blocked_by_nwb_checksum.md` update; the spec's status line; the
  `replay_detection_comparison` PLAN; the memory notes): after the lead's review, by the plan.
  **Done 2026-10-16** on `full-pipeline-refresh-code` (`bd3aa2c`, `0134a63`), not merged; the coordinator
  updated the memory notes (ledger `progress.md`, 09:00 and 09:10 PDT).

## 5. Open items

Severity:
- **High:** a number or statement shipped or quoted now is wrong or unsupported on the refreshed data.
- **Medium:** a gap or defect to close before a specific result is cited, or before the next
  regeneration.
- **Low:** tooling, cosmetics, documentation hygiene, or pre-existing behaviour that changes no
  reported number.

Every row was checked against the cited evidence in this session unless it says otherwise.

| # | item | evidence | severity | proposed owner / action |
|---|---|---|---|---|
| 1 | Hand-maintained docs quote pre-refresh numbers; 8 conclusions change | section 6 (78 claims; 6.1 lists the 8) | **High** | lead: review section 6, then requote (plan R8 STOP) |
| 2 | `fit_provenance.git_hash()` reads the checkout's HEAD at every write, so a long run in a checkout whose HEAD moves stamps a commit it did not run | `src/continuum_swr_replay/analysis/replay/fit_provenance.py:85-96` (subprocess at call time), `:195` (called in each `write_fit_provenance`); ledger 2026-10-11 22:45 | Medium | code task: read HEAD once at process start and pass it down; until then, freeze the checkout during runs (the R6b/R7 runbooks did). **Resolved 2026-10-16** on `full-pipeline-refresh-code` (not merged): Task 27 item 1 (`b7dc1d6`, `e6e3b9e`) reads the commit once at import; Task 28 `929b1b9` does the same for the content runner's `_provenance` (`task-27-report.md`, `task-28-report.md`) |
| 3 | R09 / R05 / R07 z- and SD-references are taken over all cohort stops after the qvals join, including zero-bin stops that are never fit | `r6b_cpu_report.md` §4.2 and concern 1: ≤ 0.2 % in any SD; R09's own comment says "the SD of exactly the rows that get fit" | Low | design decision (lead); a code task if references should be the fitted rows |
| 4 | `switch_aligned`: a NaN `is_stay` drops the stop from the sequence, which creates 6–8 spurious "first-switch" episodes per source | `r6b_cpu_report.md` §4.4 (non_local 21 lost / 8 gained episodes). Built only by `scripts/content/compare_event_source_models.py:575-628` and stored in the X03–X05 bundles; grep over `scripts/` and `src/` finds **no figure that reads it** | Low | code task: an unknown stop breaks the run instead of being skipped. **Resolved 2026-10-16**: Task 27 fix round 1 (`28e00cc`) decides a first switch from the stop's own arriving trip. R6c rebuilt X03–X05: 2,319 first switches; episodes 1,961 / 1,721 / 1,693 (non_local / SWR / MUA), equal to Task 27's counts (`r6c_report.md` §4.3) |
| 5 | Per-bin and contrast-rate fit pickles save rows in a non-deterministic order | `r6b_cpu_report.md` §4.1 (R01–R03, R05, R06 bit-equal values, rows reordered) | Low | code task: sort by key before saving. **Resolved in code 2026-10-16** (not merged): Task 27 item 3 (`c0b69fd`, `a4ba6d2`) sorts by the full row key. The rate pickles on disk were not refit (R6c reran no rate run) |
| 6 | The R12b (Widloski) runner exits 1 at cluster teardown after every record is saved | `logs/memoryless_r12b_20261011.out` ends `TimeoutError` then `EXIT=1`; `.claude/research/refresh_r13/r12b_gate.json` 242/242 ok, inputs re-hashed equal | Low | code task: close the cluster with a longer timeout, or exit by the manifest |
| 7 | `generate_replay_content_figures.py` asks for an M7 model and for behaviour / mixed-effects inputs that do not exist (those figures are silently skipped) | `scripts/content/generate_replay_content_figures.py:127` (`key_models = ["M2", "M4", "M6", "M7"]`), `:89`, `:146`, `:161`; R7a report "Tracked-output decisions" | Low | code task: drop M7, decide whether those inputs belong in the pipeline |
| 8 | The event-gallery catalog globs the decode folder instead of reading the register | `scripts/figures/generate_event_gallery.py:167-173` (`list_epoch_pkls`); equal to the 242 today | Low | code task: `analysed_decode_files` |
| 9 | A stale "218" in the `SWR_INVENTORY_RUN_ID` docstring | `src/continuum_swr_replay/analysis/hippocampus/swr_source.py:176` ("the 218 analysed and the 37 pending"); same text in `docs/design/2026-10-09-swr-source.md:27` (6.4 M3) | Low | doc pass. **Resolved 2026-10-16** on `full-pipeline-refresh-code` (not merged): `bd3aa2c` fixes both texts (ledger 09:10 PDT) |
| 10 | The model-checking ladder's top-left legend covers a few points | R7 report, wave 2b checks (by eye) | Low | batch with the next figure fix |
| 11 | The gallery's excluded-span hatch has never been seen by eye | R7a "Gallery checks": 0 of 1,000 selected windows touch excluded time | Low | render one window that crosses excluded time and look |
| 12 | Theta power is raised 0.3–1 s beyond the mask (analysed time): SD 1.2–1.26 and p99 4.3–4.9 there, against SD ≈ 1 and p99 2.5–2.8 far from any mask | ledger 2026-10-11 R13c promotion note (coordinator's `theta_reach.py`, 3 epochs). No model reads `theta_z` (R8 grep: no hit in `model_specs.py`, `covariate_registry.py`, `scripts/{rate,content,pfc}`, `analysis/pfc`); Fig 2 / theta-QA panels and the event table's `theta_z` do | Medium | lead: decide whether theta displays need a wider reach |
| 13 | The theta SD gap after task 26a: median 0.68 (85 epochs with excluded time) vs 0.74 (157 without), p = 0.003 | `r13c_report.md` §5; the coordinator showed the normaliser itself is sound (rows > 10 s from a mask have SD 0.97–1.03) | Low | open question; test epoch composition if theta is used |
| 14 | 22 zero-span stops (exposure 0) keep their labels; 5 are at_well and stay in the well-anchored cohort | `r13c_report.md` concern 3 | Low | code task or accept (no exposure, no events) |
| 15 | No stamp separates pre- and post-26a tables | task-26a review minor 8; ruling (ledger) to enforce at promotion: both roots built at ≥ `bb19449`, PFC at ≥ `bb19449` | Low | accepted; revisit at the next schema change |
| 16 | R09 `uncertainty_rate` degeneracy: 218 of 711 rows not converged, 374 separated; `reward_history_incomplete` SEs ≤ 1e-6 | `r6b_cpu_report.md` §4.2, concern 3 | Medium | read R09 coefficients only from converged, non-separated rows; consider dropping the degenerate term |
| 17 | Choice block `fit_failed`: non_local 49/242, SWR 83/242, MUA 94/242, mostly `C_behavior` / `C_full` | R8 read of the X03–X05 bundles: non_local was 57/218 in September (same predictors). The golden SWR/MUA bundles are **June** files with a different 4-predictor set (4/218, 6/218), so they are not a like-for-like "before" | Medium | do not cite the SWR/MUA choice block without the fitted share; check identifiability of `C_behavior`/`C_full` |
| 18 | Per-bin GLM log lines "NaN, inf or invalid value detected in weights" | `r6b_cpu_report.md` §2 notes: R01 99, R02 78, R03 13, R11 7, the same counts as the previous run | Low | pre-existing; trace which fits they hit if a per-session value is cited |
| 19 | PFC: 432 non-converged model fits of 166,968 | ledger 2026-10-12 05:25 (PFC gate): identical to the pre-refresh 432 / 155,952, the same wilbur sessions | Low | pre-existing |
| 20 | The two PFC sigma-sensitivity bundles were not rerun | ledger 05:25; R8a §0 | Medium | rerun before citing the sigma sensitivity |
| 21 | Per-session immediate-reward β is widely spread | R8 read of `all_models.pkl` `M7r`: IQR 2.53 (SWR), 1.95 (NL), 2.68 (NL w/ SWR), 2.53 (NL no-SWR); 5–95 % −1.45 to +4.86 (SWR) | Low | quote only across-session medians or pooled values; check the high-side sessions for near-separation |
| 22 | `M3` (and the whole content ladder) fits 241 of 242 sessions: **wilbur20210403_08_r4** drops | R8: the session has 124 events, 110 at-well (= the cohort audit's `n_excluded_session` 110). In `logs/r6b_refits_20261011/C01.log` the `M5` and `M6` rungs (the `swr_x` rungs) raise `ValueError: need covariance of parameters` in statsmodels `ConditionalPoisson` for one session, and the ladder keeps only sessions with all 8 rungs ("Validated complete 8-rung ladder on 241 shared sessions"). The log does not name the session; the link is by the count (hypothesis, strongly supported). September had the same 217/218 pattern | Low | accept; or name the session in the log |
| 23 | Earlier parked: label-agreement keeps every bin (task 26b concern C10) | task-26b report; ruling "accept, R8 note"; no reported statistic reads the half-split state numbers | Low | note in the state-evidence docs |
| 24 | Earlier parked: `covariate_dictionary.md` numbers (84,069 / 107,400) | task-26a review minor 6 | Low | in section 6 (D1–D3) |
| 25 | Earlier parked: task 25 minors — `R1` never fit on a builder-made table (synthetic only); a zero-row table returns empty blocks silently; `fit_per_stop_rate_model` returns None on a missing column for other callers; the module docstring says R0–R2 | ledger, task 25 entry | Low | small code task |
| 26 | Earlier parked: the M3 specification changed since April (pipeline-flow panel F: 205 sessions / 6 coefficients then, 241 / 3 now) | R7a "pipeline_data_flow" | Low | caption / methods note |
| 27 | Earlier parked: final-review-2 M9 (`masks.py:60-61` and the artifact design note quote two populations as one), M10 (the supplementary data-flow figure reads a raw frame; its caption must not call the event an analysed one), M11 (PLAN Tasks 16–18 bodies still describe per-consumer masks) | `.superpowers/sdd/PLAN/final-review-2.md` M9–M11 | Low | doc pass |
| 28 | Earlier parked: provenance sidecars — 10 runs write none (provenance is in the research run records); `check_fit_provenance` has blind spots (root-level Stage 5B/6 and `content_estimator_comparison` need explicit paths; `_smoke` files are not skipped) | ledger R6 ruling; `r6b_cpu_report.md` §5 ran the explicit paths | Low | code task: sidecars for those scripts |
| 29 | Earlier parked: `pfc_marked_model_family/` (852 sidecars) is stale on the old tables | ledger R6 ruling (out of scope) | Low | do not cite; refit if the family is used |
| 30 | Earlier parked: the PFC smoke check fails near-silent units (0 usable null draws → NaN p) | ledger: senor20201101_04_r2 units 111/124, identical in September | Low | relax the smoke gate for near-silent units |
| 31 | Earlier parked: ContFrag `arm_fraction_posterior` divides by `n_timebins`, which includes undecoded bins, so its per-event sum can be < 1 | `src/continuum_swr_replay/analysis/contfrag/tables.py:672-675` (`arm_fraction` beside it uses `n_timebins − n_undecoded_bins`); only an optional column in `choice_prediction` / `switch_analysis`, not the default | Low | code task: same denominator as `arm_fraction` |
| 32 | Earlier parked: the main venv has drifted from `uv.lock` (numba 0.68 vs 0.62.1, llvmlite 0.50 vs 0.45.1, cuda-pathfinder, extra mutmut/textual/rich/libcst, jax-cuda12 plugin) | ledger; the tables and R11 ran in it; freeze `.claude/research/refresh_r13/main_venv_freeze_20261011.txt` | Medium | lead: lock the freeze or re-sync; a `uv sync` today would change the environment the results were made in. **Resolved 2026-10-16** on `full-pipeline-refresh-code` (not merged): the lead asked for a re-lock; `01b3051` pins what ran (numba 0.68.0, llvmlite 0.50.0, the CUDA plugin 0.8.0 and 14 NVIDIA CUDA 12.9 packages). The lock now matches the freeze except other-platform packages and hand-installed tools (`venv_drift.md`; ledger 07:40 PDT) |
| 33 | Figure 2 panel e text must say "per-session median Local-state decoding error" | task-26c review; text at `paper/figure_claims_plan.md:33-34` (6.4 F1) | Low | doc pass |
| 34 | The plan's R7 line is wrong: `summarize_per_bin_rate_models.py` only prints; `docs/per_bin_rate_results.md` is hand-transcribed | `.claude/docs/plans/full-pipeline-refresh/PLAN.md:1660`; the script writes no file (R7b) | Low | plan edit |
| 35 | Upstream: `plot_rescaled_ISI_autocorrelation`'s lag slice selects 0 points (LorenFrankLab/non_local_detector#64, filed); a local workaround is in task 26d | R7b report; ledger | Low | track upstream; drop the workaround after the fix |
| 36 | Task 12 (held-out spike evidence per decoder state) was never run on the R11 decodes; only the 5-epoch pilot on pre-R11 decodes exists | section 4.6; no output in `Processed-Data/_refresh_*/state_evidence` | Medium | run `scripts/qa/run_state_evidence.py` on the 242 epochs (CPU, ~2.6 h at 16 workers by the task-12 estimate). **Resolved 2026-10-16**: F1 ran it on 240 of 242 epochs (section 4.6). Left: j1620210709_10_r5 (running) and chimi20200221_08_r4 (after the merge; items 50 and 54) |
| 37 | Task 13 (half-tetrode held-out labels): GPU decodes never run | section 4.7 | Medium | run after R11 on free GPUs, per the ledger ruling. **Resolved 2026-10-16**: 20 half decodes plus 2 checks, all ok (section 10.6). Follow-up: item 52 |
| 38 | R13 additions not run: sensitivity without the > 2 % masked epochs (restated on 242), the masked-time table, and the robustness reruns (no mask; thresholds ±25 %; 1 s after each step) | section 4.7; plan "R13" | Medium | lead: schedule before the artifact rule is described as robust. **Partly resolved 2026-10-16**: the >2 % sensitivity, the masked-time table, V0 and V3 are done (R13 interim, section 10.10). The rest (V1, V2, X_R10nr) is item 49 |
| 39 | The spike-evidence ("reality") outputs are stale against the R11 decodes, and `paper/figures/reality/event_spike_evidence_per_animal.csv` still labels j16 "j"; claims row 19a cites its ΔLL ≈ 0.73 | ledger 2026-10-11 22:45 (task 26c note) | Medium | rerun `analyze_event_spike_evidence.py` before citing. **Resolved 2026-10-16**: F1 reran the engine on 242/242 epochs; outputs in `paper/figures/reality_r11/` (its per-animal CSV names j16). Row 19a's ≈ 0.73 was a one-session prototype; F1's replacement is in section 10.2 |
| 40 | `compare_pfc_cohorts.py` cannot compare two held-out PFC trees (it always reads the "before" tree as likelihood-ratio bundles) | section 2.4; `compare_pfc_cohorts.py:2627-2629` | Low | code task: read each tree by its sidecars' estimator |
| 41 | `compare_fits.py`'s default mode leaves the "before" reward-step cells empty when both sides use the current rung names | section 2.2 | Low | code task: pair by the current names when both sides have them |
| 42 | 33 at-well stops in chimi20200218_06_r3 disagree on reward between `qvals` and the stop table (old and new tables alike) | `r13a_report.md` concern 5 | Low | known-issues note (plan R8 Docs step) |
| 43 | Plan R8 "Docs" step pending: re-decode known-issues note, `three_sessions_blocked_by_nwb_checksum.md`, the spec's status line, the replay-detection PLAN, memory notes | plan R8 | Low | after the lead's review. **Resolved 2026-10-16** on `full-pipeline-refresh-code` (not merged): `bd3aa2c`, `0134a63`; memory notes updated by the coordinator (ledger 09:00, 09:10 PDT) |
| 44 | The audit's `--check-marks` path keeps every NWB analysis file it opens (about 22 per recording day) and does not raise the open-file limit; over 45 days it reaches about 1,000 open files against a soft limit of 1,024 | R8 run: 158 NWB handles open after 8 chimi days (`/proc/<pid>/fd`), soft limit 1,024 (`/proc/<pid>/limits`). R8 raised the running process's soft limit with `prlimit --nofile=65536` at 18:02; no code or data was touched. `scripts/qa/audit_refresh_consistency.py` calls neither `decoding_common.raise_open_file_limit` nor `release_nwb_files` | Low | code task: call both in the marks check |
| 45 | The fitter takes only a single-column separated term to its limit; a multi-column separated direction refuses the session. X03 non_local patch_switch keeps 157 of 189 sessions (PS3 refuses 33, and the common cohort drops them from PS0–PS2) | `r6c_report.md` §5.2, concern 1; ruling 2 (ledger 14:05 PDT) | Medium | code task (parked): a multi-column limit, or a per-rung cohort for that block; do not cite patch_switch without the n |
| 46 | The same BFGS stopping problem is still in `fit_conditional_logit`, `choice_prediction.fit_choice_model`, the `model_checking` CV folds and (unmeasured) the DM L-BFGS-B | `task-28-report.md` §3 and fix round 1 §7; ledger 09:40 PDT ("Parked") | Medium | code task: apply Task 28's fix; measure the DM gap first |
| 47 | X01 (`compare_content_estimators.py`) stores no optimizer diagnostics (`newton_decrement`, `max_abs_grad`, `n_newton_steps`, `unidentified_terms`) | `r6c_report.md` §3.1, concern 4; all 3,872 of its conditional fits are `converged` (decrement < 1e-8) | Low | small code task |
| 48 | C01's directory swap (`_publish_staged_directory`) deletes the K04–K07 DM contrast files in `replay_content/models/` | `r6c_report.md` §2.3, concern 7; R6c restored the 44 files byte for byte | Medium | before any C01 rerun: restore or rerun K04–K07; code task: keep files the runner does not own |
| 49 | R13's V1 (thresholds ×0.75), V2 (×1.25) and X_R10nr (R10 labels, no rule) are paused. V1's and V2's detector reruns are done and kept; no V1, V2 or X_R10nr table or fit is built | `r13_robustness_report.md` §9 (resume commands; about 4–8 h); the lead paused R13 at 13:48 PDT | Medium | lead: resume or drop before the artifact rule is called robust to its thresholds |
| 50 | Post-merge reruns: chimi20200221_08_r4's state evidence and reality engine with Task 29's checks; every figure that reads the content fits (stale against R6c) | `task-29-report.md` (the rerun waits for the merge); `r6c_report.md` concern 8 | Medium | after the merges: rerun that epoch (about 10–20 min of CPU, D1) and the content figures; the lead confirms keeping the epoch (D1 option (a)) |
| 51 | Phase 3, the detector comparison, was out of the refresh's scope; its inputs are ready, its adapters, runner and report are not built | ledger 08:30 PDT (answer to the user) | Low | the next project |
| 52 | Half-tetrode No-Spike labels fail the log-ratio test (0/5), and the sign tracks the selecting half's spike share (Spearman 0.80); "held out" and "less power" are confounded | `t13_half_tetrode_report.md` §3, §5, concern 1 | Medium | spike-balanced half splits (or more partitions) before any held-out No-Spike claim |
| 53 | The non_local_detector issue on device-dependent subnormal occupancy is drafted, not filed | D1 recommendation 4; draft `scratchpad/nld_issue_subnormal.md` (session scratchpad); ledger 12:20 PDT: no GitHub credentials on breeze | Low | file it once push access works |
| 54 | F1's final full pass of Task 12 is still to run (see below) | `f1_evidence_report.md` §5 | Low | watch it; stop it if it recomputes all 242 epochs |

**Count by severity, 2026-10-16: 1 High, 10 Medium, 33 Low still open (44 of 54 items).** Closed: items 2,
4, 5, 9, 32, 36, 37, 38 (in part, the rest is item 49), 39 and 43. On 2026-10-12 it was 1 High, 10 Medium,
33 Low (44 items).

**The F1 final pass (item 54).** Source: `f1_evidence_report.md` §5; ledger 17:50 PDT.
- After j1620210709_10_r5 ends (about 03:00–04:00 on 2026-10-17), `logs/f1_evidence_20261016/run_state_xargs.sh`
  runs a final full pass from the main tree (`--n-workers 4`, no `--epochs`).
- It reuses the 241 written epochs only if the commit (`d29cc3c`), settings and decode files are unchanged and
  the tree has no tracked changes. It recomputes chimi20200221_08_r4, which fails the same check (about
  10 min), records that, writes the pooled outputs over the 241 ok epochs, and exits 1.
- **If the main tree has a new commit by then** (the `memoryless-sequence-detection` branch moves), the commit
  check fails for every epoch and the pass recomputes all 242 (10 h or more). In that case stop it and use
  `.claude/research/refresh_followup/f1_outputs/b_pooled_240/`. Do not stop the j16 epoch itself.
- A second risk: the dask shutdown `TimeoutError` seen in 5 epochs could recur at the end, before the pooled
  summary is written. The same fallback applies.

## 6. Hand-maintained doc claims that need new numbers

**No doc was edited.** Search: `.claude/research/refresh_r8/grep_doc_claims.py` over every tracked
`docs/**/*.md` (dated `docs/audits/` records excluded), `paper/claims_table.md`,
`paper/figure_claims_plan.md`, the three `*_caption.md` files and `CLAUDE.md`, for the pre-refresh
values of every quantity that moved (518 raw hits in `doc_claims_hits.tsv`), then read and grouped by
hand. One row below is one claim: one statement (one or a few adjacent lines) that quotes moved
numbers. "Cause" is R8a's attribution where R8a covers the quantity; "not attributed" means R8a's
table does not include it. "Not re-derived" means none of R8's sources gives the new value; it must be
read from the named artifact before requoting.

Sources of the new values: R8a §0–§6; the summarizer log
`logs/r7_figures_20261011/r7b_summarize_per_bin_rate_models.log` (rerun identically as
`logs/r8_audit_20261012/summarize_per_bin_rate_models_rerun.log`); `compare_fits_rate.md` and
`compare_fits_content.md` (section 2); the X01 three-estimator outputs
`Processed-Data/replay_content/content_estimator_comparison/content_estimator_{comparison_summary,bootstrap}.csv`
(rewritten 2026-10-12 00:05 at `110c9ae`); direct reads of
`models_per_bin_rate/all_models{,_allstops}.pkl` (frozen cohort, `n_cv`, `attrs["arrival_clock_cohort"]`).

CLAUDE.md quotes no moved number. Its only refresh item is the wording proposal in section 7.
`uncertainty_content_caption.md` and `content_estimators_caption.md` are written by their scripts
(rewritten in R7/R6b); `memoryless_comparison_caption.md` is simulation-only. None of the three is
listed.

### 6.1 Claims whose conclusion changes, not only the number (read these first)

| # | claim and where | was | now | cause |
|---|---|---|---|---|
| V1 | "SWR-coupled non-local events are the rarest type"; all four baseline contrasts significant. `docs/per_bin_rate_results.md:61-88`; `docs/replay_rate_content_swr_synthesis.md:110-114`, `:636-637` | NL w/ SWR 0.184 < NL no-SWR 0.225 events/s; within-session 0.83×, p = 5.1e-09 | NL w/ SWR 0.212 ≈ NL no-SWR 0.216; within-session 0.98×, **p = 0.30**; three of four contrasts significant | not attributed (hypothesis: R-T moves events into the SWR class, R8a §2–§3) |
| V2 | "SWR-coupled non-local events are the most tightly bursty stream (dead time ≤ 45 ms, peak 2.43×)". `per_bin_rate_results.md:220-235`; `docs/rate_models.md:241-242`; synthesis `:186-188`, `:650`; `paper/claims_table.md:53` (19c) | peaks 2.59 / 2.02 / **2.43** / 1.57 (SWR / NL / NL w/ SWR / NL no-SWR), NL w/ SWR refractory ≤ 45 ms | 2.67 @105 ms / 1.95 @120 / **1.98** @120 / 1.56 @105; all refractory ≤ 60 ms. Detected SWR is now the burstiest; NL w/ SWR ≈ NL all | not attributed |
| V3 | "On the well-anchored cohort the non-SWR time-course peak is inside [1, 3] s, a located maximum (2.68 s)". `per_bin_rate_results.md:99-106`; `rate_models.md:396`; synthesis `:116-117` | 2.63 / 2.82 / 2.31 / 2.68 s | 2.62 / 2.96 / 2.48 / **3.00 s = the right edge** (peak fraction of support 1.0), as it was on all stops | not attributed |
| V4 | "After full control the non-SWR immediate component is a real response of its own … both streams respond … a difference of degree"; Family A "earns its parameters in every population". `per_bin_rate_results.md:157-163`, `:197-206`, `:299-305`; `rate_models.md:221-222`, `:259`, `:514-516`, `:527-529`, `:648-654`; synthesis `:135-152`, `:160-172`, `:254`; `claims_table.md:50`, `:53` (19c), `:130-131` | M7r bump-0 RR 2.00 (66 %, p 3.1e-06); kernel lag 1 = 1.09; marginal `M1−M0` −0.75 (child better 58.3 %); Family A `R1−R0` −0.44; paired NL w/ SWR > no-SWR 69.4 % of 216, p 1.1e-11 | RR **1.42** (58 %, p 2.6e-03, 5/5); kernel lag 1 = **1.01**; marginal **+0.09** (child better 49.2 %); Family A **+0.88**; the paired dissociation is stronger: **76.7 % of 240, 5/5, p 1.1e-19** | R-T + mask −0.36 and added epochs −0.26 for the RR (R8a); the rest not attributed |
| V5 | CS5 (`swr_x_reward_belief_mean_candidate_arm_preoutcome`) direct effect "estimator-dependent: count and DM negative, fraction positive; only the DM's interval excludes zero; DM 0/5". `claims_table.md:57` (19f); `docs/design/content_estimator_comparison.md` §8 (`:53-64`, `:127`, `:229-230`, `:259`, `:370-371`); synthesis `:78-85`, `:452-457`, `:582`, `:605` | count −0.398 (1/5) [−0.904, +0.199]; fraction +0.237 (3/5) [−1.322, +0.539]; DM −0.316 (0/5) [−1.136, −0.057]; n 216 | count −0.173 (2/5) [−1.125, +0.176]; fraction −0.012 (2/5) [−0.806, +0.202]; DM **+0.046 (3/5)** [−0.730, +0.103]; n 238. **No interval excludes zero; the sign pattern reversed.** Total `CS5nt` (fraction) −0.651 → −0.259 (0/5, q < 1e-4) | within noise (Kay 2.0 −0.19, added epochs −0.19; R8a) |
| V6 | "SWR content is less current-locked" (`CS1` −0.373, 0/5 animals positive). `claims_table.md:51` (19a); synthesis `:283`, `:622`, `:657`; `content_estimator_comparison.md:54`, `:122`, `:248`, `:259` | fraction −0.373 (0/5); count −0.461; DM −0.297 | fraction **−0.069 (1/5)**, count −0.093, DM −0.036; all three animal-bootstrap intervals include zero ([−0.632, +0.145] fraction); the session sign-rank q is still < 1e-4 (`compare_fits_content.md`). The total `CS1nt` stays clearly negative: −0.599 → −0.411 (0/5). Split `C2_base`: SWR +0.797 → **+0.884**, non-SWR +0.725 → **+0.530** | the decode, +0.21, bootstrap excludes 0 (R8a) |
| V7 | Content occupancy #14: "non-SWR content is well-dominated (72 % at wells = ~42 % remote + ~30 % own well; 27 % paths); SWR ~76 % paths". `claims_table.md:43` | remote well 0.42, own well 0.30, remote path 0.27; SWR paths 0.76; 218 sessions | remote well 0.45, own well **0.13**, remote path **0.41** (wells 0.58); SWR remote path 0.73, wells 0.21 + 0.05; 242 sessions | own well −0.169 = script definition change of 2026-08-12 (−0.162) + decode −0.028 + cohort/label (R8a §5.2): **the 0.30 was already stale before the refresh** |
| V8 | "Arm value's main effect is positive too (`CS5_armvalue` main term +0.264, 5/5, bootstrap [+0.031, +0.818])". `claims_table.md:52` (19b), `:56` (19g); synthesis `:417-421` | fraction +0.264 (217, 5/5) [+0.031, +0.818]; count +0.228 (5/5); DM +0.168 (4/5); leaf `Mv` +0.2512 (215, 4/5) | fraction +0.256 (242, **3/5**) [**−0.268**, +0.608]; count +0.701 (4/5); DM +0.300 (4/5); `Mv` +0.2409 (237, 3/5). The fraction interval now includes zero | not attributed (R8a does not split this term) |

### 6.1a Follow-up changes to these conclusions (2026-10-16)

Sources: F2 ([`f2_robustness_report.md`](../../.claude/research/refresh_followup/f2_robustness_report.md)),
R6c ([`r6c_report.md`](../../.claude/research/refresh_r13/r6c_report.md)) and F1
([`f1_evidence_report.md`](../../.claude/research/refresh_followup/f1_evidence_report.md)). The claim rows of
6.2–6.4 are not requoted here; they wait for the lead's review.

| # | what changes | evidence | effect on the wording |
|---|---|---|---|
| F2-a | **The SWR reward rate ratio does not rest on j16.** Without j16 it rises, 6.39 → 6.89 (animal-balanced 6.80). j16 sits at 5.35; it raised the pooled value from 3.44 by ceasing to be a low outlier (1.16 before R-T), not by being high. | F2 §1a, §1c; leave-one-animal-out range 5.36–6.98; 81–84 % positive in every set | A cohort-robust statement is "about 5–7×, every animal positive" (F2). §0's and §3's attribution of the *change* (R-T in j16, plus the cohort) still holds |
| F2-b | **About a fifth of 6.39 rests on the 31 added early peanut epochs:** 5.29 without them (5.27 animal-balanced); 5.29 without them and j16 (4.29 animal-balanced). The added peanut epochs' SWR RR is 17.54 against 3.49 for the 13 original ones (Mann-Whitney p 0.025). | F2 §1a, §1c | Quote a range or the animal-balanced value, and name the cohort. F2's day-trend explanation is a hypothesis |
| F2-c | **V4: the NL-no-SWR immediate response is fragile.** Without wilbur: RR 1.21, 55 % positive, p 0.083 (G0 nonSWR session p 0.053). In the 31 added peanut epochs: 0.92. The paired dissociation holds in every leave-out: 72.4–78.5 % of sessions, every remaining animal positive, signed-rank p ≤ 4.9e-9. | F2 §1a, §1b, concern 1 | "Both streams respond" is not robust; the dissociation is. The joint-model contrast S1_rew is 4/5 (chimi negative), against 5/5 for the paired M7r test; cite 4/5 if the joint model is cited |
| F2-d | **V6: CS1 is negative because SWR content shifts onto the trunk arms.** The pooled term shares the arm fixed effects between classes, so it measures the current arm's *share*. SWR events put about 0.35 of their mass on trunk arms, non-SWR 0.18. Giving SWR its own arm baseline moves CS1 −0.065 → +0.208 (3/5) and CS1nt −0.405 → +0.058 (3/5). | F2 §2e (sandbox refit; 215 sessions where every model fits) | Name the share estimand: SWR content devotes a smaller share to the current arm (CS1nt −0.41, 0/5; descriptive log-share −0.32, 0/5); among leaf arms only, −0.10 (1/5). F2 gives recommended wording (Item 2 conclusions) |
| F2-e | **V6: CS1 is not null at the session level.** The animal median −0.069 is the least negative summary. On the 216 shared sessions: session median −0.191, fixed- and random-effects pools −0.226 and −0.228; session sign-rank p 2e-7. | F2 §2d, concern 5 | Do not call the direct effect null |
| F2-f | **V6: the "+0.88 vs +0.53" split compared unpaired animal medians** (senor's for SWR, chimi's for non-SWR). Paired within session: +0.092 (3/5) at the pooled mean length; −0.11 (2/5) to +0.24 (4/5) at other reference lengths. | F2 §2b; R6c's split values +0.8823 / +0.5252 change nothing here | "SWR content favours the current arm more" is not supported. Report paired within-session differences |
| V9 | **CS5b no longer estimates the arm-value term under a previous-goal control.** All strata: −0.4234 (155, 1/5) → −0.0608 (198, 2/5, q 0.16); many added sessions have the control unidentified, so the estimand is mixed. Rewarded stratum: the published row (−0.0526, n 237) is the CS5 model, because both controls are unidentified in every session. | R6c §5.2, §5.4, concern 2 | Do not quote either CS5b row as "controlling for the previous goal" until the lead decides how to treat it. Claim C9 (`paper/claims_table.md:57`) and synthesis `:568-583` (S13) quote CS5b |
| F1 | **No conclusion change.** Non-SWR non-local events under the R11 decodes carry spike evidence for their decoded location, less per spike than SWR events: engine, per spike, nonSWR-close 0.51, nonSWR-remote 0.71, SWR-remote 1.14, each above baseline in 5/5 animals; held out, 0.20 vs 0.60 per held-out spike, 5/5. | F1 Verdict, §2, §3 | Row 19a's "≈ 0.73" (C4) came from a one-session prototype. F1's replacement: nonSWR-close dll_animal 0.79, per spike 0.51, 5/5 animals and 242/242 epochs above baseline (F1 concern 5) |

R6c changes no other 6.1 conclusion. Its values for V5 (CS5 count / fraction / DM −0.1732 / −0.0142 /
+0.0461), V6 (fraction CS1 −0.0709, 1/5) and V8 (CS5 main term +0.2560, 3/5) keep the same signs and animal
counts (R6c §5.4). F3 found no conclusion change from the optimizer gap (`f3_optimizer_report.md` §5).

### 6.2 The two items the brief names

| file:line | old text | new value | cause |
|---|---|---|---|
| `docs/rate_models.md:424-433` | arrival-window retention of event bins, well-anchored cohort "94.3 % (detected SWR), 92.9 % (non-local with SWR), 90.6 % (non-local, all) and **88.9 %** (non-local without SWR)"; all stops "91.2 / 90.2 / 85.0 / **81.4 %**" | well-anchored **94.7 / 94.1 / 90.8 / 88.0 %** (`all_models.pkl` `attrs["arrival_clock_cohort"]`; the figure note reads "95 % of SWR vs 88 % of non-SWR"); all stops 91.3 / 91.3 / 85.7 / 81.3 % (`all_models_allstops.pkl`) | not attributed |
| `paper/claims_table.md:53` (row 19c) | kernel `rr_lag1` session-pooled 1.228 / 1.214 / 1.408 / 1.086 and animal-balanced 1.300 / 1.200 / 1.405 / 1.089; paired 69.4 % of 216, p 1.1e-11 (all stops 79.2 %, 7.8e-22); `rr_immediate` animal-balanced 4.91 / 2.96 / 6.86 / 2.28 (all stops 4.15 / 2.58 / 7.24 / 1.23); across-session 3.44 / 3.44 / 7.18 / 2.00 (no-SWR p 3.1e-06); "about 3.6× smaller"; marginal ΔAIC −3.23 / −8.18 / −3.96 / −0.75; conditional +1.58 / +1.19 / +2.41 / +2.28; Family A −1.30 / −5.97 / −2.10 / −0.07 (animal-balanced) and −2.43 / −5.37 / −1.79 / −0.44 (session-pooled, 217); 76.0 % / 52.1 % child-better; all-stops Family A −1.41 / −5.12 / −2.43 / +0.30 (218); self-history 2.59 / 2.02 / 2.43 (≤45 ms) / 1.57; retention 0.943 / 0.929 / 0.906 / 0.889 (all stops 0.912 / 0.902 / 0.850 / 0.814); "218 sessions" | session median kernel 1.38 / 1.19 / 1.51 / 1.01; paired **76.7 % of 240, 5/5, p 1.1e-19**; across-session RR 6.39 / 3.27 / 9.91 / 1.42 (no-SWR p 2.6e-03), ratio ≈ 7.0×; marginal −6.83 / −7.84 / −5.37 / +0.09; conditional +0.94 / +1.01 / +1.62 / +2.44; Family A session-pooled −4.26 / −4.20 / −3.14 / +0.88 (241); self-history 2.67 / 1.95 / 1.98 / 1.56 (all ≤ 60 ms); retention 0.947 / 0.941 / 0.908 / 0.880 (all stops 0.913 / 0.913 / 0.857 / 0.813); 242 sessions. **Not re-derived:** the animal-balanced kernel and `rr_immediate`, the Family A animal-balanced values and child-better shares, and every all-stops value except retention (read `all_models.pkl` / the audited per-bin aggregate / `models_per_stop_rate` and the `_allstops` bundles) | RR rows: R8a §0 (R-T + mask, cohort); the rest not attributed |

### 6.3 `docs/per_bin_rate_results.md`: the new numbers, transcribed (not applied)

From the summarizer log (S), `compare_fits_rate.md` (C), or a direct read of `all_models.pkl` (P).

| lines | quantity | old | new | source |
|---|---|---|---|---|
| 3, 7 | fit date; corpus | 2026-09-12; 218 sessions | 2026-10-11/12 (R6b, `bb19449` tables); 242 sessions | S |
| 15-16 | well-anchored stops; frozen per-bin cohort | 84,069 of 107,400; 83,549 stops over 6,777,953 bins | 93,560 of 120,320; 92,942 stops over 7,535,050 bins | R13c; P |
| 23-25 | in-transit sessions clearing the gates | 50 of 218 | not re-derived (`per_bin_rate_cohort_audit_intransit.csv`) | – |
| 28-29 | `M7r` sessions fit (SWR / NL / NL w/ SWR / NL no-SWR) | 218 / 217 / 216 / 218 | 242 / 242 / 242 / 240 | S |
| 34 | stops whose reward features are recomputed at the outcome boundary | 13.4 % | 16,066 of 120,320 shifted (13.35 %); 16,353 cross (13.59 %) | C |
| 61-68 | baseline rate, median (IQR), NL all / SWR / NL no-SWR / NL w/ SWR | 0.407 (0.319–0.534) / 0.289 (0.223–0.346) / 0.225 (0.156–0.286) / 0.184 (0.137–0.249) | 0.431 (0.325–0.562) / 0.267 (0.229–0.307) / 0.216 (0.148–0.300) / 0.212 (0.175–0.255) | S |
| 70, 74-79 | ordering; pairwise within-session ratios, p | NL w/ SWR / no-SWR 0.83× (5.1e-09); SWR / NL 0.63× (1.3e-24); NL w/ SWR / SWR 0.75× (9.6e-37); NL no-SWR / SWR 0.81× (2.4e-09) | 0.98× (**3.0e-01**); 0.63× (6.3e-39); 0.80× (3.0e-41); 0.77× (8.3e-11). Ordering NL all > SWR > NL no-SWR ≈ NL w/ SWR (V1) | S |
| 81-86 | additive split; coupling shares | 0.184 + 0.225 ≈ 0.407; 0.75 SWR-coupled NL events per SWR; 0.45 of NL events SWR-coupled | 0.212 + 0.216 ≈ 0.431; 0.80; 0.49 | S; P |
| 99-106 | time-course peak, [1, 3] s, over unflagged fits | 2.63 / 2.82 / 2.31 / 2.68 s over 168–217 fits | 2.62 / 2.96 / 2.48 / 3.00 s (edge) over 192–236 fits (V3) | C |
| 108-113 | duration caveat (3.6×, 1.858 vs 0.519 s; 2.3×; 20 % at 2 s, 12 % at 4 s) | as quoted | not re-derived (the doc does not state its query) | – |
| 127-139 | arrival vs stop clock; separated share of `M5a`; retention | 2.31–2.82 vs 2.80–3.80 s, KS 0.06–0.11; 42.9–63.6 %; 88.9–94.3 % (all stops 81.4–91.2 %) | 2.48–3.00 vs 2.94–4.00 s, KS 0.075–0.090; 58.2–63.1 %; 88.0–94.7 % (all stops 81.3–91.3 %) | C; P |
| 150-155 | §2 table: raw RR (`M1_rew`); controlled RR (`M7r`); %pos; p; n | 20.0 / 32.4 / 99.5 / 18.8; 3.44 / 3.44 / 7.18 / 2.00; 76 / 81 / 85 / 66 %; 3.5e-19 / 3.0e-21 / 1.3e-26 / 3.1e-06; 218 / 217 / 216 / 218 | 58.13 / 30.30 / 125.39 / 11.87; 6.39 / 3.27 / 9.91 / 1.42; 83 / 79 / 87 / 58 %; 7.9e-28 / 1.3e-22 / 8.0e-33 / 2.6e-03; 242 / 242 / 242 / 240 (animals 5/5 throughout) | S |
| 158-163 | paired dissociation; non-SWR response | 69.4 % of 216, p 1.1e-11; RR 2.00, p 3.1e-06, "about 3.6×" | 76.7 % of 240, 5/5, p 1.1e-19; RR 1.42, p 2.6e-03, ≈ 7.0× (V4) | S |
| 166-167 | `rr_immediate` animal-balanced | 4.91 / 2.96 / 6.86 / 2.28 | not re-derived (the audited per-bin aggregate) | – |
| 174-179 | bump 1 RR (%pos); bump 2 RR (%pos) | 0.05 (21) / 0.04 (17) / 0.01 (15) / 0.08 (31); 5.39 (72) / 5.19 (80) / 13.84 (80) / 3.29 (67) | 0.02 (14) / 0.05 (18) / 0.01 (13) / 0.17 (35); 7.69 (80) / 5.23 (80) / 15.08 (83) / 2.45 (61); 5/5 each | S |
| 194-199 | §3 kernel, lags 1–10 | SWR 1.23 0.99 0.93 0.90 0.91 0.92 0.94 0.97 1.00 1.03; NL 1.21 0.99 0.92 0.90 0.91 0.92 0.94 0.98 1.00 1.04; w/ SWR 1.41 1.05 0.93 0.89 0.90 0.91 0.94 0.98 1.02 1.06; no-SWR 1.09 0.95 0.92 0.91 0.92 0.93 0.95 0.97 0.98 1.00 | SWR 1.38 1.04 0.93 0.90 0.90 0.92 0.94 0.97 1.00 1.04; NL 1.19 0.99 0.93 0.91 0.91 0.93 0.95 0.98 1.01 1.04; w/ SWR 1.51 1.07 0.94 0.89 0.89 0.91 0.93 0.98 1.02 1.07; no-SWR 1.01 0.95 0.94 0.93 0.94 0.96 0.97 0.98 1.00 1.01 | S |
| 201-206 | kernel summary | lag 1 dissociates 1.41 vs 1.09; suppression from lag 3 0.89–0.97 | 1.51 vs 1.01; lags 3–8 0.89–0.98; the at-well non-SWR lag 1 is now ≈ 1, like the all-stops 0.99 the text contrasts it with (V4) | S |
| 220-235 | §4 self-history (controlled); raw `M3r` peaks | 2.59 @135 / 2.02 @120 / 2.43 @150 (≤45 ms) / 1.57 @105; raw 3.45 / 3.02 / 4.04 / 2.18 | 2.67 @105 / 1.95 @120 / 1.98 @120 / 1.56 @105, all ≤ 60 ms; raw (max over lags 1–10) 4.18 / 2.83 / 3.49 / 2.15 (V2) | S |
| 272-280 | §6 ladder: median AIC; CV bits/event; `n_cv`; KS-in-band | M0 2172/0.000/218/0.44; M1 2159/0.032/214/0.38; M5 2083/0.266/208/0.91; M3r 2165/0.044/75/0.33; M6r 2078/0.262/73/0.91; M7r 2077/0.244/74/0.93; M9r 2083/0.200/43/0.91 | M0 2227/0.000/242/0.45; M1 2221/0.024/238/0.37; M5 2138/0.238/226/0.93; M3r 2207/0.049/67/0.37; M6r 2128/0.220/62/0.96; M7r 2128/0.216/61/0.96; M9r 2133/0.177/40/0.95 | S; P (`n_cv`) |
| 282-298 | ladder bullets | KS 0.44 → 0.91; 0.38 and 0.33; 0.266 over 208 of 218; 0.200–0.262 on 43–74; M3r 0.044 over 75; M7r 2077 and M6r 2078 below M5 2083 | 0.45 → 0.93; 0.37 and 0.37; 0.238 over 226 of 242; 0.177–0.220 on 40–62; 0.049 over 67; M6r and M7r 2128 below M5 2138 | S; P |
| 299-312 | reward step marginal (child-better share, no-SWR); conditional | −3.23 / −8.18 / −3.96 / −0.75 (58.3 %); +1.58 / +1.19 / +2.41 / +2.28 | −6.83 / −7.84 / −5.37 / **+0.09** (49.2 %); +0.94 / +1.01 / +1.62 / +2.44 | C; P |
| 313-316 | `M9r` vs `M7r` | 2083 vs 2077; 0.200 vs 0.244 | 2133 vs 2128; 0.177 vs 0.216 | S |

(The summarizer's CV column is the median over the sessions whose CV is finite, i.e. over `n_cv`,
the same convention as the doc's.)

### 6.4 Other claims, per file

| # | file:line | old | new | cause |
|---|---|---|---|---|
| R1 | `docs/rate_models.md:23-24`, `:80`, `:173-174`, `:282`, `:410` | 84,069 of 107,400 stops; 515 dropped for no arrival clock; 83,549 stops and 6,777,953 bins | 93,560 of 120,320; 575 stops (72,059 bins) dropped for no arrival clock; 92,942 stops and 7,535,050 bins | cohort (R13c) |
| R2 | `docs/rate_models.md:31` | in-transit: 50 of 218 sessions clear the gates | not re-derived | – |
| R3 | `docs/rate_models.md:95-96`, `:108` | Family A animal-balanced −5.97 / −2.10 / −1.30 / −0.07; all-stops +0.30 over 218 | session-pooled −4.26 / −4.20 / −3.14 / +0.88 (241); animal-balanced and all-stops not re-derived | not attributed |
| R4 | `docs/rate_models.md:185-193` | corpus 218; `M7r` 218 / 217 / 216 / 218; `M5_time` 218 | 242; 242 / 242 / 242 / 240; 242 | cohort |
| R5 | `docs/rate_models.md:221-222`, `:527-529` | paired 69.4 % of 216, p 1.1e-11; kernel 1.41 vs 1.09 | 76.7 % of 240, p 1.1e-19; 1.51 vs 1.01 (V4) | R8a for RR; rest not attributed |
| R6 | `docs/rate_models.md:241-242` | self-history 2.59 / 2.43 (≤45 ms) / 2.02 / 1.57 | V2 values | not attributed |
| R7 | `docs/rate_models.md:259`, `:262` | `rr_lag1` 1.228 / 1.214 / 1.408 / 1.086 (and animal-balanced); "218 sessions" | session median 1.38 / 1.19 / 1.51 / 1.01; animal-balanced not re-derived; 242 | not attributed |
| R8 | `docs/rate_models.md:293`, `:452` | 107,400 stops; 14,494 cross (13.5 %), 14,343 shifted (13.4 %) | 120,320; 16,353 cross (13.59 %), 16,066 shifted (13.35 %) | cohort |
| R9 | `docs/rate_models.md:320`, `:403` | `M0a` fits all 218; "not on 218 sessions" | 242 | cohort |
| R10 | `docs/rate_models.md:396` | non-SWR peak moves inside, to 2.68 s | 3.00 s, the edge (V3) | not attributed |
| R11 | `docs/rate_models.md:436-438` | arrival rungs' CV NaN 91.5 % (`M5a`) / 93.5 % (`M7ra`) of non-local fits; 99.5 % / 100 % on `non_local_swr` | 92.4 % / 95.3 %; 100 % / 100 % | not attributed |
| R12 | `docs/rate_models.md:514-516` | Family A −5.97 / −5.37 (NL) … 217 sessions; 76.0 % vs 52.1 % child-better | session-pooled −4.20 (NL), 241 sessions; shares not re-derived | not attributed |
| R13 | `docs/rate_models.md:594`, `:607` | `S1_rew` flags 44 of 218 fits; "218 sessions" | n = 242; flag counts not re-derived (`swr_contrast_rate_term_diagnostics.csv`) | cohort |
| R14 | `docs/rate_models.md:628-640` | `M5_time` 0.266 on 208 of 218; AIC 2083 vs 2077 / 2078; 0.200–0.262 | 0.238 on 226 of 242; 2138 vs 2128 / 2128; 0.177–0.220 | not attributed |
| R15 | `docs/rate_models.md:648-654` | marginal −8.18 / −3.96 / −3.23 / −0.75; conditional +1.58 / +1.19 / +2.41 / +2.28 | −7.84 / −5.37 / −6.83 / **+0.09**; +0.94 / +1.01 / +1.62 / +2.44 (V4) | not attributed |
| C1 | `paper/claims_table.md:25` (#2) | 39,500 / 65,913 immobile non-local events non-SWR (~60 %); 57–68 % per animal | 42,974 / 75,675 (56.8 %); classes 21 / 22 / 57 %; per animal not re-derived | decode, labels, cohort (R8a §5; counts reproduce R7) |
| C2 | `paper/claims_table.md:43` (#14) | quadrants (V7); "all 218 sessions" | V7 values; 242 | V7 |
| C3 | `paper/claims_table.md:50` (framing row) | kernel 1.41 vs 1.09; 69.4 % of 216, p 1.1e-11; "both streams respond" | 1.51 vs 1.01; 76.7 % of 240, p 1.1e-19 (V4) | V4 |
| C4 | `paper/claims_table.md:51` (#19a) | `C2_base` `is_current_arm` +0.708 (218); three-estimator +0.626 / +0.681 / +0.737 (n 218); split non-SWR +0.725 (215), SWR +0.797 (199); `CS1` −0.373 (0/5), `CS1nt` −0.5985; DM `log_phi_is_swr` +1.799 [+1.447, +2.205], "≈ 6×"; close non-SWR ΔLL ≈ 0.73 (spike-evidence engine) | +0.7278 (242, 5/5); +0.668 / +0.728 / +0.728 (n 241); non-SWR +0.5300 (236, 5/5), SWR +0.8839 (221, 4/5); `CS1` −0.0685 (1/5), `CS1nt` −0.4111 (0/5) (V6); `log_phi_is_swr` +1.726 [+1.455, +2.125], 5/5, ≈ 5.6×; ΔLL not re-derived (the reality engine has not been rerun on the R11 decodes, section 5) | decode (C1, CS1, log_phi; R8a) |
| C5 | `paper/claims_table.md:52` (#19b) | `C3_armreward` non-SWR +0.0515 (215, q 0.336), SWR −0.2139 (197, 1/5, q 0.0058), pooled −0.0492 (218); `CS5` main +0.264; `C4_armvalue` non-SWR +0.0465; `CS4` −1.1975 (218) | non-SWR +0.0299 (238, 3/5, q 0.42), SWR −0.1617 (222, 0/5, q 0.0011), pooled −0.0830 (241, 2/5, q 0.26); `CS5` main +0.256 (V8); non-SWR +0.0376 (234); `CS4` −1.0032 (241) | CS4 at R-T, borderline (R8a); others not attributed |
| C6 | `paper/claims_table.md:53` (#19c) | see 6.2 | see 6.2 | – |
| C7 | `paper/claims_table.md:55` (#19e) | `is_stay` missing for 0 of 23,331 dropped stops and all 14,845 `mixed`; 67,606 strictly at-well stops; "not the 84,069"; smallest FDR p 0.156 / 0.749 | cohort now at_well 77,499 + mixed 16,061 = 93,560 (R6b Stage 6 sidecars); the other counts and p-values not re-derived (`stage6_is_stay_*.pkl`, `stop_replay_summary.pkl`) | cohort |
| C8 | `paper/claims_table.md:56` (#19g) | `Mu` +0.3509 (215, 5/5); `CS5` main +0.264 (5/5); `Mv` +0.2512 (215, 4/5); ΔAIC `M4→Mu` +1.80 (103/215), `M4→Mv` −2.34 (20/215), `Mu→Mvu` −1.61 (11/215); `C13` SWR +0.3680 (186), non-SWR +0.2949 (210); `CS6` +0.0039, `CS6nt` +0.1807; DM CS6 −0.027 / +0.146 | `Mu` +0.3081 (237, 5/5); `CS5` main +0.256 (3/5, V8); `Mv` +0.2409 (237, 3/5); +1.39 (110/237, 5/5), −2.42 (22/237), −1.55 (12/237); `C13` SWR +0.3398 (217), non-SWR +0.1701 (233); `CS6` −0.0585 (2/5), `CS6nt` +0.1764 (5/5); DM CS6 not re-derived (X01 does not fit CS6) | CS6 within noise (R8a); others not attributed |
| C9 | `paper/claims_table.md:57` (#19f) | `CS4` −1.1975 / `CS4nt` −1.3346 (218); DM −0.840 [−1.002, −0.682]; count −1.449; `CS5` three-way (V5); `CS5nt` −0.6506 (216), DM total −0.638; rewarded stratum +0.1303 / −0.3514; ΔAIC vs `C2_base_leaf` +2.524 (fraction) / +2.252 (DM); `CS5b` +0.1855 (134 sessions) | −1.0032 / −1.1653 (241, 0/5); DM −0.782 [−0.907, −0.443]; count −1.383; V5; `CS5nt` −0.2594 (239, 0/5), DM total not re-derived; rewarded −0.0622 (225) / −0.3116 (228); ΔAIC +2.205 / +1.817 (X01 summary, same harness); `CS5b` −0.4234 (155, 1/5, q 0.21); factorial (−0.980 / −0.398 / −0.114) not re-derived | CS4 at R-T (R8a); CS5 within noise |
| C10 | `paper/claims_table.md:59` (#20) | 71 sessions; 10,907 / 10,898 units; engagement 24.7 %; SWR raw-p 28.5 %; at-well bins 92.3 % (19,732,623 of 21,384,252), events 91.9 % (15,143 of 16,477) | 75 sessions; 11,604 units; engagement 23.4 %; SWR raw-p 30.8 % (R8a §6); bin/event fractions not re-derived (PFC sidecars) | sessions; SWR raw-p rise is j16 (hypothesis R-T) |
| C11 | `paper/claims_table.md:130-133` (summary) | kernel 1.41 vs 1.09; 69.4 % of 216; `CS4` −1.1975 (218) | 1.51 vs 1.01; 76.7 % of 240; −1.0032 (241) | as above |
| S1 | `docs/replay_rate_content_swr_synthesis.md:10-25`, `:44`, `:517` | refit-status box: 84,069 of 107,400; 50 of 218; "218-session corpus" | 93,560 of 120,320; in-transit not re-derived; 242 | cohort |
| S2 | synthesis `:78-85` | three-estimator CS5, CS1 (−0.461 / −0.373 / −0.297), CS4 (−1.449 / −1.198 / −0.840), `is_current_arm` (+0.626 / +0.681 / +0.737) | CS5 V5; CS1 −0.093 / −0.069 / −0.036; CS4 −1.383 / −1.003 / −0.782; +0.668 / +0.728 / +0.728 | R8a |
| S3 | synthesis `:108-123` | Family B overview: 218 sessions, 83,549 / 6,777,953; baseline rates; "rarest type"; peaks; KS 0.44 → 0.91; `M5` 0.266 over 208 of 218; `M7r` AIC 2077, KS 0.93, CV on 74 sessions | per 6.3 rows (242; 92,942 / 7,535,050; V1; V3; 0.45 → 0.93; 0.238 over 226; 2128, 0.96, 61) | mixed |
| S4 | synthesis `:135-152` | kernel / RR table (1.23 / 1.21 / 1.41 / 1.09; 3.44 / 3.44 / 7.18 / 2.00 with %pos, p); animal-balanced 4.91 …; paired 69.4 % of 216; RR 2.00 | per 6.3 (V4) | R8a for RR |
| S5 | synthesis `:160-172` | marginal / conditional ΔAIC; Family A (animal-balanced and session-pooled, 217) | per 6.3 and R3 | not attributed |
| S6 | synthesis `:184-188`, `:193` | self-history; `M9r` vs `M7r` 2083 vs 2077, 0.200 vs 0.244 | V2; 2133 vs 2128, 0.177 vs 0.216 | not attributed |
| S7 | synthesis `:202-206` | arrival-clock peaks, KS; retention 0.943 / 0.929 / 0.906 / 0.889 (all stops 0.912 / 0.902 / 0.850 / 0.814) | per 6.3; 0.947 / 0.941 / 0.908 / 0.880 (0.913 / 0.913 / 0.857 / 0.813) | not attributed |
| S8 | synthesis `:250-256` | SWR-power model (reward +0.141/SD, 5/5, p 5.5e-17; RPE 5/5; exploration lowers it); kernel 1.23 / 1.41 / 1.09 | power model refit in R6 (run R10) but not re-read here (`swr_power_model.pkl`); kernel 1.38 / 1.51 / 1.01 | not attributed |
| S9 | synthesis `:278-292` | content table: `C2_base` split; `CS1` / `CS1nt`; `CS2` −0.6457 (217); `C3` split; `CS4` / `CS4nt`; `CP0` +1.1116 (208); `CPS1` / `CPS1nt` −0.0708 / −0.2615 | +0.7278 / +0.5300 / +0.8839; −0.0685 / −0.4111; −0.4251 (239, 1/5); C5 row; −1.0032 / −1.1653; +1.0515 (230); +0.0144 (239, 3/5, q 0.0023) / −0.2079 (242, 1/5) | R8a for CS1, CS2, CS4 |
| S10 | synthesis `:354-363` | covariate table (`C1`, `C4`, `C6`, `C7`, `C8`, `C9`, `C10`, `C11`, `C12` × all / SWR / non-SWR) | all rows in `compare_fits_content.md` "SWR-content contrast (audited aggregate)", e.g. `C1` +0.8732 / +0.8999 / +0.8939 → +0.7813 / +0.8403 / **+0.6391**; `C10` non-SWR +0.0209 → +0.1071; `C11` non-SWR −0.0615 → −0.0116 | C1 at the decode (R8a); others not attributed |
| S11 | synthesis `:376-379` | `C4` split null: +0.1275 (217), +0.0465 (209), −0.2912 (191) | +0.0856 (242), +0.0376 (234), −0.1263 (218) | not attributed |
| S12 | synthesis `:417-431` | `Mv`, `CS5` main, `C13` split, `CS6` / `CS6nt` fraction and DM | as C8 (V8) | as C8 |
| S13 | synthesis `:452-458`, `:478-490`, `:568-583` | `CS5` three-way table; rewarded stratum; factorial; `CS5b` 133 / 134 sessions, +0.1855; total −0.6506 / −0.638 | as C9 (V5) | as C9 |
| S14 | synthesis `:600-625`, `:636-677`, `:735-736`, `:765` | summary bullets: 69.4 %; `CS4` −1.20; `CS2` −0.65; `CS5` +0.24; `CS5nt` −0.65; `CS6nt` +0.18; kernel 1.41; `CS1` −0.37; `CS1nt` −0.60; baseline rates; KS; self-history; `CS6` 0.0039 / 0.1807; `CS5b` 133 | as the rows above | as above |
| E1 | `docs/design/content_estimator_comparison.md:5`, `:120-136`, `:301` | "218-session cohort"; the §8 table (count / fraction / DM for CS1, CS4, CS5, `is_current_arm`); 65,913 events | X01 rerun values (S2, V5, V6); 241–242 sessions; 75,675 events | R8a |
| E2 | `docs/design/content_estimator_comparison.md:53-64`, `:229-230`, `:248`, `:259`, `:370-371`, `:384` | CS5 fraction +0.237 / DM −0.316 / total −0.638; CS1 −0.373; CS4 −1.198 / −1.449 / −0.840; `CS5b` 133 sessions | V5, V6, S2; `CS5b` DM not re-derived | R8a |
| E3 | `docs/design/content_estimator_comparison.md:136`, `:277`, `:317`, `:340` | `log_phi_is_swr` 1.817 / +1.8, "≈ 6×" | 1.724 (C1 rung; 1.726 on `C2_base`), ≈ 5.6× | decode −0.21 (R8a) |
| D1 | `docs/covariate_dictionary.md:458`, `:759` | 65,913 events | 75,675 | cohort, decode |
| D2 | `docs/covariate_dictionary.md:471-476`, `:518` | 84,069 well-anchored (69,224 at_well + 14,845 mixed); 13.5 % (14,494 of 107,400) cross the boundary | 93,560 (77,499 + 16,061); 13.59 % (16,353 of 120,320) | cohort |
| D3 | `docs/covariate_dictionary.md:752` | `+1.8`, ≈ e^1.8 ≈ 6× | 1.72, ≈ 5.6× | decode |
| T1 | `docs/design/cohort_and_timing_rules.md:33`, `:51-57` | 107,400 stops → 84,069 (69,224 + 14,845); 65,913 events → 59,699 at-well (90.6 %); Family A 84,064; Family B 83,549 / 6,777,953 | 120,320 → 93,560 (77,499 + 16,061); 75,675 → 69,011 at-well events (91.2 %, C01 cohort audit); Family B 92,942 / 7,535,050; Family A not re-derived (`models_per_stop_rate/all_models_cohort_audit.csv`) | cohort |
| T2 | `docs/design/cohort_and_timing_rules.md:240`, `:327`, `:339` | 71-session batch, 10,907 units; the `senor20201101_14_r7` example (40 of 162, 24.7 %) | 75 sessions, 11,604 units; that epoch is now excluded (`SUSTAINED_NOISE`), so the example needs replacing | cohort |
| M1 | `docs/README.md:77`, `:122`, `:170` | "one 218-session cohort"; "all 218 analysed epochs are in every table"; "50 of 218 sessions" | 242 (in-transit count not re-derived) | cohort |
| M2 | `docs/known_issues/three_sessions_blocked_by_nwb_checksum.md:3-4`, `:16`, `:55-58` | "the cohort is 218, not 215" and the 218 counts | the cohort is 242 (plan R8 "Docs" step asks for this update) | cohort |
| M3 | `docs/design/2026-10-09-swr-source.md:27` | "all 255 epochs ok (the 218 analysed and the 37 pending)" | true when written; the register now holds 242 analysed and 13 `SUSTAINED_NOISE` of those 255 (same stale text as `swr_source.py:176`) | register |
| F1 | `paper/figure_claims_plan.md:33-34` | "e: median Local-state decoding error across sessions" | "e: per-session median Local-state decoding error" (task 26c parked minor; median now 2.99 cm over 242 sessions) | wording |
| F2 | `paper/figure_claims_plan.md:66-70` | close subset ~28 % of non-SWR (~23 cm), remote ~72 % (~1.8 m) | not re-derived; the R11 decode lost mostly short own-arm non-SWR events (R8a §4), so these shares have likely moved (hypothesis) | decode (hypothesis) |
| F3 | `paper/figure_claims_plan.md:116` | Fig 3c duration-controlled GEE ×4.91, 5/5 | not re-derived | – |
| F4 | `paper/figure_claims_plan.md:185` | "the ~60 %" non-SWR share | 57 % (42,974 / 75,675) | as C1 |

### 6.5 Dated measurements left out of the count

These lines record a measurement with its date or on a named older table, in a design record or known
issue. They are true as records. Re-measure only if one is cited for the current data:
`docs/known_issues/current_arm_trunk_fallback.md:27-113` (2026-08-16, 218 sessions / 65,913 events);
`docs/known_issues/contains_path_length_confound.md:25`, `:72` (65,913 events; Stage 5B 218 rows);
`docs/known_issues/degenerate_mua_events.md:12` (218 ContFrag sessions);
`docs/known_issues/swr_mua_count_conventions.md:12` (218 sessions; CLAUDE.md already says the Δ82/Δ250
magnitudes must be re-derived); `docs/design/2026-10-03-full-pipeline-refresh-design.md:20`, `:58`,
`:518`, `:633`; `docs/design/2026-10-09-artifact-detection.md:217`, `:340`, `:354`;
`docs/design/2026-10-09-swr-tetrode-selection.md:37-242` (C1c on 218 epochs);
`docs/design/model_history.md:113-130`, `:197`, `:256`, `:407`;
`docs/design/content_model_design.md:66`, `:134`. The grep's remaining hits are false positives
(other quantities that share a digit string).

### 6.6 Count

**78 claims need new numbers:** the 2 named items (6.2), 22 rows of `docs/per_bin_rate_results.md`
(6.3) and 54 rows elsewhere (6.4; row C6 is 6.2 and is not counted twice). **8 of these 78 change a
stated conclusion** (6.1 lists them; they are the same lines, read for their conclusion, and are not
counted again). In 26 of the 78 at least one new value is "not re-derived" and must be read from the
named artifact first. F1 is a wording fix, not a number.

**Status changes, 2026-10-16** (the rows of 6.2–6.4 are unchanged):
- **Fixed on `full-pipeline-refresh-code`, not merged: 2 claims.** M2 (the three-sessions note now states
  the 242 cohort, keeping its history) and M3 (the swr-source note's "218 analysed and 37 pending"), both by
  `bd3aa2c` (ledger 09:10 PDT). M1 is fixed in one of its three lines only: `0134a63` rewrote the
  `docs/README.md` index line; "one 218-session cohort" (`:77`) and "50 of 218 sessions" (now `:172`) remain.
  **76 claims still need new numbers.**
- **Conclusion changes: 9, not 8.** CS5b in C9 (and S13) is the ninth (6.1a, V9). F2 corrects V4 and V6
  further (6.1a); they were already counted.
- **"Not re-derived": 24, not 26.** C4's ΔLL is now derived (F1: nonSWR-close dll_animal 0.79, per spike
  0.51). C8's DM CS6 is now derived (R13 interim §4, canonical bundles, R8a's accessor: DM CS6 −0.070, DM
  CS6nt +0.122). Both rows still need their other values re-read: R6c moved C4's and C8's content values
  (section 3).

## 7. For the lead: proposed CLAUDE.md wording

Source: `.superpowers/sdd/PLAN/task-26a-report.md`, "Proposed `CLAUDE.md` wording ("One analysed-time
rule")". Not applied. CLAUDE.md's current sentence (section "Canonical per-event classification table"):

> Inputs use the mask itself, not excluded time: the decoders' missing bins, the SWR detectors'
> normalization, and the ripple-power covariate (`SWR_zscore`, the Kay consensus over the tetrodes rule
> R-T keeps).

Proposed replacement:

> Inputs use the mask itself, not excluded time: the decoders' missing bins, the SWR detectors'
> normalization, the ripple-power covariate (`SWR_zscore`, the Kay consensus over the tetrodes rule R-T
> keeps), and a decode frame's two epoch normalizers. `with_swr_source` re-z-scores
> `theta_power_zscore` over the bins outside the mask (NaN inside it) and takes `run_speed_baseline`
> (the 90th percentile of head speed) over the bins outside the mask.

Proposed addition after "so every table inherits it.":

> The read also adds `in_excluded_time` (True in excluded time) and drops every decode-time event
> count. A per-bin summary leaves the `in_excluded_time` rows out. A stop's location, segment and
> distance summaries (`occupancy_state`, `occupancy_well_id`, `modal_observed_track_segment_id`,
> `nearest_well_distance_cm_*`) read only its rows outside excluded time. A stop wholly inside it is
> `unknown`, so it is in no well-anchored cohort. The bout-start context columns still read the
> stop's first row.

## 8. Cleanup list (nothing deleted)

What can be removed once the lead accepts this report. Sizes by `du -sh` (no symlink is followed),
measured 2026-10-12 (`.claude/research/refresh_r8/cleanup_sizes.{sh,txt}`). The plan keeps `$B` and the
golden baseline; this list does not include them, nor any `_pre_*_backup_*` folder.

**Staging folders (the brief's list).**

| path | size | files | note |
|---|---|---|---|
| `Processed-Data/replay_content/_staging_r13c_20261011/` | 992 MB | 294 | promoted R13c build; its tables equal the canonical ones |
| … of which `shard_{chimi,j16,peanut,senor,wilbur}/` | 103 + 71 + 65 + 58 + 37 MB | 6 each | intermediate shards, never recorded or promoted |
| … of which `_session_cache/` | 6 MB | 253 | sha256-equal to the canonical cache |
| `Processed-Data/contfrag_replay_content/_staging_r13c_20261011/` | 71 MB | 9 | promoted |
| `Processed-Data/contfrag_replay_content/_staging_r13b_20261011/` | 71 MB | 8 | superseded by R13c, never promoted |

**Other refresh staging folders found** (not in the brief's list; the lead decides):

| path | size | files | note |
|---|---|---|---|
| `Processed-Data/replay_content/_staging_r13_tables_20261011/` | 992 MB | 294 | R13a build, promoted then superseded by R13c |
| `Processed-Data/contfrag_replay_content/_staging_r13_20261011/` | 75 MB | 17 | R13a ContFrag build, superseded |
| `Processed-Data/_staging_redecode_20261010/` | 35 GB | 104 | **the 13 S epochs' R11 decodes** (left out of canonical by the 2026-10-10 ruling); keep if they may be wanted |
| `Processed-Data/_staging_redecode_20261003/` | 18 GB | 40 | R1-era staging (pilot decodes) |
| `Processed-Data/_staging_redecode_20261004/` | 419 KB | 0 | empty folder tree |
| `Processed-Data/replay_content/_staging_{stoponly_full_20260904, stop_regen_full_20260904, patch_fraction_full_20260905, patch_fraction_smoke_20260905, stop_clocks_full_20260906}/` | 5 + 284 + 284 + 17 + 285 MB | 1–6 each | September, before the refresh |

**Sandbox and run checkouts.**

| path | size | note |
|---|---|---|
| `/cumulus/edeno/continuum-swr-replay-run/Processed-Data/` | 2.6 GB (501 files, 4,173 file symlinks) | R8a's sandbox, a real directory now (it was a symlink to the canonical tree) |
| `/cumulus/edeno/continuum-swr-replay-run/Processed-Data_L4smoke/` | 321 MB (296 files, 4,173 symlinks) | R8a's L4 smoke |
| `/cumulus/edeno/continuum-swr-replay-run/` (whole checkout, incl. the two above and its `.venv`) | 8.2 GB | detached at `bb19449`; remove with `git worktree remove` from the main tree |
| `/cumulus/edeno/continuum-swr-replay-run2/` | 5.3 GB (its `Processed-Data` is a symlink to the main tree's, not counted) | detached at `bb19449`, clean; same removal |
| `/cumulus/edeno/r11a_scratch/` | 1.7 GB | R11a/R12 scratch (smoke, Triton spike) |

**Stale tmux sessions** (all panes dead unless noted; the server keeps dead panes):

- zephyr, refresh: `r7_*` (14: `r7_content_occ`, `r7_content_occ_smoke`, `r7_fig02`, `r7_fig03`,
  `r7_fig04`, `r7_fig04_d29`, `r7_gallery_catalog`, `r7_gallery_first`, `r7_gallery_render`,
  `r7_pipeline_flow_default`, `r7_pipeline_flow_ev275`, `r7_supp_dv`, `r7_uncert_content`, `r7_wave1`),
  `r13b_build`, `r13b_smoke`, `r13c_*` (11), `swr_r10`, `artifacts_full2`, `refresh_pbsens`,
  `ablation*` (25), `memoryless_pilot`, `memoryless_refcheck`. No `r8a_*` or `r6b_*` session remains
  (both were closed by their agents).
- zephyr, R8's own: `r8_audit` (alive until the audit ends), `r8_probe` (dead).
- breeze: `r12b_full` is already gone. `pfc_refresh2` remains with a dead pane. `nld_em_profile`
  (alive) is the lead's and is not on this list.
- Not listed: the `nld*` sessions of 2026-10-08 and the lead's own sessions; ownership unclear, the
  lead decides.

## 9. Files written by R8

- `Processed-Data/_refresh_20261012/`: `audit.json`, `compare_fits_rate.md`, `compare_fits_content.md`.
- `logs/r8_audit_20261012/`: `audit.log`, `compare_fits_rate.log`, `compare_fits_content.log`,
  `compare_pfc_cohorts.log`, `compare_pfc_cohorts_cohortmode.log`, `probe_marks_wilbur.log`,
  `summarize_per_bin_rate_models_rerun.log` (a read-only rerun of the summarizer; identical to the R7b
  log).
- `.claude/research/refresh_r8/`: `kay_checks.{py,txt}`, `kay_checks_per_epoch.csv`,
  `probe_marks_wilbur.{py,csv}`, `nonlocal_events_per_epoch.{py,csv,txt}`, `grep_doc_claims.py`,
  `doc_claims_hits.tsv`, `cleanup_sizes.{sh,txt}`.
- This report, in the dev worktree only.
- The 2026-10-16 update wrote only this report (section 10). It ran no analysis; every number in it is
  copied from the cited follow-up reports.

Nothing else was written. No canonical table, fit, decode, backup, PFC output or figure was touched; no
tracked file in the main tree or in `-run2` changed.

## 10. Follow-ups, 2026-10-16

One subsection per follow-up. Each gives what was run, the result, what it changes in this report, and what is
still open. The source reports are untracked files in the main tree (`/cumulus/edeno/continuum-swr-replay`);
the links resolve there. The ledger is `.superpowers/sdd/PLAN/progress.md` (entries of 2026-10-16, PDT); it
holds the coordinator's rulings. Code changes are on `full-pipeline-refresh-code` (head `e224a09`) and are
neither pushed nor merged.

### 10.1 R8a attribution

Source: [`r8a_report.md`](../../.claude/research/refresh_attribution/r8a_report.md) (2026-10-12).

- **Run:** Stage 1 descriptives and the Stage 2 sandbox ladder on 205 shared epochs. No new run today.
- **Result:** already in section 3.
- **Changes here:** none to its numbers. Two of its open items are now answered elsewhere:
  - R-T against mask normalisation (its §11 item 2): R13 finds that R-T, in j16 alone, accounts for all of
    the SWR reward-RR rise (section 10.10);
  - the j16 reading: F2 shows the SWR reward RR does not rest on j16 (section 10.3).
- **Still open:** which decoder change moves CS1, C1 and `log_phi_is_swr` (its §11 item 3), and the PFC
  changes (not refit, its §11 item 4).

### 10.2 F1: spike and state evidence on 242 epochs

Source: [`f1_evidence_report.md`](../../.claude/research/refresh_followup/f1_evidence_report.md).

- **Run:** (A) the spike-evidence engine `analyze_event_spike_evidence.py` on 242/242 epochs, with a fresh
  marks cache; outputs `paper/figures/reality_r11/`. (B) Task 12's `run_state_evidence.py` on the R11
  decodes; 240 ok, 1 failed, 1 running. CPU only, main tree `d29cc3c`.
- **Result, A** (event medians, per spike, in nats): SWR-remote 1.14, nonSWR-remote 0.71, nonSWR-close 0.51.
  Every class is above its epoch's matched baseline in 5/5 animals and 242/242 epochs. SWR-remote exceeds
  nonSWR-remote in 5/5 animals and 240/242 epochs (+0.40). Uncapped, on 241 epochs: NL with SWR 1.05, NL
  without SWR 0.65 per spike. A is not held out (the same spikes choose and score the MAP), so its "above
  baseline" is close to built in.
- **Result, B:** section 4.6. Immobile NL-Continuous against the animal's position: 6.10 [4.25, 7.94],
  p 0.0008, 5/5, 237/240 epochs.
- **June vs R11:** the P30-B transitions do not explain the rise in evidence. SWR-remote's rise is the decode.
  For nonSWR-close, about half is the event-definition code and half the decode (F1 §4).
- **Changes here:** §4.6 rewritten; items 36 and 39 closed; claims row 19a's ≈ 0.73 replaced (6.1a, 6.6).
- **Still open:** j1620210709_10_r5 (running); chimi20200221_08_r4 after the merge (item 50); the final
  pass (item 54). Untested hypotheses: why non-SWR events are weaker per spike (fewer cells, or a weaker
  sequence), and which decode input raised R11 above June.
- **Disagreement, resolved:** F1 calls the chimi20200221_08_r4 failure "not known". D1, written later the same
  day, explains it (section 10.7).

### 10.3 F2: robustness and leave-one-out checks

Source: [`f2_robustness_report.md`](../../.claude/research/refresh_followup/f2_robustness_report.md).

- **Run:** read-only leave-outs of the per-bin M7r immediate reward RR and of the paired dissociation, with
  the summarizer's own functions (reproduces 6.39 / 3.27 / 9.91 / 1.42 and 76.7 % of 240 exactly). One
  sandbox content refit (run checkout, `bb19449`) for the current-arm question.
- **Result, rate:** SWR RR 6.39; without j16 6.89; without the 31 added peanut epochs 5.29; without both
  5.29 (4.29 animal-balanced); leave-one-animal-out 5.36–6.98. The dissociation holds in every set (72.4–78.5 %,
  p ≤ 4.9e-9). NL-no-SWR without wilbur: 1.21, p 0.083. S1_rew is 4/5 (chimi negative).
- **Result, content:** the brief's hypothesis (per-class length reference) is refuted: length is z-scored
  once, over the whole at-well cohort. The cause is the shared arm fixed effects; SWR content moves onto the
  trunk arms (0.35 vs 0.18 of its mass).
- **Changes here:** §0; section 6.1a rows F2-a to F2-f (V4, V6).
- **Still open:** whether the peanut day trend is real (a hypothesis; 1–5 sessions per day). F2's BFGS
  observation led to F3.

### 10.4 F3: did the content fits stop short of their optimum?

Source: [`f3_optimizer_report.md`](../../.claude/research/refresh_followup/f3_optimizer_report.md).

- **Run:** Newton refits of all 2,631 shipped headline content fits (C2_base in three populations, CS1,
  CS1nt, CS4, CS4nt, CS5, CS5nt, CS6, CS6nt), read-only. The shipped coefficients reproduce exactly.
- **Result:** every fit was flagged converged, yet the log-likelihood gap to the optimum is median 1e-4, p90
  0.074, p99 1.38, max 4.81; above 0.1 in 243 fits and above 1 in 57. The gap is in a nuisance term, the third
  recency-spline column (tiny SD). Headline terms move by median 0.001 SE, max 0.35 SE. No sign changes and no
  animal count falls; C2_base SWR rises 4/5 → 5/5 on a knife edge (wilbur −0.008 → +0.009).
- **Cause (confirmed):** BFGS stops on gtol 1e-5 applied to −LL/nobs, and `ConditionalPoisson.fit` silently
  ignores a `gtol` argument.
- **Changes here:** led to Task 28 and R6c; the note under section 3's table; item 46.
- **Still open:** the same gap in other fitters (item 46). ΔAIC across a step that adds the spline may carry
  the gap (a hypothesis, not measured).
- **Disagreement:** the coordinator's own read of `f3_per_fit.csv` (ledger 07:35 PDT) gives median 5e-5 and
  p99 1.35; F3's report gives 1e-4 and 1.38. Both agree on p90 0.074 and max 4.81.

### 10.5 Venv drift and the uv re-lock

Sources: [`venv_drift.md`](../../.claude/research/refresh_followup/venv_drift.md); commit `01b3051`; ledger
07:40 PDT.

- **Run:** `uv sync --frozen --dry-run --all-extras` in the main tree (no change made).
- **Result:** the refresh ran with numba 0.68.0 / llvmlite 0.50.0 (lock 0.62.1 / 0.45.1). The lock resolved no
  CUDA plugin or NVIDIA package, so it could not rebuild the GPU environment of the R11 decodes and the PFC
  fits. Root cause: `[tool.uv] override-dependencies` "jax>=0.7.2" replaced every jax requirement, the extras'
  `jax[cuda12]` included.
- **Re-lock (the lead's request):** `01b3051` names `jax-cuda12-plugin[with-cuda]==0.8.0` directly and pins
  what ran. The lock now matches the freeze except other-platform packages and hand-installed tools.
- **Changes here:** item 32 closed.
- **Still open:** the merge. numba's effect on the position-linearization Viterbi is judged low risk and was
  not tested.

### 10.6 Task 13: half-tetrode decodes

Source: [`t13_half_tetrode_report.md`](../../.claude/research/refresh_followup/t13_half_tetrode_report.md).
Outputs: `Processed-Data/_refresh_20261016/half_tetrode/`.

- **Run:** 20 half decodes and 2 all-tetrode checks on the 5 pilot epochs, on idle breeze GPUs, from the main
  tree at `d29cc3c`, after four R11-compatibility checks. All ok. S = all reproduces the canonical decode (max |ΔP|
  2.4e-4; events identical, 257/257 and 245/245).
- **Result** (animal means, n = 5; half labels against canonical labels scored by the same four runs):
  - Local immobile 0.98 vs 1.17; Local running 9.27 vs 9.27; all 5/5.
  - NL-Continuous immobile, against the mismatched location, 2.91 [0.88, 4.95] p 0.017 vs 4.46; against the
    animal, 3.41 vs 4.48.
  - NL-Fragmented immobile, against the mismatched location, 2.33 vs 1.58.
  - **No-Spike fails the log-ratio test:** −55.9 immobile and −78.7 running (p 0.016), 0/5, against +6.65 and
    +3.93. Yet the other half is silent in 34 % of these bouts against 1 % expected (excess +0.32, 5/5).
- **Caveat:** the two label sources select different bouts (event-bin Jaccard 0.32–0.80). The comparison holds
  the scoring runs fixed, not the bouts.
- **Changes here:** §4.7; item 37 closed; item 52 added.
- **Still open:** whether the No-Spike failure is the half decoder's lower spike count (the sign tracks the
  selecting half's spike share, Spearman 0.80; a hypothesis). Spike-balanced splits would test it.

### 10.7 D1: the chimi20200221_08_r4 mismatch

Source: [`d1_decode_mismatch_report.md`](../../.claude/research/refresh_followup/d1_decode_mismatch_report.md).

- **Run:** a read-only rebuild of the whole epoch on the CPU, a per-tetrode breakdown, a numpy float32
  emulation, and an occupancy scan of all 242 epochs.
- **Result (confirmed):** the animal never entered patch 1. Occupancy is exactly 0 at bins 0–70 and a float32
  subnormal at bins 71–72 (5.4e-43, 5.6e-40). The GPU keeps subnormals; JAX on the CPU flushes them to zero.
  So CPU rebuilds differ by up to 20 nats per spike at bins 71–78. The numpy emulation reproduces the stored
  values, so the stored decode is correct float32 arithmetic. Only this epoch is affected: the next-lowest
  minimum occupancy is 4e-6.
- **A separate, model-level effect** in this epoch: never-visited bins attract MAPs in silent bins. 192 of its
  6,779 non-local event bins have their MAP in never-visited bins; in the event-arm table the never-visited arms 1 and 6
  hold 90 and 176 MAP bins (3.9 % of its event bins). This explains Task 28's arm-0-unidentified fits.
- **Ruling** (ledger 12:20 PDT): option (a), keep the epoch and document it; fix the checks (Task 29); draft a
  non_local_detector issue. Keeping the epoch is listed for the lead in §0.
- **Changes here:** the cause in §4.6; items 50 and 53.
- **Still open:** the issue is not filed (no GitHub credentials on breeze). D1 infers, without checking, that
  the epoch's ContFrag decode behaves the same.

### 10.8 Tasks 27, 28 and 29 (code, on `full-pipeline-refresh-code`)

Sources: [`task-27-report.md`](../../.superpowers/sdd/PLAN/task-27-report.md),
[`task-28-report.md`](../../.superpowers/sdd/PLAN/task-28-report.md),
[`task-29-report.md`](../../.superpowers/sdd/PLAN/task-29-report.md), and the ledger. All three were reviewed
and approved (ledger 08:50, 13:10 and 14:25 PDT).

- **Task 27** (`b7dc1d6`, `cfb2f4e`, `c0b69fd`; fix round `28e00cc`, `e6e3b9e`, `a4ba6d2`; `ec04edf`):
  - fit provenance records the commit the process loaded;
  - a first switch is decided by the stop's own arriving trip (ruling 08:10 PDT). On the shipped stop table
    first switches go 2,290 → 2,319; episodes 1,936 → 1,961 (non_local), 1,705 → 1,721 (SWR), 1,675 → 1,693
    (MUA);
  - the rate runners save rows sorted by the full row key.
  - 1,077 tests passed. Closes items 2, 4 and 5.
- **Task 28** (`6e329de` … `f05fa4a`):
  - the conditional-Poisson fitter scales columns by their within-event SD, adds a Newton finish and analytic
    SEs, and stores `llf`, `newton_decrement`, `max_abs_grad`, `n_newton_steps` and `unidentified_terms`;
  - a single-column separated term is fit at its limit and reported unidentified; a column constant within
    every event is unidentified; a mixed direction is refused; aggregators count `n_unidentified` and gate on
    it.
  - On the 2,662 F3 tasks: 2,658 fit, 4 refused (3 with too few events, and j16 20210707 04_r2 C2_base nonSWR
    for a mixed direction). The 2,583 identified fits are within 1.8e-9 log-likelihood of F3's optimum and
    about 11× faster. No headline term is unidentified in any fit.
  - Also `929b1b9`: the content runner's provenance records the loaded commit.
  - Parked: the same stopping problem in other fitters (item 46); the multi-column limit (item 45); N1 (the
    clustered SE, `sd_*` and separation flag read left-out constant-response events; 4 real fits, ≤ 0.4 %).
- **Task 29** (`da34665`, `e224a09`):
  - the state-evidence checks and the reality engine leave out bins with 0 < occupancy < 1e-20 (exact zeros
    stay in: both devices agree there; leaving them out would be D1's option (c), not approved);
  - on chimi20200221_08_r4 it flags 9 bins (71–79), and the reality engine leaves out 19 of 184 windows (17 of
    53 SWR-remote). Other epochs give byte-identical outputs (2 real and 2 simulated checked). 188 tests
    passed.
  - Still open: the rerun of that epoch after the merge (item 50); whether dropping 32 % of its SWR-remote
    windows is acceptable (the lead's); the half-decode ground-intensity check at bins 71–79 is unchecked.
- **Counts that differ by definition:** D1 counts 22 of 182 events with a MAP bin in 71–78; Task 29 counts 19
  of 184 windows in 71–79 (the engine's three groups plus baseline windows). Task 29 says they are the same
  order; they were not reconciled.

### 10.9 R6c: the content refit with the fixed fitter

Source: [`r6c_report.md`](../../.claude/research/refresh_r13/r6c_report.md).

- **Run:** 19 runs (C01–C07, K01–K03, X01–X05, F01–F04) from `continuum-swr-replay-run2` at `f05fa4a`, clean,
  CPU only, writing the canonical content bundles; backup `Processed-Data/_pre_r6c_backup_20261016/fits/`
  (267 files, SHA256SUMS verified). Not rerun: DM (C08, K04–K07), rate (R01–R11), P/S.
- **Gates:** all pass. Headline coefficients equal F3's optimum to ≤ 5.6e-6 SE and llf to ≤ 1.8e-9;
  inclusion equals Task 28's table; switch counts equal Task 27's; `check_fit_provenance` fresh on all four
  roots. One documented exception: X01 stores no optimizer columns (item 47).
- **Result:** headlines equal F3's predictions in the third or fourth decimal (section 3). No log-likelihood
  fell. 48 single-session moves above 0.35 SE, all traced to a stop-short fit, a limit term, or an R6b fit
  flagged separated. Five rows move by cohort and must be re-read (section 3). X03 non_local patch_switch
  falls 189 → 157 sessions (item 45).
- **C01's directory swap** deleted the 44 K04–K07 DM files; they were restored byte for byte (item 48).
- **Changes here:** section 3 content rows and the re-read table; 6.1a V9; items 45, 47, 48 and 50.
- **Still open:** figures and docs that read the content fits are stale until the merge (item 50); the X
  runners silence the fitter's log (`compare_event_source_models.py:78-82`), so refusal reasons are not in the
  X03–X05 logs.

### 10.10 R13 interim: artifact-rule robustness

Source: [`r13_robustness_report.md`](../../.claude/research/refresh_robustness/r13_robustness_report.md).
**Interim:** the lead paused R13 at 13:48 PDT.

- **Run (done):** V0 (no mask, with a sandbox SWR run under R-T and no mask normalisation), V3 (+1 s after each
  step), the >2 % masked-epoch sensitivity, the masked-time table and the R-T against mask split. Code
  `bb19449` in the run checkout; 242 epochs; reference = a copy of the R6b fits taken at 12:44, before R6c.
- **Result, V0 and V3:** no headline moves beyond its paired session bootstrap interval. Per-bin M7r immediate
  RR, canonical → V0 / V3: SWR 6.39 → 6.39 / 6.38; NL with SWR 9.91 → 10.35 / 9.82; NL no-SWR 1.42 → 1.64 /
  1.56. Content and DM terms move ≤ 0.04 in V0 and ≤ 0.06 in V3, except the null CS5 (−0.012 → −0.116 in V0).
- **Result, >2 % sensitivity:** 3 epochs over 2 % masked; leaving them out moves nothing beyond its interval.
  The 10 epochs over 2 % excluded (per-session families only): SWR RR 5.88, Δ −0.083 log [−0.161, +0.014].
- **Result, the split** (205 R3 epochs): SWR RR 3.15 (R3) → 5.44 with R-T and no mask (+0.547 log [0.310,
  0.840]; j16 1.16 → 5.20) → 5.27 with mask normalisation and the rule (−0.031 [−0.175, 0.138]). R-T acts in j16
  alone; mask normalisation acts in chimi, senor and peanut and raises `is_swr` by about 0.015.
- **Result, the masked-time table:** the mask covers 345.5 s (0.111 % of 86.4 h) in 85 epochs, all from steps;
  excluded time 691.9 s. The rule removes 251 Kay events at detection, 9 non-local events, 0.26 % of the 4-ms
  stop bins and 436 at-well-cohort stops.
- **Changes here:** §4.7; item 38 closed in part, item 49 added; C8's DM CS6 now derived (6.6).
- **Still open:** V1, V2 and X_R10nr (item 49; resume commands in its §9). V0's two channels are inferred, not
  measured (X_R10nr would measure them). The rule's 9 non-local events here against R8a's 21 are not
  reconciled (different definitions). R13's content fits used the old fitter, so its deltas include optimizer
  noise (up to 0.35 SE by F3), and its canonical content values are the R6b ones (e.g. C2_base SWR 0.884,
  n 221).
