# Parity audit for the readability-hygiene change set

**Date:** 2026-09-06
**Commit range audited:** `e38f014..55c68f6` (11 commits; `55c68f6` is the audited HEAD)
**Frozen baseline:** `Processed-Data/replay_content/_golden_baseline_pre5a/`
(frozen at `e38f014`, before the first edit, by
`uv run python scripts/qa/freeze_regression_baseline.py --label pre5a`)

## Verdict

**PASS.** Every canonical table is byte-identical to the frozen baseline; the builders,
re-run at HEAD on a five-session slice, reproduce the live rows exactly (the only
differences are the four cohort-scoped `_z` columns, which are an affine consequence of
standardizing over 5 sessions instead of 218, not a code difference); the content fit is
byte-identical; the SWR-power fit is exactly zero-delta; the uncertainty-rate fit drifts
only in the four rows and seven columns already characterized as optimizer noise on a flat
likelihood direction.

One suite test fails — `test_rate_ladder_rungs_share_a_cohort` — for a reason that predates
this change set and is independent of it (§ "Full unit suite").

---

## 1. Canonical-table parity gate

```bash
uv run python scripts/qa/compare_to_golden_baseline.py \
    --baseline Processed-Data/replay_content/_golden_baseline_pre5a
```

Result: `byte-identical to baseline across 8 table(s)`, exit 0.
Wall clock 1:04.77, peak RSS 13.58 GB. The gate's default `--candidate` is the live
`Processed-Data/replay_content` root and its default `--atol` is `0.0` (exact), with no
`--expected-drift-columns` and no `--expected-added-columns` declared, so this is exact
parity with nothing waived.

`compare_to_golden_baseline.py` compares on each table's registered row key via
`table_parity.compare_table_roots`. Independently, the on-disk parquet files are also
identical byte for byte (`sha256sum`):

| table | rows | sha256 (live == baseline) |
|---|---:|---|
| `event_table` | 65,913 | `68ffdc1bdce6881cd53a6f55ae813020907cd6eb86860a3d90a07a699869491c` |
| `event_arm_table` | 593,217 | `6b11ce36bcf23fa2b0d81cd56519ba6973a196e2f512ca1fd3011b7f2cca83f3` |
| `event_patch_table` | 197,739 | `dff5b2bb8312273da6a739dc69067a1e3aec455fdce358047b60ad6fa886c126` |
| `event_sequence_table` | 2,971,614 | `9f23c2b6071201c32434cf66492fe32112f43ef7f12f8006d84abcad73d7b8ed` |
| `event_onset_table` | 137,745 | `4ff2f1635c5b159a843a2c72a92cc13c658a46aabe250c603dc035a826eec57c` |
| `stop_table` | 107,400 | `ea62e0b65fbec97c8fce64185a6627856a7056b1f5e3915aef52bf32d944610f` |
| `stop_timebin_table` | 32,533,075 | `d508ecf5384a58fb4fdba3716de773e3d18f9f19466effe4e9446cd8eff86c7e` |
| `qvals_table` | 105,359 | `686f0985b7abcf60d5c174140944be2b5a98868840268fa053b8eea94f167c38` |

(The live `qvals_table.parquet` sits under `models_per_bin_rate/`; the frozen copy is flat.)

## 2. Builder re-run: the tables are still *produced* identically

The canonical tables were not regenerated during this change set, so § 1 alone proves only
that nothing overwrote them. To prove the *builders* still produce the same values at HEAD,
one session per animal was rebuilt from the pinned behavioral cache into a scratch
directory:

```bash
uv run python -m continuum_swr_replay.analysis.replay.prepare_data \
    --from-cache \
    --max-per-animal 1 \
    --output-dir <scratch>/rebuild \
    --behavior-cache-dir Processed-Data/replay_content/_session_cache \
    --contingency-dir  Processed-Data/replay_content
```

Wall clock 3:40.21, peak RSS 11.78 GB, exit 0. Sessions rebuilt:
`chimi20200213 02_r1`, `j1620210706 02_r1`, `peanut20201127 14_r7`, `senor20201027 02_r1`,
`wilbur20210328 02_r1`. Driver totals: `events: 3354, event_arm rows: 30186, stops: 5046,
sessions: 5`; cross-table invariant `Σ n_selected_events == len(event_table) == 3354` held.

Each rebuilt table was then diffed against the same five sessions' rows in the *live*
canonical table with `table_parity.diff_table(..., atol=0.0)` on the table's registered row
key:

| table | rebuilt rows | live-subset rows | cols | row key | drifted columns |
|---|---:|---:|---:|---|---|
| `event_table` | 3,354 | 3,354 | 103 | `animal, session_date, epoch, event_id` | **none (exact)** |
| `event_arm_table` | 30,186 | 30,186 | 84 | `… , event_id, candidate_arm` | 4 `_z` columns (below) |
| `event_patch_table` | 10,062 | 10,062 | 29 | `… , event_id, patch` | **none (exact)** |
| `event_sequence_table` | 142,320 | 142,320 | 18 | `… , event_id, t_in_event` | **none (exact)** |
| `stop_table` | 5,046 | 5,046 | 55 | `… , stop_id` | **none (exact)** |

Row counts, key sets, column sets, dtypes and NaN positions matched everywhere.

### The four `event_arm_table` `_z` columns are a cohort-scope artifact, not a code difference

`_z` columns are standardized over the fit cohort, so a 5-session rebuild computes its
reference over 5 sessions while the live table's reference spans 218. The raw source column
of each `_z` is **bit-identical** between the rebuild and the live rows, and the two `_z`
versions are related by an exact affine map (residual at float round-off):

| `_z` column | raw source identical | max &#124;Δz&#124; | affine slope | intercept | max &#124;residual&#124; |
|---|---|---:|---:|---:|---:|
| `reward_belief_mean_candidate_arm_preoutcome_z` | yes | 1.70598 | 1.402821632141 | 0.784726988888 | 6.66e-15 |
| `reward_belief_variance_candidate_arm_preoutcome_z` | yes | 1.01549 | 1.276040466670 | −0.476152059228 | 1.67e-15 |
| `reward_fraction_candidate_arm_last10_visits_z` | yes | 0.677261 | 1.122552953417 | 0.510957821017 | 1.22e-15 |
| `task_reward_probability_candidate_arm_z` | yes | 0.906110 | 1.206073638849 | 0.599393592206 | 4.88e-15 |

The slopes are exactly the ratio of the two references' SDs — e.g. for
`reward_belief_mean_candidate_arm_preoutcome`, the rebuild's
`standardization_reference.json` records `std = 0.16691575250266708` over 19,962 reference
rows and the canonical one `std = 0.11898572753538102` over 390,120, whose ratio is
1.402821632141, the fitted slope to 12 decimals. Rebuilding the whole cohort would restore
the canonical reference and hence the canonical `_z`.

## 3. Fit smokes at HEAD versus their pre-edit runs

### 3a. Content fit — byte-identical

```bash
uv run python scripts/content/fit_replay_content_models.py --models M0,M0a \
    --output-dir <scratch>/content
```

(`fit_replay_content_models.py` has no `--max-per-animal`, so this is the full cohort at two
rungs: a 436 × 70 frame, 218 sessions × 2 rungs.) Wall clock 0:54.34, peak RSS 2.09 GB.

| file | sha256 pre-edit | sha256 at HEAD | |
|---|---|---|---|
| `all_models.pkl` | `dadc7da64f108063968a94ec276937dec24e03b7f4ab0cf10c60569985dbebef` | same | identical |
| `M0.pkl` | `e6e6a468e042b1db225c0f998cbb6ede404d3b46670bba26cf4be1b62ded9a86` | same | identical |
| `M0a.pkl` | `ab70b24d7da6065b4da706837aded7a0fe34f676bebf2ccbe9c2699e16d01922` | same | identical |
| `model_comparison.pkl` | `080b8a42d1f454d3de80984049502594b569f3be24076366ebabb4ee68724c46` | same | identical |
| `all_models_cohort_audit.csv` | `6a77887a784a7c2af8502a5600a0214af6487aad60fc37f9a07a6a59ee1af340` | same | identical |

The `.provenance.json` sidecars carry a wall-clock field and are excluded by construction.
Max &#124;Δ&#124; over every coefficient is therefore **0.0 exactly**, not merely below 1e-12.

### 3b. SWR power model — exactly zero delta

```bash
uv run python scripts/rate/fit_swr_power_model.py --max-per-animal 1 \
    --output-dir <scratch>/power
```

Wall clock 0:23.57, peak RSS 0.98 GB. Frame 5 × 32 (5 sessions, one per animal); identical
columns, dtypes and NaN positions; no non-numeric column differs. All 25 numeric columns are
**exactly zero** delta, the 7 coefficient columns among them:

| coefficient column | max &#124;Δ&#124; |
|---|---:|
| `known_reward_outcome_lag_1` | 0.0 |
| `task_reward_probability_chosen_arm` | 0.0 |
| `rpe_postoutcome` | 0.0 |
| `patch_switch_fraction_last5_completed_trips` | 0.0 |
| `task_reward_probability_patch_mean_gap_best_vs_second` | 0.0 |
| `reward_belief_mean_chosen_arm_preoutcome` | 0.0 |
| `time_since_stop_start_s` | 0.0 |

Both side-output CSVs (`swr_power_cohort_audit_smoke.csv`,
`swr_power_post_outcome_gate_audit_smoke.csv`) are byte-identical
(`82841e2e9968f086…`, `5ca88ce879565834…`).

The canonical `models_per_bin_rate/swr_power_model.pkl` was **not** touched: sha256
`2f14c5996d760d578c89ecb920d5762de04be5540212061c49b46fc4edbbf9b7` before and after.

### 3c. Uncertainty rate — drift confined to four flat-likelihood rows

```bash
uv run python scripts/rate/fit_uncertainty_rate.py --max-per-animal 1
```

(No `--output-dir`; it writes `uncertainty_rate_*_smoke.*` into
`Processed-Data/replay_content/models_per_bin_rate/`, from where they were copied out.)
Wall clock 0:33.74, peak RSS 1.40 GB. Frame 15 × 164 (5 sessions × 3 populations), identical
columns, identical NaN positions, no non-numeric column differs.

Summary: 152 numeric columns, 49 exactly zero. **Max &#124;Δ&#124; over all `coef_*` columns
= 5.38e-06**; **max &#124;Δ&#124; over every `coef_*` column outside the seven named below
= 9.86e-13**, i.e. inside the 1e-12 gate.

Every column with max &#124;Δ&#124; > 1e-12:

| column | max &#124;Δ&#124; | NaN-position mismatch |
|---|---:|---:|
| `rr_hist_rcos_0` | 1.307290e+20 | 0 |
| `rr_hist_rcos_2` | 2.771966e+18 | 0 |
| `pval_reward_history_incomplete` | 6.869565e-01 | 0 |
| `coef_hist_rcos_0` | 5.377567e-06 | 0 |
| `coef_Intercept` | 5.377560e-06 | 0 |
| `coef_reward_history_incomplete` | 2.068258e-06 | 0 |
| `rr_reward_history_incomplete` | 2.068255e-06 | 0 |
| `rr_Intercept` | 1.395096e-07 | 0 |
| `bse_Intercept` | 1.165721e-07 | 0 |
| `bse_hist_rcos_0` | 1.165721e-07 | 0 |
| `coef_hist_rcos_1` | 1.501164e-08 | 0 |
| `bse_hist_rcos_1` | 1.310723e-08 | 0 |
| `coef_hist_rcos_2` | 7.055378e-09 | 0 |
| `bse_hist_rcos_2` | 6.161263e-09 | 0 |
| `deviance` | 4.994945e-09 | 0 |
| `aic` | 4.994718e-09 | 0 |
| `coef_hist_rcos_3` | 4.547881e-09 | 0 |
| `bse_hist_rcos_3` | 3.970959e-09 | 0 |
| `rr_hist_rcos_1` | 3.346486e-09 | 0 |
| `bse_reward_history_incomplete` | 3.125334e-09 | 0 |
| `llf` | 2.497359e-09 | 0 |
| `pval_hist_rcos_1` | 7.019647e-10 | 0 |
| `pval_hist_rcos_3` | 6.571109e-10 | 0 |
| `coef_task_reward_probability_patch_mean_gap_best_vs_second` | 5.364614e-10 | 0 |
| `pval_hist_rcos_2` | 3.615738e-10 | 0 |
| `rr_task_reward_probability_patch_mean_gap_best_vs_second` | 2.094066e-10 | 0 |
| `rr_hist_rcos_3` | 1.255136e-10 | 0 |
| `rr_reward_history_rcos_2` | 3.066702e-11 | 0 |
| `bse_task_reward_probability_patch_mean_gap_best_vs_second` | 1.148593e-11 | 0 |
| `rr_bs(tc, …)[4]` | 8.668621e-12 | 0 |
| `rr_reward_history_rcos_0` | 4.110490e-12 | 0 |
| `rr_rpe_postoutcome` | 3.637979e-12 | 0 |
| `rr_bs(tc, …)[1]` | 2.426503e-12 | 0 |
| `rr_reward_history_rcos_1` | 2.387424e-12 | 0 |
| `rr_bs(tc, …)[2]` | 2.085443e-12 | 0 |
| `rr_bs(tc, …)[0]` | 1.003642e-12 | 0 |

The `coef_*` movers are exactly `coef_Intercept`, `coef_hist_rcos_0..3`,
`coef_reward_history_incomplete`, and the nominal
`coef_task_reward_probability_patch_mean_gap_best_vs_second` — the accepted pattern. `rr_*`
columns are `exp(coef)`, so the 1e+20 entries are `exp` of a ±5e-6 wobble at
`coef ≈ −100`; `pval`, `bse`, `llf`, `deviance` and `aic` are downstream of the same fits.

Top `coef_*` by drift, showing where the cut falls:

| `coef_*` column | max &#124;Δ&#124; |
|---|---:|
| `coef_hist_rcos_0` | 5.377567e-06 |
| `coef_Intercept` | 5.377560e-06 |
| `coef_reward_history_incomplete` | 2.068258e-06 |
| `coef_hist_rcos_1` | 1.501164e-08 |
| `coef_hist_rcos_2` | 7.055378e-09 |
| `coef_hist_rcos_3` | 4.547881e-09 |
| `coef_task_reward_probability_patch_mean_gap_best_vs_second` | 5.364614e-10 |
| — end of the named set; everything below is inside the gate — | |
| `coef_reward_history_rcos_1` | 9.858780e-13 |
| `coef_reward_history_rcos_2` | 4.050094e-13 |
| `coef_reward_history_rcos_0` | 3.614886e-13 |
| `coef_patch_switch_fraction_last5_completed_trips` | 3.410605e-13 |
| `coef_bs(tc, …)[4]` | 2.664535e-13 |

The four covariates whose scale the changed z-scoring actually sets —
`coef_relative_value_switch_minus_stay_preoutcome`,
`coef_reward_belief_differential_entropy_leaf_mean_preoutcome`,
`coef_reward_belief_mean_update_l1_all_ports_postoutcome`,
`coef_reward_belief_mean_update_l1_alternative_patches_postoutcome` — move by at most
**2.47e-13**.

**Row confinement.** Only **4 of 15** rows carry any `coef_*` drift above 1e-12, and all four
are the rows whose self-history coefficient is enormous:

| animal | session_date | epoch | model | `coef_hist_rcos_0` | `separated` | `converged` |
|---|---|---|---|---:|---|---|
| j16 | 20210706 | 04_r2 | MU_uncertainty | 67.136198 | False | True |
| peanut | 20201128 | 06_r3 | MU_uncertainty | −104.842792 | False | True |
| senor | 20201028 | 02_r1 | MU_uncertainty | −109.547599 | False | True |
| wilbur | 20210328 | 04_r2 | MU_uncertainty | −88.070061 | False | True |

The eleven non-drifting rows span &#124;`coef_hist_rcos_0`&#124; from 0.051 to 59.4, so the
partition is not exactly at 50 — one stable row sits at 59.4 while every drifting row is
above 67. The likelihood is flat along the self-history direction in those four fits, so the
optimizer's stopping point is set by its tolerance rather than by the data, and the
last-bit change in the standardized inputs is amplified there. No scientific quantity moved.

**Related observation (not a parity finding):** three of the four drifting rows are flagged
`separated = False, converged = True` despite a self-history coefficient near −100, so the
separation detector under-reports on this ladder.

## 4. Full unit suite

```bash
uv run pytest tests/unit -q
```

Collected count (`uv run pytest tests/unit --co -q | tail -1`): **2132 tests collected**.

```
1 failed, 2131 passed, 63 warnings in 295.04s (0:04:55)
FAILED tests/unit/test_replay_tables_audit_regressions.py::test_rate_ladder_rungs_share_a_cohort
```

This run includes the `slow` marks, which the working runs during the change set deselected
(`-m "not slow"`, 2118 passed / 14 deselected).

**The failure predates this change set and is independent of it.** Evidence:

1. The test reads exactly one input, the materialized artifact
   `Processed-Data/replay_content/models_per_bin_rate/all_models.pkl`, and asserts every
   per-bin rate rung reports the same `n_bins`. It observed four distinct cohorts:
   `M5a_time_arrival` 6,477,947; `M7ra_time_arrival_histcos_rew` 6,480,795;
   `M0a_const_arrival` 6,618,107; the remaining seven rungs 7,315,746.
2. That artifact is **byte-identical** to the copy frozen into
   `_golden_baseline_pre5a/models_per_bin_rate/` before the first edit (sha256
   `475986413800b84c4462e77cf5ae1d674c8f90f59e25ac181240c1fb1869497b`), and its provenance
   sidecar records it was fit at `eecf54a5…`, a commit that is an ancestor of the change
   set's branch point `e38f014`.
3. The only edit to the failing test file in this change set is an import repoint
   (`analysis.replay.tables` → `tables_arm_patch` / `tables_event`); the assertion is
   untouched.
4. The test's own docstring states it "holds only once they have been refit with the frozen
   cohort (hence `slow`)". The per-bin rate ladder has not been refit; that refit is tracked
   as an open follow-up elsewhere.

## 5. Module size

```bash
wc -l src/continuum_swr_replay/analysis/replay/model_specs.py
```

**1394** lines (from 1535 before the narrative move). The design journal now lives in
`docs/design/model_history.md`. The sub-900 target is unreachable without moving executable
content: the module's measured floor is 857 executable lines (non-comment, non-docstring,
non-blank) plus 79 blank lines required by PEP 8 spacing = 936, before a single explanatory
comment.

## 6. `--help` sweep of the scripts touched by the export trim and the deletions

Every script under `scripts/content/`, `scripts/rate/` and `scripts/qa/`, plus
`scripts/figures/plot_swr_contrast_coefs.py` and `scripts/decoding/regenerate_stop_table.py`
— **35 scripts** — was checked for an `argparse.ArgumentParser` first (all 35 have one), then
run as `uv run python <script> --help`:

**35 of 35 exited 0 and printed a `usage:` line. Zero failures.** So the trimmed package
export surface breaks no importer, and no deleted symbol is still reached at import time.

Two notes on how the sweep was made safe:

- 34 of the 35 guard their entry point with `if __name__ == "__main__"`.
  `scripts/figures/plot_swr_contrast_coefs.py` deliberately has no `main()` — it runs top to
  bottom — but it constructs its parser and calls `parse_args()` at module level *ahead* of
  its first table read, so `--help` exits during parsing and reads nothing. (It does create
  `paper/figures/swr_contrast/` at import; that directory already existed.)
- `scripts/rate/fit_swr_power_model.py` previously had no argparse at all, so any argument
  including `--help` was ignored and the full 217-session fit ran, overwriting the canonical
  `swr_power_model.pkl`. It now has `--max-per-animal` / `--output-dir`, and `--help` prints
  and exits without fitting; the canonical artifact's sha256 was verified unchanged across
  this entire audit.

## 7. Deleted symbols and their grep evidence

Every symbol was grepped over `src/ scripts/ tests/ docs/` (and, for the second group, also
`notebooks/ paper/`) before deletion, and re-grepped at HEAD for this audit.

### Module

| deleted | evidence |
|---|---|
| `src/continuum_swr_replay/analysis/replay/tables.py` (58-line pure re-export) | 26 `from …tables import` statements across 20 files were repointed to the four owning modules (`tables_event`, `tables_stop`, `tables_arm_patch`, `_tables_common`), including the walkthrough notebook's `.py` and `.ipynb`. At HEAD: `grep -rn "analysis\.replay\.tables\b\|analysis/replay/tables\.py" src/ scripts/ tests/ notebooks/` → **0 hits**; the file does not exist. |

### Unreferenced definitions

| symbol | hits before deletion | hits at HEAD |
|---|---|---:|
| `_helpers.immobility_bout_starts` | 2: the definition + its name in the generated `docs/regression_pipeline_map.md` (callers use `immobility_bout_intervals`) | 0 |
| `retrospective._PATCH_TO_ARM_SET` | 1: the definition | 0 |
| `simulation._BASE_BINARY` | 1: the definition | 0 |
| `simulation._CONTINUOUS_ARM` | 1: the definition | 0 |
| `simulation._EVENT_LEVEL` | 1: the definition | 0 |

`_spatial_bandit_rl.set_nan_at_max2` was listed as a deletion candidate and **kept**: it is
not a no-op. It assigns `trials_to_next_switch_with_nans`, which the next line reads, and its
NaN branch censors the final countdown group when the epoch's last trial is not a switch.

### Test-only surfaces

| symbol | production callers | disposition |
|---|---|---|
| `visit_recency_per_arm` | none (package re-exports, 3 tests, 1 benchmark, generated map) | deleted; tests rewritten against the `*_all_events` vectorized form |
| `arm_never_visited_per_arm` | none (same shape) | deleted, same |
| `rolling_reward_count_per_arm` | none (same shape) | deleted, same |
| `recent_reward_rate_per_arm` | none (same shape) | deleted, same |
| `behavioral_location_at_times` | none; its only test was of its own `DeprecationWarning` | deleted with that test; the explicit API `behavioral_location_from_port_experiences` is untouched |
| `clocks.trial_cutoff_times` | a pure wrapper over `_trial_arrival_times` | wrapper deleted; the implementation took the public name `trial_arrival_times` and 4 call sites were repointed |
| `clocks._trial_outcome_times` | a pure wrapper's implementation | implementation took the public name `trial_outcome_times`; wrapper deleted |

At HEAD, `grep -rn "\b<symbol>\b" src/ scripts/ tests/ docs/ notebooks/` returns **0 hits**
for `visit_recency_per_arm`, `arm_never_visited_per_arm`, `rolling_reward_count_per_arm`,
`recent_reward_rate_per_arm`, `behavioral_location_at_times`, `trial_cutoff_times`,
`_trial_arrival_times` and `_trial_outcome_times`.

### Private re-exports dropped from two package `__init__`s

Thirteen `_x as _x` re-exports were removed — 7 from `covariates/__init__.py`
(`_trial_arrival_times`, `_trial_outcome_times`, `_prepare_trials_for_context`, `_fit_alpha`,
`_leaky_q_value_diff`, `_omission_streak`, `_q_chosen`) and 6 from
`covariates/history/__init__.py` (`_prepare_port_experiences`, `_fit_alpha`, `_q_chosen`,
`_observed_port_outcomes`, `_leaky_q_value_diff`, `_omission_streak`). Consumers that went
through the package were repointed to the owning modules (`history.diagnostics`,
`history.value_uncertainty`, `history._common`, `history.reward_history`). The definitions
themselves stayed: at HEAD `_prepare_trials_for_context` has 3 hits, all in
`covariates/context.py`, and `_leaky_q_value_diff` has 2, both in
`covariates/history/value_uncertainty.py`.

### Package export surface

`analysis/replay/__init__.py`'s `__all__` went from **90 names to 51** (verified by parsing
the `__all__` assignment at the trim commit's parent and at the trim commit: 90 → 51; the
branch point had 91, one of which the clock rename consumed). The 39 dropped names had zero
package-level importers across `src/ scripts/ tests/ notebooks/ paper/` (`.ipynb` cells
included). Kept despite having no regex-visible importer: the four lazy Dirichlet-multinomial
exports resolved through the PEP-562 `__getattr__`, and four names a dynamic-`getattr`
contract test pins as package API (`UNCERTAINTY_COVARIATES`, `add_trial_context`,
`behavioral_location_from_port_experiences`, `summarize_behavioral_location`).

Verified at HEAD:

```python
import continuum_swr_replay.analysis.replay as m
len(m.__all__)                       # 51
exec("from continuum_swr_replay.analysis.replay import *", ns)
[n for n in m.__all__ if n not in ns]  # []
```

i.e. the trimmed surface really star-imports, with every listed name resolving.

## 8. Scope hygiene

Nothing under `Processed-Data/`, `logs/` or the scratch tree is committed by this audit; the
only tracked change is this file. Smoke artifacts were written to a scratch directory except
the uncertainty runner's `uncertainty_rate_*_smoke.*`, which it writes into
`models_per_bin_rate/` by design and which are untracked. No canonical table or fit artifact
was modified: every sha256 in § 1 matches the frozen baseline, and the canonical
`swr_power_model.pkl` sha256 was re-checked after every fit run.

> Postscript: the assertion in `test_rate_ladder_rungs_share_a_cohort` was rewritten after this audit's range (commit 8d3c923) to compare `n_bins` per (session, time clock) and to bound rung coverage; it passes at the branch head. The arrival-clock rungs `M5a`/`M7ra` fit 213 of 218 sessions (five fits returned None); per-session cohorts are equal in 436/436 (session, clock) groups.
