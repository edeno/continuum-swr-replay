# Event-Rate Model Layer Statistical Review

## Findings

1. **Per-bin `TMAX=4.0` is implemented as 3.99 s, silently dropping a small tail of events.**  
   `TMAX` is documented as 4 s in [`model_specs.py:516`](../../../src/continuum_swr_replay/analysis/replay/model_specs.py#L516), but `aggregate_to_timebins` uses `nmax = int(tmax / bin_s)` at [`_helpers.py:1128`](../../../src/continuum_swr_replay/analysis/replay/_helpers.py#L1128). With `BIN_S=0.015`, this creates 266 bins = 3.99 s, while onsets are filtered with `t_since_stop_onset < tmax` at [`_helpers.py:1145`](../../../src/continuum_swr_replay/analysis/replay/_helpers.py#L1145). Artifact check: 169 onsets in `[3.99, 4.0)` are dropped across 103 sessions (`86` non-local, `43` SWR, `40` MUA), and about 124.32 s of intended exposure is absent (~0.11% of first-4-s exposure). Small, but silent and exactly in a model-boundary convention.

2. **Strict reward lags are leakage-free; `is_after_omission` / `is_after_switch` are not equivalent and should not carry the main immediate-reward claim.**  
   `reward_lag1..10` and `recent_trial_reward_rate` are computed at bout starts using the goal-arrival cutoff in [`tables.py:841`](../../../src/continuum_swr_replay/analysis/replay/tables.py#L841) and [`_helpers.py:704`](../../../src/continuum_swr_replay/analysis/replay/_helpers.py#L704). That is the right past-only definition. By contrast, `is_after_reward` / `is_after_omission` are attached from the current trip selected by trial start at [`_helpers.py:83`](../../../src/continuum_swr_replay/analysis/replay/_helpers.py#L83) and [`_helpers.py:126`](../../../src/continuum_swr_replay/analysis/replay/_helpers.py#L126). The model spec explicitly notes this can encode the future outcome for stops during approach [`model_specs.py:481`](../../../src/continuum_swr_replay/analysis/replay/model_specs.py#L481). Artifact check: `reward_lag1` agrees with `is_after_reward` on only 0.8995 of finite rows. Use `Rlag` / per-bin `reward_lag1` for the claim; keep R1/R2 omission as literature-comparable but caveated.

3. **Claim #19c has stale per-bin values relative to the unified 15 ms run.**  
   [`paper/claims_table.md:53`](../../../paper/claims_table.md#L53) reports per-bin lag-1 RR `1.42 / 3.10 / 1.08` and mixed "4/15 ms bins"; [`docs/rate_models.md:95`](../../../docs/rate_models.md#L95) says the current unified-15 ms run shifted absolute coefficients. Artifact check from `models_per_bin_rate/all_models.pkl`: M7 lag-1 RR medians are `swr=1.413`, `non_local_swr=3.930`, `non_local_no_swr=1.108`; M7r gives `3.829` for `non_local_swr`. The qualitative dissociation holds, but paper numbers should be updated or explicitly labeled as superseded.

4. **The headline dissociation is supported, but current inference is session-level, not fully hierarchical.**  
   Summaries use Wilcoxon signed-rank over session coefficients plus per-animal sign consistency [`summarize_per_bin_rate_models.py:40`](../../../scripts/summarize_per_bin_rate_models.py#L40), [`summarize_per_bin_rate_models.py:53`](../../../scripts/summarize_per_bin_rate_models.py#L53). That is useful, but 218 sessions are not independent animals. The reportable claim should lean on paired within-session direction and 5/5 animal consistency, and add an animal/session hierarchical bootstrap, clustered sandwich, or animal-level permutation before treating p-values like population-level evidence.

5. **Per-bin dummy self-history models are useful for kernel shape, not prediction.**  
   The repo documents dummy-lag separation and negative CV [`docs/rate_models.md:149`](../../../docs/rate_models.md#L149). Artifact check confirms M7 dummy CV bits/event are negative for all split outcomes (`non_local=-0.108`, `non_local_no_swr=-0.760`, `non_local_swr=-0.819`, `swr=-0.442`), while M7r improves prediction (`non_local=0.199`, `non_local_swr=0.061`, `swr=0.063`, though `non_local_no_swr=-0.057`). Reward lag-1 coefficients are stable across M7/M7r, but model-checking and prediction claims should cite M7r, not dummy M7.

6. **Rank/collinearity safeguards are uneven.**  
   Per-stop GLMs have a full-rank design check before fitting [`per_stop_rate_model.py:108`](../../../src/continuum_swr_replay/analysis/replay/per_stop_rate_model.py#L108). Per-bin GLMs rely on statsmodels exceptions and only check `reward_lag1` contrast [`per_bin_rate_model.py:114`](../../../src/continuum_swr_replay/analysis/replay/per_bin_rate_model.py#L114). The runner suppresses all warnings globally [`fit_per_bin_rate_models.py:70`](../../../scripts/fit_per_bin_rate_models.py#L70), so separation and convergence warnings are not persisted. M9 has visible fit loss in the artifact (`swr` only 182/218 sessions; `non_local_swr` 194/218), consistent with the documented separation risk.

7. **"Pooled non-local masks the dissociation" is defensible descriptively, but not yet a formal nested model comparison.**  
   The code fits separate outcome processes via `POPULATIONS` [`_helpers.py:1173`](../../../src/continuum_swr_replay/analysis/replay/_helpers.py#L1173), not one multitype point-process model with a population-by-reward interaction. The paired split contrast is strong (M7 `non_local_swr > non_local_no_swr` in 87.2% of sessions, 5/5 animal median differences), but the formal test should be a joint model or paired hierarchical contrast.

## Verified Behavior

- Per-stop outcomes are the canonical four count columns (`n_SWR`, `n_non_local`, `n_non_local_SWR`, `n_non_local_no_SWR`) in [`fit_per_stop_rate_models.py:52`](../../../scripts/fit_per_stop_rate_models.py#L52). Generated artifact check: 107,400 stops, 218 sessions, 0 additivity failures for `n_non_local_SWR + n_non_local_no_SWR == n_non_local`.
- Per-stop exposure is `log(immobility_time)` with zero/negative durations dropped [`per_stop_rate_model.py:85`](../../../src/continuum_swr_replay/analysis/replay/per_stop_rate_model.py#L85). Synthetic tests cover offset recovery, zero-duration exclusion, rank deficiency, and CV improvement.
- Per-bin exposure is one row per `(stop, bin)` with partial final-bin exposure [`_helpers.py:1127`](../../../src/continuum_swr_replay/analysis/replay/_helpers.py#L1127), and self-history shifts reset within stop [`_helpers.py:1156`](../../../src/continuum_swr_replay/analysis/replay/_helpers.py#L1156). Tests cover exposure, truncation, bin counts, and lag reset.
- Per-bin CV splits by whole `stop_id`, not individual bins [`per_bin_rate_model.py:194`](../../../src/continuum_swr_replay/analysis/replay/per_bin_rate_model.py#L194), which prevents within-stop adjacency leakage.
- `attach_is_swr_to_onsets` uses within-stop temporal rank to map non-local onsets to canonical event-table `is_swr` [`_helpers.py:993`](../../../src/continuum_swr_replay/analysis/replay/_helpers.py#L993). Real-session test checks onset split totals against stop-table split counts.
- GOF binarizes `count > 0` [`per_bin_rate_model.py:282`](../../../src/continuum_swr_replay/analysis/replay/per_bin_rate_model.py#L282). Artifact check found zero `count > 1` bins in the current 15 ms designs for all four populations, so this is currently safe.

## Statistical Risks

- Ordinary Poisson per-stop models report model-based SE/p-values under independent-stop and Poisson-variance assumptions. Given burstiness and repeated sessions per animal, use these as per-session coefficient estimators, not final inferential p-values.
- Per-bin GLMs handle event clustering through self-history and use time-rescaling GOF [`per_bin_rate_model.py:243`](../../../src/continuum_swr_replay/analysis/replay/per_bin_rate_model.py#L243), which is appropriate. Remaining issue: session drift and stop departure hazard are acknowledged but not modeled; the within-stop time curve at late `tc` is selected on long rewarded stops.
- AIC/CV comparisons are valid within Family B's shared design, but cross-family CV is explicitly non-comparable [`docs/rate_models.md:184`](../../../docs/rate_models.md#L184). Keep Family A for literature comparability and Family B for controlled conditional intensity.
- Lag 2-10 "shared suppression" is plausible from medians, but I did not find a formal equality/interaction test across populations. Treat "shared" as descriptive unless backed by a joint contrast.

## Test Integrity

Focused run:  
`PYTHONDONTWRITEBYTECODE=1 uv run pytest -p no:cacheprovider tests/unit/test_per_stop_rate_model.py tests/unit/test_per_bin_rate_model.py tests/unit/test_per_bin_rate_helpers.py tests/unit/test_fit_per_bin_rate_models.py tests/unit/test_stop_table_bout_start_no_leak.py tests/unit/test_recent_trial_reward_rate.py tests/unit/test_uncertainty_covariates.py`

Result: 57 passed, 1 failed. The failure is the slow uncertainty-rate smoke test: [`tests/unit/test_uncertainty_covariates.py:229`](../../../tests/unit/test_uncertainty_covariates.py#L229) expects `scripts/fit_uncertainty_rate.py` to expose `UNCERTAINTY_COVARIATES`, but that script only defines `VALUE_COVARIATES` and `CURATED_COVARIATES` [`fit_uncertainty_rate.py:72`](../../../scripts/fit_uncertainty_rate.py#L72), [`fit_uncertainty_rate.py:86`](../../../scripts/fit_uncertainty_rate.py#L86). This is outside the core Family A/B rate ladder but is a real test/API drift for the uncertainty extension.

I did not rerun full data-heavy model fitting. I read existing generated artifacts under `Processed-Data/replay_content/`.

## Next Checks

1. Fix or document the `TMAX / BIN_S` edge: use `ceil(tmax / bin_s)` and clip the last bin's exposure to `tmax`, or set/report `TMAX_EFFECTIVE=3.99`. Add a regression test for non-divisible `tmax`.
2. Refit per-bin models after the edge fix and update [`paper/claims_table.md:53`](../../../paper/claims_table.md#L53) from the unified 15 ms artifacts.
3. Make `Rlag` / per-bin `reward_lag1` the primary immediate-reward evidence; demote R1/R2 `is_after_omission` to a caveated literature-comparable positive control.
4. Add condition-number/VIF/separation summaries to per-bin fits and persist warnings/counts per `(session, outcome, model)`.
5. Add a joint multitype point-process or paired hierarchical model testing `population x reward_lag1` and equality of lag 2-10 kernels.
6. Add animal-level or clustered inference for the paper numbers, with sensitivity to bin width, `TMAX`, raised-cosine basis size, and stop-duration/departure-hazard selection.
