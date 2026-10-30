# Phase C review fixes: ledger

Branch `review-fixes-phase-c` (worktree `/cumulus/edeno/continuum-swr-replay-fix-phase-c`, off master `28e66ac9`).
The lead approved the fixes. Nothing is merged or pushed.

## Commits

| commit | what |
|---|---|
| `f0eff84f` | `build_baseline_columns` adds `well_unknown` after the well indicators, only when a fitted bin lies in a stop with no `occupancy_well_id` (a mixed stop). Bundle contract 3 in both marked runners, so a contract-2 bundle is never skipped as current. Design doc: the typical kernel is the response at the reference well. Tests: unit tests for the column; a planted test where mixed stops run at ×2 rate with denser events. With the column, the intercept and the typical kernels stay at their true values. Without it, both move. |
| `0b177f34` | The runner writes `neuron_index` = the kept units' recorded indices, not the row number. New slow, unmocked round trip: `run_analysis` on the planted simulation, with only the Spyglass loaders replaced. It reads the bundle back and checks the planted peak (argmax within ±1 bin), that the 2 silent units are dropped, `kept_unit_indices`, `neuron_index` and the `well_unknown` column. The planted strength test now requires argmax within ±1 bin, with one onset 0.4 bin off the grid. |
| `8637e8d7` | `compare_pfc_marked_family_trees.py` compares only classes that both trees fitted. Before this fix, a class the reference session had dropped crashed the script. Test added. |
| `6677e696` | Stale wording fixed. "Unpenalized" now reads "free in marked_glm; in the family, free unless baseline_penalty_sigma is given". The family module's no-shuffle sentence now describes the optional `slope_permutations` reference. In the summary docstring: the max-statistic p exists only when null draws exist, and the population kernel and its SEM are over animals. The summary now uses `PRE_ONSET_MAX_S` / `POST_ONSET_WINDOW_S`. The example commands now reproduce the shipped run. |

`stop_context` stays in `FOLLOWUP_MODEL_SPECS`, as the lead asked; it is not in `models.txt`.

## Tests

`uv run pytest tests/analysis/pfc tests/unit -q -k "marked or pfc"`: 637 passed, 5 skipped, in 17 min on CPU. Tests were red before the fix for: `well_unknown` names, the planted mixed-stop test, `neuron_index` (`[0,1,2,3,…]` vs `[0,1,2,4,…]`) and the compare crash (`FileNotFoundError`).

## Runs

| run | tmux | cards | logs | output | start | end | time / memory |
|---|---|---|---|---|---|---|---|
| smoke: j1620210710 12_r6, `replay_content` + `marginal_current_arm_fraction` | `pfc_c_well_unknown_smoke` | 1 | `logs/pfc_response_followups/marked_family_well_unknown_smoke/` | `Processed-Data/pfc_response_followups/marked_family_well_unknown_smoke/` | 2026-10-29 23:42 | 23:47 | 4 min 48 s (preparation 127 s, `replay_content` fit 95 s, marginal ≈ 40 s); RSS 9.5 GB |
| full refit: 75 sessions × 10 models (`models.txt`), `--model-set followup --event-set with_swr_without_replay --slope-permutations 0 --baseline-penalty-sigma 3`, same session partition as `marked_family_baseline_prior` | `pfc_c_well_unknown` (closed) | 1–9 | `logs/pfc_response_followups/marked_family_well_unknown/` (`commit.txt` = `6677e696`) | `Processed-Data/pfc_response_followups/marked_family_well_unknown/` | 2026-10-29 23:53 | 2026-10-30 03:06 | 2 h 40 min – 3 h 12 min per card (≈ 26 GPU-h); max RSS 13–23 GB; 9/9 EXIT 0 |
| summary | — | — | — | `…/marked_family_well_unknown/summary.parquet`; `.claude/research/pfc_response_followups_outputs/marked_family_well_unknown/report.{md,csv}`, `fig/` | 03:10 | | 1 min 38 s |

Summary command: `summarize_pfc_marked_family.py --bundle-dir Processed-Data/pfc_response_followups/marked_family_well_unknown --out .claude/research/pfc_response_followups_outputs/marked_family_well_unknown/report.md --figure-dir .claude/research/pfc_response_followups_outputs/marked_family_well_unknown/fig`. This is the same call as for `marked_family_baseline_prior`; only the paths differ.

## Results

- 750/750 bundles `complete`, all contract 3. No non-converged full fit.
- Units without a spike: 48 in 35 sessions, left out. Fitted units: 18,504. Both counts are the same as in the old tree.
- Non-finite held-out deltas are the same as in the old tree in every bundle (470 bundles; one example is a 1-spike unit in j16 12_r6).
- Mixed-stop bins: **all 75 sessions** carry `well_unknown`. Share of fitted bins: median 43.0 %, range 1.7–68.0 %.
- Old tree `neuron_index`: in the 35 sessions with a dropped unit it held the row number. No reader used it: summary and Part 4 do not, and the compare script pairs units by `kept_unit_indices`.
- Per-unit curves compared with `marked_family_baseline_prior` (`compare_pfc_marked_family_trees.py --min-rate 0`, `tree_comparison_vs_baseline_prior.csv`):
  - Slope curves: the median of each pair's median absolute difference is 0.0022 log rate, and the largest is 0.29.
  - Typical curves: the median of each pair's median absolute difference is 0.0056, and the largest is 0.77, mostly in low-rate units.
  - Held-out deltas: the median of the per-pair maximum absolute difference is 3.5.
- Values Part 4 reads, old tree `summary.parquet` → new tree `summary.parquet`. They were extracted with `reported.part4.marked_family_values`; the full table is in `quoted_values_old_vs_new.csv` in the research directory. **The movement is not a finding.**
  - Current-arm content slope peak: SWR replay −0.0305 → −0.0306 at 60 ms (lag unchanged); non-SWR −0.0268 → −0.0269 at 74 ms (lag unchanged). Same sign in 5/5 animals for both.
  - 0–150 ms means: largest change 2.7e-4 log rate.
  - Every quoted peak moves ≤ 2.7e-4. Peak lags move by ≤ 2 ms (one bin) in three ripple-power rows; the others are unchanged.
  - Animals with the sign: unchanged in every row.
  - Held-out gain shares: change ≤ 0.9 points. Examples: current arm non-SWR 33.1 → 34.0 %, SWR 35.4 → 35.1 %; MUA non-SWR 28.2 → 29.1 %; ripple power non-SWR 26.9 → 26.2 %.
  - Ripple–MUA correlation: unchanged (0.79 SWR, 0.65 non-SWR); it is a per-event quantity.
