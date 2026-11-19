# Codebase review, 2026-10-31 (master 9f26ca1e)

Read-only review of the code behind `paper/reported/part{1-4}.json`, from three reviewers:
correctness (code-review skill), statistical practice, and organization/clarity. The findings
were then checked by the main session.

Verification labels:
- **V**: checked by the main session against code or the live artifacts.
- **R**: checked by the reviewer on live artifacts, not re-checked.
- **U**: an unverified lead.

Freshness at review time: all summary inputs match their recorded sha256, all 143 fits are
fresh (`scripts/qa/check_fit_provenance.py`), and 0 values are pending.

## Headline numbers re-checked

Every headline survives the re-checks below. The defects are in how the numbers are gated,
tested and described.

## A. Statistics and reporting (ranked)

1. **The p and q values treat epochs as independent replicates (V).**
   - What happens:
     - `contrast_aggregation.py:633` (signed-rank over per-epoch coefficients, then BH),
       `reported/part3.py:384-405` (`_paired_immediate`) and `_immediate_session` test about
       240 epochs. These come from 45 recording days and 5 animals.
     - The overview quotes `part3.rate_paired_p` = 1.1e-19 and `part3.armuncert_all_q` =
       5.4e-27 with no caveat.
     - The same paired test aggregated to days gives p = 1.4e-12 (R).
     - "Sessions" in the JSON descriptions means epochs.
   - Fix: give a day- or animal-level statistic (sign-flip or permutation over days, or a
     mixed model) next to every quoted p or q, and say "epochs".

2. **The "immediate reward" rate ratio is a basis coefficient, not an effect size (V).**
   - What happens:
     - `rate_immediate_session_non_local_swr` = 9.9× is exp of the coefficient on
       `reward_history_rcos_0`, the first of three raised-cosine basis functions. It is shown
       in bold in `docs/per_bin_rate_results.md:145`.
     - Across sessions the three basis coefficients correlate −0.95 and +0.89 (rcos_0 with
       rcos_1 and rcos_2); their medians are +2.29, −5.04 and +2.71.
     - The reconstructed kernel at lag 1 is 1.51×. By lag it runs 1.51, 1.07, 0.94, 0.89,
       0.89, 0.91, 0.93, 0.98, 1.02, 1.07.
   - Fix: quote the reconstructed kernel (at lag 1, or its integral) as the effect size and
     drop exp(rcos_0) from the prose. A paired test on the lag-1 kernel gives p of the same
     order (R).

3. **The rate headline values skip the separation and convergence gate that other values use (V).**
   - What happens:
     - In `M7r_time_histcos_rew` (non_local_swr), 118 of 242 epoch fits are flagged
       separated, 86 with `rcos_0` separated, and 83 are not converged. non_local_no_swr
       shows 103, 81 and 70.
     - `_time_course` drops separated fits. `_lag_one_kernel`, `_paired_immediate`,
       `_immediate_session` and `_immediate_animal` (`part3.py:372-430`) keep all of them.
   - Effect of gating on separated or non-converged reward terms:
     - The paired test keeps 107 of 240 pairs, still 74 % positive, 5/5 animals,
       p 4.8e-10.
     - The lag-1 kernel moves from 1.51 to 1.43 (SWR-coupled) and from 1.012 to 1.011
       (non-SWR).
   - "Non-SWR events barely respond to reward" depends on the model. M1_rew gives lag-1
     ratios of 2.51 against 1.55 (R). The SWR/non-SWR ratio of about 1.5 is stable across
     models; the absolute "no response" is not.
   - The per-bin fits' model SEs are not clustered by stop. This is the about 30×
     pseudo-replication from the 2026-09-03 review, so the rate table's `pct_sig_session`
     is inflated and its SE is not comparable with the content table's clustered SE (U).
   - Fix: one inclusion rule for every rate value; report the gated and ungated values side
     by side.

4. **The "direct" (length-controlled) content estimand is described too strongly (V).**
   - What happens:
     - `model_specs.py:1437-1444`: event length enters only as z(log n_bins) × current arm
       and × previous arm.
     - The swr × reward, value and uncertainty terms get no length interaction.
     - The overview says "once event length is held fixed".
   - The Part 3 Reading lists "SWR avoidance of recently rewarded arms" as robust. That is
     supported only for the total effect: `swr_armreward_total` = −0.39, 1/5 animals
     positive. The direct effect is `swr_armreward_direct` = −0.18, q 0.72, 2/5. The
     within-SWR fit `armreward_swr` = −0.26 (0/5) is a different estimand.
   - Fix: describe it as "adjusted for length × current/previous arm", call the reward
     avoidance a total effect, and add a sensitivity rung with length × covariate terms.

5. **The Part 2 nested-bootstrap intervals are optimistic with 5 top-level units (R).**
   - Animal-level t intervals are wider. Extent is 22–64 cm against 29–58 cm; matched speed
     is 70–264 cm/s against 107–250 cm/s.
   - Signs hold in 5/5 animals except HPD.
   - Matched differences are driven by the longest-duration strata. For extent the stratum
     differences run from about +2 cm (shortest) to about +165 cm (longest, open-ended).
   - Fix: give an animal-level interval or a leave-one-animal-out range next to each
     bootstrap interval.

6. **The Part 4 `marked_*_animals_same_sign` counts carry no information (R).**
   - All 16 are 4/5 or 5/5, including slopes the text treats as null.
   - The sign is read at a lag chosen from the population peak, and 4/5 or better occurs
     about 19 % of the time by chance before that selection.
   - Fix: drop the counts, or use a per-animal permutation.

7. **Winner's curse in the history peaks (R).**
   - `part3.py:~2669` `_peak` takes the argmax over lags and then a Wald interval with no
     adjustment.
   - The end-locked peaks are safe (z up to 7). The small onset-locked non-SWR peaks are not
     (wilbur z 2.9, j16 1.7).

8. **Limitations to state more prominently (R):**
   - The held-out tetrode windows are labeled by the all-tetrode decode.
   - `rate_baseline_ratio` 0.98, p 0.30, is read as "similar" with no equivalence test.
   - "SWR-coupled" is defined by overlap with an SWR, so reward gating may be SWR gating
     (the `swr` lag-1 kernel is 1.38).
   - PLAUSIBLE: the PFC matched controls are uniform within a bout, not matched on time
     since stop start.

## B. Code correctness

9. **AIC counts unestimated terms (V; no reported effect).**
   - `analysis/replay/fitting.py:905` sets `n_params = len(predictors)`, including
     `no_within_event_variation` terms.
   - In the CS5_armvalue / C2_base_swr_leaf comparison only 4 epochs per estimator carry
     such a term, and they carry it in both rungs, so ΔAIC is unaffected.
   - Fix: count only estimated terms.
10. **`llf_null` counts events the fitted likelihood omits (U).** In `fitting.py:~920`,
    events with a constant response are dropped from `llf` but kept in `llf_null`, which
    inflates pseudo-R² and ΔAIC against the null rung. Check how many events this touches.
11. **The odds ratio has no zero guard (V; latent).** `reported/part4.py:1033`: an animal
    with an empty cell would raise ZeroDivisionError or report 0. Not triggered now (the
    reported range is 2.7–5.7).
12. **The last spike-count bin is degenerate (V; low).** `intervals.py:603` keeps spikes
    ≤ `time[-1]`, so the last bin holds only spikes at exactly `time[-1]`. This is one bin
    per call; it matters only for short peri-event windows (`visualization/glm_plots.py:224`).
13. **Bout exposure and NaN speed (U).** `analysis/replay/bouts.py:134-143`: exposure is one
    sample short, and a NaN speed sample splits a bout (`NaN < threshold` is False).
14. **Per-stop cross-validation drops NaN rows per rung (V; low).**
    `per_stop_rate_model.py:255`: unlike the fitter, it doesn't require a frozen design.
    The only caller passes the frozen table.

## C. Organization and clarity (R unless marked)

15. **Reported keys can't be traced to the code that computes them.**
    - Keys are built by f-strings over dicts, and `Value` has no function reference.
    - Fix: add `computed_by` and print it in `values_index.md`.
16. **`source` fields hold absolute `/cumulus/...` paths (V):** part3 935, part4 615,
    part2 191. Fix: write them relative to the data root.
17. **Analysis rules live in the reporting layer, and the PFC scoring lives in a script.**
    - The history measurability gate is only in `part3.py:2619-2676`.
    - `part4.null_geometry_values` re-implements a script.
    - PFC held-out p-values are computed in `scripts/pfc/run_pfc_analysis_bandit.py`
      (`score_session`, `run_analysis` 493 lines), and their column names are prose strings
      retyped in 4 places.
    - The cohort firing-rate cut is `MIN_FIRING_RATE_HZ = 0.0` in `reported/part4.py:88`,
      but `load_population_data(min_firing_rate=1.0)` (V).
18. **The same population has about six names:** `non_local_no_swr`, `non_swr`,
    `non_swr_replay`, `nonSWR`, `non_SWR`, `non_local_no_SWR`.
19. **Duplication and size.**
    - `_value_factory` and `rel()` are copied in all four `reported/part*.py`, and the
      copies have drifted.
    - `_SPLIT_RUNGS` / `_POOLED_RUNGS` retype `model_specs` key terms.
    - Functions of 300–520 lines: `part4.timing_values`, `part3.per_bin_values`,
      `per_bin_design.aggregate_to_timebins` (17 params), `pfc/session.process_neuron`
      (17 args, 6-tuple).
20. **Smells:**
    - Module globals reassigned by `run_prepare_data` (`prepare_data.py:102-115`).
    - A comment contradicts the code (`prepare_data.py:428-432`).
    - 25 imports of private `_` names across modules.
    - `learning.py:494,636` catch every exception.
    - Unused builder parameters (`build_event_arm_table(trials)` and others).
21. **The layering is not documented for `reported/`.** It is missing from the Code layout
    and from the layering tests.

## D. Checked and sound

- Content fits: 12,638/12,638 converged, with real Newton decrements.
- The per-term gates drop 0–7 epochs per headline term.
- Rate offsets are log(exposure), and rungs are fit on one frozen cohort.
- PFC matched controls use a fixed jointly matchable subset (0.27–0.6 % unmatchable).
- The PFC overlap odds ratio is robust to stratification (Mantel-Haenszel by session 3.21;
  by session × rate tercile 2.70).
- The interval primitives, the reward-kernel reconstruction, the spline rebuild and the
  post-outcome masking in the rate contrast are sound.
- The import-layering tests pass with no allowlist; `src` never imports `scripts`.
- `ruff check` is clean.
