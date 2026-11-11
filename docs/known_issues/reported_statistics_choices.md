# Choices in the reported statistics that have not been decided

**Status:** open. Each entry is a choice the code makes today; none has been
changed, and none has been judged right or wrong.
**Found:** 2026-11-11, while the statistics moved from the summary builders
(`src/continuum_swr_replay/reported/`) into `analysis/`. The moves kept every
reported value bit for bit, so each choice below is as it was before the move.
**Severity:** not known per entry. An entry marked *current data: no effect* was
checked against the current data on 2026-11-11, as the entry says; the others
were read in the code and not measured.

Each entry was read in the code on 2026-11-11 and says where the choice is made
(file and function), what the code does, and what a different choice would
change. To find the values a function reaches, look up the builder function in
`paper/reported/questions.md` ("Builder function").

When an entry is decided, write the decision here (kept, with the reason; or
changed, with the commit) and do not delete the entry.

## Behavior summary

| # | Where | What the code does | What is open |
|---|---|---|---|
| B1 | `analysis/replay/rate_fit_statistics.py`: `reward_kernel_statistics` | The lag-1 reward kernel over sessions is the median of the sessions' rate ratios (`per_bin_design.reward_kernel_rate_ratios`: `median(exp(kernel))`). The animal-balanced value is `exp` of the median over animals of each animal's median log kernel. | The two orders agree when a median is one middle value and differ when it averages two. The two reported numbers are therefore not the same statistic at two levels. |
| B2 | `reported/behavior.py`: `power_values` (`power_sessions`) | The session count is every row of the SWR-power model's table. Each covariate's statistics (`coefficient_statistics.coefficient_statistics`) are over the sessions that have that coefficient. | A covariate missing in some sessions is summarized over fewer sessions than `power_sessions` states. |
| B3 | `reported/behavior.py`: `rate_contrast_separation_values` (`rate_contrast_term_separated_fraction`) | The share is over the fits of every population that carry `is_swr:reward_history_rcos_1`. The neighbouring values are over the `contrast` population's fits only. | Whether the share should be over the contrast fits only, as its neighbours are. |
| B4 | `analysis/replay/history_kernel_statistics.py`: `self_history_statistics`, `kernel_peak` | The counts of animals whose first lag is raised or suppressed use an unadjusted interval (mode ± 1.96 SE). The peak uses an interval widened for the number of lags (Bonferroni). | Two interval rules in one set of values. |
| B5 | `analysis/replay/history_kernel_statistics.py`: `self_history_statistics` | The pooled records are stored by population and animal. A second record for the same population and animal replaces the first with no error. | Whether a duplicate record should be refused. |
| B6 | `analysis/replay/learning.py`: `pooled_gee_role_summary` | A converged GEE fit with no p-value enters the Benjamini–Hochberg correction as p = 1. `scripts/behavior/run_stay_leave.py` does the same. | A missing p counted as a test changes the other roles' adjusted p-values. |
| B7 | `analysis/replay/learning.py`: `smallest_adjusted_p` | When two strata tie for the smallest adjusted p, the one first in alphabetical order is reported. | Only which stratum is named on a tie. |
| B8 | `analysis/replay/stop_statistics.py`: `still_immobile_fraction`, `onset_time_statistics` | "Still immobile at t" counts a stop whose duration equals t (`>=`). "Onset within the window" does not count an onset at the window's end (`<`). | The boundary rule differs between the two; it matters only for a value exactly on the boundary. |
| B9 | `reported/behavior.py`: `_population_ranges` | The descriptions of the range values are built by splitting the text of other values' descriptions. | Not a statistic. A change to the wording of those descriptions changes these silently. |

## Detection summary

| # | Where | What the code does | What is open |
|---|---|---|---|
| D1 | `analysis/hippocampus/memoryless_sequence/sweep_statistics.py`: `heldout_fold_statistics` | The lowest and highest fold rates are Python `min` and `max` over the folds in order; the spread is `numpy.ptp`. | With a fold whose rate is NaN, `min` and `max` depend on the folds' order and the spread is NaN, so the builder's "one rate for shared replays" refusal would not fire. *Current data: no effect on a reported value.* On the committed held-out summary the statistics whose lowest rate, highest rate or spread the builder reports or tests have no NaN fold. The place-less statistics do have a NaN fold (their spread is NaN), and the builder reads only their counts. |
| D2 | same file: `no_replay_statistics` | The immobile minutes and the number of recordings are read from the first matching row. | The function does not check that there is only one such row, as `controls_value` does. |
| D3 | same file: `carving_statistics` | The pooled two-part conditions are all read at the levels of the first condition. | Correct only while every pooled condition is run at the same levels. |
| D4 | `reported/detection.py`: `controls_values`; `scripts/figures/generate_memoryless_comparison.py`: `check_controls_claims` | The builder refuses the no-replay block when no output was run or an output reported an event. The figure script also requires one row per detector output. | Two refusals for one claim. |

## mPFC summary

| # | Where | What the code does | What is open |
|---|---|---|---|
| P1 | `analysis/pfc/population_statistics.py`: `called_fractions`, `called_fraction_by_animal` | The share is `n_called / n_tested` with whole numbers, so a test with no tested unit raises `ZeroDivisionError`. | Figure 4d and the `scripts/pfc/` reports call these functions; before 2026-11-11 their own copies drew a blank for such a test. *Current data: no effect* (every test has tested units). |
| P2 | `analysis/pfc/population_statistics.py`: `bh_called_fraction` | It refuses when the number of finite p-values differs from the tested count of `significant_fraction`, which counts the finite p-values. | The two counts come from the same rule, so the refusal cannot fire. |
| P3 | `analysis/pfc/population_statistics.py`: `prior_binding_percent`; `scripts/pfc/compare_pfc_cohorts.py`: `peak_shrinkage` | The function reads a missing `mle_converged` flag as not converged. The script reads a missing flag as converged, and a missing column as all converged. | *Current data: no effect seen*: the script's report and the summary's lowest and highest share agree to the digits the report prints. |
| P4 | `reported/mpfc.py`: `marked_family_values` | The marked-family summary is pivoted with pandas' default aggregation, the mean. | Two rows for the same model, event class, target and statistic would be averaged with no error. |
| P5 | `scripts/pfc/measure_pfc_null_geometry.py`, `scripts/pfc/compare_pfc_cohorts.py` | `--min-firing-rate` defaults to 1 Hz. The summary and Figure 4 use no firing-rate cutoff (`reported.mpfc.MIN_FIRING_RATE_HZ`). | A default run of either script reports another population than the summary's. |
| P6 | `analysis/pfc/timing.py`: `called_response_timing` and the functions beside it | They return milliseconds; the project's convention is seconds. | Kept because converting afterwards changed the last digits of some medians. The docstrings state the unit. |

## Every summary

| # | Where | What the code does | What is open |
|---|---|---|---|
| A1 | every `pd.read_csv` in `reported/` | CSV inputs are read with pandas' default float parser, which is not exact and differs by machine. | A summary is bit-identical only when rebuilt on the same kind of machine; see `TODO.md` ("Summary input paths and rebuild comparison"). |
