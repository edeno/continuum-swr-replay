# Review of the reported-numbers code written on 2026-11-11

**What was reviewed:** everything written or moved on 2026-11-11 for the reported
numbers, at commits `9c8e06dc` and `7e785a78` of branch `questions-and-numbers`:
the statistics modules in `analysis/` for the events, detection, behavior and
mpfc summaries, their builders in `reported/`, the Figure 2 change, four pfc and
Figure 4 scripts, the spike-evidence launcher, and the machinery that carries a
value to the manuscript (`reported/values.py`, `code_record.py`, `release.py`,
`emit.py`, `formatting.py`, `scripts/reported/`).

**How:** seven reviewers, one part each, each with a fresh context and no
account of the code's quality. Each read its part in full, reworked the tests'
expected values by hand, and ran its own mutants on a scratch copy. None could
read the lab data.

**What was already known:** on the lab data, every reported value is bit for bit
what it was before the statistics moved. No reviewer found a value in today's
summaries that is computed wrongly. The findings are of four kinds: a number
that would be wrong or mislabelled on other data with no error; text that does
not say what the code does; code that is not clean; tests that cannot fail.

**Status of each finding** is at its end: `todo`, `FIXED <commit>`, `record` (an
undecided choice for `docs/known_issues/reported_statistics_choices.md`), or
`ask` (the data owner's decision). `HIGH` marks the ones to do first. A finding
is as its reviewer reported it unless it says it was confirmed afterwards.
Confirmed afterwards in the code or the summaries: L1 (fixed), M-1, E-1, T-2,
K-1, K-2, K-3.

Mutants killed, of those each reviewer ran:

| Part | Killed | Run |
|---|---|---|
| behavior: fit and rate statistics | 87 | 103 |
| behavior: stops, content, history, stay/leave | 93 | 117 |
| detection | 77 | 98 |
| mpfc: population and timing | 99 | 116 |
| pfc scripts and Figure 4 | 16 | 25 |
| events statistics, detector comparison, builder, Figure 2 | 65 | 143 |
| machinery | 62 | 73 or 83 (the report's total and its per-module table differ) |

Finding ids: L launcher, S scripts, D- detection, B- behavior fits and rates, T-
behavior stops and content, M- mpfc, E- events, K- machinery.

## R5: four pfc scripts + launcher (reported 15:12)
- L1 launcher says ok for failed epoch, always DONE/exit 0 -> FIXED d01ef44b, 7e785a78 (tests + cluster run).
- S3 part4_rate_threshold_sensitivity.py:269-276 loads "every unit" at MIN_FIRING_RATE_HZ (0.0 today); if raised,
  lower-gate rows silently wrong. Fix: load at 0.0 (or min(GATES.values())), drop the import. Docstring line 5
  ("with a 1 Hz gate") and funnel docstring (:167-170) untrue. todo
- S4 compare_pfc_cohorts.py:1359-1371 shrinkage_table override is an identity (same rule as row construction);
  delete the nested function, restore one-line percent. New test passes on old code. todo
- S5 compare_pfc_cohorts.py:1464-1477 heldout_fraction_table convoluted; call called_fractions per scope. todo
- S6 P1 (ZeroDivisionError) bites per animal / per band / non_positive_section, not only pooled; two conventions
  side by side (legacy_fdr_table, significance_table still blank; band_rows 5 classes raise, SWR_no_non_local NaN).
  Update record entry P1; decide one convention. todo
- S7 cohort_count_statistics (population_statistics.py:94-103) sums six columns in one DataFrame.sum(): one float
  column upcasts all totals -> "1000.0" in table. Fix: sum per column. todo
- S8 half-moved: session_cohort_table sums ripple/SWR totals itself (:362-367), build_report calls
  cohort_count_statistics twice (:2087-2100); excluded_section computes worst percent itself
  (measure_pfc_null_geometry.py:502); overlap_section keeps n_sessions/worst inline (:362-371); gate_counts copies
  a table for len (:484-486); rerun_sections builds a frame to sum 3 columns (:1946-1954). todo (judge each)
- S9 band_rows (part4:233-243, 258-267) pulls swr_without_replay_unit_statistics for 2 numbers; needs sessions
  table copied from builder (mpfc.py:1681-1690). todo (judge)
- S10 tests: 7/17 mutants survive in compare_pfc_cohorts: heldout fixture has every unit called (H1,H2,H4,H5);
  test_compare_pfc_cohorts.py:1058 substring "4 of 54" matches "54 of 54" -> assert "corpus 4 of 54"; total row
  nonlocal_kept_pct never asserted (C5); SWR kept % in prose not asserted (R1); Figure 4 alpha argument never
  varied (F3). measure_pfc_null_geometry.py and part4 script have no tests. todo

## R3: detection (reported 15:17) — sweep_statistics.py, event_table.py addition, reported/detection.py
- D-1 "shortest/longest" = whichever durations have rows (sweep_statistics.py:63-69, 124-150, 153-211;
  detection.py:534-545, 606-620); caption hard-codes 40 ms / 500 ms
  (paper/figures/memoryless_comparison/memoryless_comparison_caption.md:91-92). sweep_timing_statistics does not
  return its shortest duration; builder formats duration into a description itself (1000*..:.0f). NaN duration_s
  sorts last = "longest". Fix: return shortest_duration_s; refuse missing duration_s; builder refuses when
  durations differ between blocks / from the caption's; docstrings "durations with a row". todo
- D-2 pooled_heldout_rate (:504-517): pivot_table averages duplicates; class with events and NaN/absent rate
  counts as zero support; unknown fold/truth/estimator -> n_events 0 silently (place-less refusal passes
  vacuously); two .fillna(0) are dead code. Fix: pivot/unstack that raises on duplicates; refuse class with weight
  and no finite rate; check fold/truth/estimator exist; drop fillna. todo
- D-3 no test pins which statistic goes to which key in detection builder (9/9 wiring mutants survive). Cheap pin
  test: sweep_values/controls_values/heldout_values on the committed CSVs == committed JSON (76 values; one
  differs by the CSV-parser last digit -> use approx or round_trip). todo
- D-4 outside_window_call_statistics (event_table.py:694-698): plain `>` where detector uses detect._is_above
  (relative 1e-9 tolerance); NaN pieces dispersion counts as decided (undocumented, mutant survives);
  artifact-dropped candidates counted? Fix: use the detector's comparison (make public); state+test NaN rule. todo
- D-5 n_supported rebuilt by round(rate*n) (:564) instead of the numerator; banker's rounding 2.5->2; int() mutant
  survives though it changes fold A's loose count on real CSV. Name wrong for informative_share. Fix: return
  numerator via _whole_count and sum. todo
- D-6 folds passed as a string "AB" (detection.py:908-933; tests :465-496). Fix: _TILING_FOLDS = ("A","B"). todo
- D-7 burst_share_range (:339-344) no detector filter/no row check; lowest_estimator_agreement (:584-588) skips a
  NaN fold; no_replay_statistics counts a duplicated row as another output. Fix: one row-selecting helper that
  asserts the count. todo
- D-8 trials refusal compares totals only (detection.py:682-683) though message says "different multi-part
  trials". Fix: per condition. todo
- D-9 record: D1-D3 accurate; D1 fix = np.min/np.max + refusal on NaN spread. Missing entries: D-2, D-4, D-5, D-7,
  and ms units in sweep_timing_statistics (:210). todo
- D-10 small: `summary` names both the frame and the block (:219) -> `block`; numpy scalars vs dict[str,float]
  annotations; detection.py:969 recomputes shared["rate_min"]; pilot description hard-codes the epoch name
  (detection.py:278) though MEMORYLESS_PILOT_EPOCH exists; event_table tests in two places; B6/B7/B10 refusal
  mutants survive (trials refusal one direction; shared-rate refusal only fold B perturbed; recording refusal
  untested); state_evidence_values and recording_values have no test. todo
- Good, keep: controls_value's exactly-one-row check, _whole_count, test tables with decoy rows, refusal pattern.

## R1: behavior — coefficient_statistics.py, rate_fit_statistics.py, builder parts (reported 15:18)
Verdict: readable; tests mostly real; no arithmetic left in the builder functions reviewed.
- B-1 gated_coefficient_statistics (coefficient_statistics.py:162; callers behavior.py:429,711,828,930,994,2114,2210)
  reads gate.kept only: ignores gate.refusal (aggregate_contrast emits no row when refused,
  contrast_aggregation.py:607-625) and returns no n_dropped/n_unidentified. 10 sessions, 6 flagged -> n_sessions 4,
  median, p, no sign of the 6. Fix: return n_dropped, n_unidentified; raise or return refusal so builder can pend;
  or record as a choice. todo (was this so before the move? check removed lines)
- B-2 one row per session assumed: merges without validate="1:1" at coefficient_statistics.py:257
  (aic_difference_statistics), rate_fit_statistics.py:148 (baseline_rate_ratio_statistics), :288
  (paired_lag_one_kernel_statistics); only :227 validates. Fix: validate="1:1" on the three; shared check
  n_sessions(fits) == len(fits) in single-frame functions. todo
- B-3 group_range (coefficient_statistics.py:453) order-dependent with NaN; clock_shape_statistics with no usable
  curve returns time_peak_s 1.0, edge 1, sessions 0, clock_ks nan (rate_fit_statistics.py:449, 493-504);
  _population_ranges (behavior.py:1540) then NaN or silently over 3 populations. Fix: group_range raises on NaN;
  fitted_time_course raises / returns NaN peaks when n_curves == 0. todo
- B-4 ladder_fit_statistics (rate_fit_statistics.py:341,346; docstring :326,332): ks_band_fraction = mean of flag
  (denominator = fits with a flag) while separated_fraction/cv_not_finite_fraction divide by all fits; aic_median
  includes inf (test pins it, :330). Decide + document -> record. todo
- B-5 untrue text: coefficient_statistics.py:249-250 "finite difference" but .dropna() keeps inf;
  :159-160 significant_fraction denominator = sessions with a usable SE (falls back to model p when no SE);
  behavior.py:919-921 "read with summarize_per_bin_rate_models.py's rule" (script ungated, p only when n>10; builder
  gated); behavior.py:1760 "finite cv_ll_per_stop_mean" but known_fraction is notna (coefficient_statistics.py:298;
  per-bin twin uses isfinite); rate_fit_statistics.py:85-89 "populations differ in stops" compares totals only;
  behavior.py:1683 most_dropped_of "sessions fit for that term" = kept+dropped (no n_unidentified);
  behavior.py:412,420,531,539,547,629,653-654,958 descriptions hard-code "15 ms", "lags 3-10", "lag 10", "[0, 8] s",
  "[1, 3] s", "[1, 6] s" while code reads constants (changing descriptions changes summary text -> rebuild). todo
- B-6 fitted_time_course re-types the spline term (rate_fit_statistics.py:53,434,441): lower_bound=0.0 hard-coded,
  bare 401, new STOP_CLOCK_COLUMN="tc"; design has _SPLINE/_ARRIVAL_SPLINE (model_specs.py:538-541). Mutants on
  arrival upper bound survive (arrival basis tested only on a flat curve). Fix: build from the design's own term.
  Also: plot_per_bin_rate_summary.py:303 and summarize_per_bin_rate_models.py:190 draw/print the UNGATED kernel
  while the summary reports the gated one (rule differs, not only code). todo
- B-7 reward_kernel_statistics gates and reconstructs twice (rate_fit_statistics.py:245-247); dead: :205
  .dropna(subset=_REWARD_COEFFICIENTS), :437 np.isfinite(intercept), :493 nanmax==max. todo
- B-8 three rules for when a signed-rank p is reported: coefficient_statistics.py:134 always (1 session -> p=1.0);
  aggregate_contrast needs >5 (contrast_aggregation.py:633); summarize_per_bin_rate_models.py:98 needs >10. record.
- B-9 builder leftovers: behavior.py:617/:652 `window` reused; :1150-1156 in-transit shares recovered by scanning
  values for a key prefix (collect in a dict at :1045-1057); :1602 rung set as a regex passed into analysis
  (mutant match->contains survives); :2428 stray "# --- the summary ---" header; docs/regression_pipeline_map.md
  :1141-1165 new modules under "Unassigned" (write_pipeline_map.py needs a stage rule). todo
- B-10 weak tests: baseline rate test mean==median (rates 1..4) (:119); baseline ratio test (:140) unpaired ratio
  also gives 4; exp-median-ratio test (test_coefficient_statistics.py:185) log4/log2 = 4/2; KS distance bound loose
  (value 0.073869) (:453). No test of builder wiring (B3,B4,B5 mutants survive); _read_per_bin_fits any->all
  survives. Uncovered: NaN in group_range, no usable curve, duplicate rows in merges, inf AIC, missing
  ks_within_band, misaligned index in session_median_statistics (n_animals 0 silently), median exactly 0, gate that
  refuses, non-flat arrival curve, >=10 spline columns (text sort of basis columns!), empty frames. todo
- Record: B1,B2,B3,B9 accurate (B1 confirmed with numbers). Add B-1, B-4, B-8.
- Good, keep: session_median_statistics as the single base; validate="1:1" pattern at :227; reward-kernel tests
  from single-bump coefficients; frozen_cohort_statistics refusal.

## R4: mpfc — population_statistics.py, timing.py additions, mpfc builder (reported 15:19)
Verdict: readable, trusted on today's data; all hand-worked test values agree; 99/116 mutants killed. But the move
was a transliteration: duplication and inconsistent edge rules came along. No statistic left in the builder.
- M-1 pre-event level fractions clipped to [0,1] (timing.py:625-631, :670-672); builder descriptions
  (mpfc.py:1291-1292, :1416-1418) do not say clipped. A VALUE IN TODAY'S SUMMARY IS AT THE CLIP: mpfc.json key
  timing_population_control_pre_event_level_fraction_non_swr_replay_enhanced is exactly 0.0. Not cited in main.tex.
  Fix: signed ratio, or say "clipped" in descriptions; record. Mutant removing the clip survives (T14). todo/ask
- M-2 population_size_vs_control (timing.py:660-669): event mean over all chosen units, control mean over units
  with a finite control curve (one-sided filter); same in population_control_pre_event_level_fraction. Fix: both
  means over units with a control. Not verified whether any called unit lacks a control today. todo (check on data)
- M-3 two implementations of the 2x2 overlap: replay_class_overlap_statistics (population_statistics.py:510-617)
  vs swr_without_replay_unit_statistics (:840-934): empty one-only cell raises (odds_ratio :479-507) vs NaN
  (:900-901); NaN peak -> whole rank correlation NaN vs unit dropped; notna vs isfinite; different "called" set
  for abs_peak_median. The raise stops the whole summary (reached via rate_cutoff_check_values too). "Best-model
  share" has 3 denominators (:336 every unit; :645-647; :928-932 units with a best model). Tested/called mask
  written 6 times (:210, :305-306, :398, :556-561, :878-879, :971-972, :1053-1054); called_units unused in its
  module. Fix: one call_overlap_statistics; one best_model rule; one tested/called helper. todo (values must stay
  identical or change knowingly)
- M-4 swr_without_replay_timing_statistics (population_statistics.py:986-1081) duplicates called_response_timing
  (timing.py:616-638, 653-659): bit-identical on random input; differs only in min units (NaN below 2 vs computed
  from 1). Module docstring (:8-9) says timing lives in timing. Fix: one population_rise_timing in timing.py.
  Also peak_and_onset recomputed in loop (:1055); `if tested.any()` redundant (:973). todo
- M-5 half_window_s parameter (timing.py:499, 598-600) duplicates lag_time; wrong on another grid. Fix: test
  peak_lag == lag_time[0] or [-1], drop the parameter. todo
- M-6 untrue text: timing.py:25-28 "give lags in ms" but called_response_timing_by_animal returns onset_lag,
  rise_onset_lag, steepest_lag in seconds plus rise_onset_ms (record P6 incomplete; also omits
  swr_without_replay_timing_statistics); mpfc.py:1308,1338 "session resamples" = recording days (18 descriptions;
  44 others say "recording-day resamples"); mpfc.py:1498 "share of all units" = units with a usable control
  (timing.py:756-758); mpfc.py:1137-1138 "units called for SWR-coupled replay" = also tested for non-SWR replay
  (no effect today); population_statistics.py:550-554 Raises text; :3-6 "each function returns named numbers"
  (called_units returns a mask, units_at_or_above_rate a table: repeats loader's gate load_population.py:510);
  mpfc.py:2078 "# --- stop context (Q8)" cites a numbering. todo (description changes -> rebuild)
- M-7 two definitions of the mean curve's peak in one row: timing.py:627,663,671 use max/argmax; population_peak_ms
  and population_size use largest |value|. PopulationTiming.size docstring (:262-265) and mpfc.py:1387-1388 then
  untrue. Fix: one peak index. Not verified on real curves. todo
- M-8 swr_without_replay_event_statistics (population_statistics.py:689-741; builder mpfc.py:1686-1690): sums via
  a dict -> duplicate session deduplicated in n_events but not in denominator; missing count -> 0 via
  int(s.get(...) or 0). Fix: column sums + refuse duplicate sessions. todo
- M-9 no single rule for empty/missing group: called_fractions raises (P1) while bh_called_fraction,
  prior_binding_percent, called_without_improvement return NaN+warning; cohort_count_statistics:104,115 drops a
  0/0 session from the median silently; called_without_improvement:398-400 NaN improvement counts as improved;
  called_response_timing: zero/NaN curve unit in neither direction; steepest_median drops units peaking at first
  lag. Fix: state each; one rule. record/todo
- M-10 small: best_model_fractions(column=...) unused option (:315); "SWR_no_non_local: p_heldout" and two peak
  column names hard-coded 3 times (:841-844, 967, 1041), missing from HELDOUT_P_COLUMNS; FutureWarning in
  prior_binding_percent (tests silence); onset_before_onset_fraction naming; three loop idioms in
  called_response_timing (timing.py:632-659); timing_values calls called_response_timing_by_animal twice;
  generate_figure04.py:453 writes 1/(n_control_draws+1) instead of empirical_pvalue_floor; mpfc.py:2018
  row.get("post_onset_mean", np.nan) where neighbours use row[...]. todo
- M-11 weak tests: BH test input gives 0.25 under BH/BY/Bonferroni (test_population_statistics.py:148; use
  p=[0.01,0.02,0.03,0.9]); odds ratio (a*b)/(c*d) survives (b==d in both tables); CI low/high swap survives
  (low==high); resampling by epoch vs day survives; n_boot not passed survives (bootstrap untested); `<0` vs `<=0`
  for peak/onset/animal counts (no unit at lag 0); edge counted on right only; resample by animals vs groups
  (DAYS and ANIMALS coincide, test_timing_statistics.py:65-66); clip removal; argmax(abs). todo
- Good, keep: fixtures with comment tables (step/ramp curves, twelve-unit overlap with hand ranks); DIRECTIONS;
  odds_ratio error message; called_fractions delegating to significant_fraction.

## R2: behavior — stop/content/history_kernel statistics, learning additions, builder parts (reported 15:20)
Verdict: readable, the move from the builder was faithful (compared with 1e3593e8); no arithmetic left in the nine
builder functions. Not yet trustworthy on other data.
- T-1 trunk_shares (content_statistics.py:68-76): share.where(on_trunk, 0.0) + groupby.sum() -> an event with NaN
  arm_fraction becomes trunk share 0 and enters the animal mean (upstream contract: undefined is NaN, never zero,
  analysis/replay/summaries.py:441). Events dropped silently (unknown is_swr, not exactly one current arm), count
  not returned. Not verified whether today's table has such events. Fix: drop/refuse NaN arm_fraction; return the
  count left out. todo (check on data)
- T-2 history peak statistics have no denominator (history_kernel_statistics.py:188-222; behavior.py:2458-2460,
  2483-2507, 2559-2600): ends/onsets only over animals with a measurable lag; lengths not returned; no onset kernel
  at all -> NaN+warning while first lag raises. Builder docstring says descriptions give how many were kept (true
  for first lag and cross, not the six peak values). TODAY: behavior.json history_onset_animals_non_local = 4,
  described without "of N". Fix: return n_end_peaks, n_onset_peaks; put in descriptions; raise when none. todo
- T-3 trunk_share_statistics / current_patch_trunk_fraction (content_statistics.py:114-120, 141-146): n_animals
  counts animals lacking one class; medians over different animals; builder writes "median over the N animals".
  Fix: raise when an animal lacks a class. todo
- T-4 still_immobile_fraction and dwell (stop_statistics.py:98-113, 68-95; descriptions behavior.py:1261, 1309) use
  immobility_duration_s = span less excluded time (tables_stop.py:70-73), described as "still immobile t after
  they began"; denominator includes zero-duration bouts. Fix: say "analysed immobility (excluded time removed)" or
  compute from stop_end - stop_start. record/ask (size not measured)
- T-5 outcome_boundary_statistics (stop_statistics.py:116-177): "shifted" count built without at_risk_events while
  shipped fits pass them (scripts/rate/fit_per_bin_rate_models.py:223-235) -> an upper bound; rule re-derived
  (ceil(delta/BIN_S) vs per_bin_design.py:939-944; differ on exact bin edges). Fix: say so in docstring and
  description; better: aggregate_to_timebins reports shifted stops. record/todo
- T-6 lag_index (history_kernel_statistics.py:53-66): lag_index(5) or (0) -> -1 (reads last lag silently);
  lag_index(40) -> 2. Fix: raise unless positive whole multiple of the bin. pooled_gee_role_summary
  (learning.py:740-761): a stratum with no converged role raises ZeroDivisionError from statsmodels (neighbours
  guard). smallest_adjusted_p with nothing finite: "min() arg is an empty sequence". todo
- T-7 "measurable" means two things: module docstring (history_kernel_statistics.py:10-14) = support>=3 and SE<=0.5;
  cross_history_statistics (:326-333) applies support only; builder docstring (behavior.py:2454-2459) says cross
  kernel is Mselfcross_dur under both rules; code reads Mselfcross_long_dur under support only (predates the move).
  Fix docstrings; record. todo
- T-8 outcome_boundary_statistics reads the SWR source via aggregate_to_timebins default (test monkeypatches
  per_bin_design.excluded_intervals_by_session): take excluded_intervals param. Returns totals only, so
  scripts/qa/compare_fits.py:742-812 keeps a line-for-line copy: return one row per stop. Dead code :163-168
  (errstate, nan_to_num), :152 fillna(False). todo
- T-9 small: content_statistics.py:23-27 PATCH_TRUNK via well mapping (data_loaders.constants.TRUNK_ARMS exists);
  :64,73 on_trunk & on_current == on_current; :164 event_length_statistics calls class_medians with dummy "lower"
  and needs `animal` only for that; :166 median_bins includes unknown-is_swr events the class medians exclude;
  stop_statistics.py:85 redundant dropna; history_kernel_statistics.py:104,197 mask named "ok" like the fit flag;
  learning.py:789-802 "one row per attempted fit" untrue (skipped sessions have rows; description "logistic fits
  attempted" behavior.py:2296); behavior.py:2362 "mixed stops (every one without an is_stay)" unchecked claim (true
  today: stops_mixed_with_stay = 0); :2472 compares fit outcomes to key words; :1320, 2311 stop_cohort_counts called
  in two builder functions (two function names for one statistic); _arm_table() pasted in two test files. todo
- T-10 record: B5 other half (cross_history_statistics counts a duplicate record twice); B6 other half
  (cross_session_role_summary leaves a missing p out; GEE rule counts it as 1; both feed smallest_adjusted_p).
  New: support-only rule for cross kernel; lags in ms; median lag over an even number of animals falls between
  lags (37.5 ms printed 38); length_share unbounded (1.2). record
- T-11 weak tests: no test anywhere for stop_table_values, stay_leave_values, breadth_derived_values,
  cohort_audit_values, content_contrast_values, estimator_values (wiring mutants survive: rewarded/omitted swapped,
  GEE q reads sign-rank q, total/length-adjusted swapped); history fixture has identical populations and blocks
  (low reads high survives; onset count reads end-locked count; cross read from self-only rung); Bonferroni
  0.05/m vs 0.025/m survives; _WALD_Z 1.96 vs 1.645 survives; onset share / n_stops survives (5 stops and 5
  onsets); boundary `<` vs `<=` and last-bin `>=` vs `>` survive; excluded time untested for the boundary statistic.
- Good, keep: exact move; small tables with working in comments; loud refusals naming the animal; kernel_peak's
  docstring; reward_history_incomplete_statistics failing on duplicate stop_id.

## R6: events — event_statistics.py, nonlocal_comparison.py, events builder, Figure 2 (reported 15:33)
Verdict: readable; moved code matches the removed builder code line for line; no statistic left in the builder.
Tests do NOT earn "checked by hand": about half of the mutants survive (event_statistics 39/76 survived,
nonlocal_comparison 17/36, builder 16/20, figure 6/11). Figure 2 does draw b, d from the table with the summary's
functions; a stale cache cannot bypass the refusal.
- E-1 event_signal_statistics (event_statistics.py:244-247 docstring vs :265-269 code): per-animal running/rest
  levels are medians over EVENT ROWS joined to the reference (event-weighted), docstring says "median of their
  epochs' levels". The one place where the new docstring says something the removed code did not. Example gives
  n_animals_below_running 0 vs 1. Fix: decide; correct docstring + builder descriptions (events.py:643-652) or
  compute from reference (changes animal counts). record + ask
- E-2 overlap_class_statistics (:109-112, 119): value_counts(normalize).unstack() leaves NaN for a class an animal
  lacks; .min() skips it (0.5 instead of 0.0; dtype-dependent). A NaN class row is in n_events but in no class.
  Fix: unstack(fill_value=0); raise when the three counts do not sum to the total. todo
- E-3 spike_evidence_statistics (:406-427): epoch/animal with no baseline -> NaN > False, still in n_scored; a
  wrong baseline name -> all zeros, no error. Fix: raise when baseline group empty or an epoch/animal lacks one.
  todo (check on data whether any epoch lacks a baseline)
- E-4 Figure 2 refusal (generate_figure02.py:410-420) compares only four class counts though message says
  "different immobile non-local events"; panel d uses boundary_offsets(events) without the immobility filter while
  panel b's counts filter whole_event_speed_eligible (summary refuses such a table). Fix: compare a digest of
  sorted (event_id, overlap_class); refuse a non-immobile row in event_table_panels. todo
- E-5 numbers printed on panels b, d are recomputed and rounded in the plotters (figure_panels.py:1031-1045,
  :1155-1176: np.median, f"{:+.0f}") — half-to-even vs reported.formatting half-up; docstring
  generate_figure02.py:33 "the panels show the numbers the text cites" not guaranteed for printed digits. Fix: pass
  medians from boundary_offset_statistics and class shares formatted by the reported formatter. todo
- E-6 two NaN rules for theta-like (:355-360): event_signal_statistics drops an event with no theta;
  rest_matched_theta_statistics counts it as not theta-like; rest_windows NaN passes the "no window" refusal.
  Fix: refuse NaN in theta_z, rest_windows, rest_theta_like_share. todo
- E-7 event_signal_statistics checks only theta's reference (:259-260 hard-coded theta_running_median though
  signals is a parameter); rest reference never checked. Fix: check <signal>_running_median and _rest_median for
  every signal. todo
- E-8 smaller silent cases: threshold_step_statistics (:513) n_fits == n_animals counts rows (use nunique);
  peak_ripple_mode_statistics (:457-464) no animal-count check, NaN dip_p counts as rejection; class_medians (:558)
  any direction other than "lower" = higher (raise on unknown); _weighted_shares (:606-612) NaN share drops from
  numerator but stays in denominator. todo
- E-9 left_out_statistics (nonlocal_comparison.py:230-255): per-hour rates = counts over known-membership events /
  whole exposure; time fractions use every compared event. record (+ no events section in the record at all;
  add E-1 and strict-vs-inclusive boundaries).
- E-10 untrue text left by the Figure 2 change: figure_panels.py:1087-1091; generate_figure02.py:300-303 "Single
  classification source for panels a/b/d"; tests/analysis/hippocampus/test_classified_event_table.py:500-530
  _timing_offsets "Mirror the inline panel-d computation" (code gone; tests test only the mirror), :440;
  generate_figure02.py:143 cites "task 26b"; analysis/hippocampus/__init__.py:72-74 heading. todo
- E-11 unclean: rename start_offset_s->start_diff only to suit the plotter (generate_figure02.py:418-420; change
  the plotter); classes tallied twice (:306-319, :354); mixed import path (:84-93); "reality" names in events.py
  (reality_dir, REALITY_METRICS, ...); REALITY_EPOCHS=242 reused as the size of four unrelated inputs
  (events.py:1389,1438,1475), n_animals=5 hard-coded (:999); EVENT_FEATURES_RUN defined 3 times; closure in loop
  (:771-773); literal session keys (:1442-1443); summarize is 210 lines (:1374-1586); class_medians does two jobs
  (content_statistics.py:164 calls with dummy "lower"); threshold_step_statistics half-parametrized; n_epochs
  (:122) repeats tables.session_keys.n_sessions; no_nonlocal_fraction_animal_min = 1 - matched.max()
  (nonlocal_comparison.py:135-136); I/O mixed with statistics in nonlocal_comparison (left_out_seconds returns
  zeros when no features files); module docstring vs return types (most_steps tuple, int, per-event table). todo
- E-12 weak tests: symmetric inputs, two-value groups (mean==median), no ties; not asserted: tetrodes_median,
  no-spike/uncertain medians, Widloski fraction/rate, kay_time_s, outside-seconds keys; no test:
  other_detector_events, check_against_directional, every builder function's wiring (stale output not pended,
  gamma staleness, missing-row refusal all survive); generate_figure untested (refusal-against-itself, cache
  without counts, version left at 2 survive; version test asserts only >= 2); misleading test names
  (weigh_windows_by_event; "is left out" though removing dropna survives). The spike-evidence test with deliberate
  ties is the model. todo

## R7: machinery — values.py, code_record.py, release.py, emit.py, formatting.py, scripts/reported (reported 15:38)
Verdict: main path followable; cannot yet be trusted to enforce its two headline rules.
- K-1 canonical summary written from UNCOMMITTED REPOSITORY DATA: committed-code rule covers loaded modules, the
  script, non-Python files under the package, uv.lock; NOT data files builders read elsewhere in the repo
  (paper/tables/table_s1_literature_parameters.csv for survey; docs/design/replay_detection_comparison/*.csv|json
  for detection, detection.py:94-97). Recorded as inputs; --code-only skips every input even tracked ones in the
  checkout. Reproduced: drop 5 CSV rows uncommitted -> canonical survey.json with clean label, 6/32 values differ;
  restore CSV -> check_release --code-only says current. Manuscript cites 29 of 32 SURVEY values. Fix: treat any
  recorded input under the repository like a build file (tracked, equal to HEAD); --code-only skips only inputs
  outside the repo. (summarize.py:59-71, code_record.py:120-147) todo HIGH
- K-2 assume-unchanged / skip-worktree files pass as committed (code_record.py:197-202: git ls-files + git diff
  honour index flags). Fix: compare content: git ls-tree -r HEAD blob ids vs git hash-object of working files
  (with -z; also fixes non-ASCII names). todo HIGH
- K-3 a script or module OUTSIDE the checkout is dropped from the record silently (code_record.py:146-147), and an
  EMPTY record is written (summarize.py:78-84): test_summarize_writes_no_canonical_summary_from_uncommitted_code
  passes at its last step with "a code record of 0 files". Fix: build_files raises when a loaded package module or
  the running script is outside repo; summarize refuses an empty record; test needs a tiny package inside its
  repo. todo HIGH
- K-4 formatting.py has NO direct test (no test calls `formatted`); 8/10 mutants killed only by
  test_emitted_files_hold_the_committed_summaries, which passes again after `make reported-values`. Same for macro
  names, LaTeX style, marker @style, cell escaping, by-figure view. Fix: table-driven test of formatted, one row
  per docstring sentence (probe rows: 0.0995->0.10, 9.95->10, 99.5->100, 1234->1200, -0.004 dec2->0.00,
  37.5 int->38, 9.96e-5 pval->1.0e-4, 0.00099996 pval->0.0010, LaTeX 75{,}675, \ensuremath{-}0.071). Decide: pct
  of 0.004 prints "0 %", 0.996 "100 %"; p exactly 0 prints "0"; @int on a fraction prints 1; negative pval raises
  math domain error. todo HIGH
- K-5 stale-input rule is neither one rule nor "never refused" (docs/development.md:447-450):
  a) refusals: events.py:1389-1401 (extras not covering every event), :1438, :1454, :1475;
     behavior.py:2756-2768 (_read_per_bin_fits "refit it"); mpfc.py:331, :385; fit_staleness crashes
     (FileNotFoundError/KeyError) on odd sidecars. Fix: narrow the sentence or make these pend.
  b) fit_staleness passes: sidecar with inputs {} -> None (event_table_staleness treats "records none" as stale);
     replaced artifact bytes with same sidecar -> None.
  c) no staleness check in events.summarize for spike-evidence metrics (:1538), heldout_by_swr (:1551),
     content_occupancy (:1561), event-features run (:1497-1536).
  d) pending selected by key prefix (events.py:1411, :1467).
  e) marked_family_staleness uses file mtimes (mpfc.py:390-399).
  f) a fit's own git_hash "-dirty" is never read by any builder.
  Fix for one rule: one function in values.py stale_inputs(recorded, current, what). todo/ask
- K-6 release check does not look at reported_values.tex, the PDF, or \input files: hand-edited macro 300->999
  still "current" (only the unit test catches; make release-check does not run it); cited_macros reads main.tex
  only (main.tex:165 \inputs tables/table_s1_literature.tex, 0 macros today). Fix: check_release asserts
  tex_file(summaries) == reported_values.tex and reads the .tex files main.tex inputs. todo HIGH
- K-7 trial build with an uncommitted lazily-imported module can never be written; message untrue ("code changed
  while built") (summarize.py:74-82): before-record lacks the module. Fix: compare before["head"] with
  after["head"]; set summary.commit from the after-record. todo
- K-8 pending="" is pending in one place, not in three (values.py:119,201,286-293; release.py:74; emit.py:110,198;
  summarize.py:85). Fix: Value.__post_init__ rejects empty pending. todo
- K-9 stale docstrings: check_release.py:9-12 (three removed states); summarize.py:17-18 ("only the requested
  builder is imported": events imports reported.detection, so a detection-builder edit stales events);
  code_record.py:44-46 NOT_RECORDED; emit.py:16-17 marker claim (unmatched markers left alone silently);
  values.py:150; Summary docstring; test_reported_code_record.py:5. todo
- K-10 untested: Value validation (digits in key, NaN/inf, fraction outside [0,1], no question, non-identifier
  function, unknown kind/style), duplicate key overwrite in Summary.add, files_record hashing names only. todo
- K-11 more machinery than needed: dead condition code_record.py:240-241 (parses), dead branch :270
  (is_absolute); one fact stored four ways (uncommitted, -dirty label, "Code record" column, staleness text);
  Summary.commit and produced_by threaded through five builder signatures (set in summarize.py); duplicates:
  code_record._git vs runs.provenance.git; sha256 in four places vs runs.io.sha256_file; rel() defined identically
  in four builders beside values._shown; SUMMARY_DIR in three scripts, MANUSCRIPT in two; two table renderers in
  emit.questions_view (:188-208, :290-305); input_changes lives in code_record.py but only release.py uses it;
  mpfc.staleness vs release.staleness name clash. Record breadth 17-71 files -> any shared-module refactor stales
  every summary (trains people to ignore the check). todo/ask
- K-12 small: gitignored .DS_Store under src/continuum_swr_replay blocks every canonical build (record only tracked
  non-Python files); non-ASCII filename (no -z); --output-dir Paper/Reported on macOS writes a -dirty summary into
  paper/reported (use samefile); newline in a description/pending escapes the TeX comment (emit.py:111-115);
  release check never looks at recorded head (amend); spike-evidence input record path reads as one epoch's file
  and the digest hashes path.name (identical for all 242) -> THE DIGEST OF THE SET hashes names+contents? check
  files_record; pending value filled into a markdown marker carries no flag; committed-code rule is in
  docs/reproduce.md, not in the guide's rules. todo
- Sound: docstring-insensitive digest; macro names injective; citation over-counts are in the safe direction;
  staged edits caught. Good, keep: cited read from main.tex; pend; Summary.write/read asymmetry; emitter fills in
  memory first; throwaway-repo tests.
