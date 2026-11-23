# Appendix: scratch prototypes and outside references

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

## Scratch map

The prototypes, checks and explorations the design cites are scratch, ignored by git, in the MAIN checkout:

    S = /cumulus/edeno/continuum-swr-replay/.claude/research/2026-11-20-regression-prototypes/

They are not kept. Read them for method; import nothing from them. Each finding a phase depends on becomes a
test (in the suite) or a stored analysis (a committed script that writes an output under Processed-Data and
that a summary or a figure can read). Some of the scripts read intermediates from an old session's
scratchpad under `/tmp/claude-39494/...`; those intermediates may be gone, so rebuild the inputs from the
canonical tables, never from those paths.

| Finding (design section) | Scratch to start from | Becomes | Phase |
|---|---|---|---|
| P-spline Poisson REML reproduces mgcv (2.4) | `S/proto_basic/pspline.py`, `S/proto_basic/fit_mgcv.R`, `S/proto_basic/agreement_detail.py`, `S/proto_basic/results/agreement_one_clock.csv`, `agreement_detail_two_clock.txt` | `scripts/checks/compare_regression_to_mgcv.py` + `.R` (documented outside reference) and committed mgcv outputs on three small synthetic data sets as test fixtures | 1 |
| REML can have two optima; a local search stops at the worse one (2.4) | `S/proto_basic/check/two_disagree.py`, `S/proto_basic/check/pspline_localsearch_v1.py`, `S/proto_basic/check/investigate.py` | test: a planted data set where a start at `rho = 0` ends at the worse optimum and the grid start finds the better one | 1 |
| Closed-form checks of the fitter (2.4) | `S/proto_basic/check_closed_form.py`, `check_closed_form.txt` | hand-case tests (check 4) | 1 |
| Penalized spline false-positive rate, power, interval coverage (2.4) | `S/proto_basic/sim.py`, `S/proto_basic/run_all.py`, `S/proto_basic/analyze.py`, `results/report.txt` | slow test of coverage and size on 200 simulated data sets; the full grid as a stored analysis | 1 |
| Shared penalty beats per-session (2.5) | `S/proto_basic/check_aggregation.py` | history only; replaced by the stored check of the declared-penalty rule, `scripts/checks/simulate_declared_penalty.py`, and the test `test_sessions.py::test_the_declared_penalty_recovers_a_planted_curve` (86ad3fc5), moved onto the simulation engine in Phase 2b | 2, 2b |
| Held-out scores per session cannot show a small effect (2.5) | `S/proto_content/sim_lib.py` (`cv_gains`), `S/verify_rate/s6_refit_variants.py` | no test; the fold rule's docstring states the use of folds | 2 |
| Stop-end counting makes a false reward effect in M7r; a second clock removes it (2.1) | `S/proto_rate/proto_core.py` (`simulate`, `bin_events`, `design_proposed`), `S/proto_rate/p2_m7r_try.py`, `S/proto_rate/p3_run.py`, `log_full_null.txt` | unit test on synthetic stops (no reward effect, counting rule on: model A's ratio within tolerance of 1); stored analysis over the real visits with simulated events (null and planted) | 5a |
| SWR start drop before a stop ends is in the data, not the counting rule (2.7) | `S/explore_leave/part_a.py`, `S/explore_leave/epoch_worker.py`, `tables/a3_*.csv` | stored model-free curves against the three clocks per kind (the baseline for Q1 and Q7a) | 3 (curves), 5a (compare) |
| Most stop ends are small movements; three units compared (2.7) | `S/explore_leave/part_b.py`, `S/explore_leave/part_b4.py`, `tables/b1_*.csv`, `b4_curves_*.csv` | the stop-end kind column and its counts per animal in the visit tables, with a test | 3 |
| Clock overlap (4.7) | `S/my_checks/three_clock_overlap.py` | the collinearity record of every rate fit; Phase 5a's rule decides on `f_m` | 3, 5a |
| Leave definition: last stop end against the last poke-out (9) | `S/my_checks/leave_definition_check.py`, `leave_vs_pause_check.py` | test on synthetic visits of the clock arithmetic; per-animal counts in the table build log | 3 |
| Current-arm share rises toward leaving (2.2, 2.7) | `S/my_checks/departure_timecourse.py`, `S/explore_leave/part_b4.py` | model-free baseline for Q7b | 6a |
| Slopes differ by arm role; the pooled slope hides them (2.2) | `S/my_checks/role_slopes_check.py`, `S/verify_content/42_role_slopes.py`, `S/proto_content/run_sim.py`, `sim_all_roles.csv` | test: simulated content with opposite role effects, pooled slope near 0, role slopes recovered; model-free per-role slopes as a baseline | 6a |
| Centered recency sends signal to the uncertainty slope (2.3) | `S/verify_uncertainty/s06_null.py`, `S/proto_content/diag_centering.py`, `S/proto_content/calibrate.py`, `sim_all_uncertainty_summary.csv` | test: truth recency only, the uncertainty slope's false-positive rate near nominal with the uncentered smooth, and above it with the centered spline (shows the reason) | 6a |
| Straight-line recency fails on a curved truth (2.3) | `S/proto_content/run_sim.py` | test | 6a |
| The uncertainty measures against recency, reward fraction and epoch drift (tenth and twelfth rounds): belief variance calibrated only with a flexible unshrunk recency curve, low power; reward variability not separable from reward-fraction curvature (dropped); sample count needs the trial number | `S/proto_uncertainty_measures/` (`REPORT.md`, `sim_main_key_table.csv`, `sim_control_table.csv`, `sim.py`, `fit_lib.py`, `build_measures.py`, `check_measures.py`, `collinearity.py`; the repeat with the shared fitter's recency penalty, same folder) | the Phase 6a gate, then `tests/analysis/replay_models/content/test_uncertainty_measures_identifiable.py` and the uncertainty scenario list for the simulation package (Phase 2b), `replay_models/content/uncertainty_scenarios.py` | 6a |
| Uncertainty is mostly a clock of time since the last visit (2.3) | `S/verify_uncertainty/s01_covariate.py`, `s03_dose_response.py`, `s03b_recency_strata.py` | the collinearity record of the uncertainty models | 6a |
| Reward covariates use no outcome before it is known (2) | `S/verify_rate/s1_covariates.py`, `S/verify_content/10_covariates.py`, `11_next_timing.py` | tests of the new columns' timing against synthetic port experiences | 3 |
| Next arm equals previous arm in most events (3, Q6) | `S/verify_content/43_next_controlled.py` | the support count of the next-arm role | 7 |
| Stay or leave fits mostly do not converge now (7 table) | `S/verify_content/31_stay.py` | baseline count before the refit | 7 |
| Behavior summary values reproduce from stored fits (2) | `S/verify_rate/s3_reproduce.py`, `S/verify_content/30_reproduce.py` | baseline values recorded before each family is replaced | 5a, 6a |
| Inference session to day to animal (4.5) | `S/verify_rate/s5_inference.py` (`infer`), `S/verify_uncertainty/s07_inference.py` | functions in `shared_statistics.py` with tests | 2 |
| Events outside the visits (9) | `S/explore_outside_visits/analyze.py`, `analyze2.py`, `tables/q1_*`, `q6_examples*.csv` | counts per label in each summary (`count_visit_context`); the two data points checked in Phase 4 | 4 |
| Realistic rate curves (open question b): model A on the real bins with planted curves shaped like the real ones | `S/proto_rate_realistic/` (running on 2026-11-23), `S/proto_rate_smoothing/firth_check/model_form_diagnostics.py`, `S/proto_rate_smoothing/outcome_curves/` | the rate model's scenario list for the simulation package | 5a |
| Reward-term estimators inside per-session fits (support rule, forced fit; the bias-reduced fit is not used, lead 2026-11-23) | `S/proto_rate_smoothing/REPORT.md` (follow-up section), `firth_check/` | evidence for open question a; scenarios of the rate list | 5a |
| One JAX fitter: port design and its experiments | `S/jax_fitter/DESIGN.md` (copied as [jax-fitter-design.md](jax-fitter-design.md)) | Phase 2c | 2c |
| Model inventory: 15 families and their rules (2.6) | `S/model_inventory_2026-11-20.md` | the list of families each phase moves or retires ([overview](overview.md#what-happens-to-each-existing-family)) | all |

## Outside references (read for method, not imported)

| Reference | Path | Commit on 2026-11-20 | What to read |
|---|---|---|---|
| Penalized Poisson with REML, NumPy | `/cumulus/edeno/neurospatial/src/neurospatial/encoding/_glm_numpy.py` | `d8a5643a2` | `_newton_fit_numpy` (260), `_reml_objective_numpy` (408), `select_penalty_by_reml` (481): pooled REML over units, convergence gate |
| The same, JAX | `/cumulus/edeno/neurospatial/src/neurospatial/encoding/_glm_jax.py` | `d8a5643a2` | `_newton_loop_jax` (215), `_reml_score_jax` (349), `select_penalty_by_reml_jax` (385) |
| Laplace rate model | `/cumulus/edeno/state_space_practice/src/state_space_practice/temporal_rate_gp.py` | `28c237a` | `poisson_log_rate_site` (194), `infer_log_rate` (402): IRLS sites and Laplace |
| Softmax Newton | `/cumulus/edeno/state_space_practice/src/state_space_practice/multinomial_choice.py` | `28c237a` | `_softmax_update_core` (125): softmax curvature |
| Spline Poisson regression | `/cumulus/edeno/non_local_detector/src/non_local_detector/likelihoods/sorted_spikes_glm.py` | `4fe14a8fc` | `make_spline_design_matrix` (68), `fit_poisson_regression` (135) |
| Ridge Poisson batch, this repository | `src/continuum_swr_replay/analysis/replay_models/rate/ridge_history_cv.py` | branch head | `fit_ridge_poisson` (144), `fit_ridge_poisson_batch` (383), `nested_cv` (453) |
| Batched unit GLM, this repository | `src/continuum_swr_replay/analysis/pfc/batched_glm.py` | branch head | `_sweep` (460), `make_ridge_penalty` (677), `fit_poisson_designs` (1422), `DEFAULT_PENALTY_SIGMA` (167) |
| mgcv | `csr-mgcv` conda env (R 4.5.3, mgcv 1.9-4) | — | `gam(method = "REML")`, `s(bs = "ps")` with fixed knots, `s(L, by = R)` linear functionals, `concurvity(full = TRUE)`, `predict(type = "lpmatrix")` |
| Wood (2011), JRSSB 73(1):3-36 | — | — | Laplace REML and its derivatives (as built in `analysis/regression/reml.py`; `docs/design/regression_fitter.md`) |
