# Simplify roundabout code and stale workarounds

**Branch:** `simplify-workarounds-2026-10-24` (from master `99034aab`).
**Spec:** the 2026-10-24 four-area survey reported to the lead in session
(findings restated per task below; the lead asked to "fix all"). No separate
spec file — rulings made against this plan are provisional.

**Goal:** remove the roundabout code, stale workarounds, duplicated sources of
truth and dead compatibility leftovers the survey found, **without changing any
scientific output**.

## Global Constraints

1. **Behavior-preserving.** Every canonical table (`event_table`,
   `event_arm_table`, `event_sequence_table`, `event_onset_table`,
   `event_patch_table`, `stop_table`, timebin tables, contfrag tables,
   `qvals_table`), every fitted-model output and every figure's data must be
   byte-identical (or, for floating-point refits, within 1e-10 relative) to the
   pre-branch code, unless the task text explicitly names an output change. An
   unexpected output difference stops the task: report it as BLOCKED with the
   diff, do not "fix" it by changing the reference.
2. **No writes to live data.** Never write into the live
   `Processed-Data/replay_content/` tables, model directories or decode
   directories. Regenerations and smoke fits go to a scratch root
   (`Processed-Data/_scratch_simplify_20261024/` or the session scratchpad).
   No re-decoding, no full model refits.
3. **Untracked files are not yours:** `.claude/docs/plans/finish-covariate-rename/`,
   `.claude/docs/plans/representation-selection/`, `docs/reference_*_2026-09-17.md`,
   `paper/figures/replay_story/`, `scripts/figures/generate_replay_story_figures.py`,
   `tests/integration/test_replay_story_figures.py`. Do not edit, stage or
   delete them. Their 7 test failures are pre-existing.
4. **Surgical.** Modify only what the task names. `ruff check` and
   `ruff format` on touched files only (never repo-wide format).
5. **Tests.** Run the tests covering touched code; add an equivalence test
   (old vs new on simulated data) whenever a computation is rewritten. Keep
   `tests/unit/test_lazy_imports.py`, `tests/unit/test_no_new_script_imports.py`
   and `tests/unit/test_models_table_mirrors_specs.py` passing. Real-data
   checks use `-m real_data` on the smallest slice that exercises the path.
6. **Docs follow code.** When a change makes CLAUDE.md, `docs/`, `tests/README.md`
   or a docstring wrong, fix that text in the same commit.
7. **Commits:** one or more per task, message style of the repo (short
   sentence, e.g. "Cohort audit: the qvals gap is decided per row"), ending with
   the session's Co-Authored-By / Claude-Session lines. No push, no merge.
8. **Workers never dispatch subagents.**

## Phase 0 — baseline

### Task 0: Freeze the reference outputs
- Record `git rev-parse HEAD` and the default-suite result
  (`uv run pytest -q -p no:cacheprovider`, ~31 min; expect 7 failures from the
  untracked replay_story files) in the report.
- Establish a table reference the final parity check can diff against.
  Find how the canonical tables are regenerated (`scripts/tables/`,
  `scripts/qa/compare_to_golden_baseline.py`, `scripts/qa/freeze_regression_baseline.py`,
  `tables/parity.py`) and whether a run can be pointed at a scratch root
  (`CSR_PROCESSED_DATA_DIR`, `parameters.py:86`). Regenerate the canonical
  tables for a 3-session slice (one chimi, one j16, one session with PFC) with
  the **pre-branch** code into the scratch root and check them against the
  live tables restricted to those sessions. If identical, the live tables are
  a valid reference for the end-of-branch full regeneration; if not, keep the
  scratch slice as the reference and say why they differ.
- Time the one-session regeneration; extrapolate the full-cohort cost.
- Write the exact commands to `.claude/docs/plans/simplify-workarounds/parity.md`
  so later tasks rerun the slice parity with one command. Commit that file.

## Phase A — stale docs and dead leftovers

### Task 1: Fix stale docs and comments
- CLAUDE.md "Canonical per-event classification table": the text describes a
  `label_event_*` rename → bit-for-bit check → drop round trip in
  `build_event_table` and says `label_event_*` survives as an alias in
  `covariate_registry.py`. Current code (`analysis/replay/tables_event.py:~470`)
  derives timing once and renames; `label_event_*` survives only in the
  `LEGACY_COLUMN_NAMES` deny-list (`tables/covariate_registry.py:381`). Rewrite
  that paragraph to match the code.
- `analysis/replay/simulation.py:1056` `simulate_full_pipeline` docstring names
  `compute_arm_counts → collapse_to_events → make_long_format_arm → add_event_uid`,
  none of which exist. Route the simulator through the real builder
  (`build_event_arm_table` or the current equivalent) if that is feasible
  without changing what the tests assert; otherwise reword the docstring to say
  exactly which production functions it exercises. Drop the "backward compat"
  events frame (~:1221) if nothing reads it.
- `analysis/behavior/_spatial_bandit_rl.py`: header says "copied unchanged… no
  edits" but six marked perf-shim/pandas-compat edits exist. State that it is
  vendored with those edits, listed.
- `scripts/figures/generate_inventory_examples.py:~182`: docstring blames
  `run_state_evidence.py`, which it does not import; the CPU pin comes from
  `run_inventory_comparison.py:94`. (Task 16 may remove the pin; fix the text now.)
- `analysis/replay/_tables_common.py:144` comment names `event_start/...`
  columns that no longer exist.
- `data_loaders/utils.py:~408` `get_labels` docstring says "vectorized" but loops.

### Task 2: Remove dead compatibility leftovers
For each item, grep all of `src/`, `scripts/` (including `scripts/figures/archive/`),
`tests/`, `docs/`. If a persisted table schema or on-disk artifact reader
depends on it, keep it and say so in the report.
- `data_loaders/task_variables.py:396-480, 571-620`: the `parallel=` flag and
  `_process_single_trial`; no caller passes `parallel`.
- `data_loaders/trials.py:772,809`: `is_boundary_marker` (always False, read
  only by tests). Remove column and the test assertions on it.
- `data_loaders/trials.py:~933`: `goal_arrival_time` compatibility alias of
  `destination_first_poke_in_time`, and `start_time` duplicating
  `movement_start_time`. Migrate readers to the canonical name and drop the
  alias — **unless** either name is a column of a persisted canonical table, in
  which case leave it.
- `make_well_visits_df`: unused `position_info` / `track_graph` parameters.
- `get_start_end_node_id` (`trials.py:636-737`): called only by tests; its
  "for test compatibility" fallback to all graph nodes contradicts its
  docstring. Make the tests build a graph with junctions and drop the fallback.
- `analysis/replay/covariates/context.py:569-571`: `current_well` alias, never read.
- `analysis/replay/model_comparison.py:104`: legacy `"ll"` fallback (no producer).
- `analysis/replay/retrospective.py:640`: `include_patch` legacy flag (no caller).
- `analysis/constants.py:47`: `DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC` duplicates
  `MAX_SPEED_THRESHOLD_CM_PER_SEC` under a second name. Keep one name (the one
  with more uses) and migrate the rest. `constants.py:58`: drop the `float()` cast.
- `figure_utilities.py:435`: dead `paper/figures/styles/paper.mplstyle` branch.
- `tables/covariate_registry.py`: `alias`/`retire`/`derive`/`audit` action
  branches unused by `REGISTRY`; `finalize_covariate_schema` (:904) has no
  production caller. Remove the dead branches (and the exploratory caller's
  dependency, or leave the exploratory script broken-and-noted if it is a
  one-off under `scripts/exploratory/`).
- `analysis/hippocampus/nonlocal_detection.py`: `other_data` parameter of
  `event_strict_overlap_keys/_mask` has no caller passing it.
- `runs/cli.py:filter_epoch_info`: unreachable `isinstance(..., pd.Series)`
  branch. Also the `--epoch` filter splits `"02_r1"` and ignores `_r1` — fix it
  to match the full epoch name (this is a bug fix; add a test).
- `hippocampus/event_analysis.py`: `get_event_start_end_times` duplicates
  `immobility.intervals_from_labels`; delegate to it. Keep `get_immobile_events`
  (archive figure scripts use it) but drop its backward-compat modes if no
  caller uses them.

### Task 3: Repo config and small script redundancies
- `pyproject.toml`: merge the `[dependency-groups] dev` (pytest-cov only) into
  the `dev` extra so `uv sync --extra dev` is complete; deduplicate the
  `jax-cuda12-plugin` line shared by `cuda` and `gpu` (e.g. `gpu` depends on the
  `cuda` extra). Check whether `optax`, `panel`, `kachery`, `kachery-client`,
  `kachery-cloud` are needed (imported anywhere, or required by figurl /
  sortingview at runtime); remove only the truly unneeded. Keep
  `override-dependencies` (still required by Spyglass's pins). Run `uv lock`
  and `uv sync --extra dev`; report the lock diff summary.
- `scripts/rate/fit_cross_history_rate.py:37`: remove `src/` `sys.path` insert
  (package is installed editable).
- `scripts/behavior/fetch_spatial_bandit_behavior.sh`: reads `$PROCESSED_DATA_DIR`
  while Python reads `CSR_PROCESSED_DATA_DIR`; use the same variable.
- BLAS-thread pin block copied into 5 entry points (`fit_per_bin_rate_models.py:69`,
  `run_event_features.py:78`, `run_pfc_analysis_bandit.py:63`, …) plus a variant
  at `run_memoryless_detection.py:778`: one helper in `runs/` called before
  numpy is imported (it must still run before the numpy import, so the helper
  module must not import numpy).

## Phase B — one source of truth

### Task 4: Interaction provenance in one table
`analysis/replay/contrast_aggregation.py:1066-1130` builds ~33 `*_x_*` product
columns by hand; `covariate_timing.py:79-205` (`DERIVED_COVARIATE_SOURCES`)
lists their sources by hand; `cohort_audit.py:83-129` (`_factor_forms`) recovers
factors by splitting names on `_x_`. The copies already disagree
(`log1p_time…` lists `is_current_arm`, `rpe_…_x_is_current_arm` does not).
Make one `{product: factors}` declaration that builds the columns and answers
the timing and qvals-gap questions. Before changing anything, list every
disagreement among the three copies in the report and say which is right; if
resolving one would change an output, that is a named output change — report
it and keep the current output (BLOCKED only if the outputs cannot be kept).
Verify: the built columns are byte-identical; `covariate_timing` and
`cohort_audit` answers identical for every column on the slice.

### Task 5: Per-bin model terms stated once
`PerBinRateModelSpec.formula()` (`model_specs.py:650`) and `_design_columns`
(`per_bin_rate_model.py:77`, also imported by `scripts/rate/fit_uncertainty_rate.py`)
rebuild the same term list; `_BLOCK_F`/`_BLOCK_COLS` (`model_specs.py:1011/1029`)
are parallel dicts; `longest_history_window_s` (~1150) guesses history columns
from name prefixes. One list of (term, columns) producing both. Verify: for
every per-bin spec, the formula string and column list are identical before
and after (write the test against a frozen copy of the old outputs).

### Task 6: One stop-context helper
`analysis/replay/tables_stop.py:300-340` and `analysis/contfrag/tables.py:896-940`
are near-identical (~95 lines). Both loop over `grouped` calling `strict_first`
although `bout_first_row = grouped.first(skipna=False)` already exists
(`tables_stop.py:271`). One shared vectorized helper. Verify stop and contfrag
tables byte-identical on the slice.

### Task 7: Decode naming and detector classes in `decoding/`
`decoding/files.py:decode_identifier` hardcodes `"Kay"`/`"no_ripple"`;
`data_loaders/utils.py` defines `DEFAULT_SWR_DEFINITION`, `DEFAULT_TRAINING_TYPE`,
`OTHER_RIPPLE_DETECTORS` and the identifier parser; `parameters.TrainingType`
is a third copy. Put the constants and both directions (build/parse) in one
module under `decoding/`. `parameters.py` `DETECTOR_CONFIG`/`CONTFRAG_DETECTOR_CONFIG`
store class names as strings that scripts map back via `{cls.__name__: cls}`
(`run_decoding_bandit.py:73`, `run_decoding_contfrag.py:64`); hold the real
classes in `decoding/` (respect `test_lazy_imports.py`: if `non_local_detector`
must stay lazy, use a function that returns the mapping). Verify every
existing decode identifier round-trips (build → parse) identically; decode
file names for the whole epoch register unchanged.

### Task 8: qvals ingestion in one function
`tables/covariate_registry.py:904`, `analysis/behavior/qvals_table.py:355` and
`:915` hand-copy the canonicalize → `apply_rename` → `rename_dynamic_columns`
→ drop chain; it runs per animal (`:678`) and again on the concat (`:1052`);
`qvals_table.py:301` writes `q_port{i}` as a copy of `Q{i+1}` and then checks
the copy against itself (only `Q` vs `betadist_mu` is a real check). One
ingestion function with a plain rename and the one real check. Verify
`qvals_table` byte-identical (regenerate it to scratch; the source CSVs under
`/stelmo/alison/behavior_model_csvs/` are READ-ONLY).

## Phase C — compute once

### Task 9: Analyzed-time drops computed once
- `analysis/events/eligibility.py:203-275` `event_drop_reasons` applies the
  speed rule with a per-event loop (`filter_event_intervals_by_speed`,
  `immobility.py:376`) and again with cumsum counts, raising if they disagree.
  Keep only the vectorized version (equivalence test vs the loop on simulated
  events incl. empty, NaN-speed, boundary cases).
- `analysed_time._drops` (:483-504) computes dropped-SWR bounds on the decode
  grid then discards them, and re-codes `dropped_for_excluded_time` inline.
  Return the grid bounds as columns of `artifact_event_drops` and use the one
  predicate; make `inventory_comparison/adapters.py` `_sampled_bounds` (:331) /
  `_dropped_on_grid` (:358) use those columns instead of the private
  `_speed_eligible_labels`.
- `scripts/detection/run_memoryless_detection.py:450` uses raw inventory bounds
  for the same drops. Measure whether it differs from the grid bounds on the
  slice and report; do not change its output in this task.
- Do NOT move theta/run-speed normalization (`analysed_time._normalize_outside_mask`
  :609) — Ruling P1 in the ledger.

### Task 10: Loader computations without the long way round
Each with an old-vs-new equivalence test on simulated data plus a real-data
check on ≥2 epochs:
- `data_loaders/spikes.py:116-185` `detect_coincident_spikes`: replace the
  `ndimage.label` + off-by-one shift + `groupby().apply` rebuild with a
  cumsum cluster id, distinct-tetrode count per cluster, keep mask, `np.split`.
- `data_loaders/lfp.py:778-797`: `get_multiunit_hse` re-implements
  `intervals.get_spikecount_per_time_bin` (`intervals.py:573`); call it.
- `data_loaders/utils.py:408-450,494`: `get_training_timepoints` turns
  `get_labels` output back into a boolean with `> 0`; use
  `intervals.interval_mask`. Check NaN-bound behavior (interval_mask refuses
  NaN bounds): confirm on real ripple tables that none have NaN bounds, else
  report.
- `analysis/hippocampus/swr_detectors.py:836` `_mask_intervals` turns a bool
  mask into NaN to reuse `gap_free_stretches`. If splitting at timestamp gaps is
  intended, leave it with a one-line comment saying so; otherwise use
  `intervals.contiguous_true_runs`.
- `analysis/events/swr_source.py:282` `swr_epoch_id_of_session` builds a fake
  `"{animal}{day}_.nwb"` so `format_epoch_identifier` can strip it; build the
  id directly.

### Task 11: Per-stop GLM without the intercept relabel
`analysis/replay/per_stop_rate_model.py:138-207`: array design named
"intercept", relabeled "Intercept" mid-fit, so outputs carry `coef_intercept`
next to `sd_Intercept`; BIC hand-computed at :195 though statsmodels'
`result.bic_llf` exists. Use one intercept name throughout and statsmodels' BIC.
If any reader of saved Family-A bundles (grep `coef_intercept`, `sd_Intercept`)
depends on the current names, keep the saved names and only remove the
internal relabel; otherwise unify and update readers in the same commit. Verify
coefficients/SEs/BIC on a smoke fit match to 1e-10 relative.

## Phase D — error handling and stale workarounds

### Task 12: Errors that hide bugs
- `data_loaders/spikes.py:479-495` `get_spike_data` treats every `ValueError`
  as "area not recorded". Raise/catch a dedicated exception from the place that
  knows the area is absent.
- `data_loaders/position.py:223-262` catches `DataJointError` around `fetch1`,
  conflating "more than one row" with "no row". Verify, then distinguish.
- Fitters catching `Exception` and returning None: `per_stop_rate_model.py:175,277`
  (no `exc_info`), `fitting.py:1308`, `choice_prediction.py:533`,
  `model_checking.py:133`; `fitting.py:966` swallows diagnostics at debug,
  `:1455` swallows VIF. Reuse `per_bin_rate_model.py:68` `_EXPECTED_FIT_ERRORS`
  (move it somewhere shared) so coding errors propagate. Log the expected ones
  with `exc_info`.

### Task 13: Lazy-export layer and pickle fallback
- `analysis/hippocampus/__init__.py`: 169-line lazy-export layer justified by a
  DataJoint import at `data_aggregation` import time; the survey found no
  spyglass/datajoint module loads when importing `data_aggregation`,
  `quality_metrics`, `result_loading`, `triggered_analysis`, `hpc_raw_stats`.
  Re-verify for every submodule it exports (`python -X importtime` or
  `sys.modules` check); if true, replace with plain imports. Keep
  `test_lazy_imports.py` passing (update its pinned list if it pins this layer,
  saying why).
- `tables/io.py`: pickle fallback (`read_table`, `read_attrs`,
  `resolved_table_path`, `table_exists` redirecting `.pkl` → `.parquet`). All
  live tables are parquet. Make callers build parquet paths
  (`contrast_aggregation.py:724-727`, `prepare_data.py:971`,
  `qvals_table.py:965`, `scripts/rate/fit_uncertainty_rate.py:145`, others found
  by grep) and remove the redirect. If a script still needs to read `_backup`
  pickles, it reads them explicitly.

### Task 14: Parameters with one legal value
Seven builders take `speed_threshold` and call `require_canonical_speed_threshold`,
raising on anything but the default: `tables_event.py:700,813`,
`tables_stop.py:132`, `tables_arm_patch.py:114`, `contfrag/tables.py:292,597,772`.
`pfc/session.py:1160` hand-checks `immobility_threshold` the same way. Drop the
parameter (use the constant). Leave `pfc/canonical_events.py:265`
(`allow_stricter=True`) alone. Update callers and tests.

### Task 15: Import-time side effects
- `warnings.filterwarnings(...)` for pynwb/hdmf/datajoint copied at module level
  in `data_loaders/utils.py`, `position.py`, `spikes.py`, `lfp.py`, `events.py`,
  and `bandit_task.py:39` importing `PositionIntervalMap` (noqa F401) only to run
  Spyglass's logger setup: one setup function called from `load_data` (and
  any other public loader entry point that needs it).
- `analysis/behavior/qvals_table.py:51` disables pandas `PerformanceWarning`
  process-wide: scope it with `warnings.catch_warnings()` around the vendored call.
- `jax.config.update("jax_enable_x64", True)` at import in
  `analysis/replay/dirichlet_multinomial.py:62` and `analysis/pfc/batched_glm.py:124`.
  Check whether the pinned JAX (see `uv.lock`) offers a scoped x64 context that
  works under `jit`. If yes, scope it to the fitting entry points and remove the
  import-time switch, verifying fits identical; if no, leave it and write one
  line in the report why.
- `runs/batch.py:126` `release_nwb_files` reads Spyglass's private
  `__open_nwb_files` only to log a count, with a warn-once global. Keep the
  `close_nwb_files()` call (Spyglass #1442 workaround), drop the private read.

## Phase E — scripts and test seams

### Task 16: Shared inventory-comparison code into `src/`
~19 non-exploratory scripts edit `sys.path` to import other scripts;
`scripts/detection/run_inventory_comparison.py` (908 lines) is imported by 5
scripts for its roster, `decode_path`, `output_path`, `SAMPLE_INTERVAL_S` and
the private `_speed_trace`; `sweep_memoryless_detection.py` by 3. The allowlist
in `tests/unit/test_no_new_script_imports.py` (21 entries) freezes this.
Move the shared pieces into `src/continuum_swr_replay/analysis/inventory_comparison/`
(or the module where each belongs), make the scripts import from the package,
remove their `sys.path` inserts and shrink the allowlist accordingly. Also:
`run_inventory_comparison.py:94` sets `JAX_PLATFORMS=cpu` when imported, and
`generate_inventory_examples.py:186-188` forces CUDA back; device choice moves
into each entry point's `main()`. Other script-to-script imports (e.g.
`run_carey_detection.decode_marks`) move too if they are library code. Verify:
every touched script's `--help` runs; the integration tests covering them pass;
a one-session detection/comparison output is identical.

### Task 17: `main(argv=None)` everywhere
~70 scripts define `main() -> None` reading `sys.argv`; 16 test files patch
`sys.argv`. Give every script `main(argv: Sequence[str] | None = None)` passing
`argv` to `parse_args`, and make the tests call `main([...])`.

### Task 18: Figures saved explicitly
`figure_utilities.py:581` `save_figure` takes no figure and uses `plt.gcf()`.
Give it an explicit `fig` argument and update all callers; replace the
hand-rolled png+pdf loops (`generate_supp_pipeline_schematic.py:188`,
`generate_supp_pipeline_data_flow.py:468`, `diagnose_trial_detection.py:162,257`)
and the direct `.savefig` calls that duplicate `save_figure`'s job (19 scripts —
convert where the output path/format is the same; leave ones that need other
options). `scripts/figures/generate_all_figures.py:83-91` shells out to
`uv run python <script>`: keep the subprocess (isolates matplotlib state) but
use `sys.executable`, report full stderr on failure, and say in its docstring
which figures it covers.

### Task 19: One patch point for data roots
`PROCESSED_DATA_DIR` is bound into ~45 modules at import, so tests patch it
module by module (28 setattrs; `test_table_builders_read_the_register.py`
patches two modules six times); `paths.model_dir`'s docstring says pass a root
instead. `swr_source.ARTIFACT_INTERVALS_RUN_ID` / `SWR_INVENTORY_ROOT` /
`ARTIFACT_INTERVALS_DIR` are module globals that an autouse fixture
(`tests/conftest.py:59-80`) nulls and tests patch 46 times across 21 files.
Make data roots resolved at call time through `paths` (so one patch point, or
an explicit `root=` argument where a function already has one), and replace
the module-by-module test patches with that one point. For the artifact
source, prefer an explicit argument threaded from the entry points if it
touches ≤ ~30 call sites; otherwise resolve it through one accessor and keep the
autouse fixture patching that accessor. Record the choice as a ruling in the
report. Verify the slice parity and the full default suite.

### Task 20: Fitters report their own drops
Four runners (`scripts/rate/fit_swr_contrast_rate.py`, `fit_uncertainty_rate.py`,
`fit_per_bin_rate_models.py`, `scripts/coupling/fit_swr_power_model.py`) each
have a `_cohort_records` that reconstructs which rows the fitter dropped,
though the fitter already returns part of it (`result.attrs["cohort_stop_drops"]`).
First verify (the survey marked this suspected). If true, make the fitter
report every drop it makes and the runners read it; cohort records must be
identical on a smoke fit.

## Phase F — final parity

### Task 21: Full regeneration parity
Regenerate every canonical table for the full cohort with the branch code into
the scratch root (tmux session `simplify_parity`, commands from `parity.md`)
and diff against the Task 0 reference. Run `uv run pytest -m real_data`.
Report every difference; there must be none outside named output changes.

## Phase G — the lead's answers to the final review (2026-10-26)

The lead answered the final review's questions. Not changed, by the lead's
answer: the senor20201030 linearization tie-break (leave alone), the
memoryless sweeps (stay on the GPU when one is visible), the 4 regenerated
`.claude/research` outputs (accepted). Every task keeps the Global
Constraints; Tasks 22–26 change no canonical table.

### Task 22: Hippocampus imports lazy again
Restore a lazy-export `__init__` for `analysis.hippocampus` only, so
`import continuum_swr_replay.analysis.hippocampus.<leaf>` no longer loads
JAX/non_local_detector/statsmodels/dask for leaves that do not need them.
Start from the layer Task 13 removed (`git show 7b5fec82`), but keep what
Task 13 got right: no fallback that hides an ImportError, and the guard
tests (`test_lazy_imports`, `test_import_layering`) pin the new behavior.
Measure a leaf import's time before and after.

### Task 23: Provenance records the arguments main received
Every writer that records `sys.argv` (fit sidecars via `write_fit_provenance`,
`_invocation`, `scripts/qa/_report.py`, the PFC runner's `cli=`, about 12
sites) records `[program, *argv]` where `argv` is what `main(argv)` parsed
(`sys.argv[1:]` when `main` is called with `None`). A CLI run's sidecar stays
byte-identical in that field; an in-process `main([...])` records its own
arguments, not pytest's. Tests that patch `sys.argv` only to pin the recorded
command line pass argv instead.

### Task 24: The conditional-logit robustness ladder skips a failed session
`scripts/content/fit_replay_content_models.py --logit` raises when one
session's fit fails with an expected fit error (live: M4, senor20201106
02_r1, "need covariance"). Skip such a session instead: log it with its
reason, keep every logit rung on one common session set, publish the skipped
sessions beside the logit fits, and keep the check that the logit sessions are
a subset of the primary ladder's. An unexpected error still propagates. Run
`--logit` on the full cohort into a scratch root (never the live one) and
report the sessions skipped per rung.

### Task 25: Archive figure-4 script imports; schematic names current columns
`scripts/figures/archive/generate_figure04_panels.py` imports `plot_AICs`,
renamed `plot_aics` in c2f73132: import the current name and confirm the
script imports. In `scripts/figures/generate_supp_pipeline_schematic.py`,
steps 5–7 name columns that no longer exist; name the current ones (from
the event/event-arm table schemas and `model_specs.py`) and redraw the figure.

### Task 26: Research scripts redirect through the data root
`.claude/research` scripts (gitignored, the lead's) that redirect output by
assigning a module root (`mod.RC = …`, `*.PROCESSED_DATA_DIR = …`,
`swr_source.SWR_INVENTORY_ROOT = …`) have done nothing since Task 19, so they
read and write the live root. Change each to set `$CSR_PROCESSED_DATA_DIR`
before its first package read, as `scripts/exploratory/_content_cohort_control.py`
does. Back up every file before editing; never run one.

### Task 27: The power model's cohort CSV splits its excluded events
`scripts/coupling/fit_swr_power_model.py:_cohort_records` builds
`n_events_excluded_pre_outcome` and `n_events_excluded_outcome_imbalance`, but
`write_cohort_audit` writes only `COHORT_COLUMNS`, so neither reaches
`swr_power_cohort_audit.csv`, whose `n_excluded_session` lumps them with the
<30-event sessions. Let `write_cohort_audit` take extra columns written after
the canonical ones, and have the power runner pass these two; no other
runner's CSV changes. Drop `post_outcome_gate`, `post_outcome_gate_rule` and
`outcome_time_semantics` from the cohort record (the per-session
`swr_power_post_outcome_gate_audit.csv` carries them). On a smoke fit, the two
counts plus the session-gate remainder add up to `n_excluded_session`.

## Ruled out of scope
- Theta/run-speed re-normalization at read time (Ruling P1).
- `fitting.py:280` `_rows_of_events_whose_response_varies` (statsmodels
  `ConditionalPoisson` drop rule): retiring it changes the fit cohort (Ruling P2).
- Spyglass merge-entry workaround for j1620210710, `ripple_detection` private
  `_valid_blocks` import, netCDF `reset_index`, `override-dependencies`, the
  hand-written fitter: justified (Ruling P3).
