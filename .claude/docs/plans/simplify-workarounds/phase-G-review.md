# Phase G review: c20709d5..4c8ed8f8 (Tasks 22-27)

Reviewer: phase-boundary reviewer (read-only, one pass over the diff plus the code it touches).
Inputs: PLAN.md Phase G, task-22..27 briefs, common.md, task-22..27 reports, the range diff,
the Task 26 diff in task-26-report.md (checked against the backup file).

Re-run by the reviewer at 4c8ed8f8: test_lazy_imports, test_import_layering,
test_fit_replay_content_models, test_cohort_audit, test_rate_cohort_audits,
test_swr_contrast_rate, test_run_pfc_analysis_bandit, test_compare_fits_cohort_contrast,
test_paths, test_standardization, test_replay_content_runner_schema: 402 passed, 1 skipped
(sentinel data root). The full suite at 4c8ed8f8 (t25_suite, from the Task 27 report):
5807 passed, 7 known failures.

Canonical tables: none changed. No table builder, schema, or `tables/` module is in the diff.
The only `src/` changes are the hippocampus `__init__`, `cohort_audit.write_cohort_audit`
(audit CSV writer, new optional argument) and a `fit_provenance` docstring.

## Verdicts

| Task | Spec | Quality |
|---|---|---|
| 22 Hippocampus lazy imports | ✅ | approved |
| 23 Provenance records main's argv | ✅ | approved |
| 24 Logit ladder skips a failed session | ✅ code / one deliverable still open (I1) | approved, with M1 |
| 25 Archive fig-4 import; schematic columns | ✅ | approved |
| 26 Research scripts redirect through data root | ✅ | approved |
| 27 Power cohort CSV splits excluded events | ✅ | approved, with M3 |

## Critical

None.

## Important

**I1. Task 24: the full-cohort skips per rung are not reported yet.** The brief asks for a `--logit`
run on the full cohort into scratch and the sessions skipped per rung. At review time (09:23)
tmux `t24_logit` was still running. The last log line was at 09:16, and
`Processed-Data/_scratch_simplify_20261024/full_t21/replay_content/models/logit/logit_skipped_sessions.csv`
did not exist yet. The log shows 6 skips so far, all `ValueError: need covariance of parameters ...`:

- M4: senor/20201106/02_r1, wilbur/20210330/06_r3
- M5: senor/20201106/02_r1, wilbur/20210330/06_r3, wilbur/20210331/06_r3, wilbur/20210331/08_r4

This is a reporting gap, not a code defect. Close it before Phase G is signed off: when the run
finishes, read the published CSV, give the per-rung counts and the size of the common logit
session set (the "N of M primary sessions staged" lines), and confirm that the publish
succeeded. The open-ended part is the size of the common set. Every skip at any rung removes
the session from all rungs, and the skips grow with rung size (2 at M4, 4 at M5), so the larger
rungs may shrink it noticeably. The lead should see that number.

## Minor

**M1. Task 24: the skip reason comes from the fitter's last log record, which couples a published CSV
to log text** (`scripts/content/fit_replay_content_models.py:1607-1641`).

I judge this acceptable as it stands. I checked every `return None` path in
`fit_conditional_logit` (`src/continuum_swr_replay/analysis/replay/fitting.py:1225-1310`):

- each one after the first log statement logs its own reason, so today's reason is right;
- the empty-predictors return logs nothing, and the wrapper handles that case explicitly;
- unexpected exceptions are not caught, so they propagate (pinned by `test_an_unexpected_logit_error_still_propagates`).

The failure scenario: someone later adds a guard that returns None without logging, after the
NaN-drop `logger.info` at line 1252. The published reason would then read "dropping N event(s)
with NaN ...", which is wrong but looks plausible. The skip itself stays correct.

The alternative is small. In `fitting.py`, move the body into `conditional_logit_or_reason(...) ->
tuple[dict | None, str]`, so each of the 5 refusal sites returns the string it already logs, and
make `fit_conditional_logit` return `[0]` of that call. That is about 20 changed lines in
`fitting.py`. It deletes `_RecordCollector` and `_fit_conditional_logit_with_reason` (~35 lines), and
it ends the wrapper's temporary change of the fitting logger's level (see M2). The 25 existing call
sites keep working. One runner test that monkeypatches `runner.fit_conditional_logit`
(`test_replay_content_runner_schema.py:759`) would patch the new name instead.

I recommend it if the lead accepts a `fitting.py` edit in this phase. It changes no fit, cohort or
table.

**M2. Task 24: the wrapper lowers the fitting logger to INFO for each call**
(`fit_replay_content_models.py:1627`). The fitter's INFO records, such as the NaN-drop notes,
then reach the root handlers during logit fits, even when a caller had set that logger to WARNING.
The cost is log noise only, and M1's alternative removes it.

**M3. Task 27: no test checks that `_write_outputs` passes `extra_columns`.** The identity test
(`tests/unit/test_rate_cohort_audits.py:276`) calls `write_cohort_audit` itself, with
`mod.COHORT_EXTRA_COLUMNS`. `test_paths` and `test_standardization` check only file names.
Failure scenario: `extra_columns=COHORT_EXTRA_COLUMNS` is dropped from
`scripts/coupling/fit_swr_power_model.py:437-441`, the CSV quietly loses both columns, and every
test still passes. The fix is one `pd.read_csv` column assertion in `test_paths.py:265`'s test,
whose stub record already carries the two fields.

**M4. Task 22: the `TYPE_CHECKING` import mirror is not pinned to `_LAZY_EXPORTS`**
(`src/continuum_swr_replay/analysis/hippocampus/__init__.py:126-165`). The worker flagged this
in a ruling. A name added to the table but not to the mirror only degrades static resolution for
mypy and IDEs; nothing breaks at runtime. The replay/pfc inits have the same pattern.

**M5. Task 25: unverified wording kept in the schematic.** The "Real data" numbers in steps 4, 5, 8
and 9 (0.12 % straddle, median fragmentation 0, rho~0.94) were left unverified, as the worker
reported. Step 9 still names `fit_conditional_poisson` as the fit, while the shipped content
estimator is fraction-based. The brief limited the task to steps 5-7, so this is noted, not counted
against it.

Minor count: 5.

## Per-task notes (evidence)

### Task 22
- **Exports.** The new `_LAZY_EXPORTS` holds the same 32 names, each mapped to the same
  submodule as the eager imports at c20709d5.
- **No caller lost an attribute.** I grepped for package-level users of submodule attributes
  (`import ...hippocampus as h; h.data_aggregation`) and for `patch("...hippocampus.<name>")`.
  Neither exists.
- **`__getattr__` has no try/except.** An ImportError propagates and an unknown name raises
  AttributeError.
- **The guard tests are meaningful:**
  - the light-leaf subprocess test fails if any listed module loads jax, non_local_detector,
    statsmodels or dask;
  - the ImportError test (`sys.modules[...] = None`) would fail under any fallback that turns
    ImportError into AttributeError or None;
  - the worker's mutation check (eager init restored → 5 tests fail) is consistent with the
    test code.
- **Timing.** The leaf import went from about 4.9 s to 0.83 s wall, measured 3 times each,
  before and after.

### Task 23
- **Every writer site is converted.** At c20709d5 `git grep sys.argv` finds 20 writer sites
  outside `exploratory/`. All are converted. What remains at head:
  - `argv = sys.argv[1:] if argv is None else list(argv)` and `sys.argv[0]` in each `main`;
  - `_report._invocation`, which still reads the program from `sys.argv[0]`, as specified;
  - `table_parity.py:452`, which sets `sys.argv` for an in-process child and records
    nothing, so the worker's ruling is right;
  - a docstring.
- **No other command-line recorder reads `sys.argv`.** `run_refits.py:441` records its child's
  argv, and `runs/provenance.py` has no argv field.
- **CLI byte-identity holds by construction.** `[sys.argv[0], *sys.argv[1:]] == sys.argv`. The
  worker also confirmed it on a base-versus-head subprocess run.
- **Dask path.** `run_analysis` takes `cli` as a required keyword argument. Main's `cli` is
  passed to both the serial path (`run_pfc_analysis_bandit.py:1758`) and the delayed path
  (`:1790`), so a worker no longer reads its own `sys.argv`.

### Task 24
- **Skips.** A session that `fit_conditional_logit` refuses is logged, recorded in
  `LOGIT_SKIP_COLUMNS`, and skipped with `continue`.
- **One common session set.** `validate_and_match_fit_results`
  (`table_contracts.py:762-790`) intersects the session sets of all rungs. It still raises
  when a rung is empty or the intersection is empty.
- **Subset check kept.** `logit_sessions <= shared_sessions` (`:1749`). It is also implied by
  the matcher's frozen-cohort check, which is harmless.
- **The skip CSV** is written beside `<rung>_logit.pkl` inside the staged bundle, so it is
  published atomically, and it is written even when empty. No reader globs the `logit/`
  directory.
- **Tests.** The 4 new tests and the rewritten abort test are meaningful. The skip and
  publish tests fail at base.

### Task 25
- **Every name in the schematic exists.**
  - Event-arm table (`_EVENT_ARM_TABLE_BUILD_COLUMNS`, `_tables_common.py:231`): `arm_count_hard`,
    `arm_fraction`, `n_nonlocal_bins`, `is_current_arm`, `is_prev_arm`, `is_next_arm`,
    `visit_recency_trials*`, `reward_belief_mean/variance_candidate_arm_preoutcome`, `swr_x_*`.
  - `model_specs.py`: the `*_x_is_current_arm` terms (23 occurrences).
  - `add_trial_context`, for step 2: `prev_well`, `next_well`, `last_rewarded_well`.
- **Step 2's new assumption** matches the `add_trial_context` docstring (`context.py:434-436`).
- **`plot_aics(glm_results, model_list=...)`** matches `glm_plots.py:251`, and no `plot_AICs`
  remains in the repository.

### Task 26
- **The edit matches the report.** I diffed it against the backup; the backup's sha256 is
  `016392fa...61b1da`.
- **The environment variable is scoped correctly.** `session_content_distances`
  (`scripts/content/analyze_content_occupancy.py:166-176`) reads the root only through
  `decode_paths(epoch_id, processed_data_dir())`; the dot read is patched. Setting and restoring
  `$CSR_PROCESSED_DATA_DIR` around that one call therefore reproduces the old redirect.
- **The files left unchanged** are justified one by one. They are pinned to bb19449, read
  the live root on purpose, or use a layout the variable cannot express. None writes the live
  root.

### Task 27
- **The identity holds in the code.**
  - In `_cohort_records`, `complete` is finite over `_QVALS_PREDICTORS ∪ PREDICTORS ∪ {POWER_COL,
    event_start_s}`.
  - Every non-timing qvals predictor is in `PREDICTORS` (`fit_swr_power_model.py:78-86`), so
    `complete == finite & fit_complete` from `_post_outcome_masks` / `_power_fit_complete_mask`.
  - `selected ⊆ eligible & fit_complete`, so `n_in - qvals - own = pre_outcome +
    outcome_imbalance + selected.sum()`.
  - `fit_power_model` models the selected events of sessions with ≥ 30 events, so the
    remainder `selected.sum() - n_modeled ≥ 0` is exactly the session gate.
  - The unit test exercises all three buckets (24 + 20 + 20). The smoke fit gives 52 + 0 + 0 = 52.
- **No other runner's CSV changes.** No other caller passes `extra_columns`, and
  `df[[*COHORT_COLUMNS]]` is the same selection as before.
- **The dropped record fields** were never written to the cohort CSV, and the per-session gate
  audit still carries them (`:236-238`).

## Declined to judge

- Step 9's estimator name and the "Real data" numbers in the schematic: the brief scoped the task
  to steps 5-7 (noted as M5).
- Stale module imports in the `.claude/research` scripts (Task 26 concern 1), the inert ablation
  guards (concern 2), and `run_pfc_smoke.py`'s `DEFAULT_EVENT_TABLE_PATH` (concern 3): outside
  Task 26's brief, and gitignored lead files.
- `scripts/figures/archive/generate_figure04_panels.py` still has `def main() -> None` with no argv:
  it is an archive script, and Task 25 asked only for the import fix.
- No minimum session count for a published logit rung: the spec does not ask for one. The
  per-rung common-set size is surfaced under I1 instead.
- The heavy chain loaded by `data_aggregation`, `posterior_analysis`, `result_loading` and
  `memoryless_sequence.sweep` through their own imports: inherent to those modules, outside Task 22.
