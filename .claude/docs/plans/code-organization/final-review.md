# Final review: code-organization-2026-10-19 (470a393..6e511d39, 51 commits)

Reviewer: opus, read-only, 2026-10-22. Phase C (Tasks 12–16) got most of the time. Phases A and B were
re-checked only for the whole-branch invariants and the open items.

## How I split the passes

1. **Classify the commits.** I ran `git show -M --stat` on every Phase C commit and `git diff -M --name-status 470a393 HEAD`
   for the whole branch.
   - 46389e03 is 27 renames (R084–R100) plus path strings.
   - 7783e97a and 994417dd are near-pure renames (99% / 98%).
   - 26844eea and 6de3c467 are splits that git cannot detect as renames.
   - The real edits are 779c08c7, 7376d457, 8c0aeca8, 5b956bd2, 03258d60, 89d9cb00 and the guards.
2. **Verbatim checks on the splits.** My own AST script compares each top-level definition's source text, old against new
   (`scratchpad/final_review/verbatim.py`).
   - Task 14: glm.py at 3782844e against the six modules at 26844eea. 49 = 49 definitions, none missing, none extra, none differing.
   - Task 15: session_tables.py at 779c08c7 against the package at 6de3c467. 92 = 92, none missing, extra or differing.
   - Task 16: `git diff types.py schemas.py` changes only the 3 `>>>` doctest lines.
   - swr_source.py at BASE against analysed_time + swr_inventory: 15 definitions differ. Every difference is one of these:
     - a docstring;
     - a module-qualified call (`swr_inventory.<fn>`, from Phase-B M6);
     - `swr_epoch_id` → `format_epoch_identifier`, whose body is the same expression;
     - `ARTIFACT_INTERVALS_ROOT` → `_DIR`.
3. **Read the real edits line by line.** This covered:
   - 779c08c7 (`process_neuron` and `run_session` stages);
   - 7376d457 (contrast assembly: the `RC` and `FIT_INPUTS` handling against BASE);
   - the allowlist history of `test_no_new_script_imports`;
   - `write_pipeline_map.py`;
   - `test_produced_by_paths_exist`;
   - the README, CLAUDE.md and docs diffs.
4. **Greps over tracked files.** I searched for 46 old module path forms (dotted and slash), the 27 moved script paths,
   and bare `<old_module>.<attr>` references. Historical plans/reviews and `.superpowers` were excluded.
5. **Scientific invariants.**
   - Every call site of `excluded_intervals` / `with_swr_source` / `artifact_event_drops`, at BASE and at HEAD.
   - The `classified_table` diff, and the `detection` / `eligibility` / `immobility` diffs.
   - Whether `spatial_mapping` changed. It did not.
6. **Execution (targeted only).**
   - The guards: layering, lazy imports, doc paths, script imports, PRODUCED_BY, the stdlib shadow test, the pipeline map, paths. 81 passed.
   - 7 Phase C test files: 180 passed. `write_pipeline_map.py --check` is current.
   - `--help` exits 0 for 4 moved scripts. `run_refits.py --dry-run` works.
   - The lead's WIP imports resolve.
   - A scan of all 405 `monkeypatch.setattr(<module>, "<name>")` calls in tests found no vacuous patch the moves created. Every
     flagged hit was a module-attribute read at call time, or an alias my heuristic got wrong.
   - `ruff check src scripts tests`.
   - The `io` / `statistics` stdlib-shadowing probes.

## Strengths

- **The splits are verbatim.** I confirmed all three independently: PFC GLM 49/49, session_tables 92/92, and the schemas rename,
  which changes doctests only.
- **The layering is complete.** `KNOWN_VIOLATIONS` is empty for all six rules, including `no_package_cycles`.
  `HIPPOCAMPUS_ALLOWED_REPLAY_MODULES` and `JAX_FREE_PENDING` are empty too, and the stale-entry check stops the allowlists from rotting.
- **Old paths are gone.** No old dotted or slash module path is left in tracked code or docs. The remainders are deliberate:
  - the two frozen manifest strings, each with a comment;
  - the historical diff statement at `compare_pfc_cohorts.py:2041`;
  - the "(now …)" annotations in one design doc;
  - the four committed CSV provenance rows.
- **The analysed-time rule is untouched.** The set of read-point call sites is identical at BASE and HEAD (only the paths differ).
  `classified_table` changes are imports and the `match_swr` rename only, `spatial_mapping` is untouched, and strict overlap is
  still `events.detection.strict_overlap_matrix`.
- **The decomposition preserves behaviour.**
  - Stage bodies are contiguous blocks of the original.
  - `neuron_stats` key order is preserved (the `dict.update` order matches the original assignment order).
  - The fallback path's `_firing_rates` is the same three expressions.
  - The one ordering change: the argument checks now run before the deferred imports. It is error-path only and is an
    improvement, since a bad threshold now fails without DataJoint.
  - The moved canonical_events import pulls in no DataJoint or Spyglass (checked), so it cannot fail later than before.
- **Script moves keep their paths straight.**
  - Every `PRODUCED_BY` and `"produced_by"` literal names its own file.
  - Nothing compares a stored `produced_by` with the current path, so runs from before the move stay readable.
  - The `parents[2]` repo-root shims are still correct, because every move keeps the same depth.
- **The guards were mutation-checked** (script-import probe, PRODUCED_BY probe, stdlib red/green). `load_population` got lighter:
  at BASE it pulled in jax, statsmodels, non_local_detector and dask; now it pulls in jax only.

## Issues

### Critical
None.

### Important

**I1. The five PFC audits fell out of the regression pipeline map.**
- Where: `scripts/qa/write_pipeline_map.py:27-41` (`ROOTS`).
- What is wrong: at 8f690706, `docs/regression_pipeline_map.md` had entries for `scripts/qa/{check_pfc_ridge_lambda0,
  compare_pfc_cohorts, compare_pfc_fitters, diagnose_pfc_event_definition, measure_pfc_null_geometry}.py` (lines 733–793).
  After 46389e03 they are gone, because `scripts/pfc` is not a root.
- Why it matters: `--check` passes, because it only checks that the map is current. A pure reorganisation silently removed
  documentation coverage.
- Fix: add `"scripts/pfc"` to `ROOTS`, and add `scripts/pfc/run_pfc_analysis_bandit.py` to `EXCLUDED_PREFIXES`, so coverage
  matches BASE exactly. Regenerate the map, and check that those five entries come back with the same stage and body.

**I2. Nothing a researcher reads states the new layout's rules or its old→new module map.**
- Where: `CLAUDE.md:66-91`, `docs/README.md`.
- What is wrong:
  - The foundation → `analysis.events` → `analysis/*` → `visualization` / `scripts` layering exists only in
    `tests/unit/test_import_layering.py`. "Foundation" appears in no tracked doc.
  - About 40 `src` modules moved with no shims. `scripts/README.md` has an old→new table for the scripts, but nothing maps
    `hippocampus.swr_source` → `events.analysed_time` / `events.swr_inventory`, `decoding_common` → `runs.*` / `decoding.config`,
    `table_io` → `tables.io`, `pfc.glm` → six modules, `types` → `schemas`, and so on.
  - Out-of-repo notebooks and the lead's auto-memory (which still says `swr_source.ARTIFACT_INTERVALS_RUN_ID`) get a bare
    `ModuleNotFoundError` with no pointer.
- Why it matters: the plan's goal is that "a researcher can find each rule in one place". The rule that decides where new code
  may go is in a test, and the map from old names is in the git log.
- Fix:
  1. Add about five lines under CLAUDE.md "Project Structure": the layer order, what each layer may import, and that
     `test_import_layering.py` enforces it.
  2. Add a "Moved modules (2026-10)" table, generated from `git diff -M --name-status 470a393 HEAD -- src`, to `docs/README.md`
     (or CLAUDE.md), mirroring the scripts table.

  Both are docs-only edits. The CLAUDE.md edit needs the lead's approval.

### Minor

- **M1. Task 16 brought three ruff I001 regressions.** BASE 470a393 passes `ruff check src scripts tests`; HEAD has 3 unsorted
  import blocks. The `schemas` import now sorts before `tables`. Files:
  - `scripts/tables/build_timebin_tables.py:17`
  - `scripts/tables/regenerate_stop_table.py:19`
  - `scripts/exploratory/_lagged_reward_kernel.py:24`

  Fix: `ruff check --fix` on those three files (import order only).
- **M2. The script-import guard cannot see two forms of script-to-script import.**
  - Path loads:
    - `scripts/rate/plot_uncertainty_rate.py:52`, a pipeline plotter that `exec_module`s the whole fitter for
      `POPULATIONS_TO_FIT` / `FIT_INPUTS`;
    - `scripts/exploratory/_audit_covariate_accuracy.py:59`;
    - `scripts/exploratory/_cs5_response_by_design_factorial.py:13`.
  - Bare sibling imports:
    - `scripts/tables/assemble_contfrag_shards.py:11`;
    - `scripts/tables/refresh_contfrag_behavioral_covariates.py:21` (already broken at BASE and HEAD).

  Fix:
  1. Move `POPULATIONS_TO_FIT` / `FIT_INPUTS` of `fit_uncertainty_rate.py` to `src` (beside `MU_UNCERTAINTY_CURATED_COVARIATES`).
  2. Extend `test_no_new_script_imports.py` to flag `spec_from_file_location` on a `scripts/` path, and bare imports of a
     script basename, with the remaining cases in `ALLOWED`.

  This can be a follow-up.
- **M3. Three names collide or nearly collide.** A researcher will open the wrong file.
  - **SWR inventory.** There are three things with this name: `analysis.events.swr_inventory` (the paper's SWR source read),
    `analysis.hippocampus.swr_inventories` (the four direct detectors that write the inventories), and the
    `inventory_comparison.adapters.swr_inventory` function. The function is already imported `as source_inventory` in 4 files.
  - **Detection.** `analysis.events.detection` is non-local event detection and SWR-overlap classification, while
    `scripts/detection/` holds the SWR detectors and the detection comparison.
  - **Schema.** `schemas.py` holds the TypedDicts. `inventory_comparison/schema.py` and `tables/` hold the table schemas
    (CLAUDE.md says `tables/` holds "schema").

  Fix: a one-line "not to be confused with …" in each module docstring. No rename is needed; the plan named
  `swr_inventory` and `schemas`.
- **M4. `model_specs` now holds row filters and imports pandas** (`model_specs.py:40-44`, `:1503-1560`). For the lead's WIP this
  is acceptable: nothing it imports was renamed, and the module stays JAX/statsmodels-free (it is in `JAX_FREE`, and I checked
  in a fresh interpreter). But `restrict_to_leaf_arms` and `leaf_only_cohort` filter rows; they do not define specs. They would
  sit naturally in `content_response`, which already owns `renormalize_fraction_over_candidates`. That move would also drop the
  `model_specs` → `content_response` edge. `rung_predictors` can stay. Optional follow-up, a pure move.
- **M5. `contrast_aggregation.py` is now 1,139 lines in two halves.** Lines 1–546 are the contrast aggregation shared by rate and
  content; lines 547–1139 assemble the content-contrast cohort. Optional follow-up pure move of the second half to
  `analysis/replay/contrast_content_cohort.py`, as the Task 12 report suggests. Not needed for the merge.
- **M6. `session.py` imports private names from sibling modules.**
  - Where: `analysis/pfc/session.py:23,33-36`.
  - What: `_check_fitter`, `_fit_diagnostics`, `_stack_fit_diagnostics`, `_fit_immobility_models` and `_make_design_matrices` come
    from `fit` / `design`. `pfc/__init__` also re-exports two of them under public aliases.
  - Fix: since they now cross module boundaries, make them public in a separate rename commit. Follow-up.
- **M7. Strict overlap, the `is_swr` convention, is not in the interval module.** `intervals.py`'s docstring calls the module
  "the one home of the project's interval primitives", but its conventions table points out of the module, to
  `events.detection.strict_overlap_matrix` (`detection.py:141`). It is a pure array primitive. Optional: move it into
  `intervals.py`; importers rewrite and nothing else changes. Follow-up.
- **M8. A docstring promises a patch point that does not work.** `analysis/events/swr_inventory.py:153-156`
  (`swr_inventory_dir`) says "under ``paths.SWR_INVENTORY_ROOT``, read when called". The module binds the name at import
  (`:76`), so a patch on `paths` does not reach it; tests correctly patch `swr_inventory.SWR_INVENTORY_ROOT`. Reword to "this
  module's ``SWR_INVENTORY_ROOT`` (from ``paths``)". The same applies to `artifact_intervals_dir` and `ARTIFACT_INTERVALS_DIR`.
- **M9. `git log --follow` does not reach the history of `pfc/session.py`.** Git does not detect the glm.py → session.py
  rename; `--follow -M30%` does (checked). Add "formerly `analysis/pfc/glm.py`; `git log --follow -M30%`" to the session.py
  module docstring or the moved-modules table (I2). No history rewrite.
- **M10. The stdlib-shadow test checks only the top-level package directory.**
  - It does not see `tables/io.py`, `runs/io.py` (both new in this branch), `state_evidence/io.py` or
    `memoryless_sequence/statistics.py`.
  - The `io` modules are harmless: `io` is loaded during interpreter startup, before `sys.path` is consulted.
    `cd src/continuum_swr_replay/tables && python -S -c "import io"` resolves to the stdlib.
  - `statistics.py` does shadow (verified), but it is pinned by Q5.
  - Optional: walk every package directory, with an allowlist of `{io: preloaded, statistics: Q5 pickles}`, so no new
    shadowing name can land.

## Rulings on the open items

1. **The `trial_rail` panel bug: confirmed, pre-existing, not a merge blocker.**
   - Evidence: `chimi20200212_02_r1_HPC.pkl` (v4 cache) has `trials.index == [1, 2, 3, …]`
     (`trial_number`) and `port_experiences.movement_trial_number == [NA, 1, 2, …]`.
   - `session_tables/responses.py:161` does `reset_index(drop=True)`, which renumbers from 0, before
     `trial_context_at_times` compares the index with `movement_trial_number` (`covariates/context.py:161-167`).
   - So on real data the `trial_rail` panel always raises.
   - Ruling: fix it on its own branch after the merge. It is a behaviour change, so it does not belong in a refactor branch.
     Write a failing test first, with a fixture whose trial ids start at 1, as the cache does. Then keep the trial ids instead
     of resetting the index, and check that the later `iterrows` loop reads position, not label.
2. **The `lfp`-style local copies: none of the kind M7 removed remain in `src`. Leave the rest.** What is left:
   - script-local scalar helpers (`sweep_memoryless_detection.overlaps`);
   - `generate_event_gallery._intervals_overlap_any`, which only wraps `strict_overlap_matrix`;
   - pairwise operations of a different kind (`inventory_comparison.overlap.overlap_pairs`, `heldout.overlapping`,
     `components.overlap_components`);
   - `qa/compare_memoryless_to_reference.overlap`. This one **must stay independent**: it belongs to a parity check against an
     independent re-implementation, and routing it through `intervals` would make the check compare the code with itself.
   - The only consolidation worth doing is M7 (moving the strict primitive into `intervals.py`), and that is optional.
3. **The narrower NWB-name check placement (`session_data_over_time_path`): accept.** It restores exactly what Task 11 removed.
   Moving it into `format_epoch_identifier` costs edits to 11 test files with 1-digit fake dates, for no gain in this branch.
   A malformed name still fails loudly later, with FileNotFound. Follow-up if wanted: switch the fixtures to 8-digit dates,
   then apply the saved wide diff.
4. **The atomic writer that `tables/io` keeps for itself: accept.**
   - `runs` sits above `tables` (`runs/io.py:30` imports `tables.io`), so `tables` cannot call `runs.io.write_atomically`.
   - Deduplicating would need a new foundation node for about 15 lines.
   - Both writers share `umask_file_mode`, and `runs/io.py:96` already points at it.
5. **The PFC audits in the pipeline map: yes.** Make `scripts/pfc` a mapped root, excluding the runner (I1).
6. **The 4 committed CSVs that keep old `produced_by` rows: accept.** They are records of past runs, each beside its
   `git_commit`. `scripts/README.md` "Moved scripts" says so.
7. **Phase-B leftovers M8 (int default under a `float` annotation) and M11 (`SWR_INVENTORY_ROOT` keeps `_ROOT`): leave.**
   Neither changes any output.
8. **Whether `overlaps_dropped_events` should become strict: leave it.** It differs from strict in 0 cases across the cohort
   (Task 4).

## Fix in the final wave (ordered)

1. **I1:** add `scripts/pfc` to `ROOTS`, exclude `run_pfc_analysis_bandit.py`, regenerate the map, and diff the five restored
   entries against the 8f690706 map.
2. **I2:** write the moved-modules table into `docs/README.md` (generated from `git diff -M --name-status 470a393 HEAD -- src`),
   including the glm → six-module split and the M9 note. Add the layering paragraph to CLAUDE.md "Project Structure"
   (lead approval).
3. **M1:** `uv run ruff check --fix` on the three files; confirm `ruff check src scripts tests` is clean.
4. **M8:** reword the two `*_dir()` docstrings in `swr_inventory.py`.
5. **M3:** add one-line "not to be confused with" notes to the module docstrings of `events/swr_inventory.py`,
   `hippocampus/swr_inventories.py`, `events/detection.py`, `schemas.py` and `inventory_comparison/schema.py`.
6. Optional in this wave, otherwise follow-ups:
   - M2 (guard extension, and the `fit_uncertainty_rate` constants into `src`);
   - M10 (shadow test over all directories).

Out of this wave, as follow-ups:
- M4, M5 and M7 (pure moves);
- M6 (making the private names public);
- the `trial_rail` fix on its own branch.

## Declined to judge

- **Re-running the golden tables, the PFC CPU runs and the render comparisons.** The ledger and task reports have the logs; I
  verified the code text instead, which is what those runs vouch for, and the full suites are running elsewhere.
- **Whether the 21 allowlisted detection and memoryless script imports should be refactored now.** That is a subsystem
  refactor; the plan recorded them as exceptions, and their list only shrinks.
- **The implementer's placement calls** (`analyze_event_spike_evidence` → `detection/`, `refresh_contfrag` → `tables/`, the
  `_report` split). I agree with all three, and none needs a ruling.
- **Rewriting history** for the glm → session rename or for Task 8's mixed commit. It is not worth rewriting 50 commits; M9
  covers discoverability.
- **Phase A/B rulings** already made by those reviews and the ledger (`MODEL_DIRS` removal, `require_swr_source=`, the private
  names in `analysed_time`, `replace_outputs` in `runs.io`, `figure_utilities` outside the foundation). Not re-argued.
- **Pre-existing broken scripts** (`refresh_contfrag_behavioral_covariates` imports, `_univariate_coefs` →
  `analysis.replay.rate_model`). They fail at BASE and at HEAD; that is out of scope.
- **`tests/README`'s "tests mirror src" convention** while most tests are still in `tests/unit/`. The README itself says the
  migration is gradual.
- **The `real_data` suite at HEAD.** It is running elsewhere; I had no result to judge.
- **Import-time cost of `model_specs` beyond "no heavy dependencies".** I did not time it; pandas is already loaded by every
  caller.
- **The logger name change** (`…pfc.glm` → `…pfc.<module>`). Nothing configures those loggers (Task 14 grep), and the
  per-session loggers are named by epoch.
- **The content of the lead's untracked WIP.** I checked only that its imports still resolve.
- **The scientific content of the model ladders and fits.** The branch does not touch them.

## Assessment

**Ready to merge: with fixes.** There are no correctness defects. Every split is verbatim, the decomposition preserves
behaviour, the layering allowlists are empty, and the analysed-time, strict-overlap, `event_id` and junction-bin paths are
unchanged. The fixes that should land first are I1 (restore the lost pipeline-map coverage), I2 (write down the layering and
the old→new module map) and M1 (3 ruff regressions). All three are small docs and tooling edits; M4–M7 and the `trial_rail`
bug are follow-ups.
