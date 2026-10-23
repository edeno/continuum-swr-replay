# Code Organization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

Status: complete, not merged (branch `code-organization-2026-10-19`, from master 470a393). Ledger: ledger.md; final review: final-review.md; Phase D review: phase-D-review.md.

**Goal:** Give the package a foundation layer. Shared infrastructure stops living inside leaf analysis packages, the six package-level import cycles go away, and a researcher can find each rule in one place.

**Architecture:**
- **Shared code moves first.** Shared code moves out of the leaf analyses into a set of foundation homes:
  - `intervals.py`, `paths.py`, `tables/`, `runs/`, `decoding/`;
  - an `analysis/events/` package for the analysed-time and event substrate.
- **Each move is followed by a byte-identical check.** Every task moves code, rewrites every import in the repo, and proves the golden tables did not change.
- **Then the remaining work.** Guards land first so each move can only shrink their allowlists. Then come the moves, then scripts-as-libraries, the scripts regroup and the god-module splits.

**Tech Stack:** Python 3.11, uv, pytest, ruff, `ast` (static import graph). Golden-table parity is checked with `continuum_swr_replay.table_parity.compare_table_roots`; that module moves to `tables/parity.py` in Task 5.

**Spec:** `.claude/docs/reviews/2026-10-19-code-organization-review.md` (findings F1–F12, the suggested sequence, and Q1–Q9). The lead's answers are under Decisions.

## Decisions (lead, 2026-10-19)

- **Event substrate:** option (a). Promote it to a new `analysis/events/` package.
- **Moved scripts (Q1):** record the new path (`PRODUCED_BY` and friends), with no stub at the old path. `scripts/README.md` gets an old→new path table.
- **Real-data tests (Q6):** opt-in everywhere. A `real_data` marker skips them unless they are run with `-m real_data`.
- **`docs/audits/` (Q3):** move it to `.claude/docs/reviews/` and update every link.
- **Controller defaults, which the lead accepted without objection:**
  - Q4: test-pinned exploratory scripts stay where they are, as documented exceptions.
  - Q7: the `models_per_bin_rate/` rename stays deferred.
  - Q8: the tracked `.claude/projects/.../memory/` copy is left alone.
  - Q9: `compare_fits.py` and `compare_pfc_cohorts.py` stay scripts; only logic other code imports moves to `src`.
- **Q5 (checked by the controller).** 706 memoryless candidate pickles under `Processed-Data/replay_detection_comparison/memoryless/` reference the modules `analysis.hippocampus.memoryless_sequence.config` and `.detect`. No fitted model bundle references a package class. So those two modules, and the classes they define, do not move.
- **No re-export shims.** Every in-repo import (`src/`, `scripts/`, `tests/`, `notebooks/`, docs) is rewritten to the new path. The only exception is a module that pickled artifacts reference (Q5), which keeps an alias module.

## Global Constraints

- **Byte-identical tables.** Every task that touches a table builder, the read point, an interval primitive or an import used on the table path must reproduce the 5-session golden tables exactly.
  - The rebuild commands are in the SDD workspace `global-constraints.md`.
  - Compare against `/tmp/claude-39494/-cumulus-edeno-continuum-swr-replay/abb6ddbf-eeb3-43b5-9505-88838c9b5af2/scratchpad/golden_e70ab12/rc` with `compare_table_roots`, at atol=0, for all 7 rebuilt tables.
- **Moves use `git mv`,** so rename detection keeps history. A move and a behaviour change never share a commit.
- **Rewrite imports with a script, not by hand.** Each move task keeps its codemod in scratchpad and reports its output. After the codemod, check with grep that the old dotted path appears nowhere in tracked files. Docs and docstrings count as tracked files.
- **Grep bare references too.** Grep tracked `*.md` and docstrings for bare `<old_module_basename>.` references, such as `swr_source.with_swr_source`. Rewrite those that name the moved module, and list them in the report. Phase-A review I2 found that `test_doc_paths_resolve` cannot see these references.
- **These paths do not move:**
  - `analysis.hippocampus.memoryless_sequence.config` and `.detect` (pickled, Q5);
  - `analysis.pfc.load_population`, `analysis.replay.model_specs` and `analysis.replay.per_bin_design`, which the lead's untracked WIP `scripts/figures/generate_replay_story_figures.py` imports.
- **Do not touch the untracked WIP files:**
  - `scripts/figures/generate_replay_story_figures.py`;
  - `tests/integration/test_replay_story_figures.py`;
  - `paper/figures/replay_story/`;
  - `.claude/docs/plans/{finish-covariate-rename,representation-selection}/`;
  - `docs/reference_*_2026-09-17.md`;
  - `.vscode/`.
- **Known baseline failures:** the 7 tests in `test_replay_story_figures.py` (×5) and `test_session_keys.py` (×2). Both come from the untracked WIP script.
- **Full suite.** After Task 2, a full run is `uv run pytest -q -p no:cacheprovider` plus `uv run pytest -q -p no:cacheprovider -m real_data`. There is no `-n`; xdist is not installed.
- **Environment.**
  - Use `uv run`.
  - Never write into `Processed-Data/`.
  - No GPU decodes; the GPUs are shared.
  - Never `git stash`, `reset --hard`, `checkout -- <file>` or `clean`.
- **Layering.** After this plan, `tests/unit/test_import_layering.py` enforces:
  - foundation (`intervals`, `paths`, `tables`, `runs`, `decoding`, `parameters`, `types`/its rename, `data_loaders`) imports no `analysis`, `visualization` or `scripts`;
  - `analysis.events` imports only foundation and itself;
  - `analysis` imports no `visualization`;
  - `src` imports no `scripts`;
  - `hippocampus` imports no `pfc` or `inventory_comparison`;
  - there are no package-level cycles.
- **Commits.** Every commit message ends with the two attribution lines from the SDD implementer instructions.

## Review Focus

1. **Lazy `__init__`s and import-time side effects.**
   - The risk: a module that sets global state at import (`jax.config.update("jax_enable_x64", True)`, warning filters, matplotlib backends) may stop running before the computation that relied on it.
   - What should happen: lazy `__init__`s change no numbers.
   - The test: Task 1 greps for module-level `jax.config`, `warnings.filterwarnings` and `matplotlib.use` reached through the made-lazy packages. It also runs the golden tables.
2. **Merging interval primitives that differ in subtle ways.**
   - The risk: closed, half-open and strict overlap primitives, and the two `json_ready` variants, can disagree on edge cases.
   - What should happen: merging never silently changes semantics. Each convention stays a separate, named function.
   - The test: Tasks 4 and 8 run an old-vs-new check on edge cases (touching endpoints, zero-length intervals, empty input, NaN) for every replaced local copy.
3. **Scripts imported as `scripts.x.y` by code that crosses process boundaries.**
   - The risk: dask workers unpickle `DecodeRunner` by module path, subprocess-spawning tests launch scripts by path, and provenance strings record script paths.
   - What should happen: decodes and runners keep working after a move.
   - The test: Tasks 9 and 13 run a CPU `LocalCluster` worker probe that unpickles the moved runner record. Task 13 also runs a test that every `PRODUCED_BY` and `DESIGN_SPEC` string names an existing file.
4. **Docs and docstrings that cite moved paths.**
   - The risk: CLAUDE.md, `docs/` and `scripts/README.md` name module paths, and a stale reference sends a reader to a file that no longer exists.
   - The test: from Task 3 on, a test checks that every `scripts/...py`, `src/continuum_swr_replay/...py` and dotted `continuum_swr_replay.*` reference in tracked `*.md` resolves. It has an allowlist for historical plan and review docs. The pipeline map is checked with `--check`.
5. **Output paths after the `paths.py` migration.**
   - The risk: a migrated constant resolves to a different `Path` and silently reads a stale artifact.
   - The test: Task 10 asserts that every new constant equals the literal it replaced. It also runs `table_path()` over every canonical table name.

---

## Phase A — guards (no code moves)

### Task 1: Import guards and lazy package `__init__`s (F1 step 1, F11)

**Files:**
- Modify: `src/continuum_swr_replay/analysis/replay/__init__.py`, `src/continuum_swr_replay/analysis/pfc/__init__.py`. Use PEP 562 lazy re-exports, copying the pattern in `analysis/hippocampus/__init__.py:75-138`.
- Modify: `tests/unit/test_import_layering.py`. Add the rules and a known-violations allowlist, and drop the unused `bouts` entry from rule 3.
- Modify: `tests/unit/test_lazy_imports.py`. Add a check that `jax`, `non_local_detector` and `statsmodels` are not loaded.

**Interfaces:**
- Produces:
  - `KNOWN_VIOLATIONS: dict[str, set[tuple[str, str]]]` in `test_import_layering.py`, keyed by rule name, holding `(importer_module, imported_module)` pairs. Later tasks delete entries from it, never add.
  - The helper `package_graph() -> dict[str, set[str]]` over `src/continuum_swr_replay`.

- [ ] **Step 1: Record the baseline import facts**
  - With `python -X importtime`, record the cold import cost of each of these modules into scratchpad: `analysis.replay.event_ids`, `.cohort_selection`, `.covariate_registry`, `.versions`, `.standardization`, `.design_constants`, `table_parity`, `analysis.pfc.cohort`, `analysis.replay.model_specs`. Also record whether `jax` ends up in `sys.modules`.
  - Grep for module-level `jax.config`, `warnings.filterwarnings`, `matplotlib.use` and `np.seterr` in every module the replay and pfc `__init__`s import today. List them in the report.
- [ ] **Step 2: Write the layering rules as failing tests**
  - The rules: `analysis` ↛ `visualization`; `src` ↛ `scripts`; foundation ↛ `analysis`/`visualization`/`scripts`; `hippocampus` ↛ `pfc`/`inventory_comparison`; no package-level cycles.
  - The package graph uses the first two dotted components under `continuum_swr_replay`, with `analysis.<pkg>` counted as its own package. Copy the review's `ast` approach. Count imports inside functions too, since lazy imports are still edges.
  - Each rule asserts `violations - KNOWN_VIOLATIONS[rule] == set()`, and also that `KNOWN_VIOLATIONS[rule] - violations == set()`, so the allowlist cannot go stale.
  - Run the tests to see the exact violation sets, then copy those sets into `KNOWN_VIOLATIONS`. Expect the review's F2/F11 edges: `swr_inventories → pfc.glm`, `swr_inventories → inventory_comparison`, `artifacts/detect → decoding_common`, `state_evidence/io → decoding_common`, `decode_files → analysis.constants`, `table_io → hippocampus`/`replay`, `table_parity → replay`, plus the six cycles.
- [ ] **Step 3: Make `replay/__init__.py` and `pfc/__init__.py` lazy**
  - Keep `__all__` identical.
  - Add a test that `from continuum_swr_replay.analysis.replay import <name>` works for every name in `__all__`, and the same for pfc.
- [ ] **Step 4: Add the leaf import check**
  - The test: `test_leaf_modules_do_not_load_jax`, which runs in a subprocess so `sys.modules` starts clean. For each module in a `JAX_FREE` list, it imports the module and asserts that none of `jax`, `non_local_detector` or `statsmodels` is in `sys.modules`.
  - `JAX_FREE` holds the modules from Step 1 that are now JAX-free. A module that still loads JAX goes into `JAX_FREE_PENDING` with the import chain that causes it.
- [ ] **Step 5: Verify**
  - Golden tables at atol=0.
  - Targeted tests: `test_import_layering.py`, `test_lazy_imports.py`, and every test file that imports `analysis.replay` or `analysis.pfc` at package level (find them with grep).
  - Report each Step 1 module's import time before and after.
- [ ] **Step 6: Commit** with the message "Replay and PFC packages import lazily; the import layering is enforced".

### Task 2: Test markers, real-data opt-in, tests README (F9 small items)

**Files:**
- Modify: `pyproject.toml` (`[tool.pytest.ini_options]`): register the `real_data` and `integration` markers.
- Modify: `tests/conftest.py`. Create it if it is absent; otherwise extend the root conftest. Skip `real_data` tests unless the `-m` expression selects them.
- Modify: every test that reads `Processed-Data`, `PROCESSED_DATA_DIR`, `/stelmo` or the Spyglass DB. Find them by grepping the review's list (F9) and every `skipif(not ...exists())` guard in `tests/`.
- Modify: every tracked file in `tests/integration/`. Add `pytestmark = pytest.mark.integration`, except in `test_replay_story_figures.py` (untracked WIP).
- Modify: `tests/README.md`.

- [ ] **Step 1: Write the failing marker test**
  - `tests/unit/test_markers.py` collects a tiny temporary test file with `pytester` (or a subprocess `pytest --collect-only -q`).
  - It asserts that a `@pytest.mark.real_data` test is skipped by default with reason "real_data: opt in with -m real_data", and that it runs under `-m real_data`.
- [ ] **Step 2: Implement the conftest hook**
  - In `pytest_collection_modifyitems`, check `config.getoption("-m")`. When the expression does not mention `real_data`, add a skip marker to each `real_data` item.
- [ ] **Step 3: Apply the markers**
  - Turn each existing silent `skipif(not path.exists())` into `@pytest.mark.real_data`. Under `-m real_data`, missing data must fail loudly rather than skip; that is the point of opting in.
  - List every test you marked in the report, with its old guard.
- [ ] **Step 4: Rewrite `tests/README.md`**
  - Give the real test count and runtime, taken from the last full run: about 31 min for 5,128 tests.
  - Describe the folder layout and the policy: mirror `src` and move tests when a module is touched, with no big-bang move. Tests of scripts go under `tests/scripts/` going forward.
  - Describe the two markers and how to run them.
- [ ] **Step 5: Verify**
  - Two full-suite runs: default, and `-m real_data`.
  - Default: the known 7 failures, minus any that are now real_data-skipped; say which.
  - `-m real_data`: every marked test passes.
  - Save both outputs.
- [ ] **Step 6: Commit** with the message "Real-data tests are opt-in; integration tests are marked".

### Task 3: Documentation homes and stale structure text (F10, F12 archive check)

**Files:**
- Move: `docs/audits/*` → `.claude/docs/reviews/` with `git mv`. Prefix a date to any file name that lacks one, using its git first-commit date. List each mapping.
- Modify: every tracked reference to `docs/audits/`, including `docs/README.md:180,261` and code docstrings.
- Modify: `docs/README.md`. Index the nine unindexed docs listed in F10, each with a one-line description; mark one superseded only if its own text says so.
- Modify: `CLAUDE.md`.
  - Regenerate the "Project Structure" tree from the real tree, to package depth plus one level under `analysis/`.
  - Line ~430: `PROCESSED_DATA_DIR` is defined in `continuum_swr_replay.parameters`.
  - Line ~446: "nodes 0–5 are wells", consistent with the 10-node description above it.
- Modify: the stale docstrings. These are `replay/posterior_io.py:5`, `visualization/palettes.py:6,43` and `model_specs.py:996`, plus the two dead doc references (`docs/design/2026-10-03-full-pipeline-refresh-design.md:693,750`, `docs/replay_rate_content_swr_synthesis.md:794`). Point each at the file that exists now, or mark it "(removed)" with the commit that removed it.
- Create: `tests/unit/test_doc_paths_resolve.py` (Review Focus 4). It checks every `scripts/…\.py`, `src/continuum_swr_replay/…\.py` and backticked dotted `continuum_swr_replay\.[\w.]+` in tracked `*.md` files outside `.claude/docs/plans/`, `.claude/docs/reviews/` and `docs/known_issues/` (historical). A dotted name must resolve to a module, or to a module attribute found statically with `ast`. Paths that do not resolve today go into an allowlist, which the task works down to empty or justifies line by line.
- Check: `scripts/figures/archive/` references from `generate_figure02.py`, `generate_all_figures.py`, `figure_panels.py` and three tests. Mentions are fine. Report any import.

- [ ] **Step 1:** Write `test_doc_paths_resolve.py` and run it to see the failures, which are the current dead references.
- [ ] **Step 2:** Do the `docs/audits` move, update the links, and run the test again.
- [ ] **Step 3:** Make the CLAUDE.md, docs/README.md and docstring fixes. The test must pass with the allowlist empty, or with each remaining entry justified in a comment.
- [ ] **Step 4:** `uv run python scripts/qa/write_pipeline_map.py --check`.
- [ ] **Step 5: Commit** with the message "Docs live in their declared homes and every cited path resolves".

## Phase B — foundation layer

### Task 4: One interval algebra in `intervals.py` (F2 intervals, F6 binning helper)

**Files:**
- Create: `src/continuum_swr_replay/intervals.py`. It may import only numpy, pandas and the standard library.
- Move into it:
  - the generic interval functions of `analysis/hippocampus/artifacts/masks.py` (`merge_intervals`, `subtract_intervals`, `overlaps_intervals`, `covered_seconds`, and `masked_seconds` if it is generic). The excluded-time padding rule and anything that names artifacts stays in `masks.py`, which moves in Task 6;
  - every function of `analysis/inventory_comparison/intervals.py` (`normalize`, `union`, `intersect`, `subtract`, `total_duration`, `contains`). Delete that file;
  - `get_spikecount_per_time_bin` from `analysis/pfc/glm.py:49`.
- Replace the local copies with calls: `data_loaders/lfp.py:758` `_overlap_any`; `pfc/glm.py:1957` `_contiguous_true_runs`, unless it is the same as `quality_metrics._run_slices` or `_contiguous_true_runs` elsewhere; `pfc/glm.py:2002` `_overlaps_any`.
- Each overlap convention keeps its own name, and the conventions sit side by side: `overlaps_closed`, `overlaps_half_open`, and the strict, positive-duration one. `event_detection.strict_overlap_matrix` stays in events (Task 6), with its docstring pointing at `intervals` for the other two.
- Test: `tests/unit/test_intervals.py`.

**Interfaces:**
- Produces: `continuum_swr_replay.intervals` with the functions above under their existing names, so no call site has to change semantics. The two families keep their array and DataFrame signatures, and the module docstring has a table of conventions (closed, half-open, strict).
- `hippocampus/swr_inventories.py` now imports `get_spikecount_per_time_bin` from `intervals`. That deletes the `hippocampus → pfc` edge from `KNOWN_VIOLATIONS`.

- [ ] **Step 1: Write edge-case tests for every moved and replaced function**
  - Cover touching endpoints, zero-length intervals, nested intervals, unsorted input, empty input and NaN bounds.
  - Run each replaced local copy (`lfp._overlap_any`, `glm._overlaps_any`, `glm._contiguous_true_runs`) through the same cases in scratchpad, against the old copy (from `git show HEAD:<file>`).
  - If any replaced copy disagrees with the canonical function, stop and report it.
- [ ] **Step 2:** Use `git mv` where a whole file moves (`inventory_comparison/intervals.py` → `intervals.py`), then append the masks functions and the binning function.
- [ ] **Step 3:** Run the codemod over `src/`, `scripts/`, `tests/`, `notebooks/` and `*.md`. Then grep that none of the old dotted paths of the moved names remain.
- [ ] **Step 4:** Update `KNOWN_VIOLATIONS` (deletions only), check `test_lazy_imports` (`intervals` must not load dask or JAX), and regenerate the pipeline map.
- [ ] **Step 5: Verify**
  - Golden tables.
  - PFC: binning parity on the five golden sessions' spike times. Compare the counts from the old `glm.get_spikecount_per_time_bin` (via `git show`) with the new one, exactly.
  - Targeted tests.
- [ ] **Step 6: Commit** with the message "One interval algebra; spike binning leaves the PFC GLM".

### Task 5: The `tables/` schema layer (F1 step 2, F2 tables)

**Files:**
- Create the package `src/continuum_swr_replay/tables/` and move these into it with `git mv`:
  - `table_io.py` → `tables/io.py`;
  - `table_parity.py` → `tables/parity.py`;
  - `session_keys.py` → `tables/session_keys.py`;
  - `analysis/replay/covariate_registry.py` → `tables/covariate_registry.py`;
  - `analysis/replay/versions.py` → `tables/versions.py`;
  - `analysis/replay/event_ids.py` → `tables/event_ids.py`;
  - `analysis/replay/standardization.py` → `tables/standardization.py`, if it imports nothing from `analysis` (check first; if it does, leave it and report why).
- `tables/` imports nothing from `analysis`. Two back edges have to go:
  - `table_io.py:118` imports `swr_source` lazily. Find what it uses. If it is the SWR-source stamp, move that one function into `tables/` or invert the call so the caller passes it. If neither is a clean move, stop and report.
  - `table_io.py:122` imports `covariate_registry` lazily. That becomes an intra-package import.
- `tables/session_keys.py`: drop the literal-plus-test workaround at `session_keys.py:19-24` and import `covariate_registry` directly, now that it is light. Keep `SESSION_COLUMNS` value-identical.

**Interfaces:**
- Produces: `continuum_swr_replay.tables.{io, parity, session_keys, covariate_registry, versions, event_ids, standardization}`, keeping all public names. `compare_table_roots` is now `tables.parity.compare_table_roots`.

- [ ] **Step 1:** Record `CANONICAL_TABLE_BASENAMES`, `SESSION_COLUMNS`, the covariate-registry key set and `versions.*` values into scratchpad.
- [ ] **Step 2:** `git mv`, run the codemod, and grep that the old paths are gone everywhere.
- [ ] **Step 3:** Remove the two back edges, and add `tables` to the layering rules' foundation set.
- [ ] **Step 4:** Add `tables.*` to `JAX_FREE` in the lazy-import test. Move entries from `JAX_FREE_PENDING`.
- [ ] **Step 5: Verify**
  - The Step 1 values are identical.
  - Golden tables. The comparison now imports `tables.parity`, so first run it once against the golden root to confirm that a `{}` comparison still means equal.
  - Targeted tests, and the pipeline map.
- [ ] **Step 6: Commit** with the message "Table schema, ids and parity form a foundation tables/ package".

### Task 6: The `analysis/events/` package (F2, lead's option (a))

**Files:**
- Create the package `src/continuum_swr_replay/analysis/events/` and move these into it with `git mv`, in one commit with no edits beyond imports:
  - `hippocampus/swr_source.py` → `events/swr_source.py`;
  - `hippocampus/event_eligibility.py` → `events/eligibility.py`;
  - `hippocampus/event_immobility.py` → `events/immobility.py`;
  - `hippocampus/event_detection.py` → `events/detection.py`;
  - `hippocampus/event_table.py` → `events/classified_table.py`;
  - `hippocampus/epoch_normalizers.py` → `events/epoch_normalizers.py`;
  - `hippocampus/artifacts/masks.py` → `events/artifact_masks.py` (only the part left after Task 4);
  - `hippocampus/artifacts/io.py` → `events/artifact_intervals.py`. The detector, `hippocampus/artifacts/detect.py`, keeps writing through it;
  - `replay/cohort_selection.py` → `events/cohort_selection.py`.
- Modify: `analysis/hippocampus/__init__.py` lazy re-exports. Drop the moved names; don't re-export them.
- Modify: CLAUDE.md, and every doc and docstring that names these modules.
- Cut the remaining `hippocampus → inventory_comparison` edge: `swr_inventories.py:115-116` imports `inventory_comparison.overlap` and `.schema`.
  - First find what it uses. If the schema it uses is the inventory row schema that `swr_source` reads too, it belongs in `events/` (`events/inventory_schema.py`).
  - If `overlap` is interval math, it now lives in `intervals`.
  - If neither fits, stop and report what `swr_inventories` needs and why.
- Layering: add a rule that `analysis.events` imports only foundation and `analysis.events`, and `analysis.constants`.

**Interfaces:**
- Produces: `continuum_swr_replay.analysis.events.{swr_source, eligibility, immobility, detection, classified_table, epoch_normalizers, artifact_masks, artifact_intervals, cohort_selection}`, each keeping its public names. `build_classified_nonlocal_event_table` is now in `events.classified_table`, and `with_swr_source` in `events.swr_source` until Task 7 splits it.

- [ ] **Step 1:** List what each moved module imports (the controller's graph is in scratchpad `orgplan/graph.py`). Confirm the moved set is closed: no moved module imports a hippocampus or replay module that stays behind, apart from what the edge cuts above remove. If a moved module needs a module that is staying, stop and report.
- [ ] **Step 2:** `git mv` everything, run the codemod over all trees and docs, and grep that the old paths are gone.
- [ ] **Step 3:** Cut the `swr_inventories` edge, update `KNOWN_VIOLATIONS` (deletions only), and add the `events` rule.
- [ ] **Step 4: Verify**
  - Golden tables.
  - `artifact_event_drops` output identical on the 5 golden sessions, as in the Task 7 site check of the previous plan: `scratchpad/task7/compare_sites.py`.
  - Targeted tests for events, hippocampus, replay builders, contfrag, inventory comparison and PFC canonical events.
  - The pipeline map, and `test_doc_paths_resolve`.
- [ ] **Step 5: Commit** with the message "The analysed-time and event substrate is its own analysis.events package".

### Task 7: Split `events/swr_source.py` by responsibility (F6 swr_source)

**Files:**
- Split `analysis/events/swr_source.py` (1,364 lines) in two:
  - `events/swr_inventory.py`: the SWR inventory read and run ids, `swr_source.py:385-747` before the move.
  - `events/analysed_time.py`: excluded intervals, artifact drops, `with_swr_source`, `read_dot_with_swr_source`, `analysed_bins`, `_frame_epoch` and the time-axis and label checks.
  - Delete `swr_source.py` afterwards. `stamp_swr_source` goes with whichever half writes the stamp.
- Modify CLAUDE.md "One analysed-time rule". The rule's intervals live in `events.analysed_time.excluded_intervals`, the padding in `events.artifact_masks`, and the eligibility rule in `events.eligibility`.

**Interfaces:**
- Produces: `analysis.events.analysed_time.with_swr_source` (and the rest of the read-point API), and `analysis.events.swr_inventory.*`. The module-level constants (`ARTIFACT_INTERVALS_RUN_ID`, the detection roots) go to whichever half uses them. If both halves use one, it goes in `swr_inventory`, and `analysed_time` imports it from there.

- [ ] **Step 1:** Classify every top-level name in `swr_source.py` into one of the two halves, as a table in the report. A function both halves call goes in `swr_inventory` (the lower layer). `analysed_time` imports `swr_inventory`, never the reverse.
- [ ] **Step 2:** Make the split. Code moves with no edits beyond imports.
- [ ] **Step 3:** Run the codemod and grep.
- [ ] **Step 4: Verify** golden tables, the site check, targeted tests, the pipeline map and the doc paths.
- [ ] **Step 5: Commit** with the message "The SWR inventory read and the analysed-time read point are separate modules".

### Task 8: `runs/` run infrastructure (F3)

**Files:**
- Create `src/continuum_swr_replay/runs/` (foundation; it imports no `analysis`):
  - `runs/io.py`: one `write_atomically`, `write_json`, `json_ready` and `sha256_file`/`array_sha256`. Take them from `decoding_common.py:423-518`, `state_evidence/io.py:23,48` and `run_swr_detection.py:394,401`, and replace the atomic writers in `table_io` (now `tables/io.py:198`), `replay/prepare_data.py:338-357` and `artifacts/io.py:206` (now `events/artifact_intervals.py`) with calls to it.
  - `runs/provenance.py`: one git and provenance record, from `decoding_common.py:546-580`. `replay/fit_provenance.py`, `decoding/write_regeneration_manifest.py:58`, `qa/run_refits.py:241` and `qa/sweep_memoryless_detection.py:731` call it.
  - `runs/cli.py`: `parse_run_id`, `RUN_ID_PATTERN`, `add_common_args`, `filter_epoch_info`.
  - `runs/batch.py`: `run_batch`, `run_epochs`, `run_lock`, `run_epoch_logged`, and logger setup. Take them from `decoding_common.py:341-408,519-878`, `run_swr_detection.py:1058,1178`, `run_memoryless_detection.py:944`, `qa/run_state_evidence.py:1186`, and the five `_run_epoch_logged` copies.
- Migrate the four runners off `scripts/decoding/run_swr_detection.py` and `run_memoryless_detection.py` as libraries: `run_artifact_detection`, `run_carey_detection`, `qa/run_inventory_comparison`, `qa/run_event_features`. They import `runs.*`, and any detection-domain helper they share (`load_marks`, `marks_cache_path`, `MARKS_RULE`, `input_paths`, `epoch_artifact_intervals`) moves into `analysis/events/` or `analysis/hippocampus/` as fits its imports. Make that placement call in the report, by layering.

**Interfaces:**
- Produces: `continuum_swr_replay.runs.{io, provenance, cli, batch}`. `replace_outputs(paths, outputs)` moves to `runs.io` with its rollback semantics and the subset assertion from the last fix wave unchanged.

- [ ] **Step 1: Check the variants agree before merging them**
  - For `json_ready`, re-serialise every existing JSON sidecar and manifest under `Processed-Data/replay_detection_comparison/` (read-only: load, pass through the old and new `json_ready`, compare the dumps). Also run constructed cases: Mapping, ndarray, numpy scalars, NaN, Path, datetime.
  - For each `run_epochs` and `_run_epoch_logged` variant, write down its differences (lock, logging, error handling) before merging. If two variants differ in behaviour a runner relies on, keep it as a named option, not a silent default.
  - If anything is not identical, stop and report.
- [ ] **Step 2:** Create the `runs/` modules, using `git mv` where a whole module moves. Run the codemod and grep.
- [ ] **Step 3:** Migrate the runners, and delete the duplicates.
- [ ] **Step 4: Verify**
  - Simulated manifest bytes: re-run the previous plan's `scratchpad/task1/bytes_check.py` against the new code.
  - Golden tables (`prepare_data`'s writer changed).
  - Targeted tests for every runner, and the layering and lazy-import tests.
- [ ] **Step 5: Commit** with the message "One run infrastructure: atomic writes, provenance, CLI and batch live in runs/".

### Task 9: The `decoding/` package, and `decoding_common` dissolved (F2 decoding, F3, F8 results reader)

**Files:**
- Create `src/continuum_swr_replay/decoding/` and fill it:
  - `decode_files.py` → `decoding/files.py`;
  - `decoding_transitions.py` → `decoding/transitions.py`;
  - `DecodingConfig` and friends → `decoding/config.py`;
  - one `results.nc` reader → `decoding/results.py`. It merges `replay/posterior_io.py` (move it) with `hippocampus/result_loading.py:141-144`'s open plus `state_bins` rebuild. Route the other raw opens to it: `state_evidence/io.py:115`, `scripts/qa/run_state_evidence.py:447`, `prepare_contfrag_replay_content_data.py:226`, `generate_inventory_examples.py:445`.
- Move the HPC decode-mask functions (`hpc_artifact_intervals`, `hpc_decode_masks`, `decoding_common.py:202-340`) to `analysis/hippocampus/decode_masks.py`. They read events, so they cannot live in foundation.
- `hippocampus/artifacts/detect.py:80,190`: read `parameters.WAVEFORM_STD`, not `DecodingConfig`. That deletes the `analysis → decoding_common` edge.
- Delete `decoding_common.py` once it is empty, with every caller rewritten.
- `scripts/decoding/_decode_runner.py` → `src/continuum_swr_replay/decoding/runner.py`, if nothing in it is CLI.
  - The dask worker must unpickle `DecodeRunner` from its new path. Probe this with a CPU `distributed.LocalCluster(n_workers=1, processes=True)` from `cwd=/tmp`, using no GPU (Review Focus 3).
  - Drop the runners' `sys.path` repo-root insertion if nothing else needs it.
- `decoding/` imports nothing from `analysis`. `analysis.constants.DEFAULT_NON_LOCAL_THRESHOLD`, which `decode_files` imports, moves to `parameters.py`, and `analysis/constants.py` re-exports it from there. Check whether `analysis.constants` already wraps `parameters` (F8 says it does at lines 44 and 52), and follow that pattern.

**Interfaces:**
- Produces:
  - `continuum_swr_replay.decoding.{files, transitions, config, results, runner}`;
  - `analysis.hippocampus.decode_masks.{hpc_artifact_intervals, hpc_decode_masks}`;
  - `decoding.results.open_results(path) -> xr.Dataset`, with the `state_bins` multi-index rebuilt.

- [ ] **Step 1: Prove the results reader merge is safe**
  - Open each of the 5 golden sessions' `_results.nc` (bandit and contfrag) with every old reader and with the new one. Assert `xr.Dataset.identical` (read-only).
  - If the readers differ, stop and report.
- [ ] **Step 2:** `git mv`, run the codemod and grep. `decoding_common` must be gone.
- [ ] **Step 3:** The worker probe.
- [ ] **Step 4: Verify**
  - Golden tables.
  - Path parity: re-run the previous plan's `scratchpad/task9` path dump, old vs new, for all 242 epochs.
  - The `test_decoding_save_dir` and runner tests, the layering and lazy-import tests, and the pipeline map.
- [ ] **Step 5: Commit** with the message "Decoding config, files, transitions, results reader and runner live in decoding/".

### Task 10: `paths.py`, one place for output locations (F5)

**Files:**
- Create: `src/continuum_swr_replay/paths.py` (foundation). It holds:
  - `REPLAY_CONTENT_DIR`, `CONTFRAG_REPLAY_CONTENT_DIR`, `DETECTION_COMPARISON_DIR`, `ARTIFACT_INTERVALS_DIR`;
  - `MODEL_DIRS: dict[str, Path]`, one per `models_*` subdirectory in use;
  - `CANONICAL_TABLE_BASENAMES` / `BASENAME_TO_TABLE`, moved from `tables/parity.py`, which imports them from here;
  - `table_path(name: str, root: Path = REPLAY_CONTENT_DIR, *, fmt: str = "parquet") -> Path`, which raises `KeyError` for a name outside the registry.
- Migrate:
  - every `PROCESSED_DATA_DIR / "replay_content"` (50 sites in 32 files);
  - every literal canonical-table file name (`stop_table`, `event_table`, `qvals_table`, `event_arm_table` and the rest; count them with grep);
  - every `"models_per_bin_rate"` and other `models_*` literal;
  - the PFC roots `pfc/canonical_events.py:71` and `pfc/cohort.py:29`;
  - the detection roots in `events/swr_inventory.py`, which move here and are imported back.
  - Don't migrate inside `scripts/figures/archive/`.
- Delete `replay/prepare_data.py:100`'s `OUTPUT_DIR` if nothing imports it, or point it at `paths.REPLAY_CONTENT_DIR`.
- Test: `tests/unit/test_paths.py`.

- [ ] **Step 1: Write the failing test**
  - In scratchpad, a script records each site's old literal `Path`, resolved, by evaluating the old expression.
  - `test_paths.py` asserts each new constant equals the literal it replaced: `REPLAY_CONTENT_DIR == PROCESSED_DATA_DIR / "replay_content"` and so on.
  - It asserts `table_path(n)` for every registry name.
  - It asserts a `KeyError` for an unknown name.
- [ ] **Step 2:** Create `paths.py`, then run a codemod that rewrites only exact-match expressions. Review the hits it skipped by hand, and list them in the report.
- [ ] **Step 3: Verify**
  - Every migrated site's resolved path is identical to the Step 1 record.
  - Golden tables.
  - Targeted tests, and the pipeline map.
- [ ] **Step 4: Commit** with the message "Output locations and canonical table names live in paths.py".

### Task 11: The remaining parallel definitions (F8)

**Files:**
- `data_loaders/constants.py`: add these and import them everywhere the review lists:
  - `LEAF_ARMS: tuple[int, ...] = (0, 1, 2, 3, 4, 5)`;
  - `TRUNK_ARMS: tuple[int, ...] = (6, 7, 8)`;
  - `N_LEAF_ARMS = len(LEAF_ARMS)`;
  - `port_columns(prefix: str) -> list[str]`, which returns `[f"{prefix}_port_{i}" for i in range(N_PORTS)]` where `N_PORTS` is the existing well count.
  
  The sites to change: `uncertainty_content.py:39`, `session_tables.py:114`, `trajectory.py:26`, `table_contracts.py:234`, `model_specs.py:1457` (this keeps its own name, set from `N_LEAF_ARMS`), `analyze_content_occupancy.py:88`, the 8+ `qvals_table.py` sites, `value_block.py:27-29`, `prepare_data.py:516` and `covariate_registry.py:206-214`. A site that uses a set (`{6,7,8}`) gets `frozenset(TRUNK_ARMS)`, so the type is unchanged.
- `data_loaders/epoch_register.py`: add `analysed_animals() -> tuple[str, ...]`, in register order. Replace the four spelled-out animal lists. If a list's order differs from the register's, keep that site's order and say so; order can drive plot order.
- `data_loaders/utils.py`: add `format_epoch_identifier(nwb_file_name: str, epoch: str, brain_area: str, suffix: str = "") -> str`, the inverse of `parse_epoch_identifier`. Add a round-trip test over `analysed_epochs()` × areas × variants. Replace the 9 inline formatters (F8), and route `swr_source`'s epoch-id builder (now `events/swr_inventory.py`) through it if its output is byte-identical.
- Replace the `500.0` sampling-rate defaults with `parameters.SAMPLING_FREQUENCY`, at the 12 signature sites F8 lists.
- `COLORS` / `CONTFRAG_COLORS`: move from `parameters.py` to `visualization/palettes.py`, and rewrite the importers.

- [ ] **Step 1:** Write a parity script in scratchpad that evaluates every replaced literal or inline expression, old and new, including the epoch-id formatter over all 242 epochs × variants. It must show 0 differences.
- [ ] **Step 2:** Make the changes and run the codemod.
- [ ] **Step 3: Verify** golden tables, the parity script, targeted tests and the pipeline map.
- [ ] **Step 4: Commit.** One commit per constant family is fine.

## Phase C — scripts, regroup, splits

### Task 12: Scripts stop being libraries (F4)

**Files:**
- Content-fit output naming moves to `src/continuum_swr_replay/analysis/replay/fit_outputs.py`: `canonical_output_subdir`, `canonical_output_subdirs` and `belief_chain_output_subdir`, from `scripts/content/fit_replay_content_models.py:269,350,417`. The fitter and the three importers import it from there.
- `restrict_to_leaf_arms`, `leaf_only_cohort` and `rung_predictors` (`fit_replay_content_models.py:469-524`) move to `analysis/replay/model_specs.py` next to `leaf_only_predictors`. If those functions are about cohorts rather than specs, they go to `analysis/events/cohort_selection.py` instead; make the call by what they read.
- `_assemble_arm`, `_assemble_patch` and `_prepare_session` (`scripts/content/fit_swr_contrast_content.py`) move to `analysis/replay/contrast_aggregation.py` as public names.
- The report helpers that `qa/report_event_features.py:99-109` imports from `qa/report_inventory_comparison.py` move to `scripts/qa/_report.py`, as public names.
- Find the other script-to-script imports with grep (`from scripts\.` in `scripts/`). For each, either move the shared code to `src` or to a `_`-prefixed sibling module, or record it as an exception with a reason. The exceptions include the test-pinned exploratory scripts (Q4).
- Create `tests/unit/test_no_new_script_imports.py`. It fails on any `from scripts.<path>` or `import scripts.<path>` in `scripts/` or `src/` where no component of `<path>` starts with `_`, and it carries an allowlist. Tests in `tests/` may import scripts.
- Modify `scripts/README.md`: only `_`-prefixed modules are importable by other scripts, and the allowlist is the current exception list.

- [ ] **Step 1: Pin the output locations.** Record `canonical_output_subdirs()` and every `canonical_output_subdir(...)` / `belief_chain_output_subdir(...)` value for every rung and config the fitter accepts, before and after. This is the review's F4 risk.
- [ ] **Step 2:** Make the moves, run the codemod, and grep.
- [ ] **Step 3: Verify**
  - The Step 1 values are identical.
  - The fitter's `--help` and a dry-run listing, if it has one, are unchanged.
  - Targeted tests.
- [ ] **Step 4: Commit.**

### Task 13: Regroup `scripts/` (F7, Q1 = new path, no stubs)

**Files:**
- `scripts/detection/`: `run_swr_detection`, `run_carey_detection`, `run_memoryless_detection` and `run_artifact_detection` (from `decoding/`); `run_inventory_comparison`, `report_inventory_comparison`, `run_event_features`, `report_event_features`, `run_state_evidence` and `sweep_memoryless_{detection,controls,heldout}` (from `qa/`); and the `_` helpers they need.
- `scripts/tables/`: `prepare_*`, `regenerate_stop_table`, `build_timebin_tables`, `assemble_*_shards`, `convert_tables_to_parquet` and `write_regeneration_manifest` (from `decoding/`).
- `scripts/decoding/` keeps `run_decoding_bandit`, `run_decoding_contfrag` and anything else that decodes.
- `scripts/pfc/` gains `compare_pfc_cohorts`, `compare_pfc_fitters`, `check_pfc_ridge_lambda0`, `measure_pfc_null_geometry` and `diagnose_pfc_event_definition` (from `qa/`).
- `qa/` keeps parity, audits, checks, the mutation check and the doc writers.
- Use `git mv` for every move. Update every `PRODUCED_BY`, `DESIGN_SPEC` and recorded script path to the new location.
- Add `tests/unit/test_produced_by_paths_exist.py`: every `PRODUCED_BY` / `DESIGN_SPEC` string in `scripts/` and `src/` names an existing tracked file.
- Add a "Moved scripts (2026-10)" old→new table to `scripts/README.md`, and restructure its sections to match the folders.
- Update every doc and test reference. The doc-path test from Task 3 catches misses. Run `write_pipeline_map.py --check`.

- [ ] **Step 1:** Decide the final placement of every script in `decoding/`, `qa/` and `pfc/` as a table in the report. A script that does not fit a group stays where it is, with a reason.
- [ ] **Step 2:** `git mv`, then run the codemod (imports, `scripts/...` path strings, docs). Run the subprocess-launching tests, since they launch scripts by path.
- [ ] **Step 3:** The CPU `LocalCluster` probe again, for any moved runner that dispatches dask work.
- [ ] **Step 4: Verify**
  - Targeted tests for every moved script.
  - The two new path tests.
  - The pipeline map `--check`.
  - Golden tables, because the table builders' scripts moved; rebuild using the new script paths.
- [ ] **Step 5: Commit** with the message "scripts/ is grouped by pipeline: detection, tables, decoding, pfc, qa".

### Task 14: Split the PFC GLM (F6 PFC)

**Files:**
- Split `analysis/pfc/glm.py` (3,099 lines) into:
  - `pfc/design.py`: bases and design matrices, `glm.py:149-655`;
  - `pfc/fit.py`: fitters, penalties and diagnostics;
  - `pfc/modulation.py`: deviance and modulation statistics, `:739-990`;
  - `pfc/event_sets.py`: fit masks, event sets and coverage, `:1489-1934`;
  - `pfc/matched_control.py`: `:1934-2303`;
  - `pfc/session.py`: `process_neuron` and `run_session`.
- The code moves with no edits beyond imports. Decomposing `process_neuron` (497 lines) and `run_session` (595 lines) into stages is a second commit within this task, and only if the bundle diff below stays identical.
- `pfc/glm.py` is deleted once empty. Do not move `pfc/load_population.py`, because the lead's WIP imports it.

- [ ] **Step 1: Baseline.** Pick 2 sessions from the PFC cohort that run on CPU. Run the PFC fit for them at the current HEAD into scratchpad, never into `Processed-Data`, using the existing CLI's output-dir option. If there is no such option, stop and report. Measure the runtime of one session before running the second.
- [ ] **Step 2:** The pure-move split. Run the codemod and grep. The diff of `neuron_stats` and the modulation outputs must be identical (`DataFrame.equals`, plus bitwise arrays).
- [ ] **Step 3:** Optional decomposition commit, with the same diff.
- [ ] **Step 4: Verify** the PFC tests, the layering test and the pipeline map.
- [ ] **Step 5: Commit.**

### Task 15: Split `visualization/session_tables.py` into a package (F6)

**Files:**
- `visualization/session_tables.py` (3,492 lines) becomes `visualization/session_tables/`:
  - `__init__.py` holds `plot_session_tables` and the registry;
  - one module per panel group: `responses.py`, `arm_covariates.py`, `patch_covariates.py`, `stop_covariates.py`, and `_common.py` for shared helpers.
- The code moves with no edits beyond imports.

- [ ] **Step 1:** Render the session-tables figure for 2 sessions at HEAD into scratchpad. Save the PNG pixel arrays, and the figure's panel and axis titles in the order they are drawn.
- [ ] **Step 2:** Make the split.
- [ ] **Step 3:** Re-render. The pixel arrays and titles must be identical; check matplotlib determinism by rendering twice before the split. Run the targeted tests.
- [ ] **Step 4: Commit.**

### Task 16: Rename `types.py` (F12)

**Files:**
- `git mv src/continuum_swr_replay/types.py src/continuum_swr_replay/schemas.py`. If `schemas` clashes with an existing name, use `typing_defs.py`.
- Rewrite its importers (about 16 in src and 9 in scripts and tests).

- [ ] **Step 1:** `git mv`, run the codemod and grep.
- [ ] **Step 2:** Add a test: `cd src/continuum_swr_replay && python -c "import ast"` succeeds. That is the review's repro.
- [ ] **Step 3: Verify** targeted tests and the layering test, then commit.

## Phase D — names and homes a reader can trust (lead, 2026-10-23)

The lead's goal for this branch: an outside scientist can understand the analysis by reading the code. Phase D removes three places where a name or a location says something that is not true.

### Task 17: Each output lives in the folder that names it

**Why.** `table_path()` special-cases `qvals_table`, the one canonical table stored inside the per-bin rate-model folder (`paths.py:142`). `swr_power_model` (a per-event ripple-power model) also sits in `models_per_bin_rate/`. Everything else there is a per-bin rate fit, so the folder name is right once these two leave.

**Files:** `src/continuum_swr_replay/paths.py`, `analysis/behavior/qvals_table.py`, `scripts/tables/write_regeneration_manifest.py`, `scripts/rate/fit_swr_power_model.py`, every reader of these files (grep `qvals_table`, `qvals_standardization_reference`, `swr_power_model`, `swr_power_cohort_audit`, `swr_power_post_outcome_gate_audit`, `z_reference_by_session`), `docs/README.md` ("Where results live"), the tests that pin paths.

- [ ] **Step 1: Failing tests.** `table_path(name, root)` is `root / f"{name}.{fmt}"` for every canonical table, `qvals_table` included. `model_dir("per_event_swr_power", root)` is `root / "models_per_event_swr_power"`. Every swr_power output resolves under it.
- [ ] **Step 2:** `qvals_table.parquet` and `qvals_standardization_reference.json` are read and written at the replay-content root, beside the other six tables. Delete the special case. Add `"per_event_swr_power": "models_per_event_swr_power"` to `MODEL_SUBDIRS` and move every swr_power output there. Remove the `MODEL_SUBDIRS` comment about the deferred rename; `models_per_bin_rate` now holds only per-bin rate fits and keeps its name.
- [ ] **Step 3: Provenance check.** Find every recorded path or hash that names the old locations (fit provenance JSONs, the regeneration manifest, `check_fit_provenance.py`, resume settings). A frozen record of the past stays as written. Anything compared at run time must still pass against files at the new locations; report each case and how it resolves.
- [ ] **Step 4: Verify.** GOLDEN IDENTICAL from a scratch staging root (the golden baseline is flat and unaffected). Targeted tests, layering, doc paths.
- [ ] **Step 5: The disk move.** Never write into `Processed-Data/`. Write the exact `mv` commands for the canonical root (and any staging root with these files) to the report. The lead runs them. Backups (`*.bak_*`, `*_pre*_2026*`, `_pre_*_backup_*`) stay where they are. The exploratory outputs (`contingency_table.pkl`, `contingency_csv_alpha.json`, `covariate_ranking.pkl`) stay: unsupported one-offs.

### Task 18: Names that do not collide

**Why.** Final review M3: a reader opens the wrong file.

**Renames (`git mv`, codemod, grep; one commit per rename):**
- `analysis/events/swr_inventory.py` → `analysis/events/swr_source.py`. Its docstring opens "The paper's one SWR source".
- `analysis/hippocampus/swr_inventories.py` → `analysis/hippocampus/swr_detectors.py`. It runs the four direct detectors.
- `analysis/events/detection.py` → `analysis/events/nonlocal_detection.py`. It is non-local event detection and SWR-overlap classification, not the SWR detectors of `scripts/detection/`. If its contents show that name is wrong, pick a name from what it does and report why.
- `schemas.py` → `data_types.py`. It holds the TypedDicts for loaded session data and decoder parameters, not table schemas. `inventory_comparison/schema.py` keeps its name.
- The `inventory_comparison.adapters.swr_inventory` function → `source_swr_inventory`. Four files already import it `as source_inventory`; drop those aliases.

**Kept, with the reason in a docstring:**
- `runs/io.py`, `tables/io.py`, `state_evidence/io.py`: `io` is loaded during interpreter startup, so they cannot shadow it (final review M10), and `package.io` is a common pattern.
- `memoryless_sequence/statistics.py`: renaming it breaks unpickling of the memoryless candidate pickles (Q5). It does shadow `statistics` when a tool runs from inside its directory.

- [ ] **Step 1: Pickle check first.** Scan every pickle under `Processed-Data/` (read-only) for the four old module paths, with a positive control (`memoryless_sequence`). A rename whose old path appears in a pickle stops and is reported, not done.
- [ ] **Step 2:** Each rename: `git mv`, codemod, grep for the old dotted path and the bare `<basename>.` form in tracked files, docs included. Frozen manifest strings compared on resume stay as written (list them).
- [ ] **Step 3:** Extend the stdlib-shadow test to walk every package directory, with an allowlist `{io: preloaded at startup, statistics: pickled (Q5)}`. A new shadowing module fails it.
- [ ] **Step 4:** Update the old→new table in `docs/README.md` ("Code layout"), `tests/unit/test_import_layering.py` and `test_lazy_imports.py` sets, and CLAUDE.md paths (path updates only).
- [ ] **Step 5: Verify.** GOLDEN IDENTICAL, targeted tests, layering, doc paths, ruff.

## End of plan

- Final whole-branch review (opus) → one fix wave → a scoped re-review.
- Full suite, both runs.
- `KNOWN_VIOLATIONS` must be empty, or each remaining entry must have a reason the lead accepts.
- Archive the ledger next to this plan.
- No merge to master without the lead's approval.
- **Phase D disk move (done 2026-10-23, controller, lead-approved).** `qvals_table.parquet` and `qvals_standardization_reference.json` moved to the root of `replay_content/` and of 6 staging roots; the 4 `swr_power_*` files moved to `replay_content/models_per_event_swr_power/`. Hashes and mtimes unchanged. A relative symlink sits at every old path, because the `covariate-explorer`, `phase3` and `phase4-run` worktrees share `Processed-Data/` and still read the old paths. **Hazard until cleanup:** a write through an old path replaces the symlink with a regular file (`tables/io.write_atomically` ends in `os.replace`), leaving two different copies, and nothing reports it. So no `build_qvals_table.py --write` and no `fit_swr_power_model.py` run from an older branch until the symlinks are gone. This check must print nothing (it names the 7 roots that got symlinks; backups and the frozen `replay_content_regen_staging_pre_v4_2333f07` still hold real files at the old paths by design, so a whole-tree `find` prints 53 of them): `for r in replay_content replay_content_staging replay_content_pe_staging replay_content_regen_staging replay_content_v3_staging replay_content/_staging_r13_tables_20261011 replay_content/_staging_r13c_20261011; do find "Processed-Data/$r/models_per_bin_rate" -maxdepth 1 \( -name qvals_table.parquet -o -name qvals_standardization_reference.json -o -name 'swr_power_*' \) ! -type l; done`. **Cleanup:** once those branches are on the merged master, delete the symlinks: `find Processed-Data -path '*models_per_bin_rate/*' -type l \( -name 'qvals_*' -o -name 'swr_power_*' \) -delete`. **Cleanup done 2026-10-23 (lead-approved):** the 18 symlinks were deleted, so the hazard is gone; the frozen `replay_content_regen_staging_pre_v4_2333f07` got the same move (its two qvals files to its root, hashes and mtimes unchanged, the empty `models_per_bin_rate/` removed).
