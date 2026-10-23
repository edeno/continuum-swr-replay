# Phase D review (be88764e..ee0f9ca1)

Reviewer: phase-boundary review, read-only. Inputs: task-17/18 briefs, progress.md Phase D rulings
(statistics.py renamed to window_statistics.py; io.py modules kept), task-17-18-report.md, the diff,
PLAN.md Global Constraints. Processed-Data/ was read, never written.

## Verdict: With fixes

The code change is clean and matches the briefs once the rulings are applied. No reader or writer in
src/, scripts/ or tests/ builds the old qvals or swr_power paths. No old dotted path or bare
`basename.` reference is left, apart from the "(removed: renamed)" rows. Every rename commit is a
pure move. Tests and ruff pass.

The fixes concern two things:
- Through the 18 symlinks the controller left at the old paths, some tools in this branch still read
  or write the old layout without saying so, and a write from an older branch would leave two
  different copies of a file.
- One new name (`source_swr_inventory`) says something untrue.

## What was checked

| Check | Result | Evidence |
|---|---|---|
| T17 S1: tests fail first | yes | `tests/unit/test_paths.py`: `table_path` is `root / name` for every name, qvals included. The `per_event_swr_power` literal and `test_every_swr_power_output_resolves_under_its_own_model_dir` are new. The report says 7 failed before the fix. |
| T17 S2: special case gone, new subdir, comment removed | yes | `paths.py:47-53`, `paths.py:137`. `qvals_table.py:1034-1042` writes at `replay_content_dir`. `fit_swr_power_model.py:76,450` |
| T17 S3: provenance | yes, with one gap (I1) | The sidecars record name + sha256 + mtime, not a path. The frozen `tables_manifest.json` and `_golden_baseline_*/manifest.json` are not read at run time. |
| T17 S4: GOLDEN | IDENTICAL | `$SP/golden17/log.txt`, `$SP/golden18/log.txt` (7 tables, `DataFrame.equals=True`) |
| T17 S5: disk move | done by the controller, hashes identical | `$SP/disk_move.log`. Canonical: `qvals_table.parquet` and its reference at the root, 4 files in `models_per_event_swr_power/`. 18 symlinks. The cleanup `find` in PLAN.md matches exactly those 18 (dry run: 18, all made 2026-10-23). |
| T18 S1: pickle scan | 0 hits for the old paths; control 706 | `$SP/pickle_scan/` |
| T18 S2: renames, one commit each, `git mv` | yes | `git show -M --stat` gives R100 for `swr_inventories→swr_detectors` and `statistics→window_statistics`. In the other renamed files only docstrings change. Each file was diffed against its pre-image: swr_source changes 2 docstring lines, nonlocal_detection 2 doctest imports, data_types 3 doctest imports, window_statistics loses the 4-line keep paragraph. |
| Old names gone from tracked files | yes | `git grep` (excluding `.claude/docs`) for `events.swr_inventory`, `events/swr_inventory.py`, `swr_inventories`, `events.detection`, `events/detection.py`, `continuum_swr_replay.schemas`, `schemas.py`, `import schemas`, `adapters.swr_inventory`, `source_inventory`, `memoryless_sequence.statistics`/`/statistics`, and bare `swr_inventory.X`, `swr_inventories.X`, `schemas.X`, `statistics.X`. The only hits are the two "(removed: renamed)" rows in docs/README.md:212-213 and local variables named `statistics` in pfc scripts. |
| T18 S3: shadow test | yes, and it fails on the old layout | It walks every `__init__.py` directory (every .py directory under src has one, checked), with allowlist `{io}`. `statistics.py` would have failed it. The report gives a red check (`json.py`). |
| T18 S4: README table, layering and lazy sets, CLAUDE.md | yes, with gaps (M3) | `test_import_layering.py` FOUNDATION has `data_types`. `test_lazy_imports.py` strings updated. CLAUDE.md has only path edits. |
| WIP script | imports only `pfc.load_population`, `replay.model_specs`, `replay.per_bin_design`; reads `models_per_bin_rate/all_models.pkl`, which did not move | `generate_replay_story_figures.py:42-51,67` |
| Commit trailers | all 11 commits carry both lines | |
| Tests run here | 79 passed (layering, lazy imports, doc paths, shadow, paths). 110 passed (fit provenance, golden-baseline scripts, pipeline map, retired names, regeneration manifest, inventory adapters). `ruff check src scripts tests`: All checks passed | |

## Critical

None.

## Important

### I1. Three tools in this branch still see the old layout through the symlinks

With the symlinks in place, a new record names the old path, and the run gives no warning.

1. **`scripts/qa/check_fit_provenance.py:61-75` (`resolve_input`)** searches the sidecar's own
   directory before the root. For the 3 per-bin sidecars that record `qvals_table`, it resolves the
   table through the old-path symlink:
   ```
   models_per_bin_rate/swr_contrast_rate.provenance.json -> .../replay_content/models_per_bin_rate/qvals_table.pkl
   models/swr_contrast_content_arm.provenance.json       -> .../replay_content/qvals_table.pkl
   ```
   The bytes are the same today, so the "93/93 fresh" result holds. But the report prints the old
   path. The order of the search is also wrong without the symlinks: a leftover real copy in the
   sidecar's directory takes precedence over the canonical table.
   **Fix:** resolve a name in `CANONICAL_TABLE_BASENAMES` at `root` only. Keep the `sidecar.parent`
   search for non-canonical names, which are the aggregate-only inputs. Add a unit test: a sidecar in
   `models_per_bin_rate/` that records `qvals_table`, with a decoy file beside it, resolves to
   `root/qvals_table.pkl`.
2. **`scripts/tables/write_regeneration_manifest.py:485-498`** inventories `root.rglob("*")` with
   `is_file()`, which follows symlinks. On the canonical root as it stands now, a new manifest would
   list 7 old-path entries:
   - `models_per_bin_rate/qvals_table.parquet` and `qvals_standardization_reference.json`
   - `models_per_bin_rate/swr_power_model.z_reference_by_session.json`
   - the same qvals pair in both `_staging_r13*` roots

   These are duplicates of files already listed at their new paths (dry-run listing above).
3. **`scripts/qa/freeze_regression_baseline.py:102`**: `shutil.copytree(src, ...)` uses its default
   `symlinks=False`. A baseline frozen now would copy qvals and swr_power into its
   `models_per_bin_rate/` as real files, so a new baseline would be laid out the old way.

**Fix (2, 3):** add `and not path.is_symlink()` to the manifest inventory. Pass
`ignore=lambda d, names: [n for n in names if (Path(d) / n).is_symlink()]` to the freeze's
`copytree`. Or, at minimum, write in the PLAN.md cleanup entry that no regeneration manifest or
baseline freeze runs until the symlinks are deleted.

### I2. A write from an older branch replaces a symlink and leaves two different copies

`runs/io.py:110-118` (`write_atomically`) writes to a temp file and then calls
`os.replace(temporary, path)`. When `path` is one of the old-path symlinks, `os.replace` puts a
regular file where the symlink was; the root file is left unchanged. The `covariate-explorer`,
`phase3` and `phase4-run` worktrees share this `Processed-Data/` (checked with `readlink -f`), and
their code still writes to the old paths. Suppose one of them runs `build_qvals_table.py --write`
(through `publish_canonical`), or `fit_swr_power_model.py` (its pickle; its CSVs go through
`to_csv`, which follows the link). Then that worktree reads its new file, while this branch's
readers (`table_path` → root) read the stale one, and neither run reports it. The ruling in
progress.md ("a symlink is the same file, so no stale copy") holds for reads only.

**Fix:** add two things to the PLAN.md cleanup entry and to progress.md:
- Until cleanup, those worktrees must not run `build_qvals_table.py --write` or
  `fit_swr_power_model.py`.
- A check that must print nothing:
  `for r in replay_content replay_content_staging replay_content_pe_staging replay_content_regen_staging replay_content_v3_staging replay_content/_staging_r13_tables_20261011 replay_content/_staging_r13c_20261011; do find "Processed-Data/$r/models_per_bin_rate" -maxdepth 1 -type f \( -name qvals_table.parquet -o -name qvals_standardization_reference.json -o -name 'swr_power_*' \); done`

  It prints nothing today. The `-maxdepth 1` and the explicit list of roots leave out the backups
  and the frozen `_pre_v4_2333f07` snapshot. Those still hold real files at the old paths, by
  design.

### I3. `source_swr_inventory` tells a reader something untrue

`analysis/inventory_comparison/adapters.py:428`: the function reads **any one of the four**
detectors' inventories. Its docstring says "One SWR detector's inventory". It is called once per
detector in `run_inventory_comparison.py:340`, and the tests call it with `karlsson` and `zugaro`.
In this package, "source" now means "the paper's one SWR source" (Kay): see the `swr_source` module
docstring and `events/__init__.py`. So `source_swr_inventory(run_dir, "yu", ...)` reads as "the
source's inventory", which it is not. Its siblings are `carey_inventory`, `mua_inventory`,
`state_space_inventory` and `memoryless_inventory`. The name collision that prompted the rename is
gone, because the module is now `swr_source`.

**Fix:** rename it to `swr_detector_inventory`, or back to `swr_inventory`, which no longer collides
with anything. This was a lead ruling ("cost if wrong: one more rename"), so the lead should decide.

## Minor

- **M1. The nonlocal_detection docstring claims ripple detection.**
  `analysis/events/nonlocal_detection.py:1` says "Detection and analysis of neural events
  (ripples, non-local states)". The module detects no ripples: it holds `get_confident_times` and
  the SWR-overlap classification. Fix: "Non-local event detection (`get_confident_times`) and the
  SWR-overlap classification of those events." The name itself is acceptable, because the package
  docstring (`events/__init__.py`) describes both halves.
- **M2. The moved-modules section misses one rename and names the wrong origin for another.**
  `docs/README.md` "Moved modules" does not list the function rename
  `inventory_comparison.adapters.swr_inventory` → `source_swr_inventory` (or whatever I3 settles
  on). That function existed on master (`470a393:.../adapters.py:429`), and outside callers will get
  an `AttributeError`. Also, `kay_version_check` is listed as moved "from
  `analysis/hippocampus/swr_detectors.py`"; on master it was in `swr_inventories.py`. Write "from
  `swr_inventories.py` (now `swr_detectors.py`)".
- **M3. The "Where results live" row for `models_per_bin_rate/` is incomplete.** It names one
  producer (`fit_per_bin_rate_models.py`), but the folder also holds outputs from
  `fit_swr_contrast_rate.py`, `fit_uncertainty_rate.py` and `fit_cross_history_rate.py`. It also says
  "per-bin rate fits only", but on disk the folder still holds `contingency_table.pkl`,
  `contingency_csv_alpha.json` and `covariate_ranking.pkl`, which the brief kept. Name all four
  producers, and add "plus three exploratory one-offs nothing supported reads".
- **M4. A runbook step still names the old path.**
  `docs/design/2026-10-03-full-pipeline-refresh-design.md:591`: the Promote step tells the reader to
  copy `models_per_bin_rate/qvals_standardization_reference.json`. This is the refresh runbook, which
  will likely be followed again. A dated note such as "(since 2026-10-23 at the root)" keeps the
  record and stops the step from misleading.
- **M5. Two writer locations have no test.**
  - `test_every_swr_power_output_resolves_under_its_own_model_dir` pins `OUTPUT_DIR`, but not the
    `--output-dir` default (`fit_swr_power_model.py:450`).
  - Nothing pins where `run_build_qvals_table` writes. It passes `replay_content_dir` to
    `publish_canonical` and does not go through `table_path`.

  Either location could move back without a test failing. Fix: assert the argparse default. For
  qvals, derive the directory from `table_path("qvals_table", root).parent`, or test the writer with
  a stubbed table.
- **M6. Unrelated, noted only; not part of this phase.** `fit_swr_power_model.py` (a per-event
  ripple-power model) still lives in `scripts/rate/`, and `run_refits.py:109` and
  `analysis_map.md:107,192` list it under the analysis id `rate.ripple_power`. That is the same kind
  of misleading folder name that Task 17 fixed for the outputs.

## Checked and fine

- The frozen string `"excluded time (swr_source.excluded_intervals)"` (`run_state_evidence.py:1080`)
  is compared on resume. The comment above it (1076-1078) already says the function is
  `analysed_time.excluded_intervals`. `swr_source.py` defines no `excluded_intervals`, so nobody is
  misled into a wrong import.
- The frozen dictionary keys `swr_inventory_run_id` are data keys, not module names.
- The six renamed modules: `swr_source`, `swr_detectors`, `data_types` and `window_statistics` say
  what each module does, and their docstrings agree. `nonlocal_detection` is acceptable (see M1).
- `docs/README.md` "Code layout" is otherwise accurate:
  - the foundation list includes `data_types.py`;
  - the range `470a393..2e55b21b` reproduces the table from `git diff -M --name-status`;
  - every R/D entry has a row.
- `models_per_bin_rate/` now holds only per-bin rate fits, apart from M3's three one-offs. All of
  `swr_contrast_rate`, `uncertainty_rate` and `cross_history` are Family-B per-bin fits.
