# Code organization review — 2026-10-19

Branch `simplify-repo-2026-10-19` at `7e54217`. Read-only review of architecture, module
boundaries and layout. The consolidations that just landed (epoch-id parser,
`decode_files.py`, the `swr_source` read point, `_decode_runner.py`, the strict-overlap
predicate, `model_specs.py`) were not re-reviewed. Untracked WIP was ignored.

Sizes: `src/` 157 modules / 81,897 lines; `scripts/` 72,739 lines (tracked `.py`);
`tests/` 104,924 lines, about 4,236 `def test_` (3,647 of them under `tests/unit/`).

Method: a static `ast` import graph of `src/` (all imports, with top-level and
function-local ones kept apart), `python -X importtime` on chosen modules (run from
`/tmp`, see F12), and grep sampling. Every number below comes from those commands.

---

## Summary

The package's upstream layering mostly holds. `data_loaders` never imports `analysis`,
`analysis` never imports `visualization`, `src` never imports `scripts`, and the
generated docs carry checks. The main structural problem is that the repo has no
foundation layer. Shared infrastructure lives inside leaf analysis packages:
- the table schema and ids in `analysis/replay`;
- the excluded-time rule, the interval algebra and the decode-frame read in
  `analysis/hippocampus`;
- spike binning in `analysis/pfc/glm.py`;
- run and orchestration helpers in `decoding_common.py` and inside one script.

That one fact accounts for most of the symptoms. There are six package-level import
cycles. Importing an 83-line leaf such as `event_ids` costs 4.5 s and loads JAX. Helpers
are re-implemented across the layer boundary (`json_ready` ×2, atomic write ×4,
`run_epochs` ×3, overlap tests ×4). And 35 script-to-script imports plus 56 test files
use scripts as libraries. Second-tier issues:
- output paths and track topology are spelled inline dozens of times;
- the tests are a flat 230-file `unit/` folder with real-data tests mixed in;
- a few stale structure docs.

## What is well organized

- **Upstream layering of the loaders.** Rule 2 holds (`data_loaders` imports no
  `analysis`). Spyglass imports are lazy and guarded by
  `tests/unit/test_lazy_imports.py`. `data_loaders/__init__.py`,
  `analysis/hippocampus/__init__.py` and the package root all use PEP 562 lazy
  re-exports.
- **Visualization is a true leaf.** No module under `analysis/` or `data_loaders/`
  imports `visualization` or matplotlib (grep for `import matplotlib|plt\.` under
  `analysis/` and `data_loaders/` returns nothing). `src` never imports `scripts` (the
  only hit is a comment at `model_specs.py:996`).
- **Single sources that are enforced, not just declared.**
  - `model_specs.py` is mirrored by `docs/models_table.md` through
    `test_models_table_mirrors_specs.py`.
  - `docs/regression_pipeline_map.md` is generated with a `--check` mode.
  - The epoch register (CSV plus `epoch_register.py` plus a doc generator) is the cohort.
  - `session_keys.SESSION_COLUMNS` is pinned against the registry by a test.
  - Covariate builders read `design_constants`, not `model_specs` (rule 1).
- **Cohesive subpackages.** `hippocampus/artifacts/`, `hippocampus/memoryless_sequence/`
  (with one `config.py` for every threshold), `hippocampus/state_evidence/`,
  `replay/covariates/history/` and `inventory_comparison/` each have one job. Every
  module opens with a one-line docstring that says what it is for, which makes
  `regression_pipeline_map.md` usable.
- **Reuse rather than forks.** `analysis/contfrag/` builds on the hippocampus helpers
  (`contfrag/result_loading.py:9`, `contfrag/data_aggregation.py:21-38`) instead of
  copying them.
- **Docs point at real files.** Of every `scripts/...py` path cited in `CLAUDE.md`,
  `docs/*.md`, `docs/design/*.md`, `scripts/README.md` and `tests/README.md`, only two
  do not exist (F10).

---

## Findings (ranked by impact)

Risk column: risk to the golden-table outputs (`event_table`, `event_arm_table`,
`event_patch_table`, `event_sequence_table`, `stop_table`, `qvals_table`) if the
proposal is done carefully. Every item below except a function body rewrite is a
relocation, so the expected diff is byte-identical. The check is
`scripts/qa/compare_to_golden_baseline.py` against `_golden_baseline_v5_v4cache`.

### F1. `analysis/replay/__init__.py` eagerly imports the whole package, and the package hosts shared leaves — every consumer pays ~4.5 s and JAX

**Evidence.**
- `analysis/replay/__init__.py` (161 lines) eagerly imports `tables_arm_patch`, `fitting`,
  `choice_prediction`, `per_bin_*`, `retrospective` and others (lines 8-77). The import
  chain is `tables_arm_patch → _tables_common → trajectory →
  hippocampus.posterior_analysis → non_local_detector` (2.06 s), plus `jax` and
  `statsmodels`.
- Measured cold import times (`-X importtime`):

  | module | lines | own deps | import cost | JAX loaded |
  |---|---|---|---|---|
  | `analysis.replay.event_ids` | 83 | numpy, pandas | 4.55 s | yes |
  | `analysis.replay.cohort_selection` | 553 | numpy, pandas | 4.50 s | yes |
  | `table_parity` | 182 | → `covariate_registry` | 4.57 s | yes |
  | `analysis.pfc.cohort` | 111 | → `cohort_selection` | 4.25 s | yes |
  | `analysis.replay.model_specs` | 1490 | none heavy | 4.32 s | yes |
  | `data_loaders.epoch_register` (for contrast) | 498 | — | 0.52 s | no |

- Many modules inside `replay/` are dependency-free leaves that other packages need:
  `covariate_registry` (imports only `versions`), `versions`, `standardization`,
  `event_ids`, `cohort_selection`, `design_constants` and `fit_provenance` (only
  `table_io`). Their importers outside `replay/`:
  - `pfc/cohort.py:21`, `pfc/glm.py:34` (`cohort_selection`);
  - `pfc/load_population.py:34` (`fit_provenance`);
  - `behavior/qvals_table.py:21,29` (`covariate_registry`, `standardization`);
  - `contfrag/tables.py:47,55,70`;
  - `inventory_comparison/adapters.py:82`, `inventory_comparison/features.py:53`;
  - `hippocampus/event_table.py:241` (lazy);
  - `table_io.py:122` (lazy), `table_parity.py:19`.
- The code already works around the cost. `session_keys.py:19-24` keeps a literal plus a
  test *because* importing `covariate_registry` "would pull in the heavy
  `analysis.replay` package". `table_io.py:116-125` uses function-local imports to dodge
  the resulting cycle. `posterior_io.py:55-59` does the same.
- The policy is inconsistent: `data_loaders` and `hippocampus` are lazy; `replay`, `pfc`
  (`pfc/__init__.py:63-76` eagerly imports `glm`, so statsmodels and dask) and
  `visualization` (`__init__.py:173-194`) are eager.

**Why it matters.**
- Any tool, test or worker that needs a row key or a cohort rule loads JAX. That
  switches float64 process-wide and touches the GPU allocator.
- It inflates test time, and it hides the real dependency graph: everything appears to
  depend on everything.

**Proposal.**
1. (S) Make `analysis/replay/__init__.py` and `analysis/pfc/__init__.py` lazy through PEP
   562, copying the existing `hippocampus/__init__.py:75-138` pattern. Add the leaf
   modules above to a "no JAX / no non_local_detector / no statsmodels" import test,
   next to `test_lazy_imports.py`.
2. (M, after F2) Move the schema leaves out of `replay/` into a foundation `tables/`
   package. See the target layout in F2.

**Effort** S for step 1, M for step 2. **Risk** none; imports only. Re-export names stay
importable through the lazy `__getattr__`.

### F2. No foundation layer: shared infrastructure sits in leaf analysis packages, giving six package-level cycles

**Evidence.** Package-level cycles in the static graph (module-level cycles are broken
by lazy imports; the only module SCCs are
`{table_io, swr_source, artifacts.io}` and `{visualization.figurl, visualization.interactive}`):

| cycle | forward edge | back edge |
|---|---|---|
| hippocampus ↔ pfc | `pfc.glm → hippocampus.*` (9 edges) | `hippocampus/swr_inventories.py:120` imports `pfc.glm.get_spikecount_per_time_bin` (top level) |
| hippocampus ↔ inventory_comparison | 18 edges | `swr_inventories.py:115-116` imports `inventory_comparison.overlap`, `.schema` |
| hippocampus ↔ replay | 15 edges | `hippocampus/event_table.py:241` → `replay.event_ids` (lazy) |
| hippocampus ↔ decoding_common | `decoding_common.py:41-45` → `swr_source`, `artifacts.masks` | `hippocampus/artifacts/detect.py:80` and `state_evidence/io.py:17` → `decoding_common` |
| hippocampus ↔ table_io | `swr_source`, `artifacts/io.py:32` → `table_io` | `table_io.py:118` → `swr_source` (lazy) |
| replay ↔ table_io | 3 edges | `table_io.py:122` → `covariate_registry` (lazy) |

The cause is that `analysis/hippocampus/` holds two kinds of modules.

- **Foundation code that the whole repo uses.** Fan-in counts from the graph:
  `swr_source` 18 (and 45 script files), `event_immobility` 16, `artifacts/masks` 11,
  `event_detection` 9, plus `event_eligibility` and `epoch_normalizers`.
- **Leaf analyses.** `swr_inventories`, `quality_metrics`, `state_evidence/` and
  `memoryless_sequence/`. Some of these reach sideways into sibling analysis packages.

Concrete misplacements:
- `get_spikecount_per_time_bin` is a generic binning primitive. It lives in
  `pfc/glm.py:49` (a 3,099-line module that imports dask and statsmodels at top level)
  and is consumed by hippocampus SWR detection (`swr_inventories.py:837`).
- `hippocampus/artifacts/detect.py:80,190` imports `decoding_common.DecodingConfig` only
  to read `WAVEFORM_STD`, which is `parameters.WAVEFORM_STD`. The cost is that importing
  the artifact detector loads `dask.distributed` (measured: `decoding_common` 2.01 s of
  `artifacts.detect`'s 2.24 s).
- Interval algebra has two homes with overlapping functions:
  - `inventory_comparison/intervals.py`: `normalize`, `union`, `intersect`, `subtract`,
    `total_duration`, `contains`, on NDArrays;
  - `hippocampus/artifacts/masks.py`: `merge_intervals`, `subtract_intervals`,
    `overlaps_intervals`, `covered_seconds`, `masked_seconds`, on DataFrames.

  Further local copies:
  - `data_loaders/lfp.py:758` `_overlap_any`, which reimplements
    `masks.overlaps_intervals` (both closed-interval). It cannot import `masks` because of
    rule 2, which shows the primitive is in the wrong layer.
  - `pfc/glm.py:1957,2002` `_contiguous_true_runs` and `_overlaps_any` (half-open).

  The three overlap conventions (closed, half-open, strict) may each be intentional. They
  should sit side by side in one module so that the choice is visible.
- `decode_files.py` sits at the top level only because `data_loaders` may not import
  `analysis` (its docstring, lines 11-14). It also imports
  `analysis.constants.DEFAULT_NON_LOCAL_THRESHOLD` (line 22), so a "top-level" module
  depends on `analysis`.

**Why it matters.** A contributor cannot tell what is safe to import from where, and a
"leaf" change can break a sibling package. The existing layering test cannot express
the rule that matters, which is "foundation imports nothing above it".

**Proposal.** Create a foundation layer that imports only numpy, pandas, networkx and
other foundation modules. One possible layout:

```
continuum_swr_replay/
  core/            parameters.py, constants (merge analysis/constants.py + data_loaders/constants.py
                   track topology), types.py (see F12), session_keys.py, paths.py (F5)
  intervals.py     one interval algebra: masks.{merge,subtract,overlaps,covered}_intervals +
                   inventory_comparison/intervals.py + the strict/half-open/closed predicates,
                   side by side, + spike binning (get_spikecount_per_time_bin)
  tables/          io.py (table_io), parity.py (table_parity), schema.py (covariate_registry +
                   versions), ids.py (event_ids), standardization.py
  decoding/        files.py (decode_files), transitions.py (decoding_transitions),
                   config.py (DecodingConfig), results.py (the one results.nc reader, F8)
```

`swr_source`, `event_eligibility`, `event_immobility`, `artifacts/masks`, `bouts` and
`cohort_selection` are the "event substrate". There are two options:
- (a) promote them to `analysis/events/`;
- (b) keep them in `hippocampus/` and forbid `hippocampus` from importing `pfc` and
  `inventory_comparison`.

(b) is the smaller change. It needs only `get_spikecount_per_time_bin` moved and
`swr_inventories` given its own overlap and schema imports from the new `intervals.py`
or an `inventory_comparison` leaf that has no hippocampus imports.

**Effort** M. **Risk** low. These are relocations of pure functions. Run the
golden-table byte diff, and keep re-export shims for one release because scripts import
these names (`masks` alone appears in 9 script files).

### F3. Run and orchestration infrastructure is split across `decoding_common.py`, `_decode_runner.py`, and a script that acts as a library

**Evidence.**
- `decoding_common.py` (878 lines, top level, imports `dask` and `dask.distributed` at
  module load, lines 28-31) mixes five jobs:
  - decode configs (`DecodingConfig`, :65);
  - HPC decode masks (`hpc_artifact_intervals`, `hpc_decode_masks`, :202-340), which is
    hippocampus logic;
  - logging (:341-408);
  - atomic I/O (`write_atomically`, `write_json`, `replace_outputs`, :423-518);
  - CLI and batch (`parse_run_id`, `add_common_args`, `run_batch`, :519-878) plus git
    provenance (:546).
- `scripts/decoding/run_swr_detection.py` is the de facto run framework for the
  detection runners. It exports `epoch_id` (:228), `array_sha256` (:394), `json_ready`
  (:401), `epoch_artifact_intervals` (:533), `run_epochs` (:1058) and `run_lock` (:1178).
  Its importers:
  - `run_artifact_detection.py:118-121`;
  - `run_carey_detection.py:132-136`;
  - `qa/run_inventory_comparison.py:187-188`;
  - `qa/run_event_features.py:195-198`;
  - 5 test files.

  `run_memoryless_detection.py` likewise exports `load_marks`, `marks_cache_path`,
  `MARKS_RULE` and `input_paths` to `run_carey_detection.py:121-124` and
  `run_event_features.py:188`.
- Duplicate definitions (grep `^def NAME`):

  | helper | where defined |
  |---|---|
  | `json_ready` | `hippocampus/state_evidence/io.py:23`, `scripts/decoding/run_swr_detection.py:401` (the latter also accepts `Mapping`/ndarray) |
  | `write_json` | `decoding_common.py:453`, `state_evidence/io.py:48` |
  | atomic replace | `table_io.py:198`, `decoding_common.py:448`, `replay/prepare_data.py:338-357`, `hippocampus/artifacts/io.py:206` |
  | `run_epochs` | `run_swr_detection.py:1058`, `run_memoryless_detection.py:944`, `qa/run_state_evidence.py:1186` |
  | `_run_epoch_logged` | 5 scripts (swr, carey, artifact, event_features, state_evidence) |
  | git provenance | `decoding_common.py:546-580`, `replay/fit_provenance.py:71-111`, `decoding/write_regeneration_manifest.py:58`, `qa/run_refits.py:241`, `qa/sweep_memoryless_detection.py:731` |

**Why it matters.** There are four atomic writers and three batch runners. A fix to
crash safety or to provenance must be made N times, and the manifests the runners write
are not guaranteed to agree in format. A new runner author cannot see which helper is
canonical.

**Proposal.** Create a `continuum_swr_replay/runs/` subpackage (no analysis imports):
- `io.py`: one `write_atomically`, `write_json`, `json_ready`, `sha256_*`; `table_io`
  then calls it.
- `provenance.py`: one git or provenance record, which `fit_provenance` reuses.
- `cli.py`: `parse_run_id`, `add_common_args`, `filter_epoch_info`.
- `batch.py`: `run_batch`, `run_epochs`, `run_lock`, `run_epoch_logged`, logger setup.

Move the HPC decode-mask functions to `analysis/hippocampus/` (or to `decoding/` from
F2) and `DecodingConfig` to `decoding/config.py`. Fold `scripts/decoding/_decode_runner.py`
into `decoding/runner.py` if it has no CLI of its own (565 lines). The detection runners
then import `runs.*` rather than each other.

**Effort** M. **Risk** none to the tables. Run manifests and `*.json` sidecars could
change format if the `json_ready` variants are merged (the two differ on
`Mapping`/ndarray handling). Diff one smoke run's manifest before and after.

### F4. Scripts used as libraries: 35 script-to-script imports, 56 test files, and output-naming rules that live only in a runner

**Evidence.**
- `scripts/README.md` states that a script does not import another script. It then
  grandfathers "the existing `from scripts.… import …` lines". Excluding
  `_decode_runner` and `_report`, there are 35 such import lines (F3 lists the runner
  ones). 56 tracked test files import `scripts.*`; the top targets are
  `qa.run_inventory_comparison` (5), `qa.compare_fits` (5) and
  `decoding.run_swr_detection` (5).
- `scripts/content/fit_replay_content_models.py` (2,203 lines) owns rules that other
  code reads:
  - `canonical_output_subdir` (:269), `canonical_output_subdirs` (:350) and
    `belief_chain_output_subdir` (:417). These decide where content fits land, and
    `analyze_uncertainty_content.py:56`, `generate_replay_content_figures.py:46` and
    `qa/freeze_regression_baseline.py:38` import them.
  - `restrict_to_leaf_arms` (:492), `leaf_only_cohort` (:524) and `rung_predictors`
    (:469, a wrapper over `model_specs.leaf_only_predictors`).
  - `compare_content_estimators.py:117-122` imports `_assemble_arm`, `_assemble_patch`
    and `_prepare_session`, which are private helpers of
    `scripts/content/fit_swr_contrast_content.py`.
- `scripts/qa/report_event_features.py:99-109` imports 11 names, several private
  (`_caption`, `_per_session`, `_ratio`, `_summed`), from
  `qa/report_inventory_comparison.py`. `qa/sweep_memoryless_heldout.py` imports two
  other sweep scripts.
- Exploratory scripts that have become load-bearing (pinned by unit tests):
  - `exploratory/_verify_harmonization_parity.py` (`test_harmonization_parity_guard.py`);
  - `exploratory/_statescript_contingency.py` (`test_statescript_contingency.py`);
  - `exploratory/_augment_data_over_time.py` (`test_augment_discover.py`);
  - `exploratory/_trunk_zscore_{extract,timebin}.py` (`test_swr_source_consumers.py`).

  `scripts/README.md:296-298` calls this folder "unsupported one-offs … May be stale".
- Large scripts with library-scale logic and test coverage: `qa/compare_fits.py`
  (3,936 lines, 80 top-level defs, 5 test files) and `qa/compare_pfc_cohorts.py` (2,632
  lines, 46 defs).

**Why it matters.**
- Output-location rules defined in a CLI module mean any figure or QA script that finds
  fits must import the fitting runner, with its argparse, logging setup and heavy
  imports.
- Private-name imports across scripts break silently on a rename.
- "Exploratory" no longer means "safe to change".

**Proposal.**
- Move the content-fit output naming into `analysis/replay/fit_outputs.py`, or into
  `paths.py` from F5.
- Move the leaf-only cohort helpers next to `model_specs.leaf_only_predictors` or into
  `cohort_selection`.
- Move the shared contrast-content assembly helpers into
  `analysis/replay/contrast_aggregation.py` (which already exists for this purpose).
- Promote the report-formatting helpers shared by the two `report_*` scripts into
  `scripts/qa/_report.py`, which is already the sanctioned shared module.
- For each test-pinned exploratory script: promote it to `qa/` without the underscore,
  or record it as an exception. This is a question for the lead (Q4). None is proposed
  for deletion.
- Tighten the README rule to "only `_`-prefixed modules are importable". Add a static
  test that fails on a new `from scripts.<non-underscore>` import, with an allowlist that
  shrinks over time.

**Effort** M to L (incremental). **Risk** low to moderate. The output-subdir strings
decide where fits are written and read, so pin them with a test that compares
`canonical_output_subdirs()` before and after the move.

### F5. Output-path conventions are spelled inline across ~30 files

**Evidence.**
- `PROCESSED_DATA_DIR / "replay_content"` appears 50 times in 32 files (4 in `src`, 28 in
  `scripts`). `analysis/replay/prepare_data.py:100` already defines
  `OUTPUT_DIR = PROCESSED_DATA_DIR / "replay_content"`, and nothing imports it under
  that meaning.
- Literal table filenames: `"stop_table.{parquet,pkl}"` 30 times in 19 files,
  `"event_table…"` 25 times in 17 files, `"qvals_table…"` 15 times in 11 files,
  `"event_arm_table…"` 14 times in 9 files. `"models_per_bin_rate"` appears 45 times.
- The registry of canonical table basenames (`BASENAME_TO_TABLE`,
  `CANONICAL_TABLE_BASENAMES`) lives in `table_parity.py:28-39`, a module about diffing
  tables, so producers do not use it.
- The PFC path constants re-derive the root: `pfc/canonical_events.py:71` and
  `pfc/cohort.py:29`. `swr_source.py:192,208` holds the detection roots.
- `docs/README.md` ("Where results live") notes that `models_per_bin_rate/` also holds
  `qvals_table.parquet`, `swr_contrast_rate.pkl`, `uncertainty_rate.pkl` and
  `swr_power_model.pkl`, and that the rename was "considered and deferred".

**Why it matters.** Relocating an output (staging, backups, the deferred rename) is a
32-file hunt, and a missed spelling silently reads a stale artifact. That is exactly the
`event_id` mis-join hazard CLAUDE.md warns about for partial regeneration.

**Proposal.** Create `continuum_swr_replay/paths.py` (foundation) holding:
- `REPLAY_CONTENT_DIR`, `CONTFRAG_REPLAY_CONTENT_DIR`, `DETECTION_COMPARISON_DIR`,
  `ARTIFACT_INTERVALS_DIR`, `MODEL_DIRS`;
- `table_path(name, root=REPLAY_CONTENT_DIR)`, validated against the canonical-table
  registry, which moves there from `table_parity.py`.

Migrate mechanically. The deferred directory rename then becomes a one-line change
(still a question for the lead, Q7).

**Effort** M (mechanical). **Risk** low. Add a test asserting that every migrated
constant resolves to the same `Path` as the literal it replaced.

### F6. God modules, worst in the PFC path

**Evidence** (lines; longest functions from an `ast` scan):
- `analysis/pfc/glm.py`, 3,099 lines and 39 top-level defs. It holds:
  - spike binning (:49);
  - design and basis builders (:149-655);
  - fitting and penalties (:200-315, :656);
  - deviance and modulation statistics (:739-990);
  - `process_neuron` (:990, **497 lines**);
  - session fit masks and event sets (:1489-1934);
  - matched-control sampling (:1934-2303);
  - `run_session` (:2505, **595 lines**, Spyglass orchestration).

  Its partners are `pfc/batched_glm.py` (1,459) and `pfc/heldout.py` (950;
  `control_draw_null` is 369 lines), plus `scripts/pfc/run_pfc_analysis_bandit.py`
  (1,834; `run_analysis` is 447 lines).
- `visualization/session_tables.py`, 3,492 lines with 71 defs: one public
  `plot_session_tables` (298 lines) plus 69 private panel plotters behind a registry. It
  is cohesive but too large to navigate.
- `analysis/hippocampus/swr_source.py`, 1,364 lines and 38 defs. Despite the name it holds
  the SWR inventory read (:385-747), the artifact and excluded-time intervals
  (:528-594, :1183-1256), artifact drops (:870-1034), the decode-frame read point
  (`with_swr_source` :1066, `read_dot_with_swr_source` :1257) and `analysed_bins`
  (:1271). CLAUDE.md sends readers to `event_eligibility.py` for "the rule". In fact the
  rule's intervals are in `swr_source.excluded_intervals` and its padding is in
  `artifacts/masks`.
- Other modules over 1,200 lines: `replay/retrospective.py` (1,359),
  `replay/simulation.py` (1,236), `hippocampus/quality_metrics.py` (1,482),
  `hippocampus/state_evidence/summary.py` (1,323), `hippocampus/swr_inventories.py`
  (1,272), `data_loaders/trials.py` (1,357).

**Why it matters.** The PFC GLM is the hardest place in the repo to make a surgical
change. Its two 500-to-600-line functions combine data loading, event construction and
fitting, so a change to the event definition and a change to the penalty share one
function.

**Proposal.**
- Split `pfc/glm.py` into `pfc/design.py` (bases and design matrices), `pfc/fit.py`
  (fitters and diagnostics), `pfc/modulation.py` (peak modulation and deviance
  fractions), `pfc/event_sets.py` (fit masks, event sets, coverage),
  `pfc/matched_control.py`, and `pfc/session.py` (`run_session` and `process_neuron`,
  then decomposed). The binning helper goes to `intervals.py` (F2).
- Split `session_tables.py` into a `visualization/session_tables/` package by panel group
  (responses, arm covariates, patch covariates, stop covariates), keeping the registry in
  `__init__`.
- Rename or split `swr_source.py` into `swr_source.py` (inventory) and
  `analysed_time.py` (excluded intervals, drops, `with_swr_source`), and fix the CLAUDE.md
  pointer.

**Effort** L (PFC), M (session_tables), M (swr_source). **Risk**: for PFC, none to the
golden tables but real risk to the PFC bundles; diff `neuron_stats` from 2 sessions
before and after. `swr_source` is on every table's read path, so a byte diff of the
golden tables is mandatory.

### F7. The `scripts/` domain grouping is inconsistent; `qa/` is a catch-all

**Evidence.**
- `scripts/qa/` (33 files; its README section runs from line 124 to line 277, half the
  README) mixes true QA with science pipelines:
  - QA: parity and audits (`compare_to_golden_baseline`, `audit_*`, `check_*`,
    `mutation_check`, `write_*_doc`).
  - Science runners: `run_inventory_comparison` (922), `run_event_features` (1,010),
    `run_state_evidence` (1,747), `sweep_memoryless_{detection,controls,heldout}`,
    `report_{inventory_comparison,event_features}`, `run_quality_metrics`,
    `compute_hpc_raw_stats`.
- `scripts/decoding/` ("produce the decoded data + event/stop tables") holds the two
  decoders and four detection runners (`run_swr_detection`, `run_carey_detection`,
  `run_memoryless_detection`, `run_artifact_detection`). It also holds seven table-build
  scripts: `prepare_*`, `regenerate_stop_table`, `build_timebin_tables`,
  `assemble_*_shards`, `convert_tables_to_parquet`, `write_regeneration_manifest`.
- `scripts/pfc/` has one script. The PFC diagnostics (`compare_pfc_cohorts`,
  `compare_pfc_fitters`, `check_pfc_ridge_lambda0`, `measure_pfc_null_geometry`,
  `diagnose_pfc_event_definition`) are in `qa/`.

**Why it matters.** The replay-detection comparison is one of the paper's three goals.
Its pipeline is spread over `decoding/`, `qa/` and `figures/`, which explains much of the
cross-script importing in F4.

**Proposal.**
- `scripts/detection/`: the four detectors, the inventory comparison, event features,
  state evidence, the memoryless sweeps and their reports.
- `scripts/tables/`: the table builders.
- `decoding/` keeps the two decoders and the contfrag decode.
- `qa/` keeps parity, audits and doc writers.
- PFC diagnostics move to `pfc/` (or `pfc/qa_*`).

**Effort** M. **Risk** none to the tables, with one catch: runners record their own path
in provenance (for example `run_swr_detection.py:221`,
`PRODUCED_BY = "scripts/decoding/run_swr_detection.py"`), so a move changes manifest
strings that later checks may compare (Q1). Doc references must move too; the
pipeline-map `--check` and a grep catch them.

### F8. Remaining parallel definitions (constants, topology, column lists, epoch-id builder, results reader)

**Evidence.**
- **Track topology.** `data_loaders/constants.py:194,234` defines `WELL_ARMS` and
  `ALL_ARMS`, but the leaf and trunk sets are re-spelled in:
  - `uncertainty_content.py:39` (`LEAF_ARMS = tuple(range(6))`);
  - `visualization/session_tables.py:114` (`_LEAF_ARMS`);
  - `replay/trajectory.py:26` (`_TRUNK_ARMS = {6, 7, 8}`);
  - `replay/table_contracts.py:234` (`isin(range(6))`);
  - `model_specs.py:1457` (`LEAF_ONLY_N_CATEGORIES = 6`);
  - `scripts/content/analyze_content_occupancy.py:88` (`N_LEAF_ARMS = 6`).
- **Per-port column lists.** `f"…_port_{i}" for i in range(6)` is rebuilt in
  `behavior/qvals_table.py` (at least 8 sites: :87-88, :134-135, :177, :845-847,
  :866-867), `covariates/history/value_block.py:27-29`, `replay/prepare_data.py:516` and
  `covariate_registry.py:206-214`.
- **Sampling rate.** `parameters.SAMPLING_FREQUENCY = 500` and
  `analysis.constants.DEFAULT_SAMPLING_FREQUENCY_HZ` exist, yet `500.0` is the literal
  default in about 12 signatures:
  - `quality_metrics.py:232,336,563,1071,1302`;
  - `posterior_analysis.py:340`, `triggered_analysis.py:29`;
  - `pfc/glm.py:152,319`, `contfrag/data_aggregation.py:48`;
  - `memoryless_sequence/config.py:85`, `task_variables.py:484`.
- **Animal list.** `behavior/qvals_table.py:46`,
  `scripts/decoding/prepare_replay_content_data.py:72`,
  `scripts/figures/plot_swr_contrast_coefs.py:155` and
  `scripts/figures/generate_uncertainty_content_figure.py:85` each spell it. It is
  derivable from the epoch register.
- **Epoch-id builder.** The parser is unified (`data_loaders/utils.parse_epoch_identifier`)
  but the formatter `f"{stem}_{epoch}_{area}"` is still inline in 9 places:
  - `hippocampus/result_loading.py:90`, `swr_source.py:382`;
  - `pfc/canonical_events.py:189`, `pfc/glm.py:2703`;
  - `contfrag/result_loading.py:60`, `data_loaders/epoch_register.py:456`;
  - `scripts/decoding/_decode_runner.py:389`, `prepare_contfrag_replay_content_data.py:267`,
    `refresh_contfrag_behavioral_covariates.py:129`.

  They use both `.replace('_.nwb','')` and `.removesuffix('_.nwb')`.
- **`results.nc` reader.** The netCDF open plus the `state_bins` multi-index rebuild is
  duplicated in `replay/posterior_io.py:61-63` ("mirrors the same fix") and
  `hippocampus/result_loading.py:141-144`. Other raw opens are in
  `state_evidence/io.py:115`, `scripts/qa/run_state_evidence.py:447`,
  `scripts/decoding/prepare_contfrag_replay_content_data.py:226` and
  `scripts/figures/generate_inventory_examples.py:445`. `posterior_io` is a decode reader
  but lives in `replay/`.
- **Five constants homes.** `parameters.py` mixes paths, decoder parameters and
  `COLORS`. `analysis/constants.py` re-wraps `parameters` values (lines 44, 52). The
  others are `data_loaders/constants.py`, `replay/design_constants.py` and
  `replay/versions.py`.

**Why it matters.** Each case is low-risk alone. Together they make "what is the leaf set"
or "where is the epoch id built" a grep rather than a lookup.

**Proposal.**
- Add `LEAF_ARMS`, `TRUNK_ARMS`, `PORT_COLUMNS(prefix)` and `ANIMALS` (derived from the
  register) to the core constants (F2), and import them everywhere.
- Add `format_epoch_identifier(nwb_file_name, epoch, area, suffix="")` beside the parser
  in `data_loaders/utils.py`.
- Use one results reader in `decoding/results.py` (F2), with `posterior_io` moved there.
- Replace the `500.0` defaults with the constant.
- Move `COLORS`/`CONTFRAG_COLORS` from `parameters.py` to `visualization/palettes.py`.

**Effort** S to M. **Risk** low. The epoch-id variants should give the same string for
every register entry; add a parity test over `analysed_epochs()`.

### F9. Tests: a flat `unit/`, an ad-hoc `analysis/` split, real-data tests unmarked, stale README

**Evidence.**
- `tests/unit/` holds 230 flat files (3,647 tests). `tests/analysis/` mirrors only
  `hippocampus` and `pfc` (19 files). There is no home for `replay`, the largest package
  (23k lines), so its tests are in `unit/` under topic names such as
  `test_replay_tables_audit_regressions.py` and `test_figure_script_fixes.py`.
- Some modules are tested in both trees: quality metrics in
  `tests/analysis/hippocampus/test_quality_metrics.py` and
  `tests/unit/test_quality_metrics_{episode_gap,excluded_time}.py`; PFC load_population
  in `tests/analysis/pfc/test_load_population.py` and
  `tests/unit/test_pfc_load_population_nan_filter.py`. One name exists in two trees:
  `tests/unit/test_run_event_features.py` and `tests/integration/test_run_event_features.py`.
- Real-data-dependent tests sit in `unit/` and skip silently when `Processed-Data` is
  absent. Without a marker, the same command tests different things on different
  machines:
  - `test_uncertainty_covariates.py:941-977`;
  - `test_covariate_registry.py:506-517`;
  - `test_replay_tables_audit_regressions.py:277-316`;
  - `test_statescript_contingency.py:192-197`;
  - `test_per_bin_rate_helpers.py:34` (RC);
  - `tests/analysis/pfc/test_matched_control.py:397-426`;
  - `tests/integration/test_from_cache_parity.py:33-48`.
- The `integration` marker is applied in only 2 of 9 tracked integration files
  (`test_memoryless_figure_one_simulation.py:69`,
  `test_memoryless_simulated_replay.py:64`). 17 unit files spawn subprocesses or dask
  clusters.
- About 25% of unit test files (56) test `scripts/` modules (F4).
- `tests/README.md` is stale. It says the suite "runs in under a minute (~37 s)" (there
  are now about 4,236 tests) and that "`tests/integration/` is reserved for future
  DB-gated tests … currently unused" (it has 9 tracked files).

**Why it matters.** "Find the tests for module X" has no rule. A green run on a laptop
without data means something different from one on the server.

**Proposal.**
- Register a `real_data` marker (skip by default, opt in with `-m real_data`) and apply it
  to the tests above.
- Apply `integration` to every file in `tests/integration/`.
- Going forward, mirror `src` (`tests/analysis/replay/…`, `tests/data_loaders/…`) and
  move files opportunistically when touched, rather than in a big-bang move.
- Put tests of scripts under `tests/scripts/`.
- Update `tests/README.md`.

**Effort** S (markers, README), M (incremental moves). **Risk** none.

### F10. Documentation homes: one undeclared home, unindexed specs, stale structure text

**Evidence.**
- `docs/audits/` (20 tracked files) is a live home cited from `docs/README.md:180,261`.
  It is missing from the "Where each doc TYPE lives" table, whose nearest type,
  "review → `.claude/docs/reviews/`", says point-in-time reports belong under `.claude/`.
- Nine docs are not in the `docs/README.md` index:
  - `docs/design/2026-05-27-figure01-revision-design.md`;
  - `docs/design/2026-06-23-replay-trajectory-metrics-design.md`;
  - `docs/design/2026-06-25-non-swr-event-reality-design.md`;
  - `docs/design/2026-06-27-state-space-point-process-event-model.md`;
  - `docs/design/2026-10-03-full-pipeline-refresh-design.md` (cited by
    `run_swr_detection.py:222` as `DESIGN_SPEC`);
  - `docs/design/2026-10-04-decoder-transition-prior.md`;
  - `docs/design/content_model_design.md`;
  - `docs/design/replay_detection_comparison/memoryless_simulation_controls.md`;
  - `docs/known_issues/degenerate_mua_events.md`.
- Stale or contradictory points in `CLAUDE.md`:
  - the "Project Structure" tree (lines 66-72) lists only `hippocampus/` and `pfc/` under
    `analysis/` and omits `replay/` (the largest), `behavior/`, `contfrag/` and
    `inventory_comparison/`;
  - line 430 says `PROCESSED_DATA_DIR` is "defined in `src.parameters`";
  - line 446 says "nodes are wells (0-5)", which contradicts the 10-node description
    earlier in the same file.
- Stale script paths in docstrings: `replay/posterior_io.py:5`
  (`scripts/run_decoding_bandit.py`), `visualization/palettes.py:6,43`
  (`scripts/plot_replay_content_time_slice.py`, which does not exist), and
  `model_specs.py:996`.
- Two doc references point at files that do not exist:
  `scripts/qa/audit_refresh_consistency.py`
  (`docs/design/2026-10-03-full-pipeline-refresh-design.md:693,750`) and
  `scripts/_bandit_csv_contingency.py` (`docs/replay_rate_content_swr_synthesis.md:794`).
- A tracked `.claude/projects/-cumulus-edeno-continuum-swr-replay/memory/` (MEMORY.md and
  `project_failed_epochs.md`, last touched 2026-04-02) is a stale in-repo copy of
  auto-memory.

**Proposal.**
- Add `audits` as a declared type in the `docs/README.md` table (or move the audits to
  `.claude/docs/reviews/`; Q3).
- Index the nine docs, or mark them as superseded.
- Regenerate the CLAUDE.md structure tree from the real tree and fix the two lines.
- Fix the four docstrings and two doc references.
- Remove or ignore the tracked memory copy (Q8).

**Effort** S. **Risk** none.

### F11. The layering test encodes three good rules but misses the ones that are actually broken

**Evidence.**
- `tests/unit/test_import_layering.py` checks:
  1. covariates ↛ `model_specs`;
  2. `data_loaders` ↛ `analysis`;
  3. `hippocampus` ↛ `replay` except `{bouts, event_ids}`.

  All three pass. However, the rule-3 allowlist entry `bouts` is unused (no hippocampus
  module imports it). Using it would also close a cycle, since `replay/bouts.py:21,27`
  imports `hippocampus.artifacts.masks` and `replay.covariates.context`.
- Unguarded edges that hold today: `analysis` ↛ `visualization`, and `src` ↛ `scripts`.
- Unguarded edges that are broken today:
  - `hippocampus` ↛ `pfc` and `inventory_comparison` (F2);
  - `analysis` ↛ `decoding_common` (F2: `artifacts/detect.py:80`,
    `state_evidence/io.py:17`);
  - top-level foundation modules ↛ `analysis`: `decode_files.py:22`, `table_io.py:118,122`
    (lazy), `table_parity.py:19`.
- `test_lazy_imports.py` guards Spyglass and `dask.distributed` but not JAX or
  `non_local_detector`, which is why F1 went unnoticed.

**Proposal.** Add the following rules to `test_import_layering.py` now, with a
known-violations allowlist that F1 and F2 work down to empty:
- `analysis` ↛ `visualization`;
- `src` ↛ `scripts`;
- foundation ↛ `analysis`/`visualization`/`decoding_common`;
- `hippocampus` ↛ `pfc`/`inventory_comparison`;
- no package-level cycles. The `ast` graph in this review computes this in about 40
  lines.

Drop `bouts` from the rule-3 allowlist. Add a JAX-free import check for the F1 leaves.

**Effort** S. **Risk** none.

### F12. Minor

- **A local `types.py` shadows the stdlib `types`.** Running
  `python -c "import ast"` with the working directory set to
  `src/continuum_swr_replay/` fails with
  `ImportError: cannot import name 'StrEnum' from partially initialized module 'enum'`,
  because `types.py:13` is picked up as the stdlib `types`. This only bites tooling run
  from inside the package directory, but the failure is obscure. Renaming it (for
  example `schemas.py` or `typing_defs.py`) touches about 16 src importers plus 9
  in scripts and tests. Effort S, golden risk none.
- **Leftover directory.** `src/continuum_swr_replay/analysis/hippocampus/replay_content/`
  is untracked and holds only a stale `__pycache__/` from April. It is harmless but shows
  up in `find`.
- **Archived figures.** `scripts/figures/archive/` (15 scripts) carries the old literal
  thresholds (`head_speed < 4.0`, `> 0.8`). That is fine for archived provenance, but
  `generate_figure02.py`, `generate_all_figures.py`, `figure_panels.py` and three tests
  still reference `archive/generate_figure02.py`. Check that those are mentions, not
  imports.

---

## A suggested sequence

1. **Guards first (S, golden risk none).** Extend `test_import_layering.py` with the F11
   rules and a known-violations allowlist. Add the JAX-free import check. Make
   `analysis/replay/__init__.py` and `analysis/pfc/__init__.py` lazy (F1, step 1). Add
   the `real_data` and `integration` markers and fix the tests README (F9). Make the doc
   fixes (F10). Each item is independent and immediately useful.
2. **Foundation moves (M; golden byte diff after each).**
   - `core/` constants (topology, ports, animals, sampling rate) and
     `format_epoch_identifier` (F8).
   - `intervals.py`, with spike binning moved out of `pfc/glm.py`.
   - Point `artifacts/detect.py` at `parameters.WAVEFORM_STD` (F2).
   - `paths.py`, with the canonical-table registry moved out of `table_parity` (F5).

   Keep re-export shims; shrink the layering allowlist as you go.
3. **`tables/` schema layer (M).** `table_io`, `table_parity`, `covariate_registry`,
   `versions`, `event_ids`, `standardization`, `session_keys` (F1, step 2). This removes
   the `table_io ↔ replay/hippocampus` cycles and the `behavior → replay` edge. Before
   moving any class, run the pickle check in Q5.
4. **`runs/` infrastructure (M).** Unify atomic write, JSON, provenance, `run_epochs` and
   the lock. Migrate the detection runners off `run_swr_detection.py` (F3). Diff one
   smoke run's manifest.
5. **Scripts as libraries (M to L, incremental).** Move the content-fit output naming and
   the leaf-only helpers to `src`, consolidate the report helpers into `_report.py`, and
   turn on the "no new `from scripts.` imports" test (F4).
6. **Regroup `scripts/` (M).** Create `detection/` and `tables/`, and slim `qa/` (F7). Do
   this after step 5 so that fewer cross-imports have to be rewritten, and after the lead
   decides how `PRODUCED_BY` provenance should be handled.
7. **God-module splits (L).** `pfc/glm.py` first (diff the PFC bundles on two sessions),
   then `swr_source.py` (golden byte diff), then `session_tables.py` (F6).
8. **Tests mirror `src` (ongoing).** Move test files when their module is touched (F9).

## Questions for the lead

1. **Provenance strings.** Runners record their own path (`PRODUCED_BY`,
   `run_swr_detection.py:221`). Is changing those strings on a script move acceptable,
   or should moved scripts keep a thin stub at the old path?
2. **Hippocampus scope.** Should the event substrate (`swr_source`, `event_eligibility`,
   `event_immobility`, `artifacts/masks`, `bouts`, `cohort_selection`) become its own
   `analysis/events/` package (option a)? Or should it stay in `hippocampus/`, with
   `hippocampus` forbidden from importing `pfc` and `inventory_comparison` (option b,
   the smaller change)?
3. **`docs/audits/`.** Is it an intended published home (then add it to the type table),
   or should those 20 reports move to `.claude/docs/reviews/`?
4. **Test-pinned exploratory scripts.** `_verify_harmonization_parity`,
   `_statescript_contingency`, `_augment_data_over_time` and `_trunk_zscore_*` are
   exercised by unit tests. Should they be promoted to `qa/` or kept as documented
   exceptions? None is proposed for removal.
5. **Pickled package classes.** Do any fitted `.pkl` bundles (`all_models.pkl`, the
   contrast bundles, the PFC `*_modulation.pkl`, the `_session_cache` pickles written at
   `prepare_data.py:344`) contain instances of classes defined in this package, such as
   `ModelSpec`, `PerBinRateModelSpec` or NamedTuples? If so, moving those modules breaks
   unpickling of existing artifacts unless an alias shim is kept. This was not verified
   here (it needs loading the artifacts); it should be checked before step 3.
6. **Real-data tests.** Should the real-data tests run in the default suite on the server
   (opt-out marker) or be opt-in everywhere?
7. **Output directory rename.** Is the `models_per_bin_rate/` holding `qvals_table` and the
   contrast and uncertainty bundles still deliberately deferred? With F5 done, it becomes
   a one-line change.
8. **Tracked memory copy.** Is the tracked `.claude/projects/.../memory/` copy intentional?
9. **Comparison scripts.** Are `qa/compare_fits.py` (3,936 lines) and
   `qa/compare_pfc_cohorts.py` (2,632) ongoing tools whose logic should move to `src`, or
   one-time audit generators that should stay as scripts?
