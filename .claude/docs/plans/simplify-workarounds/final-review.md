# Final whole-branch review: simplify-workarounds (99034aab..8c522b4c)

Reviewer: opus, read-only. 52 commits, 365 files. I did not edit, commit, run a no-CLI
script, or touch the tmux sessions `simplify_parity` / `simplify_phaseF_suite`. Scratch
checks are in
`/tmp/claude-39494/-cumulus-edeno-continuum-swr-replay/bf9a3aff-72a3-4a3b-8ef1-aaefb5dcbcfe/scratchpad/final/`.

## Verdict

**Ready to merge: with fixes (Minor only), once Task 21 reports 0 parity differences and the
default suite shows only the 7 known failures.**

- No Critical or Important issue across the phases.
- The per-phase reviews' one Important (E2: `_content_cohort_control.py` wrote into the live
  `models_swr_contrast/`) is fixed in 8c522b4c, and I re-read the guard (below).
- The cross-phase interactions I probed hold:
  - hippocampus plain imports × x64 scoping × device choice;
  - narrowed handlers × cohort records;
  - call-time roots × parquet-only paths × the parity tool;
  - the conftest sentinel × real_data tests.
- What is left is text, one formatting slip, one test-order leak and one latent VIF mislabel.

Strengths:
- Every removal is backed by a grep or AST scan.
- Frozen-copy equivalence tests were added wherever a computation was rewritten.
- Parity was checked on the slice after each task.
- The data-root sentinel makes "a test writes live data" structurally impossible in the default run.
- The ledger records each scope departure as a ruling with its cost.

## Critical

None.

## Important

None.

## Minor

1. **`src/continuum_swr_replay/analysis/pfc/batched_glm.py:148`: Task 15 left the file
   unformatted.** One blank line is missing after `_in_float64`, before the `#:` comment.
   - It is the only touched file that was ruff-formatted at base and is not at HEAD. I checked
     every `.py` in the diff against its base blob with `ruff format --check`. The other 50
     "would reformat" files were already unformatted at base.
   - Scenario: the next `ruff format` on this file adds an unrelated hunk to someone's diff,
     against Global Constraint 4.

2. **`tests/unit/test_figure_script_fixes.py::test_the_panel_e_headers_fit_their_cells`
   depends on test order.** I reproduced it: `pytest tests/unit/test_figure_utilities.py
   <that test>` gives 1 failed, 54 passed. The failure predates the branch (T18 Concern 2).
   - Cause: the `set_figure_defaults` tests in `tests/unit/test_figure_utilities.py`
     (:113-170) leave `paper` rcParams set. The data-flow script sets no defaults, so the test
     is right and the leak is the bug.
   - Scenario: random ordering, `-k` selection or an xdist split turns the suite red.
   - Checked: an autouse `mpl.rc_context()` fixture around those tests makes the pair pass
     (55 passed), using a scratch plugin.

3. **`src/continuum_swr_replay/analysis/replay/fitting.py:1452`:
   `sm.add_constant(X)` defaults to `has_constant="skip"`** (T12 Concern 2, Phase D Minor 4).
   - With a column that is a nonzero constant overall, no intercept is added. Each `vif_<pred>`
     then reports its neighbour's VIF, and the last index raises IndexError.
   - Since T12-2, that IndexError is not in `EXPECTED_FIT_ERRORS`. It escapes the `:967`
     handler and throws away a conditional-Poisson fit that had otherwise succeeded, with the
     term marked unidentified.
   - I confirmed that `has_constant="add"` gives an identical matrix for plain designs and for
     all-zero columns. The NaN VIFs in the saved fits come from all-zero columns. The result
     differs only in the nonzero-constant case. No current fit hits it (T12 survey).

4. **`tests/README.md:136-140` became false during the branch.** It says `bandit_task.py` "is
   the Spyglass entry point and is uncoverable offline".
   - 2f7f7af8 removed its module-level `from spyglass.common import PositionIntervalMap`.
   - `tests/unit/test_position.py:293-319` now imports it offline and tests `load_data`'s
     orchestration with faked loaders.
   - The T15 report calls this pre-existing, but the base file did import Spyglass at :39.

5. **`CLAUDE.md:79`: the `runs/` line ("provenance, CLI, batch loops") leaves out the two
   modules this branch added**, `runs/threads.py` (Task 3) and `runs/device.py` (Task 16).
   - Scenario: an agent looking for where a runner pins BLAS threads or picks its JAX device
     does not find it from CLAUDE.md.

6. Already deferred, confirmed still present (see the triage table):
   - `fit_provenance.py:25-27` and the pickle-era text;
   - the `_LOG1P_STOP_TIME` alias;
   - the `"_contfrag"` literals that make CLAUDE.md convention 3 ("built and parsed only by
     `decoding/identifiers.py`") overstated;
   - `scripts/rate/plot_uncertainty_rate.py:52-58` runs `fit_uncertainty_rate.py`'s whole top
     level. That includes `pin_blas_threads()`, its `logging.basicConfig` (so the plotter logs
     in the fitter's format) and its `RuntimeWarning` ignore. Pre-existing.

## Triage of deferred minors

Sources: progress.md (PG) and the phase reviews (P0, PA, PB, PC, PD, PE1, PE2), plus the task
reports they point to (T#).

| # | Item | Source | Verdict | Action / question |
|---|---|---|---|---|
| 1 | Conditional-Poisson VIF uses `has_constant="skip"`; since T12-2 the IndexError discards a good fit | PG T12, PD M4, T12 C2 | **FIX** | `analysis/replay/fitting.py:1452`: `sm.add_constant(X)` → `sm.add_constant(X, has_constant="add")`. Add a unit test in `tests/unit/test_fit_error_handling.py`: a design with a nonzero-constant column returns each `vif_<pred>` under its own name and raises nothing. Plain designs are unchanged (verified). |
| 2 | `_LOG1P_STOP_TIME = LOG1P_STOP_TIME` leftover alias | PB T4 | **FIX** | `analysis/replay/contrast_aggregation.py:769`: delete the line. Replace `_LOG1P_STOP_TIME` with `LOG1P_STOP_TIME` at :771, :775, :1127, :1205. Verify with `tests/unit/test_contrast_aggregation.py` and `test_interaction_provenance.py`. |
| 3 | `fit_provenance` module docstring describes the removed `.pkl`→parquet redirect | PD M1 | **FIX** | `analysis/replay/fit_provenance.py:25-27` → "The hashed file is the file that was read: a table input is its `.parquet` path (the readers refuse `.pkl`); a `.pkl` input is a fitted bundle, hashed as it is." |
| 4 | Text Task 13 made false: `convert_tables_to_parquet.py:15-17` ("`read_table` still falls back to pickle") and `prepare_data.py:825-828` (cites `read_table(INPUT_DIR / "event_sequence_table.pkl")`) | PD M2 | **FIX** | :15-17 → "...They are historical snapshots; `read_table` reads only parquet, so read a backup's pickles with `pd.read_pickle`." :825/:828 → `event_sequence_table` and `read_table(table_path("event_sequence_table", INPUT_DIR))`. |
| 5 | Pre-existing messages and docstrings naming `*_table.pkl` for tables that are parquet | PD M2 | **FIX** (text only; no test matches these strings, checked) | Drop the `.pkl` suffix (name the table, e.g. `stop_table`) at: `prepare_data.py:387,448,561`; `retrospective.py:118`; `covariates/history/value_uncertainty.py:155,363`; `covariates/history/reward_history.py:44`; `model_specs.py:958`; `scripts/rate/fit_uncertainty_rate.py:14,257`; `scripts/coupling/fit_swr_power_model.py:172`; `scripts/rate/fit_per_stop_rate_models.py:8`; `scripts/tables/prepare_replay_content_data.py:13-20`; `scripts/tables/regenerate_stop_table.py:2,7`; `scripts/behavior/build_qvals_table.py:4`; `scripts/content/analyze_content_occupancy.py:233`. Leave `trial_leaf_table.pkl` (a real pickle). |
| 6 | `"_contfrag"` literal at 4 call sites; CLAUDE.md convention 3 is overstated | PB T7 | **FIX** | Add `CONTFRAG_SUFFIX = "_contfrag"` to `decoding/identifiers.py`. Use it in the parser (:278, `rest[:1] == [CONTFRAG_SUFFIX[1:]]`) and at `analysis/contfrag/result_loading.py:65`, `scripts/decoding/run_decoding_contfrag.py:81`, `scripts/qa/run_refits.py:365`, `scripts/tables/prepare_contfrag_replay_content_data.py:194`. Verify with `tests/unit/test_decode_identifiers.py` (its round trip) and the contfrag loader tests. |
| 7 | `TestFullPipelineRecovery` docstring overclaims | PA T1, PG | **FIX** | `tests/unit/test_replay_content_simulation.py:283` → "Simulated events through the production MAP-to-arm mapping and hard arm count (`simulate_full_pipeline`), then the production fitter." |
| 8 | test_figure_script_fixes order dependence (rcParams leak) | PG T18, T18 C2 | **FIX** | In `tests/unit/test_figure_utilities.py`, add `@pytest.fixture(autouse=True) def _restore_rcparams(): with mpl.rc_context(): yield` (mpl is already imported at :8). Verified with a scratch plugin: the failing pair passes. |
| 9 | `tests/README.md` coverage note: `bandit_task.py` "uncoverable offline" | PG T15 | **FIX** | `tests/README.md:136-140` → "`bandit_task.py` imports Spyglass only inside its loaders; its orchestration is tested with faked loaders (`unit/test_position.py`), and its Spyglass reads need `-m real_data`." |
| 10 | `batched_glm.py` left unformatted (this review) | Minor 1 | **FIX** | `uv run ruff format src/continuum_swr_replay/analysis/pfc/batched_glm.py` (one blank line at :148). |
| 11 | CLAUDE.md `runs/` line omits `threads` / `device` (this review; T16 report) | Minor 5 | **FIX** | `CLAUDE.md:79` → `├── runs/                  # run infrastructure: provenance, CLI, batch loops, BLAS thread pin (threads), JAX device choice (device)` |
| 12 | T12-1: several `IntervalLinearizedPosition` rows → the session-named graph (senor20201030_ 02_r1 has a stray "6 arm" linearization) | PG T12 | **LEAD** | Keep the tie-break to the session graph, or delete the stray "6 arm" linearization in the DB and make several rows a plain raise? |
| 13 | Content runner `--logit` aborts on live (M4, senor20201106 02_r1, "need covariance"); pre-existing | PG T12, T12 C1 | **LEAD** | Should the conditional-logit robustness ladder skip a session whose fit fails with an expected error, or keep refusing to publish? |
| 14 | T13-1: each `analysis.hippocampus.<leaf>` import now ~3 s and loads JAX/nld/statsmodels/dask | PG T13, T13 C1 | **LEAD** | Accept the cost, or keep the leaves light (restore a lazy `__init__` for hippocampus only)? No defect either way: no backend starts and x64 stays off at import (checked). |
| 15 | T16-3: `sweep_memoryless_controls` / `sweep_memoryless_heldout` compute on the GPU when one is visible, as at base | PG T16, PE1 M2 | **LEAD** | Pin them to the CPU (`default_jax_platforms("cpu")`)? That changes where future numbers are computed. The base numbers were GPU-computed. |
| 16 | T17-1: sidecars and `_invocation` record `sys.argv`, so an in-process `main([...])` records pytest's command line | PG T17 | **LEAD** | Should provenance record the parsed argv (about 12 writer sites to thread)? CLI runs are unaffected. |
| 17 | `scripts/figures/archive/generate_figure04_panels.py` cannot import (`plot_AICs` gone, at base too) and is the last zero-argument `parse_args()` | PG T17 | **LEAD** | Delete the archive script, restore `plot_AICs`, or keep it broken and noted? |
| 18 | T18 incident: 4 gitignored `.claude/research` outputs overwritten with current-table versions, with no backup (`event_rasters_triptych.png`, `dates_trends.png`, `peri_departure_rate.png/.csv`) | PG T18 | **LEAD** | The lead needs to be told. Are the regenerated versions acceptable, or does anything cite the July ones? |
| 19 | `.claude/research` scripts that redirect by assigning module roots (`mod.RC`, `*.PROCESSED_DATA_DIR`, `swr_source.SWR_INVENTORY_ROOT`) are no-ops since Task 19; `refresh_followup/f2_item2_refit.py` would write the live root if run at HEAD | PE2 M1, T19 round 2 | **LEAD** | These are the lead's own gitignored files. Update them to set `$CSR_PROCESSED_DATA_DIR`, or retire them? docs/README.md now carries the warning. |
| 20 | Power-model cohort record fields (`post_outcome_gate`, `…_rule`, `outcome_time_semantics`, `n_events_excluded_*`) never reach the CSV | PG T20 | **LEAD** | Extend `write_cohort_audit`'s columns, or drop the fields? Only a unit test reads them. |
| 21 | Pipeline schematic still shows stale columns: step 5 `arm_count_0 … arm_count_8, total_count`, step 6 `is_current, is_prev…`, step 7 `visit_recency, cumulative_reward` (`generate_supp_pipeline_schematic.py:52,57,62`) | PG T18, T18 C3 | **LEAD** | Fixing them changes a supplement figure, and step 7 needs a choice of which current covariates to name. Which names, and should it be redrawn now? |
| 22 | Commit trailer model name | P0 M6 | LEAVE | Already ruled: no change. |
| 23 | `--restrict` strict categorical dtype could false-positive | P0 rereview | LEAVE | No categorical column in the 16 tables. |
| 24 | `get_event_start_end_times` categorical → object | PA T2 | LEAVE | No reader depends on the dtype. |
| 25 | `clocks.trial_arrival_times` whole-frame `end_time` fallback | PA T2 | LEAVE | Unreachable in production, but `test_trial_sort_invariance._toy_trials` relies on it. Removing it means changing the fixture, for no gain. |
| 26 | 25 `# noqa: E402` in `run_pfc_analysis_bandit.py` | PA T3 | LEAVE | The same pattern as the other runners. |
| 27 | Walkthrough `.py`/`.ipynb` out of sync, stale stored outputs (incl. `qvals_table.pkl`), cell-35 docstring | PA rereview, PG T13 | LEAVE | Pre-existing. Refresh on the next notebook run. |
| 28 | `std_factor` / `sd_covariate` parse names | PB T4 | LEAVE | Works. Reading SD units from the declaration is a design change, not cleanup. |
| 29 | `interactions._FactorColumns` Mapping contract | PB T4 | LEAVE | Only `__getitem__` is used. A rewrite touches `interaction_product`'s signature. |
| 30 | `per_bin_rate_model._design_columns` one-line wrapper | PB T5 | LEAVE | 11 callers, including tests and a script. Churn without benefit. |
| 31 | `fit_per_bin_rate_models.py:458` uses `DERIVED_COVARIATE_SOURCES` but not the product declaration | PB T5 | LEAVE | No per-bin column is a declared product. The alternative is a private `covariate_timing._column_sources`. |
| 32 | 87 unshipped per-bin flag combinations reorder columns | PG T5 | LEAVE | Ruled. Columns are read by name. |
| 33 | `data_loaders.utils` still imports `parse_nwb_filename` | PB T7 | LEAVE | It is used there (`utils.py:180`). Not a leftover. |
| 34 | qvals ingestion raises `ValueError` where it raised `RegistryError` | PB T8 | LEAVE | `RegistryError` subclasses `ValueError` (`covariate_registry.py:450`). Nothing catches either. |
| 35 | Exploratory harmonization script cannot read a native-name qvals baseline | PB T8 | LEAVE | Unsupported one-off, documented in the script. |
| 36 | `eligibility.py:244`: ValueError before KeyError on multi-fault input | PC M1 | LEAVE | Only inputs with more than one fault. |
| 37 | `adapters.py:379,:670`: no check that `drops` and `dot` share a grid | PC M2 | LEAVE | Both always come from one frame. An assertion would guard a hypothetical caller. |
| 38 | `swr_detectors.py:844`: an event strictly inside a timestamp gap is now dropped | PC M3 | LEAVE | 0 gaps in real grids. The assumption is in the docstring. |
| 39 | T10 edge changes (views, 1-sample shape, dtype, NaN-bound raise) | PC M4 | LEAVE | Ledgered, 0 of 43,028 ripples affected. |
| 40 | Family-A `coef_intercept` → `coef_Intercept` on the next refit | PC M5, T11-1 | LEAVE | Ruled. Say it in the merge summary so an old-vs-new Family-A comparison knows. |
| 41 | `run_memoryless_detection` raw vs grid bounds | T9 | LEAVE | Measured: ≤1.78 ms, 0/40,053 reasons change. Out of scope by the brief. |
| 42 | per_bin logs keep no traceback | T12-3 | LEAVE | Ruled (about 500 expected failures per rung). |
| 43 | `choice_prediction.py:552` logs about 139 tracebacks on live | PD M6 | LEAVE | Log volume only. |
| 44 | Exploratory/plotter scripts edited by grep only, not run | PG T13 | LEAVE | `--help` 79/79 (T17). Exploratory scripts are unsupported. My static resolver finds no new broken import (below). |
| 45 | `check_fit_provenance --json` prints `.parquet` input paths | T13 C2 | LEAVE | The same bytes were hashed. |
| 46 | `write_table` maps any suffix to `.parquet` | T13-2 | LEAVE | Ruled. The converter relies on it. |
| 47 | `require_canonical_speed_threshold(allow_stricter=False)` is test-only | PG T14 | LEAVE | Simplifying it means editing `pfc/canonical_events.py:261`, which the plan's Task 14 fences off. |
| 48 | `data_loaders/__init__` lazy layer and Spyglass test stubs may be obsolete | PG T15 | LEAVE | The layer still keeps the loader modules out of a constants-only import. Stubs are harmless. |
| 49 | `test_contfrag_hash_safety.py` docstring/AST-load workaround is stale | T15 | LEAVE | Pre-existing (the lazy `data_loaders` predates the branch). Rewriting the test to import directly is not a small change. A follow-up candidate. |
| 50 | `quiet_spyglass` `@cache` loses its filters if first called inside `catch_warnings` | PD M3 | LEAVE | No production caller wraps a loader. Dropping the cache changes filter precedence on every call. Log only. |
| 51 | `swr_detectors.band_reproduction_error` and `release_nwb_files` import Spyglass without `quiet_spyglass` | PD M5 | LEAVE | They always run after a loader. Inert. |
| 52 | `_grid_rows` / `_report._invocation` / `_md` private names imported across modules | T16 C4 | LEAVE | As at base. |
| 53 | 4 scripts left unformatted | PG T17 | LEAVE | Unformatted at base. Formatting them is not this branch's diff. |
| 54 | `data_flow` and `diagnose_trial_detection` hand png+pdf loops; 16 `fig.savefig` sites | T18-1, T18-2 | LEAVE | Ruled. Converting changes PDF bytes or loses options. |
| 55 | Figures 02-04 not rendered after `save_figure(fig)` | T18-3 | LEAVE | `gcf()` is the returned figure in each (PE2 read the code). A render is Task 21 / lead territory. |
| 56 | `docs/design/2026-10-03-full-pipeline-refresh-design.md:200` mentions `PROCESSED_DATA_DIR` | PG T19 | LEAVE | A point-in-time design note. |
| 57 | Sentinel is a file, so `exists()` on the default root is False in tests | R19-8 | LEAVE | Ruled. That is the intent. |
| 58 | Contrast bundle has no `cohort_stop_drops` attr | PG T20 | LEAVE | Adding it changes a fitted bundle, so it waits for a refit. |
| 59 | `plot_uncertainty_rate.py:52-58` importlib-runs the fitter's top level | PE1 M1 | LEAVE | Pre-existing. Moving `POPULATIONS_TO_FIT` would change the plotter's log format and warning filters (Minor 6). A follow-up with the lead's choice of home. |
| 60 | `run_state_evidence.py:103`, `generate_figure01.py:37` set `JAX_PLATFORMS` at import | PE1 M3 | LEAVE | Both set it before any JAX import. `run_state_evidence` deliberately overrides the shell, which `default_jax_platforms`' setdefault would not. `scripts/README.md:31-32` describes the comparison and sweep families. |
| 61 | Archive bodies call `processed_data_dir()` at module level | PE2 M4 | LEAVE | No-CLI notebook bodies. The docstring was narrowed in 8c522b4c. |

**Counts: FIX 11, LEAD 10, LEAVE 40.**

## What I checked

- **Global constraints.**
  - `git status` lists exactly the 7 untracked paths of Constraint 3.
  - None of them is in `git diff --name-only 99034aab..HEAD`.
  - None has an mtime after the branch start (2026-10-24 21:38:55).
  - `find -newer <branch start>` over `Processed-Data/{replay_content,contfrag_replay_content,artifact_intervals,replay_detection_comparison}` to depth 3 finds nothing. The only Processed-Data child that changed is `_scratch_simplify_20261024`.
  - Outside the branch's own diff, the only files newer than the branch start are the 4 `.claude/research` outputs of the T18 incident.
- **Task 13 × Task 15 × Task 16 (imports, x64, device).**
  - In a fresh interpreter, importing `analysis.hippocampus` and `analysis.inventory_comparison` leaves `xla_bridge._backends == {}`, `jax_enable_x64 False`, `jax_platforms None`, and loads neither spyglass nor datajoint.
  - Every in-process device choice after imports goes through `jax.config.update`: `runs.device.default_jax_platforms` and `visualization.inventory_examples.use_memoryless_device`.
  - The two import-time pins (`run_state_evidence`, `generate_figure01`) come before any JAX import.
  - `pfc/gpu.py` sets only backend-init variables (`CUDA_VISIBLE_DEVICES`, `XLA_PYTHON_CLIENT_*`).
  - `runs.threads` imports no numpy, and `test_runs_threads` passes.
- **Task 12 × Task 20.**
  - The power runner's `n_modeled = Σ coefs["n_events"]` is restricted to fitted sessions exactly as `fit_keys` was.
  - An expected per-session failure drops the session from both counts.
  - `detection_inputs` (which received the moved `decode_marks`) has no exception handlers, so `AreaNotRecordedError` narrowing is untouched.
- **Task 19 × Task 13 × Task 0.**
  - `table_parity.py` changed only `_copy_cache` (`replay_content_dir()`) and two identifier imports since Task 0.
  - `table_path` builds `.parquet` under the call-time root.
  - parity.md's commands are unchanged and say to keep the variable unset.
- **Task 19 × tests.**
  - The real_data test files whose root patches were removed now resolve roots inside the test (`table_path(...)` / `replay_content_dir()` at call) or take `live_processed_data_dir`.
  - The one that writes (`test_from_cache_parity`) writes its cache to `tmp_path`.
  - No test subprocess drops the variable except `test_run_event_features.py:75`, which only imports.
- **Task 19 fix round 2 (not separately re-reviewed).**
  - I read `use_output_root` and every output site of `fit_swr_contrast_content.py`: 9 `model_dir("swr_contrast")` writes plus sidecars beside them.
  - No write goes through the symlinked `replay_content` entries.
  - A root under the live `Processed-Data` (scratch included) is refused by the final `is_relative_to(live)` check. That is conservative but correct.
- **Renamed/removed names across the tree.**
  - A static resolver over every tracked `.py` (including exploratory, archive, tests, notebooks) imported each `from continuum_swr_replay… import X` target and checked `X`.
  - A second pass checked `alias.attr` loads and `setattr(alias, "X")` / dotted-string targets.
  - The only failures are pre-existing: `exploratory/_univariate_coefs.py` (`replay.rate_model`, absent at base) and `archive/generate_figure04_panels.py` (`plot_AICs`). The negative tests in `test_paths.py` and local-variable shadowing in `test_heldout.py` are false positives.
  - Module-attribute stores are all sanctioned overrides: `prepare_data.SESSION_CACHE_DIR`, `epoch_register.REGISTER_PATH`, `figure_utilities.FIGURE_DIR` and the cohort-control monkeypatch.
  - The untracked replay_story script's package imports still resolve.
- **Dead code left behind.** Users at HEAD remain for `filter_event_intervals_by_speed`, `gap_free_stretches`, `get_labels`, `_speed_eligible_labels`, `resolved_table_path`, `table_exists`, `first_row_floats`, `bout_start_context` and `get_immobile_events` / `get_event_start_end_times` (archive). No calls pass the removed `speed_threshold=` / `immobility_threshold=` to the 8 builders. No imports of the dropped dependencies (optax, chex, absl, jinjaroot, kachery_client). sortingview and figurl need `kachery`, not `kachery-client`.
- **Docs.**
  - The CLAUDE.md diff matches the code.
  - Every `file.py` / `file.py:symbol` reference in CLAUDE.md resolves.
  - docs/README.md, tests/README.md and scripts/README.md describe the call-time roots, the moved modules and the lazy packages as implemented, except Minors 4-5.
  - The removed names appear only in "(removed: moved)" rows, point-in-time design notes and the notebooks' sanctioned legacy reads.
- **Tests run** (focused, no full suite):
  - guards (lazy imports, script imports, models-table mirror, layering, doc paths, paths, sentinel, pipeline map): 144 passed, 1 skipped;
  - cross-phase areas (runs, fit errors, table io, fit provenance, identifiers, interaction provenance, contrast aggregation, cohort audits, first-row context, per-bin terms, eligibility, speed guards): 592 passed, 1 skipped;
  - `test_runs_threads`, `test_filter_epoch_info`, `test_generate_all_figures` and `test_table_parity_script`: 27 passed;
  - the order-dependence reproduction and its rc_context fix.
- **Lint.**
  - `ruff check` is clean on every touched non-notebook `.py`. The 30 notebook findings are notebook-style expressions.
  - `ruff format --check` against base blobs: `batched_glm.py` is the only touched file the branch left unformatted.

## Declined to judge

- Whether `allow_stricter=False` should go: the plan fences off `pfc/canonical_events.py`.
- The scientific choices behind items 12-13, 15 and 20-21: those are the lead's.
- The quality of the moved inventory/memoryless code: Phase E1 AST-verified it as verbatim.
- Formatting of the 50 files that were already unformatted at base.

## What I cannot verify

- **Task 21:** the full-cohort regeneration parity against the live tables, `-m real_data` (including `test_matched_control`), and the default suite at 8c522b4c. All were still running in tmux.
- The implementer-reported equivalences that the phase reviews already took on report:
  - the slice SHA matches;
  - the 242-session rate and power refits;
  - the 85-epoch artifact drops;
  - the 199-file detection chain;
  - 83 argparse defaults and 57 root values with the variable set and unset;
  - the byte-identical figure renders.
- GPU-path behaviour (breeze) of the sweeps and example scripts after Task 16. I checked config and import order only.
- DB-dependent paths:
  - the DLC position branch (0 DLCPosV1 rows);
  - the several-row linearization tie-break;
  - `fetch_spatial_bandit_behavior.sh`.
- Figures 02-04 rendering through `save_figure(fig)`.
- Out-of-repo callers (other checkouts, notebooks elsewhere) of:
  - removed names;
  - `*_table.pkl` paths;
  - Family-A `coef_intercept`;
  - assignments to module roots.
