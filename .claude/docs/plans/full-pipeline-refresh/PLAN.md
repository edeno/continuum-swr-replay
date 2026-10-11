# Full pipeline refresh: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Status (2026-10-03):** the plan was revised after an independent review (3 blocking, 16
should-fix and 11 minor findings, all addressed below), and implementation has started at
Task 1. Task 1 pinned `ripple_detection` to `68e1784`. On the five pilot epochs, 763 HSE
events matched, 0 were missed and 62 were added; the largest SWR z-score difference was
2.56, at a trace end (at most 8.8e-4 more than 1 s from both ends); and non-local events
went from 2,365 under the 1.7.0 rule to 2,405 under 2.0's, with minimum sample counts {7}. Tasks 1–3 complete and reviewed; `C0` = `de156501c05b4381714919be54a872b3ac0e58fc`, the commit both decoders launch from. R0 done (backup root `Processed-Data/_pre_redecode_backup_20261003`). An incident during Task 2 overwrote the April decode of `chimi20200212_02_r1` (accepted loss; see the backup's `MISSING_APRIL_DECODES.md`).

**Status (2026-10-09):** Tasks 14–15 complete (one epoch register; one SWR source, the 2.0 Kay inventory). The lead chose to mask artifacts with a new, validated detector and to re-decode all 255 epochs with one SWR source; see Phase C (Tasks 16–18, R9–R13). Study C1 and its follow-ups (C1b, C1c) found that 43 % of j16's Kay events were not SWRs. Three tetrodes caused it. The lead chose a tetrode-selection rule for all four SWR detectors, and Task 17 now carries it. On 2026-10-09 the lead also chose a simplified design: one mask for everything, decoded events within 0.3 s of it dropped, the spike-line rule kept, no reference veto, and no post-step second (see the Phase C introduction).

**Status (2026-10-04):** the decoders' discrete-transition settings had silently followed changed library defaults (+32% non-local events on identical inputs). After validation the project lead chose setting P30-B (decision R10). Task 11 pins it. Its commit `C0'` replaces `C0` as the commit both decoders launch from, and R1 restarts from `C0'` into a new staging folder. Tasks 12 and 13 (held-out spike evidence per decoder state) were added at the project lead's request. **2026-10-05:** an exclusion audit led to Task 14 (one epoch register); 40 restorable epochs are populated and decoded as an incremental update after the 218-epoch analysis, which proceeds now. wilbur20210401 is registered as blocked.

**Goal:** rebuild every long-running computation on one set of inputs, and prove it. That
covers:
- the NonLocal and ContFrag HPC decodes;
- the Widloski run;
- the four SWR inventories;
- the tables, model refits, PFC analyses and figures.

**Architecture:** the existing runners are extended rather than replaced:
- `run_decoding_bandit.py` and `run_decoding_contfrag.py` gain a staging `--save_dir` and a
  safe resume;
- `run_memoryless_detection.py` gains the project's spike loader and a GPU-parallel driver;
- a new CPU runner, `run_swr_detection.py`, calls `ripple_detection` at `68e1784` on
  read-only Spyglass inputs;
- a new audit script checks that everything shares inputs.

Long computations run from runbooks (Phase B), with smoke and pilot gates and stop points
for the project lead's go-ahead.

**Tech stack:** Python 3.11, uv, NumPy/SciPy/pandas, JAX (GPU via a hand-installed
`jax-cuda12-plugin`), dask and dask-cuda, Spyglass (DataJoint, read-only),
`ripple_detection`, `non_local_detector`, pytest, ruff.

**Spec:** [docs/design/2026-10-03-full-pipeline-refresh-design.md](../../../../docs/design/2026-10-03-full-pipeline-refresh-design.md).
Read it first. This plan argues from it, and where the two disagree the spec wins.

## Global Constraints

**Environment**
- `ripple_detection` is pinned to
  `git+https://github.com/Eden-Kramer-Lab/ripple_detection.git@68e1784b183f161a568d3154325cf770730bfb83`.
  Manifests and logs record this commit, because the installed version string comes from
  the last tag (1.7.x) and does not identify it.
- Never run a plain `uv sync` in the main tree. Use `uv sync --inexact`: the CUDA plugin
  (`jax-cuda12-plugin[with-cuda]==0.8.0`) was hand-installed and is not in `uv.lock`.
- `dask-cuda==25.10.0`, the locked version, must be importable. Without it `run_batch`
  silently runs on a CPU cluster. After Task 2, any device list given without `dask_cuda`
  is refused.
- GPU checks run with `XLA_PYTHON_CLIENT_PREALLOCATE=false` on this shared host.
- Run everything through `uv run`. Times are in seconds.

**Scientific settings**
- Non-local events (R9): the package's minimum, `round(0.015 s / step)` samples. Every
  epoch's minimum must be the same (7 at our 500 Hz grids, as the package computes on absolute timestamps); a different one stops the run.
- Speed: every SWR detector runs with `speed_threshold=4.0`.
- Sampling rate: the nominal 1000 Hz is passed to every SWR detector and filter design.
- Zugaro:
  - filter `butter(3, [120, 180], btype="bandpass", fs=1000, output="sos")` with
    `sosfiltfilt`, odd padding;
  - padding and edge exclusion come from each filter's single-pass settling time to 1e-3,
    rounded up to 10 ms: 90 for Zugaro, about 170 for channel selection;
  - `low_threshold=2.0`, `high_threshold=5.0`, `minimum_inter_ripple_interval=0.050`,
    `minimum_duration=0.0`, `maximum_duration=None`, then keep `duration <= 0.150`.
- Zugaro channel choice, following `bz_GetBestRippleChan`:
  - 140–180 Hz, Butterworth order 4, zero-phase;
  - a centered RMS window of 12 ms rounded up to an odd count, so 13 samples;
  - score = mean RMS ÷ median RMS;
  - exclude channels with mean or median RMS below 1, in the stored LFP units;
  - an exact tie goes to the lowest electrode id;
  - if every channel is excluded, the epoch fails.
- Widloski full run: 100 shuffles, `--run-seed 20260922`.
- Canonical decode settings: HPC, `no_ripple` training, stationary transitions,
  non-local threshold 0.8, no figurl.

**What must not change**
- Spyglass is read-only: no inserts, populates or parameter rows.
- No figure script is edited; figures are regenerated by running them.
- Decodes launch from one commit (`C0`). Tasks 4–10 are developed in a separate git
  worktree and merged only after both decoders finish (see "Branch and worktrees").

**Conventions**
- One backup root is used throughout: `Processed-Data/_pre_redecode_backup_<R0 date>`,
  written `$B`, dated once in R0. Staging directories are named `_staging_redecode_<date>/`.
- Unit tests run offline. Any module that imports Spyglass at import time (for example
  `run_decoding_bandit`, `build_timebin_tables` through `bandit_task`) is imported with
  `spyglass` stubbed in `sys.modules`, as `tests/unit/test_events.py:330-345` does. A test
  that needs the database or a GPU is a smoke step in a runbook, not a unit test.
- GPUs: pass `--cuda_visible_devices` explicitly and name only cards idle at launch (≤ 2 GiB
  used, 0% utilization), since decoders claim 99% of a card's memory and the host is shared.
- Commit messages end with:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01KFtJm5LJcVyD5mXspXkcoY
  ```
- Lint and format with `uv run ruff check <files> && uv run ruff format <files>`.

## Review Focus

These five conditions are the ones most likely to break things, and no task would exercise
them by accident. Each has a test in the task that owns the code.

1. **A staged decode resumes after a crash.** The results must be read back from the
   staging folder, not the canonical one, and the reopened `results.nc` must not be
   rewritten (Task 2, `test_run_decode_reloads_from_save_dir`,
   `test_resume_rewrites_only_data_over_time`).
2. **The marks cache holds April-rule marks for the same epoch id.** They keep more spikes,
   so they pass the encoding-mark guard. The cache must never serve marks made for another
   decode or another rule (Task 4, `test_a_cache_from_another_decode_is_not_read`,
   `test_an_april_rule_cache_is_never_read`).
3. **Raw LFP columns are positions, not electrode ids.** A wrong column-to-electrode map
   references and filters the wrong channels. The band-reproduction check must fail on a
   permuted map (Task 8, `test_band_check_catches_a_permuted_channel_map`).
4. **The LFP has NaN rows or timestamp jumps.** The filters must never cross them, edges and
   short stretches must be unobserved, and kept rows must be free of edge transients
   (Task 7, `test_filter_never_crosses_a_gap`, `test_edges_are_unobserved`,
   `test_short_stretch_is_unobserved`, `test_kept_rows_have_no_edge_transient`).
5. **An artifact silently lacks epochs.** For example, a Widloski or SWR manifest missing an
   epoch with no recorded reason. The audit must flag it, not pass it (Task 10,
   `test_an_unexplained_absence_fails_the_audit`).

## Branch and worktrees

- **Main tree.** Work happens on `memoryless-sequence-detection`, in place.
  1. Tasks 1–3 are committed in the main tree.
  2. That commit is `C0`. Both decoders launch from it (runbook R1), and the main tree must
     not change until both finish. The package is installed editable, and each epoch's log
     records `git rev-parse HEAD` at its start.
  3. During R1, plan status updates are committed on the worktree branch, never in the main
     tree.
- **Worktree for Tasks 4–10.**
  1. Create it from `C0` with
     `git worktree add ../continuum-swr-replay-refresh-dev -b full-pipeline-refresh-code C0`.
  2. Inside it, run `uv sync --extra dev` to give it its own CPU environment for unit tests.
  3. Develop and commit there.
  4. After both decoders finish, fast-forward `memoryless-sequence-detection` to the
     worktree branch (`C1`).
  5. Run `uv sync --inexact` in the main tree, re-check `dask_cuda` and the GPUs, then
     remove the worktree.
- **Database and GPU smoke steps** (Tasks 6, 8 and 9) run in the main tree after the
  merge.

## Files

| File | Responsibility | Task |
|---|---|---|
| `pyproject.toml`, `uv.lock` | the pin | 1 |
| `.claude/research/refresh_version_shift/measure_version_shift.py` (gitignored) | MUA bursts, SWR z-score and non-local events under 1.7.0 vs `68e1784` | 1 |
| `src/continuum_swr_replay/decoding_common.py` | `--save_dir`, `--allow_cpu`, refusing a CPU fallback, the provenance line | 2 |
| `src/continuum_swr_replay/analysis/hippocampus/result_loading.py`, `.../contfrag/result_loading.py` | a `root=` argument | 2 |
| `scripts/decoding/run_decoding_bandit.py`, `scripts/decoding/run_decoding_contfrag.py` | save to and resume from `save_dir`; on resume, rewrite only `data_over_time` | 2 |
| `scripts/decoding/build_timebin_tables.py` | `--output-dir` | 3 |
| `scripts/decoding/prepare_contfrag_replay_content_data.py` | `--behavior-cache-dir` | 3 |
| `scripts/decoding/run_memoryless_detection.py` | marks via the loader, cache key, parallel driver, filters, provenance | 4, 5 |
| `src/continuum_swr_replay/analysis/hippocampus/memoryless_sequence/decode_marks.py`, `tests/unit/test_memoryless_decode_marks.py` | **deleted** | 4 |
| `src/continuum_swr_replay/data_loaders/lfp.py` | `_ripple_keys`, `get_ripple_band_inputs`, `check_band_electrodes` | 6 |
| `src/continuum_swr_replay/data_loaders/position.py`, `.../bandit_task.py` | `get_epoch_position_info`, used by both | 6 |
| `src/continuum_swr_replay/types.py` | `RippleBandInputs` | 6 |
| `src/continuum_swr_replay/analysis/hippocampus/swr_inventories.py` | filters, channel choice, detector settings, Yu diagnostics, band check, Kay checks | 7, 8 |
| `scripts/decoding/run_swr_detection.py` | SWR runner | 9 |
| `scripts/qa/audit_refresh_consistency.py` | Stage 8 audit and the Stage 5 tables manifest | 10 |
| docs: `scripts/README.md`; `docs/design/replay_detection_comparison/{overview,detector_comparison,01-yu-swr-detector,02-memoryless-sequence}.md` | what changed | 4, 5, 9, 10 |

---

# Phase A: code

## Task 1: pin `ripple_detection`, install dask-cuda, measure the version shift

**Files:**
- Modify: `pyproject.toml`, `uv.lock`
- Create (gitignored scratch):
  `.claude/research/refresh_version_shift/measure_version_shift.py`, writing to
  `.claude/research/refresh_version_shift/outputs/`

**Interfaces:**
- Consumes: `load_data(nwb_file_name, epoch_name, ripple_detector_name="Kay")`, which
  returns `hse_times` (dict by brain area) and `zscored_ripple_consensus_trace` (Series);
  and the April `Processed-Data/{epoch_id}_80_data_over_time.pkl`, whose
  `Non-Local_Continuous_prob` and `Non-Local_Fragmented_prob` columns feed the event count.
- Produces: the pinned environment that every later task assumes. `Zugaro_ripple_detector`,
  `Yu_ripple_detector`, `noise_threshold_diagnostics`, `minimum_sample_count` and
  `simulate.simulate_session` become importable, and `dask_cuda` imports.

The pilot epochs:

| NWB file | Epoch |
|---|---|
| `chimi20200225_.nwb` | `02_r1` |
| `wilbur20210329_.nwb` | `12_r6` |
| `j1620210707_.nwb` | `06_r3` |
| `peanut20201202_.nwb` | `10_r5` |
| `senor20201101_.nwb` | `06_r3` |

- [ ] **Step 1: measure under 1.7.0, before pinning.** For each pilot epoch, call
  `load_data` and write:
  - `outputs/<label>/<epoch_id>_hse.parquet`: `start_time` and `end_time` of
    `hse_times["HPC"]`;
  - `outputs/<label>/<epoch_id>_swr_zscore.parquet`: the consensus z-score Series.

  Here `<label>` is `before` or `after`. The run also records `ripple_detection.__version__`
  and
  `importlib.metadata.distribution("ripple-detection").read_text("direct_url.json")`.

  Run: `uv run python .claude/research/refresh_version_shift/measure_version_shift.py --label before`.
  Expected: five epochs, about 1.5 min each.
- [ ] **Step 2: pin, and install dask-cuda.**
  ```bash
  cp uv.lock /tmp/uv.lock.before_pin
  uv add --no-sync "ripple_detection @ git+https://github.com/Eden-Kramer-Lab/ripple_detection.git@68e1784b183f161a568d3154325cf770730bfb83"
  uv sync --inexact
  uv pip install "dask-cuda==25.10.0"
  XLA_PYTHON_CLIENT_PREALLOCATE=false uv run python -c "import jax, dask_cuda, ripple_detection; from ripple_detection import Zugaro_ripple_detector, Yu_ripple_detector, noise_threshold_diagnostics, minimum_sample_count; print(ripple_detection.__file__); print(jax.devices())"
  diff <(grep -A2 'name = "non-local-detector"' /tmp/uv.lock.before_pin) <(grep -A2 'name = "non-local-detector"' uv.lock)
  diff <(grep -A2 'name = "spyglass-neuro"' /tmp/uv.lock.before_pin) <(grep -A2 'name = "spyglass-neuro"' uv.lock)
  ```
  Expected:
  - `jax.devices()` lists `CudaDevice` entries;
  - `dask_cuda` imports;
  - both diffs are empty (`non_local_detector` `2d730c3`, Spyglass `18b2bff`).

  If `jax.devices()` shows only CPU, run
  `uv pip install "jax-cuda12-plugin[with-cuda]==0.8.0"` and check again. If either diff is
  not empty, stop and report.
- [ ] **Step 3: run the unit suite against the new package.**
  `uv run pytest tests/unit -q -x`.
  - A failure caused by the decided minimum-duration rule (R9) is fixed by updating the test
    to the new rule. Note each such test in the commit message.
  - Any other failure is a 2.0 API change: fix the call site in the same commit. Never pin
    back.
- [ ] **Step 4: measure under the pin.** Run the same script with `--label after`.
- [ ] **Step 5: compare, and write `outputs/version_shift_report.md`,** one row per epoch:
  - **HSE events:** each matched, missed or added by strict positive-duration overlap.
  - **SWR z-score:** the largest absolute difference on shared timestamps.
  - **Non-local events (R9):** threshold the April `Non-Local_Continuous_prob +
    Non-Local_Fragmented_prob` at > 0.8, then count runs under both rules:
    - 1.7.0: keep a run when its last minus first timestamp is ≥ 0.015;
    - 2.0: keep a run of at least `minimum_sample_count(time, 0.015)` samples.

    Record each epoch's `minimum_sample_count`.

  Put the totals in this plan's status line.
- [ ] **Step 6: commit** `pyproject.toml`, `uv.lock` and any test updates, with message
  "Pin ripple_detection to 68e1784 for the full refresh". `dask-cuda` stays a manual
  install, like the CUDA plugin; note it in `docs/README.md` beside the plugin's note.

**Acceptance:**
- the pin is in the lock, and `non_local_detector` and Spyglass are unchanged;
- the GPU is visible and `dask_cuda` imports;
- the unit suite passes;
- the report exists, with its numbers in the status line.

## Task 2: decoders write to staging, resume safely, refuse a CPU fallback, record provenance

**Files:**
- Modify:
  - `src/continuum_swr_replay/decoding_common.py`: `add_common_args`, `run_batch`, and a new
    `provenance_record`;
  - `src/continuum_swr_replay/analysis/hippocampus/result_loading.py`: `load_results`;
  - `src/continuum_swr_replay/analysis/contfrag/result_loading.py`: `load_contfrag_results`;
  - `scripts/decoding/run_decoding_bandit.py`: `run_decode` and `save_results_and_model`;
  - `scripts/decoding/run_decoding_contfrag.py`: the same two.
- Test: `tests/unit/test_decoding_save_dir.py` (new, offline: stub `spyglass` before
  importing the scripts)

**Interfaces:**
- Produces:
  - `add_common_args` adds `--save_dir` (`type=Path`, default `PROCESSED_DATA_DIR`) and
    `--allow_cpu` (`store_true`).
  - `run_batch` passes `save_dir=args.save_dir` to `run_decode_fn`. When `HAS_CUDA` is False
    and `--allow_cpu` was not given, it raises `RuntimeError` naming `dask_cuda`, before any
    cluster starts.
  - Both `run_decode(..., save_dir: Path = PROCESSED_DATA_DIR)`:
    - call `save_dir.mkdir(parents=True, exist_ok=True)`;
    - check `results_exist(save_dir, ...)`;
    - reload with the loader's `root=save_dir`;
    - save into `save_dir`.
  - `save_results_and_model(..., resumed: bool)`: when `resumed` is True, it writes only the
    `data_over_time` pickle and never reopens `results.nc`, the detector or the environment
    for writing.
  - `load_results(..., root: Path = PROCESSED_DATA_DIR)` and
    `load_contfrag_results(..., root: Path = PROCESSED_DATA_DIR)` read every file from
    `root`.
  - `provenance_record(repo: Path) -> dict` returns:
    - `commit` (`git rev-parse HEAD`) and `dirty` (`git status --porcelain
      --untracked-files=no` non-empty);
    - `uv_lock_sha256`;
    - `ripple_detection_commit` (`commit_id` from `direct_url.json`, or None);
    - `non_local_detector_version`;
    - `cuda_plugin_version` (from `importlib.metadata` for `jax-cuda12-plugin`, or None).

    Each `run_decode` logs it once after START as one line:
    `PROVENANCE ` followed by `json.dumps(record, sort_keys=True)`.

- [ ] **Step 1: write the failing tests.**
  - `test_save_dir_and_allow_cpu_defaults`: `parse_args([])` gives
    `save_dir == PROCESSED_DATA_DIR` and `allow_cpu is False`.
  - `test_run_batch_refuses_cpu_fallback`: with `decoding_common.HAS_CUDA = False` and no
    `--allow_cpu`, `run_batch` raises `RuntimeError` mentioning `dask_cuda`, and no cluster
    is constructed (monkeypatch `LocalCluster` to fail if called).
  - `test_run_batch_passes_save_dir`: set `HAS_CUDA = False` and `--allow_cpu`, monkeypatch
    `LocalCluster` to `lambda **kw: LocalCluster(n_workers=1, processes=False)`, and pass a
    one-row `get_epoch_info` and a fake `run_decode_fn` that records kwargs. Assert
    `save_dir == args.save_dir`.
  - `test_load_results_reads_from_root` and `test_load_contfrag_results_reads_from_root`:
    monkeypatch `xr.open_dataset` and the detector's `load_model` to record their paths.
    Assert both paths are under `root`.
  - `test_run_decode_reloads_from_save_dir` (Review Focus 1), for each script:
    - monkeypatch `load_data` to a minimal dict with HPC spikes;
    - make `results_exist` return True;
    - make the loader record kwargs, then raise a sentinel.

    Assert the loader got `root=save_dir`.
  - `test_resume_rewrites_only_data_over_time` (Review Focus 1), for each script:
    - write placeholder `results.nc`, `detector.pkl` and `environment.pkl` files into
      `tmp_path`;
    - call `save_results_and_model(..., resumed=True)` with stub objects whose save methods
      raise if called.

    Assert the three files' sha256 values are unchanged and the `data_over_time` pickle was
    written.
  - `test_save_dir_is_created`: a `save_dir` that doesn't exist yet is created before any
    write.
  - `test_provenance_record_fields`: run against the repo. All six keys are present,
    `commit` has 40 hex characters, and `uv_lock_sha256` equals the sha256 of `uv.lock`.
- [ ] **Step 2: run them and confirm they fail.**
  `uv run pytest tests/unit/test_decoding_save_dir.py -q`.
- [ ] **Step 3: implement.** Keep every default identical to today's behavior, except the
  CPU-fallback refusal.
- [ ] **Step 4: run the tests and confirm they pass.** Also run
  `uv run pytest tests/unit -q -k "decod or result_loading"`.
- [ ] **Step 5: lint, format and commit** ("Let both decoders stage, resume safely, refuse a
  CPU fallback and log provenance").

**Acceptance:**
- the tests pass;
- with no `--save_dir`, behavior is unchanged;
- no code path reads `PROCESSED_DATA_DIR` when a `save_dir` was given;
- a resume never rewrites `results.nc`;
- every epoch log carries one `PROVENANCE` line.

## Task 3: staging options for the timebin and ContFrag table builders

**Files:**
- Modify: `scripts/decoding/build_timebin_tables.py` (`main`) and
  `scripts/decoding/prepare_contfrag_replay_content_data.py` (its argument parser, and line
  347, where `SESSION_CACHE_DIR` is set from `--replay-content-dir`)
- Test: `tests/unit/test_timebin_tables.py` (extend) and
  `tests/unit/test_contfrag_prepare_cli.py` (new). Both run offline with Spyglass stubbed.

**Interfaces:**
- Produces:
  - `build_timebin_tables.py --output-dir` (`type=Path`, default `RC`): both
    `publish_canonical` calls write there.
  - `prepare_contfrag_replay_content_data.py --behavior-cache-dir` (`type=Path`, default
    `None`): when given, `replay_prepare_data.SESSION_CACHE_DIR` is set to it. When not
    given, it stays `replay_content_dir / "_session_cache"`, as today.

- [ ] **Step 1: write the failing tests.**
  - `test_output_dir_receives_both_tables`:
    - monkeypatch `find_sessions` to two fake sessions;
    - make `build_one` return small valid frames;
    - stub `validate`;
    - make `publish_canonical` record its directory.

    `main()` with `--output-dir tmp_path` sends both writes to `tmp_path`.
  - `test_behavior_cache_dir_overrides_the_replay_content_cache`: parse
    `["--replay-content-dir", "/a", "--behavior-cache-dir", "/b"]`, apply the script's
    configuration step, and assert `SESSION_CACHE_DIR == Path("/b")`. Without the flag it is
    `Path("/a/_session_cache")`. If the configuration lives inside `main`, refactor it into a
    function `configure_paths(args)` first, so it can be tested without running the build.
- [ ] **Step 2: run them and confirm they fail.**
- [ ] **Step 3: implement.**
- [ ] **Step 4: run the tests and confirm they pass,** including the files' existing tests.
- [ ] **Step 5: lint, format and commit** ("Let the timebin and ContFrag table builders work in
  staging").

  This commit is `C0`. Record its hash in the status line before runbook R1, commit the
  status line on the worktree branch, and create the worktree.

**Acceptance:** both tests pass, and both defaults are unchanged.

## Task 4: Widloski runner reads marks through the project's loader

Work in the worktree.

**Files:**
- Modify: `scripts/decoding/run_memoryless_detection.py`, changing:
  - the module docstring;
  - `marks_cache_path` and `load_marks`;
  - `fetch_raw_hpc_unit_marks`, which is deleted;
  - the `marks_loader` call in `run_one`;
  - the manifest's `"marks"` record.
- Delete:
  `src/continuum_swr_replay/analysis/hippocampus/memoryless_sequence/decode_marks.py` and
  `tests/unit/test_memoryless_decode_marks.py`.
- Test: `tests/unit/test_run_memoryless_detection.py`: rewrite `TestLoadMarks`, and update
  every `marks_loader` lambda to the new signature.
- Docs: `docs/design/replay_detection_comparison/02-memoryless-sequence.md`, the
  paragraphs naming `decode_marks`, the April rule or the marks cache.

**Interfaces:**
- Consumes:
  - `continuum_swr_replay.data_loaders.spikes.get_hpc_marks(nwb_file_name)`, which returns
    `(list[NDArray], list[NDArray])`;
  - `filter_spike_times(spike_times: dict, spike_waveform_features: dict, position_time: NDArray)`,
    which returns `(dict, dict)`, keeps spikes in `[position_time[0], position_time[-1]]`,
    and drops electrodes left empty.
- Produces:
  - `MARKS_RULE: str = "load_data: get_hpc_marks (tetrode sort groups; coincident spikes counted) then filter_spike_times over the decode's time span"`;
  - `MARKS_RULE_TAG: str = "loader"`;
  - `marks_cache_path(key: EpochKey, cache_dir: Path, detector_sha256: str) -> Path`,
    returning `cache_dir / f"{key.epoch_id}_{MARKS_RULE_TAG}_{detector_sha256[:16]}_marks.pkl"`;
  - `load_marks(key: EpochKey, cache_dir: Path, span_s: tuple[float, float], detector_sha256: str) -> tuple[list[NDArray], list[NDArray]]`;
  - the `marks_loader` callable signature becomes
    `(key, cache_dir, span_s, detector_sha256)`;
  - `run_one` passes the detector sha256 it already computes for its input hashes;
  - the manifest's `"marks"` record is
    `{"rule": MARKS_RULE, "detector_sha256": ..., "cache": <file name>, "sha256": marks_sha256(...)}`.

- [ ] **Step 1: write the failing tests.**
  - `test_marks_come_from_the_project_loader`: monkeypatch `get_hpc_marks` to return three
    electrodes: one with a spike before the span, one with all spikes outside it, and one
    normal. Assert the second is dropped, the out-of-span spike removed, and features kept
    aligned with times.
  - `test_a_cache_from_another_decode_is_not_read` (Review Focus 2): with a counting fake
    loader, `detector_sha256="a"*64` then `"b"*64` runs the loader twice, and the second
    result is not the first cache's content.
  - `test_an_april_rule_cache_is_never_read` (Review Focus 2): with
    `{epoch_id}_decode_time_marks.pkl` (the old name) holding sentinel arrays in
    `cache_dir`, `load_marks` calls the loader and returns its marks.
  - Update `test_the_record_names_the_marks_rule`: `rule == MARKS_RULE`, and
    `detector_sha256` equals the sha256 of the fixture's detector file.
- [ ] **Step 2: run them and confirm they fail.**
  `uv run pytest tests/unit/test_run_memoryless_detection.py -q`.
- [ ] **Step 3: implement.**
  - Delete `fetch_raw_hpc_unit_marks` and the `decode_marks` imports.
  - Delete the two `decode_marks` files.
  - Rewrite the docstring's marks paragraph.
  - Update spec 02's marks text.
- [ ] **Step 4: run the tests and confirm they pass.** Also run
  `uv run pytest tests/unit -q -k memoryless` and
  `git grep -n decode_marks -- src scripts tests docs`. Expected: no hits outside plan and
  spec history.
- [ ] **Step 5: lint, format and commit** ("Read the Widloski runner's marks through the
  project's loader").

**Acceptance:**
- the tests pass;
- `decode_marks` is gone from the code;
- the cache can only serve marks made for the same decode file under the same rule.

## Task 5: Widloski runner runs in parallel, like the decoder

Work in the worktree.

**Files:**
- Modify: `scripts/decoding/run_memoryless_detection.py` (`main`, `run_settings`, and new
  `select_epochs` and `run_epochs`)
- Test: `tests/unit/test_run_memoryless_detection.py` (extend `TestMain`, add
  `TestSelectEpochs`)
- Docs: spec 02's runner usage, and `scripts/README.md`'s entry for the runner

**Interfaces:**
- Consumes: `run_one(...)`, as left by Task 4; `decoding_common.provenance_record`,
  `setup_logger` and `HAS_CUDA`.
- Produces:
  - new CLI options: `--cuda_visible_devices` (str, default None, which keeps today's serial
    in-process run), `--local_scratch_dir` (default `.local_scratch`), `--date` (int, such as
    `20200225`) and `--epoch` (str, such as `02_r1`);
  - `select_epochs(epoch_ids: list[str], *, date: int | None, epoch: str | None) -> list[str]`,
    where the date is `int(nwb_stem[-8:])`, as `prepare_data.py:135` reads it, and the epoch
    is the key's `epoch_name`;
  - `run_epochs(epoch_ids, run_one_kwargs: dict, cluster_factory: Callable[[], Cluster]) -> Iterator[tuple[str, dict]]`.
    It submits one `run_one` per epoch to `Client(cluster_factory())` and yields
    `(epoch_id, record)` in completion order.
    - A failure is caught in the worker and returned as
      `{"status": "failed", "error": repr(e)}`.
    - Each task logs to `logs/{epoch_id}_memoryless.log` via `setup_logger`.
  - Only `main` calls `write_manifest`, after each yielded record.
  - A device list given while `HAS_CUDA` is False exits with an error naming `dask_cuda`.
  - With a device list, `cluster_factory` builds
    `LocalCUDACluster(CUDA_VISIBLE_DEVICES=..., local_directory=args.local_scratch_dir)`.
  - `run_settings` also records `provenance_record(repo)` without its `dirty` field (the run
    id already says dirty). A rerun under any other `uv_lock_sha256`,
    `ripple_detection_commit` or `cuda_plugin_version` is refused, as library versions
    already are.

- [ ] **Step 1: write the failing tests.**
  - `test_date_and_epoch_filters_select_epochs`: of `["chimi20200225_02_r1_HPC",
    "chimi20200225_04_r2_HPC", "chimi20200226_02_r1_HPC", "j1620210707_06_r3_HPC"]`:
    - `date=20200225, epoch="02_r1"` selects the first only;
    - `date=20210707` selects the j16 epoch;
    - both None return everything.
  - `test_parallel_run_records_each_epoch_and_a_failure`: use
    `cluster_factory = lambda: LocalCluster(n_workers=2, processes=False)` and a fake
    `run_one` that raises for one epoch. The manifest has `ok` for the rest and `failed`
    with the error for that one, `main` exits non-zero, and the other outputs exist.
  - `test_only_the_main_thread_writes_the_manifest`: monkeypatch `write_manifest` to record
    `threading.current_thread().name`. Every call comes from `MainThread`, while the fake
    `run_one` runs in worker threads.
  - `test_a_device_list_without_dask_cuda_is_refused`: monkeypatch `HAS_CUDA=False` and pass
    `--cuda_visible_devices 0`. `main` exits with a message naming `dask_cuda` and writes no
    manifest.
  - `test_settings_record_the_environment`: `run_settings(...)` contains `uv_lock_sha256`,
    `ripple_detection_commit` and `cuda_plugin_version`. Changing any of them makes `main`
    refuse the existing run id.
  - The existing serial-path tests stay green unchanged.
- [ ] **Step 2: run them and confirm they fail.**
- [ ] **Step 3: implement.**
- [ ] **Step 4: run the whole runner test file and confirm it passes.**
- [ ] **Step 5: lint, format, update the docs and commit** ("Run the Widloski detector across
  GPUs like the decoder").

**Acceptance:**
- the tests pass;
- a serial run (no `--cuda_visible_devices`) behaves exactly as before;
- the manifest has a single writer.

## Task 6: ripple-band inputs and the shared position helper, read-only

Work in the worktree for the pure parts. Do the database smoke in the main tree after the
merge.

**Files:**
- Modify:
  - `src/continuum_swr_replay/data_loaders/lfp.py`: refactor `get_ripple`'s restriction dicts
    into `_ripple_keys`; add `get_ripple_band_inputs` and `check_band_electrodes`;
  - `src/continuum_swr_replay/data_loaders/position.py`: add `get_epoch_position_info`;
  - `src/continuum_swr_replay/data_loaders/bandit_task.py`: `load_data` calls
    `get_epoch_position_info` in place of its own `PositionIntervalMap` lookup,
    `get_position_info` call and NaN-row drop (`bandit_task.py:335-353`);
  - `src/continuum_swr_replay/types.py`: add `RippleBandInputs`.
- Test: `tests/unit/test_lfp.py` and `tests/unit/test_position.py` (extend or create,
  offline)

**Interfaces:**
- Produces:
  ```python
  def _ripple_keys(nwb_file_name: str, epoch_name: str) -> tuple[dict, dict]:
      """(ripple_key, lfp_band_restriction) exactly as get_ripple builds them today."""

  def get_epoch_position_info(nwb_file_name: str, epoch_name: str) -> PositionData:
      """PositionIntervalMap lookup, get_position_info, then load_data's drop of rows with
      NaN head_position_x or head_position_y, with the same log line. load_data and the
      SWR runner both call this, so they share speed and span."""

  def check_band_electrodes(band_electrode_ids, selection_electrode_ids) -> None:
      """Raise ValueError naming missing and extra electrodes when the sets differ."""

  class RippleBandInputs(TypedDict):
      time: NDArray[np.float64]              # (n_time,), the band's timestamps, seconds
      ripple_band: NDArray[np.float64]       # (n_time, n_channels), stored LFPBandV1 as float64
      ripple_band_dtype: str                 # the band's stored dtype
      raw_referenced: NDArray[np.float64]    # (n_time, n_channels), referenced raw LFP as float64
      raw_referenced_native: NDArray         # (n_time_raw, n_channels), referenced in the stored dtype, as LFPBandV1 does it
      raw_timestamps: NDArray[np.float64]    # (n_time_raw,), timestamps of raw_referenced_native
      reference_overflow_count: int          # samples where native-dtype subtraction wrapped
      electrode_ids: NDArray[np.int64]       # (n_channels,), column order of every array
      reference_ids: NDArray[np.int64]       # (n_channels,)
      lfp_band_elect_index: NDArray[np.int64]  # Spyglass's electrode index, for filter_data
      band_valid_interval: object            # the Spyglass Interval make filtered over (_compute_interval_times)
      decimation: int
      sampling_frequency: float              # nominal: lfp_band_sampling_rate (1000)
      band_filter_coeff: NDArray[np.float64] # FirFilterParameters filter_coeff
      band_filter_name: str                  # "ripple_150_250"
      lfp_filter_name: str                   # the LFPV1 filter that made the raw LFP
      lfp_units: str                         # the electrical series' unit, for the dead-channel guard

  def get_ripple_band_inputs(nwb_file_name: str, epoch_name: str) -> RippleBandInputs: ...
  ```
- Requirements for `get_ripple_band_inputs`:
  - Take the band key from `_ripple_keys`.
  - Read the band's electrodes and references from `LFPBandSelection.LFPBandElectrode`
    restricted by that band key.
  - Call `check_band_electrodes` against `RippleLFPSelection.RippleLFPElectrode` restricted
    by the same band key plus `group_name="CA1"`. `RippleLFPSelection` is keyed on
    `LFPBandV1` (Spyglass `ripple.py:33-43`), so an unrestricted query unions every epoch.
  - Reproduce `LFPBandV1.make`'s inputs (Spyglass `lfp_band.py:498-560`):
    - the interval from `LFPBandV1()._compute_interval_times(key, lfp_key)`;
    - the referenced data from
      `LFPBandV1()._load_lfp_data_with_referencing(lfp_object, elect_ids, ref_ids, interval)`,
      which works in the stored dtype;
    - the electrode index and the decimation.
  - Count overflow by comparing the native-dtype subtraction with a float64 subtraction of
    the same columns.
  - Restrict the float64 arrays to the band's timestamps.
  - Insert nothing.

- [ ] **Step 1: write the failing tests.** All offline.
  - `test_band_electrode_sets_must_match`: equal sets pass. A missing or extra electrode
    raises `ValueError` naming it.
  - `test_ripple_keys_are_what_get_ripple_used`: pin both dicts for `("chimi20200225_.nwb",
    "02_r1")` to the literal values `get_ripple` builds today. These are `group_name "CA1"`,
    `ripple_param_name "default"`, `filter_name "ripple_150_250"`, `filter_sampling_rate
    1000`, `target_interval_list_name "02_r1 noPrePostTrialTimes"`, and
    `lfp_band_sampling_rate 1000`.
  - `test_epoch_position_drops_nan_rows_like_load_data`: monkeypatch the two Spyglass calls
    to return a position frame with two NaN-position rows. Both rows are dropped and the rest
    kept exactly.
- [ ] **Step 2: run them and confirm they fail.**
- [ ] **Step 3: implement.**
- [ ] **Step 4: run the tests and confirm they pass,** including all of `test_lfp.py` and
  every `load_data` unit test.
- [ ] **Step 5: run the database smoke, in the main tree after the merge.**
  - Call `get_ripple_band_inputs("chimi20200225_.nwb", "02_r1")`. Expected: 22 channels,
    electrode ids `[0, 4, 8, 12, 20, 24, …, 92]`, references `{16, 81}`,
    `sampling_frequency == 1000.0` and `reference_overflow_count == 0`.
  - `get_ripple(...)` output, and `get_epoch_position_info(...)` against `load_data`'s
    `position_info`, are identical to the `C0` reference pickled in R1. Compare with
    `pandas.testing.assert_frame_equal`.

  Record both results in the status line.
- [ ] **Step 6: lint, format and commit** ("Read the ripple-band inputs and share the epoch's
  position with load_data").

**Acceptance:**
- the tests pass;
- `get_ripple` and `load_data`'s position are byte-identical to `C0` on the smoke epoch;
- the fetch writes nothing.

## Task 7: Zugaro filtering and channel choice

Work in the worktree.

**Files:**
- Create: `src/continuum_swr_replay/analysis/hippocampus/swr_inventories.py`
- Test: `tests/unit/test_swr_inventories.py`

**Interfaces:**
- Produces:
  ```python
  ZUGARO_BAND_HZ = (120.0, 180.0); ZUGARO_FILTER_ORDER = 3
  CHANNEL_SELECTION_BAND_HZ = (140.0, 180.0); CHANNEL_SELECTION_FILTER_ORDER = 4
  CHANNEL_SELECTION_RMS_WINDOW_S = 15 / 1250      # 0.012 s, bz_GetBestRippleChan's 15 samples at 1250 Hz
  DEAD_CHANNEL_RMS = 1.0                          # bz_GetBestRippleChan's guard, in stored LFP units
  SETTLING_TOLERANCE = 1e-3
  ZUGARO_MAXIMUM_DURATION_S = 0.150

  def design_bandpass_sos(sampling_frequency: float, band_hz: tuple[float, float],
                          order: int) -> NDArray:
      """butter(order, band_hz, btype="bandpass", fs=sampling_frequency, output="sos")."""

  def settling_samples(sos: NDArray, sampling_frequency: float,
                       tolerance: float = SETTLING_TOLERANCE) -> int:
      """Samples until the single-pass impulse response's magnitude stays below tolerance
      x its peak, rounded up to a whole 10 ms. The padding and the edge exclusion."""

  def odd_window_samples(window_s: float, sampling_frequency: float) -> int:
      """round(window_s * fs), then up to an odd count (the package's convention)."""

  def gap_free_stretches(time: NDArray, values: NDArray) -> NDArray:
      """(n_stretches, 2) [start, stop) row indices. A row with any non-finite value, or a
      step over 1.5 x the median step (the package's rule), ends a stretch."""

  def zero_phase_bandpass(values: NDArray, time: NDArray, sampling_frequency: float,
                          band_hz: tuple[float, float], order: int) -> NDArray:
      """design_bandpass_sos, applied with sosfiltfilt(padtype="odd",
      padlen=settling_samples(...)) per gap-free stretch, in float64. Rows outside
      stretches, the first and last settling_samples rows of each stretch, and every row
      of a stretch too short for both are NaN.
      values: (n_time, n_channels); returns the same shape."""

  class NoUsableChannelError(ValueError): ...

  @dataclass(frozen=True)
  class ChannelChoice:
      electrode_id: int
      scores: dict[int, float]          # electrode id -> mean RMS / median RMS
      excluded: dict[int, str]          # electrode id -> reason

  def best_ripple_channel(values: NDArray, time: NDArray, sampling_frequency: float,
                          electrode_ids: NDArray) -> ChannelChoice:
      """bz_GetBestRippleChan: zero_phase_bandpass at 140-180 Hz (order 4); centered moving
      RMS over odd_window_samples(0.012, fs) samples within each stretch; score
      mean/median over finite samples; exclude mean or median < DEAD_CHANNEL_RMS; the
      highest score wins; an exact tie goes to the lowest electrode id; raise
      NoUsableChannelError when every channel is excluded."""
  ```

- [ ] **Step 1: write the failing tests.** All are deterministic, with seeded RNGs.
  - `test_zugaro_band_gains`: with
    `sos = design_bandpass_sos(1000.0, ZUGARO_BAND_HZ, ZUGARO_FILTER_ORDER)`, assert these
    zero-phase gains, as `abs(H)**2` from `sosfreqz`:

    | Frequency (Hz) | Gain |
    |---|---|
    | 120 | 0.500 ± 1e-3 |
    | 180 | 0.500 ± 1e-3 |
    | 150 | 1.000 ± 1e-3 |
    | 100 | 0.0222 ± 5e-4 |
    | 200 | 0.0573 ± 5e-4 |
    | 250 | ≤ 0.0011 |
    | 60 | ≤ 1e-4 |
  - `test_selection_band_gains`: the 140–180 Hz order-4 design has zero-phase gain
    0.500 ± 1e-3 at 140 and 180 Hz.
  - `test_settling_lengths`: `settling_samples` is 90 for the Zugaro design at 1000 Hz, and
    160–180 for the selection design (the review measured about 166 ms).
  - `test_sos_matches_transfer_function_form`: on seeded white noise (200,000 samples),
    `sosfiltfilt(sos, x, padtype="odd", padlen=90)` and
    `filtfilt(b, a, x, padtype="odd", padlen=90)` with `b, a` from the same design differ by
    under 1e-10 relative.
  - `test_passes_ripples_and_rejects_neighbors_at_1000_hz`: on a 10 s, 1000 Hz signal, a
    150 Hz unit sinusoid keeps an interior amplitude within 1% of 1, and a 100 Hz one falls
    below 0.03.
  - `test_stretches_split_at_nan_and_time_jump`: a time grid with one 1.6× median step and
    one NaN row gives three stretches with exact bounds.
  - `test_filter_never_crosses_a_gap` (Review Focus 4): a large pulse at the end of stretch
    1 leaves stretch 2's output exactly equal to filtering stretch 2 alone.
  - `test_edges_are_unobserved` (Review Focus 4): the first and last `settling_samples`
    rows of each stretch are NaN, and the interior is finite.
  - `test_short_stretch_is_unobserved` (Review Focus 4): a stretch of at most
    `2 * settling_samples` rows is all NaN.
  - `test_kept_rows_have_no_edge_transient` (Review Focus 4): for each design, a 3 s
    stretch cut from a 60 s seeded noise signal, filtered on its own, matches the long
    signal's filtered output over the kept rows to within 1e-3 of the output's standard
    deviation.
  - `test_bursty_channel_wins`: channel A carries sparse 150–170 Hz bursts over low noise;
    channel B carries steady 160 Hz power of equal total energy. Both have a baseline RMS of
    at least 5 after filtering. A wins.
  - `test_dead_channel_guard_is_pinned`: a channel scaled so its filtered mean RMS is 0.99
    is excluded, and one at 1.01 is kept.
  - `test_all_channels_excluded_raises`: two all-zero channels raise
    `NoUsableChannelError`.
  - `test_tie_goes_to_lowest_electrode_id`: two identical channels with ids 40 and 8 select
    8.
  - `test_rms_window_is_13_samples_at_1000_hz`:
    `odd_window_samples(0.012, 1000.0) == 13`.
- [ ] **Step 2: run them and confirm they fail.**
  `uv run pytest tests/unit/test_swr_inventories.py -q`.
- [ ] **Step 3: implement,** with NumPy-style docstrings giving shapes.
- [ ] **Step 4: run the tests and confirm they pass.**
- [ ] **Step 5: lint, format and commit** ("Filter for Zugaro and pick its channel as
  bz_GetBestRippleChan does").

**Acceptance:** every test passes, and every gain, length and setting in Global Constraints is
pinned by a test.

## Task 8: detector settings, Yu diagnostics, band check, Kay version checks

Work in the worktree.

**Files:**
- Modify: `src/continuum_swr_replay/analysis/hippocampus/swr_inventories.py`
- Test: `tests/unit/test_swr_inventories.py` (extend)

**Interfaces:**
- Consumes:
  - from Task 6: `RippleBandInputs`;
  - from Task 7: everything;
  - from `ripple_detection`: `Kay_ripple_detector`, `Karlsson_ripple_detector`,
    `Yu_ripple_detector`, `Zugaro_ripple_detector`, `filter_ripple_band`,
    `noise_threshold_diagnostics`, and `simulate.simulate_session` / `simulate_time`;
  - from Phase 3: `analysis.inventory_comparison.schema.inventory_from_intervals`,
    `combine_inventories`, `SessionIdentity`, and
    `analysis.inventory_comparison.overlap.overlap_pairs`.
- Produces:
  ```python
  DETECTOR_SETTINGS: Mapping[str, Mapping[str, object]] = MappingProxyType({
      "kay": {"speed_threshold": 4.0, "minimum_duration": 0.015, "zscore_threshold": 2.0,
              "smoothing_sigma": 0.004, "close_ripple_threshold": 0.0},
      "karlsson": {"speed_threshold": 4.0, "minimum_duration": 0.015, "zscore_threshold": 3.0,
                   "smoothing_sigma": 0.004, "close_ripple_threshold": 0.0},
      "yu": {"speed_threshold": 4.0, "minimum_duration": 0.020, "percentile": 99.99,
             "smoothing_sigma": 0.004, "close_ripple_threshold": 0.0, "zscore_per_channel": True},
      "zugaro": {"speed_threshold": 4.0, "low_threshold": 2.0, "high_threshold": 5.0,
                 "minimum_inter_ripple_interval": 0.050, "minimum_duration": 0.0,
                 "maximum_duration": None},
  })

  def cap_zugaro_duration(events: pd.DataFrame) -> tuple[pd.DataFrame, int]:
      """Keep duration <= ZUGARO_MAXIMUM_DURATION_S; return the kept events and the count
      dropped."""

  def run_swr_detectors(time: NDArray, ripple_band: NDArray, zugaro_lfp: NDArray,
                        speed: NDArray, sampling_frequency: float
                        ) -> dict[str, pd.DataFrame | Exception]:
      """Native event tables keyed "kay", "karlsson", "yu", "zugaro". A detector that
      raises is returned as its exception, so the others still report. zugaro_lfp is
      (n_time, 1), already filtered (NaN rows unobserved); Zugaro's table is capped."""

  def yu_diagnostics(time: NDArray, ripple_band: NDArray, speed: NDArray,
                     sampling_frequency: float) -> dict:
      """noise_threshold_diagnostics on the Yu consensus over immobile samples, with the
      settings in DETECTOR_SETTINGS["yu"]: histogram grid and edges, mode, threshold, and
      the normalization center and scale. Computed whether or not any event is found."""

  def band_reproduction_error(inputs: RippleBandInputs,
                              fir_filter: Callable | None = None) -> tuple[float, float]:
      """(max |re-filtered - stored| on shared timestamps, tolerance). fir_filter defaults
      to spyglass FirFilterParameters().filter_data, called with LFPBandV1.make's exact
      arguments: raw_timestamps, raw_referenced_native, band_filter_coeff,
      band_valid_interval.times, lfp_band_elect_index, decimation. The tolerance is 1.0
      for an integer band dtype, else 1e-6 x the stored band's standard deviation."""

  def kay_version_check(direct: pd.DataFrame, canonical: pd.DataFrame, *,
                        session: SessionIdentity, run_id: str,
                        speed_time_s: NDArray, speed_cm_per_s: NDArray) -> pd.DataFrame:
      """One row per event of either set: detector ("kay" or "kay_spyglass"), event_key,
      status ("matched", "added" or "missed"), partner_event_key (largest-overlap partner
      when matched), start_shift_s, end_shift_s, whole_event_speed_eligible. Overlap is
      overlap_pairs' strict positive-duration rule. Every event is kept, eligible or not;
      the column lets the report split them."""
  ```

- [ ] **Step 1: write the failing tests.**
  - `test_detector_settings_are_pinned`: an exact equality of `DETECTOR_SETTINGS` with the
    literal above.
  - `test_zugaro_keeps_150_ms_and_drops_longer`: durations `[0.149, 0.150, 0.151]` keep two
    and drop one.
  - `test_detectors_find_injected_ripples_at_1000_hz`:
    - generate `time = simulate_time(60_000, 1000)` and
      `simulate_session(time, [5, 15, 25, 35, 45], rng=0, ripple_frequency=(150, 175))`;
    - the band is `filter_ripple_band(session.lfps, sampling_frequency=1000)`;
    - Zugaro's input is `zero_phase_bandpass` of the raw channel chosen by
      `best_ripple_channel`.

    Each detector overlaps at least 4 of the 5 injected times. If Yu cannot estimate its
    threshold on simulated data, stop and report rather than loosen the test.
  - `test_zugaro_ignores_a_230_hz_ripple`: the same, but with
    `ripple_frequency=(229, 231)`. Zugaro detects none of the five, which confirms its band.
  - `test_a_failing_detector_does_not_hide_the_others`: monkeypatch `Yu_ripple_detector` to
    raise. `run_swr_detectors` returns the exception for `"yu"` and tables for the other
    three.
  - `test_yu_diagnostics_match_the_detector`: on the simulated session, `yu_diagnostics`
    gives a threshold equal (to 1e-9) to the events' `detection_threshold_zscore`.
  - `test_band_check_passes_on_its_own_filter`: a fake `fir_filter` (`np.convolve` with a
    fixed kernel) builds the stored band from fake raw input. The error is 0.
  - `test_band_check_catches_a_permuted_channel_map` (Review Focus 3): the same fake, but
    the raw columns are permuted against the band, so the error far exceeds the tolerance.
  - `test_kay_check_statuses`: direct `[(1.0, 1.1), (2.0, 2.1), (5.0, 5.1)]`, canonical
    `[(1.05, 1.12), (3.0, 3.1), (5.1, 5.2)]`. Expected:
    - the 1.0 event is matched both ways;
    - direct 2.0 and 5.0 are `added`, since 5.0 only touches 5.1 with zero duration;
    - canonical 3.0 and 5.1 are `missed`;
    - every row has `whole_event_speed_eligible`.
- [ ] **Step 2: run them and confirm they fail.**
- [ ] **Step 3: implement.**
- [ ] **Step 4: run the tests and confirm they pass.**
- [ ] **Step 5: lint, format and commit** ("Run Kay, Karlsson, Yu and Zugaro on shared inputs,
  with input and version checks").

**Acceptance:** the tests pass, and the settings exactly match spec Stage 3.

## Task 9: the SWR runner

Work in the worktree. Run the smoke in the main tree after the merge.

**Files:**
- Create: `scripts/decoding/run_swr_detection.py`
- Test: `tests/unit/test_run_swr_detection.py`
- Docs:
  - `scripts/README.md`: a runner entry;
  - `docs/design/replay_detection_comparison/overview.md`: seven inventories, a Zugaro row,
    and D1 resolved for Zugaro;
  - `detector_comparison.md`: Zugaro's row;
  - `01-yu-swr-detector.md`: the direct path now produced by this runner for all four
    detectors, and `zscore_per_channel` replacing `zscore_per_tetrode`.

**Interfaces:**
- Consumes:
  - Task 6: `get_ripple_band_inputs`, `get_epoch_position_info`;
  - Task 8: everything;
  - `decoding_common.setup_logger`, `filter_epoch_info`, `provenance_record`;
  - `data_loaders.get_epoch_info`;
  - Spyglass `RippleTimesV1`, read through `_ripple_keys`. Its `fetch1_dataframe` gives
    the canonical events, and `get_ripple_lfps_and_position_info(key)` gives Spyglass's own
    inputs for the version-only Kay check.
- Produces:
  - `KNOWN_NO_INPUT_EPOCHS: Mapping[tuple[str, str], str]`, mapping
    `(nwb_file_name, epoch_name)` to a reason. It holds the epochs documented in
    `data_loaders/constants.py:150-162`:
    - `senor20201104_` 06_r3 and 08_r4: no `LFPBandV1`;
    - `senor20201105_` 09_r4: no `TrackGraph`;
    - `wilbur20210404_` 02_r1, 04_r2, 06_r3, 08_r4 and 10_r5: no CA1 extraction.
  - `epoch_id(nwb_file_name, epoch_name) -> str`, returning `"{stem}_{epoch}_HPC"`.
  - `run_epoch(nwb_file_name, epoch_name, *, output_dir: Path, fetch_inputs=get_ripple_band_inputs, fetch_position=get_epoch_position_info, fetch_canonical_kay=..., fetch_spyglass_kay_inputs=...) -> dict`.
    It writes, under `output_dir`:
    - `{epoch_id}_{detector}_events.parquet` for each detector that succeeded;
    - `{epoch_id}_swr_observed.parquet`, with columns `scope` (`"stored_band"` or
      `"zugaro"`), `start_s` and `end_s`;
    - `{epoch_id}_kay_version_check.parquet`, with a `comparison` column:
      `"version_only"` or `"version_and_inputs"`.

    It returns the manifest record:
    - `status` (`ok` when every detector succeeded, `partial` when some did), and a
      per-detector `detectors: {name: "ok" | error}`;
    - `provenance_record(...)` and `numpy` / `scipy` versions;
    - `electrode_ids`, `reference_ids`;
    - `zugaro_channel` (`{electrode_id, scores, excluded}`);
    - `band_check` (`{max_abs_difference, tolerance, band_dtype, reference_overflow_count}`);
    - `yu_diagnostics`, from `yu_diagnostics`;
    - `sampling` (`{nominal, measured}`), `lfp_filter_name`, `lfp_units`;
    - `position_span_s`, `n_events` per detector, `zugaro_dropped_over_150ms`;
    - `n_detecting_tetrodes`, `meets_published_tetrode_criterion: "unknown"`;
    - `input_sha256` (`{time, ripple_band, raw_referenced}`).
  - Speed: `head_speed` from `get_epoch_position_info(...)`, interpolated onto LFP times with
    `np.interp(..., left=np.nan, right=np.nan)`. LFP rows outside
    `[position_time[0], position_time[-1]]` are dropped.
  - Skips and failures:
    - an epoch in `KNOWN_NO_INPUT_EPOCHS` is `{"status": "skipped", "reason": ...}` without
      touching the database;
    - `NoUsableChannelError` makes Zugaro's entry an error and the status `partial`;
    - any other exception is `{"status": "failed", "error": repr(e)}`.
  - `main(argv)` takes `--animal`, `--date`, `--epoch`, `--overwrite`, `--n-workers`
    (default 8), `--run-id` (default `git rev-parse HEAD`[:12]) and `--output-root`
    (default `PROCESSED_DATA_DIR / "replay_detection_comparison" / "swr"`).
    - It runs a dask
      `LocalCluster(n_workers=args.n_workers, processes=True, threads_per_worker=1)`.
    - Only the main thread writes `manifest.json`, atomically, after each completion.
    - Each epoch logs to `logs/{epoch_id}_swr.log`.
    - A rerun without `--overwrite` skips epochs already `ok`.

- [ ] **Step 1: write the failing tests.** All inject fakes; none touches the database.
  - `test_outputs_and_manifest_fields`: synthetic inputs from `simulate_session` at 1000 Hz
    with `ripple_frequency=(150, 175)`, and a fake position at 2 cm/s throughout. Expected:
    - every file exists;
    - the manifest has every field above;
    - `zugaro_channel.electrode_id` is in `electrode_ids`;
    - the band check passes with the fake filter.
  - `test_known_no_input_epochs_are_skipped_without_fetching`: a fetch that fails if called
    is never called for `("senor20201105_.nwb", "09_r4")`.
  - `test_a_failed_epoch_is_recorded_and_the_rest_run`: run with
    `LocalCluster(processes=False)` on two epochs, one raising. Expect one `ok`, one
    `failed`, and a non-zero exit.
  - `test_one_detector_failing_makes_the_epoch_partial`.
  - `test_rerun_without_overwrite_skips_done_epochs`.
  - `test_lfp_outside_the_position_span_is_dropped`: a fake position spanning 2–8 s of a
    10 s LFP leaves no event or observed interval outside [2, 8].
- [ ] **Step 2: run them and confirm they fail.**
- [ ] **Step 3: implement, and update the docs.**
- [ ] **Step 4: run the tests and confirm they pass.**
- [ ] **Step 5: lint, format and commit** ("Add the runner for the four SWR inventories").
- [ ] **Step 6: run the smoke, in the main tree after the merge.**
  `uv run python scripts/decoding/run_swr_detection.py --animal chimi --date 20200225 --epoch 02_r1 --n-workers 1`.
  Expected:
  - status `ok`;
  - the band check is within tolerance, with zero overflow;
  - a chosen Zugaro channel;
  - event counts for all four detectors;
  - both Kay comparisons written.

  Then run the five pilot epochs, and record wall clock, peak RSS, event counts and the Kay
  check totals in the status line.

**Acceptance:**
- the tests pass;
- the pilot smoke passes the band check on all five epochs;
- the docs list seven inventories.

## Task 10: the same-data audit and the tables manifest

Work in the worktree.

**Files:**
- Create: `scripts/qa/audit_refresh_consistency.py`
- Test: `tests/unit/test_audit_refresh_consistency.py`
- Docs: an entry for it in `scripts/README.md`

**Interfaces:** every check is a pure function over paths and frames, and returns
`{"ok": bool, "failures": list[str], "details": ...}`.
```python
def decode_provenance(log_dir: Path, root: Path, epoch_ids: list[str]) -> pd.DataFrame
    # epoch_id, decoder ("nonlocal" | "contfrag"), and the provenance fields: from
    # results.nc attrs["provenance"] (written at the fresh save, Task 2) when present, else
    # from the JSON after the first "PROVENANCE " substring in logs/{id}.log or
    # logs/{id}_contfrag.log; a column says which source was used
def check_one_commit(provenance: pd.DataFrame, allowed: Mapping[str, str]) -> dict
    # one commit, dirty False, and one uv_lock_sha256 across all rows, or listed exceptions
    # with reasons
def check_grids_match(nonlocal_dot: pd.DataFrame, contfrag_dot: pd.DataFrame) -> dict
    # identical time index, linear_position and head_speed
def check_minimum_sample_counts(times: Mapping[str, NDArray]) -> dict
    # minimum_sample_count(time, 0.015) is the same for every epoch (R9)
def check_widloski_hashes(manifest: dict, root: Path) -> dict
    # the recorded input sha256s equal the current files'
def check_swr_spans(swr_manifest: dict, decode_spans: Mapping[str, tuple[float, float]],
                    tolerance_s: float = 0.002) -> dict
def record_tables(table_dir: Path, commit: str, inputs: Mapping[str, str]) -> Path
    # writes table_dir/tables_manifest.json: each *.parquet / *.json file's sha256, the
    # commit, the inputs (decode commit, qvals source, cache dir)
def check_tables_recorded(table_dirs: list[Path], decode_commit: str) -> dict
    # each promoted directory has a tables_manifest.json whose sha256s match the files
    # present, and whose commit descends from decode_commit (git merge-base --is-ancestor)
def check_pfc_sidecars(sidecars: list[dict], event_table_sha256: str) -> dict
def check_epoch_sets(sets: Mapping[str, set[str]],
                     reasons: Mapping[str, Mapping[str, str]]) -> dict
    # an epoch missing from an artifact passes only with a recorded reason for that artifact
def check_fit_provenance(repo: Path, roots: list[Path]) -> dict
    # runs scripts/qa/check_fit_provenance.py on each root; ok when every exit code is 0
def check_encoding_marks(epoch_id: str, root: Path) -> dict
    # needs the database; runs only with --check-marks. Reloads marks through get_hpc_marks
    # plus filter_spike_times, and applies check_marks_match_encoding to both decodes'
    # encoding models
```
`main` has two subcommands:
- `record-tables <dir> --commit <sha> --inputs <json>`, used in R5;
- `audit --output <path> [--check-marks]`, which writes `audit.json` and exits non-zero if
  any check fails.

- [ ] **Step 1: write the failing tests.** Each check passes a case and fails a case:
  - two commits, or one dirty row, with no exception fail;
  - a one-sample-shifted ContFrag grid fails;
  - two epochs with minimum counts 8 and 7 fail;
  - an edited decode file fails the Widloski hash check;
  - a span off by 0.01 s fails;
  - a file changed after `record_tables` fails `check_tables_recorded`;
  - one mPFC sidecar naming an older event-table hash fails;
  - `test_an_unexplained_absence_fails_the_audit` (Review Focus 5): an epoch absent from the
    SWR set with no reason fails, and the same epoch with the manifest's `skipped` reason
    passes.
- [ ] **Step 2: run them and confirm they fail.**
- [ ] **Step 3: implement.**
- [ ] **Step 4: run the tests and confirm they pass.**
- [ ] **Step 5: lint, format and commit** ("Audit that the refreshed pipeline shares its
  inputs").

**Acceptance:** the tests pass, and `audit` runs read-only (`record-tables` writes only its
manifest).

## Task 11: pin the decoders' transition settings (added 2026-10-04, decision R10)

Work in the worktree `../continuum-swr-replay-pin` (branch `pin-transition-settings`, from
`C0`). After review, the main branch fast-forwards to it. That commit is `C0'`, and R1
restarts from it.

**Why:** neither decoder set its discrete-transition values, so both took
non_local_detector's defaults, which changed after April: +32% non-local events on
identical inputs. After validation on the pilot epochs (held-out spike evidence, simulations,
takeover checks, prior-strength sensitivity), the project lead chose setting P30-B
([docs/design/2026-10-04-decoder-transition-prior.md](../../../../docs/design/2026-10-04-decoder-transition-prior.md)).

**Files:**
- Create: `src/continuum_swr_replay/decoding_transitions.py`, the only place the values live
- Modify: `scripts/decoding/run_decoding_bandit.py`, `scripts/decoding/run_decoding_contfrag.py`
- Test: `tests/unit/test_decoding_transitions.py`
- Docs: the design doc above; R10 in the spec's decisions table

**Interfaces:**
```python
LOCAL_DWELL_S = 3.0; NONLOCAL_DWELL_S = 0.1; NO_SPIKE_DWELL_S = 0.1   # each with its source
PRIOR_VISITS = 30; NO_SPIKE_EXIT_TO_LOCAL = 0.96
def nonlocal_transition_settings(sampling_frequency: float = 500.0) -> dict
    # DiscreteStationaryCustom start matrix from the dwell times; concentration
    # 1 + PRIOR_VISITS / 3; per-state stickiness, so each learned row's prior mean is its start
    # row, worth PRIOR_VISITS x dwell / bin pseudo-counts; prior_weight 0.0; the No-Spike row
    # frozen at [0.0192, 0.98, 0.0004, 0.0004]
def contfrag_transition_settings() -> dict
    # ContFrag's values at C0, made explicit: DiscreteStationaryDiagonal([0.98, 0.98]),
    # concentration 1.1, stickiness [0, 0], prior_weight 0.0, no frozen rows
```
Both runners refuse `--non_stationary`: the library's non-stationary detector cannot be saved
(non_local_detector #50) and applies the prior per bin (#51).

**Acceptance:**
- The tests pass:
  - row sums;
  - expected stays of 3 s and 0.1 s;
  - the frozen row;
  - prior means, and pseudo-count totals of 45,000 and 1,500;
  - literal pins: concentration 11, stickiness [44960, 1460, 1460, 1460];
  - the detectors both runners build carry these values;
  - ContFrag equals C0's construction;
  - both refusals.
- The pinned NonLocal values equal the P30-B ablation detector's, bit for bit in float32.
  ContFrag's equal the C0 staging detector's, bit for bit.
- Nothing else in the decode path changes.

## Task 12: held-out spike evidence per decoder state (added 2026-10-04)

Work in the worktree. The run happens after R2, on the swapped-in canonical decodes. Its
figure and table go into the R8 report and the paper.

**Why:** the paper must show that each decoder state captures what it claims, judged by
spikes that did not choose the location:
- Local: the animal's position;
- Non-Local: a remote location;
- No-Spike: silence.

A prototype on the five pilot epochs found:
- Local and immobile Non-Local supported;
- running Non-Local Continuous unsupported;
- No-Spike untestable by a tetrode split, but supported by population-wide silence.

The prototype's scripts and method are in `.claude/research/refresh_validation/state_evidence/`.
This task turns them into tested code.

**Files:**
- Create: `src/continuum_swr_replay/analysis/hippocampus/state_evidence.py` (pure functions)
- Create: `scripts/qa/run_state_evidence.py` (runner)
- Test: `tests/unit/test_state_evidence.py`
- Docs: a `scripts/README.md` entry

**Reuse:**
- The encoding, mark loading and per-spike likelihood code in
  `scripts/qa/analyze_event_spike_evidence.py`. Move what both need into the new module rather
  than copying it.
- The junction-bin rule: `edge_id == -1` is excluded, as in `map_estimate_to_arm`.

**Interfaces:**
```python
def per_tetrode_terms(encoding, spike_times, marks, time, interior_bins) -> TetrodeTerms
    # per spike: log joint mark intensity over the interior bins; per tetrode: ground
    # intensity; spikes binned with the decoder's np.digitize rule
def regimes(head_speed: NDArray) -> NDArray      # "immobile" < 4 cm/s, "running" > 10, else ""
def state_bouts(probability, regime, *, threshold=0.8, min_bins=5, max_scored_bins=50,
                cap=1500, seed) -> pd.DataFrame
def matched_windows(bouts, regime, *, seed) -> pd.DataFrame   # same regime, same length
def calibration_windows(regime, *, n_per_regime=2000, n_bins=10, seed) -> pd.DataFrame
def tetrode_halves(n_tetrodes, *, n_partitions=10, seed=0) -> list[tuple[NDArray, NDArray]]
def window_evidence(terms, windows, animal_bins, track_distance, halves, *,
                    partner_min_cm=20.0, n_partners=20, no_spike_rate) -> pd.DataFrame
    # local: logL_C(animal) - mean logL_C(partners' animal bins, >= 20 cm away)
    # remote: logL_C(MAP_S) - logL_C(animal), and - mean logL_C(partners' MAP_S)
    # local vs S's best bin >= 20 cm from the animal
    # No-Spike: log P(C counts | no_spike_rate) - log P(C counts | g(animal)), C's count,
    #   E_C, silent share, P(C silent | S silent) vs exp(-E_C)
    # spike-mark part primary, full log-likelihood secondary
def summarize_state_evidence(windows: pd.DataFrame) -> dict[str, pd.DataFrame]
    # bouts: median, IQR, share > 0, mean with a sign-flip p (10,000 seeded flips) and an
    #   epoch-stratified bootstrap CI, per-spike evidence, paired bout - baseline, per-epoch
    #   and per-animal consistency
    # calibration: per-bin means with CIs, window-level OLS slope with CI, top bin > 0
    # rate calibration (observed / expected held-out spikes); coordinated silence
```

**Encodings:**
- Primary: a running-only refit (speed >= 4 cm/s), asserting that no event bin is in it.
- Secondary: the decode's own encoding.

**Runner:**
- Arguments: `--epochs` (default: every canonical HPC decode), `--output-dir` (default
  `Processed-Data/_refresh_<date>/state_evidence/`) and `--n-workers`.
- Dask CPU workers. Per epoch, it writes one per-window parquet and a validation JSON.
- It then writes the summary CSVs and three figures: bouts vs matched windows, calibration
  curves and coordinated silence.
- It is read-only except for the output directory.

- [ ] **Step 1: write the failing tests**, on synthetic inputs:
  - summed per-tetrode terms reproduce the decoder's log-likelihood;
  - a junction bin is never chosen;
  - a partner is never the window itself and is at least 20 cm away;
  - run-averaged Local and No-Spike values equal half the all-tetrode values (documenting that
    they are not held out);
  - an independent per-window loop matches the vectorized evidence;
  - planted remote content gives remote evidence > 0; a planted local window gives local
    evidence > 0;
  - bouts respect the threshold, the regime, the minimum length, the cap and the seed;
  - fixed seeds give identical reruns.
- [ ] **Step 2: run them and confirm they fail.**
- [ ] **Step 3: implement.**
- [ ] **Step 4: run the tests and confirm they pass.** On the five pilot epochs (the
  swapped-in decodes, identical to the prototype's P30-B inputs), compare the P30-B bout and
  calibration statistics with the prototype's, and explain every difference.
- [ ] **Step 5: lint, format and commit** ("Add held-out spike evidence per decoder state").

**Run (after R2):**
`tmux new-session -d -s state_evidence "cd $PWD && uv run python scripts/qa/run_state_evidence.py --n-workers 16 > logs/state_evidence_<date>.out 2>&1"`.

**Acceptance:**
- the tests pass;
- the pilot statistics match the prototype's;
- the full run covers every canonical decode, with a recorded reason for any epoch it skips.

## Task 13: held-out state labels on the pilot epochs (added 2026-10-04)

Work in the worktree. The decodes need a GPU. Run them after R1's full batches, on idle cards.

**Why:** in Task 12 the bouts and probabilities come from the all-tetrode decode. The state
labels are therefore not held out; only the remote location is. Here, the labels come from a
decode that sees one tetrode half, and the other half scores them.

**Files:**
- Create: `scripts/qa/run_half_tetrode_decodes.py`
- Modify: `state_evidence.py`, so it accepts labels from a half decode
- Test: `tests/unit/test_half_tetrode_decodes.py`

**Method:**
- Use each pilot epoch and the first two partitions of `tetrode_halves`, both directions: four
  decodes per epoch.
- Fit the encoding model on half S's tetrodes only, at the canonical decode's training times,
  with no weights from any all-tetrode posterior.
- Keep the canonical fit's discrete transitions and initial conditions fixed, with no EM. That
  is 12 numbers, an aggregate leak the report states.
- Decode with S's spikes, then score with half C as in Task 12.

- [ ] **Step 1: write the failing tests.**
  - On a small simulated session, the fixed-transition decode equals the library's standard
    decode with the same transitions.
  - The half encoding model holds only S's electrodes.
  - No EM iteration runs.
- [ ] **Step 2: run them and confirm they fail.**
- [ ] **Step 3: implement.**
- [ ] **Step 4: run the tests and confirm they pass.** Then decode one pilot epoch with
  S = all tetrodes and report its agreement with the canonical decode: state-probability
  differences and shared events.
- [ ] **Step 5: lint, format and commit** ("Decode with half the tetrodes to hold out the state
  labels").

**Acceptance:**
- the tests pass;
- 20 decodes;
- the half-label summary sits beside the all-tetrode one in the R8 report.

## Task 14: one epoch register with reason codes, read by every stage (added 2026-10-05)

Work in the worktree `/cumulus/edeno/continuum-swr-replay-refresh-dev`, branch
`full-pipeline-refresh-code`, after the R2 merge. Leave the main tree untouched: long
runs (R3 and R4) import from it.

**Why.** The project lead requires a consistent exclusion rule with a good reason for
every exclusion. An audit found:
- the cohort is 218 epochs at every stage, but 36 regular epochs, the depletion filter,
  and several documented reasons were wrong or undocumented;
- each stage keeps its own skip list (for example `KNOWN_NO_INPUT_EPOCHS` in the SWR
  runner).

The audit is in `.claude/research/refresh_exclusions/`: read `exclusions_report.md` first,
then `exclusions_table.csv` (557 rows, one per run epoch, with evidence). The populate plan
for 40 restorable epochs is in `.claude/research/refresh_populate/`
(`populate_report.md`, `populate_matrix.csv`).

**The lead's decisions this task implements:**
- 40 epochs become analysable after Spyglass populates, which the controller runs after this
  task lands:
  - 36 with no `PositionIntervalMap` and no CA1 ripple times;
  - senor20201104 06_r3 and 08_r4;
  - senor20201105_09_r4;
  - wilbur20210326_03_r1.

  Until their decodes exist and the update is approved, they are PENDING. The analysis
  proceeds now on the 218 already decoded.
- All four first-exposure epochs are included: j1620210706_02_r1, senor20201027_02_r1,
  chimi20200212_02_r1 and wilbur20210326_03_r1. wilbur20210326's session description is
  "exploration session", not "Spatial bandit task (regular)", so admit it through an explicit,
  documented allowlist of first-exposure sessions.
- wilbur20210401 (8 run epochs) is excluded as BLOCKED: half ingested; re-ingestion needs a
  destructive reinsert, Comrie's trial parsing and an RL refit. Record that evidence.
- Depletion sessions (46 files, a decaying reward schedule) stay excluded as a different task
  variant. Document this; today it is undocumented.

**The rule** (state it verbatim in the module docstring and in the doc):
> An epoch is analysed if and only if it is a run epoch (`^[0-9]+_r[0-9]+$`) of a regular
> spatial-bandit session in Spyglass (or of a listed first-exposure session), with parsed
> trials (`noPrePostTrialTimes`), and every decode input resolves to exactly one row:
> position map, position info and linearization, ripple band, CA1 ripple times, HPC marks.
> Every other run epoch of a project session is excluded with exactly one enumerated
> reason code and its evidence, in one register that every stage reads and the audit
> checks.

**Files:**
- Create `src/continuum_swr_replay/data_loaders/epoch_register.py`:
  - a `Reason` enum. Suggested codes, refined from the audit's classes:
    - `NO_RUN_EPOCH`;
    - `NO_TRIAL_INTERVALS` (trial parsing failed; holds for 12 sessions);
    - `NO_NWB` (files absent);
    - `DEPLETION_TASK`;
    - `NOT_INGESTED`;
    - `PENDING_POPULATE` (with the planned fix);
    - `MISSING_INPUT` (a decode input absent with no fix planned).

    Keep the set small. Every code needs a one-line definition.
  - the register itself: a version-controlled table (CSV beside the module, or a Python
    mapping) of `(nwb_file_name, epoch or "*", status, reason, evidence)`. Status is
    `analysed`, `pending` or `excluded`. It is generated once from `exclusions_table.csv`
    and the populate matrix, then hand-maintained.
  - `analysed_epochs(include_pending: bool = False) -> pd.DataFrame`: `get_epoch_info()` rows
    whose register status is `analysed` (plus `pending` when asked). It must raise if
    `get_epoch_info()` returns an epoch with no register entry. An unregistered epoch is an
    error, never silently included.
  - `register_reason(nwb_file_name, epoch) -> Reason | None`.
- Modify `get_epoch_info()` (`data_loaders/utils.py`): admit the listed first-exposure
  sessions' descriptions. Keep its other behaviour, because other callers use it. Document
  the change.
- Every stage that enumerates epochs uses `analysed_epochs()`:
  - both decoders' `run_batch` epoch list (add an `--pending-only` flag for the
    incremental update);
  - the SWR runner, which replaces `KNOWN_NO_INPUT_EPOCHS` with register lookups;
  - the Widloski runner;
  - the table builders (`prepare_replay_content_data.py`, `build_timebin_tables.py`,
    `regenerate_stop_table.py`, and any other `get_epoch_info()` caller that builds
    analysis tables);
  - the PFC runner (mPFC adds its own unit-availability condition; record that as a
    PFC-level reason, not an HPC exclusion).

  Grep for every `get_epoch_info(` caller and list each with your decision in the report.
- The audit (`scripts/qa/audit_refresh_consistency.py`):
  - `check_epoch_sets` takes its reasons from the register;
  - add a check that every `get_epoch_info()` epoch is registered;
  - add a check that every `analysed` epoch has every artifact;
  - add a check that the PENDING set is reported.
- Create `docs/epoch_register.md`: the rule, the reason codes, counts per status and reason,
  and the full table of excluded and pending epochs with evidence. Add a test that the doc
  mirrors the module, following `tests/unit/test_models_table_mirrors_specs.py`.
- Doc fixes:
  - `constants.py` docstring: point it to the register, and fix the 18-session text, which
    has 6 entries the audit found mis-described;
  - `docs/known_issues/wilbur20210404_missing_ca1_ripples.md`: a depletion session, never in
    the cohort;
  - `docs/known_issues/three_sessions_blocked_by_nwb_checksum.md`: resolved 2026-08-19; all
    218 are in every table;
  - `docs/README.md:103-104`.

  Do NOT edit CLAUDE.md. Instead write the proposed replacement text for its "91 NWB files
  and 221 total epochs" claim into the report; the lead approves the wording first.

**Tests (write them first):**
- the register covers every row of `exclusions_table.csv`, and no epoch has two statuses;
- every non-analysed entry has exactly one reason and non-empty evidence;
- `analysed_epochs()` raises on an unregistered epoch;
- `analysed_epochs()` gives exactly the 218 decoded epoch ids today (pinned list), and 258
  with `include_pending=True`. Verify the 40 count against the populate matrix;
- the first-exposure allowlist admits wilbur20210326 and nothing else new;
- each runner selects epochs through the register (stub the register in the runners' unit
  tests);
- the doc mirrors the module.

**Acceptance:**
- tests pass;
- no stage keeps its own epoch skip list;
- the audit fails on an unregistered or unexplained epoch;
- with the current Spyglass state, `analysed_epochs()` equals the 218 staged and swapped-in
  decodes.

## Task 15: one SWR source for every analysis — the direct ripple_detection 2.0 Kay inventory (added 2026-10-09)

Work in the worktree `/cumulus/edeno/continuum-swr-replay-refresh-dev`, branch `full-pipeline-refresh-code`.
Leave the main tree untouched until the controller merges.

**Why:** the project lead decided (2026-10-05) that the paper's SWR labels (`is_swr`, `SWR_event`,
and every SWR-aligned analysis) come from the direct ripple_detection 2.0 Kay inventory produced by
the SWR runner (R3), not from Spyglass's stored `RippleTimesV1`. The 2.0 rule is at least 15 ms
continuously above z = 2, the published Kay et al. 2016 definition. Across 216 epochs the two agree
closely: ratio 1.015, with 92.8% of stored events recovered. But the stored events came from
unknown older inputs, and 1.7% of them fall inside artifacts. Artifact masking (Task 16) will be
added later through the interface below.

**Inputs:**
- R3 inventory: `Processed-Data/replay_detection_comparison/swr/<run_id>/`.
  - Files per epoch: `{epoch_id}_kay_events.parquet` (start_time, end_time, duration, …),
    `{epoch_id}_swr_observed.parquet` (where detection was possible) and `manifest.json` (status,
    provenance).
  - The run id is today `50a98592acee`. It will change when R3 reruns with the artifact mask, so it
    must be one configured constant, not a literal at each call site.
- The project's whole-event immobility rule (`analysis/hippocampus/event_immobility.py`).
- The epoch register (Task 14): `analysed_epochs()` and `analysed_decode_files()`.

**Files:**
- Create `src/continuum_swr_replay/analysis/hippocampus/swr_source.py`:
  - `SWR_INVENTORY_RUN_ID` and the detector name (`"kay"`), each defined once and documented.
  - `swr_events(epoch_id, *, artifact_intervals=None) -> pd.DataFrame`. It reads the inventory, keeps
    the whole-event-immobile events (the same canonical rule as the non-local and MUA cohorts), and
    drops events that overlap `artifact_intervals` (an interval table, a hook for Task 16; None
    today). It records a reason for every dropped event.
  - It raises if the epoch's manifest status is not `ok` or `partial` with the Kay inventory present.
    That includes the 2 j16 epochs until R3 reruns them.
  - `swr_event_labels(time, epoch_id, *, artifact_intervals=None) -> np.ndarray`: the per-bin
    `SWR_event` labels on a decode time grid. It must use exactly the convention the table builders
    use now for `SWR_event` (find it and keep it). Also give the coverage mask: bins where SWR
    detection was observed, from `swr_observed`.
- Replace the stored-ripple SWR source everywhere analyses read SWR labels or times. Grep `SWR_event`,
  `ripple_times`, `get_ripple(`, `is_swr`, `RippleTimesV1` and `swr_` across `src/` and `scripts/`, and
  list every consumer with your decision in the report. At least:
  - the event-table builder `build_classified_nonlocal_event_table` (`analysis/hippocampus/event_table.py`),
    which takes `SWR_event` from data_over_time today;
  - the stop table's `is_swr_stop`;
  - the timebin tables;
  - the ContFrag tables;
  - the PFC SWR-aligned analyses;
  - the replay-content tables;
  - the MUA/SWR count conventions doc;
  - figure helpers in `src/` (do not edit figure scripts; that is a plan constraint).
- Leave the decoders' `no_ripple` encoding-training mask unchanged in this task. It is a separate
  decision, noted in the report.
- Docs:
  - a design note `docs/design/2026-10-09-swr-source.md`: the decision, the 2.0 vs stored comparison
    numbers (from `.claude/research/refresh_r1/kay_version_ratio.csv`), and the interface;
  - update `docs/known_issues/swr_mua_count_conventions.md` if the conventions change;
  - update CLAUDE.md's description of `SWR_event` provenance only as a PROPOSAL in the report (do not
    edit CLAUDE.md).

**Tests (first):**
- `swr_events` reads a synthetic inventory, applies the immobility rule, and drops overlapping events
  with reasons.
- It raises on a failed epoch.
- `swr_event_labels` reproduces the current builder's labels exactly when fed the same intervals.
- Every listed consumer now calls the source. Use stubs; no consumer reads `RippleTimesV1` or
  data_over_time's `SWR_event` for analysis anymore. Keep `SWR_event` in data_over_time as the decode's
  own provenance, unused by analyses.

**Acceptance:**
- the tests pass;
- one source, used by every SWR consumer;
- the report lists every consumer and the decision for each;
- on 5 pilot epochs (read-only), the new `is_swr` counts per epoch are reported next to the old ones.

---

# Phase B: runbooks

Each runbook's results go in this plan's status line and in a ledger:
`.superpowers/sdd/full-pipeline-refresh.md`, archived to
`.claude/docs/plans/full-pipeline-refresh/phase-N-ledger.md`. A **STOP** line means: do not
continue without the project lead's explicit go-ahead in the conversation.

Pilot epochs:
- `chimi20200225_02_r1`
- `wilbur20210329_12_r6`
- `j1620210707_06_r3`
- `peanut20201202_10_r5`
- `senor20201101_06_r3`

## R0: environment and backups

Before Task 1's pin, run Task 1 Step 1. After Task 1:

- [ ] **Set the backup root once.** `B=Processed-Data/_pre_redecode_backup_<today>`, written
  into the status line. Every later backup goes under `$B`.
- [ ] **Disk.** Run `du -sh` on `Processed-Data/contfrag_replay_content`,
  `Processed-Data/replay_content/models_dm`,
  `Processed-Data/replay_content/content_estimator_comparison` and `paper/figures`, then
  `df -h Processed-Data`. Need room for the staged decodes of both kinds (about 1.5 TB
  while the April ones still exist) plus these copies.
- [ ] **Golden baseline.**
  `uv run python scripts/qa/freeze_regression_baseline.py --label pre_redecode_<today>`.
  Expected: a read-only `replay_content/_golden_baseline_pre_redecode_<today>/` with a
  sha256 manifest.
- [ ] **Everything else the refresh overwrites.**
  ```bash
  mkdir -p $B/logs $B/replay_content $B/figures
  cp -p logs/*_HPC.log logs/*_HPC_contfrag.log $B/logs/
  cp -rp Processed-Data/replay_content/models_dm Processed-Data/replay_content/content_estimator_comparison $B/replay_content/
  cp -rp Processed-Data/contfrag_replay_content $B/
  cp -p Processed-Data/replay_content/stop_replay_summary.pkl Processed-Data/replay_content/stage5b_*.pkl Processed-Data/replay_content/stage6_*.pkl $B/replay_content/
  rsync -a --include='*/' --include='*.png' --include='*.pdf' --exclude='*' paper/figures/ $B/figures/
  ```
  Expected: each copy's file count matches its source. Record the counts.

## R1: decodes (main tree at `C0'`)

R1 first ran from `C0` into `_staging_redecode_20261003` (smoke and pilot only). After R10 it restarts from `C0'` into `_staging_redecode_20261004`. The old staging folder is left as it is.

- [ ] **Smoke.** One epoch of each decoder, on two idle cards. Check with
  `nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv` first.
  ```bash
  S=Processed-Data/_staging_redecode_<date>
  /usr/bin/time -v uv run python scripts/decoding/run_decoding_bandit.py --brain_area HPC \
      --animal chimi --date 20200225 --epoch 02_r1 --save_dir $S --cuda_visible_devices <A>
  /usr/bin/time -v uv run python scripts/decoding/run_decoding_contfrag.py --brain_area HPC \
      --animal chimi --date 20200225 --epoch 02_r1 --save_dir $S --cuda_visible_devices <B>
  ```
  While each runs, `nvidia-smi` must show its process on card A (or B) only. Record wall
  clock, peak RSS, GPU memory and `results.nc` size. Check that both logs carry a
  `PROVENANCE` line with `commit == C0'` and `dirty == false`, and that `$S` holds four files
  per decoder.
- [ ] **Pilot.** The five pilot epochs, the same way. Record:
  - non-local events per epoch, old (canonical) against new (staged);
  - each epoch's `minimum_sample_count(time, 0.015)`, which must be 7 for all (R9);
  - NonLocal events per epoch must equal the P30-B ablation's: peanut 245, j16 236, chimi 264,
    wilbur 195, senor 223. ContFrag's results must equal the `_staging_redecode_20261003` pilot's;
  - Task 10's `check_decoder_settings` on the staged detectors, run from the worktree,
    against `expected_decoder_settings()`, which the audit builds from the code
    (`nonlocal_transition_settings()` and `contfrag_transition_settings()`). An
    `--expected-settings` file is optional, an explicit override.
- [ ] **STOP:** report the smoke and pilot numbers and a full-run estimate. Ask for the
  go-ahead to launch both full batches.
- [ ] **Full.** Two tmux sessions, on disjoint idle cards:
  ```bash
  tmux new-session -d -s redecode "cd $PWD && uv run python scripts/decoding/run_decoding_bandit.py --brain_area HPC --save_dir $S --cuda_visible_devices <cards> > logs/redecode_<date>.out 2>&1"
  tmux new-session -d -s redecode_contfrag "cd $PWD && uv run python scripts/decoding/run_decoding_contfrag.py --brain_area HPC --save_dir $S --cuda_visible_devices <cards> > logs/redecode_contfrag_<date>.out 2>&1"
  ```
- [ ] **Capture the `C0` reference for Task 6.** While the batches run, from the main tree
  (read-only), pickle these to `.claude/research/refresh_c0_reference/`:
  - `get_ripple("chimi20200225_.nwb", "02_r1", "Kay")`;
  - `load_data`'s `position_info` for that epoch.
- [ ] **Gates.** All must hold before the swap:
  1. Each decoder has its four files for every decodable epoch; expect 218. Every other
     epoch has a logged reason. The known no-input epochs are senor20201104 06_r3 and 08_r4
     and senor20201105 09_r4.
  2. Every staged `results.nc` carries `attrs["provenance"]` (and every staged log a `PROVENANCE` line) with
     `commit == C0'`, `dirty == false` and a non-null `cuda_plugin_version`. A log marked `RESUMED` is checked
     through its `results.nc` attribute, since the resume rewrote the log.
  3. Every `data_over_time` grid passes `validate_time_grid`.
  4. Every epoch's minimum sample count is 7. If not, STOP and report (R9).
  5. NonLocal and ContFrag share time grid and position per epoch: Task 10's
     `check_grids_match`, or the same comparison inline before the merge.
  6. The non-local events-per-epoch table is written, with outliers listed.

## R2: swap and merge

- [ ] **Check readers.** Confirm no other session is reading `Processed-Data/*_HPC_*`
  decode files: check `tmux ls`, and `lsof +D Processed-Data`, which can be slow.
- [ ] **STOP:** ask for the go-ahead to swap.
- [ ] **Move April out.** Write `sha256sum` of the April NonLocal and ContFrag decode files
  (8 per epoch) into `$B/SHA256SUMS`, then move them into `$B/`.
- [ ] **Move the staged files in.** Move exactly the four named files per epoch per decoder into
  `Processed-Data/` under the canonical names, never a glob or `rsync` of `$S`: a hard kill can
  leave hidden `.<name>.<pid>.tmp` files from the atomic writes, which are deleted, not moved.
  Verify counts, and hashes against their staging hashes.
- [ ] **Merge.**
  1. Merge `C0'` into the worktree branch (a merge commit, not a rebase: the ledger names the
     worktree's commit hashes), resolve any conflict, and run the unit suite there. Then
     fast-forward the main branch to the worktree branch (`C1`).
  2. Run `uv sync --inexact`.
  3. Check that `dask_cuda` imports and that `jax.devices()` lists the GPUs.
  4. Run the Task 6 and Task 9 database smokes.
  5. Remove the worktree.

## R3: SWR inventories (main tree at `C1` or later)

- [ ] **Smoke and pilot.** Task 9, Step 6. Then report the filter checks for the pilot
  epochs, computed by a gitignored scratch script,
  `.claude/research/refresh_zugaro_checks/check_filter.py`, from Welch spectra of the raw
  and filtered chosen channel:
  - the measured gain at 120, 150 and 180 Hz;
  - the share of filtered power within 120–180 Hz;
  - 60 Hz harmonic power at 120 and 180 Hz;
  - how often the same Zugaro channel wins across a day's run epochs.
- [ ] **Full.**
  `tmux new-session -d -s swr_inventories "cd $PWD && uv run python scripts/decoding/run_swr_detection.py --n-workers 16 > logs/swr_inventories_<date>.out 2>&1"`.
- [ ] **Gate.**
  - Every epoch is `ok`, or `skipped` with a reason.
  - A `partial` or `failed` epoch is diagnosed before R8.
  - Every `ok` epoch passes the band check with zero overflow.

## R4: Widloski full run

- [ ] **Smoke.** One epoch on one idle card:
  ```bash
  uv run python scripts/decoding/run_memoryless_detection.py --run-seed 20260922 \
      --epoch-id chimi20200225_02_r1_HPC --cuda_visible_devices <A>
  ```
  `nvidia-smi` must show it on card A. Compare its candidate count, replay calls and shuffle
  passes with the April pilot row. Record time and memory.
- [ ] **Full.** On cards disjoint from R6's PFC run:
  `tmux new-session -d -s memoryless_full "cd $PWD && uv run python scripts/decoding/run_memoryless_detection.py --run-seed 20260922 --all --cuda_visible_devices <cards> > logs/memoryless_full_<date>.out 2>&1"`.
- [ ] **Gate.** Every canonical decode is `ok` or has a reason, and the manifest's decode
  hashes equal the canonical files'.

## R5: tables

```bash
T=Processed-Data/replay_content/_staging_redecode_tables_<date>
TC=Processed-Data/contfrag_replay_content/_staging_redecode_<date>
CACHE=Processed-Data/replay_content/_session_cache
```

Run `--help` on each script before its first use. Where a flag differs from below, stop and
report. Every prepare call that passes `--behavior-cache-dir $CACHE` must also pass `--from-cache`:
without it the builder refetches from Spyglass and rewrites the canonical cache entries
(`prepare_data.py:219, 312`). Every gate checks the written outputs and their row counts, never the exit code
alone: the cross-table check and the no-session path return 0 and write nothing
(`prepare_data.py:971, 997-1004`; ContFrag script :463).

- [ ] **Stop table.**
  `uv run python scripts/decoding/prepare_replay_content_data.py --stop-only --from-cache --output-dir $T --behavior-cache-dir $CACHE`
  (about 6 min). Check that `$T/stop_table.parquet` exists and holds every session.
- [ ] **`qvals_table`.**
  `uv run python scripts/behavior/build_qvals_table.py --replay-content-dir $T --write`
  (about 12 min). It must come after the stop table. Check that the qvals table and its
  standardization reference exist in `$T`.
- [ ] **Replay tables.**
  `uv run python scripts/decoding/prepare_replay_content_data.py --from-cache --output-dir $T --behavior-cache-dir $CACHE --contingency-dir $T`
  (about 1.5 h). Alternatively, five `--animal` shards into `$T/shard_<animal>`, then
  `uv run python scripts/decoding/assemble_replay_content_shards.py $T/shard_* --output-dir $T --contingency-dir $T --expected-sessions <n>`.
  Check that the event, event_arm, event_patch, event_sequence and stop tables plus
  `standardization_reference.json` exist, and that their session counts agree.
- [ ] **Timebin tables.**
  `uv run python scripts/decoding/build_timebin_tables.py --output-dir $T` (about 69 min).
- [ ] **ContFrag tables.** In parallel:
  `uv run python scripts/decoding/prepare_contfrag_replay_content_data.py --event-source SWR --from-cache --output-dir $TC --replay-content-dir $T --behavior-cache-dir $CACHE`,
  and the same with `--event-source MUA` (about 2.5 h each). Check their outputs in `$TC`.
- [ ] **Drift summary.** A scratch script, `.claude/research/refresh_drift/summary.py`,
  writes rows per table and session, old (canonical) against new (staged), with every
  session present in only one side listed. `compare_to_golden_baseline.py` is not used: it
  raises at the first drift.
- [ ] **Regeneration manifest.** Read `scripts/decoding/write_regeneration_manifest.py`. It
  expects `models_validation/` bundles and a clean tree. Decide whether it can run here
  without `--baseline-replay`, and write the decision down. Either way, record both staging
  folders with Task 10:
  ```bash
  uv run python scripts/qa/audit_refresh_consistency.py record-tables $T --commit $(git rev-parse HEAD) --inputs '{"decode_commit": "<C0>", "behavior_cache": "'$CACHE'"}'
  uv run python scripts/qa/audit_refresh_consistency.py record-tables $TC --commit $(git rev-parse HEAD) --inputs '{"decode_commit": "<C0>", "qvals_from": "'$T'", "behavior_cache": "'$CACHE'"}'
  ```
- [ ] **Never use:** `regenerate_stop_table.py`, `refresh_contfrag_behavioral_covariates.py`,
  `scripts/exploratory/_augment_data_over_time.py`, or `python -m …prepare_data`.
- [ ] **Check the cohort.** Does it now hold 218 sessions, with the three formerly blocked
  ones back? Record the answer.
- [ ] **STOP:** report table counts, the drift summary and the cohort. Ask for the go-ahead
  to promote.
- [ ] **Promote.**
  1. Write the exact file list for each target: `$T` → `Processed-Data/replay_content/`,
     including `standardization_reference.json` and the qvals standardization reference
     (`models_per_bin_rate/qvals_standardization_reference.json`); and `$TC` →
     `Processed-Data/contfrag_replay_content/`.
  2. Back up every target file that will be replaced into `$B/replay_content_tables/` and
     `$B/contfrag_replay_content_tables/`.
  3. `cp -p` the listed files into place.
  4. Re-run `record-tables` on both promoted roots, and check them with
     `check_tables_recorded`.

## R6: model refits and PFC

- [ ] **Inventory the refits from what is on disk.**
  - List every fit artifact under `Processed-Data/replay_content/models*/`,
    `Processed-Data/contfrag_replay_content/models_*`, and the Stage 5B, Stage 6 and
    `stop_replay_summary` artifacts at the `replay_content` root, each with its provenance
    sidecar's recorded `cli`.
  - Write the list into the ledger. It is the refit list.
  - Move `*_smoke` bundles into `$B/smoke_bundles/` rather than refitting them.
  - Run `--help` on each script before first use, and stop and report where a recorded
    `cli` and `--help` disagree.
- [ ] **CPU refits.** Run every listed `cli`, independent ones in parallel.
  - These are expected, with their September runtimes:
    - `fit_per_bin_rate_models.py --stop-cohort at_well --workers 16` (41 min), then
      `all_stops` (57 min) and `in_transit --workers 10` (5 min);
    - `fit_per_stop_rate_models.py`;
    - `fit_swr_contrast_rate.py` for `at_well` and `all_stops`, then `--aggregate-only`;
    - `fit_uncertainty_rate.py`, `fit_swr_power_model.py`, `fit_cross_history_rate.py --write`;
    - every `fit_replay_content_models.py` ladder, including the leaf chain, the
      Dirichlet-multinomial estimator, the patch and SWR-subset variants, and the ContFrag
      `--event-source SWR` and `MUA` fits;
    - `fit_swr_contrast_content.py`: `all` (3.3 h), `rewarded` (3.0 h), `omitted`, and the
      `_dm`, `_dm_rewarded`, `_dm_omitted` and `_dm_len` variants;
    - `analyze_uncertainty_content.py` (after the leaf chain),
      `compare_content_estimators.py --n-jobs 16` and `compare_event_source_models.py`;
    - three Stage 5B runs and eight Stage 6 runs, as recorded. The first Stage 6 run takes
      `--rebuild-summary`.
  - Anything listed but not expected is still run, and noted in the ledger.
- [ ] **STOP:** ask for the go-ahead for the GPU unit GLM.
- [ ] **PFC unit GLM.**
  1. Copy `logs/pfc_ridge_heldout_20260916/launch.sh` to `logs/pfc_refresh_<date>/launch.sh`,
     updating `EXPECTED_HEAD`, `GPUS`, the 71-session partition files and the backup check
     (which now points at `$B/mpfc`).
  2. Move `Processed-Data/*_mPFC_modulation*` into `$B/mpfc/`.
  3. Launch: about 14–17 h per card on 5 cards.
- [ ] **Gate.**
  `uv run python scripts/qa/check_fit_provenance.py` exits 0 on the `replay_content` root
  and, with its root argument, on `contfrag_replay_content`. The Stage 5B and Stage 6 root
  artifacts' sidecars name the promoted tables.

## R7: figures and generated docs

- [ ] **Once R2 is done.** Confirm in each script that it reads no table, then run:
  - `generate_figure02.py --force-recompute`;
  - `generate_figure03.py`;
  - `generate_supp_decoder_validation.py`;
  - `generate_event_gallery.py catalog`, then `run_event_gallery.sh` (about 4 h).
- [ ] **Once R5 promotes the tables:** `generate_supp_pipeline_data_flow.py`.
- [ ] **Once R6 is done.** Run:
  - `generate_figure04.py`;
  - `plot_per_bin_rate_models.py`, `plot_per_bin_rate_summary.py`, `plot_model_checking.py`
    and `plot_uncertainty_rate.py`;
  - `plot_swr_contrast_coefs.py`;
  - `generate_uncertainty_content_figure.py`;
  - `generate_replay_content_figures.py`;
  - `summarize_per_bin_rate_models.py`, which regenerates `docs/per_bin_rate_results.md`.
- [ ] **Not rerun:** Figure 1 and `generate_memoryless_comparison.py`, which use simulation
  only.

## R8: audit and report

- [ ] **Audit.**
  `uv run python scripts/qa/audit_refresh_consistency.py audit --output Processed-Data/_refresh_<date>/audit.json --check-marks --widloski-manifest <R4 manifest> --swr-manifest <R3 manifest>`.
  The expected decoder settings come from the code (`expected_decoder_settings()`);
  `--expected-settings <json>` is optional, an explicit override, and `audit.json` records
  under `expected_settings` which source it used.
  Expected: exit 0. Any failure is diagnosed, not waived.
- [ ] **Report.** Write `docs/audits/<date>-full-pipeline-refresh.md`, covering:
  - the audit summary;
  - the R5 drift summary, `compare_fits.py --old-* <$B copies> --new-* <current>`, and
    `compare_pfc_cohorts.py`;
  - a headline-number table, old against new: `swr_x_q_arm`, the reward-gating rate ratios,
    CS4/CS5/CS6, the Dirichlet-multinomial `log_phi_is_swr`, and PFC engagement;
  - from Task 1: the MUA version shift and the non-local events under both rules;
  - from R1: the minimum sample counts;
  - both Kay version checks;
  - the probe marks removed per wilbur epoch;
  - non-local events per epoch;
  - held-out spike evidence per decoder state (Task 12), with the half-tetrode labels (Task 13);
  - the list of hand-maintained doc claims that need new numbers.
- [ ] **STOP:** ask the project lead to review the report before any hand-maintained doc is
  requoted.
- [ ] **Docs.**
  - a new `docs/known_issues/` note on the re-decode: date, backup path, what changed, R9;
  - update `three_sessions_blocked_by_nwb_checksum.md` from R5's cohort answer;
  - the spec's status line;
  - `.claude/docs/plans/replay_detection_comparison/PLAN.md`;
  - the project memory notes on decode drift and the cohort.
- [ ] **Clean up.** After every gate passes, remove the staging folders. Keep `$B` and the
  golden baseline.

## Order at a glance

1. Task 1 Step 1.
2. Task 1.
3. R0.
4. Tasks 2–3, ending in commit `C0`.
5. R1: smoke and pilot from `C0`; then R10 (Task 11 pins the transition settings, `C0'`); then
   R1 again from `C0'`. Tasks 4–10 run in the worktree meanwhile, and Tasks 12–13 are
   implemented there while the batches run.
6. R2, which includes the merge to `C1`.
7. R3 and R4.
8. R5.
9. R6.
10. R7.
11. R8.

R3 can start right after the merge, while the decode gates are still being checked. Task 12's
run (CPU) starts after R2, alongside R3–R5. Task 13's decodes run on idle cards after R1's full
batches.

---

# Phase C: artifact masking and one SWR source everywhere (added 2026-10-09)

**Why.** The project lead asked (2026-10-05) to mask artifacts carefully: SWRs can look like
artifacts, and Spyglass's artifact rules (algorithm and parameters) are not vetted. Artifacts must
also be kept out of decoding. A read-only design study built and validated a detector:
`.claude/research/refresh_artifact_detector/detector_report.md`. Its results:
- masks 0 of 9,208 true SWRs (95 % upper bound 0.033 %);
- catches 95 of 111 steps confirmed by eye, 3 of 3 spike-only lines and 153 of 153 sustained-noise
  windows;
- masks 402 s (0.15 %) across 74 of the 218 epochs.

The lead approved the design on 2026-10-09, subject to a literature check
(`.claude/research/refresh_artifact_detector/literature_review.md`, 23 papers and 3 code sources
whose methods were read). The check found the design consistent with field practice. Rodent labs
drop an event that also appears where an SWR cannot: a noise channel, a cortical electrode, or EMG
estimated from the LFP. None of the rodent papers read reports how many true SWRs its rule
removes. The check also found one main gap: no direct check for licking or chewing EMG. It is
covered by study C1 below, before the full artifact run. The lead also chose to
re-decode all epochs so that the decoders' encoding exclusion uses the same SWR source as the
analyses: the masked 2.0 Kay inventory.

**Rulings (2026-10-09, controller):**
- **Simplified design (the lead, 2026-10-09).** This replaces the per-consumer masks, the post-step
  second, the per-analysis dependence windows and the invariance test of earlier drafts.
  - **One mask for everything.** The detector stores unpadded cores with their evidence. One table
    pads them:
    - S blocks ±1 s (noise spikes leak up to 1 s past a block's edges);
    - T cores with an LFP signature ±0.3 s (the ripple filter's 0.27-s ringing, and the burst of odd
      spikes that follows a step);
    - spike-only T cores ±0.1 s.

    The same mask feeds the SWR detectors' normalization and event drops, MUA detection, the
    decoders (`is_missing` and the encoding exclusion), the Widloski runner and every analysis.
  - **Artifact time is unobserved.** An event of any kind that overlaps the mask leaves every
    cohort; it is not counted as non-SWR. The mask's time leaves every rate exposure. Every
    normalization leaves it out.
  - **Consequences (the lead: "We also need to make sure we are not analyzing the artifact times
    or the consequences of artifact times in the analysis").**
    - **One analysed-time rule (the lead, 2026-10-10).** Excluded time is the mask widened by
      0.3 s on each side. An event of any kind (non-local, SWR, MUA, ContFrag label, Widloski), an
      exposure second, a time bin with its history window, and a peri-event window (PFC responses,
      reward kernels, triggered averages) is analysed only when it lies wholly outside excluded
      time. A kept event that overlaps a dropped event is dropped too, because its overlap class is
      unknown. Inputs (the decoders' `is_missing` and encoding exclusion, the detectors'
      normalization and the ripple-power covariate's normalization) use the mask itself.
    - Why 0.3 s: C3 measured that a masked gap changes analysed NonLocal events within at most
      0.21 s and flips ContFrag SWR labels within at most 0.19 s; the reach does not grow with gap
      length. ContFrag's posterior settles more slowly, by about 1.3 s, but the change is small: SWR
      mean P(Continuous) moves by 0.01 or more within 0.2 s in 99 % of gaps, and by 1.36 s at most.
      The reach is applied to every cohort and every exposure, not only to decoded events, so all
      rates share one exposure and NonLocal and ContFrag cohorts agree. Excluded time is 0.28 % of
      the 255 epochs (mask alone 0.17 %; R9 run `a1083731940b`).
  - **No post-step second.** No rodent lab we read pads after an artifact; the Buzsáki lab's
    `bz_FindRipples` drops flagged events and pads nothing. R8 instead reruns the headline results
    with 1 s masked after each step, as a robustness check.
  - **The spike-line rule (T-spk) stays** (the lead). It targets a mechanism no neural signal
    produces: many tetrodes firing within 200 µs with waveforms unlike any spike on their tetrode.
    The coincidence filter only removes lines within 40 µs. R9 reviews every spike-only core.
  - **The reference veto (R-rip) is dropped.** It removed 116 Kay events, most of them j16's, which
    the tetrode rule now handles.
  - **Names.** The former "true SWRs" are the *reference SWR set*: 9,208 events that meet stated
    rules (Kay∩Karlsson, 30–200 ms, speed < 4 cm/s over ± 250 ms, a sharp wave, MUA ≥ 2×). The
    former "visual" labels are *model-reviewed* labels: one AI reviewer, no human, no second rater.
    Neither is ground truth, and documents and methods say so.
  - **Justification**, without knowing which events are SWRs:
    - mechanism: each signature is something neural tissue cannot produce;
    - precedent read from source (C2);
    - one threshold rule (rule P, below);
    - a small footprint: 0.15 % of time;
    - robustness: R8 reruns the headline results without the mask, with every threshold 25 %
      lower and 25 % higher, and with 1 s masked after each step.
- **Keep the noisy tetrodes.** The sustained-noise tetrodes are masked during their noise blocks,
  not dropped. A sensitivity analysis (R8) leaves out the 4 epochs with more than 2 % masked.
- **Masking versus dropping the noisy tetrodes.** The field usually drops a channel with sustained
  noise instead of masking its time (literature review, verdict 6). R11 starts by decoding the 12
  S epochs both ways on breeze. Masking stays unless the masked decode shows extra non-local
  probability within 3 s outside the S blocks, beyond the 1-s pad, compared with the decode that
  drops those tetrodes. If it does, the affected tetrodes are dropped for whole S epochs.
- **No amplitude cap.** Some pipelines drop events above 9–10 SD. Our true SWRs reach 44 robust z,
  so a cap would remove giant SWRs. The design note reports how many reference-set SWRs a cap
  would drop.

**Reading the task bodies below.** They were written before the simplified design. Wherever they
say the `swr`, `decoder` or `analysis` mask, read *the one artifact mask*. Task 16's fix round 2
built it, and Task 17a's runners use it. Nothing pads a second after a step.

**Rulings after study C1 (2026-10-09; reports in `.claude/research/refresh_artifact_detector/followup/`):**
- **Tetrode selection for SWR detection (the lead's decision, 2026-10-09).**
  - What C1 found: 43 % of j16's Kay events lacked SWR features, against 1–4 % in the other animals.
    They are carried by j16 tetrode groups 0, 10 and 22, on every day. Those three do not ripple with
    the rest of CA1, yet they carry 5–7× its ripple-band power (`followup_c1c_report.md`).
  - The rule was stated before its effect was measured (`c1c_rule.json`). Per epoch, drop a CA1 band
    tetrode whose ripple envelope stays low while the other tetrodes ripple.
  - Its effect: j16's featureless share falls from 43 % to 1.9 %, true SWRs detected rise from 3,767
    to 4,229, and no visually confirmed SWR is lost.
  - The same kept tetrodes feed all four detectors. Task 17 implements it.
- **Reference event veto (R-rip): dropped in the simplified design** (116 events; most were j16's,
  which R-T handles). What C1b found, for the record: drop a detected event when the reference electrode's 150–250 Hz
  envelope is more than 11.36 times the CA1 median envelope at the event's peak.
  - This is the field's noise-channel check, scaled by CA1 at the same moment
    (`followup_c1b_report.md`).
  - θ = 1.5 × the largest true-SWR value; it lost 0 true SWRs in all 7 held-out splits.
  - The field's own form (buzcode, FMAToolbox) is not used: at its default it drops 2.7 % of true
    SWRs, because large SWRs reach our white-matter reference.
- **Not adopted:**
  - a reference 250–400 Hz time mask (no margin over giant SWRs, and it removes 20 events);
  - a raw-oscillation rule for small steps (fitted on the steps themselves);
  - a per-spike amplitude ceiling (catches nothing new).
- **Coincidence filter: kept.** It removes 0.52 % of true-SWR spikes, almost all of them
  same-sample, common-mode lines.
- **Visit onset.** In every animal, events in the first second after the pump or poke-in lacked
  features more often. After the tetrode rule, the featureless share at the ports is 1.0 % (rewarded)
  and 4.8 % (unrewarded) over all animals. No visit-onset or padded-speed rule is added. The design
  note reports these numbers.

## Task 16: artifact detection in the repo, with a run folder

Work in the worktree `/cumulus/edeno/continuum-swr-replay-refresh-dev`, branch
`full-pipeline-refresh-code`.

**Inputs:**
- **Prototype:** `.claude/research/refresh_artifact_detector/scripts/`:
  - `artdet_detector.py` (the rules);
  - feature code in `artdet_lfp.py`, `artdet_lfp2.py` and `artdet_lfp3.py` (S window spectra, J5 and
    L20 steps) and in `artdet_spikes.py` (spike lines and nearest-neighbour mark distances);
  - `artdet_common.py`.

  Read the report's §1, §5.1 and §6 first.
- **Reference output:** `artifact_intervals_default.csv` in the same folder: 470 intervals in 74 of
  the 218 epochs, with `swr`-mask padding.
- **Validation sets** in `validation/`: `true_swr_set.csv`, `artifact_set.csv`,
  `visual_labels_round*.csv` and `S_reference_windows.csv`.
- **Data, read-only:**
  - `get_ripple_band_inputs` (`data_loaders/lfp.py`): the raw referenced CA1 LFP, the reference
    electrode columns, the stored band and the time;
  - `get_hpc_marks` (`data_loaders/spikes.py`), after the coincidence filter;
  - the epoch span the SWR runner uses.

**Design rules:**
- Mask only on an artifact signature: S, T-ref, T-ca1 or T-spk. Never mask on power alone.
- Every threshold is in robust per-channel units: the statistic divided by its epoch MAD × 1.4826.
  No absolute LFP units.
- Parameters are module constants, each with a one-line evidence note naming the report section.
  Values, verbatim from the report:

  | Constant | Value |
  |---|---|
  | `s_core` | 4 |
  | `s_core_run` | 2 windows of 1 s |
  | `s_ext` | 2 |
  | `s_gap` | 2 windows |
  | `t_ref` | 15 |
  | `t_ca1_k` | 20 |
  | `t_ca1_rho` | 0.5 |
  | `t_ca1_frac` | 0.15 |
  | `t_ca1_min` | 3 channels |
  | `t_spk_frac` | 0.40 |
  | `t_spk_ood` | 0.20 |
  | `t_spk_bin` | 200 µs, on two grids offset by half a bin |
  | OOD cut | nearest neighbour > 4 × the decoder's `waveform_std` (read from the decoder settings, not a literal) |
  | `t_merge` | 0.05 s |

- The stored file holds **unpadded cores and their evidence**. Consumers pad them from one table:

  | Consumer | S | T with an LFP signature | T spike-only |
  |---|---|---|---|
  | `swr`: SWR-detector normalization and event dropping | 0 s | ±0.3 s | ±0.1 s |
  | `decoder`: `is_missing`, encoding exclusion, MUA detection | ±1 s | ±0.3 s | ±0.1 s |
  | `analysis`: event eligibility and rate exposure | ±1 s | 0.3 s before the core, 1.3 s after it | ±0.1 s |

  The `analysis` mask is the `decoder` mask plus the 1 s after each LFP-signature step.
  **Superseded:** fix round 2 collapses this table into one mask (S ±1 s, T with an LFP signature
  ±0.3 s, spike-only ±0.1 s), per the simplified design. The runners Task 17a changed switch to it.
  It also sets the rule-P constants and puts C2's justification table into the design note.
- **The OOD reference must not depend on a decode.** The prototype measured marks against a fitted
  decoder's encoding marks (`*_HPC_detector.pkl`), and the new decodes will themselves use this mask.
  Use instead, per tetrode, all of the epoch's marks outside the LFP-signature intervals (S, T-ref
  and T-ca1, padded by the `decoder` table), leaving the spike itself out. Use the same mark space
  and distance as the prototype. The acceptance criteria check that this reproduces the prototype.

**Files:**
- Create the package `src/continuum_swr_replay/analysis/hippocampus/artifacts/`:
  - `features.py`: robust per-channel scaling; S window power and the specificity ratio; J5 and L20
    on CA1 and on the reference; spike lines with their OOD fractions. Vectorised, with array shapes
    documented.
  - `detect.py`: the constants and the four rules. `detect_artifacts(...) -> pd.DataFrame` returns
    cores with columns `kind`, `signatures`, `core_start`, `core_end` and the evidence columns.
  - `masks.py`:
    - `CONSUMER_PADDING`;
    - `artifact_intervals(cores, consumer) -> pd.DataFrame[start_time, end_time]`, merged and clipped
      to the span;
    - `interval_mask(time, intervals) -> np.ndarray[bool]`, with closed intervals, the project's
      convention.
  - `io.py`: read and write the per-epoch file and the run manifest.
- Create `scripts/decoding/run_artifact_detection.py`, a CPU runner in the style of
  `run_swr_detection.py`:
  - it selects epochs from the register (`analysed_epochs`, `--include-pending`, `--pending-only`);
  - it can resume, writes atomically and releases NWB files after each epoch (`release_nwb_files()`);
  - a failed epoch is recorded and does not stop the run.
  - Output: `Processed-Data/artifact_intervals/<run_id>/{epoch_id}_artifact_intervals.parquet`, with
    an empty file for an epoch with no artifacts, plus `manifest.json`. The manifest records, per
    epoch, the status, span, input row counts, constants, code commit and sha256. The run id is the
    code commit's first 12 characters, as for R3.
- Update `swr_source.py`:
  - `artifact_intervals(epoch_id)` reads the new file and returns the `analysis` mask's intervals;
  - `swr_observed_mask` and `swr_observed_stretches` exclude that mask's time;
  - `ARTIFACT_INTERVALS_RUN_ID` stays None in this task. Runbook R9 sets it.
- Create the design note `docs/design/2026-10-09-artifact-detection.md`. It covers:
  - the decision and the signatures;
  - the parameter table with its evidence (report §5.1) and the validation numbers;
  - the Spyglass comparison, including Spyglass #1695;
  - the consumer table;
  - the literature mapping (`literature_review.md`);
  - known limits: 16 small steps missed, a single rater, licking not measured.

**Tests (write them first):**
- Synthetic cases:
  - a reference step gives a T-ref core;
  - a persistent step on 3 of 20 CA1 channels gives T-ca1;
  - a large ripple burst on every channel, with jumps that do not persist, gives no core;
  - 5 s of ripple-band noise on 4 of 20 channels gives one S block, and a 1-window dip inside it is
    filled;
  - a 200-µs spike line on 50 % of tetrodes with OOD marks gives T-spk;
  - the same line with in-distribution marks gives no core.
- The padding table: each consumer's intervals from a known set of cores, including merging and
  clipping.
- `interval_mask`: closed intervals.
- The runner: epochs come from the register (stubbed); resume skips finished epochs; a failure is
  recorded; an epoch with no artifacts writes an empty file.
- `swr_source`: the hook applies the `analysis` table, coverage excludes its time, and a configured
  run with a missing file raises.

**Acceptance:**
- **Real data, read-only.** Run on the 5 pilot epochs and on the 14 epochs with the most masked time
  in `artifact_intervals_default.csv`:
  - every prototype core is overlapped by a repo core, and the reverse;
  - masked time per epoch is within 2 % of the prototype's;
  - every difference is explained.
- **Same validation verdicts in those epochs:** no true-SWR or visually confirmed SWR is masked
  beyond the prototype's 5 label errors; the visual T, SPK and S windows are caught as in the
  prototype.
- **Cost:** runtime and memory per epoch are measured on the pilots, with the extrapolation to 255
  epochs.

## Task 17: the runners use the masks

**Files and changes:**
- **SWR runner** (`scripts/decoding/run_swr_detection.py`, `analysis/hippocampus/swr_inventories.py`):
  - **One mask for all four detectors.** Kay, Karlsson, Yu and Zugaro each get
    `normalization_mask = ~swr_mask` on their own time grid. Every detector's events that overlap
    the `swr` mask are dropped, and the drops are counted per detector in the manifest. The same mask
    for all four keeps the comparison fair.
  - Do not NaN-fill the LFP before filtering (report §6.2).
  - `swr_observed` marks the `swr` mask's time as not observed.
  - The manifest records the artifact run id.
  - **Tetrode selection (rule R-T, the lead's decision of 2026-10-09).** It runs once per epoch,
    before any detector, and all four detectors use the kept tetrodes. Zugaro chooses its channel
    only among them. Put it in a new module, `analysis/hippocampus/ripple_tetrodes.py`, ported from
    `followup/scripts/c1c_*.py`. The definition, verbatim from `followup/c1c_rule.json`:
    - For each CA1 band tetrode c, take env_c = |Hilbert(stored ripple band)|. Its robust z, z_c, is
      taken over the epoch's finite samples.
    - LOO_c(t) = sqrt(gaussian_smooth(Σ over the other tetrodes of env², σ = 4 ms)). It is z-scored
      by its mean and SD over the reference samples: finite, speed < 4 cm/s, and outside the `swr`
      mask.
    - coupling_c = the median of z_c over the reference samples where LOO_c ≥ 5.
    - Drop c when coupling_c < 1.0.
    - Constants, each with a one-line evidence note: cut 1.0, LOO threshold 5, σ = 4 ms. Evidence: in
      j16, groups 0, 10 and 22 score −0.12 to 0.81 in all 58 epochs, and every other j16 tetrode
      scores ≥ 1.78.
    - **Fail closed.** The epoch fails, with its reason recorded, when the rule would drop more than
      a third of the tetrodes, or when no reference sample has LOO ≥ 5.
    - The manifest records, per epoch, every tetrode's coupling and the dropped electrode groups.
  - **Fix the 2 j16 epochs** (`j1620210710_02_r1`, `_04_r2`). They fail with "Found 2 potential
    parents" on the position merge table. Resolve the position the way the decoder's `load_data`
    does, and add a test with the failing restriction stubbed.
- **MUA / HSE detection,** wherever the project runs it (find every `multiunit_HSE_detector` caller
  and the decode's `get_data_over_time`): use the `decoder` mask as its normalization mask, and
  drop overlapping events.
- **Decoders** (`run_decoding_bandit.py` and `run_decoding_contfrag.py`, HPC only):
  - `is_missing` = the `decoder` mask on the decode time grid, passed to `estimate_parameters`.
  - Encoding training:
    `is_training = ~(Kay inventory events of the configured run) & ~decoder_mask`, on position time.
    - Use every event in the inventory after artifact dropping, not only the immobile cohort, because
      the old rule excluded every stored ripple.
    - Check which speed gating the stored ripples had at detection, match it, and report it.
    - This replaces the stored-ripple source of `get_training_timepoints(..., "no_ripple")` for HPC
      (the lead's decision of 2026-10-09).
  - `data_over_time`'s `SWR_event` comes from the same inventory, so the decode's provenance equals
    the analysis source.
  - Record, with the decode's settings provenance:
    - the SWR inventory run id;
    - the artifact run id;
    - the seconds set missing;
    - the seconds removed from encoding.
  - Refuse to run if either run id is unset (fail closed).
- **Widloski masking (Task 17b, added 2026-10-09).** The runner reads the saved decode's encoding
  model, so the new encoding exclusion reaches it. Its own likelihood still reads every bin. So:
  - `decoder`-mask bins are missing in its likelihood;
  - no candidate window may touch them;
  - they enter neither the shuffle null nor any normalization.
- **Widloski runner** (`run_memoryless_detection.py`): fix the thread cap. The dask nanny sets
  OMP/MKL/OPENBLAS_NUM_THREADS = 1 in each worker, and R4 took 18 h against an estimate of 5–6 h.
  Measure one epoch before and after: runtime, plus outputs identical bit for bit.

- **Design note** `docs/design/2026-10-09-swr-tetrode-selection.md`. It covers:
  - C1's port-period results;
  - the j16 finding and the R-T rule, with its before/after table and blind checks;
  - the R-rip veto (C1b) and the field noise-channel form, and why neither is used;
  - the visit-onset numbers;
  - the limits: licking was not measured, there was one rater, and why j16's three tetrodes are
    noisy is unknown.

  Link it from `docs/README.md` and from the artifact design note.

**Tests (write them first):**
- the SWR runner passes the same mask to all four detectors, counts the drops, and leaves the
  artifact time out of `swr_observed`;
- R-T on synthetic data:
  - a tetrode with ripple bursts of its own that do not follow the others is dropped;
  - a tetrode that ripples with the others is kept;
  - samples inside the `swr` mask do not count as reference samples;
  - more than a third dropped fails the epoch;
- all four detectors receive the same kept tetrodes, and Zugaro's channel is one of them;
- the decoders build `is_missing` and `is_training` from stubbed sources, and refuse to run when a
  run id is unset;
- the Widloski workers see the intended thread counts.

**Acceptance:**
- **One decode.** A pilot epoch with artifacts (`chimi20200225_02_r1`) is decoded into a scratch
  staging folder:
  - masked bins have a log-likelihood of 0 (spot-check);
  - the seconds set missing and the seconds removed from encoding are reported;
  - the runtime is compared with the previous decode.
- **SWR runner** on the 5 pilots and the 2 j16 epochs, with a scratch artifact run: Kay counts are
  reported next to the prototype's masked counts (`validation/kay_effect_per_epoch.csv`).
- **R-T, read-only, against C1c.** Use the 5 pilots, 4 j16 epochs on different days,
  chimi20200227 and chimi20200228 (one epoch each), and 2 senor S epochs.
  - The dropped groups match `followup/c1c_rule_drops.csv`.
  - Kay events match C1c's selected run (`followup/data/c1c_rule_events.parquet`, column `run`).
    The two runs use different artifact masks (the prototype's against the scratch run's), so
    explain every difference. C1c also applied R-rip, which is dropped: count its events
    separately.

## Task 18: tables and analyses use the mask

**Files and changes:**
- **One eligibility rule for every event cohort** (non-local, SWR, MUA, ContFrag and Widloski): the
  whole event is immobile (the existing rule) **and** it does not overlap the artifact mask.
  - Define it once, next to `analysis/hippocampus/event_immobility.py`.
  - A dropped event keeps a reason, as `swr_source`'s `DropReason` does.
  - Grep every cohort builder and list each, with your decision, in the report.
- **Exposure:** every rate denominator loses the mask's time. That includes:
  - stop-level immobility time (Family A's offset);
  - per-bin tables (drop bins inside the mask; Family B);
  - PFC per-bin rows;
  - any other exposure you find.
- **Consequences (the simplified design in the Phase C introduction).**
  - Add one function that returns the mask's intervals padded by a given distance; everything uses
    it.
  - **Decoded events** (non-local, ContFrag labels, Widloski) within 0.3 s of the mask are dropped,
    with their reason.
  - **Per-bin rows with history covariates:** drop a bin when its longest history window, taken from
    `model_specs.py`, touches the mask.
  - **Peri-event windows** (PFC responses, MUA, reward kernels): drop a window that touches the mask.
  - List every analysis output you changed, and how, in the report.
- **Stamps:** tables carry `artifacts@<run>` through the existing `swr_source` stamp.
- **Docs:**
  - update `docs/known_issues/swr_mua_count_conventions.md`;
  - write a CLAUDE.md change to the "One speed-eligible population" paragraph as a **proposal in the
    report**. Do not edit CLAUDE.md.

**Tests (write them first):**
- an event that overlaps the mask leaves every cohort;
- exposure falls by exactly the masked time inside the stop;
- per-bin rows inside the mask are dropped;
- with no artifacts, every output is identical to before.

**Acceptance:**
- On the 5 pilots with a scratch artifact run, report per cohort the counts before and after, and
  the exposure removed.

## C1: follow-up checks from the literature review (read-only, CPU; done 2026-10-09)

Output: `.claude/research/refresh_artifact_detector/followup/`. Reports: `followup_report.md` (C1),
`followup_c1b_report.md` and `followup_c1c_report.md`. The rulings are in the Phase C introduction.
1. **Events at the ports (the licking question).** There is no EMG and no lick sensor; a beam
   break shows only that the nose is in the port. So C1 tested the detected events, not licking.
   - It compared Kay events during rewarded visits, unrewarded visits and immobility away from the
     ports.
   - It measured their SWR features (sharp wave, MUA, multi-channel ripple, in-distribution marks)
     and reviewed them blind.
   - The 250–400 Hz proxies only pointed to candidates; the stored LFP is cut off at about 410 Hz.
   - **Result:** a featureless subpopulation, mostly j16. C1b calibrated the R-rip veto on
     held-out data, and C1c traced j16 to three tetrodes (the R-T rule).
2. **Raw-signal oscillation check** on the 16 missed small steps: not adopted. A two-feature rule
   catches 6 of 16, but its cuts were fitted on those 16.
3. **Per-spike amplitude ceiling:** not adopted. It catches nothing that is not already masked.
4. **Coincidence filter:** kept. It removes 0.52 % of true-SWR spikes, almost all of them in the same
   30-kHz sample. Correct the docstring's "rarely exceeds 20 %" claim when convenient: with 40-µs
   chaining, 5.5 % of true SWRs exceed 20 %, and only 2 % exceed 33 %.
5. **Amplitude cap,** for the methods: a cap at 9 or 10 robust z would drop 27.6 % or 23.7 % of
   true SWRs.

**C2 outcome and rulings (2026-10-09; `followup/justification_report.md`):**
- **Placement rule P.** Every artifact threshold is 1.5 × the largest value its statistic reaches in
  the calibration SWR sets: true SWRs, visual SWRs and giant-day events, measured over the event
  plus its padding.
  - Why 1.5: one animal's maximum, divided by the other four animals' maximum, reaches 1.27–1.59.
  - Definitional constants are exempt, and their reason is a mechanism or a precedent: window
    lengths, bins, persistence and padding.
  - R-rip (11.36) already follows the rule.
- **Constants that change under P (a Task 16 fix round, after Task 17a):**

  | Constant | Now | Under P |
  |---|---|---|
  | `t_ref` | 15 | 1.5 × 11.716 = 17.6 |
  | `t_ca1_k` | 20 | 1.5 × 16.682 = 25.0 |
  | `s_core` | 4 | 1.5 × 2.872 = 4.31 |

  - The code writes each threshold as the margin times a named calibration maximum, with an evidence
    note.
  - Measured cost: 1 of 111 visual steps is no longer caught, and no SWR count changes.
  - The reason: when the rule was placed without chimi, the current values masked 2 visual SWRs and 2
    giant-day events there.
- **T-spk stays off the rule** (0.40 of tetrodes, 0.20 OOD), with its limits stated.
  - True SWRs reach 0.364 of tetrodes in bins with at least 20 % OOD marks, so the margin is thin.
  - It rests on 3 labels and masks 5.2 s in total.
  - R9 reviews every spike-only core in a gallery, which adds labels on the full data.
- **R-T keeps its cut of 1.0 and its LOO threshold of 5**, the values stated before any effect was
  measured.
  - The cut is a mechanism constant: a rise of less than one robust SD while the rest of CA1
    ripples.
  - Task 17b's acceptance reports LOO 6.25 as a sensitivity check. It drops no rippling tetrode;
    5 drops 2 tetrode-epochs, about 1 true SWR.
- **Precedents read from source:**
  - FieldTrip `ft_artifact_jump`: jump z > 20 after a 9-point median filter; it supports T-ref and
    T-ca1.
  - PREP, EEGLAB `clean_channels` and IBL / SpikeInterface `detect_bad_channels`: low correlation
    with the other channels and high-frequency noise; they support R-T and S.
  - None of the four has a reference veto, a persistence test or an OOD spike-line rule.

## C2 and C3: justification and consequences (read-only; started 2026-10-09)

- **C2** (`followup/c2_brief.md`, report `justification_report.md`) justifies every rule (S, T-ref,
  T-ca1, T-spk, padding, R-T, R-rip) and every constant. For each it gives:
  - the mechanism, and why SWRs lack the signature;
  - field precedents read from source: FieldTrip `ft_artifact_jump`, PREP, EEGLAB
    `clean_channels`, IBL / SpikeInterface `detect_bad_channels`, and the existing review;
  - one stated rule for placing every threshold;
  - held-out validation, with folds by animal and leave-one-animal-out.

  A changed constant is a Task 16 fix round, or goes into Task 17b. R9 and Task 17b wait for C2.
- **C3** (`followup/c3_brief.md`, report `followup_c3_report.md`; GPU on breeze) measures how far a
  masked gap changes the NonLocal and ContFrag posteriors and their events. It injects gaps into
  clean epochs and compares decodes with the same fitted model. Its result sets the decoder
  consequence window in Task 18.

## R9: artifact detection, all epochs (CPU)

1. Fast-forward the main branch to Task 16 once its review passes, and push. R9 needs only Task 16.
   Tasks 17–18 continue in the worktree and merge after R9 finishes, because the run imports from
   the main tree.
2. Run on all 255 epochs (218 analysed and 37 pending) in tmux `artifacts_full`, on CPU, from a
   clean tree. Use 8 workers and about 30 GiB of memory: one epoch peaked at 3.8 GiB.
3. **Gate.**
   - **Provenance:** every manifest record has `dirty: false`; otherwise the run is not used.
   - **Reference electrodes:** list every epoch with `t_ref_disabled` and every epoch with
     `n_time_gaps` > 0. Accept each one explicitly before setting the run id.
   - **Repo-only and prototype-only cores:** list every core with no prototype core, and every
     prototype core with no repo core, and review each in a gallery. Differences are expected:
     rule P raised two thresholds, and the OOD reference no longer depends on a decode. One known
     example is a 0.2-s spike-only prototype core in chimi20200227 that the repo does not flag
     (Task 17b). Each difference must be explained, not just matched.
   - **Every spike-only core** goes into a review gallery too. T-spk rests on 3 labels (C2).
   - **S margins:** in the 12 S epochs, list every spike-only core within 1.1 s of an S core and
     check it against the true-SWR and visual-SWR sets. This is review item 2 on the decode-free OOD
     reference. If a true SWR is masked there, rule on it before going on.

   On the 218:
   - masked time is within 2 % of C2's rebuilt-detector prediction at the shipped constants and
     one mask (301.06 s against 300.66 s on the 22 acceptance epochs), in total and per epoch.
     Every core-level difference from the prototype is explained (above);
   - the validation verdicts are reproduced:
     - reference SWR set: 0 masked after adjudication;
     - model-reviewed SWRs: only the known S-padding case is masked (senor20201103_14_r7, 202.48 s,
       0.55 s before an S block), so 1 of 55;
     - model-reviewed steps: at least 94 of 111 (rule P; C2's prediction);
     - spike-only lines: 3 of 3;
     - S windows: 153 of 153.

   On the 37 pending epochs, review a gallery of their intervals.
4. Set `ARTIFACT_INTERVALS_RUN_ID` (one commit), and push.

## R10: SWR inventories rerun with the mask (CPU)

- Run `run_swr_detection.py` on all 255 epochs into a new run folder.
- **Gate:**
  - the 14 low-count epochs have SD / robust SD ≤ 1.3;
  - the 2.0 Kay total on the 218 is within 1 % of C1c's selected run, 34,281, plus the events
    R-rip removed there (at most 116; R-rip is dropped). That run has the tetrode rule and leaves
    out events overlapping the prototype's `swr` mask. The prototype's 41,743 predates the tetrode
    rule;
  - R-T drops groups 0, 10 and 22 in every j16 epoch, and the other drops match
    `followup/c1c_rule_drops.csv`, or each difference is explained;
  - no epoch fails the R-T guard, or each failure is ruled on;
  - both j16 epochs are `ok`;
  - Karlsson, Yu and Zugaro counts are reported before and after the two rules (no gate).
- Then set `SWR_INVENTORY_RUN_ID` (one commit), and push.

## R11: full re-decode, all 255 epochs (GPU, zephyr and breeze)

- NonLocal and ContFrag, both into a new staging folder, from the commit that sets both run ids.
- First, on breeze, decode the 12 S epochs with masking and with their noisy tetrodes dropped, and apply
  the Phase C ruling (superseded 2026-10-10: see Rulings).
- **Also first: j16's three tetrodes in decoding.** R-T removes groups 0, 10 and 22 from SWR
  detection only. Two of them have spikes that rise during true SWRs (own MUA 2.8× and 3.0×), so
  they stay in decoding by default.
  - Their LFP is loud at the ports, and their mark rate rises over days (C1c). So decode 4 j16
    epochs, one every two days, with and without those three tetrodes.
  - Compare the non-local events: count, overlap, and the share at the ports.
  - Keep the tetrodes unless more than 5 % of the events with them have no overlapping event
    without them. If they do, bring the comparison to the lead before the full run.
- **Rulings (project lead, 2026-10-10).**
  - **S epochs excluded.** The 13 epochs in which artifact run `fbbff144a123` finds an S block
    (11 analysed, 2 pending) are left out whole. No tetrode is dropped. The register gives them
    reason `SUSTAINED_NOISE`, and the audit's `sustained_noise` check holds the register to the run.
    The reasons: the S-epoch test failed (3 of 13 epochs fail; pooled over 14 blocks the excess is
    +0.054, 95 % CI [0.019, 0.102]; `.claude/research/refresh_r11/r11a_report.md` §5), and R1's
    decodes of the S epochs moved across the whole epoch (−8 % to −57 % non-local events in 6 S
    epochs against R11; `.claude/research/refresh_r11/gates/r11_full_gates_report.md`, gate 6).
  - **j16's groups 0, 10 and 22 kept in decoding.** They fail the 5 % rule (13.4 % of events
    lost), but 8 random sets of 3 other tetrodes per epoch, matched for spike share, lose 11.0–13.1 %
    pooled. The excess is about 20 events at the ports on one day, 07-07: about 2 % of the four
    days' events (`.claude/research/refresh_r11/j16_control/j16_control_report.md`).
- Breeze: SSH from zephyr works (2026-10-09). Check `nvidia-smi` and use idle cards only.
- **Gates:** as R1 (P30-B pinned, no CPU fallback, provenance), plus both run ids recorded in every
  decode.
- **Swap:** move the 218 old decodes out of the canonical folder, and move 242 R11 decodes in: the
  207 analysed epochs and the 35 pending ones, whose entries flip to `analysed` in the same change
  (with `docs/epoch_register.md` regenerated). The 13 S epochs' R11 decodes stay out of the
  canonical folder. Use the R2 swap procedure with a sample re-hash.

## R12: Widloski rerun, all 242 epochs

Run with the thread fix, into a new run folder.

## R13: tables, refits, PFC, figures, audit

R5–R8 as written, on 242 epochs (255 less the 13 S epochs), with stamps
`kay@<R10 run>+artifacts@<R9 run>`. R8 adds:
- the sensitivity analysis without the epochs over 2 % masked. Restate that set on the 242 epochs:
  the old "4 epochs" no longer fits, since two of the most-masked epochs (senor20201107_12_r6 and
  senor20201101_14_r7) are now excluded;
- the masked-time table, and the events and bins the mask and the 0.3-s decoder reach remove;
- robustness reruns of the headline results: without the mask; with every artifact threshold 25 %
  lower and 25 % higher; with 1 s masked after each step.

**The audit's `epoch_sets` and the S epochs (coordinator's ruling, 2026-10-10).** The SWR run
`056eee0dcb5a` holds the 13 S epochs, and it is not rerun. A new SWR run id would make the
analysis-side run-id checks refuse every R11 decode, since each records `056eee0dcb5a` in
`attrs["decode_masks"]`. So `check_epoch_sets` accepts `SUSTAINED_NOISE` epochs in run manifests
(SWR, Widloski) only, and lists them under `held_sustained_noise`. A decode, a table or the decode
folder holding one still fails.
