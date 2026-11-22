# Reproduce the analyses

How to rebuild the paper's results, from the recordings to the PDF. Run every
command from the repository root.

There are four levels, each needing more than the one before:

| What | Command | Needs |
| --- | --- | --- |
| Build the PDF | `make manuscript` | LaTeX |
| Write the reported numbers into the LaTeX macros and the docs | `make reported-values` | the committed summaries; Python |
| Rebuild the summaries and the main figures | `make summaries`, `make figures` | the lab's processed data |
| Before a milestone (a draft to the co-authors, submission, a release): check that every summary is current with the code and its inputs | `make release-check` | the committed summaries; Python; the lab's processed data (`make release-check ARGS=--code-only` checks the code alone, without it) |
| Rebuild everything, from the recordings | [the full rebuild](#the-full-rebuild) | the lab's Spyglass database, the NWB files, GPUs |

There is no public reproduction yet: every analysis reads the lab's data.
Archiving the analysis tables so that others can rebuild the figures and the
summaries is planned (see `TODO.md`).

## Requirements

- **Python environment.** [uv](https://docs.astral.sh/uv/getting-started/installation/)
  and the committed `uv.lock`; `.python-version` selects Python 3.11.
  `make sync` installs the analysis environment and `make sync-dev` adds the
  development tools. The decoders and the Part 4 fits run on NVIDIA GPUs; install
  the CUDA build of JAX and the decoders' GPU cluster (`dask-cuda`) with
  `uv sync --frozen --inexact --extra gpu` (Linux only; `--inexact` keeps the development tools).
- **LaTeX.** `latexmk`, `pdflatex` and BibTeX, from TeX Live or MacTeX, for
  `make manuscript`.
- **The lab's data.** Loading a session (`load_data`) queries the lab's
  Spyglass database and reads its NWB files, so the decodes, the detectors, the
  table builds and Figures 2–4 need a machine with database access (a DataJoint
  configuration, `dj_local_conf.json`).
- **The processed-data root.** Every script reads and writes under one root,
  `$CSR_PROCESSED_DATA_DIR` if it is set, else the checkout's `Processed-Data/`
  (`continuum_swr_replay.paths.processed_data_dir`). Point it at the shared copy
  to reuse existing outputs:

  ```bash
  export CSR_PROCESSED_DATA_DIR=/path/to/Processed-Data
  ```

[Data lineage](data-lineage.md) lists what each step reads and writes.

## The full rebuild

All scripts are in `scripts/`. Steps 1–8 are the full rebuild. After a change to the table code, start at step 4;
after a change to fitting code only, start at step 6.

1. **Cohort.** The analyzed epochs are the register's (`data_loaders/epoch_register.csv`;
   [epoch_register.md](epoch_register.md)).
2. **Artifacts and SWRs.**
   - Run `detection/run_artifact_detection.py` (the artifact mask), then `detection/run_swr_detection.py` (the
     four SWR inventories).
   - Then set the two run ids the decoders read: `ARTIFACT_INTERVALS_RUN_ID` and `SWR_INVENTORY_RUN_ID` in
     `src/continuum_swr_replay/analysis/event_definition/swr_source.py`. A decode refuses to start without them.
3. **Decodes.** `decoding/run_decoding_bandit.py` (non-local detector) and `decoding/run_decoding_contfrag.py`
   (ContFrag), on GPUs.
4. **Tables**, in this order:
   1. `behavior/fetch_spatial_bandit_behavior.sh`;
   2. `tables/prepare_replay_content_data.py --stop-only` (the stop table);
   3. `behavior/build_qvals_table.py --write` (needs the stop table);
   4. `tables/prepare_replay_content_data.py` (the event, event × arm, event × patch, sequence and stop
      tables; needs `qvals_table`);
   5. `tables/build_timebin_tables.py`;
   6. `tables/prepare_contfrag_replay_content_data.py --event-source SWR` and `--event-source MUA`.
5. **Parts 1 and 2** on the decodes: the checks, the detector comparison and the decode-only figures (Figures 1–3,
   the supplements). The runs behind Figure 2g, the supplementary held-out tetrode figure and the restriction
   and held-out tetrode values of the events summary go in this order (in `scripts/event_evidence/` unless
   named otherwise; the two GPU runners need `--gpus`):
   1. `score_nonlocal_restriction.py --gpus <cards>`: the three restriction decoders, each fold fit by EM on
      the other folds (reads the decodes, the detectors, the trials through `load_data` and the SWR source);
   2. `measure_bin_facts.py`: each bin's spikes, well-visit flag and SWR-observed flag (reads run 1's
      records, the decodes, the SWR source and the session cache);
   3. `detect_heldout_events.py`: the events detected from the held-out decode and the time sums, by
      category, well visit, event window, SWR-observed flag, margin from the SWRs and place sampling
      (needs 1 and 2; reads the event table);
   4. `measure_event_participation.py`: the Local-rest controls (reads the event table, the decodes and the
      session cache);
   5. `score_tetrode_holdout.py --gpus <cards>`: the held-out tetrode score (needs 4; reads the event table,
      the decodes, the detectors and the trials);
   6. `sum_nonlocal_restriction_windows.py`: the restriction scores over the canonical windows (needs 1, 2,
      4 and 5);
   7. `reported/summarize.py --summary events`, then `reported/emit_reported_values.py`;
   8. `figures/generate_figure02.py` and `figures/generate_supp_figure02.py` (the second is not drawn
      by `figures/generate_all_figures.py`).

   ```bash
   uv run --frozen python scripts/event_evidence/score_nonlocal_restriction.py --gpus 1 2 4 5
   JAX_PLATFORMS=cpu uv run --frozen python scripts/event_evidence/measure_bin_facts.py --workers 6
   JAX_PLATFORMS=cpu uv run --frozen python scripts/event_evidence/detect_heldout_events.py
   JAX_PLATFORMS=cpu uv run --frozen python scripts/event_evidence/measure_event_participation.py
   uv run --frozen python scripts/event_evidence/score_tetrode_holdout.py --gpus 1 2 4 5
   JAX_PLATFORMS=cpu uv run --frozen python scripts/event_evidence/sum_nonlocal_restriction_windows.py
   ```

   With two workers on a card (`--workers-per-gpu 2`), the largest epochs can run out of GPU memory; run the
   same command again with `--workers-per-gpu 1`. Each runner reruns only the epochs whose record is not `ok`.
   Steps 3 and 6 read another per-bin facts run with `--bin-facts-dir` (and another restriction run with
   `--restriction-dir`); with `--output-dir`, a trial of the CPU steps writes nothing under the data root.
6. **Part 3 fits and the figures that read them:** `workflows/run_refits.py`, or `make refits` (`make refits ARGS=--dry-run` lists the runs).
7. **Part 4:** `pfc/run_pfc_analysis_bandit.py` on GPUs, then `figures/generate_figure05.py`.
   `figures/generate_all_figures.py` draws Figures 1–3 and 5 in one go; run it only after this step.
8. **Summaries, macros, PDF.** Rebuild the five summary files (parts 0–4), write their
   values into the LaTeX macros and the docs, and build the PDF:

   ```bash
   make summaries        # scripts/reported/summarize.py --summary <name>, for each summary
   make reported-values  # scripts/reported/emit_reported_values.py
   make manuscript
   ```

   `make reported-values` lists every number that changed, on the screen and in
   `paper/reported/changes.md`. Read that list before committing.

   Each summary records the files of the checkout that built it: the package
   modules and data files the build loaded and the inputs the repository tracks
   (the literature table, the committed sweep summaries), with their digests,
   the commit, and the digest of `uv.lock`. A summary in `paper/reported/` is
   built from committed files: `summarize.py` writes nothing there when one of
   those files differs from the commit (commit first; `--output-dir` elsewhere
   makes a trial build). A docstring or comment edit does not make a summary
   stale.
   `make reported-values` names a summary whose code has changed since, but never
   refuses it. `make release-check` fails on a missing summary file, a summary with
   no code record, a stale one, one whose single-file inputs changed or are absent,
   one with tag problems, a pending value the manuscript cites, or a
   `paper/reported_values.tex` that is not what the summaries give. Run it at a
   milestone where the processed data is (`make release-check ARGS=--code-only` skips the inputs), then
   rebuild only the summaries it names.

## Decoding

The main analysis workflow is in `scripts/decoding/run_decoding_bandit.py`, which performs continuum decoding using non-local detection algorithms.

### Decoding Workflow

1. **Load data**: Use `load_data()` to get all session data
2. **Set up environment**: Create `Environment` with place bins and track graph
3. **Select training data**: Exclude ripple times or use all data (via `get_training_timepoints()`)
4. **Perform decoding**: Use detector-specific methods:
   - `NonLocalClusterlessDetector` for HPC (uses waveform features)
   - `NonLocalSortedSpikesDetector` for mPFC/OFC (uses sorted spikes)
5. **Organize results**: Compute derived measures (ahead/behind distance, HPD)
6. **Save outputs**: Results as NetCDF (.nc), models as pickle (.pkl)
7. **Generate figurl**: Interactive visualization URL (optional)

### Key Decoding Parameters

| Setting | Value | Defined in |
| --- | --- | --- |
| Place bin, linearized track | 2.0 cm | `continuum_swr_replay.parameters.PLACE_BIN_SIZE_1D` |
| Place bin, 2-D | 3.5 cm | `continuum_swr_replay.parameters.PLACE_BIN_SIZE_2D` |
| Position standard deviation of the encoding model | √12.5 ≈ 3.54 cm | `continuum_swr_replay.parameters.POSITION_STD` |
| Waveform-feature standard deviation (clusterless HPC) | 24 µV | `continuum_swr_replay.parameters.WAVEFORM_STD` |
| Local state's position standard deviation | 2.5 cm | `continuum_swr_replay.parameters.LOCAL_POSITION_STD` |
| GPU block size | 1024 | `continuum_swr_replay.parameters.BLOCK_SIZE` |
| Training data | `no_ripple` | `continuum_swr_replay.decoding.identifiers.DEFAULT_TRAINING_TYPE` |
| Non-local probability threshold for events | 0.8 | `continuum_swr_replay.parameters.DEFAULT_NON_LOCAL_THRESHOLD` |

The discrete transitions are stationary: `--non_stationary` (speed as a transition
covariate) is refused until non_local_detector can save such a model
(`continuum_swr_replay.decoding.transitions.NON_STATIONARY_REFUSAL`).
`tests/repository/test_documented_values.py` checks this table against the code.

### Command-Line Usage

```bash
# Decode all sessions for an animal
uv run --frozen python scripts/decoding/run_decoding_bandit.py --animal chimi --brain_area HPC

# Decode specific session
uv run --frozen python scripts/decoding/run_decoding_bandit.py --animal chimi --date 20200212 --epoch 02_r1

# With an interactive figurl view of each decode
uv run --frozen python scripts/decoding/run_decoding_bandit.py --animal chimi --create_figurl

# Overwrite existing results
uv run --frozen python scripts/decoding/run_decoding_bandit.py --animal chimi --overwrite
```

### GPU and Distributed Computing

The script uses Dask with CUDA for GPU-accelerated parallel processing:

```python
# Automatically sets up LocalCUDACluster
# Configure visible GPUs via command line
--cuda_visible_devices "0,1,2,3"  # Use specific GPUs

# Set XLA memory fraction (done automatically in script)
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.99"
```

Multiple sessions are processed in parallel using `dask.delayed` and `dask.compute`.

### Output Files

Results are saved to `--save_dir`, by default `paths.processed_data_dir()` (`$CSR_PROCESSED_DATA_DIR`, else the checkout's `Processed-Data/`):

- `{epoch_identifier}_results.nc` - Decoding results (xarray Dataset)
- `{epoch_identifier}_detector.pkl` - Fitted detector model
- `{epoch_identifier}_environment.pkl` - Environment configuration
- `{epoch_identifier}_{threshold}_data_over_time.pkl` - Derived time series data

Logs are written to `logs/{epoch_identifier}.log` with git hash for reproducibility.

## What is committed

| Committed | Made by |
| --- | --- |
| `paper/reported/<summary>.json` | `make summaries` |
| `paper/reported_values.tex`, `paper/reported/questions.md`, the inline values in `docs/` and `paper/` | `make reported-values` |
| `paper/tables/table_s1_literature.tex` (Supplementary Table S1) | `scripts/figures/generate_supp_literature_table.py`, from the survey CSV beside it, which is edited by hand |
| `paper/main.pdf` | `make manuscript` |
| Final figures under `paper/figures/` (each listed in `.gitignore` as an exception) | the figure scripts |

Decodes, tables, fits, logs and every other file under the processed-data root
are not committed. After a change that alters results, commit the summaries,
the macros, the docs they fill and the rebuilt PDF together.

[Figure pipeline](figure-pipeline.md) gives each figure's script, inputs and tests;
the question list gives the questions and values each figure shows.
