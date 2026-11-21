# Data lineage

Where the data comes from and where each stage writes, from the recordings to
the numbers in the paper. [reproduce.md](reproduce.md#the-full-rebuild) gives
the order to run the stages in; this page gives what each one reads and writes.

Paths below are relative to the processed-data root, `$CSR_PROCESSED_DATA_DIR`
if it is set, else the checkout's `Processed-Data/`
(`continuum_swr_replay.paths`). Use the functions there to build a path, not a
hand-joined string. Nothing under the root is committed.

## Recordings and cohort

- **NWB files and the Spyglass database.** The raw data are NWB files from five
  rats (chimi, j16, peanut, senor, wilbur), read through the lab's Spyglass
  database. `NWB_FILES` in `src/continuum_swr_replay/data_loaders/constants.py`
  lists every file the loaders search. `load_data` reads position, spikes, LFP
  and task events from the Spyglass tables listed in
  [development.md](development.md#data-integration-with-spyglass).
- **Cohort.** Which run epochs are analyzed is decided by the epoch register,
  `src/continuum_swr_replay/data_loaders/epoch_register.csv`, one row per epoch
  with its status and the reason for every exclusion
  ([epoch_register.md](epoch_register.md)). Every stage reads it; no stage globs
  a folder to find epochs.

## Stages

| Stage | Output (under the root) | Written by |
| --- | --- | --- |
| Artifact mask | `artifact_intervals/<run id>/` | `scripts/detection/run_artifact_detection.py` |
| SWR inventories (Kay, Karlsson, Yu, Zugaro) | `replay_detection_comparison/swr/<run id>/` | `scripts/detection/run_swr_detection.py` |
| Decodes, non-local detector | `{epoch_id}_results.nc`, `_detector.pkl`, `_environment.pkl`, `_80_data_over_time.pkl` | `scripts/decoding/run_decoding_bandit.py` |
| Decodes, ContFrag | `{epoch_id}_contfrag_results.nc`, `_contfrag_detector.pkl`, `_contfrag_environment.pkl`, `_contfrag_data_over_time.pkl`; the SWR or MUA event source is chosen later, by `prepare_contfrag_replay_content_data.py --event-source` | `scripts/decoding/run_decoding_contfrag.py` |
| Canonical tables | `replay_content/{event,event_arm,event_patch,event_sequence,stop,qvals,event_onset}_table.parquet`; ContFrag tables under `contfrag_replay_content/` | `scripts/tables/` and `scripts/behavior/` (order in [reproduce.md](reproduce.md#the-full-rebuild)) |
| Part 3 fits | `replay_content/models*/` and the other result folders under `replay_content/` | `scripts/replay_models/` (`rate/`, `content/`, `coupling/`, `choice/`), all run by `scripts/workflows/run_refits.py` |
| Detector comparison | `replay_detection_comparison/{memoryless,carey,inventory,event_features}/<run id>/` | `scripts/detection/` (the detectors), `scripts/detector_comparison/` (the comparison and event features) |
| Part 4 fits (mPFC) | `{epoch_id}_mPFC_modulation.pkl` and its `.npy` arrays | `scripts/pfc/run_pfc_analysis_bandit.py` |

Two settings tie the decodes to the detector runs:
`ARTIFACT_INTERVALS_RUN_ID` and `SWR_INVENTORY_RUN_ID` in
`src/continuum_swr_replay/analysis/event_definition/swr_source.py` name the artifact and
SWR runs every decode and every analysis reads. A decode refuses to start
without them. The rule for which time and which events are analyzed is
described in [development.md](development.md#canonical-per-event-classification-table).

The event, event × arm, event × patch, event sequence and stop tables share
event ids, and the qvals table joins to them, so they are rebuilt together; a
partial rebuild mis-joins silently
([development.md](development.md#canonical-per-event-classification-table)).

The canonical tables sit at the root of `replay_content/`, beside their
standardization references (`standardization_reference.json`,
`qvals_standardization_reference.json`). They are parquet: read and write them
through `continuum_swr_replay.tables.io` (`read_table` / `write_table`), never
`pd.read_pickle`; `read_table`, `read_attrs` and `table_exists` refuse a `.pkl`
path. Fitted-model artifacts (`all_models.pkl` and the others) hold live
statsmodels objects and stay pickled. Superseded table pickles are kept under
`_pickle_archive/`; nothing reads them. No library module or supported script binds
a root at import. The old upper-case names (`parameters.PROCESSED_DATA_DIR`,
`paths.REPLAY_CONTENT_DIR` and the rest) still resolve for notebooks and
`scripts/exploratory/`, but assigning one moves nothing: to redirect a run, set
`$CSR_PROCESSED_DATA_DIR`. A run folder's `manifest.json` names its inputs and
settings.

## From results to the paper

| Stage | Output (in the repository) | Written by |
| --- | --- | --- |
| Summaries | `paper/reported/<summary>.json` | `scripts/reported/summarize.py --summary <summary>` (`make summaries`) |
| Macros and inline values | `paper/reported_values.tex`; the `<!--v:…-->` values in `docs/` and `paper/` | `scripts/reported/emit_reported_values.py` (`make reported-values`) |
| Figures | `paper/figures/<topic>/` | `scripts/figures/` ([figure-pipeline.md](figure-pipeline.md)) |
| Manuscript | `paper/main.pdf` | `make manuscript` |

Each summary records the commit that built it and the SHA-256 of every file it
read (its `inputs` list), so a reported number can be traced to the files it
came from: under the root for the lab-data summaries, in the repository for the
survey summary. Each also records the code that built it, and is written only
from committed code. The trace is not yet exact for sets of input files, which
are recorded under a label (see "Summary input paths" in `TODO.md`).
