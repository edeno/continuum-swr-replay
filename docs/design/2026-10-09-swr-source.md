# One SWR source: the direct ripple_detection 2.0 Kay inventory

**Status (2026-10-09):** decided by the project lead on 2026-10-05; implemented in
`src/continuum_swr_replay/analysis/hippocampus/swr_source.py`, the only place the source is
configured. Part of the full pipeline refresh
([2026-10-03-full-pipeline-refresh-design.md](2026-10-03-full-pipeline-refresh-design.md),
Task 15). Artifact masking (Task 16) plugs into the hook described below.

## Decision

Every SWR label of the paper comes from the Kay inventory that the SWR runner
(`scripts/decoding/run_swr_detection.py`, refresh stage R3) writes. Spyglass's stored
`RippleTimesV1` events are no longer an analysis source. This covers `SWR_event`,
the event table's `is_swr` / `overlap_class` / matched SWR, the stop table's
`is_swr_stop` / `n_swr_events`, the timebin tables' SWR onsets, the ContFrag SWR-event
tables, the Widloski-detector table's SWR columns, and the PFC path's SWR intervals.

- **The rule.** ripple_detection 2.0's `Kay_ripple_detector`: an event needs at least
  15 ms continuously above z = 2. This is the published Kay et al. (2016) definition.
  ripple_detection 1.7, which Spyglass's stored events used, applied the 15 ms minimum to
  the whole event instead.
- **The settings** are R3's pinned `DETECTOR_SETTINGS["kay"]`: z threshold 2.0, minimum
  duration 0.015 s, smoothing sigma 0.004 s, close-ripple threshold 0, and a speed
  threshold of 4 cm/s at the event's endpoints. They are on the stored CA1 ripple band,
  with the head speed and time span that `load_data` uses.
- **The inventory.** Run `50a98592acee` (2026-10-05, commit `50a98592acee`): 216 epochs
  `ok`, the 3 known no-input epochs skipped, and 2 `failed`
  (j1620210710 `02_r1` and `04_r2`). Those 2 failed because Spyglass's own Kay inputs
  could not be fetched for the version check (a lab-database inconsistency, "Found 2
  potential parents"). The source refuses them until R3 reruns them. R3 will rerun with
  Task 16's artifact mask under a new run id. `SWR_INVENTORY_RUN_ID` is the one value to
  change then.

## Why: the 2.0 inventory against the stored events

Source: `.claude/research/refresh_r1/kay_version_ratio.csv`. It compares 216 epochs, using
the version check's strict positive-duration overlap.

| | stored (1.7, Spyglass) | direct (2.0) |
|---|---|---|
| events, all | 38,307 | 38,879 (ratio 1.015) |
| events, speed-eligible | 38,042 | 38,564 (ratio 1.014) |
| stored events matched by a 2.0 event | 92.8 % (2,761 missed) | 3,499 added |
| per-epoch ratio, median (IQR) | | 1.04 (0.99-1.16) |
| epochs outside [0.6, 1.3] | | 16 (14 low, 2 high) |

Pooled ratio (direct / stored) by animal:

| animal | ratio |
|---|---|
| chimi | 0.88 |
| j16 | 1.20 |
| peanut | 1.03 |
| senor | 0.88 |
| wilbur | 1.06 |

What the comparison shows:

- **The two agree closely in bulk.**
- **The stored events' inputs are unknown.** On today's Spyglass inputs, ripple_detection
  1.5.1, 1.7.0, 1.7.1 and 2.0 give identical counts in the outlier epochs. So the stored
  events came from other, older inputs (`.claude/research/refresh_artifacts/`).
- **Artifacts contaminate the stored events.** 654 of 38,824 stored events (1.7 %) lie
  inside artifacts.
- **Artifacts explain the low epochs.** In those 14 epochs (senor, chimi) artifacts
  inflate the Kay consensus trace's SD, from 1.54 to 10.5 times its robust SD against
  1.145 typically. Fewer events then cross z = 2. Task 16's mask (left out of the
  normalization) is expected to restore them: masked over stored, the median is 1.14.
  Until then those epochs have too few SWRs. See the pilot below.

## Interface

`analysis/hippocampus/swr_source.py`:

- **`SWR_INVENTORY_RUN_ID`, `SWR_DETECTOR` (`"kay"`), `SWR_INVENTORY_ROOT`.** These are
  the only definitions. The SWR runner writes under the same `SWR_INVENTORY_ROOT`.
- **`ARTIFACT_INTERVALS_RUN_ID`, `ARTIFACT_INTERVALS_ROOT`.** Task 16's hook, configured
  once like the run id. The run id is None today: no artifact is masked.
  `artifact_intervals(epoch_id)` resolves an epoch's intervals from
  `ARTIFACT_INTERVALS_ROOT / <run id> / {epoch_id}_artifact_intervals.parquet`
  (`start_time`, `end_time`). It returns none while the run id is unset, and raises when
  a configured run has no file for the epoch. No caller passes artifact intervals: every
  function below resolves them by epoch id, so every caller gets the same ones.
- **`swr_source_name()`** is the source's stamp:
  `<detector>@<inventory run id>+artifacts@<artifact run id or none>`, today
  `kay@50a98592acee+artifacts@none`.
  - Every canonical table built from these labels carries it in its `attrs`, kept in the
    parquet metadata (`stamp_swr_source`). That means event, event-arm, event-patch,
    event-sequence, stop, event-onset and stop-timebin tables, Replay and ContFrag alike,
    and the Widloski events table. Only qvals does not.
  - Its producer stamps it after reading its decode frames through the source.
  - `publish_canonical` refuses such a table without the current stamp.
  - The shard assemblers and the ContFrag covariate refresh refuse an input stamped with
    another source.
  - `record-tables` (Task 10) records the tables' stamp as `inputs["swr_source"]`, and the
    audit's `tables_recorded` requires it to be the configured source.
  - The release manifest (`write_regeneration_manifest.py`) checks every SWR-bearing
    candidate table and records the stamp.
- **`require_swr_source(frame, what=...)`** refuses a table stamped with another source.
  Every reader that combines a shipped table's SWR columns with the source's live labels
  calls it:
  - the PFC path's `load_session_canonical_events` (its `is_swr` against
    `load_session_swr_intervals`);
  - `regenerate_stop_table.py`, which splits rebuilt stops on the shipped event table's
    `is_swr`;
  - `analyze_content_occupancy.py`'s arm occupancy, reported beside its live SWR split.
- **`swr_source_files(epoch_id)`.** It serves an epoch only if the manifest status is
  `ok`, or `partial` with the Kay inventory written. Otherwise it raises `SwrSourceError`
  (a `ValueError`; the table builders' per-session failure tallies collect it). It checks
  the events and observed files against the sha256 the manifest recorded, and the
  manifest against its folder's run id.
- **`swr_events(epoch_id, *, speed)`.** It returns every inventory event with its own
  `event_number` as `SWR_event` and a `drop_reason`. The reason is None for a kept event.
  Otherwise it is the first that holds of:
  - `artifact`: the event's closed interval shares any time with one of the epoch's
    `artifact_intervals`.
  - `no_speed_sample`;
  - `moving`;
  - `speed_unknown`.

  The kept events pass the canonical whole-event immobility rule
  (`event_immobility.filter_event_intervals_by_speed`): every head-speed sample of the
  decode grid in `[start, end]` is finite and below 4 cm/s. The non-local and MUA cohorts
  use the same rule. The speed is the decode grid's `head_speed`. The inventory's own speed
  summaries are on LFP-interpolated samples, so they are not the canonical rule's samples.
- **`swr_event_labels(time, epoch_id, *, speed)`.** It returns the
  per-bin labels. The convention is unchanged: `build_speed_eligible_event_cohort(...,
  event_col="SWR_event")` with source ids preserved, as `get_data_over_time` writes
  `SWR_event`. A bin gets an event's number when its time lies in the closed interval
  `[start_time, end_time]`. A dropped event leaves a gap in the numbering and zeros its
  bins. Two events sharing a bin raise `EventDefinitionError`, as the builders do.
- **`swr_observed_stretches(epoch_id)` and `swr_observed_mask(time, epoch_id)`.** These
  say where detection was observed: the gap-free stretches of the stored ripple band
  (`swr_observed.parquet`, scope `stored_band`), as closed intervals.
- **`with_swr_source(dot)` and `read_dot_with_swr_source(path)`.** They give one epoch's
  `data_over_time` for analysis. `SWR_event` holds the source's labels and `swr_observed`
  the coverage mask. The decode's stored-ripple counts (`n_SWR`, `n_non_local_SWR`,
  `n_non_local_no_SWR`, `n_SWR_MUA`, `n_SWR_no_MUA`) are dropped. The decode's own
  `SWR_event` stays in the pickle as provenance and never reaches an analysis. The epoch id
  comes from the frame's `animal`, `day` and `epoch`, so a ContFrag or cortical decode reads
  the same CA1 SWRs. The epoch's files are checked against the manifest once, and the frame
  carries the stamp.
- **`swr_intervals(dot, *, nwb_file_name=None, epoch_name=None)`** returns one epoch's SWR
  intervals from a source-read frame: the sampled bounds of its `SWR_event` labels, which
  the event table classifies against. It refuses a frame without `swr_observed`. The figures
  use it for their SWR shading and example selection.

## Who reads it

- **Readers of a decode frame** go through `read_dot_with_swr_source`:
  - the replay-content tables (`prepare_data._process_session`, and the stop-only
    bootstrap);
  - `build_timebin_tables.py`;
  - `regenerate_stop_table.py`;
  - the ContFrag tables (`prepare_contfrag_replay_content_data.py`,
    `refresh_contfrag_behavioral_covariates.py`);
  - the pooled loaders `load_data_over_time` and `load_contfrag_data_over_time` (Figures 2
    and 3, the decoder-validation supplement);
  - `run_quality_metrics.py`, `analyze_content_occupancy.py` and
    `analyze_event_spike_evidence.py`;
  - the Widloski runner. It also takes its SWR coverage from `swr_observed_stretches`
    instead of `SWR_zscore`'s finite stretches.
- **The PFC path** (`canonical_events.load_session_swr_intervals`) reads the source's
  labels on the cached frame's grid. Its provenance (`canonical_event_provenance`) now
  records `swr_interval_source` (`swr_source_name()`, such as
  `kay@<run id>+artifacts@none`) and the events file's path and sha256. The
  PFC runner's skip check compares both, so a bundle fit on the stored ripples is refit.
- **`quantify_swr_overlap_geometry.py`** classifies against the source's intervals.
- **Builders that take a frame** (`build_classified_nonlocal_event_table`,
  `build_stop_table`, the timebin and ContFrag builders, `quality_metrics`, the figure-panel
  helpers on `load_data_over_time`) are unchanged. They read whatever `SWR_event` the
  caller's frame holds, which is now the source's.
- **Figures 2 and 3 and the event gallery** shade, select and check their SWRs with
  `swr_intervals` on the source-read frame. They never gate `load_data`'s stored
  `ripple_times`. `assert_drawn_class_matches_ripples` checks the drawn spans against
  those classes. Figure 2's render cache is keyed by `swr_source_name()`.
- **`plot_session_tables`**'s `swr_lfp` panel shades the `swr_intervals=` the caller
  passes and refuses to draw without them.

`tests/unit/test_swr_source_consumers.py` runs each reader on a frame whose stored labels
disagree with the source, under a configured artifact run that masks one source SWR. Two
static checks there keep it that way: no module outside the figure archive and an
explained list names the stored ripples, and none unpickles a decode frame without the
source.

## What does not change here

- **The decoders' `no_ripple` encoding-training mask** still excludes the stored
  `RippleTimesV1` ripples. That is a separate decision. The two definitions overlap
  93 %, and the decodes were kept.
- **`get_data_over_time` still writes the stored ripples' `SWR_event`.** It is the decode's
  own provenance.
- **`SWR_zscore`** is still Spyglass's Kay consensus trace, a signal and not a label. The
  inventory ships no trace.
- **The pipeline-data-flow supplement** unpickles a decode frame directly, but it reads
  no SWR column.
- **Conventions.**
  - Event-level `is_swr` is strict positive-duration overlap.
  - Stop-level `is_swr_stop` is the inclusive per-timebin rule.
  - Onset tables count onsets in the window.

  See [../known_issues/swr_mua_count_conventions.md](../known_issues/swr_mua_count_conventions.md).

## Pilot (read-only, 5 epochs)

Old (stored labels in the refresh decodes' `data_over_time`) against new (this source)
`is_swr` for the immobile non-local events of `build_classified_nonlocal_event_table`:

| epoch | non-local events | `is_swr` old | `is_swr` new | old → non-SWR | non-SWR → new | SWRs old | SWRs new |
|---|---|---|---|---|---|---|---|
| chimi20200225_02_r1 | 264 | 102 (38.6 %) | 15 (5.7 %) | 87 | 0 | 141 | 24 |
| j1620210708_08_r4 | 152 | 75 (49.3 %) | 82 (53.9 %) | 0 | 7 | 221 | 270 |
| peanut20201202_06_r3 | 364 | 89 (24.5 %) | 95 (26.1 %) | 0 | 6 | 135 | 144 |
| senor20201027_02_r1 | 487 | 267 (54.8 %) | 270 (55.4 %) | 3 | 6 | 273 | 277 |
| wilbur20210403_06_r3 | 126 | 49 (38.9 %) | 50 (39.7 %) | 0 | 1 | 80 | 85 |

What the pilot shows:

- **chimi20200225_02_r1 is an artifact epoch.** It is the R3 smoke epoch, with a 0.17
  direct/stored ratio. In it most of the SWR-coincident events lose their SWR until Task
  16 masks the artifacts out of the normalization.
- **Elsewhere the change is small,** and mostly gains.
- **Drops.** One inventory event (senor20201027_02_r1) was dropped as `moving`.
- **Coverage is complete.** The stored band covers at least 99.97 % of each decode grid,
  and no non-local event touches an unobserved bin.

Script and output: `task15_pilot.py` and `task15_pilot.csv` in the session scratchpad.
