# Full pipeline refresh: re-decode, comparison inventories, downstream regeneration

**Status (2026-10-03):** design agreed in conversation; this spec is under review. The
implementation plan will live at `.claude/docs/plans/full-pipeline-refresh/PLAN.md`.

## Purpose

Rebuild every long-running computation in the project on one set of inputs, so that the
replay-detection comparison's inventories, the canonical tables, and every fitted model,
PFC bundle and figure can be shown to rest on the same data.

The canonical HPC decodes were made on 2026-04-01 at `0820d39d`. Since then the spike
loader, the position loader, the decoding model and two libraries have changed (see
"What the re-decode changes"), so no current result can be reproduced from current code.
After this work, nothing canonical rests on the April decodes.

### End state

1. **Decodes.** Canonical NonLocal and ContFrag HPC decodes exist for every epoch with
   inputs (218 expected). They come from one commit and one locked environment that pins
   `ripple_detection` at `68e1784`. The April decodes sit in
   `Processed-Data/_pre_redecode_backup_YYYYMMDD/`.
2. **Widloski.** The Widloski & Foster detector has run with its 100-draw shuffle on every
   canonical decode.
3. **SWR inventories.** Kay, Karlsson, Yu and Zugaro inventories exist for every epoch with
   LFP, all from that package commit, plus the Kay version check against the canonical
   `SWR_event`.
4. **Everything downstream.** Rebuilt from item 1:
   - the canonical table set, the timebin tables, the ContFrag tables and `qvals_table`;
   - every rate, content, estimator and PFC model, refit;
   - every decode-dependent figure and generated doc.
5. **Proof.** An audit shows that items 1–4 share inputs, and a before/after report shows
   what the refresh changed.

Dates written `YYYYMMDD` in paths below are the day that stage runs.

### Not in scope

- **Phase 3 of the comparison.** Its adapters and report are the next step; this work
  produces their inputs.
- **Phase 2's deferred pilot checks** (null validation and the two sensitivity runs). They
  rerun on the new decodes afterwards.
- **Spyglass integration** of the new detectors (Phase 7).
- **Requoting hand-maintained result docs.** The before/after report lists every claim that
  needs a new number; requoting follows the project lead's review of that report.
- **Artifacts owned by another branch or worktree.** `Processed-Data/covariate_profiles` and
  `pfc_marked_model_family/` are listed as stale, not rebuilt.

## Decisions (2026-10-03)

| # | Decision | Why |
|---|---|---|
| R1 | Re-decode on current code as it stands; replace the canonical decodes after backing them up | The fixes since April are wanted. A spikes-only re-decode could not reproduce April's model on a library 79 commits newer |
| R2 | Pin `ripple_detection` at `68e1784` before any decode | The decode's MUA burst events and SWR z-score then come from the same package version as the four SWR inventories |
| R3 | Zugaro reads the same CA1 electrodes and references as Kay, Karlsson and Yu | The electrode set is shared input; only the band and filter differ between the four |
| R4 | Direct Kay at `68e1784` is the comparison's Kay inventory. The canonical `SWR_event` stays as an extra SWR definition and as the version check | Kay then matches the other three in package, band, electrodes and speed |
| R5 | The Widloski runner reloads marks through the project's loader; the decoder records nothing extra | The project lead's choice. The loader is the code that fed the decode, and the audit confirms one commit (Stage 8). The guard's limits are stated under Stage 4 |
| R6 | Re-decode all 218 epochs, including the three once blocked by an NWB checksum | All three load fully as of 2026-10-03. An epoch that still fails has its April files moved to the backup and is listed, so the canonical folder holds one decode version |
| R7 | Rebuild the ContFrag decodes and everything downstream, including model refits and figures | Nothing is left on April data, so every comparison and every quoted number shares inputs |
| R8 | A lean implementation plan: code tasks with acceptance criteria, plus a runbook per long run | The runs span days and several sessions |

## What the re-decode changes

Measured on 2026-10-03 from `chimi20200225_02_r1_HPC_detector.pkl`, the git history since
`0820d39d`, and `load_data` on three sessions.

| Input | April (`0820d39d`) | Now |
|---|---|---|
| HPC marks: sort groups | every `UnitMarks` sort group | tetrodes only (`f172084`). wilbur20210331, for one, loses probe sort groups 17–19, which have 32 channels each |
| HPC marks: coincident spikes | each cluster's leading spike removed but not counted | counted (`64bf585`). 259 more of about 358,000 marks are dropped on chimi20200225_02_r1 |
| Position | untrimmed DLC grid | trimmed to the valid interval (`e5a0def`); up to about 76 s shorter at an epoch's end |
| `non_local_detector` | `5e3a086` | `2d730c3` |
| Local state | exact position | Gaussian, `local_position_std` 2.5 cm (`ded292e`) |
| Local and No-Spike stickiness | 1e6 | 1e5 |
| Non-Local Continuous self-transition prior | 0.98 | 0.96 |
| No-Spike transition row during fitting | updated | frozen |
| `results.nc` contents | posterior | posterior and the per-bin `log_likelihood` (added 2026-06-26) |
| `ripple_detection` | 1.7.0 | `68e1784`: the changelog's 2.0.0 (2026-09-22), never tagged, so `git describe` gives `v1.7.1-365-g68e1784`. This affects MUA burst events and the SWR consensus z-score |

Every non-local event changes, and so does every `event_id`. Exact-parity gates against
earlier tables therefore cannot pass. Stage 8's audit replaces them.

## Stage order

| Stage | What | Needs | Runs on |
|---|---|---|---|
| 0 | Environment pin, version-shift report, golden baseline | — | CPU |
| 1 | NonLocal and ContFrag decodes, into staging | 0 | GPU |
| 2 | Swap staged decodes into the canonical names | 1 | file moves |
| 3 | SWR inventories | 0 | CPU |
| 4 | Widloski full run | 2 | GPU |
| 5 | Tables, built in staging, then promoted | 2 | CPU |
| 6 | Model refits and PFC analyses | 5 | CPU; PFC unit GLM on GPU |
| 7 | Figures and generated docs | 2 for decode-only figures; 6 for the rest | CPU |
| 8 | Same-data audit and before/after report | 3–7 | CPU |

**Go-ahead points, where work stops for the project lead's approval:**

1. launching the Stage 1 full batch, after the smoke and pilot numbers are in;
2. the Stage 2 swap;
3. promoting the Stage 5 tables;
4. the Stage 6 PFC launch;
5. any requote of hand-maintained docs.

**Critical path:**

| Step | Estimate |
|---|---|
| Decodes | 1–1.5 days on 3–5 free cards |
| Swap | — |
| Tables | about 3 h |
| PFC unit GLM | about 17 h on 5 cards, while the CPU fits and Widloski run alongside |
| Figures | about 5 h |
| Audit | — |

In all, about 3–4 days of wall clock, set mostly by how many cards are free. Every estimate
is re-measured on a smoke run before its stage is launched.

## Stage 0: environment and baseline

- **Pin.** Add
  `ripple_detection @ git+https://github.com/Eden-Kramer-Lab/ripple_detection.git@68e1784b183f161a568d3154325cf770730bfb83`
  with `uv add --no-sync`, then run `uv sync --inexact`. The CUDA plugin
  (`jax-cuda12-plugin[with-cuda]==0.8.0`) was installed by hand and is not in `uv.lock`
  (`docs/README.md`, "Where results live"), so an exact sync would remove it. Afterwards,
  confirm that `jax.devices()` lists the GPUs.
- **Smoke under the pin.** Run `get_ripple` (Spyglass's `RippleTimesV1` read and
  `get_Kay_ripple_consensus_trace`) and `get_multiunit_hse` on one epoch. Spyglass imports
  only `Kay_ripple_detector`, `Karlsson_ripple_detector`, `core.gaussian_smooth` and
  `core.get_envelope` from the package, and all four are still exported at `68e1784`.
- **Version-shift report.** On the five pilot epochs, measure under 1.7.0 and again under
  `68e1784`:
  - MUA burst events, each matched, missed or added by overlap;
  - the largest difference in the SWR consensus z-score.
- **Golden baseline.** Freeze the current tables and models read-only with
  `scripts/qa/freeze_regression_baseline.py --label pre_redecode_YYYYMMDD`, for the
  before/after report.
- **Provenance from here on.** Every manifest records the commit, a hash of `uv.lock`, the
  installed CUDA plugin version, and the `ripple_detection` commit. The package's installed
  version string comes from its last tag (1.7.x) and does not identify the commit.

## Stage 1: decodes

Both decoders call `load_data` and fit their own model, and neither reads the other's
output (`run_decoding_contfrag.py:50-71, 300-345`). They run in parallel on disjoint
GPUs, from one commit.

- **Staging.**
  - Add a `--save_dir` option to both decoders through `decoding_common.add_common_args`,
    defaulting to `PROCESSED_DATA_DIR`, which is hard-coded today (`parameters.py:79-81`).
  - Decodes write to `Processed-Data/_staging_redecode_YYYYMMDD/`, so the canonical folder
    never holds a mix of versions.
  - Without staging, an existing result is silently reused unless `--overwrite` is given
    (`run_decoding_bandit.py:346-355`).
- **Logs.** `run_batch` writes `logs/{epoch_identifier}.log` in mode `w`. The April logs
  are the only per-epoch record of the April commit, so they are copied into the backup
  before any launch.
- **Settings.**
  - NonLocal uses the canonical settings: HPC, `no_ripple` training (masked by canonical
    Kay), stationary transitions, non-local threshold 0.8, no figurl.
  - ContFrag uses its canonical settings, as run in
    `logs/contfrag_pipeline_20260427_072127.log`.
- **GPUs.** Pass `--cuda_visible_devices` explicitly, naming only cards idle at launch. The
  decoders claim 99% of a card's memory and the host is shared; the default is cards 0–9.
- **Staged runs.**
  1. One epoch of each decoder, recording wall clock, GPU memory, peak RSS, and the size of
     `results.nc` now that it holds `log_likelihood`.
  2. The five pilot epochs.
  3. The rest.

  April's runs give the scale: 213 NonLocal epochs took about 9.5 h on 10 cards, and 221
  ContFrag epochs took 5 h 51 m. The tmux sessions are `redecode` and
  `redecode_contfrag`.
- **Gates before the swap.**
  1. Every epoch has its four files or a recorded failure.
  2. Every `data_over_time` grid passes `validate_time_grid`.
  3. Per epoch, the NonLocal and ContFrag `data_over_time` share their time grid and
     position exactly, since both read one `load_data`. This matters downstream: the
     ContFrag tables join `qvals_table` by a recomputed `stop_id` with no clock guard
     (`analysis/contfrag/tables.py:669-676`).
  4. A table of non-local events per epoch, old against new, with outliers listed for
     review.

## Stage 2: swap

This needs the project lead's go-ahead, with no other session reading the decode files.

1. Move the April NonLocal and ContFrag decode files into
   `Processed-Data/_pre_redecode_backup_YYYYMMDD/`, with a sha256 manifest. That is eight
   files per epoch, about 590 GB.
2. Move the staged files into the canonical names.
3. Check counts and hashes.

The swap can be reversed until Stage 5 promotes tables.

## Stage 3: SWR inventories

Can start after Stage 0. It reads LFP and position, not the decodes. Each detector runs with
its own settings; what is shared is the input: electrodes, references, sampling rate,
speed trace and time span.

### Inputs: one read-only fetch per epoch

- **Stored CA1 ripple band.** Read from `LFPBandV1` (`ripple_150_250`, 1000 Hz) with its
  electrodes and references. The runner asserts the electrodes equal `RippleLFPSelection`'s
  CA1 set, the check spec 01 asks for.
- **Raw LFP behind that band.** Read from the same `LFPOutput` entry.
  - Its columns are positions, not electrode ids (24 raw columns against 22 band columns on
    chimi20200225_02_r1), so each column is mapped to its electrode through the LFP's
    electrode table, as Spyglass does.
  - Each channel then has its reference subtracted, in float64.
  - These referenced channels feed Zugaro.
- **Input check.** Spyglass's own FIR filter (`FirFilterParameters`, `ripple_150_250` at
  1000 Hz: 535 taps, band edges 140/150/250/260 Hz) is applied through
  `FirFilterParameters().filter_data` to the referenced LFP, as `LFPBandV1.make` does. The
  result must reproduce the stored band to the stored precision. This shows Zugaro reads the
  same channels and referencing as the other three. The largest difference goes in the
  manifest, and a mismatch stops that epoch.
- **Speed and span.** `head_speed` comes from `get_position_info`, the source and span the
  new decode uses, linearly interpolated onto LFP times. The LFP is restricted to the
  position span, as `filter_ripple_lfp_time` does.
- **Rate.** The nominal 1000 Hz is passed to every detector and to the Zugaro filter design,
  as Spyglass passes it to canonical Kay. The measured rate (1000.07 Hz on chimi) goes in
  the manifest; it shifts Zugaro's band by about 0.01 Hz. The LFP's anti-aliasing filter is
  recorded too.

### The four detectors, package at `68e1784`

| Detector | Input | Settings |
|---|---|---|
| Kay | stored 150–250 Hz band | Spyglass's canonical settings, equal to the package defaults: z ≥ 2 for 15 ms, 4 ms Gaussian smoothing, z-score over all valid samples, no close-event exclusion |
| Karlsson | stored band | package defaults: z ≥ 3 on any tetrode for 15 ms, 4 ms smoothing, z-score over all valid samples |
| Yu | stored band | spec 01, step 4: threshold at the 99.99th percentile of the mirrored immobility noise, 20 ms above it, 4 ms smoothing, per-tetrode z-score (`zscore_per_channel=True`). Threshold diagnostics go in the manifest |
| Zugaro | raw referenced LFP, filtered as below | squared and summed over channels; moving average of 11/1250 s (9 samples at 1000 Hz); z-score over all valid samples, which excludes the stretch edges below; boundaries strictly above 2 SD; peak strictly above 5 SD; events closer than 50 ms merged, with no length cap; no minimum duration. Events longer than 150 ms are then dropped (kept at exactly 150 ms) |

**Speed rule.** All four run with `speed_threshold = 4.0`, Zugaro included: an event is
kept only if speed at its first and last sample is at most 4 cm/s, and unknown speed fails.
Phase 3 then applies the shared rule (overview X4) to every inventory on one speed trace,
ours and Widloski's included: speed finite and below 4 cm/s for the whole event.

### The Zugaro filter

**Lineage and settings.** There is no single standard, but there is one lineage. Nearly
every Buzsáki-lab ripple inventory comes from the `FindRipples` two-threshold rule. It
descends from Csicsvari et al. 1999, was written up in FMAToolbox by Hirase and Zugaro, and
was carried into buzcode `bz_FindRipples`, then neurocode and CellExplorer; the package
implements it as `Zugaro_ripple_detector`. The settings differ by repository and by who ran
it (compiled by the project lead, 2026-10-03):

| Source | Thresholds (SD) | Merge gap / max (ms) | Min (ms) | Band (Hz) | Filter |
|---|---|---|---|---|---|
| buzcode `bz_FindRipples` defaults (`0969ddf`) | 2 / 5 | 30 / 100 | 20 | 130–200 | Butterworth order 3, zero-phase |
| **Buzsáki-lab Databank files: Petersen MS10, Girardeau Rat09, McKenzie PV8** | **2 / 5** | **50 / 150** | **none stored** | **120–180** | **Butterworth order 3** (Petersen's copy: Chebyshev order 4) |
| CellExplorer `ce_FindRipples` defaults | 2 / 5 | 20 / 150 | 20 | 130–200 | Chebyshev order 4, zero-phase |
| neurocode `FindRipples` defaults (`d166a67`, Oliva lab) | 0.5 / 2.5 | 50 / 500 | 25 | 80–250 | Butterworth order 3 |
| Harvey 2023's released callers | 1 / 3 | 50 / 300 | 20 | 100–250 | Butterworth order 3 |

This comparison runs the Databank files' settings, the row in bold. Every row shares the
core steps:

1. filter the ripple band with a zero-phase filter;
2. square, and average over 11 samples at 1250 Hz;
3. z-score over the whole recording or a chosen interval;
4. bound each event at the low-threshold crossings, and require its peak to exceed the high
   threshold;
5. merge events closer than the gap. The buzcode lineage merges with no cap, then drops
   events over the maximum.

An optional sixth step rejects events by an EMG signal and a noise channel. The package
implements neither, so neither runs here.

**Where this run departs from the lineage,** each recorded in the manifest:

- **Channels.** All CA1 channels are squared and summed (R3), where the lineage's callers
  usually pass one pyramidal-layer channel.
- **Speed.** The shared 4 cm/s endpoint rule applies. The lineage has no speed criterion.
- **Package conventions:**
  - the peak is the maximum of the normalized power;
  - an event cut off by a gap or the recording edge is kept and flagged;
  - the moving average is set in seconds, 11/1250 s, which is 9 samples or 9 ms at 1000 Hz
    against 8.8 ms at 1250 Hz;
  - duration limits are counted in samples.

- **Design rate.** The filter is designed at the LFP's 1000 Hz. The 1250 Hz in the original
  snippet is buzcode's LFP rate; the same coefficients run on 1000 Hz data would pass
  96–144 Hz instead.
- **Numerical form.** Second-order sections, `butter(..., output="sos")` with
  `sosfiltfilt`. This is the same filter as the transfer-function form: the single-pass
  responses agree to 3e-14, and outputs on white noise to 4e-14.
- **Band, as the zero-phase response.** Forward-backward filtering squares the single-pass
  gain, so the design edges sit at −6 dB, not −3 dB. This is the method's own filter, so it
  is not redesigned; the tests pin these gains.

  | Frequency | Gain |
  |---|---|
  | 120 and 180 Hz | −6 dB |
  | 123.6–175.4 Hz | above −3 dB |
  | 150 Hz | 1.00 |
  | 110 Hz | −19 dB |
  | 190 Hz | −15 dB |
  | 100 Hz | −33 dB |
  | 200 Hz | −25 dB |
  | 250 Hz | −60 dB |
  | 60 Hz | −81 dB |
- **Gaps and edges.**
  - Each gap-free stretch is filtered on its own, in float64.
  - The single-pass impulse response decays below 1e-3 of its peak within 86 ms, and the
    zero-phase one within ±68 ms. SciPy's default padding is only 21 samples, so the padding
    is set to 90 samples with odd extension.
  - The first and last 90 ms of each stretch count as unobserved: they are left out of
    Zugaro's z-score and its events, recorded, and applied to every detector through
    shared exposure (overview X11).
  - A stretch shorter than the padding counts as unobserved.
- **Checks on recorded data.** On the pilot epochs, spectra of the raw and filtered channels
  report:
  - the measured gain at 120, 150 and 180 Hz;
  - the share of filtered power that lies within 120–180 Hz;
  - the power of the 60 Hz harmonics at 120 and 180 Hz. They sit on the band edges and pass
    at half amplitude. No notch filter is added, since the method has none.

### Outputs

Under `Processed-Data/replay_detection_comparison/swr/<run_id>/`, with the run id taken
from the commit:

- **Per epoch and detector,** the package's own event table with all its statistics.
- **Per epoch,** where the LFP was observed, after edge exclusion.
- **Per epoch,** the Kay version check: each direct-Kay and canonical `SWR_event` event
  marked matched, missed or added by positive-duration overlap (the project's `is_swr`
  rule), with the boundary differences of the matched
  ones.
- **`manifest.json`**, with:
  - the package commit, every parameter, and the Zugaro filter design;
  - the electrodes and references, and the input-check result;
  - Yu's threshold diagnostics;
  - `meets_published_tetrode_criterion` (true, false or unknown);
  - input hashes and library versions.

**Epochs with no inputs** are recorded as skipped with that reason: senor20201104 06_r3 and
08_r4 have no ripple band, and wilbur20210404 has no CA1 ripple extraction. None of them has
a decode either.

### Code

- **Input fetch:** `data_loaders/lfp.py`, sharing `get_ripple`'s restriction keys through
  one helper rather than copying them.
- **Detection:** a new `analysis/hippocampus/swr_inventories.py`, holding the pinned
  settings, the Zugaro filtering, the input check and the Kay check.
- **Runner:** a new `scripts/decoding/run_swr_detection.py`.
  - It takes the decoder's options: `--animal`, `--date`, `--epoch`, `--overwrite`.
  - It runs on a dask `LocalCluster` on CPU, with one log per epoch.
  - Each epoch's failure is recorded and the run continues.

### Tests

- **Zugaro filter:**
  - the pinned gains above, at a design rate of 1000 Hz;
  - it never filters across a gap;
  - padding and edge exclusion behave as stated;
  - a stretch too short to filter is unobserved;
  - the 150 ms cut keeps an event of exactly 150 ms.
- **Other units:**
  - the electrode assertion;
  - the pinned settings for all four detectors;
  - the Kay matching on made-up intervals.
- **End to end:** the package's simulated session at 1000 Hz runs through the wrapper, and
  each detector finds the injected ripples.
- **Smoke:** the five pilot epochs, measuring runtime and producing the Kay check and the
  filter checks above.

## Stage 4: Widloski full run

The runner `scripts/decoding/run_memoryless_detection.py` is extended rather than replaced.
It already has epoch discovery, resume and retry, a per-run manifest that refuses mixed
settings, and the outputs the Phase 3 adapter reads.

- **Marks.** The runner calls the two functions `load_data` uses: `get_hpc_marks`
  (tetrode-only, current coincident-spike rule) and `filter_spike_times` over the decode's
  time span.
  - **What the guard misses.** The existing guard (`check_marks_match_encoding`) confirms
    only that the training-span marks are present. It cannot see spikes the reload adds, or
    spikes removed only during SWRs, which `no_ripple` training leaves out. Stage 8's
    one-commit check is what covers that gap.
- **Cache.** The marks cache is keyed by epoch alone today, and the April-rule caches keep
  more spikes than the new rule. A stale one would pass the guard while decoding extra
  spikes. The cache key therefore gains the rule's name and a hash of the decode's
  `_detector.pkl`.
- **The April rule retires.** `decode_marks.py` and its tests are deleted. The pilot outputs
  record which rule they used, and git keeps the code.
- **Parallel, like the decoder.**
  - New options: `--cuda_visible_devices`, `--local_scratch_dir`, and `--date` and `--epoch`
    filters beside `--animal`, `--all` and `--epoch-id`.
  - A dask `LocalCUDACluster` runs one epoch per task.
  - Only the main process writes `manifest.json`, updating it as each epoch finishes.
  - Each epoch gets its own log, `logs/{epoch_id}_memoryless.log`.
- **Full run.** `--all`, 100 shuffles, seed 20260922, run id from the commit and seed, in
  tmux session `memoryless_full`. The pilot took about 5 min per epoch with the shuffle at
  7–11 GB peak RSS: about 18 GPU-hours, or 5–6 h on 3–4 cards. This is re-measured on the
  first new-decode epoch.
- **Tests.**
  - A cache made for another decode is refused.
  - The parallel driver records a failed epoch and carries on.
  - A smoke run on one new-decode epoch, compared with its pilot row counts.

## Stage 5: tables

Everything is built into one staging directory,
`Processed-Data/replay_content/_staging_redecode_tables_YYYYMMDD/`, in this order. The
behavioral cache `replay_content/_session_cache/comrie_photogate_trials_v4` (218 entries)
does not depend on the decoder and is reused (`prepare_data.py:230-233`).

1. **Stop table bootstrap.**
   `prepare_replay_content_data.py --stop-only --from-cache --output-dir <staging>`.
   About 6 min serially.
2. **`qvals_table`.** `build_qvals_table.py --replay-content-dir <staging> --write`, about
   12 min; it reads the live `alison_*` DataJoint schemas. It must follow the stop table:
   built before it once, it left a whole session missing
   (`docs/audits/2026-08-25-covariate-regen-staging-audit.md`).
3. **Replay tables.**
   `prepare_replay_content_data.py --from-cache --output-dir <staging> --behavior-cache-dir replay_content/_session_cache --contingency-dir <staging>`.
   - Runtime: about 1.5 h serially, or about 36 min as five per-animal shards followed by
     `assemble_replay_content_shards.py`.
   - Built-in gates:
     - it fails closed on any failed session;
     - Σ`n_selected_events` must equal the event count;
     - the `qvals` clock and value-block checks run.
4. **Timebin tables.** `build_timebin_tables.py`, about 69 min; it reads pump events
   through live `load_data`. It gains an `--output-dir` option, since today it writes into
   `replay_content/` directly. These tables must be rebuilt: the last regeneration carried
   them forward, which is invalid once stop ids change.
5. **ContFrag tables.**
   `prepare_contfrag_replay_content_data.py --event-source SWR` and `--event-source MUA`,
   each with `--from-cache --output-dir <staging>`. The two run concurrently, about 2.5 h
   each.
6. **Regeneration manifest.** `write_regeneration_manifest.py`.
   - It expects `models_validation/` bundles and a clean tree, and the command that produced
     those bundles was never recorded. The plan's first task for this stage reads the script
     and settles how to run it.
   - `compare_to_golden_baseline.py` runs as a drift report, since it raises on key drift.
7. **Promote (go-ahead).**
   1. Back up the current tables to
      `replay_content/_pre_redecode_backup_YYYYMMDD/`.
   2. Copy the staged tables in with `cp -p`.

**Shortcuts not to use, because each is invalid after a re-decode:**

- `regenerate_stop_table.py` assumes the stop bouts are unchanged.
- `refresh_contfrag_behavioral_covariates.py` copies the old posterior aggregates.
- `scripts/exploratory/_augment_data_over_time.py` rewrites the `data_over_time` pickles.
- `python -m …prepare_data`: the module has had no `__main__` since `0e6c4c7`, so the
  command now does nothing.

## Stage 6: model refits and PFC analyses

Runtimes below are from the September logs, and every refit is re-measured.

### CPU, in parallel where independent

**Rate**

| Fit | Runtime |
|---|---|
| `fit_per_bin_rate_models.py --stop-cohort at_well --workers 16` | 41 min |
| same, `all_stops` | 57 min |
| same, `in_transit` | 5 min |
| `fit_per_stop_rate_models.py --stop-cohort at_well` | 2 min |
| `fit_swr_contrast_rate.py`, `at_well` (then `--aggregate-only`) | 20 min |
| same, `all_stops` | 27 min |
| `fit_uncertainty_rate.py` | 14 min |
| `fit_swr_power_model.py` | not recorded |
| `fit_cross_history_rate.py` | not recorded |

**Content**

| Fit | Runtime |
|---|---|
| `fit_replay_content_models.py` ladders: arm, `--hierarchy patch`, `--swr-subset T` and `F`, the leaf chain `--leaf-only --models M0,M0a,M1,M2,M3,M4,Mv,Mu,Mvu`, and `--estimator dirichlet_multinomial` | 1–11 min each |
| `fit_swr_contrast_content.py --outcome-stratum all` | 3.3 h |
| same, `rewarded` | 3.0 h |
| same, `omitted` | 9 min |
| the Dirichlet-multinomial variants of the content contrasts | not re-measured; checked before launch |
| `analyze_uncertainty_content.py`, after the leaf chain | not recorded |
| `compare_content_estimators.py --n-jobs 16` | 26–49 min |
| `compare_event_source_models.py`, plus the ContFrag content fits (`--event-source SWR` and `MUA`, `--brain-area HPC`) | short |

**PFC staged analyses**

| Fit | Runtime |
|---|---|
| Stage 5B, `run_stage5b_retrospective.py` | about 20 min per variant |
| Stage 6, `run_stage6_learning.py` | about 3 min |

### GPU, needs a go-ahead

The mPFC unit GLM covers 71 sessions, at σ 3, K 5, R 1,000, on the at-well cohort.

- It runs one process per card (`--n_workers 1 --gpu N --sessions <partition file>`),
  modeled on `logs/pfc_ridge_heldout_20260916/launch.sh`.
- It takes about 14–17 h per card on 5 A100s.
- The existing mPFC bundles move to the backup first, since the launcher refuses to start
  over them.
- It runs on cards that don't overlap Stage 4's.

### Gate

`check_fit_provenance.py` must exit 0 at the end of the stage.

## Stage 7: figures and generated docs

Figures are regenerated by running their scripts. No figure script is edited; the rule
against editing the Figure 1–4 scripts stands.

- **After Stage 2, from the decodes alone** (per the dependency map; the plan confirms
  each script reads no table before scheduling it this early):
  - Figure 2 with `--force-recompute` (about 16 min); its joblib cache would otherwise
    serve April data;
  - Figure 3;
  - `generate_supp_decoder_validation.py` and `generate_supp_pipeline_data_flow.py`;
  - the event gallery (about 4 h at 6 jobs).
- **After Stage 6:**
  - Figure 4 (about 2 min);
  - the rate and content plots, `plot_swr_contrast_coefs.py`, and
    `generate_uncertainty_content_figure.py`;
  - `summarize_per_bin_rate_models.py`, which writes `docs/per_bin_rate_results.md`.
- **Unaffected:** Figure 1 and `generate_memoryless_comparison.py` use simulation only.

## Stage 8: same-data audit and before/after report

A new `scripts/qa/audit_refresh_consistency.py` writes `audit.json` beside the outputs,
summarized in `docs/audits/YYYY-MM-DD-full-pipeline-refresh.md`. It checks:

1. **Commit and environment.** One commit and one `uv.lock` hash across every decode log,
   or an explicit list of exceptions with reasons.
2. **NonLocal against ContFrag.** Per epoch, they share time grid and position (the
   Stage 1 gate, re-checked on the canonical files).
3. **Marks.** Per epoch, the marks reloaded through the loader contain every encoding mark
   of both decodes. This is the existing guard, now applied to ContFrag too.
4. **Widloski.** The run manifest's decode hashes equal the canonical decode files'.
5. **SWR inventories.** Per epoch, the runner's position span and speed source equal the
   decode's.
6. **Tables.** The regeneration manifest, or failing that, each table's sha256 with the
   commit and inputs that built it.
7. **Models.** `check_fit_provenance.py` exits 0, and the PFC bundles' sidecars name the
   new event table.
8. **Epoch set.** One set across all artifacts, with every absence explained.

The **before/after report** covers:

- the drift report from `compare_to_golden_baseline.py`, plus `compare_fits.py` and
  `compare_pfc_cohorts.py`;
- a table of headline numbers, old against new: the effects the living notes and project
  memory quote;
- the MUA version shift and the Kay version check;
- the probe marks removed per wilbur epoch;
- non-local events per epoch;
- the hand-maintained doc claims that need new numbers.

## Failure handling and rollback

- **Failures.** Every runner records an epoch's failure, carries on with the rest, and exits
  non-zero.
- **Unexpected failure modes.** One found in a smoke or pilot run stops that stage for
  diagnosis; it is not retried at scale.
- **Rollback.** Before Stage 5 promotes tables, the decodes can be restored from the backup.
  After promotion, the tables restore from the `replay_content` backup and the golden
  baseline.

## Code changes

1. `decoding_common.add_common_args` gains `--save_dir`, used by `run_decoding_bandit.py`
   and `run_decoding_contfrag.py`.
2. `build_timebin_tables.py` gains `--output-dir`.
3. `run_memoryless_detection.py`, as in Stage 4. `decode_marks.py` and its tests are
   removed.
4. `data_loaders/lfp.py` gains the ripple-input fetch, sharing `get_ripple`'s keys.
5. New `analysis/hippocampus/swr_inventories.py`.
6. New `scripts/decoding/run_swr_detection.py`.
7. New `scripts/qa/audit_refresh_consistency.py`.

Every change comes with the tests listed under its stage.

## Documentation

- **This spec.** Its status line tracks progress.
- **Comparison docs:**
  - `replay_detection_comparison/overview.md`: seven inventories, adding Zugaro, and decision
    D1 resolved for Zugaro;
  - `detector_comparison.md`: Zugaro's row;
  - spec 02: the runner's spike source and its parallel mode;
  - spec 01: the package's current parameter names (`zscore_per_channel`) and the direct
    path, now covering Kay and Zugaro.
- **The re-decode.** `.claude/docs/plans/replay_detection_comparison/PLAN.md` and a new
  `docs/known_issues/` note record:
  - when the canonical decodes were replaced, and where the backup is;
  - what changed.
- **`docs/known_issues/three_sessions_blocked_by_nwb_checksum.md`** is updated once Stage 5
  shows whether the three sessions rejoin the cohort.
- **Generated docs** are regenerated in Stage 7. Hand-maintained result docs wait for the
  before/after report.
- **Project memory:** the decode-drift and cohort notes are updated.
