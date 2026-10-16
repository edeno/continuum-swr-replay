# Full pipeline refresh: re-decode, comparison inventories, downstream regeneration

**Status (2026-10-12): the refresh is done and audited.** The final audit passed (17 of 17
checks), and the report is
[docs/audits/2026-10-12-full-pipeline-refresh.md](../audits/2026-10-12-full-pipeline-refresh.md).
The implementation plan is `.claude/docs/plans/full-pipeline-refresh/PLAN.md`. What the re-decode
changed, and where the backups are, is in
[the known issue](../known_issues/2026-10-12-full-redecode-r11.md).

The spec was approved on 2026-10-03 and revised after an independent review (dask-cuda, ContFrag
staging, R9, backups, Kay version check, filter edges). It was not rewritten after the build. Its
planning numbers stand as written, among them "218 epochs". The cohort is now the epoch
register's 242 analysed epochs, with 13 more left out as `SUSTAINED_NOISE`.

Out of scope, as the spec says: Phase 3 of the comparison, Phase 2's deferred pilot checks,
Spyglass integration, and requoting the hand-maintained result docs (they wait for the project
lead's review of the report).

Not run in the refresh (report, section 4.7):

- Task 12 on the R11 decodes. Only the five-epoch pilot on pre-R11 decodes exists.
- Task 13's half-tetrode decodes. The code is done; the 20 GPU decodes were not run.
- The R13 robustness and sensitivity reruns: the sensitivity without the epochs over 2 % masked,
  the masked-time table, and the reruns without the mask, with every artifact threshold 25 % lower
  and higher, and with 1 s masked after each step.

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
| R3 | Zugaro reads one channel, picked per epoch by buzcode's `bz_GetBestRippleChan` from the same CA1 electrodes and references Kay, Karlsson and Yu read | The candidate set and referencing are shared input. Using one channel, and choosing it by the lineage's own rule, is the method's practice (revised 2026-10-03 from "all CA1 channels summed") |
| R4 | Direct Kay at `68e1784` is the comparison's Kay inventory. The canonical `SWR_event` stays as an extra SWR definition and as the version check | Kay then matches the other three in package, band, electrodes and speed |
| R5 | The Widloski runner reloads marks through the project's loader; the decoder records nothing extra | The project lead's choice. The loader is the code that fed the decode, and the audit confirms one commit (Stage 8). The guard's limits are stated under Stage 4 |
| R6 | Re-decode all 218 epochs, including the three once blocked by an NWB checksum | All three load fully as of 2026-10-03. An epoch that still fails has its April files moved to the backup and is listed, so the canonical folder holds one decode version |
| R7 | Rebuild the ContFrag decodes and everything downstream, including model refits and figures | Nothing is left on April data, so every comparison and every quoted number shares inputs |
| R8 | A lean implementation plan: code tasks with acceptance criteria, plus a runbook per long run | The runs span days and several sessions |
| R9 | Non-local events take the pinned package's minimum-duration rule as it computes on our timestamps: `minimum_sample_count(time, 0.015)` is **7** at our 500 Hz grids, where 1.7.0 needed 9 bins | The project lead's choice (2026-10-03), made after review found the pin changes `segment_boolean_series` (`event_detection.py:78`), and after measuring the pilot epochs on the April posteriors: 2,365 events under 1.7.0 (9 bins), 2,382 at 8 bins, and 2,405 at 7 bins (+1.7%; j16 +4%). It matches the duration rule the four SWR detectors use at the same version. 15 ms is exactly 7.5 samples at 500 Hz, and absolute Unix timestamps make the median float64 step 2.0000935 ms, so the count rounds to 7. Every session shares that float spacing, so every epoch should compute 7; Stage 1 records each epoch's count and stops if any differs |
| R10 | The decoders' discrete transitions are pinned in `src/continuum_swr_replay/decoding_transitions.py`, never left to the library's defaults. The NonLocal decode uses setting P30-B: a fixed-count Dirichlet prior whose mode (the M-step's zero-data estimate) gives sourced dwell times (Local 3 s, non-local 100 ms), worth 30 visits per state, with the No-Spike row frozen at base-rate exits. The ContFrag decode keeps its values, made explicit. Both runners refuse `--non_stationary`. Details: [2026-10-04-decoder-transition-prior.md](2026-10-04-decoder-transition-prior.md) | The project lead's choice (2026-10-04), after validation on the five pilot epochs. The decoders had taken `non_local_detector`'s defaults, which changed after April and added 32% non-local events on identical inputs. The library's non-stationary transitions cannot be saved (#50) and add their prior at every time bin (#51) |

## What the re-decode changes

Measured on 2026-10-03 from `chimi20200225_02_r1_HPC_detector.pkl`, the git history since
`0820d39d`, and `load_data` on three sessions. On 2026-10-04 the discrete-transition rows
were updated for R10, and the concentration and start-row row was added. Their April values
are the defaults of `non_local_detector` `5e3a086`
(`src/non_local_detector/models/_defaults.py`, read with `git show` in the library clone).
The April `chimi20200225_02_r1_HPC_detector.pkl` carries the same values.

| Input | April (`0820d39d`) | Now |
|---|---|---|
| HPC marks: sort groups | every `UnitMarks` sort group | tetrodes only (`f172084`). wilbur20210331, for one, loses probe sort groups 17–19, which have 32 channels each |
| HPC marks: coincident spikes | each cluster's leading spike removed but not counted | counted (`64bf585`). 259 more of about 358,000 marks are dropped on chimi20200225_02_r1 |
| Position | untrimmed DLC grid | trimmed to the valid interval (`e5a0def`); up to about 76 s shorter at an epoch's end |
| `non_local_detector` | `5e3a086` | `2d730c3` |
| Local state | exact position | Gaussian, `local_position_std` 2.5 cm (`ded292e`) |
| Transition prior: concentration and Local start row | 1.0; [0.99, 0.005, 0.0025, 0.0025] | 11 and [0.99933, 0.00022 ×3], pinned (R10) |
| Stickiness: Local and No-Spike / non-local | 1e6 / 300 | pinned (R10). Local 44,960 / non-local 1,460 (each learned row's prior mode, the M-step's zero-data estimate, equals its start row; 30 visits per state). The No-Spike row is frozen, so its stickiness is unused. The library's defaults (1e5 / 300) are not used |
| Non-Local Continuous self-transition prior | 0.98 | 0.98, a 100 ms stay, pinned (R10); the same for Non-Local Fragmented |
| No-Spike transition row during fitting | updated | frozen at [0.0192, 0.98, 0.0004, 0.0004] (R10) |
| `results.nc` contents | posterior | posterior and the per-bin `log_likelihood` (added 2026-06-26) |
| `ripple_detection` | 1.7.0 | `68e1784`: the changelog's 2.0.0 (2026-09-22), never tagged, so `git describe` gives `v1.7.1-365-g68e1784` |
| Non-local event minimum duration (`segment_boolean_series`) | the last bin minus the first is at least 15 ms, so 9 bins at 500 Hz | at least `round(0.015 / step)` samples, so 7 bins at our grids (R9). The changelog marks it breaking; on the pilot epochs it adds 1.7% |
| MUA burst events (`multiunit_HSE_detector`) | 1.7.0 rules | the 2.0 rules: duration counted in samples, one missing-sample policy, and a NaN speed that fails the speed rule |
| Speed smoothing (`gaussian_smooth`) and the SWR consensus trace | kernel not renormalized at the ends | kernel renormalized at the ends of the data, so edge samples change |

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
  confirm that `jax.devices()` lists the GPUs, and that `uv.lock` still holds
  `non_local_detector` `2d730c3` and Spyglass `18b2bff`.
- **dask-cuda.** `dask_cuda` is not installed (the review found `find_spec` returns None),
  so `run_batch` falls back to a CPU `LocalCluster` and ignores `--cuda_visible_devices`
  (`decoding_common.py:23-28, 235-241`). Install the locked version
  (`uv pip install "dask-cuda==25.10.0"`) and confirm `import dask_cuda`. `run_batch` and the
  Widloski driver are changed to refuse a device list when `dask_cuda` is missing.
- **Smoke under the pin.** Run `get_ripple` (Spyglass's `RippleTimesV1` read and
  `get_Kay_ripple_consensus_trace`) and `get_multiunit_hse` on one epoch, then the unit
  suite.
  - Spyglass imports `Kay_ripple_detector`, `Karlsson_ripple_detector`,
    `core.gaussian_smooth` and `core.get_envelope` from the package.
  - The project imports `segment_boolean_series`, `gaussian_smooth`,
    `get_multiunit_population_firing_rate` and `multiunit_HSE_detector`.
  - All of these are still exported at `68e1784`.
- **Version-shift report.** On the five pilot epochs, measure under 1.7.0 and again under
  `68e1784`:
  - MUA burst events, each matched, missed or added by overlap;
  - the largest difference in the SWR consensus z-score;
  - non-local events per epoch under the 1.7.0 and 2.0 minimum-duration rules, segmented
    from the April `Non-Local` probabilities, with each epoch's minimum sample count (R9).
- **Golden baseline.** Freeze the current tables and models read-only with
  `scripts/qa/freeze_regression_baseline.py --label pre_redecode_YYYYMMDD`, for the
  before/after report.
- **Other backups.** The golden baseline copies only the tables and 11 model folders
  (`freeze_regression_baseline.py:64-76`). Copy everything else the refresh overwrites into
  one backup root, `Processed-Data/_pre_redecode_backup_YYYYMMDD/`, dated once at Stage 0:
  - `models_dm/` and `content_estimator_comparison/`;
  - all of `contfrag_replay_content/`, tables and models;
  - the Stage 5B and 6 outputs at the `replay_content` root;
  - the gitignored figure PNG and PDF files under `paper/figures/`.
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
  - The decoders create `save_dir`.
  - On resume they read results back from `save_dir`, not the canonical folder: both loaders
    gain a `root`. They then rewrite only `data_over_time`, never the `results.nc` they just
    reopened lazily.
- **Logs.** `run_batch` writes `logs/{epoch_identifier}.log` in mode `w`. The April logs
  are the only per-epoch record of the April commit, so they are copied into the backup
  before any launch. Each new epoch log records:
  - the commit and whether the tree was dirty;
  - the `uv.lock` sha256 and the `ripple_detection` commit;
  - the `non_local_detector` version and the CUDA plugin version.
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
  5. Every staged log names `C0` with a clean tree.
  6. Every epoch's minimum non-local sample count, `minimum_sample_count(time, 0.015)`, is
     the same (7 expected). If one differs, stop and report (R9).

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
  CA1 set for the same band key, the check spec 01 asks for. `RippleLFPSelection` is keyed on
  `LFPBandV1`, so an unrestricted query would mix every epoch's selection.
- **Raw LFP behind that band.** Read from the same `LFPOutput` entry.
  - Its columns are positions, not electrode ids (24 raw columns against 22 band columns on
    chimi20200225_02_r1), so each column is mapped to its electrode through the LFP's
    electrode table, as Spyglass does.
  - Each channel then has its reference subtracted, in the stored dtype, as Spyglass does
    (`lfp_band.py:420-436`). Any integer overflow is counted, and the data become float64
    afterwards.
  - These referenced channels are Zugaro's candidates, from which it picks one.
- **Input check.** Spyglass's own FIR filter (`FirFilterParameters`, `ripple_150_250` at
  1000 Hz: 535 taps, band edges 140/150/250/260 Hz) is applied through
  `FirFilterParameters().filter_data` to the referenced LFP, with `LFPBandV1.make`'s exact
  arguments: the valid-time interval from `_compute_interval_times`, the native dtype, all
  columns, the electrode index and the decimation. The result must reproduce the stored band
  to the stored precision. This shows Zugaro's
  candidates are the channels the other three read, with the same referencing. The largest difference goes in the
  manifest, and a mismatch stops that epoch.
- **Speed and span.** `head_speed` comes from one helper shared with `load_data`:
  `get_position_info`, then the same drop of NaN-position rows (`bandit_task.py:343-347`).
  That is the source and span the new decode uses. It is linearly interpolated onto LFP
  times. The LFP is restricted to the
  position span, as `filter_ripple_lfp_time` does.
- **Rate.** The nominal 1000 Hz is passed to every detector and to the Zugaro filter design,
  as Spyglass passes it to canonical Kay. The measured rate, steps over elapsed time within
  gap-free stretches, goes in the manifest. On chimi20200225_02_r1 it is about 999.991 Hz:
  the recording clock runs about 8.5 ppm slow, which moves a 150 Hz band edge by about
  0.001 Hz, so the nominal rate stands. The 1000.07 Hz that the median timestamp step
  suggests is not the clock but float64 rounding of absolute Unix timestamps: near 1.58e9 s
  a float64 resolves 2^-22 s, so each 1 ms step is stored as 4194 or 4195 (rarely 4196) of
  those units, and 2^22/4194 = 1000.0725. It shifts no band. The LFP's anti-aliasing filter
  is recorded too.

### The four detectors, package at `68e1784`

| Detector | Input | Settings |
|---|---|---|
| Kay | stored 150–250 Hz band | Spyglass's canonical settings, equal to the package defaults: z ≥ 2 for 15 ms, 4 ms Gaussian smoothing, z-score over all valid samples, no close-event exclusion |
| Karlsson | stored band | package defaults: z ≥ 3 on any tetrode for 15 ms, 4 ms smoothing, z-score over all valid samples |
| Yu | stored band | spec 01, step 4: threshold at the 99.99th percentile of the mirrored immobility noise, 20 ms above it, 4 ms smoothing, per-tetrode z-score (`zscore_per_channel=True`). Threshold diagnostics from `noise_threshold_diagnostics` are kept per epoch, so they survive an epoch with no events: the scalars (threshold, mode, flank ratio, sample counts, normalization statistics) in the manifest, and every field with the histogram arrays (edges, counts, smoothed and mirrored counts) in the epoch's `{epoch_id}_yu_diagnostics.json` |
| Zugaro | one raw referenced channel, chosen and filtered as below | squared; moving average of 11/1250 s (9 samples at 1000 Hz); z-score over all valid samples, which excludes the stretch edges below; boundaries strictly above 2 SD; peak strictly above 5 SD; events closer than 50 ms merged, with no length cap; no minimum duration. Events longer than 150 ms are then dropped (kept at exactly 150 ms) |

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

**Channel selection,** following buzcode `bz_GetBestRippleChan` (`0969ddf`), per epoch, over
the CA1 candidates' referenced raw LFP:

1. Band-pass 140–180 Hz with a Butterworth filter of order 4, zero-phase. As for detection,
   it is designed at 1000 Hz in second-order sections and filtered per gap-free stretch. Its
   padding and edge exclusion come from its own settling time, about 170 ms (see "Gaps and
   edges").
2. Compute a centered moving RMS over 12 ms, the original's 15 samples at 1250 Hz. As the
   package does for the 11-sample average, the window is given in seconds and rounded up to
   an odd count, so 13 samples at 1000 Hz.
3. Score each channel as mean RMS ÷ median RMS over its valid samples. A channel with
   occasional large ripple bursts over a low baseline scores high.
4. Leave out channels whose mean or median RMS is below 1, buzcode's dead-channel guard. It
   applies in the stored LFP units, which the manifest names. If every channel is left out,
   the epoch fails with that reason.
5. The highest score wins, and an exact tie goes to the lowest electrode id.

The original's own comment calls it a placeholder ("eventually this will detect which lfp
channel has the highest SNR"). The other automated rules in the lineage are set aside:
- Harvey 2023's power ratio (100–250 Hz over total power) uses a band from other settings
  than the Databank ones.
- neurocode's `swrChannels` asks interactive questions, so it can't be reproduced faithfully.

The manifest records every candidate's score, the excluded channels and the winner. The
report says how often the same channel wins across a day's run epochs.

**Where this run departs from the lineage,** each recorded in the manifest:

- **Channel choice.** The channel is picked automatically by `bz_GetBestRippleChan`. The
  Databank files store theirs as a tag (`channelTags.Ripple` in `session.mat`), often set by
  hand in CellExplorer's session GUI, and the files don't record how it was chosen.
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
  - Padding and edge exclusion are set per filter, from the time its single-pass impulse
    response takes to decay below 1e-3 of its peak, rounded up to 10 ms:
    - the Zugaro filter takes 86 ms, so padding is 90 samples (odd extension) and 90 ms;
    - the channel-selection filter takes about 166 ms, so about 170.

    SciPy's default padding is only 21 samples.
  - The first and last stretch of that length count as unobserved. For Zugaro, they are
    left out of its z-score and events, recorded, and applied to every detector through
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
- **Per epoch,** the Kay version check, in two comparisons. Each marks every event of
  either set matched, missed or added by positive-duration overlap (the project's `is_swr`
  rule), with the boundary differences of the matched ones.
  1. **Version only.** Kay at `68e1784` on Spyglass's own inputs, from the read-only
     `RippleTimesV1.get_ripple_lfps_and_position_info(key)`, with Spyglass's stored
     `RippleParameters` for that key, against canonical `SWR_event`. The manifest records
     the stored parameters and flags any mismatch with the direct Kay's settings, which
     would also separate the second comparison.
  2. **Version and inputs.** The direct Kay inventory against canonical `SWR_event`.

  Both are reported over all events and over speed-eligible events only.
- **Per epoch,** Yu's threshold diagnostics with their histogram arrays,
  `{epoch_id}_yu_diagnostics.json`, when Yu's threshold could be estimated.
- **Status per detector,** so one detector's failure doesn't hide the others' results.
- **`manifest.json`**, with:
  - the package commit, every parameter, and the Zugaro filter design;
  - the electrodes and references, and the input-check result;
  - Yu's threshold diagnostics, as scalars; the arrays are in the epoch's sidecar;
  - `meets_published_tetrode_criterion` (true, false or unknown);
  - input hashes and library versions.

**Epochs with no inputs** are recorded as skipped with that reason, from a documented list:
- senor20201104 06_r3 and 08_r4 have no ripple band;
- senor20201105_09_r4 has no track graph, so its position can't be loaded;
- wilbur20210404 has no CA1 ripple extraction.

None of them has a decode either.

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
- **Channel selection:**
  - a channel with sparse large bursts over a low baseline wins over a channel with steady
    power;
  - a flat or near-dead channel is excluded;
  - a tie goes to the lowest electrode id;
  - the 13-sample (12 ms) window and the 140–180 Hz order-4 band are pinned;
  - the dead-channel guard is pinned at 0.99 against 1.01;
  - all channels excluded is an error;
  - each filter's kept rows are free of edge transients: a short stretch's interior equals
    filtering a long signal.
- **Other units:**
  - the electrode assertion;
  - the pinned settings for all four detectors;
  - the Kay matching on made-up intervals.
- **End to end:** the package's simulated session at 1000 Hz runs through the wrapper, and
  each detector finds the injected ripples. `simulate_session` draws ripple frequencies from
  150–250 Hz by default, and Zugaro's band stops below 200 Hz, so:
  - the Zugaro assertion uses `ripple_frequency=(150, 175)`;
  - a 230 Hz ripple must not be detected by Zugaro.
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
    filters beside `--animal`, `--all` and `--epoch-id`. The date is the last 8 characters of
    the NWB stem, as `prepare_data.py:135` reads it, so `j1620210707` gives 20210707.
  - A device list without `dask_cuda` installed is refused.
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
   `prepare_replay_content_data.py --stop-only --from-cache --output-dir <staging> --behavior-cache-dir replay_content/_session_cache`.
   About 6 min serially. Every prepare call passes the same `--behavior-cache-dir`. Without
   it the cache path becomes `<staging>/_session_cache`, and every session is refetched from
   Spyglass (`prepare_data.py:899-904`).
2. **`qvals_table`.** `build_qvals_table.py --replay-content-dir <staging> --write`, about
   12 min; it reads the live `alison_*` DataJoint schemas. It must follow the stop table:
   built before it once, it left a whole session missing
   (`docs/audits/2026-08-25-covariate-regen-staging-audit.md`).
3. **Replay tables.**
   `prepare_replay_content_data.py --from-cache --output-dir <staging> --behavior-cache-dir replay_content/_session_cache --contingency-dir <staging>`.
   - Runtime: about 1.5 h serially, or about 36 min as five per-animal shards followed by
     `assemble_replay_content_shards.py <shard dirs> --output-dir … --contingency-dir …`.
   - Built-in gates:
     - it fails closed on any failed session;
     - Σ`n_selected_events` must equal the event count;
     - the `qvals` clock and value-block checks run.
   - The cross-table check and the no-session path return exit 0 and write nothing
     (`prepare_data.py:971, 997-1004`). Every gate therefore checks the written outputs and
     their row counts, never the exit code alone.
4. **Timebin tables.** `build_timebin_tables.py`, about 69 min; it reads pump events
   through live `load_data`. It gains an `--output-dir` option, since today it writes into
   `replay_content/` directly. These tables must be rebuilt: the last regeneration carried
   them forward, which is invalid once stop ids change.
5. **ContFrag tables,** in their own staging folder,
   `Processed-Data/contfrag_replay_content/_staging_redecode_YYYYMMDD/`.
   - Run `prepare_contfrag_replay_content_data.py --event-source SWR` and
     `--event-source MUA`, each with
     `--from-cache --output-dir <contfrag staging> --replay-content-dir <staging> --behavior-cache-dir replay_content/_session_cache`.
     The two run concurrently, about 2.5 h each.
   - `--replay-content-dir` defaults to the canonical `replay_content`, which supplies both
     the `qvals_table` and the behavioral cache (script lines 336-378). Pointing it at the
     staging folder joins the new `qvals`; the new `--behavior-cache-dir` keeps the cache
     canonical.
   - The ContFrag fitters read `contfrag_replay_content/` (`fit_replay_content_models.py:154-159`).
6. **Regeneration manifest.** `write_regeneration_manifest.py`.
   - It expects `models_validation/` bundles and a clean tree, and the command that produced
     those bundles was never recorded. The plan's first task for this stage reads the script
     and settles how to run it.
   - `compare_to_golden_baseline.py` raises at the first drift (`:101-112`), so it can't
     report one. A scratch summary of rows per table and session, old against new, is the
     drift report instead.
   - A tables manifest records every staged table's sha256, the commit that built it, and
     its inputs (Stage 8, check 6).
7. **Promote (go-ahead).**
   1. Back up the current `replay_content` tables and the `contfrag_replay_content` tables
      into the Stage 0 backup root.
   2. Copy the staged files in with `cp -p`. The list of files is written down beforehand,
      including `standardization_reference.json` and
      `models_per_bin_rate/qvals_standardization_reference.json`, which the fitters require
      (`table_contracts.py:396`).

**Shortcuts not to use, because each is invalid after a re-decode:**

- `regenerate_stop_table.py` assumes the stop bouts are unchanged.
- `refresh_contfrag_behavioral_covariates.py` copies the old posterior aggregates.
- `scripts/exploratory/_augment_data_over_time.py` rewrites the `data_over_time` pickles.
- `python -m …prepare_data`: the module has had no `__main__` since `0e6c4c7`, so the
  command now does nothing.

## Stage 6: model refits and PFC analyses

**The refit list is built from what is on disk,** not from memory: every live fit
 artifact's provenance sidecar records the command that made it (`cli`). That covers:
- `replay_content/models*/`;
- `contfrag_replay_content/models_*`;
- the Stage 5B and 6 and `stop_replay_summary` artifacts at the `replay_content` root.

Smoke bundles (`*_smoke`) are moved into the backup rather than refit. Runtimes below are from
the September logs, and every refit is re-measured.

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

There are three Stage 5B runs and eight Stage 6 runs, each as its sidecar recorded. The
first Stage 6 run takes `--rebuild-summary`.

### GPU, needs a go-ahead

The mPFC unit GLM covers 71 sessions, at σ 3, K 5, R 1,000, on the at-well cohort.

- It runs one process per card (`--n_workers 1 --gpu N --sessions <partition file>`),
  modeled on `logs/pfc_ridge_heldout_20260916/launch.sh`.
- It takes about 14–17 h per card on 5 A100s.
- The existing mPFC bundles move to the backup first, since the launcher refuses to start
  over them.
- It runs on cards that don't overlap Stage 4's.

### Gate

`check_fit_provenance.py` must exit 0 at the end of the stage. It runs on both the
`replay_content` and `contfrag_replay_content` roots, and the root-level Stage 5B and 6
artifacts are checked against their sidecars.

## Stage 7: figures and generated docs

Figures are regenerated by running their scripts. No figure script is edited; the rule
against editing the Figure 1–4 scripts stands.

- **After Stage 2, from the decodes alone** (per the dependency map; the plan confirms
  each script reads no table before scheduling it this early):
  - Figure 2 with `--force-recompute` (about 16 min); its joblib cache would otherwise
    serve April data;
  - Figure 3;
  - `generate_supp_decoder_validation.py`;
  - the event gallery (about 4 h at 6 jobs), after
    `generate_event_gallery.py catalog` rebuilds its event selection.
- **After Stage 5 promotes the tables:** `generate_supp_pipeline_data_flow.py`, which reads
  the canonical `event_arm_table`.
- **After Stage 6:**
  - Figure 4 (about 2 min);
  - the rate and content plots, `plot_swr_contrast_coefs.py`, and
    `generate_uncertainty_content_figure.py`;
  - `summarize_per_bin_rate_models.py`, which writes `docs/per_bin_rate_results.md`.
- **Unaffected:** Figure 1 and `generate_memoryless_comparison.py` use simulation only.

## Stage 8: same-data audit and before/after report

A new `scripts/qa/audit_refresh_consistency.py` writes `audit.json` beside the outputs,
summarized in `docs/audits/YYYY-MM-DD-full-pipeline-refresh.md`. It checks:

1. **Commit and environment.** One commit, a clean tree and one `uv.lock` hash across every
   decode log (Stage 1 records them per epoch), or an explicit list of exceptions with
   reasons.
2. **NonLocal against ContFrag.** Per epoch, they share time grid and position (the
   Stage 1 gate, re-checked on the canonical files).
3. **Marks.** Per epoch, the marks reloaded through the loader contain every encoding mark
   of both decodes. This is the existing guard, now applied to ContFrag too.
4. **Widloski.** The run manifest's decode hashes equal the canonical decode files'.
5. **SWR inventories.** Per epoch, the runner's position span and speed source equal the
   decode's.
6. **Tables.** The Stage 5 tables manifest: each table's sha256, the commit and inputs that
   built it, and that commit descending from the decode commit.
7. **Models.** `check_fit_provenance.py` exits 0, and the PFC bundles' sidecars name the
   new event table.
8. **Epoch set.** One set across all artifacts, with every absence explained.

The **before/after report** covers:

- the Stage 5 drift summary, plus `compare_fits.py` and `compare_pfc_cohorts.py`;
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

1. **Decoders:**
   - `decoding_common` gains `--save_dir` and `--allow_cpu`, and `run_batch` refuses a
     device list when `dask_cuda` is missing;
   - both decoders create `save_dir`, resume from it, and on resume rewrite only
     `data_over_time`;
   - both log the per-epoch provenance;
   - `load_results` and `load_contfrag_results` gain `root`.
2. **Staging options for the table builders:** `build_timebin_tables.py` gains
   `--output-dir`, and `prepare_contfrag_replay_content_data.py` gains
   `--behavior-cache-dir`.
3. **Widloski runner,** as in Stage 4. `decode_marks.py` and its tests are removed.
4. **Data loaders:**
   - `data_loaders/lfp.py` gains the ripple-input fetch, sharing `get_ripple`'s keys;
   - `data_loaders/position.py` gains the position helper shared with `load_data`.
5. **New `analysis/hippocampus/swr_inventories.py`.**
6. **New `scripts/decoding/run_swr_detection.py`.**
7. **New `scripts/qa/audit_refresh_consistency.py`,** including the Stage 5 tables manifest.

Every change comes with the tests listed under its stage. Unit tests run offline: any module
that imports Spyglass at import time is stubbed in `sys.modules`, as
`tests/unit/test_events.py` does.

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
