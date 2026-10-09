# Artifact detection: mask only on signatures that SWRs lack

**Status (2026-10-09):** the project lead approved the design on 2026-10-09, subject to a
literature check, which found it consistent with field practice. Implemented in
`src/continuum_swr_replay/analysis/hippocampus/artifacts/` and
`scripts/decoding/run_artifact_detection.py` (Task 16 of the full pipeline refresh,
[2026-10-03-full-pipeline-refresh-design.md](2026-10-03-full-pipeline-refresh-design.md)).
**Not yet applied:** `swr_source.ARTIFACT_INTERVALS_RUN_ID` stays None until runbook R9 runs
the detector on all epochs and sets it. Study C1 (licking and chewing EMG) may add a fifth
signature first.

Sources: the design study
`.claude/research/refresh_artifact_detector/detector_report.md` (the "report"), its
literature review `literature_review.md`, and its prototype scripts. Task 16's check against
the prototype is in `task16_check/` there.

## Decision

- **Mask only on an artifact signature,** never on power alone. A large SWR is also a
  large, multichannel, high-frequency deflection on CA1, so any rule that counts channels
  above an amplitude also masks SWRs. The four signatures are below.
- **Robust units only.** Every threshold is a statistic over its own epoch MAD × 1.4826,
  per channel. No absolute LFP units: the stored raw LFP says "volts, conversion 1.0",
  which is suspect.
- **One set of cores, one padding table.** The detector stores unpadded cores and their
  evidence. Each consumer pads them from `masks.CONSUMER_PADDING`:
  - `swr`: the SWR detectors (they read the LFP);
  - `decoder`: the decoders and MUA detection (they read spikes);
  - `analysis`: everything downstream.
- **Artifact time is unobserved.** An event of any kind that overlaps the `analysis` mask
  leaves every cohort; it is not counted as non-SWR. The mask's time leaves every rate
  exposure. `swr_source`'s coverage (`swr_observed_*`) already excludes it.
- **The 1 s after a step.** Ripple-band power stays below its random-time level for at least
  1 s after a step (report §8 item 4). The `analysis` mask covers it.
- **Keep the noisy tetrodes.** The sustained-noise (S) tetrodes are masked during their
  noise blocks, not dropped. A sensitivity analysis (R8) leaves out the 4 epochs with more
  than 2 % masked. R11 first decodes the 12 S epochs both ways (masked and with those
  tetrodes dropped); masking stays unless the masked decode shows extra non-local
  probability within 3 s outside the S blocks, beyond the 1-s pad.
- **No amplitude cap.** Some pipelines drop events above 9–10 SD as non-biological. Our true
  SWRs reach 44 robust z. Of the 9,208 true SWRs (report §3), 2,538 (27.6 %) have a Kay
  consensus peak above 9 robust z and 2,181 (23.7 %) above 10; so do 24 and 23 of the 55
  visually confirmed SWRs. These counts are in robust units (median and MAD × 1.4826). In
  mean / SD units they are probably lower, since the consensus SD is typically about 1.15
  robust SD; study C1 reports that count.

## The signatures

| Kind | Signature | Rule (`detect.py`) |
|---|---|---|
| **S** | Ripple-band power raised on a *fixed subset* of CA1 channels for at least 2 s | `s_cores` |
| **T-ref** | An abrupt level jump on the raw reference electrode, which sits in white matter outside CA1 | `t_ref_events` |
| **T-ca1** | An abrupt *and persistent* level step on at least 3 CA1 channels at once | `t_ca1_events` |
| **T-spk** | A spike line: ≥ 40 % of tetrodes in one 200-µs bin, ≥ 20 % of those marks out of distribution | `t_spk_events` |

- **S.** For each 1-s window *w* and CA1 channel *c*: the 150–250 Hz Welch power of the raw
  referenced LFP (15 Hann segments of 64 ms), over channel *c*'s epoch-median window power,
  is r[w, c]. The specificity spec[w] is the 2nd-largest r over channels divided by the
  median r. An SWR raises ripple power on most CA1 tetrodes at once for under 0.3 s, so spec
  stays near 1; S raises it on a fixed subset for seconds. A core is ≥ 2 consecutive windows
  with spec ≥ 4; the block is the run of windows with spec ≥ 2 that holds it; blocks ≤ 2
  windows apart merge.
- **T-ref.** J5(t) = median(x[t+1..t+5]) − median(x[t−5..t−1]), a 5-ms median level jump.
  T-ref fires where the reference |J5| ≥ 15 robust units. A step on the reference appears on
  every channel referenced to it; a hippocampal SWR reaches the white-matter reference only
  weakly.
- **T-ca1.** A channel's jump persists when |J5| ≥ 20 robust units, and
  L20(t) = mean(x[t+5..t+24]) − mean(x[t−24..t−5]) has J5's sign with |L20| ≥ 0.5 |J5|.
  T-ca1 fires where at least max(3, 0.15 × n channels) channels step persistently at the
  same sample. A step stays shifted for tens of ms; a ripple cycle or a sharp-wave onset does
  not.
- **T-spk.** Spike synchrony in 200-µs bins on two grids offset by half a bin, so a line
  across one grid's bin edge is whole on the other. A mark is out of distribution (OOD) when
  its nearest reference mark of the same tetrode (Euclidean, raw marks in µV), the spike
  itself left out, is more than 4 × the decoder's `waveform_std` (24 µV, read from
  `DecodingConfig`) away: there the decoder's mark kernel has no support (report §2.4).
- **Merging.** T signatures within 50 ms form one T core. Each rule is its own function, so a
  fifth signature (C1's EMG, if needed) is one more event source.

### The OOD reference: the one deliberate change from the prototype

The prototype measured each mark against a fitted decoder's encoding marks
(`*_HPC_detector.pkl`). The new decodes will use this mask themselves, so the reference must
not depend on a decode. The repo uses, per tetrode, **all of the epoch's marks outside the
LFP-signature cores** (S, T-ref and T-ca1, padded by the `decoder` table), the spike itself
left out. The mark space and metric are the prototype's. The encoding set it replaces was
"all non-ripple time", which included the artifact spikes around steps (report §8 item 8);
leaving those out makes some lines near steps OOD that the prototype's reference called
in-distribution. Task 16's check measured the effect (below).

## Parameters and evidence

From the report, §5.1. Each is a module constant in `detect.py` or `masks.py`, with a
one-line evidence note naming the section. Sensitivity: report §5.2.

| Parameter | Value | Evidence | Margin |
|---|---|---|---|
| `t_ref` (reference \|J5\|, robust units) | 15 | Giant-day Kay∩Karlsson max 10.7 (99.9th pct 8.1); true-SWR max 6.5; benign reference-only bursts on giant days reach 10.1–13.1; visual T median 31.9 | At 10, 13 benign reference intervals and 4 giant-day events are masked; at 8, +5 true SWRs and +153 s; at 6, +31 true SWRs and 8 visual SWRs |
| `t_ca1_k` | 20 | At K ≤ 12 plain or persistent jumps occur at giant-SWR onsets; at K = 12 (ρ = 0.7), 6 of 6 giant-day events flagged were SWRs | True-SWR max fraction 0 at K = 20; K = 12: 8 visual SWRs |
| `t_ca1_rho` | 0.5 | Step morphology; changes nothing measured at K = 20 | Kept as a physiological guard |
| `t_ca1_frac` / `t_ca1_min` | 0.15 / 3 channels | SWRs reach at most 1 channel at K = 20 | 0.10 adds 1 T; 0.25 loses 2 |
| `t_spk_bin` | 200 µs, two grids | Spike lines are simultaneous; at 0.5 ms a clean-day giant SWR reaches OOD 0.200 and is masked | 0.5 ms adds 9 T but costs that SWR |
| `t_spk_frac` | 0.40 | True SWRs reach 0.50 in 200 µs, so the OOD gate carries the specificity | 0.3: +2 true SWRs and 1 visual SWR; 0.6 loses all 3 spike-only artifacts |
| `t_spk_ood` | 0.20 | True-SWR / visual-SWR / giant-day max 0; spike-only artifacts 0.27–0.42 | 0.3 loses 1 spike-only artifact |
| OOD cut | 4 × `waveform_std` | Calibrated against the decoder's own mark term (report §2.4) | — |
| `s_core` | 4 | Non-S epochs never hold 2 consecutive windows ≥ 4; confirmed S blocks have a median longest run of 11 | 2.5: +18 true SWRs and +127 s |
| `s_core_run` | 2 windows | Same evidence | 1: +11 true SWRs and +121 s |
| `s_ext` | 2 | Extension to partial windows | 3: S recall 0.993 |
| `s_gap` | 2 windows | A step inside a block drops spec for one window; without the fill senor20201101_14_r7 keeps SD / robust SD 1.61 | — |
| `t_merge` | 0.05 s | T signatures of one event | — |

Two values the report states only in its scripts:

- **A line needs at least 3 tetrodes** (`T_SPK_MIN_TETRODES`). The prototype's line table
  kept bins with ≥ max(3, 25 %) of tetrodes. Every analysed epoch has 11–22 tetrodes, so only
  the 40 % rule binds.
- **The S window** is 1 s of samples from the span's first sample; the last partial window is
  not read, as in the prototype.

## Consumers and padding

| Consumer | Who | S | T with an LFP signature | T spike-only |
|---|---|---|---|---|
| `swr` | SWR-detector normalization and event dropping (Kay, Karlsson, Yu, Zugaro) | 0 s | ±0.3 s | ±0.1 s |
| `decoder` | `is_missing`, encoding exclusion, MUA detection | ±1 s | ±0.3 s | ±0.1 s |
| `analysis` | Event eligibility for every cohort, rate exposure | ±1 s | 0.3 s before the core, 1.3 s after it | ±0.1 s |

- **S, 0 s for `swr`:** with gap fill, 0 / 0.5 / 1 s give identical SD / robust SD in all 11
  S epochs; 0.5 s already masks a visual SWR 0.55 s before a block.
- **S, ±1 s for `decoder`:** noise spikes on the affected tetrodes leak up to 1 s past some
  block edges (`validation/S_edge_spike_rates.csv`).
- **T, ±0.3 s:** consensus ringing above 2 robust z after a core has its 99th percentile at
  191 ms and its maximum at 335 ms; the stored ripple FIR's half-length is 0.267 s; and the
  post-step OOD discharge tail reaches +269 ms.
- **Spike-only, ±0.1 s:** 99.3 % of excess OOD spikes lie within ±100 ms of spike-only cores.

Intervals are closed, merged when they overlap or touch, and clipped to the epoch's span.

## Interface

- `artifacts/features.py`: `robust_scale`, `window_band_power`, `window_centers`,
  `channel_specificity`, `level_jump` (J5), `level_shift` (L20), `robust_jump`,
  `reference_jump`, `persistent_step_count`, `nearest_mark_distance`, `spike_lines`.
- `artifacts/detect.py`: the constants, the four rules, `t_cores`, `spike_ood` and
  `detect_artifacts(time, ca1_lfp, reference_lfp, spike_times, spike_marks,
  sampling_frequency, *, valid=None) -> pd.DataFrame` (the cores, unpadded).
- `artifacts/masks.py`: `CONSUMER_PADDING`, `Consumer`, `artifact_intervals(cores,
  consumer, span)`, `interval_mask(time, intervals)`, `subtract_intervals`,
  `masked_seconds`.
- `artifacts/io.py`: the cores file (`CORE_COLUMNS`: `kind`, `signatures`, `core_start`,
  `core_end`, `spec_max`, `n_windows`, `ref_j5_max`, `ca1_frac_max`, `spk_frac_max`,
  `spk_ood_max`) and the manifest; `read_epoch_cores` checks the manifest's run id, the
  epoch's status and the file's sha256.
- **Runner** `scripts/decoding/run_artifact_detection.py`, CPU only. Epochs come from the
  register (`--include-pending`, `--pending-only`). Output:
  `Processed-Data/artifact_intervals/<run_id>/{epoch_id}_artifact_intervals.parquet` (an
  empty file for a clean epoch) and `manifest.json` (per epoch: status, span, input row
  counts, constants, commit, masked seconds per consumer, input hashes, sha256). The run id
  is the commit's first 12 characters.
- **Inputs, read only:** `get_ripple_band_inputs` (the raw referenced CA1 LFP at the stored
  band's timestamps; the reference electrodes are the LFP's non-band columns, which is
  checked by electrode-id order), the position span `get_epoch_position_info` gives the SWR
  runner, and `get_hpc_marks` after the coincidence filter.
- **The SWR source:** `swr_source.artifact_intervals(epoch_id)` returns the `analysis`
  mask's intervals of the configured run, and `swr_observed_mask` /
  `swr_observed_stretches` leave that time out.

## Validation

The prototype's results on the 218 analysed epochs (report §0, §4):

| Set | Result |
|---|---|
| True SWRs (independent label: Kay ∩ Karlsson, duration, immobility, sharp wave, MUA) | 5 of 9,214 masked, all 5 visually T artifacts; **0 of 9,208 after adjudication** (95 % upper bound 0.033 %) |
| Hard true SWRs (top 5 % power, or ≥ 50 % of tetrodes in 1 ms) | 0 of 505 after adjudication |
| Visually confirmed SWRs (incl. 16 giant chimi SWRs) | 0 of 55 |
| Visually confirmed T | 95 of 111 (86 %) |
| Spike-only artifacts | 3 of 3 |
| S reference windows | 153 of 153 |
| Masked time, `swr` mask | 402 s (0.15 % of 77.25 h), in 74 epochs |
| Kay normalization | SD / robust SD of the 14 low-count epochs from 1.54–10.5 to 1.03–1.29 |

**Task 16's reproduction** (`task16_check/`, run `3e523b0a1cc6` at commit `3e523b0`; 19
epochs: the 5 pilots and the 14 other epochs with the most masked time in the prototype's
output):

- **Features.** On chimi20200225_02_r1 the S window power and specificity are bit-identical
  to the prototype's, and the T-ca1 counts are identical. The reference |J5| differs by at
  most 0.031 robust units, because the prototype cached it as float16.
- **Cores.** All 279 prototype cores are overlapped by a repo core. 282 of the 283 repo
  cores are overlapped by a prototype core.
- **Masked time** (`swr` mask): 290.11 s against 289.91 s. Every epoch is within 0.01 %,
  except chimi20200225_04_r2 at +0.20 s (+1.56 %).
- **Every difference comes from the OOD reference:**
  - 19 prototype T cores gain the `spk` signature, with no change in time;
  - 3 spike-only cores appear inside S blocks, with no change in time;
  - 1 new spike-only core (chimi20200225_04_r2, 699.01 s after the span's start): by eye a
    small step on several CA1 channels with a transient on both references and a spike
    line, over theta, not an SWR. It belongs to the small-step class the prototype missed
    (reference |J5| 4.3).

  With every mark in the reference, all 23 of these lines fall below the 0.2 OOD gate.
  Artifact marks near other steps, and S-noise marks inside S blocks, supported each other
  in the encoding-based reference.
- **Validation verdicts unchanged:**
  - true SWRs: 3 of 508 masked, the same 3 known label errors (visual T);
  - visual SWRs: 0 of 5;
  - visual T: 63 of 69 (the same 63);
  - visual S: 3 of 3;
  - S reference windows: 143 of 143.
- **Cost.** 65–83 s per epoch, CPU, and up to 207 s under I/O contention (median 74 s). Of an
  uncontended 72 s, loading takes 68 s (marks 45 s, LFP 16 s, position 7 s) and detection
  4–5 s. Worker peak RSS is 1.8–2.6 GiB. For 255 epochs: about 6 h on one worker, or 45 min
  on 8.
- **The other masks** on these epochs: the `decoder` mask is 304.3 s and the `analysis`
  mask 521.7 s. The 1 s after each LFP step nearly doubles the masked time of T-heavy
  epochs.

## Comparison with Spyglass

Measured on the same referenced LFP (report §7):

| Spyglass rule | Masked | True SWRs masked | Visual SWRs | Visual T | Spike-only | S windows |
|---|---|---|---|---|---|---|
| `default_difference` | 1,761 s | 46.0 % | 33 / 55 | 97 / 111 | 0 / 3 | 92 / 153 |
| `default_difference_ref` | 32 s | 2.6 % | 17 / 55 | 97 / 111 | 0 / 3 | 55 / 153 |
| `default_mad` | 454 s | 27.9 % | 22 / 55 | 103 / 111 | 0 / 3 | 131 / 153 |
| **This detector** | 402 s | 0 after adjudication | 0 / 55 | 95 / 111 | 3 / 3 | 153 / 153 |

- **Why its LFP rules mask SWRs.** Their thresholds are absolute raw units ("500" is about
  9 robust SD of a 3-sample difference here), and every rule that counts channels above an
  amplitude also counts SWRs. None looks for a signature SWRs lack.
- **Why they miss these artifacts.** No rule uses duration or channel specificity (S), and no
  LFP rule looks at spikes. Spyglass's ripple and decoding pipelines do not read artifact
  intervals at all.
- **Spike sorting.** The marks were sorted with `ampl_2000_prop_75` (> 2000 µV on ≥ 75 % of
  a tetrode's channels). It cannot catch S (one noisy channel per tetrode), and only 19 % of
  spike-line OOD marks exceed 2000 µV on any channel. Its z-score test z-scores across a
  tetrode's channels within a sample, so |z| can never exceed √3 and `zscore_thresh` = 5
  never fires; this is filed as Spyglass issue #1695.

## Field practice (literature review)

The literature check (23 papers and 3 code sources whose methods were read) found:

1. **The design matches field practice.** Rodent labs drop an event that also appears where an
   SWR cannot: a noise channel, a cortical electrode, or EMG estimated from the LFP
   (buzcode, FMAToolbox, Howe & Blair 2022, Deceuninck & Kloosterman 2024,
   Callaghan-Campbell & Murphy 2025). Our rules are the same idea.
2. **Our validation is stronger than any rodent paper read.** None reports how many true SWRs
   its artifact rule removes.
3. **Main gap: no direct check for licking or chewing EMG.** Study C1 tests it before R9.
4. **Padding is within the field range** (±5 ms to ±1 s) and is derived from measured ringing.
   The extra 1 s after a step matches the human-iEEG ±1 s convention.
5. **Removing masked time from rate exposure is field practice** (Chen et al. 2025).
6. **For sustained single-channel noise the field drops the channel.** R11 tests that.
7. **No amplitude cap,** documented above.
8. **T-spk must stay gated on OOD marks:** true SWRs reach 50 % of tetrodes in 200 µs, so a
   synchrony-only rule (such as Combinato's) would delete SWR spikes.
9. **Locomotion-only encoding** (Kay et al. 2020; Ciliberti et al. 2018) is a model choice we
   do not make; the methods state it.

## Known limits

- **16 of 111 visual T are missed.** They are small: reference |J5| 2.0–10.7, persistent
  steps on at most 3 channels. Each lowering tested costs real SWRs (`t_ref` 8, K = 12, or
  0.5-ms spike bins). C1 tests a raw-signal oscillation check on them.
- **One rater.** Visual review, the reference standard, was done by one reviewer with no
  second rater. The T set is a stratified sample of the artifact pool, so recall is for that
  mix. The true-SWR label has 6 known errors out of 9,214; its purity outside flagged events
  was checked on 30 random events.
- **Licking and chewing were not measured.** T-ref catches a lick artifact only if it reaches
  the reference as an abrupt jump. C1 is the check.
- **The cause of S is unknown:** fixed electrodes {0, 4, 8, 12, 80, 84, 88, 92} on senor,
  Nov 1–7 2020, which points to a headstage or connector channel.
- **The masked decode has not been run.** The effect of `is_missing` and the encoding
  exclusion is inferred from the library code (Task 17 runs one).
- **Time gaps.** Like the prototype, the features run on consecutive samples; a gap in the
  LFP's timestamps is not treated specially. Every non-finite sample is counted in the
  manifest (`n_invalid_samples`).
