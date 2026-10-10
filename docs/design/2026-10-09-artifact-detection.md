# Artifact detection: one mask, from signatures that SWRs lack

**Status (2026-10-09):** the project lead chose this design on 2026-10-09: one mask for every
stage, thresholds placed by one rule, and no post-step second. Implemented in
`src/continuum_swr_replay/analysis/hippocampus/artifacts/` and
`scripts/decoding/run_artifact_detection.py` (Task 16 of the full pipeline refresh,
[2026-10-03-full-pipeline-refresh-design.md](2026-10-03-full-pipeline-refresh-design.md)).
The SWR runner, MUA detection and both HPC decoders use the mask (Task 17a), and so does the
Widloski runner (Task 17b: the mask's bins are missing in its likelihood); tables and
analyses follow in Task 18. **Not yet applied:** `swr_source.ARTIFACT_INTERVALS_RUN_ID` stays
None until runbook R9 runs the detector on all epochs and sets it.

**Sources** (all under `.claude/research/refresh_artifact_detector/`):

- `detector_report.md` (the "report"): the prototype and its validation;
- `literature_review.md`;
- `followup/`: studies C1 (events at the ports, the tetrode rule), C2
  (`justification_report.md`: the threshold rule and the precedents read from source) and C3
  (`followup_c3_report.md`: how far a masked gap changes the decoders);
- `task16_check/`: the repo detector's checks against the prototype and against C2.

## Names: what the validation sets are, and are not

Neither validation set is ground truth.

- **The reference SWR set** is rule-defined: 9,208 events that meet all of
  - Kay ∩ Karlsson;
  - Kay duration 30–200 ms;
  - speed < 4 cm/s over ± 250 ms;
  - a sharp-wave deflection in the 30-Hz low-pass of the referenced CA1 LFP (median
    |deflection| ≥ 1 robust SD, ≥ 80 % sign agreement);
  - MUA ≥ 2× the epoch mean in ± 25 ms.

  No rule uses an artifact signature. 6 events that met the rules were labelled steps by the
  model reviewer and removed from the 9,214. The "hard" stratum (505 events) is the top 5 % by
  power, or ≥ 50 % of tetrodes in 1 ms.
- **Model-reviewed labels** come from one AI reviewer, with no human and no second rater:
  55 SWRs, 111 steps (T), 3 spike-only lines (SPK), 4 S, 5 small reference-only events, 1 pop
  and 8 unclear. The review pool was stratified, not random, so recall is for that mix.
- **S reference windows** (153 in 15 blocks) are defined from the named affected electrodes
  (their median ratio ≥ 3 for ≥ 2 windows), not from the S rule.
- **Giant-day events:** the 3,938 Kay ∩ Karlsson events of chimi Feb 13–14, the days with the
  largest SWRs.

The calibration SWR set of rule P (below) is the reference SWR set, the model-reviewed SWRs and
the giant-day events.

## Decision

- **Mask only on an artifact signature,** never on power alone. A large SWR is also a large,
  multichannel, high-frequency deflection on CA1, so any rule that counts channels above an
  amplitude also masks SWRs.
- **No rule can mask a ripple seen on one CA1 electrode alone**, by construction:
  - S needs at least 2 channels for at least 2 s;
  - T-ca1 needs at least 3 channels with a lasting step;
  - T-spk needs at least 40 % of tetrodes;
  - T-ref reads only the reference electrodes. It takes the maximum over them, so a jump on
    one reference electrode alone does fire it, as intended.
- **Robust units only.** Every threshold is a statistic over its own epoch MAD × 1.4826, per
  channel. No absolute LFP units: the stored raw LFP says "volts, conversion 1.0", which is
  suspect.
- **Thresholds by one rule (rule P, below).**
- **One mask for everything.** The detector stores unpadded cores and their evidence. One table
  (`masks.PADDING_S`) pads them into the mask. The same mask feeds:
  - the SWR detectors' normalization and event drops;
  - MUA detection;
  - the decoders' `is_missing` and encoding exclusion;
  - the Widloski runner and every analysis.
- **Artifact time is unobserved.**
  - An event of any kind that overlaps the mask leaves every cohort; it is not counted as
    non-SWR. `swr_source`'s coverage (`swr_observed_*`) excludes the mask.
  - The mask's time leaves every rate exposure, and every normalization leaves it out.
- **No post-step second.** No rodent lab we read pads after an artifact. The Buzsáki lab's
  `bz_FindRipples` drops flagged events and pads nothing. Ripple-band power does stay below its
  random-time level for at least 1 s after a step (report §8 item 4), so R8 reruns the headline
  results with 1 s masked after each step, as a robustness check.
- **The decoders' reach: 0.3 s (Task 18).** A decoded event (non-local event, ContFrag label,
  Widloski event) within 0.3 s of the mask is dropped. C3 injected 1,200 gaps into clean epochs:
  - no analysable (immobile) non-local event changed farther than 0.21 s from a gap;
  - the reach does not grow with gap length.
- **No reference veto (R-rip).** C1b's event veto (reference ripple envelope > 11.36 × the CA1
  median at the event's peak) removed 116 Kay events, most of them j16's. The tetrode rule
  (R-T, Task 17b) now handles those. One rule fewer is kept. Both are described in
  [2026-10-09-swr-tetrode-selection.md](2026-10-09-swr-tetrode-selection.md).
- **Keep the noisy tetrodes.** The sustained-noise (S) tetrodes are masked during their noise
  blocks, not dropped. R8 leaves out the 4 epochs with more than 2 % masked. R11 decodes the 12
  S epochs both ways (masked, and with those tetrodes dropped); masking stays unless it shows
  extra non-local probability within 3 s outside the S blocks.
- **No amplitude cap.** Some pipelines drop events above 9–10 SD as non-biological. Of the
  9,208 reference-set SWRs, 2,538 (27.6 %) have a Kay consensus peak above 9 robust z and
  2,181 (23.7 %) above 10; so do 24 and 23 of the 55 model-reviewed SWRs. These are robust
  units; in mean / SD units the counts are probably lower (the consensus SD is typically about
  1.15 robust SD).

## The signatures

| Kind | Signature | Rule (`detect.py`) |
|---|---|---|
| **S** | Ripple-band power raised on a *fixed subset* of CA1 channels for at least 2 s | `s_cores` |
| **T-ref** | An abrupt level jump on the raw reference electrode, which sits in white matter outside CA1 | `t_ref_events` |
| **T-ca1** | An abrupt *and persistent* level step on at least 3 CA1 channels at once | `t_ca1_events` |
| **T-spk** | A spike line: ≥ 40 % of tetrodes in one 200-µs bin, ≥ 20 % of those marks out of distribution | `t_spk_events` |

- **S.** For each 1-s window *w* and CA1 channel *c*, r[w, c] is the 150–250 Hz Welch power of
  the raw referenced LFP (15 Hann segments of 64 ms) over channel *c*'s epoch-median window
  power. The specificity spec[w] is the 2nd-largest r over channels divided by the median r.
  - An SWR raises ripple power on most CA1 tetrodes at once for under 0.3 s, so spec stays
    near 1. S raises it on a fixed subset for seconds.
  - A core is ≥ 2 consecutive windows with spec ≥ `S_CORE`. The block is the run of windows
    with spec ≥ 2 that holds it, and blocks ≤ 2 windows apart merge.
- **T-ref.** J5(t) = median(x[t+1..t+5]) − median(x[t−5..t−1]), a 5-ms median level jump. T-ref
  fires where the reference |J5| ≥ `T_REF` robust units. A step on the reference appears on
  every channel referenced to it; a hippocampal SWR reaches the white-matter reference only
  weakly.
- **T-ca1.** A channel's jump persists when |J5| ≥ `T_CA1_K` robust units and
  L20(t) = mean(x[t+5..t+24]) − mean(x[t−24..t−5]) has J5's sign with |L20| ≥ 0.5 |J5|. T-ca1
  fires where at least max(3, 0.15 × n channels) channels step persistently at the same sample.
  A step stays shifted for tens of ms; a ripple cycle or a sharp-wave onset does not.
- **T-spk.** Spike synchrony in 200-µs bins on two grids offset by half a bin, so a line across
  one grid's bin edge is whole on the other.
  - A mark is out of distribution (OOD) when its nearest reference mark of the same tetrode
    (Euclidean, raw marks in µV), its own event left out (the tetrode's marks within 1 ms of
    it, itself included; below), is more than 4 × the decoder's `waveform_std` (24 µV, read
    from `DecodingConfig`) away. There the decoder's mark kernel has no support (report §2.4).
  - Many tetrodes firing within 200 µs, with waveforms unlike any spike on their tetrode, is
    a mechanism no neural signal produces. The project's coincidence filter removes only lines
    within 40 µs.
- **Merging.** A T signature at most 50 ms after the previous one joins its core, so a core can
  chain past 50 ms. Each rule is its own function, so a further signature is one more event
  source.

### The OOD reference: the deliberate changes from the prototype

The prototype measured each mark against a fitted decoder's encoding marks
(`*_HPC_detector.pkl`). The decoders now use this mask, so the reference must not depend on a
decode. The repo uses, per tetrode, **all of the epoch's marks outside the S, T-ref and T-ca1
cores padded into the mask**, each spike's own event left out (below), in the prototype's mark
space and metric.

The encoding set it replaces was "all non-ripple time", which included the artifact spikes
around steps (report §8 item 8). Leaving those out makes some lines near steps OOD that the
prototype's reference called in distribution. In S epochs, S-noise marks inside their blocks
become OOD too. Task 16's checks measured both effects (below).

**A spike's own event is not its reference (added after R9).** A spike's nearest reference mark
is searched only among the same tetrode's reference marks more than 1 ms from it
(`T_SPK_OOD_OWN_EVENT_S`); the spike itself, at Δt = 0, falls under the same rule. A neuron
cannot fire twice within its absolute refractory period (about 1 ms), so two detections that
close on one tetrode are either two units, whose marks differ and are not each other's nearest
neighbour among the tetrode's ~10^5 marks, or one waveform detected twice. Leaving them out
takes away the support an artifact's twin detection gives it; on the 23 acceptance epochs it
also made 20 other spikes OOD, and no other core changed. R9 (run `a1083731940b`, gate report
§7) found the case: at 8.614 s in chimi20200227_02_r1 a spike line on 55 % of tetrodes was detected twice, about 0.15 ms apart, on five tetrodes; the
twins held each other in distribution, only 3 of 17 spikes were OOD (0.176 < 0.20), and T-spk
did not fire. With the own event left out, 9 of 17 are OOD (0.529) and the line is a spike-only
core. The 1 ms is a definition, not a margin, so rule P does not apply. SpikeInterface 0.99.1's
default refractory period is also 1 ms (`compute_refrac_period_violations`,
`get_potential_auto_merge`), and its `remove_duplicated_spikes` treats one unit's spikes within
0.3 ms as one spike detected twice. The prototype also let twins support each other (it left
out only the spike itself, `artdet_spikes.py`); it caught this line because its reference was the
decoder's encoding marks, which did not hold the twins. The search is exact: the k nearest
reference marks, with k = 1 + the most reference marks inside any spike's own event (4–10 on the
acceptance epochs). On the 23 acceptance epochs (Task 16's 22 and chimi20200227_02_r1) the new core is the only change:
every other core is identical, 2 of 619 line bins with ≥ 40 % of tetrodes move (this line on
both grids), and no reference-set SWR is newly masked. The search takes 1.3–2.0× as long as
before: 0.2–1.0 s more per epoch, 26 s more on j1620210709_10_r5 (3.1 M spikes).

## Thresholds: rule P

**Rule P (C2 §2).** An artifact threshold on a statistic SWRs also reach is
`THRESHOLD_MARGIN` = 1.5 × the largest value that statistic reaches in the calibration SWR set.
The value is measured where the rule could mask the event: the event widened by its padding.

- **Why 1.5.** One animal's calibration maximum, divided by the other four animals' maximum,
  reaches 1.27–1.59 (`c2_loao_ratio.csv`; T-ca1 1.59 and T-ref 1.27, both chimi). A cut placed
  on four animals with less margin failed on the fifth: placed without chimi, T-ref 13.8 and K
  15.7 masked 2 model-reviewed SWRs and 2 giant-day events on chimi (C2 §3.1).
- **Zero maximum.** If SWRs never reach the statistic, it is placed below the smallest labelled
  artifact.
- **Definitions are exempt.** Window lengths, bins, run lengths, persistence, merging and
  padding are placed by a mechanism or a measured tail.

In the code each rule-P threshold is the margin times a named calibration maximum
(`detect.py`):

| Constant | Calibration maximum | Shipped value | Prototype value (its margin) |
|---|---|---|---|
| `S_CORE` | `S_CORE_CALIBRATION_MAX` 2.8724 (a reference-set SWR, j16) | **4.3086** | 4 (1.39×) |
| `T_REF` | `T_REF_CALIBRATION_MAX` 11.7156 (a giant-day event; reference set 9.2, model-reviewed 7.2) | **17.5734** | 15 (1.28×) |
| `T_CA1_K` | `T_CA1_K_CALIBRATION_MAX` 16.6824 (chimi's giant day and a model-reviewed SWR) | **25.0236** | 20 (1.20×) |

The maxima are `followup/c2_rule_check.csv`'s rule values divided by 1.5.

### Justification table, at the shipped values

Adapted from C2 §1. "Sensitivity" is each constant at ± 25 % of its shipped value, one at a
time, scored by C2's rebuilt detector with the one mask on all 218 epochs
(`task16_check/round2/c2_sensitivity_shipped.csv`). At the shipped values that detector
masks 414.4 s, and of the calibration set it masks 0 of 9,208 reference-set SWRs, 0 of 3,938
giant-day events and 1 of 55 model-reviewed SWRs (the S-padding case, below).

| Rule · constant | What it detects | Why SWRs lack it | Field precedent | Placement | Margin | Held out (C2 §3.1) | Sensitivity (± 25 %) | Residual risk |
|---|---|---|---|---|---|---|---|---|
| **S** · `S_CORE` 4.31 | Ripple-band power rising on a fixed few CA1 channels for seconds | An SWR raises most CA1 tetrodes at once for < 0.3 s | PREP high-frequency noise, EEGLAB line noise, IBL `psd_hf`: a channel whose high-frequency power stands out from its population | Rule P | **1.5×** | P placed per split gave 4.13–4.31; 0 reference-set SWRs lost; S windows 55/55, 98/98, 153/153 | 3.23: +5.0 s; 5.39: −4.0 s; no SWR count changes, S windows 153/153 | Low. The cause of S is unknown (hardware) |
| S · `S_CORE_RUN` 2 windows, `S_EXT` 2, `S_GAP` 2 windows | That the rise lasts; partial windows; a step inside a block | SWRs last < 0.3 s | PREP `badTimeThreshold`; EEGLAB `max_broken_time`; FMAToolbox / buzcode hysteresis; gap merging (Chen 2025; Staresina 2023) | Definitions | — | — | (C2) run 1: +121 s and 11 reference-set SWRs masked | Low |
| **T-ref** · `T_REF` 17.57 | An abrupt level jump on the reference electrode | The white-matter reference sees SWRs only weakly | FieldTrip `ft_artifact_jump` (same detector family) | Rule P | **1.5×** | P without chimi failed on chimi, which sets the pooled 1.5 | 13.18: +14.8 s; 21.97: −6.8 s and 2 fewer steps; no SWR count changes | Low–medium: the maximum is a giant-day event |
| **T-ca1** · `T_CA1_K` 25.02 | A level step on several CA1 channels | Ripple cycles jump but do not persist; only giant onsets reach high \|J5\| | FieldTrip's jump-preserving median filter | Rule P | **1.5×** | as T-ref | 18.77: +5.8 s, 1 more step; 31.28: −9.8 s, 3 fewer steps; no SWR count changes | Medium: the largest cross-animal ratio (1.59) |
| T-ca1 · ρ 0.5 | That the step persists for 20 ms | A ripple cycle reverses within 3 ms | FieldTrip median filter | Mechanism | — | — | (C2) recall unchanged at ρ 0.375–0.625 | Low |
| T-ca1 · 0.15 of channels, at least 3 | Several channels at once | The calibration set steps on at most 1 channel | FieldTrip `cumulative`; PREP / EEGLAB fractions | Above the SWR count of 1 | 3–4× | Rule P gives 2; no loss | (C2) 0.1125: +5 s, 1 more step | Low |
| **T-spk** · `T_SPK_FRAC` 0.40 | A line of spikes on many tetrodes within 200 µs | Reference-set SWRs reach 0.364 in bins with OOD marks | Combinato: ≥ 50 % of channels in 3 ms (looser in time) | Between SWRs (≤ 0.364) and the lines (0.50–0.55); **off rule P** | 1.10× | P placed (0.53–0.55) catches 1 of 3 lines | (C2) 0.30: 2 reference-set SWRs and 1 model-reviewed SWR masked; 0.50: 4 fewer steps | **High:** thin margin and 3 labels; R9 reviews every spike-only core |
| T-spk · `T_SPK_OOD` 0.20 | …whose marks lack decoder support | The calibration set's maximum is 0 | none direct | Below the smallest line (0.27); **off rule P** | 1.35× below the lines | — | (C2) 0.25: 1 fewer step | Medium: 3 labels |
| T-spk · 4 × `waveform_std`, 200-µs bins on 2 grids | No kernel support; simultaneity | SWR marks are the place cells' own; SWR spikes spread over ms | Combinato 3 ms | Mechanism (the decoder's kernel floor); the prototype sweep (0.5 ms masks a giant SWR) | — | — | (C2) 5 × `waveform_std`: 2 of 3 lines | Medium |
| T · merge 0.05 s | One core per event | — | — | Definition | — | — | (C2) ± 25 %: no change | Low |
| Padding: see the next section | | | | Measured tails | | | | Low |

T-spk stays off rule P. Its SWR side reaches 0.364 and the lines sit at 0.50–0.55, so no value
has a 1.5× margin and still catches them. It masks about 5 s in total, and R9 adds labels by
reviewing every spike-only core.

## The mask

`masks.PADDING_S`, the one table:

| Core | Padding (each side) | The measurement behind it |
|---|---|---|
| S block | **1.0 s** | Noise spikes on the affected tetrodes leak up to 1 s past some block edges: within 0.5 s outside a block, a median of 1.0–1.3× baseline but 3–6× at the 75th percentile and up to 165× (`validation/S_edge_spike_rates.csv`) |
| T with an LFP signature (`ref` or `ca1`) | **0.3 s** | The stored ripple FIR has 535 taps (half-length 0.267 s). The consensus ringing ≥ 2 robust z around 425 cores has its 99th percentile at 316 ms; 0.3 s covers 96.8–100 % of held-out cores (`c2_padding_heldout.csv`). The burst of OOD spikes after a step reaches +269 ms (report §2.3) |
| Spike-only T | **0.1 s** | 98.8 % of OOD marks after a line lie within 100 ms (99th percentile 53 ms; C2 §1) |

- **Interval rules.** Intervals are closed, merged when they overlap or touch, and clipped to the
  epoch's span.
- **What the 1-s S padding costs.** The SWR detectors now lose the second around each S block
  too. One model-reviewed SWR lies 0.55 s before an S block (senor20201103_14_r7, 202.48 s),
  and the mask covers it.
  - With S padded 0 s, as the prototype's SWR mask did, it was kept.
  - Its S statistic over the event ± 1 s is 19.4; every other calibration event stays at the
    2.87 maximum. So rule P cannot remove this cost; only a smaller S padding could.
- **Unobserved samples are not masked.** A sample is unobserved when the stored band, the CA1
  LFP or a reference is not finite. No feature reads across one:
  - a sample whose J5 window (± 5 samples) or L20 window (± 24 samples) holds one has no step
    value;
  - a 1-s window holding one has no S value.

  The SWR source's coverage already leaves out non-finite samples of the stored band. Each
  record counts them (`n_invalid_samples`) and the windows without S power
  (`n_windows_without_power`).

## Field precedents, read from source

From C2 §1a (versions as fetched 2026-10-09) and the literature review:

- **FieldTrip `ft_artifact_jump`** (b68a51e).
  - Defaults: a median filter of order 9, absolute difference, per-channel z, maximum over
    channels > **20**, and no padding (`artpadding` 0).
  - It supports T-ref and T-ca1: a jump detector with a fixed cutoff.
  - Ours is in robust units (MADn), since SWRs and the artifacts themselves inflate an SD; its
    20 is not directly comparable.
  - Its median filter "preserves jumps", the same idea as our persistence test.
- **PREP `findNoisyChannels`** (678fead).
  - A channel's robust deviation z > 5, high-frequency noise z > 5, and maximum correlation
    with any other channel < 0.4 in > 1 % of 1-s windows.
  - It supports S (a channel standing out from its population) and the tetrode rule.
- **EEGLAB `clean_channels`** (2ee0662).
  - Correlation with a RANSAC reconstruction < 0.8 for > 40 % of the time; line noise z > 4.
  - It supports the same two rules.
- **IBL / SpikeInterface `detect_bad_channels`** (d7cdc97; IBL `voltage.py` 2203d0a).
  - "Dead" (similarity to the channel median < −0.5) and "noisy" (> 1, or `psd_hf` above
    threshold).
  - It supports S and the tetrode rule.
- **buzcode `bz_FindRipples`** (e615663; literature review).
  - It drops a detected event when a noise channel crosses its threshold inside the event, or
    EMG-from-LFP exceeds 0.9 at its start.
  - It drops flagged events and **pads nothing**.
- **Where these four differ from us:**
  - they drop whole channels, where we mask time;
  - they use correlation units, where we use robust z.
- **What no source covers:** none has a persistence test or an OOD spike-line rule, and
  FieldTrip's padding defaults are 0 s.

## Interface

- **`artifacts/features.py`:**
  - window spectra: `robust_scale`, `window_band_power`, `window_centers`,
    `channel_specificity`;
  - steps: `window_valid`, `level_jump` (J5), `level_shift` (L20), `robust_jump`,
    `reference_jump`, `persistent_step_count`;
  - spikes: `nearest_mark_distance`, `spike_lines`.
- **`artifacts/detect.py`:**
  - the margin, the calibration maxima, the constants and the four rules, plus `t_cores` and
    `spike_ood`;
  - `detect_artifacts(time, ca1_lfp, reference_lfp, spike_times, spike_marks,
    sampling_frequency, *, valid=None) -> pd.DataFrame`, which returns the unpadded cores.
- **`artifacts/masks.py`:** `PADDING_S`, `artifact_intervals(cores, span)` (the mask),
  `interval_mask(time, intervals)`, `subtract_intervals`, `masked_seconds`.
- **`artifacts/io.py`:**
  - the cores file (`CORE_COLUMNS`: `kind`, `signatures`, `core_start`, `core_end`, `spec_max`,
    `n_windows`, `ref_j5_max`, `ca1_frac_max`, `spk_frac_max`, `spk_ood_max`) and the manifest;
  - `read_epoch_cores` checks the manifest's run id, the epoch's status and the file's sha256.
- **The runner, `scripts/decoding/run_artifact_detection.py`:** CPU only.
  - Epochs come from the register (`--include-pending`, `--pending-only`).
  - It writes `Processed-Data/artifact_intervals/<run_id>/{epoch_id}_artifact_intervals.parquet`
    (empty for a clean epoch) and `manifest.json`.
  - Per epoch the manifest records the status, span, input row counts, timestamp gaps,
    unobserved samples, `t_ref_disabled`, the constants, the commit, the masked seconds, input
    hashes and the sha256.
  - The run id is the commit's first 12 characters.
- **Inputs, read only:**
  - `get_ripple_band_inputs`: the raw referenced CA1 LFP at the stored band's timestamps. The
    reference electrodes are the LFP's non-band columns, checked by electrode-id order.
  - The position span the SWR runner uses.
  - `get_hpc_marks`, after the coincidence filter.
- **Its readers:**
  - `swr_source.artifact_intervals(epoch_id)` returns the mask. `swr_observed_mask` and
    `swr_observed_stretches` leave it out.
  - `decoding_common.hpc_artifact_intervals` gives the same mask to the decoders and MUA.
  - The SWR runner pads the run's cores the same way.

## Validation

**The prototype**, on the 218 analysed epochs (report §0, §4; prototype constants, its SWR mask
with S padded 0 s):

| Set | Result |
|---|---|
| Reference SWR set | 0 of 9,208 masked (95 % upper bound 0.033 %) |
| Its hard stratum | 0 of 505 |
| Model-reviewed SWRs (incl. 16 giant chimi SWRs) | 0 of 55 |
| Model-reviewed steps (T) | 95 of 111 (86 %) |
| Model-reviewed spike-only lines | 3 of 3 |
| S reference windows | 153 of 153 |
| Masked time | 402 s (0.15 % of 77.25 h), in 74 epochs |
| Kay normalization | SD / robust SD of the 14 low-count epochs from 1.54–10.5 to 1.03–1.29 |

**C2's rebuilt detector at the shipped constants with the one mask**, all 218 epochs
(`task16_check/round2/c2_prediction_scores.csv`):

| Set | Result |
|---|---|
| Masked time | 414.4 s (392.3 s with S padded 0 s) |
| Reference SWR set | 0 of 9,208 |
| Model-reviewed SWRs | 1 of 55: the S-padding case above (0 of 55 with S padded 0 s) |
| Giant-day events | 0 of 3,938 |
| Model-reviewed steps | 94 of 111: rule P's T-ca1 loses 1 |
| Spike-only lines | 3 of 3 |
| S reference windows | 153 of 153 |

**Task 16's repo detector** (`task16_check/`, run `ea2047d55e81` at commit `ea2047d`, from a
clean tree). It covered 22 epochs: the 5 pilots, the 14 other epochs with the most masked time
in the prototype's output, and the 3 epochs that hold the spike-only lines. It was compared with
C2's rebuilt detector at the same constants and the one mask (`task16_check/round2/`):

- **Masked time:** 301.06 s against 300.66 s. 20 epochs agree to 0.0002 s, and every C2
  interval overlaps a repo interval.
- **Two epochs have one more repo interval each**, a spike-only core of 0.2 s; both come from
  the decode-free OOD reference:
  - chimi20200225_04_r2 at 699.0 s, a small step (accepted in fix round 1);
  - chimi20200225_06_r3 at 347.8 s, a large step on several CA1 channels. Rule P's
    `T_CA1_K` no longer flags its CA1 part. Its spike line (0.68 of tetrodes, 40 % OOD) is
    still flagged under the repo reference, but not under the prototype's
    (`round2/new_core_chimi20200225_06_r3_HPC_347.805.png`).
- **Verdicts identical to C2's** on these epochs:
  - reference SWR set: 0 of 902 masked (3 of the 905 as labelled: the known label errors, all
    model-reviewed steps);
  - model-reviewed SWRs: 1 of 7, the S-padding case;
  - steps: 63 of 69;
  - spike-only lines: 3 of 3;
  - S events: 3 of 3;
  - S reference windows: 143 of 143.
- **The SWR runner:** on the 7 epochs with an S block, padding S by 1 s instead of 0 s changes
  Kay's count from 1,017 to 1,018 (−1 to +1 per epoch), against 250 without a mask
  (`round2/kay_s_padding_effect.csv`).
- **Cost:** 61–96 s per epoch on CPU, almost all of it loading. The worker peak RSS is
  1.9–4.0 GiB; the largest is j1620210706_02_r1, with 2.5 M samples.
- **Determinism:** rerunning 17 of the epochs gave byte-identical cores files.

## Comparison with Spyglass

Measured on the same referenced LFP (report §7):

| Spyglass rule | Masked | Reference-set SWRs masked | Model-reviewed SWRs | Steps | Spike-only | S windows |
|---|---|---|---|---|---|---|
| `default_difference` | 1,761 s | 46.0 % | 33 / 55 | 97 / 111 | 0 / 3 | 92 / 153 |
| `default_difference_ref` | 32 s | 2.6 % | 17 / 55 | 97 / 111 | 0 / 3 | 55 / 153 |
| `default_mad` | 454 s | 27.9 % | 22 / 55 | 103 / 111 | 0 / 3 | 131 / 153 |
| **This detector (prototype constants)** | 402 s | 0 | 0 / 55 | 95 / 111 | 3 / 3 | 153 / 153 |

- **Why its LFP rules mask SWRs.**
  - Their thresholds are absolute raw units: "500" is about 9 robust SD of a 3-sample
    difference here.
  - Every rule that counts channels above an amplitude also counts SWRs, and none looks for a
    signature SWRs lack.
- **Why they miss these artifacts.**
  - No rule uses duration or channel specificity (S), and no LFP rule looks at spikes.
  - Spyglass's ripple and decoding pipelines do not read artifact intervals at all.
- **Spike sorting.** The marks were sorted with `ampl_2000_prop_75` (> 2000 µV on ≥ 75 % of a
  tetrode's channels).
  - It cannot catch S, which is one noisy channel per tetrode, and only 19 % of spike-line OOD
    marks exceed 2000 µV on any channel.
  - Its z-score test z-scores across a tetrode's channels within one sample, so |z| can never
    exceed √3 and `zscore_thresh` = 5 never fires. This is filed as Spyglass issue #1695.

## Field practice (literature review)

The literature check read the methods of 23 papers and 3 code sources. It found:

1. **The design matches field practice.** Rodent labs drop an event that also appears where an
   SWR cannot: a noise channel, a cortical electrode, or EMG estimated from the LFP (buzcode,
   FMAToolbox, Howe & Blair 2022, Deceuninck & Kloosterman 2024, Callaghan-Campbell & Murphy
   2025). Our rules are the same idea.
2. **None of the rodent papers read reports how many SWRs its artifact rule removes.**
3. **Licking and chewing (study C1).** There is no EMG and no lick sensor, so C1 tested the
   detected events at the ports, not licking itself.
   - The featureless events it found were mostly j16's. Three tetrodes carry them, and the
     tetrode rule (Task 17b,
     [2026-10-09-swr-tetrode-selection.md](2026-10-09-swr-tetrode-selection.md)) removes
     those tetrodes.
   - After that rule, the featureless share at the ports is 1.0 % (rewarded) and 4.8 %
     (unrewarded) over all animals. No visit-onset rule is added.
4. **Padding.**
   - The field ranges from ± 5 ms (Spyglass) to ± 1 s (human iEEG), and we found no rodent
     paper that derives its pad. Ours comes from measured tails.
   - The field does not pad after an artifact; the post-step second is an R8 check.
5. **Removing masked time from rate exposure is field practice** (Chen et al. 2025).
6. **For sustained single-channel noise the field drops the channel.** R11 tests that.
7. **No amplitude cap**, documented above.
8. **T-spk must stay gated on OOD marks.** SWRs reach 50 % of tetrodes in 200 µs, so a
   synchrony-only rule (such as Combinato's) would delete SWR spikes.
9. **Locomotion-only encoding** (Kay et al. 2020; Ciliberti et al. 2018) is a model choice we
   do not make; the methods state it.

## Known limits

- **17 of 111 model-reviewed steps are missed** at the shipped constants.
  - The prototype's 16 are small: reference |J5| 2.0–10.7, persistent steps on at most 3
    channels. Rule P's higher T-ca1 threshold misses one more.
  - Lowering a threshold costs SWRs, and a raw-oscillation rule (C1) was fitted on the misses
    themselves.
- **The labels are not ground truth.** The reference SWR set is rule-defined; its purity
  outside flagged events was checked on 30 random events. The model-reviewed labels come from
  one AI reviewer with no human and no second rater.
- **T-spk rests on 3 labelled lines** with a 1.10× margin. R9 reviews every spike-only core.
- **The decode-free OOD reference** makes S-noise marks OOD inside S blocks, so a line near an
  S block could become a spike-only core. R9 lists spike-only cores within 1.1 s of an S core.
- **The cause of S is unknown:** fixed electrodes {0, 4, 8, 12, 80, 84, 88, 92} on senor,
  Nov 1–7 2020, which points to a headstage or connector channel.
- **The 1-s S padding masks one model-reviewed SWR** (above).
- **The masked decode has not been run** yet (Task 17a's acceptance runs one).
- **Time gaps.** Like the prototype, the features run on consecutive samples; a gap in the
  LFP's timestamps is not treated specially. Each record counts them (`n_time_gaps`, steps
  longer than 1.5 sample periods), so R9 can find the affected epochs.
- **No reference electrode.** An epoch without one outside CA1 runs with T-ref disabled, and its
  record says `t_ref_disabled: true`.
