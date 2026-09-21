# Detector comparison table

*Reference table for the replay-detection comparison. Kept current with what is
implemented in `ripple_detection` and in this project; a row's status says which.*
*Governing rule and cross-cutting decisions: [overview.md](overview.md).*

One row per detection method considered for the comparison, including the ones excluded,
so a reader can see what each method needs, what it thresholds, and where it is. "Family"
follows the trigger taxonomy in the overview: LFP-triggered, rate-triggered,
posterior-triggered, or joint. Nothing here ranks the methods; that is what
[04](04-event-features.md) is for.

| method | family | reference | input | signal | normalization | threshold rule | duration / merge | speed / state | needs | status |
|---|---|---|---|---|---|---|---|---|---|---|
| **Kay** (`Kay_ripple_detector`) | LFP, consensus | Kay et al. 2016 | ripple-band LFP, all CA1 tetrodes | √(smoothed Σ envelope²), 4 ms Gaussian | z-score, whole epoch or mask | ≥ 2 SD sustained, extended to the mean | ≥ 15 ms; no merge | endpoints ≤ 4 cm/s | multiple tetrodes helpful, one suffices | in package; inventory exists (`SWR_event`) |
| **Karlsson** (`Karlsson_ripple_detector`) | LFP, per-channel | Karlsson & Frank 2009 | ripple-band LFP per tetrode | per-channel envelope, 4 ms Gaussian | z-score per channel | ≥ 3 SD on any channel, extended to that channel's mean; overlapping channel events merged | ≥ 15 ms | endpoints ≤ 4 cm/s | ≥ 1 tetrode | in package; needs a run |
| **Yu** (`Yu_ripple_detector`) | LFP, consensus | Yu et al. 2017 | ripple-band LFP, all CA1 tetrodes | median of per-tetrode z-scored 4 ms-smoothed envelopes | immobility mean/SD | 99.99th percentile of the mode-mirrored immobility histogram (data-driven), sustained, extended to the immobility mean | ≥ 20 ms (samples); no merge | endpoints ≤ 4 cm/s in package, then canonical whole-event | ≥ 1 tetrode; enough immobility to resolve 1e-4 | in package (PR merged); inventory to build ([01](01-yu-swr-detector.md)) |
| **Zugaro** / FindRipples (`Zugaro_ripple_detector`) | LFP, single or summed | Hirase; Zugaro (FMAToolbox); Csicsvari et al. 1999 as ancestor | ripple-band LFP, summed across channels | squared, 11-sample moving average at 1250 Hz | z-score | two thresholds: bounded at 2 SD, kept if peak > 5 SD | 20–100 ms; merge within 30 ms if merged span < 100 ms | endpoints (package addition; original has none) | ≥ 1 channel | in package (PR #16) |
| **Long** / DetectSWR (`Long_sharp_wave_ripple_detector`) | LFP, two-layer | Long (buzcode `bz_DetectSWR`) | **raw** LFP: pyramidal-layer channel + stratum radiatum channel | sharp-wave difference (2–50 Hz DoG, ripple − radiatum) and smoothed rectified 80–250 Hz ripple band | local ±5 s median/SD per candidate | k-means on (sharp wave, ripple power) pairs per 40 ms block, cluster percentile cuts, then local peak ≥ median + 2.5 SD on both | sharp-wave bounds at median + 0.5 SD; SW 20–500 ms or ripple ≥ 25 ms; ≥ 50 ms apart | endpoints (package addition) | a stratum radiatum channel | in package (PR #16); **not runnable on our tetrodes** |
| **Roumis** (`Roumis_ripple_detector`) | LFP, per-channel mean | none cited | ripple-band LFP | mean over channels of √(smoothed envelope²) | z-score | ≥ 2 SD sustained, extended to the mean | ≥ 15 ms | endpoints | ≥ 1 channel | in package; not in the comparison |
| **Shvartsman** (`Shvartsman_ripple_detector`) | LFP, per-channel with participation | none cited | ripple-band LFP per channel | per-channel envelope | z-score or manual per channel | ≥ 3 SD per channel; merged event needs a minimum fraction of channels | ≥ 15 ms | ≥ 50 % of samples ≤ 4 cm/s | multiple channels | in package; not in the comparison |
| **MUA HSE** (`multiunit_HSE_detector`) | rate | this package's convention; Davidson et al. 2009 and Grosmark 2016 as parameter variants | multiunit spike matrix | summed rate, 15 ms Gaussian | z-score, all samples or mask | ≥ 2 SD sustained, extended to the mean | ≥ 15 ms | endpoints ≤ 4 cm/s | spikes, sorted or clusterless | in package; inventory exists (`MUA_event`) |
| **Carey candidate** (`Carey_candidate_detector`) | joint LFP × rate | Carey, Tank & van der Meer 2019 (vandermeerlab `GenCandidateEvents`) | ripple-band LFP + multiunit matrix; optional theta channel | geometric mean of a ripple envelope score (10 ms Gaussian, rescaled to mean 1) and a capped, baseline-subtracted multiunit score | z-score of the joint score | bounded at 1 SD, kept if peak > 3 SD | ≥ 20 ms | event contained in low-speed and low-theta intervals; ≥ 5 active units | spikes and LFP | in progress |
| **State-space non-local** (ours) | posterior | Denovellis et al. 2021 lineage; this project | clusterless marks + position | 4-state posterior on the 2 ms grid | — | non-local probability > 0.8 for ≥ 15 ms | whole-event immobility | canonical rule | encoding model | exists (`non_local_event`) |
| **Memoryless sequence** (Widloski port) | posterior | Widloski & Foster 2025 | clusterless marks | 80 ms uniform-prior windows, 4 ms stride | — | spread < 10 cm, jump < 20 cm, duration > 100 ms, dispersion > 12 cm, encoding-pair shuffle p < 0.05 | merge < 20 cm and < 50 ms | per-bin < 4 cm/s then whole-event | encoding model | to build ([02](02-memoryless-sequence.md)) |

## Methods considered and not ported

| method | source | why not |
|---|---|---|
| `amSWR` spectral template score | vandermeerlab | needs a per-session template from manually identified ripples (`SWRfreak`) |
| `getSWR` | vandermeerlab | a parameter variant of the Zugaro rule: envelope instead of squared power, merge before the length filter, 140–200 Hz |
| Control-channel power difference | vandermeerlab `SWR_detection.m`; recommended by Liu et al. 2022 | artefact handling, excluded from the package by decision; available to the comparison as a deferred control since the corpus-callosum reference tetrodes exist |
| `DetectRipplelets` | neurocode (Yamamoto & Tonegawa 2017) | post-processing: bundles ripples within 200 ms; equivalent to 03's fragment-merge recount at a 200 ms gap |
| `eventSpikingTreshold`, `removeArtifactsFromEvents`, EMG exclusion | neurocode | post-filters and artefact rejection |
| `find_HSE`, `find_rippleHSE` | neurocode | parameter variants of the MUA detector (peak ≥ 3 SD, 50–500 ms) and an overlap cell 03 already produces |
| Maboudi et al. 2018 HMM, Yu & Frank 2021 theta coactivity | — | need sorted units or the wrong epochs (overview scope table) |

## Reading the table

- **Signal** and **threshold rule** are where the methods differ structurally; two rows with
  the same numbers in those columns are the same method with different defaults.
- **Speed / state** records what each method does itself; the comparison then applies the
  canonical whole-event rule to every inventory (overview X4).
- The consensus statement on SWR detection (Liu et al. 2022, Nat Commun 13:6000) asks that
  detection criteria be reported in full and that results be shown across a range of
  parameters; this table is the first half of that, and [04](04-event-features.md) the
  second.
