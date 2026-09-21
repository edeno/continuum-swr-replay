# Detector comparison table

*Reference table for the replay-detection comparison. Kept current with what is
implemented in `ripple_detection` and in this project; a row's status says which.*
*Governing rule and cross-cutting decisions: [overview.md](overview.md).*

One row per detection method considered for the comparison, including the ones excluded,
so a reader can see what each method needs, what it thresholds, and where it is. "Family"
follows the trigger taxonomy in the overview: LFP-triggered, rate-triggered,
posterior-triggered, or joint. Nothing here ranks the methods; that is what
[04](04-event-features.md) is for.

Tingley & Peyrache (2020) argue that the field's central methodological problem is not
any one method but "the degree of variability in how they are applied, both at the level
of data preprocessing and in the assumptions that each method makes with respect to the
data", with discrepancies in methodology contributing to differences in interpretation.
Their suggested remedies are direct comparison of multiple methods, a shared code base,
and a taxonomy of methods. This table is the taxonomy, `ripple_detection` is the shared
code base, and [03](03-inventory-comparison.md) is the direct comparison.

| method | family | reference | input | signal | normalization | threshold rule | duration / merge | speed / state | needs | status |
|---|---|---|---|---|---|---|---|---|---|---|
| **Kay** (`Kay_ripple_detector`) | LFP, consensus | Kay et al. 2016, [doi:10.1038/nature17144](https://doi.org/10.1038/nature17144) | ripple-band LFP, all CA1 tetrodes | √(smoothed Σ envelope²), 4 ms Gaussian | z-score, whole epoch or mask | ≥ 2 SD sustained, extended to the mean | ≥ 15 ms; no merge | endpoints ≤ 4 cm/s | multiple tetrodes helpful, one suffices | in package; inventory exists (`SWR_event`) |
| **Karlsson** (`Karlsson_ripple_detector`) | LFP, per-channel | Karlsson & Frank 2009, [doi:10.1038/nn.2344](https://doi.org/10.1038/nn.2344) | ripple-band LFP per tetrode | per-channel envelope, 4 ms Gaussian | z-score per channel | ≥ 3 SD on any channel, extended to that channel's mean; overlapping channel events merged | ≥ 15 ms | endpoints ≤ 4 cm/s (the package default; the paper restricted analysis to < 2 cm/s and applied no speed gate at detection) | ≥ 1 tetrode | in package; needs a run |
| **Yu** (`Yu_ripple_detector`) | LFP, consensus | Yu et al. 2017, [doi:10.7554/eLife.27621](https://doi.org/10.7554/eLife.27621) | ripple-band LFP, all CA1 tetrodes | median of per-tetrode z-scored 4 ms-smoothed envelopes | immobility mean/SD | 99.99th percentile of the mode-mirrored immobility histogram (data-driven), sustained, extended to the immobility mean | ≥ 20 ms (samples); no merge | endpoints ≤ 4 cm/s in package, then canonical whole-event | ≥ 1 tetrode; enough immobility to resolve 1e-4 | in package (PR merged); inventory to build ([01](01-yu-swr-detector.md)) |
| **Zugaro** / FindRipples (`Zugaro_ripple_detector`) | LFP, single or summed | Hirase; Zugaro (FMAToolbox), no paper; ancestor rule in Csicsvari et al. 1999 as ancestor | ripple-band LFP, summed across channels | squared, 11-sample moving average at 1250 Hz | z-score | two thresholds: bounded at 2 SD, kept if peak > 5 SD | 20–100 ms; merge within 30 ms if merged span < 100 ms | endpoints (package addition; original has none) | ≥ 1 channel | in package (PR #16) |
| **Long** / DetectSWR (`Long_sharp_wave_ripple_detector`) | LFP, two-layer | Long (buzcode `bz_DetectSWR`), no paper | **raw** LFP: pyramidal-layer channel + stratum radiatum channel | sharp-wave difference (2–50 Hz DoG, ripple − radiatum) and smoothed rectified 80–250 Hz ripple band | local ±5 s median/SD per candidate | k-means on (sharp wave, ripple power) pairs per 40 ms block, cluster percentile cuts, then local peak ≥ median + 2.5 SD on both | sharp-wave bounds at median + 0.5 SD; SW 20–500 ms or ripple ≥ 25 ms; ≥ 50 ms apart | endpoints (package addition) | a stratum radiatum channel | in package (PR #16); **not runnable on our tetrodes** |
| **Roumis** (`Roumis_ripple_detector`) | LFP, per-channel mean | none cited | ripple-band LFP | mean over channels of √(smoothed envelope²) | z-score | ≥ 2 SD sustained, extended to the mean | ≥ 15 ms | endpoints | ≥ 1 channel | in package; not in the comparison |
| **Shvartsman** (`Shvartsman_ripple_detector`) | LFP, per-channel with participation | none cited | ripple-band LFP per channel | per-channel envelope | z-score or manual per channel | ≥ 3 SD per channel; merged event needs a minimum fraction of channels | ≥ 15 ms | ≥ 50 % of samples ≤ 4 cm/s | multiple channels | in package; not in the comparison |
| **MUA HSE** (`multiunit_HSE_detector`) | rate | this package's convention; Davidson et al. 2009 ([doi:10.1016/j.neuron.2009.07.027](https://doi.org/10.1016/j.neuron.2009.07.027)) and Grosmark & Buzsáki 2016 ([doi:10.1126/science.aad1935](https://doi.org/10.1126/science.aad1935)) as parameter variants | multiunit spike matrix | summed rate, 15 ms Gaussian | z-score, all samples or mask | ≥ 2 SD sustained, extended to the mean | ≥ 15 ms | endpoints ≤ 4 cm/s | spikes, sorted or clusterless | in package; inventory exists (`MUA_event`) |
| **Carey candidate** (`Carey_candidate_detector`) | joint LFP × rate | Carey, Tanaka & van der Meer 2019, [doi:10.1038/s41593-019-0464-6](https://doi.org/10.1038/s41593-019-0464-6) (vandermeerlab `GenCandidateEvents`) | ripple-band LFP + multiunit matrix; optional theta channel | geometric mean of a ripple envelope score (10 ms Gaussian, rescaled to mean 1) and a capped, baseline-subtracted multiunit score | z-score of the joint score | bounded at 1 SD, kept if peak > 3 SD | ≥ 20 ms | event contained in low-speed and low-theta intervals; ≥ 5 active units | spikes and LFP | in package (PR #17) |
| **State-space non-local** (ours) | posterior | Denovellis et al. 2021 lineage, [doi:10.7554/eLife.64505](https://doi.org/10.7554/eLife.64505); this project | clusterless marks + position | 4-state posterior on the 2 ms grid | — | non-local probability > 0.8 for ≥ 15 ms | whole-event immobility | canonical rule | encoding model | exists (`non_local_event`) |
| **Memoryless sequence** (Widloski port) | posterior | Widloski & Foster 2025, [doi:10.1038/s41467-025-65181-5](https://doi.org/10.1038/s41467-025-65181-5) | clusterless marks | 80 ms uniform-prior windows, 4 ms stride | — | spread < 10 cm, jump < 20 cm, duration > 100 ms, dispersion > 12 cm, encoding-pair shuffle p < 0.05 | merge < 20 cm and < 50 ms | per-bin < 4 cm/s then whole-event | encoding model | to build ([02](02-memoryless-sequence.md)) |

## Methods considered and not ported

| method | source | why not |
|---|---|---|
| `amSWR` spectral template score | vandermeerlab | needs a per-session template from manually identified ripples (`SWRfreak`) |
| `getSWR` | vandermeerlab | a parameter variant of the Zugaro rule: envelope instead of squared power, merge before the length filter, 140–200 Hz |
| Control-channel power difference | vandermeerlab `SWR_detection.m`; recommended by Liu et al. 2022 | artifact handling, excluded from the package by decision; available to the comparison as a deferred control since the corpus-callosum reference tetrodes exist |
| `DetectRipplelets` | neurocode (Yamamoto & Tonegawa 2017, [doi:10.1016/j.neuron.2017.09.017](https://doi.org/10.1016/j.neuron.2017.09.017)) | post-processing: bundles ripples within 200 ms; equivalent to 03's fragment-merge recount at a 200 ms gap |
| `eventSpikingTreshold`, `removeArtifactsFromEvents`, EMG exclusion | neurocode | post-filters and artifact rejection |
| `find_HSE`, `find_rippleHSE` | neurocode | parameter variants of the MUA detector (peak ≥ 3 SD, 50–500 ms) and an overlap cell 03 already produces |
| Maboudi et al. 2018 HMM ([doi:10.7554/eLife.34467](https://doi.org/10.7554/eLife.34467)), Yu & Frank 2021 theta coactivity ([doi:10.1371/journal.pbio.3001393](https://doi.org/10.1371/journal.pbio.3001393)) | — | need sorted units or the wrong epochs (overview scope table) |

## Literature parameter survey

[literature_detection_parameters.csv](literature_detection_parameters.csv) is a lab-compiled
table of 57 replay papers that decode replay content (1999–2025; source: the shared Google Sheet
`12mMgiYWO7tKRhs5qhMXMZ5RggGi1J24wxIkcs4knTUI`, exported 2026-09-21; column headers
flattened to one line). **The values have been checked against the papers and corrected;
see [literature_parameter_corrections.md](literature_parameter_corrections.md)** for the
corrections with their evidence, the cells still flagged, the papers that could not be
retrieved, and the systematic issues (sliding-window width recorded as a time bin,
smoothing width against standard deviation, analysis restrictions recorded as detection
gates). The Sheet and the CSV carry the same values and were compared cell by cell; the
Sheet is sorted by year, so cite a row by paper rather than by position.
It records each paper's trigger
(MUA 23, SWR 17, both 13, decoding 4), thresholds, smoothing, bands, durations, and
decoding settings, and since 2026-09-21 also the shuffle and significance procedure, the
sliding-window step, whether decoding was sorted or clusterless, the denominator behind
"% decoded", and the bin at which reconstruction error was measured. Two uses here:

**Sweep values for 04's threshold robustness.** The literature range bounds what a
"modest relaxation and tightening" means. Reported values after correction (count with a value, then min / median / max;
`#N/A` and "not reported" excluded). Several speed values gate analysis rather than
detection, so that row overstates how many papers gate detection on speed:

Counts are of rows holding a bare number, so entries such as ">4", "33%" and
"lower quartile" are excluded rather than coerced; an earlier version of this table
counted ">1" as 1 and "33%" as 33.

| parameter | n | min | median | max | most common |
|---|---|---|---|---|---|
| SWR z-score threshold (SD) | 27 | 1 | 3 | 8 | 3, 2, 4 |
| MUA z-score threshold (SD) | 27 | 2 | 3 | 4 | 3, 2, 4 |
| speed gate (cm/s) | 42 | 0.05 | 5 | 10 | 5, 4, 2 |
| SWR smoothing (ms) | 18 | 4 | 12.5 | 100 | 4, 12.5, 15 |
| MUA smoothing (ms) | 28 | 5 | 15 | 30 | 15, 10, 5 |
| ripple band (Hz) | 30 | 80–180 low, 200–300 high | 150–250 | | 150–250 (16 rows) |
| minimum duration (ms) | 41 | 15 | 50 | 100 | 50, 100, 15 |
| maximum duration (ms) | 25 | 400 | 600 | 2000 | 500, 2000, 750 |
| merge gap (ms) | 14 | 20 | 50 | 100 | 50, 20, 40 |
| minimum active cells | 27 | 3 | 5 | 10 | 5, 4, 3 |

Our retained settings (Kay z = 2, 4 ms smoothing, 15 ms minimum, 4 cm/s; HSE z = 2,
15 ms minimum) sit at the permissive end on threshold and minimum duration, so the
"tightening" arm of the robustness plan (z = 3, 50 ms) is the literature's median, not an
extreme. The 04 pilot should draw its values from this table and say so.

**Coverage of the literature by the package.** Most rows are parameter variants of
rules already in `ripple_detection`: a z-score on a ripple-band envelope, single or
consensus channel (Kay / Karlsson family), or a z-score on smoothed population rate
(`multiunit_HSE_detector`), gated by speed. Rows the package covers as distinct
algorithms: Liu & Fernandez-Ruiz 2023 (k-means on ripple power and radiatum sharp wave,
`Long_sharp_wave_ripple_detector`), Carey 2019 (`Carey_candidate_detector`), Widloski
2022 and Mallory 2025 (posterior-triggered, the memoryless port). Yu 2017 is not in the
sheet. Recurring rules the package does **not** express, in order of how often they
appear:

| rule | papers | package | status |
|---|---|---|---|
| Maximum event duration | 25 papers, 400–2000 ms | `maximum_duration` on every detector | **added** (PR #21); applies to the reported event, while `minimum_duration` still applies to the run above threshold |
| Minimum active cells | 27 papers, 3–10 | `minimum_active_units` on `multiunit_HSE_detector` and Carey | **added** (PR #21); every burst now reports `n_active_units` |
| A ripple band other than 150–250 Hz | 14 of the 30 papers stating one | `band` on `filter_ripple_band` and `ripple_bandpass_filter` | **added** (PR #21) |
| Merge events separated by a short gap | 14 papers, 20–100 ms | `merge_close_events` | **added** (PR #21); `exclude_close_events` remains the drop-the-later convention |
| Require a ripple and a burst together | 13 papers (trigger "SWR, MUA") | `require_overlap` | **added** (PR #21); composes any two inventories |
| Extend the event from the detection threshold down to a lower boundary threshold (1 SD, 0.5, 0.15 of peak) | Harvey 2023, Yamamoto 2017, Bhattarai 2020, Michon 2021, Mou 2022 | Kay extends to 0; Yu extends to the immobility mean; Zugaro uses `low_threshold` | not settable on Kay/Karlsson/HSE |
| Theta/delta ratio or manual NREM scoring as the state gate | Lee 2002, Wikenheiser 2014, Grosmark 2016, Drieu 2018, Farooq 2019, Liu 2019, Muessig 2019, Yang 2024 | Carey's optional `theta_lfp` z < 2 | no theta/delta helper |
| Burst preceded by a silent period | Diba 2007, Ólafsdóttir 2015, Liu 2019, Bhattarai 2020 | none | absent |
| Adaptive threshold tuned to a target event rate | Gridchyn 2020 (1 event/s) | Yu's noise-percentile is data-driven but rate-agnostic | absent |

The five added rules were parameter additions to existing detectors rather than new
algorithms, and every default is unchanged, so no inventory built before PR #21 is
affected. The four remaining rules are open; none is needed for the six inventories.

## Public detector code surveyed (2026-09-21)

A web survey of published or widely used SWR detectors, made to decide what else the
package should carry. Verdicts: **duplicate** of a package rule, **not portable** (trained
weights, GPL transcription only, manual template, online-only), or **candidate** (a
distinct rule worth reimplementing from its description). Nothing below is required for
the six inventories.

| detector | source | verdict | reason |
|---|---|---|---|
| RippleNet (Hagen 2021), cnn-ripple / rippl-AI (Navas-Olive 2022, 2024), `ripple_detector_CNN` | CINPLA, PridaLab, ywatanabe1989 | not portable | value is in trained weights; cnn-ripple and `ripple_detector_CNN` are GPL-3 |
| bz_FindRipples, CellExplorer `ce_FindRipples`, Tingley & Buzsáki 2020, Fernández-Ruiz 2019, pynacollada tutorial | buzcode, CellExplorer, Peyrache lab | duplicate | parameter variants of the Zugaro rule (bz defaults 130–200 Hz, 20 ms minimum); artifact parts excluded by decision |
| neurodsp `detect_bursts_dual_threshold` (Feingold 2015) | neurodsp | duplicate | dual threshold on median-normalized amplitude; only novelty is a minimum in cycles |
| buzcode `detect_swr.m` (the path Liu et al. 2022 cite) | buzcode | duplicate | the path Liu et al. 2022 cite is dead, but the file still exists in buzcode under `analysis/SharpWaveRipples/detect_swr/`, alongside `bz_DetectSWR.m`, which is already the Long detector |
| Norman / LBCN-Stanford `ripple_detection` (human iEEG) | LBCN | not portable | Zugaro-like core; distinctive parts are IED and noise-channel artifact rejection |
| Falcon (Ciliberti & Kloosterman 2017), Open Ephys ripple-detector plugin (de Sousa 2022) | Falcon on Bitbucket (`kloostermannerflab/falcon-core`, GPL-3); open-ephys-plugins (no license) | not portable as-is | online-only, GPL / unlicensed C++; block-RMS with calibration baseline is a threshold variant |
| Zeng & Eden state-space ripple-replay detector | IEEE Trans Biomed Eng 73(1):153-167 (2026), [doi:10.1109/TBME.2025.3578583](https://doi.org/10.1109/TBME.2025.3578583); preprint arXiv:2502.05394 | not portable | needs a decoder; no code released |
| Sebastian 2023 `Topological_SWR` | PridaLab | not a detector | waveform classifier |
| RIPPLELAB / mne-hfo HFO detectors (Staba, Gardner line-length, Crépon, MNI) | epilepsy tools | low priority | different features (line length, rectified-peak count); mne-hfo is BSD-3 if ever wanted |
| `eventSpikingTreshold.m` (M. Valero, 2019; code only, no paper) | neurocode | **candidate** | post-hoc unit-firing validation of ripples (z-scored 10 ms population rate ≥ 0.5 around the peak); the spike check Liu et al. 2022 recommend; composes with any detector |
| `find_rippleHSE.m` (Grosmark & Buzsáki 2016 definition, [doi:10.1126/science.aad1935](https://doi.org/10.1126/science.aad1935)) | neurocode | **candidate** | HSE ∩ ripple by interval intersection; the sheet's "MUA with at least one LFP with SWR" rows (Grosmark 2016, Yang 2024); differs from Carey's geometric-mean score |
| Consensus-vote wrapper (Ruelas 2026, Front Syst Neurosci) | no code released | **candidate** | keep events flagged by ≥ k of several detectors, union bounds; license-free idea |
| Sharp-wave-gated threshold with amplitude ceiling (Callaghan-Campbell & Murphy 2025) | AngCamp/CampbellMurphy2024 | **candidate** | radiatum sharp-wave z > 1 **and** 3 < ripple z < 10; threshold-only counterpart to Long's k-means; the sheet's Harvey 2023 row |
| Offline emulation of causal detectors (Dutta, Ackermann & Kemere 2019; de Sousa 2022) | shayokdutta/RippleDetectionAnalysis, SpikeGadgets/RippleDetector (GPL-3) | **candidate, low** | causal FIR envelope, N-of-M channels within 15 ms, lockout; only for online-vs-offline benchmarking |

The survey also flagged two citation risks; both were checked and are already handled:
the Long docstring cites neurocode `SharpWaveRipples/DetectSWR.m` alongside the buzcode
path, and the Zugaro defaults (2/5 SD, 30 ms inter-ripple interval, 20–100 ms) are the
FMAToolbox `FindRipples` values it cites first.

## References

Verified against CrossRef on 2026-09-21; each DOI was resolved. Entries with no
DOI are code with no accompanying publication, and are cited by file and commit.

| Work | DOI |
|---|---|
| Kay et al. (2016) *Nature* 531(7593):185-190 | [10.1038/nature17144](https://doi.org/10.1038/nature17144) |
| Karlsson & Frank (2009) *Nat Neurosci* 12(7):913-918 | [10.1038/nn.2344](https://doi.org/10.1038/nn.2344) |
| Yu et al. (2017) *eLife* 6:e27621 | [10.7554/eLife.27621](https://doi.org/10.7554/eLife.27621) |
| Csicsvari et al. (1999) *J Neurosci* 19(1):274-287 | [10.1523/JNEUROSCI.19-01-00274.1999](https://doi.org/10.1523/JNEUROSCI.19-01-00274.1999) |
| Carey, Tanaka & van der Meer (2019) *Nat Neurosci* 22(9):1450-1459 | [10.1038/s41593-019-0464-6](https://doi.org/10.1038/s41593-019-0464-6) |
| Davidson, Kloosterman & Wilson (2009) *Neuron* 63(4):497-507 | [10.1016/j.neuron.2009.07.027](https://doi.org/10.1016/j.neuron.2009.07.027) |
| Widloski & Foster (2025) *Nat Commun* 16:10287 | [10.1038/s41467-025-65181-5](https://doi.org/10.1038/s41467-025-65181-5) |
| Widloski & Foster (2022) *Neuron* 110(9):1547-1558 | [10.1016/j.neuron.2022.02.002](https://doi.org/10.1016/j.neuron.2022.02.002) |
| Mallory, Widloski & Foster (2025) *Science* 387(6733):541-548 | [10.1126/science.ads4760](https://doi.org/10.1126/science.ads4760) |
| Liu et al. (2022) *Nat Commun* 13:6000, the SWR detection consensus statement | [10.1038/s41467-022-33536-x](https://doi.org/10.1038/s41467-022-33536-x) |
| Denovellis et al. (2021) *eLife* 10:e64505 | [10.7554/eLife.64505](https://doi.org/10.7554/eLife.64505) |
| Gillespie et al. (2021) *Neuron* 109(19):3149-3163 | [10.1016/j.neuron.2021.07.029](https://doi.org/10.1016/j.neuron.2021.07.029) |
| Maboudi et al. (2018) *eLife* 7:e34467 | [10.7554/eLife.34467](https://doi.org/10.7554/eLife.34467) |
| Yu & Frank (2021) *PLOS Biol* 19(9):e3001393 | [10.1371/journal.pbio.3001393](https://doi.org/10.1371/journal.pbio.3001393) |
| Grosmark & Buzsáki (2016) *Science* 351(6280):1440-1443 | [10.1126/science.aad1935](https://doi.org/10.1126/science.aad1935) |
| Yamamoto & Tonegawa (2017) *Neuron* 96(1):217-227 | [10.1016/j.neuron.2017.09.017](https://doi.org/10.1016/j.neuron.2017.09.017) |
| Navas-Olive et al. (2022) *eLife* 11:e77772 | [10.7554/eLife.77772](https://doi.org/10.7554/eLife.77772) |
| Navas-Olive et al. (2024) *Commun Biol* 7:211 | [10.1038/s42003-024-05871-w](https://doi.org/10.1038/s42003-024-05871-w) |
| Hagen et al. (2021) *Neuroinformatics* 19(3):493-514, RippleNet | [10.1007/s12021-020-09496-2](https://doi.org/10.1007/s12021-020-09496-2) |
| Dutta, Ackermann & Kemere (2019) *J Neural Eng* 16(1):016009 | [10.1088/1741-2552/aae90e](https://doi.org/10.1088/1741-2552/aae90e) |
| de Sousa et al. (2022) *J Neural Eng* 19(4):046040 | [10.1088/1741-2552/ac857b](https://doi.org/10.1088/1741-2552/ac857b) |
| Ciliberti & Kloosterman (2017) *J Neural Eng* 14(4):045004, Falcon | [10.1088/1741-2552/aa7526](https://doi.org/10.1088/1741-2552/aa7526) |
| Feingold et al. (2015) *PNAS* 112(44):13687-13692 | [10.1073/pnas.1517629112](https://doi.org/10.1073/pnas.1517629112) |
| Ruelas et al. (2026) *Front Syst Neurosci* 20:1822457 | [10.3389/fnsys.2026.1822457](https://doi.org/10.3389/fnsys.2026.1822457) |
| Callaghan-Campbell & Murphy (2025) *Sci Data* 12:1841 | [10.1038/s41597-025-06115-0](https://doi.org/10.1038/s41597-025-06115-0) |
| Sebastian et al. (2023) *Nat Neurosci* 26(12):2171-2181 | [10.1038/s41593-023-01471-9](https://doi.org/10.1038/s41593-023-01471-9) |
| Zeng & Eden (2026) *IEEE Trans Biomed Eng* 73(1):153-167 | [10.1109/TBME.2025.3578583](https://doi.org/10.1109/TBME.2025.3578583) |
| Norman et al. (2019) *Science* 365(6454):eaax1030 | [10.1126/science.aax1030](https://doi.org/10.1126/science.aax1030) |
| Tingley & Buzsáki (2020) *Neuron* 105(1):138-149 | [10.1016/j.neuron.2019.10.012](https://doi.org/10.1016/j.neuron.2019.10.012) |
| Tingley & Peyrache (2020) *Phil Trans R Soc B* 375:20190231 | [10.1098/rstb.2019.0231](https://doi.org/10.1098/rstb.2019.0231) |
| van der Meer, Kemere & Diba (2020) *Phil Trans R Soc B* 375:20190238 | [10.1098/rstb.2019.0238](https://doi.org/10.1098/rstb.2019.0238) |
| Fernández-Ruiz et al. (2019) *Science* 364(6445):1082-1086 | [10.1126/science.aax0758](https://doi.org/10.1126/science.aax0758) |
| neurocode release (AYA lab, 2023) | [10.5281/zenodo.7819979](https://doi.org/10.5281/zenodo.7819979) |
| Widloski & Foster code release (2025) | [10.5281/zenodo.15199609](https://doi.org/10.5281/zenodo.15199609) |

Code with no paper: FMAToolbox `FindRipples.m` (H. Hirase's algorithm, M. Zugaro's
implementation; no license file, GPL-3 in the file headers), buzcode `bz_DetectSWR.m`
(J. D. Long II; GPL-3), neurocode `eventSpikingTreshold.m` (M. Valero; no license
file), and E. Stark's `detect_hfos` filtering routines, which survive only inside
buzcode. The per-paper detection settings in
[literature_detection_parameters.csv](literature_detection_parameters.csv) carry a
DOI per row.

## Reading the table

- **Signal** and **threshold rule** are where the methods differ structurally; two rows with
  the same numbers in those columns are the same method with different defaults.
- **Speed / state** records what each method does itself; the comparison then applies the
  canonical whole-event rule to every inventory (overview X4).
- The consensus statement on SWR detection (Liu et al. 2022, Nat Commun 13:6000) asks that
  detection criteria be reported in full and that results be shown across a range of
  parameters; this table is the first half of that, and [04](04-event-features.md) the
  second.
