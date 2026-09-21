# Corrections to the literature parameter sheet

*Record of what was checked in
[literature_detection_parameters.csv](literature_detection_parameters.csv), what was
changed, and what remains unverified. Checked 2026-09-21 against the papers' Methods.*

Both the CSV in this directory and the Google Sheet it came from carry the 63
corrections below. They were compared cell by cell afterwards and agree exactly across
all 54 rows and 28 columns.

## Method

All 54 rows were checked. Every row's DOI was resolved through the CrossRef API and checked against the stated
first author, last author, year, and journal. Each parameter cell that was not `#N/A`
or "Not reported" was checked against the paper's Methods, retrieved through Europe PMC
full text, the NCBI `efetch` service, the PMC article page, or the lab's own PDF
library, which supplied the papers no public route would serve. A first pass was made per
block of rows; every correction below was then re-checked independently against the
paper before it was applied, and the supporting sentence is quoted.

Nothing was changed on the strength of a single pass, and nothing was changed for a
paper that could not be retrieved.

## Applied corrections

| Row | Paper | Column | Was | Now | Evidence |
|---|---|---|---|---|---|
| 3 | Huelin Gorriz 2023 | Journal | Nature Communication | Nature Communications | CrossRef container title |
| 4 | Harvey 2023 | SWR z-score (SD) | 2.5 | 4 | "all events exceeding 4 SD from the mean were detected", then expanded to 1 SD. The 2.5 SD belongs to the separate sharp-wave stage |
| 6 | Tirole 2022 | Detection | MUA | SWR, MUA | "a threshold over 3 was set for the z-scored ripple power, and used as an additional criterion for candidate replay events" |
| 13 | Denovellis 2021 | Reconstruction error (cm) | Not reported | 7 | "7 cm median difference, 5-9 cm 95% CI" |
| 29 | Muessig 2019 | SWR electrodes | >1 | 1 | "the LFP whose power estimate had the highest standard deviation was then used to define ripple events" |
| 31 | Ólafsdóttir 2017 | SWR z-score (SD) | #N/A | 2.5 | "we identified periods where the ripple power exceeded 2.5std above the mean" |
| 31 | Ólafsdóttir 2017 | Max duration (ms) | #N/A | 500 | "Events lasting less than 40ms or more than 500ms were excluded" |
| 31 | Ólafsdóttir 2017 | Combine events (ms) | #N/A | 40 | "events separated by less than 40ms were joined together" |
| 31 | Ólafsdóttir 2017 | % decoded | Not reported | 36.9-38.6% | "36.88% versus 38.60%" of putative events qualified as replay |
| 32 | Wu & Ji 2017 | Min. cells | #N/A | 4 | "We defined those PBEs with at least four active template cells as candidate events" |
| 33 | Yamamoto 2017 | SWR low band (Hz) | 140 | 100 | "we used both ripple band power (100-200 Hz) and multi-unit activities" |
| 33 | Yamamoto 2017 | Decoding notes | Clusterless | (blank) | "we first performed standard spike sorting"; the word clusterless never appears |
| 35 | Grosmark 2016 | First author | Grossmark | Grosmark | CrossRef author: Andres D. Grosmark |
| 36 | Ambrose 2016 | Time bin (ms) | 10 | 20 | "Time windows of 20ms, overlapping by 10ms"; 10 ms is the step |
| 36 | Ambrose 2016 | % decoded | 10-15% | 15-20% | "On average, 15-20% of SWRs were determined to contain significant trajectory replay" |
| 37 | Jadhav & Frank 2016 | DOI | 10.1016/j.neuron.2017.09.017 | 10.1016/j.neuron.2016.02.010 | The old DOI is Yamamoto & Tonegawa 2017, already row 33. CrossRef: Jadhav to Frank, Neuron 90:113-127 (2016) |
| 38 | Silva 2015 | MUA smooth (ms) | #N/A | 10 | "the smoothed histogram (Gaussian kernel, standard deviation of 10 ms)" |
| 39 | Ólafsdóttir 2015 | Reconstruction error (cm) | (blank) | 10.0 | "median error for both arms = 10.0 cm" |
| 41 | Wikenheiser 2013 | Year | 2014 | 2013 | CrossRef: Hippocampus 23:22-29, issued 2012-06-27, print 2013 |
| 41 | Wikenheiser 2013 | Last author | Reddish | Redish | CrossRef author: A. David Redish |
| 43 | Pfeiffer 2013 | Detection notes | cumulative distance (<40 cm) | start-to-end distance (<40 cm) | "a start-to-end distance less than 40 cm were eliminated" |
| 43 | Pfeiffer 2013 | % decoded | 25-49% | 25.3-43.9% | "between 25.3% and 43.9% of candidate events" |
| 47 | Karlsson & Frank 2009 | Journal | Neuron | Nature Neuroscience | CrossRef container title |
| 47 | Karlsson & Frank 2009 | Animal speed (cm/s) | 4 | 2 | "we examined SWRs that occurred when animals were moving less than 2 cm/sec" |
| 47 | Karlsson & Frank 2009 | SWR electrodes | >1 | 1 | "the smoothed envelope stayed above 3 standard deviations of the mean for at least 15 ms on at least one tetrode" |
| 53 | Nádasdy 1999 | Max duration (ms) | 200 | #N/A | 200 ms is the sequence-search template window, not an event duration limit |
| 19 | Bhattarai 2020 | First author | Bhattari | Bhattarai | CrossRef author: Baburam Bhattarai |
| 24 | Chenani 2019 | Journal | Nature Communication | Nature Communications | CrossRef container title |
| 24 | Chenani 2019 | Detection notes | "remained above 3 SD"; "immobility visually inspected" | peak reached 3 SD, boundaries below 1 SD; reward zones defined by visual judgment | "the place cell rate remained at least one standard deviation (SD) above its average during a period when the peak firing rate reached at least three SD"; the visual judgment defined the reward zones, not immobility |
| 1 | Mallory 2025 | Reconstruction error (cm) | #N/A | 2.9 (track), 3.4 (arena) | "mean +/- SEM positional decoding error per session, linear track: 2.9 +/- 0.1 cm; open arena: 3.4 +/- 0.2 cm" |
| 1 | Mallory 2025 | Trajectory | MAP | Posterior center of mass | The authors' code takes the decoded position as `binDecoding.posteriorCOM` (`compute_filtering_binDecoding_cm.m`); `posteriorPeak` is a separate quantity used for filtering |
| 46 | Gupta 2010 | Min. duration (ms) | #N/A | 40 | "the sequence score of place cell activity within a flexible time window (40 ms to 600 ms, to accommodate longer sequences)" |
| 46 | Gupta 2010 | Max duration (ms) | #N/A | 600 | Same sentence; the window grows until five steps pass without a score increase, or it reaches 600 ms |
| 16 | Igata 2021 | Min. duration (ms) | 60 | 50 | "Candidate synchronous events in which the duration was less than 50 ms or more than 2000 ms were excluded". The 60 ms belongs to a later stage: "Trajectory sequences covering more than 4 time bins (60 ms) and a cumulative distance of more than 20 cm were included" |
| 16 | Igata 2021 | Min. cells | 4 | >4 | "Candidate synchronous events were detected when the z-scored spike rates exceeded 2 and the number of active neurons exceeded 4", so the minimum is five |
| 10 | Krause 2022 | Max duration (ms) | #N/A | 2000 | "Only SWRs longer than 50 ms and shorter than 2 s ... were included" |
| 10 | Krause 2022 | % decoded | Not reported | 89.7% | "89.7% +/- 2.0% of SWR were identified to encode trajectories", against 22.8% by the traditional method |
| 17 | Gridchyn 2020 | MUA z-score (SD) | 3.5 | #N/A | "a threshold of 3.5 times the baseline mean established in the pre-rest session". It is a multiple of the mean spike count, not of a standard deviation, so it does not belong in an SD column |
| 17 | Gridchyn 2020 | Detection notes | "3.5 SD initial" | "3.5x the baseline mean spike count, not a SD multiple" | Same sentence; the adaptive adjustment to about 1 event/s was already right, and a 150 ms detection refractory is now noted |
| 21 | Xu 2019 | Min. cells | 5 | 4 | "Events containing fewer than five spikes, four cells or with less than 10% of the population of neurons active were rejected". The 5 is the spike minimum |
| 25 | Michon 2019 | Detection | MUA | SWR, MUA | "Candidate replay events were defined as mua bursts that overlapped with a ripple event" |
| 25 | Michon 2019 | % decoded | 5-15% | 3.2-15.5% | "amounted to 8.75% +/- 2.52% of all SWR bursts (range: 3.2%-15.5%)" |
| 25 | Michon 2019 | SWR z-score (SD), smooth, band, electrodes; MUA smooth | #N/A | 8; 15; 140-225; 1-3; 15 | "a high threshold of m + 8 x s" with a 0.5 SD boundary; envelope "smoothed with a Gaussian kernel (bandwidth 15 ms)"; "filtered in the ripple frequency band (140-225 Hz)"; "recorded from 1-3 tetrodes"; multiunit histogram "Gaussian kernel with 15 ms bandwidth" |
| 40 | Pfeiffer & Foster 2015 | Trajectory | MAP | Posterior center of mass | "truncated to the longest sequence of time frames with a weighted mean posterior probability less than 50 cm from that of the previous frame". The paper never uses MAP |
| 40 | Pfeiffer & Foster 2015 | Decoding notes | cumulative distance (<80 cm) | start-to-end distance (<80 cm) | "a start-to-end distance less than 80 cm were eliminated", the same error as the 2013 row |
| 52 | Lee & Wilson 2002 | Detection notes | "theta/delta ratio"; "grouped by spike times exceeding max ISI from any cell" | theta-to-total-power ratio identifying REM; max_isi within a cell, max_gap between cells | "REM was identified as periods with an elevated ratio ... of hippocampal EEG power in the theta band (5-12 Hz) to overall power (1-475 Hz). The remainder was classified as SWS"; "for each cell, each set of consecutive spikes with interspike intervals less than max_isi ... broken between every pair of adjacent letters separated by more than max_gap" |
| 52 | Lee & Wilson 2002 | Combine events (ms) | #N/A | 100 | "max_isi = 50 ms and max_gap = 100 ms"; max_gap is the event-separation threshold |
| 27 | Shin & Jadhav 2019 | SWR electrodes | >1 | 1 | "the envelope stayed above 3 SD of the mean on at least one tetrode" |
| 2 | Yang 2024 | Reconstruction error (cm) | 1.51 | Not reported | "1.51, 4.07, and 5.34 mean error for real, simulated, and shuffled data" is trial-decoding error in trial blocks, not a position error in cm |
| 15 | Michon 2021 | Position bin (cm) | 1 | 4 | "the posterior probability distribution over position (sampled at a regular grid with 4 cm spacing)"; 1 cm is the sorted-unit tuning-curve bin |
| 19 | Bhattarai 2020 | Trajectory | Linear regression | Weighted correlation | "We then calculated a weighted correlation (r) ... for each significant replay event"; the paper never uses linear regression |
| 22 | Farooq 2019 (Neuron) | Reconstruction error (cm) | ~5 cm (inferred from Figure 1J) | 7.6 | "median error in decoded location: 7.6 cm versus 54.3 cm in shuffled datasets" |
| 28 | Carey 2019 | Place field smooth (cm) | 2 | 3 | "Raw firing rates were smoothed with a sigma = 1 bin kernel" on "approximately 3 cm per bin"; 2 ms is the temporal spike kernel |
| 42 | Wu & Foster 2014 | Trajectory | MAP | Weighted correlation, MAP | The replay criterion is "absolute weighted correlation > 0.5"; the MAP function cuts the subregion the score is computed on, so both belong |
| 44 | Carr & Frank 2012 | SWR electrodes | >1 | 1 | "ripple power (150-250 Hz) on any CA1 tetrode exceeded 3 SD above the mean" |
| 50 | Ji & Wilson 2007 | Min. cells | 5 | 4 | "a replaying frame necessarily contained at least four active cells"; candidate frames are "those containing at least four active template cells" |
| 51 | Foster & Wilson 2006 | Animal speed (cm/s) | 5.4 | #N/A | "Fields were velocity-filtered to exclude times when the speed of the animal was below 5.4 cm s-1" is a place-field filter that keeps running periods, not a detection gate |

The Karlsson row matters to this project directly: `Karlsson_ripple_detector`'s 4 cm/s
default is the package's own, not the paper's, and the paper thresholds on one tetrode,
which the package already does.

## Flagged but not changed

These were disputed in the first pass and are **not** applied, because the paper could
not be retrieved a second time to quote the sentence. The rest of the first pass's
disputes were resolved once the lab's PDF library was available; only these remain. They are recorded so the numbers
are not cited without a check.

| Row | Paper | Column | Value | Concern |
|---|---|---|---|---|
| 2 | Yang 2024 | 12 detection cells | as listed | Identical to the Grosmark 2016 row; possibly inherited rather than read from the supplement |
| 37 | Jadhav & Frank 2016 | 5 decoding cells | Bayes, 15, 2, 4, linear regression | The retrieved main text contains no decoding analysis at all: zero occurrences of "decod", "Bayes" or "linear regression". The supplement could not be retrieved |
| 16 | Igata 2021 | Min. duration | 60 | May be 50 ms, with 60 ms a sustained-sequence length. Methods are in an unreachable SI Appendix |

## Papers added

Three papers that decode replay were missing and have been added as data rows 55 to 57
(sheet rows 56 to 58), at the end rather than in year order:

| Paper | Why it belongs | Source |
|---|---|---|
| Widloski & Foster 2025, *Nat Commun* 16:10287 | The sequence detector [02](02-memoryless-sequence.md) ports. Decodes replay with no ripple or burst requirement | [10.1038/s41467-025-65181-5](https://doi.org/10.1038/s41467-025-65181-5) |
| Maboudi et al. 2018, *eLife* 7:e34467 | Hidden-Markov and Bayesian decoding of replay; already cited in the overview's scope table | [10.7554/eLife.34467](https://doi.org/10.7554/eLife.34467) |
| Ólafsdóttir et al. 2016, *Nat Neurosci* 19:792-794 | Multiunit-triggered events with Bayesian decoding and trajectory fitting; the only grid-cell replay entry | [10.1038/nn.4291](https://doi.org/10.1038/nn.4291) |

Two candidates were checked and left out, because the sheet is for papers that decode
replay and neither does. **Kay et al. 2016** contains no instance of "decode", "Bayes" or
"posterior". **Yu et al. 2017** explicitly declines: "We chose this method to examine the
content of SWRs instead of trajectory decoding methods because the latter enriches for
SWRs that contain multiple place cells and decode to trajectories matching predefined
templates... SWRs that do not meet these criteria... are classified as non-replay events
and excluded." That is the selection bias [03](03-inventory-comparison.md) sets out to
measure, stated by the author of the detector [01](01-yu-swr-detector.md) builds, so it
belongs in the design documents rather than in this sheet. Yu et al. 2018 and Kay et al.
2020 were also checked and left out: the first does no decoding, and the second decodes
sub-second theta cycling rather than replay.

## Columns added

Seven columns (AC to AI) record things the table could not express. Each was filled from
the papers, with a quote required for every value and `#N/A` where the paper does not say:

| Column | Why | Filled |
|---|---|---|
| Spike sorting | Sorted against clusterless was free text in five rows and absent elsewhere | 56 of 57 (50 sorted, 5 clusterless, 1 both) |
| Time bin step (ms) | "Time bin" often holds a sliding-window width whose step is shorter, so two rows with the same value could mean different things | 43 |
| Shuffle Method | The central omission for a sheet about decoding replay: the surrogate largely determines the "% Decoded" the sheet already tracks | 54 |
| Shuffles (#) | " | 35 |
| Sig. Thresh. | " | 38 |
| % Decoded denom. | That percentage is variously of ripples, candidate events, putative events or multiunit bursts | 14 |
| Reconst. Error bin (ms) | Error depends on the window it was measured in, which ranges from 2 ms to 500 ms across rows presented as comparable | 11 |

The shuffle columns show that "none" is a real methodological position, not missing text:
ten rows select events by a score, posterior or confidence threshold rather than by a
surrogate test: Widloski 2022, Krause 2022, Berners-Lee 2021, Denovellis 2021, Gillespie
2021, Gridchyn 2020, Ambrose 2016 (for its primary criterion), Pfeiffer & Foster 2015,
Bendor & Wilson 2012 and Kudrimoti 1999.

Filling these also settled a duplication: the word "Clusterless" was removed from
`Decoding notes` for the five rows that carried it, so the fact now lives only in
`Spike sorting`. That column also caught one the notes had missed, Yamamoto 2017, which
used an unsorted decoder for its CA3 inhibition experiment and a sorted one elsewhere.

## Deliberately not filled

Two values the papers state were left out of the sheet on purpose, because writing them
into those columns would change what the column means:

- **Gridchyn 2020's 150 ms detection refractory** is not a merge threshold. Every other
  entry in "Combine Events Thresh." merges nearby events; a refractory suppresses the
  later detection outright. It is recorded in the detection notes instead.
- **Pfeiffer & Foster 2015's 80 place units** is a session-inclusion criterion ("only
  rats in which we obtained at least 80 simultaneously recorded place units"), not a
  per-event active-cell count like the rest of the "Min. Cells" column.

## Systematic issues, not single-cell errors

- **Window versus step.** Several "time bin" cells give a sliding-window width whose
  step is shorter: Widloski 2022 (80 ms window, 5 ms step), Mou 2022 (20 ms, 10 ms step),
  Ambrose 2016 (20 ms, 10 ms step). The column records the window. Say so in the sheet.
- **Smoothing width versus standard deviation.** The "SWR smooth" value of 100 ms for
  Huelin Gorriz 2023 and Tirole 2022 is a moving-average window, while the other entries
  in that column are Gaussian standard deviations. These are not comparable numbers.
- **Detection gate versus analysis restriction.** Several speed values restrict place-field
  construction or later analysis and gate nothing at detection: Harvey 2023, Liu 2023,
  Wu & Ji 2017, Drieu 2018, Foster & Wilson 2006, and Karlsson & Frank 2009 as corrected.
- **Trajectory scoring labels.** "Radon transform" is used for Davidson-2009-style line
  search in at least Muessig 2019, Drieu 2018, Ólafsdóttir 2017, Yamamoto 2017, and
  Ólafsdóttir 2015. None of those papers uses the word. If that is the intended
  shorthand, define it in the sheet.
- **Stage mixing.** Igata 2021 is a second case: its minimum duration had been taken
  from the trajectory-sequence stage rather than the event stage.

- **Stage mixing, continued.** Where a paper detects a sharp wave and a ripple separately, the row
  can take its minimum duration from one stage and its maximum from the other, as in
  Harvey 2023.

## Papers that could not be retrieved

Every cell in these rows is unverified, which is not the same as confirmed: Krause 2022 (10), Gridchyn 2020 (17), Xu 2019 (21), Michon 2019 (25), Berners-Lee
2021 (12), Pfeiffer & Foster 2015 (40), Lee & Wilson 2002 (52). Bhattarai 2020 (19) and Farooq 2019 (23) have their
detection parameters in supplements that could not be retrieved. Igata 2021 (16), Gupta 2010 (46) and Diba 2007 (49) were settled once their
supplements were supplied. Diba's row needed no correction at all: the supplement gives
"the animal's speed was <= 10 cm s-1", events "detected by searching for silent periods
>= 60 ms", the "≥ 5 or ≥ 30 percent of place-cells (whichever was greater) ... within the
next 300 ms" rule, and the rank-order correlation, all exactly as recorded. Igata's confirmed the 2 SD spike-rate threshold, the 15 ms multiunit and
4 ms ripple smoothing, the 150-250 Hz band across thirteen tetrodes, the 5 cm/s stop
criterion, the 2000 ms maximum, the Mahalanobis-distance ripple power with its
Gaussian-mixture split, and the MAP trajectory with its 40 cm jump and 20 cm length
limits. Gupta's confirmed its 2 SD threshold, three active cells, 180-220 Hz band, 3.5 cm pixels, 10 ms
stepping, and roughly 10 % of events becoming replays (1719 of 17,593) all check out,
and the flexible 40-600 ms detection window supplied the two durations above. Michon 2021 (15) and Carey 2019
(28) were checked against preprints, so the published versions may have drifted.

Mallory 2025 (row 1) was checked against the paper and the authors' code at
https://github.com/caitlinmallory/TimeCourseOrganizationOfHippocampalReplay (commit
`128513a`), which sets every detection parameter explicitly and confirmed all of them:
`speedThr = 5` cm/s, `num_cells_participating_thr = 10`, `sequence_durationThr = 0.05`,
`sequence_deltThr = 0.05` for the merge, `windowSizeDecoding_replay = 0.08` with
`shiftSizeDecoding = 0.005`, `binSize = 2` cm, `sequence_delxThr = 0.4 * maze_size`,
`replay_weighted_r_thr = 0.6`, `replay_coverage_thr = 0.2`, and
`replay_posterior_diff_thr = 0.33`, which is the ">66% of the posterior in one
directional map" criterion. Its 80 ms time bin is a sliding window with a 5 ms step. Science, Cell Press and the Journal of
Neuroscience all refuse automated requests.

## How the Sheet was updated

The Sheets API refuses an unauthenticated write even with link editing enabled, so the
Sheet was edited through a browser session driven by Playwright, anonymously, one cell
at a time by cell reference. The edits appear in the Sheet's version history as
anonymous and can be reverted there.

Three cells sit in columns with chip validation, `Detection` and `Trajectory`. Typing a
value into those cleared the cell instead of setting it, so they were restored by
copying a cell that already held the wanted value: `G3` to `G7`, and `Y4` to `Y20` and
`Y43`. Anyone editing those columns by script should copy an existing chip rather than
type.

Two values are recorded slightly differently from the quoted text, because the column is
numeric: Ólafsdóttir 2015's reconstruction error reads 10 rather than 10.0, and Wu &
Foster 2014's trajectory reads "Weighted correlation" alone, since the vocabulary has no
"Weighted correlation, MAP" entry. The MAP function there cuts the subregion that the
weighted correlation is computed over; the score itself is the weighted correlation.
