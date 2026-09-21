# Corrections to the literature parameter sheet

*Record of what was checked in
[literature_detection_parameters.csv](literature_detection_parameters.csv), what was
changed, and what remains unverified. Checked 2026-09-21 against the papers' Methods.*

The CSV in this directory carries the 30 corrections below. **The Google Sheet it was
exported from does not**, and has to be edited by hand; the "Sheet edits" table is
written for that.

## Method

All 54 rows were checked. Every row's DOI was resolved through the CrossRef API and checked against the stated
first author, last author, year, and journal. Each parameter cell that was not `#N/A`
or "Not reported" was checked against the paper's Methods, retrieved through Europe PMC
full text, the NCBI `efetch` service, or the PMC article page. A first pass was made per
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
| 24 | Chenani 2019 | Detection notes | "remained above 3 SD"; "immobility visually inspected" | peak reached 3 SD, boundaries below 1 SD; reward zones defined by visual judgement | "the place cell rate remained at least one standard deviation (SD) above its average during a period when the peak firing rate reached at least three SD"; the visual judgement defined the reward zones, not immobility |
| 27 | Shin & Jadhav 2019 | SWR electrodes | >1 | 1 | "the envelope stayed above 3 SD of the mean on at least one tetrode" |

The Karlsson row matters to this project directly: `Karlsson_ripple_detector`'s 4 cm/s
default is the package's own, not the paper's, and the paper thresholds on one tetrode,
which the package already does.

## Flagged but not changed

These were disputed in the first pass and are **not** applied, because the paper could
not be retrieved a second time to quote the sentence. They are recorded so the numbers
are not cited without a check.

| Row | Paper | Column | Value | Concern |
|---|---|---|---|---|
| 2 | Yang 2024 | Reconstruction error | 1.51 | May be a trial-block decoding error rather than a position error in cm. Supplement paywalled |
| 2 | Yang 2024 | 12 detection cells | as listed | Identical to the Grosmark 2016 row; possibly inherited rather than read from the supplement |
| 42 | Wu & Foster 2014 | Trajectory | MAP | The reported score may be weighted correlation |
| 44 | Carr & Frank 2012 | SWR electrodes | >1 | May be one tetrode, as in Karlsson & Frank |
| 50 | Ji & Wilson 2007 | Min. cells | 5 | May be 4 |
| 51 | Foster & Wilson 2006 | Animal speed | 5.4 | May be a place-field construction filter rather than a detection gate |
| 37 | Jadhav & Frank 2016 | 5 decoding cells | Bayes, 15, 2, 4, linear regression | The retrieved main text contains no decoding analysis at all: zero occurrences of "decod", "Bayes" or "linear regression". The supplement could not be retrieved |
| 15 | Michon 2021 | Position bin | 1 | The preprint decodes on a 4 cm grid; 1 cm may be the sorted-unit tuning-curve bin. Published version paywalled |
| 16 | Igata 2021 | Min. duration | 60 | May be 50 ms, with 60 ms a sustained-sequence length. Methods are in an unreachable SI Appendix |
| 19 | Bhattarai 2020 | Trajectory | Linear regression | May be weighted correlation. Methods are in an unreachable SI Appendix |
| 22 | Farooq 2019 (Neuron) | Reconstruction error | ~5 cm | May be 7.6 cm median. Paper not retrievable |
| 28 | Carey 2019 | Place field smooth | 2 | May be a 1-bin kernel on ~3 cm bins, with 2 ms being the temporal spike kernel. Paper not retrievable |

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
- **Stage mixing.** Where a paper detects a sharp wave and a ripple separately, the row
  can take its minimum duration from one stage and its maximum from the other, as in
  Harvey 2023.

## Papers that could not be retrieved

Every cell in these rows is unverified, which is not the same as confirmed: Mallory 2025
(row 1), Krause 2022 (10), Gridchyn 2020 (17), Xu 2019 (21), Michon 2019 (25), Berners-Lee
2021 (12), Pfeiffer & Foster 2015 (40), Lee & Wilson 2002 (52). Gupta 2010 (46), Diba 2007
(49), Igata 2021 (16), Bhattarai 2020 (19) and Farooq 2019 (23) have their detection
parameters in supplements that could not be retrieved. Michon 2021 (15) and Carey 2019
(28) were checked against preprints, so the published versions may have drifted. Science, Cell Press and the Journal of
Neuroscience all refuse automated requests.

## Sheet edits

To bring the Google Sheet into line, change these cells. Row numbers are sheet rows,
which are the data row plus one for the header.

| Sheet row | Column heading | New value |
|---|---|---|
| 4 | Journal | Nature Communications |
| 5 | SWR Z-score Thresh. (STD) | 4 |
| 7 | Detection | SWR, MUA |
| 14 | Reconst. Error (cm) | 7 |
| 30 | SWR electrodes (#) | 1 |
| 32 | SWR Z-score Thresh. (STD) | 2.5 |
| 32 | Max Duration (ms) | 500 |
| 32 | Combine Events Thresh. (ms) | 40 |
| 32 | % Decoded | 36.9-38.6% |
| 33 | Min. Cells (#) | 4 |
| 34 | SWR Low Band (Hz) | 100 |
| 34 | Decoding notes | (clear the cell) |
| 36 | First Author | Grosmark |
| 37 | Time bin (ms) | 20 |
| 37 | % Decoded | 15-20% |
| 38 | DOI | https://doi.org/10.1016/j.neuron.2016.02.010 |
| 39 | MUA smooth (ms) | 10 |
| 40 | Reconst. Error (cm) | 10.0 |
| 42 | Year | 2013 |
| 42 | Last Author | Redish |
| 44 | Decoding notes | replace "cumulative distance" with "start-to-end distance" |
| 44 | % Decoded | 25.3-43.9% |
| 48 | Journal | Nature Neuroscience |
| 48 | Animal Speed (cm/s) | 2 |
| 48 | SWR electrodes (#) | 1 |
| 54 | Max Duration (ms) | #N/A |
| 20 | First Author | Bhattarai |
| 25 | Journal | Nature Communications |
| 25 | Detection Notes | Place cell rate peak reached 3 SD, boundaries where it fell below 1 SD, and 5 or more active cells / Reward zones defined by visual judgement of the speed distribution |
| 28 | SWR electrodes (#) | 1 |
