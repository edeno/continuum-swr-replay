# Memoryless sliding-window sequence detection (Widloski & Foster 2025)

*Design spec, now implemented; not yet run on recorded data. `config.py`, `statistics.py`,
`detect.py`, `likelihood.py`, `simulation.py`, `scoring.py`, `shuffle.py`,
`event_table.py` and `epoch.py` exist and are tested, and
`scripts/decoding/run_memoryless_detection.py` runs them per recorded epoch. The decodes
are not on the development machine, so the real-data pilot is the next step. The
comparison between inventories lives in `analysis/inventory_comparison/`
([03](03-inventory-comparison.md)). The sections on what the port does on simulated
data are results; everything else is a proposed analysis.*
*Governing rule and cross-cutting decisions: [overview.md](overview.md).*

Inventory 6 of seven: the posterior-sequence trigger. How this cohort is compared against
the other six is [03-inventory-comparison.md](03-inventory-comparison.md); the
initial event-feature and selection diagnostics are in [04](04-event-features.md).
Threshold robustness is a planned later stage in 04, and performance validation remains
parked in [05](05-validation.md). Neither blocks the default-setting comparison.

## Purpose

Port the Widloski & Foster sliding-window sequence detector so that it runs on our
recordings, producing inventory 6 of the seven in [overview.md](overview.md).

The detector is memoryless by construction: each 80 ms window is decoded independently
under a uniform spatial prior, and sequences are assembled afterwards from continuity
criteria. That is the substantive contrast with our state-space procedure, which infers
position and representational state jointly across time. Holding the encoding model fixed
(overview X5) means the two differ in temporal inference and event selection, not in what
the spikes are taken to mean.

Why this particular port matters for the ripple-less question: posterior-triggered
detection is the only family that can find replay with neither a ripple nor a population
burst (an MUA trigger can find ripple-less replay, but only when a burst is present), and
this is the published instance of that family. If two posterior-triggered detectors built on the same
encoding model disagree about the non-SWR population, that is a fact about detection
rather than about hippocampus.

Implementation targets reuse across sessions. HPC clusterless decoding is the initial
case; extending to other brain areas needs other likelihood backends.

The method is Widloski & Foster (2025), *Replay without sharp wave ripples
in a spatial memory task*, Nature Communications 16.
doi:[10.1038/s41467-025-65181-5](https://www.nature.com/articles/s41467-025-65181-5) ·
[open full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC12639102/) ·
[preprint](https://www.biorxiv.org/content/10.1101/2024.05.13.593931v1).

Source code for the original is vendored at
[reference/widloski_foster_2025/](../../../reference/widloski_foster_2025/); that
directory's README records provenance, licenses, pinned commit, and the functions that
exist in neither published source. File references below of the form
`foster_lab_matlab/x.m` and `zenodo_15199609/x.m` point into it.

## What this detector selects, relative to ours

Both this detector and our state-space procedure use Bayesian spatial decoding and
threshold-based event selection; the distinction to test is how temporal structure,
competing explanations, and uncertainty enter that selection. The contrast is drawn
against our cohort specifically because those two are the pairing where the encoding
model is shared and only inference and selection differ. The corresponding contrasts
against the LFP- and rate-triggered inventories are in
[03-inventory-comparison.md](03-inventory-comparison.md).

| Aspect | Widloski & Foster procedure | Our current state-space procedure | Comparison to make |
|---|---|---|---|
| Target | Extended, smooth decoded sequences | High-probability non-local representation, continuous or fragmented | Report the full populations and a secondary comparison restricted to extended trajectories; stationary or fragmented events are outside the original sequence target |
| Time | Independent spatial decoding in overlapping 80 ms windows, followed by continuity and merge rules | Inference on the native 2 ms grid with state and position transitions and acausal smoothing | Describe boundary differences, splits/merges, and short-event populations; smoothing controls are follow-ups, and recovery accuracy needs a reference |
| Alternatives | Spatial posterior followed by sequence criteria; no explicit Local/No-Spike state competition | Local, No-Spike, Non-Local Continuous, Non-Local Fragmented | For comparator-only events, inspect local coding, sparse activity, and non-local evidence; allow that our model may miss supported sequences |
| Uncertainty | Posterior compressed to center, spread, and jump statistics for selection | Joint state/position posterior; non-local state probability is thresholded | Examine diffuse or multimodal events without treating our own posterior confidence as validation |
| Event selection | Duration > 100 ms, dispersion > 12 cm, spread/jump/merge criteria, and published shuffle significance | Combined non-local probability > 0.8, the current 15 ms minimum confidence-run duration, then whole-event immobility | Describe exclusions under both definitions; robustness of either cohort to its thresholds remains unassessed until the follow-up in 04 |

The current state-space contract comes from `get_data_over_time`,
`get_confident_times`, and `build_speed_eligible_event_cohort`, not from a new
definition written for this comparison. Read thresholds and fitted transition
parameters from the actual run and record them. The 15 ms minimum is a bare default of
`get_confident_times` that `get_data_over_time` never passes and no run config records,
so the manifest must state it explicitly rather than derive it from saved parameters.
A 2 ms grid is not a demonstrated 2 ms effective resolution: our acausal smoother can also shift or extend boundaries.
Likewise, overlapping memoryless windows reuse spikes and induce temporal dependence;
"memoryless" does not imply an absence of smoothing in the resulting trajectory.
Longer windows also accumulate evidence. Neither their temporal averaging nor our
temporal prior establishes better detection without examining the resulting events.

## The original method

Established from the paper's Methods together with the vendored code; where the two
disagree, the disagreement is recorded in the deviations register rather than silently
resolved.

1. **Rate maps.** 2 cm bins over a 90 × 90 cm arena; occupancy-normalized spike counts
   from running periods (> 10 cm/s); smoothed with an isotropic Gaussian, 8 cm SD per
   the paper (the vendored Mallory code gives 4 cm — see the discrepancies table).
   Cluster inclusion: noise overlap < 0.03, isolation > 0.95, peak SNR > 1.5. Place-cell
   inclusion: rate > 0.01 Hz, spatial information > 0.5 bits/spike.
   (`foster_lab_matlab/compute_rateMap.m`, `load_AnalysisInformation_cm.m`)
2. **Decoding.** Memoryless Bayesian, uniform spatial prior, Poisson likelihood, in
   80 ms windows stepped by 5 ms. Every tuning curve gets `+1e-3` added before decoding.
   (`foster_lab_matlab/compute_BayesianDecoding_fast.m`)
3. **Per-window statistics.** Posterior center of mass; *posterior spread* as the square
   root of the second central image moment; *jump size* as the L2 norm between
   consecutive COMs. The reference pads the jump vector at the end
   (`jumps = [jumps; jumps(end)]`, driver line 82), so window k carries the **forward**
   jump to window k+1 and the last window repeats its predecessor's; the Mallory `_cm`
   filter pads a zero at the front instead (backward jump). The port uses the forward
   convention. (`compute_centerOfMass.m`, `compute_imageSpread.m`)
4. **Bin filter.** Keep windows with spread < 10 cm, jump < 20 cm, speed < 5 cm/s.
   Rejected windows become NaN. (`compute_filtering_binDecoding_cm.m`)
5. **Sequence assembly.** Contiguous non-NaN runs; merge neighbors separated by
   < 20 cm and < 50 ms; drop runs shorter than 100 ms.
   (`compute_allSequences_NaNseparated.m`, `..._merge.m`)
6. **Replay criteria.** Duration > 100 ms and spatial dispersion > 12 cm, where
   dispersion is the RMS distance of the COM trajectory from its mean.
   (`load_ind_replayEvents.m`)
7. **Significance (required by the published replay definition).** Re-decode
   each event 100× with place-cell identities permuted; require p < 0.05 for both mean
   spread and mean jump size. (`zenodo_15199609/load_replayEvents_cellIDshuffle.m`)

### Notes on the original that bear on the port

- **Step 7 is required by the paper.** Its Methods, "Replay detection," lists the
  shuffle test as criterion (3). `load_ind_replayEvents.m` selects duration/dispersion
  candidates, but its lack of a significance filter does not establish the final
  paper cohort. The Zenodo detection driver applies both p-value filters at lines
  146–148; the ripple-statistics driver also applies them in some downstream analyses.
- **The spread statistic is a product, not a sum.** `compute_imageSpread.m` computes
  `sqrt( Σ |x−x̄|² |y−ȳ|² P / ΣP )` — a product of marginal deviations, in **bin** units,
  so in 2D it is area-like rather than a length, and the paper's "m < 10 cm" is a loose
  label. Mallory's code confirms this reading by rescaling the threshold as
  `maze_size² × 0.0012` with the comment `% equivalent to John's 10 cm`. **In 1D the
  function takes its other branch and reduces to `sqrt(Σ|x−x̄|²P)` — an ordinary SD.**
  The port uses a graph-distance RMS spread in cm. Keeping a 10 cm threshold is an
  explicit adaptation, not an equivalence between the 2D statistic and a 1D SD. In
  the reference driver the raw spread is compared with `sequence_posteriorSpreadThr /
  binSize`, so even the raw numerical threshold must be distinguished from its label.
- **Two bugs in the published detection script** (`zenodo_15199609/load_replayEvents_cellIDshuffle.m`),
  which affect that script's own significance filter: line 137 passes the all-NaN p-value
  vector to `compute_sequenceJumps` instead of the posterior COM, so every jump mean is
  NaN, `nansum(NaN < NaN)` is 0, and every event gets p = 1/101 on jump and passes that
  half; and lines 122–123 index `timeBins_sub{i}` by loop counter while writing to
  `ind_replay(i)`, re-decoding the wrong candidate. That script's filter was therefore
  spread-only, on the wrong windows. The vendored analysis script does not rely on it:
  `plot_replayEvents_rippleStats_allSessions.m` loads `replayEvents_cellIDshuffle_sameEvents`
  (line 673), from `load_replayEvents_cellIDshuffle_sameEvents.m`, which computes jumps
  from the posterior COM on the saved events, and where it filters on significance it
  requires both p-values `<= 0.05` (lines 927–928; the other uses of `replays_pvalue` in
  that script are commented out, and its main event set is loaded from a `_sigReplays`
  file whose producer is not vendored). The port follows that corrected script, and the
  sweep reports a spread-only rate beside it to bracket the difference.
- **Run lengths are index differences.** `compute_allSequences_NaNseparated_merge.m`
  recomputes `lengths = boundaries(:,2) − boundaries(:,1)` for every run, merged or not,
  so the 100 ms filter (`lengths < sequence_durationThr/shiftSizeDecoding`, driver line
  96) drops runs of index length below 20: a surviving run has at least 21 windows,
  spanning ≥ 100 ms between first and last midpoint. At our 4 ms stride that is index
  length 25, i.e. 26 windows (V10). The *pre-merge* length convention comes from
  `seqle`, which is vendored nowhere (see the reference README); the port defines run
  length as an index difference throughout and does not depend on it.
- **Forward-jump consequence.** Because window k's jump is to k+1, a window whose
  successor was rejected by the speed gate (NaN center) carries a NaN jump and is itself
  rejected. Sequences therefore end one window before the first speed-rejected window.
- **The two reference implementations order the speed gate and the jumps
  differently, and the port follows the driver.** In
  `zenodo_15199609/load_replayEvents_cellIDshuffle.m` the speed gate is
  `ind_good` at line 54, decoding is restricted to it at line 64
  (`numSpks = numSpks(:,ind_good)`), the results are scattered back into a
  NaN-initialised `posteriorCOM` at lines 79-80, and only then are jumps taken
  from that NaN-padded trajectory at line 82. Speed therefore reaches a
  neighbour's jump. The helper library does the opposite:
  `compute_filtering_binDecoding_cm.m:12` computes `jumps` from the raw
  ungated centre of mass and line 54 applies speed, spread and jump in one
  `find`, so speed never propagates. A reader who checks only the helper will
  conclude this port invented the ordering; it did not, but the disagreement is
  real and worth stating.

### Clusterless shuffle: permute encoding pairs

The practical adaptation uses the saved KDE directly. For each tetrode, let `M`
be its saved encoding waveform vectors and `X` the corresponding encoding positions.
For each candidate and shuffle draw, sample a permutation `pi` of encoding rows and
use `(M, X[pi])` in the likelihood. Permute complete position rows independently
within each tetrode; do not move encoding samples between tetrodes. This requires
neither waveform clustering nor a new encoding-model fit.

Keep the observed event spike times and their waveform vectors fixed. Also keep
occupancy, mean rates, bandwidths, and ground-process intensity fixed: permuting
`X` preserves its marginal distribution and therefore its KDE ground intensity.
Use one permutation per tetrode for all windows of a candidate in a given draw.
Saved objects remain immutable; random seeds are determined by the run, session,
candidate, draw, and tetrode. The `results.nc` fast path can supply the observed
likelihood, but shuffled likelihoods require the encoding arrays and event marks.

Re-decode the candidate's original 80 ms windows, including windows inside merged
gaps, for the observed data and 100 shuffled encodings. Hold candidate bounds fixed;
do not redetect events or reapply spread/jump gates within a shuffle. For each decode,
compute the mean posterior spread and mean consecutive-window graph-distance jump,
repeating the final jump to match the reference's length-N jump vector.
These are separate from the NaN-masked statistics used in candidate assembly.
Both observed statistics must be smaller than expected under their shuffled
distributions. The statistics and the test follow the corrected reference script
(`load_replayEvents_cellIDshuffle_sameEvents.m`, lines 72–79): both means skip NaN
(`nanmean`), `p = (1 + nansum(null_stat < observed_stat)) / (100 + 1)` counts only
draws strictly below the observed value (a tie or a NaN draw does not count), and each
half passes at `p <= 0.05`. One line is translated rather than copied. The reference's
`jumps(jumps==0) = nan` sets aside, in each draw, the window pairs whose continuous
center of mass did not move. This port's jump sizes come from binned centers, and a
literal zero test on those would also set aside every pair whose center stayed in its
2 cm bin, most pairs on a slow replay, turning the mean jump into the mean size of a bin
crossing; measured on the fixtures, that flipped the verdict the reference's own rule
gives. The set-aside pairs therefore come from a continuous center, kept apart from the
binned centers whose jumps are averaged.

- **Sorted cells: the reference's rule** (`unmoved_center_pairs`). As the reference
  does, each window is decoded from its own whole-number spike counts, so windows with
  the same spikes get the same posterior to the last bit, and in every draw a pair is
  set aside when the continuous center of mass, `nansum((1:n).*X) / nansum(X)` in bin
  units, did not move. "Did not move" is read to within 1e-9 bins
  (`CENTER_TIE_BINS`): the exact test also turns on rounding, and at every
  disagreement between the port and the reference module before that tolerance the
  center had moved by at most 2.8e-14 bins, which MATLAB would round in its own way.
  Over every sweep cell the port and the reference module set aside the same pairs in
  all 292072 window pairs of the tested candidates' observed decodes and in all but one
  of the 29.2 million pairs of their shuffled draws (`zero_jump_*` in
  `reference_check_summary.csv`). The one was first seen on the 2026-10-03 rerun, on a
  different machine from the 2026-09-25 run that found none, with every other sweep
  measure identical and no verdict count moved; it has not been traced.
- **Clusterless marks: an approximation** (`unchanged_window_pairs`). On a track graph
  there is no continuous center, so a pair is set aside, in every draw alike, when no
  spike enters or leaves it. Marks are continuous, so no spike stands in for another,
  but a change of spikes that leaves the center unmoved is kept where the reference
  would set it aside.

Earlier revisions of this port kept zero jumps and counted ties against the candidate,
copied the zero test literally onto binned centers, and read the pairs from spikes on
sorted cells too; each was replaced after review. The comparator is the reference's
test, as far as a binned center allows.
`ShuffleResult.testable` records a NaN observed statistic, which the reference's
arithmetic passes trivially; it cannot arise for a candidate that meets the dispersion
criterion.

**Interpretation.** This tests the association between waveform marks and encoding
locations. It preserves spike timing and the marginal encoding distributions, but
unlike cell-ID permutation it can blur conditional spatial tuning instead of swapping
intact fields. It therefore emulates the association-breaking purpose and the
event-scoring procedure, not the exact null distribution. Call it an **encoding-pair
shuffle** in the methods. In the pilot, inspect both null statistics: broad posteriors
can have nearly stationary centers, so the effect on the jump test must be checked
rather than assuming this shuffle is uniformly more permissive or more stringent.
Report the comparator cohort both before and after this filter; retain our native
state-space definition. No component model or general detector-calibration study is
required to implement this adaptation.

**As built** (`shuffle.py`). The shuffle is run on every candidate that meets both
replay criteria, as the reference tests every duration- and dispersion-accepted event;
this project's whole-event immobility rule is a column (`whole_event_speed_eligible`) to
filter on afterwards, so the published cohort and this project's are both readable from
one table. It permutes the encoding marks by the inverse
permutation rather than the positions: pairing `M` with `X[pi]` and `M[pi^-1]` with `X`
give the same set of pairs, and the joint mark intensity is a sum over pairs, so this is
exact and avoids permuting the `(n_encoding, n_bins)` position kernel. The per-spike
joint mark intensity is the library's own `block_estimate_log_joint_mark_intensity`, and
spikes are binned by its own `get_spike_time_bin_ind` on the full epoch grid, so the
unshuffled re-decode is the epoch decode restricted to the candidate's rows; tests pin
both facts against the library. Seeds come from the run seed, the epoch identifier, the
candidate's first window, the draw and the electrode, so draw `k` does not depend on how
many draws there are. The reference passes at `p <= 0.05` and the paper says `p < 0.05`;
at 100 draws the p-values are multiples of 1/101 and the two agree, and the port tests
`<=` as the script does.

**For sorted cells, the reference's own shuffle** (`shuffle_candidate_sorted`): whole
place fields are swapped between cells, `rateMap(ind_rand, :)`, so each cell's spikes
are decoded with another cell's field. The row likelihood is
`sum_n count_n log field_n(x) - sum_n field_n(x)` and a draw permutes only the first
term, which the code computes as `counts @ log(fields[pi])`; tests pin it to the
library run on swapped fields. The simulations, which are sorted, use it; recorded data,
decoded clusterless, use the encoding-pair shuffle. On a correctly decoded moving
trajectory it is the mirror image of the encoding-pair shuffle: the jump half separates
(observed below every one of 100 draws) and the spread half does not (p about 0.5),
because a swapped field still gives a sharp posterior, but one that hops between
fields; twenty sorted cells tiling a linear track, 30 Hz. Its effect on the sweep is
under "Parameter sweep over injected replays".

**What the encoding-pair test does on simulated data, under the reference's
conventions.** Measured before any real data, and the reason the pilot must inspect
both nulls:

- *One tetrode, five units, fields tiling a linear track, decoded in-sample.* The decode
  follows the animal to within a bin. The spread half separates by a wide margin:
  observed mean spread 3.5 cm, every one of 50 draws above 13 cm. The jump half does
  not: breaking which mark goes with which place leaves a broad posterior, whose center
  sits near the middle of the track and barely moves, so the shuffled mean jumps are
  small and a trajectory that really moves sits inside their distribution (p about 0.25
  at 80 Hz). "Observed jump below the shuffled" favours a decode that stands still, and
  the candidate does not pass. Marks scrambled against their spike times, so the decode
  carries no place at all, fail the jump half further still (p about 0.9). A literal
  zero-jump rule on binned centers had turned the real trajectory into a pass by setting
  most of its own pairs aside; reading the pairs from the spikes restores the verdict a
  continuous center gives.
- *`non_local_detector`'s clusterless simulation, a continuous replay spliced into a
  silent epoch.* Spread passes; jump fails, the observed mean above every draw. That
  simulation gives each electrode one stretch of track, a within-electrode shuffle keeps
  it, so a shuffled decode still sweeps the track, and the replay's own jumps are no
  smaller than the sweeps'.

The first is a property of the jump statistic against a broad null; the second, of
permuting within electrodes when an electrode's identity carries position. CA1 tetrodes
are not topographic, so the second should matter less on real data than in that
simulation, but how much position a tetrode's identity carries has to be measured, not
assumed. Whether the reference's cell-ID shuffle shows the first behaviour on its own
data is not known. Both are pinned in the tests
(`tests/unit/test_memoryless_shuffle.py`,
`tests/integration/test_memoryless_simulated_replay.py`) so a change to either statistic
has to confront them.

## What the port does on simulated data

Run on the simulation this project's Figure 1 uses to explain the state-space model
(`scripts/figures/generate_memoryless_comparison.py`, pinned by
`tests/integration/test_memoryless_figure_one_simulation.py`). The simulation labels a
continuous outbound replay, a fragmented replay, a continuous inbound replay, a silent
stretch and an unmanipulated pause; a remote stationary event is added here, because the
package simulates no such case.

**It detects the right events and mis-times them.** On the figure's recording, which
is also the integration test's fixture, the detector assembles 32 candidate sequences
over 65 s of which three meet both replay criteria: the two continuous replays and the
80 ms replay of panel c. The test pins which events are accepted and how each is
mis-measured, not the candidate count. No
silent, fragmented, local or remote stationary event is accepted. But the two accepted
events run 1.58 s and 1.18 s for replays that are 0.27 s: 83% and 77% of each detected
interval lies outside the labeled event, and about three quarters of their windows decode
the animal's own position rather than a replayed one.

**Why.** The method scores a window on posterior spread and on the jump to the next
window, and carries no hypothesis that competes with a replay for a window. Local coding
during immobility is smooth, narrow and slow, so it passes both criteria. The merge rule
then does most of the work: with merging disabled the same data gives 1.4x and 2.1x
inflation instead of 5.8x and 4.3x. Absorption needs a travelling event *spatially near*
the stationary one — here the replay begins at the animal's own place, which is the usual
geometry for awake replay. The added remote stationary event has no travelling neighbour,
and is declined for going nowhere; that is the method working as designed, since extended
trajectories are its stated target.

**What this is not.** It is not an artifact of the adaptations. The finding is unchanged
under the original's running-only encoding convention, and for any spread cutoff from
7.5 cm to the 20 cm the reference's own 1D branch uses (both measured before the port
decoded with the reference's rate-map floor; the durations above, merged and unmerged,
are the same with it). The original contains no
local/non-local comparison, no distance-from-animal criterion, no minimum spike or active
cell count, and no ripple or burst trigger; it decodes every speed-eligible bin; and the
merge is ported unchanged. `load_replayEvents_cm.m:218` computes a `startDistFromRat` and
no driver gates on it.

**Hedges to carry with the numbers.**

- The simulation's place fields are driven by position alone, so its immobile pauses
  emit running-strength local coding. Real immobility is sparser and would break the
  local stretch more often. Treat the magnitudes as an upper bound and the mechanism as
  the finding.
- Read the spread cutoff as the strict 2D-equivalent (about 4-5 cm) and the inflation
  falls to 1.2-1.7x. The plateau from 7.5 to 20 cm is what makes the result robust, not
  any single choice.
- These are pre-shuffle candidates. The encoding-pair shuffle would not rescue the
  boundaries: a sequence that is mostly stationary local coding has *low* mean spread and
  *low* mean jump, so it passes such a test more easily than a travelling replay does.
- One simulation is not a calibration study. What transfers is that boundary agreement,
  not event count, is where the two procedures should be expected to differ.

### How it fails to find a replay that is there

The sweep in the next section is the measured source for every rate quoted in this
document. An earlier version of this section carried two tables from scratch scripts
that were not kept (a 40 Hz grid over 20–130 cm and 80–800 ms, and a duration grid at
100 cm); they could not be regenerated from the repository and have been removed rather
than cited.

**Short-path replay is invisible, and this dominates everything else.** The
dispersion criterion is a "replay must travel" rule: a uniform traverse of path
length `L` has dispersion `L / sqrt(12)`, so `dispersion > 12 cm` needs
`L > 41.6 cm`. A replay covering less than about a quarter of this 170 cm track
is not reported at any duration or firing rate (sweep table below). **Before
attributing any cohort difference to temporal inference, measure the decoded path
length of our state-space events against this bound.** On a branched track a
within-arm replay may be excluded by construction.

**Duration is secondary for detection, and decisive for measurement.** Sub-100 ms
replays are sometimes found, because windows that only partly overlap the event
still decode it and pad the run past the 100 ms filter. That is the same
over-extension that inflates boundaries, working in the detector's favor. What it
costs is the duration itself: the 100 ms filter puts a floor under the reported
duration by construction, and the window width lifts events off it, so **the
duration distribution this detector produces is truncated from below and inflated
near the truncation**, and it must not be compared with our own event durations as
though both measured the same thing. Accuracy is worst at both ends, for two
independent reasons. Short events are padded by the window: the 80 ms replay in
the figure holds no local coding at all, 0% of its windows decode the animal's own
place, and it is still reported too long. Slow events are padded by absorbed local
coding instead. Removing either cause would not fix the other.

**Spike density gates everything else.** The sweep's three firing rates (20, 60
and 160 Hz peak) span erratic to complete detection at fixed geometry; the
per-rate rates are in the raw sweep table. Cell yield therefore changes the
cohort, and any comparison across sessions with different yields confounds
detection with biology.

**Other ways a real sequence is lost**, from the criteria rather than from
measurement, and worth checking on real data:

- Movement. A window whose midpoint speed is not below the threshold loses its
  center, and because jumps are forward the sequence ends one window *before*
  it. A replay overlapping the start of movement is truncated, not just
  trimmed. This is easy to induce accidentally: the first version of the sweep
  above placed injections across the end of a pause and measured the speed gate
  instead of what it intended.
- A single unusable window inside a replay removes its predecessor as well, and
  splits the run in two.
- Splitting. A gap longer than 50 ms, or a bridge longer than 20 cm, blocks the
  merge, and each piece must then clear 100 ms and 12 cm on its own. Measured on
  a replay of 60 cm injected in two halves: with a 40 ms pause the halves merge
  and the event is accepted at 17 cm of dispersion, and with a 60 ms pause
  neither half travels far enough. This is not an independent failure mode. The
  merge only rescues pieces that would each fail dispersion, so it reduces to the
  short-path case above, and the state-space model splits the same replay in two
  as well. It is listed because the merge threshold, not the biology, decides
  where one event ends and the next begins.
- Arm switches. On a branched track a jump to a non-adjacent arm is a large
  graph distance and fails the jump criterion, so a replay that changes arm is
  cut at the switch. Untested here; this simulation is a single linear track.

### Parameter sweep over injected replays

`scripts/qa/sweep_memoryless_detection.py` writes made-up replays into the
spike trains of the simulated recording, one at a time, each with a path
length, duration, firing rate and distance from the animal's own place that we
choose. Because the right answer is known, what each detector reports can be
scored against it. The text below calls these injected replays. 4536 cells, sorted
spikes on a linear track. Every number below comes from
[detection_sweep_summary.csv](detection_sweep_summary.csv), which that script
writes; the full per-cell table goes to `Processed-Data` and is not tracked.
The sweep takes about four minutes to reproduce, and
`scripts/figures/generate_memoryless_comparison.py` draws the lower panels of
the supplemental figure from the same file.

**Two things the sweep is careful about**, both because getting them wrong
produced a wrong answer here first.

Injections are spread over every immobile stretch of the recording, with the
recording's own labeled events cut out of them. An earlier version discarded a
whole stretch whenever it held a labeled event, which discards all of them, and
a silent fallback then kept the last one: all 4536 cells landed in a single
2.2 s window of one pause.

A detector counts as finding an event only when what it reports **is** the
injected event, not merely something overlapping it in time. Over the whole
reported event its decoded positions must sit on the injected path (the path's
nearer end, outside the injection, so an event that only runs long in time still
counts), and it must not cover far more ground than the path does. The rule is
one function, `memoryless_sequence/scoring.py:score_against_injection`,
unit-tested on its own.
Every reported event of either detector that overlaps the injection is scored,
one event at a time over its whole extent, and the best-scoring one answers for
the cell, so an absorbed sequence that happens to start earlier cannot hide a true
detection behind it. The state-space detector's events are the project's own
definition, non-local probability above 0.8 for at least 15 ms, as the sliding
window's are the published one. Scoring temporal overlap alone credits the
sliding-window detector for sequences built out of the local coding beside the
injection: those reach 38 to 162 cm of decoded track for injections of 10 to
30 cm.

The sliding window decodes the one fitted encoding model with the reference's
`+1e-3` Hz floor under every place field (`with_reference_rate_floor`), and is held to
the whole published definition as adapted here, its significance test included: every accepted sequence that overlaps the injection meets 100 draws of the
reference's place-cell identity shuffle, and counts as the injected replay only if it
beats them on both halves. `sliding_matches` is that rate;
`sliding_matches_before_shuffle` leaves the test out, and `sliding_matches_spread_only`
applies its spread half alone, which is what the detection script's own filter could
fail.

**Detection depends on different things for the two procedures.** Percent of
injected replays found under the whole published definition, by path length and
duration:

| path (cm) | 40 ms | 60 ms | 80 ms | 120 ms | 200 ms | 300 ms | 500 ms |
|---|---|---|---|---|---|---|---|
| 10 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| 20 | 0 | 0 | 0 | 2 | 1 | 1 | 5 |
| 30 | 0 | 2 | 1 | 3 | 5 | 6 | 9 |
| 45 | 1 | 3 | 8 | 29 | 54 | 62 | 71 |
| 60 | 4 | 9 | 22 | 44 | 72 | 76 | 84 |
| 90 | 0 | 1 | 11 | 36 | 61 | 76 | 85 |
| 130 | 0 | 4 | 8 | 25 | 67 | 83 | 92 |
| any path (state-space) | 62-74 | 78-96 | 86-92 | 88-97 | 98-100 | 100 | 100 |

And before the significance test:

| path (cm) | 40 ms | 60 ms | 80 ms | 120 ms | 200 ms | 300 ms | 500 ms |
|---|---|---|---|---|---|---|---|
| 10 | 1 | 0 | 0 | 0 | 1 | 0 | 0 |
| 20 | 0 | 0 | 0 | 2 | 2 | 4 | 5 |
| 30 | 0 | 3 | 1 | 4 | 5 | 6 | 9 |
| 45 | 2 | 3 | 14 | 31 | 54 | 62 | 71 |
| 60 | 11 | 20 | 35 | 58 | 75 | 82 | 86 |
| 90 | 1 | 6 | 28 | 47 | 74 | 78 | 88 |
| 130 | 4 | 25 | 29 | 38 | 79 | 96 | 96 |

The sliding window is governed by path length, which is one of its criteria.
The state-space model is governed by duration, which is spike count, which is
the data. Below about 45 cm the sliding window finds a few percent of
injections at any duration and any firing rate.

**Several rates, not one, because they answer different questions.** Over all
4536 cells the sliding window built an overlapping candidate 81% of the time,
accepted one on duration and dispersion 27% of the time, and that sequence was
the injected event 21% of the time. The significance test then removes few: 18%
under the whole published definition, 20% with its spread half alone. The
state-space model reported non-local activity 91% of the time, and it was the
injected event 91% of the time. Quoting 27% against 91% compares "replay,
classified" with "non-local, detected"; 81% against 91% is the nearer comparison of
whether anything was seen at all.

**The significance test fails short replays on its jump half.** Among the
injected replays the sliding window reported before the test, the share passing each
half, by duration:

| true duration | 40 ms | 60 ms | 80 ms | 120 ms | 200 ms | 300 ms | 500 ms |
|---|---|---|---|---|---|---|---|
| spread half | 75% | 83% | 88% | 97% | 98% | 98% | 99% |
| jump half | 31% | 50% | 57% | 79% | 93% | 95% | 98% |

A short replay covers its path in few windows, so its own non-zero jumps are large,
and the shuffled posteriors, hopping between swapped fields, do not jump further.
Overall 97% pass the spread half and 87% the jump half (`shuffle_pass_rate_by_duration`,
and `sliding_shuffle_*_pass_rate` in `overall`).

The reference's rate-map floor is what moved the spread half. Without it the spread
half passed about two thirds of replays at every duration (65% overall; 82% the jump
half), and the whole published definition reported 11% of injected replays rather
than 18%. The likely reason, not separately measured: without a floor a spike from a
cell whose swapped field is nearly silent at a place rules that place out almost
entirely, so a shuffled posterior stays sharp and its spread does not exceed the
observed one; the floor bounds that. Without the floor, the jump-half trend did not
depend on how the reference's zero-jump rule is read: with the pairs no spike entered
or left set aside it read 21% at 40 ms against 97% at 500 ms, keeping every zero jump
21% against 98%, and setting aside every pair whose binned center stayed put 29%
against 95%. Setting aside, in addition, a sorted cell's spike leaving as another of
its spikes enters moved no value in the summary: on the Figure 1 simulation that rule
sets aside 24 of 3313 window pairs, and no p-value there changed.

**Nearly a quarter (23%) of the sliding window's accepted events are not the
injected event.** They are absorbed local coding that overlaps it. This is the
same over-call the figure shows, measured as a rate.

**What is being compared, and what is not.** Each procedure runs with its own
acceptance criteria, the sliding window's as adapted here (graph RMS spread, binned
centers, a 4 ms stride, a 4 cm/s immobility cutoff); those criteria are part of each
pipeline and are not harmonised, so the sweep compares the event inventories of two
complete pipelines, and a disagreement cannot be put down to the triggers alone. The
figure caption leads with the shared target, where both are held to the published
duration and dispersion criteria (`overall_in_domain`, `state_space_replay_matches`). What is held identical is the scoring: both detectors are asked
whether the injected replay came back, and a reported event counts only when,
over its whole extent, its decoded positions track the injected path and it does
not cover far more ground than that path. Both are measured over each detector's
whole reported event, on both sides, and over the windows the detector kept.
Measuring either over only the overlapping part credits an event that tracks the
injection and then sits elsewhere, which is the very thing the test is for; the
span allowance alone (two 20 cm tolerances) is looser than a path under 30 cm, so
the position test over the whole event is what excludes an absorbed sequence
beside a short injection.
The state-space model runs on the library's default discrete transitions:
`discrete_transition_stickiness` is a Dirichlet concentration read only by the
EM M-step, and this sweep calls `fit`, so setting one would change nothing.
Each reported event is scored on its own for both detectors, and the
state-space detector's events carry the project's 15 ms minimum; an earlier
version scored the union of every state-space stretch above threshold, which
read 94% rather than 91%. About 4% of the state-space detector's reports come
as more than one interval, and the best of them answers for the cell, as the
best of several accepted sliding-window sequences does (0.5% of cells).

**No cell in 4536 was found by the sliding window and missed by the
state-space model** under the published definition, and five before the significance
test (`found_by_sliding_window_only` and its `_before_shuffle` partner in
`overall`). 73% of cells went the other way. Of those, 65% built a
candidate that then failed dispersion, 23% built no candidate at all, and 4% were
the injected replay but failed the significance test, while the median share of
windows passing the spread criterion inside the event was 1.00. The spread gate is
not what loses these events; dispersion is, then building no candidate at all, and
only a few are lost to the significance test.

**The path-length limit belongs to the replay definition, not to the inference.**
A uniform traverse of length L has dispersion L/sqrt(12), so the published 12 cm
dispersion criterion cannot accept a path under about 42 cm however well it is
decoded. Scoring our own state-space events under that same criterion confirms
it: their median dispersion is 2.5, 4.5 and 6.5 cm for injected paths of 10, 20
and 30 cm, and the rule would reject 100% of them. Above the floor the two are
of the same order under that rule — 34% against 26% at 45 cm, 53% against 43% at
60 cm, 52% against 46% at 130 cm (the sliding window first, before its significance
test, since the state-space events are held to duration and dispersion alone;
`by_path` in the summary). This is recorded as `state_space_replay_matches` in the
summary. It is a diagnostic of *why* the sliding window misses short paths, not
a rate to quote against `sliding_matches`: the two procedures are compared as
published, against the injected event.

**Where the published criteria can accept an event at all** — path over about
42 cm and duration over 100 ms, 25% of the grid — the sliding window reports 63%
under the whole published definition (68% before its significance test, 67% with the
spread half alone) and the state-space model 99% (`overall_in_domain`); held to the
published duration and dispersion criteria, the state-space model reports 68%, the
same as the sliding window before its test. The remaining three
quarters of the grid are events the published method is defined not to call
replays. That is a property of the method worth measuring, and it is why the
whole-grid rate is much lower than the in-domain rate.

**Reported duration, among events both procedures reported** (the sliding window's
before its significance test, which never moves a sequence's bounds):

| true duration | 40 ms | 60 ms | 80 ms | 120 ms | 200 ms | 300 ms | 500 ms |
|---|---|---|---|---|---|---|---|
| sliding window | 3.3x | 2.2x | 1.7x | 1.5x | 1.2x | 1.2x | 1.1x |
| state-space | 1.10x | 1.00x | 0.97x | 0.98x | 0.96x | 0.98x | 0.98x |

Medians over firing rate, path length and placement. A single event can do
better than its row: the 80 ms replay in the figure, which fires strongly and
travels 100 cm, comes in at 1.6x rather than 1.7x. An absorbed sequence whose
local stretch sits more than 20 cm from the path fails the position test and is
counted among the 23% above rather than here, so this table is the window floor
plus whatever absorption falls within the tolerance.

**Where the extra length goes.** The same events' signed bound errors, reported minus
true, median (`timing_by_duration`):

| true duration | 40 ms | 60 ms | 80 ms | 120 ms | 200 ms | 300 ms | 500 ms |
|---|---|---|---|---|---|---|---|
| sliding window, start (ms) | -28 | -28 | -23 | -22 | -22 | -22 | -18 |
| sliding window, end (ms) | +65 | +57 | +33 | +30 | +21 | +26 | +25 |
| state-space, start (ms) | -4 | -3 | -1 | 0 | +1 | +1 | +2 |
| state-space, end (ms) | +1 | -2 | -2 | -1 | -5 | -3 | -5 |

The sliding window starts early and ends late at every duration; its bounds are window
midpoints, and its 80 ms windows reach 40 ms past each. Its late ends carry a long tail,
the upper quartile reaching 231 ms at 80 ms. The state-space detector's medians sit
within 5 ms of the truth throughout.

**Absorption reaches about 80 cm.** Among events both reported, by the distance
between the replay and the animal's own place:

| distance | 40-80 cm | 80-120 cm | over 120 cm |
|---|---|---|---|
| duration ratio | 1.65x | 1.16x | 1.13x |
| share of windows on the animal's place | 0.30 | 0.00 | 0.00 |

**What the sweep cannot say.** The animal rests at one end of the track and
pauses nowhere else, so distance from its place is a deterministic function of
where the replay is put: the five midpoints give 20, 50, 80, 110 and 140 cm and
nothing else. The table above is therefore a contrast across three groups of
midpoints with the animal's position held at one value, not a distance
contrast, and it cannot be decoupled without a recording in which the animal
pauses in more than one place. The `(0, 40]` bin is empty because the nearest
midpoint only admits short paths, which are the ones the detector does not
report. Injected replays also run at constant speed with Poisson firing from
the same fields the encoding model was fitted on, so there is no model
mismatch and the surface is optimistic. Sorted spikes on a linear track:
neither the clusterless likelihood nor the jump criterion of a branched track
is exercised.

**On the encoding model.** Both procedures read one set of place fields. The
state-space detector fits them, and the sliding-window likelihood is handed the
same object, so the two differ in how they use the encoding model and not in
what it is. An earlier version fitted the model twice, once for each side;
those two fits differed by about 4% of peak firing rate, and the sweep's
answers were the same either way.

### Activity that is not one clean replay

`scripts/qa/sweep_memoryless_controls.py` runs both detectors, through the sweep's own
run steps, on activity designed in
[memoryless_simulation_controls.md](memoryless_simulation_controls.md); every number is
in `detection_controls_summary.csv`. Our detector is read two ways: its non-local events
(continuous and fragmented states together, the project's definition) and the
continuous state alone.

- **No replay.** Over 1.7 minutes of immobility with no replay, from eight simulated
  recordings, no output reported any event: fewer than 1.8 per minute at 95%
  confidence. The simulator's seed changes the spikes, not the trajectory, so all of
  it is local coding at the track's end, where the animal pauses.
- **Place-less bursts** (every cell at one rate unrelated to its field; no spatial
  content), pooled over peak rates of 20, 60 and 160 Hz. Our non-local events cover
  90% of them, the bursts spending 59-80% of their time in the fragmented state; 75%
  even at 20 Hz, where such a burst fires about as fast as the local coding it
  replaces. The continuous state alone covers 1%; the sliding window 8% (15% before
  its significance test). A burst with no place in it is therefore a non-local event
  under the project's definition, which counts the fragmented state; what in it the
  fragmented state takes up, incoherence or merely a departure from the local pattern,
  these simulations do not separate.
- **Jumping bursts** (a random place every 15 ms). Our non-local events cover 97%,
  which is right; the continuous state alone 50% and the sliding window 7% (12% before
  its test), each calling a sequence that is not there.
- **Mid-event jumps**, their far end held 45 cm from where the animal rests. Our
  detector reports one event over both parts in 73-85% of trials at every jump up to
  80 cm; the sliding window does so for jumps up to 40 cm and never across 80 cm.
- **Chains**, reaching no nearer the animal than 20 cm. With the second part
  continuing from the first, both report one event over both at a 10 ms gap and one
  per part at 200 ms. With it 140 cm away, the sliding window never bridges the gap
  and misses a part in 38-58% of trials; our detector reports one event per part in
  17% of trials at 10 ms and 83% at 200 ms.
- **Each part alone**, decoded with exactly the spikes it had in its event and the
  recording's own spikes around it. Pooled over jumps and chains, recovery is higher
  inside the event than alone: the sliding window 72% against 61%, ours 89% against
  86%. The pooled gain hides losses. Paired by trial, the sliding window recovers fewer
  parts inside the event than alone in 15 of 432 trials and more in 95; ours fewer in 2
  and more in 22 (`n_trials_fewer_in_context`, `n_trials_more_in_context`). The sliding
  window's losses fall most in chains far apart (8 of 120 trials, against 1-3 of 96-120
  elsewhere), and its gains most at a 120 ms jump (32 of 96), where a merged event can
  clear the duration criterion a lone part misses. A part counts as recovered only where the
  event's decoded positions over that part sit on its path, not merely where the event
  as a whole matches; that stricter rule changed no result here.

### Held-out evidence on simulated data

`scripts/qa/sweep_memoryless_heldout.py` tries the held-out test of spec 05 where the
truth is known: both detectors run on discovery cells alone, in three folds (two
interleaved with the roles swapped, one whose evaluation cells cover half the track),
over every injected replay and controls trial, and every event is scored on the
evaluation cells by `inventory_comparison.heldout` under both content estimates, ours
and the sliding window's. Numbers are in `detection_heldout_summary.csv`; the rule and
its choices are in [memoryless_simulation_controls.md](memoryless_simulation_controls.md).

Each event is counted once, as our detector bounds it; the sliding window's bounds on
shared events are in the summary under their own `detector`.

- **Our additional detections carry spatial information in cells they never saw.**
  Among informative events (at least 5 evaluation spikes expected along the content),
  true replays only our detector reported have held-out support in 93-97% of cases in
  the interleaved folds and 87% in the half-coverage fold; shared replays in 100%.
- **The test did not support events without content, on few of them.** None of the 19
  informative place-less burst events has support; at a floor of 2 expected spikes, 4
  of 75 do. Jumping bursts, which do represent places, 57-62%. On the no-replay
  stretches neither detector reported an event from discovery cells, so no event of
  pure local coding reached the test: the false-support rate rests on place-less
  bursts alone.
- **The content estimate barely matters here.** Ours and the sliding window's agree on
  99% or more of the events informative under both.
- **The sliding window reported no event our detector missed**, so there is no
  comparator-only class to test.
- **Power is the limit.** Only 6-7% of the replays only our detector reported are
  informative at all, against 16-23% of shared ones: most additional detections are
  too short or too sparse for a dozen evaluation cells to judge, so an event without
  support is mostly one that cannot be judged, not one shown false.
- Events are labelled by their time inside the injected activity's parts: half or
  more, its truth; partly, mixed; wholly inside a chain's silent gap, content-free
  (none occurred); on the recording's own labeled events, kept apart.

### Against the reference's own arithmetic

Passing tests show the port is consistent with itself; they do not show it decides
as the reference would. `scripts/qa/memoryless_reference.py` re-implements the
reference's sorted-cell path as its MATLAB computes it: a Poisson decode with `+1e-3`
Hz on every rate map, a continuous center of mass, the 1D spread about it, forward
jumps, zero jumps set aside by value in the significance test, the merge of
`compute_allSequences_NaNseparated_merge.m`, and dispersion as the RMS distance of the
retained centers from their mean. `scripts/qa/compare_memoryless_to_reference.py` runs
it beside the port on the same place fields, window spike counts, thresholds and cell
permutations, once as published and once without its floor. Results are in
`reference_check_summary.csv`.

- **The port decides as the reference does, apart from its binned center.** It decodes
  sorted cells with the reference's floor (`with_reference_rate_floor`), so the
  posteriors agree to 1e-6, and applies the reference's zero-jump rule, so the
  set-aside pairs agree in every observed draw and all but one shuffled pair in 29.2
  million (see above); what remains is the binned
  center (V7). On the Figure 1 simulation every gate, candidate and verdict agrees. Over
  every sweep cell (4536 cells, 17632 candidates, 2.4 million windows) 17512 candidates
  have identical bounds, the reference calls 2198 of them replay against the port's
  2205, and 1945 pass its significance test against 1949; 34 cells (0.7%) differ in
  either count. Near-equal totals can hide changes in opposite directions: tested on
  the same candidate windows, 17 of the 2205 significance verdicts differ, all from
  the binned center, since the posteriors and the set-aside pairs are the reference's.
- **The floor is not a detail.** Without it (V3) the posteriors of 5% of sweep windows
  move by more than 0.001, the reference calls 2473 candidates replay and passes 1705,
  and 465 cells (10%) differ from the reference as published. The port ran without
  the floor until 2026-09-25; its sweep then reported 11% of injected replays under
  the whole published definition, against 18% with it.

What this does not establish: it covers the sorted path on a simulated linear track,
not the clusterless encoding-pair null used on recorded data (see "What the
encoding-pair test does on simulated data"); `seqle` and `compute_sequenceDispersion`
are not vendored, so the run-length convention and the dispersion are taken from the
driver's arithmetic and the paper's Methods; and it holds the thresholds at the port's
adapted values (10 cm spread, 4 ms stride, 4 cm/s), so it checks implementation, not
parameter choice.

### Summary: what the port misses, and what it over-calls

*measured* means a script or test in this repository produces the number:
`scripts/qa/sweep_memoryless_detection.py` and its summary CSV, or one of the
`test_memoryless_*` files. *from the criteria* means it follows from the
thresholds and is not demonstrated. Rows marked *measured elsewhere* were run
during this work but by scratch scripts that were not kept, so they cannot be
reproduced from this repository and should be re-run before being cited.

| direction | case | outcome | status |
|---|---|---|---|
| miss | path shorter than about 42 cm | reported in a few percent of injections at any duration or firing rate, against 71-100% for the state-space model | measured |
| miss | replay that changes arm without passing the junction | cut at the switch, and neither half then clears dispersion | measured |
| miss | sparse ensembles, about one spike per window | erratic, 0-50% where dense spiking gives 100% | measured elsewhere |
| over-call | any event shorter than about 150 ms | reported 1.5x to 3.3x too long in the sweep, and 1.6x for a strongly firing 80 ms replay; the detector cannot report a short event as short | measured |
| miss | whole path inside one window | posterior smears, spread rejects. The sweep's fastest cell is 3250 cm/s and is still found sometimes, so the threshold is above that | measured elsewhere |
| miss | replay overlapping the onset of movement | truncated one window *before* the first moving window | measured |
| by design | remote place held still | decoded confidently, assembled, declined for going nowhere | measured |
| miss | one unusable window inside a replay | removes its predecessor too and splits the run | from the criteria |
| miss | gap over 50 ms, or bridge over 20 cm | merge blocked, each piece must clear 100 ms alone | from the criteria |
| over-call | local coding beside a replay | absorbed, 5.8x and 4.3x duration inflation | measured |
| neither | spurious events with no replay in them | none in 65 s of simulation | measured |

**The state-space model finds every one of these.** Run on the same recording, with the same encoding settings, it reports all six labeled events at P(non-local) = 1.00 and stays silent through the silent stretch (peak 0.04). Its bounds sit within about 20 ms of the truth, and all of its intervals are real events. So these cases are limits of the sliding-window criteria, not limits of the data or of decoding in general. Note what this does and does not show: both procedures read the same encoding model, but the state-space model was given the labeled events as its held-out set, exactly as the sliding-window port was, and no threshold on either side was tuned for this comparison.

**The over-call is a boundary error, not a spurious event.** In the one
simulation tested end to end the detector accepted two events and both were
real replays; nothing was invented. What is wrong is the extent: about three
quarters of each accepted event is the animal's own place. Any per-event
content summary computed over those bounds - represented distance, arm,
ahead/behind - is therefore dominated by local coding rather than by the
replay, so for downstream content analysis the event behaves as a false
positive over most of its duration even though the event itself is real.

**The same mechanism cuts both ways.** Absorbing neighbouring windows is also
what lets a sub-100 ms replay clear the duration criterion at all. Removing the
over-extension would lose those events. Sensitivity and boundary accuracy are
not independent knobs here.

**The two directions do not cancel.** A missed short-path replay and an
over-extended long one are different errors: the first removes events from the
cohort, the second corrupts the timing and content of the events that remain.
Reporting a single agreement rate against our cohort would hide both.

## Decisions register

Decisions taken during design, with the reasoning, so the methods section can state them
without re-litigation.

| # | Decision | Reasoning |
|---|---|---|
| **D1** | Explain event agreement and disagreement with a Foster-style comparator using the same encoding model | Isolates temporal inference and selection choices; directly addresses reviewer concerns about what our model adds or misses |
| **D2** | Use the encoding-pair shuffle above for the Foster-style comparator; report candidates before and after filtering | Simple adaptation using saved KDE arrays, with fixed event spikes and the original spread/jump scoring intent. It does not preserve whole cell tuning curves, so its p-values are not equivalent to cell-ID shuffle p-values |
| **D3** | Use our canonical whole-event immobility rule (< 4 cm/s), not theirs (per-bin < 5 cm/s) | Makes both cohorts draw from the same speed-eligible population, so a cohort difference is attributable to the detection algorithm rather than the eligibility rule |
| **D4** | Center = discrete Fréchet mean over real bin centers using track distance; spread = sqrt of the minimized expected squared distance | Avoids averaging unrelated linearized coordinates. Bin restriction approximates COM/SD on a straight segment; deterministic ties use bin order, without a temporal preference. Dispersion applies the same minimization to the retained sequence centers |
| **D5** | Rebuild the posterior from the saved encoding model rather than reusing our state-space posterior | Holds encoding fixed while removing the state-space temporal prior. This is an algorithmic comparator, not independent evidence about the encoding model |
| **D6** | The ported cohort gets its own key (`ms_event_id`); never joined on `event_id` | `event_id` hashes are cohort-dependent; a different cohort sharing the key would mis-join silently (see CLAUDE.md on atomic regeneration) |
| **D7** | `results.nc` fast path is opt-in and verified, never an automatic fallback | The `log_likelihood` variable only exists in decodes written after 2026-06-26 (`c8aee2c`); an automatic fallback would silently change which epochs are analyzed |
| **D8** | Keep native event definitions for the main comparison; use common selection ranges only as a secondary analysis | Prevents redefining our cohort to force agreement while distinguishing a different representational target from better detection |
| **D9** | Keep all overlap pairs, detector-specific event denominators, and both SWR labels | A long sequence can overlap several short events; shared time does not imply identical boundaries or SWR status |

## Deviations register

Every departure from the original. **Effect** states what a reader should worry about,
not a reassurance.

### Forced by recording modality

| # | Original | Ours | Why | Effect |
|---|---|---|---|---|
| **V1** | Sorted units, per-cell rate maps | Clusterless KDE encoding model over waveform features × position | HPC in this project is decoded clusterless; there are no sorted units | The encoding model is not a set of place fields. Cell-level inclusion criteria (rate, spatial information, isolation, SNR) have no analogue and are not applied |
| **V2** | Cell-ID shuffle, p <= 0.05 on spread and jump | Within-tetrode encoding-pair shuffle; the reference's statistics, count and threshold unchanged, its zero-jump rule read from the spikes | Uses existing clusterless arrays without component fitting (D2) | Scrambling individual mark–position pairs can blur tuning; original cell-ID permutation exchanges intact curves. Null distributions and accepted cohorts can differ |
| **V3** | Tuning curves get `+1e-3` before decoding | Clusterless KDE with its own `EPS` clipping of the ground-process intensity | Different likelihood construction | Low-occupancy bins are regularized differently. Not tunable to match for the clusterless decode. For sorted cells, as in the sweep and the figure, it is matched: the sliding window decodes with the floor (`with_reference_rate_floor`), and the state-space detector reads the fields as fitted. It matters: without it the sweep's significance test passed 69% of replay candidates rather than 88%, and the published-definition rate read 11% rather than 18% ("Against the reference's own arithmetic") |
| **V4** | Rate maps fit on running periods (> 10 cm/s) | Encoding model fit on the decode's `is_training` mask (`training_type = "no_ripple"`) | Holds encoding fixed between our two procedures | Non-ripple training includes immobility. It differs from the paper, but is shared in the main comparison; any running-only sensitivity must refit both procedures on that same mask |

### Forced by track geometry

| # | Original | Ours | Why | Effect |
|---|---|---|---|---|
| **V5** | 2D open field, 90 × 90 cm | 1D linearized three-legged track graph, 9 arms | The task's geometry | Sequences are constrained to the graph; "direction" is along-arm rather than an angle |
| **V6** | Euclidean distance for jump and dispersion | Track-graph distance | On a linearized branched track, `abs(Δlinear_position)` is not the distance between two places — bins on different arms can be adjacent in linear coordinate | Using linear differences would under-count jumps at arm boundaries, producing spuriously smooth and spuriously long sequences. Track distance avoids this; it also makes our jumps non-comparable in absolute terms to an open-field jump. Note the asymmetry this creates with the encoding model: `Environment` exposes graph distances (`get_manifold_distances`, `_bin_distance_matrix_`) and this detector uses them, but `KDEModel` smooths with a Euclidean kernel in whatever coordinate it is given, which for a linearized track is the linear one. Measured on a Y track with 20 cm edge spacing: a kernel fitted at one bin gives a density at a second bin 6 cm away on the graph and 26 cm away in linear coordinate of 8.4e-05 of its peak, matching `exp(-0.5*(26/6)^2)` rather than `exp(-0.5*(6/6)^2)`. Place fields therefore do not wrap through a junction. Both procedures in this comparison read the same encoding model (X5, D5), so this does not bias the contrast between them; it is a caveat on absolute decoding near junctions for either |
| **V7** | Posterior COM | Discrete Fréchet mean over track distance (D4) | Linearized-coordinate COM does not respect branching | On straight segments the center is quantized to a bin; near junctions geometry also changes it. Against a continuous center on the simulated linear track, 99.3% of candidates keep identical bounds, the replay count moves by 7 in about 2200, and 17 of 2205 significance verdicts differ on the same windows ("Against the reference's own arithmetic") |
| **V8** | Unvisited arena bins excluded | Junction placeholder bins (`edge_id == -1`) excluded before normalization | These bins are linearization artifacts from `edge_spacing > 0`, not places. In `get_track_grid` the `-1` rows are built from `place_bin_centers[~is_track_interior]`, so they are non-interior by construction and the primary likelihood path never returns them; the guard matters on the `results.nc` path, where `log_likelihood` is re-expanded to every `state_bins` column | If not excluded, probability mass lands on non-places and drags the center toward gaps. Same hazard `map_estimate_to_arm` guards against |
| **V9** | Spread is a 2D product moment (area-like, bin units) | Graph-distance RMS spread (a length, cm) | Provides a spread measure on our geometry, approximating SD on a straight segment | The 10 cm cutoff is an adapted choice prioritized in the later threshold-robustness study (04); its robustness is not established by the initial report, and these spreads are not numerically comparable to the paper's 2D spreads. It also limits which replays survive, but **not through speed alone**. A window's spread grows only when its posterior actually smears over the swept path, and whether it does depends on spike density and field overlap. On `non_local_detector`'s sparse clusterless simulation, where each of 20 units fires once, a window straddling several fields is multimodal and the detector returns one accepted sequence at 375 cm/s but breaks the same trajectory into rejected pieces at 450 cm/s and above (`tests/integration/test_memoryless_simulated_replay.py`). With denser spiking the posterior stays peaked and there is no such cap: injected replays are found at 1625 cm/s with a median spread of 3.6 cm, and spread only breaks down near 4300 cm/s, where the whole path fits inside one window. The binding quantity is how far the path travels *within a window* relative to how well the spikes localize it, not the speed as such. An earlier version of this entry gave a flat 430 cm/s cap; that was a measurement on the sparse simulation, over-generalized. Note the direction of the adaptation: the helper library's own 1D branch (`compute_imageSpread.m:20-30`) computes the same COM-based RMS this port does, and `load_replayEvents_cm.m:32` gates it at **20 cm**, so 10 cm is twice as strict as the only 1D precedent in the reference. That branch also uses a 50 ms duration and a `0.4 * maze_size` jump, both looser than what this port takes from the paper |

### Forced by our sampling grid

| # | Original | Ours | Why | Effect |
|---|---|---|---|---|
| **V10** | 5 ms step | 4 ms step (2 samples at 500 Hz) | 5 ms is 2.5 samples on our native grid | Slightly finer temporal resolution on sequence boundaries. Every bin-count threshold must be **derived from the stride**, never carried over as the original's literal constants |
| **V11** | 80 ms window | 80 ms window (40 samples) | Exact at 500 Hz | None |

### Deliberate, for comparability

| # | Original | Ours | Why | Effect |
|---|---|---|---|---|
| **V12** | Per-bin speed < 5 cm/s | Per-bin < 4 cm/s for candidacy, then our canonical whole-event rule | D3 | Slightly stricter than theirs. Shared with our cohort by construction |
| **V13** | Paper's ripple detection and ripple-less classification | This cohort is labeled under **each** SWR definition in the comparison (Kay, Karlsson, Yu 2017), each speed-qualified and sampled the same way, with strict positive-duration overlap. No single definition is imposed (overview X6) | Imposing one SWR definition on a sequence cohort would blend methods; reporting the matrix also supplies label sensitivity without a separate sweep | Both ripple detection and replay/ripple association differ from the paper, so absolute non-SWR fractions cannot reproduce its headline percentages. The reported non-SWR fraction is a function of the SWR definition as well as the events, and must always be quoted with the definition attached |

### Retained settings and adapted thresholds

Place bin size (2.0 cm = their `binSize`), decoding window (80 ms), uniform spatial
prior, memoryless decoding, jump threshold (20 cm), merge
thresholds (20 cm / 50 ms), minimum duration (100 ms), dispersion threshold (12 cm).

Their application changes with track distance and the 4 ms stride. The spread cutoff
of 10 cm is an additional adapted setting (V9), not an exactly preserved criterion.
The initial comparison reports these retained settings and their adaptations explicitly.
Sensitivity to them is deferred to 04's cost pilot and limited robustness study. Preserve
window statistics, gate masks, candidate bounds, and shuffle provenance so valid work can
be reused later; changes that produce new candidate bounds require corresponding shuffle
calculations. Do not imply that robustness has been established before that study runs.

### Discrepancies within the original, and how we resolved them

| Item | Paper | Zenodo scripts | Helper library | We use |
|---|---|---|---|---|
| Ripple band | 100–220 Hz | — | `[150,250]`, comment says "John used 120-170" | N/A (V13) |
| Ripple merge gap | merge if < 50 ms | `eventMergeThr = 0` (no merge) | — | N/A (V13) |
| Spike-density smoothing | 80 ms SD | 100 ms SD | — | N/A (bursts not ported) |
| Rate-map smoothing | 8 cm SD Gaussian | — | `compute_rateMap.m` passes `numSpatialBins_smoothing/binSize = 4` to `setUp_gaussFilt`, which uses it as the `mvnpdf` *covariance*: SD 2 bins = 4 cm | N/A (V1: no rate maps) |
| Sequence / replay duration | 100 ms | 100 ms | 50 ms, in the **live** `spatialDim==1` branch of `load_replayEvents_cm.m:38-39`, not only in the commented "John's criterion" block | 100 ms, following the paper |
| Significance | Required for replay | Detection driver filters by both p-values, with the bugs noted above; downstream usage varies | `load_ind_replayEvents` alone selects duration/dispersion candidates | Encoding-pair shuffle adaptation, with candidates retained for comparison (D2/V2) |

## Architecture

New subpackage `src/continuum_swr_replay/analysis/hippocampus/memoryless_sequence/`.
Named for what it does; the citation lives in docstrings, not the module path.

| module | responsibility | depends on |
|---|---|---|
| `config.py` | frozen dataclass of every threshold — the `Analysis_Information.mat` equivalent, one source of truth | — |
| `likelihood.py` | prior-excluded `(n_time, n_interior_bins)` log-likelihood on the native 2 ms grid | `detector.pkl`; a `results.nc` path was designed (D7) and is not built |
| `statistics.py` | pure functions: window aggregation, normalization, center / spread / jump / dispersion | numpy, bin-distance matrix |
| `detect.py` | bin filter → sequence assembly → merge → criteria → `CandidateSequences`, retaining rejection diagnostics | the above |
| `simulation.py` | inject a replay of known shape into a simulated recording; place it clear of movement | numpy |
| `scoring.py` | the one rule both detectors are scored by against an injected replay | numpy |
| `shuffle.py` | encoding-pair permutations, re-decoding fixed candidate windows, spread/jump p-values | encoding arrays, event marks, likelihood and statistics helpers |
| `event_table.py` | one row per candidate: timing, jumps within pieces and bridges between them, speed eligibility, SWR labels against the canonical intervals, shuffle columns; the shuffle-draw sidecar | the canonical SWR classifier and matcher, the immobility rule |
| `epoch.py` | `run_epoch`: the whole detector over one epoch, arrays in and tables out, in the one fixed order | all of the above |

The comparison itself (interval-pair table, event summaries, selection diagnostics,
per-session and per-animal comparisons) belongs to [03](03-inventory-comparison.md) and
is not part of this subpackage.

`scripts/decoding/run_memoryless_detection.py` is the per-epoch runner: it reads each
epoch's saved decode, calls `run_epoch`, and writes the tables and a run manifest.

**Spike marks.** The runner reads them through the project's loader, with the two
functions `load_data` uses: `get_hpc_marks` (tetrode sort groups, coincident spikes
counted), then `filter_spike_times` over the decode's time span, which is the position
grid `load_data` filtered over when the decode ran. `check_marks_match_encoding` refuses
them unless every mark the encoding model was fitted on is among them. Those are the
training span's, so the check cannot see a spike the reload adds, or one dropped only
during SWRs, which `no_ripple` training leaves out; the
[full-pipeline refresh](../2026-10-03-full-pipeline-refresh-design.md) leaves that gap to
its Stage 8 one-commit check. The marks are cached per decode file: a cache file is named
for the epoch, the rule (`loader`) and the first 16 hex digits of the decode's
`_detector.pkl` sha256. A cache made for another decode of the epoch is therefore never
read, and neither is the pilot's `{epoch_id}_decode_time_marks.pkl`, made under the April
decodes' rule, which keeps more spikes and so could pass the guard.

**Running it.** `--epoch-id`, `--animal` or `--all` chooses the decodes, and `--date`
(such as `20200225`, the last 8 characters of the NWB file's stem, so `j1620210707` is
20210707) and `--epoch` (such as `02_r1`) narrow the choice. By default the epochs run one
after another in one process. With `--cuda_visible_devices` they run in parallel, as the
decoders do: a dask `LocalCUDACluster` with one worker per listed GPU runs one epoch per
task, and its workers' scratch goes to `--local_scratch_dir`. The main process alone
writes `manifest.json`, after each epoch finishes, and each epoch logs to
`logs/{epoch_id}_memoryless.log`. In either mode a failed epoch is recorded and the rest
run on. A device list is refused when `dask_cuda` is not importable. The full run:

```bash
uv run python scripts/decoding/run_memoryless_detection.py --run-seed 20260922 \
    --all --cuda_visible_devices <cards>
```

## Data flow, per epoch

1. Load `encoding_model_[("", 0)]` from `{epoch}_detector.pkl`; compute prior-excluded
   log-likelihood once at 2 ms via `predict_clusterless_kde_log_likelihood(is_local=False)`.
2. Restrict to interior bins **before** normalizing (V8). `predict_clusterless_kde_log_likelihood`
   already returns only `is_track_interior_` bins, and junction placeholders are
   non-interior by construction, so this is a no-op on the primary path. On the
   `results.nc` path (D7) `log_likelihood` carries every `state_bins` column with masked
   fill, so the interior mask must be applied there explicitly.
3. Sliding 80 ms windows as a float64 cumulative sum over 40 rows, stride 2 rows;
   normalize across real bins under a uniform prior. Validate uniform 2 ms spacing
   and model sampling units; do not aggregate across missing samples or epoch bounds.
4. Per-window center and spread (D4, V7). Then, **before any jump is computed**, blank
   the center of every window whose midpoint speed is not below 4 cm/s (the integer
   sentinel `NO_CENTER`, the port's NaN), as the reference does by decoding only
   speed-eligible windows. Then forward jumps (V6):
   window k's jump is to window k+1, the last window repeats the previous jump, and a
   window whose successor has a NaN center gets a NaN jump. The ordering is load-bearing:
   taking jumps before the speed gate lets a sequence extend one window further into the
   speed-rejected region and can turn a rejected 100 ms candidate into an accepted one.
5. Keep windows with spread < 10 cm and finite jump < 20 cm (the speed gate has already
   been applied through the NaN centers).
6. Contiguous runs of kept windows → merge across gaps < 20 cm and < 50 ms → drop runs
   shorter than 100 ms.
7. Keep events with duration > 100 ms and dispersion > 12 cm.
8. Record whole-event immobility via the existing `event_speeds_are_immobile`, over
   every native speed sample between the endpoints, as a column; it does not gate the
   shuffle.
9. Use the same SWR intervals as `build_classified_nonlocal_event_table`: apply
   `filter_event_labels_by_speed` to canonical `SWR_event` labels and reconstruct
   bounds with `intervals_from_labels`. Classify with `classify_swr_overlap_geometry`
   and the canonical matched-SWR convention. Raw Kay intervals are not interchangeable
   with this qualified, sampled population.
10. Apply the encoding-pair shuffle to every candidate that meets the replay criteria,
    retaining p-values and pass/testability flags on every candidate row. Derive the filtered comparator
    cohort by subsetting; candidate IDs and bounds do not change.

Both cohorts use the same canonical time axis and speed trace. Step 8 evaluates every
native speed sample between sequence endpoints, including rejected windows inside
merged gaps, not just retained window centers. Step 9 uses an identical SWR interval
table on both sides; calling the same overlap helper alone would not ensure parity.

**Window and event timing.** A window with support rows `[k, k + 40)` is timestamped at
the grid sample `k + 20`, half a window into its support, rather than at `time[k] +
0.040`: on a recording's clock that sum misses the sample by up to 2.4e-7 s, and event
bounds must be samples for the speed and SWR checks to be exact. Read speed at that
sample for the step 4 gate. Event start/end are the
first/last retained midpoint, and the duration criterion is their difference, as in
the reference driver. Save the outer decoding support separately; it is not the
event interval used for overlap or speed eligibility. Do not add the 80 ms window
width to event duration. Retain original timing fields for our existing cohort,
whose canonical duration includes a sample-width convention; compute a common
`interval_span_s = event_end_s - event_start_s` for duration comparisons and overlap.
Native probability/duration criteria remain unchanged.

**Merged gaps.** Retain the per-window keep mask and the pre-merge sequence IDs.
Compute dispersion from the finite retained centers, as in the reference's NaN-masked
trajectory. Specify whether each reported jump is a consecutive-window jump or a
bridge between retained pieces; report bridge distances separately. This makes
merging inspectable instead of silently treating rejected windows as decoded paths.

**Why the speed rule is two-stage.** A whole-event rule cannot gate bins before events
exist. Step 4 establishes candidacy; step 8 governs. Step 8 is not redundant: merging at
step 6 can span windows that step 4 rejected.

**The efficiency claim.** In the `uv.lock` version of `non_local_detector`
(`2d730c3a86ad0c62a817d0356b9248d43b6c32f8`),
[`fit_clusterless_kde_encoding_model`](https://github.com/LorenFrankLab/non_local_detector/blob/2d730c3a86ad0c62a817d0356b9248d43b6c32f8/src/non_local_detector/likelihoods/clusterless_kde.py)
divides spike count by the number of training samples. The non-local likelihood
adds a per-row `-summed_ground_process_intensity` and per-spike mark terms. Thus
row sums give a window likelihood when the sample duration and spike support match.
Validate those units for the saved model as well as the current library version.

The direct-call test must respect the library's edge convention. In
`predict_clusterless_kde_log_likelihood` spikes are kept if `time[0] <= t <= time[-1]`
(inclusive at both ends), and `get_spike_time_bin_ind` assigns them with
`np.digitize(spike_times, time[1:-1])`, so the last row index is never produced: every
spike with `t >= time[-2]`, including one exactly at `time[-1]`, folds into the
penultimate row, and every spike before `time[1]` lands in row 0. A call on 40
timestamps therefore spans 78 ms of edges with the two end rows absorbing the boundary
spikes. An interior 40-row slice of a full-epoch likelihood has no such fold: row k
holds the spikes in `[time[k], time[k+1])`, so the slice includes spikes from the 2 ms
after its last timestamp. Compare against a direct marked-process sum over the explicit
half-open 80 ms support, with 40 ground-process contributions; test boundary spikes,
the end-row fold, and epoch tails. A naive call on the sliced timestamp vector is not a
valid reference. Also check long-epoch numerical
accuracy, because float32 cumulative sums can lose precision when subtracted.

## Output schema

Table `memoryless_event_table`, one row per candidate sequence, accepted or not (D2),
built by `event_table.build_memoryless_event_table`. Its columns and dtypes are declared
once, in `MEMORYLESS_EVENT_TABLE_DTYPES`, and held for an empty epoch too.

- **Identity** — `animal`, `day`, `epoch`, `brain_area`, `epoch_id` (the decode's full
  identifier; one epoch can be decoded more than one way), `run_id` (joins to the run
  manifest), `swr_definition` (the ripple detector the SWR labels come from, read from
  the epoch identifier), `sequence_label` (1-based, in time order), `ms_event_id`
  (hashed from the decode's identifier, the run and the sequence's windows, so two
  decodes or two runs never share a key)
- **Timing (seconds)** — `event_start_s`, `event_end_s`, `event_duration_s`,
  `event_midpoint_s`: the first and last retained window midpoint
- **Decoding support (seconds)** — `support_start_s`, `support_end_s`, half a window wider
  at each end, separate from the event bounds
- **Detector statistics** — `n_windows` (every window spanned, merged gaps included),
  `n_pieces` (pre-merge runs), `mean_posterior_spread_cm` (over retained windows),
  `max_jump_cm` and `mean_jump_cm` (consecutive-window jumps within pieces only),
  `max_bridge_cm` (between the facing centers of adjacent pieces; NaN for one piece),
  `dispersion_cm` (over every kept window of the merged span, as the reference's
  NaN-masked trajectory, so a single window dropped before the merge still counts),
  `pieces_dispersion_cm` and `n_kept_windows_outside_pieces` (the same dispersion from
  the pieces' windows alone, and how many kept windows lie outside every piece: added
  2026-10-02 after the first pilot, where 2 of 4 replay calls on chimi20200225_02_r1
  rested on one such window about a meter away; diagnostics, not criteria),
  `path_distance_cm` (summed track-graph distance along the retained
  centers, bridges included; the analogue of the original's `compute_sequenceDistance`,
  reported and not used as a criterion), `is_replay` (both replay criteria)
- **Speed (cm/s)** — `animal_max_speed_cm_s`, `animal_mean_speed_cm_s`,
  `whole_event_speed_eligible`, over every native speed sample between the endpoints.
  Named as the canonical classified table names them.
- **SWR** — `is_swr` (nullable), `overlap_class`, `overlap_subclass`,
  `n_swrs_overlapping`, `matched_swr_event`, `matched_swr_start`, `matched_swr_end`,
  against `canonical_swr_intervals`. An event not wholly inside an interval where the
  labels were observed (`observed_intervals` on `SWR_zscore`) has `overlap_class = "unknown"`, never
  `non_swr`.
- **Shuffle** — `shuffle_observed_spread_cm`, `shuffle_observed_jump_cm` (the reference's
  statistics over every window of the span, zero jumps set aside), `p_spread`, `p_jump`,
  `shuffle_testable` (both observed statistics finite; informational), `shuffle_pass`,
  `n_shuffles`. `n_shuffles == 0` marks a candidate that was not tested. Every candidate
  meeting the replay criteria is tested, so `shuffle_pass` alone selects the published
  replay definition and `shuffle_pass & whole_event_speed_eligible` this project's
  cohort.
- **Windows** — `first_window`, `last_window`: the sequence's window indices, which with
  the decode key the event id; `n_windows` is their difference plus one.

`ms_event_id` is hashed from the decode (`animal`, `day`, `epoch`, `brain_area`,
`epoch_id`) and the window bounds, not from `run_id`, so two runs of one decode join on
it and `run_id` separates them as a column.

Sidecar `memoryless_shuffle_draws`: `ms_event_id`, `draw`, `null_spread_cm`,
`null_jump_cm`, one row per draw, for the null-distribution audit. Sidecar
`memoryless_observed`: `kind`, `start_s`, `end_s`, what each epoch observed independent
of what it detected: the span its windows cover (`decoded_windows`) and the stretches
where SWR labels were observed (`swr_labels`), for 03's availability accounting. The per-window
statistics and gates, and the pre- and post-merge runs, are saved as the
`CandidateSequences` object itself.

Timing column names mirror the canonical event table so comparison code is symmetric.
`ms_event_id` is deliberately a different key from `event_id` (D6).
The run manifest records model/artifact hashes, code versions, thresholds, time grid,
encoding/training provenance, the marks' rule, cache file and hash,
shuffle definition/seeds/scoring conventions, and any skipped epochs. Its settings also
hold the environment (`uv.lock`'s sha256, the commit `ripple_detection` was installed
from, the CUDA plugin's version), and a rerun under the same run id in another
environment is refused, as one under another library version is.
Export 03's observation intervals from input and output availability before applying
detection gates, accounting for full window support at edges and gaps. An empty event
table alone cannot distinguish an observed interval from a failed or unavailable run.
Retain per-window statistics and individual gate masks, pre/post-merge candidates,
and candidate rejection flags in diagnostic outputs; accepted events alone cannot
explain why a detector rejected activity. These outputs are scientific diagnostics,
not additional acceptance criteria.

## Testing

| test | what it protects |
|---|---|
| **Additivity** — cumsum-window log-likelihood equals a direct marked-process sum on identical half-open spike support, including boundary spikes, epoch tails, and long traces | Correct exposure and spike assignment, as well as numerical stability |
| Statistics against hand-constructed posteriors with closed-form answers | Center / spread / jump / dispersion correctness |
| Discrete Fréchet mean matches a brute-force minimum and approaches COM/SD as a straight-segment grid is refined; deterministic tie cases | Quantization and the precise D4 contract |
| Sequence assembly against the reference's merge semantics: run length is an index difference, a single-window run has length 0 and is dropped (the port's convention; whether `seqle` agrees is unverified, and `drop_single_window_runs=False` is the sensitivity), and the 100 ms filter keeps runs of ≥ 21 windows at a 5 ms stride and ≥ 26 at 4 ms | Off-by-one and merge behavior |
| Jump alignment: window k's jump is to k+1, the last window repeats it, and a window before a NaN center is rejected | Forward-jump convention; boundaries would shift by one stride under the backward convention |
| Speed gate precedes jumps: a candidate whose last eligible window precedes a speed-rejected window is one window shorter than it would be with jumps taken first; construct the case where that changes acceptance | The ordering in data-flow step 4 |
| Junction-bin test: mass on an `edge_id == -1` bin must not move the center | V8 |
| Stride-derived thresholds: changing the stride must change bin-count thresholds | V10 — the constant that would silently become a 50 ms threshold |
| Window midpoints determine duration and overlap; outer support is separate; rejected merged gaps still participate in speed eligibility | Temporal conventions do not change the cohort silently |
| Identical event intervals receive identical SWR labels in both pipelines, including a movement-spanning SWR and sampled boundary contacts | Shared qualified SWR population and strict overlap |
| One-to-many/many-to-many overlaps count each event once per detector; matched events may have different SWR labels; disjoint and empty cohorts | Denominators, matching, and empty-cohort handling |
| Rejection diagnostics and gate-specific reruns agree on synthetic examples | Explanations of selection follow the actual pipeline |
| Encoding-pair shuffle preserves observed spike/mark arrays, encoding marginals, and ground intensity; identity permutation reproduces the original likelihood | The intended null operation and absence of accidental refitting or input mutation |
| One permutation spans all windows in a draw; observed/null scores use identical bounds; any single draw regenerates from its key; ties and NaN are handled as the reference handles them; on sorted cells the set-aside pairs are the reference's, windows whose continuous center did not move, in every draw, and on clusterless marks those no spike entered or left; each shuffle's behaviour on a real trajectory and on scrambled marks is pinned | Reproducibility, fidelity to the reference's test, and the controls |
| The reference's sorted-cell arithmetic on hand-worked cases, and the port pinned to it on a sorted fixture: same posteriors on the port's fields, same kept windows, same candidates and verdicts (`test_memoryless_reference.py`) | That the port decides as the reference would, apart from the documented adaptations |
| Simulated-data smoke test, then one real epoch | End-to-end |

## Failure behavior

No silent fallbacks. Missing `detector.pkl` or a missing `("", 0)` encoding key raises,
following `scripts/qa/analyze_event_spike_evidence.py`. An epoch with no eligible
immobility returns an empty frame with correct columns **and explicit dtypes** —
`build_classified_nonlocal_event_table`'s empty return is all-object, and concatenating
such frames across epochs silently yields object columns; the memoryless table must not
inherit that. The `results.nc` fast path is never substituted
automatically (D7).

## Run-to-run reproducibility

Two runs of the detector on the same epoch at identical settings are bit-for-bit
equal, because the runner runs XLA in deterministic mode with Triton GEMM off. Every
run made after this change does so, so every epoch runs under one setting (the
coordinator's ruling, 2026-10-11). R12 ran in deterministic mode with Triton GEMM on,
and lost 3 epochs for that reason (below).

Every process that runs JAX gets two flags added to the `XLA_FLAGS` it inherits:
`--xla_gpu_deterministic_ops=true` and `--xla_gpu_enable_triton_gemm=false`. These are
the Dask GPU workers, or the runner itself when it runs the epochs one after another.
The user's own flags are kept. When a flag is given more than once, XLA uses its last
value, so the runner adds each flag unless the last occurrence of that flag already has
the value the runner needs. A flag that is already right is not added twice. XLA reads
`XLA_FLAGS` once, when JAX starts in a process, so the runner sets them before that: in
each worker's environment when the worker is started, and in its own environment while
it runs an epoch. Its own environment is restored afterwards. The runner works the
flags out once and records them in the run manifest's settings (`xla_flags`). The
workers and the runner get that same value, so the manifest says what they ran with. A
resume under the same run id with other flags is refused, like any other change of
settings.

Triton GEMM is off because deterministic mode alone failed some epochs. Deterministic
ops turn off XLA's autotuning, and the default Triton GEMM setup then fails on some
shapes. With Triton GEMM off, XLA sends these matrix products to cuBLAS. R12 (run
`c55cf318d199-seed20260922`, 242 epochs) used deterministic mode with Triton GEMM on and
lost 3 epochs: chimi20200224_08_r4_HPC, chimi20200227_04_r2_HPC and
chimi20200228_02_r1_HPC. Each failed in the shuffle with
`JaxRuntimeError('CANCELLED: Too small divisible part of the contracting dimension.')`,
at `non_local_detector/likelihoods/clusterless_kde.py:89`
(`spike_waveform_feature_distance.T @ position_distance`). All three ran without error
on 2026-10-05, with neither flag. The reason given above is the working
explanation (autotuning off); the spike below tested the fix, not the mechanism.

The R12 Triton spike (2026-10-11) measured it on breeze, one epoch per GPU, with the
canonical R11 decodes and seed 20260922:

- With deterministic ops alone, chimi20200224_08_r4_HPC fails again, 4:34 into the run.
- With `--xla_gpu_enable_triton_gemm=false` added, it runs: 233 candidates, 11 replay,
  7 passing the shuffle. The 2026-10-05 run, with neither flag, had 232, 11 and 7.
- Two such runs are bit-for-bit equal; only `run_id` differs.
- On a passing epoch, chimi20200225_02_r1_HPC, the output with Triton GEMM off differs
  a little from R12's (Triton GEMM on, both deterministic). The counts are the same:
  237 candidates, 4 replay, 3 passing the shuffle. 38 spread gates and 73 jump gates
  flip, out of about 291,000 windows that have a spread (295,369 windows in all). One
  event's bounds move by a window. The null spread differs by at most 1.7e-4 relative.
  Wall time is 4:56 against 4:20.

So Triton GEMM on and off agree on the counts and differ slightly at the window level.
R12 is rerun whole under a new run id, so that every epoch runs under one setting.

Every run manifest records this in its settings (`known_limits.bit_reproducible`,
`run_memoryless_detection.KNOWN_LIMITS`): the measurement above, the earlier ones as
history with the flags each used, and a note that R12 used Triton GEMM and lost 3
epochs.

Before the Triton flag, the runner set deterministic mode only. The R12 smoke test
(2026-10-10) measured that on one epoch: chimi20200225_02_r1_HPC, seed 20260922, with
the decodes read from the R11 staging folder. The only flag was
`--xla_gpu_deterministic_ops=true`, so Triton GEMM was on.

- Two runs with the flag are bit-for-bit equal. Every column of the events, shuffle
  draws, observed and candidates outputs matches; only `run_id` differs.
- Each run took 4:20 wall clock, against 5:10 without the flag.
- A run without the flag differs from them: 111 of 295,369 windows flip a gate, the
  spread differs by up to 0.35 % relative, and one event's bounds move by a window.
  The counts are equal: 237 candidates, 4 replay, 3 passing the shuffle.

Before deterministic mode, with neither flag, two runs were not bit-for-bit equal. Task
17a (2026-10-09) measured it twice:

- chimi20200225_02_r1, two runs on one GPU, both with one thread (the runner before
  and after its thread fix, which with one thread does the same computation):
  19,199 windows' numbers differ, the largest relative difference is 1.2e-4, and one
  window's gate flips. The report gives no event-level comparison for this pair.
- j1620210706_02_r1, the runner before and after its thread fix: 6,326 of 631,632
  windows' spreads differ (largest relative difference 1.3e-5), and one window's gate
  flips; every event-level output is equal (431 candidates, 51 replay, 50 passing
  the shuffle, `is_swr`).

Task 17a could not say why (its best guess was the GPU likelihood). On the R12 epoch
the flag removes the differences, which points to XLA's GPU operations that are not
deterministic by default. Runs made without the flag can be compared only on their event-level
outputs, not on window-level numbers or file hashes, as Task 17a's acceptance was.

## Unverified assumptions

Carried deliberately; each must be checked against real data before results are cited.

1. **`log_likelihood` layout in `results.nc`** — its `(n_time, n_state_bins)` column
   structure is inferred from the writing code, not from a file. Gates D7 only.
2. **The saved environments were built with `edge_spacing > 0`** — junction
   placeholders exist only then; with zero spacing every bin is interior and there is
   nothing to exclude. Check `environment.pkl` for the epochs used. (Whether the
   placeholders are interior is no longer open: `get_track_grid` builds the `-1` rows
   from `~is_track_interior`, so they are excluded by construction.)
3. **Which epochs' decodes carry `log_likelihood`** — depends on when each was last run
   relative to 2026-06-26.
4. **Window likelihoods match direct computation for saved models and long epochs** —
   the additive algebra depends on matching sample units, spike support, and boundary
   conventions; numerical accuracy still needs the explicit tests above.
5. **Missing LFP is not visible in `SWR_event`** (resolved, kept for the record). The
   label is written as an integer and is never NaN, so an event during missing LFP would
   read as observed and non-SWR. Coverage is read from `SWR_zscore` instead, which
   `get_data_over_time` reindexes onto the grid with a 4 ms tolerance and leaves NaN
   where there is no LFP (`observed_intervals`). The canonical event table has the same gap.
6. **`data_over_time` sits on the decode's uniform 500 Hz grid, without gaps** — the
   runner uses its index as the decode grid, and `validate_time_grid` refuses anything
   else, so a violation stops the run rather than passing silently. Checked on the five
   pilot epochs (2026-10-02): every grid is uniform to 1.1e-4 of a sample about its own
   rate, and that rate is 0.4-1.7 ppm fast of 500 Hz, which over an epoch drifts most
   of a sample from an exact 500 Hz grid. Uniformity is therefore judged about the
   grid's own mean rate, and that rate must be within 1e-4 of the configured one.
7. **How much position a tetrode's identity carries** — it decides how weak the
   within-electrode shuffle is (see "What the jump half of the test does on simulated
   data"). Measure it on the pilot epochs.
8. **Open decision: non-SWR events are decoded partly from their own spikes.** The
   decode trains on `no_ripple`, which excludes only ripple times
   (`data_loaders/utils.py:get_training_timepoints`), so the spikes of an immobile
   non-SWR event are also encoding spikes, while an SWR event's are not. In the observed
   decode each such spike's own mark-position pair puts weight at the animal's place;
   the shuffle removes it. The reference is out-of-sample throughout (rate maps from
   running above 10 cm/s). The asymmetry falls on the SWR/non-SWR split, and it applies
   to our state-space cohort as well, since both read one encoding model. Two remedies
   are open: leave-event-out encoding, dropping encoding spikes inside the candidate's
   decoding support from the kernel sums; or the running-only refit spec 03 already
   defers. **Decided 2026-09-22: accepted for now.** Both cohorts carry the asymmetry
   equally; revisit if a non-SWR fraction becomes a headline claim. Reconfirmed
   2026-09-25: no running-only refit on the pilot.
