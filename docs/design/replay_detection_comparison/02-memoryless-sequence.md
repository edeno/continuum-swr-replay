# Memoryless sliding-window sequence detection (Widloski & Foster 2025)

*Design spec. Status: revised after review and clarification of the scientific goal;
not yet implemented. The commitments below are proposed analyses, not results.*
*Governing rule and cross-cutting decisions: [overview.md](overview.md).*

Inventory 6 of six: the posterior-sequence trigger. How this cohort is compared against
the other five is [03-inventory-comparison.md](03-inventory-comparison.md); the
initial event-feature and selection diagnostics are in [04](04-event-features.md).
Threshold robustness is a planned later stage in 04, and performance validation remains
parked in [05](05-validation.md). Neither blocks the default-setting comparison.

## Purpose

Port the Widloski & Foster sliding-window sequence detector so that it runs on our
recordings, producing inventory 6 of the six in [overview.md](overview.md).

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
directory's README records provenance, licences, pinned commit, and the functions that
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
| Uncertainty | Posterior compressed to centre, spread, and jump statistics for selection | Joint state/position posterior; non-local state probability is thresholded | Examine diffuse or multimodal events without treating our own posterior confidence as validation |
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
3. **Per-window statistics.** Posterior centre of mass; *posterior spread* as the square
   root of the second central image moment; *jump size* as the L2 norm between
   consecutive COMs. The reference pads the jump vector at the end
   (`jumps = [jumps; jumps(end)]`, driver line 82), so window k carries the **forward**
   jump to window k+1 and the last window repeats its predecessor's; the Mallory `_cm`
   filter pads a zero at the front instead (backward jump). The port uses the forward
   convention. (`compute_centerOfMass.m`, `compute_imageSpread.m`)
4. **Bin filter.** Keep windows with spread < 10 cm, jump < 20 cm, speed < 5 cm/s.
   Rejected windows become NaN. (`compute_filtering_binDecoding_cm.m`)
5. **Sequence assembly.** Contiguous non-NaN runs; merge neighbours separated by
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
- **Two bugs in the published shuffle script** (`zenodo_15199609/load_replayEvents_cellIDshuffle.m`),
  which affect that script's significance-filtered outputs: line 137 passes the all-NaN p-value
  vector to `compute_sequenceJumps` instead of the posterior COM, so that script's
  jump-size p-values are meaningless; and lines 122–123 index `timeBins_sub{i}` by loop
  counter while writing to `ind_replay(i)`, re-decoding the wrong candidate. Recorded
  as implementation discrepancies, not evidence that the paper omitted significance.
  `load_replayEvents_cellIDshuffle_sameEvents.m` computes jumps from the posterior COM.
  The clusterless adaptation below uses a simpler encoding-pair permutation.
- **Run lengths are index differences.** `compute_allSequences_NaNseparated_merge.m`
  recomputes `lengths = boundaries(:,2) − boundaries(:,1)` for every run, merged or not,
  so the 100 ms filter (`lengths < sequence_durationThr/shiftSizeDecoding`, driver line
  96) drops runs of index length below 20: a surviving run has at least 21 windows,
  spanning ≥ 100 ms between first and last midpoint. At our 4 ms stride that is index
  length 25, i.e. 26 windows (V10). The *pre-merge* length convention comes from
  `seqle`, which is vendored nowhere (see the reference README); the port defines run
  length as an index difference throughout and does not depend on it.
- **Forward-jump consequence.** Because window k's jump is to k+1, a window whose
  successor was rejected by the speed gate (NaN centre) carries a NaN jump and is itself
  rejected. Sequences therefore end one window before the first speed-rejected window.

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
distributions. Use `p = (1 + count(null_stat <= observed_stat)) / (100 + 1)` and
require both p-values < 0.05. Keep zero jumps and count ties conservatively, because
our discrete graph centres can repeat exactly. These differ from the MATLAB script's
zero-jump omission and strict `<` comparison. Non-finite observed or null statistics
make the test untestable; they must not silently count as a pass or shrink the denominator.

**Interpretation.** This tests the association between waveform marks and encoding
locations. It preserves spike timing and the marginal encoding distributions, but
unlike cell-ID permutation it can blur conditional spatial tuning instead of swapping
intact fields. It therefore emulates the association-breaking purpose and the
event-scoring procedure, not the exact null distribution. Call it an **encoding-pair
shuffle** in the methods. In the pilot, inspect both null statistics: broad posteriors
can have nearly stationary centres, so the effect on the jump test must be checked
rather than assuming this shuffle is uniformly more permissive or more stringent.
Report the comparator cohort both before and after this filter; retain our native
state-space definition. No component model or general detector-calibration study is
required to implement this adaptation.

## Decisions register

Decisions taken during design, with the reasoning, so the methods section can state them
without re-litigation.

| # | Decision | Reasoning |
|---|---|---|
| **D1** | Explain event agreement and disagreement with a Foster-style comparator using the same encoding model | Isolates temporal inference and selection choices; directly addresses reviewer concerns about what our model adds or misses |
| **D2** | Use the encoding-pair shuffle above for the Foster-style comparator; report candidates before and after filtering | Simple adaptation using saved KDE arrays, with fixed event spikes and the original spread/jump scoring intent. It does not preserve whole cell tuning curves, so its p-values are not equivalent to cell-ID shuffle p-values |
| **D3** | Use our canonical whole-event immobility rule (< 4 cm/s), not theirs (per-bin < 5 cm/s) | Makes both cohorts draw from the same speed-eligible population, so a cohort difference is attributable to the detection algorithm rather than the eligibility rule |
| **D4** | Centre = discrete Fréchet mean over real bin centres using track distance; spread = sqrt of the minimized expected squared distance | Avoids averaging unrelated linearized coordinates. Bin restriction approximates COM/SD on a straight segment; deterministic ties use bin order, without a temporal preference. Dispersion applies the same minimization to the retained sequence centres |
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
| **V2** | Cell-ID shuffle, p < 0.05 on spread and jump | Within-tetrode encoding-pair shuffle; both p-values < 0.05 | Uses existing clusterless arrays without component fitting (D2) | Scrambling individual mark–position pairs can blur tuning; original cell-ID permutation exchanges intact curves. Null distributions and accepted cohorts can differ. Quantized graph centres also motivate retaining zero jumps and conservative ties |
| **V3** | Tuning curves get `+1e-3` before decoding | Clusterless KDE with its own `EPS` clipping of the ground-process intensity | Different likelihood construction | Low-occupancy bins are regularized differently. Not tunable to match; the two decoders differ in their floor behaviour |
| **V4** | Rate maps fit on running periods (> 10 cm/s) | Encoding model fit on the decode's `is_training` mask (`training_type = "no_ripple"`) | Holds encoding fixed between our two procedures | Non-ripple training includes immobility. It differs from the paper, but is shared in the main comparison; any running-only sensitivity must refit both procedures on that same mask |

### Forced by track geometry

| # | Original | Ours | Why | Effect |
|---|---|---|---|---|
| **V5** | 2D open field, 90 × 90 cm | 1D linearized three-legged track graph, 9 arms | The task's geometry | Sequences are constrained to the graph; "direction" is along-arm rather than an angle |
| **V6** | Euclidean distance for jump and dispersion | Track-graph distance | On a linearized branched track, `abs(Δlinear_position)` is not the distance between two places — bins on different arms can be adjacent in linear coordinate | Using linear differences would under-count jumps at arm boundaries, producing spuriously smooth and spuriously long sequences. Track distance avoids this; it also makes our jumps non-comparable in absolute terms to an open-field jump |
| **V7** | Posterior COM | Discrete Fréchet mean over track distance (D4) | Linearized-coordinate COM does not respect branching | On straight segments the centre is quantized to a bin; near junctions geometry also changes it |
| **V8** | Unvisited arena bins excluded | Junction placeholder bins (`edge_id == -1`) excluded before normalization | These bins are linearization artifacts from `edge_spacing > 0`, not places. In `get_track_grid` the `-1` rows are built from `place_bin_centers[~is_track_interior]`, so they are non-interior by construction and the primary likelihood path never returns them; the guard matters on the `results.nc` path, where `log_likelihood` is re-expanded to every `state_bins` column | If not excluded, probability mass lands on non-places and drags the centre toward gaps. Same hazard `map_estimate_to_arm` guards against |
| **V9** | Spread is a 2D product moment (area-like, bin units) | Graph-distance RMS spread (a length, cm) | Provides a spread measure on our geometry, approximating SD on a straight segment | The 10 cm cutoff is an adapted choice prioritized in the later threshold-robustness study (04); its robustness is not established by the initial report, and these spreads are not numerically comparable to the paper's 2D spreads |

### Forced by our sampling grid

| # | Original | Ours | Why | Effect |
|---|---|---|---|---|
| **V10** | 5 ms step | 4 ms step (2 samples at 500 Hz) | 5 ms is 2.5 samples on our native grid | Slightly finer temporal resolution on sequence boundaries. Every bin-count threshold must be **derived from the stride**, never carried over as the original's literal constants |
| **V11** | 80 ms window | 80 ms window (40 samples) | Exact at 500 Hz | None |

### Deliberate, for comparability

| # | Original | Ours | Why | Effect |
|---|---|---|---|---|
| **V12** | Per-bin speed < 5 cm/s | Per-bin < 4 cm/s for candidacy, then our canonical whole-event rule | D3 | Slightly stricter than theirs. Shared with our cohort by construction |
| **V13** | Paper's ripple detection and ripple-less classification | This cohort is labelled under **each** SWR definition in the comparison (Kay, Karlsson, Yu 2017), each speed-qualified and sampled the same way, with strict positive-duration overlap. No single definition is imposed (overview X6) | Imposing one SWR definition on a sequence cohort would blend methods; reporting the matrix also supplies label sensitivity without a separate sweep | Both ripple detection and replay/ripple association differ from the paper, so absolute non-SWR fractions cannot reproduce its headline percentages. The reported non-SWR fraction is a function of the SWR definition as well as the events, and must always be quoted with the definition attached |

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
| Sequence / replay duration | 100 ms | 100 ms | 50 ms in the commented "John's criterion" block | 100 ms |
| Significance | Required for replay | Detection driver filters by both p-values, with the bugs noted above; downstream usage varies | `load_ind_replayEvents` alone selects duration/dispersion candidates | Encoding-pair shuffle adaptation, with candidates retained for comparison (D2/V2) |

## Architecture

New subpackage `src/continuum_swr_replay/analysis/hippocampus/memoryless_sequence/`.
Named for what it does; the citation lives in docstrings, not the module path.

| module | responsibility | depends on |
|---|---|---|
| `config.py` | frozen dataclass of every threshold — the `Analysis_Information.mat` equivalent, one source of truth | — |
| `likelihood.py` | prior-excluded `(n_time, n_interior_bins)` log-likelihood on the native 2 ms grid | `detector.pkl` (primary), `results.nc` (opt-in) |
| `statistics.py` | pure functions: window aggregation, normalization, centre / spread / jump / dispersion | numpy, bin-distance matrix |
| `detect.py` | bin filter → sequence assembly → merge → criteria → event table, retaining rejection diagnostics | the above, plus existing repo helpers |
| `shuffle.py` | encoding-pair permutations, re-decoding fixed candidate windows, spread/jump p-values | encoding arrays, event marks, likelihood and statistics helpers |
| `compare.py` | interval-pair table, event summaries, selection diagnostics, and per-session/animal comparisons | both event tables, diagnostic traces, canonical SWRs |

`likelihood.py` exposes one function with two implementations behind it, so the
`results.nc` fast path (D7) is a swap, not a fork in the caller.

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
4. Per-window centre and spread (D4, V7). Then, **before any jump is computed**, set
   the centre of every window whose midpoint speed is not below 4 cm/s to NaN, as the
   reference does by decoding only speed-eligible windows. Then forward jumps (V6):
   window k's jump is to window k+1, the last window repeats the previous jump, and a
   window whose successor has a NaN centre gets a NaN jump. The ordering is load-bearing:
   taking jumps before the speed gate lets a sequence extend one window further into the
   speed-rejected region and can turn a rejected 100 ms candidate into an accepted one.
5. Keep windows with spread < 10 cm and finite jump < 20 cm (the speed gate has already
   been applied through the NaN centres).
6. Contiguous runs of kept windows → merge across gaps < 20 cm and < 50 ms → drop runs
   shorter than 100 ms.
7. Keep events with duration > 100 ms and dispersion > 12 cm.
8. Re-check whole-event immobility via the existing `event_speeds_are_immobile` /
   `whole_event_immobility_mask`.
9. Use the same SWR intervals as `build_classified_nonlocal_event_table`: apply
   `filter_event_labels_by_speed` to canonical `SWR_event` labels and reconstruct
   bounds with `intervals_from_labels`. Classify with `classify_swr_overlap_geometry`
   and the canonical matched-SWR convention. Raw Kay intervals are not interchangeable
   with this qualified, sampled population.
10. Apply the encoding-pair shuffle to eligible candidates, retaining p-values and
    pass/testability flags on every candidate row. Derive the filtered comparator
    cohort by subsetting; candidate IDs and bounds do not change.

Both cohorts use the same canonical time axis and speed trace. Step 8 evaluates every
native speed sample between sequence endpoints, including rejected windows inside
merged gaps, not just retained window centres. Step 9 uses an identical SWR interval
table on both sides; calling the same overlap helper alone would not ensure parity.

**Window and event timing.** A window with support `[t, t + 0.080)` is timestamped at
`t + 0.040`. Interpolate speed at that midpoint for the step 4 gate. Event start/end are the
first/last retained midpoint, and the duration criterion is their difference, as in
the reference driver. Save the outer decoding support separately; it is not the
event interval used for overlap or speed eligibility. Do not add the 80 ms window
width to event duration. Retain original timing fields for our existing cohort,
whose canonical duration includes a sample-width convention; compute a common
`interval_span_s = event_end_s - event_start_s` for duration comparisons and overlap.
Native probability/duration criteria remain unchanged.

**Merged gaps.** Retain the per-window keep mask and the pre-merge sequence IDs.
Compute dispersion from the finite retained centres, as in the reference's NaN-masked
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

Table `memoryless_event_table`, one row per detected candidate sequence (D2).

- **Identity** — `animal`, `day`, `epoch`, `brain_area`, `sequence_label`, `ms_event_id`,
  `run_id` (joins to the frozen configuration and source-artifact manifest)
- **Timing (seconds)** — `event_start_s`, `event_end_s`, `event_duration_s`, `event_midpoint_s`
- **Decoding support (seconds)** — `support_start_s`, `support_end_s`, separate from event bounds
- **Detector statistics** — `n_windows`, `mean_posterior_spread_cm`, `max_jump_cm`,
  `mean_jump_cm`, `dispersion_cm`, `path_distance_cm` (summed track-graph distance
  along the centre trajectory, the analogue of the original's `compute_sequenceDistance`;
  reported, not used as a criterion)
- **Speed (cm/s)** — `max_speed`, `mean_speed`, `whole_event_speed_eligible`
- **SWR** — `is_swr`, `overlap_class`, `overlap_subclass`, `n_swrs_overlapping`,
  `matched_swr_event`, `matched_swr_start`, `matched_swr_end`; nullable under 03's
  observation-availability rule, with unavailable labels recorded as unknown
- **Shuffle** — `shuffle_observed_spread_cm`, `shuffle_observed_jump_cm`,
  `p_spread`, `p_jump`, `shuffle_testable`, `shuffle_pass`, `n_shuffles`;
  retain each draw's two scores in a sidecar for the pilot/null-distribution audit

Timing column names mirror the canonical event table so comparison code is symmetric.
`ms_event_id` is deliberately a different key from `event_id` (D6).
The run manifest records model/artifact hashes, code versions, thresholds, time grid,
encoding/training provenance, shuffle definition/seeds/scoring conventions, and any skipped epochs.
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
| Statistics against hand-constructed posteriors with closed-form answers | Centre / spread / jump / dispersion correctness |
| Discrete Fréchet mean matches a brute-force minimum and approaches COM/SD as a straight-segment grid is refined; deterministic tie cases | Quantization and the precise D4 contract |
| Sequence assembly against the reference semantics: run length is an index difference, a single-window run has length 0 and is dropped, and the 100 ms filter keeps runs of ≥ 21 windows at a 5 ms stride and ≥ 26 at 4 ms | Off-by-one and merge behaviour |
| Jump alignment: window k's jump is to k+1, the last window repeats it, and a window before a NaN centre is rejected | Forward-jump convention; boundaries would shift by one stride under the backward convention |
| Speed gate precedes jumps: a candidate whose last eligible window precedes a speed-rejected window is one window shorter than it would be with jumps taken first; construct the case where that changes acceptance | The ordering in data-flow step 4 |
| Junction-bin test: mass on an `edge_id == -1` bin must not move the centre | V8 |
| Stride-derived thresholds: changing the stride must change bin-count thresholds | V10 — the constant that would silently become a 50 ms threshold |
| Window midpoints determine duration and overlap; outer support is separate; rejected merged gaps still participate in speed eligibility | Temporal conventions do not change the cohort silently |
| Identical event intervals receive identical SWR labels in both pipelines, including a movement-spanning SWR and sampled boundary contacts | Shared qualified SWR population and strict overlap |
| One-to-many/many-to-many overlaps count each event once per detector; matched events may have different SWR labels; disjoint and empty cohorts | Denominators, matching, and empty-cohort handling |
| Rejection diagnostics and gate-specific reruns agree on synthetic examples | Explanations of selection follow the actual pipeline |
| Encoding-pair shuffle preserves observed spike/mark arrays, encoding marginals, and ground intensity; identity permutation reproduces the original likelihood | The intended null operation and absence of accidental refitting or input mutation |
| One permutation spans all windows in a draw; observed/null scores use identical bounds; ties, zero jumps, and non-finite scores are handled explicitly | Reproducibility and valid spread/jump comparisons |
| Simulated-data smoke test, then one real epoch | End-to-end |

## Failure behavior

No silent fallbacks. Missing `detector.pkl` or a missing `("", 0)` encoding key raises,
following `scripts/qa/analyze_event_spike_evidence.py`. An epoch with no eligible
immobility returns an empty frame with correct columns **and explicit dtypes** —
`build_classified_nonlocal_event_table`'s empty return is all-object, and concatenating
such frames across epochs silently yields object columns; the memoryless table must not
inherit that. The `results.nc` fast path is never substituted
automatically (D7).

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
