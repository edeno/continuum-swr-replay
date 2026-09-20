# Comparing state-space events with Widloski & Foster sequence detection

*Design spec. Status: revised after review and clarification of the scientific goal;
not yet implemented. The revisions below are proposed analysis commitments, not results.*

## Purpose

Anticipate reviewer concerns by determining **which events the two methods detect,
how those events differ, and which modeling choices explain the differences**.
The non-SWR fraction is one outcome, alongside event duration, spatial extent,
trajectory structure, spike evidence, and event boundaries.

The central comparison uses the same recordings and fitted encoding model. It changes
the temporal inference and event-selection procedure. Agreement is evidence of
robustness to those choices; it is not independent validation of the recordings or
encoding model. Disagreement can reflect different scientific targets as well as
different sensitivity or false detections.

Our claim to a more principled treatment is specific: joint inference of position
and representational state, explicit Local and No-Spike alternatives, and propagation
of spatial and state uncertainty through time. Whether those features improve detection
must be tested. More events, smoother trajectories, or larger model posteriors do not
by themselves establish superiority. The implementation should be reusable across
sessions; HPC clusterless decoding is the initial comparison, with other likelihood
backends needed before extending it to other brain areas.

The comparison method is Widloski & Foster (2025), *Replay without sharp wave ripples
in a spatial memory task*, Nature Communications 16.
doi:[10.1038/s41467-025-65181-5](https://www.nature.com/articles/s41467-025-65181-5) ·
[open full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC12639102/) ·
[preprint](https://www.biorxiv.org/content/10.1101/2024.05.13.593931v1).

Source code for the original is vendored at
[reference/widloski_foster_2025/](../../reference/widloski_foster_2025/); that
directory's README records provenance, licences, pinned commit, and the functions that
exist in neither published source. File references below of the form
`foster_lab_matlab/x.m` and `zenodo_15199609/x.m` point into it.

## What the methods select

Both methods use Bayesian spatial decoding and threshold-based event selection.
The distinction to test is how temporal structure, competing explanations, and
uncertainty enter that selection.

| Aspect | Widloski & Foster procedure | Our current state-space procedure | Comparison to make |
|---|---|---|---|
| Target | Extended, smooth decoded sequences | High-probability non-local representation, continuous or fragmented | Report the full populations and a secondary comparison restricted to extended trajectories; stationary or fragmented events are outside the original sequence target |
| Time | Independent spatial decoding in overlapping 80 ms windows, followed by continuity and merge rules | Inference on the native 2 ms grid with state and position transitions and acausal smoothing | Measure boundary shifts, splits/merges, short-event recovery, and sensitivity to smoothing assumptions |
| Alternatives | Spatial posterior followed by sequence criteria; no explicit Local/No-Spike state competition | Local, No-Spike, Non-Local Continuous, Non-Local Fragmented | For comparator-only events, inspect local coding, sparse activity, and non-local evidence; allow that our model may miss supported sequences |
| Uncertainty | Posterior compressed to centre, spread, and jump statistics for selection | Joint state/position posterior; non-local state probability is thresholded | Examine diffuse or multimodal events without treating our own posterior confidence as validation |
| Event selection | Duration > 100 ms, dispersion > 12 cm, spread/jump/merge criteria, and published shuffle significance | Combined non-local probability > 0.8, the current 15 ms minimum confidence-run duration, then whole-event immobility | Attribute expected selection differences before making sensitivity claims; our probability/duration thresholds also require sensitivity analysis |

The current state-space contract comes from `get_data_over_time`,
`get_confident_times`, and `build_speed_eligible_event_cohort`, not from a new
definition written for this comparison. Read thresholds and fitted transition
parameters from the actual run and record them. A 2 ms grid is not a demonstrated
2 ms effective resolution: our acausal smoother can also shift or extend boundaries.
Likewise, overlapping memoryless windows reuse spikes and induce temporal dependence;
"memoryless" does not imply an absence of smoothing in the resulting trajectory.

## Reviewer questions and discriminating analyses

| Concern | Analysis | What would support the claimed advantage |
|---|---|---|
| Are the extra events just shorter or less spatially extended by definition? | Separate duration, dispersion, and continuity failures; report both native cohorts and a comparison within common duration/extent ranges | Additional evidence-supported events within comparable ranges, or a clearly stated expansion to a different representational target |
| Does the state-space prior manufacture non-local events? | Prior-excluded spike-evidence diagnostics, transition-prior sensitivity, and held-out evidence or simulation where available | Remote evidence persists without selecting and scoring on the same posterior; conclusions survive reasonable prior changes |
| Does the sliding window manufacture smoothness or combine distinct events? | Window/stride and merge-rule sensitivity; boundary and split/merge analysis | Observed changes follow the predicted temporal operation; simulations establish which boundaries are recovered correctly |
| Are comparator-only events real sequences our model rejects? | Inspect all four state probabilities and spike evidence on those intervals, including well-supported counterexamples | Explain rejection without assuming our labels are truth; document failures of our Local model or temporal prior if found |
| Is the apparent benefit driven by encoding or threshold choices? | Hold encoding fixed in the main comparison; vary detector thresholds and, separately, fit both procedures on the same running-only training mask | The relevant differences persist under these controls; if they do not, narrow the claim accordingly |
| Does a detector difference explain the non-SWR finding? | Use identical qualified SWRs; show duration-stratified non-SWR fractions and boundary-related label disagreements | Non-SWR conclusions persist beyond differences in interval duration or boundary placement |

**Two levels of evidence.** The first deliverable is a descriptive event comparison
and a record of algorithmic selection differences. Claims about better sensitivity,
specificity, or calibration additionally require a validation study. Use simulations
with planted traveling, stationary remote, fragmented, local, and no-sequence activity,
varying spike counts and durations; include generators that do not reproduce our
transition assumptions. Score non-local representation and extended traveling sequences
as separate targets, so a method is not penalized for omitting events outside its remit.

A possible real-data validation is detection on a discovery set of tetrodes followed
by scoring neural evidence on held-out tetrodes, with encoding and threshold tuning
kept separate from evaluation. This requires rerunning **both** detectors on the
discovery set; scoring held-out channels after using all channels to select events
does not qualify. Define a common predictive scoring rule before performing that
study. Prior-excluded likelihood evaluated on the discovery spikes remains a useful
diagnostic, but is not independent validation. The initial comparison does not wait
for this study; its interpretation remains descriptive until validation is available.

## The original method

Established from the paper's Methods together with the vendored code; where the two
disagree, the disagreement is recorded in the deviations register rather than silently
resolved.

1. **Rate maps.** 2 cm bins over a 90 × 90 cm arena; occupancy-normalized spike counts
   from running periods (> 10 cm/s); smoothed with an isotropic Gaussian, 8 cm SD.
   Cluster inclusion: noise overlap < 0.03, isolation > 0.95, peak SNR > 1.5. Place-cell
   inclusion: rate > 0.01 Hz, spatial information > 0.5 bits/spike.
   (`foster_lab_matlab/compute_rateMap.m`, `load_AnalysisInformation_cm.m`)
2. **Decoding.** Memoryless Bayesian, uniform spatial prior, Poisson likelihood, in
   80 ms windows stepped by 5 ms. Every tuning curve gets `+1e-3` added before decoding.
   (`foster_lab_matlab/compute_BayesianDecoding_fast.m`)
3. **Per-window statistics.** Posterior centre of mass; *posterior spread* as the square
   root of the second central image moment; *jump size* as the L2 norm between
   consecutive COMs. (`compute_centerOfMass.m`, `compute_imageSpread.m`)
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
| **V8** | Unvisited arena bins excluded | Junction placeholder bins (`edge_id == -1`) excluded before normalization | These bins are linearization artifacts from `edge_spacing > 0`, not places | If not excluded, probability mass lands on non-places and drags the centre toward gaps. Same hazard `map_estimate_to_arm` guards against |
| **V9** | Spread is a 2D product moment (area-like, bin units) | Graph-distance RMS spread (a length, cm) | Provides a spread measure on our geometry, approximating SD on a straight segment | The 10 cm cutoff is an adapted choice requiring sensitivity analysis; these spreads are not numerically comparable to the paper's 2D spreads |

### Forced by our sampling grid

| # | Original | Ours | Why | Effect |
|---|---|---|---|---|
| **V10** | 5 ms step | 4 ms step (2 samples at 500 Hz) | 5 ms is 2.5 samples on our native grid | Slightly finer temporal resolution on sequence boundaries. Every bin-count threshold must be **derived from the stride**, never carried over as the original's literal constants |
| **V11** | 80 ms window | 80 ms window (40 samples) | Exact at 500 Hz | None |

### Deliberate, for comparability

| # | Original | Ours | Why | Effect |
|---|---|---|---|---|
| **V12** | Per-bin speed < 5 cm/s | Per-bin < 4 cm/s for candidacy, then our canonical whole-event rule | D3 | Slightly stricter than theirs. Shared with our cohort by construction |
| **V13** | Paper's ripple detection and ripple-less classification | Canonical speed-qualified, sampled `SWR_event` intervals and strict positive-duration overlap | Both cohorts must use the identical SWR population and overlap convention | Both ripple detection and replay/ripple association differ from the paper. Absolute non-SWR fractions cannot reproduce its headline percentages; within-recording detector comparisons are the target |

### Retained settings and adapted thresholds

Place bin size (2.0 cm = their `binSize`), decoding window (80 ms), uniform spatial
prior, memoryless decoding, jump threshold (20 cm), merge
thresholds (20 cm / 50 ms), minimum duration (100 ms), dispersion threshold (12 cm).

Their application changes with track distance and the 4 ms stride. The spread cutoff
of 10 cm is an additional adapted setting (V9), not an exactly preserved criterion.
Report sensitivity to these choices alongside the default comparison.

### Discrepancies within the original, and how we resolved them

| Item | Paper | Zenodo scripts | Helper library | We use |
|---|---|---|---|---|
| Ripple band | 100–220 Hz | — | `[150,250]`, comment says "John used 120-170" | N/A (V13) |
| Ripple merge gap | merge if < 50 ms | `eventMergeThr = 0` (no merge) | — | N/A (V13) |
| Spike-density smoothing | 80 ms SD | 100 ms SD | — | N/A (bursts not ported) |
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
2. Drop junction bins (`place_bin_centers_nodes_df_.edge_id == -1`) **before** normalizing (V8).
3. Sliding 80 ms windows as a float64 cumulative sum over 40 rows, stride 2 rows;
   normalize across real bins under a uniform prior. Validate uniform 2 ms spacing
   and model sampling units; do not aggregate across missing samples or epoch bounds.
4. Per-window centre, spread, jump (D4, V6, V7).
5. Keep windows with spread < 10 cm, jump < 20 cm, per-bin speed < 4 cm/s.
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
`t + 0.040`. Interpolate speed at that midpoint for step 5. Event start/end are the
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
exist. Step 5 establishes candidacy; step 8 governs. Step 8 is not redundant: merging at
step 6 can span windows that step 5 rejected.

**The efficiency claim.** In the `uv.lock` version of `non_local_detector`
(`2d730c3a86ad0c62a817d0356b9248d43b6c32f8`),
[`fit_clusterless_kde_encoding_model`](https://github.com/LorenFrankLab/non_local_detector/blob/2d730c3a86ad0c62a817d0356b9248d43b6c32f8/src/non_local_detector/likelihoods/clusterless_kde.py)
divides spike count by the number of training samples. The non-local likelihood
adds a per-row `-summed_ground_process_intensity` and per-spike mark terms. Thus
row sums give a window likelihood when the sample duration and spike support match.
Validate those units for the saved model as well as the current library version.

The direct-call test must respect the library's edge convention: a call on 40
timestamps spans only 78 ms between its first and last timestamp, and the likelihood
function clips spikes at the last timestamp. An interior 40-row slice of a full-epoch
likelihood can contain spikes from the following 2 ms interval. Compare against a
direct marked-process sum over the explicit half-open 80 ms support, with 40
ground-process contributions; test boundary spikes and epoch tails. A naive call on
the sliced timestamp vector is not a valid reference. Also check long-epoch numerical
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
  `matched_swr_event`, `matched_swr_start`, `matched_swr_end`
- **Shuffle** — `shuffle_observed_spread_cm`, `shuffle_observed_jump_cm`,
  `p_spread`, `p_jump`, `shuffle_testable`, `shuffle_pass`, `n_shuffles`;
  retain each draw's two scores in a sidecar for the pilot/null-distribution audit

Timing column names mirror the canonical event table so comparison code is symmetric.
`ms_event_id` is deliberately a different key from `event_id` (D6).
The run manifest records model/artifact hashes, code versions, thresholds, time grid,
encoding/training provenance, shuffle definition/seeds/scoring conventions, and any skipped epochs.
Retain per-window statistics and individual gate masks, pre/post-merge candidates,
and candidate rejection flags in diagnostic outputs; accepted events alone cannot
explain why a detector rejected activity. These outputs are scientific diagnostics,
not additional acceptance criteria.

## Comparison artifact

### Matching and counting

Use complete native cohorts from the same available epochs. Neither cohort is a
truth set. Present the comparison against both the unfiltered candidates and the
shuffle-filtered comparator, clearly labeled. Our state-space cohort stays fixed.
Build a table of **every** strict positive-duration overlap pair, carrying
`event_id`, `ms_event_id`, intersection duration, temporal intersection-over-union,
the fraction of each event covered by the other, signed onset/offset differences
(`memoryless - state_space`), and both events' SWR labels. The existing
`event_strict_overlap_keys` can check membership using timebin labels, but returns
neither pairs nor overlap lengths; pairing must operate on the canonical intervals.

Report separately the proportion of state-space events overlapping any comparator
event and the reverse proportion. Each event contributes once to its own denominator,
regardless of its number of matches. Show overlap-length distributions to distinguish
small edge contacts from substantial agreement. Also report time coverage of the
intersection and union of the two cohorts; this limits dependence on segmentation.
Summarize connected components of the bipartite overlap graph as one-to-one,
one-state-space-to-many-comparator, the reverse, or many-to-many. Do not turn these
components into replacement event intervals.

Use **state-space only / comparator only / matched** as display groups, with matched
events still counted separately by detector. Compute non-SWR fractions within each
detector and its own matched subset. Keep a separate table of paired SWR-label
agreement/disagreement; a pair's two labels can differ solely because of its bounds.
Pair counts describe the matching relation and are not independent event counts.

### What differs about the events

Report duration/span, spike count, active tetrode count, spike rate, spatial extent,
distance from the animal, continuous/fragmented state occupancy, posterior uncertainty
and multimodality, trajectory reversals/pauses, ripple power, and task context (well,
time in stop, reward/patch context where available). Use existing trajectory and
spike-evidence helpers where their timing and definitions apply. Compute summaries
for both detectors' intervals from the same underlying traces. Namespace summaries
that depend on a particular posterior (`ss_*`, `ms_*`); differences in posterior
confidence or smoothness partly follow from the procedures that generated them.

For state-space-only events, retain whether comparator window gates fail, no candidate
is assembled, or an assembled candidate fails duration/dispersion or whole-event
speed. Relate each event to nearby rejected candidates as well as accepted candidates.
Retain every failing criterion, not a single assumed cause. Gate-failure summaries
explain selection; they do not independently validate our extra events. Confirm
claimed effects of a gate by rerunning with only that gate changed.

For comparator-only events, examine the state-space probability trace and the Local
and No-Spike alternatives across the same interval, plus prior-excluded spatial
evidence and shorter constituent events. Distinguish boundary/segmentation differences
from a sustained disagreement about representation. Treat credible missed sequences
as limitations of our model, not errors of the comparator by definition.

All main group summaries show per-session values and animal-level consistency.
Use animal-level resampling for uncertainty where justified, with sessions nested
within animal; report pooled event summaries as descriptive. Show native duration
distributions first, then duration-stratified comparisons, since longer intervals
have more opportunity to overlap a ripple. Report missing features and skipped epochs
explicitly; model/data availability defines the paired analysis population.

### Controlled changes and deliverables

Start with the default same-encoding comparison. Then vary one operation at a time:
window width/stride, gap merging, spread/jump gates, and duration/dispersion cutoffs.
Include sensitivity to our non-local probability cutoff and fitted temporal priors.
Freeze the sensitivity configurations before inspecting cohort differences, retain
each run's configuration, and report all tested settings. The main comparison changes
multiple algorithmic choices together; it cannot isolate the contribution of the
temporal prior without these additional controls. Running-only encoding is a separate
control applied to both procedures, not an asymmetric improvement to one comparator.

The initial report contains:

1. The modeling/selection comparison table and deviations from the published method.
2. Event and time-overlap summaries, split/merge counts, and boundary/SWR-label changes.
3. Event-feature distributions and selection diagnostics for unmatched and matched groups.
4. Reproducibly selected examples from each group and failure type, displaying spikes,
   both posterior representations, state probabilities, gate masks, speed, and LFP on
   a shared time axis; include counterexamples and show the sampling rule.
5. The sensitivity results needed to support the stated interpretation. Better
   sensitivity, specificity, or calibration remains a separate validation claim.

## Testing

| test | what it protects |
|---|---|
| **Additivity** — cumsum-window log-likelihood equals a direct marked-process sum on identical half-open spike support, including boundary spikes, epoch tails, and long traces | Correct exposure and spike assignment, as well as numerical stability |
| Statistics against hand-constructed posteriors with closed-form answers | Centre / spread / jump / dispersion correctness |
| Discrete Fréchet mean matches a brute-force minimum and approaches COM/SD as a straight-segment grid is refined; deterministic tie cases | Quantization and the precise D4 contract |
| Sequence assembly against the vendored MATLAB semantics, including its dropping of index-length-0 runs | Off-by-one and merge behaviour |
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
immobility returns an empty frame with correct columns, matching
`build_classified_nonlocal_event_table`. The `results.nc` fast path is never substituted
automatically (D7).

## Unverified assumptions

Carried deliberately; each must be checked against real data before results are cited.

1. **`log_likelihood` layout in `results.nc`** — its `(n_time, n_state_bins)` column
   structure is inferred from the writing code, not from a file. Gates D7 only.
2. **Junction bins are inside `is_track_interior_`** — inferred from the CLAUDE.md
   warning and `map_estimate_to_arm`. If they are already excluded, step 2 is a no-op;
   if they are not, omitting it corrupts every centre estimate.
3. **Which epochs' decodes carry `log_likelihood`** — depends on when each was last run
   relative to 2026-06-26.
4. **Window likelihoods match direct computation for saved models and long epochs** —
   the additive algebra depends on matching sample units, spike support, and boundary
   conventions; numerical accuracy still needs the explicit tests above.
