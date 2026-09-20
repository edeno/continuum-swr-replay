# Memoryless sliding-window sequence detection (Widloski & Foster port)

*Design spec. Status: approved, not yet implemented.*

## Purpose

Detect replay events with a second, methodologically independent detector, and compare
the resulting cohort — split by SWR overlap — against the non-local event cohort our
state-space pipeline produces. The question being answered is whether our
"non-local events without SWRs" population is a property of the events or an artifact
of the state-space detector. Secondary goal: the detector should be reusable across
sessions and brain areas, not a one-off script.

The method being ported is Widloski & Foster (2025), *Replay without sharp wave ripples
in a spatial memory task*, Nature Communications 16.
doi:[10.1038/s41467-025-65181-5](https://www.nature.com/articles/s41467-025-65181-5) ·
[open full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC12639102/) ·
[preprint](https://www.biorxiv.org/content/10.1101/2024.05.13.593931v1).

Source code for the original is vendored at
[reference/widloski_foster_2025/](../../reference/widloski_foster_2025/); that
directory's README records provenance, licences, pinned commit, and the functions that
exist in neither published source. File references below of the form
`foster_lab_matlab/x.m` and `zenodo_15199609/x.m` point into it.

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
7. **Significance (separate validation, not part of the event definition).** Re-decode
   each event 100× with place-cell identities permuted; require p < 0.05 for both mean
   spread and mean jump size. (`zenodo_15199609/load_replayEvents_cellIDshuffle.m`)

### Notes on the original that bear on the port

- **Step 7 is not part of the main event definition.** `load_ind_replayEvents.m` selects
  on duration and dispersion alone. The shuffle lives in a separate script.
- **The spread statistic is a product, not a sum.** `compute_imageSpread.m` computes
  `sqrt( Σ |x−x̄|² |y−ȳ|² P / ΣP )` — a product of marginal deviations, in **bin** units,
  so in 2D it is area-like rather than a length, and the paper's "m < 10 cm" is a loose
  label. Mallory's code confirms this reading by rescaling the threshold as
  `maze_size² × 0.0012` with the comment `% equivalent to John's 10 cm`. **In 1D the
  function takes its other branch and reduces to `sqrt(Σ|x−x̄|²P)` — an ordinary SD.**
  Porting to a linearized track therefore makes the statistic a genuine length and the
  10 cm threshold literally correct. This is a simplification, not a deviation.
- **Two bugs in the published shuffle script** (`zenodo_15199609/load_replayEvents_cellIDshuffle.m`),
  neither reachable from the main event definition: line 137 passes the all-NaN p-value
  vector to `compute_sequenceJumps` instead of the posterior COM, so that script's
  jump-size p-values are meaningless; and lines 122–123 index `timeBins_sub{i}` by loop
  counter while writing to `ind_replay(i)`, re-decoding the wrong candidate. Recorded
  for completeness; we do not implement step 7.

## Decisions register

Decisions taken during design, with the reasoning, so the methods section can state them
without re-litigation.

| # | Decision | Reasoning |
|---|---|---|
| **D1** | Purpose is a methods comparison against our existing cohort, built to be reusable | Set by project need; determines that faithfulness matters only where it affects comparability |
| **D2** | Omit the cell-ID shuffle; define events by duration + dispersion only | Matches the original's *main* event definition (`load_ind_replayEvents.m`). Also unavoidable — clusterless data has no cell identities to permute. A hook is left for adding a null later |
| **D3** | Use our canonical whole-event immobility rule (< 4 cm/s), not theirs (per-bin < 5 cm/s) | Makes both cohorts draw from the same speed-eligible population, so a cohort difference is attributable to the detection algorithm rather than the eligibility rule |
| **D4** | Centre = Fréchet mean over track-graph distance; spread = sqrt of the same minimized value | A posterior mean is undefined on a branched graph. The Fréchet mean generalizes COM, yields centre and spread from one computation, and reduces exactly to COM/SD on locally straight track |
| **D5** | Rebuild the posterior from the saved encoding model rather than reusing our state-space posterior | Our posterior carries the state-space prior, whose temporal smoothing is precisely what the memoryless decoder avoids. Reusing it would make the "independent" detector a relabelling of ours |
| **D6** | The ported cohort gets its own key (`ms_event_id`); never joined on `event_id` | `event_id` hashes are cohort-dependent; a different cohort sharing the key would mis-join silently (see CLAUDE.md on atomic regeneration) |
| **D7** | `results.nc` fast path is opt-in and verified, never an automatic fallback | The `log_likelihood` variable only exists in decodes written after 2026-06-26 (`c8aee2c`); an automatic fallback would silently change which epochs are analyzed |

## Deviations register

Every departure from the original. **Effect** states what a reader should worry about,
not a reassurance.

### Forced by recording modality

| # | Original | Ours | Why | Effect |
|---|---|---|---|---|
| **V1** | Sorted units, per-cell rate maps | Clusterless KDE encoding model over waveform features × position | HPC in this project is decoded clusterless; there are no sorted units | The encoding model is not a set of place fields. Cell-level inclusion criteria (rate, spatial information, isolation, SNR) have no analogue and are not applied |
| **V2** | Cell-ID shuffle, p < 0.05 on spread and jump | Not implemented | No cell identities exist to permute (see D2) | The ported cohort has no significance filter. Since the original's main cohort has none either, this does not separate us from their headline numbers — but it does mean neither cohort is shuffle-controlled |
| **V3** | Tuning curves get `+1e-3` before decoding | Clusterless KDE with its own `EPS` clipping of the ground-process intensity | Different likelihood construction | Low-occupancy bins are regularized differently. Not tunable to match; the two decoders differ in their floor behaviour |
| **V4** | Rate maps fit on running periods (> 10 cm/s) | Encoding model fit on the decode's `is_training` mask (`training_type = "no_ripple"`) | Reuses the already-fitted model rather than refitting | **The training set is genuinely different**: non-ripple times include immobility, theirs excludes it. This changes the encoding model, not just the decoder, and is the deviation most likely to move results |

### Forced by track geometry

| # | Original | Ours | Why | Effect |
|---|---|---|---|---|
| **V5** | 2D open field, 90 × 90 cm | 1D linearized three-legged track graph, 9 arms | The task's geometry | Sequences are constrained to the graph; "direction" is along-arm rather than an angle |
| **V6** | Euclidean distance for jump and dispersion | Track-graph distance | On a linearized branched track, `|Δlinear_position|` is not the distance between two places — bins on different arms can be adjacent in linear coordinate | Using linear differences would under-count jumps at arm boundaries, producing spuriously smooth and spuriously long sequences. Track distance avoids this; it also makes our jumps non-comparable in absolute terms to an open-field jump |
| **V7** | Posterior COM | Fréchet mean over track distance (D4) | COM undefined on a graph | On straight segments these agree to within a bin; near junctions they can differ |
| **V8** | Unvisited arena bins excluded | Junction placeholder bins (`edge_id == -1`) excluded before normalization | These bins are linearization artifacts from `edge_spacing > 0`, not places | If not excluded, probability mass lands on non-places and drags the centre toward gaps. Same hazard `map_estimate_to_arm` guards against |
| **V9** | Spread is a 2D product moment (area-like, bin units) | Spread is a 1D SD (a length, cm) | `compute_imageSpread` branches on dimensionality; the 1D branch is an ordinary SD | Makes the 10 cm threshold literally meaningful. Our spread values are **not** on the same scale as theirs and must not be compared numerically to any 2D figure in the paper |

### Forced by our sampling grid

| # | Original | Ours | Why | Effect |
|---|---|---|---|---|
| **V10** | 5 ms step | 4 ms step (2 samples at 500 Hz) | 5 ms is 2.5 samples on our native grid | Slightly finer temporal resolution on sequence boundaries. Every bin-count threshold must be **derived from the stride**, never carried over as the original's literal constants |
| **V11** | 80 ms window | 80 ms window (40 samples) | Exact at 500 Hz | None |

### Deliberate, for comparability

| # | Original | Ours | Why | Effect |
|---|---|---|---|---|
| **V12** | Per-bin speed < 5 cm/s | Per-bin < 4 cm/s for candidacy, then our canonical whole-event rule | D3 | Slightly stricter than theirs. Shared with our cohort by construction |
| **V13** | Ripples: 100–220 Hz, z > 2 for ≥ 15 ms, tetrode-averaged, z-scored on stopping periods | Our existing Kay-detector SWR intervals and `classify_swr_overlap` | Both cohorts must be SWR-classified identically or the comparison is confounded | Our SWR definition, not theirs, governs both sides. Absolute SWR-overlap rates are therefore not comparable to the paper's; the *difference between detectors* is the meaningful quantity |

### Preserved exactly

Place bin size (2.0 cm = their `binSize`), decoding window (80 ms), uniform spatial
prior, memoryless decoding, spread threshold (10), jump threshold (20 cm), merge
thresholds (20 cm / 50 ms), minimum duration (100 ms), dispersion threshold (12 cm).

The threshold *values* are preserved, but note per V9 that the spread threshold's
*meaning* is not: ours gates a 1D SD in cm, theirs gates a 2D product moment in bin
units. The shared number 10 reflects their own 1D/2D branch, not a coincidence — but
our spread values are not numerically comparable to theirs.

### Discrepancies within the original, and how we resolved them

| Item | Paper | Zenodo scripts | Helper library | We use |
|---|---|---|---|---|
| Ripple band | 100–220 Hz | — | `[150,250]`, comment says "John used 120-170" | N/A (V13) |
| Ripple merge gap | merge if < 50 ms | `eventMergeThr = 0` (no merge) | — | N/A (V13) |
| Spike-density smoothing | 80 ms SD | 100 ms SD | — | N/A (bursts not ported) |
| Sequence / replay duration | 100 ms | 100 ms | 50 ms in the commented "John's criterion" block | 100 ms |

## Architecture

New subpackage `src/continuum_swr_replay/analysis/hippocampus/memoryless_sequence/`.
Named for what it does; the citation lives in docstrings, not the module path.

| module | responsibility | depends on |
|---|---|---|
| `config.py` | frozen dataclass of every threshold — the `Analysis_Information.mat` equivalent, one source of truth | — |
| `likelihood.py` | prior-excluded `(n_time, n_interior_bins)` log-likelihood on the native 2 ms grid | `detector.pkl` (primary), `results.nc` (opt-in) |
| `statistics.py` | pure functions: window aggregation, normalization, centre / spread / jump / dispersion | numpy, bin-distance matrix |
| `detect.py` | bin filter → sequence assembly → merge → criteria → event table | the above, plus existing repo helpers |

`likelihood.py` exposes one function with two implementations behind it, so the
`results.nc` fast path (D7) is a swap, not a fork in the caller.

## Data flow, per epoch

1. Load `encoding_model_[("", 0)]` from `{epoch}_detector.pkl`; compute prior-excluded
   log-likelihood once at 2 ms via `predict_clusterless_kde_log_likelihood(is_local=False)`.
2. Drop junction bins (`place_bin_centers_nodes_df_.edge_id == -1`) **before** normalizing (V8).
3. Sliding 80 ms windows as a cumulative sum over 40 rows, stride 2 rows; normalize
   across real bins under a uniform prior.
4. Per-window centre, spread, jump (D4, V6, V7).
5. Keep windows with spread < 10 cm, jump < 20 cm, per-bin speed < 4 cm/s.
6. Contiguous runs of kept windows → merge across gaps < 20 cm and < 50 ms → drop runs
   shorter than 100 ms.
7. Keep events with duration > 100 ms and dispersion > 12 cm.
8. Re-check whole-event immobility via the existing `event_speeds_are_immobile` /
   `whole_event_immobility_mask`.
9. Classify SWR overlap via the existing `classify_swr_overlap` / `event_strict_overlap_*`.

Steps 8–9 call our functions rather than reimplementing them, so both cohorts are
speed-gated and SWR-classified by identical code.

**Why the speed rule is two-stage.** A whole-event rule cannot gate bins before events
exist. Step 5 establishes candidacy; step 8 governs. Step 8 is not redundant: merging at
step 6 can span windows that step 5 rejected.

**The efficiency claim.** In `non_local_detector`, `mean_rate` is spike count divided by
*number of position samples*
([`weighted_mean_rate`](https://github.com/LorenFrankLab/non_local_detector/blob/main/src/non_local_detector/likelihoods/common.py)),
so rates are per-sample, not per-second, and the per-row term
`-1.0 * summed_ground_process_intensity`
([`clusterless_kde.py:460`](https://github.com/LorenFrankLab/non_local_detector/blob/main/src/non_local_detector/likelihoods/clusterless_kde.py#L460))
is exactly `−Δt·Λ(x)` with Δt one sample. The per-row log-likelihood is therefore
**exactly additive across rows**, and an 80 ms window is the sum of 40 rows. This is a
verified property of the library, but it is the load-bearing assumption of the whole
design and gets a dedicated test.

## Output schema

Table `memoryless_event_table`, one row per detected sequence.

- **Identity** — `animal`, `day`, `epoch`, `sequence_label`, `ms_event_id`
- **Timing (seconds)** — `event_start_s`, `event_end_s`, `event_duration_s`, `event_midpoint_s`
- **Detector statistics** — `n_windows`, `mean_posterior_spread_cm`, `max_jump_cm`,
  `mean_jump_cm`, `dispersion_cm`, `path_distance_cm` (summed track-graph distance
  along the centre trajectory, the analogue of the original's `compute_sequenceDistance`;
  reported, not used as a criterion)
- **Speed (cm/s)** — `max_speed`, `mean_speed`, `whole_event_speed_eligible`
- **SWR** — `is_swr`, `overlap_class`, `overlap_subclass`, `n_swrs_overlapping`,
  `matched_swr_event`, `matched_swr_start`, `matched_swr_end`

Timing column names mirror the canonical event table so comparison code is symmetric.
`ms_event_id` is deliberately a different key from `event_id` (D6).

## Comparison artifact

A script producing, per epoch: events from the two cohorts matched by strict
positive-duration temporal overlap (reusing `event_strict_overlap_keys`), bucketed
**ours-only / theirs-only / both**, each bucket crossed with `is_swr`. The headline
quantity is the non-SWR fraction within each detector's cohort and within the shared
subset.

## Testing

| test | what it protects |
|---|---|
| **Additivity** — cumsum-window log-likelihood equals a direct `predict_clusterless_kde_log_likelihood` call on the same 80 ms window, to floating-point tolerance, on simulated marks | The load-bearing assumption. If it fails, the design degrades to per-window recomputation and we must know immediately |
| Statistics against hand-constructed posteriors with closed-form answers | Centre / spread / jump / dispersion correctness |
| Fréchet mean equals ordinary COM on a straight segment | That D4 really does reduce to the original (V7) |
| Sequence assembly against the vendored MATLAB semantics, including its dropping of index-length-0 runs | Off-by-one and merge behaviour |
| Junction-bin test: mass on an `edge_id == -1` bin must not move the centre | V8 |
| Stride-derived thresholds: changing the stride must change bin-count thresholds | V10 — the constant that would silently become a 50 ms threshold |
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
4. **Additivity holds numerically at our block sizes** — proven from the source, tested
   explicitly rather than assumed.
