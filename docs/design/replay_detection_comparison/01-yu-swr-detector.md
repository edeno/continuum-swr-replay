# SWR detection with a noise-percentile threshold (Yu et al. 2017)

*Design spec. Status: revised after review; not yet implemented.*
*Governing rule and cross-cutting decisions: [overview.md](overview.md).*

Inventory 3 of six. Supplies a third SWR detection *algorithm* alongside
`Kay_ripple_detector` and `Karlsson_ripple_detector` — not a third threshold on one
algorithm.

The initial comparison uses the retained settings below. Scientific sensitivity checks
are follow-ups under [04](04-event-features.md), beginning with a cost pilot after the
default-setting report; they are not prerequisites for that report. Implementation
correctness and the synthetic checks below still precede use of this detector.

## The method as published

Yu JY, Kay K, Liu DF, Grossrubatscher I, Loback A, Sosa M, Chung JE, Karlsson MP,
Larkin MC, Frank LM (2017). *Distinct hippocampal-cortical memory representations for
experiences associated with movement versus immobility.* eLife 6:e27621.
doi:[10.7554/eLife.27621](https://doi.org/10.7554/eLife.27621) ·
[PMC5576488](https://pmc.ncbi.nlm.nih.gov/articles/PMC5576488/)

Verbatim from the Methods:

> We first normalized the consensus envelope power to times during immobility
> (speed < 4 cm/s). We assumed that the full distribution of the envelope power was the
> result of a lower power noise component and a higher power signal component that
> contributes to the long tail. We approximated the noise distribution by finding the
> mode for the histogrammed consensus envelope power and creating a symmetric
> distribution by mirroring the histogram below the peak around the mode. We then used a
> cutoff (99.99th percentile) in envelope power based on this empirically estimated noise
> distribution as the detection threshold for SWRs. This method allowed us to capture the
> distinct noise distributions for immobility periods for each day. SWRs were then
> detected when the power of the consensus envelope exceeded the detection threshold for
> a minimum of 20 ms. The starts and ends of the events were set to times when the
> envelope power returns to the mean. Only SWR events occurring at speed < 4 cm/s were
> included in our analyses.

Also from the Methods, on how the consensus is formed:

> The raw CA1 LFP was referenced to an electrode in corpus callosum and then filtered
> (150–250 Hz) to isolate the SWR band. The SWR envelope was then obtained using the
> Hilbert transform and then smoothed by convolving with a Gaussian kernel (σ = 4 ms). A
> consensus SWR envelope was calculated by taking the median of the envelopes across all
> available tetrodes.

In the paper the SWR band is sampled at 1.5 kHz. Our ripple-band LFP is at 1 kHz
(`data_loaders/lfp.py`, `filter_sampling_rate`); the band is the same 150–250 Hz.

**Why the noise model.** A z-score threshold is contaminated by the very events it means
to detect: ripples inflate the variance used to normalize, so the threshold moves with
ripple rate and amplitude, and its effective stringency differs between days and animals.
Mirroring the left flank estimates the noise scale from the part of the distribution
ripples cannot have contributed to. This is the method's whole point and is what makes it
a distinct algorithm rather than a parameter choice.

## The lab's own implementation

The published code trail was searched on 2026-09-21. The paper's data-availability
statement points only to CRCNS hc-13; no code link. One Frank lab implementation of the
pipeline is public:

- `DFFunctions/AG_extractRipplesJY.m` in
  [bitbucket.org/franklab/filterframework_shared](https://bitbucket.org/franklab/filterframework_shared)
  (commit `f69ed5a`, `develop`; also in the `filterframework_shared_open` mirror and in
  [github.com/droumis/FFPhy](https://github.com/droumis/FFPhy)). Added June 2019 by
  Demetris Roumis; header: "based on jai's version of kk_extractconsensus2". Not
  vendored: the repository states no licence.

What it settles, beyond the paper text:

| step | as implemented in the wrapper |
|---|---|
| Envelope | per-tetrode ripple envelope; samples above 2000 zeroed as "wild noise", epoch abandoned above 5 s of them |
| Smoothing | 4 ms Gaussian **per tetrode** before aggregation |
| Per-tetrode normalization | each tetrode's smoothed envelope **z-scored over the whole epoch** before the median |
| Consensus | `nanmedian` across tetrodes |
| Noise sample | consensus values during immobility (`kk_get2dstate`, velocity < 4, no buffer), pooled across all run epochs of the day |
| Baseline | mean of that immobility sample; `extractevents(trace, thresh, baseline, 0, mindur, 0)` ends events where the trace returns to it |
| Threshold | `jy_variableripthreshold_corecalculation({powervals}, ...)`, returning the threshold and a histogram pdf |
| Immobility rule | keep an event only if **both endpoints** fall in immobility |

What is still missing everywhere: `jy_variableripthreshold_corecalculation` (the
histogram → mode → mirror → percentile step) and `kk_extractconsensus2`. Neither is on
any branch or in the history of the two public Bitbucket repositories or the FFPhy
fork, GitHub code search finds only the call site, and no public Bitbucket workspace
carries them (a workspace `jyu` exists but is deactivated for inactivity). The MEX
source `extractevents.cpp` is in `droumis/FFPhy` and fixes the boundary semantics.
The threshold estimator is therefore reconstructed from the paper text (Algorithm
step 3); obtaining the original from the authors would let it be checked line by line.

## Mapping onto the existing package

`Kay_ripple_detector` already runs: NaN-row drop → consensus trace → `normalize_signal`
→ `threshold_by_zscore` → `exclude_movement` → `exclude_close_events` → stats. Two parts
of Yu's method are already implemented there:

- `normalize_signal` accepts a `normalization_mask`, which is exactly "normalize to
  immobility."
- `threshold_by_zscore` applies the minimum duration to the above-threshold run and then
  returns the whole containing above-mean segment — Yu's boundary rule and duration
  rule, already written.

Two parts are **not** shared and must be written:

- **The consensus trace.** Yu takes the *median across tetrodes* of individually
  smoothed Hilbert envelopes. `get_Kay_ripple_consensus_trace` sums squared envelopes
  across tetrodes, smooths the sum, and takes the square root. The two aggregations
  respond differently to activity confined to a few tetrodes (a median suppresses it; a
  root-sum-of-squares does not) and produce noise distributions of different shape —
  which is the very thing the threshold estimator measures. Reusing Kay's trace would
  import another method's internals, the hazard the governing rule names.
- **The threshold estimator**, the method's defining piece.

Plus a thin detector that assembles those two around the shared parts. Yu's 4 cm/s speed
threshold matches our canonical rule; `minimum_duration` changes from 15 ms to 20 ms.

## Decisions register

| # | Decision | Reasoning |
|---|---|---|
| **Y1** | Implement in `ripple_detection` first (the lab's own package, local clone at `~/Documents/GitHub/ripple_detection`), pin the project to that commit, and produce the inventory through `RippleTimesV1` by registering the detector in Spyglass's algorithm dict from the project | Puts the third SWR algorithm beside the two it is compared with, under that package's tests and naming. Spyglass dispatches by name through a module-level dict, so no Spyglass change is needed to run it; a two-line Spyglass PR is optional cleanup that removes the registration line (see Producing the inventory) |
| **Y2** | Estimate the percentile from **mirrored data**, not a mirrored histogram | See V-Y2 below; this is a numerical convenience, not a correction of the noise model |
| **Y3** | Use the mode as stated (histogram argmax). The robustified multi-bin mode is a sensitivity check, not the default | The multi-bin mode is another method's choice (it appears in the vendored Widloski code). Importing it would violate the governing rule |
| **Y4** | Reuse the package's normalization, thresholding, and close-event exclusion; implement Yu's median consensus trace and the threshold estimator locally | Normalization to a mask and mean-crossing extension are written identically in Yu's text and in the package, so sharing them introduces no difference to misattribute. The consensus aggregation is *not* shared (median vs root-sum-of-squares) and must be Yu's own |
| **Y5** | Apply the canonical whole-event immobility rule (`event_speeds_are_immobile`) to every event before export; the package's `exclude_movement` is not sufficient | `exclude_movement` tests speed only at the two endpoints, with `<=`, so it admits interior movement, interior NaN speed, and an endpoint at exactly 4 cm/s — all rejected by the canonical rule (overview X4). The lab wrapper's own rule is also endpoint-only, so this is a deliberate substrate departure from Yu, applied after detection as for the other inventories |
| **Y6** | Estimate the noise distribution and threshold **per epoch**, not per day | Matches how every other inventory is produced (one epoch, one run). The paper and the lab wrapper pool a day's run epochs; our epochs carry enough immobility to resolve the percentile on their own (Algorithm step 3), and a per-epoch estimate lets the threshold track within-day drift. Report the within-day spread of thresholds as a diagnostic |
| **Y7** | Z-score each tetrode's smoothed envelope over the epoch before taking the median, as the lab wrapper does; expose it as a flag so the text-only reading (median of raw envelopes) can be run | The paper text does not say whether tetrodes are equalized before the median; the only concrete implementation does, and it changes which tetrodes dominate the consensus. Following the implementation is the closer reading; the flag keeps the alternative one run away |

## Deviations register

### Forced

| # | Original | Ours | Why | Effect |
|---|---|---|---|---|
| **V-Y1** | Their recording geometry and tetrode count | Ours | Different dataset | The estimated threshold is per-session by construction, so this is what the method is designed to absorb. But the noise distribution's shape depends on electrode count and placement; thresholds are not comparable in absolute terms to the paper's |

### Method-internal (the source underspecifies; we choose and document)

| # | Underspecified | Our choice | Effect |
|---|---|---|---|
| **V-Y2** | How the percentile is taken from the mirrored distribution | Mirror the **samples**, not the histogram: estimate the mode from a histogram, then form the reflected sample `{x : x ≤ m} ∪ {2m − x : x ≤ m}` and take `np.percentile` on it | Removes bin-width dependence from the tail estimate; the mode is the only quantity that still depends on binning. This is a numerical convenience, **not** a correction of the method's support: the reflected distribution ends at `2m − min(x)` whichever way it is built (see the inherited assumptions), and a histogram mirrored at the same mode converges to the same percentile as bins are refined |
| **V-Y3** | Histogram bin rule for locating the mode | `bins="fd"` (Freedman–Diaconis), passed explicitly through the detector and Spyglass parameters. Use equal-width bins over the full finite immobility sample range; take the midpoint of the highest-count bin, breaking ties toward the lowest bin | This is our implementation choice, not a published default. Mode location still shifts the whole reflected distribution. Characterize bin sensitivity synthetically; a real-data control belongs to the follow-ups in 04 |
| **V-Y4** | Whether "the mode" is a single bin or a robust estimate | Single-bin argmax, per Y3 | Mode-estimation variability propagates to the threshold. A robustified estimate is a possible follow-up, not the retained method |

### Assumptions the method makes, which we inherit

- **The left flank is noise-only.** Ripples or artifacts below the mode can distort
  both the mode estimate and reflected distribution. The resulting threshold change
  depends on the contamination and must be measured rather than assumed conservative.
- **Noise is symmetric about its mode.** Mirroring yields a distribution whose support
  ends at `2m − min(x)`, whether built from samples or a histogram. On a single
  Rayleigh(σ = 1) distribution the true 99.99th percentile is 4.29 and the reflected
  estimate is about 2, illustrating possible underestimation of a skewed noise tail.
  That example does not establish the size or direction of error for the median of
  smoothed tetrode envelopes in these recordings. Characterize the actual consensus
  operation in synthetic checks and report its estimated thresholds. Do not assume that
  Yu is more permissive than Kay from the Rayleigh example or from thresholds expressed
  in different traces' z-score units.

## Architecture

Three additions to `ripple_detection`, following its naming (`Kay_ripple_detector`,
`get_Kay_ripple_consensus_trace`) and its idioms (`ValueError` on bad input, pandas
`DataFrame` of event stats out).

| function | file | responsibility |
|---|---|---|
| `estimate_noise_threshold(values, percentile=99.99, bins="fd")` | `core.py` | mode → reflected sample → percentile. Pure, array-in/scalar-out, no LFP concepts; `bins` accepts a positive integer or a NumPy equal-width bin-rule name |
| `get_Yu_ripple_consensus_trace(filtered_lfps, sampling_frequency, smoothing_sigma=0.004, zscore_per_tetrode=True)` | `detectors.py` | per-tetrode Hilbert envelope → per-tetrode Gaussian smoothing → per-tetrode z-score over the epoch (Y7, flag) → median across tetrodes. `(n_time, n_tetrodes)` in, `(n_time,)` out; sibling of the Kay trace |
| `Yu_ripple_detector(time, filtered_lfps, speed, sampling_frequency, speed_threshold=4.0, minimum_duration=0.020, percentile=99.99, smoothing_sigma=0.004, close_ripple_threshold=0.0, normalization_mask=None, normalization_time_range=None, bins="fd", zscore_per_tetrode=True)` | `detectors.py` | Kay's keyword interface with `percentile` in place of `zscore_threshold` and additional `bins` and `zscore_per_tetrode` parameters, so Spyglass can pass `ripple_detection_params` straight through. Forward `bins` unchanged to the estimator. Block-wise over contiguous valid LFP (Algorithm step 1); threshold estimated once per call |

The canonical whole-event immobility rule (Y5) is **not** part of the package function:
it is substrate applied to every inventory after detection and lives in
`continuum_swr_replay`. The package's newer `exclude_movement_by_majority` does not match
the canonical rule either and is not used.

Splitting them this way keeps the two novel pieces independently testable — the estimator
against closed-form distributions and the consensus against synthetic envelopes — inside
the package's own test suite, with no Spyglass fixtures required.

## Producing the inventory

1. **Package.** Branch the local clone from its current development head, add the three
   functions and their tests, and record the commit.
2. **Pin.** Add `ripple_detection` as a direct git dependency of the project at that
   commit. Today the project only receives it transitively through Spyglass at 1.7.0;
   the pin also picks up the commits the clone is ahead by, which is a deliberate
   version bump to note in the changelog, not a side effect.
3. **Register.** Spyglass's `RippleTimesV1.make` looks the algorithm up by name in the
   module-level dict `spyglass.ripple.v1.ripple.RIPPLE_DETECTION_ALGORITHMS`, which
   holds only Kay and Karlsson. A `RippleParameters` row can name `"Yu_ripple_detector"`
   (the field is a blob), but populate raises `KeyError` until the name is registered.
   The project registers it in `data_loaders/lfp.py`, next to where `ripple_detector_name`
   is validated, before any populate:

   ```python
   from ripple_detection import Yu_ripple_detector
   from spyglass.ripple.v1 import ripple as sg_ripple

   sg_ripple.RIPPLE_DETECTION_ALGORITHMS.setdefault("Yu_ripple_detector", Yu_ripple_detector)
   ```

   This is module state, not a private API, but a fresh Spyglass install re-populating
   the row fails until the same line runs. That is the reproducibility cost, and it is
   carried explicitly: the registration lives in the loader, and the run manifest records
   the `ripple_detection` commit that provides the function.
4. **Parameter row.** Insert `ripple_param_name = "yu_2017"` with
   `ripple_detection_algorithm = "Yu_ripple_detector"`, `speed_name = "head_speed"`, and
   `ripple_detection_params = {"speed_threshold": 4.0, "minimum_duration": 0.020,
   "percentile": 99.99, "smoothing_sigma": 0.004, "bins": "fd",
   "zscore_per_tetrode": True}`. Record the bin rule,
   realized histogram edges, selected mode, threshold, and NumPy version in the epoch's
   diagnostic artifact/manifest. Any sensitivity variant (V-Y3, the robustified mode,
   the squared-envelope check) is its own named row.
5. **Populate and read back.** Populate `RippleTimesV1` for the CA1 group with that
   parameter name; extend `get_ripple` to accept `ripple_detector_name = "Yu"` mapped to
   that row, alongside the existing Kay mapping. Then apply Y5 project-side and key the
   inventory and observation availability as 03 requires.
6. **Follow-up, optional.** A two-line Spyglass PR (import plus dict entry) makes the
   registration line unnecessary. It is cleanup, not a prerequisite.

One check when wiring step 5: assert that the `LFPBandV1` frame handed to the detector
carries exactly the `RippleLFPSelection` electrodes for the CA1 group, since the Kay path
in the loader has assumed so without asserting it.

## Algorithm

1. Consensus envelope via `get_Yu_ripple_consensus_trace`: per-tetrode Hilbert envelope
   of the 150–250 Hz band, each smoothed with a 4 ms Gaussian and z-scored over the
   epoch (Y7), then the median across tetrodes.
   **Missing LFP is handled by block, not by row-dropping.** The package's
   `_preprocess_detector_inputs` drops every row with a NaN channel and hands the
   compressed time axis downstream, so smoothing runs across the gap and
   `segment_boolean_series` measures duration across it as if it were contiguous: two
   short exceedances on either side of missing data can merge into one qualifying event.
   Instead, split the epoch into maximal contiguous blocks of valid samples, run steps
   1, 4 and 5 within each block, and never let an event span missing LFP. Detect both
   non-finite rows and timestamp gaps, including disjoint intervals concatenated by
   Spyglass; retained adjacent rows need not be adjacent samples. The noise
   threshold (step 3) is estimated once per epoch from the immobility samples of all
   blocks. The same row-dropping applies to the existing Kay inventory as run through
   Spyglass; see the note under Unverified assumptions.
2. Normalize with `normalize_signal(..., normalization_mask=speed < 4)` — z-scored to
   immobility.
3. `estimate_noise_threshold(..., percentile=percentile, bins=bins)` on the
   immobility-restricted values:
   a. `counts, edges = np.histogram(values, bins=bins)`, with no range clipping;
      `j = np.argmax(counts)` and `m = (edges[j] + edges[j + 1]) / 2` (V-Y3);
   b. reflected sample `{x : x ≤ m} ∪ {2m − x : x ≤ m}` (V-Y2);
   c. return `np.percentile(reflected, percentile, method="linear")`, with 99.99 retained.
4. `threshold_by_zscore(trace, time, minimum_duration=0.020, zscore_threshold=threshold)`
   — exceedance for ≥ 20 ms, boundaries extended to mean crossings.
5. `exclude_movement` at 4 cm/s (package parity), then `exclude_close_events`, then
   event stats.
6. Canonical whole-event immobility (Y5): drop any event with a non-finite speed sample
   or a sample at or above 4 cm/s anywhere inside it, via `event_speeds_are_immobile` on
   the speed trace over the event interval. Record how many events step 5 passed and
   step 6 removed.

Step 3's sample size matters and is worth stating: at our 1 kHz, an epoch with 20
minutes of immobility gives ~1.2 M samples, roughly half below the mode, so the reflected
sample has ~1.2 M values and the 99.99th percentile is its ~120th largest — comfortably
resolved. Epochs with little immobility will not resolve it, which is a failure condition
rather than a degraded estimate (see below).

## Testing

| test | what it protects |
|---|---|
| Threshold recovery: on a **symmetric** synthetic distribution with no signal component, `estimate_noise_threshold` returns the true analytic percentile | The estimator itself, independent of LFP |
| Mirroring a histogram vs mirroring samples converge to the same percentile as bins are refined, on both a symmetric and a skewed distribution | V-Y2 is a numerical choice, not a change of estimate |
| Support bound: on a Rayleigh sample with no signal, the estimate lies below `2m − min(x)` and far below the true percentile; record the ratio | The inherited assumption. A characterization test that records the bias, not a pass/fail |
| Consensus trace: median across tetrodes of individually smoothed envelopes; a large ripple on one of N tetrodes moves the median far less than it moves the Kay root-sum-of-squares trace | Y4 — the aggregation is Yu's, not Kay's |
| Per-tetrode z-scoring: with one tetrode's envelope scaled by 10×, the flag on gives the same consensus as with unit scaling, the flag off does not | Y7 — the flag does what it claims |
| Whole-event immobility: an event with one interior sample ≥ 4 cm/s, or a NaN speed sample, or an endpoint at exactly 4 cm/s, passes `exclude_movement` and is removed by step 6 | Y5 — parity with the canonical rule the other inventories use |
| Contaminated left flank: record how below-mode contamination changes the mode and threshold | The noise-only-flank assumption; the direction is measured rather than prescribed |
| Mode sensitivity: threshold as a function of bin rule, across plausible defaults | V-Y3, V-Y4 |
| Synthetic LFP with planted ripples at known times and amplitudes: recovery and false-positive rate | End-to-end detection |
| Pure-noise LFP: characterize detections after median aggregation, smoothing, reflection, and the duration rule | The nominal percentile of a reflected distribution is not a guaranteed event false-positive rate |
| Boundaries extend to mean crossings, not to threshold crossings; 20 ms minimum applies to the **exceedance**, not the extended event | Convention parity with the published rule |
| Degenerate inputs: all-NaN channels, no immobility samples, too few immobility samples to resolve the percentile, mode at the histogram edge | Failure behaviour below |
| Histogram configuration: default and explicit `bins="fd"` agree; fixed inputs reproduce edges, midpoint mode, and threshold, including tied maxima | V-Y3 is a reproducible choice |
| Spyglass pass-through: the parameter dict with `percentile` and `bins` yields the same frame as a direct call; a non-default bin count reaches the estimator unchanged | The signature contract and configuration path |
| Missing-LFP gap: two sub-minimum exceedances separated by a NaN gap or timestamp gap yield **no** event, and an exceedance that reaches a gap edge ends there | Block-wise processing; no event spans missing LFP |

Tests live in the package: `tests/test_core.py` for the estimator, `tests/test_detectors.py`
for the consensus trace and the detector, alongside the Kay and Karlsson cases.

A side-by-side run against Kay and Karlsson on the same session is a **sanity check, not
a test** — there is no ground truth, and disagreement is the object of study.

## Failure behavior

No silent fallbacks, and in particular no falling back to an SD threshold — that would
quietly convert the Yu inventory into a second Kay inventory and corrupt the comparison.
Package functions raise `ValueError`, as the existing detectors do; the project-side
loader raises if the `"Yu"` parameter row is absent rather than substituting Kay.

- Too few immobility samples to resolve the requested percentile: raise, naming the epoch
  and the sample count. The required count follows from the percentile and is checkable
  before estimating.
- Mode at the first or last histogram bin: raise. A mode at the edge means the reflected
  distribution is degenerate and the threshold is meaningless.
- No immobility at all in the epoch: return an empty event table with correct columns,
  matching the convention elsewhere in the pipeline.

## Unverified assumptions

1. **"Envelope power."** The consensus is a median of Hilbert *envelopes*; the threshold
   is then set on "consensus envelope power." Whether "power" means the envelope or its
   square is not stated. The design uses the envelope. Squaring is monotone but
   non-linear, so it moves the mode and the mirrored support bound; retain it as a
   possible later sensitivity check. (The 150–250 Hz band and the 4 ms per-tetrode
   smoothing are stated in the paper and are no longer assumptions.)
2. **Whether the existing Kay inventory is affected by row-dropping.** `RippleTimesV1`
   calls `Kay_ripple_detector`, which drops NaN rows before smoothing and segmentation.
   Whether the ripple-band LFP handed to it contains interior NaN runs or timestamp
   gaps in our sessions is not established. If it does, some `SWR_event` intervals may
   bridge missing data and are flagged and excluded from comparison matches under 03's
   availability rule; if it does not, the issue is moot for inventory 1. Check both
   interior NaN runs and departures from the expected sample spacing per session.
The lab wrapper resolves two earlier entries: the noise sample is taken from
immobility-restricted consensus values only, and the paper's per-day estimation is
what the lab code does; per-epoch is our decision (Y6), not a reading of the paper.
