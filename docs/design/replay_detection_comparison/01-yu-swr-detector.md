# SWR detection with a noise-percentile threshold (Yu et al. 2017)

*Design spec. Status: revised after review; not yet implemented.*
*Governing rule and cross-cutting decisions: [overview.md](overview.md).*

Inventory 3 of six. Supplies a third SWR detection *algorithm* alongside
`Kay_ripple_detector` and `Karlsson_ripple_detector` — not a third threshold on one
algorithm.

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
| **Y1** | Implement locally in `continuum_swr_replay` first; upstream to `ripple_detection` once behaviour is settled | Keeps the three SWR detectors comparable in one place while the implementation is in flux. Upstreaming later also requires a Spyglass change before it reaches us via `ripple_detector_name` |
| **Y2** | Estimate the percentile from **mirrored data**, not a mirrored histogram | See V-Y2 below; this is a numerical correction, not a change of method |
| **Y3** | Use the mode as stated (histogram argmax). The robustified multi-bin mode is a sensitivity check, not the default | The multi-bin mode is another method's choice (it appears in the vendored Widloski code). Importing it would violate the governing rule |
| **Y4** | Reuse the package's normalization, thresholding, and close-event exclusion; implement Yu's median consensus trace and the threshold estimator locally | Normalization to a mask and mean-crossing extension are written identically in Yu's text and in the package, so sharing them introduces no difference to misattribute. The consensus aggregation is *not* shared (median vs root-sum-of-squares) and must be Yu's own |
| **Y5** | Apply the canonical whole-event immobility rule (`event_speeds_are_immobile`) to every event before export; the package's `exclude_movement` is not sufficient | `exclude_movement` tests speed only at the two endpoints, with `<=`, so it admits interior movement, interior NaN speed, and an endpoint at exactly 4 cm/s — all rejected by the canonical rule (overview X4). Applied after detection, as for the other inventories |

## Deviations register

### Forced

| # | Original | Ours | Why | Effect |
|---|---|---|---|---|
| **V-Y1** | Their recording geometry and tetrode count | Ours | Different dataset | The estimated threshold is per-session by construction, so this is what the method is designed to absorb. But the noise distribution's shape depends on electrode count and placement; thresholds are not comparable in absolute terms to the paper's |

### Method-internal (the source underspecifies; we choose and document)

| # | Underspecified | Our choice | Effect |
|---|---|---|---|
| **V-Y2** | How the percentile is taken from the mirrored distribution | Mirror the **samples**, not the histogram: estimate the mode from a histogram, then form the reflected sample `{x : x ≤ m} ∪ {2m − x : x ≤ m}` and take `np.percentile` on it | Removes bin-width dependence from the tail estimate; the mode is the only quantity that still depends on binning. This is a numerical convenience, **not** a correction of the method's support: the reflected distribution ends at `2m − min(x)` whichever way it is built (see the inherited assumptions), and a histogram mirrored at the same mode converges to the same percentile as bins are refined |
| **V-Y3** | Histogram bin rule for locating the mode | Fixed default, recorded in config; mode location is the only thing it affects under V-Y2 | Mode location still shifts the whole reflected distribution, so bin sensitivity must be reported even though the tail no longer depends on binning |
| **V-Y4** | Whether "the mode" is a single bin or a robust estimate | Single-bin argmax, per Y3 | Noisier than a robustified mode, and the noise propagates to the threshold. Quantified by the sensitivity check rather than pre-empted |

### Assumptions the method makes, which we inherit

- **The left flank is noise-only.** If ripples or artifacts contribute below the mode,
  the noise distribution is overestimated and the threshold is conservative.
- **Noise is symmetric about its mode, with a left flank long enough to reach the
  requested percentile.** The consensus envelope is strictly positive, so its noise
  distribution is bounded below and right-skewed (Rayleigh-like) even without ripples.
  Mirroring the left flank yields a distribution whose support ends at `2m − min(x)` —
  about twice the mode in raw units — so the "99.99th percentile" cannot exceed that
  bound whatever the true tail does. On a Rayleigh(σ = 1) sample the true 99.99th
  percentile is 4.29 and the mirrored estimate is ≈ 1.97, for the histogram and the
  sample variant alike (checked at 2 M samples, 100–10 000 bins). The threshold is
  therefore biased **low** by a large factor, and in practice the method behaves like
  "twice the mode of the immobility envelope" rather than like a tail percentile. This
  is inherent to the published method, not to our implementation. It must be
  characterized (see testing) and stated, not silently corrected — correcting it would
  make the detector no longer Yu's. After z-scoring the bound is `2m − min(z)` with
  `m < 0`, so the threshold in z-units is small; report its value next to Kay's 2 SD so
  the difference in stringency is visible.

## Architecture

Module `src/continuum_swr_replay/analysis/hippocampus/swr_detectors.py`, written so the
threshold estimator can be lifted into `ripple_detection` unchanged when upstreamed (Y1).

| function | responsibility |
|---|---|
| `yu_consensus_trace(filtered_lfps, sampling_frequency, smoothing_sigma=0.004)` | per-tetrode Hilbert envelope → per-tetrode Gaussian smoothing → median across tetrodes. Pure, `(n_time, n_tetrodes)` in, `(n_time,)` out |
| `estimate_noise_threshold(values, percentile, bins)` | mode → reflected sample → percentile. Pure, array-in/scalar-out, no LFP concepts |
| `Yu_ripple_detector(time, filtered_lfps, speed, sampling_frequency, ...)` | assembles the two functions above with the package's normalization, thresholding, and exclusions, then applies the canonical whole-event immobility rule (Y5) |

Splitting them this way keeps the two novel pieces independently testable — the estimator
against closed-form distributions and the consensus against synthetic envelopes, with no
Spyglass fixtures required.

## Algorithm

1. Consensus envelope via `yu_consensus_trace`: per-tetrode Hilbert envelope of the
   150–250 Hz band, each smoothed with a 4 ms Gaussian, then the median across tetrodes.
   **Missing LFP is handled by block, not by row-dropping.** The package's
   `_preprocess_detector_inputs` drops every row with a NaN channel and hands the
   compressed time axis downstream, so smoothing runs across the gap and
   `segment_boolean_series` measures duration across it as if it were contiguous: two
   short exceedances on either side of missing data can merge into one qualifying event.
   Instead, split the epoch into maximal contiguous blocks of valid samples, run steps
   1, 4 and 5 within each block, and never let an event span missing LFP. The noise
   threshold (step 3) is estimated once per epoch from the immobility samples of all
   blocks. The same row-dropping applies to the existing Kay inventory as run through
   Spyglass; see the note under Unverified assumptions.
2. Normalize with `normalize_signal(..., normalization_mask=speed < 4)` — z-scored to
   immobility.
3. `estimate_noise_threshold` on the immobility-restricted values:
   a. histogram, take the mode `m`;
   b. reflected sample `{x : x ≤ m} ∪ {2m − x : x ≤ m}` (V-Y2);
   c. return its 99.99th percentile.
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
| Whole-event immobility: an event with one interior sample ≥ 4 cm/s, or a NaN speed sample, or an endpoint at exactly 4 cm/s, passes `exclude_movement` and is removed by step 6 | Y5 — parity with the canonical rule the other inventories use |
| Contaminated left flank: adding signal below the mode raises the threshold | The other inherited assumption |
| Mode sensitivity: threshold as a function of bin rule, across plausible defaults | V-Y3, V-Y4 |
| Synthetic LFP with planted ripples at known times and amplitudes: recovery and false-positive rate | End-to-end detection |
| Pure-noise LFP: false-positive rate consistent with the nominal percentile and the 20 ms duration requirement | That the threshold means what it claims |
| Boundaries extend to mean crossings, not to threshold crossings; 20 ms minimum applies to the **exceedance**, not the extended event | Convention parity with the published rule |
| Degenerate inputs: all-NaN channels, no immobility samples, too few immobility samples to resolve the percentile, mode at the histogram edge | Failure behaviour below |
| Missing-LFP gap: two sub-minimum exceedances separated by a NaN gap yield **no** event, and an exceedance that reaches a gap edge ends there | Block-wise processing; no event spans missing LFP |

A side-by-side run against Kay and Karlsson on the same session is a **sanity check, not
a test** — there is no ground truth, and disagreement is the object of study.

## Failure behavior

No silent fallbacks, and in particular no falling back to an SD threshold — that would
quietly convert the Yu inventory into a second Kay inventory and corrupt the comparison.

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
   non-linear, so it moves the mode and the mirrored support bound, and must be run as a
   sensitivity check. (The 150–250 Hz band and the 4 ms per-tetrode smoothing are stated
   in the paper and are no longer assumptions.)
2. **Whether the percentile is taken over immobility-restricted values only.** The
   Methods normalize to immobility and report per-day noise distributions, which implies
   it, but does not state it outright. The alternative — estimating over all time — would
   shift the mode. Both are cheap to run; the design assumes immobility-restricted and the
   comparison should report the other as a sensitivity check.
3. **Whether the existing Kay inventory is affected by row-dropping.** `RippleTimesV1`
   calls `Kay_ripple_detector`, which drops NaN rows before smoothing and segmentation.
   Whether the ripple-band LFP handed to it contains interior NaN runs in our sessions
   is not established. If it does, some `SWR_event` intervals may bridge missing data
   and should be flagged in 03; if it does not, the issue is moot for inventory 1.
   Check by counting interior NaN runs per session in the ripple-band LFP.
4. **How the paper handles multiple recording days per animal.** "Distinct noise
   distributions for immobility periods for each day" implies per-day estimation; our
   epochs are finer-grained than days. The design estimates per epoch, which is a
   plausible reading but not a verified one.
