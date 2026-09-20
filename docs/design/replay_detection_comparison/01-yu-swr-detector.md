# SWR detection with a noise-percentile threshold (Yu et al. 2017)

*Design spec. Status: approved, not yet implemented.*
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

The SWR band is sampled at 1.5 kHz; the band is 150–250 Hz.

**Why the noise model.** A z-score threshold is contaminated by the very events it means
to detect: ripples inflate the variance used to normalize, so the threshold moves with
ripple rate and amplitude, and its effective stringency differs between days and animals.
Mirroring the left flank estimates the noise scale from the part of the distribution
ripples cannot have contributed to. This is the method's whole point and is what makes it
a distinct algorithm rather than a parameter choice.

## Mapping onto the existing package

`Kay_ripple_detector` already runs: consensus trace → `normalize_signal` →
`threshold_by_zscore` → `exclude_movement` → `exclude_close_events` → stats. Three parts
of Yu's method are already implemented there:

- `normalize_signal` accepts a `normalization_mask`, which is exactly "normalize to
  immobility."
- `threshold_by_zscore` extends events to mean crossings after applying a minimum
  duration — Yu's boundary rule and duration rule, already written.
- `get_Kay_ripple_consensus_trace` produces the consensus envelope power.

So the only new component is **the threshold estimator**, plus a thin detector that
assembles the existing parts around it. Yu's 4 cm/s speed threshold already matches the
package default and our canonical rule; `minimum_duration` changes from 15 ms to 20 ms.

## Decisions register

| # | Decision | Reasoning |
|---|---|---|
| **Y1** | Implement locally in `continuum_swr_replay` first; upstream to `ripple_detection` once behaviour is settled | Keeps the three SWR detectors comparable in one place while the implementation is in flux. Upstreaming later also requires a Spyglass change before it reaches us via `ripple_detector_name` |
| **Y2** | Estimate the percentile from **mirrored data**, not a mirrored histogram | See V-Y2 below; this is a numerical correction, not a change of method |
| **Y3** | Use the mode as stated (histogram argmax). The robustified multi-bin mode is a sensitivity check, not the default | The multi-bin mode is another method's choice (it appears in the vendored Widloski code). Importing it would violate the governing rule |
| **Y4** | Reuse the package's consensus trace, normalization, thresholding, and movement exclusion rather than reimplementing | These are substrate shared with Kay by construction — Yu's method *is* the Kay pipeline with a different threshold rule. Reimplementing would introduce differences the comparison would then misattribute |

## Deviations register

### Forced

| # | Original | Ours | Why | Effect |
|---|---|---|---|---|
| **V-Y1** | Their recording geometry and tetrode count | Ours | Different dataset | The estimated threshold is per-session by construction, so this is what the method is designed to absorb. But the noise distribution's shape depends on electrode count and placement; thresholds are not comparable in absolute terms to the paper's |

### Method-internal (the source underspecifies; we choose and document)

| # | Underspecified | Our choice | Effect |
|---|---|---|---|
| **V-Y2** | How the percentile is taken from the mirrored distribution | Mirror the **samples**, not the histogram: estimate the mode from a histogram, then form the reflected sample `{x : x ≤ m} ∪ {2m − x : x ≤ m}` and take `np.percentile` on it | A mirrored *histogram* has finite support ending at `2 × mode`, so a 99.99th percentile read from it lands at or past the final bin and the threshold silently degenerates to ≈ `2 × mode`, set by bin width rather than by the data. Mirroring the samples is equivalent in the limit and exact in the tail. Reported as a deliberate departure from the literal text |
| **V-Y3** | Histogram bin rule for locating the mode | Fixed default, recorded in config; mode location is the only thing it affects under V-Y2 | Mode location still shifts the whole reflected distribution, so bin sensitivity must be reported even though the tail no longer depends on binning |
| **V-Y4** | Whether "the mode" is a single bin or a robust estimate | Single-bin argmax, per Y3 | Noisier than a robustified mode, and the noise propagates to the threshold. Quantified by the sensitivity check rather than pre-empted |

### Assumptions the method makes, which we inherit

- **The left flank is noise-only.** If ripples or artifacts contribute below the mode,
  the noise distribution is overestimated and the threshold is conservative.
- **Noise is symmetric about its mode.** The consensus trace is `sqrt(sum(envelope²))`,
  strictly positive and right-skewed even without ripples. Mirroring the left flank
  therefore builds a distribution with a **shorter right tail than the true noise**, so
  the threshold is biased **low** and detections are biased high. This is inherent to the
  published method, not to our implementation. It must be characterized (see testing) and
  stated, not silently corrected — correcting it would make the detector no longer Yu's.

## Architecture

Module `src/continuum_swr_replay/analysis/hippocampus/swr_detectors.py`, written so the
threshold estimator can be lifted into `ripple_detection` unchanged when upstreamed (Y1).

| function | responsibility |
|---|---|
| `estimate_noise_threshold(values, percentile, bins)` | mode → reflected sample → percentile. Pure, array-in/scalar-out, no LFP concepts |
| `Yu_ripple_detector(time, filtered_lfps, speed, sampling_frequency, ...)` | assembles the package's consensus trace, normalization, thresholding, and exclusions around that threshold |

Splitting them this way keeps the one genuinely novel piece independently testable
against closed-form distributions, with no LFP fixtures required.

## Algorithm

1. Consensus envelope power via `get_Kay_ripple_consensus_trace` (150–250 Hz, 4 ms
   Gaussian smoothing).
2. Normalize with `normalize_signal(..., normalization_mask=speed < 4)` — z-scored to
   immobility.
3. `estimate_noise_threshold` on the immobility-restricted values:
   a. histogram, take the mode `m`;
   b. reflected sample `{x : x ≤ m} ∪ {2m − x : x ≤ m}` (V-Y2);
   c. return its 99.99th percentile.
4. `threshold_by_zscore(trace, time, minimum_duration=0.020, zscore_threshold=threshold)`
   — exceedance for ≥ 20 ms, boundaries extended to mean crossings.
5. `exclude_movement` at 4 cm/s, then `exclude_close_events`, then event stats.

Step 3's sample size matters and is worth stating: at 1.5 kHz, an epoch with 20 minutes
of immobility gives ~1.8 M samples, roughly half below the mode, so the reflected sample
has ~1.8 M values and the 99.99th percentile is its ~180th largest — comfortably
resolved. Epochs with little immobility will not resolve it, which is a failure condition
rather than a degraded estimate (see below).

## Testing

| test | what it protects |
|---|---|
| Threshold recovery: on a **symmetric** synthetic distribution with no signal component, `estimate_noise_threshold` returns the true analytic percentile | The estimator itself, independent of LFP |
| Mirroring a histogram vs mirroring samples: the histogram variant saturates near `2 × mode` while the sample variant tracks the analytic percentile as bin count varies | V-Y2 — the trap that motivated the departure |
| Skew characterization: on a right-skewed noise distribution with no signal, record how far below the true percentile the estimate falls | The inherited assumption. A characterization test that records the bias, not a pass/fail |
| Contaminated left flank: adding signal below the mode raises the threshold | The other inherited assumption |
| Mode sensitivity: threshold as a function of bin rule, across plausible defaults | V-Y3, V-Y4 |
| Synthetic LFP with planted ripples at known times and amplitudes: recovery and false-positive rate | End-to-end detection |
| Pure-noise LFP: false-positive rate consistent with the nominal percentile and the 20 ms duration requirement | That the threshold means what it claims |
| Boundaries extend to mean crossings, not to threshold crossings; 20 ms minimum applies to the **exceedance**, not the extended event | Convention parity with the published rule |
| Degenerate inputs: all-NaN channels, no immobility samples, too few immobility samples to resolve the percentile, mode at the histogram edge | Failure behaviour below |

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

1. **Band and smoothing.** 150–250 Hz is stated in the paper; the 4 ms consensus
   smoothing is the package default rather than a published Yu parameter. Whether Yu used
   the same smoothing is not established from the available text.
2. **Whether the percentile is taken over immobility-restricted values only.** The
   Methods normalize to immobility and report per-day noise distributions, which implies
   it, but does not state it outright. The alternative — estimating over all time — would
   shift the mode. Both are cheap to run; the design assumes immobility-restricted and the
   comparison should report the other as a sensitivity check.
3. **How the paper handles multiple recording days per animal.** "Distinct noise
   distributions for immobility periods for each day" implies per-day estimation; our
   epochs are finer-grained than days. The design estimates per epoch, which is a
   plausible reading but not a verified one.
