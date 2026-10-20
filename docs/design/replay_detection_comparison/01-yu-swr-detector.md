# SWR detection with a noise-percentile threshold (Yu et al. 2017)

*Design spec. Status (2026-10-03): revised after review. Package side implemented
(`ripple_detection` at commit 659cc91 and later: `Yu_ripple_detector`; the project pins
`68e1784`). The project-side inventory is computed directly from the package (§Producing
the inventory, direct path) by `scripts/detection/run_swr_detection.py`, which runs all four
SWR detectors on shared inputs; it is implemented and tested offline, not yet run on
recorded data. The Spyglass steps are deferred to the end.*
*Governing rule and cross-cutting decisions: [overview.md](overview.md).*

Inventory 3 of seven. Supplies a third SWR detection *algorithm* alongside
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
The paper includes only days with at least three tetrodes in or near the CA1 cell
layer. Record whether our selected recording sites satisfy that condition (see Producing
the inventory), rather than assuming any CA1-labeled group does.

The consensus statement on SWR detection (Liu et al. 2022, *Nat Commun* 13:6000, their
ref. 81) names this method as the data-driven alternative to arbitrary SD thresholds —
"the distribution of ripple frequency events exceeding 20 ms can be modeled against
background 'noise' and the termination of the SPW-R envelope occurs with return to the
session mean" — with the caveat that even this is "inadequate if quantitative SPW-R
counts are compared across sleep and wake because these states have different background
LFP power." Our comparison is within-epoch and within-state, so the caveat bears on
cross-epoch rate comparisons, not on the inventory itself.

**Why the noise model.** A z-score threshold is contaminated by the very events it means
to detect: ripples inflate the variance used to normalize, so the threshold moves with
ripple rate and amplitude, and its effective stringency differs between days and animals.
Mirroring the left flank estimates the noise scale from the part of the distribution
ripples cannot have contributed to. This is the method's whole point and is what makes it
a distinct algorithm rather than a parameter choice.

## The lab's own implementation

The published code trail was searched on 2026-09-21. The paper's data-availability
statement points to CRCNS dataset records rather than source code. A later Frank lab
implementation of the pipeline is public:

- `DFFunctions/AG_extractRipplesJY.m` in
  [bitbucket.org/franklab/filterframework_shared](https://bitbucket.org/franklab/filterframework_shared)
  (commit `f69ed5a`, `develop`; also in the `filterframework_shared_open` mirror and in
  [pinned FFPhy wrapper](https://github.com/droumis/FFPhy/blob/fce2048379d942c07f79d9192265ddf44f279d88/DFFunctions/AG_extractRipplesJY.m)). Added June 2019 by
  Demetris Roumis; header: "based on jai's version of kk_extractconsensus2". Not
  vendored: the repository states no license.

This is a later derivative, not an archived copy of the code used for the 2017 paper.
It supplies implementation evidence where the paper is underspecified. In this wrapper:

| step | as implemented in the wrapper |
|---|---|
| Envelope | per-tetrode amplitude envelope from input column 3; the upstream `DR_filtereeg2.m` stores `int16(abs(hilbert(filtfilt(kernel, 1, raw))))` there, without squaring |
| Band filter | `ripplefilter.mat` (2008-01-25, 1982 bytes, md5 `ed158b07…`), applied with `filtfilt`. **Byte-identical to the `ripplefilter.mat` shipped in `ripple_detection`** and used by `filter_ripple_band`, so that function gives the same FIR, not merely the same nominal 150–250 Hz. **The inventory does not use it:** R10 (run `056eee0dcb5a`) reads Spyglass's stored band (`LFPBandV1`, filter `ripple_150_250` at 1 kHz: pass 150–250 Hz, stop below 140 and above 260 Hz, 535 taps, linear phase with its delay removed), which Kay, Karlsson and Carey also read. Same nominal band and similar edges; the difference from `ripplefilter.mat` on Yu's events is not measured (SWR detector audit, 2026-10-19) |
| Artifact handling | samples with absolute envelope > 2000 are zeroed if their total duration is at most 5 s (calculated assuming 1.5 kHz). Above that, the code enters `keyboard`; if execution resumes, `continue` skips the current tetrode, not the entire epoch |
| Smoothing | 4 ms Gaussian **per tetrode** before aggregation: `gaussian(0.004·Fs, ceil(8·0.004·Fs))` is an area-normalized kernel with σ in samples and length 8σ (±4σ; N = 48 at 1.5 kHz, 32 at 1 kHz), centered at `mean(1:N)`, so an even N carries a half-sample offset, applied by `smoothvect` as a full convolution cropped from `round(N/2)`. The package's `gaussian_smooth` truncates at ±8σ, not ±4σ, and uses a symmetric kernel; measured on a positive envelope the difference is under 1e-4 relative, and the half-sample offset is not reproduced |
| Per-tetrode normalization | each tetrode's smoothed envelope **z-scored over the whole epoch** before the median |
| Consensus | `nanmedian` across tetrodes |
| Noise sample | consensus values during immobility (`kk_get2dstate`, velocity < 4, no buffer), pooled across all run epochs of the day |
| Baseline | mean of that immobility sample; `extractevents(trace, thresh, baseline, 0, mindur, 0)` ends events where the trace returns to it |
| Minimum duration | `mindur = round(min_suprathresh_duration * Fs)`; the MEX routine counts consecutive samples at or above threshold |
| Threshold | `jy_variableripthreshold_corecalculation({powervals}, ...)`, returning the threshold and a histogram pdf |
| Immobility rule | keep an event only if **both endpoints** fall in immobility |

The relevant supporting sources are
[`filtereeg2.m`](https://github.com/droumis/FFPhy/blob/fce2048379d942c07f79d9192265ddf44f279d88/DFFunctions/filtereeg2.m#L138-L159)
and [`extractevents.cpp`](https://github.com/droumis/FFPhy/blob/fce2048379d942c07f79d9192265ddf44f279d88/DFFunctions/extractevents.cpp#L113-L169).
Together they support unsquared amplitude and a sample-count duration rule for this
code lineage. Its debugger, trace-duplication workaround, and index-to-time conversion
are not a specification to copy; the retained timing rules are explicit below.

**Both original functions have since been recovered** (2026-09-21) from Anna Gillespie's
home directory on the lab server and are vendored verbatim at
[reference/yu_2017_franklab/](../../../reference/yu_2017_franklab/), whose README records
paths, modification dates (2017-09-04 and 2017-09-22), md5 sums, and license status:

- `jy_variableripthreshold_corecalculation.m` — Jai Yu's threshold estimator. It is the
  authoritative reference for Algorithm step 2 and is transcribed there.
- `kk_extractconsensus2.m` — the consensus-and-extraction driver, in Anna Gillespie's
  2017 edit, from which the 2019 wrapper above was consolidated. It reads its behavior
  flags (`useenvelope`, `normeachtet`, `adaptivefilt`) from `varargin` with **no defaults
  in the file**, so the flag values used for the paper are not recoverable from source.
  The 2019 wrapper's hard-coded choices — envelope, per-tetrode z-scoring, no adaptive
  filter — are the retained reading. The driver has **no caller anywhere** in the trees
  searched; the lab's production path is `Trodes_dayprocess.m` (line 316, Anna
  Gillespie's tree) calling `AG_extractRipplesJY` with `nstd = []` (data-driven
  threshold) and `ripmindur = 0.015`. That 15 ms minimum is the lab's later production
  setting and differs from the paper's 20 ms; the port retains the paper's value and
  records the discrepancy (below). Every per-animal driver in that tree sets
  `extractRipplesJY = 0`, so the JY path is present but switched off in the lab's
  later production runs; the code lineage is dormant, not maintained. The envelope reaching the wrapper is stored as
  `int16(abs(hilbert(...)))` by `DR_filtereeg2.m`, i.e. integer-quantized and unsquared;
  the quantization is absorbed by the per-tetrode z-scoring and is not reproduced. The driver also carries an optional slow-baseline
  rejection (`adaptivefilt`: a 0–0.5 Hz low-pass of the sub-2z trace; events whose start
  sits above that baseline's mean + 2 SD are dropped) that the paper does not describe
  and the wrapper does not use; it is **not** ported.

### Discrepancies between the paper and the lab code

| item | paper | lab code | we use |
|---|---|---|---|
| Minimum duration | 20 ms | `AG_extractRipplesJY` doc default and `Trodes_dayprocess.m` production call: 0.015 s | 20 ms — the published value; 15 ms is a later production setting, and Kay already runs at 15 ms |
| Noise estimate scope | per day | per day, pooled over the day's run epochs | per epoch (Y6) |
| Immobility rule | events at speed < 4 cm/s | both endpoints in immobility | endpoint rule for package parity, then whole-event (Y5) |
| Left-flank reflection | mirror about the mode | `abs(b) + 2m` | `2m − b`, warn if mode > 0 (Y3, V-Y2) |
| Flags (`useenvelope`, `normeachtet`, `adaptivefilt`) | not described | driver: caller-supplied, no defaults; wrapper: envelope, z-score, no adaptive filter | the wrapper's values, as a flag for `zscore_per_channel` (Y7) |

## Mapping onto the existing package

`Kay_ripple_detector` already runs: NaN-row drop → consensus trace → `normalize_signal`
→ `threshold_by_zscore` → `exclude_movement` → `exclude_close_events` → stats. Reuse its
normalization and post-detection utilities, with a separate Yu event-extraction helper:

- `normalize_signal` accepts a `normalization_mask`, which is exactly "normalize to
  immobility."
- `threshold_by_zscore` also extends qualifying threshold runs to mean crossings, but
  qualifies duration using `last_sample_time - first_sample_time`. At 1 kHz, its
  0.020 s setting requires 21 samples; the lab MEX rule accepts 20. The Yu helper uses
  integer run lengths and the boundary convention in Algorithm step 4. Do not adjust
  the duration by subtracting a floating-point sample interval or change Kay/Karlsson's
  shared helper to implement this convention.

Two parts are **not** shared and must be written:

- **The consensus trace.** Yu takes the *median across tetrodes* of individually
  smoothed Hilbert envelopes. `get_Kay_ripple_consensus_trace` sums squared envelopes
  across tetrodes, smooths the sum, and takes the square root. The two aggregations
  respond differently to activity confined to a few tetrodes (a median suppresses it; a
  root-sum-of-squares does not) and produce noise distributions of different shape —
  which is the very thing the threshold estimator measures. Reusing Kay's trace would
  import another method's internals, the hazard the governing rule names.
- **The threshold estimator**, the method's defining piece.

Plus a thin detector and private sample-count extraction helper that assemble those
pieces around the shared utilities. The retained speed cutoff is 4 cm/s and the
minimum above-threshold sample duration is 20 ms.

## Decisions register

| # | Decision | Reasoning |
|---|---|---|
| **Y1** | Implement in `ripple_detection` first (the lab's own package, local clone at `~/Documents/GitHub/ripple_detection`), pin the project to that commit, and produce the inventory through `RippleTimesV1` by registering the detector in Spyglass's algorithm dict from the project | Puts the third SWR algorithm beside the two it is compared with, under that package's tests and naming. Spyglass dispatches by name through a module-level dict, so no Spyglass change is needed to run it; a two-line Spyglass PR is optional cleanup that removes the registration line (see Producing the inventory) |
| **Y2** | Implement the threshold estimator as the original does: fixed histogram grid, mode from a smoothed histogram, mirrored **histogram**, threshold one bin past the CDF crossing (Algorithm step 2) | The original is now in hand (`reference/yu_2017_franklab/`). Under the governing rule its internals are Yu's, not ours to redesign; every earlier guess (sample mirroring, a bin rule, a single-bin mode) is retired |
| **Y3** | Reflect left bins as `2m − b`, and record that the original writes `|b| + 2m` | The two agree whenever every left-of-mode bin is ≤ 0, which holds for a z-scored right-skewed envelope; they diverge for a positive mode. Implementing the intended reflection and warning on `m > 0` keeps ours equal to the original where the original was run, without copying a latent defect, and makes any divergence on our data visible per epoch (V-Y2) |
| **Y4** | Reuse normalization and post-detection utilities; implement Yu's consensus, noise estimator, and a private sample-count event extractor | The package's existing duration test needs 21 samples for 20 ms at 1 kHz. Yu retains the lab routine's count rule, accepting 20; mean-crossing bounds use native timestamps as specified below |
| **Y5** | Apply the canonical whole-event immobility rule (`event_speeds_are_immobile`) to every event before export; the package's `exclude_movement` is not sufficient | `exclude_movement` tests speed only at the two endpoints, with `<=`, so it admits interior movement, interior NaN speed, and an endpoint at exactly 4 cm/s — all rejected by the canonical rule (overview X4). The lab wrapper's own rule is also endpoint-only, so this is a deliberate substrate departure from Yu, applied after detection as for the other inventories |
| **Y6** | Estimate the noise distribution and threshold **per epoch**, not per day | Matches how every other inventory is produced (one epoch, one run). The paper and the lab wrapper pool a day's run epochs; our epochs carry enough immobility to resolve the percentile on their own (Algorithm step 2), and a per-epoch estimate lets the threshold track within-day drift. Report the within-day spread of thresholds as a diagnostic |
| **Y7** | Smooth within valid blocks, then z-score each tetrode using all valid samples across the epoch before taking the median; expose the z-scoring as a flag | Follows the later lab wrapper's epoch-wide equalization. A data gap must not create a new normalization population. Median raw envelopes remain a deferred alternative |
| **Y8** | Apply available upstream artifact masks from the shared LFP substrate; add no fixed 2000-unit cutoff or automatic tetrode dropping | The wrapper's amplitude units and manual intervention do not define a transferable artifact policy. Record mask provenance or its absence, keep the selected montage fixed, and handle masked samples as unavailable data |

## Deviations register

### Forced

| # | Original | Ours | Why | Effect |
|---|---|---|---|---|
| **V-Y1** | Their recording geometry and tetrode count | Ours | Different dataset | The estimated threshold is per-session by construction, so this is what the method is designed to absorb. But the noise distribution's shape depends on electrode count and placement; thresholds are not comparable in absolute terms to the paper's |

### Method-internal (the source underspecifies; we choose and document)

| # | Underspecified | Our choice | Effect |
|---|---|---|---|
| **V-Y2** | Reflection of the left flank | `2m − b` for every left bin `b` (Y3) | The original's `abs(b) + 2m` equals this only for `b ≤ 0`. Inactive difference when the mode is ≤ 0; the implementation warns when it is not and records the mode per epoch |
| **V-Y3** | None: histogram grid, mode smoothing, and percentile rule are all fixed by the original | `-10:0.01:50` (left-closed bins, in median-of-z units), 11-point moving-average mode, `1 − 1e-4` CDF crossing plus one bin | Not a choice. Sensitivity of the threshold to grid width (0.005, 0.02) is a characterization for 04, not a design freedom |
| **V-Y4** | MATLAB `smooth(x, 11)` edge handling and `histc` binning | Centred 11-point mean with shrinking symmetric windows at the ends; left-closed bins with the final bin counting values exactly equal to 50 | Edge handling cannot affect an interior mode; documented so the transliteration is exact rather than approximate |

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
- **Ripples must not dominate the immobility variance.** The mirrored distribution
  reaches at most `2m − min(x)`, and the event rule extends events to the immobility
  *mean*, so the method is only defined when the left flank is wider than the
  mode-to-mean distance: `(m − min) > (mean − m)`. Large or frequent ripples inflate the
  immobility SD and drag the mean above the noise ceiling, and the ratio falls below 1.
  Measured on the package's simulated LFPs (2026-09-21; 20 s, four tetrodes, six planted
  ripples): pure noise gives a threshold of 2.9 SD with one false event; ripples up to
  ~30× the in-band background give ratios of 2–8, full recovery, and thresholds of
  0.2–1.1 SD; ~70× background gives a ratio of 0.96 and a threshold at the mean; ~135×
  (the simulator's default amplitude) gives 0.56 and no defined threshold. The original
  MATLAB runs regardless and produces ill-defined events; the port raises.
  `estimate_noise_threshold` reports the ratio as `flank_ratio` in its diagnostics; the
  per-epoch manifest must record it so an epoch in this regime is visible rather than
  silently absent from the inventory.

## Architecture

Three public additions to `ripple_detection`, plus a private event-extraction helper,
following its naming (`Kay_ripple_detector`,
`get_Kay_ripple_consensus_trace`) and its idioms (`ValueError` on bad input, pandas
`DataFrame` of event stats out).

| function | file | responsibility |
|---|---|---|
| `estimate_noise_threshold(values, percentile=99.99, histogram_edges=None, mode_smoothing_window=11)` | `core.py` | transliteration of `jy_variableripthreshold_corecalculation`: fixed-grid histogram → smoothed-histogram mode → mirrored histogram → threshold one bin past the CDF crossing. Pure, array-in/scalar-out, no LFP concepts. `histogram_edges=None` means the original grid `np.arange(-10, 50 + 0.01/2, 0.01)`; returns the threshold and, on request, the diagnostic (edges, counts, mode) |
| `get_Yu_ripple_consensus_trace(ripple_filtered_lfps, sampling_frequency, smoothing_sigma=0.004, zscore_per_channel=True, *, time=None)` | `detectors.py` | One call per epoch: envelope and smoothing within valid blocks, then pooled epoch-wide tetrode z-scores (Y7), then median. `(n_time, n_tetrodes)` in, `(n_time,)` out with unavailable rows retained as NaN. `time` identifies timestamp gaps; omission declares a regular sample grid |
| `Yu_ripple_detector(time, filtered_lfps, speed, sampling_frequency, *, speed_threshold=4.0, minimum_duration=0.020, percentile=99.99, smoothing_sigma=0.004, close_ripple_threshold=0.0, normalization_mask=None, zscore_per_channel=True, maximum_duration=None)` | `detectors.py` | Kay's keyword interface with `percentile` in place of `zscore_threshold` and an additional `zscore_per_channel` parameter, so Spyglass can pass `ripple_detection_params` straight through. The histogram grid and mode smoothing are method constants, not parameters. Pass the full epoch and `time` to the consensus helper; estimate the threshold on the raw consensus (Algorithm step 2); extract events separately within valid blocks |
| `_extract_Yu_ripple_events(trace, time, sampling_frequency, minimum_duration, threshold)` | `detectors.py` | Private helper for one contiguous block of normalized consensus: integer run-length qualification and mean-crossing extension under Algorithm step 4; leaves existing detectors' duration conventions intact |

The canonical whole-event immobility rule (Y5) is **not** part of the package function:
it is substrate applied to every inventory after detection and lives in
`continuum_swr_replay`. The package's newer `exclude_movement_by_majority` does not match
the canonical rule either and is not used.

Splitting them this way keeps the two novel pieces independently testable — the estimator
against closed-form distributions and the consensus against synthetic envelopes — inside
the package's own test suite, with no Spyglass fixtures required.

## Producing the inventory

**Decided 2026-10-02:** for the comparison the inventory is computed directly by calling
the package; Spyglass is not altered until the final product. The direct path and the
deferred Spyglass integration produce the same events from the same inputs and
parameters, and the Spyglass path is accepted only if it reproduces the direct inventory.

### Direct path (now)

`scripts/detection/run_swr_detection.py` produces this path for all four SWR detectors at
once (Kay, Karlsson, Yu and Zugaro) on shared inputs, as Stage 3 of the refresh design
([2026-10-03-full-pipeline-refresh-design.md](../2026-10-03-full-pipeline-refresh-design.md))
specifies. It is implemented and tested offline, not yet run on recorded data.

1. **Package.** Done: `Yu_ripple_detector` and its tests are on the package's `master`.
2. **Pin.** Done: `ripple_detection` is a direct git dependency of the project at commit
   `68e1784` (refresh design R2). The project used to receive it only transitively
   through Spyglass at 1.7.0, which lacks the detector, so the pin is a deliberate
   version bump. Spyglass imports only the Kay and Karlsson detectors and two `core`
   helpers from the package, so the existing Kay reads keep working.
3. **Read inputs.** `get_ripple_band_inputs` reads the CA1 ripple-band `LFPBandV1` frame
   and its electrodes through the read-only restrictions `get_ripple` uses, and asserts
   that the electrodes are the `RippleLFPSelection` CA1 electrodes (the check below).
   `get_epoch_position_info` gives the head speed and time span `load_data` uses. No
   Spyglass row is inserted or populated.
4. **Detect.** Call `Yu_ripple_detector` with `speed_threshold = 4.0`,
   `minimum_duration = 0.020`, `percentile = 99.99`, `smoothing_sigma = 0.004`,
   `zscore_per_channel = True`, and `Karlsson_ripple_detector` at its package
   defaults. Record in the epoch's manifest the package commit, the histogram grid,
   realized histogram edges, selected mode, threshold, tetrode normalization statistics,
   normalization population, and NumPy version. The runner records the package
   commit and the NumPy version in its manifest, and every field of the package's
   `noise_threshold_diagnostics` with the immobility sample's mean and SD, so they
   survive an epoch with no events. The scalars (the threshold, mode, flank ratio,
   out-of-grid fraction and sample counts, and the mean and SD) go in the manifest;
   every field, with the histogram arrays (edges, counts, smoothed and mirrored
   counts), goes in the epoch's `{epoch_id}_yu_diagnostics.json`. The per-tetrode
   z-score statistics are not recorded. Any later sensitivity variant gets its own
   named parameter set in the manifest.
5. **Gate and key.** Apply Y5 project-side, then key the inventory and observation
   availability as 03 requires. The runner writes one event parquet per epoch and
   detector, and the epoch's observed intervals; Y5 and the keying are left to the
   comparison's adapters, which apply the shared rule to every inventory (overview X4).

### Spyglass integration (final product, deferred)

6. **Register.** Spyglass's `RippleTimesV1.make` looks the algorithm up by name in the
   module-level dict `spyglass.ripple.v1.ripple.RIPPLE_DETECTION_ALGORITHMS`, which
   holds only Kay and Karlsson. Either upstream a two-line Spyglass PR (import plus dict
   entry) or register it in `data_loaders/lfp.py` before any populate:

   ```python
   from ripple_detection import Yu_ripple_detector
   from spyglass.ripple.v1 import ripple as sg_ripple

   sg_ripple.RIPPLE_DETECTION_ALGORITHMS.setdefault("Yu_ripple_detector", Yu_ripple_detector)
   ```

   Module-level registration means a fresh Spyglass install fails to re-populate until
   the same line runs; the upstream PR removes that cost.
7. **Parameter row.** Insert `ripple_param_name = "yu_2017"` with
   `ripple_detection_algorithm = "Yu_ripple_detector"`, `speed_name = "head_speed"`, and
   the step-4 parameters.
8. **Populate, read back and check.** Populate `RippleTimesV1` for the CA1 group with
   that parameter name; extend `get_ripple` to accept `ripple_detector_name = "Yu"`.
   Check that the populated events equal the direct inventory before switching any
   reader over.

One check when reading the inputs (step 3): assert that the `LFPBandV1` frame handed to the detector
carries exactly the `RippleLFPSelection` electrodes for the CA1 group, since the Kay path
in the loader has assumed so without asserting it.
Record electrode/tetrode identities, the number of distinct detecting tetrodes, and
whether anatomical metadata verifies at least three in or near the CA1 cell layer.
Use `true`, `false`, or `unknown` for `meets_published_tetrode_criterion`; a CA1 group
name alone is insufficient verification. Report any departure from that published
recording condition alongside the inventory, without silently changing the shared cohort.

Under Y8, propagate available upstream artifact masks into LFP availability before
processing and record their provenance. Without such a mask, record its absence and
retain finite samples; the initial detector adds no numerical amplitude clipping.
Any row invalid on a selected channel is unavailable for the consensus. Do not zero-fill
it or change the montage within the epoch. This differs from the wrapper's artifact
patch and `nanmedian` handling and is part of the shared-input adaptation.

## Algorithm

1. Call `get_Yu_ripple_consensus_trace` **once for the entire epoch**, passing `time`.
   Within each contiguous valid block, take each tetrode's Hilbert amplitude envelope
   of the 150–250 Hz band and smooth it with a 4 ms Gaussian. Retain the smoothed
   per-tetrode arrays on the original time axis. Then compute each tetrode's mean and
   sample standard deviation (`ddof=1`, matching the wrapper) across **all valid blocks
   of the epoch**, including movement. Apply those same statistics to every block
   before taking the median across tetrodes (Y7). With `zscore_per_channel=False`, take
   the median of the smoothed envelopes directly. Invalid rows stay NaN in both cases.
   **Missing LFP is handled by block, not by row-dropping.** The package's
   `_preprocess_detector_inputs` drops every row with a NaN channel and hands the
   compressed time axis downstream, so smoothing runs across the gap and
   `segment_boolean_series` measures duration across it as if it were contiguous: two
   short exceedances on either side of missing data can merge into one qualifying event.
   Split the epoch into maximal contiguous blocks for envelope calculation, smoothing,
   and event extraction (steps 4 and 5), never for separate normalization. Detect both
   non-finite rows and timestamp gaps, including disjoint intervals concatenated by
   Spyglass; retained adjacent rows need not be adjacent samples. The noise
   threshold (step 2) is estimated once per epoch from the immobility samples of all
   blocks. The same row-dropping applies to the existing Kay inventory as run through
   Spyglass; see the note under Unverified assumptions.
2. `estimate_noise_threshold` on the **raw** consensus values (median of per-tetrode
   z-scores, step 1) at all finite samples with finite speed < 4 cm/s, across all
   blocks. This is the original's `powervals` and its units; the grid below is only
   meaningful in those units, so the estimator must run **before** any further
   normalization. Transliterating `jy_variableripthreshold_corecalculation`:
   a. `edges = -10:0.01:50`; `counts = histc(values, edges)` — left-closed bins, the
      last bin counting values exactly equal to 50; values outside the grid are
      dropped, and their fraction is recorded (Failure behavior);
   b. `smoothed = smooth(counts, 11)` — centered 11-point moving average with shrinking
      symmetric windows at the ends; `j = first argmax(smoothed)`; `m = edges[j]`;
   c. mirrored histogram: the bins with `edges ≤ m` keep their **raw** counts; each bin
      with `edges < m` is reflected to `2m − edges` with its raw count (Y3, V-Y2); the
      union is sorted by position;
   d. `cdf = cumsum(mirrored counts) / total`; `k = first index with cdf ≥ 1 − 1e-4`;
      the threshold is the position of bin `k + 1`, i.e. one bin past the crossing;
   e. also record the immobility mean `baseline` and sample SD `stdev` of `values`, as
      the original does for `extractevents` and `maxthresh`.
   The original also computes and discards two alternatives (the last empty mirrored
   bin; a Gaussian fit at 2.5758 SD); neither is ported.
3. Normalize the consensus with `normalize_signal` once across the epoch, using the same
   immobility sample as step 2, so the immobility mean is zero and the SD is one for the
   package's boundary extension. Convert the threshold to those units,
   `threshold_z = (threshold − baseline) / stdev`, and use it as `zscore_threshold`.
   Thresholding the raw trace at `threshold` with a return to `baseline` and
   thresholding the normalized trace at `threshold_z` with a return to zero select
   identical samples, so the package's `threshold_by_zscore` is reused unchanged.
   Preserve the time axis and unavailable rows.
4. Run `_extract_Yu_ripple_events` within each valid block. Set
   `n_min = max(1, floor(minimum_duration * sampling_frequency + 0.5))`, matching the
   lab wrapper's positive-number rounding. Qualify a consecutive run at or above the
   threshold when its sample count is at least `n_min`: 20 samples at 1 kHz or 30 at
   1.5 kHz for the retained 20 ms setting. Extend to the containing run strictly above
   the immobility mean (zero after step 2), and emit that containing run only once if
   several qualifying exceedances share it. Use the first and last included samples'
   native timestamps as event bounds. A return to or below the mean ends a run.
   At an epoch or missing-data edge, truncate to the available block and flag the
   clipped boundary. Require a finite threshold strictly above the baseline; otherwise
   the threshold/mean-crossing rule is not well-defined and the run fails explicitly.
   Record the longest qualifying run's sample count and its duration
   `count / sampling_frequency` separately from `event_end_s - event_start_s`; do not
   add a sample width to overlap bounds or reproduce the MATLAB wrapper's
   one-based-index timestamp offset.
5. `exclude_movement` at 4 cm/s (package parity), then `exclude_close_events`, then
   event stats. Any duration-dependent Yu statistics, including `max_thresh`, use
   the same sample-count convention as selection; shared package stats must not
   silently reintroduce the first-to-last duration test.
6. Canonical whole-event immobility (Y5): drop any event with a non-finite speed sample
   or a sample at or above 4 cm/s anywhere inside it, via `event_speeds_are_immobile` on
   the speed trace over the event interval. Record how many events step 5 passed and
   step 6 removed.

Step 2's sample size matters and is worth stating: at our 1 kHz, an epoch with 20
minutes of immobility gives ~1.2 M samples, roughly half below the mode, so the mirrored
histogram holds ~1.2 M counts and the `1 − 1e-4` crossing sits ~120 counts from its top —
comfortably resolved. Epochs with little immobility will not resolve it, which is a failure condition
rather than a degraded estimate (see below).

## Testing

| test | what it protects |
|---|---|
| Threshold recovery: on a **symmetric** synthetic distribution with no signal component, `estimate_noise_threshold` returns the true analytic percentile to within one grid bin | The estimator itself, independent of LFP |
| Parity with the original: on a stored immobility sample, the Python estimator returns the same threshold as `jy_variableripthreshold_corecalculation` run in MATLAB (reference value stored with the fixture; the discarded Gaussian-fit branch is the only Curve-Fitting-Toolbox dependency and does not affect the returned threshold) | Y2 — the transliteration is exact, not approximate |
| Reflection: with every left bin ≤ 0, `2m − b` and the original's `abs(b) + 2m` give identical thresholds; with a positive mode they differ, and the implementation warns on `m > 0` and returns the correctly reflected threshold | Y3 / V-Y2 — ours equals the original where the original was run |
| Support bound: on a Rayleigh sample with no signal, the estimate lies below `2m − min(x)` and far below the true percentile; record the ratio | The inherited assumption. A characterization test that records the bias, not a pass/fail |
| Consensus trace: median across tetrodes of individually smoothed envelopes; a large ripple on one of N tetrodes moves the median far less than it moves the Kay root-sum-of-squares trace | Y4 — the aggregation is Yu's, not Kay's |
| Per-tetrode z-scoring: with one tetrode's envelope scaled by 10×, the flag on gives the same consensus as with unit scaling, the flag off does not | Y7 — the flag does what it claims |
| Blocks with different means/variances use one pooled epoch mean and sample SD per tetrode, matching a reference built from separately smoothed blocks; movement contributes to these statistics but not the subsequent immobility normalization | Gaps cannot silently change the normalization population |
| Whole-event immobility: an event with one interior sample ≥ 4 cm/s, or a NaN speed sample, or an endpoint at exactly 4 cm/s, passes `exclude_movement` and is removed by step 6 | Y5 — parity with the canonical rule the other inventories use |
| Contaminated left flank: record how below-mode contamination changes the mode and threshold | The noise-only-flank assumption; the direction is measured rather than prescribed |
| Grid sensitivity: threshold at grid widths 0.005, 0.01, 0.02 on synthetic noise, recorded not asserted; the smoothed-histogram mode differs from the raw argmax on a noisy histogram and matches it on a smooth one | V-Y3, V-Y4 — characterization of fixed choices |
| Synthetic LFP with planted ripples at known times and amplitudes: recovery and false-positive rate | End-to-end detection |
| Pure-noise LFP: characterize detections after median aggregation, smoothing, reflection, and the duration rule | The nominal percentile of a reflected distribution is not a guaranteed event false-positive rate |
| At 1 kHz, 19 consecutive qualifying samples fail and 20/21 pass; at 1.5 kHz, 29 fail and 30 pass; fractional-sample durations use the declared positive rounding | The sample-count convention, including the package/MEX difference |
| Qualifying exceedances sharing one above-mean run produce one event; equality to the mean ends the run; native sample timestamps define bounds and clipped block edges are flagged | Mean-crossing extension and explicit timestamp conventions |
| An upstream artifact mask removes coverage without zero-filling or changing the montage; its absence is recorded and adds no 2000-unit clipping | The declared project artifact policy, separate from the later wrapper |
| Degenerate inputs: all-NaN channels, no immobility samples, too few immobility samples to resolve the percentile, mode at the histogram edge, more than the stated fraction of values outside the grid | Failure behavior below |
| Grid semantics: default edges equal `-10:0.01:50`; a value of exactly 50 is counted and a value of 50.001 is dropped; tied smoothed maxima select the first; the threshold is one bin past the crossing | V-Y3, V-Y4 — `histc`, `smooth`, and `find(..., 1, 'first') + 1` transliterated |
| Spyglass pass-through: the parameter dict with `percentile` and `zscore_per_channel` yields the same frame as a direct call | The signature contract and configuration path |
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
- Positive mode: **warn** and proceed with the correct reflection `2m − b`. This is the
  one case where the original's `abs(b) + 2m` would have given a different (higher)
  threshold; the warning names the epoch and the mode, the manifest records both, and
  04 reports how often it occurred. Ours is then the method as intended, not what the
  original code would have produced on that epoch (V-Y2).
- Values outside the `[-10, 50]` grid: the original drops them silently. Record their
  fraction per epoch and raise above a stated ceiling (1e-3), since a heavy out-of-grid
  tail means the trace is not in the units the grid assumes.
- Non-finite or non-positive tetrode SD when z-scoring is enabled, or too few valid
  samples to compute it: raise rather than silently dropping the tetrode.
- A non-finite threshold or one at or below the immobility baseline: raise and record
  the diagnostic values; do not substitute a different threshold or baseline.
- No immobility at all in the epoch: return an empty event table with correct columns,
  matching the convention elsewhere in the pipeline.

## Unverified assumptions

1. **Which flags the 2017 analysis ran with.** The recovered driver takes `useenvelope`,
   `normeachtet`, and `adaptivefilt` from its caller with no defaults, so the paper's
   settings are not recoverable from source. The retained reading — unsquared envelope,
   per-tetrode z-scoring, no adaptive filter — is the 2019 wrapper's, supported by the
   preprocessing code (`filtereeg2.m` stores `abs(hilbert(...))`). The threshold
   estimator itself is no longer a reconstruction: it is transliterated from the
   recovered `jy_variableripthreshold_corecalculation.m` (Algorithm step 2).
2. **Whether the existing Kay inventory is affected by row-dropping.** `RippleTimesV1`
   calls `Kay_ripple_detector`, which drops NaN rows before smoothing and segmentation.
   Whether the ripple-band LFP handed to it contains interior NaN runs or timestamp
   gaps in our sessions is not established. If it does, some `SWR_event` intervals may
   bridge missing data and are flagged and excluded from comparison matches under 03's
   availability rule; if it does not, the issue is moot for inventory 1. Check both
   interior NaN runs and departures from the expected sample spacing per session.

The wrapper uses immobility-restricted consensus values for its noise sample and pools
them across run epochs of a day, consistent with the paper's per-day description.
Per-epoch estimation is our adaptation (Y6), not a reading of the paper.
