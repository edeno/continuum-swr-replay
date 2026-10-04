# Event differences, selection diagnostics, and threshold robustness

*Design spec. Status: agreed, not yet implemented. Default-setting diagnostics follow
[03](03-inventory-comparison.md). Threshold robustness is planned for a later stage and
does not block the initial report.*
*Governing rule and cross-cutting decisions: [overview.md](overview.md).*

The initial comparison asks which events each method finds, how the overlapping and
non-overlapping populations differ, and which selection rules account for disagreement.
It describes strengths and limitations for the scientific question being asked; it does
not rank detectors against an assumed truth set.

## Expectations and interpretation

Record these expectations before inspecting the comparison, but treat them as possible
tradeoffs rather than a checklist of predictions that must receive binary verdicts.

| method | expected selection properties | what remains an empirical question |
|---|---|---|
| SWR and MUA triggers | Select LFP or population-rate events; do not explicitly distinguish local, non-local, or absent spatial coding; thresholds exclude lower-amplitude activity | How much spatial evidence the admitted and excluded intervals carry, and how often their bounds differ from decoded representations |
| Widloski-style sequence detector | Accumulates evidence in overlapping 80 ms windows and selects extended, concentrated, continuous sequences | How much evidence the longer windows recover, which representations the criteria exclude, and how boundaries and segmentation differ |
| Our state-space detector | Compares Local, No-Spike, Continuous, and Fragmented explanations with temporal inference | Whether additional events have prior-excluded evidence; whether tracking uncertainty, sparse activity, priors, or probability thresholding explain apparent non-locality, boundary shifts, or fragmentation |

A duration or dispersion cutoff excludes events by definition. The useful measurement
is how many intervals it excludes and what those intervals contain. Exclusion alone
does not establish poor sensitivity to the method's intended target. Likewise, an SWR
with local coding is not an incorrectly detected SWR; it may be a poor proxy for
non-local representation.

Distinguish **temporal boundary differences** from **the proportion of events carrying
the content of interest**. Neither becomes boundary accuracy or detection precision
without a defensible reference. A continuous amplitude distribution motivates examining
threshold dependence; it does not establish that its lower tail contains replay.

Use the same primary diagnostics for every inventory. Our model's Local or No-Spike
probability is a model-dependent description, not an independent verdict on another
detector's event. Prior-excluded likelihood uses the same encoding and spikes that
contributed to selection, so it is also a diagnostic rather than independent validation.

Record the initial analysis configuration and report all primary contrasts, including
those unfavorable to our method. Additional analyses prompted by the results are allowed:
label them exploratory and record their motivation, configuration, and outcome. Keep the
original report and settings identifiable; do not silently replace unfavorable results.

## Initial report: default settings

### Event features

For shared events, each method's unmatched events, and sequences matched by neither SWR
nor MUA, determine membership using shared observation time as in 03. Report unavailable
comparisons separately. For these populations, report:

- **Timing:** interval span, native duration, and separation from nearby events.
- **Population engagement:** spike count, active tetrode count, and spike rate.
- **LFP and MUA:** ripple consensus amplitude and population rate, using a common trace
  and normalization for each feature across inventories. Identify the Kay trace when it
  supplies the common ripple feature; each SWR detector still uses its own detection trace.
- **Spatial representation:** extent, distance from the animal, posterior uncertainty,
  and continuous/fragmented or local/no-spike composition where defined.
- **Context:** animal position and tracking confidence where available; well, time in
  stop, reward, and patch context from existing tables.

Compute each feature with the same rule on every inventory's intervals. Namespace
posterior-derived features (`ss_*`, `ms_*`, and `cf_*` if ContFrag results are used),
and compute each available posterior's features for all inventories rather than choosing
a posterior according to which detector selected the event. Missing features remain
missing and their frequency is reported.

Show per-session values and animal-level consistency. Keep native populations primary,
with duration-stratified views so differences in opportunity to accumulate spikes or
overlap an event remain visible. Examples follow 03's reproducible sampling rule and
include supported comparator-only activity as well as questionable ours-only activity.

### Common spike-evidence diagnostic

Reuse the likelihood backend and spatial mapping in
`scripts/qa/analyze_event_spike_evidence.py`, applying a common diagnostic to every
inventory: mean native-bin log-likelihood difference between the best remote real bin
and the animal's position. Define remote bins using the same track-distance cutoff
(`REMOTE_CM`, currently 100 cm) for every interval and record it in the configuration.
Use identical timing, valid-bin, and spike-support conventions throughout. Report
spike counts and valid-bin coverage alongside the score; intervals with no valid remote
bin or missing position are explicitly unscorable.

If trajectory-specific likelihood is also reported, evaluate each named trajectory on
all intervals where it is defined. Do not compare a remote maximum for SWRs against a
trajectory-constrained score for our events as if they were the same statistic.

Score baseline windows with the same rule. Select them outside the union of the default
inventories within all seven detectors' shared observation exposure during eligible
immobility, with each full baseline window contained in that exposure. Match session,
duration, and animal location where feasible; report matching failures. Do not reuse the
QA script's low non-local probability condition as a model-neutral definition of
baseline. These windows are a reference population, not known negatives. Both the remote
maximum and event selection
can favor large scores, and silence can carry likelihood structure: a positive score or
an excess over baseline is not a calibrated declaration of replay.

### Boundaries and reasons for disagreement

From 03's matched pairs, report signed onset/offset differences, one-to-many matches,
and the fragment-merge recount. Ripple and MUA onsets are additional reference times,
not known true boundaries of a representation.

For ours-only activity, retain every comparator rejection reason: window gates, failure
to assemble a candidate, duration or dispersion, whole-event speed, or shuffle failure
or untestability. Unavailable comparator inputs or outputs are recorded separately as
unknown comparisons, not rejection reasons. Report the adapted memoryless cohort before
and after shuffle filtering as required by 02; its graph statistics and encoding-pair
null do not reproduce the published sorted-cell method exactly.

For comparator-only activity, examine our state probabilities, common spike evidence,
and any shorter constituent events. Distinguish boundary/segmentation differences from
sustained disagreement. Retain counterexamples to our method's expected advantages.

Gate diagnostics describe what happened under the default settings. Attribution of a
change in the cohort to one parameter requires a controlled rerun; such reruns belong
to the follow-up below and are not needed to release the descriptive report.

## Planned follow-up: threshold robustness

The consensus statement on SWR detection (Liu et al. 2022, *Nat Commun* 13:6000) asks
that published methods detail their detection criteria in full and "report how their
results manifest under a range of SWR detection parameters," and that feature
distributions, which are log-normal, be reported rather than means. The robustness plan
below and the per-cell distributions above are the second half of that; the detector
table ([detector_comparison.md](detector_comparison.md)) is the first.

Tingley & Peyrache's `RnR_methods`
(https://github.com/DavidTingley/RnR_methods, GPL-3, accompanying
[doi:10.1098/rstb.2019.0231](https://doi.org/10.1098/rstb.2019.0231)) is the nearest
precedent for the sweeps below. It ships the replay scores this project also uses,
Bayesian Radon and weighted correlation and rank-order correlation, together with
scripts that compare those scores against each other, across decoding bin sizes, under
added noise, and for rank-order false positives as a function of sequence length and
shuffle. Two things to take from it: bin size and shuffle choice move a score enough to
deserve their own arm of any sweep, and a false-positive rate is a property of the
score-and-shuffle pair rather than of the detector. It is GPL-3 MATLAB, so it is a
reference for what to vary, not code to port.

This remains part of the comparison, but is **deferred until after the default-setting
report**. Do not launch broad sweeps with the first run. Until it is performed, state
that threshold robustness has not been assessed; do not call the default results robust
to threshold choice.

Start with a small cost pilot on a declared session subset, using the same subset for
the methods being compared. Measure runtime, peak memory, and new shuffle work before
choosing the scope of a larger run. Specify the subset, parameter values, and compute
budget before examining the pilot's cohort differences. A limited pilot supports claims
only about those settings and sessions.

Begin with the retained setting and one modest relaxation and tightening of a selected
threshold, one parameter at a time. Take the candidate values from the literature
parameter survey in [detector_comparison.md](detector_comparison.md), so "modest" is
defined by the published range rather than by us. Prioritize our probability threshold, SWR/MUA
amplitude thresholds, and the memoryless detector's adapted spread cutoff; choose exact
values before that analysis. Additional duration, jump, dispersion, or other sweeps
follow only when needed to answer a stated question. There is no mandatory exhaustive
grid or factorial combination.

Reuse work only where the changed parameter leaves it valid:

| change | reusable work | work that must be repeated |
|---|---|---|
| Our probability or minimum-duration cutoff, with fitted model fixed | Saved state probabilities | Event extraction, whole-event speed qualification, features, and overlap accounting |
| SWR/MUA detection threshold, with preprocessing and normalization fixed | Consensus/per-channel traces or population-rate trace and normalization | Thresholding, event assembly, qualification, and downstream summaries |
| Memoryless spread/jump/duration/dispersion cutoff, with encoding, window, and stride fixed | Window likelihoods and statistics; valid fixed-bound shuffle results | Gate/assembly/selection stages affected by the change, qualification, and shuffles for new or changed candidate bounds |
| Encoding, temporal model, decoding window, or stride | Only artifacts whose inputs remain identical | Refit or re-decode and downstream calculations as required; these are separate controls, not cheap threshold changes |

Cache keys must include the relevant input/model hashes, timing support, configuration,
and shuffle seeds. Do not reuse a shuffle score for an event whose bounds changed.
Record all settings attempted and any computations not completed.

For each setting, compare with every other inventory at its retained setting: report
the fraction of its events captured and total time admitted, with the same spike-evidence
diagnostic where computed. Show native counts and segmentation effects. This describes
the tradeoff between coverage and evidence; another inventory is not a truth set and
these curves are not sensitivity/specificity curves.

## Other follow-ups: not initial deliverables

- **Recording robustness:** leave-one-tetrode-out re-detection/re-decoding if engagement
  diagnostics suggest that conclusions depend on a few tetrodes. Compare methods on the
  same declared session subset.
- **Alternative eligibility or encoding:** native speed gates and running-only encoding
  controls in 03, only when a specific disagreement warrants them.
- **Task-effect robustness:** refit rate/content models with alternative inventories if
  needed to assess dependence of a paper conclusion on the event definition.
- **Scoring and model controls:** fixed-candidate trajectory-score comparisons, window
  changes, or temporal-prior changes when needed to isolate an explanation.
- **Performance validation:** simulation or held-out evidence in [05](05-validation.md)
  if the intended claim extends to accuracy, sensitivity, precision, or calibration.

## Testing and deliverables

Initial implementation checks cover identical features and common scores for identical
intervals, posterior namespaces, missing data, baseline selection, overlap/segmentation
accounting, and agreement between rejection diagnostics and synthetic gate failures.
Method implementation tests in 01 and 02 still apply; deferring scientific sweeps does
not defer correctness checks.

The initial deliverables are the default-inventory feature and evidence summaries,
boundary/segmentation summaries, rejection diagnostics, and reproducibly selected
examples, with configuration and availability recorded. The later robustness run must
reproduce the primary inventory at its retained setting and preserve correct cache
invalidation. Its cost pilot, sweep results, and limitations are separate deliverables.
