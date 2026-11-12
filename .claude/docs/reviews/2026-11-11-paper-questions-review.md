# Review of the paper's questions, scope, and argument

Reviewed on 2026-11-11 at `a2e94d17b60b0ef9e942571ceae8c5cafa3363df`.
This records the scientific and editorial review requested by the project lead.
The accompanying [paper-development plan](../../../docs/design/2026-11-11-paper-development-plan.md)
turns these recommendations into a sequence of work with completion criteria.

**Terminology clarification, 2026-11-12:** the project lead specifies that replay
includes stationary representations as well as trajectories. The terminology and
comparator recommendations below incorporate that clarification. The subsequent
[question outline](../../../docs/design/2026-11-12-paper-question-outline.md)
gives the proposed main questions and subquestions.

The assessment uses the [question registry](../../../src/continuum_swr_replay/reported/questions.py),
[generated questions and values](../../../paper/reported/questions.md),
[manuscript](../../../paper/main.tex), [figure pipeline](../../../docs/figure-pipeline.md),
reported summaries, relevant design documents, and [publication follow-ups](../../../TODO.md).
It evaluates the argument and the coverage of its evidence. It does not rerun
analyses, validate the recordings, or certify that every upstream estimator is
correct. The [analysis-validity review](2026-11-11-analysis-validity-review.md)
and [reported-statistics choices](../../../docs/known_issues/reported_statistics_choices.md)
are separate inputs whose outstanding findings must be checked before affected
results are used.

Question identifiers and status descriptions below refer to this snapshot.
This is a review of the outline, not a second maintained question registry.

## Assessment

Most of the right questions are present. Their hierarchy is the main problem:
the list mixes scientific questions, supporting controls, individual measurements,
methodological framing, and abandoned interpretations. It is useful as an analysis
inventory but too detailed and uneven for a paper outline.

The strongest organizing question is:

> What non-local hippocampal representations occur outside detected SWRs, how
> confidently can we identify them, and how are they associated with behavior
> and cortical activity?

The paper's novelty needs to extend beyond the existence of replay outside SWRs,
which [Widloski and Foster demonstrated](https://www.nature.com/articles/s41467-025-65181-5).
The behavioral contribution also needs a clear relationship to
[Comrie et al.'s work on alternative representations in this task](https://www.sciencedirect.com/science/article/pii/S0896627326006744).
The promising contribution is the combination of explicit local/non-local
detection, independent validation, and characterization of the additional activity,
including its cortical associations. This is a proposed argument, not an assertion
that all its evidence is complete.

The most consequential unfinished work concerns validity and scope. Adding more
descriptive outcomes will not resolve those gaps.

## Proposed main questions

| Main question | Existing questions | Purpose |
|---|---|---|
| Can the model identify non-local representations with demonstrable accuracy and understood failure modes? | 1.1; 2a.1–2; 2a.4; 2c.2–7 | Establish what the detector measures and when to trust it. |
| What independently supported non-local activity falls outside detected SWRs? | 2a.5; 2b.1–5; 2c.1 | Establish the empirical phenomenon and its sensitivity to detection choices. |
| What content and dynamics characterize these events? | 2c.8; 2d.1–6 | Describe distance, content, duration, and dynamics without assuming distinct event types. |
| How are event occurrence and content associated with behavioral history? | 3.1–7 | Separate occurrence, content selection, and subsequent choice. |
| Is mPFC activity associated with these events beyond shared behavioral and hippocampal activity? | 4.1, divided into subquestions | Establish cortical association, then assess whether it depends on represented content. |

Use neutral section titles about validation, event occurrence, content, behavior,
and cortical associations. Titles such as "it works" and "the events are real"
assume the result of the checks they contain.

## Highest-priority gap: validation coverage

The validation population must match the population used for the headline claims.
The existing summaries make this concern concrete:

- The canonical detector calls most simulated place-less bursts. The source is
  [detection.json](../../../paper/reported/detection.json), key
  `controls_placeless_nonlocal_fraction`, attributed to `controls_values`.
- The held-out evidence reported separately by SWR status is specifically described
  for **Non-Local Continuous bouts**. The source is
  [events.json](../../../paper/reported/events.json), key
  `heldout_non_swr_per_spike`, attributed to `heldout_by_swr_values`.
- The canonical non-local event definition also includes Fragmented activity;
  the [simulation-control design](../../../docs/design/replay_detection_comparison/memoryless_simulation_controls.md)
  scores non-local representation and a continuous-trajectory target separately.
  That trajectory benchmark is narrower than the paper's replay scope, which also
  includes stationary representations.

The simulation result is not an estimate of real-data false discovery and does
not invalidate the observed events. It does mean that the non-local label alone
does not establish identifiable spatial content. Conversely, validation of
Continuous bouts does not automatically validate the full event inventory.

Promote the following question to the main validation argument:

> Which detected events contain independently supported spatial information,
> and does that evidence cover Continuous, Fragmented, close, and remote events?

Audit existing evidence before commissioning another analysis. Where coverage is
missing, either extend the appropriate independent test or explicitly restrict the
claim. A remote or Continuous subset can support a useful paper if it is named
consistently in prevalence, behavioral, and cortical results. Its results cannot
be used to validate an unrestricted population by implication.

## Reviewer questions and required responses

These are predictions about this paper's review, not statements that every issue
is currently an observed defect.

| Likely question | Required response | Current connection |
|---|---|---|
| Are you detecting spatial representation, broad uncertainty, or model mismatch? | Evaluate plausible local and nonspatial alternatives; state the tested event classes. Local-position decoding accuracy alone is insufficient. | 1.1; 2a.1–2; 2a.4; 2c.3–7 |
| Are events outside ripples, or outside this ripple detector? | Threshold sensitivity, electrode-level checks, alternative detection choices, and continuous ripple-power analyses. | 2a.5; 2b.2–5; SWR-source and tetrode-selection designs |
| Do priors, smoothing, or the Local-kernel width create apparent events? | Test sensitivity of the substantive conclusions, including the close/remote distinction. | 2a.2; 2a.5; 2c.6; decoder-transition-prior design |
| Is validation independent of event selection? | Explain which spikes determine selection, content estimates, and evaluation, including which stages exclude evaluation tetrodes. | 2c.3 and the held-out pipelines |
| Does the new method improve detection or detect a broader category by definition? | Compare shared targets and consider sensitivity together with false-positive behavior. Replay includes stationary representations and trajectories; a sequence detector may target only the latter. | 1.1; 2a.3; detector-comparison designs |
| Do effects replicate across animals and days? | Show animal-level effects and uncertainty, account for nesting, and examine sensitivity to individual animals. | Recording-day inference in TODO |
| Are behavioral associations explained by occupancy, visit history, stop duration, or task structure? | Use appropriate baselines, temporally valid covariates, and held-out evaluation for predictive claims. | 3.1–7; behavior and content-model designs |
| Does mPFC activity reflect representation or common reward, movement, MUA, and ripple-related activity? | Behavioral controls, continuous activity covariates, matched windows, and population inference. Test content dependence separately if claimed. | 4.1; marked-model designs |
| Are differences between SWR classes caused by differences in duration or decoding quality? | Show descriptive contrasts and appropriately adjusted comparisons with their different interpretations stated. | 2d.2–6; duration caveat in TODO |
| What is new relative to prior replay and task papers? | Identify the measurement or biological conclusion added by each main result and its limits. | Introduction and proposed main questions |

The [public peer reviews of Widloski–Foster](https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-025-65181-5/MediaObjects/41467_2025_65181_MOESM3_ESM.pdf)
specifically questioned sparse-spike detections, smoothing, ripple normalization
and electrode averaging, circular validation, and claims of distinct function.
These provide a targeted rehearsal; the complete table above also includes this
review's assessment of the present project's specific risks.

Real-data windows in which replay is considered unlikely are not known negative
ground truth. Shuffles test specified nulls; thinning tests robustness to reduced
information. Neither should automatically be described as a direct measurement
of real-data false-positive rate. Likewise, a high posterior state probability
does not by itself establish an empirical false-discovery guarantee. The relevance
of calibration and null choice is illustrated by
[Evaluating hippocampal replay without a ground truth](https://elifesciences.org/articles/85635).

## Consolidation and allocation of space

| Existing questions | Recommendation |
|---|---|
| 0.1 | Keep as introduction/methodological background and the literature survey. |
| 2a.3 | Treat the single-epoch comparator implementation diagnostic as supporting evidence; frame the main comparison around independently supported discoveries and shared targets. |
| 2b.6 | Treat as methodological framing, not a standalone empirical question. |
| 2c.2–5 and 2c.7 | Group under independent spike evidence; retain counts, likelihoods, matched windows, and remote subsets as complementary checks. |
| 2d.2–6 | Group under event dynamics and decoding uncertainty. Promote measurements that change the interpretation. |
| 3.1, 3.2, 3.6, 3.7 | Group under content selection while retaining reward history, value, and uncertainty as distinct explanatory variables. |
| 3.3 | Keep occurrence separate from content. |
| 3.4–5 | Group under incremental information about subsequent choice; retain informative negative results. |
| 3.8 | Use as supporting evidence on behavioral context and graded ripple coupling if it advances a main conclusion. |
| 4.1 | Separate event-associated responses, responses beyond shared covariates, and content dependence. |

Keep merged/superseded analyses distinct from unsupported interpretations. A
negative answer is still an answer and belongs under the surviving scientific
question. "Computed" is a bookkeeping status, not a guarantee that inference is
complete or that the evidence supports the question's full scope.

The plan should prioritize a small number of primary comparisons based on their
scientific role. It should not select them solely because their current estimates
are favorable. Report that these analyses were developed with knowledge of the
data; do not retrospectively describe them as preregistered.

## Interpretations that need narrower language

- **Replay and its dynamics:** replay includes stationary representations as well
  as trajectories. Require evidence for represented content; do not require that
  the representation move. Define non-locality and decoder states explicitly, and
  distinguish the paper's replay scope from narrower trajectory-only comparators.
- **Outside SWRs:** use "outside detected SWRs" unless stronger evidence supports
  an absence claim. Detection thresholds and recorded electrodes limit the scope.
- **Question 2b.7:** a ripple-power bump weakens a claim of absent ripple activity;
  it does not establish that the two populations have the same mechanism. The
  recorded reason for dropping this question is stronger than that evidence permits.
- **A continuum:** smooth distributions and threshold dependence alone do not
  establish a common biological mechanism. Graded empirical relationships can be
  described without settling mechanism.
- **Question 3.6:** use "length-adjusted" and "unadjusted associations" unless a
  causal interpretation of "direct" and "total effects" is explicitly justified.
- **mPFC:** event-associated activity alone does not establish importance,
  transmission, directionality, or function. The introduction's inference from a
  response to "demonstrating their importance" is too strong.
- **Similarity and null results:** failure to detect a difference is not equivalence;
  absence of detected incremental choice prediction does not establish absence of
  a behavioral role. Define equivalence bounds only if an equivalence claim is needed.

## Paper and figure scope

The questions fit one paper if the behavioral section remains focused. A complete
theory of value, uncertainty, reward memory, choice, and cortical communication
would substantially expand the project. Establishing additional measurable events
and their associations does not require explaining every possible function.

The current Figures 2 and 3 overlap in event examples and ripple-power description,
while the behavioral section lacks a dedicated main figure. A proposed allocation is:

| Figure | Main purpose |
|---|---|
| 1 | Model and decisive validation, including an understandable failure case. |
| 2 | Independently supported events outside detected SWRs and essential characterization. |
| 3 | A focused behavioral result with its relevant baseline or negative result. |
| 4 | mPFC associations and the controls needed to interpret them. |

Detailed parameter sweeps, additional dynamics measures, comparator diagnostics,
and alternative behavioral specifications can support these figures in the
supplement. Essential validity evidence must remain visible in the main argument.
If behavioral evidence cannot sustain a main result, reduce its narrative role
rather than filling a reserved figure with exploratory associations.

The first work should resolve validation coverage, SWR/decoder sensitivity, and
unfinished mPFC inference. Those determine which claims can be drafted. The
[development plan](../../../docs/design/2026-11-11-paper-development-plan.md)
specifies how to proceed without turning every possible reviewer question into
a mandatory new study.
