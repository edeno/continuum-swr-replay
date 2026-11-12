# Plan for developing the paper's questions, evidence, and figures

**Date:** 2026-11-11. **Status:** proposed scientific work plan, recorded at the
project lead's request after the
[paper-questions review](../../.claude/docs/reviews/2026-11-11-paper-questions-review.md).
The review records the assessment and its sources; this plan specifies the work,
dependencies, and completion criteria. Recording the plan does not mark any
analysis complete or change the canonical event definition.

The [question outline](2026-11-12-paper-question-outline.md) gives the proposed
main questions and subquestions. **Terminology clarified by the project lead on
2026-11-12: replay includes stationary representations as well as trajectories.**
Validation and comparison targets in this plan follow that scope; a trajectory
requirement applies only to explicitly narrower analyses or comparators.

The objective is a coherent, defensible paper on independently supported non-local
hippocampal representations outside detected SWRs, their content and behavioral
associations, and their relationship to mPFC activity. The eventual scope must
follow the evidence, including negative and inconclusive results.

The [question registry](../../src/continuum_swr_replay/reported/questions.py)
remains the maintained source of questions. Identifiers below refer to the review
snapshot, before any reorganization. [TODO](../../TODO.md) remains the record of
existing publication follow-ups; this plan orders the scientific work rather than
replacing all of those tasks.

## Priorities and dependencies

| Priority | Work | What depends on it |
|---|---|---|
| Essential | Define the target phenomenon, claim population, and inference level. | Every headline and validation comparison. |
| Essential | Match independent validation to the claimed event population. | Prevalence, content, behavior, and cortical interpretation. |
| Essential | Establish sensitivity to SWR detection and consequential decoder choices. | Claims about activity outside detected SWRs. |
| Essential for a main behavioral result | Resolve affected estimator/inference issues and establish an appropriate behavioral baseline. | Claims about occurrence, content selection, or prediction. |
| Essential for a main cortical result | Complete population inference and controls for shared activity. | The mPFC section and its interpretation. |
| Conditional | Wider Local-kernel re-decode; extended Fragmented validation; formal model comparison; predictive or equivalence analyses. | Required only for the claims whose scope demands them. |
| Optional | A full joint marked point-process model, an extensive value/uncertainty theory, or a mechanistic continuum account. | Extensions beyond the proposed core paper. |

Reuse existing controls and sensitivity runs where they answer the required
question. Audit their population, provenance, and relevance before repeating them.
Do a representative pilot before a costly re-decode or refit. Narrow a claim when
the evidence supports a smaller population; do not silently generalize the result.

Stages 1–3 establish scope and validity. Behavioral and cortical work can then
proceed independently on the agreed population. Figure layout and methods drafting
can begin earlier, with unresolved results labeled explicitly.

## Stage 1: establish the argument and the claim populations

**Work**

- Organize the registry around the review's main questions: detection validity;
  activity outside detected SWRs; content and dynamics; behavioral associations;
  cortical associations. Retain detailed measurements as subordinate questions.
- Define non-local representation, replay (including stationary representations
  and trajectories), Continuous and Fragmented activity, close and remote events,
  and SWR overlap. Distinguish a decoder label from independently established
  spatial information, and trajectory-only analyses from the broader replay scope.
- For each intended main result, specify the population, primary outcome,
  principal comparison, replication level, essential control, intended figure,
  and evidence limitation. These can initially be notes in the existing registry;
  do not build a separate reporting system just to support this plan.
- Select primary comparisons for scientific relevance and record the exploratory
  history. Keep negative answers visible; distinguish merged questions from
  abandoned interpretations.
- Check what this paper adds beyond the related replay-detection and spatial-bandit
  papers. Describe a concrete additional measurement or finding for each result.
- Resolve the [reported-statistics choices](../known_issues/reported_statistics_choices.md)
  that affect selected outcomes. Check the current disposition of findings in the
  [analysis-validity review](../../.claude/docs/reviews/2026-11-11-analysis-validity-review.md)
  before using affected fits; an older finding is not automatically an outstanding
  defect after subsequent code changes.

**Deliverable:** a concise Results outline in the maintained question structure,
with one primary scientific comparison per intended main result and explicit
population/inference notes.

**Completion criterion:** each proposed claim has a defined estimand, population,
supporting evidence, and identified unresolved dependency. A result may be negative
or inconclusive; it must not be called ready solely because values exist.

## Stage 2: establish what the detector's discoveries mean

**Work**

- Map the evidence already available for Continuous, Fragmented, close, and remote
  events, separately with and without detected SWRs. Distinguish state bouts from
  canonical events and document how the two populations correspond.
- Audit the independence of event selection, content estimation, and evaluation.
  State which tetrodes or spikes are withheld at each step and which encoding fits
  use held-out time. Where a test reuses selection data, report that limitation or
  revise the test before making an independent-validation claim.
- Use existing simulations to assess known non-local content, local coding, and
  place-less activity. Check coverage of stationary representations as well as
  trajectories, extending validation where needed. Both are positive replay
  targets when spatial content is supported. Examine failure modes as well as recovery.
- Establish the independent spatial evidence for every population supporting a
  main claim. Extend tests where necessary, or restrict the validated population.
  A favorable pooled result must not conceal an unsupported subclass.
- For real-data nulls, specify what structure each null preserves and destroys.
  Report null-conditioned detection rates and their uncertainty. Treat thinning as
  an information/robustness test and quiet windows as comparison windows, without
  treating either as known negative ground truth.
- Keep held-out local decoding accuracy distinct from validation of non-local state
  labels. If the paper claims that the four-state architecture improves on reduced
  models, assess that comparison with an appropriate held-out criterion.
- Compare detectors on shared targets and common eligible time. Respect each
  method's published definition; separate target-definition differences from
  performance differences, including a sequence detector's exclusion of stationary
  representations. The single-epoch implementation diagnostic is supporting evidence,
  not a general superiority result.

**Deliverable:** a validation figure and methods account that show independent
support, important failure modes, and exactly which populations are validated.
Any new reported quantities enter the existing analysis → summary → figure/text
pipeline.

**Completion criterion:** prevalence and downstream claims refer to the population
that the validation actually supports. A posterior threshold is not presented as
a measured false-discovery guarantee. If validation supports only a subset, that
restriction is propagated consistently through the argument and affected analyses.

## Stage 3: establish robustness of activity outside detected SWRs

**Work**

- Complete the consequential threshold sensitivity work of 2a.5. Examine the
  stability of the main conclusion and independent support, not only the number
  of events or their overlap classes.
- Review existing SWR-source, normalization, artifact, reference, and tetrode-selection
  checks. Add targeted electrode-level or alternative-detector checks where needed
  to assess missed local ripples. State the limits of the sampled electrodes.
- Examine ripple power continuously alongside the binary overlap definition.
  Keep empirical graded relationships separate from an assertion of a shared
  mechanism or a new biological event type.
- Use existing transition-prior sensitivity results where applicable. Identify
  which remaining choices could alter the claimed event population, spatial
  distances, or substantive conclusion.
- If close events or own-well enrichment support a headline, complete the wider
  Local-kernel analysis in 2c.6 on a representative subset before expanding it.
  If that work is deferred, restrict the relevant main claims to a validated remote
  subset and describe close events as unresolved.
- Describe duration, extent, speed, and posterior uncertainty compactly. Where
  comparing SWR classes, distinguish unadjusted descriptions from duration/quality-
  adjusted associations. Do not indiscriminately adjust away properties that form
  part of the scientific question; state the estimand each comparison answers.
- Show animal-level effects and sensitivity to individual animals. Justify an
  inferential approach that respects animal/day/epoch nesting and the limited number
  of independent animals; changing a clustering option alone is not sufficient.

**Deliverable:** the main event/prevalence figure, a bounded robustness supplement,
and clear language specifying what "outside detected SWRs" means.

**Completion criterion:** the main conclusion's sensitivity is quantified and its
scope is defensible across the settings and animals examined. If it depends on
particular settings, report that dependence and revise the claim. This criterion
does not require an invariant event count or a desired positive result.

If the primary cohort or event definition changes, regenerate the affected
canonical tables and fits together under the existing lineage rules before
reporting downstream results. Do not combine evidence from incompatible event
inventories.

## Stage 4: make the behavioral section answer a focused question

**Work**

- Keep occurrence, content selection, and subsequent choice separate. Choose a
  primary behavioral comparison that advances the paper; use other models as
  controls or clearly labeled secondary analyses.
- For occurrence, state the exposure/risk set and the role of stop timing, outcome
  timing, and event history. Do not interchange per-stop rates and conditional
  onset intensity.
- For content, establish an appropriate spatial baseline and distinguish occupancy,
  current location, visit recency, realized reward history, inferred value, and
  inferred uncertainty. The belief-model uncertainty/recency relationship needs
  explicit treatment.
- Audit selected content fits and model comparisons against current estimator and
  convergence diagnostics. Resolve any applicable defects before interpreting
  coefficients, uncertainty, or model ranking; regenerate affected outputs.
- Use animal-level estimates and defensible hierarchical inference. Define primary
  comparison families and report excluded/separated fits and their implications.
- Use "length-adjusted" and "unadjusted associations" unless a causal estimand
  and its assumptions are established. A difference between coefficients is not
  established by significance in only one SWR class.
- If claiming prediction, use information available before the choice and evaluate
  incremental performance over a behavioral baseline with splits that respect
  temporal dependence. Otherwise describe a conditional association. Retain the
  informative null result without interpreting it as proof of no behavioral role.

**Deliverable:** a focused behavioral figure and Results subsection showing the
primary effect, baseline, uncertainty, and relevant negative or sensitivity result.

**Completion criterion:** the reader can identify the behavioral information added
by the events, or the limits of that information, beyond the stated baseline.
Broad value/uncertainty modeling is optional unless it is essential to this answer.
If the evidence remains exploratory, reduce this section's main-text role.

## Stage 5: establish the scope of the mPFC association

**Work**

- Divide 4.1 into event-associated responses, responses beyond shared covariates,
  and content dependence. Do not let the existence of many reported quantities
  stand in for separate tests of these questions.
- Complete population inference and the relevant SWR-controlled comparison. Align
  the figure's population and aggregation with the summary used by the text.
- Evaluate matched immobile windows, shared behavioral context, hippocampal MUA,
  and graded ripple power as appropriate to the claim. Include SWRs without
  non-local events where they provide an interpretable comparison.
- Inspect common support and collinearity before interpreting adjusted effects.
  Matching on ripple power across threshold-defined classes may be infeasible;
  explain the comparison a continuous model can actually identify.
- Show population response estimates and uncertainty, with the inference respecting
  the animal/day/session structure and relevant repeated recordings. Do not rely
  only on a fraction of individually selected units.
- Pursue content-dependent responses if the paper claims that mPFC activity tracks
  represented content. A positive held-out improvement fraction alone is not a
  calibrated population test; respect the limitations of available shuffle references.
- Test a between-class contrast if asserting a difference. Define and justify
  equivalence margins only if asserting equivalence. Timing and association alone
  do not establish hippocampus-to-cortex transmission or causal importance.

**Deliverable:** a cortical-association figure with the primary response, controls,
population inference, and any supported content dependence.

**Completion criterion:** each cortical claim distinguishes an event-associated
response from an association beyond modeled shared activity and from a
content-dependent response. If controls are incomplete, use the narrower claim.
Perturbation experiments or a new mPFC decoder are not required for an explicitly
observational paper.

## Stage 6: draft the paper around the evidence

**Work**

- Build the main figures around model/validation, events outside detected SWRs,
  focused behavior, and mPFC associations. Combine redundant event examples and
  ripple descriptions; move secondary dynamics, sweeps, and comparator diagnostics
  to the supplement while retaining decisive validity evidence in the main text.
- Write captions and Results together. For each result, state the question,
  population, primary comparison, estimate/uncertainty, and limitation. Write
  negative or inconclusive findings where they constrain the interpretation.
- Revise the introduction and title to match the actual contribution and scope.
  Define replay/non-local terminology and remove unsupported implications about
  function, mechanisms, importance, or broadcast.
- Explain the relationship to prior work, remaining detection limits, inference
  across animals, observational design, and dataset/task scope in the Discussion.
- Reorganize question/figure tags through `questions.py` and the builder tag tables;
  regenerate the summaries and generated question view when required. Do not edit
  `paper/reported/questions.md` by hand or silently detach results from their sources.
- Add references through the project's Zotero workflow. Preserve the single
  reported-value pipeline; main-text and caption numbers come from summaries.

**Deliverable:** a readable coauthor draft with complete main figures, captions,
Results, Methods, Discussion, and a targeted supplement.

**Completion criterion:** a reader can follow the question → evidence → conclusion
chain for every main result. All cited results have current provenance, compatible
populations, appropriate uncertainty, and no pending status hidden by wording.
The draft's ambition matches the evidence rather than the inventory of analyses.

## Stage 7: verify the draft and prepare for submission

- Read the draft against the reviewer-question table in the review. For each concern,
  identify the figure/methods response or state the limitation; commission further
  work only where it changes a necessary claim.
- Regenerate the affected figures and reported values from the final analyses,
  complete applicable tests and `make release-check`, and build the manuscript.
  Technical reproducibility checks complement scientific review; they do not
  establish scientific validity by themselves.
- Complete the existing coauthor/journal metadata and public-reproduction items
  in [TODO](../../TODO.md), including the planned archival inputs and data statement
  before submission. Do not treat a successful manuscript build as completion of
  those publication tasks.

**Completion criterion:** the coauthor/submission version is traceable to its
analyses, its principal reviewer concerns have explicit responses, and its
reproduction and reporting commitments are satisfied at the relevant milestone.
