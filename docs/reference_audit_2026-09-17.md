# Scientific Reference Check

**Input**: The eight references used in [the project comparison](reference_comparison_2026-09-17.md), with emphasis on potential counterexamples to its novelty claims.

**Tools used**: Local PDF text extraction, selected page images, publisher/PubMed/PMC pages, and accessible supplementary documents. The literature-corpus helper could not start because its dependency download failed. This audit uses the papers themselves, not a successful corpus search.

**Summary**: All eight references are verified to exist and their main papers were read. The 15 claim checks below comprise 12 direct, one method, one indirect and one weak classification. Three claims require narrower wording; four substantive omissions were corrected. Mallory's separate supplement remains inaccessible. This is a comparison against these references, not a systematic priority review or a reanalysis of their data.

## Reading coverage

Page numbers below are physical pages of the supplied PDF, including any cover matter, not journal pagination. Reading covered main results, discussion, figure captions and methods. Supplement coverage is specified rather than assumed. Figures were checked principally through their captions and associated results; selected page images were also inspected, including Gupta's image-only supplementary pages and Comrie Figure S4. Supplementary videos and raw data were not analyzed.

| Reference and verified identity | Supplied file | Additional material checked | Most relevant evidence |
| --- | --- | --- | --- |
| [Yu et al., 2017, eLife](https://doi.org/10.7554/eLife.27621), *Distinct hippocampal-cortical memory representations for experiences associated with movement versus immobility* | `eLife-2017-Yu et al-Distinct hippocampal-cortical memory representations for experiences associated.pdf`, 26 pages | Online figure supplements on SWR thresholds, remote IAP coactivity, cell/spike matching and PFC controls; relevant author response about remote coactivity | Fig. 1, Fig. 3 and supplement 3, Fig. 4 matching controls, Fig. 6D–E; pp. 3, 6–8, 13–17, methods |
| [Yu et al., 2018, Nature Communications](https://doi.org/10.1038/s41467-018-04498-w), *Specific hippocampal representations are linked to generalized cortical representations in memory* | `Nature Comm.-2018 -Yu et al.-Specific hippocampal representations are linked to generalized cortical representations in memory.pdf`, 11 pages | Text/captions of all 14 supplementary figures from the publisher's DOCX | Figs. 5–6, pp. 5–10; Supplementary Figs. 13–14 |
| [Yu & Frank, 2021, PLOS Biology](https://doi.org/10.1371/journal.pbio.3001393), *Prefrontal cortical activity predicts the occurrence of nonlocal hippocampal representations during spatial navigation* | `PLOS Biology-2021-Yu_Frank-Prefrontal cortical activity predicts the occurrence of nonlocal hippocampal.pdf`, 27 pages | S1–S8 captions in the full article XML | Figs. 2–7, pp. 4–14; methods pp. 16–21; S5 |
| [Gupta et al., 2010, Neuron](https://doi.org/10.1016/j.neuron.2010.01.034), *Hippocampal Replay Is Not a Simple Function of Experience* | `Neuron-2010-Gupta et al-Hippocampal Replay Is Not a Simple Function of Experience.pdf`, 24 pages | Bundled supplementary figures/captions S1–S4 and methods; image-only pages 13, 17 and 18 inspected | Figs. 4–6, pp. 5–8; supplementary sequence detection and Bayesian-decoding control |
| [Gillespie et al., 2021, Neuron](https://doi.org/10.1016/j.neuron.2021.07.029), *Hippocampal replay reflects specific past experiences rather than a plan for subsequent choice* | `Neuron-2021-Gillespie et al-Hippocampal replay reflects specific past experiences rather than a plan for.pdf`, 33 pages | Bundled S1–S8 captions and STAR Methods | Figs. 2–8, pp. 5–14; event detection p. 19; stationary-state control p. 20; choice controls S8 |
| [Mallory et al., 2025, Science](https://doi.org/10.1126/science.ads4760), *The time course and organization of hippocampal replay* | `Science-2025-Mallory et al. - 2025 - The time course and organization of hippocampal replay.pdf`, 9 pages | Separate supplementary PDF requested but publisher returned HTTP 403; supplementary controls only assessed through descriptions in the main text | Figs. 1–4, article pp. 1–8; timing, content self-avoidance, adaptation model and MEC manipulation |
| [Comrie et al., spatial-bandit manuscript](https://doi.org/10.1101/2024.09.23.613567), *Hippocampal representations of alternative possibilities are flexibly generated to meet cognitive demands* | `Spatial_bandit_manuscript_v3.2_biorxiv_figs.pdf`, 51 pages | Supplied v3.2 methods, S1–S5 and video descriptions; videos not watched | Figs. 2–5; speed restriction p. 30; S4 pp. 39–40; S5 pp. 41–42 |
| [Widloski & Foster, 2025, Nature Communications](https://doi.org/10.1038/s41467-025-65181-5), *Replay without sharp wave ripples in a spatial memory task* | `Widloski and Foster - 2024 - Replay without sharp wave ripples in a spatial memory task.pdf`, 14 pages | Text/captions of Supplementary Figs. 1–6 from the publisher PDF | Fig. 1, pp. 2–4; stationary exclusion p. 9; detection p. 11; Supplementary Fig. 3 |

The Widloski filename's 2024 date is misleading: the supplied article was published in November 2025. The linked [April 2026 correction](https://www.nature.com/articles/s41467-026-72252-8) changes reference entries. For Comrie, the public DOI establishes identity; the supplied v3.2 document supplies the claims and page anchors here.

## Definitions that affect the comparison

| Paper | What the relevant representation/event means | Consequence for this project |
| --- | --- | --- |
| Yu 2017 | IAPs have well-specific ongoing firing outside SWRs. Remote reactivation is tested using their firing and coactivity during SWRs occurring away from their fields. | Ongoing immobility coding is established; the demonstrated remote reactivation has an SWR restriction. |
| Yu 2018 | PFC reactivation is related to hippocampal content within SWRs, with controls for task firing and ripple power. | Generic PFC modulation is weaker evidence than their content-specific mapping. |
| Yu 2021 | Temporally isolated spikes during theta/locomotion, with spatial/directional incongruity and pairwise coactivity. Locomotion is >2 cm/s. | A direct non-SWR nonlocal precedent. This is not the same event definition as a decoded stationary representation; its speed range overlaps this project's <4 cm/s criterion. |
| Gupta 2010 | Ordered spike sequences passing controls and an SWR-coincidence requirement. | A sequence-first search does not mean the retained events are non-SWR. |
| Gillespie 2021 | Remote content places most event probability on a different track segment; continuous-state content can remain stationary. Alternative MUA detection is also tested. | Remote is not identical to this project's nonlocal latent state. MUA detection does not prove SWR absence. |
| Mallory 2025 | The main linear-track set begins near the animal and progresses away; open-field retrospective/prospective content is defined relative to paths. | A local event start can coexist with substantial nonlocal content. Event self-history and repeated-content avoidance are different quantities. |
| Comrie v3.2 | High-confidence representations on another segment during running >10 cm/s; nearby alternatives and distant patches are analyzed separately. | Choice prediction for nearby paths cannot be assigned to all remote content. Maximum actual-to-represented distance is not within-event spatial extent. |
| Widloski 2025 | Spatially coherent trajectories, duration >100 ms and dispersion >12 cm, decoded in 80 ms windows. | Brief stationary events are excluded. Prevalence, extent and duration cannot be compared directly without accounting for different detection and measurement rules. |

## Claim 1

**Text**: “Extended replay can occur without ripples or population bursts; ripple expression depends on represented location.”

**Citation(s)**: Widloski & Foster 2025.

**Verification**: verified_exists.

**Classification**: direct.

**Evidence**: Figure 1 establishes ripple/burst-less replay; Figures 2–5 establish spatially structured ripple expression during replay. Individual-tetrode and threshold controls go beyond a single consensus-threshold observation. Supplementary Figure 3 addresses movement/theta, spike-matched decoding quality and behavioral context.

**Recommendation**: keep. The existence of non-SWR replay and detection without an obligatory ripple/burst trigger are established precedents.

## Claim 2

**Text**: “Brief, nearly stationary well representations were excluded by their detector.”

**Citation(s)**: Widloski & Foster 2025.

**Verification**: verified_exists.

**Classification**: direct.

**Evidence**: Methods p. 11 require >100 ms duration and >12 cm spatial dispersion; discussion p. 9 explicitly excludes stationary replay. Their ripple/burst-less trajectories are already shorter than other trajectories. Supplementary Figure 3M reports a higher prevalence at the Random well before memory-guided navigation to Home.

**Recommendation**: keep. The useful extension is the excluded focal repertoire and its joint characterization, not shorter duration or any behavioral association by themselves. An 80 ms decoding window further limits direct comparisons with this project's much shorter events.

## Claim 3

**Text**: “Well-specific ongoing firing and hippocampal–PFC coactivity occur outside SWRs.”

**Citation(s)**: Yu et al. 2017.

**Verification**: verified_exists.

**Classification**: direct.

**Evidence**: Figure 1 defines IAPs using immobility firing after excluding SWRs. Figure 6D–E and pp. 13–15 describe hippocampal–PFC coactivity outside SWRs. The discussion distinguishes ongoing local representation from reactivation. It relates IAP properties to N-wave-coupled cells but does not identify N-waves directly in the foraging recordings.

**Recommendation**: keep. The initial comparison underemphasized this result; neither immobility-associated firing nor non-SWR hippocampal–PFC coordination is new here.

## Claim 4

**Text**: “Remote well and path representations reactivate largely separately during SWRs, with distinct PFC activity.”

**Citation(s)**: Yu et al. 2017.

**Verification**: verified_exists.

**Classification**: direct.

**Evidence**: Figure 3 and its supplement 3 show remote IAP firing/coactivity during SWRs outside their place fields. Figures 4–6 distinguish movement- and immobility-associated content and PFC responses. Approximately 93% of SWRs contain one class rather than both; well-associated SWRs are shorter/weaker, including after cell/spike matching. The author response about remote IAP coactivity also specifies an SWR analysis.

**Recommendation**: keep. Do not turn this into a demonstrated class of remote, immobile, non-SWR events. Conversely, do not portray the well/path and short/weak-ripple relationships as entirely new.

## Claim 5

**Text**: “Specific hippocampal representations coactivate generalized PFC task representations during SWRs.”

**Citation(s)**: Yu et al. 2018.

**Verification**: verified_exists.

**Classification**: direct.

**Evidence**: Figures 5–6, pp. 5–9, relate generalized task-phase coding in PFC to SWR participation and show different path locations engaging common PFC representations. Supplementary Figure 14 uses a ripple-power control. The task was familiar; the finding does not directly demonstrate the formation of a new schema.

**Recommendation**: keep. A PFC onset response in this project supports modulation; demonstrating which information PFC represents would be a stronger and separate result.

## Claim 6

**Text**: “Sparse, theta-associated nonlocal activity during movement is coordinated with PFC, whose activity predicts its occurrence.”

**Citation(s)**: Yu & Frank 2021.

**Verification**: verified_exists.

**Classification**: direct.

**Evidence**: Isolated spiking occurs outside a cell's usual spatial/directional context. Figure 5 supports coordinated assemblies rather than independent anomalous spikes. Figures 6–7 compare PFC activity with behaviorally matched controls and predict isolated spiking using cross-validation, including preceding activity 4–8 theta cycles earlier. SWRs are excluded; S5 examines theta. These are sparse nonlocal representations, not a decoded trajectory analysis.

**Recommendation**: keep. This directly answers the question about Jai's remote representations outside SWRs: yes. Nonlocality without SWRs and cross-area prediction are already established.

## Claim 7

**Text**: “Extend the cross-area question to immobility and low-theta events.”

**Citation(s)**: Yu & Frank 2021, as the comparison being extended.

**Verification**: verified_exists.

**Classification**: indirect.

**Evidence**: Yu's methods p. 16 use >2 cm/s; the project uses <4 cm/s. Behavioral labels therefore overlap at 2–4 cm/s. A distinct physiological regime is plausible but requires the project's theta/speed evidence, preferably including a stricter stillness subset. Temporal isolation of spikes also does not by itself measure the duration of an underlying representation.

**Recommendation**: narrow_claim.

**Suggested rewrite**: “Test cortical modulation around decoded nonlocal events at low speed and low theta, and quantify their relationship to Yu's theta-associated isolated-spiking regime.”

## Claim 8

**Text**: “Replay emphasizes previous goals and less recently visited places, with little reliable relationship to subsequent choice.”

**Citation(s)**: Gillespie et al. 2021.

**Verification**: verified_exists.

**Classification**: direct.

**Evidence**: Figures 4–8 show increased replay after repeated rewards at the goal, persistence for previous goals, and enrichment of less recently visited unrewarded arms. Choice analyses, including S8, do not establish a reliable effect across subjects and analyses. This is not a statement that every individual choice coefficient is exactly zero, or that replay cannot support planning in any task.

**Recommendation**: keep. Reward-history relationships, selective remote sampling and weak next-choice prediction are substantial precedents. A posterior reward-belief mean is not identical to a previous-goal label.

## Claim 9

**Text**: “Their continuous decoding state includes stationary remote events; an explicit stationary classifier labels fewer than 30% of remote events stationary, and removing them leaves the conclusions unchanged (Methods, PDF p. 20).”

**Citation(s)**: Gillespie et al. 2021; this statement was added during the second audit.

**Verification**: verified_exists.

**Classification**: direct.

**Evidence**: STAR Methods p. 20 explicitly describes this classifier check and fraction. Main-text p. 5 and methods p. 19 also report that MUA-based detection leaves the conclusions unchanged. The stationary result is an empirical precedent; the MUA control is not an analysis demonstrating that the events lack SWRs.

**Recommendation**: keep. Stationary remote content and non-ripple-triggered analysis cannot themselves carry novelty claims. This was a substantive omission from the earlier comparison.

## Claim 10

**Text**: “Gillespie did use Poisson GLMs of **arm-specific replay counts/rates**, not only normalized content summaries (Figures 3D–E, 4 and 8).”

**Citation(s)**: Gillespie et al. 2021.

**Verification**: verified_exists.

**Classification**: method.

**Evidence**: Figures 3D–E, 4 and 8 and the statistical methods fit arm-specific replay counts per trial with Poisson models. The described model should not be treated as equivalent to this project's time-resolved occurrence model with within-stop timing and self-history; an immobility-duration exposure offset is not specified in that arm/trial model.

**Recommendation**: keep. Describe the present occurrence/allocation separation and temporal controls as the extension, rather than claiming that earlier studies did not model replay rates.

## Claim 11

**Text**: “Gupta et al. already found replay of paths unvisited for more than ten minutes, greater opposite-side replay when that side was infrequently experienced, and constructed shortcut sequences (Figures 4–6; PDF pp. 5–8).”

**Citation(s)**: Gupta et al. 2010.

**Verification**: verified_exists.

**Classification**: direct.

**Evidence**: Figures 4–6, pp. 5–8, support these claims. Supplementary trajectory counts distinguish rarely experienced from never-experienced paths: the strict never-experienced top-shortcut result includes 11 events in one rat, within a larger set of 19 shortcut events. The sequence-selection methods retain an SWR-coincidence requirement. Reward-belief uncertainty is not estimated.

**Recommendation**: keep. Do not label all 19 shortcuts never experienced or treat preferential sampling of less-visited places as a direct test of uncertainty computation. The present uncertainty account is an interpretation that could explain related selectivity, not a result established by Gupta.

## Claim 12

**Text**: “Mallory et al. find early avoidance of the just-traversed path for roughly three seconds, avoidance of recently replayed content over roughly one second, and later retrospective enrichment.”

**Citation(s)**: Mallory et al. 2025.

**Verification**: verified_exists.

**Classification**: direct.

**Evidence**: Main-text Figures 2–4 establish these content/timing relationships. The adaptation model reproduces aspects of the organization; MEC inhibition reduces late retrospective bias and changes replay extent/recruitment. Neither directly establishes neuronal fatigue as the biological cause of all replay selection. Main text reports robustness to imposing an SWR requirement and alternative detectors, but the separate supplement could not be inspected.

**Recommendation**: keep. Occurrence self-history in this project is not a replication or refutation of content self-avoidance. Reduced repetition of content can coexist with an increased event rate after a preceding event.

## Claim 13

**Text**: “During running, alternative-path content tracks relative value and predicts choice; representational extent increases around periods of uncertain local values and updating.”

**Citation(s)**: Comrie et al., supplied v3.2.

**Verification**: verified_exists.

**Classification**: direct, for the restricted content and uncertainty relationship tested.

**Evidence**: Figures 2–3 concern nearby Stay/Switch alternatives. Figure S4D, pp. 39–40, finds that distant patches do not favor the subsequently selected or immediately previous patch. Figures 4–5 and S5F–H align maximum represented distance with the time course of uncertainty about the upcoming/chosen port and local updating. That is not the same as an adjusted association between a remote candidate's uncertainty and its allocation.

**Recommendation**: narrow_claim.

**Suggested rewrite**: “Nearby alternative-path representations during running track relative value and predict choice. Distant representations do not favor the next or previous patch; their spatial reach follows the time course of local uncertainty and updating.”

## Claim 14

**Text**: “The absence of a stay/leave association here and its presence during running in Comrie are suggestive of state dependence; different analyses and cohorts do not yet establish a formal behavioral double dissociation.”

**Citation(s)**: Comrie et al., supplied v3.2, compared with this project's results; this is the earlier wording that was replaced.

**Verification**: verified_exists.

**Classification**: weak.

**Evidence**: Although qualified, the inference underplays the distinction within Comrie between nearby alternatives and distant patches. Different content classes can produce the apparent contrast without a running-versus-immobility explanation. This project also retains a modest, non-distance-controlled choice correlation in one analysis, so a blanket choice-null description is too broad.

**Recommendation**: narrow_claim.

**Suggested rewrite**: “A behavioral-state comparison requires matched content definitions, cohorts and analyses; the present results do not establish a general running-versus-immobility dissociation in choice prediction.”

## Claim 15

**Text**: “Maximum distance between actual and represented position increases around patch entry and follows a time course similar to uncertainty about the upcoming/visited port and local value updating (Figures 4–5 and S5F–H; pp. 13–20, 42).”

**Citation(s)**: Comrie et al., supplied v3.2; revised wording.

**Verification**: verified_exists.

**Classification**: direct.

**Evidence**: Figures 4–5 and S5F–H support aligned changes across trials around patch switches. The Beta–Bernoulli model updates the visited port with the observed outcome and relaxes beliefs about unvisited ports toward a uniform distribution. Uncertainty is therefore model-derived and linked to visit history. The metric is actual-to-decoded distance, not distance traversed during a decoded event.

**Recommendation**: keep. The project's more specific candidate contribution is uncertainty of the represented alternative predicting allocation during immobility, shared across SWR classes. Identifying that association independently of value, recency and event length remains an analysis question.

## Overall notes

The four material omissions corrected in the comparison are Gillespie's stationary remote content and MUA control; Comrie's distinct result for distant patch content; Widloski's duration and behavioral-context precedents; and the overlap between Yu 2021's and this project's speed criteria. Yu 2017's non-SWR ongoing activity and Yu 2021's non-SWR remote activity were also checked again rather than treated as interchangeable findings.

The defensible contribution is the **joint characterization** of brief, focal remote reward-location representations across ripple expression, their candidate-location uncertainty association, and potentially their cortical accompaniment. Each component must be compared with its closest precedent. The shared uncertainty relationship across SWR classes is presently better supported than a class-specific uncertainty mechanism.

The project comparison retains its own evidence limits: incomplete full-dataset spike-evidence and detector-sensitivity validation, pending stronger within-arm value/uncertainty identification, and preliminary mPFC population inference. No reference establishes that this project has already resolved those questions. Nor does a lack of precedent within these eight papers establish worldwide priority.

**Access limitation**: the publisher returned HTTP 403 for Mallory's separate supplementary PDF. Its main paper was read in full, but supplementary results have not been independently checked beyond their descriptions in that paper. The remaining source gap should stay explicit when using this audit.
