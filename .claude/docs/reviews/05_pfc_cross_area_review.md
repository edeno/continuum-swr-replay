## Verdict

> **Historical review.** This document describes the pre-refit artifacts it
> inspected. The canonical-event, matched-control and ridge/held-out work has
> since landed. As of 2026-09-18, Figure 4 is descriptive and includes all
> 17,280 units with four finite primary curves; it applies neither the 1 Hz
> screen recorded below nor a p-value/held-out-statistic inclusion screen. See
> `docs/audits/2026-09-18-pfc-ridge-heldout.md` for the current audit.

Changes requested before using Fig. 4 / claim #20 as publication evidence. The current PFC code supports a preliminary neuron-level association between mPFC firing and detector-defined non-SWR events, but it does not yet implement the canonical replay-table `is_swr` definition, the matched immobile non-event control promised in the claims table, or animal/session-level inference needed for a cross-area claim.

## Main Findings

1. **Claim/figure scope is narrower than the current figure language.** [paper/claims_table.md](../../../paper/claims_table.md:58) frames claim #20 as mPFC GLM modulation for non-SWR replay plus a matched immobile non-event control, with SWR comparison only as reference and equivalence recut pending. [paper/figure_claims_plan.md](../../../paper/figure_claims_plan.md:42) still says Fig. 4 shows PFC neurons respond to replay regardless of SWR detection. The current implemented figure/statistics do not yet justify equivalence or “regardless” language.
2. **The PFC event split is not the canonical replay-table split.** [src/continuum_swr_replay/analysis/pfc/glm.py](../../../src/continuum_swr_replay/analysis/pfc/glm.py:1002) recomputes non-local events from summed acausal state probabilities, filters by `max_speed < immobility_threshold`, and splits them with `get_overlap_times(..., data["ripple_times"])` at [glm.py](../../../src/continuum_swr_replay/analysis/pfc/glm.py:1017). The canonical event table says the paper-wide `is_swr` is strict positive-duration overlap derived from `SWR_event` labels over the full non-local label window, and explicitly says threshold-crossing `ripple_times` are not used for classification [event_table.py](../../../src/continuum_swr_replay/analysis/hippocampus/event_table.py:32).
3. **The saved population artifacts appear stale relative to current code.** A local pickle inspected from `Processed-Data` still contained `SWR + non_local_no_SWR vs. non_local` columns, while current [glm.py](../../../src/continuum_swr_replay/analysis/pfc/glm.py:727) intentionally removed that non-nested LR comparison and the test pins that removal [test_glm_pipeline.py](../../../tests/analysis/pfc/test_glm_pipeline.py:557). Any Fig. 4 generated from existing artifacts needs provenance/rerun confirmation.

## Definition-Drift Risk

The canonical replay/content tables retain an event if at least one label-window bin survives the immobile + probability filter, but classify `is_swr` from the full label window and label-derived `SWR_event` intervals [tables.py](../../../src/continuum_swr_replay/analysis/replay/tables.py:219). PFC instead reconstructs events directly from threshold crossings, uses filtered interval boundaries, and overlaps with `data["ripple_times"]` [glm.py](../../../src/continuum_swr_replay/analysis/pfc/glm.py:1007). This can reassign boundary/partial-overlap events and can change the non-SWR denominator used for mPFC modulation.

The example-neuron path in [scripts/generate_figure04.py](../../../scripts/generate_figure04.py:330) repeats the same noncanonical event construction. It also uses `<=` for immobility at [generate_figure04.py](../../../scripts/generate_figure04.py:357), while the batch PFC run uses `<` at [glm.py](../../../src/continuum_swr_replay/analysis/pfc/glm.py:1042).

## Alignment/Binning Risks

Spike counts and event impulses are binned against `position_info.index` using `get_spikecount_per_time_bin` [glm.py](../../../src/continuum_swr_replay/analysis/pfc/glm.py:30), and event design matrices use event `start_time` only [glm.py](../../../src/continuum_swr_replay/analysis/pfc/glm.py:336). This is plausible for onset modulation, but it needs a real-data alignment audit: confirm event starts, spike bins, and position bins all use the same left-edge convention and that no events fall just outside valid bins.

Lag sign is documented by tests: positive lags shift an event forward in the design matrix [test_glm_pipeline.py](../../../tests/analysis/pfc/test_glm_pipeline.py:121). Before interpretation, confirm the plotted “time from event” sign matches the fitted convolution convention, not just the synthetic test.

The GLM fitting uses only immobile rows for fitting [glm.py](../../../src/continuum_swr_replay/analysis/pfc/glm.py:470), while SWR event regressors include all ripple times by design [glm.py](../../../src/continuum_swr_replay/analysis/pfc/glm.py:1028). That can be valid, but needs an explicit check that ripple regressors outside immobility do not leak edge effects into retained immobile rows near movement transitions.

## GLM/Statistical Risks

Good: current code detects rank-deficient designs from absent event classes and routes affected tests/modulations to `NaN` [glm.py](../../../src/continuum_swr_replay/analysis/pfc/glm.py:703). It also removed at least one non-nested LR comparison [glm.py](../../../src/continuum_swr_replay/analysis/pfc/glm.py:724).

Still missing: the claim needs matched immobile non-event controls. Current models compare event regressors against constant/SWR/non-local alternatives, but there is no matched non-event onset design or permutation/control distribution in the reviewed PFC pipeline.

The figure summarizes neuron-level FDR across pooled neurons [load_population.py](../../../src/continuum_swr_replay/analysis/pfc/load_population.py:181) and plots percent significant neurons [generate_figure04.py](../../../scripts/generate_figure04.py:250). That does not account for neuron dependence within session/animal. Local loaded artifacts had 10,980 filtered neurons across 71 sessions and 5 animals, heavily imbalanced by animal, so animal/session-level summaries or hierarchical/cluster-robust inference are required.

Equivalence/similarity between SWR and non-SWR responses is not established. Heatmaps and population means can be shown descriptively, but equivalence requires a pre-specified effect-size margin, power check, and paired session/animal-level comparison.

## Tests Run Or Not Run

Ran: `uv run pytest tests/analysis/pfc tests/visualization/test_glm_plots.py -q` passed: 48 tests.

Artifact check performed: `load_population_data(Path("Processed-Data"), min_firing_rate=1.0)` loaded 10,980 neurons, 71 sessions, 5 animals, 201 lags, finite FDR columns. This validates loader mechanics, not event-definition correctness.

Not run: no data-heavy Spyglass/DataJoint rerun of `run_pfc_analysis_bandit.py`; no canonical event-table vs PFC event-table join; no regeneration of Fig. 4; no matched non-event control analysis.

## Next Checks

1. Rebuild PFC event sets from canonical `event_table.event_start/event_end/is_swr` and rerun mPFC GLMs; save event counts, thresholds, code hash, and definition metadata per session.
2. Compare old PFC threshold/ripple-times split against canonical label-derived `is_swr` per session: number reclassified, boundary cases, event-start shifts, and effect on GLM coefficients.
3. Add matched immobile non-event onset controls with the same session, immobility-bout time, duration/event-rate structure, and spike-bin alignment.
4. Report inference at animal/session level: per-animal fractions/effects, cluster bootstrap or mixed model, and FDR family defined by claim rather than by pooled neurons alone.
5. Add tests that fail if PFC event construction diverges from canonical replay tables, if figure/example immobility inequality differs from batch analysis, or if stale non-nested LR columns are present in consumed artifacts.
