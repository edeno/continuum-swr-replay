# Spike-Evidence / Reality Gate Review

## Findings

1. **High: current implementation supports a prototype/smoke claim, not the full-strength paper claim.** The claims table correctly says Goal 2 is validated on one session per animal and not all events yet ([paper/claims_table.md:98](../../../paper/claims_table.md#L98)). The script still samples at most 30 events per group per session ([scripts/analyze_event_spike_evidence.py:65](../../../scripts/analyze_event_spike_evidence.py#L65), [scripts/analyze_event_spike_evidence.py:320](../../../scripts/analyze_event_spike_evidence.py#L320)) and only recomputes per-event likelihoods. The saved-likelihood all-events workflow remains documented as TODO/pending validation ([docs/design/2026-06-25-non-swr-event-reality-design.md:147](../../../docs/design/2026-06-25-non-swr-event-reality-design.md#L147)).

2. **High: event definitions drift from the canonical event table.** The reality script reconstructs events from thresholded probabilities and per-bin immobility with `MIN_EVENT_BINS = 3` ([scripts/analyze_event_spike_evidence.py:57](../../../scripts/analyze_event_spike_evidence.py#L57), [scripts/analyze_event_spike_evidence.py:153](../../../scripts/analyze_event_spike_evidence.py#L153)). Canonical detection uses `get_confident_times(..., minimum_duration=0.015)` ([src/continuum_swr_replay/analysis/hippocampus/event_detection.py:48](../../../src/continuum_swr_replay/analysis/hippocampus/event_detection.py#L48)) and canonical classification is based on `non_local_event` labels and strict positive-duration SWR overlap ([src/continuum_swr_replay/analysis/hippocampus/event_table.py:206](../../../src/continuum_swr_replay/analysis/hippocampus/event_table.py#L206)). The reality script instead sets `is_swr = swr[sl].any()` on the sliced bins ([scripts/analyze_event_spike_evidence.py:159](../../../scripts/analyze_event_spike_evidence.py#L159)), which can disagree with the strict boundary convention tested elsewhere.

3. **Medium: the core likelihood path is prior-excluded, but not held-out.** `event_likelihood` directly calls `predict_clusterless_kde_log_likelihood(..., is_local=False)` rather than reading posterior confidence ([scripts/analyze_event_spike_evidence.py:245](../../../scripts/analyze_event_spike_evidence.py#L245)). That does test likelihood-only evidence at the posterior-derived MAP and animal positions. It is not independent spike evidence: the same spikes influence event detection, MAP selection, and likelihood scoring, so the matched baseline and saved-likelihood parity checks are essential.

4. **Medium: baseline windows are not matched tightly enough for a definitive null.** Baselines are all contiguous immobile windows with low non-local probability and 3-40 bins ([scripts/analyze_event_spike_evidence.py:170](../../../scripts/analyze_event_spike_evidence.py#L170)); they are not duration-matched event-by-event and do not exclude SWR/MUA periods. Missing matching variables include duration, animal position/arm/well, distance to reward well, time within stop, reward context, ripple/theta/MUA context, and session-local occupancy/encoding coverage. Do not match on firing rate if spike presence is being tested, but do match or stratify the behavioral/spatial opportunity variables that can shift `logL(MAP)-logL(animal)`.

5. **Medium: MAP/animal binning is mostly correct in the spike-evidence script, but surrounding distance code is not fully consistent.** The core ΔLL lookup uses `env.get_distances_to_interior_bins(...).argmin(1)` for MAP and animal bins ([scripts/analyze_event_spike_evidence.py:263](../../../scripts/analyze_event_spike_evidence.py#L263), [scripts/analyze_event_spike_evidence.py:279](../../../scripts/analyze_event_spike_evidence.py#L279)), which matches the design. However, there is no finite-distance guard for disconnected graph components, no assertion that `L.shape[1] == len(pos)`, and no assertion that `gpi` is aligned to interior bins ([scripts/analyze_event_spike_evidence.py:199](../../../scripts/analyze_event_spike_evidence.py#L199)). The shared content-occupancy script still uses linear-position argmin shortcuts for wells/MAP bins ([scripts/analyze_content_occupancy.py:133](../../../scripts/analyze_content_occupancy.py#L133), [scripts/analyze_content_occupancy.py:141](../../../scripts/analyze_content_occupancy.py#L141)).

6. **Medium: spike-window slicing and per-spike normalization need direct validation.** Spikes are sliced inclusively from `time[0]` to `time[-1]` before likelihood evaluation ([scripts/analyze_event_spike_evidence.py:249](../../../scripts/analyze_event_spike_evidence.py#L249)). If decode times are bin centers, this may omit edge-bin spikes or double-count exact-boundary spikes depending on the likelihood function's bin convention. `dll_per_spike` divides the summed point-process log-likelihood difference by observed spike count ([scripts/analyze_event_spike_evidence.py:258](../../../scripts/analyze_event_spike_evidence.py#L258), [scripts/analyze_event_spike_evidence.py:290](../../../scripts/analyze_event_spike_evidence.py#L290)); that is useful, but it is not purely "per mark" because no-spike/intensity terms are still in the numerator.

7. **Low/Medium: the GPI control is incomplete in the script.** The script computes event distance to the GPI peak ([scripts/analyze_event_spike_evidence.py:282](../../../scripts/analyze_event_spike_evidence.py#L282)) and plots MAP distributions, but the documented random-position reference is not implemented; `GPI_REMOTE_CM` is declared but unused ([scripts/analyze_event_spike_evidence.py:62](../../../scripts/analyze_event_spike_evidence.py#L62)). Distance from an arithmetic mean MAP to the GPI peak can also be misleading for events that jump between graph branches.

## Verdict

**Approve only as a prototype reality gate.** The implementation is directionally testing prior-excluded clusterless likelihood rather than posterior confidence, and the core MAP/animal bin lookup uses track distance. It should not yet be used as the full paper-level evidence for all non-SWR events until event definitions are made canonical, baselines are strengthened, and saved-likelihood/all-event validation is complete.

## Verified Behavior

- Reviewed Goal 2 context in the claims table: #6-#7c explicitly frame spike count/tetrode engagement, prior-excluded likelihood, GPI specificity, and matched non-event baseline, with full-dataset caveats ([paper/claims_table.md:28](../../../paper/claims_table.md#L28), [paper/claims_table.md:86](../../../paper/claims_table.md#L86)).
- Verified the script recomputes likelihood from the saved encoding model and marks, not the posterior array, via `predict_clusterless_kde_log_likelihood` with `is_local=False` ([scripts/analyze_event_spike_evidence.py:236](../../../scripts/analyze_event_spike_evidence.py#L236)).
- Verified ΔLL is computed as `L[MAP_bin] - L[animal_bin]` after track-distance bin lookup ([scripts/analyze_event_spike_evidence.py:278](../../../scripts/analyze_event_spike_evidence.py#L278)).
- Ran lightweight adjacent tests: `110 passed` for spatial mapping, SWR-overlap consistency, classified event table, and event detection. These validate nearby contracts, not the reality script itself.

## Numerical / Indexing Risks

- Add explicit checks that `L` is 2-D, `L.shape[0] == e - s + 1`, and `L.shape[1] == len(pos)` before indexing.
- Add guards that `_bins_for` distances are finite; otherwise disconnected components can silently map to bin 0 via `argmin`.
- Validate `summed_ground_process_intensity` is already interior-bin-aligned, or apply the same interior mask used for `pos`.
- Replace GPI distance from `mean(map_estimate)` with a per-timebin or event-median graph distance to the GPI peak.
- Compare per-event sliced likelihood against full-session saved likelihood sliced to the same bins; this is the highest-value test for endpoint/time alignment.

## Scientific Validation Status

The current result is a **5-session, sampled-event validation** consistent with the design spec's prototype/scaling notes ([docs/design/2026-06-25-non-swr-event-reality-design.md:119](../../../docs/design/2026-06-25-non-swr-event-reality-design.md#L119)). It is not yet a full-dataset, all-events, decode-pinned result. The local-kernel re-decode for close-event place-field leakage is also still out of scope/pending ([docs/design/2026-06-25-non-swr-event-reality-design.md:169](../../../docs/design/2026-06-25-non-swr-event-reality-design.md#L169)).

## Test Gaps

- No direct unit tests for `build_session_events`, `event_likelihood`, `_bins_for`, `event_metrics`, or `summarize`.
- No test that the reality script's `is_swr` agrees with canonical strict-positive-duration overlap.
- No synthetic known-place likelihood test where spikes are generated from one known bin and ΔLL must be positive for that bin over the animal bin.
- No saved-vs-recomputed likelihood parity test, despite being required by the design ([docs/design/2026-06-25-non-swr-event-reality-design.md:161](../../../docs/design/2026-06-25-non-swr-event-reality-design.md#L161)).
- No baseline-matching test showing sampled baselines match event duration/position/opportunity distributions.

## Next Checks

1. Rebuild the reality event set from `build_classified_nonlocal_event_table` or prove row-level parity with it.
2. Implement saved-likelihood reader and compare saved vs recomputed ΔLL on one re-decoded session.
3. Replace baseline sampling with per-event matched windows on session, duration, position/arm/well or distance-to-well, time-within-stop/reward context, and SWR/MUA/theta exclusion.
4. Add finite-distance/interior-shape assertions and a disconnected-graph regression test.
5. Run the all-events analysis, aggregate at animal/session level, and report the sampled prototype only as a feasibility result until that completes.
