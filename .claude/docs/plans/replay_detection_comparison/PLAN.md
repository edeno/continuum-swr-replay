# Replay detection comparison — implementation plan

**Status (2026-10-18):** Phase 0 done. Phase 2 implemented, tested and reviewed four
times (through 2026-09-25), including the shuffle, the event table, a per-epoch runner and
the simulation controls ([memoryless-simulation-controls](../memoryless-simulation-controls/PLAN.md),
done). The descriptive real-data pilot is done (2026-10-02, five epochs, with the shuffle,
after two input fixes: decode-time marks, grid rate): 28 of 1,409 candidates meet duration
and dispersion, 26 pass the shuffle, 26 overlap an SWR, and 2 rest on one kept window
outside their pieces (now panel f of the comparison figure). The pilot's three checks
(null validation, two sensitivities, reference-output rerun): the reference-output
rerun is done (2026-10-03, decisions unchanged). The full run with the shuffle is done on all
242 analyzed epochs (2026-10-11, R12b of the full pipeline refresh, run
`d0964cc76ed2-seed20260922`, on the R11 decodes). The other two pilot checks (null validation,
two sensitivities) were dropped by the lead on 2026-10-28: the Widloski results are reported
without them, and the port stays a documented adaptation whose clusterless null was not
separately validated.
Phase 1 is done. It was **re-scoped (2026-10-02): the SWR inventories are computed directly** by
calling the pinned `ripple_detection` detectors on the ripple-band LFP, with no Spyglass
registration, parameter row or populate; Spyglass integration moves to a final Phase 7. The
inventories (Kay, Karlsson, Yu, Zugaro) were computed in the full pipeline refresh (R3, then
R10 run `056eee0dcb5a`, 2026-10-10). Phase 3's accounting is implemented and tested on
synthetic inventories, and its inputs are ready. **Phase 3 Tasks 1-9 are built and reviewed on
branch `replay-detection-phase3`** (2026-10-17): the Carey runner, the adapters, the comparison
runner, the report, the sampled examples and the docs, tested on simulated epochs. The R1 Carey
smoke passed (run `a16413a1eeaf`, `chimi20200225_02_r1_HPC`, 104 events). R2-R6 are done: Carey on all
242 epochs (run `fa6537e48bcf`), the comparison smoke, the pilot, the full run (run `afeb897d850f`,
242 epochs `ok`), the report and the sampled examples; the lead reviewed them on 2026-10-18. Task plan:
[phase-3-inventory-comparison.md](phase-3-inventory-comparison.md), with eight detectors read as
nine inventories (Carey added as inventory 8, Long dropped, the memoryless detector read under
two cohorts; D1, D7 below). The lead deferred the two remaining
Phase 2 pilot checks (2026-10-17) and dropped them (2026-10-28); the Widloski column is
reported without them. Phase 4's task plan is written
([phase-4-event-features.md](phase-4-event-features.md)). **Phase 4 Tasks 1-10 are built and reviewed on
branch `replay-detection-phase4`** (2026-10-18): the features, the common spike-evidence diagnostic, the
selection rules, the populations, `availability.stored_coverage`, the runner `scripts/qa/run_event_features.py`,
the report `scripts/qa/report_event_features.py`, the examples `scripts/figures/generate_event_feature_examples.py`
and the docs. The R1 smoke passed (run `68f51dabdb3c`, `chimi20200225_02_r1_HPC`, 13.5 min and 4.95 GB on one
worker). The pilot ran on 5 epochs (run `86ed2e7fa90a`: 4 `ok`, and peanut, which failed there on a bug at
its last window, re-checked `ok` under `f8a6984ae781`); its report was written. The full run (R3) is pending,
after the final fix wave (an interval ending on the grid's last sample no longer fails 4 epochs). The lead
dropped the baseline comparison from Phase 4 (2026-10-18; D8). Phase 5 deferred,
Phase 6 parked, Phase 7 last.
The full pipeline refresh and its audit are in
[docs/audits/2026-10-12-full-pipeline-refresh.md](../../reviews/2026-10-12-full-pipeline-refresh.md).

The scientific design lives in
[docs/design/replay_detection_comparison/](../../../../docs/design/replay_detection_comparison/)
(overview + specs 01–05 + the detector table). This file is the *process* record: what is
built, where it lives, what order the remaining work goes in, and which decisions are still
open. It does not restate the design; when the two disagree, the spec wins and this file
is stale.

## Reading order

1. **Picking up the next phase?** Read the phase row below, then the spec it points at.
2. **Need the governing rule (harmonize substrate, not method internals) or the
   cross-cutting decisions X1–X11?** [overview.md](../../../../docs/design/replay_detection_comparison/overview.md).
3. **Need to know what a detector does relative to the others?**
   [detector_comparison.md](../../../../docs/design/replay_detection_comparison/detector_comparison.md).

## Phases

| phase | spec | deliverable | status |
|---|---|---|---|
| 0 | [01](../../../../docs/design/replay_detection_comparison/01-yu-swr-detector.md) §Architecture, [table](../../../../docs/design/replay_detection_comparison/detector_comparison.md) | Detectors implemented in `ripple_detection` with tests, README, changelog | **done** 2026-09-21: PRs #14–#19 merged to package `master` (b0431a4) after independent review |
| 1 | [01](../../../../docs/design/replay_detection_comparison/01-yu-swr-detector.md) §Producing the inventory, direct path | Yu and Karlsson SWR inventories for every epoch, **computed directly**: pin the package, read the CA1 ripple-band LFP and head speed (read-only fetches, as `get_ripple` already does), call `Yu_ripple_detector` / `Karlsson_ripple_detector`, Y5 speed gate, observation-availability table, per-epoch parquet + manifest recording the package commit and the threshold diagnostics. No Spyglass table, parameter row or registration is written | **done** 2026-10-10 in the full pipeline refresh; re-scoped 2026-10-02 (was: via Spyglass `RippleTimesV1`). `scripts/decoding/run_swr_detection.py` computed all four inventories directly, with `ripple_detection` pinned at `68e1784`: R3 first (run `50a98592acee`), then R10 (run `056eee0dcb5a`, with artifact run `fbbff144a123` and tetrode rule R-T). 255 epochs `ok` (the 242 analyzed and the 13 `SUSTAINED_NOISE`). Each epoch has the four event files, a `swr_observed` file, the Yu diagnostics and a Kay version check. No Spyglass table was written |
| 2 | [02](../../../../docs/design/replay_detection_comparison/02-memoryless-sequence.md) | Memoryless sliding-window sequence detector on the clusterless posterior; shuffle null; adapted cohort before/after shuffle | code done and independently reviewed twice, review fixes in (2026-09-22); the sweep and figure apply the paper's place-cell identity shuffle, with the reference's conventions where they carry over (zero jumps set aside, strict-below count, p ≤ 0.05; a third review found the port's "conservative" departures were manufacturing the by-duration jump trend, and they were reverted 2026-09-22). A fourth review (2026-09-25) led to the sorted zero-jump fix and a check against the reference's own arithmetic (02 §"Against the reference's own arithmetic"): the port lacked the reference's `+1e-3` Hz rate-map floor, which moved the significance-test pass rate from 88% to 69% on the sweep. The sorted path of the sweep and figure now decodes with it (D5), and the port then decides as the reference does on 99% of candidates; the published-definition rate reads 18% (was 11%). The port stays a documented adaptation, not a validated reproduction: the clusterless null used on recorded data is unchecked, and claims of reproduction or detector superiority wait for the pilot checks below. **Full run with the shuffle done (2026-10-11): R12b of the full pipeline refresh, run `d0964cc76ed2-seed20260922`, 242 epochs `ok`, 100 draws, on the R11 decodes with the SWR source `kay@056eee0dcb5a+artifacts@fbbff144a123`.** **The deferred pilot checks (null validation, two sensitivities) were dropped by the lead on 2026-10-28.** 02 assumption 8 (non-SWR events decoded partly from their own spikes) stays accepted; see D6 |
| 3 | [03](../../../../docs/design/replay_detection_comparison/03-inventory-comparison.md); task plan [phase-3-inventory-comparison.md](phase-3-inventory-comparison.md) | Nine inventories (eight detectors) on the common schema, observation-availability tables, all-pairs overlap table, single/no-detector cells, SWR label matrix, sampled examples | accounting done 2026-09-22 in `analysis/inventory_comparison/` (`intervals`, `schema`, `availability`, `overlap`, `components`, `recount`, `cells`, `swr_matrix`), test-first with mutation checks and an end-to-end simulated epoch; **inputs ready (2026-10-12):** the four SWR inventories (Phase 1), the Widloski table (Phase 2), the canonical classified event table and the decodes' MUA events exist for the 242 analyzed epochs. **Built and tested on simulated epochs, 2026-10-17, on `replay-detection-phase3` (Tasks 1-9, reviewed):** the Carey inventory (inventory 8, a fifth SWR definition, with four recorded adaptations and the package's `normalization_mask`; runner `scripts/decoding/run_carey_detection.py`), per-detector adapters (Kay, Karlsson, Yu and Zugaro from the direct SWR inventories, Carey, MUA HSE, the canonical classified table, the memoryless table under two cohorts) that also build each detector's coverage from its inputs (`inventory_comparison/adapters.py`), a runner writing the tables per epoch (`scripts/qa/run_inventory_comparison.py`), per-session/animal reporting (`scripts/qa/report_inventory_comparison.py`), the sampled examples (`scripts/figures/generate_inventory_examples.py`), and excluded time by reason. **Run (R1-R6 done):** the R1 Carey smoke (passed); R2, Carey on all 242 epochs (run `fa6537e48bcf`); R3-R5, the comparison smoke, the pilot and the full run (run `afeb897d850f`, 242 epochs `ok`, nine inventories); R6, the report (`report/` in the run folder, with agreement under stricter match rules and boundary offsets) and the sampled examples (`paper/figures/inventory_comparison/`). The lead reviewed the report and examples on 2026-10-18. The Widloski columns are reported without the Phase 2 pilot checks, which the lead dropped on 2026-10-28. An independent review (2026-09-22) found cross-epoch key collisions, a partial-observation SWR label, float-rounded recount gaps, window support off by one sample, and dropped missing-run rows, all fixed test-first. Both items still open from it are fixed in Phase 3 (excluded time by reason for every comparison; still runs reach the neighboring samples). The full run's manifest records each epoch's wall clock and peak RSS. **Task plan written 2026-10-17** ([phase-3-inventory-comparison.md](phase-3-inventory-comparison.md)): the Carey detector and runner, both review items, one adapter per inventory, the per-epoch runner, the report, the sampled examples and a runbook; the lead answered its eleven questions the same day. **Done:** built, run and reviewed; Phase 4 starts from run `afeb897d850f`'s tables |
| 4 | [04](../../../../docs/design/replay_detection_comparison/04-event-features.md) §Initial report | Event features, common spike-evidence diagnostic, boundary and rejection diagnostics, at default settings | **task plan written 2026-10-18** ([phase-4-event-features.md](phase-4-event-features.md)), on branch `replay-detection-phase4`: per-event features (timing, engagement, a common ripple and MUA trace, `ss_*` and `ms_*` spatial features, context) on every inventory, the common spike-evidence diagnostic with matched baseline windows, the populations from Phase 3's membership, the memoryless rejection stages for ours-only events and the comparator-only checks, the report, sampled examples and a runbook; ten questions for the lead. Boundaries are already in Phase 3's report. **Built and reviewed, 2026-10-18** (Tasks 1-10, on `replay-detection-phase4`): `inventory_comparison/{features,evidence,selection,populations}.py`, `availability.stored_coverage`, the runner `scripts/qa/run_event_features.py`, the report `scripts/qa/report_event_features.py` and the examples `scripts/figures/generate_event_feature_examples.py`. **Run:** R1 smoke passed (run `68f51dabdb3c`, 13.5 min, 4.95 GB, one worker); the pilot ran on 5 epochs (run `86ed2e7fa90a`: 4 `ok`, and peanut re-checked `ok` under `f8a6984ae781`) and its report was written; the full run is pending. **Baseline comparison dropped (lead, 2026-10-18; D8):** the runner draws no baseline windows and computes the evidence on the events' rows only, and the report gives the evidence beside the spikes per row; `selection`'s baseline functions are kept for a later design matched on spike count. **Implemented; the full run is pending** |
| 5 | [04](../../../../docs/design/replay_detection_comparison/04-event-features.md) §Planned follow-up | Threshold robustness: cost pilot, then one-at-a-time relax/tighten | deferred until Phase 4 is reported |
| 6 | [05](../../../../docs/design/replay_detection_comparison/05-validation.md) | Simulation or held-out validation | parked; reopen only for a performance claim |
| 7 | [01](../../../../docs/design/replay_detection_comparison/01-yu-swr-detector.md) §Producing the inventory, Spyglass integration | Final product: register Yu in Spyglass (or upstream the two-line PR), `RippleParameters` rows, populate `RippleTimesV1`, `get_ripple("Yu")`, and check the populated events equal the Phase 1 direct inventories | last; not started |

Order: 1 and 2 are independent and can proceed in parallel; 3 needs both; 4 needs 3; 7
comes last and changes no result if its equality check passes. Karlsson (inventory 2) is
computed directly in Phase 1 alongside Yu, by the same code path.

## Phase 0: package state (`~/Documents/GitHub/ripple_detection`)

That local clone is on the original development machine, not on the current one; here,
clone [Eden-Kramer-Lab/ripple_detection](https://github.com/Eden-Kramer-Lab/ripple_detection)
(GitHub `master` carries `Yu_ripple_detector` and has 7245c16 in its history). Run tests with `.venv/bin/python -m pytest -q --no-cov` (not `uv run`, which
drops a stray `uv.lock`). Package `master` at **7245c16** (2026-09-21) carries every
detector the comparison needs and is the commit Phase 1 pins. Merged in this order, each
after an independent Opus review whose findings were fixed before merge:

| PR | what | review outcome |
|---|---|---|
| #18 | ruff is the single formatter/linter; black and flake8 removed; ruff pinned to a minor version | merged after three one-line fixes |
| #14 | ripple-band filter designed for the actual rate; `minimum_sample_count` duration convention (23 not 24 samples at 1500 Hz/15 ms, about 20 % more marginal noise-only events, **detection-affecting**); `max_thresh` redefined as the largest threshold that still detects the event (greedy peak expansion could return 0.1); Karlsson stats on per-channel max; HSE validates inputs incl. NaN speed | merged after fixes |
| #15 | `simulate_LFP` `ripple_snr` (per-ripple, from each burst's filtered peak; unit-gain assumption was 40-47 % off at band edges), frequency/duration ranges; docs no longer state a z-score mapping | merged after fixes |
| #16 | `Zugaro_ripple_detector`, `Long_sharp_wave_ripple_detector`; edge rule and FIR rewritten in prose/reflect-convolution (licensing), upfront length check, `*_local_percentile` output columns, odd `smoothing_window` enforced | merged after fixes; no discrepancy vs the MATLAB |
| #17 | `Carey_candidate_detector` | merged as-is; no discrepancy vs the MATLAB |
| #19 | README "Choosing a detector" table, docstring NaN policies, `__all__`, changelog regroup, Carey theta filter order | merged |
| #20 | Correctness pass after four more reviews: per-event speed lookup, tap count scaled with the sampling rate, close-event chaining, unified duration and immobility rules, Kay trace across gaps, Long `max_thresh`, verified DOIs, American spelling | merged |

Resolved in #20: duration limits are one inclusive round-half-up sample-count rule
(`sample_count_within`) and immobility is `speed <= speed_threshold` everywhere; the
close-event chaining bug is fixed. Still open: `close_event_threshold` (the multiunit
detector) versus `close_ripple_threshold` (everything else, and the name Spyglass
hard-codes), which needs a deprecation alias rather than a rename.

**#20 changes event counts.** The filter tap count, the duration rule, and the immobility
comparison all move marginal events. Anything detected with an earlier commit must be
re-run before it is compared with output from this one.

Standing decisions for package work (from the user, this session):

- No artifact, EMG, or noise-channel exclusion logic in the package. Artifact handling is
  the project's job (03 lists the reference-channel control as deferred).
- Detectors are named after the algorithm's author and cite the paper and the lab code
  they follow in the docstring and changelog. GPL (FMAToolbox) and unlicensed (neurocode)
  code is reimplemented from the algorithm, never transcribed.
- Yu noise threshold is estimated **per epoch**, not per day as the lab code did; a
  histogram mode above zero raises a warning, not an error.
- Package code is written test-first; commits are gated on the pytest exit code.

## Phase 1 details not in the spec

- **Direct computation, decided 2026-10-02.** Spyglass is not altered until Phase 7: no
  registration line, no `RippleParameters` row, no populate. Phase 1 reads inputs through
  the same read-only fetches `get_ripple` already makes (`LFPBandV1` for the CA1 ripple
  band, `RippleLFPSelection` electrodes, head speed) and calls the pinned package
  functions on them. Spec 01 §Producing the inventory carries the direct step list; its
  former Spyglass steps are Phase 7.
- Pinning the package directly is still needed (the project env has 1.7.0, transitively
  through Spyglass, which has no `Yu_ripple_detector`). Spyglass's `ripple/v1/ripple.py`
  imports only `Karlsson_ripple_detector`, `Kay_ripple_detector`, `core.gaussian_smooth`
  and `core.get_envelope`, all still exported on package `master`, so the pin does not
  break the existing Kay reads.
- The project's `get_multiunit_hse` call dropped `use_speed_threshold_for_zscore=False`
  (200c94c) because ripple_detection 2.0 removes it; on 1.7.0 the default is the same
  `False`, so it is behavior-neutral there.
- The original lab code is vendored byte-exact under `reference/yu_2017_franklab/` with
  its provenance README; the spec's discrepancy table is derived from it.
- Recording condition: the published method assumes ≥3 CA1 tetrodes; record
  `meets_published_tetrode_criterion` as `true`/`false`/`unknown` per epoch.

## Open decisions

| # | question | current position |
|---|---|---|
| D1 | Do Zugaro, Long, and Carey get project inventories, or stay as package-only comparison rows? | **Resolved, 2026-10-17.** Zugaro is inventory 7 (2026-10-03, refresh design R3), computed directly by `scripts/decoding/run_swr_detection.py` on one CA1 channel per epoch, chosen by buzcode's `bz_GetBestRippleChan` rule (overview, "Inventory 7"). **Long is dropped:** it needs a stratum radiatum channel, and every tetrode targets stratum pyramidale (`docs/task_description.md:159-161`). **Carey is inventory 8:** `Carey_candidate_detector` (PR #17; the project now pins `9766fd4`; the SWR inventories were computed at `68e1784`) at the package default, the code's Hilbert ripple score, on the same shared inputs and speed rule as the other SWR detectors (4 cm/s, the artifact mask, rule R-T for the LFP) and the decode's marks for the multiunit trains. Four adaptations: each tetrode's multiunit train is one "unit" (for `minimum_active_units` and the per-unit `spike_cap`), a coarser count than sorted cells; the Hilbert configuration, not the published spectral one, which needs hand-picked ripples this dataset does not have; the stored 150-250 Hz band averaged over the R-T kept tetrodes, not one 140-250 Hz channel; and no theta restriction. The lead settled the input choices (2026-10-17): the LFP's grid, the package's `normalization_mask` for the artifact mask (`9766fd4`, pinned from `e1decf2`), and Carey as a fifth SWR definition in the SWR label matrix and the triggered set. |
| D2 | Which `ripple_detection` commit to pin | **Reopened 2026-10-02.** Package `master` is now 68e1784, 309 commits past 7245c16 (literature methods, `save_events`/`load_events` that keep the detecting version, a dataset catalog). Latest tag is still v1.7.1. Pin one tagged commit for Phase 1 and record it in each inventory's manifest; tag a release first if possible. |
| D3 | Spyglass PR for the registration line? | Moved into Phase 7 (final product). |
| D7 | Inventory 1 (Kay): keep the canonical `SWR_event` populated under the old package, or also recompute Kay directly with the pinned package? | **Closed, 2026-10-17.** Inventory 1 is the directly computed Kay inventory (run `056eee0dcb5a`), which is now the paper's SWR source (`swr_source.py`). The canonical Spyglass `SWR_event` is its version check (the SWR runner's `kay_version_check`), not an inventory. |
| D4 | Widloski adaptation choices (02 §deviations) | Documented in 02; not re-decided here. |
| D5 | Add the reference's `+1e-3` Hz floor to the sorted path of the sweep and figure? | **Yes, done 2026-09-25.** The sliding window decodes the one fitted encoding model with the floor (`with_reference_rate_floor`); the state-space detector reads it as fitted. Recorded data, decoded clusterless, have no equivalent (02 V3). |
| D8 | The event features' baselines are matched on duration and location, not on spike count: keep the baseline comparison? | **Decided by the lead, 2026-10-18: dropped from Phase 4.** Reason: the windows were matched on duration and location but not on spike count, and the per-row remote maximum grows with spikes, so "above baseline" mostly measured engagement (on the pilot the median spikes per row is 1.10 for events and 0.26 for baseline windows, and corr(evidence_dll, spikes per row) is 0.84). Phase 4 reports the evidence score and its spike counts per population, with no baseline. `selection.baseline_pool`, `window_room`, `baseline_draws` and `matched_baselines` stay, with their tests, for a later design matched on spike count. |
| D9 | Non-local events keep runs of at least 7 samples (12 ms span), not 8: `minimum_sample_count` rounds 15 ms down on the quantized 500 Hz grid (1.1 % of state-space events; SWR and MUA detectors unaffected). Fix now? | **Decided by the lead, 2026-10-18: keep for now; fix at the next full rebuild**, since the fix renumbers the non-local labels and needs `data_over_time`, the event table set and the refits rebuilt together. See [nonlocal_minimum_duration_rounding.md](../../../../docs/known_issues/nonlocal_minimum_duration_rounding.md). |
| D10 | Carey runs with no theta restriction (adaptation 4), though Carey et al. 2019 exclude events during high theta (6–10 Hz power, z < 2 over the whole record; `GenCandidateEvents.m:199-224`). Add the theta gate? | **Decided by the lead, 2026-10-19: add it at the next full rebuild**, with the rerun of Carey, the Phase 3 inventory comparison and the Phase 4 event features that it needs. Input: the raw LFP of the corpus callosum tetrode the project already uses for theta (the first electrode of the `theta_5_11` band, `get_theta`; ground-referenced, `reference_elect_id = -1`), passed as `theta_lfp` to `Carey_candidate_detector` at its defaults (6–10 Hz, `theta_threshold` 2). It is a Carey-specific rule; the shared speed, immobility and artifact rules are unchanged. Report any state-space events it turns into ours-only against Carey. |
| D6 | Run the running-only encoding refit for both procedures on the pilot, as the 2026-09-25 review asked? | **No, decided 2026-09-25.** 02 assumption 8 stays as accepted on 2026-09-22: the `no_ripple` asymmetry is shared by both cohorts, revisited only if a non-SWR fraction becomes a headline claim. |

## Branch state (project repo)

Phases 1-3 are on `master` (at 25b413d): Phase 1's inventories come from
`scripts/decoding/run_swr_detection.py`, Phase 2's code from `memoryless-sequence-detection`, and
Phase 3 from `replay-detection-phase3`, which `master` has reached. Phase 4's code is on
`replay-detection-phase4`, from 25b413d, in the worktree `/cumulus/edeno/continuum-swr-replay-phase3`;
it is not merged.

**Scope rule (2026-10-02):** this branch does not edit the Figure 1–4 scripts or other
paper-figure scripts. A `ruff format` sweep and a lazy-import change had touched 21 of
them (6221117, 86d2a92); they were restored to `master`. Format or fix those on their
own branch.

Phase 2 pilot, on the machine that has the decodes and Spyglass. The epoch the plan first
named, `chimi20200212_02_r1_HPC`, has no decode here. Pilot epochs (chosen 2026-10-02: per
animal, the canonical decode whose non-local event count and SWR fraction sit closest to
the 218-epoch medians of 231 events and 0.42; tetrode counts from the saved encoding
model):

| epoch | events | SWR fraction | tetrodes | role |
|---|---|---|---|---|
| `chimi20200225_02_r1_HPC` | 248 | 0.39 | 22 | **first pilot** |
| `wilbur20210329_12_r6_HPC` | 221 | 0.43 | 13 | low tetrode count, for the null check |
| `j1620210707_06_r3_HPC` | 209 | 0.41 | 22 | other animals, for the null check |
| `peanut20201202_10_r5_HPC` | 202 | 0.41 | 20 | |
| `senor20201101_06_r3_HPC` | 229 | 0.43 | 21 | |

**Decode-time marks (2026-10-02).** The first pilot attempt failed the
encoding-mark check: the decodes (2026-04-01, 0820d39d) predate 64bf585, which
made `detect_coincident_spikes` count each cluster's leading spike, so
`load_data` now drops 259 more of chimi20200225's ~358k marks than the decode
saw. The runner now rebuilds the decode's marks from raw `UnitMarks` with the
0820d39d rule (`memoryless_sequence/decode_marks.py`), bounded to the decode's
span; on chimi20200225_02_r1 and wilbur20210329_12_r6 the kept indices equal
the 0820d39d code's and every encoding mark is found. `load_data` keeps the
corrected rule.

**First pilot run (2026-10-02, 6d03535, `--no-shuffle`, run
`6d035353e0d7-seed20260922-noshuffle`):** chimi20200225_02_r1, 2 min, 7.5 GB
peak RSS, 295,369 windows, 198 candidates, 4 meeting duration > 100 ms and
dispersion > 12 cm (2 SWR, 2 not). Median candidate dispersion 1.1 cm: most
candidates are stationary, and only 38 of 198 overlap one of our 248
non-local events (41 of ours overlap a candidate). Candidate 28 (dispersion 18.6 cm,
2 cm path) was checked the same day: not a port defect. Dispersion is on the
track graph; it counts a kept window inside the merged gap that was dropped as
a single-window run, exactly as the reference does (`load_replayEvents_cm.m`
takes `compute_sequenceDispersion` over the merged span of `x_NaN`, where a
dropped singleton is not NaN; spec 02 §Merged gaps). That one window decodes
108 cm away; from the pieces alone dispersion is 0.6 cm. 2 of the 4 replay
calls depend on such a window (the other at 102 cm, pieces-only 1.7 cm), and
36 of 198 candidates have one. This is a property of the published definition,
kept as the default; it needs a diagnostic column (pieces-only dispersion,
count of out-of-piece windows) for Phase 4, and the port's `path_distance_cm`,
which counts pieces and bridges only, does not describe the same windows as
`dispersion_cm`. Both diagnostics were added the same day (df6f48f); rerun
`df6f48fb2873-seed20260922-noshuffle` reproduces 0.6 / 1.7 cm and 36 of 198.
Between that rerun and the first, every column but `run_id` is identical except
`mean_posterior_spread_cm`, which moves by at most 1.5e-5 cm (3e-6 relative) on
43 rows: GPU run-to-run variation in the likelihood, below any decision.

**Shuffle pilot (2026-10-02, 0c2548c, run `0c2548cacf2d-seed20260922`, 100
draws):** 4 min 48 s wall (vs 2 min without), 7.4 GB peak RSS, so about 42 s
per tested candidate on one A100. All 4 replay calls are testable and all 4
pass (p_spread / p_jump 0.01-0.03; 0.0099 is the floor at 100 draws),
including both whose dispersion rests on one window outside their pieces
(candidates 28 and 195): the shuffle does not catch that pattern. All 4 are
whole-event speed eligible; 2 overlap a speed-qualified SWR.

**Five-epoch pilot (2026-10-02, shuffle, seed 20260922; chimi at 0c2548c, the
other four at 6f79548 in one 25 min 47 s run, 11.4 GB peak RSS):**

| epoch | candidates | with a stray kept window | replay | replay resting on it | pass shuffle | replay overlapping SWR |
|---|---|---|---|---|---|---|
| chimi20200225_02_r1 | 198 | 36 | 4 | 2 | 4 | 2 |
| wilbur20210329_12_r6 | 169 | 21 | 5 | 0 | 5 | 5 |
| j1620210707_06_r3 | 126 | 11 | 8 | 0 | 6 | 8 |
| peanut20201202_10_r5 | 268 | 21 | 2 | 0 | 2 | 2 |
| senor20201101_06_r3 | 648 | 55 | 9 | 0 | 9 | 9 |
| total | 1409 | 144 | 28 | 2 | 26 | 26 |

The stray-window replay call is chimi-only on this pilot (2 of 28 calls).
Every replay call is whole-event speed eligible. A spike-level reproduction in
the sorted 25-cell simulation (about 1,200 trials of remote bursts and of
remote spikes inside a one-window silence) produced no stray kept window: the
jump gate needs the next window broad yet centered on the same remote place,
which the simulator's likelihood rarely gives. A hand-placed construction does
reproduce it (`hand_placed.py`): two remote spikes alone in window W, a
local/remote tie in W-1, two flanking remote spikes in W+1 (broad, centered
2 cm from W), and local firing either side. The event is called replay at
16.5 cm dispersion against 3.0 cm from its pieces; the spread half of the
shuffle passes (p = 0.01) and the jump half does not (p = 0.12), unlike
chimi's candidate 28 (0.02 / 0.03). Figure beside candidate 28:
`stray_window_outputs/stray_window_figure.png` (`plot_stray_window.py`). The
research directory is gitignored, so these stay local.

**The construction as a figure example (2026-10-03, 53b4b72 then 5db999a).** The
builder is `stray_window_spikes` in `memoryless_sequence/simulation.py`, and the
event is panel f of `generate_memoryless_comparison.py`, placed at 31.40 s (the
stillness before the inbound replay; placed before the outbound replay it
flipped that replay's verdict) and written into the decoded spikes only, so the
encoding model is fitted as before. The first version (five local cells,
142-170 cm) decoded its still parts at about 155 cm against the animal's 170 cm,
and the state-space detector read 124 of its 201 ms as non-local. The current
version carries the still parts with the two cells nearest the animal for 120
ms either side: 99% of the kept windows decode within 15 cm of the animal, the
state-space detector reports no non-local time, and dispersion is 14.4 cm with
the far window against 0.5 cm without. It fails the shuffle on both halves
(p = 0.41 spread, 0.32 jump). A scratch sweep of 25+ local-cell variants found
none that both decodes at the animal and passes the jump half: with only the
170 and 163 cm cells covering the animal's place, about one shuffle draw in five
puts the pair close together and the shuffled event is as still as the real one
(read from individual draws; not a caption claim). Variants with cells further
back pass the spread half but move the decode off the animal, and the
state-space detector then reports up to 169 ms non-local. On recorded data,
where many cells fire, both stray-window calls passed the shuffle.

```bash
python scripts/decoding/run_memoryless_detection.py --run-seed 20260922 \
    --epoch-id chimi20200225_02_r1_HPC --no-shuffle   # detector alone, fast
python scripts/decoding/run_memoryless_detection.py --run-seed 20260922 \
    --epoch-id chimi20200225_02_r1_HPC                # with the shuffle
```

Measure the runtime of the second before running an animal: the shuffle re-decodes each
tested candidate's spikes 100 times.

**Reference-output check rerun (2026-10-03, c045526, 26 min 39 s, 1.8 GB):** every
sweep decision count equals the 2026-09-25 baseline (17,632 candidates, 2205 / 2198
replay, 1949 / 1945 published, 17 verdicts differing on the same windows, 34 cells
differing). One shuffled zero-jump pair of 29.2 million now differs (was 0), untraced;
the run is on a different machine. Figure 1 measures moved only because the figure
gained the stray-window event (replay 3 → 4 in both port and reference, published 2 / 2,
every gate still agreeing with the floor). `reference_check_summary.csv` replaced.

**Overlap with SWRs and our non-local events (2026-10-03, five pilot epochs,
`.claude/research/memoryless_overlap/overlap.py`, local).** Strict overlap; SWRs speed
gated as the runner's; ours are the canonical table's speed-eligible events. Of 1,409
candidates, 179 overlap an SWR, 202 one of ours, 1,170 neither. All 28 replay calls
(duration + dispersion) overlap one of ours and 26 an SWR; the 26 shuffle passes, all
ours, 24 SWR. Coverage the other way: replay calls touch 27 of 781 SWRs and 30 of 1,109
of our events (27 of 458 SWR-overlapping, 3 of 651 not); even all candidates touch only
204 of ours (59 of the non-SWR ones). Chimi's two stray-window calls are its two SWR
calls; its two non-SWR calls have pieces-only dispersion 14.9 and 24.0 cm.

Checks the pilot carries, from the 2026-09-25 review (its running-only encoding control
was declined, D6). None needs the full dataset; all
come before a reproduction or superiority claim, not before the descriptive comparison.
**Deferred by the lead (2026-10-17):** checks 1 and 2 wait; Phase 3 runs without them, and
no claim about the Widloski column is made until they are done:

1. **Validate the encoding-pair null** (02 D2, assumption 7). On a few pilot epochs from
   different animals and tetrode counts, inject known moving sequences, stationary
   events, and incoherent activity (marks scrambled against spike times), and run the
   detector with the shuffle. Inspect both null statistics, spread and jump, for each,
   and report the cohorts before and after the shuffle. On simulations the jump half
   fails real moving trajectories against a broad null (02 §"What the encoding-pair test
   does on simulated data"); measure how much position a tetrode's identity carries. A
   failure here is a failure of the adapted null, not of the published algorithm, and is
   reported as such.
2. **Two adaptation sensitivities, not a search** (02 V9 and the singleton convention):
   the spread cutoff at 10 cm and at 20 cm (the reference's only 1D precedent,
   `load_replayEvents_cm.m:32`), and `--keep-single-window-runs`. Run on the simulation
   sweep and the pilot epoch; report whether conclusions change. Do not tune either
   setting toward agreement with our detector.
3. **Reference-output check stays green.** Rerun
   `scripts/qa/compare_memoryless_to_reference.py` after any change to the detector or
   the shuffle and compare against `reference_check_summary.csv`.
