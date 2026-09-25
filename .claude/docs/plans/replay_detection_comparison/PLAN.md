# Replay detection comparison — implementation plan

**Status (2026-09-22):** Phase 0 done. Phase 2 implemented and tested, including the
shuffle, the event table and a per-epoch runner, but not yet run on recorded data: the
decodes are not on the development machine. Phase 3's accounting is implemented and
tested on synthetic inventories; its recorded-data side waits on Phases 1 and 2. Phases 1
and 4 not started. Phase 5 deferred, Phase 6 parked.

The scientific design lives in
[docs/design/replay_detection_comparison/](../../../docs/design/replay_detection_comparison/)
(overview + specs 01–05 + the detector table). This file is the *process* record: what is
built, where it lives, what order the remaining work goes in, and which decisions are still
open. It does not restate the design; when the two disagree, the spec wins and this file
is stale.

## Reading order

1. **Picking up the next phase?** Read the phase row below, then the spec it points at.
2. **Need the governing rule (harmonize substrate, not method internals) or the
   cross-cutting decisions X1–X11?** [overview.md](../../../docs/design/replay_detection_comparison/overview.md).
3. **Need to know what a detector does relative to the others?**
   [detector_comparison.md](../../../docs/design/replay_detection_comparison/detector_comparison.md).

## Phases

| phase | spec | deliverable | status |
|---|---|---|---|
| 0 | [01](../../../docs/design/replay_detection_comparison/01-yu-swr-detector.md) §Architecture, [table](../../../docs/design/replay_detection_comparison/detector_comparison.md) | Detectors implemented in `ripple_detection` with tests, README, changelog | **done** 2026-09-21: PRs #14–#19 merged to package `master` (b0431a4) after independent review |
| 1 | [01](../../../docs/design/replay_detection_comparison/01-yu-swr-detector.md) §Producing the inventory, steps 2–6 | Yu SWR inventory for every epoch via Spyglass: pin the package, register the algorithm, `RippleParameters` row `yu_2017`, populate, `get_ripple("Yu")`, Y5 speed gate, observation-availability table | not started |
| 2 | [02](../../../docs/design/replay_detection_comparison/02-memoryless-sequence.md) | Memoryless sliding-window sequence detector on the clusterless posterior; shuffle null; adapted cohort before/after shuffle | code done and independently reviewed twice, review fixes in (2026-09-22); the sweep and figure apply the paper's place-cell identity shuffle, with the reference's conventions where they carry over (zero jumps set aside, strict-below count, p ≤ 0.05; a third review found the port's "conservative" departures were manufacturing the by-duration jump trend, and they were reverted 2026-09-22). A fourth review (2026-09-25) led to the sorted zero-jump fix and a check against the reference's own arithmetic (02 §"Against the reference's own arithmetic"): the port lacked the reference's `+1e-3` Hz rate-map floor, which moved the significance-test pass rate from 88% to 69% on the sweep. The sorted path of the sweep and figure now decodes with it (D5), and the port then decides as the reference does on 99% of candidates; the published-definition rate reads 18% (was 11%). The port stays a documented adaptation, not a validated reproduction: the clusterless null used on recorded data is unchecked, and claims of reproduction or detector superiority wait for the pilot checks below. **Next: real-data pilot**, with the checks under "Phase 2 pilot". 02 assumption 8 (non-SWR events decoded partly from their own spikes) stays accepted; see D6 |
| 3 | [03](../../../docs/design/replay_detection_comparison/03-inventory-comparison.md) | Six inventories on the common schema, observation-availability tables, all-pairs overlap table, single/no-detector cells, SWR label matrix, sampled examples | accounting done 2026-09-22 in `analysis/inventory_comparison/` (`intervals`, `schema`, `availability`, `overlap`, `components`, `recount`, `cells`, `swr_matrix`), test-first with mutation checks and an end-to-end simulated epoch; **remaining, needs recorded data:** per-detector adapters (Kay/Karlsson/Yu from `RippleTimesV1`, MUA HSE, the canonical classified table, the memoryless table) that also build each detector's coverage from its inputs, a runner writing the tables per epoch, per-session/animal reporting, and the sampled examples. An independent review (2026-09-22) found cross-epoch key collisions, a partial-observation SWR label, float-rounded recount gaps, window support off by one sample, and dropped missing-run rows, all fixed test-first. Still open from it: excluded time by reason (the `observation_table` reasons are not yet joined to `Coverage`); an event ending between the last still and first moving sample is speed-eligible yet outside exposure, so it reads unknown; throughput at 221 epochs x 6 detectors unmeasured |
| 4 | [04](../../../docs/design/replay_detection_comparison/04-event-features.md) §Initial report | Event features, common spike-evidence diagnostic, boundary and rejection diagnostics, at default settings | not started |
| 5 | [04](../../../docs/design/replay_detection_comparison/04-event-features.md) §Planned follow-up | Threshold robustness: cost pilot, then one-at-a-time relax/tighten | deferred until Phase 4 is reported |
| 6 | [05](../../../docs/design/replay_detection_comparison/05-validation.md) | Simulation or held-out validation | parked; reopen only for a performance claim |

Order: 1 and 2 are independent and can proceed in parallel; 3 needs both; 4 needs 3.
Karlsson (inventory 2) needs only a Spyglass populate with the existing
`Karlsson_ripple_detector` registration and can be done alongside Phase 1.

## Phase 0: package state (`~/Documents/GitHub/ripple_detection`)

Local clone; run tests with `.venv/bin/python -m pytest -q --no-cov` (not `uv run`, which
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

- `docs/design/replay_detection_comparison/01-yu-swr-detector.md` §Producing the inventory
  is the step list. Step 1 is done. Step 3's registration line goes in
  `src/continuum_swr_replay/data_loaders/lfp.py` next to the `ripple_detector_name`
  validation.
- The original lab code is vendored byte-exact under `reference/yu_2017_franklab/` with
  its provenance README; the spec's discrepancy table is derived from it.
- Recording condition: the published method assumes ≥3 CA1 tetrodes; record
  `meets_published_tetrode_criterion` as `true`/`false`/`unknown` per epoch.

## Open decisions

| # | question | current position |
|---|---|---|
| D1 | Do Zugaro, Long, and Carey get project inventories, or stay as package-only comparison rows? | The six inventories in overview are Kay, Karlsson, Yu, MUA HSE, ours, Widloski. The three newer detectors are in the table as "in package, no run". Long needs a raw stratum-radiatum channel, which this dataset may not have; check before scheduling. |
| D2 | Which `ripple_detection` commit to pin | `master` b0431a4 (all PRs merged). No release tag yet; the changelog marks the duration change as minor-version, so a `v1.8.0` tag before pinning would be cleaner. |
| D3 | Spyglass PR for the registration line? | Optional cleanup after Phase 1 produces results (spec 01 step 6). |
| D4 | Widloski adaptation choices (02 §deviations) | Documented in 02; not re-decided here. |
| D5 | Add the reference's `+1e-3` Hz floor to the sorted path of the sweep and figure? | **Yes, done 2026-09-25.** The sliding window decodes the one fitted encoding model with the floor (`with_reference_rate_floor`); the state-space detector reads it as fitted. Recorded data, decoded clusterless, have no equivalent (02 V3). |
| D6 | Run the running-only encoding refit for both procedures on the pilot, as the 2026-09-25 review asked? | **No, decided 2026-09-25.** 02 assumption 8 stays as accepted on 2026-09-22: the `no_ripple` asymmetry is shared by both cohorts, revisited only if a non-SWR fraction becomes a headline claim. |

## Branch state (project repo)

Work is on `memoryless-sequence-detection`, ahead of `master`, no PR. Phase 2's code and
Phase 3's accounting are on it (see the phase table); Phases 1 and 4 have no code yet.

Phase 2 pilot, on the machine that has the decodes and Spyglass:

```bash
python scripts/decoding/run_memoryless_detection.py --run-seed 20260922 \
    --epoch-id chimi20200212_02_r1_HPC --no-shuffle   # detector alone, fast
python scripts/decoding/run_memoryless_detection.py --run-seed 20260922 \
    --epoch-id chimi20200212_02_r1_HPC                # with the shuffle
```

Measure the runtime of the second before running an animal: the shuffle re-decodes each
tested candidate's spikes 100 times.

Checks the pilot carries, from the 2026-09-25 review (its running-only encoding control
was declined, D6). None needs the full dataset; all
come before a reproduction or superiority claim, not before the descriptive comparison:

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
