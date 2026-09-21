# Replay detection comparison — implementation plan

**Status:** Phase 0 (detectors in the `ripple_detection` package) implemented and in
review. Phases 1–4 not started. Phase 5 deferred, Phase 6 parked.

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
| 0 | [01](../../../docs/design/replay_detection_comparison/01-yu-swr-detector.md) §Architecture, [table](../../../docs/design/replay_detection_comparison/detector_comparison.md) | Detectors implemented in `ripple_detection` with tests, README, changelog | Yu merged to package `master`; four detector/fix PRs + one CI PR open (below) |
| 1 | [01](../../../docs/design/replay_detection_comparison/01-yu-swr-detector.md) §Producing the inventory, steps 2–6 | Yu SWR inventory for every epoch via Spyglass: pin the package, register the algorithm, `RippleParameters` row `yu_2017`, populate, `get_ripple("Yu")`, Y5 speed gate, observation-availability table | not started |
| 2 | [02](../../../docs/design/replay_detection_comparison/02-memoryless-sequence.md) | Memoryless sliding-window sequence detector on the clusterless posterior; shuffle null; adapted cohort before/after shuffle | not started |
| 3 | [03](../../../docs/design/replay_detection_comparison/03-inventory-comparison.md) | Six inventories on the common schema, observation-availability tables, all-pairs overlap table, single/no-detector cells, SWR label matrix, sampled examples | not started |
| 4 | [04](../../../docs/design/replay_detection_comparison/04-event-features.md) §Initial report | Event features, common spike-evidence diagnostic, boundary and rejection diagnostics, at default settings | not started |
| 5 | [04](../../../docs/design/replay_detection_comparison/04-event-features.md) §Planned follow-up | Threshold robustness: cost pilot, then one-at-a-time relax/tighten | deferred until Phase 4 is reported |
| 6 | [05](../../../docs/design/replay_detection_comparison/05-validation.md) | Simulation or held-out validation | parked; reopen only for a performance claim |

Order: 1 and 2 are independent and can proceed in parallel; 3 needs both; 4 needs 3.
Karlsson (inventory 2) needs only a Spyglass populate with the existing
`Karlsson_ripple_detector` registration and can be done alongside Phase 1.

## Phase 0: package state (`~/Documents/GitHub/ripple_detection`)

Local clone; run tests with `.venv/bin/python -m pytest -q --no-cov` (not `uv run`, which
drops a stray `uv.lock`). Package `master` at `659cc91` carries the Yu detector
(`Yu_ripple_detector`, `get_Yu_ripple_consensus_trace`, `estimate_noise_threshold`). That
is the commit Phase 1 pins unless the PRs below are merged first, in which case pin the
merge.

| PR | branch | what | depends on |
|---|---|---|---|
| #14 | `fix/detector-conventions` | ripple-band filter designed for the actual sampling rate; `minimum_sample_count` duration convention; Karlsson statistics on the per-channel max; HSE input validation | — |
| #15 | `simulator-realism` | `simulate_LFP` ripple size in ripple-band units, per-ripple frequency/duration ranges | — |
| #16 | `zugaro-and-long-detectors` | `Zugaro_ripple_detector` (FMAToolbox FindRipples rule), `Long_sharp_wave_ripple_detector` (neurocode DetectSWR rule) | — |
| #17 | `carey-joint-detector` | `Carey_candidate_detector` (vandermeerlab GenCandidateEvents rule) | — |
| #18 | `ci-ruff-format` | Replace black with `ruff format` in CI; drop black/flake8 pins. Fixes the Code Quality job, which failed on `master` too | merge **first** |

Merge notes:

- Each PR adds bullets under `## [Unreleased]` in `CHANGELOG.md` and a section to
  `README.md`, so every merge after the first conflicts there. Resolve by keeping both
  sets of bullets; nothing else overlaps.
- After #18 merges, rebase #14–#17 to get a green Code Quality job; no code change needed.
- Codecov patch/project pass on #15–#17 after the 2026-09-21 coverage tests; only the
  Code Quality job fails, and only because those branches still run black. Rebase after #18.
- Branch `consistency-pass` (pushed, no PR) merges all five and adds the README
  "Choosing a detector" table, per-detector missing-sample notes, `__all__` with a
  public-surface test, and a regrouped changelog. Open its PR after #14–#18 merge, or
  merge it directly if the five are merged in any order first (its merge commits resolve
  the changelog/README conflicts).
- #16 is the only branch defining `_boolean_run_bounds`; `master` has `_boolean_runs`. If
  #16 lands after #17, keep one helper.

Standing decisions for package work (from the user, this session):

- No artefact, EMG, or noise-channel exclusion logic in the package. Artefact handling is
  the project's job (03 lists the reference-channel control as deferred).
- Detectors are named after the algorithm's author and cite the paper and the lab code
  they follow in the docstring and changelog. GPL (FMAToolbox) and unlicensed (neurocode)
  code is reimplemented from the algorithm, never transcribed.
- Yu noise threshold is estimated **per epoch**, not per day as the lab code did; a
  histogram mode above zero raises a warning, not an error.
- Package code is written test-first.

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
| D2 | Pin `ripple_detection` at `master` 659cc91 or wait for #14–#18? | Wait if the merges land within days; #14 changes the duration convention, which changes Yu event counts at exact-boundary durations. |
| D3 | Spyglass PR for the registration line? | Optional cleanup after Phase 1 produces results (spec 01 step 6). |
| D4 | Widloski adaptation choices (02 §deviations) | Documented in 02; not re-decided here. |

## Branch state (project repo)

Work is on `memoryless-sequence-detection`, ahead of `master`, not pushed, no PR.
Commits are documentation only (specs, table, vendored reference code); no analysis code
has been written for Phases 1–4.
