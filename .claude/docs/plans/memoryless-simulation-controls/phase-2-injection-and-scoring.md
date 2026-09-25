# Phase 2 — General injector, builders, multi-part scoring

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#general-injector)

**Inputs to read first:**

- `src/continuum_swr_replay/analysis/hippocampus/memoryless_sequence/simulation.py:34-160`
  — `InjectedReplay` and `inject_replay`.
- `src/continuum_swr_replay/analysis/hippocampus/memoryless_sequence/scoring.py` —
  `score_against_injection`, `best_scored`.
- `tests/unit/test_memoryless_simulation.py`, `tests/unit/test_memoryless_scoring.py`.

**Contracts referenced:** [Truth labels](shared-contracts.md#truth-labels).

**Designs referenced:** [General injector](designs.md#general-injector),
[Builders](designs.md#builders), [Multi-part scoring](designs.md#multi-part-scoring).

## Tasks

- **Baseline**: for three fixed seeds and three replay shapes, save `inject_replay`
  outputs (spike arrays and bounds) from the current code.
- **General injector**: add `InjectedActivity` and `inject_activity`; rebuild
  `inject_replay` on them; verify bit-identical outputs against the baseline, then turn
  the comparison into a unit test with hard-coded expectations from a tiny case.
- **Builders**: `field_rates`, `piecewise_replay`, `placeless_burst`, `jumping_burst`,
  each returning `InjectedActivity` with its `parts` and `truth`. Export them.
- **Scoring**: NaN-in-gap support in `score_against_injection`; `recovered_parts`;
  `segmentation`; a helper giving onset/offset errors for a multi-part event per the
  design.
- **Docs**: module docstrings of `simulation.py` and `scoring.py` describe the new
  conditions and the multi-part rule.

## Deliberately not in this phase

- Any script or figure change: the controls sweep that uses these is phase 3.

## Validation slice

| Test | Asserts |
| --- | --- |
| `inject_replay` unchanged | Same spikes and bounds as the recorded baseline for fixed seeds |
| piecewise replay | A two-segment jump represents the first segment, then jumps J cm with no gap; a chain is silent in its gap (no injected spikes from any cell there) |
| place-less burst | Every cell has one rate; expected population count matches a mid-track replay of the same duration and peak rate within 1% |
| jumping burst | Position is piecewise constant over 15 ms holds, uniform on the track over many seeds |
| NaN-gap scoring | Decoded positions in a gap are ignored for error but count in span; single-part scores unchanged on the existing tests |
| `recovered_parts` | Half-duration overlap boundary: 49% not recovered, 51% recovered |
| `segmentation` | One spanning event, one per part, a missing part, and a part split in two give the four labels |

Mutation-check the scoring tests (python -B): break the half-duration threshold, the
spanning test, and the NaN mask; each must fail a test.

## Fixtures

Hand-built spike trains and event bounds; `place_field_means(25, 170)` for builders.

## Review

Before merging, dispatch `code-reviewer` (read-only) against the diff. Confirm the
injector is bit-identical, the tests are not tautological, and nothing references this
plan.
