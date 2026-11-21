# Phase 9 — Inference: the declared family, FDR and simultaneous bands

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#session-to-day-to-animal)

Branch: `regression-inference` from master after Phases 5a, 6a and 7 (every declared estimand has session
values). CPU only.

The per-estimand summaries (estimate, interval, animals with the sign, day-level p) already exist from the
family phases. This phase adds what needs the whole family: the false discovery rate over the declared
tests, the leave-one-animal-out check, and the curve bands Figure 4 draws.

**Inputs to read first:**

- The design, section 4.5 (inference) and the "What would make this a failure" list of section 1.
- `analysis/regression/declared_tests.py` (the list, fixed before Phase 5a), `analysis/regression/bands.py`,
  `analysis/shared_statistics.py` (`estimand_summary`, `benjamini_hochberg` `:75`,
  `nested_bootstrap_draws` `:377`, `bootstrap_interval` `:456`).
- The estimand modules of Phases 5a, 6a and 7 (`visit_rate_statistics.py`, `role_content_statistics.py`)
  and their stored session values.
- `reported/behavior/visit_rate.py`, `role_content.py` (the values to extend).
- Scratch: `S/verify_rate/s5_inference.py` (leave-one-animal-out and day-level pattern),
  `S/verify_uncertainty/s07_inference.py`.

**Contracts referenced:** [C5](shared-contracts.md#c5-the-convention) rule 5,
[C8](shared-contracts.md#c8-the-declared-family-of-tests), [C10](shared-contracts.md#c10-rebuilding-reported-values-and-figures).

**Designs referenced:** [session to day to animal](designs.md#session-to-day-to-animal),
[simultaneous bands](designs.md#simultaneous-bands).

## Tasks

- **Family table** (`analysis/regression/family.py`, new): read every declared estimand's per-kind and
  difference session values from the stored records (one function per family already returns them), apply
  `estimand_summary`, and compute Benjamini-Hochberg q-values over the day-level p-values of the whole
  family (all members times SWR, No SWR and difference, as decided in L1). A missing p is left out of the
  family (`benjamini_hochberg`'s rule) and listed. The output is one table (estimand, kind, estimate,
  interval, animals with sign, day p, q) written to `models_convention/family.parquet`.
- **Leave-one-animal-out** for each member: the estimate and day-level p without each animal; stored beside
  the family table and reported as a range (no claim rests on it; it shows dependence on one animal).
- **Bands for every Figure 4 curve** (`analysis/regression/bands.py` callers in the family modules): the
  animal-balanced curve with its simultaneous band from nested bootstrap draws for `f_s`, `f_u` (and `f_m`
  if kept), model D's lag curve, the recency curve and the current-arm `f_u`, per kind and for the
  difference; stored as curves (grid, estimate, pointwise and simultaneous bounds) in
  `models_convention/curves.parquet`.
- **Coverage check of the bands** (stored analysis, `scripts/checks/simulate_band_coverage.py`, new):
  simulated sessions nested in five animals with between-animal variation; the simultaneous band of the
  animal-balanced curve covers the true population curve at the stated rate. If it does not, report and
  stop; the band construction is a call ([designs.md#simultaneous-bands](designs.md#simultaneous-bands)).
- **Summary values**: add the q-value and the leave-one-animal-out range to each declared value's group in
  `reported/behavior/`; descriptions say "q over the declared family of N tests" with N read from
  `declared_tests.py`. Curves are figure inputs, not values.
- **Docs**: a short section in `docs/behavior_models.md` (or `docs/rate_models.md`) on the declared family,
  its members, and the reduction; with value markers for the counts.

## Deliberately not in this phase

- The pooled model with grouping terms (stage 3), even if a member is borderline; that is the lead's call.
- Changing the declared list. If a member cannot be computed (support or collinearity), it stays in the
  family with a missing p and is listed; the family is not re-declared after the fits.
- Figure 4 (Phase 10).

## Validation slice

| Test | Asserts |
| --- | --- |
| `tests/analysis/regression/test_family.py::test_q_values_over_whole_family` | on a hand table of p-values the q-values equal `benjamini_hochberg` over the full list; a missing p is left out and listed |
| `test_family.py::test_family_matches_declared_list` | the table holds exactly the declared members times three |
| `test_family.py::test_null_family_false_discovery` (slow) | on simulated sessions with every member null, the share of q < 0.05 is at most 5 % over 200 replicates |
| `test_bands.py::test_animal_balanced_band_coverage` (slow) | the across-animal band covers the population curve at 93-97 % on simulated nested data |
| `tests/reported/behavior/test_family_values.py::test_q_and_loao_values` | each declared value group gains its q and its leave-one-animal-out range |

## Fixtures

Synthetic nested data (animals, days, sessions) with fixed seeds in `tests/analysis/regression/conftest.py`.

## Verification and runtime

- Spot-check two members by hand from the stored session values (recompute the day values and the signed-rank
  p in a notebook cell; they must match the table).
- Check the q-values are not driven by duplicates: one row per (member, kind) in the family table.
- The bootstrap draws for the bands are the long part (10,000 draws per curve); time one curve and
  extrapolate.

## Rebuilt

`models_convention/family.parquet`, `curves.parquet`, the coverage simulation; `paper/reported/behavior.json`
and the emitted files.

## Exit criteria

Every declared member has its summary and q-value (or a listed reason it has none); bands exist for every
curve Figure 4 draws, and their coverage check passes.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind).
- User-facing documentation listed as tasks is updated, not deferred.
