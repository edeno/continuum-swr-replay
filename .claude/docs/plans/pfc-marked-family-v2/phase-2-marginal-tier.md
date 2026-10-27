# Phase 2 — Marginal tier: every target fit alone

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Why: the joint models report each mark's contribution after every other mark in the model has taken what it can, so shared variance is credited to no one. The content family reports "total" beside "direct" for the same reason; this tier is the family's total.

**Inputs to read first:**

- [src/continuum_swr_replay/analysis/pfc/marked_model_family.py:135-190](../../../../src/continuum_swr_replay/analysis/pfc/marked_model_family.py#L135) — `MarkedFamilySpec` (`tier` field already exists with values `core` and `interaction`).
- [src/continuum_swr_replay/analysis/pfc/marked_model_family.py:195-390](../../../../src/continuum_swr_replay/analysis/pfc/marked_model_family.py#L195) — the specs after phase 1b.
- [scripts/pfc/run_pfc_marked_model_family.py:800-907](../../../../scripts/pfc/run_pfc_marked_model_family.py#L800) — `--models` and `--include-interactions` handling; artifact naming in `family_artifact_path`.
- [scripts/pfc/summarize_pfc_marked_family.py:324-385](../../../../scripts/pfc/summarize_pfc_marked_family.py#L324) — `build_report`; one table per model today.

**Contracts referenced:**

- [Fit contract version 2](shared-contracts.md#fit-contract-version-2) — the `tier` key.
- [Bundle directory suffixes](shared-contracts.md#bundle-directory-suffixes) — marginal artifact names.

## Tasks

- **Generated marginal specs.** Add `marginal_specs() -> Mapping[str, MarkedFamilySpec]` in `marked_model_family.py`: for every distinct target mark across `CORE_MODEL_NAMES`, one spec named `marginal_<mark>` with `scientific_marks=(mark,)`, `tier="marginal"`, and adjustments `(EVENT_DURATION_MARK, LEAF_CONTENT_FRACTION, DECODE_ENTROPY)` when the mark is content-weighted or a role fraction, else `(EVENT_DURATION_MARK, DECODE_ENTROPY)`. Interaction targets get a marginal spec with only their two constituents plus duration and entropy as adjustments. Export `MARGINAL_MODEL_NAMES`.

- **Runner.** `--tier {joint,marginal,all}` (default `joint`); `marginal` resolves `--models` to the generated names. `family_artifact_path` puts `marginal_<mark>` in the model slot so the name is `<epoch_identifier>_marginal_<mark>_marked_family.pkl`. The contract's `tier` key distinguishes the bundles; the skip check keys on the artifact path as now.

- **Summary.** `load_bundles` reads `tier` from the sidecar into the unit table. `build_report` adds, after the per-model tables, one "joint versus marginal" table per target mark with columns: class, joint post-onset mean, marginal post-onset mean, joint fraction gaining, marginal fraction gaining, and the joint model the target came from. `plot_model` overlays the marginal population curve as a dashed black line on the joint panel when a marginal bundle exists for that target.

- **Run** the marginal tier on the corpus (compact designs; about 4 GPU-hours on three cards), then regenerate the summary. Add a paragraph to the corpus-v2 audit naming the targets whose marginal and joint estimates differ by more than a factor of two in post-onset mean or by more than 0.05 in fraction gaining; those are the targets whose joint reading depends on the adjustment set.

- **Docs.** `docs/design/pfc_marked_model_family.md`: a "Marginal tier" subsection saying what it is, that it is the family's total effect, and that neither tier supersedes the other. `scripts/README.md`: the `--tier` flag.

## Deliberately not in this phase

- No new marks; marginal specs are generated from the phase-1b inventory.
- No permutation reference (phase 3); the marginal and joint tiers share the phase-3 machinery later.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_marginal_specs_cover_every_core_target_once` | one spec per distinct core target, `tier == "marginal"`, one scientific mark each, adjustments per the rule above. |
| `test_marginal_specs_for_interactions_keep_both_constituents` | an interaction's marginal spec has exactly its two constituents plus duration and entropy as adjustments. |
| `test_runner_tier_flag_selects_the_generated_models` | `--tier marginal` yields `MARGINAL_MODEL_NAMES`; `--tier all` yields core, interactions and marginal in that order; artifact names carry `marginal_<mark>`. |
| `test_summary_pairs_joint_and_marginal_for_each_target` | on synthetic bundles for one target in both tiers, the comparison table has one row per class with both post-onset means. |
| `test_marginal_single_target_equals_joint_when_the_model_has_one_target` | for `within_stop_timing`, whose joint model has one target and the same adjustments, joint and marginal bundles are numerically identical (`atol 1e-8`). |
| corpus marginal run (`slow`, GPU) | one bundle or marker per session × marginal spec, one contract. |

## Fixtures

Synthetic bundles from `tests/unit/test_summarize_pfc_marked_family.py` helpers, extended with a `tier` field; the eligibility fixture for the spec tests.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind).
- User-facing documentation listed as tasks is updated, not deferred.
