# Readability Decomposition Implementation Plan

**Status:** Complete. Phases 1 (`15b5e92`), 2 (`9f63352`), 3 (`540b039`), 4 (`945e815`+`8593086`) done; qvals peel (`f7bf001`) + review fixes (`b8b20c3`) applied. Phase 5 consolidating byte-diff **byte-identical across all 5 canonical tables** vs `_golden_baseline_v5_v4cache` (218-session `--from-cache` regen, 0 errors). Note: that regen ran on the phase-1–4b tree; the later qvals peel is verbatim function motion covered by the builder-parity suite, not re-regenerated.

Finish the readability work the `package-readability-refactor` plan started: break the two surviving god-functions (`build_event_table`, the content-fit `main()`) into readable stage pipelines, split the oversized `history.py` into a concern-per-module package, and rename the internal functions whose names still lie. A scientist should be able to open the code and trace raw data → tables → covariates → fits without fighting 500-line functions and misnamed helpers. No science, no persisted column names, and no published paths change; every numerical-path phase is output-preserving.

## Reading order

For agent invocation, **load only the slice you need**:

1. **Working a specific phase?** Open the matching phase file. Each is self-contained.
2. **Need the `history/` package API or the stage-helper conventions?** [shared-contracts.md](shared-contracts.md).
3. **Need scope / risks / the between-phase review policy / verification?** [overview.md](overview.md).

## Files

- [overview.md](overview.md) — scope, integration points, verification policy, the between-phase review gate, risks.
- [shared-contracts.md](shared-contracts.md) — the `history/` re-export API and the stage-function conventions.
- Phases (each ships as a separable PR):
  - [phase-1-honest-names.md](phase-1-honest-names.md) — rename lying internal functions; annotate the interaction block.
  - [phase-2-content-fit-decomp.md](phase-2-content-fit-decomp.md) — content-fit `main()` into named stages.
  - [phase-3-event-table-pipeline.md](phase-3-event-table-pipeline.md) — `build_event_table` into a stage pipeline.
  - [phase-4-history-split.md](phase-4-history-split.md) — split `history.py` by concern; unify the reward-rate functions.
  - [phase-5-consolidating-byte-diff.md](phase-5-consolidating-byte-diff.md) — one regen byte-diff confirming the whole decomposition preserved output.
