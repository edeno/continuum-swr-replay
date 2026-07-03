# Repository Reorganization Implementation Plan

**Status:** Not started.

Reorganize the repo so every artifact type has exactly one home: published docs in
`docs/`, design specs / plans / reviews under `.claude/docs/`, exploratory scratch
in `.claude/research/`, model definitions consolidated in `model_specs.py`, and
`scripts/` grouped into domain subdirectories with an index. This is a physical
reorganization (files move, all references are fixed), executed one shippable PR at
a time so the repo builds and tests pass at every phase boundary.

## Reading order

For agent invocation, **load only the slice you need**:

1. **Working a specific phase?** Open the matching phase file — each is self-contained.
2. **Need the target directory structure / the "one home per type" rules?** [overview.md](overview.md#target-structure) — the canonical spec every phase implements a slice of.
3. **Need broader scope / risks / the move-safety findings?** [overview.md](overview.md).

## Files

- [overview.md](overview.md) — target structure, move-safety analysis, risks, verification
- Phases (each ships as a separable PR; recommended order, but 1/2 are independent of each other):
  - [phase-1-docs.md](phase-1-docs.md) — design specs → `docs/design/` (human-discoverable); historical plans + reviews → `.claude/docs/`; add `docs/README.md`
  - [phase-2-model-specs.md](phase-2-model-specs.md) — move the `is_swr`-contrast ladders from the runner scripts into `model_specs.py`
  - [phase-3-scripts.md](phase-3-scripts.md) — group `scripts/` into domain subdirs + `exploratory/`; add `scripts/README.md`; fix all path/idiom/test/config references
  - [phase-4-figures.md](phase-4-figures.md) — file the 27 loose `paper/figures/` outputs into topic subdirs; move exploratory figure output to `.claude/research/`
  - [phase-5-data-dirs.md](phase-5-data-dirs.md) — rename the cryptic `Processed-Data/…/models*/` dirs to descriptive names + un-mix; update all readers/writers
  - [phase-6-outputs-claudemd.md](phase-6-outputs-claudemd.md) — document the results + figure output conventions; final CLAUDE.md "Project Structure" rewrite + straggler sweep
