# Phase 2 — Move the `is_swr`-contrast ladders into `model_specs.py`

[← back to PLAN.md](PLAN.md) · [overview](overview.md#current-codebase-integration-points)

Restore single-source-of-truth: the rate (`G/S`) and content (`C/CP`) `is_swr`-contrast
model definitions currently live only in the runner scripts. Move the **spec** (which
predictors/formula each model uses) into `model_specs.py`; leave the **data assembly and
fitting** in the runners. Behavior-preserving — verified by smoke parity.

**Inputs to read first:**

- `scripts/fit_swr_contrast_rate.py:76-169` — the rate spec block to extract: `_SPLINE`,
  `_REWK_PREV`, `_HISTCOS`, `_BLOCK_F`, `_BLOCK_COLS`, `STOP_COVARS`, `_blocks_formula`,
  `_blocks_cols`, `_g_specs()`, `_s_specs()`, `G_SPECS`/`S_SPECS`. Everything below `_fit`
  (data load + fit) stays in the runner.
- `scripts/fit_swr_contrast_content.py` — the content spec symbols are **interleaved** with
  code that STAYS, so **move by symbol, not by a line range**: `ARMFE` (@58), `BASEc` (@59),
  `_TSC` (@110), `ARM_LADDER` (@111), `ARM_KEY_TERM` (@135), `BASEc_PATCH` (@148),
  `_LR_PATCH` (@173), `PATCH_LADDER` (@174), `PATCH_KEY_TERM` (@186) move out; the `_z`
  helper (@66),
  `_assemble_arm` (@71-109), and `_assemble_patch` (@151-173) STAY (they build the
  interaction columns the ladders name — that coupling is expected). Watch the couplings:
  `BASEc` depends on `ARMFE`; the ladders reference `BASEc`/`_TSC`/`_LR_PATCH`; the
  staying `_assemble_*` builders reference no spec symbol, but the runner's `_run`/`_summ`
  reference the ladders + key-term dicts, so re-import those.
- `src/continuum_swr_replay/analysis/replay/model_specs.py` — target module; see how the
  existing `MODEL_*` dicts and `ModelSpec` are declared and exported (`__all__`).
- `docs/models_table.md` — the is_swr section; its "single source of truth is model_specs.py"
  note becomes accurate after this phase.

## Tasks

- **Add a rate-contrast spec section to `model_specs.py`**: relocate `_SPLINE`,
  `_REWK_PREV`, `_HISTCOS`, `_BLOCK_F`, `_BLOCK_COLS`, `STOP_COVARS`, `_blocks_formula`,
  `_blocks_cols`, `_g_specs()`, `_s_specs()` verbatim; export `SWR_CONTRAST_RATE_G_SPECS`,
  `SWR_CONTRAST_RATE_S_SPECS`, `SWR_CONTRAST_RATE_STOP_COVARS` (rename `G_SPECS`/`S_SPECS`
  on the way out to avoid a bare-name clash). Add to `__all__`. Note `N_REWARD_LAGS` /
  `N_HISTORY_BASIS` are already package constants — import them, don't duplicate.
- **Add a content-contrast spec section to `model_specs.py`**: relocate `ARMFE`, `BASEc`,
  `_TSC`, `ARM_LADDER`, `ARM_KEY_TERM`, `BASEc_PATCH`, `_LR_PATCH`, `PATCH_LADDER`,
  `PATCH_KEY_TERM`;
  export as `SWR_CONTRAST_CONTENT_{ARM_LADDER,ARM_KEY_TERM,PATCH_LADDER,PATCH_KEY_TERM,
  BASEC,BASEC_PATCH}`. Add the exported names to `__all__`; `_TSC` and `_LR_PATCH` may remain
  private helpers inside `model_specs.py`.
- **Repoint the runners to import** the specs from `model_specs.py` (delete the moved
  definitions from `fit_swr_contrast_rate.py` and `fit_swr_contrast_content.py`; keep the
  local short aliases if convenient, e.g. `G_SPECS = SWR_CONTRAST_RATE_G_SPECS`). Verify no
  other symbol the runner still uses was moved out.
- **Update `docs/models_table.md`** provenance note for the is_swr section to state the
  ladders now live in `model_specs.py` (removing the implicit "defined in the runner"
  exception).

## Deliberately not in this phase

- Moving the runner files into `scripts/rate/` and `scripts/content/` — that's phase 3.
  Reconcile in place first; the package import (`from …model_specs import …`) is move-safe,
  so phase 3 relocates the thinned runners without touching the spec import.
- Any change to the model *definitions* themselves (predictors, formulas). This is a pure
  relocation; the ladders must be byte-for-byte the same.

## Validation slice

| Test | Asserts |
| --- | --- |
| `uv run pytest` | full suite green (unchanged pass count) |
| `uv run python scripts/fit_swr_contrast_rate.py --max-per-animal 1` | runs; skip tally + coefficient columns identical to a pre-change run on the same seed/data (baseline: re-run once before the edit and diff the printed summary) |
| `uv run python scripts/fit_swr_contrast_content.py --max-per-animal 2` | runs; the `_summ` table (medCoef/%pos/p per rung) matches a pre-change baseline run |
| `python -c "from continuum_swr_replay.analysis.replay.model_specs import SWR_CONTRAST_RATE_G_SPECS, SWR_CONTRAST_CONTENT_ARM_LADDER"` | new exports import cleanly |

**Baseline capture (do first, before editing):** run both smokes on the current code and
save their printed summaries; after the move, re-run and assert identical. Spec extraction
that changes a coefficient is a bug.

## Fixtures

Uses the existing `Processed-Data/replay_content/` tables (smoke reads a slice). No new
fixtures.

## Review

Before opening the PR, dispatch `code-reviewer` against the diff. Confirm:
- The runners contain **no** ladder definitions — only imports of the `SWR_CONTRAST_*`
  specs plus their own assembly/fit code.
- Both smokes reproduce their pre-change summaries (parity captured and diffed).
- `model_specs.py.__all__` exports the new names; `docs/models_table.md` provenance note is
  updated.
- Private content helpers (`_TSC`, `_LR_PATCH`) live beside the ladders they feed, so importing
  `SWR_CONTRAST_CONTENT_PATCH_LADDER` cannot fail on a missing helper.
- No duplicated constants (`N_REWARD_LAGS` etc. imported, not re-declared).
- Docstrings / test names don't reference this plan.
