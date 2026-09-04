# Phase 4 — Split tables.py by grain

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Break the 1,850-line god-module into per-grain builder modules so a scientist finds "how the per-event / per-arm / per-stop table is built" in one focused file. **Pure code motion — zero logic change.** Numerical-path phase: regen byte-diff required, and it must be exact because nothing computational moved.

**Inputs to read first:**

- `src/continuum_swr_replay/analysis/replay/tables.py` — the whole module. Map each top-level symbol to a grain before moving anything: `build_event_table:259`, `build_event_sequence_table:797`, `build_stop_table:900`, `build_event_arm_table:1225`, `build_event_patch_table:1389`, plus their private helpers and the module-level constants (`EVENT_ARM_STANDARDIZED_COLS`, `add_nonlocal_swr_split`, etc.).
- Every importer of `tables.py`: `grep -rn "from .* import.*tables\|analysis.replay.tables\|replay import build_" src scripts tests` — the re-export shim must keep all these working unchanged.

## Tasks

- **Decide the split and write it down first** (in this phase file's execution notes or a scratch map): three new modules under `analysis/replay/tables/` — `event.py` (event + event_sequence), `arm_patch.py` (event_arm + event_patch + `EVENT_ARM_STANDARDIZED_COLS` + `add_nonlocal_swr_split`), `stop.py` (stop). Shared private helpers used by ≥2 grains go to `tables/_common.py`. Confirm the grouping by tracing which helpers each `build_*` calls before moving.
- **Move code verbatim.** Cut each function/constant to its new module with zero edits to the body. Update intra-module imports to point at `_common` where a helper moved. This is the whole risk surface — resist any "while I'm here" cleanup (that would break the byte-diff and is out of scope).
- **Turn `tables.py` into a thin re-export shim:** `from .tables.event import *`-style explicit re-exports of every previously-public name (list them; no `import *` with a `__all__` gap). Every existing `from ...tables import build_event_arm_table` keeps working. Add a module docstring: "Canonical table builders, split by grain — see tables/event.py, tables/arm_patch.py, tables/stop.py."
- **Note the removed path:** the old monolithic `tables.py` body is replaced by the shim in this same phase — no parallel `tables_v2`. The shim *is* the compatibility surface.
- **Docs:** update the `docs/README.md` pipeline description and any `CLAUDE.md` "Project Structure" reference that points at `tables.py` to name the new per-grain modules.
- **Regen byte-diff + full suite.** Rebuild canonical tables via `--from-cache`, run `compare_to_golden_baseline` (must be byte-identical), and run the full suite (the builder-parity and from-cache-parity integration tests are the real proof here).

## Deliberately not in this phase

- Any change to builder *logic*, signatures, or the covariate/disk-read coupling (phase 5). If a function is hard to move because it reads disk, note it for phase 5 — do not fix it here.
- Renaming builders or changing their public API — the re-export shim preserves every name exactly.
- Splitting other large modules (`history.py`, `covariate_registry.py`) — out of scope; this phase is `tables.py` only.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_tables_shim_reexports_all_builders` | Every previously-public name importable from `analysis.replay.tables` still is, and is the same object as in its new home. |
| existing `test_builder_parity` / `test_from_cache_parity` (integration) | Canonical tables built post-split equal pre-split. |
| regen + `compare_to_golden_baseline` (slow, integration) | Byte-identical to the frozen v5 baseline. |
| full unit suite | Green (imports across ~40 modules unbroken). |

## Fixtures

None new — this phase leans entirely on the existing parity/integration tests and the golden baseline. That reliance is why phases 0–1 come first.

## Review

Before opening the PR, dispatch `code-reviewer` against the diff. Confirm:
- The diff is pure motion: no function body changed (review as "moved" hunks; any real edit is a red flag).
- The shim re-exports every prior public name; no importer broke.
- Regen byte-diff clean; parity integration tests pass.
- Docs/CLAUDE.md structure refs updated to the new module names.
- No `tables_v2` / parallel path left behind.
- No name references this plan.
