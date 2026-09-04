# Overview — Scope, dependencies, integration, verification, risks

[← back to PLAN.md](PLAN.md)

## Current codebase integration points

References verified against the tree at planning time; re-check line numbers before editing (see Risks).

**Session-key identity (phase 2)** — the tuple `(animal, session_date, epoch)` and its helpers are redefined in ~20 sites with divergent types. Canonical home already exists but is unused for this:

- `src/continuum_swr_replay/analysis/replay/covariate_registry.py:74` — `_ROW_KEYS`: authoritative per-table keys; `:117` `row_keys(table=...)` accessor. The new module wraps these.
- `src/continuum_swr_replay/analysis/replay/table_contracts.py:21` `_SESSION_COLUMNS`; `_helpers.py:26` `_SESSION_COLS`; `covariates/history.py:56`; `choice_prediction.py:49`; and 15 `scripts/**` copies (rate ×6, decoding ×6, content, pfc ×2, exploratory). All replaced by the shared module.
- `src/continuum_swr_replay/analysis/hippocampus/event_table.py:77` — `_SESSION_COLS = ["animal", "day", "epoch"]`: **the one outlier using `day`**. This grain names its session-date column `day`, not `session_date`; the shared module must expose a `day`-named variant or this site keeps a local mapping. Do not blindly rewrite it to `session_date` — that column name is load-bearing for the hippocampus event table.

**Version literals (phase 2)** — four hand-edited, semantically-coupled strings with no cross-check:

- `covariate_registry.py:71` `COVARIATE_SCHEMA_VERSION`; `:72` `COVARIATE_SCHEMA_SCOPE`.
- `analysis/replay/prepare_data.py:117` `BEHAVIORAL_CONTEXT_CACHE_VERSION`; `:245` reads `cache_schema_version` from a cache bundle.
- `scripts/decoding/write_regeneration_manifest.py:527` `manifest_schema`.

**model_specs.py prose (phase 2)** — `analysis/replay/model_specs.py` is 1,465 lines, the majority design-journal prose (reversed decisions, retracted numbers). The `ModelSpec` catalog stays; the narrative moves to `docs/design/model_history.md`.

**Canonical-table writers (phase 3)** — `write_table` (`table_io.py`) is called from ~14 sites that persist canonical tables, each independently applying `stamp_covariate_schema` (or not). Sites: `prepare_data.py:674,1009`; `analysis/behavior/qvals_table.py:1073`; `scripts/decoding/prepare_contfrag_replay_content_data.py:496`; `assemble_replay_content_shards.py:170,202`; `build_timebin_tables.py:133,134`; `regenerate_stop_table.py:202,204`; `assemble_contfrag_shards.py:109`; `refresh_contfrag_behavioral_covariates.py:293`. (`convert_tables_to_parquet.py:81` is a one-off migration tool — left alone.) All canonical writers route through the new `publish_canonical()`.

**tables.py builders (phase 4)** — `analysis/replay/tables.py` (1,850 lines) hosts `build_event_table:259`, `build_event_sequence_table:797`, `build_stop_table:900`, `build_event_arm_table:1225`, `build_event_patch_table:1389`. Split by grain; `tables.py` becomes a thin re-export shim.

**Covariate builder disk reads (phase 5)** — `analysis/replay/covariates/history.py:46` imports `read_table`/`table_exists`; `:49` imports `require_covariate_schema`; `load_contingency:1322` reads a persisted table at `:1334`. These become arguments passed in by callers.

**Mutation survivors (phase 1)** — 36 survivors from the objective AST pass over `covariates/candidate_arm.py` (5) and `covariates/history.py` (114 mutations, 31 survivors). Highest-signal: five `searchsorted side="right"→"left"` flips at `history.py:462,547,698,747,1519` — the reward-history-window semantics.

## Scope and dependency policy

### Goals

- A naive scientist can trace raw → parse → tables → covariates → fits with each stage in one obvious module.
- One definition of a "session"; one door onto the canonical tables; covariate values as explicit functions of inputs.
- A measured, non-eroding builder test floor.

### Non-Goals

- **No change to the science.** Estimands, model specs, cohorts, covariate definitions, standardization references are untouched. This is structure and tests only.
- **No change to published file paths or on-disk layout** beyond what `publish_canonical` standardizes internally. Paths a scientist or downstream analysis already references stay put.
- **No new dependencies.** `mutmut` is not adopted; phase 1 uses the in-repo AST harness.
- No documentation-narrative layer (the user chose code-structure-first); readability comes from the code and its docstrings.

### Dependency policy

No dependency changes. The mutation harness is stdlib `ast` + `subprocess` only.

## Verification methodology (the spine — applies to every numerical-path phase)

1. **Golden baseline (coarse, total).** Phase 0 freezes the current v5 canonical tables and model outputs read-only. After any numerical-path change, regenerate from the warm `--from-cache` path and byte-diff against the frozen set. The `read_table`-based parity harness proven this session (timebin rebuild matched 32.5M rows; `swr_x_q_arm` matched to 6 decimals) is the reference implementation.
2. **Value-level unit tests (precise, local).** Phase 1 raises the builder test floor by killing the mutation survivors, so a later builder change that alters a covariate value fails a *named* test at the line, not an opaque table diff.
3. **The gate for every phase:** existing suite green + new tests green + (numerical-path phases) regen byte-identical to the frozen baseline. A phase that cannot show all three does not merge.

## Metrics

- Parity tolerance: **exact** (byte-identical parquet content on the canonical tables; the harness compares per-column with `atol=0` on numerics and equality on objects). Model-output coefficients: identical to ≥6 decimals on the headline terms.
- Mutation kill rate on `candidate_arm.py` + `history.py`: from 70% (36 survivors) to **≥95%** after phase 1, with **zero surviving `side=` mutants**.
- Session-key definition sites: from ~20 to **1**.
- Canonical-table write paths: from ~14 ad-hoc to **1** chokepoint.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| A "pure code-motion" split silently changes a covariate value | Phase 1 test floor + phase-0 regen byte-diff catch it; the split phase is gated on byte-identical output. |
| The `day` vs `session_date` outlier (`event_table.py:77`) gets normalized and breaks the hippocampus table | Overview + phase 2 flag it explicitly as do-not-touch-the-column-name; the shared module exposes a `day`-named accessor. |
| Line numbers drift between planning and execution | Every phase lists "Inputs to read first"; the executor re-greps the anchor before editing rather than trusting the number. |
| Regenerating for the byte-diff picks up unrelated Spyglass drift | Use `--from-cache` (frozen behavioral cache) so only code changes can move the output; never the live Spyglass path. |
| Equivalent mutants inflate the phase-1 target | Phase 1 triages each survivor as real-gap vs equivalent before writing a test; equivalent mutants are documented, not forced. |

## Rollout Strategy

Replace-in-place, no feature flags. Backwards compatibility is preserved where external callers exist: `tables.py` keeps re-exporting every builder name (phase 4), and no published file path changes. Each phase is an independent PR that leaves the suite green and the canonical tables byte-identical, so partial rollout is safe. The mutation harness ships as a repo tool (phase 0) and a CI-runnable check.

## Open Questions

1. **Publish-chokepoint scope for contfrag writers** — the ContFrag reducers (`assemble_contfrag_shards.py`, `refresh_contfrag_behavioral_covariates.py`) write source-tagged tables. Best answer: route them through `publish_canonical` with a `basename` override; confirmed feasible since they already call `stamp_covariate_schema` then `write_table`. Deferred detail to phase 3.
2. **Where centralized versions live** — best answer: a new `analysis/replay/versions.py` holding the three literals plus a `assert_versions_compatible()` the manifest calls. Confirmed in phase 2.

## Estimated Effort

Roughly: phase 0 ~150 LOC (harness relocation + baseline script); phase 1 ~400 LOC (tests); phase 2 ~250 LOC net-negative (delete 19 copies, add 1 module + versions); phase 3 ~200 LOC (chokepoint + 14 call-site edits); phase 4 ~0 net (code motion across 3 new files + shim); phase 5 ~150 LOC (signature changes + caller updates). Phases 4–5 are the largest diffs but lowest net-new logic.
