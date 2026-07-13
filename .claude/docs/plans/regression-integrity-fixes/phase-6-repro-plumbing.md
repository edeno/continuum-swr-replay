# Phase 6 — Reproducibility: fold `_z`/`q_arm_hmm` into the builder; document count convention

> Historical baseline: canonical standardization/rebuild plumbing landed before the 2026-07-10 refresh. Phase 7 adds run-level provenance; do not execute this old phase.

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Close the reproducibility hazard where a clean `prepare_data.py` rebuild silently drops the `_z`
columns (and historically `q_arm_hmm`) because they are stamped in-place by exploratory scripts. After
this phase, one canonical builder run produces the fitted tables in full. Also document the SWR/MUA
stop-vs-timebin count convention so the two totals aren't cross-substituted in offsets.

**If phase 4 chose the encode branch, run this phase FIRST** (overview regen-order note) so the table is
regenerated once from a builder that already owns `_z`/`q_arm_hmm`, before phase 4 adds `has_arm_history`.

**Inputs to read first:**

- Historical deleted path `scripts/exploratory/_add_standardized_cols.py:1-50` — the in-place `_z` stamping that this completed phase folded into the builder and removed.
- [src/continuum_swr_replay/analysis/replay/tables.py:1116](../../../../src/continuum_swr_replay/analysis/replay/tables.py) (`_event_arm_table_columns`) and [:1296-1345](../../../../src/continuum_swr_replay/analysis/replay/tables.py) — where continuous covariates are attached; the `_z` columns must be appended to the canonical column order.
- [src/continuum_swr_replay/analysis/replay/prepare_data.py](../../../../src/continuum_swr_replay/analysis/replay/prepare_data.py) — the assembly driver; per-session loop at `:263`, all-session `pd.concat` at `:308-310`, `to_pickle` at `:386`. Owns the **event_arm-side** `_z` (add after concat). `q_arm_hmm` already emitted natively (`tables.py` `_attach_hmm_arm_value`).
- [src/continuum_swr_replay/analysis/behavior/qvals_table.py:416-468](../../../../src/continuum_swr_replay/analysis/behavior/qvals_table.py) — the qvals builder `main()`: all-epoch `pd.concat` at `:453`, required-column guard, `to_pickle` at `:468`. Owns the **qvals-side** `_z` (add after concat, before write) — `prepare_data` only *reads* qvals (`:242`) and must **not** standardize it (F3).
- [docs/known_issues/](../../../../docs/known_issues/) — home for the SWR/MUA count-convention note.
- Audit evidence: `.claude/docs/reviews/06_data_provenance_regression_audit.md` §3 (SWR/MUA Δ82/Δ250) and §2 (post-hoc schema divergence).

## Tasks

- **Fold standardization into the correct builder for each table — they are two different owners (F3).**
  `_z` columns are pooled-global (means/SDs over the *full* table — `_add_standardized_cols.py:9`), and
  `_add_standardized_cols.py` currently stamps *two* tables owned by *two* builders. Add a shared
  `analysis/replay/standardization.py` with `attach_standardized(df, cols, reference=None) ->
  (df, reference)` (computes `(x - nanmean)/nanstd` pooled, or applies a supplied reference), then wire
  it into **each owning builder**:
  - **event_arm-side `_z`** (`q_arm`, `q_arm_hmm`, `reward_rate_arm_k10`, `p_reward_arm` — the
    `ARM_CONT` set in `_add_standardized_cols.py:27`): call from `prepare_data.main()` **after the
    all-session `pd.concat`** (`prepare_data.py:308-310`), before `to_pickle` (`:386`). The per-session
    loop (`:263`) cannot do global z. Append these `_z` names to `_event_arm_table_columns`.
  - **qvals-side `_z`** (`hmm_relative_value_z`, `hmm_rpe_z`, `reward_entropy_z`, `hmm_value_update_z`,
    `state_entropy_z`, `trials_since_*_z`): `prepare_data.py` only *reads* `qvals_table.pkl` as
    contingency input (`:242`) — it does **not** own it. Add the qvals `_z` in
    `qvals_table.py:main()` **after its own all-epoch `pd.concat`** (`:453`) and before its `to_pickle`
    (`:468`). Append these names to the qvals column order there.
  Keep the pooled-global scope (audit confirmed safe: the only CV family fits on raw columns). **Persist
  each builder's fitted means/SDs** to `standardization_reference.json` (event_arm side, under
  `Processed-Data/replay_content/`) and `qvals_standardization_reference.json` (qvals side, under
  `models_per_bin_rate/`) so both transforms are reproducible and auditable.
- **SCOPE — builder `_z` is NOT the fit-time `_z` the contrast runners use (F2). Do not unify them.**
  This phase only makes the *persisted table* `_z` (what `_add_standardized_cols.py` writes today)
  reproducible. The SWR-contrast **content** runner computes its **own** `_z` at fit time over the *event
  cohort* (`fit_swr_contrast_content.py:129`, `_ZSCALARS`) and over the *stop cohort* for peri-switch
  (`:166`); the **rate** contrast runner computes its own per-SD reference (`fit_swr_contrast_rate.py:346`).
  These are event/stop-*cohort* scopes, **different from** the qvals/event_arm stop-level *global* scope —
  the means/SDs differ, so pointing a runner at the builder `_z` would **change coefficients**. Decision
  (default): **preserve the runners' fit-time `_z` unchanged**, and instead **persist each run's own z
  reference** (the means/SDs it computed) into that run's output metadata (`*_meta.json`) so the fit is
  reproducible. Repointing a runner to the builder `_z` is explicitly **out of scope** and, if ever done,
  requires an old-vs-new coefficient comparison — see Non-goals.
- **Delete `scripts/exploratory/_add_standardized_cols.py`** (it stamped *both* the event_arm and qvals
  tables — both are now owned by their builders per the split above) and its `.bak_prestd` note once
  folded. Grep for consumers of the **persisted** `_z` columns and confirm they now get them from the
  rebuilt tables — note the SWR-contrast runners are **not** such consumers (they recompute their own
  fit-time `_z`, so the delete cannot change their coefficients). (Old code path removed — no orphan.)
- **Regenerate and verify parity — correctly for global z (F5).** A one-session rebuild computes `_z`
  over one session and will **not** match the current all-session pooled `_z`, so a naive one-session
  parity gate is structurally broken. Two valid gates, implement both:
  1. **Raw-column one-session parity** (fast): rebuild one session and assert every **non-`_z`** column
     matches the current materialized file within `atol=1e-10`. This proves the per-session build is inert.
  2. **`_z` parity against a frozen reference** (fast): for the event_arm table load
     `standardization_reference.json` and for the qvals table load `qvals_standardization_reference.json`,
     apply each to the one-session/one-epoch raw columns, and assert the resulting `_z` match the current
     materialized `_z` for those rows within `atol=1e-10`. This proves both standardization transforms are
     inert given the pooled references — without a full 218-session rebuild in CI.
  A **full-rebuild** parity (all sessions, `_z` recomputed) is the belt-and-suspenders check run once
  operationally after merge (smoke-test one session first per CLAUDE.md long-job rules), not in CI.
- **Document the SWR/MUA count convention.** Add `docs/known_issues/swr_mua_count_conventions.md`: the
  stop-level `n_SWR`/`n_MUA` (inclusive per-timebin `is_swr_stop` rule over the full bout) exceed the
  timebin/onset onset-in-window counts by ~0.2%/0.75% (Δ82/Δ250) because of boundary-straddling events;
  `n_non_local` matches exactly and is the safe cross-table count. State the rule: **do not cross-substitute
  `n_SWR`/`n_MUA` between stop-level and bin-level tables in an exposure/offset term.** Link it from
  `docs/README.md` and `CLAUDE.md`'s "Canonical per-event classification table" note (which already
  flags the dual `is_swr_stop` vs event-`is_swr` convention).
- **Add a rebuild-completeness check** to the builder: after assembly, assert the `_z` columns are
  present and non-empty, so a future refactor can't silently drop them again (the exact failure this
  phase fixes).

## Deliberately not in this phase

- Changing the standardization *scope* (global → per-fold) — the audit showed global is safe under the
  current inference design; a per-fold change would only be needed if a `_z` column enters a CV'd
  formula, which it does not. Out of scope.
- **Unifying the runners' fit-time `_z` with the builder `_z` (F2).** The contrast runners' event/stop-
  cohort `_z` and the tables' stop-level global `_z` are different quantities; repointing a runner at the
  builder `_z` changes coefficients and would require an explicit old-vs-new coefficient comparison.
  Explicitly out of scope — this phase preserves the runners' fit-time `_z` and only persists its
  per-run reference (see the SCOPE task).
- Encoding `has_arm_history` / mechanism A — that is [phase 4](phase-4-covariate-causal-hygiene.md).
- Re-running the full 218-session table build — the parity gate uses one session; the full regen is an
  operational step run after merge (smoke-test first per CLAUDE.md long-running-job rules).

## Validation slice

| Test | Asserts |
| --- | --- |
| `tests/unit/test_standardization.py::test_z_formula` | folded `_attach_standardized_columns` reproduces `_add_standardized_cols`'s `(x-mean)/std` (incl. `sd==0 → NaN`) on a fixed array |
| `::test_reference_roundtrip` | applying a persisted `standardization_reference.json` to raw columns reproduces the `_z` used to fit that reference |
| `tests/unit/test_runner_z_reference.py::test_run_meta_has_z_reference` | a content/rate contrast run writes its fit-time `_z` means/SDs into `*_meta.json`, and re-applying them reproduces the run's standardized columns (F2 — runner `_z` preserved + reproducible, not replaced by builder `_z`) |
| `tests/integration/test_builder_parity.py` (slow) | one-session rebuild: **non-`_z`** columns == current materialized within `atol=1e-10`; **`_z`** columns == current within `atol=1e-10` **when applied via the frozen reference** (not recomputed per-session, F5); ×9-arm/×3-patch invariants hold |
| `::test_z_columns_present` | rebuild-completeness assertion fires if a `_z` column is missing/empty |
| manual | `_add_standardized_cols.py` deleted; no runner imports it (`grep -rl _add_standardized_cols scripts src` → empty) |

## Fixtures

One cached session for the parity gate (see `docs/cached_trials_for_testing.md`). Unit tests use fixed
NumPy arrays including a zero-variance column to exercise the `sd==0 → NaN` branch.

## Review

Dispatch `code-reviewer`. Confirm:
- Standardization runs in the **all-session** assembly step (`prepare_data.main` after concat) for event_arm and in `qvals_table.main` after its concat for qvals — reviewer confirms placement (F3/F5).
- The runners' fit-time `_z` is **unchanged**; only its per-run reference is persisted to `*_meta.json`. No runner was repointed at the builder `_z` (F2) — reviewer confirms no coefficient drift.
- Raw-column one-session parity passes at `atol=1e-10`; `_z` parity passes **against the frozen reference**, not a per-session recompute — the gate is not the structurally-broken one-session global-z comparison.
- `standardization_reference.json` is persisted and the transform reproduces from it.
- `_add_standardized_cols.py` is deleted and nothing imports it (no orphan/parallel path).
- `_z` names are in `_event_arm_table_columns` and the qvals column order (a clean rebuild carries them).
- The count-convention doc states the "don't cross-substitute in offsets" rule and is linked from README/CLAUDE.md.
- The rebuild-completeness assertion actually fails when a `_z` column is dropped (not a tautology).
