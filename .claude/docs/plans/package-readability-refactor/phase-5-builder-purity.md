# Phase 5 — Make covariate builders I/O-pure

**Finding on execution: already satisfied.** The covariate builders are already
I/O-pure. The only disk read in `history.py` is inside `load_contingency` itself
(the intentional single loader), which callers (`prepare_data`, the ContFrag
reducers) invoke once at the top and pass the frame down; every builder receives
`contingency`/`qvals` as an argument. Verified empirically by monkeypatching
`read_table` to raise and running the builders on in-memory frames. So this
phase reduces to a regression guard (`test_history_builders_are_io_pure`) that
pins the contract, plus this note. No builder signature or logic changed, so no
regeneration byte-diff is needed. The tasks below are the original plan, kept
for the record.

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Remove hidden disk reads from covariate construction so a reader sees covariates as a function of explicit inputs, and unit tests run on in-memory frames instead of on-disk fixtures. Highest-risk phase (changes builder signatures and their callers); lands last, on the phase-1 test floor and the golden baseline. Numerical-path phase: regen byte-diff required.

**Inputs to read first:**

- `src/continuum_swr_replay/analysis/replay/covariates/history.py:46` (`read_table`/`table_exists` import), `:49` (`require_covariate_schema`), `:1322-1340` (`load_contingency` reads a persisted qvals table).
- Every caller of `load_contingency` and of the covariate-attach functions that trigger the disk read: `grep -rn "load_contingency\|read_table" src/continuum_swr_replay/analysis/replay/covariates scripts` — these callers already hold or can cheaply load the qvals/contingency frame and will pass it in.
- `analysis/replay/prepare_data.py` and the content/rate runners — the top-level callers that materialize qvals once and thread it down.

## Tasks

- **Change the signatures.** Make the covariate builders that currently call `load_contingency`/`read_table` internally accept the already-loaded frame as a parameter (e.g. `contingency: pd.DataFrame`). Keep `load_contingency` as a thin public loader callers use *once* at the top, then pass its result down — the builders no longer touch disk. Move the `read_table`/`table_exists`/`require_covariate_schema` imports out of `history.py` if nothing else there needs them (confirm by grep).
- **Update every caller** to load the contingency/qvals frame once and pass it through. The natural owner is the top-level runner (`prepare_data`, the fit runners); intermediate builders just forward the argument. This restores the clean build→persist→consume direction.
- **Update the phase-1 tests** that used on-disk contingency fixtures to pass in-memory frames instead (the point of the change) — they get simpler, not just different.
- **Note the removed path:** the in-builder disk read is deleted in this phase, not left as a fallback. A builder must not silently read disk when the argument is omitted — make the frame a required argument so a missing pass is a loud `TypeError`, not a hidden read.
- **Docs:** update the covariate-package docstring (`covariates/__init__.py`) and `docs/README.md` covariate section to state builders are pure functions of their frames; the only disk read is the caller's one-time `load_contingency`.
- **Regen byte-diff + full suite + mutation re-check.** Rebuild canonical tables via `--from-cache`, run `compare_to_golden_baseline` (byte-identical), full suite green, and re-run `scripts/qa/mutation_check.py` to confirm the phase-1 floor held through the signature change.

## Deliberately not in this phase

- Any change to what a covariate *computes* — only *where its inputs come from* changes. The byte-diff proves the values are identical.
- Further module splits or renames. `history.py` stays one module (splitting it is a separate future effort, noted in overview as out of scope).

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_history_builders_are_io_pure` | The covariate builders run to a correct result given only in-memory frames, with no filesystem access (assert via a `read_table` monkeypatch that raises if called). |
| `test_missing_contingency_raises_loudly` | Omitting the required frame raises `TypeError`, never a silent disk read. |
| phase-1 value tests (now in-memory) | Same asserted covariate values as phase 1, now without on-disk fixtures. |
| regen + `compare_to_golden_baseline` (slow, integration) | Byte-identical to the frozen v5 baseline. |
| `mutation_check` re-run (slow) | Kill rate still ≥95%, 0 `side=` survivors. |
| full unit suite | Green. |

## Fixtures

In-memory qvals/contingency frames replacing the phase-1 on-disk fixtures. A `read_table` monkeypatch fixture that raises, to prove no builder reads disk.

## Review

Before opening the PR, dispatch `code-reviewer` against the diff. Confirm:
- No covariate builder reads disk; the required-argument change makes omission a loud error.
- Every caller threads the frame from a single top-level `load_contingency`.
- Regen byte-diff clean; mutation floor held; phase-1 tests simplified to in-memory.
- Covariate-package docstring updated. Tests exercise behavior (the monkeypatch proves purity), not tautologies.
- No name references this plan.
