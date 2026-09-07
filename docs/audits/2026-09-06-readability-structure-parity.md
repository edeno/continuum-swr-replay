# Parity audit for the readability-structure change set

**Date:** 2026-09-06
**Commit range audited:** `cd06a31..bf21a5e` (14 commits; `bf21a5e` is the audited HEAD)
**Frozen baseline:** `Processed-Data/replay_content/_golden_baseline_pre5b/`
(frozen at `cd06a31`, before the first edit, by
`uv run python scripts/qa/freeze_regression_baseline.py --label pre5b`)

## Verdict

**PASS.** The change set is structure-only: it moved constants, split one helper module into
three, collapsed sixteen duplicate session-key aliases and five copies of one indicator,
derived event timing once instead of twice-and-comparing, trimmed the covariate registry to
its runtime half, and relocated two command lines out of the package into `scripts/`. No
fitter, spec, formula, or persisted column changed.

Evidence:

- Every canonical table is **byte-identical** to the frozen baseline, on both the keyed
  comparison and the raw parquet bytes (§ 1).
- The builders, re-run at HEAD on a five-session slice, reproduce the live rows **exactly**
  on `event_table`, `event_patch_table`, `event_sequence_table` and `stop_table`, and on
  every `event_arm_table` column except the four cohort-scoped `_z` columns, whose raw
  source columns are bit-identical and whose `_z` values differ by exactly the affine map
  the two standardization references imply (§ 2).
- The content fit at HEAD is **byte-identical** to its pre-edit run, so max |Δ| over every
  coefficient is 0.0 exactly, not merely below 1e-12 (§ 3).
- `uv run pytest tests/unit -q` (slow marks included): **2144 passed**, zero failures (§ 4).
- All three import-layering rules went from 16/2/3 violations to 0/0/0 (§ 7).

---

## 1. Canonical-table parity gate

```bash
uv run python scripts/qa/compare_to_golden_baseline.py \
    --baseline Processed-Data/replay_content/_golden_baseline_pre5b
```

Result: `byte-identical to baseline across 8 table(s)`, exit 0.
Wall clock 1:02.56, peak RSS 13.62 GB. The gate's default `--candidate` is the live
`Processed-Data/replay_content` root and its default `--atol` is `0.0` (exact), with no
`--expected-drift-columns` and no `--expected-added-columns` declared, so this is exact
parity with nothing waived.

`compare_to_golden_baseline.py` compares each table against the baseline on the table's
registered row key via `table_parity.compare_table_roots`. Independently, the on-disk
parquet files were hashed on both sides and are identical byte for byte:

| table | rows | sha256 (live == baseline) |
|---|---:|---|
| `event_table` | 65,913 | `68ffdc1bdce6881cd53a6f55ae813020907cd6eb86860a3d90a07a699869491c` |
| `event_arm_table` | 593,217 | `6b11ce36bcf23fa2b0d81cd56519ba6973a196e2f512ca1fd3011b7f2cca83f3` |
| `event_patch_table` | 197,739 | `dff5b2bb8312273da6a739dc69067a1e3aec455fdce358047b60ad6fa886c126` |
| `event_sequence_table` | 2,971,614 | `9f23c2b6071201c32434cf66492fe32112f43ef7f12f8006d84abcad73d7b8ed` |
| `event_onset_table` | 137,745 | `4ff2f1635c5b159a843a2c72a92cc13c658a46aabe250c603dc035a826eec57c` |
| `stop_table` | 107,400 | `ea62e0b65fbec97c8fce64185a6627856a7056b1f5e3915aef52bf32d944610f` |
| `stop_timebin_table` | 32,533,075 | `d508ecf5384a58fb4fdba3716de773e3d18f9f19466effe4e9446cd8eff86c7e` |
| `qvals_table` | 105,359 | `686f0985b7abcf60d5c174140944be2b5a98868840268fa053b8eea94f167c38` |

(The live `qvals_table.parquet` sits under `models_per_bin_rate/`; the frozen copy is flat.)

## 2. Builder re-run through the relocated command line

The canonical tables were not regenerated during this change set, so § 1 alone proves only
that nothing overwrote them. To prove the *builders* still produce the same values at HEAD —
and that the command line still works from its new home under `scripts/decoding/` — one
session per animal was rebuilt from the pinned behavioral cache into a scratch directory:

```bash
uv run python scripts/decoding/prepare_replay_content_data.py \
    --from-cache \
    --max-per-animal 1 \
    --output-dir <scratch>/post5b/rebuild \
    --behavior-cache-dir Processed-Data/replay_content/_session_cache \
    --contingency-dir  Processed-Data/replay_content
```

Wall clock 3:38.46, peak RSS 11.68 GB, exit 0 (the driver's own total: 213 s). Sessions
rebuilt: `chimi20200213 02_r1`, `j1620210706 02_r1`, `peanut20201127 14_r7`,
`senor20201027 02_r1`, `wilbur20210328 02_r1`. Driver totals: `events: 3354, event_arm rows:
30186, stops: 5046, sessions: 5`; the cross-table invariant
`Σ n_selected_events == len(event_table) == 3354` held.

Each rebuilt table was then diffed against the same five sessions' rows in the *live*
canonical table with `table_parity.diff_table(..., atol=0.0)` on the table's registered row
key:

| table | rebuilt rows | live-subset rows | cols | row key | drifted columns |
|---|---:|---:|---:|---|---|
| `event_table` | 3,354 | 3,354 | 103 | `animal, session_date, epoch, event_id` | **none (exact)** |
| `event_arm_table` | 30,186 | 30,186 | 84 | `… , event_id, candidate_arm` | 4 `_z` columns (below) |
| `event_patch_table` | 10,062 | 10,062 | 29 | `… , event_id, patch` | **none (exact)** |
| `event_sequence_table` | 142,320 | 142,320 | 18 | `… , event_id, t_in_event` | **none (exact)** |
| `stop_table` | 5,046 | 5,046 | 55 | `… , stop_id` | **none (exact)** |

Row counts, key sets, column sets, dtypes and NaN positions matched everywhere.
`diff_table` raises on structural drift and on any undeclared value drift, so "exact" here
means the comparison ran to completion with an empty report at `atol = 0.0`.

### The four `event_arm_table` `_z` columns are a cohort-scope artifact, not a code difference

`_z` columns are standardized over the fit cohort, so a 5-session rebuild computes its
reference over 5 sessions while the live table's reference spans 218. Two facts separate
that from a code difference:

1. **The raw source column of each `_z` is bit-identical** between the rebuild and the live
   rows — equal under `np.array_equal(..., equal_nan=True)` *and* under raw IEEE bit
   comparison (`float64.view(np.int64)` over the non-NaN entries).
2. **The two `_z` versions are related by an exact affine map**, and that map is exactly the
   one the two `standardization_reference.json` files imply
   (`slope = std_live / std_rebuild`, `intercept = (mean_live − mean_rebuild) / std_rebuild`).

Regressing the rebuild's `_z` on the live `_z`:

| `_z` column | raw source bit-identical | max &#124;Δz&#124; | fitted slope | fitted intercept | max &#124;residual&#124; |
|---|---|---:|---:|---:|---:|
| `reward_belief_mean_candidate_arm_preoutcome_z` | yes | 1.70598 | 0.712849001675 | −0.559391850616 | 4.88e-15 |
| `reward_belief_variance_candidate_arm_preoutcome_z` | yes | 1.01549 | 0.783674206360 | 0.373148087122 | 1.55e-15 |
| `reward_fraction_candidate_arm_last10_visits_z` | yes | 0.677261 | 0.890826572551 | −0.455174804415 | 2.00e-15 |
| `task_reward_probability_candidate_arm_z` | yes | 0.906110 | 0.829136768924 | −0.496979266356 | 8.88e-16 |

Each fitted slope and intercept agrees with the reference-derived prediction to ≤ 4.3e-13.
For example, `reward_belief_mean_candidate_arm_preoutcome`: the rebuild's reference records
`std = 0.16691575250266708` over 19,962 reference rows, the canonical one
`std = 0.11898572753538102` over 390,120; their ratio is 0.712849001675, the fitted slope to
twelve decimals. Rebuilding the whole cohort restores the canonical reference and hence the
canonical `_z`.

## 3. Content fit smoke at HEAD versus its pre-edit run

```bash
uv run python scripts/content/fit_replay_content_models.py \
    --models M0,M0a --output-dir <scratch>/post5b/content
```

(`fit_replay_content_models.py` has no `--max-per-animal`, so this is the full 218-session
cohort at two rungs.) Wall clock 1:00.33, peak RSS 2.10 GB, exit 0.

| file | sha256 pre-edit | at HEAD | |
|---|---|---|---|
| `M0.pkl` | `e6e6a468e042b1db225c0f998cbb6ede404d3b46670bba26cf4be1b62ded9a86` | same | identical |
| `M0a.pkl` | `ab70b24d7da6065b4da706837aded7a0fe34f676bebf2ccbe9c2699e16d01922` | same | identical |
| `all_models.pkl` | `dadc7da64f108063968a94ec276937dec24e03b7f4ab0cf10c60569985dbebef` | same | identical |
| `model_comparison.pkl` | `080b8a42d1f454d3de80984049502594b569f3be24076366ebabb4ee68724c46` | same | identical |
| `all_models_cohort_audit.csv` | `6a77887a784a7c2af8502a5600a0214af6487aad60fc37f9a07a6a59ee1af340` | same | identical |

The `.provenance.json` sidecars carry a `written_at` wall-clock field and are excluded by
construction. As a second, value-level check the pickles were dumped to column-sorted
`%.17g` CSVs on both sides and diffed: `diff -r <scratch>/pre5b/coefs <scratch>/post5b/coefs`
is **empty**. Max |Δ| over every coefficient is therefore **0.0 exactly**, not merely below
1e-12.

## 4. Full unit suite

Collected count:

```bash
uv run pytest tests/unit --co -q | tail -1
2144 tests collected in 22.87s
```

```bash
uv run pytest tests/unit -q
2144 passed, 63 warnings in 287.21s (0:04:47)
```

This run includes the `slow` marks. Zero failures. (The
`test_rate_ladder_rungs_share_a_cohort` failure recorded in the preceding hygiene audit is
gone: it reads the materialized per-bin rate ladder, which has since been refit onto a
single cohort.)

`uv run ruff check src/ scripts/ tests/` — all checks passed.

## 5. Module size

```bash
wc -l src/continuum_swr_replay/analysis/replay/_helpers.py \
      src/continuum_swr_replay/analysis/replay/covariate_registry.py \
      src/continuum_swr_replay/analysis/replay/model_specs.py \
      scripts/content/fit_replay_content_models.py \
      src/continuum_swr_replay/analysis/replay/table_contracts.py
```

| file | before (`cd06a31`) | at HEAD | note |
|---|---:|---:|---|
| `analysis/replay/_helpers.py` | 1273 | **405** | split into `bouts` / `event_ids` / `per_bin_design`; 364 after the split alone, then +41 for the shared `is_stay` pair and its type annotation |
| `analysis/replay/covariate_registry.py` | 1535 | **908** | ledger published as documentation, runtime half kept |
| `analysis/replay/model_specs.py` | 1394 | **1343** | 17 shared design constants moved to `design_constants.py` |
| `scripts/content/fit_replay_content_models.py` | 2601 | **2172** | three contract validators moved to `table_contracts.py` |
| `analysis/replay/table_contracts.py` | 333 | **772** | receiving side of the same move |

New modules and their sizes: `design_constants.py` 127, `bouts.py` 152, `event_ids.py` 83,
`per_bin_design.py` 756, `scripts/decoding/prepare_replay_content_data.py` 140,
`scripts/behavior/build_qvals_table.py` 73.

Two of these did not reach their nominal targets and the trimming was stopped rather than
pushed into a behaviour change:

- **`_helpers.py` at 405 (target < 400).** The split alone left it at 364; the shared
  `stay_indicator` / `stay_indicator_from_patches` pair (39 lines with docstrings) and their
  type annotation account for the rest. The kept set is exactly the shared-primitive set.
- **`covariate_registry.py` at 908 (target < 400).** Roughly a quarter is data — 189 lines of
  `qvals` `Entry` rows and 68 lines of frozen legacy names — and the kept *functions* alone
  are 578 lines. Reaching the target needs a module split (a separate `qvals_ingestion.py`),
  not more trimming.

The science path in the content runner (`_load_and_filter_cohort` 204 lines,
`_fit_content_ladder` 80, `fit_model_per_session` 143; 427 total) is **unchanged** — none of
the moved validators was inside those functions, they were called from them.

## 6. `--help` sweep

Every `.py` file under `scripts/content/`, `scripts/rate/`, `scripts/qa/`,
`scripts/decoding/`, `scripts/behavior/` and `scripts/figures/` — **56 scripts**, including
both relocated command lines — was checked for an `argparse.ArgumentParser` first (all 56
have one), then run as `COLUMNS=100 uv run python <script> --help`.

**56 of 56 exited 0 and printed a `usage:` line. Zero failures.** So no module split, no
deleted symbol and no relocated entry point breaks an importer at import time.

The two relocated command lines were additionally diffed against their pre-move `--help`
output rendered from a pristine `git archive` of the old tree: the only differences are the
program name (and its usage-line rewrap) and the sentences that name the invocation path.
All flags, defaults, choices and help strings are otherwise byte-identical.

`grep -rn "python -m continuum_swr_replay" docs/ scripts/ src/ tests/ CLAUDE.md` returns
nothing outside `docs/audits/`, where historical records are left as written.

## 7. Import layering

`tests/unit/test_import_layering.py` parses the package with `ast` — it imports nothing, so a
violation is reported as a plain `file -> module` edge rather than as an `ImportError` from a
cycle. Three rules:

1. No module under `analysis/replay/covariates/` may import `model_specs`. A builder writes
   columns; a ladder names them. The shared numbers live in `design_constants` so both sides
   can read them without the edge.
2. `data_loaders/` may not import `analysis/`. Loading is upstream of every analysis.
3. `analysis/hippocampus/` may not import `analysis/replay/` except the two leaf modules the
   event tables genuinely share (`bouts`, `event_ids`).

Violation counts, measured with the same checker against a pristine `git archive cd06a31 src`
tree and against HEAD:

| rule | pre-move (`cd06a31`) | at HEAD |
|---|---:|---:|
| (i) `covariates/` → `model_specs` | **16** (5 files) | **0** |
| (ii) `data_loaders/` → `analysis` | **2** (1 file) | **0** |
| (iii) `hippocampus/` → `replay` outside the leaves | **3** (1 file) | **0** |

The 16 rule-(i) edges were in `covariates/candidate_arm.py` (2),
`covariates/definitions.py` (6), `covariates/history/reward_history.py` (3),
`covariates/history/value_uncertainty.py` (2) and `covariates/history/visit_recency.py` (3).
Rule (ii) was `data_loaders/trial_clocks.py` reaching for
`analysis.constants.OMISSION_KNOWN_LATENCY_SEC`. Rule (iii) was
`analysis/hippocampus/event_table.py` reaching for `replay._helpers.hash_event_id` /
`validate_event_ids`.

The test file carries **no `xfail` marker** (`grep -n xfail tests/unit/test_import_layering.py`
→ no hits): all three rules pass outright. A fourth test pins the checker's own resolver — a
relative import inside a package `__init__.py` must resolve against the containing package,
or `from ..model_specs import X` would name a nonexistent module and escape rule (i).

`uv run python scripts/qa/write_pipeline_map.py --check` →
`docs/regression_pipeline_map.md is current`, exit 0.

## 8. Deleted and moved symbols, with grep evidence

Every symbol was grepped over `src/ scripts/ tests/ docs/` before deletion and re-grepped at
HEAD for this audit. All counts below are hits at HEAD.

### Modules and constants that moved

| moved | from | to | note |
|---|---|---|---|
| 17 design constants (`ARM_FE_PREDICTORS`, `RECENCY_SPLINE_*`, `REWARD_HISTORY_*`, `N_REWARD_LAGS`, `BIN_S`, `TMAX`, `SPLINE_KNOTS`, `TMAX_ARRIVAL`, `ARRIVAL_SPLINE_KNOTS`, `ARRIVAL_CLOCK_COLUMN`, `N_HISTORY_LAGS`, `N_HISTORY_BASIS`) | `analysis/replay/model_specs.py` | `analysis/replay/design_constants.py` | `model_specs.__all__` still re-exports all 17 for scripts; every name in `__all__` (47) still resolves |
| `OMISSION_KNOWN_LATENCY_SEC` | `analysis/constants.py` | `data_loaders/constants.py` | 6 importers updated, no shim; `analysis/constants.py` now *imports* it to interpolate `OUTCOME_TIME_SEMANTICS` (a genuine use, and the permitted direction) |
| 4 bout/immobility helpers | `analysis/replay/_helpers.py` | `analysis/replay/bouts.py` | |
| 3 id-hashing helpers | `analysis/replay/_helpers.py` | `analysis/replay/event_ids.py` | |
| 9 per-bin design helpers + `POPULATIONS` + 4 private column tuples | `analysis/replay/_helpers.py` | `analysis/replay/per_bin_design.py` | |
| `_build_arg_parser`, `main`, the `__main__` guard | `analysis/replay/prepare_data.py` (1134 → 1022) | `scripts/decoding/prepare_replay_content_data.py` | `run_prepare_data` and every library symbol stayed |
| the argparse half of `main` | `analysis/behavior/qvals_table.py` (1091 → 1045) | `scripts/behavior/build_qvals_table.py` | the other 16 statements became `run_build_qvals_table`, AST-identical |
| 3 contract validators + 4 constants | `scripts/content/fit_replay_content_models.py` | `analysis/replay/table_contracts.py` | bodies AST-identical; `_CV_RESULT_COLUMNS` → public `CV_RESULT_COLUMNS` |

Grep at HEAD: each of the 16 moved `_helpers` definitions has **0** hits in
`analysis/replay/_helpers.py` (`nearest_well_distance_cm`, `immobility_bout_intervals`,
`time_since_stop`, `assign_bout`, `hash_event_id`, `hash_stop_id`, `validate_event_ids`,
`attach_is_swr_to_onsets`, `raised_cosine_history_basis`, `reward_basis_from_lags`,
`reward_kernel_weights`, `reconstruct_reward_kernels`, `reconstruct_self_history_kernels`,
`_arrival_spanning_stop_ids`, `aggregate_to_timebins`, `POPULATIONS`). The move text is
byte-identical apart from `ruff format`; an AST comparison of all 36 top-level symbols of the
pre-split module against the four post-split modules reports `missing: []`, `extra: []`,
`AST-different: []`.

`grep -n "^def main\|^def _build_arg_parser\|^if __name__" ` over `prepare_data.py` and
`qvals_table.py` → **no hits**: neither package module carries a command line any more.

### Deleted symbols

| symbol | where it was | hits at HEAD |
|---|---|---:|
| `_merge_classified_and_verify_timing` | `analysis/replay/tables_event.py` | **0** |
| `_SESSION_COLS` | `analysis/replay/_helpers.py` (and its 15 sibling aliases) | **0** |
| `_alias` | `analysis/replay/covariate_registry.py` | **0** |
| `rename_map` | `analysis/replay/covariate_registry.py` | **0** |
| `retired_columns` | `analysis/replay/covariate_registry.py` | **0** |
| `dropped_columns` | `analysis/replay/covariate_registry.py` | **0** |
| `audit_columns` | `analysis/replay/covariate_registry.py` | **0** |
| `all_current_columns` | `analysis/replay/covariate_registry.py` | **0** |
| `export_json` | `analysis/replay/covariate_registry.py` | **0** |
| `_validate_requested_standardization` | `scripts/content/fit_replay_content_models.py` | **0** |
| `_validate_canonical_table_contract` | `scripts/content/fit_replay_content_models.py` | **0** |
| `_validate_and_match_fit_results` | `scripts/content/fit_replay_content_models.py` | **0** |
| `_CV_RESULT_COLUMNS` | `scripts/content/fit_replay_content_models.py` | **0** |

Two grain *names* went with the trimmed rows: `_entries_for` (and through it
`assert_canonical_vocabulary`) now validates `table=` against the eight `_ROW_KEYS` grains
rather than the ten that used to carry registry entries, so `table="choice"` and
`table="fit_time"` raise `RegistryError` instead of resolving to an empty entry set — both
have **0** callers at HEAD.

`_merge_classified_and_verify_timing` computed the event timing twice — once in the
aggregator as `event_*_s`, once from the classified table as `label_event_*` — and then
asserted the two agreed. Before deleting it, the comparison was reproduced outside the
builder on five cached sessions (3,354 events × 4 timing columns): equal under
`np.array_equal(..., equal_nan=True)` *and* under raw IEEE bit comparison, same dtypes, zero
mismatches. The replacement `_merge_classified_event_facts` renames the classified timing to
the canonical `event_*_s` names in the merge that already brings in the SWR classification,
keeps `validate="one_to_one"`, and keeps the one guard that can actually fire (an event with
no classified row). Full `event_table` builds on those five sessions are byte-identical
before and after across all 95 columns (`%.17g` CSV sha256 equal on all five, same dtype
vector).

### Collapsed duplication

| what | before | after | pin |
|---|---:|---|---|
| module-level session-key aliases | **16** (6 in `src/`, 10 in `scripts/`) | 0; every site uses `session_keys.SESSION_COLUMNS` / `SESSION_COLUMNS_DAY` directly | `tests/unit/test_session_keys.py::test_no_local_session_key_definitions`; `grep -rnE "^_?SESSION_COL(S\|UMNS)(_DAY)?\s*(:[^=]*)?=" src/ scripts/` (excluding `session_keys.py`) → no hits |
| copies of the `is_stay` rule | **5** (3 vectorized, 2 scalar, across `tables_stop.py`, `tables_event.py`, `shuffles.py`, `contfrag/tables.py` ×2) | one definition pair in `_helpers.py`, called from all five | `tests/unit/test_stay_indicator.py` (12 tests) |
| covariate-registry `Entry` rows | **237** across 10 grains | **53** (`qvals` only); the other 184 published as an appendix table in `docs/design/2026-08-21-covariate-naming-scheme.md` (565 → 877 lines) | `test_deny_list_matches_the_ledger_it_was_extracted_from` (recomputes both frozen sets from the pre-trim module) and `test_deny_list_is_documented_in_the_naming_scheme_appendix` (git-free) |

All 16 session-key aliases were derived from the canonical `session_keys` constants — none
retyped the tuple and none disagreed, so the collapse is a naming change, not a semantic one.
A missed `list(...)` wrap would be loud rather than silent: pandas 2.3.3 raises `KeyError`
for a tuple passed to `groupby` / `df[...]` / `sort_values`.

All five `is_stay` sites compared **patches** (not arms) and propagated NaN identically, so
there was no divergence to resolve. The shared pair keeps both shapes:
`stay_indicator_from_patches` carries the equality/NaN semantics for scalars and arrays (the
three vectorized sites keep their vectorized path), and `stay_indicator` maps arms through
`arm_to_patch` first.

The registry's per-grain vocabulary check became one global deny-list. Two measurements
bound the change: no canonical name of any grain is a legacy name of any other
(`⋃canonical ∩ ⋃legacy = ∅` over all 10 grains), and for each of the eight materialized
tables the global set flags exactly the columns the per-grain set did (386 distinct columns,
zero difference). So on the columns that actually exist — canonical vocabularies and
materialized tables alike — the two checks are indistinguishable. They are *not* identical
in general: cross-grain the global set is deliberately stricter, because a legacy spelling
retired from one grain (or a `coef_`/`bse_`/`_z` decoration of one) is now rejected in every
grain, including grains that never carried it. That direction only ever rejects more, and
nothing in the pipeline relies on re-introducing another grain's retired spelling. At HEAD
the registry holds 53 entries and 219 frozen legacy names.

---

## Hygiene

- `git diff --name-only cd06a31..HEAD | grep -E "^(Processed-Data|logs|\.superpowers)/"` →
  nothing. 81 files changed, +3,436 / −3,225.
- `paper/claims_table.md` untouched.
- No `git checkout` / `stash` / `clean` was run; pre-existing untracked user files were left
  alone.
- The frozen baseline `Processed-Data/replay_content/_golden_baseline_pre5b/` was not
  touched.
