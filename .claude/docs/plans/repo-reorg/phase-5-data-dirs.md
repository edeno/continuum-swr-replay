# Phase 5 — Rename the cryptic `models*/` data dirs to descriptive names

[← back to PLAN.md](PLAN.md) · [overview](overview.md#target-structure)

> **DISCOVERED COMPLEXITY (2026-07-03, during execution) — this plan under-scoped the
> content dirs.** The content-model output dir names are **dynamic**, built by the
> unit-tested `resolve_contfrag_paths` (`fit_replay_content_models.py:91-123`): `models`,
> `models_patch`, and for ContFrag sources `models_{swr,mua}_{brain_area}` (e.g.
> `models_swr_HPC`). These **collide** in namespace with the is_swr-subset dirs appended at
> `:378` (`models_swr_T`/`_F`) — same `models_swr_*` prefix, different meaning. Renaming the
> content dirs therefore requires disambiguating the two families + rewriting
> `resolve_contfrag_paths` + `test_contfrag_paths` + the suffix logic + every reader — a
> dedicated redesign, NOT the mechanical 8-dir sweep this plan assumed. **Deferred pending a
> decision.** The RATE dirs (`models_per_bin_rate`, `models_per_stop_rate`) are still a clean
> uniform rename if wanted, but marginal benefit for ~26-file churn.

The 8 output dirs under `Processed-Data/replay_content/` have opaque names
(`models_swr_F`, `models_patch_swr_T`, …) and `models/` is a **mixed** dir (M0–M7
content results *plus* the newer `swr_contrast_content_*.pkl`). Rename to descriptive
names + un-mix, and update every reader/writer. The data moves with the dir (`mv`
preserves the pkls — no recompute); the risk is reference breakage, so the audit must
use **precise** patterns (`"models"` collides as a substring of `models_patch`, etc.).

**Inputs to read first:**

- Producers (verified): `scripts/content/fit_replay_content_models.py` writes `models`,
  `models_patch`, `models_swr_{T,F}`, `models_patch_swr_{T,F}` (`:111-112,121,315`; the
  is_swr suffix — **T=is_swr True, F=False** — is chosen at `:439`);
  `scripts/rate/fit_per_bin_rate_models.py:253` → `models_per_bin_rate`;
  `scripts/rate/fit_per_stop_rate_models.py:110` → `models_per_stop_rate`;
  `scripts/rate/fit_swr_contrast_rate.py` → `models_per_bin_rate/swr_contrast_rate.pkl`;
  `scripts/content/fit_swr_contrast_content.py` → `models/swr_contrast_content_{arm,patch}.pkl`;
  `src/…/analysis/behavior/qvals_table.py` → `models_per_bin_rate/qvals_table.pkl`.
- (Paths above assume phases 3 already moved the runners into subdirs — run phase 5 after.)

## Rename map

| old | new | notes |
|---|---|---|
| `models/` | `content_arm/` | but first move `swr_contrast_content_*.pkl` OUT (see un-mix) |
| `models_patch/` | `content_patch/` | |
| `models_swr_T/` | `content_arm_swr/` | T = SWR |
| `models_swr_F/` | `content_arm_nonswr/` | F = non-SWR |
| `models_patch_swr_T/` | `content_patch_swr/` | |
| `models_patch_swr_F/` | `content_patch_nonswr/` | |
| `models_per_bin_rate/` | `rate_per_bin/` | but move `swr_contrast_rate.pkl` + `qvals_table.pkl` OUT |
| `models_per_stop_rate/` | `rate_per_stop/` | |
| — (new) | `swr_contrast/` | un-mix: `swr_contrast_rate.pkl`, `swr_contrast_content_{arm,patch}.pkl` |
| — (new) | `behavior/qvals_table.pkl` | `qvals_table.pkl` is a behavioral table, not a rate model |

## Tasks

- **Audit exact references** (substring-safe). Find every reader/writer with:
  `rg -n '"(models|models_patch|models_swr_[TF]|models_patch_swr_[TF]|models_per_bin_rate|models_per_stop_rate)"|replay_content"?\s*/\s*"models' scripts src docs`
  and also the string forms in `docs/*.md`. Build the reader list before moving anything.
- **`mv` each dir** to its new name (if `Processed-Data` is gitignored, plain `mv`; the pkls
  move with the dir — no recompute). Create `swr_contrast/` and `behavior/`; move the
  un-mixed pkls in.
- **Update every writer**: the 4 runners + `qvals_table.py` + the content runner's
  is_swr-suffix logic (`fit_replay_content_models.py:315,439` builds `models_swr_<subset>` —
  rewrite to emit `content_arm_<swr|nonswr>` / `content_patch_<swr|nonswr>`; make the F/T→
  nonswr/swr mapping explicit there).
- **Update every reader**: plotters (`plot_per_bin_rate_*`, `plot_model_checking`,
  `plot_uncertainty_rate`), `summarize_per_bin_rate_models.py`, `compare_event_source_models.py`,
  the analysis/audit/reality scripts, and any `docs/*.md` (e.g. `per_bin_rate_results.md` cites
  `models_per_bin_rate/all_models.pkl`). Drive from the audit list.
- **Docstrings/comments** that name the old dirs (e.g. the CLAUDE.md output-file list —
  fully rewritten in phase 6, but any script docstring citing `models_per_bin_rate` updates here).

## Deliberately not in this phase

- Documenting the convention / CLAUDE.md rewrite — that's phase 6 (it reads the new names).
- Re-running any model (rename preserves the pkls; no recompute).
- Renaming the pkl *files* inside the dirs (only the dirs + the un-mix move).

## Validation slice

| Test | Asserts |
| --- | --- |
| `rg -n '"models(_patch|_swr_[TF]|_patch_swr_[TF]|_per_bin_rate|_per_stop_rate)?"' scripts src docs` | zero old dir-name literals remain |
| `uv run python scripts/rate/fit_per_bin_rate_models.py --max-per-animal 1` | writes to `rate_per_bin/`, reads run clean |
| `uv run python scripts/rate/plot_per_bin_rate_summary.py` (**slow**) | reads from `rate_per_bin/` (a reader still finds its data) |
| `ls Processed-Data/replay_content/` | only the descriptive names; no `models*/` remains; `swr_contrast/` + `behavior/` exist |
| `uv run pytest` | green (any test referencing an output dir updated) |

## Fixtures

Uses the existing pkls (moved, not regenerated). No new fixtures.

## Review

Before opening the PR, dispatch `code-reviewer` against the diff. Confirm:
- Every old `models*/` literal is gone from writers, readers, and docs (audit list fully applied).
- The is_swr suffix logic in `fit_replay_content_models.py` emits the new `_swr`/`_nonswr` names
  with an explicit T→swr / F→nonswr mapping.
- `swr_contrast_*` and `qvals_table.pkl` are un-mixed into their own dirs; no dir is still mixed.
- A reader smoke actually finds its data at the new path (not just "writers updated").
- No pkl was regenerated (rename only).
