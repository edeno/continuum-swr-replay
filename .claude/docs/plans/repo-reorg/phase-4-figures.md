# Phase 4 — Organize figure outputs: topic subdirs in `paper/figures/`, exploratory out of it

[← back to PLAN.md](PLAN.md) · [overview](overview.md#target-structure)

`paper/figures/` currently has **27 loose top-level files** mixed with 14 subdirs, and
several exploratory one-offs write *into* it. Enforce: `paper/figures/` = publication figures
in topic subdirs only; exploratory figures live in `.claude/research/<topic>_outputs/`.
Runs after phase 3 (figure scripts are in `scripts/figures/` and `scripts/rate/` by then;
exploratory one-offs in `scripts/exploratory/`).

**Inputs to read first:**

- [overview.md](overview.md#target-structure) — the `paper/figures/` target + the
  figure-output rule.
- The identified loose-file writers and their savefig idiom (`FIGS = P.parent / "paper" /
  "figures"` where `P = PROCESSED_DATA_DIR`, so **move-safe** — the fix is appending a topic
  subdir or redirecting to `.claude/research/`, not fixing a broken path):
  - `scripts/rate/plot_per_bin_rate_models.py:42,121,145` →
    `reward_kernel_per_bin_rate.png`, `selfhistory_autocorr.png`
  - `scripts/rate/plot_per_bin_rate_summary.py:35` → `per_bin_rate_ladder_summary.png`
  - `scripts/rate/plot_model_checking.py:48,82,147,217-219` →
    `model_checking_ladder.png`, `model_checking_example_{rich,sparse}.png`
  - `scripts/rate/plot_uncertainty_rate.py:172,174` → `uncertainty_rate_by_population.png`
  - `scripts/exploratory/_session_raster.py:184,230-233` →
    dynamic `session_raster_*.png` names (exploratory)
  - `scripts/exploratory/_timeresolved_psth.py` → `rate_timecourse_vs_pump.png` (exploratory)
  - `scripts/exploratory/_event_rasters.py` → `event_rasters_triptych.png` (exploratory)

## Tasks

- **Redirect the exploratory one-off writers out of `paper/figures/`**: `_timeresolved_psth.py`,
  `_event_rasters.py`, and `_session_raster.py` (now in `scripts/exploratory/`) must write to
  `.claude/research/<topic>_outputs/` (e.g. `rate_psth_outputs/`), matching the convention
  the `_fig2b_*`/`_peri_departure` scripts already follow. Move their existing outputs
  (`rate_timecourse_vs_pump.png`, `event_rasters_triptych.png`, `session_raster_*.png`) there
  too, unless a manuscript/figure index proves a specific raster is publication material.
- **Point the publication plot scripts at topic subdirs**: `plot_per_bin_rate_models.py`,
  `plot_per_bin_rate_summary.py`, `plot_model_checking.py`, and `plot_uncertainty_rate.py`
  write into `FIGS / "rate"` (create the subdir). Adjust each `savefig` target accordingly.
  For scripts that pass filenames dynamically (e.g. `plot_model_checking.py` example quartets),
  update the base output directory before moving the current files.
- **File the 27 loose top-level `paper/figures/*.png` into the right destination.** For each
  loose file, `rg -n "<basename>" scripts docs paper src README.md CLAUDE.md` to find its
  writer and references; update the writer first, then move the current file into either the
  matching publication subdir (`rate/`, `content/`, `reality/`, `quality/`, or a new one) or
  the appropriate `.claude/research/<topic>_outputs/` directory for exploratory outputs.
  Publication main-figure panels already live in `figure0N/` — leave those.
- **Resolve true orphans** (`reward_glm_effects.png`, `reward_kernel_timeresolved.png`,
  `stop_duration_distribution.png`, and any file whose basename grep has no writer):
  grep the whole repo for the basename; if a renamed/exploratory script writes it, file by
  that topic and update the writer; if truly stale (no writer, not referenced by any
  figure-index or manuscript), list for deletion in the PR body (don't delete silently).
  Do **not** treat `model_checking_example_rich.png`/`sparse.png` as orphans — they are
  written dynamically by `plot_model_checking.py`.
- **`paper/figures/scratch/`**: move to `.claude/research/` (scratch is exploratory, not
  publication) — or, if the manuscript references it, keep and document why.
- **Update `docs/README.md`** "Where results live" (from phase 3/5) to add the figure-output
  tier: `paper/figures/<topic>/` = publication; `.claude/research/*_outputs/` = exploratory.

## Deliberately not in this phase

- The main-text `figure0N/` panels and the `supp_*`/`qc_diagnosis` subdirs — already
  correctly filed; only the *loose* files and *exploratory* outputs move.
- Renaming `Processed-Data/…/models*/` data dirs (phase 5 documents those, never renames).
- CLAUDE.md "Project Structure" rewrite — that's the phase-5 capstone (this phase's figure
  convention feeds into it).

## Validation slice

| Test | Asserts |
| --- | --- |
| `find paper/figures -maxdepth 1 -type f` | empty — no loose files remain at the top level |
| `uv run python scripts/rate/plot_model_checking.py` (needs fitted models; **slow**) | writes into `paper/figures/rate/`, not the top level |
| `grep -rn "paper.*figures" scripts/exploratory/_timeresolved_psth.py scripts/exploratory/_event_rasters.py scripts/exploratory/_session_raster.py` | no `paper/figures` target remains (redirected to `.claude/research/`) |
| `rg -n 'paper.*figures|FIGS' scripts/rate scripts/exploratory` | every hit appends a publication topic subdir or redirects exploratory output to `.claude/research/*_outputs/`; no save target writes directly to top-level `paper/figures` |
| manual: `ls paper/figures/` | only subdirs (main `figure0N/`, `supp/`-style, topic dirs); orphans resolved |

Mark the plot-script re-runs **slow** (they read fitted-model pkls).

## Fixtures

Uses existing fitted-model pkls under `Processed-Data/replay_content/models*/`. No new fixtures.

## Review

Before opening the PR, dispatch `code-reviewer` against the diff. Confirm:
- `paper/figures/` has zero loose top-level files; publication outputs are in topic subdirs
  and exploratory outputs are under `.claude/research/*_outputs/`.
- The publication plot scripts write into `paper/figures/<topic>/`; the exploratory
  scripts write into `.claude/research/*_outputs/`.
- Orphans are resolved (filed or listed-for-deletion with evidence), not left dangling.
- `docs/README.md` names the figure-output tier.
- No manuscript/figure-index reference to a moved figure path was left stale (grep `paper/`
  and `docs/` for the moved basenames).
