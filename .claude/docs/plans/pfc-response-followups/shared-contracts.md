# Shared contracts

[← back to PLAN.md](PLAN.md)

1. [Event classes and curves](#event-classes-and-curves) — phases 1, 2, 4.
2. [Response window and unit sets](#response-window-and-unit-sets) — phases 1, 2, 3b, 4.
3. [Bundle trees](#bundle-trees) — phases 2, 3b, 4.
4. [Reported-value keys](#reported-value-keys) — phases 1, 4.
5. [Commits and GPU runs](#commits-and-gpu-runs) — every phase.

## Event classes and curves

| class | definition | curve read for it | model the curve comes from |
|---|---|---|---|
| `non_local_SWR` ("SWR-coupled replay") | canonical non-local event, `is_swr` true | `non_local_SWR` | `non_local_SWR + non_local_no_SWR` (block 1); in the extended tree also the three-class model below |
| `non_local_no_SWR` ("non-SWR replay") | canonical non-local event, `is_swr` false | `non_local_no_SWR` | `non_local_SWR + non_local_no_SWR` (block 0) |
| `SWR` ("all SWRs") | Kay SWR interval from `load_session_swr_intervals` | `SWR` | `SWR + non_local_no_SWR` (block 0) |
| `SWR_no_non_local` ("SWRs without replay") | a kept, cohort-restricted Kay SWR whose open interval overlaps no canonical non-local event of the session (cohort or not) and no artifact-dropped event window | `SWR_no_non_local` | `non_local_no_SWR + non_local_SWR + SWR_no_non_local` (block 2) |

Semantics, do not weaken:

- Overlap is strict positive-duration overlap of open intervals (`intervals.overlaps_open`), the convention of `is_swr`.
- `SWR_no_non_local` is a subset of `SWR`, so `make_matched_control_times` and every existing class's null see the same avoided windows with or without it.
- A kept SWR that overlaps an artifact-dropped event window has an unknown class and is left out of `SWR_no_non_local` (it stays in `SWR`); the count is recorded as `n_swr_no_non_local_unknown`.
- Onsets are each class's own `start_time`: an SWR onset and a non-local onset differ by a few ms (Part 2 `start_offset_median_ms`).

## Response window and unit sets

- Curves are read on the ±200 ms report window (`REPORT_HALF_WINDOW_S`, 201 lags at 500 Hz), never on the ±250 ms fit window. The per-unit `*_max_mod_time` columns in the bundles are taken on the fit window and are **not** used.
- "Called for class k" = `HELDOUT_P_COLUMNS[k] < 0.05` (raw empirical p against class k's own matched null), the Figure 4 rule. The population is Figure 4's (`load_population_data`, 1 Hz gate).
- Peak = lag of the largest `|curve|` on the report window, as `peak_log_modulation`; the peak sign splits enhanced from suppressed units.
- Onset (default, see overview Open Questions 2): from the peak, walk toward negative lags while the curve keeps the peak's sign and `|curve| >= 0.5 * |peak|`; the onset is the earliest lag reached. A unit whose run reaches the window edge (−200 ms) has onset `-0.2` and is counted as `onset_at_edge`.

## Bundle trees

| directory | written by | contents |
|---|---|---|
| `Processed-Data/` (`*_mPFC_modulation.*`) | canonical runner, default flags | Figure 4 bundles; read only |
| `Processed-Data/pfc_response_followups/timing_calibration.json` | `scripts/pfc/calibrate_pfc_kernel_timing.py` | phase 1 smoothing reference |
| `Processed-Data/pfc_response_followups/swr_without_replay/` | canonical runner, `--event-set with_swr_without_replay --output-dir <this>` | phase 2 bundles, same file names as canonical plus `.SWR_no_non_local_modulation.npy` and `.non_local_SWR_3class_modulation.npy` |
| `Processed-Data/pfc_marked_model_family_v1_single_intercept/` | rename of the 2026-09-19 corpus | contract 1, evidence only |
| `Processed-Data/pfc_response_followups/marked_family/` | marked-family runner, contract 2 | phase 3b joint and marginal bundles and slope nulls |

Every sidecar in a new tree records `event_set` (phase 2) or `bundle_contract_version` (phase 3); readers refuse a directory that mixes values.

## Reported-value keys

- One summary file: `paper/reported/part4.json`, written by `src/continuum_swr_replay/reported/part4.py` through `uv run python scripts/reported/summarize_part.py --part 4`, then `uv run python scripts/reported/emit_reported_values.py`. Each new block is one function `<topic>_values(..., make) -> list[Value]` added to `summarize`, as `overlap_and_effect_values` (`part4.py:809`) is.
- Keys hold no digits (LaTeX macro rule): class words are `swr`, `swr_replay`, `non_swr_replay`, `swr_no_replay`, `matched_control`; animals by `ANIMAL_WORDS`. Prefixes: `timing_`, `swr_no_replay_`, `marked_`.
- Times are stored in ms (`unit="ms"`), fractions as fractions (`style="pct"`).
- `overview.md` and docs get markers `<!--v:part4.<key>-->…<!--/v-->` only; numbers are never typed, in docs, commit messages or the ledger.

## Commits and GPU runs

- Explicit pathspecs: `git commit -m "..." -- <paths>` so a concurrent worker's staged files are never swept in. Never `git add` the symlinks `Processed-Data`, `logs`, `.claude/research`, `paper/figures/<dir>`.
- Messages end with the two attribution lines given in the task.
- GPU runs: `nvidia-smi` first; never GPU 0; one tmux session per batch named `pfc_<phase>_<what>`, one window per card, logs under `logs/pfc_response_followups/<run>/`; record the session name, log path, commit and start time in [ledger.md](ledger.md) before launching.
