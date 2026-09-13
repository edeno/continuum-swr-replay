# PFC analyses on the shared well-anchored stop cohort

**Status:** Not started.

The user's instruction (2026-09-13): "we also need the pfc analysis to be only on the at well
stops." The hippocampal rate and content families fit one shared cohort
(`analysis/replay/cohort_selection.py`: stop grain `at_well` = `occupancy_state ∈ {at_well, mixed}`,
84,069 of 107,400 stops; event grain `at_well` = `occupancy_state == at_well`, 59,699 of 65,913
events; see `docs/design/cohort_and_timing_rules.md` §"Shared stop cohort"). The three analyses
under `scripts/pfc/` do not:

- `scripts/pfc/run_pfc_analysis_bandit.py` (mPFC unit GLMs, Figure 4 / claim 20) fits every
  immobile 2 ms bin in the epoch (`analysis/pfc/glm.py` `run_session` ~1021–1064,
  `_fit_immobility_models` ~472) and defines its own events (`get_confident_times` +
  `build_speed_eligible_event_cohort` + `get_overlap_times`) — no stop, no occupancy.
- `scripts/pfc/run_stage6_learning.py` (stop grain) pools every stop (`stop_replay_summary(st, ev, ea)`
  at ~104, cached to `replay_content/stop_replay_summary.pkl`).
- `scripts/pfc/run_stage5b_retrospective.py` (event grain) uses `retrospective.at_well_event_cohort`
  (`at_well is True` AND finite `occupancy_well_arm`, plus the three-table key alignment) — the same
  events as `select_event_cohort` today, but a second rule.

`scripts/figures/generate_figure04.py` ~337–380 duplicates `run_session`'s event/mask construction
for panel a. `Processed-Data/*_mPFC_modulation.{pkl,*.npy}` (71 sessions × 5 files, April 2026,
no provenance sidecars) are the current Fig 4 inputs; OFC was never run and is not run here.
The last mPFC session log shows ~5.3 h wall-clock per session (`logs/wilbur20210331_06_r3_mPFC.log`);
with 16 workers the full rerun is roughly a day — smoke and extrapolate before launching.

## Decisions (binding)

1. **PFC GLM cohort = bins inside well-anchored stops, events inside well-anchored stops.** The fit
   mask becomes `is_immobile & in_cohort_stop`, where `in_cohort_stop` marks position-time samples
   that fall inside a `[stop_start, stop_end]` interval of a stop kept by
   `select_stop_cohort(stop_table_session, cohort=...)`. The non-local and ripple event sets that
   enter the design are restricted to events whose onset lies inside a kept stop. The regressor
   basis still spans ±250 ms around each kept event; bins outside the cohort are simply not fit.
2. `--stop-cohort {at_well,all_stops}` on all three runners, default `at_well`; `at_well` publishes
   the canonical artifact names, `all_stops` the `_allstops` suffix. `in_transit` is not offered
   for PFC (no power at the unit level; can be added later).
3. Stage 5B keeps its finite-`occupancy_well_arm` conjunct and its three-table key alignment, but
   its at-well membership comes from `select_event_cohort` (one rule).
4. Every PFC artifact gets a provenance sidecar (`fit_provenance.write_provenance` pattern, as the
   rate runners do) recording `stop_cohort`, the stop-table hash, bins before/after the mask, and
   events before/after.
5. Existing artifacts are never deleted: move to `Processed-Data/_pre_atwell_pfc_backup_20260913/`
   (the 355 mPFC files and their logs) and `Processed-Data/replay_content/_pre_atwell_pfc_backup_20260913/`
   (Stage 5B/6 pickles) before the first write.
6. Out of scope (parked): `choice_prediction.build_choice_dataset` uses strict `at_well` at the
   stop grain (claim 19d) — a fourth rule, content-side, refit + requote needed; the
   `pfc-fig4-recut` plan (canonical events, matched control) — this plan's helper must be reusable
   by it, nothing else from it is pulled forward; OFC.

## Global Constraints

- `uv run …`; `ruff check`/`ruff format --diff` clean on added lines; no plan/phase/task references
  in shipped code/tests/docs/commit messages; nothing under `Processed-Data/`, `logs/`,
  `.superpowers/`, `paper/figures/**/*.png` committed; commit trailer `Co-Authored-By: Claude Fable 5.1
  <noreply@anthropic.com>` then `Claude-Session: https://claude.ai/code/session_01JGZPceMjeWcjYZQ26byzzz`;
  no `git checkout`/`stash`/`clean`; stage by name; never stage `anna_yamls*`, `.vscode/`,
  `.claude/docs/plans/finish-covariate-rename/`.
- Import layering test (`tests/unit/test_import_layering.py`) stays green: put the generic
  interval mask in `analysis/replay/cohort_selection.py`; `analysis/pfc/` may import from
  `analysis/replay/`.
- Long runs: smoke one session with `/usr/bin/time -v`, extrapolate, then tmux (session names below)
  with logs under `logs/pfc_atwell/`. Never overwrite a canonical artifact without moving it first.
- Tests value-level; `slow` marks over 20 s.

### Task 1: One cohort helper on the PFC GLM path and on Figure 4 panel a

Files: `src/continuum_swr_replay/analysis/replay/cohort_selection.py` — add
`stop_interval_mask(times: np.ndarray, stops: pd.DataFrame) -> np.ndarray` (bool, `(n_time,)`;
True where `stop_start <= t <= stop_end` for any row; vectorized via sorted starts/ends +
`np.searchsorted`; asserts non-overlapping stops) and
`events_in_stops(event_starts: np.ndarray, stops: pd.DataFrame) -> np.ndarray` (bool per event,
onset inside a kept stop). New `src/continuum_swr_replay/analysis/pfc/cohort.py` —
`load_session_stop_cohort(nwb_file_name, epoch_name, *, cohort, stop_table_path=None) ->
tuple[pd.DataFrame, dict]`: reads `Processed-Data/replay_content/stop_table.parquet` (via
`table_io.read_table`), restricts to the session's `(animal, session_date, epoch)` (derive the key
the way `data_loaders`/`session_keys` do from `nwb_file_name` like `chimi20200212_.nwb` and
`epoch_name` `02_r1`; raise if the session has no stops), applies `select_stop_cohort`, returns
the kept stops and its report plus the stop-table sha256. `analysis/pfc/glm.py` `run_session` —
new keyword `stop_cohort: str = "at_well"` and `cohort_stops: pd.DataFrame | None`; after
`is_immobile` (~1064) compute `in_cohort = stop_interval_mask(position_info.index.to_numpy(),
cohort_stops)` and set `fit_mask = is_immobile & in_cohort`; restrict `non_local_times` and
`ripple_times` (~1027–1041) with `events_in_stops` BEFORE `get_overlap_times`; log and return in
the per-session result dict `n_bins_immobile`, `n_bins_fit`, `n_nonlocal_before/after`,
`n_ripple_before/after`, `stop_cohort`. For `all_stops` the mask is all-True over stops (so a
bin outside every stop is still dropped? — NO: `all_stops` must reproduce today's behaviour
exactly: `fit_mask = is_immobile`, events unrestricted; implement `all_stops` as "no cohort
restriction" and say so in the docstring). `scripts/pfc/run_pfc_analysis_bandit.py` —
`--stop-cohort {at_well,all_stops}` default `at_well`; suffix `_allstops` on every output name
for `all_stops`; provenance sidecar `{epoch_identifier}_modulation[.suffix].provenance.json`
via `fit_provenance` with `extra={"stop_cohort": …, "stop_table_sha256": …, counts…}`;
`results_exist` checks the suffixed name. `scripts/figures/generate_figure04.py` ~337–380 —
replace the duplicated construction with a call into the same `run_session` sub-steps (extract
`build_session_event_sets(data, results, *, cohort_stops, stop_cohort)` from `run_session` so
both call one function) and add `--stop-cohort` to the figure script (default `at_well`, reading
the canonical artifacts). Tests: `tests/unit/test_cohort_selection.py` (`stop_interval_mask`
boundaries, non-overlap assert, empty stops → all False; `events_in_stops`),
`tests/analysis/pfc/test_glm_pipeline.py` (a stubbed `run_session` on the synthetic fixture with
two stops, one at-well and one in-transit: the fit mask equals `is_immobile & inside the at-well
stop`; the events outside are dropped; `all_stops` reproduces the pre-change mask bit-for-bit),
`tests/unit/test_run_pfc_analysis_bandit.py` (argparse default; suffix; sidecar written with
`stop_cohort`). Report the smoke on one session (`--max-sessions 1` or equivalent — add the flag
if missing) with `/usr/bin/time -v`, to a scratch `--output-dir` (add the flag if missing),
comparing bins/events before/after the mask.

### Task 2: Stage 6 and Stage 5B on the shared rule

Files: `scripts/pfc/run_stage6_learning.py` — `--stop-cohort {at_well,all_stops}` default
`at_well`; `select_stop_cohort(stop_table, cohort=…)` BEFORE `stop_replay_summary` (~104); the
cached summary and every output carry the `_allstops` suffix for `all_stops` (canonical names for
`at_well`); a `stop_cohort` column on the per-session and pooled result frames; provenance
sidecars. `src/continuum_swr_replay/analysis/replay/retrospective.py` `at_well_event_cohort`
(~134–186) — membership from `select_event_cohort(event_table, cohort="at_well")` then the finite
`occupancy_well_arm` conjunct and the three-table alignment unchanged; delete the docstring
sentence saying it is on its own path; `scripts/pfc/run_stage5b_retrospective.py` gains
`--stop-cohort {at_well}` (only value; documents the rule) and a provenance sidecar. Tests:
`tests/unit/test_fdr_nan_pvalue.py`/a new `test_run_stage6_learning.py` (cohort applied: a
synthetic stop table with in-transit stops → they are absent from the summary; suffix; column),
`tests/unit/test_retrospective.py` (existing cohort tests stay green; add one asserting
`at_well_event_cohort` and `select_event_cohort` keep the same `event_id` set on a fixture with
`mixed`/`unknown` events). Then back up (`Processed-Data/replay_content/_pre_atwell_pfc_backup_20260913/`)
and rerun Stage 6 (`--rebuild-summary`, all strata that exist today) and Stage 5B (all three
variants; smoke one session first and record the runtime); verify outputs + sidecars. Write
`docs/audits/2026-09-13-stage5b-stage6-atwell.md`: per-stratum n stops / n events before vs after,
and the Stage 6 per-role FDR outcome before vs after (the claim 19e "null everywhere" must be
re-checked and stated).

### Task 3: mPFC refit, Figure 4, audit, docs

Move the 355 `Processed-Data/*_mPFC_modulation*` files and the 71 `logs/*_mPFC.log` into the
backup dir. Launch in tmux `pfc_atwell` (`--n_workers 16`, logs under `logs/pfc_atwell/`); the
Task-1 smoke's runtime × 71 / 16 is the budget — record it. Verify 71 `*_mPFC_modulation.pkl`
+ 4 npy each + 71 sidecars; `load_population_data` loads them. Regenerate Figure 4
(`scripts/figures/generate_figure04.py`, output `paper/figures/figure04/`, not committed).
Write `docs/audits/2026-09-13-pfc-atwell-cohort.md` (generated by a small
`scripts/qa/compare_pfc_cohorts.py` with a `Command:` line): per session bins fit before/after,
events before/after; population summary before (backup dir) vs after: n neurons passing
`min_firing_rate`, FDR-significant fraction per test (`SWR`, `no_SWR`, `non_local_SWR`,
`non_local_no_SWR`), per animal. Docs: `docs/README.md` "Where results live" gains the PFC
rows (canonical + `_allstops`, Stage 5B/6 outputs) and the cohort sentence; `docs/design/cohort_and_timing_rules.md`
§"Shared stop cohort" states the PFC bin-level rule and the event-onset rule;
`paper/claims_table.md` row 20 states the cohort (and a dated REVISED note if any number is
quoted there — today none is) and row 19e requoted from the Stage 6 audit with a dated note;
`scripts/README.md` `pfc/` entries mention `--stop-cohort`; `docs/regression_pipeline_map.md`
regenerated (`scripts/qa/write_pipeline_map.py`). Retired-name, mirror, map tests green; full
`uv run pytest tests/unit -q`.

## Review

Final review (Fable): one cohort rule reaches all three PFC runners and Figure 4 through the
shared selector; `all_stops` reproduces the pre-change mask/events bit-for-bit (test) and the
Stage 6 `_allstops` outputs match the backed-up pickles; every number in the audits/docs traces;
no artifacts deleted; no plan references; suite green.

## Deliberately not in this plan

Canonical events / matched control for the PFC GLM (`pfc-fig4-recut`); OFC; the
`choice_prediction` strict at-well rule (claim 19d); `in_transit` PFC bundles.
