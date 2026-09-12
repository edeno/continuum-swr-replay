# Phase 6 execution plan — one at-well cohort for the rate and content families

Spec authority: this file (the user's instruction of 2026-09-12: "the rate and content models
really need to be consistent with each other"), `shared-contracts.md` §"Common-cohort freeze for
rate designs" (preserved), and `docs/design/cohort_and_timing_rules.md` (gains the shared rule).

**The defect.** Content models fit only events whose stop is `occupancy_state == "at_well"`
(59,699 of 65,913 events, `scripts/content/fit_replay_content_models.py` ~1692). Rate models fit
every immobility stop (107,400 stops: 69,224 at-well, 14,845 mixed, 22,850 in-transit, 481
unknown at the stop grain). In-transit stops carry a disproportionate share of non-SWR non-local onsets, so the
rate claim "non-SWR replay is not reward-gated" and every content claim about non-SWR replay
describe different populations.

**The decision (binding, amended 2026-09-12 after Task 1's review).** Well-anchored stops are the
shared canonical cohort for BOTH families. At the stop grain that is `occupancy_state ∈ {at_well,
mixed}` (84,069 stops): a `mixed` bout is one whose first settling samples read as in-transit —
every mixed stop begins after the last well arrival (median 0.73 s), sits at 0 cm from the well,
and 97 % of its events are at-well at the event grain. At the event grain the rule stays
`occupancy_state == at_well` (59,699 events); content ⊂ rate up to a documented 1 % residual.
A stop-grain `at_well`-only rule would have dropped 30.6 % of the content cohort's events and made
the two cohorts cross rather than nest. The
all-stops rate fits become a supplementary sensitivity bundle; in-transit stops may be fit
separately as their own bundle but are never pooled with at-well stops.

## Global Constraints

- `uv run …`; `ruff check`/`ruff format --diff` clean on added lines; no plan/phase/task
  references in shipped code/tests/docs/commit messages; nothing under `Processed-Data/`,
  `logs/`, `.superpowers/` committed; commit trailer `Co-Authored-By: Claude Fable 5.1
  <noreply@anthropic.com>` then `Claude-Session: https://claude.ai/code/session_01JGZPceMjeWcjYZQ26byzzz`;
  no `git checkout`/`stash`/`clean`; do not sweep in `anna_yamls*`, `.vscode/`,
  `.claude/docs/plans/finish-covariate-rename/`.
- Never overwrite a canonical fit without first moving it (with sidecars) into
  `Processed-Data/replay_content/_pre_atwell_backup_20260912/`. Never delete data.
- Long runs: smoke first (`--max-per-animal 1`, `/usr/bin/time -v`), extrapolate, then tmux with
  logs under `logs/rate_refit_atwell/`; `--workers 16` (96 cores).
- Every refit artifact carries a fresh provenance sidecar and records the cohort rule it was fit
  under (`extra={"stop_cohort": "at_well"}` or `"all_stops"`).
- Tests value-level; `slow` marks over 20 s.

### Task 1: The shared cohort rule, in one place, used by every runner

Files: `src/continuum_swr_replay/analysis/replay/cohort_selection.py` (`freeze_common_cohort` ~33) —
add `select_stop_cohort(stop_table, *, cohort: str) -> tuple[pd.DataFrame, dict]` with
`cohort ∈ {"at_well", "all_stops", "in_transit"}` keyed by grain: at the stop grain `at_well` keeps
`occupancy_state ∈ {"at_well", "mixed"}` (well-anchored; see the amended decision above) and at the
event grain `occupancy_state == "at_well"`; `in_transit` keeps `occupancy_state == "in_transit"` at
both grains; `all_stops` keeps everything; returns the
frame and a report (`n_stops_before/after`, counts by `occupancy_state`, the rule string). Raise
if `occupancy_state` is absent. `scripts/content/fit_replay_content_models.py` ~1692 and
`scripts/content/fit_swr_contrast_content.py` (find its at-well filter in `_assemble_arm`) call it
with `cohort="at_well"` (no behaviour change — assert the same event set as today, and the content
smoke `--models M0,M0a` byte-identical; the SWR-contrast runner's cohort audit must show the same
`n_in`). `scripts/rate/fit_per_bin_rate_models.py` (design built ~163, `aggregate_to_timebins(`),
`scripts/rate/fit_swr_contrast_rate.py` (`_load` ~515, ~377), `scripts/rate/fit_per_stop_rate_models.py`
(~251-270), `scripts/rate/fit_uncertainty_rate.py` (its `_load`) gain `--stop-cohort
{at_well,all_stops,in_transit}` with default `at_well`, applied to the stop table BEFORE the
common-cohort freeze, logged, and recorded in the row (`stop_cohort` column) and the provenance
`extra`. The all-stops bundle publishes with suffix `_allstops` (per-bin: `all_models_allstops.pkl`
+ sidecars + `per_bin_rate_cohort_audit_allstops.csv`; contrast: `swr_contrast_rate_allstops*.pkl`;
per-stop: `all_models_allstops.pkl`; uncertainty likewise); `in_transit` publishes `_intransit`;
`at_well` publishes the canonical names. `model_comparison.METADATA_COLUMNS` gains `stop_cohort` so
a comparison across cohorts is NaN'd with a legible reason. `docs/design/cohort_and_timing_rules.md`
gains a "Shared stop cohort" section stating the rule, the counts, and why. Tests:
`tests/unit/test_cohort_selection.py::test_select_stop_cohort_*` (each cohort, the raise, the
report); a test that the content runner and the per-bin runner agree on the eligible `stop_id`
set for a synthetic session (`tests/unit/test_shared_stop_cohort.py`); runner tests for the flag
and the suffix; `compare_nested_models` NaNs a cross-cohort pair.

### Task 2: Refits and the comparison audit

Backups first (move `models_per_bin_rate/all_models.pkl`, `per_bin_rate_cohort_audit.csv`,
`swr_contrast_rate*.pkl`, `uncertainty_rate*.pkl`, `models_per_stop_rate/all_models.pkl` and every
sidecar into `_pre_atwell_backup_20260912/<dir>/`). Smoke each runner at `--stop-cohort at_well
--max-per-animal 1`; then in tmux: per-bin at-well (`rate_perbin_atwell`, `--workers 16`, ~30 min);
SWR-contrast rate at-well (`rate_contrast_atwell`, ~20 min); per-stop at-well (~1 min);
uncertainty-rate at-well (~12 min); then the per-bin and SWR-contrast rate ladders again with
`--stop-cohort all_stops` (these regenerate the previous canonical numbers under the new suffix —
verify they reproduce the backed-up at-all-stops fits byte-for-byte except the new `stop_cohort`
column and provenance, which proves the flag is a pure filter); and per-bin `--stop-cohort
in_transit` (`rate_perbin_intransit`, cheaper). Verify every artifact (sessions, rungs, provenance
fresh, `stop_cohort` column). Extend `scripts/qa/compare_fits.py` with a `--cohort-contrast` mode
that prints, per rung and population, the at-well vs all-stops vs in-transit values of: the
immediate reward-component RR (`reward_history_rcos_0` on `M7r`), the paired within-session
SWR-coupled-vs-non-SWR immediate-coefficient test, the `M7r − M6r` and `M1 − M0` ΔAIC steps, the
burst peak, and the retained arrival share; generated audit
`docs/audits/2026-09-12-rate-cohort-atwell.md` committed, with a `Command:` line.

### Task 3: Docs and the claims table

Requote from the new audit: `docs/rate_models.md` (state the shared cohort in the Family A/B
headers; the reward-gating numbers at-well with the all-stops numbers as the sensitivity;
retention by population on the arrival ladder), `docs/per_bin_rate_results.md` (header: cohort =
at-well stops; §1–§6 requoted), `docs/replay_rate_content_swr_synthesis.md` (Layer 1 numbers; one
sentence that both families now share the at-well cohort), `docs/models_table.md` (the
`--stop-cohort` convention and the `_allstops`/`_intransit` bundles), `docs/README.md` ("Where
results live"), `paper/claims_table.md` row 19c (numbers requoted; a dated REVISED note: "cohort
unified to at-well stops, matching rows 19a–19g"; keep RETRACTED annotations), and any other row
quoting a rate number (grep the old values). If the dissociation weakens at wells, say so plainly
in 19c and the synthesis; do not keep the all-stops number as the headline. Retired-name scan and
the mirror test stay green.

## Review

Final review (Fable): the cohort rule is one function used by all six runners; the all-stops
refits reproduce the previous fits (pure-filter proof); every requoted number traces to the new
audit; the at-well reward-gating result is stated with its all-stops sensitivity; no plan/phase
references; suite green.
