# Stage 5B and Stage 6 on the shared well-anchored cohort

Written 2026-09-13, after moving the two stop/event-grain analyses under
`scripts/pfc/` onto the shared selector in
`analysis/replay/cohort_selection.py` and rerunning every variant that had a
shipped artifact.

- **Stage 6** (`scripts/pfc/run_stage6_learning.py`, stop grain) pooled every
  immobility stop in the epoch. It now applies
  `select_stop_cohort(stop_table, cohort="at_well")` before the per-stop
  summary, which is the population the hippocampal rate and content ladders are
  read on. `--stop-cohort all_stops` reproduces the previous population as a
  sensitivity bundle under `_allstops` names.
- **Stage 5B** (`scripts/pfc/run_stage5b_retrospective.py`, event grain) kept a
  second statement of the at-well rule inside
  `retrospective.at_well_event_cohort`. Membership now comes from
  `select_event_cohort(event_table, cohort="at_well")`; the finite
  `occupancy_well_arm` conjunct and the three-table key alignment are unchanged.
  `--stop-cohort` accepts only `at_well` — the event grain has one rule, and the
  flag exists so the artifact states it.

Every artifact these two scripts previously shipped was **moved, not deleted**,
into `Processed-Data/replay_content/_pre_atwell_pfc_backup_20260913/` before the
first write (12 pickles, listed at the end). Run logs are under
`logs/pfc_atwell/`. Nothing under `Processed-Data/` or `logs/` is committed.

## Stage 6 — stop grain

### Cohort

| | stops | rule |
| --- | --- | --- |
| before (`all_stops`) | 107,400 | no occupancy restriction |
| after (`at_well`) | 84,069 | `occupancy_state` in {`at_well`, `mixed`} |

Dropped: 23,331 stops — 22,850 `in_transit` and 481 `unknown`. The kept set is
69,224 `at_well` + 14,845 `mixed`, exactly the counts
`docs/design/cohort_and_timing_rules.md` records for the shared cohort.

Per-role usable stops (the stops surviving `dropna` on the role's content
column), `all_stops` → `at_well`, stratum `all`:

| role | all_stops | at_well |
| --- | --- | --- |
| current | 21,209 | 21,209 |
| prev | 23,965 | 21,459 |
| last_rewarded | 23,963 | 21,453 |
| best | 24,225 | 21,681 |
| current_patch | 21,209 | 21,209 |
| prev_patch | 23,965 | 21,459 |
| last_rewarded_patch | 23,963 | 21,453 |
| best_patch | 24,225 | 21,681 |

`current`/`current_patch` were already at-well-only: `frac_on_current` is
undefined off-well by construction (`is_current_arm` is NaN there), so the
current-arm models never saw an in-transit stop.

### Every fitted number is unchanged

Command: `uv run python scripts/qa/summarize_stage6_fdr.py <bundle>.pkl`

On the current tables, the `at_well` and `all_stops` bundles are **identical**:
all four `stage6_is_stay_logistic*` frames agree row-for-row on every column
except the new `stop_cohort`, and all four pooled-GEE frames agree on every
coefficient, SE, p-value and `n_stops`.

The reason is checkable and worth recording: of the 23,331 dropped stops,
**`is_stay` is non-missing for none of them** (0 of 23,331). A stop the animal
was not standing at a well for has no stay/leave decision, so the response was
already undefined and the model's own complete-case rule already removed it.
2,573 of the dropped stops did carry replay events and 2,506 a defined
`frac_on_prev`, but not one of those had a response to fit.

So the cohort change at Stage 6 is a change in what the bundle *states*, not in
what it estimates. That is the useful outcome: the population is now explicit,
named on every artifact, and cannot drift from the rate/content population by
an edit to one script.

### Claim 19e — "null everywhere" re-checked

Still null everywhere, in all four strata, on both estimators.

Per-session Wilcoxon on β_inter across converged sessions, FDR across the eight
roles (stratum `all`, cohort `at_well`; `p_FDR` column):

| role | n sessions | median β_inter | pos/neg | p | p_FDR |
| --- | --- | --- | --- | --- | --- |
| current | 10 | −4.404 | 2/8 | 0.0195 | 0.156 |
| prev | 5 | +9.563 | 3/2 | 0.3125 | 0.568 |
| last_rewarded | 8 | −2.288 | 3/5 | 0.3828 | 0.568 |
| best | 7 | −0.109 | 3/4 | 0.6875 | 0.782 |
| current_patch | 13 | +0.856 | 9/4 | 0.4143 | 0.568 |
| prev_patch | 14 | −1.391 | 6/8 | 0.4263 | 0.568 |
| last_rewarded_patch | 12 | −1.412 | 5/7 | 0.1294 | 0.518 |
| best_patch | 16 | +1.183 | 9/7 | 0.7820 | 0.782 |

Smallest FDR-adjusted p-value across every stratum and estimator:

| stratum | per-session Wilcoxon (min p_FDR) | pooled GEE (min p_FDR) |
| --- | --- | --- |
| all | 0.156 (`current`) | 0.906 |
| swr | 1.000 (`last_rewarded_patch`; the other seven roles have <5 converged sessions) | 0.763 |
| non_swr | 0.661 (`prev_patch`) | 0.941 |
| mua | 1.000 | 0.749 |

Nothing approaches 0.05. Claim 19e stands as stated, now on the shared cohort.

**The per-session test is thin in the stratified runs.** In `swr`, `non_swr` and
`mua`, most roles have fewer than five converged sessions and are not tested at
all (the per-session logistic fails or is skipped for the rest: 1,744 rows per
bundle; 85 converged in `all`, 25 in `swr`, 44 in `non_swr`, 29 in `mua`, out
of 1,744 session x role fits each). The pooled GEE, which uses all 218 sessions, is the
estimator to read there; the per-session column is an absence of power, not
evidence of absence.

### Comparison with the backed-up bundles

The backed-up `stage6_is_stay_logistic.pkl` (fit 2026-09-02) does **not** match
the new `all_stops` bundle: 388 of 1,744 rows differ in `n_stops` and 191 in
`status`. That difference is **not** the cohort, and not this change. It is the
v3 → v4 behavior-cache reclassification that landed after that fit:

- the new `all_stops` per-stop summary matches the backed-up
  `stop_replay_summary.pkl` on all 48 columns and all 107,400 rows **except**
  `is_rewarded_outcome` and `is_omitted_outcome`;
- those differ on exactly 1,361 stops, and every one is `0.0` (omission) in the
  backup and NaN now — the reclassification recorded in the project memory as
  "v3→v4 reclassifies 1361 stops/12 events omission→NaN".

Since `is_omitted_outcome` is the model's other predictor, those 1,361 stops
leave the fits, which is what moves `n_stops` and `status`. Every replay-content
column is identical.

The `_swr`, `_non_swr` and `_mua` backups date from **2026-06-04**, carry the
retired `day` column instead of `session_date`, and predate the v5 DIO-photogate
regeneration entirely. They are stale for reasons unrelated to the cohort and
are not comparable to anything; they were rerun rather than compared.

## Stage 5B — event grain

### Cohort: the same events, one rule

| | events |
| --- | --- |
| before (`at_well` flag) | 59,699 of 65,913 |
| after (`select_event_cohort`) | 59,699 of 65,913 |

The two predicates select the same events on the canonical table, verified
row-wise: `at_well.eq(True).fillna(False)` and `occupancy_state == "at_well"`
disagree on **0 of 65,913** events, and all 59,699 have a finite
`occupancy_well_arm` (so the retained conjunct drops nothing on this table). The
backed-up run's own counters agree: `n_events_input_arm_family` =
`n_events_after_at_well_arm_family` = 59,699.

### The bundle is unchanged

The `all` variant's 218 rows are **identical to the backed-up run on all 77
shared columns** — every metric, every shuffle-null summary, every retained /
excluded counter — with the single new column `stop_cohort` added and
`elapsed_s` excluded from the comparison. The per-session status distribution is
identical too (160 `partial_movable_families`, 35 `ok`, 23 `no_movable_family`).

| metric | real (median) | Δ vs shuffle | sessions with Δ>0 |
| --- | --- | --- | --- |
| chosen_arm_fraction | 0.0744 | +0.0017 | 109/194 |
| ends_at_next | 0.0759 | +0.0019 | 110/194 |
| contains_path_to_next | 0.0000 | +0.0000 | 0/35 |
| chosen_patch_fraction | 0.4199 | +0.0034 | 115/195 |
| ends_at_next_patch | 0.4118 | +0.0012 | 107/195 |

Before and after are the same numbers; the table is the after.

### The two subset variants

`--swr-only` and `--non-swr-only` were rerun because their shipped files dated
from **2026-06-04**, carried 27 columns against today's 79, and predate both the
visit-context shuffle calibration and the v5 DIO-photogate regeneration. Their
old numbers are not comparable to the new ones and are not a before/after of
this change — for example the old `swr_only` `chosen_arm_fraction` Δ was −0.0395
(14/218 sessions positive) against +0.0007 (86/166) now, which is the null
changing, not the cohort.

| variant | at-well events | modelled (arm family) | `ok` / `partial` / `none` |
| --- | --- | --- | --- |
| all | 59,699 | 25,275 | 35 / 160 / 23 |
| swr_only | 25,470 | 8,614 | 20 / 148 / 50 |
| non_swr_only | 34,229 | 13,662 | 26 / 157 / 35 |

The two subsets partition the cohort exactly (25,470 + 34,229 = 59,699).

### Runtimes

Smoke (one session, scratch output): **28.1 s** wall, 2.6 GB max RSS, of which
~14 s is table loading and 18.1 s the session. Full runs, in tmux
`stage5b_atwell`: `all` **21 min**, `swr_only` **15 min**, `non_swr_only`
**17 min**, all exit 0, 218 sessions each.

The `all` variant was run twice (the second time so that all three sidecars name
one tree). The two runs are identical on every column but `elapsed_s`, which is
the shuffle seeding behaving as documented.


## Artifacts

Written by this sitting (`Processed-Data/replay_content/`), each with a
`*.provenance.json` sidecar recording the cohort, the cohort report and the
SHA-256 of every table read:

| artifact | population |
| --- | --- |
| `stop_replay_summary.pkl` | at_well (84,069 rows) |
| `stop_replay_summary_allstops.pkl` | all_stops (107,400 rows) |
| `stage6_is_stay_logistic{,_swr,_non_swr,_mua}.pkl` | at_well |
| `stage6_is_stay_pooled_gee{,_swr,_non_swr,_mua}.pkl` | at_well |
| `stage6_is_stay_logistic{,_swr,_non_swr,_mua}_allstops.pkl` | all_stops |
| `stage6_is_stay_pooled_gee{,_swr,_non_swr,_mua}_allstops.pkl` | all_stops |
| `stage5b_retrospective{,_swr_only,_non_swr_only}.pkl` | at_well |

Moved into `_pre_atwell_pfc_backup_20260913/` beforehand:
`stop_replay_summary.pkl` (2026-09-02),
`stage6_is_stay_logistic.pkl` / `stage6_is_stay_pooled_gee.pkl` (2026-09-02),
`stage6_is_stay_{logistic,pooled_gee}_{swr,non_swr,mua}.pkl` (2026-06-04),
`stage5b_retrospective.pkl` (2026-09-02),
`stage5b_retrospective_{swr_only,non_swr_only}.pkl` (2026-06-04).
