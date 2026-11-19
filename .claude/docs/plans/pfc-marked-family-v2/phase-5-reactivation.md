# Phase 5 — Does the mPFC population reactivate the replayed place?

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Why: every content result so far is relative to the animal. The sharper question is whether mPFC represents *which* place the hippocampus replays. mPFC units have well tuning during behavior, so the test is whether the population pattern in the 200 ms after a replay of arm X resembles the population pattern the animal shows while sitting at well X. Independent of phases 1–3; CPU only.

**Inputs to read first:**

- [src/continuum_swr_replay/analysis/pfc/marked_glm.py:307-437](../../../../src/continuum_swr_replay/analysis/pfc/marked_glm.py#L307) — `prepare_marked_session_context`: fitted bins, counts, both replay frames with `event_id`, `start_time`.
- [src/continuum_swr_replay/analysis/pfc/marked_model_family.py:472-510](../../../../src/continuum_swr_replay/analysis/pfc/marked_model_family.py#L472) — `_validate_event_arm_geometry` (nine rows per event, `candidate_arm` 0–8, `arm_fraction` sums to one); reuse it, and see `:609` for how the table is read.
- [src/continuum_swr_replay/analysis/pfc/cohort.py:32](../../../../src/continuum_swr_replay/analysis/pfc/cohort.py#L32) — stops with `occupancy_well_id`.
- [src/continuum_swr_replay/analysis/pfc/canonical_events.py:179](../../../../src/continuum_swr_replay/analysis/pfc/canonical_events.py#L179) — SWR intervals, excluded from templates.
- `CLAUDE.md` "Spatial Bandit Task Structure" — leaf arm *i* is well *i* for *i* in 0–5; trunk arms 6–8 belong to no well.
- [src/continuum_swr_replay/analysis/pfc/heldout.py:407](../../../../src/continuum_swr_replay/analysis/pfc/heldout.py#L407) — `empirical_pvalue` for the plus-one tail fraction.

**Contracts referenced:**

- [Bundle directory suffixes](shared-contracts.md#bundle-directory-suffixes) — `Processed-Data/pfc_replay_reactivation/`.

## Tasks

- **Module** `src/continuum_swr_replay/analysis/pfc/replay_reactivation.py` with four pure functions:
  - `well_templates(counts, fit_mask, cohort_stops, exclusion_mask, *, fold_stops)`: for each well, the mean count vector `(n_units,)` over fitted bins that lie in the given fold's stops at that well and are not excluded (bins within ±250 ms of any replay onset or inside an SWR interval). Returns templates `(n_wells, n_units)`, per-unit mean and SD over the same bins (for z-scoring), and the bin count per well. A well with fewer than 500 bins (1 s) in the fold has no template.
  - `event_vectors(counts, fit_mask, onsets, *, window=(0.0, 0.2))`: summed counts in the window after each onset, `(n_events, n_units)`, divided by window length; events whose window leaves the fitted bins are dropped and counted.
  - `reactivation_statistic(event_vectors, templates, leaf_fractions, current_well, *, unit_mean, unit_sd, min_off_current_mass=0.5)`: z-score vectors and templates with the template-period stats (units with `unit_sd == 0` are excluded from the correlation); `s[e, w]` = Pearson correlation across units between event *e* and template *w*; with `f` the leaf fractions renormalized over wells that have a template and are not the current well, the statistic is `Σ_w f[e, w]·s[e, w] − mean_{w ≠ current} s[e, w]`. Events with off-current leaf mass below `min_off_current_mass`, or with no current well, are excluded and counted. Also return the arg-max well of `s[e, ·]` over the same candidate set and whether it equals the arg-max of `f[e, ·]` (a decoder accuracy against chance `1 / n_candidates`).
  - `reactivation_null(rng, leaf_fractions, n_draws)`: permutes the rows of `leaf_fractions` across events within class (onsets, current wells and vectors fixed).
- **Two-fold design.** Templates from odd-indexed cohort stops score events in even-indexed stops and vice versa, so no bin contributes to both a template and an event vector. The per-session statistic is the mean over both folds' events.
- **Script** `scripts/pfc/test_pfc_replay_reactivation.py`: per session, per replay class and pooled: the mean statistic, the number of events used and excluded (by reason), the decoder accuracy and its chance level, and a 1,000-draw within-class label shuffle giving a plus-one tail fraction. One parquet per session under `Processed-Data/pfc_replay_reactivation/` with a provenance sidecar. `scripts/pfc/summarize_pfc_replay_reactivation.py`: session → animal medians, animals with a positive statistic, sessions below tail fraction 0.05 by animal, and one figure (per-session statistic by animal and class, null band).
- **Audit** `docs/audits/<date>-pfc-replay-reactivation.md`; `docs/README.md` and `scripts/README.md` entries.

## Deliberately not in this phase

- No per-arm kernels in the GLM family.
- No decoding of position within an arm; the well is the unit of place.
- No trunk-content events: an event whose leaf mass is entirely on trunks has no represented well and is excluded by the off-current-mass rule.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_templates_use_only_the_fold_stops_and_exclude_event_bins` | synthetic session with two wells and planted per-well rates: templates equal the planted rates on the fold's bins only; bins inside a planted onset window or SWR interval are excluded; a well with under 500 bins has no template. |
| `test_event_vectors_window_and_drop_rule` | a window that runs past the last fitted bin is dropped and counted; others equal the planted counts divided by 0.2 s. |
| `test_statistic_is_positive_when_the_population_reactivates_the_replayed_well` | population with well tuning (each unit prefers one well) and event vectors drawn from the replayed well's template: statistic > 0.3 and decoder accuracy > 0.8; the same with event vectors drawn from a random well: statistic within ±0.05 of zero and accuracy near chance. |
| `test_current_well_is_excluded_from_numerator_and_comparison` | an event with all leaf mass on the current well is excluded; one with mass split current/other uses only the other well in `f`. |
| `test_null_permutes_fractions_within_class_only` | the shuffled rows are a permutation within each class; onsets and vectors untouched; same seed same draws. |
| `test_two_fold_assignment_separates_template_and_event_bins` | no fitted bin index appears in both a fold's template set and its scored events. |
| real-data smoke (`slow`) | one session end to end (senor 20201101 08_r4, 276 units): finite statistic, exclusion counts reported, runtime under 5 min. |

## Fixtures

A synthetic population with planted well tuning built in `tests/analysis/pfc/test_replay_reactivation.py`; real-data smoke on one median session.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind).
- User-facing documentation listed as tasks is updated, not deferred.
