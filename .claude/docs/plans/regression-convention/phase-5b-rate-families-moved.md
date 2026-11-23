# Phase 5b — The other rate families on the convention

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#rate-design)

Branch: `rate-families-convention` from master after Phase 5a. CPU only.

Decision 4 of the design (lead: "move it to the convention and fix if needed") covers the per-stop family,
the SWR-contrast rate covariates and the uncertainty rate model. This phase moves them onto Phase 5a's visit
design and fitter.

**Inputs to read first:**

- The design, section 7 ("What happens to each existing family") and decision 3 (both forms of the reward
  effect: the total effect per second of stop, and the direct effect at the same time before leaving).
- Phase 5a's `analysis/replay_models/rate/visit_rate.py` and `visit_rate_statistics.py`.
- `src/continuum_swr_replay/analysis/replay_models/rate/per_stop_rate_model.py:87`
  (`fit_per_stop_rate_model`), `:331` (`OFFSET_COL = "immobility_duration_s"`), `:356`
  (`fit_per_stop_rate_hierarchy`); `model_specs.py:490` (`MODEL_PER_STOP_RATE`).
- `src/continuum_swr_replay/analysis/replay_models/rate/swr_contrast_rate.py:133`
  (`fit_post_outcome_cohort`), `:208` (`fit_contrast_session`), `:603` (`covariate_sds`);
  `model_specs.py:1294-1330` (`_BLOCKS`: the covariates PERISWITCH, BLOCK, SWITCHRATE, VALUE, RPE, UNCERT,
  UPDATE, REWXTIME), `:1374-1428` (`_g_specs`, `_s_specs`).
- `src/continuum_swr_replay/analysis/replay_models/rate/uncertainty_rate.py`; `model_specs.py:1084`
  (`MU_UNCERTAINTY_COVARIATES`), `:1090` (`MODEL_UNCERTAINTY_RATE`).
- `src/continuum_swr_replay/analysis/replay_models/covariate_timing.py` (which covariates are post-outcome).
- `reported/behavior/rate.py` (`per_stop_values`, `rate_contrast_audit_values`) for the values these
  replace.

**Contracts referenced:** [C4](shared-contracts.md#c4-the-selection-function-is-the-only-filter),
[C5](shared-contracts.md#c5-the-convention), [C6](shared-contracts.md#c6-kinds-contrasts-and-the-two-meanings-of-direct),
[C7](shared-contracts.md#c7-the-fit-row-record), [C9](shared-contracts.md#c9-where-new-fits-are-stored).

**Decisions this phase applies** (settled 2026-11-20): L6, lead: "We need to be consistent in the repo".
The visit is the unit; the total reward effect is per visit, and no per-stop variant is kept beside it, not
even as a literature check (C6, C14). If a per-stop number is ever wanted, it is a question for the lead.

## Tasks

- **Baseline**: save the current `rate.per_stop.*`, `rate.swr_contrast.*` and `rate.uncertainty.*` values
  from `paper/reported/behavior.json` and the stored fits' session values for the same covariates.
- **Total reward effect** (`analysis/replay_models/rate/visit_total_rate.py`, new): one row per visit (L6),
  the count of the kind's events that start inside the visit (C3) after the outcome is known, offset `log` of the at-risk seconds after the outcome is known, and the visit's own
  outcome (rewarded, 0/1). Poisson through the shared fitter, no smooth, support rule on the 0/1 term (the rule the lead settles
  at Phase 5a, `OPEN (lead): reward-term support`). The
  estimand: the log rate ratio, rewarded against omitted, "per second of in-visit at-risk time" (total),
  reported beside Phase 5a's Q2 ("at fixed time before leaving", direct), each named (C6). It is not a
  member of the declared family (L1).
- **SWR-contrast covariates** (`analysis/replay_models/rate/visit_rate_covariates.py`, new): each covariate
  of `_BLOCKS` other than BASE and REWXTIME, added one at a time to model A as a pooled-SD slope per kind
  (C5 rule 2), with its kind difference. A post-outcome covariate (per `covariate_timing.py`) enters only on
  bins at or after `outcome_known_time_s` (the cohort step restricts and counts; the fitter drops nothing).
  REWXTIME is replaced by nothing: its question (does the reward effect change over the visit) is answered
  by adding a by-outcome smooth of `s` to model A (call; state it in the docs).
  That by-outcome smooth, like model A's clocks, is a P-spline at a declared fixed log penalty, set on
  this model's own bins by a target median edf and checked by its scenario list (design, seventeenth
  round). Whether model A itself gets this form is `OPEN (lead): rate model form` (Phase 5a); follow the
  lead's answer there.
- **Uncertainty rate model**: the `MU_UNCERTAINTY_COVARIATES` added to model A as pooled-SD slopes, one model
  with all of them as now, with the collinearity record; a covariate over the threshold is reported only
  jointly with its partner (C5).
- **A scenario list for the simulation package (Phase 2b)** (`replay_models/rate/visit_rate_covariates_scenarios.py`,
  new; run by Phase 2b's runner): one planted slope per covariate over the real design; the fit must
  recover it and not move it to another covariate or to the reward term (C5 collinearity rule, last
  bullet). Stored in `models_convention/simulations/rate_covariates.parquet`.
- **Runner** `scripts/replay_models/rate/fit_visit_rate_covariates.py` (new) and `fit_visit_total_rate.py`
  (new), registered in `run_refits.py`.
- **Summary values** (`reported/behavior/visit_rate.py`, extended): the total effect per kind and difference;
  each covariate's slope per kind and difference with interval, animals with sign and day-level p, labeled as
  outside the declared family; collinearity flags.
- **Docs**: `docs/rate_models.md` (the two forms of the reward effect, named; what moved from the
  SWR-contrast and uncertainty models).

- **Retire what this phase replaced** (C13; the items are the REMOVE rows of
  [legacy-inventory.md#phase-5b](legacy-inventory.md#phase-5b)). In this PR, or in a second PR of this phase
  that merges right after it:
  1. Write the old-against-new comparison for the lead: each old value the section lists, read from the
     copy of the summary saved before the rebuild, beside its replacement from the rebuilt summary, with the
     reason for each change. Put it in the PR description.
  2. List the macros `paper/*.tex` cites from those old values (none of `behavior.json` or `mpfc.json` was
     cited on 2026-11-20; check again). Ask the lead about any that are; remove a cited value only after
     the yes.
  3. Remove every REMOVE item of the section: modules, specs, runners, `RUNS` entries, plotters,
     value groups and their summary wiring, question-list entries (mark the replaced 3.x questions
     `dropped`, reason "superseded by" the new id), tests and doc sections.
  4. Check: `git grep -n -w <name> -- src scripts tests docs` gives no match for any removed symbol, file or
     column (dated records exempt, C13); `scripts/checks/write_pipeline_map.py` regenerated; one row per
     removal in `docs/history/2026-module-moves.md`; the full suite passes.
  The ASK rows of the section are not removed here; list them in the PR for the lead.

## Deliberately not in this phase

- The history models and the shared rate helpers they still use (Phase 11).
- The duration-aware and ridge history models (Phase 11).
- Any content model.

## Validation slice

| Test | Asserts |
| --- | --- |
| `tests/analysis/replay_models/rate/test_visit_total_rate.py::test_total_effect_hand_case` | on two synthetic visits per outcome, the fitted ratio equals the ratio of events per at-risk second |
| `test_visit_total_rate.py::test_counts_only_after_outcome_known` | events before `outcome_known_time_s` are not counted, and their time is not exposure |
| `test_visit_rate_covariates.py::test_post_outcome_covariate_bins_only_after_outcome` | the cohort step keeps only post-outcome bins for a post-outcome covariate and records the count |
| `test_visit_rate_covariates.py::test_planted_slope_recovered_and_not_leaked` | a planted slope is recovered; other covariates and the reward term stay null within nominal |
| `tests/reported/behavior/test_visit_rate_values.py::test_total_and_covariate_values` | values exist per kind and difference, with the "outside the declared family" description |

## Fixtures

Phase 5a's synthetic visit helpers, extended with the covariates.

## Verification and runtime

- Simulated first, then five real sessions, then all 242. Each covariate model costs about what model A
  costs (Phase 5a's measurement); eight covariates times two kinds: run in `tmux` (`csr-5b-rate`).
- Compare with the baseline: the old per-stop reward ratio (Family A, from the saved summary) against the new
  per-visit total effect, per kind, in the old-against-new comparison (no per-stop fit is made);
  each covariate's sign against the old SWR-contrast rung's. A sign change is explained in the PR (cohort,
  scaling or the end-of-visit clock).

## Rebuilt

`models_convention/rate_total/`, `models_convention/rate/covariates_*.parquet`, the simulation file;
`paper/reported/behavior.json` and the emitted files.

## Exit criteria

Both forms of the reward effect are in the summary, each named; every moved covariate has a record and a
value or a recorded reason it has none.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind).
- User-facing documentation listed as tasks is updated, not deferred.
