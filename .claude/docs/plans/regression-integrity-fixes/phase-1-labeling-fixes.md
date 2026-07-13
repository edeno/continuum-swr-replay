# Phase 1 — Correct stale result labels and docstrings

> Historical baseline: this work landed before the 2026-07-10 plan refresh. Retained for audit trail; do not execute as an active phase.

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Pure labeling/doc correctness. No fitted numbers change. Ships first so mislabeled provenance stops
propagating into saved outputs and the paper. No dependencies on other phases.

**Inputs to read first:**

- [scripts/content/fit_swr_contrast_content.py:295-320](../../../../scripts/content/fit_swr_contrast_content.py) — the saved provenance dict (`inference`, `q_arm_source`, `deferred`).
- [src/continuum_swr_replay/analysis/replay/model_specs.py:876-901](../../../../src/continuum_swr_replay/analysis/replay/model_specs.py) — confirms `C4_armvalue`/`CS5` use `q_arm_hmm`, not beta `q_arm`.
- [src/continuum_swr_replay/analysis/replay/choice_prediction.py:305-340](../../../../src/continuum_swr_replay/analysis/replay/choice_prediction.py) — choice predictor set: replay_frac, recency, reward-rate, distance/rank; **no value covariate**.
- [docs/covariate_dictionary.md:60](../../../../docs/covariate_dictionary.md) — the `q_arm_hmm` row claiming "content/**choice** value main effect".
- [src/continuum_swr_replay/analysis/behavior/qvals_table.py:10-18](../../../../src/continuum_swr_replay/analysis/behavior/qvals_table.py) and [:305-315](../../../../src/continuum_swr_replay/analysis/behavior/qvals_table.py) — the two contradictory interval docstrings.

## Tasks

- **Fix the stale value-construct provenance** in `fit_swr_contrast_content.py` (the dict around
  `:302`–`:315`). Change `"q_arm_source": "beta-model per-arm value (HMM q_arm harmonization deferred)"`
  to `"q_arm_source": "HMM Global-Model per-arm value (q_arm_hmm); beta q_arm retained as robustness column only"`,
  and remove `"HMM per-arm q_arm"` from the `"deferred"` list (it is implemented — `model_specs.py:880`
  `C4_armvalue = [*BASEc, "q_arm_hmm"]`, built at `tables.py:1300`-ish via `_attach_hmm_arm_value`).
  Leave the other `deferred` entries (mixed-effects model, common cohort, within-group VIF) — those are
  still genuinely deferred and are addressed in later phases (5, 5, and a documented non-goal).
- **Re-emit already-saved stale provenance.** Grep the outputs dir for the stale string and either
  re-run the affected content-contrast fits or patch the saved `*_meta.json` in place:
  `grep -rl "harmonization deferred\|beta-model per-arm value" Processed-Data/`. If any are the source of
  a paper figure/table, note them for regeneration. (Do not fabricate — only patch strings, never numbers.)
- **Correct the choice claim in `docs/covariate_dictionary.md:60`.** The `q_arm_hmm` row says it is the
  value main effect "used by C4/CS5/Mv" (content — correct) **and** a "choice value main effect"
  (incorrect: `_CHOICE_BEHAVIOR`/`choice_prediction.py:313` carry no value covariate). Change to
  "content value main effect — the construct used by C4/CS5/Mv; **not** currently used in the Stage-5A
  choice model (choice carries replay_frac + recency + reward-rate + distance/rank only)." If the choice
  model is paper-facing, phase 5 **requires** a value-controlled choice sensitivity (`C_value`) and this
  line is revised there; it stands as-is only if the choice model is explicitly not paper-facing.
- **Correct the audit artifacts themselves.** `shared-contracts.md` treats
  `.claude/docs/reviews/06_covariate_provenance.csv` as the authoritative timing source phase 4 reads,
  so its accuracy is load-bearing. Fix two known staleness items in that CSV: (a) the `q_arm`/`q_arm_hmm`/
  `p_reward_arm` rows must note **identical missingness** (8,064 rows / 896 events — verified) so no
  reader treats beta `q_arm` as a fuller-cohort proxy (this is the F3 issue that killed phase 5's original
  design); (b) ensure every derived-interaction covariate's `time_availability` matches the phase-4
  `DERIVED_COVARIATE_SOURCES` map (e.g. `update_x_current` inherits `hmm_value_update`'s post-outcome
  flag). Also re-read `06_data_provenance_regression_audit.md` §4 and correct any "silent drop" wording to
  "documented deliberate NaN" for mechanism A (`covariates.py:189`).
- **Fix the interval docstring contradiction in `qvals_table.py`.** The module docstring (`:14`) states
  `trial t spans (poke_out_ts[t-1], poke_out_ts[t]]`, which is the wrong convention; the function
  docstring (`:309`), the inline comment (`:395`), and `tests/unit/test_qvals_table.py:141` all correctly
  state `[poke_out[t-1], poke_out[t])` (matching `searchsorted(..., side="right")`). Change `:14` to the
  left-closed/right-open form so the module has one convention. Numerically inert (boundary is
  measure-zero); documentation-truth fix only.

## Deliberately not in this phase

- Adding a value-controlled choice model — that is built in [phase 5](phase-5-inference-validity.md),
  where it is **required for any paper-facing choice claim** (skippable only if the choice model is
  explicitly not a claim). This phase only corrects the doc to match current code.
- Any change to fitted coefficients, cohorts, or diagnostics — later phases.

## Validation slice

| Test | Asserts |
| --- | --- |
| `grep -c "harmonization deferred" scripts/content/fit_swr_contrast_content.py` | 0 |
| new `tests/unit/test_provenance_labels.py::test_q_arm_source_is_hmm` | the runner's provenance dict reports an HMM source string and `"HMM per-arm q_arm"` not in `deferred` |
| existing `tests/unit/test_qvals_table.py::test_stop_trial_interval` (already passing) | still passes — confirms interval semantics unchanged by the docstring edit |
| new `tests/unit/test_provenance_csv.py::test_csv_corrected` | `06_covariate_provenance.csv` **fails the test unless** it contains (a) the identical-missingness note on the `q_arm`/`q_arm_hmm`/`p_reward_arm` rows and (b) rows for the derived interactions (`update_x_current`, `tss_x_current`, `swr_x_q_arm_hmm`, …) with `time_availability` matching the phase-4 `DERIVED_COVARIATE_SOURCES` timing. Phase 4 consumes this CSV as authoritative, so this gate must pass before phase 4 runs (Medium, round 6). |
| manual doc diff | `docs/covariate_dictionary.md:60` no longer asserts choice uses `q_arm_hmm` |

## Fixtures

None new. The provenance-label test imports the runner's metadata-building function (refactor the dict
into a small `_build_provenance()` helper if it is currently inlined, so it is unit-testable without a
full fit run).

## Review

Before opening the PR, dispatch `code-reviewer` against the diff. Confirm:
- The provenance string change matches `model_specs.py:880` (HMM, not beta) — reviewer re-reads that line.
- No fitted numbers changed (diff is strings/docs + one test + optional helper extraction).
- The interval docstring now matches `:309`/`:395`/the test.
- Docstrings/test names don't reference this plan or "phase 1".
