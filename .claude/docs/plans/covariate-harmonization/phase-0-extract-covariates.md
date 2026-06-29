# Phase 0 — Extract `covariates.py` (behavior-preserving)

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

A pure **move**: consolidate the covariate-definition functions (today scattered in
the 1,226-line `_helpers.py` grab-bag) and the Q-learner (today in a scratch script)
into one purpose-named module `covariates.py`, with a re-export shim so nothing else
changes. **No definitions change here** — that's phase 1. The whole point is a
reviewable, behavior-identical diff (move + shim), so phase 1's diff is *only* the
definition changes.

**Inputs to read first:**
- `_helpers.py` covariate functions to move: `visit_recency_all_events:486`, `cumulative_reward_all_events:561`, `recent_reward_rate_all_events:618`, `recent_trial_reward_rate:704`, `lagged_trial_rewards:757`, `_leaky_q_value_diff:814`, `_omission_streak:856`, `add_uncertainty_covariates:883`, `visit_recency_per_arm:1189`, `cumulative_reward_per_arm:1203`, `recent_reward_rate_per_arm:1217`.
- `scripts/_bandit_csv_contingency.py:58` `_q_chosen`, `:77` `_fit_alpha` — the fitted-α leaky-Q to promote.
- The ~16 importers of these names (from the audit): `tables.py`, `model_specs.py`, `per_bin_rate_model.py`, `per_stop_rate_model.py`, `analysis/contfrag/tables.py`, `__init__.py`, and several `scripts/*` — these must keep working unchanged.

## Tasks

- **Green the baseline first** (precondition for the "suite passes unchanged" gate): the audit found two pre-existing failures from our own recent drift — `test_uncertainty_covariates` (expects the removed `UNCERTAINTY_COVARIATES` in `fit_uncertainty_rate.py`) and `test_compare_event_source_models` (expects `R0/R1/R2`; the source set now includes `Rlag`). Fix the stale expectations so `uv run pytest` is green **before** the move. (The uncertainty-covariate test is further updated to the canonical set in phase 1.)
- **Create `src/continuum_swr_replay/analysis/replay/covariates.py`** and **move** the covariate-definition functions listed above out of `_helpers.py` **verbatim** (no logic edits). Keep their signatures/docstrings exactly.
- **Promote the Q-learner:** move `_q_chosen` and `_fit_alpha` from `scripts/_bandit_csv_contingency.py` into `covariates.py` verbatim; update `_bandit_csv_contingency.py` to import them from `covariates` (so the contingency builder keeps working).
- **Re-export shim:** in `_helpers.py`, add `from .covariates import (<moved names>)  # re-export: covariates moved to covariates.py` so every existing `from ..._helpers import recent_reward_rate_all_events` etc. keeps resolving. Update `__init__.py` to import the public covariate names from `covariates` (keep the same exported surface).
- **No doc updates needed** (no public-behavior change); a one-line note in `covariates.py`'s module docstring pointing at `docs/covariate_dictionary.md` as the spec it implements.

## Deliberately not in this phase
- **No definition changes** — `recent_reward_rate_all_events` stays k=5-callable, `leaky_q_value_diff` stays, no `p_reward_arm`/`q_arm`/`reward_rate_*` renames. All of that is phase 1.
- No table builder changes, no re-fits.
- Do not move the genuinely-misc `_helpers.py` functions (trial context, position helpers, id-hashing, `aggregate_to_timebins`, `raised_cosine_history_basis`, `add_arm_fixed_effects`) — only the covariate definitions.

## Validation slice

| Test | Asserts |
| --- | --- |
| full existing suite (`uv run pytest`) | Passes unchanged — the move is behavior-preserving. |
| `test_covariates::test_reexport_shim` | The moved names resolve identically via `covariates`, via the `_helpers` shim, and via the package `__init__`. |
| smoke parity (slow) | One session: `event_arm_table` covariate columns and a per-family fit are **bit-identical** to a pre-move run (pure move ⇒ no numeric change). |

## Fixtures
Existing test fixtures; one real session for the smoke parity.

## Review
Dispatch `code-reviewer`. Confirm: the diff is a **pure move + shim** (no logic edits to any moved function — `git diff` shows deletions in `_helpers.py` matching additions in `covariates.py`); the re-export shim covers every moved name; all existing imports still resolve; the full test suite passes unchanged; no definitions were "improved" in passing (those belong to phase 1); the Q-learner is imported by the contingency builder from its new home.
