# Shared Contracts

[← back to PLAN.md](PLAN.md)

- [history package layout + re-export API](#history-package) — phase 4.
- [stage-function convention](#stage-convention) — phases 2 and 3.

---

## history-package

`covariates/history.py` (1536 lines) becomes a package `covariates/history/`. The public API is preserved exactly by `history/__init__.py`, so every existing `from ...covariates.history import X` and every re-export in `covariates/__init__.py` keeps working. Sibling modules relative-import from each other and from `_common`.

**Symbol → module** (verify each function's helper calls before moving; a helper used by ≥2 modules goes to `_common`):

- `history/_common.py` — the shared plumbing: `_prepare_port_experiences` (history.py:93), `_observed_port_outcomes` (:226), `_completed_visit_context` (:846), `_PORT_EXPERIENCE_COLUMNS` (:77), and the port-column constant lists `_P_REWARD_PORT_COLS`/`_Q_PORT_COLS`/`_BETADIST_VAR_COLS` (:58-60) if referenced by ≥2 modules (else co-locate with their sole user).
- `history/reward_history.py` — `rolling_reward_count_all_events` (renamed from `cumulative_reward_all_events`, :495), `recent_reward_rate_all_events` (:560, **vectorized in this phase**), `recent_trial_reward_rate` (:659), `lagged_trial_rewards` (:708), `_omission_streak` (:988), and the per-arm wrappers `rolling_reward_count_per_arm` (renamed, :1218) + `recent_reward_rate_per_arm` (:1232).
- `history/visit_recency.py` — `center_visit_recency_within_event` (:236), `spline_visit_recency_within_event` (:311), `visit_recency_all_events` (:364), `arm_never_visited_all_events` (:384), `visit_recency_and_never_visited_all_events` (:402), `visit_recency_per_arm` (:1196), `arm_never_visited_per_arm` (:1207).
- `history/value_uncertainty.py` — the Beta value/uncertainty concern: `load_contingency` (:1323), `attach_contingency_value_block` (:1349), `attach_arm_uncertainty` (:1460), `add_uncertainty_covariates` (:1029), `UNCERTAINTY_COVARIATES` (:70), and the **canonical** value/switch history `_leaky_q_value_diff` (:759), `canonical_port_value_history` (:803), `canonical_trial_switch_history` (:820) with `CANONICAL_SWITCH_WINDOW`/`CANONICAL_LEAKY_Q_ALPHA` (:66-67). These are covariate producers, not diagnostics.
- `history/diagnostics.py` — the per-animal diagnostic learner ONLY: `_q_chosen` (:1246), `q_all_arm` (:1265, its own comment says "DIAGNOSTIC … not a fitted covariate"), `_fit_alpha` (:1310).

**`history/__init__.py`** re-exports every name that `covariates/history.py` currently exposes (check `covariates/__init__.py`'s `from .history import ...` list and every external importer), and carries a reading-order docstring in the style of `covariates/__init__.py:1-18`. Keep the OLD names alive too: `cumulative_reward_all_events = rolling_reward_count_all_events` is NOT re-added — phase 1 already renamed callers — but confirm no external caller still imports the old name before deleting it (grep first).

**Cross-phase invariants (do not weaken):**
- The re-export shim must expose the exact set of names the old module did. A grep-style test asserts every previously-importable public name still resolves through `history` (or `covariates.history`).
- No persisted column name changes. `rolling_reward_count_all_events` still fills `reward_count_candidate_arm_last10_known_trials`.
- No cycle: `_common` imports nothing from the sibling concern modules.

## stage-convention

For `build_event_table` (phase 3) and content-fit `main()` (phase 2), each extracted stage is a module-private function named for the scientific step (not the phase number), taking the frames/arrays it needs and returning what the next stage consumes. The orchestrating spine reads as a sequence of named calls:

```python
def build_event_table(dot, environment, trials, ...):
    dot = _annotate_dot_for_events(dot, environment)          # was "# --- 1 ---"
    in_event = _select_in_event_timebins(dot, ...)            # "# --- 2 ---"
    per_event = _aggregate_event_features(dot, in_event, ...) # "# --- 3, 4, 5 ---"
    per_event = _attach_posterior_rollups(per_event, ...)     # "# --- 7 ---"
    per_event = _attach_trajectory_rollups(per_event, ...)    # "# --- 7b ---"
    per_event = _merge_classified_timing(per_event, dot, ...) # "# --- 9 ---"
    return stamp_covariate_schema(per_event, table="event")
```

Invariants: each stage computes exactly what the corresponding banner block computed, in the same order, with the same values — pure reorganization. Stage names use scientific vocabulary, never "phase N" or plan references. The cross-pass timing verification (`tables_event.py:544-633`) stays as its own named stage (`_verify_builder_pass_timing`) — it is load-bearing, not plumbing to hide.
