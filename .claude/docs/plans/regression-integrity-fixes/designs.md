# Designs — Regression-validity remediation

[← back to PLAN.md](PLAN.md)

- [Atomic run creation and promotion](#atomic-runs)
- [Canonical trial timebase and switch history](#trial-timebase)
- [Post-outcome gating](#outcome-gating)
- [Exact cohorts and blocked folds](#cohorts-folds)
- [Dependence-aware estimation](#dependence)
- [Decision-grain replay models](#decision-models)
- [Identified SWR-power models](#swr-power)

## Atomic run creation and promotion {#atomic-runs}

Add `analysis/replay/run_artifacts.py` with one context-managed write path:

```python
@contextmanager
def staged_analysis_run(
    analysis: str,
    root: Path,
    *,
    model_specs: Mapping[str, Any],
    inputs: Sequence[Path],
    argv: Sequence[str],
) -> Iterator[AnalysisRun]:
    """Yield a staging run; validate, finalize manifest, rename, and promote on success.

    Any exception leaves CURRENT.json unchanged. The incomplete staging directory is retained with
    completed=false for diagnosis and is never returned by resolve_current_run().
    """
```

`AnalysisRun.write_dataframe` writes via a temporary sibling followed by `os.replace`, records exact
columns/dtypes/key uniqueness, and returns the output fingerprint. `AnalysisRun.finalize` validates
expected cells and identity artifacts, writes `manifest.json` last, renames `.staging` to the final run
directory, then replaces `CURRENT.json`. `resolve_current_run` validates before returning paths.

## Canonical trial timebase and switch history {#trial-timebase}

Persist one per-trial behavior table from `qvals_table._per_trial`, retaining exact poke/reward times,
trial number, chosen leaf/patch, pre-outcome value vectors, nominal probabilities, outcome, and update
variables. Define:

```python
def recent_completed_switch_rate(
    query_times: np.ndarray,
    outcome_known_times: np.ndarray,
    is_patch_change: np.ndarray,
    *,
    window: int = 5,
) -> np.ndarray:
    """Mean patch-change indicator over up to five trials completed by each query time."""
    # For query q: j = searchsorted(outcome_known_times, q, side="right")
    # return mean(is_patch_change[max(0, j-window):j]); NaN when j == 0.
```

Use the helper independently at each response grain: stop start for per-stop models, absolute bin
time for per-bin models, and full event start for content/power. Adding, removing, or splitting stops
cannot change the value at an unchanged query timestamp.

Rename the last-ten-trials arm reward count to `reward_count_arm_k10`. Emit both
`has_arm_history = arm_visit_count_prior > 0` and `reward_rate_arm_k10`; an unvisited arm has
`has_arm_history=0` and rate 0, while a visited zero-reward arm has `has_arm_history=1` and rate 0.

## Post-outcome gating {#outcome-gating}

For trial `t`, define `outcome_known_time` as `reward_on_ts[t]` when a reward onset is finite,
otherwise `poke_in_ts[t]` (arrival makes omission known). Preserve `outcome_time_source` as
`reward_on` or `poke_in_omission`.

- Event/content/power row is post-outcome only if full event start is at or after the timestamp.
- Per-bin row is post-outcome only if `stop_start + tc` is at or after the timestamp.
- A stop-level retrospective row must start after the timestamp; otherwise it is excluded rather than
  partially labeled.
- Stage-6 replay windows start exactly at this timestamp.

No `is_reward_stop` flag is consulted.

## Exact cohorts and blocked folds {#cohorts-folds}

Generalize the existing event-source helper into `analysis/replay/analysis_identity.py`:

```python
@dataclass(frozen=True)
class AnalysisIdentity:
    row_key_columns: tuple[str, ...]
    group_key_columns: tuple[str, ...]
    n_rows: int
    n_groups: int
    cohort_hash: str
    group_hash: str

def freeze_comparison_cohort(
    frame: pd.DataFrame,
    *,
    specs: Mapping[str, Any],
    row_keys: Sequence[str],
    group_keys: Sequence[str],
    required_category_values: Mapping[str, Sequence[Any]] | None = None,
) -> tuple[pd.DataFrame, AnalysisIdentity]: ...
```

Hash a canonical typed representation—not `str(value)`—after duplicate-key and complete-group
validation. Fitter strict mode receives the frozen keys and raises if internal filtering changes them.

Blocked folds are created once from unique groups sorted by canonical time, serialized, and mapped to
all descendant rows. Content groups by stop even though the likelihood conditions on event. Per-bin
and per-stop also block chronologically by stop/decision. Model code consumes assignments; it does not
construct its own `KFold`.

## Dependence-aware estimation {#dependence}

- **Per-bin and stacked SWR-rate GLMs:** primary covariance is stop-cluster sandwich; the same stop
  cluster captures temporal bins and paired SWR/non-SWR rows. Keep model-based SE under explicit
  `bse_model_*` names.
- **Per-stop GLM:** cluster on decision/trial ID after Phase 8; if unavailable for a pre-Phase-11
  auxiliary run, use prespecified chronological HAC and label it.
- **Conditional content:** resample whole stops within session, carry all their events/category rows,
  and assign fresh bootstrap event IDs. Report percentile/BCa interval, successful replicate fraction,
  and seed. A small fixed replicate count tests mechanics; production count is set after smoke timing.
- **Animal summary:** aggregate valid session effects within animal before any across-animal center.
  Bootstrap animals as whole units; with five animals, always show raw effects and leave-one-out range.

## Decision-grain replay models {#decision-models}

Target trial `t` (for `t >= 1`) defines one decision:

```text
window_start = outcome_known_time[t-1]
window_end   = poke_out_ts[t-1]          # next departure / start of target trial
current_arm  = chosen arm on trial t-1
chosen_arm   = chosen arm on trial t
behavior     = pre-outcome values on trial t
```

Aggregate every canonical event whose full start lies in `[window_start, window_end)`. Sum soft
posterior arm mass separately for all/SWR/non-SWR events; also retain hard-count equivalents. Reindex
to all six leaf arms and fill absent replay mass with zero. Never condition dataset inclusion on replay.

Primary choice ladder:

```text
D0_behavior: chosen ~ arm_dist_rank + visit_recency + reward_rate_arm_k10
                      + has_arm_history + q_arm_hmm
D1_replay:   D0 + replay_mass_soft
D2_split:    D0 + replay_mass_soft_swr + replay_mass_soft_non_swr
```

Fit conditional logit by `decision_id`; compare D0/D1 on exact common rows/folds. Hard mass and nominal
probability replace Q only as declared sensitivities. Do not call in-sample coefficients prediction.

Stage-6 decision summary uses the same window and includes zero-replay decisions. Primary descriptive
formula is `is_stay_next ~ previous_reward * replay_mass_current_patch + switch_value_advantage +
recent_completed_switch_rate_k5 + log1p(window_duration) + trial/block terms`, with nonlinear content
sensitivity. It is an association, not mediation or replay gating.

## Identified SWR-power models {#swr-power}

Move fitting to `analysis/replay/swr_power_model.py`. Gate events post-outcome and fit separate models:

```text
P0: power ~ log(event_duration) + time_since_stop
P1: P0 + trial_reward + q_current
P2: P0 + rpe                         # sensitivity, not compared by LRT to P1
P3: P1 + p_reward_current + value_diff_patch
```

Use `trial_reward` from the aligned behavior table, not stop-level `reward_lag1`. Add the corrected
completed-trial switch history only in a separately declared sensitivity. Preflight scaled SVD rank,
fit statsmodels OLS, use stop-cluster covariance, save residual/heteroskedasticity/influence diagnostics,
and score common blocked folds. Never co-fit reward, Q, and RPE.
