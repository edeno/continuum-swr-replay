# Designs — Regression-validity remediation

[← back to PLAN.md](PLAN.md)

- [Session input validation and selective refresh](#session-input-validation)
- [Atomic run creation and promotion](#atomic-runs)
- [Canonical trial timebase and switch history](#trial-timebase)
- [Post-outcome gating](#outcome-gating)
- [Exact cohorts and blocked folds](#cohorts-folds)
- [Dependence-aware estimation](#dependence)
- [Decision-grain replay models](#decision-models)
- [Identified SWR-power models](#swr-power)

## Session input validation and selective refresh {#session-input-validation}

Add `analysis/replay/source_integrity.py` around the
[session input-bundle contract](shared-contracts.md#session-input-contract). The validator receives
the resolved Spyglass position/DIO metadata, cached trials, decoder results, and saved
`data_over_time`; it returns a complete manifest plus an audit row or a typed failure:

```python
@dataclass(frozen=True)
class SessionInputAudit:
    session_id: str
    status: Literal["ok", "source_error", "stale_cache", "time_mismatch", "qc_failure"]
    failure_reason: str | None
    n_outside_valid_interval: int
    outside_valid_duration_s: float
    invalid_immobility_exposure_s: float
    position_source: str
    position_merge_id: str | None
    input_manifest_sha256: str | None

def validate_session_input_bundle(
    *,
    session_id: str,
    position_info: pd.DataFrame,
    valid_times: np.ndarray,
    position_source: str,
    position_merge_id: str | None,
    dio_events: Mapping[str, pd.DataFrame],
    trials: pd.DataFrame,
    results_time: np.ndarray,
    data_over_time: pd.DataFrame,
) -> tuple[dict[str, Any], SessionInputAudit]: ...
```

Build the valid-time mask by OR-ing every closed interval in `valid_times`; do not collapse multiple
intervals to their outer envelope. Require exact position and track-graph source cardinality before
fetching; a preferred-source fallback is triggered only by zero rows, not by a query/fetch error.
Measure raw DLC frame gaps before interpolation and reject a session when any internal gap exceeds
0.100 seconds. Trial construction records endpoint-to-well distance and pump-to-visit cardinality,
then applies the 8-cm and exact-one rules from the contract.

Refresh in two tiers, both writing beneath a new explicit release-candidate output root:

1. Reuse a saved decoder result/model only when its source, detector, training type, parameters,
   software, result bytes, and time-axis provenance can be reconstructed from stored evidence and all
   hashes match the locked run specification. Then rerun `organize_results` from freshly validated
   source data and stage a new `data_over_time`. Equal time arrays alone are insufficient.
2. If any required provenance is absent/mismatched, or if the axes differ, the decoder output is
   incompatible: run that session with `--overwrite --output-root <candidate-root>`, regenerate
   results/model/environment/derived data together, and validate the staged bundle. `--overwrite`
   applies only inside the new root and never overwrites the legacy source artifact.

Audit all sessions before mutating any promoted pointer. The current four out-of-bounds files are a
baseline diagnostic, not an allowlist: the audit decides which sessions require tier 2. Never trim a
saved posterior/result array post hoc to make it fit.

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
    """Yield a staging run; validate, finalize its manifest, and rename on success.

    Any exception leaves the global release pointer unchanged. The incomplete staging directory is
    retained with completed=false for diagnosis and is never returned by resolve_current_run().
    """

def promote_analysis_release(
    *,
    release_id: str,
    required_runs: Mapping[str, Path],
    regression_root: Path,
) -> Path:
    """Validate a complete cross-family release and atomically advance CURRENT.json."""

def freeze_release_candidate(
    *,
    release_id: str,
    required_runs: Mapping[str, Path],
    regression_root: Path,
) -> Path:
    """Write and validate an immutable, non-current descriptor for PFC/reporting."""

def resolve_release_candidate(candidate_path: Path) -> ReleaseCandidate:
    """Resolve only the explicit frozen descriptor; never consult CURRENT.json."""
```

`AnalysisRun.write_dataframe` writes via a temporary sibling followed by `os.replace`, records exact
columns/dtypes/key uniqueness, and returns the output fingerprint. `AnalysisRun.finalize` validates
expected cells and identity artifacts, writes `manifest.json` last, renames `.staging` to the final run
directory, and returns its manifest hash without changing canonical state.
`freeze_release_candidate` validates every required run/hash/source set and writes the immutable
descriptor without changing canonical state. PFC and report generators accept only its explicit path
and write repository-relative outputs beneath
`release_outputs/<release_id>.staging/`. `promote_analysis_release` validates the candidate, optional
PFC completion, and every staged output hash; atomically finalizes the release-output tree; writes the
completed release manifest; then replaces the one global `CURRENT.json`.
`resolve_current_run(analysis)` and `resolve_release_component(name)` follow that pointer and validate
the release plus selected run/component before returning paths. Until the first global release exists,
development consumers must call `resolve_analysis_run(analysis, run_id=...)` or
`resolve_release_candidate(candidate_path)` through an explicit CLI option; paper-facing/default
resolution fails closed. After promotion, `publish_release_outputs` copies and verifies the immutable
bytes at their repository-relative destinations rather than regenerating them.

## Canonical trial timebase and switch history {#trial-timebase}

Persist `behavior_trial_table.parquet` directly from the complete
`TrialsInfoByEpoch.ByTrial`/trial-source rows before any behavior-model join. It retains exact
poke/reward times, trial number, chosen leaf/patch, nominal probabilities, and observed outcome for
every source trial. Do **not** derive this table from `qvals_table._per_trial`: that function currently
inner-joins model rows and would delete the very non-Q terminal trials the timebase must preserve.
Phase 9a separately writes `behavior_model_trial_table.parquet` with pre-outcome Beta value vectors,
entropy/update variables, and exact coverage keys. Downstream code left-joins the model table to the
complete behavior timebase and filters only model-dependent specifications.

Build and persist `session_epoch_crosswalk.parquet` before either table. Its unique source key is
`(nwb_file_name, epoch_number)` and its unique canonical key is
`(session_id, animal, day, epoch)`, where `epoch_number` is the integer DataJoint epoch and `epoch`
is the canonical string label. The mapping must be one-to-one in both directions and is a manifest
input. Never recover an integer epoch by splitting the canonical label or construct an NWB filename
from an animal/day string. Trial, stop, event, and decision IDs are derived only after this crosswalk
join succeeds. Define:

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

Use the helper independently at each response grain: stop start for per-stop models,
`bin_start_time = stop_start + binj * bin_s` for per-bin models, and event start for
content/power. In `aggregate_to_timebins`, construct trial/history covariates at each bin's left edge
instead of repeating stop-start values; keep exposure, response counts, self-history, and
cross-history mechanics otherwise unchanged. Adding, removing, or splitting stops cannot change the
value at an unchanged query timestamp.

Predictive switch context is `previous_completed_trial_is_switch`, obtained from the most recent trial
whose outcome/choice is known by the query timestamp. Preserve the current trip's eventual
patch-change only as `current_trip_is_switch`, a descriptive field that is not registered
`PRE_OUTCOME_SAFE`.

For arm history, let `j = searchsorted(trial_cutoff_times, query_time, side="right")`, so exactly `j`
trials are complete:

- visited leaf: `visit_recency_trials = j - last_visit_trial_index`;
- unvisited leaf with `j > 0`: `visit_recency_trials = j + 1` and `has_arm_history = 0`;
- no completed trial: history fields are `NaN`;
- trunk: `is_reward_site = 0`, `has_arm_history = 0`, and recency/reward-rate numeric fields are 0.

All leaf history fields use only the prefix `trials[:j]`; appending future trials cannot change them.
Delete persisted `visit_recency_demeaned`. Persist only raw `visit_recency_trials`. The fixed
nonlinear map is `log1p(visit_recency_trials)`. For an estimation fit, learn mean/SD on that fit's
declared cohort and record them; for CV, learn mean/SD on training rows only and apply unchanged to
test rows. Conditional content/choice likelihoods already remove choice-set constants, so do not
persist a future-informed event/session demeaning column.

Keep the two reward-history windows distinct:

- `reward_count_arm_global_k10` and `visit_count_arm_global_k10` count rewarded arrivals and all
  arrivals to that arm among the last ten completed trials globally;
- `reward_count_arm_visits_k10`, `visit_count_arm_visits_k10`, and
  `reward_rate_arm_visits_k10` use the last up-to-ten completed visits to that arm, with
  `rate = reward_count / visit_count`;
- `has_arm_history = (visit_count_arm_visits_k10 > 0)`.

Use exact patch analogs with `patch` in place of `arm`. An unvisited target has counts/denominator
0, rate 0, and history 0; a visited but never-rewarded target has denominator >0, rate 0, and history
1. Before any completed trial, all leaf/patch history fields are `NaN`. Trunk arms are marked
`is_reward_site=0` and carry structural zeros. Primary experienced-rate models use the visit-window
rate plus its denominator/history; global-window counts are separately named recent rewarded-exposure
sensitivities. No model or invariant reconstructs a visit-window rate from global-window counts.

## Post-outcome gating {#outcome-gating}

For trial `t`, define `outcome_known_time` as `reward_on_ts[t]` when a reward onset is finite,
otherwise `poke_in_ts[t]` (arrival makes omission known). Preserve `outcome_time_source` as
`reward_on` or `poke_in_omission`.

- Event/content/power row is post-outcome only if `event_start_time >= outcome_known_time`.
- A per-bin row stores
  `bin_start_time = stop_start + binj * bin_s` and
  `bin_end_time = bin_start_time + exposure`. It is post-outcome only if
  `bin_start_time >= outcome_known_time`; a bin with
  `bin_start_time < outcome_known_time < bin_end_time` is dropped rather than labeled by its center.
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

Lock `alpha=0.05` and base seed `20260729`. Derive each cell's RNG seed from the first eight bytes of
SHA-256 over the UTF-8 string
`20260729|analysis|comparison_set|model|population|session_id|cohort_hash`; never use Python's
process-randomized `hash()`.

- **Per-bin and stacked SWR-rate GLMs:** primary covariance is finite-sample-corrected CR1 clustered
  on `stop_id`; the same cluster captures temporal bins and paired SWR/non-SWR rows. Use the
  \(t_{G-1}\) reference distribution for a term, where \(G\) is the number of clusters. Keep
  model-based SE under explicit `bse_model_*` names. Fewer than 20 clusters makes robust inference
  unavailable while leaving a valid likelihood/CV fit selectable.
- **Per-stop GLM:** use the same CR1/\(t_{G-1}\) rule clustered on Phase-8 decision/trial ID. A
  pre-Phase-11 auxiliary run with no stable decision ID is descriptive-only; do not silently
  substitute HAC in a canonical run. Define material overdispersion as finite Pearson
  \(\chi^2/df_{resid} \ge 1.5\) with at least 30 clusters. When triggered, fit one declared
  quasi-Poisson sensitivity with the Pearson scale on the identical rows. Do not choose between
  negative-binomial and quasi-Poisson after seeing results; schema v1 has no NB sensitivity and no
  likelihood/AIC comparison to quasi-Poisson.
- **Conditional content:** run exactly 2,000 whole-stop bootstrap attempts per production cell,
  carrying all events/category rows and assigning fresh event IDs. Use the deterministic cell seed,
  the 2.5/97.5 percentiles for the 95% interval, and the sample covariance of replicate coefficient
  vectors for robust SE diagnostics. CI tests use 100 attempts and are marked as mechanics-only.
  Fewer than 1,800 finite, fit-valid joint coefficient vectors makes robust inference unavailable;
  no extra attempts are added selectively.
- **Calibration:** use 400 fixed-seed autocorrelated null simulations. The block-aware 95% interval
  must cover at least 90% and exceed naive model-based coverage by at least 5 percentage points.
  Report the realized Monte Carlo counts; changing either threshold requires a reviewed code/config
  commit before production.
- **Animal summary:** take the unweighted median of valid session effects within each animal, then
  report all five animal effects and their median. A 10,000-resample percentile bootstrap of whole
  animals (same deterministic seed derivation without `session_id`) gives a descriptive 95% interval;
  always include the leave-one-animal-out range. Missing any declared animal blocks headline and
  confirmatory inference.
- **Confirmatory p-values:** use the five animal effects, not session/neuron p-values. For a term,
  enumerate all \(2^5=32\) common animal sign flips and compare the signed or absolute median
  according to the registry alternative. For a joint test, require `alternative="two-sided"`, apply
  the same sign to every term within an animal, and use
  \(\sum_j[\operatorname{median}_a(\beta_{aj})/
  \operatorname{RMS}_a(\beta_{aj})]^2\); a zero/non-finite RMS is invalid. Include equality in the
  tail count and divide by 32—no asymptotic or mid-p correction. Apply BH at `q=0.05` once across the
  single locked registry family, sorting equal raw p-values by `hypothesis_id` for deterministic
  output. Session-level robust intervals remain estimation diagnostics and are never supplied to
  BH-FDR.

## Decision-grain replay models {#decision-models}

Target trial `t` (for `t >= 1`) defines one decision:

```text
window_start = outcome_known_time[t-1]
window_end   = poke_out_ts[t-1]          # next departure / start of target trial
current_arm  = chosen arm on trial t-1
chosen_arm   = chosen arm on trial t
behavior     = pre-outcome values on trial t
```

Require a positive-duration canonical event to be fully contained in the decision window:
`event_start_time >= window_start` and `event_end_time <= window_end`. Thus an event that starts
before the previous outcome or straddles the departure boundary is excluded; equality at either
boundary is retained. Sum soft posterior arm mass separately for all/SWR/non-SWR events; also retain
hard-count equivalents. Reindex to all six leaf arms and fill absent replay mass with zero. Never
condition dataset inclusion on replay.

Primary choice ladder:

```text
D0_behavior: chosen ~ arm_dist_rank + visit_recency_log1p_z
                      + reward_rate_arm_visits_k10 + visit_count_arm_visits_k10
                      + has_arm_history + q_arm
D1_replay:   D0 + replay_mass_soft
D2_split:    D0 + replay_mass_soft_swr + replay_mass_soft_non_swr
```

`visit_recency_log1p_z` is a fit-time design column, never a persisted canonical-table field; its
training/estimation transform parameters are recorded with the cohort hash.

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
