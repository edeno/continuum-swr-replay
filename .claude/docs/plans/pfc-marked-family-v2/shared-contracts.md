# Shared contracts

[← back to PLAN.md](PLAN.md)

1. [Baseline columns](#baseline-columns) — phases 1a, 1b, 1c, 2, 3.
2. [Fit contract version 2](#fit-contract-version-2) — phases 1a, 1c, 2, 3.
3. [Construct mark names](#construct-mark-names) — phases 1b, 1c, 2, 3.
4. [Bundle directory suffixes](#bundle-directory-suffixes) — phases 1c, 2, 3, 4, 5.

## Baseline columns

Per-stop baseline terms enter every marked design as unpenalised columns immediately after the intercept, before any event-kernel block. They are constant within a stop and zero outside every cohort stop (fitted bins are always inside a cohort stop, so "outside" never occurs on fitted rows).

```python
BASELINE_REFERENCE_WELL = 0  # its indicator is omitted; the intercept carries it

def baseline_column_names(kept_wells: Sequence[int]) -> tuple[str, ...]:
    """Column names in design order.

    ``well_<w>`` for each kept well other than the reference, then
    ``rewarded_outcome`` (1 when the stop's outcome is known and rewarded,
    else 0) and ``outcome_unknown`` (1 when ``is_rewarded_outcome`` is NaN).
    """
    return (
        *(f"well_{w}" for w in kept_wells if w != BASELINE_REFERENCE_WELL),
        "rewarded_outcome",
        "outcome_unknown",
    )
```

Semantics, do not weaken:

- `kept_wells` are the wells (`occupancy_well_id` of the cohort stops, integers 0–5) with at least one fitted bin in the session; a well with no fitted bins contributes no column, and the sidecar records `baseline_column_names`.
- If the reference well itself is absent, the lowest kept well becomes the reference for that session and the sidecar says so (`baseline_reference_well`).
- `mixed` stops carry the `occupancy_well_id` the stop table assigns them; a stop with NaN `occupancy_well_id` (0.2 % of well-anchored events) contributes to no well column and is flagged in `n_bins_without_well` in the sidecar.
- The columns are not standardized and not penalised: `make_ridge_penalty(width, sigma, n_unpenalized=1 + n_baseline)`.
- `KernelDesign.n_baseline` is the count; `KernelDesign.__post_init__` requires the first block to start at `1 + n_baseline`.
- The typical-event kernel is then the response at the reference well after an unrewarded, known outcome; document this on the summary.

## Fit contract version 2

`BUNDLE_CONTRACT_VERSION = 2` in both marked runners. New keys, all recorded in the sidecar and compared by the skip check:

| key | value |
|---|---|
| `baseline_terms` | `"well_and_reward_outcome"` (or `"none"` when `--no-baseline` is passed) |
| `baseline_column_names` | the session's kept names, in order |
| `baseline_reference_well` | int |
| `tier` | `"joint"` (phase 1c) or `"marginal"` (phase 2) |
| `slope_permutations` | int, 0 when none (phase 3) |
| `slope_permutation_seed` | int (phase 3) |

A contract-1 sidecar never matches a contract-2 contract, so old bundles are refit, never skipped.

## Construct mark names

| name | source | definition |
|---|---|---|
| `content_distance_cm` | `event_table.mean_abs_ahead_behind_cm` | mean over the event of the represented position's absolute distance from the animal along the track, cm; replaces `current_arm_fraction` and `current_patch_fraction` as the spatial-role target |
| `log1p_spatial_extent_cm` | unchanged | the single trajectory-size target; `log1p_n_reversals` and `pause_fraction` become adjustments in that model |
| `current_arm_fraction` | unchanged | kept as an adjustment everywhere it was one, and as the target of the single-mark runner |
| `<a>_x_<b>` | product of the two constituents' within-class z-scores | replaces the raw products; then standardized again within class like any mark |

`MARK_DESCRIPTIONS` carries one sentence per name; the design doc's table is regenerated from it.

## Bundle directory suffixes

| directory | contents |
|---|---|
| `Processed-Data/pfc_marked_model_family_v1_single_intercept/` | the contract-1 corpus of 2026-09-19, kept as evidence, never read by v2 summaries |
| `Processed-Data/pfc_marked_model_family/` | contract-2 joint tier (phase 1c), marginal tier (phase 2) under distinct artifact names `*_marginal_<mark>_marked_family.pkl`, and slope nulls (phase 3) as `*.{class}.{target}.slope_null.npy` companions |
| `Processed-Data/pfc_stop_replay_rate/` | phase 4 per-session tables |
| `Processed-Data/pfc_replay_reactivation/` | phase 5 per-session tables |
