# Phase 1 — Tier-1 per-timebin signals in `data_over_time`

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Adds the per-timebin signals the rollups need but that are not yet stored, then
regenerates the `data_over_time` pkls. **Most of what the spec lists as Tier-1
already exists** (the four state probabilities and the z-scored ripple/MUA/theta
columns — see overview integration points). The genuinely new columns are:
`step_distance`, raw `ripple_power`, raw `mua_rate`, raw `theta_power`, and the
per-session scalar `run_speed_baseline`.

**Inputs to read first:**

- [posterior_analysis.py:121-261](../../../../src/continuum_swr_replay/analysis/hippocampus/posterior_analysis.py#L121-L261) — `get_map_speed`: the all-pairs `shortest_path_lengths` dict (~187) and the `node_ids = place_bin_center_ind_to_node[map_position_ind]` mapping (~174-179). `step_distance` is the per-bin graph distance these already imply, *before* the `/dt` and smoothing.
- [posterior_analysis.py:264-369](../../../../src/continuum_swr_replay/analysis/hippocampus/posterior_analysis.py#L264-L369) — `get_posterior_stats`: where map arrays are produced.
- [data_aggregation.py:300-382](../../../../src/continuum_swr_replay/analysis/hippocampus/data_aggregation.py#L300-L382) — `get_data_over_time` assembly dict (333-380) and `_nearest` aligner (312-322).
- [run_decoding_bandit.py:179-246](../../../../scripts/run_decoding_bandit.py#L179-L246) — save path `{epoch_identifier}_{threshold_int}_data_over_time.pkl` and the `organize_results` call site.
- `data_loaders/lfp.py` — where `ripple_consensus_trace`, `multiunit_firing_rate`, `theta_power` are created (raw signals to surface).

## Tasks

- **Add `get_map_step_distance` to `posterior_analysis.py`.** A function that
  returns the per-timebin unsmoothed track-graph step distance array (cm),
  shape `(n_time,)`, with `step_distance[0] = 0.0`. Factor it out of / share
  the node-id and `shortest_path_lengths` logic in `get_map_speed`
  (lines ~174-204) rather than duplicating: compute `node_ids`, look up
  `shortest_path_lengths[node1][node2]` for consecutive valid bins, and emit
  `np.nan` where either bin has an all-NaN posterior (reuse the existing
  `has_valid_data` mask at ~167-179). Do **not** divide by `dt` and do **not**
  smooth — this is raw per-bin displacement.
- **Surface raw `ripple_power`, `mua_rate`, `theta_power` in `get_posterior_stats` or pass-through.** These come from the `data` dict, not the posterior, so add them directly in the `get_data_over_time` assembly dict (next task) via `_nearest` — no change to `get_posterior_stats` needed unless `step_distance` is returned there. Recommended: have `get_data_over_time` call `get_map_step_distance` itself (it already holds `env` and `posterior`).
- **Extend the `get_data_over_time` assembly dict** (`data_aggregation.py:333-380`) with:
  ```python
  "step_distance": step_distance.squeeze(),
  "ripple_power": _nearest(data["ripple_consensus_trace"]).squeeze(),
  "mua_rate": _nearest(data["multiunit_firing_rate"][brain_area].squeeze()),
  "theta_power": _nearest(data["theta_power"].squeeze()).squeeze(),
  "run_speed_baseline": run_speed_baseline,  # per-session scalar, see next task
  ```
  Keep names exactly as above — Phase 2 reads them. Do not touch existing keys.
- **Compute `run_speed_baseline`** as a per-session scalar before the dict:
  `run_speed_baseline = np.nanpercentile(data["position_info"]["speed"], 90)`
  (robust "typical run speed"; see overview Open Question 1). Assign as the
  constant column above (broadcast like `max_coverage`).
- **No docstring/column-doc references to this plan.** If `get_data_over_time`
  has a column-list docstring, update it to describe the new columns in
  domain terms only.
- **Regenerate `data_over_time` (smoke test → extrapolate → full run).** See
  Validation slice; this is the long-running step.

## Deliberately not in this phase

- All per-event rollups and event-table columns — Phase 2. This phase only
  produces per-timebin signals.
- Recomputing z-scores: the per-timebin z-scored ripple/MUA/theta columns
  already exist (`SWR_zscore`, `MUA_zscore`, `theta_power_zscore`). Phase 2
  rolls those up; do not duplicate them.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_get_map_step_distance_stationary` | constant MAP posterior → all `step_distance == 0` (after index 0). |
| `test_get_map_step_distance_known_move` | two-bin synthetic posterior jumping a known graph distance → `step_distance` equals that distance (`rtol=1e-9`). |
| `test_get_map_step_distance_nan_safe` | all-NaN posterior bins → `np.nan`, no exception. |
| `test_data_over_time_new_columns_present` | on a simulated session, the five new columns exist, are finite where inputs are finite, and `step_distance >= 0`. |
| `test_data_over_time_existing_columns_unchanged` (slow, real-data) | regenerate one real session; assert every pre-existing column is bit-identical to a captured baseline (`pd.testing.assert_frame_equal` on the old column subset). |
| Smoke-test regeneration (slow) | one real epoch end-to-end through `run_decoding_bandit` save path; record wall-clock + peak memory; extrapolate to 218 sessions before the full run. |

## Fixtures

- Simulated posterior + minimal `env` (reuse `_EnvStub` / `_bandit_graph` from
  `tests/unit/test_replay_tables.py:27-68`) for the `step_distance` unit tests.
- A small simulated `data` dict (position_info + the three LFP/MUA series) for
  the assembly test; synthesize in `conftest.py`.
- One real session (any HPC epoch, e.g. `chimi20200212_02_r1_HPC`) for the
  baseline-parity and smoke-test steps. **Capture the baseline
  `data_over_time` to a pickle before editing** so the parity test has a
  reference.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` against the diff. Confirm:
- All five new columns are added with the exact names above; no existing column changed.
- `get_map_step_distance` shares (not duplicates) the node-mapping/distance logic in `get_map_speed`.
- Baseline-parity test passes on real data; smoke-test timing is recorded and extrapolated before the full regeneration.
- Validation tests exercise real behavior (known-distance move, not `assert True`); slow/real-data tests are marked.
- No docstring or test name references this plan or "Phase 1".
- The full 218-session regeneration is run (or explicitly handed off) — additive, in place.
