# Plan: Switch Decoding from 1D to 2D

## Summary

Migrate the decoding pipeline from 1D linearized position (along track graph) to 2D (x, y) position on a spatial grid. The `non_local_detector` library already supports 2D decoding, 2D distance functions, and 2D visualization. The strategy is: **decode in 2D, project back to 1D where needed** so that existing analyses and figures remain reproducible.

**Branch:** `feature/2d-decoding`

---

## Phase 1: Enable 2D Decoding Pipeline

### 1.1 Expose `is_1d` flag in CLI

**File:** `scripts/run_decoding_bandit.py`

- Add `--2d` argument to the argparse block (~line 710+)
- Pass this through `run_decode()` → `perform_decoding()` (currently `is_1d=True` is hardcoded at line 364 and never passed from `run_decode()` at line 659)
- The `Environment` creation branch already exists (lines 399-409):
  - 1D: `Environment(place_bin_size=2.0, track_graph=..., edge_order=..., edge_spacing=...)`
  - 2D: `Environment(place_bin_size=3.5)`
- Position input is already 2D (`["head_position_x", "head_position_y"]` at line 438-439) so detector input doesn't change

**Effort:** Small (~10 lines). Plumbing exists, just needs to be connected.

### 1.2 Validate 2D Environment parameters

- `PLACE_BIN_SIZE_2D = 3.5` cm is defined in `parameters.py` (line 92) but has not been empirically validated
- Run a few test sessions to check: bin count, GPU memory usage, decoder convergence
- May need to adjust bin size or add `is_track_interior` masking parameters

**Effort:** Small. Empirical check, not code change.

### 1.3 Update results file naming

- 2D results should be distinguishable from 1D results (different bin counts, different posterior structure)
- Option A: Add `_2D` suffix to epoch identifier (e.g., `chimi20200212_02_r1_HPC_2D_results.nc`)
- Option B: Store in a separate subdirectory (e.g., `Processed-Data/2D/`)
- Decide based on how you want to manage coexistence during testing

**Effort:** Small.

---

## Phase 2: Fix 2D-Incompatible Analysis Functions

### 2.1 `get_posterior_stats()` — central bottleneck

**File:** `src/continuum_swr_replay/analysis/hippocampus/posterior_analysis.py` (lines 234-338)

This function computes all derived metrics. Needs a 2D code path (detect from Environment or posterior dimensionality):

| Metric | Current 1D | 2D approach |
|--------|-----------|-------------|
| MAP estimate | `maximum_a_posteriori_estimate(posterior)` → scalar | Already works — returns (x, y) automatically |
| MAP speed | `get_map_speed()` using 1D graph shortest paths | Use `get_2D_distance()` with `make_2D_track_graph_from_environment()` between successive 2D MAP positions (preserves track-respecting distances) |
| Ahead/behind distance | `get_ahead_behind_distance(track_graph, ...)` | `get_ahead_behind_distance2D(head_pos, head_dir, map_pos, track_graph_2d)` from `non_local_detector.analysis.distance2D` |
| HPD spatial coverage | `get_HPD_spatial_coverage(posterior, threshold)` | **Broken for 2D** — see 2.3 |
| `max_linear_position` | From `edge_spacing` | Replace with 2D grid extent (e.g., diagonal or x/y range) |
| `max_distance` | From `linear_position` range | Replace with max pairwise graph distance on 2D track graph, or grid diagonal |

### 2.2 `get_map_speed()` — replace graph shortest-path logic

**File:** `src/continuum_swr_replay/analysis/hippocampus/posterior_analysis.py` (lines 119-231)

Current implementation uses `nx.shortest_path_length()` on the 1D track graph with bin-center nodes. For 2D:

- Build 2D track graph via `make_2D_track_graph_from_environment(env)` (from `non_local_detector.analysis.distance2D`)
- Use `get_2D_distance(map_pos[t], map_pos[t+1], track_graph_2d)` between successive MAP positions
- Or use `get_speed(get_velocity(map_positions_2d))` for Euclidean speed (simpler, but doesn't respect track boundaries)
- **Recommend graph-distance approach** for consistency with 1D semantics

### 2.3 Fix `get_HPD_spatial_coverage` for 2D

**File:** `non_local_detector` package — `analysis/highest_posterior_density.py`

The function is hardcoded to use `.position` coordinate (1D). For 2D posteriors with `x_position`/`y_position` coordinates, it will raise a KeyError.

**Options:**
- **Option A:** Fix upstream in `non_local_detector` — generalize to detect 1D vs 2D coordinates, compute bin width (1D) or bin area (2D)
- **Option B:** Write a wrapper in this repo that handles 2D HPD before calling the upstream function
- Note: HPD units change from cm (length) in 1D to cm^2 (area) in 2D

### 2.4 `get_data_over_time()` — update column structure

**File:** `src/continuum_swr_replay/analysis/hippocampus/data_aggregation.py` (lines 117-274)

- `map_estimate`: scalar → two columns (`map_estimate_x`, `map_estimate_y`) or keep as 2D array
- `max_linear_position`: remove or replace with grid extent
- `max_coverage`: units change (cm → cm^2) if native 2D HPD; or stays cm if computed after 1D projection
- All other columns (state probabilities, event labels, physiological signals, task variables) are position-agnostic and unchanged

**Effort for Phase 2:** Medium. The 2D functions exist in `non_local_detector.analysis.distance2D`; main work is wiring them in, dispatching on 1D/2D, and the HPD fix.

---

## Phase 3: Add 2D-to-1D Projection Utility

### 3.1 Project 2D posterior to 1D

**Currently missing** from both this repo and `non_local_detector`.

Need a function that:
1. Takes a 2D posterior over (x, y) bins
2. Maps each (x, y) bin to its nearest position on the linearized track (using track graph node positions + `track_linearization` logic)
3. Sums/accumulates probabilities into 1D position bins (marginalizing along the track-perpendicular axis)

This enables computing all existing 1D metrics (1D HPD in cm, 1D ahead/behind distance, normalized position scatter) from 2D decoding results.

**Where to put it:** `src/continuum_swr_replay/analysis/hippocampus/projection.py` (new file), or add to `posterior_analysis.py`.

### 3.2 Project 2D MAP estimate to 1D

For any MAP (x, y) position, find the nearest point on the track graph and its linear position. The `track_linearization` package already does this for raw position data — reuse that logic:
- Project (x, y) onto nearest track segment → get `projected_x`, `projected_y`, `track_segment_id`
- Convert to `linear_position` using edge order and edge spacing

### 3.3 Validate projection round-trip

- Decode a few sessions in both 1D and 2D
- Project 2D results to 1D
- Compare with native 1D decoding results
- Expect similar but not identical results (2D decoding has different spatial smoothing and bin geometry)
- This comparison itself is scientifically interesting and could be a supplementary figure

**Effort:** Medium. Conceptually straightforward but track junction geometry needs care (a single (x, y) point near a junction could project to multiple track segments).

---

## Phase 4: Update Figures

### Figure 1 — Model schematic + simulated example
- Uses simulated 1D data to illustrate the SSSM
- **No change needed.** 1D is pedagogically clearer for a methods figure.

### Figure 2 — State probability distributions + triggered averages

| Panel | Data used | Change needed |
|-------|-----------|---------------|
| 2a: State probability histograms | `Local_prob`, `Non-Local_*_prob`, `No-Spike_prob` | None — position-agnostic |
| 2b: Triggered averages (theta, MUA, SWR, speed) | Physiological signals aligned to event times | None — position-agnostic |
| Distance from animal | `ahead_behind_distance` | Recomputed via `get_ahead_behind_distance2D`, or project to 1D. Scalar output either way. |

### Figure 3 — Event characteristics (HPD, distance, position correlations)

| Panel | Data used | Change needed |
|-------|-----------|---------------|
| Event duration distribution | Event start/end times | None |
| HPD distribution | `hpd` column | **Decision:** show 2D HPD (cm^2) natively, or project to 1D (cm) for comparability with literature. Recommend showing projected 1D HPD for main figure, 2D HPD as supplement. |
| Distance from animal | `abs(ahead_behind_distance)` | Works with 2D-computed distance (scalar output) |
| Actual vs decoded position scatter | `linear_position / max_linear_position` vs `map_estimate / max_linear_position` | **Decision:** (a) project 2D MAP to 1D and keep this plot as-is, (b) replace with 2D scatter (actual x,y vs decoded x,y), or (c) show both. Recommend (a) for main figure, (b) as supplement. |
| MAP speed distribution | `map_speed` | Recomputed from 2D graph distances. Histogram shape should be similar. |
| Triggered averages by SWR overlap | Physiological signals | None |

### Figure 4 — mPFC GLM modulation
- Uses only event times and spike data
- **No change needed** — entirely position-agnostic

### Figurl interactive visualization
- `create_2D_decoding_plot` **already exists** in `visualization/figurl.py` (lines 717-870)
- Can offer both 1D (projected) and 2D views
- `create_interactive_1D_decoding_figurl` still works with projected data

### New 2D-specific figures (optional, for supplement)
- 2D replay trajectory visualization (MAP position over time on arena map)
- Head direction similarity distribution (`head_direction_similarity` from distance2D)
- 2D HPD spatial coverage maps
- 1D vs 2D decoding comparison (round-trip validation from Phase 3.3)

**Effort:** Low-Medium. Most figures work as-is with projected 1D data. Figure 3 position scatter is the main decision point.

---

## Phase 5: Rerun Decoding

- ~109 sessions across 5 animals (primarily HPC)
- 2D is more expensive: more position bins (2D grid vs ~253 1D bins), larger likelihood matrices
- Estimated 2-5x slower per session depending on arena size and bin count
- GPU memory may constrain bin count — monitor during test runs in Phase 1.2
- Save both 2D results and 1D-projected derivatives

**Effort:** High (compute time), Low (code — just rerun with `--2d` flag).

---

## Cost-Benefit Analysis

### Benefits

1. **Richer spatial information**: 2D captures off-track replay and lateral position that 1D linearization discards. If mental representations include 2D structure (shortcuts, off-track locations), 1D misses this.

2. **Eliminates linearization artifacts**: Track junctions where two segments are spatially close but linearly distant create ambiguity in 1D. 2D preserves true spatial relationships.

3. **Better distance metrics**: `get_ahead_behind_distance2D` and `get_2D_distance` with the 2D track graph give distances that respect physical space while following track structure — no forced linearization choices.

4. **New analyses become possible**: Direction of replay relative to heading (via `head_direction_similarity`), 2D trajectory classification, off-track decoding events.

5. **Backward compatible**: Can project back to 1D for any existing analysis. All current figures and metrics are reproducible.

6. **Infrastructure exists**: `non_local_detector` already supports 2D decoding, 2D distance functions (`distance2D.py`), 2D MAP estimation, 2D figurl. The gap is smaller than it appears.

### Costs

1. **Compute cost**: 2D decoding is 2-5x more expensive per session due to larger state space. ~109 sessions need reprocessing.

2. **HPD meaning changes**: 1D HPD is a length (cm), 2D HPD is an area (cm^2). Cross-study comparison with 1D literature is harder. Mitigation: always compute projected 1D HPD alongside native 2D HPD.

3. **`get_HPD_spatial_coverage` is broken for 2D** in `non_local_detector` (hardcoded `.position` coordinate). Needs a fix upstream or a wrapper. This is the biggest dependency risk.

4. **2D-to-1D projection utility doesn't exist yet**: Must be written and validated. Track junction geometry needs care (ambiguous nearest-segment assignment near junctions).

5. **Place bin size tuning**: `PLACE_BIN_SIZE_2D = 3.5 cm` is untested. May need empirical adjustment for resolution vs. computation tradeoff.

6. **Storage**: 2D results are larger (more position bins per time step). Not a major concern but nonzero.

7. **Figure 3 position scatter**: The 1D actual-vs-decoded plot is clean and interpretable. A 2D equivalent is harder to visualize compactly. Mitigation: project to 1D for this figure.

8. **Reviewer familiarity**: Most replay literature uses 1D decoding. 2D results may need extra explanation. Mitigation: show projected 1D in main figures, 2D in supplement.

### Verdict

**Benefits outweigh costs.** The projection-back-to-1D strategy means nothing is lost. The main new code is:
- Phase 1: CLI plumbing (~10 lines)
- Phase 2: Wire in existing 2D functions + HPD fix (~100-200 lines)
- Phase 3: 2D→1D projection utility (~100-150 lines)
- Phase 4: Minimal figure changes (most work as-is with projection)

The biggest risks are the `non_local_detector` HPD bug and the projection utility correctness at track junctions. Both are tractable.

---

## Implementation Order

1. **Create branch** `feature/2d-decoding`
2. **Phase 1.1-1.2**: Enable flag, test on 2-3 sessions
3. **Phase 3.1-3.2**: Build projection utility (needed before Phase 2 can produce 1D-compatible output)
4. **Phase 2.1-2.4**: Wire in 2D analysis functions with 1D/2D dispatch
5. **Phase 3.3**: Validate 1D vs projected-2D comparison
6. **Phase 4**: Update figures (minimal if projecting to 1D)
7. **Phase 5**: Full rerun of all sessions
8. **Merge** after validation

---

## Key Files Modified

| File | Change | Phase |
|------|--------|-------|
| `scripts/run_decoding_bandit.py` | Add `--2d` CLI flag, pass `is_1d` through | 1 |
| `src/.../parameters.py` | Validate `PLACE_BIN_SIZE_2D`, add naming conventions | 1 |
| `src/.../analysis/hippocampus/posterior_analysis.py` | 2D dispatch in `get_posterior_stats`, `get_map_speed` | 2 |
| `src/.../analysis/hippocampus/data_aggregation.py` | Update column structure for 2D MAP | 2 |
| `non_local_detector` (upstream or wrapper) | Fix `get_HPD_spatial_coverage` for 2D | 2 |
| `src/.../analysis/hippocampus/projection.py` (new) | 2D→1D posterior and MAP projection | 3 |
| `scripts/generate_figure03.py` | Update position scatter panel (if not projecting) | 4 |
| `src/.../visualization/figurl.py` | Wire up `create_2D_decoding_plot` in pipeline | 4 |

## Key Files Unchanged

- All event detection/analysis (`event_detection.py`, `event_analysis.py`, `triggered_analysis.py`) — position-agnostic
- Data loading (`load_data_bandit.py`, `position.py`, `task_variables.py`, `trials.py`) — linearization still runs, used for task variables and projection
- Figure 1, Figure 2a/2b, Figure 4 scripts — position-agnostic or use simulated data
- All neural data loading (spikes, LFP, ripples) — independent of position representation
