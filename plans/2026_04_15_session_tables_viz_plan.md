# Session-tables visualization plan

Plan for `plot_session_tables` — a single-session inspection figure that
visualizes the transformation from raw decoder output through the four
canonical tables, aligned on a shared time axis.

**Primary goal**: see how data flows through the pipeline and verify
that every derived variable is correct. Each panel corresponds to one
step of the derivation chain, so a bug in a derived variable (e.g.,
`arm_count_hard` not matching the arm-posterior heatmap above it, or
`next_arm` disagreeing with the ground-truth trial rail below it)
surfaces as a visual mismatch between adjacent panels.

**Secondary goal**: the future-covariate debug row specifically
targets `next_arm` / `next_well` / `is_next`, where the only way to
catch a bug is to compare the canonical value against the independent
`trials` ground truth.

## Scope

- Positioned as the canonical-tables analog of
  `plots.plot_detector`. Same aesthetic template (panels stacked
  `sharex=True`, `constrained_layout`, `height_ratios` with the
  posterior row weighted 3, `sns.despine(offset=5)`).
- Reuses as much existing infrastructure as possible — no reinvented
  palettes, posterior heatmaps, interval masks, or scaffolding.
- Single-session; multi-session summary is a separate tool.
- Static matplotlib. Interactivity is out of scope for v1.
- Returns both a Figure and a mismatch DataFrame so CI / batch audit
  scripts can call the function headless.

## Design principle: visualize the transformation

Each panel beyond the core shows one step of the canonical-table
derivation chain. A bug is visible as a mismatch between adjacent
layers.

```
raw conditional posterior (time × position_bin)       ← plot_conditional_non_local_posterior
        │   marginalize via posterior_to_arm_marginal
        ▼
arm posterior (time × 9 arms)                          ← optional heatmap panel
        │   per-bin MAP → decoded_arm
        │   w_t-weighted aggregation
        ▼
event-level scalars                                    ← optional per-event panels
(posterior_peak_mean, entropy, HPD width)
        │   hard-threshold counts
        ▼
arm_count_{hard,soft,posterior_mass} / arm_fraction   ← optional 9-row panels
per (event, arm)
        │   trial-context join
        ▼
current_arm / prev_arm / next_arm / last_rewarded_arm ← arm-marker row
per event
        │   compare against trials
ground truth: trials                                   ← trial rail
        │
mismatch pointers (red where disagreeing)              ← future-covariate debug row
```

## Core panels (always shown — reuse directly)

From `plots.plot_detector`:

- `plot_ripple_consensus_trace_zscore` (SWR LFP z-score)
- `plot_multiunit_firing` (MUA)
- `plot_acausal_probabilities` (Local / Non-Local / MUA state probs)
- `plot_conditional_non_local_posterior` — **core panel**, existing
  convention already overlays animal's `linear_position` on the
  position-bin heatmap
- `plot_speed`

New:

- **Trial rail** — step plot of `from_well → to_well` over time from
  `trials`. Rewards at trial end marked with a star; patch changes
  marked with a color segment. This is the independent ground truth
  for the future-covariate debug.

## Optional panels (opt-in by name in `panels=(...)`)

### Event-arm content group (posterior → per-arm count)

| Panel name | Source | Representation |
|---|---|---|
| `"arm_posterior_heatmap"` | `event_sequence_table.arm_posterior_{0..8}` | 9-row heatmap; `decoded_arm` overlaid (black dots), `track_segment_id` overlaid (colored line for leaf arms; hatched for trunk) |
| `"arm_count_hard"` | `event_arm_table.arm_count_hard` | 9-row heatmap at `event_midpoint_time` |
| `"arm_count_soft"` | `event_arm_table.arm_count_soft` | same |
| `"arm_posterior_mass"` | `event_arm_table.arm_posterior_mass` | same |
| `"arm_fraction"` | `event_arm_table.arm_fraction` | same, scale 0–1 |

### Event-level scalar group (arm-marginal → event scalar)

| Panel name | Source | Representation |
|---|---|---|
| `"posterior_peak_mean"` | `event_table` | stem at `event_midpoint_time`, y ∈ [1/9, 1] |
| `"posterior_entropy_mean"` | `event_table` | stem, y ∈ [0, log 9] |
| `"hpd_widths"` | `event_table` (3 cols) | three overlaid lines 50/80/95 |
| `"time_since_stop"` | `event_table` | stem, y in seconds |
| `"n_nonlocal_bins"` | `event_table` | stem, y = bin count |

### Context / future-covariate debug group

| Panel name | Shows |
|---|---|
| `"arm_markers"` | `current_arm` (circle), `prev_arm` (▲), `next_arm` (▼), `last_rewarded_arm` (★) at `event_midpoint_time` |
| `"mismatch_pointers"` | Per event: arrow up to canonical `next_arm` in the arm-markers row, arrow down to trial rail's next-trip `to_well`. Red where the two disagree. Paired with returned mismatch DataFrame. |
| `"context_flags"` | `is_after_reward`, `is_after_omission`, `is_after_switch`, `is_after_stay`, `is_swr`, `is_mua` as colored ticks at `event_midpoint_time`, stacked on one short row |

### Per-arm predictor group (trials → per-arm event-arm row)

| Panel name | Source | Representation |
|---|---|---|
| `"visit_recency"` | `event_arm_table.visit_recency` | 9-row heatmap |
| `"cumulative_reward"` | `event_arm_table.cumulative_reward` | 9-row heatmap |
| `"recent_reward_rate"` | `event_arm_table.recent_reward_rate` | 9-row heatmap, 0–1 |
| `"nominal_reward_prob"` | `event_arm_table.nominal_reward_prob` | 9-row heatmap (currently NaN — deferred column) |
| `"dist_to_arm"` | `event_arm_table.dist_to_arm` | 9-row heatmap |
| `"arm_dist_rank"` | `event_arm_table.arm_dist_rank` | 9-row heatmap, 1 = closest |

### Event / stop activity

| Panel name | Source | Representation |
|---|---|---|
| `"event_bars"` | `event_table` | one bar per event spanning `event_start` → `event_end`; height = `n_nonlocal_bins`; fill = SWR/MUA type |
| `"stop_activity"` | `stop_table` | vertical bar at stop center; height = `n_events`; hollow vs filled = `has_replay`; "S"/"M" marks for `is_swr_stop`/`is_mua_stop` |

## Data contract

```python
def plot_session_tables(
    data: dict,                          # load_data-style: trials, position_info, hse_times, ...
    detector,                            # for plot_conditional_non_local_posterior
    results: xr.Dataset,                 # same
    event_table: pd.DataFrame,
    event_arm_table: pd.DataFrame,
    event_sequence_table: pd.DataFrame,
    stop_table: pd.DataFrame,
    *,
    time_window: tuple[float, float] | None = None,
    panels: Sequence[str] = DEFAULT_PANELS,
    highlight_mismatches: bool = True,
    save_path: Path | None = None,
) -> tuple[plt.Figure, pd.DataFrame]
```

`DEFAULT_PANELS` renders the core panels plus the future-covariate
debug group. Users expand with, e.g.,
`panels=("core", "arm_posterior_heatmap", "arm_markers",
"mismatch_pointers", "hpd_widths")`.

## Promoted helpers (chunk 1, pure refactor)

Two helpers currently live in script-local code and need promotion
before any viz panels land:

1. **`build_arm_color_map`** — currently inline at
   `scripts/plot_replay_content_time_slice.py:67`. Patch-grouped
   9-arm palette. Move to the visualization package; update the one
   call site.
2. **`intervals_to_timebins(intervals, time_idx, *, value_col=None,
   fill_value=np.nan)`** — generalization of
   `plots.py::_intervals_to_mask` that carries a value (or returns
   a boolean mask when `value_col=None`). Essential for step-holding
   per-event columns onto a fast timebin axis.

## Conventions (confirmed)

- **Name**: `plot_session_tables` (pairs with `plot_detector`).
- **Arm palette**: patch-grouped (intra-patch lightness variation).
- **Trial rail**: step plot of well ID, stars at reward, colored
  segments at patch change.
- **Mismatch tolerance**: strict equality; `(NaN, NaN)` at session
  end counts as a match.
- **Panel ordering**: user's specified order in `panels=(...)`;
  `DEFAULT_PANELS` defines the fallback.
- **Track vs. linear position**: inherit the existing
  `plot_conditional_non_local_posterior` convention (animal's
  `linear_position` overlaid on position-bin heatmap). The arm-posterior
  heatmap panel overlays `track_segment_id` and `decoded_arm` at the
  arm level.
- **Panel placement for opt-ins**: user's order.

## What's intentionally left out

- `event_id`, `stop_id`, `event_duration` — identifiers /
  redundant; surfaced as tooltip / annotation, not a panel.
- `animal`, `day`, `epoch`, `non_local_event`, `stop_idx` — figure
  title or annotations.
- `arm_patch_id`, `current_patch`, `prev_patch`, `next_patch` —
  encoded via palette grouping.
- `n_unique_arms` — redundant with heatmap.
- `n_junction_bins` — warn-only annotation when nonzero.
- All interaction terms (`time_x_*`, `swr_x_*`, `mua_x_*`,
  `switch_x_*`, `omission_x_*`) — derived products; atomic columns
  already shown.
- `dot["head_position_x/y"]`, `head_orientation`, `head_velocity_x/y`,
  `projected_x_position/y_position`, `max_coverage`, `max_distance`,
  `max_linear_position`, `patch_id` — not alignment-critical /
  redundant with shown panels.
- `dot["theta_phase"]`, `dot["theta_power_zscore"]` — orthogonal;
  existing `plot_theta_*` panels available via `plot_detector`.
- `dot["map_estimate"]` (raw), `dot["ahead_behind_distance"]`,
  `dot["hpd"]` — posterior heatmap already shows the same info;
  `hpd` can be added as an optional panel if needed.
- Individual state-probability columns (`Local_prob`, `No-Spike_prob`,
  continuous/fragmented split) — `plot_acausal_probabilities` shows
  all four; `nonlocal_prob` sum is what drives event detection.
- `SWR_event`, `MUA_event` integer labels — SWR/MUA presence shown
  via event bars and stop marks.
- `trials` columns `duration`, `goal_arrival_time`, `from_patch`,
  `to_patch` — redundant with start/end/well columns.

## Implementation order (one commit per line)

1. Promote `build_arm_color_map` + `intervals_to_timebins`; update
   script consumers; test the promoted helpers.
2. Scaffold `plot_session_tables` with core panels only (reuses
   existing `plot_detector` helpers) + new trial rail.
3. Future-covariate debug row: `"arm_markers"` + `"mismatch_pointers"`
   + returned mismatch DataFrame + fixture test that injects an
   off-by-one and asserts the mismatch table catches it.
4. Event-arm content group panels (`"arm_posterior_heatmap"`,
   `"arm_count_*"`, `"arm_fraction"`, `"arm_posterior_mass"`).
5. Event-level scalar group panels.
6. Per-arm predictor group panels.
7. Event / stop activity panels.
8. CLI driver at `scripts/plot_canonical_session.py`.
9. Reviewer pass + real-data run on a cached j16 session.

## Out of scope for v1

- Multi-session summary plots.
- Interactive zoom/pan (bokeh, mpl_interactions).
- Raw spike overlay (plot_detector has this).
- GLM coefficient plots.
- Paper-quality polish (use `scientific-figures-paper` skill later).
- Refactoring `plot_event_example_stack` to use the promoted helpers.
