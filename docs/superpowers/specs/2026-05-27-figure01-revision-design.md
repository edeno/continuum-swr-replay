# Figure 1 Revision — Design

**Date:** 2026-05-27
**File touched:** `scripts/generate_figure01.py`
**Output:** `paper/figures/figure01/figure01.{pdf,png}`

## Motivation

Three pedagogical gaps in the current Figure 1:

1. **Panel (b) only covers 2 of 4 states "winning."** Local-wins and Non-Local-wins
   scenarios are shown; No-Spike-wins is absent. The reader cannot see what makes
   the No-Spike state light up.
2. **Spikes do not visibly "get in" to the model.** Panel (b) shows likelihood
   curves λ(x) but never the observed spike trains that produce them. Panel (d)
   has a raster, but it lives downstream of the model description in (b).
3. **Panel (a)'s self-loops are visually subliminal**, and the cross-state
   transitions carry no information beyond "this connection exists." The
   self-probability labels in the figure (0.98 for the two non-local states) are
   wrong — the model uses 0.96 for both Non-Local states.

## Scope

In scope:

- Panel (a): self-loops more prominent and correctly labeled; cross-state
  arrows redrawn as two-way directional arrows with thickness encoding the
  actual transition probabilities.
- Panel (b): restructured into 3 scenarios (Local wins, Non-Local wins,
  No-Spike wins) with a shared spike raster strip above them.
- Panel (c), Panel (d): height-only adjustments; no content changes.
- Overall figure: row heights and width ratios re-tuned to fit the new
  content within `PAGE_HEIGHT`.

Out of scope:

- Any change to the underlying detector code, simulation parameters, or fitted
  decoding for Panel (d).
- Any change to figure 2/3/4 — even the related Local-position-std behavior or
  state-color palette.

## Design

### Panel (a): State Diagram with Full Transitions

Same 2×2 spatial arrangement of state circles. Two changes:

**Self-loops.** Replace the current tangent arcs with prominent self-loops
drawn on the outside corner of each state circle (NW for Local, NE for
No-Spike, SW for Continuous, SE for Fragmented — routed away from the diagonal
cross-arrow paths). The self-probability label sits on the outward-facing
midpoint of the arc. Values match the actual transition matrix:

| State | Self-probability label |
|----|----|
| Local | 0.99 |
| No-Spike | 0.98 |
| Non-Local Continuous | **0.96** (current figure says 0.98 — bug) |
| Non-Local Fragmented | **0.96** (current figure says 0.98 — bug) |

**Cross-state arrows.** All 6 state pairs are connected by a **single
double-headed line** (one bidirectional arrow per pair, arrowheads at both
ends). 6 lines total. **No thickness encoding; no numeric labels.** The
diagram conveys topology only: every state can transition to every other
state. The actual numbers live in the transition kernels in panel (c) and
implicitly in panel (d)'s decoded trajectories.

Rationale for dropping thickness: with only 3 visual tiers across 12 arrows
in a small panel, the middle tier (0.005 / 0.0067, 1.34× apart) is
indistinguishable at print resolution, and pruning the encoding back to "just
show connectivity" matches the pedagogical role of panel (a) — introduce the
state machine. The asymmetry can be discussed in the figure caption if
needed.

**Self-prob source.** The self-probabilities (0.99, 0.98, 0.96, 0.96) are
**pulled from the detector at runtime** rather than hardcoded:

```python
from non_local_detector import NonLocalSortedSpikesDetector
matrix = NonLocalSortedSpikesDetector().discrete_transition_type.values
self_probs = matrix.diagonal()
```

This prevents the figure from going silently stale if upstream defaults
change. The full transition matrix (for reference; not drawn in the figure):

```
              Local   No-Spike  Continuous  Fragmented
Local         0.99    0.005     0.0025      0.0025
No-Spike      0.0067  0.98      0.0067      0.0067
Continuous    0.0175  0.005     0.96        0.0175
Fragmented    0.0175  0.005     0.0175      0.96
```

**Self-loop geometry.** Loop radius ≈ 0.08 in panel-A axis units, outer
extent ~0.10 beyond the circle edge, routed to the outward corner of each
state's grid cell: NW for Local, NE for No-Spike, SW for Continuous, SE for
Fragmented. This keeps the loops in figure margin space and away from the
diagonal cross-arrows that pass through the panel center.

### Panel (b): 3 Scenarios with Shared Spike Raster

Restructured into two vertical zones inside the row 0 right-hand slot.

**Top zone — shared spike raster strip** spanning the full width of panel (b):

- ~6–8 simulated neurons, ordered by place-field peak position.
- Raster x-axis on `[0, 1]`. Three highlighted time bins centered at
  `x = 0.17`, `0.50`, `0.83` so each bin sits directly above its scenario
  block. Highlight bands span `±0.08` around each center and use light gray
  shades (e.g., `#dddddd`, `#cccccc`, `#bbbbbb` for bin₁/bin₂/bin₃) —
  scenario state colors are reserved for the cells below.
- Spikes drawn as black ticks. Bins labeled `bin₁`, `bin₂`, `bin₃` along the
  top.
- **Animal-position trace** drawn in a separate thin axis at the bottom of
  the raster strip (not overplotted on spike rows), magenta line, height
  ~15% of the raster strip's vertical extent. This keeps the spike ticks
  legible and the position trace clearly attributed.
- A short downward connector under each highlighted bin pointing into the
  scenario block below.

**Bottom zone — 3 scenario blocks side by side**, in order left to right:

| Scenario | Bin content | Visual rule |
|----|----|----|
| **1. Local wins** | spikes from cells whose place fields sit at the animal's position | Local cell shows a tall bump at `x_animal`; other three cells show modest/diffuse curves (current behavior). |
| **2. Non-Local wins** | spikes from cells whose place fields are far from the animal | Continuous/Fragmented cells show prominent bumps away from the magenta `x_animal` line; Local cell is small (current behavior). |
| **3. No-Spike wins** | empty bin (no spikes) | No-Spike cell shows its horizontal likelihood line drawn at `y = 0.7`; Local/Continuous/Fragmented cells show small bumps peaking at `y = 0.4`. |

Each scenario block keeps the existing 2×2 layout (Local + No-Spike on top,
Continuous + Fragmented on bottom).

**Note on the No-Spike "draw it taller" rule.** The four cells in panel (b)
show **per-state likelihoods**, not posteriors. The No-Spike state is
bin-less — its likelihood is a single scalar (not a function of position),
so drawing it as a horizontal line whose height equals the likelihood value
is faithful. The Local/Non-Local likelihoods *are* functions of position,
shown as curves whose peak height represents the maximum likelihood across
bins. In an empty bin, No-Spike's scalar likelihood exceeds the peak of each
of the other states' likelihood curves — so a horizontal line at `y = 0.7`
above curves peaking at `y = 0.4` faithfully says "the No-Spike likelihood
value is the largest of the four state likelihood values," which is the
state-marginal story the panel is trying to teach.

**Raster-strip height share.** Inside row 0 of the figure, the panel-B slot
is split into a 2-row sub-gridspec with `height_ratios = [0.3, 0.7]`: the
shared raster occupies the top 30%, the three scenario blocks share the
bottom 70%.

### Panels (c) and (d): height-only changes

No content changes. Adjustments:

- Row heights change from `[1.4, 0.55, 1.7]` to approximately
  `[1.7, 0.55, 1.55]`. Row 0 grows ~20% to accommodate the shared raster; row 2
  shrinks ~10% in compensation. Panel (d) is currently over-allocated for 30
  neurons so this is safe.
- Total figure height stays at `min(TWO_COLUMN × 1.247, PAGE_HEIGHT)`, or
  grows by at most ~5% if the height math doesn't fit.

### Overall layout

```
┌──────────────────────────────────────────────────────────────────────┐
│ a               b                                                    │
│                 [   shared spike raster (3 highlighted bins)   ]     │
│ ┌──┐  ┌──┐                                                           │
│ │L │  │NS│      Local wins      Non-Local wins      No-Spike wins    │
│ └──┘  └──┘      ┌─┬─┐           ┌─┬─┐               ┌─┬─┐            │
│ ┌──┐  ┌──┐      │L│NS│          │L│NS│              │L│NS│           │
│ │C │  │F │      ├─┼─┤           ├─┼─┤               ├─┼─┤            │
│ └──┘  └──┘      │C│F│           │C│F│               │C│F│            │
│                 └─┴─┘           └─┴─┘               └─┴─┘            │
├──────────────────────────────────────────────────────────────────────┤
│ c   Local→L    No-Spike→NS    Continuous→C    Fragmented→F           │
│     [unif]     [identity]     [gaussian]      [uniform]              │
├──────────────────────────────────────────────────────────────────────┤
│ d   spike raster                                                     │
│     non-local posterior + true position                              │
│     state probabilities                                              │
│     speed                                                            │
└──────────────────────────────────────────────────────────────────────┘
```

Row 0 width split changes from `[1, 2.2]` to `[1, 3.3]` so panel (a) keeps
roughly its current absolute width while panel (b) absorbs the extra width for
the third scenario.

## Implementation Phases

Three phases, each independently runnable and committable. After each phase
the figure builds and renders; the user reviews before moving on.

### Phase 1 — Panel (b) scenarios (3 columns, no raster yet)

Goal: extend panel (b) from 2 scenarios to 3, including the new No-Spike-wins
scenario. No shared raster yet — each scenario still stands alone.

Changes:

- Add a third scenario tuple to `plot_observation_models` for No-Spike wins
  with header `"No-Spike wins"`.
- Modify `_draw_state_cells` so that when a `no_spike_wins=True` flag is
  passed, the No-Spike cell draws its horizontal line at `y = 0.7` and the
  Local/Continuous/Fragmented cells draw small bumps peaking at `y = 0.4`.
  The existing No-Spike cell drawing (`axhline(0.05, …)`) is kept as the
  default for Local-wins and Non-Local-wins scenarios.
- Extend the outer 2×2 gridspec in `plot_observation_models` from 2 columns
  to 3 columns.
- Adjust `gs_row0` width ratios from `[1, 2.2]` to `[1, 3.3]`.
- Adjust panel (a)'s axes data limits / anchor so the circles still align
  with the bottom of the scenario blocks.
- Confirm `label_panels(b_axes[0], …)` still labels the top-left Local-wins
  cell — `b_axes` now has length 12 (4 cells × 3 scenarios) but index 0 is
  still the first cell drawn (Local-wins → Local).

Verification:

- Figure renders without overflow.
- Panel (b) shows 3 scenarios; in the No-Spike-wins one, the gray horizontal
  No-Spike line sits visually above the four state bumps.
- Panel (a) circles still align with the bottom of the leftmost scenario
  block.
- Panel "b" label still attaches to the top-left Local-wins cell.

### Phase 2 — Shared spike raster above the 3 scenarios

Goal: add the shared spike raster strip and the bin-to-scenario connectors.

Changes:

- Add helper `_draw_shared_spike_raster(fig, gs_slot, animal_pos_local,
  animal_pos_nonlocal, bin_colors)` that:
  - Simulates / hand-builds a small set of spike trains (~6–8 neurons,
    ordered by place-field peak).
  - Draws ticks for the spikes.
  - Highlights 3 vertical bands corresponding to the 3 scenario bins, in
    light gray shades.
  - Draws the animal-position trace as a magenta line.
  - Returns the x-positions of the 3 bin centers so the caller can draw
    downward connectors.
- Restructure the panel (b) slot from `gs_outer = subgridspec(2, 2,
  height_ratios=[0.12, 1.0])` (title + content for 2 scenarios) to
  `gs_outer = subgridspec(3, 3, height_ratios=[raster, title, content])` so
  the raster spans all 3 columns above the titles, and the title and content
  rows are repeated per scenario.
- Re-tune row 0 height in `height_ratios` (and figure height if needed).

Verification:

- Raster ticks line up visually with the corresponding scenario bins below.
- Animal-position trace is visible and uses magenta (matching the dashed
  lines in the scenario cells).
- The 3 highlighted bins map left-to-right to Local-wins → Non-Local-wins →
  No-Spike-wins.

### Phase 3 — Panel (a) self-loops and cross-arrows

Goal: redraw panel (a) with prominent self-loops and double-headed
cross-state arrows (one line per pair, no thickness encoding, no labels on
cross-arrows).

Changes:

- Pull the actual diagonal probabilities from
  `NonLocalSortedSpikesDetector().discrete_transition_type.values.diagonal()`
  at the top of `plot_state_diagram`. This corrects the current `0.98` for
  Continuous and Fragmented to `0.96` automatically.
- Replace `loop_offsets` / `FancyArrowPatch(connectionstyle="arc3,rad=0.4")`
  with a proper self-loop drawn as a `FancyArrowPatch` arc of radius ≈ 0.08
  in axis units, outer extent ≈ 0.10 beyond the circle edge, routed per
  state to its outward corner (NW for Local, NE for No-Spike, SW for
  Continuous, SE for Fragmented). The self-probability label sits on the
  outward midpoint of the arc.
- Replace `cross_transitions = [...]` (currently an irregular subset of 8
  one-directional arrows) with a list of 6 unordered state pairs.
- For each pair, draw a **single** `FancyArrowPatch` from circle A to circle
  B with `arrowstyle="<->"` (arrowheads at both ends). No curved arc, no
  thickness encoding, no label.

Verification:

- All four self-loops are visible at print resolution and carry the correct
  probability label (0.99, 0.98, 0.96, 0.96).
- Each pair of state circles is connected by exactly one double-headed line.
- Self-loops do not visually collide with the cross-arrows or with the
  state labels inside the circles.

## Risks and Edge Cases

- **Width budget.** The figure is locked at `TWO_COLUMN` width. With 3
  scenarios in panel (b), each scenario cell becomes ~⅔ of its current width.
  Place-field bumps may look squished. Mitigation: reduce per-cell whitespace
  (`wspace` in the inner gridspec) and accept slightly smaller bumps.
- **Height budget.** Adding the raster strip without growing the figure
  requires shrinking panel (d). Panel (d)'s raster (30 neurons) is currently
  over-tall, so this is safe — but if the layout still doesn't fit,
  `fig_height` can grow up to `PAGE_HEIGHT` before being clamped.
- **Phase 1 standalone rendering.** Phase 1's figure is valid on its own —
  the 3 scenarios stand without a shared raster, just as the current 2
  scenarios do.
- **Self-loop / cross-arrow collision.** Routing self-loops to the outward
  corner of each circle (NW/NE/SW/SE) keeps them in the figure margin where
  no cross-arrows live. The diagonal cross-arrows (Local↔Fragmented and
  No-Spike↔Continuous) cross through the center of the 2×2 — self-loops on
  the outward corners stay out of that path.
- **Bin-to-scenario x-alignment.** The shared raster's three highlighted
  bins must visually fall above their scenario blocks. The scenario
  sub-gridspec uses `wspace` between columns; the raster axis uses absolute
  positions on `[0, 1]`. Bin centers `0.17 / 0.50 / 0.83` are chosen to
  match the scenario column centers under the new width split — verify
  alignment after rendering and tune if needed.
- **Magenta-on-gray contrast.** The animal-position trace (magenta, color
  `#FF00FF`-ish) must remain visible against the light-gray highlight bands
  (`#dddddd` / `#cccccc` / `#bbbbbb`). At the chosen shade range the
  contrast is fine; if shades darken the trace can switch to a thicker
  line.
- **Self-prob bug fix scope.** Prior versions of this figure (commits
  through `fa5bb0a`) display `0.98` for Continuous and Fragmented self-
  probabilities. The correction to `0.96` should be flagged in the commit
  message so reviewers and slide-deck users know to refresh derived
  materials.

## Self-Review Notes

- Placeholders: none. All `y ≈ 0.7` / `y ≈ 0.4` etc. values are concrete
  starting points; implementation can tune within the existing
  `set_ylim(-0.05, 1.2)` range.
- Internal consistency: scenario order (Local → Non-Local → No-Spike) is used
  consistently in the layout diagram, the table, and the implementation
  phases.
- Scope: each phase is one focused commit; the design is single-figure.
- Ambiguity: bin highlight shade and exact No-Spike `y` placement are left as
  implementation tuning, not specified contractually.
