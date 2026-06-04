# Figure 1 Revision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Revise Figure 1 of the paper to (1) add a third "No-Spike wins" scenario to panel (b), (2) add a shared spike raster above the three scenarios so the data flow into the model is visible, and (3) redraw panel (a) with prominent self-loops (correctly labeled) and double-headed cross-state arrows.

**Architecture:** All changes live in a single script — `scripts/generate_figure01.py`. No new files. Three phases, each independently renderable and committable. Verification is by rendering the figure with `uv run python scripts/generate_figure01.py` and visually inspecting `paper/figures/figure01/figure01.png`. There are no unit-level tests for the figure script itself (matches existing convention in this repo).

**Tech Stack:** Python 3.x, matplotlib (`gridspec`, `patches`), numpy, `non_local_detector` package (read-only — used to source the actual transition matrix diagonal).

**Spec:** `docs/superpowers/specs/2026-05-27-figure01-revision-design.md`

---

## File Structure

Only one file is modified:

- **Modify:** `scripts/generate_figure01.py` — three logical edits, one per phase:
  - Phase 1: extend `_draw_state_cells` with a `no_spike_wins` branch; extend `plot_observation_models` from 2 scenarios to 3; widen panel-B in `gs_row0`.
  - Phase 2: add a `_draw_shared_spike_raster` helper; restructure `plot_observation_models`'s sub-gridspec from 2-row to 3-row; tune main figure row heights.
  - Phase 3: pull self-probabilities from `NonLocalSortedSpikesDetector` defaults; redraw self-loops as prominent arcs in outward corners; replace cross-arrow list with 6 unordered pairs drawn as double-headed `FancyArrowPatch`es.

The verification artifact is `paper/figures/figure01/figure01.png` (and `figure01.pdf`).

---

## Phase 1 — Three Scenarios in Panel (b)

### Task 1: Extend `_draw_state_cells` to support a No-Spike-wins branch

**Files:**
- Modify: `scripts/generate_figure01.py:180-279` (the `_draw_state_cells` function)

- [ ] **Step 1: Open the file and read the existing function**

Read `scripts/generate_figure01.py:180-279` to understand the current signature and per-cell drawing logic. The function currently takes `(fig, gs_inner, x, likelihood, animal_pos)` and draws 4 cells in a 2×2 grid.

- [ ] **Step 2: Modify the function signature and body**

Replace the entire `_draw_state_cells` function (lines 180-279) with:

```python
def _draw_state_cells(fig, gs_inner, x, likelihood, animal_pos, no_spike_wins=False):
    """Draw the 4-state likelihood cells for one scenario.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    gs_inner : SubplotSpec
        A 2x2 sub-grid slot. Cells are laid out row-major:
        (Local, No-Spike) on top; (Continuous, Fragmented) on bottom.
    x : np.ndarray, shape (n_pts,)
        Position grid on [0, 1] for the schematic.
    likelihood : np.ndarray, shape (n_pts,)
        The place-field curve λ(x).
    animal_pos : float
        The animal's position in [0, 1]. Controls where the Local
        kernel sits and where the magenta dashed line is drawn.
    no_spike_wins : bool, default False
        When True, render the No-Spike-wins scenario: the No-Spike cell's
        horizontal line is drawn at y=0.7 (instead of y=0.05), and the
        Local/Continuous/Fragmented curves are scaled to peak at y=0.4 so
        the No-Spike line literally sits above them on the y-axis.

    Returns
    -------
    list of matplotlib.axes.Axes
        The four sub-axes (Local, No-Spike, Continuous, Fragmented).
    """
    configs = [
        ("Local", STATE_COLORS_SHORT[0]),
        ("No-Spike", STATE_COLORS_SHORT[1]),
        ("Non-Local\nContinuous", STATE_COLORS_SHORT[2]),
        ("Non-Local\nFragmented", STATE_COLORS_SHORT[3]),
    ]
    sub_axes = []

    # Scale factor for Local/Continuous/Fragmented bumps in the No-Spike-wins
    # scenario. 0.4 keeps the bumps clearly below the No-Spike line at 0.7.
    bump_scale = 0.4 if no_spike_wins else 1.0
    nospike_y = 0.7 if no_spike_wins else 0.05

    # Pick annotation placement so the arrow doesn't cross the formula text.
    # The text is anchored on its edge nearest the bump (ha=left/right) so it
    # only extends away from the bump; the arrow tail then sits at that edge
    # and the shaft sweeps across empty space toward xy.
    if animal_pos >= 0.5:
        annot_xtext = animal_pos - 0.25
        annot_ha = "right"
    else:
        annot_xtext = animal_pos + 0.25
        annot_ha = "left"

    for idx, (name, color) in enumerate(configs):
        row, col = idx // 2, idx % 2
        sub_ax = fig.add_subplot(gs_inner[row, col])
        sub_axes.append(sub_ax)

        if name == "Local":
            sigma_local = 0.05  # schematic kernel width on the [0,1] x-axis
            kernel = np.exp(-0.5 * ((x - animal_pos) / sigma_local) ** 2)
            local_like = likelihood * kernel * bump_scale
            sub_ax.plot(x, likelihood * bump_scale, color="#aaaaaa",
                        linewidth=0.8, alpha=0.4)
            sub_ax.fill_between(x, likelihood * bump_scale, alpha=0.2,
                                color="#aaaaaa")
            sub_ax.plot(x, local_like, color=color, linewidth=1.0)
            sub_ax.fill_between(x, local_like, alpha=0.45, color=color)
            sub_ax.axvline(animal_pos, color=ANIMAL_POS_COLOR, linewidth=0.8,
                           linestyle="--")
            # Arrow tip stays visible even when the bump is tiny
            peak_y = max(float(local_like.max()), 0.10)
            sub_ax.annotate(
                r"$\lambda(x)\,\mathcal{N}(x_{\mathrm{animal}}, \sigma_L)$",
                xy=(animal_pos, peak_y),
                xytext=(annot_xtext, 0.95),
                fontsize=5.5, color=color, fontweight="bold",
                arrowprops={"arrowstyle": "->", "color": color, "lw": 0.8},
                ha=annot_ha,
            )

        elif name == "No-Spike":
            sub_ax.axhline(nospike_y, color=color, linewidth=1.2, linestyle="-")
            sub_ax.axvline(animal_pos, color=ANIMAL_POS_COLOR, linewidth=0.8,
                           linestyle="--")
            # Position the "λ ≈ 0" annotation so it doesn't overlap the line.
            # In the default scenario, line sits at y=0.05 and text sits above
            # at axes-y=0.12. In the No-Spike-wins scenario, line moves up to
            # y=0.7 (≈ axes-y=0.6); place the text just below the line.
            annot_axes_y = 0.50 if no_spike_wins else 0.12
            sub_ax.text(
                0.5, annot_axes_y, r"$\lambda \approx 0$",
                ha="center", va="bottom", fontsize=6, color=color,
                fontweight="bold", transform=sub_ax.transAxes,
            )

        else:  # Continuous or Fragmented Non-Local
            sub_ax.plot(x, likelihood * bump_scale, color=color, linewidth=1.0)
            sub_ax.fill_between(x, likelihood * bump_scale, alpha=0.3,
                                color=color)
            sub_ax.axvline(animal_pos, color=ANIMAL_POS_COLOR, linewidth=0.8,
                           linestyle="--", alpha=0.4)

        sub_ax.set_xlim(0, 1)
        sub_ax.set_ylim(-0.05, 1.2)
        sub_ax.set_xticks([])
        sub_ax.set_yticks([])
        for spine in ["top", "right", "left"]:
            sub_ax.spines[spine].set_visible(False)
        sub_ax.spines["bottom"].set_linewidth(0.5)

        sub_ax.set_title(
            name, fontsize=6, color=color, fontweight="bold", pad=2,
        )
        # Label the position axis on the bottom row of each scenario only;
        # all four cells share the same [0, 1] position axis.
        if row == 1:
            sub_ax.set_xlabel("position", fontsize=6, labelpad=3)

    return sub_axes
```

- [ ] **Step 3: Render and verify nothing breaks for the existing two scenarios**

Run:

```bash
cd /cumulus/edeno/continuum-swr-replay
uv run python scripts/generate_figure01.py
```

Expected: command exits 0 and writes `paper/figures/figure01/figure01.png`. Open it. Local-wins and Non-Local-wins panels should look identical to before this step (because `no_spike_wins` defaults to `False`).

- [ ] **Step 4: Commit**

```bash
git add scripts/generate_figure01.py
git commit -m "Add no_spike_wins branch to _draw_state_cells

When called with no_spike_wins=True, draws the No-Spike cell at y=0.7
and scales the Local/Continuous/Fragmented curves to peak at y=0.4.
Default behaviour (no_spike_wins=False) is unchanged."
```

### Task 2: Add the third scenario in `plot_observation_models`

**Files:**
- Modify: `scripts/generate_figure01.py:282-337` (the `plot_observation_models` function)

- [ ] **Step 1: Replace the entire `plot_observation_models` function**

Replace the function with:

```python
def plot_observation_models(fig, gs_slot):
    """Show how each state's spike likelihood reacts under three scenarios.

    Three scenarios are placed side-by-side, each containing the 4-state
    2x2 grid (Local / No-Spike on top; Continuous / Fragmented Non-Local
    on bottom):

    - "Local wins": x_animal coincides with the place-field peak. The
      cell fires where its tuning says it should, so the Local
      likelihood (λ(x)·N(x_animal, σ_L)) is large at the animal's
      position. The Non-Local likelihood λ(x) is also peaked there, but
      spread across all bins, so per-bin Non-Local mass is lower.
    - "Non-Local wins": x_animal is far from the place-field peak. The
      cell fires where it "shouldn't" — Local likelihood is tiny
      because λ(x_animal) is near zero. Non-Local likelihood is still
      peaked at the place-field center, far from the animal,
      consistent with replay of a different location.
    - "No-Spike wins": the bin contains no spike. No-Spike's bin-less
      likelihood (a single scalar) exceeds the peak of each of the
      other states' position-dependent likelihood curves; the cell
      draws this as a horizontal line at y=0.7 sitting above the
      others' bumps at y=0.4.

    Returns all sub-axes (12 total), Local-wins block first then
    Non-Local-wins then No-Spike-wins. The first is used by
    ``label_panels`` for the panel "b" label.
    """
    n_pts = 200
    x = np.linspace(0, 1, n_pts)
    likelihood_center = 0.4
    likelihood = np.exp(-0.5 * ((x - likelihood_center) / 0.10) ** 2)

    # Outer 2x3: thin title row above content row, three scenarios as columns.
    gs_outer = gs_slot.subgridspec(
        2, 3, height_ratios=[0.12, 1.0], wspace=0.20, hspace=0.10,
    )

    scenarios = [
        # (header, animal_pos, no_spike_wins)
        ("Local wins", likelihood_center, False),
        ("Non-Local wins", 0.55, False),
        ("No-Spike wins", likelihood_center, True),
    ]

    all_axes = []
    for col, (header, animal_pos, no_spike_wins) in enumerate(scenarios):
        title_ax = fig.add_subplot(gs_outer[0, col])
        title_ax.axis("off")
        title_ax.text(
            0.5, 0.0, header,
            ha="center", va="bottom", fontsize=7, fontweight="bold",
            transform=title_ax.transAxes,
        )

        gs_scen = gs_outer[1, col].subgridspec(2, 2, wspace=0.10, hspace=0.55)
        all_axes.extend(
            _draw_state_cells(
                fig, gs_scen, x, likelihood, animal_pos,
                no_spike_wins=no_spike_wins,
            )
        )

    return all_axes
```

- [ ] **Step 2: Update the panel-B width allocation in `generate_figure`**

In `generate_figure` (around line 604), change:

```python
gs_row0 = gs_main[0].subgridspec(1, 2, wspace=0.15, width_ratios=[1, 2.2])
```

to:

```python
gs_row0 = gs_main[0].subgridspec(1, 2, wspace=0.15, width_ratios=[1, 3.3])
```

- [ ] **Step 3: Render and verify**

Run:

```bash
cd /cumulus/edeno/continuum-swr-replay
uv run python scripts/generate_figure01.py
```

Open `paper/figures/figure01/figure01.png` and verify:
- Panel (b) shows three scenarios in one row: Local wins, Non-Local wins, No-Spike wins.
- In the No-Spike-wins scenario, the gray horizontal No-Spike line is visibly above the Local/Continuous/Fragmented bumps.
- Panel (a) circles still roughly align with the bottom of the Local-wins scenario block.
- Panel "b" label is on the Local-wins Local cell (top-left of scenario 1).

If the cells look too cramped horizontally, try reducing `wspace=0.20` to `wspace=0.15` inside `gs_outer`.

- [ ] **Step 4: Commit**

```bash
git add scripts/generate_figure01.py
git commit -m "Add No-Spike-wins scenario to Figure 1 panel (b)

Extends panel (b) from two scenarios to three. New rightmost scenario
shows what happens in an empty bin: No-Spike's horizontal likelihood
line sits above the other three states' position-dependent likelihood
curves, illustrating why the model picks No-Spike in that bin.

Panel-B width ratio bumped from 2.2 to 3.3 to absorb the third scenario
column."
```

---

## Phase 2 — Shared Spike Raster Above the Three Scenarios

### Task 3: Add the `_draw_shared_spike_raster` helper

**Files:**
- Modify: `scripts/generate_figure01.py` (insert new helper just above `plot_observation_models`)

- [ ] **Step 1: Insert the new helper function**

Insert this function in `scripts/generate_figure01.py` immediately before the `plot_observation_models` definition (i.e., just before the `# ════…` separator at line 282):

```python
def _draw_shared_spike_raster(fig, gs_slot, seed=42):
    """Draw a shared spike-raster strip above the 3 scenario blocks.

    The strip shows ~7 simulated neurons (ordered by place-field peak)
    over a normalized time window x ∈ [0, 1]. Three highlighted bins —
    centered at 0.17, 0.50, 0.83 — sit above the three scenarios:
        bin₁ (Local wins): spikes from cells whose place fields cover
            the animal's position.
        bin₂ (Non-Local wins): spikes from cells whose place fields are
            far from the animal.
        bin₃ (No-Spike wins): empty bin (no spikes).
    Below the spikes a thin axis shows the animal-position trace
    (magenta line) — drawn on a separate axis so it doesn't visually
    compete with the spike ticks.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    gs_slot : SubplotSpec
        Slot in which to draw the raster (already a single row spanning
        the 3 scenario columns above).
    seed : int
        Random seed for the spike pattern (so the figure is
        reproducible).

    Returns
    -------
    list of matplotlib.axes.Axes
        [raster_ax, position_ax]
    """
    rng = np.random.default_rng(seed)

    # Inside the slot, split into a raster zone (~85%) and a thin position
    # trace below it (~15%).
    gs_inner = gs_slot.subgridspec(2, 1, height_ratios=[0.85, 0.15], hspace=0.05)
    raster_ax = fig.add_subplot(gs_inner[0])
    position_ax = fig.add_subplot(gs_inner[1], sharex=raster_ax)

    n_neurons = 7
    # Place-field peak positions, ordered so neuron 0 prefers x≈0.05 and
    # neuron 6 prefers x≈0.95. Local-wins bin (animal at 0.4) gets spikes
    # from neuron ~2-3; Non-Local-wins bin (animal at 0.55) is drawn with
    # spikes from neurons preferring x ≈ 0.4 (i.e., neurons 2-3 again
    # but at a different time, so the same cells fire "in the wrong place").
    pf_peaks = np.linspace(0.05, 0.95, n_neurons)

    bin_centers = np.array([0.17, 0.50, 0.83])
    bin_half_width = 0.04

    # Bin highlight bands (light grays) BEHIND the spikes
    bin_shades = ["#dddddd", "#cccccc", "#bbbbbb"]
    for center, shade in zip(bin_centers, bin_shades):
        raster_ax.axvspan(
            center - bin_half_width, center + bin_half_width,
            color=shade, alpha=1.0, zorder=0,
        )
        position_ax.axvspan(
            center - bin_half_width, center + bin_half_width,
            color=shade, alpha=1.0, zorder=0,
        )

    # Background spikes scattered across the strip (to give the raster
    # texture — these are spikes from time bins NOT highlighted).
    for i in range(n_neurons):
        n_bg = rng.integers(3, 8)
        bg_times = rng.uniform(0.0, 1.0, size=n_bg)
        # Filter out spikes that land in the highlighted bins; we want
        # to control bin contents explicitly below.
        bg_times = bg_times[
            np.all(
                np.abs(bg_times[:, None] - bin_centers[None, :]) > bin_half_width,
                axis=1,
            )
        ]
        raster_ax.scatter(
            bg_times, np.full_like(bg_times, i),
            marker="|", color="black", s=15, linewidths=0.6, zorder=2,
        )

    # bin₁ (Local wins, animal at 0.4): spikes from neurons whose PF
    # peaks are near 0.4 — that's neurons 2 and 3.
    for i in [2, 3]:
        raster_ax.scatter(
            [bin_centers[0]], [i],
            marker="|", color="black", s=15, linewidths=0.8, zorder=3,
        )

    # bin₂ (Non-Local wins, animal at 0.55): spikes from neurons whose PF
    # peaks are at 0.4 (so the cells fire "in the wrong place"). Same
    # neurons (2, 3) fire — but the animal is now at 0.55, far from
    # their place fields.
    for i in [2, 3]:
        raster_ax.scatter(
            [bin_centers[1]], [i],
            marker="|", color="black", s=15, linewidths=0.8, zorder=3,
        )

    # bin₃ (No-Spike wins): no spikes — leave the bin empty.

    raster_ax.set_xlim(0, 1)
    raster_ax.set_ylim(-0.5, n_neurons - 0.5)
    raster_ax.set_yticks([0, n_neurons - 1])
    raster_ax.set_yticklabels(["1", str(n_neurons)])
    raster_ax.set_ylabel("Neuron", fontsize=6, labelpad=2)
    raster_ax.tick_params(labelsize=5.5, labelbottom=False)
    for spine in ["top", "right"]:
        raster_ax.spines[spine].set_visible(False)
    raster_ax.spines["left"].set_linewidth(0.5)
    raster_ax.spines["bottom"].set_visible(False)

    # Bin labels along the top, above each highlight band
    for i, center in enumerate(bin_centers):
        raster_ax.text(
            center, n_neurons - 0.3, f"bin$_{i+1}$",
            ha="center", va="bottom", fontsize=6, fontweight="bold",
            transform=raster_ax.transData,
        )

    # Animal-position trace: a single line going from 0.4 → 0.55 → 0.4
    # (Local-wins position → Non-Local-wins position → back). The
    # No-Spike-wins bin can stay at 0.4 (animal doesn't have to move).
    t = np.linspace(0, 1, 200)
    # Anchor positions at the 3 bin centers; linear interpolation between.
    anim_xs = np.array([0.0, 0.17, 0.50, 0.83, 1.0])
    anim_ys = np.array([0.4, 0.4, 0.55, 0.4, 0.4])
    position = np.interp(t, anim_xs, anim_ys)

    position_ax.plot(t, position, color=ANIMAL_POS_COLOR, linewidth=0.8)
    position_ax.set_xlim(0, 1)
    position_ax.set_ylim(0.35, 0.6)
    position_ax.set_xticks([])
    position_ax.set_yticks([0.4, 0.55])
    position_ax.set_yticklabels(["0.4", "0.55"])
    position_ax.set_ylabel("x", fontsize=6, labelpad=2)
    position_ax.tick_params(labelsize=5)
    for spine in ["top", "right", "bottom"]:
        position_ax.spines[spine].set_visible(False)
    position_ax.spines["left"].set_linewidth(0.5)

    return [raster_ax, position_ax]
```

- [ ] **Step 2: Render and verify the helper is importable (don't wire it in yet)**

Run:

```bash
cd /cumulus/edeno/continuum-swr-replay
uv run python -c "from scripts.generate_figure01 import _draw_shared_spike_raster; print('ok')"
```

Expected output: `ok`. (Confirms no syntax errors.)

- [ ] **Step 3: Commit**

```bash
git add scripts/generate_figure01.py
git commit -m "Add _draw_shared_spike_raster helper for Figure 1 panel (b)

Defines but does not yet wire in the helper that will draw the shared
spike raster above the three panel-B scenarios. Lays out a 7-neuron
strip with three highlighted bins (Local-wins, Non-Local-wins,
No-Spike-wins) on x ∈ [0, 1], plus a thin position-trace axis below.
Helper is unused at this commit."
```

### Task 4: Wire the raster into `plot_observation_models`

**Files:**
- Modify: `scripts/generate_figure01.py` — the `plot_observation_models` function (created in Task 2) and the row-0 layout in `generate_figure`.

- [ ] **Step 1: Restructure `plot_observation_models` to use a 3-row sub-gridspec**

Replace the body of `plot_observation_models` from the `gs_outer` line through the end of the function with:

```python
    # Outer 3x3: raster strip on top, per-scenario title row, per-scenario
    # content row. Title-to-content ratio (0.08 : 0.62 ≈ 0.12 : 1.0) matches
    # the prior 2-row layout.
    gs_outer = gs_slot.subgridspec(
        3, 3, height_ratios=[0.30, 0.08, 0.62],
        wspace=0.20, hspace=0.10,
    )

    # Raster spans all 3 columns in row 0
    raster_slot = gs_outer[0, :]
    raster_axes = _draw_shared_spike_raster(fig, raster_slot)

    scenarios = [
        # (header, animal_pos, no_spike_wins)
        ("Local wins", likelihood_center, False),
        ("Non-Local wins", 0.55, False),
        ("No-Spike wins", likelihood_center, True),
    ]

    all_axes = []
    for col, (header, animal_pos, no_spike_wins) in enumerate(scenarios):
        title_ax = fig.add_subplot(gs_outer[1, col])
        title_ax.axis("off")
        title_ax.text(
            0.5, 0.0, header,
            ha="center", va="bottom", fontsize=7, fontweight="bold",
            transform=title_ax.transAxes,
        )

        gs_scen = gs_outer[2, col].subgridspec(2, 2, wspace=0.10, hspace=0.55)
        all_axes.extend(
            _draw_state_cells(
                fig, gs_scen, x, likelihood, animal_pos,
                no_spike_wins=no_spike_wins,
            )
        )

    return all_axes
```

- [ ] **Step 2: Tune main figure row heights to accommodate the raster**

In `generate_figure`, change:

```python
gs_main = GridSpec(
    3, 1, figure=fig,
    height_ratios=[1.4, 0.55, 1.7],
    hspace=0.20,
)
```

to:

```python
gs_main = GridSpec(
    3, 1, figure=fig,
    height_ratios=[1.7, 0.55, 1.55],
    hspace=0.20,
)
```

- [ ] **Step 3: Render and inspect**

Run:

```bash
cd /cumulus/edeno/continuum-swr-replay
uv run python scripts/generate_figure01.py
```

Open `paper/figures/figure01/figure01.png` and verify:

- Panel (b) now has a spike raster strip across the top of the row.
- Three light-gray vertical bands in the raster sit directly above each of the three scenario blocks below (Local-wins → Non-Local-wins → No-Spike-wins, left to right).
- The animal-position trace sits in a thin strip below the raster, magenta line, moving from 0.4 → 0.55 → 0.4 across the three bins.
- The No-Spike-wins bin (third highlight) contains no spike ticks.
- Panel (a) still aligns visually with the bottom of the leftmost scenario block (set_anchor="S" handles this automatically).
- The figure still fits within `PAGE_HEIGHT` (no clipping).

If the bin centers don't align with the scenario columns visually, tune `bin_centers` in `_draw_shared_spike_raster` (try 0.18 / 0.50 / 0.82 if the inner `wspace=0.20` of `gs_outer` shifts them slightly).

- [ ] **Step 4: Commit**

```bash
git add scripts/generate_figure01.py
git commit -m "Wire shared spike raster into Figure 1 panel (b)

Restructures plot_observation_models to a 3-row sub-gridspec
(raster / per-scenario titles / per-scenario content rows). The raster
spans all 3 columns and visually shows which spikes feed each scenario:
bin1 has spikes from cells with PFs near the animal (Local wins),
bin2 has spikes from cells with PFs far from the animal (Non-Local
wins), bin3 is empty (No-Spike wins). Row-0 height bumped 1.4→1.7;
panel (d) row trimmed 1.7→1.55 to compensate."
```

---

## Phase 3 — Panel (a) Redraw

### Task 5: Redraw `plot_state_diagram` end-to-end

**Files:**
- Modify: `scripts/generate_figure01.py:75-173` (the `plot_state_diagram` function — three coordinated changes: source self-probs from detector, redraw self-loops, replace cross-arrows).

This task makes three coordinated edits to `plot_state_diagram`. Because the in-between states (e.g., self-loops updated but cross-arrows still using the old irregular subset) are not visually meaningful, all three edits land in one commit at the end of the task.

- [ ] **Step 1: Sanity-check that the detector's diagonal matches expectations**

Run:

```bash
cd /cumulus/edeno/continuum-swr-replay
uv run python -c "
from non_local_detector import NonLocalSortedSpikesDetector
d = NonLocalSortedSpikesDetector().discrete_transition_type.values.diagonal()
print(d)
"
```

Expected output: `[0.99 0.98 0.96 0.96]`. If this is not what's printed, stop and investigate — the spec assumes these values, and silently mismatching values would change the figure's labels.

- [ ] **Step 2: Replace the hardcoded `self_probs` dict with a detector-sourced version**

In `plot_state_diagram`, delete the current hardcoded dict (lines 102-107):

```python
    self_probs = {
        "Local": "0.99",
        "No-Spike": "0.98",
        "Non-Local\nContinuous": "0.98",
        "Non-Local\nFragmented": "0.98",
    }
```

Replace it with:

```python
    # Pull the actual self-probabilities from the detector so the figure
    # cannot go silently stale if upstream defaults change.
    _diag = NonLocalSortedSpikesDetector().discrete_transition_type.values.diagonal()
    self_probs = {
        "Local": f"{_diag[0]:.2f}",
        "No-Spike": f"{_diag[1]:.2f}",
        "Non-Local\nContinuous": f"{_diag[2]:.2f}",
        "Non-Local\nFragmented": f"{_diag[3]:.2f}",
    }
```

- [ ] **Step 3: Replace the self-loop drawing block**

Replace the block currently at `scripts/generate_figure01.py:128-148` (from `# Self-loop arrows` through the close of the `for` loop drawing the self-loops) with:

```python
    # Self-loop arrows — drawn as prominent arcs in the outward corner of
    # each state (NW for Local, NE for No-Spike, SW for Continuous, SE for
    # Fragmented). Each loop goes from one tangent point on the circle to
    # another, bowed outward.
    loop_geometry = {
        # state -> (start_offset, end_offset, label_offset) in (dx, dy)
        # units. Offsets are relative to the circle center; magnitudes
        # chosen so loops sit clearly outside the circle.
        "Local": (
            (0.0, radius),        # start: top of circle
            (-radius, 0.0),       # end: left of circle
            (-radius * 1.6, radius * 1.4),  # label: NW outward
        ),
        "No-Spike": (
            (radius, 0.0),        # start: right of circle
            (0.0, radius),        # end: top of circle
            (radius * 1.6, radius * 1.4),   # label: NE outward
        ),
        "Non-Local\nContinuous": (
            (-radius, 0.0),       # start: left of circle
            (0.0, -radius),       # end: bottom of circle
            (-radius * 1.6, -radius * 1.4), # label: SW outward
        ),
        "Non-Local\nFragmented": (
            (0.0, -radius),       # start: bottom of circle
            (radius, 0.0),        # end: right of circle
            (radius * 1.6, -radius * 1.4),  # label: SE outward
        ),
    }
    for state_name, (cx, cy) in positions.items():
        (sx, sy), (ex, ey), (lx, ly) = loop_geometry[state_name]
        arrow = mpatches.FancyArrowPatch(
            (cx + sx, cy + sy),
            (cx + ex, cy + ey),
            arrowstyle="->,head_length=4,head_width=3",
            connectionstyle="arc3,rad=1.2",
            mutation_scale=1,
            linewidth=1.0, color="black", alpha=0.7, zorder=6,
        )
        ax.add_patch(arrow)
        ax.text(
            cx + lx, cy + ly, self_probs[state_name],
            ha="center", va="center", fontsize=6, color="black", alpha=0.8,
        )
```

- [ ] **Step 4: Replace the cross-state arrow block**

Replace the block currently at `scripts/generate_figure01.py:150-172` (from `# Cross-state arrows` through the close of the `for` loop drawing cross-state arrows) with:

```python
    # Cross-state arrows: one double-headed line per state pair (6 pairs
    # total), showing topology only — no thickness encoding, no labels.
    # Numbers live in the transition kernels (panel c) and panel (d)'s
    # decoded trajectories.
    state_pairs = [
        ("Local", "No-Spike"),
        ("Local", "Non-Local\nContinuous"),
        ("Local", "Non-Local\nFragmented"),
        ("No-Spike", "Non-Local\nContinuous"),
        ("No-Spike", "Non-Local\nFragmented"),
        ("Non-Local\nContinuous", "Non-Local\nFragmented"),
    ]
    for from_state, to_state in state_pairs:
        x1, y1 = positions[from_state]
        x2, y2 = positions[to_state]
        dx, dy = x2 - x1, y2 - y1
        dist = np.sqrt(dx**2 + dy**2)
        shrink = radius * 1.15
        arrow = mpatches.FancyArrowPatch(
            (x1 + dx / dist * shrink, y1 + dy / dist * shrink),
            (x2 - dx / dist * shrink, y2 - dy / dist * shrink),
            arrowstyle="<->,head_length=3,head_width=2",
            mutation_scale=1,
            linewidth=0.6, color="black", alpha=0.35, zorder=4,
        )
        ax.add_patch(arrow)
```

- [ ] **Step 5: Render and inspect the full new panel (a)**

Run:

```bash
cd /cumulus/edeno/continuum-swr-replay
uv run python scripts/generate_figure01.py
```

Open `paper/figures/figure01/figure01.png` and verify:

- Each of the four state circles has a clearly visible self-loop arc in its outward corner.
- Self-loops are labeled with the correct probabilities: Local=0.99, No-Spike=0.98, Continuous=0.96, Fragmented=0.96.
- Each pair of state circles is connected by exactly one line, with arrowheads at both ends. Total cross-arrows visible: 6 (4 between adjacent rows/cols + 2 diagonals).
- Self-loops do not visually collide with cross-arrows or with state labels.

If `rad=1.2` makes the self-loop arc too pinched or too bulgy, tune it (try 0.8–1.5). If labels sit too close to or too far from the loops, scale `lx`/`ly` multipliers (currently `1.6 * radius` and `1.4 * radius`).

- [ ] **Step 6: Commit**

```bash
git add scripts/generate_figure01.py
git commit -m "Redraw Figure 1 panel (a): prominent self-loops + bidirectional cross-arrows

Three coordinated changes to plot_state_diagram:
- Self-probabilities are pulled from
  NonLocalSortedSpikesDetector().discrete_transition_type.values.diagonal()
  at runtime. This corrects a bug where Continuous and Fragmented were
  labeled 0.98 instead of the actual 0.96.
- Self-loops are redrawn as prominent arcs routed to each state's
  outward corner (NW for Local, NE for No-Spike, SW for Continuous,
  SE for Fragmented). The connectionstyle 'arc3,rad=1.2' gives a
  clearly visible loop.
- Cross-state arrows are redrawn as 6 double-headed FancyArrowPatch
  lines, one per state pair. No thickness encoding, no labels —
  topology only. The previous 8 directional arrows had an irregular
  subset that misleadingly implied some transitions weren't possible."
```

---

## Final Verification

### Task 6: Compare against the prior figure and refresh PDF output

- [ ] **Step 1: Confirm the PDF and PNG both render**

Run:

```bash
cd /cumulus/edeno/continuum-swr-replay
uv run python scripts/generate_figure01.py
ls -la paper/figures/figure01/figure01.{png,pdf}
```

Expected: both files exist and have recent timestamps. PNG is ~few-hundred KB; PDF is ~1-2 MB.

- [ ] **Step 2: Visual sanity-check checklist**

Open `paper/figures/figure01/figure01.png`. Verify all of:

1. **Panel (a):** 4 state circles arranged 2×2. Each has a prominent outward-corner self-loop labeled with the correct probability (0.99 / 0.98 / 0.96 / 0.96). 6 double-headed lines connect each pair of circles. No labels on cross-arrows. No visual collisions.

2. **Panel (b) raster strip:** 7 neurons in a horizontal raster. Three light-gray vertical bands aligned with the scenario columns below. Bin labels (bin₁ / bin₂ / bin₃) along the top. Magenta animal-position line in a thin strip below the raster, moving from 0.4 to 0.55 and back to 0.4 across the three bins.

3. **Panel (b) scenarios:** Three columns, left to right: Local wins, Non-Local wins, No-Spike wins. Each column has a title row and a 2×2 of state-likelihood cells underneath.

4. **Panel (b) No-Spike-wins cell:** the gray horizontal No-Spike likelihood line is at y=0.7, clearly above the Local/Continuous/Fragmented bumps at y=0.4.

5. **Panel (c):** Four transition-kernel heatmaps, unchanged from before this revision.

6. **Panel (d):** Spike raster, posterior, state probabilities, speed — unchanged content, slightly less vertical space than before.

7. **Panel labels:** a / b / c / d appear in the correct positions (top-left of each panel).

- [ ] **Step 3: Final commit if any tuning was needed; otherwise skip**

If any tuning changes were made during the visual check (e.g., shifting bin centers, adjusting `rad` on self-loops, scaling label positions), commit them with:

```bash
git add scripts/generate_figure01.py paper/figures/figure01/
git commit -m "Tune Figure 1 visual details after final inspection

- [list any specific tweaks made]"
```

If nothing needed tuning, no commit is required.

---

## Self-Review Notes

**Spec coverage:**

- Panel (a) self-loops, detector-sourced labels, double-headed cross-arrows, self-prob bug fix (0.98 → 0.96) → Task 5.
- Panel (b) 3rd "No-Spike wins" scenario (the `_draw_state_cells` branch + the new scenario tuple) → Tasks 1, 2.
- Panel (b) shared spike raster (helper + wiring) → Tasks 3, 4.
- Panel (b) width adjustment (1:2.2 → 1:3.3) → Task 2.
- Panel (b) 3-row sub-gridspec `[0.30, 0.08, 0.62]` → Task 4.
- Row 0 / row 2 height tuning to fit the new content → Task 4.
- Animal-position trace on a separate axis → Task 3.
- Bin centers at 0.17 / 0.50 / 0.83 → Task 3.
- Highlight band shades (`#dddddd` / `#cccccc` / `#bbbbbb`) → Task 3.
- No-Spike line at y=0.7, others at y=0.4 → Task 1.
- Final visual checklist → Task 6.

**Type / naming consistency:**

- `_draw_state_cells` adds a `no_spike_wins` kwarg in Task 1; `plot_observation_models` passes it positionally as a 3-tuple entry (`(header, animal_pos, no_spike_wins)`) in Tasks 2 and 4 — names match.
- `_draw_shared_spike_raster` signature is `(fig, gs_slot, seed=42)` in Task 3; called as `_draw_shared_spike_raster(fig, raster_slot)` in Task 4 — matches.
- `STATE_COLORS_SHORT`, `ANIMAL_POS_COLOR`, `WONG` constants are referenced as already defined module-level constants (lines 60-68 of the existing file).

**No placeholders:** All code blocks are complete; no "TBD" or "similar to above" steps.

**Verification:** Each phase ends with `uv run python scripts/generate_figure01.py` + visual inspection. No unit tests added — matches existing convention for figure scripts in this repo (no figure script under `scripts/` has its own unit test).
