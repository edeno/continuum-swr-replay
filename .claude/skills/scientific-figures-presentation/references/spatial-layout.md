# Spatial Layout and Overlap Prevention Guide

**Purpose**: Prevent overlap and minimize white space in matplotlib figures through explicit spatial reasoning before coding.

**Problem**: Cannot see figures during creation, so must predict spatial layout without visual feedback.

**Solution**: Systematic spatial planning + constrained_layout + subplot_mosaic for optimal space usage.

---

## CRITICAL: Visual Layout Reasoning Before Coding

**Before writing any plotting code, explicitly reason through spatial layout.**

### Pre-Coding Spatial Checklist (MANDATORY)

Answer these questions BEFORE writing code:

1. **Layout structure**: Regular grid or complex? → Use subplot_mosaic for complex layouts
2. **Panel sizes**: All equal or varying? → Set width_ratios and height_ratios
3. **Legend placement**: Inside or outside? Where? → Reserve space in layout
4. **Axis label lengths**: Long or short? → Calculate space needed
5. **Title requirements**: Per-panel or figure-level? → Plan vertical space
6. **Tick label complexity**: Dates, long numbers, categories? → Plan for rotation/formatting
7. **White space goal**: Minimal spacing or breathing room? → Adjust constrained_layout padding

---

## Minimal White Space Strategy

### Philosophy

Every pixel should contribute to understanding. Remove all non-essential space while maintaining clarity.

### Implementation

```python
# 1. Use constrained_layout with tight padding
fig, axes = plt.subplot_mosaic(
    layout,
    figsize=(7, 4),
    dpi=450,
    constrained_layout=True,
    # Reduce default padding for tighter layout
    layout_pad=0.02,      # Padding around entire figure (default: 0.05)
)

# 2. Minimize subplot spacing
fig, axes = plt.subplots(2, 2, figsize=(7, 5),
                         constrained_layout={'h_pad': 0.02, 'w_pad': 0.02})

# 3. Remove unnecessary spines
for ax in axes.flat:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# 4. Tight bounding box on save
plt.savefig('figure.pdf', bbox_inches='tight', pad_inches=0.02)
```

---

## subplot_mosaic for Spatial Efficiency

### Why subplot_mosaic Minimizes White Space

1. **Precise panel sizing**: Control exact proportions with width_ratios/height_ratios
2. **No wasted grid cells**: Panels span only needed space
3. **Integrated legend space**: Reserve exact width needed, no more
4. **Flexible layouts**: Adapt to content needs, not grid constraints

### Basic mosaic Usage

```python
layout = """
    AB
    CD
    """

fig, axes = plt.subplot_mosaic(
    layout,
    figsize=(7, 4),
    dpi=450,
    constrained_layout=True
)

# Access by name
axes['A'].plot(x, y)
```

### Advanced: Asymmetric Layouts for Minimal Waste

```python
# Different-sized panels - no wasted space
layout = """
    AAAB
    CCDB
    """

fig, axes = plt.subplot_mosaic(
    layout,
    figsize=(7, 4),
    width_ratios=[2, 2, 2, 1],  # B column is narrow
    height_ratios=[1, 1],
    constrained_layout=True
)
```

### Pattern: Dedicated Legend Panel

```python
# Reserve minimal space for legend - no data area wasted
layout = """
    AABBL
    CCDEL
    """

fig, axes = plt.subplot_mosaic(
    layout,
    figsize=(7, 4),
    width_ratios=[1, 1, 1, 0.15, 0.4],  # Small gap, narrow legend
    constrained_layout=True
)

# Plot in main panels
for label in ['A', 'B', 'C', 'D', 'E']:
    axes[label].plot(x, y, label=f'Series {label}')

# Use L for shared legend
axes['L'].axis('off')
handles, labels = axes['A'].get_legend_handles_labels()
# Collect from all subplots if needed
axes['L'].legend(handles, labels, loc='center left', 
                fontsize=5, frameon=False)
```

### Pattern: Wide Bottom Panel

```python
# Top panels small, bottom spans full width
layout = """
    ABC
    DDD
    """

fig, axes = plt.subplot_mosaic(layout, figsize=(7, 4.5),
                                constrained_layout=True)

# A, B, C are 1x1, D is 3x1 (spanning width)
```

### Pattern: Main Plot with Side Panels

```python
# Large main plot, smaller supplementary panels
layout = """
    AAAB
    AAAC
    AAAD
    """

fig, axes = plt.subplot_mosaic(
    layout,
    figsize=(7, 5),
    width_ratios=[2, 2, 2, 1],  # B, C, D are narrow
    constrained_layout=True
)

# A is main, B/C/D show details or insets
```

### Pattern: Empty Spaces for Asymmetry

```python
# Use '.' for intentional empty space (not white space waste!)
layout = """
    AA.B
    CCDB
    """

fig, axes = plt.subplot_mosaic(layout, figsize=(7, 4),
                                constrained_layout=True)

# Top-right '.' leaves space for figure legend or caption
```

---

## Spatial Planning Workflow

### Step 1: Design Layout String

```python
# BEFORE any plotting, design the layout:

# Example: 4 panels with shared legend
layout = """
    AABBL
    CCDEL
    """

# Questions to answer:
# - How many unique panels? (5: A, B, C, D, E, L)
# - Which span multiple cells? (A=2 cells, B=2, C=2, D=2, E=2)
# - Legend location? (L column on right)
# - Any empty space needed? (small gap between D and L)
```

### Step 2: Calculate Figure Size

```python
# Start with journal width constraint
max_width = 7.0  # inches (full page width)

# Calculate height based on content and layout
# Rule of thumb: Each panel needs ~2-3 inches minimum
n_rows = 2
panel_height = 2.0  # inches per row
title_space = 0.3 if has_titles else 0
fig_height = n_rows * panel_height + title_space

figsize = (max_width, fig_height)
```

### Step 3: Set Width/Height Ratios

```python
# Make precise adjustments for content
fig, axes = plt.subplot_mosaic(
    layout,
    figsize=(7, 4),
    # Width ratios: match layout columns
    # For "AABBL": 5 columns, L should be narrow
    width_ratios=[1, 1, 1, 0.15, 0.4],  # [A, A, B, gap, L]
    
    # Height ratios: match layout rows
    height_ratios=[1, 1],  # Equal height rows
    
    constrained_layout=True
)
```

### Step 4: Configure constrained_layout Padding

```python
# For minimal white space
fig, axes = plt.subplot_mosaic(
    layout,
    figsize=(7, 4),
    constrained_layout={
        'h_pad': 0.02,     # Horizontal padding between subplots
        'w_pad': 0.02,     # Vertical padding between subplots  
        'hspace': 0.02,    # Height space between subplot groups
        'wspace': 0.02,    # Width space between subplot groups
    }
)

# For more breathing room (presentations)
fig, axes = plt.subplot_mosaic(
    layout,
    figsize=(10, 7),
    constrained_layout={
        'h_pad': 0.08,
        'w_pad': 0.08,
    }
)
```

---

## Legend Placement Strategies

### Strategy 1: Dedicated Legend Panel (BEST for minimal waste)

```python
layout = """
    AABBL
    CCDEL
    """

fig, axes = plt.subplot_mosaic(
    layout,
    figsize=(7, 4),
    width_ratios=[1, 1, 1, 0.15, 0.4],
    constrained_layout=True
)

# Hide legend panel axes
axes['L'].axis('off')

# Collect all legend items
all_handles = []
all_labels = []
for ax_name in ['A', 'B', 'C', 'D', 'E']:
    h, l = axes[ax_name].get_legend_handles_labels()
    all_handles.extend(h)
    all_labels.extend(l)

# Place in legend panel
axes['L'].legend(all_handles, all_labels, 
                loc='center left',
                fontsize=5,
                frameon=False)
```

### Strategy 2: Outside Right (Standard)

```python
# When not using mosaic, place outside
ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1),
          fontsize=5, frameon=False)

# Account for legend width in figure size
fig_width = panel_width + legend_width + margin
```

### Strategy 3: Below Subplot (Horizontal)

```python
# For many items, horizontal layout below
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          ncol=4, fontsize=5, frameon=False,
          columnspacing=1.0)  # Tight column spacing

# Account for legend height
fig_height = panel_height + legend_height + margin
```

### Strategy 4: Inside Only If Necessary

```python
# ONLY if truly no other option and verified no overlap
ax.legend(loc='upper right',
          fontsize=5,
          frameon=True,
          framealpha=0.95,  # Semi-transparent
          edgecolor='none')
```

---

## Handling Long Tick Labels

### X-Axis Labels (Dates, Categories)

```python
# Option 1: Rotate (most common)
ax.tick_params(axis='x', rotation=45, labelsize=5)
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

# Option 2: Reduce number of ticks
ax.xaxis.set_major_locator(plt.MaxNLocator(5))

# Option 3: Custom formatting
def format_func(x, pos):
    return f'{x:.1f}' if x < 1000 else f'{x/1000:.1f}k'
ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))

# Option 4: Abbreviate labels
labels = ['Jan', 'Feb', 'Mar']  # Instead of full month names
ax.set_xticklabels(labels, fontsize=5)
```

### Y-Axis Labels (Large Numbers)

```python
# Option 1: Scientific notation
ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

# Option 2: Custom units
def format_func(x, pos):
    if abs(x) >= 1e6:
        return f'{x/1e6:.1f}M'
    elif abs(x) >= 1e3:
        return f'{x/1e3:.1f}K'
    return f'{x:.0f}'
ax.yaxis.set_major_formatter(plt.FuncFormatter(format_func))

# Option 3: Reduce precision
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0f}'))
```

---

## Padding and Spacing Control

### Label Padding (Minimum for Readability)

```python
# Default labelpad=4 is often too small
# Use 8-10 for papers, 12-15 for presentations

# Papers
ax.set_xlabel('Time (s)', fontsize=7, labelpad=8)
ax.set_ylabel('Response (mV)', fontsize=7, labelpad=10)
ax.set_title('Results', fontsize=7, pad=12)

# Presentations
ax.set_xlabel('Time (s)', fontsize=24, labelpad=15)
ax.set_ylabel('Response (mV)', fontsize=24, labelpad=15)
ax.set_title('Results', fontsize=28, pad=20)
```

### Spine Removal for Cleaner Look

```python
# Remove top and right spines (Tufte principle)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Or remove all spines for minimal look
for spine in ax.spines.values():
    spine.set_visible(False)

# Keep only left and bottom
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(top=False, right=False)
```

---

## Common Overlap Scenarios and Solutions

### Scenario 1: Subplot Titles Overlap

**Problem**: Titles from top row overlap with bottom row

**Solution**:
```python
# Increase vertical spacing
fig, axes = plt.subplots(2, 2, figsize=(7, 5),
                         constrained_layout={'h_pad': 0.1, 'w_pad': 0.05})

# Or increase title pad
for ax in axes.flat:
    ax.set_title('Title', fontsize=7, pad=15)  # Increased pad
```

### Scenario 2: Y-Labels Overlap Between Columns

**Problem**: Left column y-labels overlap right column

**Solution**:
```python
# Increase horizontal spacing
fig, axes = plt.subplots(2, 2, figsize=(7, 5),
                         constrained_layout={'w_pad': 0.08})

# Or use subplot_mosaic with explicit gap
layout = """
    A.B
    C.D
    """
# The '.' creates a gap
```

### Scenario 3: Panel Labels Overlap Titles

**Problem**: a, b, c labels overlap subplot titles

**Solution**:
```python
# Adjust panel label position
label_panels(axes.flatten(),
             xpos=-0.15,  # Further left
             ypos=1.08,   # Further up
             fontsize=8)

# Or remove subplot titles if panel labels sufficient
```

### Scenario 4: Colorbar Overlaps Plot

**Problem**: Colorbar placed by default overlaps

**Solution with subplot_mosaic**:
```python
layout = """
    AAC
    BBC
    """

fig, axes = plt.subplot_mosaic(
    layout,
    figsize=(7, 4),
    width_ratios=[2, 2, 0.2],  # C is narrow colorbar column
    constrained_layout=True
)

# Heatmaps in A and B
im1 = axes['A'].imshow(data1, cmap='viridis')
im2 = axes['B'].imshow(data2, cmap='viridis')

# Colorbar in C
axes['C'].axis('off')
plt.colorbar(im1, ax=axes['C'], fraction=0.8)
```

---

## Multi-Panel Spatial Planning Template

```python
"""
==========================================
MULTI-PANEL SPATIAL PLAN
==========================================
Figure: 7.0" × 4.5" (full page width)
Layout: 2 rows × 3 columns + legend

Layout string:
    AABBL
    CCDEL

Panel contents:
- A (spans 2 cols): Main time series
- B (1 col): Scatter plot
- C (spans 2 cols): Bar chart
- D (1 col): Distribution
- E (1 col): Box plot
- L (narrow): Shared legend

Width ratios: [1, 1, 1, 0.15, 0.4]
- First 3 columns equal (data panels)
- Column 4 is gap (0.15)
- Column 5 is legend (0.4)

Height ratios: [1, 1] (equal row heights)

Spacing strategy:
- Minimal padding (h_pad=0.02, w_pad=0.02)
- No top/right spines
- Legend outside data area in dedicated panel
- All labels ≥7pt with labelpad=8

Expected issues: None with this layout
==========================================
"""

from scripts.figure_utilities import set_figure_defaults, save_figure
from scripts.color_palettes import WONG
import matplotlib.pyplot as plt

set_figure_defaults(context="paper")

layout = """
    AABBL
    CCDEL
    """

fig, axes = plt.subplot_mosaic(
    layout,
    figsize=(7, 4.5),
    width_ratios=[1, 1, 1, 0.15, 0.4],
    height_ratios=[1, 1],
    dpi=450,
    constrained_layout={'h_pad': 0.02, 'w_pad': 0.02}
)

# [Plotting code here]

# Shared legend in L panel
axes['L'].axis('off')
# [Legend code here]

save_figure('multi_panel')
plt.close()
```

---

## Validation Checklist

Before saving any figure:

1. [ ] Used constrained_layout=True
2. [ ] Legends don't overlap data (outside or dedicated panel)
3. [ ] All labels have adequate padding (≥8pt)
4. [ ] Tick labels don't overlap each other
5. [ ] No matplotlib warnings
6. [ ] Minimal white space achieved
7. [ ] All text ≥7pt font size
8. [ ] Panel labels positioned correctly

### Validation Code

```python
import warnings

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    plt.tight_layout()
    if w:
        print("⚠️  Layout warning detected!")
        print("Message:", w[-1].message)
        print("Fix required before saving")
        plt.close()
        # Increase spacing, adjust ratios, or resize figure
    else:
        print("✓ No layout issues detected")
        save_figure('figure_name')
        plt.close()
```

---

## Key Principles Summary

1. **Plan before code**: Design layout string and calculate space requirements first
2. **Use subplot_mosaic**: Superior spatial control and minimal white space
3. **constrained_layout always**: Most reliable spacing method
4. **Legends outside data**: Use dedicated panel in mosaic or bbox_to_anchor
5. **Minimal padding**: Start tight, add only where needed
6. **Validate warnings**: Fix all issues before saving
7. **Width/height ratios**: Precise control over panel proportions
8. **Remove non-data ink**: Hide unnecessary spines and elements

---

## Quick Reference: subplot_mosaic Patterns

```python
# Equal grid
"""
AB
CD
"""

# Wide bottom
"""
ABC
DDD
"""

# Main + side panels
"""
AAAB
AAAC
"""

# With legend space
"""
AABBL
CCDEL
"""

# Asymmetric with gaps
"""
AA.B
CCDB
"""

# Complex with empty corners
"""
.AB.
CCDD
EEFF
.GH.
"""
```
