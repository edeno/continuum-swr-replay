---
name: scientific-figures-paper
description: Creates publication-ready figures for scientific papers using matplotlib, following Tufte, Gelman, and Heer principles. Use when generating figures for manuscripts, preprints, peer review, or any journal submission (Nature, Science, Cell, etc.).
license: MIT
---

# Scientific Figure Creation for Papers

## Overview

Create publication-ready scientific figures that meet journal requirements and follow visualization best practices from Tufte (data-ink ratio), Gelman (statistical clarity), and Heer (perceptual effectiveness).

**Keywords**: matplotlib, scientific visualization, publication figures, Nature, Science, journal figures, data visualization, academic figures, research plots, manuscript figures, peer review, paper figures

## When to Read Reference Files

This skill uses progressive disclosure. Read reference files when you need detailed guidance:

- **Always first**: `cat references/spatial_layout.md` - Spatial planning and overlap prevention (read before ANY plotting)
- **Design guidance**: `cat references/design_principles.md` - Tufte, Gelman, Heer principles explained
- **Examples**: `cat references/common_patterns.md` - Complete examples for common figure types

## Core Principles

### Minimal White Space Philosophy

Publication-quality figures maximize information density while maintaining clarity:

- **No wasted space**: Every pixel should contribute to understanding
- **Tight layouts**: Use `constrained_layout=True` or `tight_layout()`
- **Strategic spacing**: Space only where needed for clarity
- **Legends outside or compact**: Never waste data area on legends
- **Use subplot_mosaic**: Organize complex layouts efficiently with minimal gaps

### Mandatory Typography Standards

- **Minimum font size**: 7pt (Nature/Science requirement)
- **Axis labels**: 7pt
- **Tick labels**: 5-6pt
- **Panel letters**: 8pt bold
- **Font family**: Arial or sans-serif
- **Editable fonts**: `pdf.fonttype: 42` and `ps.fonttype: 42`

### Figure Dimensions

- **Full width**: 7 inches (178mm) - two-column
- **Single column**: 3.5 inches (89mm)
- **1.5 column**: 5.5 inches (140mm)
- **DPI**: 450 for final production
- **Aspect ratios**: Follow the data structure

### Color Requirements

- **Primary palettes**: Wong or Okabe-Ito (colorblind-friendly)
- **Sequential data**: viridis, plasma, cividis
- **Avoid**: Red-green combinations
- **Grayscale compatible**: Must work in B&W

## Setup (REQUIRED FIRST STEP)

**Before creating any figures, add the skill's scripts directory to your Python path:**

```python
import sys
import os

# Add scripts directory to path (works for both local and cloud)
# Try local first (~/.claude/skills), then cloud (/mnt/skills)
skill_paths = [
    os.path.expanduser("~/.claude/skills/scientific-figures-paper/scripts"),
    "/mnt/skills/user/scientific-figures-paper/scripts",
    "/mnt/skills/scientific-figures-paper/scripts",
]

for path in skill_paths:
    if os.path.exists(path) and path not in sys.path:
        sys.path.insert(0, path)
        print(f"✓ Added to path: {path}")
        break
else:
    print("⚠ Warning: Could not find skill scripts directory")
```

Now you can import the utilities normally. All examples below assume you've completed this setup.

**Alternative: Copy to working directory**

```bash
# For local use
cp ~/.claude/skills/scientific-figures-paper/scripts/*.py .

# For cloud use
SKILL_DIR=$(find /mnt/skills -name "scientific-figures-paper" -type d 2>/dev/null | head -1)
cp $SKILL_DIR/scripts/*.py .
```

## Standard Workflow

### Single Panel Figure

```python
# Import utilities (after copying scripts - see Setup section)
from figure_utilities import set_figure_defaults, save_figure
from color_palettes import WONG
import matplotlib.pyplot as plt

# Step 1: Set paper defaults
set_figure_defaults(context="paper")

# Step 2: Create figure with tight layout
fig, ax = plt.subplots(figsize=(3.5, 2.5), dpi=450,
                       constrained_layout=True)

# Step 3: Plot data
ax.plot(x, y, color=WONG[1], linewidth=1.5, label='Data')

# Step 4: Configure axes
ax.set_xlabel('Time (s)', fontsize=7, labelpad=10)
ax.set_ylabel('Signal (mV)', fontsize=7, labelpad=10)

# Step 5: Legend outside to save space
ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1),
          fontsize=5, frameon=False)

# Step 6: Save
save_figure("figure_1")
plt.close()
```

### Multi-Panel with subplot_mosaic (RECOMMENDED)

**subplot_mosaic provides superior spatial control and minimal white space:**

```python
from figure_utilities import set_figure_defaults, save_figure
from label_panels import label_panels
from color_palettes import WONG
import matplotlib.pyplot as plt

set_figure_defaults(context="paper")

# Define layout with subplot_mosaic for precise control
# Use letters to name subplots, '.' for empty space
layout = """
    ABC
    DDD
    """

# Calculate figure size for tight layout
fig_width = 7.0  # Full page width
fig_height = 4.5  # Calculated for 2 rows

fig, axes = plt.subplot_mosaic(layout, figsize=(fig_width, fig_height),
                                dpi=450, constrained_layout=True)

# Access axes by name (not index!)
axes['A'].plot(x, y1, color=WONG[1], linewidth=1.5)
axes['A'].set_xlabel('Time (s)', fontsize=7, labelpad=8)
axes['A'].set_ylabel('Signal', fontsize=7, labelpad=8)

axes['B'].scatter(x, y2, s=20, color=WONG[2], alpha=0.7)
axes['B'].set_xlabel('X', fontsize=7, labelpad=8)
axes['B'].set_ylabel('Y', fontsize=7, labelpad=8)

axes['C'].bar(categories, values, color=WONG[3], alpha=0.8)
axes['C'].set_xlabel('Category', fontsize=7, labelpad=8)
axes['C'].set_ylabel('Count', fontsize=7, labelpad=8)

# Panel D spans full width
axes['D'].plot(time_series, color=WONG[4], linewidth=1.5)
axes['D'].set_xlabel('Time', fontsize=7, labelpad=8)
axes['D'].set_ylabel('Response', fontsize=7, labelpad=8)

# Label panels (converts mosaic dict to list automatically)
label_panels(list(axes.values()),
             labels=['a', 'b', 'c', 'd'],
             xpos=-0.12, ypos=1.05, fontsize=8)

save_figure("multi_panel")
plt.close()
```

### Why subplot_mosaic is Superior

1. **Named axes**: `axes['A']` is clearer than `axes[0, 0]`
2. **Flexible layouts**: Panels can span multiple cells
3. **Minimal white space**: Precise control over spacing
4. **Legend integration**: Reserve space for legends efficiently
5. **Intuitive design**: Visual layout string matches output

See `references/common_patterns.md` for complete examples.

## Spatial Layout Checklist (CRITICAL)

**Before writing ANY plotting code, read the complete spatial guide:**

```bash
cat references/spatial_layout.md
```

**Quick checklist (detailed in reference):**

1. ✓ Calculate required space for all elements
2. ✓ Use `constrained_layout=True` (mandatory)
3. ✓ Place legends outside plot area (95% of cases)
4. ✓ Use `labelpad=10` minimum (not default 4)
5. ✓ Handle long tick labels (rotate, reduce, format)
6. ✓ Validate with matplotlib warnings before saving

## Critical Rules

### ALWAYS Use constrained_layout

```python
# ✅ Correct - automatic optimal spacing
fig, ax = plt.subplots(figsize=(3.5, 2.5), dpi=450,
                       constrained_layout=True)

# ✅ Or for subplot_mosaic
fig, axes = plt.subplot_mosaic(layout, figsize=(7, 4), dpi=450,
                                constrained_layout=True)
```

### NEVER Scale figsize to Make Text Bigger

```python
# ❌ Wrong - makes figure physically too large
fig = plt.figure(figsize=(14, 14))

# ✅ Correct - increases resolution, maintains physical size
fig = plt.figure(figsize=(7, 7), dpi=450)
```

### ALWAYS Use Object-Oriented Interface

```python
# ✅ Preferred - explicit and maintainable
fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_xlabel('X')

# ❌ Avoid - stateful and error-prone
plt.plot(x, y)
plt.xlabel('X')
```

### Default to Legends Outside or in other subplot

```python
# ✅ Outside plot - saves data area (95% of cases)
ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1),
          fontsize=5, frameon=False)

# ✅ Below plot - for horizontal legends
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          ncol=3, fontsize=5, frameon=False)

# ❌ Inside plot - only if truly no other option
ax.legend(loc='best')  # Unreliable and wastes space
```

## Utility Scripts

Always use provided utilities for consistency:

```python
from figure_utilities import set_figure_defaults, save_figure
from figsizing import figure_size_from_panels, cm_to_inch
from label_panels import label_panels
from color_palettes import WONG, OKABE_ITO, get_palette
```

### Core Functions

**set_figure_defaults(context="paper")**: Apply journal style from `assets/paper.mplstyle`

**save_figure(name)**: Save both PDF (vector) and PNG (raster) automatically

**figure_size_from_panels()**: Calculate figure size from panel dimensions

```python
figsize = figure_size_from_panels(
    nrows=2, ncols=2,
    panel_width_cm=4.5,
    panel_height_cm=4.0,
    wspace_cm=0.8,  # Horizontal spacing
    hspace_cm=0.8   # Vertical spacing
)
```

**label_panels(axes)**: Add a, b, c, d labels to multi-panel figures

```python
# For regular subplots
label_panels(axes.flatten())

# For subplot_mosaic (convert dict to list)
label_panels(list(axes.values()), labels=['a', 'b', 'c'])
```

### Color Palettes

```python
from color_palettes import WONG, OKABE_ITO

# Wong palette (8 colors, most common)
colors = WONG  # ['#000000', '#E69F00', '#56B4E9', ...]

# Get specific palette
colors = get_palette('tol_bright')  # Options: wong, okabe_ito, tol_bright, tol_muted
```

## Common Figure Types

For complete examples with code, read:

```bash
cat references/common_patterns.md
```

**Available patterns:**

- Time series with error bands
- Multi-panel comparisons
- Scatter with regression
- Bar plots with error bars
- Heatmaps with colorbars
- Box plots
- Violin plots
- Complex multi-panel layouts

## Pre-Save Validation

**Always validate before finalizing:**

```python
import warnings

# Check for layout warnings
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    plt.tight_layout()
    if w:
        print("⚠️  Layout issue detected:", w[-1].message)
        # Fix and regenerate - don't ignore warnings!
        plt.close()
        # Increase spacing or figure size, then retry

# If no warnings, save
save_figure('figure_name')
plt.close()
```

## Figure Quality Checklist

Before submission:

**Typography:**

- [ ] All fonts ≥ 7pt
- [ ] Fonts are editable (pdf.fonttype: 42)
- [ ] Consistent font family throughout

**Layout:**

- [ ] Used constrained_layout=True
- [ ] Minimal white space (no wasted area)
- [ ] Panel labels if multi-panel
- [ ] No text cut off at edges
- [ ] Legends don't obscure data

**Scientific Standards:**

- [ ] Axes labeled with units
- [ ] Colorblind-friendly palette
- [ ] Works in grayscale
- [ ] Aspect ratio appropriate for data
- [ ] Uncertainty shown where relevant
- [ ] use `rasterized=True` to save space

**Export:**

- [ ] Both PDF and PNG generated
- [ ] PDF opens in Illustrator
- [ ] File size reasonable (<5MB)

## subplot_mosaic Best Practices

Use `subplot_mosaic` when panels need different sizes, non-rectangular layouts, or dedicated legend space.

**Layout patterns** (see `references/spatial_layout.md` for details):

- Main + side panels: `"AAAB\nAAAC"`
- Wide bottom: `"ABC\nDDD"`
- Legend space: `"AABBL\nCDDEL"`
- Empty cells: Use `'.'` for gaps

**Control sizing with ratios:**

```python
fig, axes = plt.subplot_mosaic(layout, figsize=(7, 4),
    width_ratios=[2, 1, 0.3],   # Relative widths
    height_ratios=[1, 1.5],     # Relative heights
    constrained_layout=True)
```

## Design Principles Reference

For detailed explanation of Tufte, Gelman, and Heer principles:

```bash
cat references/design_principles.md
```

**Quick summary:**

- **Tufte**: Maximize data-ink ratio, remove chartjunk
- **Gelman**: Statistical clarity, show uncertainty, appropriate scales
- **Heer**: Perceptual effectiveness, colorblind-safe, effective encodings

## Troubleshooting

**Legends overlap data**: Use `bbox_to_anchor=(1.02, 1)` or create legend panel with subplot_mosaic

**Text too small**: Increase DPI (e.g., 450), not figure size

**Elements cut off**: Add padding with `labelpad` or increase figure margins

**Warnings about layout**: Fix the issue (spacing, size) - don't ignore

**Too much white space**: Use `constrained_layout=True` and reduce margins

**Figure too large in memory**: Use `rasterized=True` in plots

For detailed troubleshooting, see `references/spatial_layout.md`

## Resources

**Style files**: `assets/paper.mplstyle`, `assets/nature.mplstyle`

**Utility scripts**: `scripts/figure_utilities.py`, `scripts/figsizing.py`, `scripts/label_panels.py`, `scripts/color_palettes.py`

**Reference docs**: `references/spatial_layout.md`, `references/design_principles.md`, `references/common_patterns.md`

## Quick Start Template

```python
"""
Standard figure creation template with subplot_mosaic
"""
from figure_utilities import set_figure_defaults, save_figure
from label_panels import label_panels
from color_palettes import WONG
import matplotlib.pyplot as plt

# Set defaults
set_figure_defaults(context="paper")

# Define layout (customize as needed)
layout = """
    AB
    CD
    """

# Create figure
fig, axes = plt.subplot_mosaic(
    layout,
    figsize=(7, 5),
    dpi=450,
    constrained_layout=True
)

# Plot in each panel
axes['A'].plot(x1, y1, color=WONG[1], linewidth=1.5)
axes['A'].set_xlabel('X', fontsize=7, labelpad=10)
axes['A'].set_ylabel('Y', fontsize=7, labelpad=10)

# ... plot B, C, D ...

# Label panels
label_panels(list(axes.values()))

# Validate and save
import warnings
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    plt.tight_layout()
    if w:
        print("⚠️ Warning:", w[-1].message)

save_figure('figure_name')
plt.close()
```
