---
name: scientific-figures-presentation
description: Creates clear, readable figures for scientific presentations and talks using matplotlib. Use when generating figures for conference talks, seminar slides, lecture materials, or any presentation where the audience views from a distance.
license: MIT
---

# Scientific Figure Creation for Presentations

## Overview

Create clear, readable figures optimized for projection and distance viewing. Every element must be visible and comprehensible from 30+ feet away.

## Core Principles

**Distance viewing** - Design for the back row (30+ feet)
**Quick comprehension** - Main point obvious in 3 seconds
**Visual impact** - Bold, saturated colors and thick lines
**Aggressive simplification** - Show only essential information

## Critical First Step

**ALWAYS read spatial layout guide before any plotting:**

```bash
cat references/spatial-layout.md
```

This prevents legend/label overlaps that are hard to debug after creation.

## Essential Standards

### Typography
- **Body text**: 16-24pt
- **Titles**: 32-48pt  
- **Axis labels**: 20-24pt
- **Tick labels**: 16-18pt
- **Font**: Arial, Helvetica (clean sans-serif)

### Dimensions
- **16:9 slides**: 10" × 5.625"
- **DPI**: 150 (screen resolution)
- **Margins**: 0.5-1" from slide edges

### Visual Elements
- **Line width**: 3-5pt (vs 1.5pt in papers)
- **Markers**: 8-12pt (vs 4pt in papers)
- **Colors**: Bold, saturated (Wong palette)
- **Contrast**: High (dark on light OR light on dark)

## Standard Workflow

### Step 1: Load Utilities and Set Context

```python
from scripts.figure_utilities import set_figure_defaults, save_figure
from scripts.color_palettes import WONG
import matplotlib.pyplot as plt

# Set presentation defaults (large fonts, bold lines)
set_figure_defaults(context="presentation")
```

### Step 2: Create Figure with Proper Dimensions

```python
# Standard 16:9 slide
fig, ax = plt.subplots(figsize=(10, 5.625), dpi=150,
                       constrained_layout=True)
```

**Why constrained_layout?**
- Prevents label overlaps automatically
- Better than tight_layout for presentations
- Mandatory for all figures

### Step 3: Plot with Bold, Visible Elements

```python
# THICK lines (3-5pt)
ax.plot(x, y, linewidth=4, color=WONG[1], 
        marker='o', markersize=10, label='Data')

# Or LARGE markers for scatter
ax.scatter(x, y, s=100, alpha=0.8, color=WONG[1])
```

### Step 4: Add Large, Readable Labels

```python
# BOLD and LARGE
ax.set_xlabel('Time (s)', fontsize=24, fontweight='bold', labelpad=15)
ax.set_ylabel('Signal (mV)', fontsize=24, fontweight='bold', labelpad=15)
ax.set_title('Key Finding', fontsize=32, fontweight='bold', pad=20)
```

### Step 5: Configure Legend

```python
# Large legend outside plot area (preferred)
ax.legend(fontsize=20, frameon=False, 
          bbox_to_anchor=(1.02, 1), loc='upper left')

# Or inside if space available
ax.legend(fontsize=20, frameon=False, loc='upper right')
```

### Step 6: Simplify and Clean

```python
# Remove clutter
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Large tick labels
ax.tick_params(labelsize=18, width=2, length=8)

# Limit tick density
ax.xaxis.set_major_locator(plt.MaxNLocator(5))
ax.yaxis.set_major_locator(plt.MaxNLocator(5))
```

### Step 7: Save

```python
save_figure("slide_figure")
plt.close()
```

## Key Differences from Paper Figures

| Aspect | Papers | Presentations |
|--------|--------|--------------|
| Line width | 1.5pt | 3-5pt |
| Font size | 7-10pt | 20-32pt |
| Markers | 4pt | 8-12pt |
| Colors | Subtle OK | Bold required |
| Complexity | Detailed OK | Simplify aggressively |
| Panels | Multi-panel OK | 1-2 max |

## Common Patterns

### Single Plot with Emphasis

```python
set_figure_defaults(context="presentation")
fig, ax = plt.subplots(figsize=(10, 5.625), dpi=150,
                       constrained_layout=True)

# Bold data
ax.plot(x, y, linewidth=4, color=WONG[1], label='Treatment')
ax.plot(x, y_control, linewidth=4, color='gray', 
        alpha=0.7, label='Control')

# Large labels
ax.set_xlabel('Time (s)', fontsize=24, fontweight='bold')
ax.set_ylabel('Response', fontsize=24, fontweight='bold')
ax.set_title('Treatment Doubles Response', fontsize=32, 
             fontweight='bold', pad=20)

# Large legend
ax.legend(fontsize=22, frameon=False)

# Clean axes
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=18)

save_figure('treatment_effect')
plt.close()
```

### Bar Chart Comparison

```python
set_figure_defaults(context="presentation")
fig, ax = plt.subplots(figsize=(10, 5.625), dpi=150,
                       constrained_layout=True)

# Bold bars
x_pos = [0, 1]
heights = [10, 30]
ax.bar(x_pos, heights, color=[WONG[1], WONG[2]], 
       alpha=0.8, width=0.6)

# Direct labels on bars (better than legend)
for i, (x, h) in enumerate(zip(x_pos, heights)):
    ax.text(x, h + 1, f'{h}', ha='center', fontsize=24, 
            fontweight='bold')

ax.set_xticks(x_pos)
ax.set_xticklabels(['Control', 'Treatment'], fontsize=24)
ax.set_ylabel('Response', fontsize=24, fontweight='bold')
ax.set_title('3× Increase with Treatment', fontsize=32,
             fontweight='bold', pad=20)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=18)

save_figure('bar_comparison')
plt.close()
```

## Dark Background Option

For dark rooms or visual impact:

```python
# Use dark background style
set_figure_defaults(context="presentation_dark")

fig, ax = plt.subplots(figsize=(10, 5.625), dpi=150,
                       facecolor='#1a1a1a',
                       constrained_layout=True)
ax.set_facecolor('#1a1a1a')

# Bright colors on dark
ax.plot(x, y, linewidth=4, color=WONG[4], label='Data')  # Yellow

# WHITE text
ax.set_xlabel('X', fontsize=24, color='white', fontweight='bold')
ax.set_ylabel('Y', fontsize=24, color='white', fontweight='bold')
ax.set_title('Title', fontsize=32, color='white', fontweight='bold')

# Configure for dark background
ax.tick_params(colors='white', labelsize=18)
for spine in ax.spines.values():
    spine.set_edgecolor('white')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

legend = ax.legend(fontsize=20, frameon=False)
for text in legend.get_texts():
    text.set_color('white')

save_figure('dark_figure', facecolor='#1a1a1a')
plt.close()
```

See `references/dark-background-guide.md` for complete details.

## Reference Files

**Always read before plotting:**
- `spatial-layout.md` - Prevents overlaps (CRITICAL!)

**Read as needed:**
- `examples.md` - Detailed examples for common plot types
- `dark-background-guide.md` - Complete dark theme guide
- `checklist.md` - Pre-submission validation

## Available Scripts

**Core utilities:**
- `figure_utilities.py` - set_figure_defaults(), save_figure()
- `color_palettes.py` - WONG, OKABE_ITO palettes
- `figsizing.py` - Dimension calculations
- `label_panels.py` - Panel labeling for multi-panel figures

**Style files:**
- `presentation.mplstyle` - Light background style
- `presentation_dark.mplstyle` - Dark background style

## Quick Rules

**Think BIG**:
- Lines: 4pt (not 1.5pt)
- Fonts: 24pt labels, 32pt titles (not 10pt)
- Markers: 10pt (not 4pt)

**Simplify MORE**:
- 1-2 comparisons per slide (not 6)
- Max 5 ticks per axis (not 10)
- Remove all non-essential elements

**High CONTRAST**:
- Bold, saturated colors from Wong palette
- Dark on light OR light on dark
- Avoid pastels and subtle colors

**Test from DISTANCE**:
- Can you read it from 30 feet?
- Is main message obvious in 3 seconds?
- Would it work on a poor projector?

## Validation Checklist

Before adding to slides:

- [ ] Read spatial-layout.md first
- [ ] All fonts ≥16pt (20pt+ preferred)
- [ ] Title ≥32pt
- [ ] Lines ≥3pt width
- [ ] Markers ≥8pt size
- [ ] High contrast colors
- [ ] Main message obvious in 3 seconds
- [ ] Maximum 1-2 key comparisons
- [ ] No overlapping elements
- [ ] Legend large and clear
- [ ] Tested: readable from distance?

See `references/checklist.md` for complete list.

## Common Mistakes

**❌ Too small**: Everything should feel oversized
**❌ Too complex**: Split into multiple slides instead
**❌ Too subtle**: Colors must be bold and saturated
**❌ Too cluttered**: Remove non-essential elements
**❌ Wrong dimensions**: Must match slide aspect ratio

## Integration with Other Skills

### With scientific-presentations Skill

This skill creates the figures. The scientific-presentations skill designs the slides that contain them.

**Workflow:**
1. Use this skill to create plots
2. Use scientific-presentations skill for slide layout
3. Both skills use same color palettes
4. Both emphasize readability at distance

### With pptx Skill

If using pptx skill for programmatic slide creation:
1. Generate figures with this skill
2. Insert into slides with pptx skill
3. Follow layout principles from scientific-presentations skill

## Key Principle

**Everything must be visible and comprehensible from the back row.**

If you squint at your screen and can't read it, your audience won't be able to either.

## When in Doubt

- Make it bigger
- Make it bolder  
- Make it simpler
- Remove decoration
- Test from distance
