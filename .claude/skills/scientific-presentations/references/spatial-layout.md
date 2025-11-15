# Spatial Layout and Overlap Prevention Guide

**Purpose**: Prevent overlap of visual elements (legends, labels, tick marks, annotations) in matplotlib figures through explicit spatial reasoning before coding.

**Problem**: Claude Code cannot "see" figures after creation, so it must predict spatial layout without visual feedback. This guide provides a systematic approach to spatial reasoning that prevents overlap issues.

---

## CRITICAL: Visual Layout Reasoning Before Coding

**Before writing any plotting code, explicitly reason through spatial layout to prevent overlap.**

### Pre-Coding Spatial Checklist (MANDATORY)

Ask and answer these questions BEFORE writing code:

1. **Figure dimensions**: What is the physical size in inches? What is the aspect ratio?
2. **Number of subplots**: How many panels? What's the arrangement (rows × cols)?
3. **Legend placement**: Where will the legend go? Will it fit without covering data?
4. **Axis label lengths**: How long are the labels? Will they overlap with tick labels?
5. **Title length**: Is the title long? Will it fit in the space above the plot?
6. **Tick label lengths**: Are tick labels long (dates, large numbers)? Will they overlap?
7. **Margin space**: Is there enough space around the plot for all labels?

---

## Spatial Layout Workflow

### Step 1: Calculate Required Space

Before plotting, explicitly calculate space requirements:

```python
# Example: Single panel figure for paper
panel_width = 3.5  # inches
panel_height = 2.5  # inches

# Space needed for elements:
# - Title: ~0.3-0.5" above plot
# - Y-axis label: ~0.3-0.5" left of plot  
# - X-axis label: ~0.3-0.5" below plot
# - Tick labels: ~0.2-0.3" on each side
# - Legend (if inside): Reserve 25% of plot area OR place outside
# - Margins: 0.1-0.15" minimum on all sides

# Total figure size calculation
title_space = 0.4
ylabel_space = 0.4
xlabel_space = 0.4
tick_space = 0.25
margin = 0.15

fig_width = panel_width + ylabel_space + tick_space + margin
fig_height = panel_height + title_space + xlabel_space + margin

# If legend outside right:
legend_width = 1.0  # Typical legend width
if legend_outside_right:
    fig_width += legend_width

# If legend below:
legend_height = 0.5  # Typical legend height
if legend_below:
    fig_height += legend_height
```

### Step 2: Use constrained_layout (MANDATORY)

```python
# ALWAYS use constrained_layout for automatic spacing
fig, ax = plt.subplots(figsize=(fig_width, fig_height), 
                       dpi=450,  # or 150 for presentations
                       constrained_layout=True)

# constrained_layout automatically:
# - Prevents axis labels from overlapping tick labels
# - Prevents titles from overlapping with axes
# - Adjusts spacing between subplots
# - Makes room for colorbars and legends (when properly configured)
```

**Why constrained_layout?**
- More modern and reliable than tight_layout
- Handles complex layouts better
- Works during figure creation (not just at save time)
- Automatically adjusts to content

### Step 3: Place Legend Outside Plot Area (PREFERRED)

```python
# PREFERRED: Always place legend outside the data area

# Option 1: Right of plot (most common, works for most cases)
ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), 
          fontsize=5,  # paper: 5-6pt, presentation: 16-20pt
          frameon=False)

# Option 2: Below plot (good for horizontal legends with many items)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), 
          ncol=3,  # Arrange horizontally
          fontsize=5, 
          frameon=False)

# Option 3: Above plot (rare, only when space permits)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), 
          ncol=3, 
          fontsize=5, 
          frameon=False)

# ONLY use legend inside plot if ALL of these are true:
# - There's clearly empty space in a corner (verified visually)
# - Using semi-transparent background (framealpha=0.9)
# - Certain no data will be obscured
# - Fewer than 4 legend items
if must_place_legend_inside:
    ax.legend(loc='upper right',  # Choose specific corner
              fontsize=5,
              frameon=True, 
              framealpha=0.95,  # Semi-transparent
              edgecolor='gray', 
              fancybox=False)
```

**Legend placement decision tree:**
- **1-2 items** → Can go inside if there's space, prefer outside
- **3-4 items** → Outside strongly preferred
- **5+ items** → MUST go outside, consider ncol for horizontal layout

### Step 4: Handle Long Tick Labels

```python
# For long x-axis labels (dates, long category names)
# Option 1: Rotate
ax.tick_params(axis='x', rotation=45)
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

# Option 2: Reduce number of ticks
ax.xaxis.set_major_locator(plt.MaxNLocator(5))  # Max 5 ticks

# Option 3: Use abbreviations/formatting
ax.xaxis.set_major_formatter(plt.FuncFormatter(
    lambda x, p: f'{x:.1f}k' if x >= 1000 else f'{x:.0f}'
))

# For long y-axis labels (large numbers, scientific notation)
# Option 1: Scientific notation
ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

# Option 2: Custom formatting
ax.yaxis.set_major_formatter(plt.FuncFormatter(
    lambda x, p: f'{x:.1e}' if abs(x) > 1000 else f'{x:.1f}'
))

# Option 3: Reduce precision
ax.yaxis.set_major_formatter(plt.FuncFormatter(
    lambda x, p: f'{x:.0f}'  # No decimal places
))
```

### Step 5: Add Extra Padding to Labels

```python
# Increase padding if labels are close to ticks/edges
# Default labelpad is 4 points - usually too small

# For papers:
ax.set_xlabel('X Label', fontsize=7, labelpad=10)  # 2.5x default
ax.set_ylabel('Y Label', fontsize=7, labelpad=10)
ax.set_title('Title', fontsize=7, pad=15)  # Default is 6

# For presentations:
ax.set_xlabel('X Label', fontsize=24, labelpad=15)
ax.set_ylabel('Y Label', fontsize=24, labelpad=15)
ax.set_title('Title', fontsize=32, pad=20)

# For multi-panel figures, increase subplot spacing
if multiple_subplots:
    fig.subplots_adjust(wspace=0.4, hspace=0.5)
    # wspace: horizontal space between subplots (as fraction of subplot width)
    # hspace: vertical space between subplots (as fraction of subplot height)
```

---

## Common Overlap Scenarios and Solutions

### Scenario 1: Legend Covers Data

**Problem**: Legend placed at 'best' or inside plot covers important data points

**Why it happens**: Matplotlib's 'best' algorithm doesn't always work, especially with complex data

**Solutions**:
```python
# ❌ NEVER do this - unreliable
ax.legend(loc='best')

# ✅ ALWAYS: Place legend outside
ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), 
          fontsize=5, frameon=False)

# ✅ ALTERNATIVE: If must be inside, use transparent background
ax.legend(loc='upper right', fontsize=5, 
          frameon=True, framealpha=0.95, 
          edgecolor='gray', linewidth=0.5)
```

### Scenario 2: Y-Axis Label Overlaps Tick Labels

**Problem**: Long numbers on y-axis (e.g., "10000", "1.23456") overlap with y-axis label text

**Why it happens**: Default labelpad (4pt) is too small for long tick labels

**Solutions**:
```python
# ❌ Default padding too small
ax.set_ylabel('Response (units)')

# ✅ Increase label padding
ax.set_ylabel('Response (units)', labelpad=15)

# ✅ OR use scientific notation to shorten ticks
ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
# Example: 10000 → 1.0×10⁴

# ✅ OR reduce number of ticks
ax.yaxis.set_major_locator(plt.MaxNLocator(5))
# Fewer ticks = more space between them

# ✅ OR reduce tick label font size
ax.tick_params(axis='y', labelsize=4)  # Smaller font
```

### Scenario 3: X-Axis Labels Overlap Each Other

**Problem**: Long date labels or category names overlap horizontally

**Why it happens**: Too many ticks or labels too long for available space

**Solutions**:
```python
# ❌ Horizontal labels overlap
ax.set_xticklabels(long_labels)

# ✅ Solution 1: Rotate labels
ax.set_xticklabels(long_labels, rotation=45, ha='right')
# ha='right' aligns text properly when rotated

# ✅ Solution 2: Reduce number of labels
ax.xaxis.set_major_locator(plt.MaxNLocator(5))
# Shows fewer ticks, more space between

# ✅ Solution 3: Abbreviate labels
short_labels = [label[:10] + '...' if len(label) > 10 else label 
                for label in long_labels]
ax.set_xticklabels(short_labels)

# ✅ Solution 4: Vertical labels (90 degrees)
ax.set_xticklabels(long_labels, rotation=90, ha='center')
# Works for very long labels
```

### Scenario 4: Title Overlaps with Top of Plot

**Problem**: Title sits too close to the top of the axes

**Why it happens**: Default pad (6pt) insufficient for larger fonts or tight layouts

**Solutions**:
```python
# ❌ Default padding
ax.set_title('Title')

# ✅ Increase title padding
ax.set_title('Title', fontsize=7, pad=20)
# pad in points, increase as needed

# ✅ OR use figure suptitle instead
fig.suptitle('Title', fontsize=8, y=0.98)
# y controls vertical position (0-1, 1=top)

# ✅ OR make figure taller
figsize = (width, height + 0.3)  # Add 0.3" for title space
```

### Scenario 5: Subplots Overlap Each Other

**Problem**: Multi-panel figure has panels overlapping or touching

**Why it happens**: Default spacing insufficient for labels, titles, or tick labels

**Solutions**:
```python
# ❌ Tight spacing causes overlap
fig, axes = plt.subplots(2, 2, figsize=(7, 7))

# ✅ Solution 1: Use constrained_layout (BEST)
fig, axes = plt.subplots(2, 2, figsize=(7, 7), 
                         constrained_layout=True)

# ✅ Solution 2: Manually adjust spacing
fig.subplots_adjust(wspace=0.4,  # Horizontal space
                    hspace=0.5,  # Vertical space
                    left=0.15,   # Left margin
                    right=0.95,  # Right margin
                    top=0.95,    # Top margin
                    bottom=0.1)  # Bottom margin

# ✅ Solution 3: Calculate proper figure size upfront
from scripts.figsizing import figure_size_from_panels
figsize = figure_size_from_panels(
    nrows=2, ncols=2,
    panel_width_cm=4.5,
    panel_height_cm=4.5,
    wspace_cm=1.5,  # More horizontal space
    hspace_cm=1.5   # More vertical space
)
fig, axes = plt.subplots(2, 2, figsize=figsize, 
                         constrained_layout=True)
```

### Scenario 6: Annotations Cover Data

**Problem**: Text annotations obscure important data points or features

**Why it happens**: Text placed directly at data point coordinates

**Solutions**:
```python
# ❌ Text directly on data point
ax.text(x, y, 'Label', fontsize=6)

# ✅ Offset annotation with arrow
ax.annotate('Label', 
            xy=(x, y),              # Point to annotate
            xytext=(10, 10),        # Offset in points
            textcoords='offset points',
            fontsize=6,
            bbox=dict(boxstyle='round,pad=0.3', 
                     facecolor='white', 
                     edgecolor='gray',
                     alpha=0.9),
            arrowprops=dict(arrowstyle='->', 
                           connectionstyle='arc3,rad=0.3',
                           lw=0.5, 
                           color='gray'))

# ✅ Position strategically in empty space
# Identify empty regions, place text there with arrow pointing to data
empty_space_coords = (x + 0.5, y + 0.5)  # Calculate empty space
ax.annotate('Label', 
            xy=(x, y),
            xytext=empty_space_coords,
            textcoords='data',
            fontsize=6,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
            arrowprops=dict(arrowstyle='->', lw=0.5))

# ✅ Use adjustText library if available (automatic placement)
from adjustText import adjust_text
texts = [ax.text(x[i], y[i], labels[i], fontsize=6) 
         for i in range(len(x))]
adjust_text(texts, 
            arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))
```

---

## Explicit Spatial Reasoning Template

**Use this template for EVERY figure - document your spatial plan in comments:**

```python
"""
==========================================
SPATIAL LAYOUT PLAN
==========================================
Figure size: 3.5" × 2.5" (single column paper figure)
Plot area: ~3.0" × 2.0" (after margins)

CONTENT:
- Number of data series: 3
- Data type: Time series lines
- X-axis range: 0-10 seconds
- Y-axis range: 0-100 mV

ELEMENTS:
- Legend: 3 items (Treatment A, B, Control)
- X-axis label: "Time (s)" - 9 characters
- Y-axis label: "Response (mV)" - 13 characters  
- Title: "Treatment Effects Over Time" - 28 characters
- Tick labels: X: 0,2,4,6,8,10 (short), Y: 0,25,50,75,100 (medium)

SPATIAL ANALYSIS:
✓ Legend: 3 items → place outside right → needs +1.0" width
✓ Y-axis label: 13 chars → OK with labelpad=10
✓ Title: 28 chars → OK with pad=15, might be tight
✓ Tick labels: Numbers short → no overlap expected
✓ X-axis labels: Numbers short → no rotation needed

SPATIAL DECISIONS:
- Total figure width: 3.5 + 1.0 (legend) = 4.5"
- Total figure height: 2.5" (sufficient)
- Legend placement: outside right at (1.02, 1)
- Use constrained_layout: YES
- Label padding: xlabel=10, ylabel=10, title=15
- No tick label rotation needed

PROCEED WITH CODING
==========================================
"""

# Import required modules
from scripts.figure_utilities import set_figure_defaults, save_figure
from scripts.color_palettes import WONG
import matplotlib.pyplot as plt

# Set figure defaults
set_figure_defaults(context="paper")

# Implement spatial plan
fig, ax = plt.subplots(figsize=(4.5, 2.5), dpi=450,
                       constrained_layout=True)

# Plot data with explicit z-order for layering
for i, (data, label) in enumerate(zip(datasets, labels)):
    ax.plot(x, data, linewidth=1.5, color=WONG[i], 
            label=label, zorder=2)

# Labels with calculated padding
ax.set_xlabel('Time (s)', fontsize=7, labelpad=10)
ax.set_ylabel('Response (mV)', fontsize=7, labelpad=10)
ax.set_title('Treatment Effects Over Time', fontsize=7, pad=15)

# Legend outside as planned
ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1),
          fontsize=5, frameon=False)

# Clean up spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Validate before saving
import warnings
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    plt.tight_layout()
    if w:
        print(f"⚠️  Layout warning detected: {w[-1].message}")
        print("   Fix issues before saving!")
        # If warnings, STOP and fix

# Save only if no warnings
save_figure('treatment_effects')
plt.close()
```

---

## Conservative Layout Rules

**When in doubt, err on the side of MORE space:**

### Rule 1: Legends Default to Outside
- **Always**: Place legend outside unless you have a specific reason not to
- **Inside only if**: <4 items AND verified empty space AND transparent background

### Rule 2: Figure Size Padding
- **Calculate minimum size**: Based on content
- **Add 10-20% margin**: Make figure slightly larger than calculated
- **Example**: Calculated 3.5" → use 3.8-4.0"

### Rule 3: Label Padding
- **Minimum labelpad**: 10 points (not default 4)
- **For long tick labels**: 15 points
- **Title pad**: 15-20 points minimum

### Rule 4: Subplot Spacing
- **Minimum wspace**: 0.4 (not default ~0.2)
- **Minimum hspace**: 0.5 (not default ~0.2)
- **Increase if labels long**: wspace=0.5-0.6, hspace=0.6-0.7

### Rule 5: Margins
- **Minimum margin**: 0.15 inches on all sides
- **With legends outside**: 0.2 inches
- **Multi-panel figures**: 0.2 inches

### Rule 6: Font Sizes
- **Papers**: Labels 7pt, ticks 5pt, legend 5pt
- **Presentations**: Labels 20-24pt, ticks 16-18pt, legend 16-20pt
- **Never below**: 16pt for presentations, 5pt for papers

---

## Validation After Creation

**Always check for overlap indicators:**

```python
import warnings
import matplotlib.pyplot as plt

# Create your figure
fig, ax = plt.subplots(...)
# ... plotting code ...

# VALIDATION STEP: Check for warnings
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    
    # Try tight_layout (even if using constrained_layout)
    # This will warn about potential issues
    try:
        plt.tight_layout()
    except Exception as e:
        print(f"⚠️  tight_layout failed: {e}")
    
    # Check for warnings
    if w:
        print("\n" + "="*60)
        print("⚠️  POTENTIAL OVERLAP DETECTED")
        print("="*60)
        for warning in w:
            print(f"  - {warning.message}")
        
        print("\nSUGGESTED FIXES:")
        print("  1. Increase figure size by 10-20%")
        print("  2. Increase label padding (labelpad=15)")
        print("  3. Place legend outside plot area")
        print("  4. Rotate tick labels")
        print("  5. Reduce font sizes slightly")
        print("  6. Increase subplot spacing (wspace/hspace)")
        print("="*60 + "\n")
        
        # DO NOT SAVE - fix issues first
        plt.close()
        raise ValueError("Fix overlap issues before saving!")

# If no warnings, safe to save
save_figure('figure_name')
plt.close()
```

---

## Multi-Panel Figure Spatial Planning

**For complex multi-panel figures, extra planning is required:**

```python
"""
==========================================
MULTI-PANEL SPATIAL PLAN
==========================================
Layout: 2 rows × 2 cols = 4 panels
Panel size: 4.0 cm × 4.0 cm each
Spacing: 1.5 cm horizontal, 1.5 cm vertical

PANEL CONTENTS:
- Panel A (0,0): Line plot, legend outside right
- Panel B (0,1): Scatter plot, no legend
- Panel C (1,0): Bar plot, no legend  
- Panel D (1,1): Box plot, legend outside right

SPATIAL CONCERNS:
- Panels A & D: Need extra width for legends
- All panels: Need space for axis labels
- Panel letters: Need space at top-left corners
- Y-axis labels: Will be on left side of panels A & C
- X-axis labels: Will be on bottom of panels C & D

SPACING ANALYSIS:
- Horizontal: Need 2.0 cm for panel D legend
- Vertical: Standard 1.5 cm sufficient
- Panel letters: Place at (-0.15, 1.05) in axes coords

FINAL FIGURE SIZE:
- Using figure_size_from_panels with wspace=2.0 cm
- Total: ~10" × 7" (approximate)
==========================================
"""

from scripts.figsizing import figure_size_from_panels
from scripts.label_panels import label_panels

# Calculate figure size with generous spacing
figsize = figure_size_from_panels(
    nrows=2, ncols=2,
    panel_width_cm=4.5,   # Slightly larger for legends
    panel_height_cm=4.0,
    wspace_cm=2.0,        # Extra space for legends
    hspace_cm=1.5
)

fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=450,
                         constrained_layout=True)

# Panel A (top-left): Line plot with legend
axes[0, 0].plot(x, y1, linewidth=1.5, color=WONG[1], label='Condition 1')
axes[0, 0].plot(x, y2, linewidth=1.5, color=WONG[2], label='Condition 2')
axes[0, 0].legend(loc='upper left', bbox_to_anchor=(1.02, 1), 
                  fontsize=5, frameon=False)
axes[0, 0].set_xlabel('Time (s)', fontsize=7, labelpad=8)
axes[0, 0].set_ylabel('Response', fontsize=7, labelpad=8)

# Panel B (top-right): Scatter plot, no legend
axes[0, 1].scatter(x, y3, s=20, alpha=0.6, color=WONG[1])
axes[0, 1].set_xlabel('X', fontsize=7, labelpad=8)
axes[0, 1].set_ylabel('Y', fontsize=7, labelpad=8)

# Panel C (bottom-left): Bar plot
axes[1, 0].bar(categories, values, color=WONG[1], alpha=0.8)
axes[1, 0].set_xlabel('Category', fontsize=7, labelpad=8)
axes[1, 0].set_ylabel('Count', fontsize=7, labelpad=8)

# Panel D (bottom-right): Box plot with legend
bp = axes[1, 1].boxplot([data1, data2], labels=['A', 'B'])
# Manual legend for boxplot
axes[1, 1].legend([bp['boxes'][0]], ['Treatment'], 
                  loc='upper left', bbox_to_anchor=(1.02, 1),
                  fontsize=5, frameon=False)
axes[1, 1].set_xlabel('Group', fontsize=7, labelpad=8)
axes[1, 1].set_ylabel('Value', fontsize=7, labelpad=8)

# Add panel labels with safe positioning
label_panels(axes.flatten(), 
             xpos=-0.15,  # Left of plot area
             ypos=1.05,   # Above plot area
             fontsize=8,
             fontweight='bold')

# Validate
with warnings.catch_warnings(record=True) as w:
    plt.tight_layout()
    if w:
        print("⚠️  Overlap detected in multi-panel figure!")
        # Fix and regenerate

save_figure('multi_panel_figure')
plt.close()
```

---

## Key Principles Summary

### 1. Plan Before Code
- Document spatial layout in comments before writing any plot code
- Answer all 7 questions in pre-coding checklist
- Calculate required space explicitly

### 2. Always Use constrained_layout
- Most reliable automatic spacing method
- Handles complex layouts better than tight_layout
- Required for all figures

### 3. Default to Legends Outside
- Outside placement: 95% of cases
- Inside placement: Only with explicit justification
- Use transparent backgrounds if inside

### 4. Conservative Padding
- Use labelpad=10 minimum (not default 4)
- Use pad=15 for titles (not default 6)
- Add 10-20% extra space to calculated dimensions

### 5. Validate Before Saving
- Check for matplotlib warnings
- Fix all warnings before saving
- Don't rely on visual inspection alone

### 6. Iterate When Needed
- If warnings appear, fix and regenerate
- Don't try to "work around" overlap
- Making figure larger is usually easiest fix

---

## Integration into Workflow

**Standard figure creation workflow with spatial reasoning:**

```python
# ============================================
# STEP 1: PLAN spatial layout (in comments)
# ============================================
"""
Figure: 3.5" × 2.5"
Content: [describe]
Elements: [list all text elements]
Spatial concerns: [identify potential overlaps]
Solutions: [how to prevent overlaps]
"""

# ============================================
# STEP 2: Set figure defaults
# ============================================
from scripts.figure_utilities import set_figure_defaults, save_figure
from scripts.color_palettes import WONG
import matplotlib.pyplot as plt

set_figure_defaults(context="paper")  # or "presentation"

# ============================================
# STEP 3: Calculate figure size with margins
# ============================================
base_width = 3.5
base_height = 2.5
legend_space = 1.0  # if legend outside
margin = 0.2

figsize = (base_width + legend_space + margin, base_height + margin)

# ============================================
# STEP 4: Create figure with constrained_layout
# ============================================
fig, ax = plt.subplots(figsize=figsize, dpi=450, 
                       constrained_layout=True)

# ============================================
# STEP 5: Plot data with explicit z-order
# ============================================
ax.plot(x, y, linewidth=1.5, color=WONG[1], 
        label='Data', zorder=2)

# ============================================
# STEP 6: Add labels with generous padding
# ============================================
ax.set_xlabel('X Label', fontsize=7, labelpad=10)
ax.set_ylabel('Y Label', fontsize=7, labelpad=10)
ax.set_title('Title', fontsize=7, pad=15)

# ============================================
# STEP 7: Place legend outside
# ============================================
ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1),
          fontsize=5, frameon=False)

# ============================================
# STEP 8: Validate for warnings
# ============================================
import warnings
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    plt.tight_layout()
    if w:
        print("⚠️  Overlap detected, fixing...")
        plt.close()
        # Fix issues and regenerate

# ============================================
# STEP 9: Save only if no warnings
# ============================================
save_figure('figure_name')
plt.close()
```

---

## Quick Reference: Common Fixes

| Problem | Quick Fix |
|---------|-----------|
| Legend covers data | `ax.legend(bbox_to_anchor=(1.02, 1))` |
| Y-label overlaps ticks | `ax.set_ylabel(..., labelpad=15)` |
| X-labels overlap | `ax.tick_params(rotation=45)` |
| Title too close | `ax.set_title(..., pad=20)` |
| Subplots overlap | `fig.subplots_adjust(wspace=0.4, hspace=0.5)` |
| Annotations cover data | Use `ax.annotate` with `xytext` offset |
| Everything cramped | Increase figure size by 20% |
| Warnings about layout | Use `constrained_layout=True` |

---

## When to Increase Figure Size

Increase figure dimensions if:
- Matplotlib warnings about tight layout
- More than 3 legend items (need more space)
- Long axis labels (>15 characters)
- Multiple subplots (>4 panels)
- Complex annotations
- Many data series (>5 lines)

**How much to increase:**
- Start with +0.5" in the dimension with issues
- For legends: +1.0" width
- For rotated labels: +0.5" height
- For multi-panel: Calculate using figure_size_from_panels

---

**Remember**: It's always easier to make a figure larger than to try to cram elements into too-small space. When in doubt, add more space!
