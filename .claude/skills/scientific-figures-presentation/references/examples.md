# Detailed Plotting Examples for Presentations

This guide provides complete, tested examples for common plot types in scientific presentations.

## Example 1: Simple Line Plot with Treatment Effect

**Use case**: Show treatment vs control over time

```python
from scripts.figure_utilities import set_figure_defaults, save_figure
from scripts.color_palettes import WONG
import matplotlib.pyplot as plt
import numpy as np

# Generate example data
x = np.linspace(0, 10, 50)
y_control = np.sin(x) + np.random.normal(0, 0.1, 50)
y_treatment = 2 * np.sin(x) + np.random.normal(0, 0.1, 50)

# Set presentation defaults
set_figure_defaults(context="presentation")

# Create figure
fig, ax = plt.subplots(figsize=(10, 5.625), dpi=150,
                       constrained_layout=True)

# Plot with bold lines
ax.plot(x, y_control, linewidth=4, color='gray', 
        alpha=0.7, label='Control')
ax.plot(x, y_treatment, linewidth=4, color=WONG[1], 
        label='Treatment')

# Large, bold labels
ax.set_xlabel('Time (s)', fontsize=24, fontweight='bold', labelpad=15)
ax.set_ylabel('Response', fontsize=24, fontweight='bold', labelpad=15)
ax.set_title('Treatment Doubles Response', fontsize=32, 
             fontweight='bold', pad=20)

# Large legend
ax.legend(fontsize=22, frameon=False, loc='upper left')

# Clean axes
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=18, width=2, length=8)

# Limit ticks
ax.xaxis.set_major_locator(plt.MaxNLocator(5))
ax.yaxis.set_major_locator(plt.MaxNLocator(5))

save_figure('treatment_effect')
plt.close()
```

**Key features:**
- Bold lines (4pt) for visibility
- Large fonts (24pt labels, 32pt title)
- Clear legend with large text
- Clean axes (no top/right spines)
- Limited tick density

---

## Example 2: Bar Chart Comparison

**Use case**: Compare two conditions with direct labels

```python
from scripts.figure_utilities import set_figure_defaults, save_figure
from scripts.color_palettes import WONG
import matplotlib.pyplot as plt
import numpy as np

set_figure_defaults(context="presentation")

fig, ax = plt.subplots(figsize=(10, 5.625), dpi=150,
                       constrained_layout=True)

# Data
categories = ['Control', 'Treatment']
values = [10, 30]
x_pos = np.arange(len(categories))

# Bold bars
bars = ax.bar(x_pos, values, 
              color=[WONG[1], WONG[2]], 
              alpha=0.8, width=0.6)

# Direct labels on bars (better than legend)
for i, (bar, val) in enumerate(zip(bars, values)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{val}', ha='center', va='bottom',
            fontsize=24, fontweight='bold')

# Add significance marker
y_max = max(values)
ax.plot([0, 1], [y_max*1.1, y_max*1.1], 'k-', linewidth=3)
ax.text(0.5, y_max*1.15, '***', ha='center', fontsize=28,
        fontweight='bold')

# Labels
ax.set_xticks(x_pos)
ax.set_xticklabels(categories, fontsize=24, fontweight='bold')
ax.set_ylabel('Response', fontsize=24, fontweight='bold', labelpad=15)
ax.set_title('3× Increase with Treatment', fontsize=32,
             fontweight='bold', pad=20)

# Clean axes
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=18, width=2, length=8)

save_figure('bar_comparison')
plt.close()
```

**Key features:**
- Direct labels on bars (no legend needed)
- Statistical significance markers
- Bold, clear category labels
- High contrast colors

---

## Example 3: Scatter Plot with Trend Line

**Use case**: Show correlation with key point emphasized

```python
from scripts.figure_utilities import set_figure_defaults, save_figure
from scripts.color_palettes import WONG
import matplotlib.pyplot as plt
import numpy as np

set_figure_defaults(context="presentation")

# Generate example data
np.random.seed(42)
x = np.random.randn(50)
y = 2 * x + np.random.randn(50) * 0.5

# Fit line
coeffs = np.polyfit(x, y, 1)
x_fit = np.linspace(x.min(), x.max(), 100)
y_fit = np.polyval(coeffs, x_fit)

fig, ax = plt.subplots(figsize=(10, 5.625), dpi=150,
                       constrained_layout=True)

# Main data - large markers
ax.scatter(x, y, s=100, alpha=0.6, color=WONG[1], 
           label='Data', zorder=2)

# Regression line - bold
ax.plot(x_fit, y_fit, linewidth=4, color=WONG[6],
        linestyle='--', label='Trend', zorder=1)

# Highlight one key point
x_key, y_key = x[25], y[25]
ax.scatter([x_key], [y_key], s=500, color=WONG[6],
           marker='*', zorder=5, edgecolors='black',
           linewidths=3, label='Key observation')

# Labels
ax.set_xlabel('Predictor', fontsize=24, fontweight='bold', labelpad=15)
ax.set_ylabel('Outcome', fontsize=24, fontweight='bold', labelpad=15)
ax.set_title('Strong Correlation (r=0.89)', fontsize=32,
             fontweight='bold', pad=20)

# Legend
ax.legend(fontsize=20, frameon=False, loc='upper left')

# Clean axes
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=18, width=2, length=8)

save_figure('scatter_correlation')
plt.close()
```

**Key features:**
- Large scatter points (100 size)
- Bold trend line (4pt)
- Emphasized key point with star marker
- Correlation in title

---

## Example 4: Side-by-Side Comparison

**Use case**: Direct comparison of two conditions (use sparingly!)

```python
from scripts.figure_utilities import set_figure_defaults, save_figure
from scripts.color_palettes import WONG
import matplotlib.pyplot as plt
import numpy as np

set_figure_defaults(context="presentation")

# Data
x = np.linspace(0, 10, 50)
data1 = np.sin(x) + np.random.normal(0, 0.1, 50)
data2 = 2 * np.sin(x) + np.random.normal(0, 0.1, 50)

# Create side-by-side
fig, axes = plt.subplots(1, 2, figsize=(10, 5.625), dpi=150,
                         constrained_layout=True)

titles = ['Control', 'Treatment']
datasets = [data1, data2]

for ax, title, data in zip(axes, titles, datasets):
    # Bold line
    ax.plot(x, data, linewidth=4, color=WONG[1])
    
    # Large title
    ax.set_title(title, fontsize=28, fontweight='bold', pad=15)
    
    # Labels
    ax.set_xlabel('Time (s)', fontsize=20, fontweight='bold')
    ax.tick_params(labelsize=16, width=2, length=6)
    
    # Clean axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Limit ticks
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))

# Only left axis gets y-label
axes[0].set_ylabel('Response', fontsize=20, fontweight='bold', labelpad=10)

save_figure('side_by_side')
plt.close()
```

**Key features:**
- Only use when direct comparison is essential
- Shared y-axis scale
- Consistent styling across panels
- Left panel gets y-label only

---

## Example 5: Emphasis on Data Region

**Use case**: Highlight specific region or time period

```python
from scripts.figure_utilities import set_figure_defaults, save_figure
from scripts.color_palettes import WONG
import matplotlib.pyplot as plt
import numpy as np

set_figure_defaults(context="presentation")

# Data
x = np.linspace(0, 10, 100)
y = np.sin(x) * np.exp(-x/10)

# Define region of interest
x_start, x_end = 3, 6
highlight_idx = (x >= x_start) & (x <= x_end)

fig, ax = plt.subplots(figsize=(10, 5.625), dpi=150,
                       constrained_layout=True)

# Background data in gray
ax.plot(x, y, linewidth=3, color='gray', alpha=0.5, zorder=1)

# Highlighted region - bold and colored
ax.plot(x[highlight_idx], y[highlight_idx],
        linewidth=5, color=WONG[1], zorder=2)

# Add shaded region
ax.axvspan(x_start, x_end, alpha=0.2, color=WONG[1], zorder=0)

# Annotation with arrow
peak_idx = highlight_idx.nonzero()[0][np.argmax(y[highlight_idx])]
ax.annotate('Critical\nperiod',
            xy=(x[peak_idx], y[peak_idx]),
            xytext=(20, 30), textcoords='offset points',
            fontsize=24, fontweight='bold', ha='center',
            arrowprops=dict(arrowstyle='->', lw=3, color='black'),
            bbox=dict(boxstyle='round,pad=0.5', 
                     facecolor=WONG[4], alpha=0.8,
                     edgecolor='black', linewidth=2))

# Labels
ax.set_xlabel('Time (s)', fontsize=24, fontweight='bold', labelpad=15)
ax.set_ylabel('Signal', fontsize=24, fontweight='bold', labelpad=15)
ax.set_title('Peak Response in Critical Period', fontsize=32,
             fontweight='bold', pad=20)

# Clean axes
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=18, width=2, length=8)

save_figure('emphasis_region')
plt.close()
```

**Key features:**
- Gray background data for context
- Bold highlighted region
- Shaded area for emphasis
- Large annotation with arrow and colored box

---

## Example 6: Error Bars (Use Sparingly)

**Use case**: Show uncertainty when essential

```python
from scripts.figure_utilities import set_figure_defaults, save_figure
from scripts.color_palettes import WONG
import matplotlib.pyplot as plt
import numpy as np

set_figure_defaults(context="presentation")

# Data with errors
x = np.array([1, 2, 3, 4, 5])
y = np.array([10, 15, 13, 17, 20])
errors = np.array([2, 2.5, 2, 3, 2.5])

fig, ax = plt.subplots(figsize=(10, 5.625), dpi=150,
                       constrained_layout=True)

# Plot with BOLD error bars
ax.errorbar(x, y, yerr=errors, 
            linewidth=4, capsize=10, capthick=4,
            marker='o', markersize=12,
            color=WONG[1], ecolor=WONG[1],
            label='Mean ± SEM')

# Labels
ax.set_xlabel('Condition', fontsize=24, fontweight='bold', labelpad=15)
ax.set_ylabel('Response', fontsize=24, fontweight='bold', labelpad=15)
ax.set_title('Consistent Response Across Conditions', fontsize=32,
             fontweight='bold', pad=20)

# Legend
ax.legend(fontsize=20, frameon=False, loc='upper left')

# Clean axes
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=18, width=2, length=8)

# Set x-axis limits for better spacing
ax.set_xlim(0.5, 5.5)

save_figure('error_bars')
plt.close()
```

**Key features:**
- THICK error bars (4pt) for visibility
- Large cap size (10pt)
- Only show error when essential to message
- Consider showing individual points instead

---

## Example 7: Heatmap

**Use case**: Show matrix data with clear colorbar

```python
from scripts.figure_utilities import set_figure_defaults, save_figure
from scripts.color_palettes import WONG
import matplotlib.pyplot as plt
import numpy as np

set_figure_defaults(context="presentation")

# Generate example matrix
data = np.random.randn(8, 8)

fig, ax = plt.subplots(figsize=(10, 5.625), dpi=150,
                       constrained_layout=True)

# Create heatmap
im = ax.imshow(data, cmap='RdBu_r', aspect='auto',
               vmin=-2, vmax=2)

# Add colorbar with large labels
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=18, width=2, length=6)
cbar.set_label('Value', fontsize=20, fontweight='bold',
               rotation=270, labelpad=25)

# Labels
ax.set_xlabel('Column', fontsize=24, fontweight='bold', labelpad=10)
ax.set_ylabel('Row', fontsize=24, fontweight='bold', labelpad=10)
ax.set_title('Data Matrix', fontsize=32, fontweight='bold', pad=20)

# Large tick labels
ax.tick_params(labelsize=18, width=2, length=6)

save_figure('heatmap')
plt.close()
```

**Key features:**
- High contrast colormap
- Large colorbar labels
- Clear axis labels
- Limited matrix size for readability

---

## Example 8: Multi-Panel Figure (2 panels max!)

**Use case**: When comparison requires side-by-side (use rarely)

```python
from scripts.figure_utilities import set_figure_defaults, save_figure
from scripts.color_palettes import WONG
from scripts.label_panels import label_panels
import matplotlib.pyplot as plt
import numpy as np

set_figure_defaults(context="presentation")

# Data
x = np.linspace(0, 10, 50)
y1 = np.sin(x)
y2 = np.cos(x)

fig, axes = plt.subplots(1, 2, figsize=(10, 5.625), dpi=150,
                         constrained_layout=True)

# Panel A
axes[0].plot(x, y1, linewidth=4, color=WONG[1])
axes[0].set_title('Sine Wave', fontsize=24, fontweight='bold', pad=10)
axes[0].set_xlabel('X', fontsize=20, fontweight='bold')
axes[0].set_ylabel('Y', fontsize=20, fontweight='bold')
axes[0].tick_params(labelsize=16)
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

# Panel B  
axes[1].plot(x, y2, linewidth=4, color=WONG[2])
axes[1].set_title('Cosine Wave', fontsize=24, fontweight='bold', pad=10)
axes[1].set_xlabel('X', fontsize=20, fontweight='bold')
axes[1].tick_params(labelsize=16)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

# Add panel labels
label_panels(axes, xpos=-0.15, ypos=1.05, fontsize=20, fontweight='bold')

save_figure('two_panels')
plt.close()
```

**Key features:**
- Maximum 2 panels for presentations
- Panel labels (A, B) for reference
- Consistent styling across panels
- Consider: could this be split into separate slides?

---

## General Tips

### Simplification Strategy

**Before creating figure, ask:**
1. What's the ONE key message?
2. What's the MINIMUM data needed to show it?
3. Can I split this into multiple slides?

### Progressive Disclosure

**Instead of one complex figure:**
```
Slide 1: Empty axes + question
Slide 2: Add baseline data
Slide 3: Add treatment data  
Slide 4: Highlight difference
```

**Create multiple versions:**
```python
# Save progressive versions
save_figure('plot_v1_axes_only')
# Add baseline
save_figure('plot_v2_with_baseline')
# Add treatment
save_figure('plot_v3_with_treatment')
# Add emphasis
save_figure('plot_v4_final')
```

Use slide animations/transitions to reveal incrementally.

### Testing

**View your figure:**
1. From across the room (simulate back row)
2. On actual projector (if possible)
3. With reduced brightness (simulate bad projector)

**Ask:**
- Can I read all text from 20+ feet?
- Is main message obvious in 3 seconds?
- Are colors distinct?
- Is anything overlapping?

---

## Common Mistakes and Fixes

**Mistake**: Lines too thin
**Fix**: Use linewidth=4 (not 1.5)

**Mistake**: Fonts too small
**Fix**: Labels ≥24pt, titles ≥32pt

**Mistake**: Too many elements
**Fix**: Show only 1-2 key comparisons

**Mistake**: Subtle colors
**Fix**: Use bold, saturated colors from WONG

**Mistake**: Complex multi-panel
**Fix**: Split into separate slides

**Mistake**: Overlapping labels
**Fix**: Read spatial-layout.md FIRST, use constrained_layout=True

**Mistake**: Dense ticks
**Fix**: ax.xaxis.set_major_locator(plt.MaxNLocator(5))
