# Design Principles for Scientific Figures

Comprehensive guide to Tufte, Gelman, and Heer principles for publication-quality figures.

---

## Edward Tufte: Data-Ink Ratio

### Core Principle

**Maximize the data-ink ratio**: The proportion of ink devoted to displaying actual data should be maximized relative to the total ink used in the graphic.

### Data-Ink Ratio = (Ink used to display data) / (Total ink used in graphic)

### Key Guidelines

1. **Erase non-data ink**
   - Remove unnecessary grid lines
   - Eliminate redundant labels
   - Remove decorative elements (shadows, 3D effects, gradients)
   - Remove chart borders when possible

2. **Erase redundant data-ink**
   - Don't repeat the same information
   - Use range frames instead of full axes
   - Remove duplicate legends

3. **Revise and edit**
   - Simplify, simplify, simplify
   - Each element must earn its place
   - If removing something doesn't reduce understanding, remove it

### Implementation in matplotlib

```python
# ❌ High non-data ink (chartjunk)
ax.grid(True, linestyle='-', linewidth=1, alpha=0.3)
ax.set_facecolor('#f0f0f0')
ax.legend(frameon=True, shadow=True, fancybox=True)

# ✅ Minimal non-data ink
ax.grid(False)
ax.set_facecolor('white')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(frameon=False)

# ✅ Even more minimal (range frame)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_bounds(min(y), max(y))
ax.spines['bottom'].set_bounds(min(x), max(x))
```

### Small Multiples

Tufte's most powerful technique: repeat the same graph structure for different data subsets.

```python
# Small multiples: same scale, easy comparison
fig, axes = plt.subplots(2, 3, figsize=(7, 4.5), 
                         sharex=True, sharey=True,
                         constrained_layout=True)

for ax, condition in zip(axes.flat, conditions):
    ax.plot(x, data[condition], color=WONG[1], linewidth=1.5)
    ax.set_title(condition, fontsize=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Only show labels on edge subplots
for ax in axes[-1, :]:
    ax.set_xlabel('Time (s)', fontsize=7)
for ax in axes[:, 0]:
    ax.set_ylabel('Response', fontsize=7)
```

### Sparklines

Tufte's intense, simple, word-sized graphics.

```python
def sparkline(ax, data):
    """Create a minimal sparkline"""
    ax.plot(data, color='black', linewidth=0.5)
    ax.fill_between(range(len(data)), data, alpha=0.1, color='black')
    ax.set_xlim(0, len(data)-1)
    ax.set_ylim(min(data)*0.9, max(data)*1.1)
    # Remove all chart elements
    ax.axis('off')

# Use in table or alongside text
fig, axes = plt.subplots(5, 1, figsize=(1, 2))
for ax, data in zip(axes, datasets):
    sparkline(ax, data)
```

---

## Andrew Gelman: Statistical Clarity

### Core Principle

**Graphs should make statistical thinking transparent**: Display data, show uncertainty, facilitate understanding of statistical relationships.

### Key Guidelines

1. **Show the data**
   - Plot actual data points when possible
   - Show distributions, not just summaries
   - Display uncertainty (confidence intervals, error bars)

2. **Avoid misleading scales**
   - Don't truncate y-axis without good reason (show context)
   - Use appropriate aspect ratios
   - Be consistent with scales across related plots

3. **Make comparisons easy**
   - Use consistent scales
   - Align baselines
   - Order categories meaningfully

4. **Label everything clearly**
   - Units on all axes
   - Clear titles
   - Descriptive legends

### Implementation

#### Show Uncertainty

```python
# ✅ Show confidence intervals
ax.fill_between(x, y_lower, y_upper, alpha=0.2, color=WONG[1],
                label='95% CI')
ax.plot(x, y_mean, color=WONG[1], linewidth=1.5, label='Mean')

# ✅ Error bars with caps
ax.errorbar(x, y, yerr=std, capsize=3, capthick=1,
            color=WONG[1], linewidth=1.5, marker='o', markersize=4)

# ✅ Show individual points with summary
ax.scatter(x_jitter, y, alpha=0.3, s=20, color=WONG[1])
ax.plot(x_groups, y_means, color=WONG[6], linewidth=2, 
        marker='D', markersize=6, label='Group mean')
```

#### Appropriate Scales

```python
# ❌ Truncated y-axis without context
ax.set_ylim(98, 102)  # Makes 1% difference look huge

# ✅ Show full context or indicate break
ax.set_ylim(0, 105)  # Shows true magnitude of difference

# ✅ Or use broken axis if justified
from matplotlib.patches import Rectangle
# [Code for broken axis visualization]
```

#### Facilitate Comparisons

```python
# ✅ Shared y-axis for easy comparison
fig, axes = plt.subplots(1, 3, figsize=(7, 2.5),
                         sharey=True,  # Same scale!
                         constrained_layout=True)

for ax, condition in zip(axes, conditions):
    ax.bar(categories, data[condition], color=WONG[1])
    ax.set_title(condition, fontsize=7)
    
# Only leftmost needs y-label
axes[0].set_ylabel('Count', fontsize=7)
```

#### Clear Labels with Units

```python
# ❌ No units
ax.set_xlabel('Time')
ax.set_ylabel('Response')

# ✅ Units specified
ax.set_xlabel('Time (s)', fontsize=7)
ax.set_ylabel('Response (mV)', fontsize=7)
ax.set_title('Signal over time (n=50)', fontsize=7)
```

### Regression Uncertainty

```python
# Show regression with confidence band
from scipy import stats

# Fit
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
y_pred = slope * x + intercept

# Prediction interval
predict_error = np.sqrt(std_err**2 + residual_std**2)
y_upper = y_pred + 1.96 * predict_error
y_lower = y_pred - 1.96 * predict_error

# Plot
ax.scatter(x, y, alpha=0.6, s=20, color=WONG[1], label='Data')
ax.plot(x, y_pred, color=WONG[6], linewidth=2, label='Fit')
ax.fill_between(x, y_lower, y_upper, alpha=0.2, color=WONG[6],
                label='95% prediction interval')
ax.legend(fontsize=5, frameon=False)
```

---

## Jeffrey Heer: Perceptual Effectiveness

### Core Principle

**Visual encodings should match human perceptual capabilities**: Use encodings that are accurately perceived and easily discriminated.

### Perceptual Ranking

From most to least accurately perceived:

1. **Position along common scale** (best)
2. Position along non-aligned scale
3. Length
4. Angle/Slope
5. Area
6. Volume
7. Color saturation
8. Color hue (worst for quantitative data)

### Key Guidelines

1. **Use position when possible**
   - Dot plots > bar charts > pie charts
   - Scatter plots for correlation

2. **Colorblind-friendly palettes**
   - Use Wong or Okabe-Ito palettes
   - Test in grayscale
   - Avoid red-green combinations

3. **Perceptually uniform color scales**
   - Use viridis, plasma, cividis for sequential data
   - Avoid jet, rainbow (perceptually non-uniform)

4. **Appropriate visual channels**
   - Nominal data: color hue, shape
   - Ordinal data: color saturation, size
   - Quantitative data: position, length

### Implementation

#### Choose Effective Encodings

```python
# ❌ Pie chart - angle encoding (poor)
ax.pie(values, labels=labels)

# ✅ Bar chart - length encoding (better)
ax.barh(labels, values, color=WONG[1])
ax.set_xlabel('Value', fontsize=7)

# ✅ Dot plot - position encoding (best)
ax.scatter(values, range(len(labels)), s=40, color=WONG[1])
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=6)
ax.set_xlabel('Value', fontsize=7)
```

#### Colorblind-Friendly Palettes

```python
from color_palettes import WONG, OKABE_ITO

# ✅ Use tested colorblind-safe palettes
colors = WONG  # 8 colors, all distinguishable

# Plot with clear discrimination
for i, (color, label) in enumerate(zip(colors, labels)):
    ax.plot(x, y[i], color=color, linewidth=1.5, label=label)

# ❌ Avoid red-green
colors_bad = ['red', 'green', 'blue']  # Red-green not distinguishable

# ✅ Test by converting to grayscale
# Simulate grayscale perception
from PIL import Image
import numpy as np

def to_grayscale(fig):
    """Convert figure to grayscale to test accessibility"""
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    gray = Image.fromarray(img).convert('L')
    return gray
```

#### Perceptually Uniform Colormaps

```python
# ❌ Jet colormap - perceptually non-uniform
im = ax.imshow(data, cmap='jet')

# ✅ Viridis - perceptually uniform
im = ax.imshow(data, cmap='viridis')

# ✅ Other good options
# Sequential: viridis, plasma, inferno, magma, cividis
# Diverging: RdBu, PRGn (if colorblind-safe)

# Colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Value (units)', fontsize=7)
```

#### Multiple Visual Channels

```python
# Encode multiple dimensions effectively
# Size + Color + Shape for different attributes

scatter = ax.scatter(x, y,
                    s=size_vals * 10,        # Size for magnitude
                    c=category_vals,          # Color for category
                    cmap='tab10',
                    marker='o',               # Could vary by another attribute
                    alpha=0.6,
                    edgecolors='black',
                    linewidth=0.5)

ax.set_xlabel('X variable', fontsize=7)
ax.set_ylabel('Y variable', fontsize=7)

# Add legend for categories
handles, labels = scatter.legend_elements(prop='colors')
ax.legend(handles, category_names, title='Category',
         fontsize=5, frameon=False, loc='best')
```

---

## Combining All Three Principles

### Complete Example: Publication-Quality Time Series

```python
from figure_utilities import set_figure_defaults, save_figure
from color_palettes import WONG
import matplotlib.pyplot as plt
import numpy as np

set_figure_defaults(context="paper")

# Data
np.random.seed(42)
x = np.linspace(0, 10, 100)
y_mean = np.sin(x)
y_std = 0.2

# Simulate confidence interval
y_upper = y_mean + 1.96 * y_std
y_lower = y_mean - 1.96 * y_std

# Simulate raw data points
n_points = 50
x_data = np.random.uniform(0, 10, n_points)
y_data = np.sin(x_data) + np.random.normal(0, y_std, n_points)

fig, ax = plt.subplots(figsize=(3.5, 2.5), dpi=450,
                       constrained_layout=True)

# Tufte: Minimal non-data ink
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(False)

# Gelman: Show data and uncertainty
ax.scatter(x_data, y_data, s=15, alpha=0.4, color=WONG[1],
          label='Observed data', zorder=2)
ax.fill_between(x, y_lower, y_upper, alpha=0.2, color=WONG[1],
                label='95% CI', zorder=1)
ax.plot(x, y_mean, color=WONG[1], linewidth=1.5,
        label='Mean trend', zorder=3)

# Heer: Effective encodings
# - Position encoding (scatter plot)
# - Colorblind-safe palette (Wong)
# - Clear visual hierarchy (zorder)

# Clear labels with units (Gelman)
ax.set_xlabel('Time (s)', fontsize=7, labelpad=10)
ax.set_ylabel('Response (mV)', fontsize=7, labelpad=10)
ax.set_title('Signal over time (n=50)', fontsize=7, pad=15)

# Minimal legend (Tufte)
ax.legend(fontsize=5, frameon=False, loc='upper right')

save_figure('example_time_series')
plt.close()
```

---

## Common Mistakes and Fixes

### Mistake 1: Excessive Grid Lines

```python
# ❌ Violates Tufte
ax.grid(True, which='both', linestyle='-', linewidth=1)

# ✅ Minimal or no grid
ax.grid(False)
# Or subtle grid only if needed
ax.grid(True, which='major', linestyle=':', linewidth=0.5, alpha=0.3)
```

### Mistake 2: No Uncertainty

```python
# ❌ Violates Gelman
ax.plot(x, y_mean, color=WONG[1])

# ✅ Show uncertainty
ax.plot(x, y_mean, color=WONG[1])
ax.fill_between(x, y_lower, y_upper, alpha=0.2, color=WONG[1])
```

### Mistake 3: Poor Color Choices

```python
# ❌ Violates Heer
colors = ['red', 'green']  # Not colorblind-safe

# ✅ Colorblind-friendly
from color_palettes import WONG
colors = [WONG[1], WONG[2]]
```

### Mistake 4: Misleading Scales

```python
# ❌ Violates Gelman
ax.set_ylim(99, 101)  # Exaggerates small differences

# ✅ Show context
ax.set_ylim(0, 105)
```

### Mistake 5: Pie Charts

```python
# ❌ Violates Heer (angle is poorly perceived)
ax.pie(values, labels=labels)

# ✅ Bar chart (length is well-perceived)
ax.bar(labels, values, color=WONG[1])
```

---

## Quality Checklist Based on Principles

### Tufte Checklist
- [ ] Removed all non-data ink (no unnecessary grids, borders, backgrounds)
- [ ] Removed redundant labels
- [ ] No 3D effects, shadows, or gradients
- [ ] Used small multiples for comparisons
- [ ] High data-ink ratio

### Gelman Checklist
- [ ] Shows actual data points or distributions
- [ ] Displays uncertainty (error bars, confidence intervals)
- [ ] Axes not misleadingly truncated
- [ ] Consistent scales for comparisons
- [ ] All axes labeled with units
- [ ] Appropriate aspect ratio

### Heer Checklist
- [ ] Used effective visual encodings (position > length > angle)
- [ ] Colorblind-friendly palette
- [ ] Perceptually uniform colormaps for sequential data
- [ ] Works in grayscale
- [ ] Sufficient contrast
- [ ] Clear visual hierarchy

---

## Further Reading

- **Tufte, E. R.** (2001). *The Visual Display of Quantitative Information* (2nd ed.). Graphics Press.
- **Gelman, A., Pasarica, C., & Dodhia, R.** (2002). "Let's practice what we preach: Turning tables into graphs." *The American Statistician*, 56(2), 121-130.
- **Heer, J., & Bostock, M.** (2010). "Crowdsourcing graphical perception: Using mechanical turk to assess visualization design." *Proceedings of the SIGCHI Conference on Human Factors in Computing Systems*, 203-212.
- **Wong, B.** (2011). "Color blindness." *Nature Methods*, 8(6), 441.

---

## Summary

| Principle | Focus | Key Technique |
|-----------|-------|---------------|
| **Tufte** | Data-ink ratio | Remove chartjunk, use small multiples |
| **Gelman** | Statistical clarity | Show uncertainty, avoid misleading scales |
| **Heer** | Perceptual effectiveness | Use position encoding, colorblind-safe palettes |

Apply all three simultaneously for optimal scientific figures.
