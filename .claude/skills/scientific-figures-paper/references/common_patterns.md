# Common Figure Patterns

Complete code examples for common scientific figure types.

---

## Time Series Plot

### Basic Time Series with Error Band

```python
from figure_utilities import set_figure_defaults, save_figure
from color_palettes import WONG
import matplotlib.pyplot as plt
import numpy as np

set_figure_defaults(context="paper")

# Generate data
time = np.linspace(0, 10, 100)
signal = np.sin(time)
error = 0.2 * np.ones_like(time)

# Create figure
fig, ax = plt.subplots(figsize=(3.5, 2.5), dpi=450,
                       constrained_layout=True)

# Plot with error band
ax.plot(time, signal, linewidth=1.5, color=WONG[1], label='Signal')
ax.fill_between(time, signal - error, signal + error,
                alpha=0.3, color=WONG[1], label='±1σ')

# Configure
ax.set_xlabel('Time (s)', fontsize=7, labelpad=10)
ax.set_ylabel('Signal (mV)', fontsize=7, labelpad=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(fontsize=5, frameon=False, loc='upper right')

save_figure('timeseries')
plt.close()
```

### Multiple Time Series

```python
fig, ax = plt.subplots(figsize=(3.5, 2.5), dpi=450,
                       constrained_layout=True)

# Multiple conditions
for i, condition in enumerate(['Control', 'Treatment A', 'Treatment B']):
    ax.plot(time, signals[i], linewidth=1.5, 
           color=WONG[i+1], label=condition)

ax.set_xlabel('Time (s)', fontsize=7, labelpad=10)
ax.set_ylabel('Response (mV)', fontsize=7, labelpad=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Legend outside to save space
ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1),
          fontsize=5, frameon=False)

save_figure('timeseries_multi')
plt.close()
```

---

## Multi-Panel Comparison

### Using subplot_mosaic

```python
from figure_utilities import set_figure_defaults, save_figure
from label_panels import label_panels
from color_palettes import WONG
import matplotlib.pyplot as plt

set_figure_defaults(context="paper")

# Define layout
layout = """
    ABC
    """

fig, axes = plt.subplot_mosaic(
    layout,
    figsize=(7, 2.5),
    dpi=450,
    constrained_layout=True
)

# Plot in each panel
conditions = ['Control', 'Treatment A', 'Treatment B']
for ax_name, condition, data in zip(['A', 'B', 'C'], conditions, datasets):
    axes[ax_name].plot(x, data, linewidth=1.5, color=WONG[1])
    axes[ax_name].set_title(condition, fontsize=7, pad=10)
    axes[ax_name].set_xlabel('X', fontsize=7, labelpad=8)
    axes[ax_name].spines['top'].set_visible(False)
    axes[ax_name].spines['right'].set_visible(False)

# Y-label only on leftmost
axes['A'].set_ylabel('Response', fontsize=7, labelpad=10)

# Label panels
label_panels(list(axes.values()), xpos=-0.15, ypos=1.08)

save_figure('comparison')
plt.close()
```

### Small Multiples (Tufte)

```python
layout = """
    ABC
    DEF
    """

fig, axes = plt.subplot_mosaic(
    layout,
    figsize=(7, 4.5),
    dpi=450,
    constrained_layout=True,
    sharex=True,
    sharey=True  # Same scale for easy comparison
)

# Plot each condition
for ax_name, condition in zip(['A', 'B', 'C', 'D', 'E', 'F'], conditions):
    axes[ax_name].plot(x, data[condition], linewidth=1.5, color=WONG[1])
    axes[ax_name].set_title(condition, fontsize=7, pad=8)
    axes[ax_name].spines['top'].set_visible(False)
    axes[ax_name].spines['right'].set_visible(False)

# Labels only on edges
for ax_name in ['D', 'E', 'F']:
    axes[ax_name].set_xlabel('Time (s)', fontsize=7, labelpad=8)
for ax_name in ['A', 'D']:
    axes[ax_name].set_ylabel('Response', fontsize=7, labelpad=10)

label_panels(list(axes.values()))
save_figure('small_multiples')
plt.close()
```

---

## Scatter Plot with Regression

### Basic Scatter with Fit

```python
from scipy import stats

set_figure_defaults(context="paper")

fig, ax = plt.subplots(figsize=(3.5, 3.5), dpi=450,
                       constrained_layout=True)

# Data
x = np.random.randn(100)
y = 2 * x + 1 + np.random.randn(100) * 0.5

# Regression
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
x_fit = np.linspace(x.min(), x.max(), 100)
y_fit = slope * x_fit + intercept

# Plot
ax.scatter(x, y, s=20, alpha=0.6, color=WONG[1], 
          edgecolors='none', label='Data')
ax.plot(x_fit, y_fit, linewidth=2, color=WONG[6],
        linestyle='--', label=f'Fit (R²={r_value**2:.3f})')

# Configure
ax.set_xlabel('Predictor', fontsize=7, labelpad=10)
ax.set_ylabel('Response', fontsize=7, labelpad=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(fontsize=5, frameon=False, loc='upper left')

save_figure('scatter_regression')
plt.close()
```

### Scatter with Confidence Interval

```python
from scipy.stats import t

# Calculate confidence interval
residuals = y - (slope * x + intercept)
residual_std = np.sqrt(np.sum(residuals**2) / (len(x) - 2))
x_mean = np.mean(x)
sxx = np.sum((x - x_mean)**2)

# Standard error of prediction
se_pred = residual_std * np.sqrt(1 + 1/len(x) + 
                                 (x_fit - x_mean)**2 / sxx)

# 95% confidence interval
t_val = t.ppf(0.975, len(x) - 2)
y_upper = y_fit + t_val * se_pred
y_lower = y_fit - t_val * se_pred

# Plot
fig, ax = plt.subplots(figsize=(3.5, 3.5), dpi=450,
                       constrained_layout=True)

ax.scatter(x, y, s=20, alpha=0.6, color=WONG[1], 
          edgecolors='none', label='Data')
ax.plot(x_fit, y_fit, linewidth=2, color=WONG[6], label='Fit')
ax.fill_between(x_fit, y_lower, y_upper, alpha=0.2, 
                color=WONG[6], label='95% CI')

ax.set_xlabel('Predictor', fontsize=7, labelpad=10)
ax.set_ylabel('Response', fontsize=7, labelpad=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(fontsize=5, frameon=False, loc='upper left')

save_figure('scatter_with_ci')
plt.close()
```

---

## Bar Plots

### Simple Bar Chart with Error Bars

```python
set_figure_defaults(context="paper")

fig, ax = plt.subplots(figsize=(3.5, 2.5), dpi=450,
                       constrained_layout=True)

categories = ['A', 'B', 'C', 'D']
means = [25, 32, 28, 35]
errors = [3, 2.5, 4, 3.5]

x_pos = np.arange(len(categories))

ax.bar(x_pos, means, yerr=errors, 
       capsize=3, capthick=1,
       color=WONG[1], alpha=0.8,
       error_kw={'linewidth': 1, 'elinewidth': 1})

ax.set_xticks(x_pos)
ax.set_xticklabels(categories, fontsize=6)
ax.set_ylabel('Response (units)', fontsize=7, labelpad=10)
ax.set_xlabel('Condition', fontsize=7, labelpad=10)

# Clean appearance
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim(0, max(means) + max(errors) + 5)

save_figure('barplot')
plt.close()
```

### Grouped Bar Chart

```python
fig, ax = plt.subplots(figsize=(4, 2.5), dpi=450,
                       constrained_layout=True)

categories = ['Group 1', 'Group 2', 'Group 3']
condition_a = [25, 32, 28]
condition_b = [30, 28, 35]

x = np.arange(len(categories))
width = 0.35

bars1 = ax.bar(x - width/2, condition_a, width, 
              color=WONG[1], alpha=0.8, label='Condition A')
bars2 = ax.bar(x + width/2, condition_b, width,
              color=WONG[2], alpha=0.8, label='Condition B')

ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=6)
ax.set_ylabel('Response (units)', fontsize=7, labelpad=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.legend(fontsize=5, frameon=False, loc='upper left')

save_figure('grouped_barplot')
plt.close()
```

---

## Heatmaps

### Basic Heatmap with Colorbar

```python
set_figure_defaults(context="paper")

fig, ax = plt.subplots(figsize=(4, 3), dpi=450,
                       constrained_layout=True)

# Data
data_matrix = np.random.randn(10, 8)

# Heatmap
im = ax.imshow(data_matrix, aspect='auto', 
              cmap='viridis', interpolation='nearest')

# Colorbar
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Value (units)', fontsize=7)
cbar.ax.tick_params(labelsize=5)

# Labels
ax.set_xlabel('Column', fontsize=7, labelpad=10)
ax.set_ylabel('Row', fontsize=7, labelpad=10)
ax.set_xticks(range(8))
ax.set_yticks(range(10))

save_figure('heatmap')
plt.close()
```

### Heatmap with Colorbar in subplot_mosaic

```python
layout = """
    AAC
    BBC
    """

fig, axes = plt.subplot_mosaic(
    layout,
    figsize=(5, 4),
    width_ratios=[2, 2, 0.2],  # C is narrow colorbar
    dpi=450,
    constrained_layout=True
)

# Two heatmaps
im1 = axes['A'].imshow(data1, aspect='auto', cmap='viridis')
axes['A'].set_title('Condition 1', fontsize=7)
axes['A'].set_ylabel('Row', fontsize=7, labelpad=10)

im2 = axes['B'].imshow(data2, aspect='auto', cmap='viridis')
axes['B'].set_title('Condition 2', fontsize=7)
axes['B'].set_xlabel('Column', fontsize=7, labelpad=10)
axes['B'].set_ylabel('Row', fontsize=7, labelpad=10)

# Shared colorbar in C
axes['C'].axis('off')
cbar = plt.colorbar(im1, ax=axes['C'], fraction=0.9)
cbar.set_label('Value', fontsize=7)
cbar.ax.tick_params(labelsize=5)

save_figure('heatmap_mosaic')
plt.close()
```

---

## Box Plots and Violin Plots

### Box Plot with Individual Points

```python
set_figure_defaults(context="paper")

fig, ax = plt.subplots(figsize=(3.5, 2.5), dpi=450,
                       constrained_layout=True)

# Data
data = [np.random.normal(0, std, 100) for std in range(1, 4)]

# Box plot
bp = ax.boxplot(data, positions=[1, 2, 3],
               widths=0.6,
               patch_artist=True,
               boxprops=dict(facecolor=WONG[1], alpha=0.6),
               medianprops=dict(color='black', linewidth=1.5),
               whiskerprops=dict(color='black', linewidth=1),
               capprops=dict(color='black', linewidth=1),
               flierprops=dict(marker='o', markersize=3, 
                             alpha=0.5, color=WONG[1]))

ax.set_xticklabels(['Group A', 'Group B', 'Group C'], fontsize=6)
ax.set_ylabel('Value (units)', fontsize=7, labelpad=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

save_figure('boxplot')
plt.close()
```

### Violin Plot

```python
fig, ax = plt.subplots(figsize=(3.5, 2.5), dpi=450,
                       constrained_layout=True)

# Violin plot
parts = ax.violinplot(data, positions=[1, 2, 3],
                     widths=0.7,
                     showmeans=True,
                     showmedians=True)

# Style the violin
for pc in parts['bodies']:
    pc.set_facecolor(WONG[1])
    pc.set_alpha(0.6)
    pc.set_edgecolor('black')
    pc.set_linewidth(1)

ax.set_xticks([1, 2, 3])
ax.set_xticklabels(['Group A', 'Group B', 'Group C'], fontsize=6)
ax.set_ylabel('Value (units)', fontsize=7, labelpad=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

save_figure('violinplot')
plt.close()
```

---

## Complex Multi-Panel Layouts

### Main Plot with Marginal Distributions

```python
layout = """
    .A
    BC
    """

fig, axes = plt.subplot_mosaic(
    layout,
    figsize=(4, 4),
    width_ratios=[1, 3],
    height_ratios=[1, 3],
    dpi=450,
    constrained_layout=True
)

# Main scatter
axes['C'].scatter(x, y, s=20, alpha=0.6, color=WONG[1])
axes['C'].set_xlabel('X', fontsize=7, labelpad=10)
axes['C'].set_ylabel('Y', fontsize=7, labelpad=10)

# X marginal (top)
axes['A'].hist(x, bins=30, color=WONG[1], alpha=0.6, edgecolor='none')
axes['A'].set_xlim(axes['C'].get_xlim())
axes['A'].axis('off')

# Y marginal (left)
axes['B'].hist(y, bins=30, orientation='horizontal',
              color=WONG[1], alpha=0.6, edgecolor='none')
axes['B'].set_ylim(axes['C'].get_ylim())
axes['B'].axis('off')

save_figure('marginal_distributions')
plt.close()
```

### Complex Grid with Shared Legend

```python
layout = """
    AABBL
    CCDEL
    FFGHL
    """

fig, axes = plt.subplot_mosaic(
    layout,
    figsize=(7, 5.5),
    width_ratios=[1, 1, 1, 0.15, 0.4],
    height_ratios=[1, 1, 1],
    dpi=450,
    constrained_layout={'h_pad': 0.02, 'w_pad': 0.02}
)

# Plot in each panel
panel_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
for i, label in enumerate(panel_labels):
    axes[label].plot(x, data[i], color=WONG[i % 8], 
                    linewidth=1.5, label=f'Series {label}')
    axes[label].set_xlabel('X', fontsize=7, labelpad=8)
    axes[label].set_ylabel('Y', fontsize=7, labelpad=8)
    axes[label].spines['top'].set_visible(False)
    axes[label].spines['right'].set_visible(False)

# Shared legend in L
axes['L'].axis('off')
all_handles = []
all_labels = []
for label in panel_labels:
    h, l = axes[label].get_legend_handles_labels()
    all_handles.extend(h)
    all_labels.extend(l)

axes['L'].legend(all_handles, all_labels,
                loc='center left',
                fontsize=5,
                frameon=False)

# Label panels
from label_panels import label_panels
label_panels([axes[l] for l in panel_labels],
            labels=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'],
            xpos=-0.15, ypos=1.05)

save_figure('complex_grid')
plt.close()
```

### Asymmetric Layout

```python
layout = """
    AAB
    CCC
    DDE
    """

fig, axes = plt.subplot_mosaic(
    layout,
    figsize=(7, 5),
    width_ratios=[2, 2, 1.5],
    height_ratios=[1, 1.2, 1],
    dpi=450,
    constrained_layout=True
)

# A: Time series (spans 2 cols)
axes['A'].plot(time, signal, color=WONG[1], linewidth=1.5)
axes['A'].set_ylabel('Signal', fontsize=7, labelpad=10)

# B: Bar chart (1 col)
axes['B'].bar(range(5), values, color=WONG[2], alpha=0.8)
axes['B'].set_ylabel('Count', fontsize=7, labelpad=10)

# C: Heatmap (spans all cols, taller)
im = axes['C'].imshow(matrix, aspect='auto', cmap='viridis')
axes['C'].set_ylabel('Feature', fontsize=7, labelpad=10)

# D: Scatter (spans 2 cols)
axes['D'].scatter(x, y, s=20, color=WONG[3], alpha=0.6)
axes['D'].set_xlabel('X', fontsize=7, labelpad=10)
axes['D'].set_ylabel('Y', fontsize=7, labelpad=10)

# E: Histogram (1 col)
axes['E'].hist(data, bins=20, color=WONG[4], alpha=0.6, edgecolor='none')
axes['E'].set_xlabel('Value', fontsize=7, labelpad=10)

# Clean spines
for ax in axes.values():
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

label_panels(list(axes.values()))
save_figure('asymmetric_layout')
plt.close()
```

---

## Distribution Plots

### Histogram with KDE Overlay

```python
set_figure_defaults(context="paper")

fig, ax = plt.subplots(figsize=(3.5, 2.5), dpi=450,
                       constrained_layout=True)

# Histogram
n, bins, patches = ax.hist(data, bins=30, density=True,
                           color=WONG[1], alpha=0.6, 
                           edgecolor='none', label='Data')

# KDE overlay
from scipy.stats import gaussian_kde
kde = gaussian_kde(data)
x_range = np.linspace(data.min(), data.max(), 100)
ax.plot(x_range, kde(x_range), color=WONG[6], 
       linewidth=2, label='KDE')

ax.set_xlabel('Value (units)', fontsize=7, labelpad=10)
ax.set_ylabel('Density', fontsize=7, labelpad=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(fontsize=5, frameon=False, loc='upper right')

save_figure('histogram_kde')
plt.close()
```

### Overlapping Distributions

```python
fig, ax = plt.subplots(figsize=(3.5, 2.5), dpi=450,
                       constrained_layout=True)

# Multiple distributions
for i, (data, label) in enumerate(zip(datasets, labels)):
    ax.hist(data, bins=30, density=True, alpha=0.5,
           color=WONG[i+1], edgecolor='none', label=label)

ax.set_xlabel('Value (units)', fontsize=7, labelpad=10)
ax.set_ylabel('Density', fontsize=7, labelpad=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(fontsize=5, frameon=False, loc='upper right')

save_figure('overlapping_distributions')
plt.close()
```

---

## Summary Table: When to Use Each Pattern

| Figure Type | Use Case | Layout Recommendation |
|-------------|----------|----------------------|
| Time series | Temporal data | Single panel or horizontal layout |
| Scatter | Correlation | Square aspect ratio |
| Bar chart | Categorical comparison | Horizontal for many categories |
| Heatmap | Matrix data | Use subplot_mosaic for colorbar |
| Box/Violin | Distribution comparison | Side-by-side layout |
| Multi-panel | Multiple related plots | subplot_mosaic with shared legend |
| Small multiples | Repeated structure | Grid with shared scales |

---

## Template: Custom Pattern

```python
"""
Template for creating custom figure patterns
"""
from figure_utilities import set_figure_defaults, save_figure
from label_panels import label_panels
from color_palettes import WONG
import matplotlib.pyplot as plt

set_figure_defaults(context="paper")

# 1. Design layout
layout = """
    AB
    CD
    """

# 2. Create figure
fig, axes = plt.subplot_mosaic(
    layout,
    figsize=(7, 5),
    width_ratios=[1, 1],
    height_ratios=[1, 1],
    dpi=450,
    constrained_layout=True
)

# 3. Plot in each panel
# [Your plotting code here]

# 4. Configure aesthetics
for ax in axes.values():
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# 5. Label panels
label_panels(list(axes.values()))

# 6. Save
save_figure('custom_pattern')
plt.close()
```
