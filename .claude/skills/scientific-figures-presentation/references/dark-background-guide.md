# Dark Background Guide for Presentations

## When to Use Dark Backgrounds

**Ideal for:**
- Presentations in dark auditoriums
- Evening or nighttime presentations
- Reducing eye strain for audience
- Modern, high-impact visual style
- When projector has high contrast capability

**Avoid when:**
- Room is bright or has windows
- Projector is low quality or old
- Printing handouts (wastes ink)
- Mixed with light background slides (inconsistent)

## Setting Up Dark Background

### Basic Setup

```python
from scripts.figure_utilities import set_figure_defaults, save_figure
from scripts.color_palettes import WONG
import matplotlib.pyplot as plt

# Load dark background style
set_figure_defaults(context="presentation_dark")

# Create figure with dark background
fig, ax = plt.subplots(figsize=(10, 5.625), dpi=150,
                       facecolor='#1a1a1a',  # Dark gray background
                       constrained_layout=True)

# Set axes background to match
ax.set_facecolor('#1a1a1a')
```

### Why #1a1a1a (not pure black)?

- Pure black (#000000) can be too harsh
- Dark gray (#1a1a1a) is easier on eyes
- Better contrast for colored elements
- More professional appearance
- Reduces projector "glow"

## Color Selection for Dark Backgrounds

### Best Colors

**Bright, saturated colors that pop on dark:**

```python
# Excellent on dark
bright_orange = WONG[1]  # '#E69F00' - Orange
bright_blue = WONG[2]    # '#56B4E9' - Sky Blue
bright_yellow = WONG[4]  # '#F0E442' - Yellow
bright_cyan = WONG[2]    # '#56B4E9' - Sky Blue
white = '#FFFFFF'        # Pure white
```

**Usage:**
```python
# Data lines
ax.plot(x, y, linewidth=4, color=WONG[1])  # Bright orange

# Emphasis
ax.plot(x, y_key, linewidth=5, color=WONG[4])  # Bright yellow

# Reference lines
ax.axhline(0, linewidth=2, color='white', alpha=0.5)
```

### Colors to Avoid

**Don't use on dark backgrounds:**

❌ Black or dark gray (invisible)
❌ Purple (too dark, loses visibility)
❌ Dark blue (insufficient contrast)
❌ Dark red (looks muddy)
❌ Pastels (too subtle)

## Text Configuration

### All Text Must Be White

```python
# Axis labels - WHITE
ax.set_xlabel('Time (s)', fontsize=24, color='white', 
              fontweight='bold', labelpad=15)
ax.set_ylabel('Signal', fontsize=24, color='white',
              fontweight='bold', labelpad=15)

# Title - WHITE
ax.set_title('Key Finding', fontsize=32, color='white',
             fontweight='bold', pad=20)

# Tick labels - WHITE
ax.tick_params(colors='white', labelsize=18, width=2, length=8)
```

## Axes Configuration

### White Axes on Dark Background

```python
# Set all spine colors to white
for spine in ax.spines.values():
    spine.set_edgecolor('white')
    spine.set_linewidth(2)

# Remove top and right as usual
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# White ticks
ax.tick_params(axis='both', colors='white', 
               labelsize=18, width=2, length=8)
```

## Legend for Dark Backgrounds

### White Legend Text

```python
# Create legend
legend = ax.legend(fontsize=20, frameon=False, loc='upper right')

# Make legend text white
for text in legend.get_texts():
    text.set_color('white')

# Optional: Add semi-transparent background
legend = ax.legend(fontsize=20, frameon=True,
                  loc='upper right',
                  facecolor='#1a1a1a', 
                  edgecolor='white',
                  framealpha=0.8)
for text in legend.get_texts():
    text.set_color('white')
```

## Complete Dark Theme Example

### Standard Line Plot

```python
from scripts.figure_utilities import set_figure_defaults, save_figure
from scripts.color_palettes import WONG
import matplotlib.pyplot as plt
import numpy as np

# Data
x = np.linspace(0, 10, 50)
y1 = np.sin(x)
y2 = 1.5 * np.sin(x)

# Dark theme setup
set_figure_defaults(context="presentation_dark")

fig, ax = plt.subplots(figsize=(10, 5.625), dpi=150,
                       facecolor='#1a1a1a',
                       constrained_layout=True)
ax.set_facecolor('#1a1a1a')

# Plot with bright colors
ax.plot(x, y1, linewidth=4, color=WONG[1], 
        label='Baseline')  # Bright orange
ax.plot(x, y2, linewidth=4, color=WONG[4], 
        label='Treatment')  # Bright yellow

# WHITE text
ax.set_xlabel('Time (s)', fontsize=24, color='white', 
              fontweight='bold', labelpad=15)
ax.set_ylabel('Signal', fontsize=24, color='white',
              fontweight='bold', labelpad=15)
ax.set_title('Treatment Enhances Response', fontsize=32,
             color='white', fontweight='bold', pad=20)

# WHITE axes
for spine in ax.spines.values():
    spine.set_edgecolor('white')
    spine.set_linewidth(2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# WHITE ticks
ax.tick_params(colors='white', labelsize=18, width=2, length=8)

# WHITE legend
legend = ax.legend(fontsize=22, frameon=False, loc='upper left')
for text in legend.get_texts():
    text.set_color('white')

# Save with dark background
save_figure('dark_line_plot', facecolor='#1a1a1a')
plt.close()
```

### Bar Chart on Dark

```python
set_figure_defaults(context="presentation_dark")

fig, ax = plt.subplots(figsize=(10, 5.625), dpi=150,
                       facecolor='#1a1a1a',
                       constrained_layout=True)
ax.set_facecolor('#1a1a1a')

# Data
categories = ['Control', 'Treatment']
values = [10, 30]
x_pos = np.arange(len(categories))

# Bright colored bars
bars = ax.bar(x_pos, values,
              color=[WONG[1], WONG[4]],  # Orange and Yellow
              alpha=0.9, width=0.6,
              edgecolor='white', linewidth=2)

# WHITE labels on bars
for bar, val in zip(bars, values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{val}', ha='center', va='bottom',
            fontsize=24, fontweight='bold', color='white')

# WHITE axis configuration
ax.set_xticks(x_pos)
ax.set_xticklabels(categories, fontsize=24, 
                   fontweight='bold', color='white')
ax.set_ylabel('Response', fontsize=24, color='white',
              fontweight='bold', labelpad=15)
ax.set_title('3× Increase', fontsize=32, color='white',
             fontweight='bold', pad=20)

# WHITE spines
for spine in ax.spines.values():
    spine.set_edgecolor('white')
    spine.set_linewidth(2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# WHITE ticks
ax.tick_params(colors='white', labelsize=18, width=2, length=8)

save_figure('dark_bar_chart', facecolor='#1a1a1a')
plt.close()
```

### Scatter Plot on Dark

```python
set_figure_defaults(context="presentation_dark")

fig, ax = plt.subplots(figsize=(10, 5.625), dpi=150,
                       facecolor='#1a1a1a',
                       constrained_layout=True)
ax.set_facecolor('#1a1a1a')

# Data
np.random.seed(42)
x = np.random.randn(50)
y = 2 * x + np.random.randn(50) * 0.5

# Bright scatter points with WHITE edges
ax.scatter(x, y, s=120, alpha=0.7, 
           color=WONG[1],  # Bright orange
           edgecolors='white', linewidths=1.5)

# Fit line in contrasting bright color
coeffs = np.polyfit(x, y, 1)
x_fit = np.linspace(x.min(), x.max(), 100)
y_fit = np.polyval(coeffs, x_fit)
ax.plot(x_fit, y_fit, linewidth=4, color=WONG[4],  # Yellow
        linestyle='--')

# WHITE configuration
ax.set_xlabel('X', fontsize=24, color='white', 
              fontweight='bold', labelpad=15)
ax.set_ylabel('Y', fontsize=24, color='white',
              fontweight='bold', labelpad=15)
ax.set_title('Strong Correlation (r=0.89)', fontsize=32,
             color='white', fontweight='bold', pad=20)

for spine in ax.spines.values():
    spine.set_edgecolor('white')
    spine.set_linewidth(2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.tick_params(colors='white', labelsize=18, width=2, length=8)

save_figure('dark_scatter', facecolor='#1a1a1a')
plt.close()
```

## Annotations on Dark Backgrounds

### White Annotations

```python
# Annotation with white text and arrow
ax.annotate('Key point',
            xy=(x_point, y_point),
            xytext=(20, 30), textcoords='offset points',
            fontsize=24, fontweight='bold',
            color='white',  # White text
            ha='center',
            arrowprops=dict(arrowstyle='->', lw=3, 
                          color='white'),  # White arrow
            bbox=dict(boxstyle='round,pad=0.5',
                     facecolor=WONG[4],  # Bright yellow box
                     edgecolor='white',
                     linewidth=2))
```

### Reference Lines on Dark

```python
# Horizontal reference line
ax.axhline(y=threshold, linewidth=3, 
           color='white', alpha=0.7, linestyle='--')

# Vertical reference line
ax.axvline(x=cutoff, linewidth=3,
           color='white', alpha=0.7, linestyle='--')

# Shaded region with bright color
ax.axvspan(x1, x2, alpha=0.3, color=WONG[4])  # Yellow highlight
```

## Grid on Dark Background

**If you must use a grid:**

```python
# Subtle WHITE grid
ax.grid(True, alpha=0.2, color='white', linewidth=1)

# Or specify which grid lines
ax.grid(True, alpha=0.2, color='white', linewidth=1,
        which='major', axis='both')
```

**Better**: Don't use grid at all - cleaner appearance

## Colorbars on Dark

```python
# Create colorbar
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# WHITE colorbar labels
cbar.ax.tick_params(labelsize=18, colors='white', 
                    width=2, length=6)

# WHITE colorbar label
cbar.set_label('Value', fontsize=20, fontweight='bold',
               color='white', rotation=270, labelpad=25)

# WHITE colorbar outline
cbar.outline.set_edgecolor('white')
cbar.outline.set_linewidth(2)
```

## Saving Dark Figures

### Critical: Specify Background

```python
# MUST specify facecolor when saving
save_figure('my_dark_figure', facecolor='#1a1a1a')

# Without facecolor, background may be white in saved file!
```

### For PowerPoint/Keynote

**PNG format recommended:**
```python
# PNG preserves dark background
save_figure('figure_name', facecolor='#1a1a1a')
# Creates: figure_name.png with dark background
```

**Slide setup:**
- Use dark slide background (#1a1a1a)
- Insert PNG figure
- Figure blends seamlessly with slide

## Testing Dark Figures

### Checklist

- [ ] Figure background is #1a1a1a (not black)
- [ ] All text is white
- [ ] All axes/spines are white
- [ ] All tick labels are white
- [ ] Legend text is white
- [ ] Colors are bright and saturated
- [ ] Annotations use white text/arrows
- [ ] Saved with facecolor='#1a1a1a'
- [ ] Tested on actual projector
- [ ] Tested in dark room

### Common Issues

**Problem**: Text appears gray, not white
**Fix**: Explicitly set color='white' for all text

**Problem**: Background is white in saved file
**Fix**: Add facecolor='#1a1a1a' to save_figure()

**Problem**: Colors look dim
**Fix**: Use bright colors (WONG[1], WONG[4])

**Problem**: Hard to see in bright room
**Fix**: Don't use dark theme for bright rooms

## Style File

The dark theme is configured in `scripts/presentation_dark.mplstyle`:

**Key settings:**
- Background: #1a1a1a
- Text: white
- Axes: white
- Color cycle: Bright colors from Wong palette
- Font size: Large (24pt labels, 32pt titles)
- Line width: Bold (3-4pt)

**To use:**
```python
set_figure_defaults(context="presentation_dark")
```

This automatically applies all dark theme settings.

## Mixing Light and Dark

**DON'T mix in same presentation!**

Pick ONE style and use consistently:
- All light background, OR
- All dark background

Mixing creates jarring transitions and looks unprofessional.

## Recommendations

**Use dark backgrounds when:**
- ✅ Presenting in dark auditorium
- ✅ Evening/nighttime presentation
- ✅ High-contrast projector available
- ✅ Modern, impactful visual style desired

**Use light backgrounds when:**
- ✅ Bright room or windows
- ✅ Daytime presentation
- ✅ Unknown projector quality
- ✅ Printing handouts
- ✅ Traditional academic setting

When in doubt, use light background (safer).

## Summary

**Dark theme essentials:**
1. Background: #1a1a1a (dark gray)
2. Text: white everywhere
3. Axes: white
4. Colors: bright and saturated
5. Save with facecolor specified
6. Test on projector
7. Use consistently (no mixing)

**The main rule:**
Everything that was black on light backgrounds must be white on dark backgrounds.
