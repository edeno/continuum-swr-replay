# Presentation Figure Checklist

Use this checklist before adding figures to slides to ensure they meet presentation standards.

## Pre-Creation Checklist

**Before writing any code:**

- [ ] Read `references/spatial-layout.md` (prevents overlaps)
- [ ] Know the ONE key message this figure will show
- [ ] Decided: light or dark background?
- [ ] Planned: which 1-2 comparisons to show?
- [ ] Confirmed: figure will fit 16:9 slide (10" × 5.625")

## Readability Standards

### Typography
- [ ] All fonts ≥16pt minimum
- [ ] Axis labels ≥20pt (24pt preferred)
- [ ] Title ≥32pt
- [ ] Tick labels ≥16pt (18pt preferred)
- [ ] Legend ≥16pt (20pt preferred)
- [ ] Font family is sans-serif (Arial, Helvetica)

### Visual Elements  
- [ ] Line width ≥3pt (4pt preferred)
- [ ] Markers ≥8pt size (10pt preferred)
- [ ] Error bar caps ≥8pt
- [ ] Reference lines ≥2pt width
- [ ] All elements visible from 30 feet

### Colors
- [ ] High contrast (dark on light OR light on dark)
- [ ] Colors are bold and saturated (no pastels)
- [ ] Used colorblind-friendly palette (Wong/Okabe-Ito)
- [ ] Maximum 3-4 colors per figure
- [ ] Colors distinguishable on poor projector

## Clarity Standards

### Message
- [ ] Main point obvious in 3 seconds
- [ ] Title communicates finding (not generic "Results")
- [ ] Shows only 1-2 key comparisons
- [ ] All non-essential elements removed

### Layout
- [ ] Used constrained_layout=True
- [ ] No overlapping text or elements
- [ ] Legend large and doesn't cover data
- [ ] Generous whitespace (not cramped)
- [ ] Axes labeled with units

### Simplification
- [ ] Grid removed (or very subtle if necessary)
- [ ] Top and right spines removed
- [ ] Fewer than 5 tick marks per axis
- [ ] Tick labels are round numbers
- [ ] Could this be simpler? (Seriously ask!)

## Technical Standards

### Dimensions
- [ ] Figure size matches slide: 10" × 5.625" for 16:9
- [ ] DPI set to 150 (screen resolution)
- [ ] Margins left for slide borders (0.5-1")
- [ ] constrained_layout=True used

### Export
- [ ] Format is PNG (best for slides)
- [ ] Saved with save_figure() function
- [ ] File size reasonable (<1MB)
- [ ] For dark theme: facecolor='#1a1a1a' specified

## Visual Impact

### Emphasis
- [ ] Key findings emphasized (color, size, annotation)
- [ ] Sufficient contrast for all elements
- [ ] Annotations use large, bold text (≥20pt)
- [ ] Arrows are thick (≥2pt) and obvious

### Polish
- [ ] Clean, professional appearance
- [ ] Consistent style if multiple figures
- [ ] No clipart or decorative elements
- [ ] Works well projected (test if possible)

## Dark Background Specific

**If using dark background (#1a1a1a):**

- [ ] Figure facecolor set to '#1a1a1a'
- [ ] Axes facecolor set to '#1a1a1a'
- [ ] ALL text is white
- [ ] ALL axes/spines are white
- [ ] Tick labels are white
- [ ] Legend text is white
- [ ] Colors are bright (WONG[1], WONG[4])
- [ ] Saved with facecolor='#1a1a1a'
- [ ] Tested in dark room

## Integration Checklist

### With Slides
- [ ] Figure exported as PNG
- [ ] Slide background matches figure background
- [ ] Figure doesn't need resizing in slides
- [ ] All text readable in slide context
- [ ] Tested in presentation software

### Progressive Disclosure
- [ ] If complex: split into multiple slides
- [ ] If building: created multiple versions (v1, v2, v3)
- [ ] Each version adds one element
- [ ] Versions consistent in style

## Final Validation

### Distance Test
- [ ] Readable from across the room (20+ feet)?
- [ ] Main message obvious from distance?
- [ ] All text legible without zooming?

### 3-Second Test
- [ ] Can someone understand main point in 3 seconds?
- [ ] Is visual hierarchy clear?
- [ ] Does title communicate finding?

### Projector Test (if possible)
- [ ] Tested on actual projector
- [ ] Colors still distinguishable?
- [ ] Text still readable?
- [ ] Contrast sufficient?

### Simplification Test
- [ ] Could anything be removed?
- [ ] Is every element necessary?
- [ ] Could this be split into multiple slides?
- [ ] Is the message as clear as possible?

## Common Issues to Check

### Spatial/Overlap Issues
- [ ] Legend doesn't cover data
- [ ] Axis labels don't overlap tick labels
- [ ] Tick labels don't overlap each other
- [ ] Title doesn't overlap axes
- [ ] Annotations don't overlap data

### Typography Issues
- [ ] No fonts below 16pt
- [ ] Bold used for emphasis (not underline)
- [ ] Consistent font family throughout
- [ ] Adequate label padding (labelpad≥10)

### Visual Issues
- [ ] Lines not too thin (≥3pt)
- [ ] Markers not too small (≥8pt)
- [ ] Colors not too subtle or similar
- [ ] No gradients or transparency reducing contrast

### Layout Issues
- [ ] Not overfilled or cramped
- [ ] Sufficient whitespace
- [ ] Elements properly aligned
- [ ] Consistent spacing

## Code Checklist

**Standard setup used:**
```python
- [ ] from scripts.figure_utilities import set_figure_defaults, save_figure
- [ ] from scripts.color_palettes import WONG
- [ ] set_figure_defaults(context="presentation") called
- [ ] fig, ax = plt.subplots(..., constrained_layout=True)
- [ ] save_figure() used for export
```

**Labels set properly:**
```python
- [ ] ax.set_xlabel(..., fontsize=24, fontweight='bold', labelpad=15)
- [ ] ax.set_ylabel(..., fontsize=24, fontweight='bold', labelpad=15)
- [ ] ax.set_title(..., fontsize=32, fontweight='bold', pad=20)
```

**Visual elements bold:**
```python
- [ ] ax.plot(..., linewidth=4, ...)
- [ ] ax.scatter(..., s=100, ...)
- [ ] ax.tick_params(labelsize=18, width=2, length=8)
```

**Cleanup applied:**
```python
- [ ] ax.spines['top'].set_visible(False)
- [ ] ax.spines['right'].set_visible(False)
- [ ] ax.xaxis.set_major_locator(plt.MaxNLocator(5))
```

## Pre-Submission Final Check

**Ask these questions:**

1. **Can it be read from 30 feet away?**
   - If no → Make fonts larger, lines thicker

2. **Is main message obvious in 3 seconds?**
   - If no → Simplify or change title

3. **Does every element serve the message?**
   - If no → Remove that element

4. **Could this be simpler?**
   - If yes → Simplify it

5. **Would this work on a bad projector?**
   - If no → Increase contrast, use bolder colors

6. **Is anything overlapping?**
   - If yes → Fix with spatial-layout.md guidance

## Score Your Figure

**Count YES answers in Readability + Clarity + Technical sections:**

- 25-28: ✅ Excellent, ready for slides
- 20-24: ⚠️ Good, minor improvements needed
- 15-19: ⚠️ Needs work, several issues to fix
- <15: ❌ Major revision needed

**Any NO in critical items = must fix before using**

**Critical items:**
- Fonts ≥16pt
- constrained_layout used
- No overlapping elements
- High contrast colors
- Main message clear
- Readable from distance

## After Using in Presentation

**Post-talk evaluation:**

- [ ] Which figures were most effective?
- [ ] Where did audience seem confused?
- [ ] Any technical issues?
- [ ] What would you change?

**Update for next time:**
- [ ] Simplify confusing figures
- [ ] Increase contrast where needed
- [ ] Fix any overlap issues
- [ ] Update based on feedback

## Quick Reference

**Minimum standards:**
- Fonts: 16pt minimum, 20pt+ preferred
- Lines: 3pt minimum, 4pt preferred
- Colors: High contrast, colorblind-safe
- Layout: constrained_layout=True
- Export: PNG at 150 DPI

**Best practices:**
- Read spatial-layout.md FIRST
- One message per figure
- Bold, oversized elements
- Aggressive simplification
- Test from distance

**When in doubt:**
- Make it bigger
- Make it bolder
- Make it simpler
- Remove decoration
