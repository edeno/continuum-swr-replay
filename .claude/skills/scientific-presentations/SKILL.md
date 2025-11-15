---
name: scientific-presentations
description: Creates clear, effective scientific presentations optimized for comprehension and retention. Use when building conference talks, seminar slides, lecture materials, or any scientific presentation requiring professional design and cognitive load management.
license: MIT
---

# Scientific Presentation Design

## Overview

Create scientifically rigorous presentations that maximize audience comprehension and retention through evidence-based design principles. This skill emphasizes visual clarity, cognitive load management, and progressive disclosure.

## When to Use This Skill

Use this skill for creating:

- Conference talks and academic presentations
- Seminar and lecture slides
- Scientific posters converted to slides
- Lab meeting presentations
- Thesis defense presentations
- Grant proposal presentations

## Core Workflow

### Step 1: Plan Content Structure

Before creating slides:

1. Define single message per slide
2. Identify which content is visual vs. text
3. Plan progressive disclosure for complex topics
4. Map out section flow and transitions

### Step 2: Read Relevant References

Based on your needs, read the appropriate reference files:

**Always read first:**

- `references/spatial-layout.md` - CRITICAL for preventing overlaps in figures/text
- `references/design-checklist.md` - Quick reference for all standards

**Read as needed:**

- `references/typography-standards.md` - Font sizes, styles, formatting rules
- `references/layout-guide.md` - Spacing, alignment, visual hierarchy
- `references/animation-guide.md` - When and how to use animations
- `references/examples.md` - Good vs bad examples with explanations
- `references/accessibility-guide.md` - WCAG compliance and colorblind design

### Step 3: Set Up Presentation

```python
# Standard 16:9 aspect ratio (10" × 5.625")
aspect_ratio = (16, 9)
slide_dimensions = (10, 5.625)  # inches

# Load colorblind-safe palette
import json
with open('assets/color-palettes.json') as f:
    palettes = json.load(f)
    wong_colors = palettes['palettes']['wong']['colors']
```

### Step 4: Create Slides Following Standards

**Mandatory standards for every slide:**

**Typography:**

- Title: 32pt minimum (36-44pt preferred)
- Body text: 18-24pt (never <16pt)
- Sans-serif fonts only (Arial, Helvetica, Calibri)
- Maximum 2 font families

**Layout:**

- 50% whitespace rule (aim for half of slide empty)
- Single clear focal point per slide
- Safe margins: 0.5-1.0" from all edges
- Nothing touching slide boundaries

**Color:**

- Maximum 3 colors per slide
- Use Wong or Okabe-Ito colorblind-safe palettes
- WCAG AA contrast: 4.5:1 for text
- Color + shape (never color alone)

**Content:**

- Descriptive titles (not "Results" or "Methods")
- Maximum 7 bullet points (5 preferred)
- 3-5 words per bullet
- No paragraphs on slides

### Step 5: Handle Data Visualization

**CRITICAL - Read spatial layout guide BEFORE creating any plots:**

```bash
# This prevents overlap issues
cat references/spatial-layout.md
```

**For all plots and figures:**

1. Use `scientific-figures-presentation` skill for creating plots
2. Follow spatial layout guide for legend placement and spacing
3. Ensure all text is ≥18pt (axis labels, ticks, captions)
4. Use direct labeling instead of legends when possible
5. One plot per slide (or max 2 for direct comparison)

### Step 6: Add Progressive Disclosure

For complex content, build incrementally:

```
Slide 1: "Does Treatment Affect Response?"
- Empty axes + labels

Slide 2: (Same title)
- Add baseline data with annotation

Slide 3: (Same title)
- Add treatment data with fade animation

Slide 4: "Treatment Doubles Response Rate"
- Highlight difference with arrow + statistical note
```

**Animation guidelines:**

- Use only to guide attention or build complexity
- Fade/wipe only (0.3-0.5s duration)
- No decorative animations (bounce, spin, fly)
- No slide transitions (or fade only)

### Step 7: Validate Presentation

```bash
# Run automated validation
python scripts/validate_presentation.py presentation.pptx
```

The validator checks:

- Font sizes meet minimums
- Aspect ratio is correct
- Bullet counts are reasonable
- Text density is appropriate

Fix all critical issues before finalizing.

## Integration with Other Skills

### With scientific-figures-presentation Skill

**Always use together for data visualization:**

```python
# 1. Use scientific-figures-presentation to create plots
# 2. Follow spatial-layout.md to prevent overlaps
# 3. Use this skill for slide design and layout
```

**Workflow:**

1. Read `references/spatial-layout.md`
2. Create plots with scientific-figures-presentation skill
3. Verify plot text sizes meet presentation standards (≥18pt)
4. Integrate plots into slides with proper layout

### With pptx Skill (if available)

- Use pptx skill for programmatic slide creation
- Use this skill for design principles and validation
- Apply constraints from this skill to pptx output

## Common Patterns

### Title Slide

- Large title (40-44pt)
- Subtitle/context (24pt)
- Author info (18pt)
- Minimal decoration

### Section Divider

- Section name only (44-52pt)
- High contrast
- Clean and bold

### Visual Content Slide

- Descriptive title (36pt)
- Large figure (60-70% of slide)
- Brief caption if needed (18pt)

### Text Content Slide

- Descriptive title (36pt)
- 3-5 short bullets (24pt)
- Maximum 50% filled
- Generous whitespace

## Quick Reference: Must-Dos

**Before creating any slide:**

- [ ] Read design-checklist.md for full requirements
- [ ] Read spatial-layout.md if including plots

**Font sizes:**

- Title: ≥32pt | Body: ≥18pt | Minimum anywhere: 16pt

**Layout:**

- 50% whitespace | Single focal point | 0.5-1.0" margins

**Color:**

- Max 3 per slide | Use Wong/Okabe-Ito | WCAG AA contrast

**Content:**

- Descriptive titles | Max 7 bullets | No paragraphs

**Validation:**

- Run validate_presentation.py before finalizing

## File Resources

### Scripts

- `validate_presentation.py` - Automated design validation

### References (Read as Needed)

- `spatial-layout.md` - CRITICAL: Read before any plotting
- `design-checklist.md` - Complete checklist for review
- `typography-standards.md` - Detailed font specifications
- `layout-guide.md` - Spacing, hierarchy, alignment details
- `animation-guide.md` - Progressive disclosure and animation
- `examples.md` - Good vs bad examples with explanations
- `accessibility-guide.md` - WCAG compliance details

### Assets

- `color-palettes.json` - Colorblind-safe color schemes

## Key Principles

**Show, don't tell** - Use visuals over bullet points

**One message per slide** - Single clear takeaway in descriptive title

**Guide attention** - Through hierarchy, animation, contrast

**Minimize cognitive load** - Progressive disclosure, whitespace, simplicity

**Maximize readability** - Large fonts, high contrast, clean layout

**Ensure accessibility** - Colorblind-safe, describable, predictable

## When in Doubt

- Make it bigger
- Make it simpler
- Add whitespace
- Remove decoration
- Test from distance (30 feet)
- Run validation script

## Next Steps After Creating Slides

1. Run `validate_presentation.py` to check standards
2. Test on actual projector
3. View from back of room
4. Practice with timing
5. Create backup PDF
6. Note: Keep original .pptx for animations
