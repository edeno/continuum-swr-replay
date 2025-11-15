# Accessibility Guide for Scientific Presentations

## Core Principle

**Design presentations that are accessible to all audience members, including those with visual, cognitive, or other disabilities.**

## WCAG Standards Overview

### WCAG Levels

**Level A**: Minimum accessibility (avoid)
**Level AA**: Recommended standard (use this)
**Level AAA**: Enhanced accessibility (ideal)

**This guide focuses on WCAG AA compliance.**

## Color and Contrast Requirements

### Contrast Ratios

**WCAG AA Standards:**

**Normal text** (<18pt or <14pt bold):
- Minimum ratio: 4.5:1
- Preferred: 7:1 or higher

**Large text** (≥18pt or ≥14pt bold):
- Minimum ratio: 3.0:1
- Preferred: 4.5:1 or higher

**Non-text elements** (icons, graphs):
- Minimum ratio: 3.0:1

### Testing Contrast

**Tools:**
- WebAIM Contrast Checker (webaim.org/resources/contrastchecker)
- Color Contrast Analyzer (desktop app)
- Built into browser developer tools

**Testing process:**
1. Select foreground color (text)
2. Select background color
3. Verify ratio meets 4.5:1 minimum
4. Test for both normal and large text

### Compliant Color Combinations

**On white background (#FFFFFF):**
- Black text (#000000): 21:1 ✅ AAA
- Dark gray (#333333): 12.6:1 ✅ AAA
- Medium gray (#595959): 7.0:1 ✅ AAA
- Dark blue (#0072B2): 5.1:1 ✅ AA

**On dark background (#1A1A1A):**
- White text (#FFFFFF): 16.1:1 ✅ AAA
- Light gray (#CCCCCC): 11.2:1 ✅ AAA
- Yellow (#F0E442): 14.7:1 ✅ AAA
- Sky blue (#56B4E9): 10.4:1 ✅ AAA

### Common Contrast Failures

**Avoid these combinations:**

❌ **Yellow on white** (#F0E442 on #FFFFFF)
- Ratio: 1.4:1 (FAIL)
- Fix: Use darker yellow or different background

❌ **Light gray on white** (#CCCCCC on #FFFFFF)
- Ratio: 1.6:1 (FAIL)
- Fix: Use darker gray (#767676 or darker)

❌ **Medium blue on black** (#56B4E9 on #000000)
- Ratio: 4.3:1 (FAIL for normal text)
- Fix: Use for large text only, or adjust shade

## Colorblind-Safe Design

### Types of Color Vision Deficiency

**Deuteranopia** (6% of males, 0.4% of females):
- Red-green confusion
- Most common type

**Protanopia** (2% of males, 0.01% of females):
- Red-green confusion (different type)

**Tritanopia** (0.01% of population):
- Blue-yellow confusion
- Very rare

**Total color blindness** (0.003% of population):
- See only grayscale
- Extremely rare

### Colorblind-Safe Palettes

**Use these pre-tested palettes:**

**Wong Palette** (Recommended):
```
#E69F00 Orange
#56B4E9 Sky Blue
#009E73 Bluish Green
#F0E442 Yellow
#0072B2 Blue
#D55E00 Vermillion
#CC79A7 Reddish Purple
#000000 Black
```

**Okabe-Ito Palette** (Alternative):
```
Similar to Wong but different ordering
Equally effective
```

**IBM Design Colorblind Safe**:
```
#648FFF Blue
#785EF0 Purple
#DC267F Magenta
#FE6100 Orange
#FFB000 Yellow
```

### Color Combinations to Avoid

**Never use together:**

❌ **Red and green** (classic problem)
- Indistinguishable to red-green colorblind
- Affects ~8% of males

❌ **Blue and purple**
- Difficult for some types

❌ **Green and brown**
- Can appear similar

❌ **Red and brown**
- Can appear similar

### Beyond Color: Redundant Encoding

**Never rely on color alone to convey information.**

**Always use color + additional encoding:**

**Color + Shape**
```
✅ Red circles vs Blue squares
❌ Red circles vs Blue circles
```

**Color + Pattern**
```
✅ Red solid vs Blue dashed
❌ Red solid vs Blue solid
```

**Color + Labels**
```
✅ Red line labeled "Control" vs Blue line labeled "Treatment"
❌ Red line vs Blue line (unlabeled)
```

**Color + Position**
```
✅ Color + legend + direct labels on plot
❌ Color only with legend
```

### Testing for Colorblindness

**Simulation tools:**

**ColorOracle** (Free, desktop)
- Real-time colorblind simulation
- Shows three main types
- Works on entire screen

**Coblis** (Web-based)
- Upload images
- Simulate different types
- Quick testing

**Process:**
1. Create presentation slide
2. Take screenshot
3. Run through simulator
4. Verify information is still clear
5. Adjust if needed

## Cognitive Accessibility

### Cognitive Load Reduction

**Working memory limits:**
- Average person: 5-7 items
- Design for: 4-5 items maximum
- This applies to: bullets, plot lines, categories, sections

**Strategies:**

**1. One message per slide**
- Single clear takeaway
- Reduces decision fatigue
- Easier to remember

**2. Progressive disclosure**
- Build complexity gradually
- Don't overwhelm
- Allow processing time

**3. Consistent structure**
- Same layout throughout
- Predictable organization
- Reduces mental effort

**4. Visual hierarchy**
- Clear focal point
- Obvious where to look first
- Natural eye movement pattern

### Reading Accessibility

**Dyslexia considerations:**

**Font selection:**
- Sans-serif (easier to read)
- Avoid narrow fonts
- Consistent letter spacing
- Adequate line spacing (1.5×)

**Text formatting:**
- Left-aligned (not justified)
- Sentence case (not ALL CAPS)
- Short lines (6-8 words)
- No italics for body text

**Content structure:**
- Short bullets (3-5 words)
- No dense paragraphs
- Visual breaks
- Whitespace for breathing

### Attention and Focus

**ADHD considerations:**

**Minimize distractions:**
- No decorative animations
- No auto-advancing slides
- No sound effects
- Clean, simple layouts

**Clear structure:**
- Section dividers
- Progress indicators (subtle)
- Logical flow
- Predictable patterns

**Engagement:**
- Frequent visual variety
- Questions to refocus
- Interactive elements (when appropriate)
- Regular breaks in long talks

## Motion and Animation

### Motion Sensitivity

**Some people experience:**
- Motion sickness from animations
- Vestibular disorders
- Migraine triggers
- Disorientation

### Safe Animation Practices

**Acceptable:**
- Fade in/out (0.3-0.5s)
- Simple wipes (0.3-0.5s)
- Gentle transitions

**Avoid:**
- Rapid movements
- Spinning/rotating
- Zooming
- Parallax effects
- Auto-play videos
- Flashing content

### Seizure Safety

**CRITICAL: Never include:**

❌ **Flashing content** (>3 flashes per second)
- Can trigger seizures
- Includes: strobe effects, rapid transitions, flickering

❌ **High-contrast rapid changes**
- Even without flashing
- Sudden dark-to-light switches

**Safe practices:**
- Animation duration ≥0.3 seconds
- No rapid color changes
- No flickering elements
- Test on colleagues

## Slide Reading Order

### Logical Reading Sequence

**Screen readers and comprehension require logical order:**

**Proper sequence:**
```
1. Slide title
2. Main content (left-to-right, top-to-bottom)
3. Supporting information
4. Captions/notes
```

**PowerPoint/Keynote:**
- Elements are read in creation order
- Verify order in "Selection Pane" (PowerPoint)
- Reorder if needed

### Describable Content

**All visual content should be describable in words.**

**Test: Can you explain the slide without showing it?**

**Good example:**
- "The plot shows treatment doubles response"
- "Three boxes show the workflow"
- Clear, concrete, specific

**Bad example:**
- "You can see the effect here"
- "As shown in this image"
- Relies on seeing specific details

## Presentation Delivery Accessibility

### For Deaf/Hard-of-Hearing

**Best practices:**

**Visual support:**
- All key points on slides (not just spoken)
- Captions for videos
- Written transcript available
- Visual indication of audio cues

**Speaking style:**
- Face audience (lip reading)
- Clear enunciation
- Repeating questions from audience
- Using microphone

### For Blind/Low-Vision

**Best practices:**

**Verbal description:**
- Describe all visual content
- Read axis labels and values
- Explain what images show
- Don't rely on "as you can see"

**Handouts:**
- Accessible format (Word, not PDF image)
- Large print available
- Digital version for screen readers

**Structure:**
- Clear verbal signposting
- Announce slide changes
- Describe slide layout

### Universal Design Principles

**Design for extremes benefits everyone:**

- Large fonts → Everyone can read better
- High contrast → Visible in bright rooms
- Simple layouts → Faster comprehension
- Descriptive titles → Clearer messages
- Visual + verbal → Redundant encoding

## Checklist for Accessible Presentations

### Visual Accessibility
- [ ] All text meets 4.5:1 contrast ratio
- [ ] Font size ≥18pt for body, ≥32pt for titles
- [ ] Colorblind-safe palette used (Wong/Okabe-Ito)
- [ ] Color + shape/pattern/labels (never color alone)
- [ ] Tested with colorblind simulator
- [ ] High contrast backgrounds (white or dark gray)

### Cognitive Accessibility
- [ ] One message per slide
- [ ] Maximum 5-7 items on screen
- [ ] Progressive disclosure for complexity
- [ ] Consistent layout and structure
- [ ] Clear visual hierarchy
- [ ] Sans-serif fonts throughout
- [ ] Left-aligned text
- [ ] 1.5× line spacing

### Motion Accessibility
- [ ] No animations >0.5s duration
- [ ] No spinning, bouncing, or decorative effects
- [ ] No flashing content (never >3 per second)
- [ ] No auto-advancing slides
- [ ] Fade/wipe only (no complex transitions)

### Content Accessibility
- [ ] All visual content is describable
- [ ] Logical reading order
- [ ] Descriptive titles (not generic)
- [ ] Key points on slides (not only spoken)
- [ ] Adequate whitespace
- [ ] No reliance on "you can see"

### Delivery Accessibility
- [ ] Will describe visual content verbally
- [ ] Will face audience when speaking
- [ ] Handouts in accessible format
- [ ] Transcript available (if recorded)
- [ ] Captions for videos

## Resources

### Color Tools

**WebAIM Contrast Checker**
- URL: webaim.org/resources/contrastchecker
- Tests foreground/background combinations
- Shows WCAG compliance level

**ColorOracle**
- Free colorblind simulator
- Real-time screen simulation
- Tests entire presentation

**Coblis**
- URL: color-blindness.com/coblis-color-blindness-simulator
- Upload image simulation
- Multiple CVD types

### Testing Process

1. **Design slide** with chosen colors
2. **Test contrast** with WebAIM checker
3. **Simulate colorblindness** with ColorOracle
4. **Verify readability** from 30 feet
5. **Check reading order** in Selection Pane
6. **Describe slide verbally** to test describability
7. **Get feedback** from diverse reviewers

### Guidelines References

**WCAG 2.1 Guidelines**
- URL: w3.org/WAI/WCAG21/quickref
- Complete accessibility guidelines
- Detailed success criteria

**WebAIM Presentations Guide**
- URL: webaim.org/articles/presentation/
- Specific to presentations
- Practical guidance

## Common Questions

**Q: Is WCAG required for academic presentations?**
A: Not legally in most cases, but ethically important. Many universities have accessibility policies.

**Q: What if I need more than 3 colors?**
A: Use shapes, patterns, or labels in addition to colors. Or split into multiple slides.

**Q: Can I use red for negative and green for positive?**
A: No. Red-green is problematic for colorblind viewers. Use orange/blue or vermillion/cyan instead.

**Q: My video has no captions. Is that okay?**
A: No. Add captions or provide transcript. Many tools auto-generate captions.

**Q: How do I make equations accessible?**
A: Use equation editor (creates readable math), describe verbally, provide written handout.

## Summary

**Core accessibility principles:**

1. **High contrast** (4.5:1 minimum)
2. **Colorblind-safe palettes** (Wong, Okabe-Ito)
3. **Color + additional encoding** (never color alone)
4. **Cognitive load management** (one message, 5-7 items max)
5. **Safe motion** (≥0.3s, no flashing)
6. **Describable content** (can explain without showing)
7. **Universal design** (benefits everyone)

**When in doubt:** Test with actual users with disabilities, or use simulation tools.
