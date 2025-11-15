# Typography Standards for Scientific Presentations

## Font Size Requirements

### Absolute Minimums

**Never use fonts smaller than these sizes:**

- **Title**: 32pt minimum
- **Section headers**: 28pt minimum
- **Body text**: 18pt minimum
- **Figure captions**: 16pt minimum
- **Fine print** (if unavoidable): 14pt absolute minimum

### Preferred Sizes

**Use these sizes for optimal readability:**

- **Title**: 36-44pt (44pt for title slide)
- **Section headers**: 28-32pt
- **Body text**: 18-24pt (24pt preferred)
- **Figure captions**: 16-18pt

**Why these sizes matter:**
- Must be readable from 30+ feet away
- Account for poor projector quality
- Reduce eye strain for audience
- Ensure accessibility for vision-impaired

## Font Selection

### Approved Font Families

**Sans-serif fonts only** - These fonts are designed for screen readability:

**Primary options:**
- Arial
- Helvetica (or Helvetica Neue)
- Calibri
- Gill Sans
- Futura
- Avenir

**Alternative options:**
- Open Sans
- Lato
- Roboto
- Source Sans Pro

### Font Family Rules

**Maximum 2 families per presentation:**
1. One for headings (can be the same as body)
2. One for body text

**Recommended combinations:**
- Gill Sans (headings) + Helvetica (body)
- Futura (headings) + Arial (body)
- Arial for everything (simplest, always works)

**Why limit font families?**
- Maintains visual consistency
- Reduces cognitive load
- Looks more professional
- Prevents "ransom note" effect

### Fonts to Avoid

**Never use these for presentations:**

❌ **Serif fonts** (Times New Roman, Georgia, Garamond)
- Hard to read on screens
- Fine details disappear at distance
- Better for print, not projection

❌ **Decorative fonts** (Comic Sans, Papyrus, Brush Script)
- Unprofessional
- Hard to read
- Distracting

❌ **Monospace fonts** (Courier, Consolas) - except for code
- Inefficient use of space
- Harder to read at distance

## Text Formatting

### Case and Capitalization

**Use sentence case:**
```
✅ "Treatment increases survival 3-fold"
❌ "TREATMENT INCREASES SURVIVAL 3-FOLD"
❌ "Treatment Increases Survival 3-Fold"
```

**Why sentence case?**
- ALL CAPS is harder to read (we recognize word shapes)
- Title Case Is Slower To Read
- Sentence case is most natural

**Exceptions where ALL CAPS is acceptable:**
- Section divider slides (large, single word)
- Acronyms (DNA, RNA, CRISPR)

### Emphasis Methods

**Use bold for emphasis:**
```
✅ "Significant **increase** in response"
❌ "Significant _increase_ in response"  (italics)
❌ "Significant __increase__ in response"  (underline)
```

**Why bold only?**
- Most visible at distance
- Clear and unambiguous
- Doesn't reduce readability

**Never use:**
- Underlines (looks like links)
- Italics (hard to read on screens)
- ALL CAPS for emphasis (too aggressive)
- Color alone for emphasis (not accessible)

### Alignment

**Left-align all body text:**
```
✅ Left aligned text
   is easiest to read
   on presentations

❌ Justified text creates
   uneven    spacing   and
   is   harder   to  read

❌ Center aligned text
   is fine for titles but
   terrible for body text
```

**Why left-align?**
- Creates consistent starting point
- Easier for eyes to track
- No awkward spacing gaps
- Faster to read

**Exceptions:**
- Titles can be centered
- Section dividers can be centered
- Very short text (1-3 words) can be centered

### Line Formatting

**Line length:**
- Maximum 6-8 words per line
- Break longer lines into multiple bullets
- Keep lines short and scannable

**Line spacing:**
- Use 1.3-1.5× line spacing
- More space = easier to read
- Single spacing is too cramped

**Example:**
```
✅ Good line spacing:

First bullet point
   (adequate space)
Second bullet point
   (easy to scan)
Third bullet point

❌ Too tight:

First bullet point
Second bullet point  ← hard to distinguish
Third bullet point
```

## Special Cases

### Equations and Formulas

**Font sizes for math:**
- Main equation: 24-28pt
- Subscripts/superscripts: 16-18pt
- Variable definitions: 18-20pt

**Best practices:**
- Use equation editor, not plain text
- Ensure all symbols are legible
- Define variables below equation

### Code Snippets

**If showing code (rare in scientific talks):**
- Use monospace font (Consolas, Courier)
- Minimum 16pt, prefer 18-20pt
- Syntax highlighting with high contrast
- Maximum 6-8 lines per slide

### Tables (Avoid!)

**If tables are unavoidable:**
- Header text: ≥20pt, bold
- Cell text: ≥18pt
- Maximum 3×5 cells
- No gridlines (use whitespace)
- Highlight key values

**Better alternatives to tables:**
- Bar charts
- Dot plots
- Heat maps
- Text summary

## Font Embedding

### Technical Requirements

**Always embed fonts in presentation file:**

**PowerPoint:**
1. File → Options → Save
2. Check "Embed fonts in the file"
3. Select "Embed all characters"

**Keynote:**
- Fonts are automatically embedded

**PDF Export:**
- Fonts are automatically embedded
- Verify in PDF properties

### Why embed fonts?

- Presentation computer may not have your fonts
- Prevents font substitution
- Maintains exact spacing and layout
- Ensures consistency across platforms

### Font Backup Strategy

**Before presenting:**
1. Embed fonts in .pptx/.key file
2. Create backup with standard fonts (Arial only)
3. Export backup PDF
4. Test on presentation computer

## Validation Checklist

Before finalizing, check:

- [ ] All fonts ≥16pt (18pt+ preferred)
- [ ] Titles ≥32pt (36pt+ preferred)
- [ ] Sans-serif fonts only
- [ ] Maximum 2 font families
- [ ] Sentence case for titles
- [ ] Bold for emphasis (not underline/italics)
- [ ] Left-aligned body text
- [ ] 1.3-1.5× line spacing
- [ ] 6-8 words per line maximum
- [ ] Fonts embedded in file

## Quick Fixes

**"Text too small"**
→ Increase to 18pt body, 36pt titles

**"Hard to read from back"**
→ Use 24pt body text minimum

**"Looks unprofessional"**
→ Use only 1-2 fonts, prefer Arial/Helvetica

**"Text looks cramped"**
→ Increase line spacing to 1.5×

**"Fonts different on presentation computer"**
→ Embed fonts in file

**"Too many different styles"**
→ Use maximum 2 font families, consistent sizing
