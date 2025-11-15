# Layout Guide for Scientific Presentations

## Slide Dimensions

### Standard Aspect Ratio

**16:9 aspect ratio only** - Modern widescreen format

**Standard dimensions:**
- Width: 10 inches
- Height: 5.625 inches
- Ratio: 1.778:1

**Why 16:9?**
- Matches modern projectors
- Matches laptop screens
- Maximizes screen real estate
- Standard for conferences

**Never use:**
- 4:3 aspect ratio (outdated)
- Mixed ratios in single presentation
- Custom non-standard ratios

### Safe Margins

**Keep all content within safe margins:**

- **Minimum margin**: 0.5 inches from all edges
- **Preferred margin**: 1.0 inch from all edges
- **Critical content**: Center 80% of slide

**Why margins matter:**
- Projectors often crop edges
- Screen bezels may cover content
- Creates breathing room
- Prevents claustrophobic feel

**Safe zone:**
```
┌─────────────────────────────┐
│ ← 1" margin                 │
│   ┌─────────────────────┐   │
│   │                     │   │
│   │   SAFE CONTENT      │   │
│   │   AREA             │   │
│   │                     │   │
│   └─────────────────────┘   │
│                1" margin → │
└─────────────────────────────┘
```

## The 50% Whitespace Rule

### Concept

**Aim for half of every slide to be empty space.**

**Benefits:**
- Reduces cognitive load
- Makes focal point obvious
- Looks professional
- Easier to process
- Memorable

### Implementation

**Before adding content, ask:**
1. Is this element essential?
2. Does it serve the message?
3. Can I make it bigger instead of adding more?

**Overfilled slide indicators:**
- Can't find focal point
- Eyes don't know where to look
- Feels cramped or busy
- Multiple competing elements

**Well-spaced slide indicators:**
- One obvious focal point
- Comfortable to look at
- Message is clear
- Professional appearance

### Examples

```
❌ Overfilled (80% covered):
┌─────────────────────────────┐
│ TITLE TEXT HERE LONG        │
│ • Bullet 1 goes here        │
│ • Bullet 2 also here        │
│ • Bullet 3 is here too      │
│ • Bullet 4 keeps going      │
│ • Bullet 5 and more         │
│ [Image] [Graph] [Diagram]   │
│ Caption and more text here  │
└─────────────────────────────┘

✅ Good spacing (50% covered):
┌─────────────────────────────┐
│                             │
│   Treatment Doubles         │
│   Response Rate             │
│                             │
│     [Large Clean Plot]      │
│                             │
│                             │
│     n=50, p<0.001           │
│                             │
└─────────────────────────────┘
```

## Visual Hierarchy

### Single Focal Point Rule

**Every slide must have one clear focal point** - the element that draws attention first.

**How to create focal point:**

1. **Size** - Make it largest element
2. **Contrast** - Highest contrast on slide
3. **Position** - Upper third of slide (F-pattern)
4. **Isolation** - Surround with whitespace
5. **Color** - Use accent color

**Common focal points:**
- Large data visualization
- Key finding statement
- Critical image
- Essential diagram

### Z-Pattern and F-Pattern

**Western audiences scan in predictable patterns:**

**Z-Pattern (for sparse content):**
```
1 → → → → 2
    ↓
3 → → → → 4
```

**F-Pattern (for text-heavy content):**
```
1 → → → → 
2 → → 
3 → → →
↓
4
```

**Design implications:**
- Place title at top left (position 1)
- Place key message at top right (position 2)
- Place supporting info in middle left (position 3)
- Place caption/note at bottom right (position 4)

### Title Placement

**Title must be at very top of slide:**

- Consistent position across all slides
- Flush with top margin
- Same size throughout
- Never move title for other content

**Why consistent title position?**
- Predictability reduces cognitive load
- Audience knows where to look
- Professional appearance
- Creates visual rhythm

## Spacing and Breathing Room

### Element Spacing

**Space between elements ≥ element size/2**

Example:
- If text block is 2 inches tall
- Leave ≥1 inch before next element

**Minimum spacing:**
- Between bullets: 0.25 inches
- Between sections: 0.5 inches
- Around focal element: 1.0 inch

### Chunking

**Group related items, separate unrelated items with whitespace.**

**Gestalt principles:**

1. **Proximity** - Items close together are related
2. **Similarity** - Similar items are related
3. **Enclosure** - Items in same container are related
4. **Connection** - Connected items are related

**Example:**
```
✅ Good chunking:
┌─────────────────────────────┐
│   Methods                    │
│                             │
│   Step 1: Sample prep       │
│   Step 2: Treatment         │
│   Step 3: Analysis          │
│                             │
│   n=50 per group            │
│   p<0.05 threshold          │
└─────────────────────────────┘

❌ Bad chunking:
┌─────────────────────────────┐
│   Methods                    │
│   Step 1: Sample prep       │
│   n=50 per group            │
│   Step 2: Treatment         │
│   p<0.05 threshold          │
│   Step 3: Analysis          │
└─────────────────────────────┘
```

## Grid Alignment

### Alignment System

**All elements must align to invisible grid.**

**Alignment rules:**

1. **Horizontal alignment**:
   - All text left-aligned to same vertical line
   - All images aligned to grid
   - Titles aligned consistently

2. **Vertical alignment**:
   - Elements align to horizontal grid lines
   - Consistent spacing between rows
   - Baseline alignment for text

3. **Center alignment**:
   - Only for titles and section dividers
   - Use sparingly

**Check alignment:**
- Draw imaginary vertical lines
- All elements should align to these lines
- No random positioning

### Misalignment Issues

**Common problems:**
- Text boxes not aligned
- Images randomly placed
- Inconsistent margins
- Bullets don't line up

**How to fix:**
- Use PowerPoint/Keynote guides
- Enable snap-to-grid
- Use alignment tools
- Check every slide

## Slide Type Layouts

### Title Slide Layout

```
┌─────────────────────────────┐
│                             │
│        LARGE TITLE          │
│      (40-44pt, bold)        │
│                             │
│    Descriptive Subtitle     │
│         (24pt)              │
│                             │
│    Author Name, PhD         │
│   Institution Name          │
│         (18pt)              │
│                             │
│     Date and Venue          │
│       (16-18pt)             │
│                             │
└─────────────────────────────┘
```

### Section Divider Layout

```
┌─────────────────────────────┐
│                             │
│                             │
│                             │
│       SECTION NAME          │
│       (44-52pt)             │
│                             │
│                             │
│                             │
│                             │
└─────────────────────────────┘
```

### Visual Content Layout

```
┌─────────────────────────────┐
│ Descriptive Title (36pt)    │
│                             │
│                             │
│    [Large Figure/Diagram]   │
│      (60-70% of slide)      │
│                             │
│                             │
│  Caption (18pt, if needed)  │
│                             │
└─────────────────────────────┘
```

### Text Content Layout

```
┌─────────────────────────────┐
│ Descriptive Title (36pt)    │
│                             │
│    • Bullet point one       │
│         (24pt)              │
│                             │
│    • Bullet point two       │
│                             │
│    • Bullet point three     │
│                             │
│   [Optional icon/diagram]   │
│                             │
└─────────────────────────────┘
```

## Position and Placement

### Optimal Placement Zones

**Attention heat map:**
```
┌─────────────────────────────┐
│ HOT    HOT     WARM         │
│ HOT    HOT     WARM         │
│ WARM   WARM    COOL         │
│ COOL   COOL    COLD         │
│ COLD   COLD    COLD         │
└─────────────────────────────┘
```

**Place important content:**
- Upper left (primary focus)
- Upper right (secondary focus)
- Center (main focal point)

**Avoid placing critical content:**
- Lower corners
- Bottom third
- Far right edge

### Multi-Element Placement

**For two elements:**
```
┌─────────────────────────────┐
│           TITLE             │
│                             │
│   [Element 1]  [Element 2]  │
│      50%          50%       │
│                             │
│   Equal size and weight     │
│                             │
└─────────────────────────────┘
```

**For three elements:**
```
┌─────────────────────────────┐
│           TITLE             │
│                             │
│ [1]     [2]      [3]       │
│  ↓       ↓        ↓        │
│                             │
│ Horizontal arrangement      │
│ (not triangle)              │
└─────────────────────────────┘
```

## Common Layout Mistakes

### ❌ Mistakes to Avoid

**1. Touching edges:**
```
❌ [Content]────────────────┘
             No margin!
```

**2. No focal point:**
```
❌ Everything same size, 
   same contrast, 
   same importance
```

**3. Random placement:**
```
❌   Text here
        And here
     And randomly here
```

**4. Overfilled:**
```
❌ Text Text Text
   Text Text Text
   [Plot][Plot][Plot]
   Text Text Text
```

**5. Inconsistent alignment:**
```
❌  Title
      Text aligned different
   More text elsewhere
       Random position
```

## Quick Fixes

**"Slide looks cluttered"**
→ Remove elements until 50% empty

**"Don't know where to look"**
→ Make one element 2× larger than others

**"Looks unprofessional"**
→ Align all elements to grid

**"Content feels cramped"**
→ Increase margins to 1.0 inch

**"Hard to follow structure"**
→ Use consistent title position

**"Projector cuts off edges"**
→ Keep content in center 80% of slide
