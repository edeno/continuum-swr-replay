# Presentation Examples: Good vs Bad

This reference shows concrete examples of common slide types, comparing poor design with effective design following our principles.

## Example 1: Results Slide with Data

### ❌ Bad Version

```
Title: "Results"

Content:
• Treatment group showed increased response
• Control group maintained baseline levels
• Statistical analysis revealed significant differences
• Effect size was large and consistent across replicates
• p-value was less than 0.001

[Small plot squeezed in bottom-right corner]
- 10pt axis labels
- Legend inside plot covering data
- No clear focal point
```

**Problems:**
- Generic title ("Results" - tells us nothing)
- Too much text (5 bullets reading like sentences)
- Plot is secondary, not primary
- Poor visual hierarchy
- Tells rather than shows

**Cognitive load**: HIGH - Audience must read, process text, then look at plot, then integrate information.

### ✅ Good Version

```
Title: "Treatment Doubles Response Rate" (36pt, bold)

Content:
[Large plot occupying 80% of slide]
- 24pt axis labels with units: "Response (μM)"
- 20pt tick labels
- Direct labels on lines (no legend needed)
- Arrow annotation pointing to treatment line: "2× increase"
- Caption: "n=50 per group, p<0.001" (18pt)

Whitespace: 50% of slide
```

**Why this works:**
- Descriptive title states the finding
- Visual evidence is primary (plot is focal point)
- Direct labeling eliminates legend
- Minimal text, maximum impact
- Clear single message

**Cognitive load**: LOW - Audience sees finding immediately in plot and title.

---

## Example 2: Methods Overview

### ❌ Bad Version

```
Title: "Methods"

Content:
We collected samples from 50 subjects in each group. 
Samples were processed using standard protocols including 
centrifugation, washing, and resuspension. Treatment was 
applied at a concentration of 10 μM for 24 hours. Analysis 
was performed using flow cytometry with appropriate 
controls and compensation. Statistical analysis used 
t-tests with Bonferroni correction.

[Tiny diagram in corner, barely visible]
```

**Problems:**
- Generic title
- Dense paragraph (unreadable from distance)
- Too much detail for overview
- Diagram is afterthought
- No visual hierarchy

**Result**: Audience stops reading after first sentence. Information lost.

### ✅ Good Version

```
Title: "Three-Step Protocol" (36pt, bold)

Content:
[Three large boxes arranged horizontally, each containing:]

Box 1: "Sample Prep"
- Icon: Test tube
- Text: "50 subjects" (20pt)

→ (Large arrow)

Box 2: "Treatment" 
- Icon: Dropper
- Text: "10 μM, 24h" (20pt)

→ (Large arrow)

Box 3: "Analysis"
- Icon: Graph
- Text: "Flow cytometry" (20pt)

Colors: Wong palette for boxes
Spacing: Generous whitespace between elements
```

**Why this works:**
- Clear workflow visualization
- Icons support comprehension
- Minimal text (just essential info)
- Visual > textual
- Easy to follow left-to-right

**Result**: Audience understands process in 3 seconds.

---

## Example 3: Comparison Slide

### ❌ Bad Version

```
Title: "Comparison of Methods"

Content:
• Method A: Traditional approach, well established,
  slower processing time, requires expertise
• Method B: New approach, faster, easier to use,
  comparable accuracy

[Two small plots side-by-side]
- Different color schemes
- Different axis scales
- Hard to compare directly
- 12pt labels
```

**Problems:**
- Tells us about methods rather than showing comparison
- Plot scales don't match (can't compare visually)
- Inconsistent styling
- Competing focal points

### ✅ Good Version

```
Title: "Method B is 3× Faster with Equal Accuracy" (32pt)

Content:
[Two matched plots, same scale, side-by-side]

Plot 1 (left): "Method A"
- Consistent axes
- Processing time: 60 min (annotation)

Plot 2 (right): "Method B"  
- Same axes for direct comparison
- Processing time: 20 min (annotation)

[Large arrow between plots: "3× faster"]
[Bottom annotation: "Accuracy: 95% (A) vs 94% (B)"]

Color scheme: Method A = gray, Method B = accent color
```

**Why this works:**
- Title states conclusion
- Direct visual comparison
- Matched scales enable comparison
- Focus on key differences
- Evidence supports claim

---

## Example 4: Hypothesis Slide

### ❌ Bad Version

```
Title: "Hypothesis"

Content:
We hypothesized that the treatment would affect the
response through multiple mechanisms including direct
receptor binding and indirect pathway modulation.
Previous work suggested this possibility but direct
evidence was lacking.
```

**Problems:**
- Wall of text
- No visual element
- Complicated sentence structure
- Unreadable from distance
- Boring

### ✅ Good Version

```
Title: "Does Treatment Activate Pathway?" (36pt)

Content:
[Simple diagram showing]
Treatment → ? → Response
         (large question mark)

Alternative:
Treatment → Pathway → Response
         (with fade-in animation)

Or simply:
"Does Treatment Activate Pathway?"
(centered, 44pt)
[Background: subtle relevant image at 20% opacity]
```

**Why this works:**
- Question format engages audience
- Visual representation aids understanding
- Sets up answer in next slides
- Creates narrative tension
- Memorable

---

## Example 5: Key Finding

### ❌ Bad Version

```
Title: "Finding"

Content:
The data demonstrates that the treatment produces
a significant effect on the response. This finding
is important because it suggests a novel mechanism.
The effect was observed across multiple experimental
conditions and remained consistent.

[Table of numbers at bottom]
```

**Problems:**
- Vague title
- Buries the finding in prose
- Table forces audience to work
- No visual evidence
- Unclear what the finding actually is

### ✅ Good Version

```
Title: "Treatment Activates Novel Pathway" (36pt)

Content:
[Large simple diagram]
Treatment → Pathway X → 2× Response
         (emphasized)    (highlighted)

Key evidence (18pt):
• Observed in all conditions
• Independent validation

Color: Pathway X in accent color
```

**Why this works:**
- Finding is in title
- Visual shows mechanism
- Evidence listed concisely
- Clear and memorable
- Focused message

---

## Example 6: Problematic Table

### ❌ Bad Version

```
Title: "Summary Statistics"

[Dense table: 8 rows × 6 columns]
- 10pt text
- Gridlines everywhere
- All values shown to 3 decimal places
- No highlighting of key values
- Inconsistent alignment
```

**Problems:**
- Tables rarely work in presentations
- Too small to read
- No clear message
- Audience gives up trying to read
- Numbers without context

### ✅ Good Version

**Option 1: Bar Chart**
```
Title: "Method B Reduces Cost 40%" (36pt)

[Simple bar chart]
- Two bars: Method A ($100), Method B ($60)
- Direct labels on bars
- Difference highlighted with arrow
- Color: A = gray, B = accent
```

**Option 2: Simplified Table (if unavoidable)**
```
Title: "Method B is Faster and Cheaper"

[Minimal table: 3 rows × 3 columns]
              Method A    Method B
Time (min)      60          20  ← (highlighted)
Cost ($)       100          60  ← (highlighted)
Accuracy (%)    95          94

- No gridlines (whitespace for separation)
- Key values highlighted in accent color
- 20pt font minimum
- Only essential comparisons
```

**Why better:**
- Visual comparison is immediate
- Key differences emphasized
- Readable from distance
- Clear message

---

## Example 7: Section Transition

### ❌ Bad Version

```
Title: "Part 2: Results"

Content:
• Overview of findings
• Detailed analysis
• Statistical testing
• Interpretation
```

**Problems:**
- Looks like content slide
- Not visually distinct
- Generic bullets
- Doesn't signal major transition

### ✅ Good Version

```
[Slide with dark background or high contrast]

Content:
"Results" (centered, 52pt, bold)

Optional: Section number or icon

Nothing else on slide.
```

**Why this works:**
- Visually distinct from content slides
- Signals major transition
- Gives audience mental break
- Sets expectation for next section
- Bold and clean

---

## Example 8: Conclusions

### ❌ Bad Version

```
Title: "Conclusions"

Content:
• We showed that treatment affects response
• This is important for understanding mechanisms
• Future work should explore other conditions
• The method could be applied elsewhere
• Further studies are needed
• This has implications for the field
```

**Problems:**
- 6 bullets (too many)
- Repeats information from talk
- Generic statements
- No clear takeaway
- Unmemorable

### ✅ Good Version

```
Title: "Key Takeaways" (36pt)

Content (24pt, with generous spacing):
1. Treatment doubles response via novel pathway

2. Method B is 3× faster with equal accuracy

3. Approach generalizes to related systems

[Optional: Memorable closing image at bottom]
```

**Why this works:**
- Maximum 3 points (rule of three)
- Each is specific and concrete
- Rephrased (not copied from earlier)
- Memorable and actionable
- Clean and simple

---

## Example 9: Acknowledgments

### ❌ Bad Version

```
Title: "Acknowledgments"

Content:
Thanks to Dr. Smith, Dr. Jones, Dr. Williams, Dr. Brown,
Dr. Davis, Dr. Miller, Dr. Wilson, Dr. Moore, Dr. Taylor,
Dr. Anderson, and everyone in the lab. Funding from NIH
grant R01-12345, NSF grant 67890, Foundation X, Foundation
Y, and University Z. Thanks also to Core Facility A, B, C,
and D.

[List continues, 14pt font]
```

**Problems:**
- Wall of text
- Unreadable from distance
- Looks overwhelming
- Takes too long to display
- Doesn't honor contributions

### ✅ Good Version

```
Title: "Acknowledgments" (36pt)

Content:
Lab members (20pt, with photos if available)
[Photos/names of key contributors in grid]

Funding (18pt)
NIH, NSF, Foundation X

Core facilities (18pt)
Imaging, Flow Cytometry

Collaborators (18pt)
[Key collaborator names/institutions]

Clean layout with adequate spacing
```

**Why this works:**
- Organized in clear categories
- Readable from distance
- Honors specific contributions
- Professional appearance
- Not overwhelming

---

## Pattern Summary

### Effective Slides Follow These Patterns:

**1. Descriptive titles**
- ✅ "Treatment Doubles Response"
- ❌ "Results"

**2. Visual > Text**
- ✅ Large plot with minimal caption
- ❌ Bullets describing what plot would show

**3. Single message**
- ✅ One clear takeaway
- ❌ Multiple competing points

**4. Generous whitespace**
- ✅ 50% empty
- ❌ Filled to edges

**5. Large fonts**
- ✅ 18pt+ everywhere
- ❌ <16pt anywhere

**6. High contrast**
- ✅ Dark on light or light on dark
- ❌ Low contrast colors

**7. Consistent style**
- ✅ Same fonts/colors throughout
- ❌ Different style each slide

## Quick Design Test

For any slide, ask:

1. **3-second test**: Can someone understand the main message in 3 seconds?
2. **Distance test**: Is this readable from 30 feet away?
3. **Purpose test**: Does every element serve the message?
4. **Simplify test**: Could I remove anything and improve this?

If any answer is "no," redesign the slide.
