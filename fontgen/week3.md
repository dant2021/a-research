# So... I Built a Font Generator
**TL;DR:** I built a tool that turns handwriting samples into functional fonts through a pipeline of image processing, AI-powered character recognition, and font engineering. It works surprisingly well, and you can try it now at [v0-ai-font-generator-beige.vercel.app](https://v0-ai-font-generator-beige.vercel.app/).

  <picture width="600px" align="center">
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/75c820a8-410b-415f-9cb4-bd7293982ab2">
    <img alt="frontend" width="600px" align="center">
  </picture>

## Why This Matters

I was sitting next to Dillon in SF and observed something strange. He needed to make slides and logos for his startup. Instead of painstakingly finding a font and designing, he would prompt AI to generate a font for him. He ended up discarding most images because the text had a misspelling or the colors were off. 

For years, creating custom fonts meant painstaking manual tracing of each character. With recent advances in AI, I figured: **why not automate the entire process?**

## How It Works: The Technical Pipeline


### 1. Image Cleanup
  <picture width="300px" align="center">
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/1670afc9-0ed5-4ba1-873f-75318f8cf26d">
    <img alt="bitmap_threshold" width="300px" align="center">
  </picture>

```
Upload → Threshold to B&W → Clean bitmap
```
Your messy handwriting sample transforms into clean, crisp black and white pixels using OpenCV's thresholding magic.

### 2. Path Creation 
```
Bitmap → Vector Paths → Bounding Boxes
```
Potrace converts those pixels into smooth SVG paths. This turned jagged edges into aestetic curves that scale perfectly.

### 3. Character Assembly

  <picture width="400px" align="center">
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/dceb901c-eef1-4124-a729-fd9c5f15ac78">
    <img alt="bounding_boxes" width="400px" align="center">
  </picture>

```
Group paths → Merge components → Identify glyphs
```

The system looks for paths that belong together (like dots over "i"s) and combines them into complete characters. This was one of the trickiest parts to get right.

### 4. Smart Baseline Detection
  <picture width="400px" align="center">
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/d196879d-cf1c-4dca-a395-84191fe8b38f">
    <img alt="baselines" width="400px" align="center">
  </picture>
  
```
Group into lines → Find descenders → Set baselines
```
The system learns where your writing sits on the line and identifies which letters dip below (g, j, p, q, y). This required some clever statistical methods:

> **Tech Detail:** For datasets with enough samples, it calculates quartiles and IQR to find outliers. For smaller samples, it uses averages and proportional thresholds.

### 5. AI-Powered Recognition
  <picture width="500px" align="center">
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/eb191019-1c30-4940-b53b-32eb5a0ba37a">
    <img alt="baselines" width="500px" align="center">
  </picture>

```
Grid arrangement → Vision model analysis → Character labeling
```
Characters get arranged in a grid, then a vision language model identifies each one. The system keeps the best example when duplicates exist.

### 6. Font Engineering
```
Import SVG paths → Scale & align → Generate font files
```
FontForge takes over, precisely positioning each character with proper spacing and alignment before generating industry-standard TTF and OTF files.

## The Hard Problems (And My Hacky Solutions)

### Descenders
Letters like "g" and "y" need special treatment. My solution uses statistical outlier detection:

```python
# When enough samples exist:
	Make a baseline
	Measure outlier distance from baseline
	Apply this scaled distance when placing the glyphs

```

**Result:** Characters align naturally along the baseline, with proper descenders that don't look awkwardly positioned.

---


### The Dot Connection Challenge
Which dot belongs to which "i"? This continues to be a pain point. The current approach uses proximity thresholds:

```python
# Simplified logic:
if (close_horizontally and small_vertical_gap and reasonable_distance):
    # Connect this dot to its base character
```

**Problem:** Sometimes a descender like "y" gets mistakenly connected to characters on the line below. It works about 85% of the time, but that remaining 15% can produce some weird-looking characters.

---

### Spacing That Feels Natural
Font spacing is harder than it looks. Current approach:

- **Tracking:** Consistent 100 EM units between characters
- **Kerning:** Adjusts based on visual gaps between specific letter pairs

**Current Issue:** The text feels oddly "alive" when typing, as spacing adjusts between different letter combinations. It's technically correct but feels unnatural.

## Where It Stands & What's Next

### What Works Now
- ✓ Processes handwriting samples into working fonts
- ✓ Supports basic Latin characters and numbers
- ✓ Exports standard TTF/OTF files
- ✓ Simple, functional interface

### The Roadmap Ahead
1. **Better Dot Detection** — Smarter connection of dots and special characters
2. **Natural-Feeling Spacing** — Fix the "live text" feeling with improved kerning
3. **Character Coverage Feedback** — Show users which letters weren't detected
4. **AI-Powered Refinement** — "Make my 'g' look more consistent with my 'q'"
5. **Enhanced UI** — Preview fonts in different contexts before downloading

## The Philosophy Behind It

I built this system to be **modular by design**. Each component can be improved independently without breaking the whole pipeline. This means:

- You get a working font generator today
- Improvements arrive incrementally 
- The system gets better over time

## Try It Yourself!

The live version is available at [v0-ai-font-generator-beige.vercel.app](https://v0-ai-font-generator-beige.vercel.app/). Upload a sample of handwriting and see your personal font in minutes.

Feedback welcome! this is very much a V0!
