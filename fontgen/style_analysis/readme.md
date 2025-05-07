# AI Font Analysis with Vision-Language Models

## Overview
This project explores how large vision-language models (VLMs) perceive and distinguish different typefaces by extracting and analyzing font embeddings. By using Qwen2.5-VL-3B-Instruct, we create a mapping of typographic relationships, revealing how AI models understand font characteristics without explicit typography training.

## Key Findings

1. **Vision Models Understand Typography**: The model distinguishes fonts along multiple dimensions simultaneously:
   - Font weight (light, regular, bold)
   - Style classification (serif vs. sans-serif)
   - Functional purpose (display vs. body text)

2. **Embedding Visualization Insights**:
   - Bold variants cluster together regardless of font family
   - Serif fonts form distinct groups from sans-serif fonts
   - Monospace fonts cluster based on their fixed-width characteristics

3. **t-SNE vs. UMAP**:
   - t-SNE provides more stable and interpretable visualizations for font relationships
   - UMAP results can vary significantly with small changes to font position or rendering
   - Both techniques reveal similar high-level clustering patterns

4. **Character-Level Analysis Limitations**:
   - Character-level embeddings show less clear patterns than font-level analysis
   - The model appears to process typography holistically rather than as individual glyphs

## Methodology

The project uses the following approach:
1. Generate font grids of standard Latin characters from various fonts
2. Extract embeddings using Qwen2.5-VL vision-language model
3. Apply dimensionality reduction (PCA + t-SNE/UMAP) to visualize relationships
4. Analyze clustering patterns across font families, weights, and styles

## Implementation Details

- **Model**: Qwen2.5-VL-3B-Instruct
- **Fonts Analyzed**: 
  - Sans-serif: Arial, Calibri, Tahoma, Verdana, Segoe UI (regular, bold, italic variants)
  - Serif: Georgia, Times New Roman (regular, bold variants)
  - Monospace: Consolas, Courier (regular, bold variants)
  - Display: Comic Sans, Impact

- **Visualization**: 
  - Principal Component Analysis (PCA) for initial dimensionality reduction
  - t-SNE and UMAP for final 2D visualization

## Usage

The code provides methods to:
- Generate character grids from font files
- Extract embeddings using Qwen2.5-VL
- Visualize font relationships through dimensionality reduction
- Compare different font weights and styles

## Requirements
- PyTorch
- Transformers (Hugging Face)
- PIL (Pillow)
- scikit-learn
- UMAP
- matplotlib

This research demonstrates that large vision models implicitly learn structured typographic knowledge through their general visual training, which could enable new applications in font analysis, design, and generation.
