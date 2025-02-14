# Multi-Token Prediction Experiments

A personal exploration into predicting multiple future tokens simultaneously, inspired by the DeepSeek paper's architecture.

## The Idea
While most approaches focus on predicting just the next token, I wanted to explore what happens when we try to predict multiple tokens ahead. The Meta/DeepSeek paper mentioned this architecture, and I was particularly curious about the consistency of predictions across different time steps.

## Implementation Notes
The core idea is relatively simple: take a frozen LLaMA model and attach multiple prediction heads, each trying to predict tokens at different positions in the future (t+1, t+2, t+3, t+4). The architecture looks like this:

![image](https://github.com/user-attachments/assets/364460de-ed66-4ac3-95c4-945c0c3b76b5)

Each head gets the same input but learns to predict further into the future. I scaled down the gradients for the further-future predictions by 0.1, thinking this might help balance the learning process.

## Key Findings

1. **Surprising Prediction Quality**: 
   - Simply switching out the last layer allowed surprisingly accurate predictions up to t+4
   - While t+1 predictions were best, t+4 predictions maintained unexpected coherence
   - This suggests the n-1 layer already contains significant future information
![image](https://github.com/user-attachments/assets/4241c301-7252-43c8-9296-1cb32ff99114)

2. **Frozen vs Unfrozen Performance**:
   - Counterintuitively, keeping the trunk frozen produced better results (although this could be because I'm GPU poor)
   - I'm curious what this means for planning capabilities in current models

3. **Output Analysis**:
   - At step 2000, predictions show coherent mathematical reasoning
   - While not perfect, even t+4 predictions maintain meaningful structure
![image](https://github.com/user-attachments/assets/2027bdfa-2438-4f09-95fd-9a6c9ce52717)

## Technical Details

### Architecture:
- Base Model: LLaMA 1.5B
- Multiple prediction heads (copies of last layer)
- Frozen embedding layer and LLaMA model
- Manual gradient accumulation for shared representations

### Training Setup:
- Dataset: Filtered reasoning dataset (5000 examples)
- Learning rate: 1e-4
- Single T4 GPU

## Future Directions

Key areas for investigation:
1. Deeper analysis of prediction consistency across time steps
2. Exploring implications for planning capabilities
3. Investigating the information content of layer n-1

## Conclusions

The results challenge my intuitions about multi-token prediction. The ability to predict multiple steps ahead with just last-layer modifications, and the superior performance of frozen models, suggests interesting properties about how future information is encoded in transformer layers.

The code is available in this repository for anyone interested in exploring these surprising findings further.
