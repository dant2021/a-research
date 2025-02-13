# Multi-Token Prediction Experiments

A personal exploration into predicting multiple future tokens simultaneously, inspired by the DeepSeek paper's architecture.

## The Idea

While most approaches focus on predicting just the next token, I wanted to explore what happens when we try to predict multiple tokens ahead. The Meta/DeepSeek paper mentioned this architecture, and I was particularly curious about the consistency of predictions across different time steps.

## Implementation Notes

The core idea is relatively simple: take a frozen LLaMA model and attach multiple prediction heads, each trying to predict tokens at different positions in the future (t+1, t+2, t+3, t+4). The architecture looks like this:

![image](https://github.com/user-attachments/assets/364460de-ed66-4ac3-95c4-945c0c3b76b5)

Each head gets the same input but learns to predict further into the future. I scaled down the gradients for the further-future predictions by 0.1, thinking this might help balance the learning process.

The implementation has some interesting details:
- The base LLaMA model is completely frozen (until step 200)
- The embedding layer is shared and frozen
- Each head is initialized with a copy of the last layer 
- Gradients are accumulated manually for the shared representations

## Challenges

Memory management turned out to be the main obstacle. I had to make some compromises:
- Couldn't implement proper gradient accumulation due to memory constraints
- Added a warm-up period (only backpropagating through the trunk after batch 200)
- Had to keep batch sizes smaller than ideal

## Current Results

The results are modest. Without proper gradient accumulation, the training wasn't as stable as I'd hoped. I suspect this significantly impacted the model's ability to learn consistent predictions across different time steps.

## Next Steps

If I revisit this experiment, I'd like to:
- Find a way to implement proper gradient accumulation
- Analyze the prediction consistency across time steps
- Try different scaling factors for the future predictions

## Technical Setup

- Base Model: LLaMA 1.5B
- Dataset: Filtered reasoning dataset from DeepSeek (5000 examples)
- Training: 3 epochs with learning rate 1e-4
- Hardware: Single GPU setup

The code is available in this repository for anyone interested in exploring similar ideas. Just be mindful of the memory requirements.

This has been an interesting exploration, even if the results weren't what I initially hoped for. Learned a lot about detaching gradients though.
