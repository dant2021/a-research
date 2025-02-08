# Latent Thought LLM

An experimental approach to teaching language models to reason in latent space while maintaining text generation capabilities.

## Overview

This project explores a novel approach to language model reasoning by progressively training a model to operate in latent space for its reasoning steps while maintaining the ability to process text inputs and generate text outputs. The model learns to "think" using hidden state representations instead of token embeddings, potentially enabling more fluid and abstract reasoning capabilities.

## Key Features

- Progressive transition from text-based to latent-space reasoning
- Dual-mode operation: text processing and latent thought chains
- Maintains alignment between embedding and hidden state spaces
- Specialized loss functions for both text prediction and latent space alignment
- Automatic answer extraction using \boxed{} format

## Technical Architecture

### Key Components
![image](https://github.com/user-attachments/assets/567aa2f7-7230-4558-aadb-27333a33bae0)

- **Embedding Mixing**: Progressive scheduler controls the ratio between token embeddings and hidden states
- **Normalization**: RMS normalization with scale matching between embedding and hidden spaces
- **Masking**: Separate handling of prompt vs. reasoning tokens

### Loss Functions

1. **Text Generation Loss**
   - Cross-entropy loss on next token prediction
   - Applied only to reasoning/answer portions
   - Maintains text generation capability

2. **Latent Space Loss**
   - MSE + Cosine similarity between predicted and target hidden states
   - Guides hidden state predictions

## Dataset

Uses the filtered_reasoning_deepseek dataset, which contains:
- Mathematical reasoning problems
- Step-by-step solutions
- Final answers in \boxed{} format (can be extracted with regex)

You can find the dataset [here](https://huggingface.co/datasets/ant-des/filtered_reasoning_deepseek)

It's derived from the amazing[ R1 dolphin dataset](https://huggingface.co/datasets/cognitivecomputations/dolphin-r1)

### Training Process

The model employs a progressive training approach:
1. Initial phase: Standard text-based reasoning
2. Transition phase: Gradual shift to latent space reasoning
3. Final phase: Mixed operation using both text and latent representations

### Training Configuration

- Base Model: Llama-1.5B
- Max Sequence Length: 1024 tokens
- Learning Rate: 1e-5

## Initial Results

![image](https://github.com/user-attachments/assets/0dc1bb94-d5a4-4e33-ab8b-9da22af3818a)

Early experiments reveal:
- Non-zero MSE/Cosine losses show distinct but stable latent representations
- Lower cross-entropy loss suggests more expressive reasoning
- MSE/Cosine loss helps with allignment of latent thoughts

### Limitations

- These are early experiments need more validation
- Current training uses teacher forcing
- Unknown performance in pure generation tasks
