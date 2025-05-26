# Latent Embedding Architecture for Transformer Enhancement

A novel parameter-efficient approach to augment transformer models with cross-layer information flow through latent projections.

## Architecture Overview

This project implements a **two-pass transformer enhancement** that allows early layers to access processed information from deeper layers through a compact latent representation.

### Core Mechanism

**Layer Mapping**: Information flows backward in the network via a simple offset:
- Extract latents from layers L5-L25 
- Inject processed latents into layers L0-L20
- Mapping: L5→L0, L6→L1, L7→L2, ..., L25→L20

**Latent Compression**: 
- **Down projection**: 1024 dimensions → 128 dimensions (19.2x compression)
- **Up projection**: 128 dimensions → 1024 dimensions (reconstruction)
- Only these projection layers are trainable (~5.5M parameters, 0.92% of total)

### Residual Stream Integration

The latent information is added as **additive bias** to the MLP down projection outputs:

```
original_output = MLP.down_proj(x)
latent_bias = up_proj(down_proj(deeper_layer_output))
final_output = original_output + gate * latent_bias
```

This preserves the original residual stream while allowing controlled integration of processed information from deeper layers.

### Training vs Inference Modes

**Training (Two-Pass)**:
1. Pass 1: Extract latents from L5-L25 using current token
2. Pass 2: Apply latent bias to L0-L20 using current token latents

**Inference (Single-Pass)**:
- Use latents extracted from the previous token
- Maintains autoregressive generation while providing "cross-temporal" information flow

### Gating Mechanism

Each bias injection point has a learnable gate (sigmoid activation):
- Starts at 0.5 (neutral scaling)
- Learns layer-specific optimal scaling during training
- Allows the model to control how much processed information each layer receives

## Results

- **Parameter efficiency**: <1% trainable parameters
- **Stable training**: Adjustment ratios converge to reasonable values (~0.1-0.15)
- **Layer specialization**: Gates learn different values per layer, indicating selective use of processed information

## Implementation Details

**Base Model**: Qwen3-0.6B (frozen)
**Latent Dimension**: 128
**Hook-based Architecture**: Clean separation of extraction and injection logic
**Training Dataset**: GSM8K mathematical reasoning problems

The architecture demonstrates that early transformer layers can effectively utilize compressed representations of deeper processing with minimal parameter overhead.
