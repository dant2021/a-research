# Full-Stack Bypass Transformer (v2)

A parameter-efficient transformer enhancement that creates a high-bandwidth "bypass channel" for cross-layer information flow through compressed latent representations.

![Screenshot 2025-06-13 155417](https://github.com/user-attachments/assets/5be98d75-9a97-4b51-ae31-7ddce2f0fb61)
## Architecture Overview

This project implements a **full-stack bypass mechanism** that allows all transformer layers to share compressed global state information, enabling better self-consistency and improved reasoning capabilities.

### Core Innovation

**Full-Stack Information Flow**: Unlike traditional transformers where information flows sequentially layer-by-layer, our bypass creates a "global communication channel":

- **Encoder**: Extracts and compresses hidden states from ALL layers (L0-L27) → 256-dim latent
- **Projectors**: 28 unique modules that decompress the shared latent → layer-specific biases  
- **Injection**: Each layer receives a tailored bias based on the global model state

### Two-Stage Training Protocol

**Stage 1 - Reconstruction Pre-training**: 
- Train the bypass to reconstruct each layer's hidden states from the compressed global representation
- Ensures the bypass learns meaningful compression of the model's internal processing

**Stage 2 - Task Fine-tuning**:
- Freeze the encoder, fine-tune only the projectors on the target task
- Leverages the learned compression to improve task performance

### Why This Matters

Traditional transformers process information sequentially, but many reasoning tasks benefit from **global consistency** across all processing stages. Our bypass mechanism:

- **Enables non-local reasoning**: Early layers can access later layer more processed KV cache
- **Maintains parameter efficiency**: <1% trainable parameters (bypass modules only)
- **Preserves base model**: Original transformer remains completely frozen

The architecture demonstrates that compressed global state sharing can significantly enhance transformer reasoning with minimal computational overhead.

---

**Note**: For the original layer-offset approach (V0), see the V0 README in this directory. 
