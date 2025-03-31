

# Font VAE: Learning Typography in Latent Space

## Research Report

This report documents my exploration into font generation using Variational Autoencoders (VAEs). The project aims to create a system where users can describe or provide an image of a font and receive a complete, functional typeface in return.

## The Week That Was

Last week started with frustration as my remote GPU workstation did not work. I couldn't connect to it for days, which seriously hampered my productivity. Apparently an overly large folder (CUDNN) was causing a timeout on their system, should get access at some point this week. 

Between GPU battles, I built my personal website at [latembed.com](https://latembed.com/) using V0. It was a actually quite fun and fast.

As it was my final week in San Francisco, I found myself saying goodbye to many friends before heading back to London. Amid the farewell dinners, I had this seemingly random idea: what if I could build a font generator?

## Approach

I implemented a VAE architecture to learn a compressed representation of font glyphs. The model processes TTF/OTF files, rendering each character as a 64×64 pixel image using FreeType. These images are then encoded into a 256-dimensional latent space.

The architecture consists of:
- Convolutional encoder network
- Glyph-specific embeddings to help the model differentiate characters
- Convolutional decoder network

## Key Challenges

The most significant was blurriness of the glyps. Basically the details would get lost which makes sense as there is a 16x compression:

I tried remediating with residual connections. But essentially the model would "cheat" by bypassing the compressed latent space entirely. This defeated the purpose of the VAE, as the model wasn't learning a meaningful latent representation.

After removing these residual connections, the model was forced to encode meaningful information in the latent space, resulting in much better yet still poor mixing capabilities.

## Results

The model demonstrates promising capabilities in three key areas:

1. **Reconstruction**: The VAE can accurately reconstruct input glyphs, preserving the essential style characteristics of the original font.
![WhatsApp Image 2025-03-28 at 23 30 32_452cbd67](https://github.com/user-attachments/assets/0a883a0a-cfd4-4a06-9108-f6cf5cba6a89)
*Example of original glyphs (top) and their reconstructions (bottom)*

The model still very much stuggles with handwritten fonts or fonts with very fine lines.

2. **Interpolation**: It's not that good yet. But the model should enable smooth interpolation between different font styles in latent space, creating a continuous spectrum of typography.
![image](https://github.com/user-attachments/assets/b0ed13b8-fa0c-4017-8074-41fae67cde9d)

## Future Directions

This initial implementation serves as a proof of concept. Now that I'm back in London, I plan to:

1. Experiment with diffusion models as an alternative approach
2. If good enough implement text-to-font capabilities

## Technical Implementation

The current implementation uses PyTorch with a convolutional VAE architecture. Training employs a combination of reconstruction loss (binary cross-entropy) and KL divergence to balance accurate reconstruction with a well-structured latent space.

The model is trained on a diverse corpus of fonts, with data augmentation to improve generalization. Each training run consists of 100 epochs with periodic visualization and checkpointing.

## Conclusion

This project emerged from a week of technical frustrations, and demonstrates that deep generative models can effectively learn the underlying patterns of typography. While this is just the beginning, the results suggest exciting possibilities for AI-assisted font design and generation.
