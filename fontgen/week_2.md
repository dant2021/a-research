## Weekly Progress Report — Font Generation & Representation

### Goal  
I’m building a font generator that lets users describe what they want and instantly get a fitting font—useful for websites, branding, or any creative use case.

---

### Training with 1000 Fonts  
Started with training on a dataset of 1000 fonts to build a model that understands stylistic variations. The main approaches explored:

- **Variational Autoencoder (VAE):**  
  - Simple 3-layer ConvNet with reparameterization.
  - It worked, but edges were blurry—unacceptable for font-quality reconstruction.

- **Autoencoder (AE) with Flux:**  
  - Used a pre-trained Flux model: 512×512×3 compressed to 128×128×16.
  - Output quality was much better, the edges got preserved.
  - Discovered that individual dimensions in the 16D latent space act as directional edge detectors (horizontal, vertical, diagonal), and also light/dark contrast filters.

  <picture width="400px" align="center">
      <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/ce292866-1ad5-40a7-9f5b-fe50e7cdb5fc">
      <img alt="side edge detection" width="400px" align="center">
    </picture>

    <picture width="400px" align="center">
      <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/6d7638c9-2c76-471d-b04b-6b72567b5be6">
      <img alt="color detection" width="400px" align="center">
    </picture>

---

### Generation Strategies

#### 1. **Latent Diffusion**  
Tried generating in the latent space.  
- Worked to some extent (see digits below) but training was slow and required too much data. There are only so many highquality fonts out there. 

![steps_00008911](https://github.com/user-attachments/assets/8cda1e4e-c53b-416a-b18b-7e2582cd9783)

#### 2. **Autoregressive on Embeddings**  
Inspired by GPT-4o image features, I tried autoregressive generation over compressed latent embeddings.

- Originally: 128×128×16 = 16,384 tokens.  
- Compressed via clustering to 4096 tokens of 64D each (4→1 with expanded dimension).
- Trained small transformers (few million parameters). It helped with speed, but:
  - Compression discarded key edge features.
  - Flux AE relies heavily on edge-sensitive channels, so quality dropped.
  - didnt get sensible outputs, but could get decent stuff with more training.

---

### Vector Arithmetic in Latents  
Tried interpolations and algebra in the Flux embedding space. Surprisingly robust results. Even weird combinations like `A + B - O` produced legible shapes:

  <picture width="400px" align="center">
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/a3c86b5b-5a03-4b59-8c4d-c1f955fd1d76">
    <img alt="A + B - O" width="400px" align="center">
  </picture>

---

### Switch to GPT-4o Image Prompting  
Rather than finish training an AR model, I pivoted to using GPT-4o image prompts as inspiration. It’s faster, more controllable, and already has refinement built in.

---

### SVG & FontForge Pipeline  
Once images are generated:
- Convert to SVG via Potrace.
- Merge related paths into glyphs using geometric proximity.
- Final output gets passed to FontForge to compile into usable fonts.

Still working on:
- Handling descender characters like `g`, `y`, `q`.
- Validating and filtering glyphs (OCR-based).
- De-duping and regenerating missing ones.

  <picture width="400px" align="center">
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/addb0266-1663-46d8-9102-3b6693ff7851">
    <img alt="Glyphs" width="400px" align="center">
  </picture>

---

### Bonus Learnings
- Got deep into Bezier curves—especially mapping between curves and raster pixels.
- Learned about perceptual detail loss when compressing AE tokens too aggressively.
- Discovered just how sensitive font quality is to minor pixel errors (more than regular image generation).
