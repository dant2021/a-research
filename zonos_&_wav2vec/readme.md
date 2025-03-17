# Zero-Shot Voice Style Transfer: Weekly Research Report

## Overview
This week, my research focused on advancing zero-shot voice style transfer capabilities, with significant breakthroughs in temporal alignment, voice arithmetic, and feature extraction. I made important progress by replacing whisper with Zonos embeddings and discovered critical issues in my approach to phoneme duration prediction.

## Key Accomplishments

### Monday: Voice Arithmetic Success
I implemented vector arithmetic operations for voice style transfer:

Example:
```
whisper =  woman_whisper - woman_voice
man_whisper = man_voice + whisper
```
This approach demonstrated that style characteristics can be mathematically manipulated in embedding space, opening new avenues for style transfer without explicit training on target voices.

### Tuesday: Kokoro Integration and VAE Concept
A meeting with the creator of Kokoro provided valuable insights into its technical limitations. This inspired a new direction: training a Variational Autoencoder (VAE) to encode audio styles. The concept involves:
- Encoding a large corpus of diverse voice samples
- Creating a latent space representation of voice styles
- Using clustering to identify and extract meaningful style patterns

I also began integrating Zonos for speaker embedding, setting the foundation for better voice cloning capabilities.

### Wednesday: Zonos Feature Extraction
Successfully implemented the Zonos feature extraction pipeline and trained an adapter model to translate between Zonos speaker embeddings and my system. This represents a significant step toward more robust zero-shot capabilities, as Zonos provides decent speaker embeddings that capture voice characteristics.

### Thursday: Temporal Loss Breakthrough
Identified and fixed a critical bug in my temporal loss implementation.  I realized that rounding operations in the code were flattening gradients during backpropagation:

This simple fix dramatically improved the model's ability to learn appropriate temporal patterns.

### Friday-Saturday: WhisperX + Wav2vec Integration
Met with a highly skilled engineer who provided feedback on my approach. I continued developing the WhisperX + wav2vec integration for phoneme duration prediction. While initial implementation seemed successful, further testing revealed issues with gaps and overlaps in predictions.

## Technical Challenges

### Temporal Alignment Issues
My experiments confirm that temporal alignment is absolutely critical for effective style transfer. Misalignments between source and target utterances cause:
- STFT losses to break down
- Background noise in generated audio
- Poor style fit between reference and generated audio

Force-aligned durations produce significantly better results than learned durations, highlighting the importance of accurate phoneme timing.

### Phoneme Mapping Complexity
Integrating WhisperX with wav2vec faces substantial technical hurdles:
- Kokoro uses a non-standard phoneme set (49 phonemes)
- Creating accurate translations between wav2vec features and force aligning the required phonemes was tedious
- Gaps and overlaps in predicted features create inconsistencies in the duration model

## Results and Comparisons

The reference duration approach (`train_v11_ref_dur.py`) demonstrates promising zero-shot capabilities:
- Works well with voices from the Kokoro system that weren't seen during training
- Still struggles with unusual human voice characteristics that aren't in the base model
- Produces significantly better results than previous methods
- Generates decent zero-shot voice transfers, though still dependent on accurate duration information

### Future Work
1. Address gaps and overlaps in WhisperX + wav2vec predictions
2. Implement the VAE approach for style clustering and extraction
3. Improve zero-shot capabilities for diverse human voices

The code is available and runnable on 24GB VRAM although it's slightly spagetti land. 
