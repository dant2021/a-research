# Zonos-Kokoro Voice Style Transfer: My Weekly Research Report

## Overview
This week, I made significant progress on my zero-shot voice style transfer system by combining Zonos embeddings with Kokoro. I discovered important insights about temporal alignment and improved my approach to feature extraction.

## Key Accomplishments

### Monday: Voice Arithmetic Success
I implemented voice style transfer through vector operations:

```
whisper = woman_whisper - woman_voice
man_whisper = man_voice + whisper
```

This shows I can extract and transfer style elements mathematically without specific training.

### Tuesday: Kokoro Insights and VAE Concept
I met with Kokoro's creator and learned about its technical limitations. This inspired my new approach: a Variational Autoencoder for audio styles that will:
- Process diverse voice samples
- Create a style latent space
- Enable clustering of similar voice characteristics

I also began integrating Zonos for better speaker embeddings.

### Wednesday: Zonos Feature Integration
I got the Zonos feature extraction working and trained an adapter model between Zonos embeddings and my system. This gives me better quality speaker representations for voice cloning.

### Thursday: Fixing the Temporal Loss
I found why my temporal loss wasn't improving - a rounding operation was flattening my gradients during backpropagation. Fixing this simple bug dramatically improved timing pattern learning.

### Friday-Saturday: WhisperX + Wav2vec Work
I discussed my approach with another engineer and continued developing the WhisperX + wav2vec integration. Initial tests looked promising, but I later found issues with gaps and overlaps in the predictions.

## Technical Challenges

### Temporal Alignment Issues
My experiments confirm that temporal alignment is critical. When timing isn't precise:
- STFT losses break down
- Audio quality suffers with background noise
- Style transfer becomes inconsistent

Forced alignments consistently outperform learned durations.

### Phoneme Mapping Complexity
Integrating WhisperX with wav2vec presents several challenges:
- Kokoro's non-standard phoneme set (49 phonemes)
- Complex translation between different phoneme systems
- Prediction gaps and overlaps causing duration inconsistencies

## Results and Next Steps

My reference duration approach (`train_v11_ref_dur.py`) works well:
- Handles unseen Kokoro voices effectively
- Struggles with unusual human voice characteristics
- Significantly outperforms my previous methods
- Produces good zero-shot transfers with accurate durations

I plan to:
1. Fix the gaps in WhisperX + wav2vec predictions
2. Implement the VAE for style clustering
3. Improve handling of diverse human voices

The code runs on 24GB VRAM, though I'll admit the implementation could use some cleanup.
