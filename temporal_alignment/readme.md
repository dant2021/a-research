# Zero-Shot Voice Style Encoder: Research Report

## Key Findings

After exploring multiple approaches to voice transfer, I've discovered that **temporal alignment is absolutely critical** for effective learning. When phoneme durations aren't correctly matched between source and target, the model struggles to learn meaningful style representations regardless of the sophistication of the loss function.

The most successful approach was using forced correct durations from reference voices, which yielded excellent results. This challenges my assumptions about style learning and suggests that temporal aspects of speech are fundamental.

## Problem Statement

This project addresses a critical limitation in current voice assistant technologies: the inability to understand the tone of voice of the user. Voice agents respond with the similar tone regardless of how the user feels. You would not respond the same way if someone shouted or whispered at you. 

I'm developing a zero-shot voice style encoder that can:
1. Extract style information from a speaker's voice using Whisper features
2. Generate responses in a similar speaking style using Kokoro speech synthesis 
3. Maintain natural prosody and style across different contexts

## Research Timeline

### Monday, March 3, 2025: Complex Losses Exploration

I started by investigating whether more sophisticated loss functions would improve style transfer. I implemented several approaches:

- **StyleLoss class** with multi-resolution STFT loss
- **Energy loss** to calculate frame-wise energy using sliding windows
- **Spectral loss** with high-frequency emphasis filters
- **Temporal loss** with a cumulative sum loss on each phoneme
- **Reference style comparison metric** this is a good metric for synthetic audio reconstruction assessment.

Despite the elegance of these approaches, the model kept plateauing during training. The losses were not providing a good learning signal, suggesting a more fundamental issue.

I also tried around with f0 crepe loss. It was not accurate enough.

### Tuesday, March 4, 2025: Codec Models and Latent Space Comparison

Shifting strategies, I explored comparing audio in the latent space using codec models (DAC). The hypothesis was that comparing compressed representations might provide a more stable learning signal.

I implemented:
- Latent space feature comparison with L1 loss
- Quantized representation loss
- Code-matching percentage metrics

This revealed interesting metrics for audio quality assessment (percentage of matching codes), but these approaches are extremely sensitive and didn't significantly improve learning.

### Wednesday, March 5, 2025: The Temporal Shift Discovery

This was the breakthrough day. Frustrated with plateauing models, I decided to add controlled noise to stable vectors to understand what made them break. Through these experiments, I made a critical discovery: **the model is extraordinarily sensitive to temporal shifts**.

Even slight misalignments in timing between source and target utterances were causing the loss functions to break down. This explained why all the sophisticated loss functions weren't helping - they were all susceptible to the fundamental issue of temporal misalignment.
```
Noise Type      Level   L1 Wave    L2 Wave    STFT L1    MEL L1     Log STFT L1 Log MEL L1 Codec Feature L1 Codec Feature L2 Codec Z L1 Codec Z L2 Codec Code Match
------------------------------------------------------------------------------------------------------------------------
time_stretch    0.00    0.0000     0.0000     0.0000     0.0000    0.0000         0.0000         0.0000         0.0000         0.0000         0.0000         1.0000         
time_stretch    0.05    0.0430     0.0047     3.2707     1.0764    1.4085         1.3064         0.6422         0.8190         1.8910         5.7889         0.0138         
time_stretch    0.10    0.0447     0.0047     3.3819     1.1293    1.6311         1.5531         0.6458         0.8173         1.8949         5.7818         0.0062         
time_stretch    0.50    0.0412     0.0043     3.1794     1.0762    1.8367         1.7420         0.6416         0.8052         1.8859         5.7406         0.0107         
time_shift      0.00    0.0000     0.0000     0.0000     0.0000    0.0000         0.0000         0.0000         0.0000         0.0000         0.0000         1.0000         
time_shift      0.05    0.0545     0.0069     4.9052     1.6048    1.8477         1.7350         0.6471         0.8226         1.9012         5.8092         0.0040         
time_shift      0.10    0.0557     0.0068     4.9760     1.6657    1.9019         1.8675         0.6590         0.8512         1.9275         5.9784         0.0078         
time_shift      0.50    0.0574     0.0067     4.9341     1.6825    1.9383         1.9136         0.6495         0.8308         1.9034         5.8204         0.0059         
gaussian        0.00    0.0000     0.0000     0.0000     0.0000    0.0000         0.0000         0.0000         0.0000         0.0000         0.0000         1.0000         
gaussian        0.05    0.0399     0.0025     2.1808     0.2079    2.8868         2.0924         0.4904         0.5480         1.4996         4.0741         0.1197         
gaussian        0.10    0.0798     0.0100     8.0497     0.7242    3.4428         2.6112         0.5715         0.6619         1.7092         4.8121         0.0299         
gaussian        0.50    0.3893     0.2286     176.12     15.055    4.7633         3.8896         0.6446         0.8084         1.8956         5.7672         0.0057  
```
### Thursday, March 6, 2025: Forced Duration Experiments

Building on yesterday's discovery, I tried forcing correct durations using a teacher model. This approach worked remarkably well! When the model was given the correct temporal alignment, it suddenly started learning effectively.

This confirmed my hypothesis that duration mismatches were the key problem. Weirdly enough a temporal loss did not work in my initial experiments. The model performance improved dramatically when phoneme durations were explicitly controlled rather than learned through a loss function. (which is still a mystery to me)

![image](https://github.com/user-attachments/assets/14922359-67fc-4ec8-bca3-456ae3496deb)

### Friday, March 7, 2025: Whisper and Wav2vec Features

With the importance of timing established, I explored using timestamped Whisper features combined with wav2vec features to create a solution that wouldn't require a teacher model.

This approach became complex very quickly, with significant challenges:
- Kokoro uses a different phoneme set (49 phonemes) than wav2vec
- Wav2vec doesn't support every text character, causing mismatches
- Converting from word-level features (Whisper) to character-level features (wav2vec) introduced additional complexity

While promising conceptually, the implementation revealed substantial technical hurdles.

### Saturday, March 8, 2025: StyleTTS2 Alignment Model Testing

I evaluated the StyleTTS2 alignment model as a potential solution for extracting correct durations. Unfortunately, the results were disappointing - the model produced duration values that were often significantly different from the Kokoro reference durations.

For example, comparing durations for identical phonemes showed major discrepancies:

```
Duration prediction:
------------------------------------------------------------
Index  Position        TTS2     Kokoro
------------------------------------------------------------
0      w               1.0      14.0    
1      ˈ               22.0     2.0     
2      ʌ               3.0      2.0     
3      n               1.0      1.0     
4      s               1.0      2.0     
5                      1.0      2.0     
6      ə               2.0      1.0     
7      p               4.0      2.0     
8      ˈ               3.0      2.0     
9      ɑ               2.0      3.0     
10     n               2.0      1.0     
11                     2.0      1.0     
12     ɐ               1.0      1.0     
13                     2.0      1.0     
14     t               3.0      2.0     
15     ˈ               1.0      3.0     
16     I               1.0      6.0        
...
```

These mismatches highlighted why the model wasn't performing well - the temporal alignment was simply not accurate enough for effective style transfer.

## Conclusions and Next Steps

This research has clearly demonstrated that temporal alignment is the critical foundation for effective voice style transfer. When durations are correctly matched, even simple loss functions (MSTFT) can produce excellent results.

**Key takeaway**: Future work should focus first on solving the alignment problem rather than developing increasingly complex loss functions.

The next phase of my research will primarily focus on obtaining accurate phoneme durations without requiring a teacher model, while maintaining the quality improvements seen with forced durations.
