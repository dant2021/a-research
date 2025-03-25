Ok I'll reveal my obsession lately. 

# Latent Space
<p align="center">
  <img src="https://github.com/user-attachments/assets/246c0c0d-ee6e-4a23-89fd-fbc584212126" width="250">
</p>
I believe AI will not communicate using human words in the future. Human words are:
inexpressive, limited, verbose and imprecise. 
As a contrast, embeddings are rich, expressive, dense and exact. 

In the future, there will be more layers of abstraction than an LM head tokenizing words. AI is limited by the information density per token. 

In the future, they will chat in tokens/embeddings. Every model will need a translation. English / human text will be one of these standards. Another will be a common token space. This will have a set size that models will project to. Most likely something like the second last layer. Not yet transformed for human output. 

There will be a part of the embedding that gets **logprobed** and **softmaxed**. And a part that stays pure. These embeddings will still be rawer and more information dense than the words we use. 

**Making it concrete.** Try to describe someone’s voice so that someone could recreate it. It’s hard right? Yet with a tiny 256-dimensional embedding we can encode the entire style of someone’s voice. 

For reference, a llama 70B model uses a 8192 dimensional embedding. If I can encode a voice in a single 256dim vector, imagine how much data I could encode across thousands of tokens in a much larger embedding. 

We are killing LLM’s. Forcing them to use the **mind-numbing english language**. Instead, they could express themselves, much more precisely. 

How to get there. For this to work, we need to do two things. 
- Allow AI to see and use its own thoughts,
- Densify the human language. 

Allowing AI to have its own persistent language. These are the thoughts, I was speaking of. Imagine if we just added 1/4th of the dimensions used for human language that do not get logprobed and softmaxed. Basically, these thought embeddings could accumulate meaning across the autoregressive generation and not get wiped after every token. 

This introduces a time dependency, but we can just be lax about that. Instead of generating every token individually, sequentially. All tokens can still be generated and thoughts be used as a refinement instead of an essential sequential thing.

Regarding densifying the human language. In audio, there is this very used thing called codecs. These are fancy auto-encoders that take 100s of samples and turn them into one token which represents the audio accurately. Very often the speech models do not predict speech but these compressed tokens, that then get decompressed into audio. 

With text, we can encode multiple text tokens that follow each other into one dense token. This denser token could be used to train a model. This would enhance the quantity of text the model can be trained on and allow for cheaper inference. 

The future of artificial intelligence lies not in copying human linguistic patterns, but in developing native, hyper-efficient embeddings. These will transcend linguistic and technological limitations.

_"You can tell the RL is done properly when the models cease to speak English in their chain of thought."_ — Andrej Karpathy

---

## Indulgence over, here's what I did last week:

### Kokoro Phoneme Alligner
I made a wav2vec + whisperX + misaki to create precise phoneme-level alignments for audio files. The code is available here. 

Here's how it works:
- Word-Level Alignment: Uses WhisperX to transcribe audio and generate word-level timestamps
- Phoneme Generation: Converts words to phonemes using Kokoro's G2P (Grapheme-to-Phoneme) pipeline
- Phoneme-Level Alignment: Employs Wav2Vec2 forced alignment to determine precise timing for each phoneme
- Fallback Mechanisms: Implements robust fallbacks for challenging cases:
  - Handles missing trailing phonemes
  - Falls back to uniform distribution when precise alignment fails
  - Detects and reports gaps in word alignments
    
The system outputs detailed JSON files with timing information for each phoneme, along with diagnostic information to help identify potential issues in the alignment process. The tool is particularly useful for kokoro finetuning which requires precise phoneme timing.

It doesnt quite work, I'll discontinue my work here for now.

### Analysis of OpenAI vs Actors
I wanted to see how good OpenAI.fm really was. I compared it with the RAVDNESS dataset. 

Here are my discoveries:
- Real audio is still more expressive, especially for subtle emotions
- The randomness/hallucinations of OpenAI make for a more diffuse clustering
- Male and female emotions are different IRL, but evenly applied in openAI's models.

Actors           |  OpenAI
:-------------------------:|:-------------------------:
![mirror_pattern_analysis](https://github.com/user-attachments/assets/c5022b2b-4ac3-4fd9-b89b-fb324c036ea0)  | ![open-emotions](https://github.com/user-attachments/assets/2558ab19-2ff2-4991-a68f-ef3367b3d27c)


Learnings 
- UMAP is better than TSNE
- This was easier than expected.

### Making Dense Tokens
Autoencoder for embeddings

I built a dense token encoder for LLM embeddings. It works by:
- Grabbing Llama 3.2 activations (layer 15)
- Compressing them with RVQ (4 stages)

It's trained with MSE + cosine similarity loss and has codebook management to prevent dead codes.
![image](https://github.com/user-attachments/assets/47be0df6-c7e5-430a-a537-f4234c31c20f)
It is somewhat inspired by moshi, doesn't work yet. 
