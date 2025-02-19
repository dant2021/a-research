import torch
import torch.nn as nn
from kokoro import KPipeline

class KokoroSynthesizer(nn.Module):
    def __init__(self, voice='af_heart', lang_code='a'):
        super().__init__()
        self.pipeline = KPipeline(lang_code=lang_code)
        self.voice = voice
        
    def forward(self, text, style_features):
        """
        Args:
            text: Raw text from Whisper
            style_features: [num_words, 256]
        """
        # Convert text to words with phonemes
        words = self.pipeline.split_words(text)
        phonemes = []
        style_vectors = []
        
        # Map style features to phonemes
        for word_idx, word in enumerate(words):
            # Get phonemes for this word
            word_phonemes = self.pipeline.phonemize(word.text)
            
            # Assign same style vector to all phonemes in the word
            for ph in word_phonemes:
                phonemes.append(ph)
                style_vectors.append(style_features[word_idx])
        
        # Create voice tensor from aligned style vectors
        voice_tensor = torch.stack(style_vectors)  # [num_phonemes, 256]
        
        # Generate audio
        generator = self.pipeline(phonemes, voice=voice_tensor)
        for _, phonemes, audio in generator:
            break
            
        return audio, phonemes
    