import torch
import torch.nn as nn
from kokoro import KPipeline

class KokoroWrapper(nn.Module):
    def __init__(self, voice='af_heart', lang_code='a'):
        super().__init__()
        self.pipeline = KPipeline(lang_code=lang_code)
        self.voice = voice
        
    def forward(self, text, style_features):
        """
        Args:
            text: Text from Whisper
            style_features: Style vectors [num_words, 256]
        """
        # Convert text to phonemes
        words = text.split()
        phonemes = []
        style_vectors = []
        
        # Map style features to phonemes
        for word_idx, word in enumerate(words):
            word_phonemes = self.pipeline.phonemize(word)
            for ph in word_phonemes:
                phonemes.append(ph)
                style_vectors.append(style_features[word_idx])
        
        # Create voice tensor
        voice_tensor = torch.stack(style_vectors)
        
        # Generate audio
        generator = self.pipeline(phonemes, voice=voice_tensor)
        for _, _, audio in generator:
            break
            
        return torch.from_numpy(audio) 