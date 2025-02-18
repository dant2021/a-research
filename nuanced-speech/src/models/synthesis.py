import torch
import torch.nn as nn
from kokoro import KPipeline

class KokoroSynthesizer(nn.Module):
    def __init__(self, voice='af_heart', lang_code='a'):
        super().__init__()
        self.pipeline = KPipeline(lang_code=lang_code)
        self.voice = voice
        
    def forward(self, text, bypass_features):
        """
        Args:
            text: Text to synthesize
            bypass_features: Emotional/intonation features from Whisper encoder
        Returns:
            audio: Generated audio waveform
        """
        # Generate base audio with Kokoro
        generator = self.pipeline(
            text, 
            voice=bypass_features,
            speed=1.0
        )
        
        # Get the first (and only) audio segment
        for _, phonemes, audio in generator:
            break
            
        return audio, phonemes
    