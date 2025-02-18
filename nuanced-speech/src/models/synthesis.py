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
            bypass_features: Style vector from bypass network [batch, 256]
        Returns:
            audio: Generated audio waveform
        """
        # Format bypass features to match Kokoro's expected shape [510, 1, 256]
        # Take first item from batch and repeat it 510 times
        voice_features = bypass_features[0].unsqueeze(0)  # [1, 256]
        voice_features = voice_features.repeat(510, 1, 1)  # [510, 1, 256]
        
        # Generate audio with formatted bypass features
        generator = self.pipeline(
            text, 
            voice=voice_features,
            speed=1.0
        )
        
        # Get the first (and only) audio segment
        for _, phonemes, audio in generator:
            break
            
        return audio, phonemes
    