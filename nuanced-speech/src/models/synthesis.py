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
            bypass_features: Style vector [batch, 256] to replace the nth token's style
        Returns:
            audio: Generated audio waveform
        """
        # Load the default voice tensor
        voice_tensor = self.pipeline.load_voice(self.voice)
        
        # Replace the nth position with our bypass features
        # Take first item from batch
        for i in range(bypass_features.shape[0]):
            voice_tensor[i] = bypass_features[i]  
        
        # Generate audio with modified voice tensor
        generator = self.pipeline(
            text, 
            voice=voice_tensor,
            speed=1.0
        )
        
        # Get the first (and only) audio segment
        for _, phonemes, audio in generator:
            break
            
        return audio, phonemes
    