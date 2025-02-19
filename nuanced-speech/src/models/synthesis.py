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
        # Load voice tensor [1500, 256]
        voice_tensor = self.pipeline.load_voice(self.voice)
        print(voice_tensor.shape)
        print(bypass_features.shape)
        
        # Remove batch dimension and align lengths
        style_features = bypass_features[0]  # [seq_len, 256]
        seq_len = min(style_features.shape[0], voice_tensor.shape[0])
        
        # Replace first 'seq_len' slots
        voice_tensor[:seq_len] = style_features[:seq_len]
        
        # Generate audio
        generator = self.pipeline(text, voice=voice_tensor, speed=1.0)
        for _, phonemes, audio in generator:
            break
            
        return audio, phonemes
    