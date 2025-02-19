import torch
import torch.nn as nn

class StyleEncoder(nn.Module):
    def __init__(self, whisper_dim=1280, style_dim=256):
        super().__init__()
        
        # Simple transformer encoder layer
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=whisper_dim,
            nhead=8,
            dim_feedforward=2048,
            batch_first=True
        )
        
        # Project to style dimension
        self.projection = nn.Linear(whisper_dim, style_dim)
        
    def forward(self, whisper_features, word_timestamps):
        # Process through transformer
        encoded = self.encoder_layer(whisper_features)
        
        # Extract word-level features using timestamps
        word_features = []
        for chunk in word_timestamps:
            start_frame = int(chunk['timestamp'][0] * whisper_features.shape[1] / 30)
            end_frame = int(chunk['timestamp'][1] * whisper_features.shape[1] / 30)
            word_feat = encoded[:, start_frame:end_frame].mean(dim=1)
            word_features.append(word_feat)
            
        word_features = torch.stack(word_features, dim=1)
        
        # Project to style dimension
        style = self.projection(word_features)
        
        return style 