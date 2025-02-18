import torch
import torch.nn as nn

class BypassNetwork(nn.Module):
    def __init__(self, whisper_hidden_dim=1280, style_dim=256):
        super().__init__()
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=whisper_hidden_dim,
                nhead=16,
                dim_feedforward=whisper_hidden_dim * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True  # Important for [batch, seq, features] format
            ) for _ in range(3)
        ])
        
        # Project from whisper dimension to style dimension
        self.style_proj = nn.Linear(whisper_hidden_dim, style_dim)

        self.mlp = nn.Sequential(
            nn.Linear(style_dim, style_dim),
            nn.GELU(),
            nn.Linear(style_dim, style_dim)
        )

    def forward(self, encoder_output):
        # encoder_output: [batch, seq_len, 1280]
        x = encoder_output
        
        # Pass through transformer layers
        for layer in self.transformer_layers:
            x = layer(x)
            
        # Project to style dimension
        style_features = self.style_proj(x)  # [batch, seq_len, 256]
        
        return style_features 