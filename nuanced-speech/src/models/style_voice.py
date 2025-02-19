import torch
import torch.nn as nn

class StyleVoice(nn.Module):
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

    def forward(self, encoder_output, word_boundaries):
        """
        encoder_output: [batch, seq_len, 1280]
        word_boundaries: List of (start, end) in frames
        """
        # Process encoder output
        x = encoder_output
        for layer in self.transformer_layers:
            x = layer(x)
        style_features = self.style_proj(x)  # [batch, seq_len, 256]
        
        # Average features within each word boundary
        batch_style = []
        for batch_idx in range(style_features.shape[0]):
            word_style = []
            for start, end in word_boundaries:
                # Get features for this word in current batch
                word_vec = style_features[batch_idx, start:end].mean(dim=0)
                word_style.append(word_vec)
            batch_style.append(torch.stack(word_style))
        
        return torch.stack(batch_style)  # [batch, num_words, 256] 