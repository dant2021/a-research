import torch
import torch.nn as nn

class TransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # Self attention
        attn_output, _ = self.self_attn(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        # Feedforward
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        
        return x

class BypassNetwork(nn.Module):
    def __init__(self, whisper_hidden_dim=1280):
        super().__init__()
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(
                d_model=whisper_hidden_dim,
                nhead=12,
                dim_feedforward=4 * whisper_hidden_dim
            ) for _ in range(3)
        ])
        
        self.mlp = nn.Sequential(
            nn.Linear(whisper_hidden_dim, whisper_hidden_dim),
            nn.GELU(),
            nn.Linear(whisper_hidden_dim, whisper_hidden_dim)
        )

    def forward(self, encoder_output):
        x = encoder_output
        for layer in self.transformer_layers:
            x = layer(x)
        return self.mlp(x) 