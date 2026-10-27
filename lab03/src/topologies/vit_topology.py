import torch
import torch.nn as nn

class UmudPatchTransformer(nn.Module):
    """
    Vision Transformer using Conv2d Patch Tokenizer
    Input: (B, 1, 28, 28)
    Tokenization: Conv2d(1, 64, kernel_size=7, stride=7) -> 16 tokens of embedding dim 64
    Transformer: 2 TransformerEncoderLayers (dim=64, nhead=4, dim_feedforward=128)
    Head: Mean-pooling over sequence tokens -> Linear(64, 10)
    """
    def __init__(self, in_channels=1, embed_dim=64, num_heads=4, num_layers=2, num_classes=10):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=7, stride=7)
        self.pos_embedding = nn.Parameter(torch.randn(1, 16, embed_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=128,
            activation="gelu",
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # x: (B, 1, 28, 28) -> patch_embed -> (B, 64, 4, 4)
        patches = self.patch_embed(x)
        B, C, H, W = patches.shape
        tokens = patches.flatten(2).transpose(1, 2)  # (B, 16, 64)
        tokens = tokens + self.pos_embedding
        encoded = self.encoder(tokens)
        encoded = self.norm(encoded)
        pooled = encoded.mean(dim=1)  # Mean pooling across 16 visual tokens
        return self.head(pooled)
