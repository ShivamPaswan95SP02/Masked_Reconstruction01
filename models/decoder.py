import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, num_layers=2, num_heads=4, enc_projection_dim=128, 
                 dec_projection_dim=64, image_size=48, layer_norm_eps=1e-6):
        super().__init__()
        self.proj = nn.Linear(enc_projection_dim, dec_projection_dim)
        self.layers = nn.ModuleList()
        transformer_units = [dec_projection_dim * 2, dec_projection_dim]
        num_patches = (image_size // 6) ** 2  # Assuming patch_size=6

        for _ in range(num_layers):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dec_projection_dim, eps=layer_norm_eps),
                nn.MultiheadAttention(dec_projection_dim, num_heads, dropout=0.1, batch_first=True),
                nn.LayerNorm(dec_projection_dim, eps=layer_norm_eps),
                nn.Sequential(
                    nn.Linear(dec_projection_dim, transformer_units[0]),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(transformer_units[0], transformer_units[1]),
                    nn.GELU(),
                    nn.Dropout(0.1),
                )
            ]))

        self.norm = nn.LayerNorm(dec_projection_dim, eps=layer_norm_eps)
        self.head = nn.Sequential(
            nn.Linear(dec_projection_dim * num_patches, image_size * image_size * 3),
            nn.Sigmoid()
        )
        self.image_size = image_size

    def forward(self, x):
        x = self.proj(x)

        for norm1, attn, norm2, ff in self.layers:
            x1 = norm1(x)
            attn_output, _ = attn(x1, x1, x1)
            x2 = x + attn_output
            x3 = norm2(x2)
            x3 = ff(x3)
            x = x2 + x3

        x = self.norm(x)
        x = x.flatten(start_dim=1)
        x = self.head(x)
        x = x.view(-1, 3, self.image_size, self.image_size)
        return x