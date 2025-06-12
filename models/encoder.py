import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, num_heads=4, num_layers=6, projection_dim=128, layer_norm_eps=1e-6):
        super().__init__()
        self.layers = nn.ModuleList()
        transformer_units = [projection_dim * 2, projection_dim]
        
        for _ in range(num_layers):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(projection_dim, eps=layer_norm_eps),
                nn.MultiheadAttention(projection_dim, num_heads, dropout=0.1, batch_first=True),
                nn.LayerNorm(projection_dim, eps=layer_norm_eps),
                nn.Sequential(
                    nn.Linear(projection_dim, transformer_units[0]),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(transformer_units[0], transformer_units[1]),
                    nn.GELU(),
                    nn.Dropout(0.1),
                )
            ]))

    def forward(self, x):
        for norm1, attn, norm2, ff in self.layers:
            x1 = norm1(x)
            attn_output, _ = attn(x1, x1, x1)
            x2 = x + attn_output
            x3 = norm2(x2)
            x3 = ff(x3)
            x = x2 + x3
        return x