import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttnTransformerBlock(nn.Module):
    def __init__(self, latent_dim, cond_dim, n_head, ff_size, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(latent_dim, n_head, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(latent_dim, n_head, dropout=dropout)


        self.ln1 = nn.LayerNorm(latent_dim)
        self.ln2 = nn.LayerNorm(latent_dim)
        self.ln3 = nn.LayerNorm(latent_dim)

        self.dropout = nn.Dropout(dropout)

        self.ff = nn.Sequential(
            nn.Linear(latent_dim, ff_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, latent_dim),
            nn.Dropout(dropout)
        )

        # 只有在 cond_dim ≠ latent_dim 时才添加投影层
        if cond_dim != latent_dim:
            self.cond_proj = nn.Linear(cond_dim, latent_dim)
        else:
            self.cond_proj = nn.Identity()

    def forward(self, x, cond, padding_mask=None, cond_padding_mask=None):
        # x: (b, l, d)
        b, l, d = x.shape

        # Step 1: transpose x to (l, b, d)
        x = x.transpose(0, 1)


        # Self-attn
        residual = x
        x = self.ln1(x)
        x, _ = self.self_attn(x, x, x, key_padding_mask=padding_mask)
        x = residual + self.dropout(x)

        # Cross-attn
        residual = x
        x = self.ln2(x)
        cond = self.cond_proj(cond)
        if cond.dim() == 2:
            cond = cond.unsqueeze(1)
        cond = cond.transpose(0, 1)  # (c_len, b, d)
        x, _ = self.cross_attn(x, cond, cond, key_padding_mask=cond_padding_mask)
        x = residual + self.dropout(x)

        # FF
        x = x.transpose(0, 1)  # back to (b, l, d)
        residual = x
        x = self.ln3(x)
        x = residual + self.ff(x)

        return x


class CrossAttnTransformer(nn.Module):
    def __init__(self, num_layers, latent_dim, cond_dim, n_head, ff_size, dropout, max_len=200):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossAttnTransformerBlock(latent_dim, cond_dim, n_head, ff_size, dropout)
            for _ in range(num_layers)
        ])
        # 可学习的位置编码
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, latent_dim))

    def forward(self, x, cond, padding_mask=None, cond_padding_mask=None):
        """
        x: (b, l, latent_dim)
        cond: (b, cond_dim) or (b, c_len, cond_dim)
        padding_mask: (b, l) - True 表示是 padding
        cond_padding_mask: (b, c_len) - True 表示是 padding
        """
        # 加位置编码
        x = x + self.pos_emb[:, :x.size(1)]

        for layer in self.layers:
            x = layer(x, cond, padding_mask, cond_padding_mask)
        return x
