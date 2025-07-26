import torch
import torch.nn as nn
import math

class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Register buffer for causal mask (1s below diagonal, 0 above)
        self.register_buffer("mask", torch.tril(torch.ones(512, 512))
                                     .unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        B, T, C = x.size()  # (batch, time, channels)

        qkv = self.qkv_proj(x)  # (B, T, 3C)
        qkv = qkv.view(B, T, self.num_heads, 3 * self.head_dim)
        q, k, v = qkv.chunk(3, dim=-1)

        # (B, heads, T, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Attention scores
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = torch.softmax(att, dim=-1)
        att = self.dropout(att)

        y = att @ v  # (B, heads, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(y)

class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
