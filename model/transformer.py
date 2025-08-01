import torch
import torch.nn as nn
from model.layers import CausalSelfAttention, FeedForward
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config')))
from config import config

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_hidden_dim, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, ffn_hidden_dim, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))  # Residual + attention
        x = x + self.ff(self.ln2(x))    # Residual + MLP
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vocab_size = config["vocab_size"]
        self.context_length = config["context_length"]
        self.embed_dim = config["embedding_dim"]

        # Token and positional embeddings
        self.token_embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.context_length, self.embed_dim))

        # Transformer blocks
        self.blocks = nn.Sequential(*[
            TransformerBlock(
                embed_dim=self.embed_dim,
                num_heads=config["num_heads"],
                ffn_hidden_dim=config["ffn_hidden_dim"],
                dropout=config["dropout"]
            )
            for _ in range(config["num_layers"])
        ])

        self.ln_f = nn.LayerNorm(self.embed_dim)
        self.head = nn.Linear(self.embed_dim, self.vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.shape
        assert T <= self.context_length, "Sequence length exceeds context window"

        tok_emb = self.token_embedding(idx)                    # (B, T, C)
        pos_emb = self.pos_embedding[:, :T, :]                 # (1, T, C)
        x = tok_emb + pos_emb                                  # (B, T, C)
        x = self.blocks(x)                                     # (B, T, C)
        x = self.ln_f(x)                                       # (B, T, C)
        logits = self.head(x)                                  # (B, T, vocab_size)

        return logits
