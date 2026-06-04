import math
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int = 8
    d_model: int = 128
    d_layers: int = 2
    d_feedforward: int = 128 * 4


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, d_feedfoward):
        super().__init__()
        self.d_model = d_model

        self.k = nn.Linear(d_model, d_model)
        self.q = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.norm_1 = nn.LayerNorm(d_model)

        self.up_proj = nn.Linear(d_model, d_feedfoward)
        self.down_proj = nn.Linear(d_feedfoward, d_model)
        self.norm_2 = nn.LayerNorm(d_model)

    def forward(self, x):
        k, q, v = self.k(x), self.q(x), self.v(x)
        attn = nn.functional.softmax((q @ k.T) / math.sqrt(self.d_model), dim=-1) @ v
        x += self.norm_1(attn)
        x += self.norm_2(self.down_proj(self.up_proj(x)))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.cfg = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(
            config.d_layers * [TransformerBlock(config.d_model, config.d_feedforward)]
        )
        self.final_linear = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, tokens):
        x = self.embedding(tokens)
        for block in self.blocks:
            x = block(x)
        x = self.final_linear(x)
        probs = nn.functional.softmax(x, dim=-1)
        return probs
