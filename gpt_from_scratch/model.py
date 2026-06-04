import math
import torch
import torch.nn as nn
from dataclasses import dataclass
from gpt_from_scratch import tokenizer


@dataclass
class GPTConfig:
    vocab_size: int = tokenizer.ByteTokenizer.vocab_size
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

    def forward(self, x, attn_mask=None):
        k, q, v = self.k(x), self.q(x), self.v(x)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_model)  # (B, T, T)
        if attn_mask is not None:
            scores = scores.masked_fill(~attn_mask, float("-inf"))
        attn = nn.functional.softmax(scores, dim=-1) @ v
        x = x + self.norm_1(attn)
        x = x + self.norm_2(self.down_proj(self.up_proj(x)))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.cfg = config
        self.tokenizer = tokenizer.ByteTokenizer()
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(
            nn.ModuleList(
                [
                    TransformerBlock(config.d_model, config.d_feedforward)
                    for _ in range(config.d_layers)
                ]
            )
        )
        self.final_linear = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, inputs: list[str]):
        tokens, pad_mask = self.tokenizer.encode_batch(inputs)  # (B, T), (B, T)
        T = tokens.size(1)

        causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=tokens.device))
        attn_mask = causal[None] & pad_mask[:, None, :]  # (B, T, T)

        x = self.embedding(tokens)
        for block in self.blocks:
            x = block(x, attn_mask)
        x = self.final_linear(x)
        probs = nn.functional.softmax(x, dim=-1)
        return nn.functional.softmax(x, dim=-1)  # (B, T, vocab_size)
