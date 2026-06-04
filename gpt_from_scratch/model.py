import math
import torch
import torch.nn as nn
from dataclasses import dataclass
from gpt_from_scratch import tokenizer


@dataclass
class GPTConfig:
    vocab_size: int = tokenizer.ByteTokenizer.vocab_size
    context_length: int = 256
    d_model: int = 384
    d_layers: int = 6
    n_heads: int = 6
    d_feedforward: int = 384 * 4


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_feedfoward):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.k = nn.Linear(d_model, d_model)
        self.q = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm_1 = nn.LayerNorm(d_model)

        self.up_proj = nn.Linear(d_model, d_feedfoward)
        self.down_proj = nn.Linear(d_feedfoward, d_model)
        self.norm_2 = nn.LayerNorm(d_model)

    def _split_heads(self, x):
        B, T, _ = x.shape
        # (B, T, d_model) -> (B, n_heads, T, head_dim)
        return x.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, x, attn_mask=None):
        B, T, _ = x.shape
        k = self._split_heads(self.k(x))  # (B, H, T, head_dim)
        q = self._split_heads(self.q(x))
        v = self._split_heads(self.v(x))

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, H, T, T)
        if attn_mask is not None:
            scores = scores.masked_fill(~attn_mask[:, None], float("-inf"))
        attn = nn.functional.softmax(scores, dim=-1) @ v  # (B, H, T, head_dim)

        # (B, H, T, head_dim) -> (B, T, d_model)
        attn = attn.transpose(1, 2).reshape(B, T, self.d_model)
        attn = self.out_proj(attn)

        x = x + self.norm_1(attn)
        x = x + self.norm_2(self.down_proj(self.up_proj(x)))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.cfg = config
        self.tokenizer = tokenizer.ByteTokenizer()
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embedding = nn.Embedding(config.context_length, config.d_model)
        self.blocks = nn.ModuleList(
            nn.ModuleList(
                [
                    TransformerBlock(
                        config.d_model, config.n_heads, config.d_feedforward
                    )
                    for _ in range(config.d_layers)
                ]
            )
        )
        self.final_linear = nn.Linear(config.d_model, config.vocab_size)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"GPT initialized with {n_params / 1e6:.2f}M parameters")

    @property
    def device(self) -> torch.device:
        return self.embedding.weight.device

    def forward_tokens(self, tokens: torch.Tensor, attn_mask=None) -> torch.Tensor:
        """Run the transformer on a (B, T) tensor of token ids -> (B, T, vocab)."""
        T = tokens.size(1)
        assert T <= self.cfg.context_length, (
            f"sequence length {T} exceeds context_length"
        )

        pos = torch.arange(T, device=tokens.device)
        x = self.embedding(tokens) + self.pos_embedding(pos)  # (B, T, d_model)
        for block in self.blocks:
            x = block(x, attn_mask)
        return self.final_linear(x)

    def forward(self, inputs: list[str]):
        tokens, pad_mask = self.tokenizer.encode_batch(inputs)  # (B, T), (B, T)
        tokens = tokens.to(self.device)
        pad_mask = pad_mask.to(tokens.device)
        T = tokens.size(1)

        causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=tokens.device))
        attn_mask = causal[None] & pad_mask[:, None, :]  # (B, T, T)

        logits = self.forward_tokens(tokens, attn_mask)

        # Next-character prediction: predict token t+1 from positions up to t.
        shift_logits = logits[:, :-1].reshape(-1, logits.size(-1))
        targets = tokens[:, 1:].clone()
        targets[~pad_mask[:, 1:]] = -100
        loss = nn.functional.cross_entropy(
            shift_logits, targets.reshape(-1), ignore_index=-100
        )

        return logits, loss

    @torch.no_grad()
    def predict(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> str:
        was_training = self.training
        self.eval()
        tokens = self.tokenizer.encode(prompt)[None].to(self.device)  # (1, T)

        for _ in range(max_new_tokens):
            # Crop to the last context_length tokens so positions stay in range.
            idx = tokens[:, -self.cfg.context_length :]
            T = idx.size(1)
            causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=self.device))
            logits = self.forward_tokens(idx, causal[None])  # (1, T, vocab)

            logits = logits[:, -1, :] / temperature  # (1, vocab)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = nn.functional.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)
            tokens = torch.cat([tokens, next_token], dim=1)

        if was_training:
            self.train()
        return self.tokenizer.decode(tokens[0])
