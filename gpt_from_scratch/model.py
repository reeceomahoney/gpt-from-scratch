import math

import torch
import torch.nn as nn
from dataclasses import dataclass
from gpt_from_scratch import tokenizer

TOKENIZERS = {"gpt2": tokenizer.GPT2Tokenizer, "byte": tokenizer.ByteTokenizer}


@dataclass
class GPTConfig:
    # Defaults describe GPT-2 small (~124M) on GPT-2 BPE.
    vocab_size: int = 50304  # 50257 padded up to a multiple of 64 for efficiency
    context_length: int = 1024
    d_model: int = 768
    d_layers: int = 12
    n_heads: int = 12
    d_feedforward: int = 768 * 4
    dropout: float = 0.0  # web-scale single pass: no overfitting to regularize
    tokenizer: str = "gpt2"


class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        d_model, n_heads = config.d_model, config.n_heads
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.dropout = config.dropout
        self.k = nn.Linear(d_model, d_model)
        self.q = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm_1 = nn.LayerNorm(d_model)
        self.resid_dropout_1 = nn.Dropout(config.dropout)

        self.up_proj = nn.Linear(d_model, config.d_feedforward)
        self.act = nn.GELU()
        self.down_proj = nn.Linear(config.d_feedforward, d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.resid_dropout_2 = nn.Dropout(config.dropout)

    def _split_heads(self, x):
        B, T, _ = x.shape
        # (B, T, d_model) -> (B, n_heads, T, head_dim)
        return x.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, x, attn_mask=None):
        x = x + self._attention(self.norm_1(x), attn_mask)
        x = x + self._feedforward(self.norm_2(x))
        return x

    def _attention(self, x, attn_mask):
        B, T, _ = x.shape
        k = self._split_heads(self.k(x))  # (B, H, T, head_dim)
        q = self._split_heads(self.q(x))
        v = self._split_heads(self.v(x))

        dropout_p = self.dropout if self.training else 0.0
        if attn_mask is not None:
            # Explicit (B, T, T) mask (padding + causal), used by the string path.
            attn = nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask[:, None], dropout_p=dropout_p
            )
        else:
            # No padding: SDPA builds the causal mask itself, enabling flash kernels.
            attn = nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=dropout_p
            )  # (B, H, T, head_dim)

        # (B, H, T, head_dim) -> (B, T, d_model)
        attn = attn.transpose(1, 2).reshape(B, T, self.d_model)
        return self.resid_dropout_1(self.out_proj(attn))

    def _feedforward(self, x):
        return self.resid_dropout_2(self.down_proj(self.act(self.up_proj(x))))


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.cfg = config
        self.tokenizer = TOKENIZERS[config.tokenizer]()
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embedding = nn.Embedding(config.context_length, config.d_model)
        self.embed_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            nn.ModuleList([TransformerBlock(config) for _ in range(config.d_layers)])
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.final_linear = nn.Linear(config.d_model, config.vocab_size)
        self.final_linear.weight = self.embedding.weight
        self.apply(self._init_weights)

        # GPT-2 residual scaling: shrink residual-stream writes by 1/sqrt(2*n_layers)
        # so activation variance stays ~constant with depth.
        residual_std = 0.02 / math.sqrt(2 * config.d_layers)
        for name, p in self.named_parameters():
            if name.endswith(("out_proj.weight", "down_proj.weight")):
                nn.init.normal_(p, mean=0.0, std=residual_std)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"GPT initialized with {n_params / 1e6:.2f}M parameters")

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

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
        x = self.embed_dropout(x)
        for block in self.blocks:
            x = block(x, attn_mask)
        return self.final_linear(self.final_norm(x))

    def forward(
        self, tokens: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Loss for pre-tokenized, densely-packed (B, T) batches.

        No padding mask: attention is causal via SDPA's is_causal path. This is
        the hot path for web-scale training, and the entry point DDP drives.
        """
        logits = self.forward_tokens(tokens)  # (B, T, vocab)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
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
