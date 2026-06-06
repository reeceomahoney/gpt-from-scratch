import functools

import tiktoken
import torch
import torch.nn as nn


class ByteTokenizer:
    vocab_size = 256

    def encode(self, text: str) -> torch.Tensor:
        return torch.frombuffer(
            bytearray(text.encode("utf-8")), dtype=torch.uint8
        ).long()

    def encode_batch(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        seqs = [self.encode(t) for t in texts]
        lengths = torch.tensor([len(s) for s in seqs])
        tokens = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0)
        mask = torch.arange(tokens.size(1))[None, :] < lengths[:, None]
        return tokens, mask

    def decode(self, tokens: torch.Tensor) -> str:
        return bytes(tokens.tolist()).decode("utf-8", errors="replace")


@functools.lru_cache(maxsize=1)
def _gpt2_encoding() -> tiktoken.Encoding:
    # Cached so we build the BPE merge table once per process.
    return tiktoken.get_encoding("gpt2")


class GPT2Tokenizer:
    """GPT-2 byte-level BPE (50257 vocab) via tiktoken."""

    vocab_size = 50257

    def __init__(self):
        self._enc = _gpt2_encoding()
        self.eot_token = self._enc.eot_token  # 50256, document separator

    def encode(self, text: str) -> torch.Tensor:
        # encode_ordinary ignores special tokens in the text (treats them as bytes).
        return torch.tensor(self._enc.encode_ordinary(text), dtype=torch.long)

    def encode_batch(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        seqs = [self.encode(t) for t in texts]
        lengths = torch.tensor([len(s) for s in seqs])
        tokens = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0)
        mask = torch.arange(tokens.size(1))[None, :] < lengths[:, None]
        return tokens, mask

    def decode(self, tokens: torch.Tensor) -> str:
        return self._enc.decode(tokens.tolist())
