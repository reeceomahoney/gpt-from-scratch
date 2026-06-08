import functools

import tiktoken
import torch


class ByteTokenizer:
    vocab_size = 256

    def encode(self, text: str) -> torch.Tensor:
        return torch.frombuffer(
            bytearray(text.encode("utf-8")), dtype=torch.uint8
        ).long()

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

    def decode(self, tokens: torch.Tensor) -> str:
        return self._enc.decode(tokens.tolist())
