import torch


class ByteTokenizer:
    vocab_size = 256

    def encode(self, text: str) -> torch.Tensor:
        return torch.frombuffer(
            bytearray(text.encode("utf-8")), dtype=torch.uint8
        ).long()

    def decode(self, tokens: torch.Tensor) -> str:
        return bytes(tokens.tolist()).decode("utf-8", errors="replace")
