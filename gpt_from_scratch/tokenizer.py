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
