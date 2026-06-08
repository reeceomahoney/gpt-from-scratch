"""Web-scale data pipeline: tokenize FineWeb-Edu into binary shards and stream
random windows from them at train time.

Prep (run once, ideally on the cluster — downloads/streams ~27GB):

    uv run gpt_from_scratch/data.py

Edit `PrepareConfig` to change the dataset, output dir, or shard sizes. This
writes uint16 token shards (`train_000000.bin`, ...) plus a held-out
`val.bin`. At train time, `TokenLoader` memmaps the shards and samples random
(B, T) windows — never loading the full corpus into memory.
"""

import multiprocessing as mp
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from gpt_from_scratch import tokenizer

# FineWeb-Edu vocab fits in uint16 (50257 < 65536), halving on-disk size.
DTYPE = np.uint16


@dataclass
class PrepareConfig:
    out_dir: str = "data/fineweb_edu_10bt"
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    dataset_config: str = "sample-10BT"
    num_proc: int | None = None  # None -> (cpu_count - 2)
    shard_tokens: int = 100_000_000  # tokens per train shard (~200MB on disk)
    val_tokens: int = 10_000_000  # held out for validation


def _tokenize(doc: dict) -> np.ndarray:
    """Tokenize one document, prefixing the EOT separator."""
    enc = tokenizer.GPT2Tokenizer()
    ids = [enc.eot_token, *enc._enc.encode_ordinary(doc["text"])]
    arr = np.array(ids, dtype=np.uint32)
    assert (arr < 2**16).all(), "token id exceeds uint16 range"
    return arr.astype(DTYPE)


def _write_shard(path: Path, tokens: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tokens.tofile(path)
    print(f"  wrote {path.name}: {len(tokens):,} tokens")


def prepare(config: PrepareConfig) -> None:
    """Stream the dataset, tokenize in parallel, and write token shards.

    The first `val_tokens` tokens become `val.bin`; the rest are split into
    `shard_tokens`-sized `train_*.bin` files.
    """
    import datasets

    out = Path(config.out_dir)
    num_proc = config.num_proc or max(1, (os.cpu_count() or 2) - 2)
    print(
        f"streaming {config.dataset_name}:{config.dataset_config} "
        f"with {num_proc} workers"
    )

    ds = datasets.load_dataset(
        config.dataset_name, name=config.dataset_config, split="train", streaming=True
    )

    buf = np.empty(config.shard_tokens, dtype=DTYPE)
    filled = 0
    shard_idx = 0
    wrote_val = False

    with mp.Pool(num_proc) as pool:
        for ids in pool.imap(_tokenize, ds, chunksize=16):
            pos = 0
            while pos < len(ids):
                take = min(len(ids) - pos, len(buf) - filled)
                buf[filled : filled + take] = ids[pos : pos + take]
                filled += take
                pos += take
                if filled == len(buf):
                    if not wrote_val:
                        # Carve the first val_tokens off as the val split.
                        _write_shard(out / "val.bin", buf[: config.val_tokens].copy())
                        _write_shard(
                            out / f"train_{shard_idx:06d}.bin",
                            buf[config.val_tokens :].copy(),
                        )
                        wrote_val = True
                    else:
                        _write_shard(out / f"train_{shard_idx:06d}.bin", buf.copy())
                    shard_idx += 1
                    filled = 0

    if filled > 0:
        _write_shard(out / f"train_{shard_idx:06d}.bin", buf[:filled].copy())
    print(f"done: {shard_idx + 1} train shard(s) + val.bin in {out}")


class TokenLoader:
    """Samples random (B, T) token windows from memmapped shards.

    Each shard is reopened as a fresh np.memmap per batch (cheap, and avoids a
    known memory leak from holding long-lived memmaps).
    """

    def __init__(self, data_dir: str, split: str, block_size: int, seed: int = 0):
        self.block_size = block_size
        self.gen = torch.Generator().manual_seed(seed)
        data = Path(data_dir)
        if split == "train":
            self.shards = sorted(data.glob("train_*.bin"))
        else:
            self.shards = [data / "val.bin"]
        if not self.shards or not all(p.exists() for p in self.shards):
            raise FileNotFoundError(
                f"no {split} shards in {data_dir}; run `python -m gpt_from_scratch.data` first"
            )

    def _memmap(self, path: Path) -> np.memmap:
        return np.memmap(path, dtype=DTYPE, mode="r")

    def batch(self, batch_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        # Pick a random shard each batch so training mixes across the corpus.
        shard = self.shards[
            int(torch.randint(len(self.shards), (1,), generator=self.gen).item())
        ]
        data = self._memmap(shard)
        max_start = len(data) - self.block_size - 1
        starts = torch.randint(0, max_start, (batch_size,), generator=self.gen)

        x = torch.empty((batch_size, self.block_size), dtype=torch.long)
        y = torch.empty((batch_size, self.block_size), dtype=torch.long)
        for i, s in enumerate(starts.tolist()):
            chunk = torch.from_numpy(data[s : s + self.block_size + 1].astype(np.int64))
            x[i], y[i] = chunk[:-1], chunk[1:]

        if device.startswith("cuda"):
            # Pinned + non-blocking overlaps the host->device copy with compute.
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y


def main():
    prepare(PrepareConfig())


if __name__ == "__main__":
    main()
