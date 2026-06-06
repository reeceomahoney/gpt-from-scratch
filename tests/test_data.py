import numpy as np
import pytest

from gpt_from_scratch.data import DTYPE, TokenLoader


def _write_shards(dir_path):
    # Synthetic shards: a ramp of token ids so we can spot-check x/y alignment.
    (dir_path / "train_000000.bin").write_bytes(np.arange(1000, dtype=DTYPE).tobytes())
    (dir_path / "val.bin").write_bytes(np.arange(500, dtype=DTYPE).tobytes())


def test_token_loader_shapes_and_shift(tmp_path):
    _write_shards(tmp_path)
    loader = TokenLoader(str(tmp_path), "train", block_size=8)
    x, y = loader.batch(batch_size=4, device="cpu")

    assert x.shape == (4, 8)
    assert y.shape == (4, 8)
    # y is x shifted by one position (next-token targets).
    assert (y[:, :-1] == x[:, 1:]).all()


def test_token_loader_missing_split(tmp_path):
    with pytest.raises(FileNotFoundError):
        TokenLoader(str(tmp_path), "train", block_size=8)
