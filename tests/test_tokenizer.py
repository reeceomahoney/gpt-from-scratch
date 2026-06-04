import torch
from gpt_from_scratch import tokenizer


def test_tokenizer():
    tk = tokenizer.ByteTokenizer()
    text = "foobar"
    tokens = torch.tensor([102, 111, 111, 98, 97, 114]).long()
    x = tk.encode(text)
    assert (x == tokens).all()

    x = tk.decode(tokens)
    assert x == "foobar"


def test_encode_batch():
    tk = tokenizer.ByteTokenizer()
    tokens, mask = tk.encode_batch(["hi", "hello"])

    expected_tokens = torch.tensor(
        [
            [104, 105, 0, 0, 0],
            [104, 101, 108, 108, 111],
        ]
    ).long()
    assert (tokens == expected_tokens).all()

    expected_mask = torch.tensor(
        [
            [True, True, False, False, False],
            [True, True, True, True, True],
        ]
    )
    assert (mask == expected_mask).all()
