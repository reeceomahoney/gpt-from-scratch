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
