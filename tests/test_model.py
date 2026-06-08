import dataclasses

from gpt_from_scratch import model
import torch


def _small_config(**kwargs) -> model.GPTConfig:
    # A tiny byte-level model so unit tests stay fast and deterministic.
    base = model.GPTConfig(
        tokenizer="byte",
        vocab_size=256,
        context_length=64,
        d_model=64,
        n_heads=8,
        d_feedforward=64 * 4,
        d_layers=2,
    )
    return dataclasses.replace(base, **kwargs)


def test_transformer_block():
    config = model.GPTConfig(d_model=64, n_heads=8, d_feedforward=64 * 4)
    tf = model.TransformerBlock(config)
    x = torch.randn((2, 8, 64))
    x = tf(x)
    assert x.shape == (2, 8, 64)


def test_forward_loss():
    config = _small_config()
    gpt = model.GPT(config)
    x = torch.randint(0, config.vocab_size, (2, 16))
    y = torch.randint(0, config.vocab_size, (2, 16))
    logits, loss = gpt(x, y)
    assert logits.shape == (2, 16, config.vocab_size)
    assert loss.ndim == 0


def test_predict():
    config = _small_config()
    gpt = model.GPT(config)
    out = gpt.predict("hi", max_new_tokens=10, top_k=40)
    assert isinstance(out, str)
    assert out.startswith("hi")


def test_predict_crops_to_context_length():
    config = _small_config(context_length=16)
    gpt = model.GPT(config)
    prompt = "a" * config.context_length
    out = gpt.predict(prompt, max_new_tokens=5)
    assert isinstance(out, str)
