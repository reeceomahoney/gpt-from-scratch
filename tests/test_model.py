from gpt_from_scratch import model
import torch


def test_transformer_block():
    tf = model.TransformerBlock(64, 64 * 4)
    x = torch.randn((8, 64))
    x = tf(x)
    assert x.shape == (8, 64)


def test_gpt():
    config = model.GPTConfig()
    gpt = model.GPT(config)
    logits, pad_mask = gpt(["hi", "hello"])
    assert logits.shape == (2, 5, config.vocab_size)
    assert pad_mask.shape == (2, 5)


def test_predict():
    config = model.GPTConfig()
    gpt = model.GPT(config)
    out = gpt.predict("hi", max_new_tokens=10, top_k=40)
    assert isinstance(out, str)
    assert out.startswith("hi")


def test_predict_crops_to_context_length():
    config = model.GPTConfig(context_length=16)
    gpt = model.GPT(config)
    prompt = "a" * config.context_length
    out = gpt.predict(prompt, max_new_tokens=5)
    assert isinstance(out, str)
