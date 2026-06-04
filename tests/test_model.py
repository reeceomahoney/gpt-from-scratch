from gpt_from_scratch import model
import torch

def test_transformer_block():
    tf = model.TransformerBlock(64, 64*4)
    x = torch.randn((8,64))
    x = tf(x)
    assert x.shape == (8,64)

def test_gpt():
    config = model.GPTConfig()
    gpt = model.GPT(config)
    x = torch.randint(config.vocab_size, (8,))
    probs = gpt(x)
    assert probs.shape == (8,config.vocab_size)

