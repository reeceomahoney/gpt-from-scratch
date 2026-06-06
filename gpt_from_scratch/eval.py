import argparse
import math

import torch

from gpt_from_scratch.data import TokenLoader
from gpt_from_scratch.model import GPT, GPTConfig
from gpt_from_scratch.train import TrainConfig, evaluate


def load_model(path: str, device: str) -> GPT:
    model = GPT(GPTConfig()).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained GPT model.")
    parser.add_argument("--checkpoint", default="gpt.pt", help="path to model weights")
    parser.add_argument(
        "--eval-steps", type=int, default=200, help="batches to average"
    )
    parser.add_argument("--prompt", default="\n", help="prompt for a sample generation")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=40)
    args = parser.parse_args()

    config = TrainConfig(eval_steps=args.eval_steps)
    torch.manual_seed(0)
    model = load_model(args.checkpoint, config.device)

    val_loader = TokenLoader(config.data_dir, "val", config.block_size)
    val_loss = evaluate(model, val_loader, config)
    print(f"val loss {val_loss:.4f} | perplexity {math.exp(val_loss):.2f}")

    sample = model.predict(
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print("\n--- sample ---")
    print(sample)


if __name__ == "__main__":
    main()
