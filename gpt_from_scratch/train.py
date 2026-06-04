import math
import time

import datasets
import torch
from dataclasses import dataclass, field

from gpt_from_scratch.model import GPT, GPTConfig


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class TrainConfig:
    block_size: int = 256
    batch_size: int = 32
    lr: float = 3e-4
    max_steps: int = 5000
    eval_every: int = 200
    eval_steps: int = 50
    save_path: str = "gpt.pt"
    device: str = field(default_factory=lambda: default_device())


ds = datasets.load_dataset("karpathy/tiny_shakespeare", revision="refs/convert/parquet")
train_text = ds["train"][0]["text"]
val_text = ds["validation"][0]["text"]


def get_batch(text: str, config: TrainConfig) -> list[str]:
    max_start = len(text) - config.block_size
    starts = torch.randint(0, max_start, (config.batch_size,))
    return [text[s : s + config.block_size] for s in starts.tolist()]


@torch.no_grad()
def evaluate(model: GPT, text: str, config: TrainConfig) -> float:
    model.eval()
    losses = [
        model(get_batch(text, config))[1].item() for _ in range(config.eval_steps)
    ]
    model.train()
    return sum(losses) / len(losses)


def main():
    config = TrainConfig()
    torch.manual_seed(0)
    model = GPT(GPTConfig()).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    last_log_step = 0
    last_log_time = time.perf_counter()
    for step in range(1, config.max_steps + 1):
        _, loss = model(get_batch(train_text, config))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % config.eval_every == 0 or step == 1:
            val_loss = evaluate(model, val_text, config)
            now = time.perf_counter()
            steps_per_sec = (step - last_log_step) / (now - last_log_time)
            last_log_step, last_log_time = step, now
            print(
                f"step {step:5d} | train {loss.item():.4f} | "
                f"val {val_loss:.4f} | perplexity {math.exp(val_loss):.2f} | "
                f"{steps_per_sec:.2f} steps/sec"
            )

    torch.save(model.state_dict(), config.save_path)
    print(f"saved model to {config.save_path}")


if __name__ == "__main__":
    main()
