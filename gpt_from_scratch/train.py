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
    min_lr: float = 3e-5
    warmup_steps: int = 100
    weight_decay: float = 0.1
    max_steps: int = 10000
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


def get_lr(step: int, config: TrainConfig) -> float:
    # Linear warmup, then cosine decay from lr down to min_lr.
    if step < config.warmup_steps:
        return config.lr * step / config.warmup_steps
    if step > config.max_steps:
        return config.min_lr
    ratio = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
    coeff = 0.5 * (1 + math.cos(math.pi * ratio))
    return config.min_lr + coeff * (config.lr - config.min_lr)


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

    # Weight-decay 2D params (matmuls, embeddings); skip biases and LayerNorm gains.
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=config.lr,
        betas=(0.9, 0.95),
    )

    last_log_step = 0
    last_log_time = time.perf_counter()
    for step in range(1, config.max_steps + 1):
        lr = get_lr(step, config)
        for group in optimizer.param_groups:
            group["lr"] = lr

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
                f"lr {lr:.2e} | {steps_per_sec:.2f} steps/sec"
            )

    torch.save(model.state_dict(), config.save_path)
    print(f"saved model to {config.save_path}")


if __name__ == "__main__":
    main()
