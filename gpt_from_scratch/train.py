import math
import time
from contextlib import nullcontext
from dataclasses import dataclass, field

import torch

from gpt_from_scratch.data import TokenLoader
from gpt_from_scratch.model import GPT, GPTConfig


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class TrainConfig:
    data_dir: str = "data/fineweb_edu_10bt"
    block_size: int = 1024
    # Effective batch = micro_batch_size * block_size * grad_accum_steps
    #                 = 128 * 1024 * 4 = 524288 tokens (~0.5M, the GPT-2 setpoint).
    # Sized for a single 97GB GH200: ~38GB activations leaves comfortable headroom.
    micro_batch_size: int = 128
    grad_accum_steps: int = 4
    lr: float = 6e-4
    min_lr: float = 6e-5
    warmup_steps: int = 700
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    max_steps: int = 20000  # ~10B tokens at 0.5M/step (one pass of sample-10BT)
    eval_every: int = 500
    eval_steps: int = 50
    compile: bool = True
    save_path: str = "gpt.pt"
    device: str = field(default_factory=lambda: default_device())


def get_lr(step: int, config: TrainConfig) -> float:
    # Linear warmup, then cosine decay from lr down to min_lr.
    if step < config.warmup_steps:
        return config.lr * step / config.warmup_steps
    if step > config.max_steps:
        return config.min_lr
    ratio = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
    coeff = 0.5 * (1 + math.cos(math.pi * ratio))
    return config.min_lr + coeff * (config.lr - config.min_lr)


def autocast_ctx(device: str):
    # bf16 on Ampere+ GPUs; otherwise run in fp32.
    if device.startswith("cuda") and torch.cuda.is_bf16_supported():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


@torch.no_grad()
def evaluate(model: GPT, loader: TokenLoader, config: TrainConfig) -> float:
    model.eval()
    losses = []
    for _ in range(config.eval_steps):
        x, y = loader.batch(config.micro_batch_size, config.device)
        with autocast_ctx(config.device):
            _, loss = model.loss_from_tokens(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def main():
    config = TrainConfig()
    torch.manual_seed(0)
    if config.device.startswith("cuda"):
        torch.set_float32_matmul_precision("high")  # TF32 for the fp32 matmuls

    model = GPT(GPTConfig(context_length=config.block_size)).to(config.device)

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
        fused=config.device.startswith("cuda"),
    )

    train_loader = TokenLoader(config.data_dir, "train", config.block_size)
    val_loader = TokenLoader(config.data_dir, "val", config.block_size)

    # Compile the hot path. The eager method stays available for eval/generation.
    loss_fn = model.loss_from_tokens
    if config.compile and config.device.startswith("cuda"):
        print("compiling model (first step will be slow)...")
        loss_fn = torch.compile(loss_fn)

    tokens_per_step = (
        config.micro_batch_size * config.block_size * config.grad_accum_steps
    )
    best_val = float("inf")
    last_log_time = time.perf_counter()
    for step in range(1, config.max_steps + 1):
        lr = get_lr(step, config)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # Gradient accumulation over micro-batches.
        optimizer.zero_grad(set_to_none=True)
        train_loss = 0.0
        for _ in range(config.grad_accum_steps):
            x, y = train_loader.batch(config.micro_batch_size, config.device)
            with autocast_ctx(config.device):
                _, loss = loss_fn(x, y)
            (loss / config.grad_accum_steps).backward()
            train_loss += loss.item() / config.grad_accum_steps

        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        if step % config.eval_every == 0 or step == 1:
            val_loss = evaluate(model, val_loader, config)
            if config.device.startswith("cuda"):
                torch.cuda.synchronize()
            now = time.perf_counter()
            tokens_per_sec = tokens_per_step * config.eval_every / (now - last_log_time)
            last_log_time = now

            is_best = val_loss < best_val
            if is_best:
                best_val = val_loss
                torch.save(model.state_dict(), config.save_path)

            print(
                f"step {step:6d} | train {train_loss:.4f} | "
                f"val {val_loss:.4f} | perplexity {math.exp(val_loss):.2f} | "
                f"lr {lr:.2e} | {tokens_per_sec / 1e3:.1f}K tok/s"
                f"{' | saved best' if is_best else ''}"
            )

    print(f"best val {best_val:.4f} | saved model to {config.save_path}")


if __name__ == "__main__":
    main()
