import math
import os
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field

import torch
import torch.distributed as dist
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP

from gpt_from_scratch.data import TokenLoader
from gpt_from_scratch.hellaswag import evaluate_hellaswag, load_val
from gpt_from_scratch.model import GPT, GPTConfig


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class DDPInfo:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int

    @property
    def is_master(self) -> bool:
        return self.rank == 0


def setup_ddp() -> DDPInfo:
    if "RANK" not in os.environ:
        return DDPInfo(enabled=False, rank=0, local_rank=0, world_size=1)
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return DDPInfo(
        enabled=True,
        rank=int(os.environ["RANK"]),
        local_rank=local_rank,
        world_size=int(os.environ["WORLD_SIZE"]),
    )


@dataclass
class TrainConfig:
    data_dir: str = "data/fineweb_edu_10bt"
    block_size: int = 1024
    # Per-GPU micro-batch
    micro_batch_size: int = 128
    # Target global batch in tokens (~0.5M, the GPT-2 setpoint)
    total_batch_size: int = int(2**19)
    lr: float = 6e-4
    min_lr: float = 6e-5
    warmup_steps: int = 700
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    max_steps: int = 20000  # ~10B tokens at 0.5M/step (one pass of sample-10BT)
    eval_every: int = 500
    eval_steps: int = 50
    eval_hellaswag: bool = True  # run the HellaSwag benchmark at each eval
    compile: bool = True
    save_path: str = "gpt.pt"
    device: str = field(default_factory=lambda: default_device())
    wandb_project: str = "gpt-from-scratch"
    wandb_run_name: str | None = None


def get_lr(step: int, config: TrainConfig) -> float:
    # Linear warmup, then cosine decay from lr down to min_lr.
    if step < config.warmup_steps:
        return config.lr * step / config.warmup_steps
    if step > config.max_steps:
        return config.min_lr
    ratio = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
    coeff = 0.5 * (1 + math.cos(math.pi * ratio))
    return config.min_lr + coeff * (config.lr - config.min_lr)


def use_bf16(device: str) -> bool:
    return device.startswith("cuda") and torch.cuda.is_bf16_supported()


def autocast_ctx(device: str):
    if use_bf16(device):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


@torch.no_grad()
def evaluate(
    model: GPT, loader: TokenLoader, config: TrainConfig, ddp: DDPInfo | None = None
) -> float:
    model.eval()
    losses = []
    for _ in range(config.eval_steps):
        x, y = loader.batch(config.micro_batch_size, config.device)
        with autocast_ctx(config.device):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    mean = sum(losses) / len(losses)
    if ddp is not None and ddp.enabled:
        t = torch.tensor(mean, device=config.device)
        dist.all_reduce(t, op=dist.ReduceOp.AVG)
        mean = t.item()
    return mean


def main():
    ddp = setup_ddp()
    config = TrainConfig()
    if ddp.enabled:
        config.device = f"cuda:{ddp.local_rank}"
    device = config.device

    # Derive grad-accum so the global batch hits total_batch_size
    per_step_tokens = config.micro_batch_size * config.block_size * ddp.world_size
    assert config.total_batch_size % per_step_tokens == 0, (
        f"total_batch_size {config.total_batch_size} not divisible by "
        f"micro_batch * block * world_size = {per_step_tokens}"
    )
    grad_accum_steps = config.total_batch_size // per_step_tokens

    if ddp.is_master:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config={
                **asdict(config),
                "grad_accum_steps": grad_accum_steps,
                "world_size": ddp.world_size,
            },
        )
    torch.manual_seed(0)
    if device.startswith("cuda"):
        torch.set_float32_matmul_precision("high")  # TF32 for the fp32 matmuls

    model = GPT(GPTConfig(context_length=config.block_size)).to(device)

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
        fused=device.startswith("cuda"),
    )

    train_loader = TokenLoader(
        config.data_dir, "train", config.block_size, seed=ddp.rank
    )
    val_loader = TokenLoader(config.data_dir, "val", config.block_size, seed=ddp.rank)

    # Load once up front; skip (don't crash) if the dataset can't be fetched.
    hellaswag = None
    if config.eval_hellaswag:
        try:
            hellaswag = load_val()
        except Exception as e:
            if ddp.is_master:
                print(f"hellaswag disabled (failed to load dataset: {e})")

    ddp_model = DDP(model, device_ids=[ddp.local_rank]) if ddp.enabled else None
    loss_fn = ddp_model if ddp_model is not None else model
    if config.compile and device.startswith("cuda"):
        if ddp.is_master:
            print("compiling model (first step will be slow)...")
        loss_fn = torch.compile(loss_fn)

    tokens_per_step = config.total_batch_size
    if ddp.is_master:
        precision = "bf16 autocast" if use_bf16(device) else "fp32"
        print(
            f"training for {config.max_steps} steps | {ddp.world_size} GPU(s) | "
            f"grad_accum {grad_accum_steps} | {tokens_per_step:,} tokens/step | "
            f"precision {precision}"
        )
    best_val = float("inf")
    last_log_time = time.perf_counter()
    for step in range(1, config.max_steps + 1):
        step_start = time.perf_counter()
        lr = get_lr(step, config)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # Gradient accumulation over micro-batches.
        optimizer.zero_grad(set_to_none=True)
        train_loss = 0.0
        for micro in range(grad_accum_steps):
            x, y = train_loader.batch(config.micro_batch_size, device)
            last_micro = micro == grad_accum_steps - 1
            sync_ctx = (
                ddp_model.no_sync()
                if ddp_model is not None and not last_micro
                else nullcontext()
            )
            with sync_ctx:
                with autocast_ctx(device):
                    _, loss = loss_fn(x, y)
                (loss / grad_accum_steps).backward()
            train_loss += loss.item() / grad_accum_steps

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        if device.startswith("cuda"):
            torch.cuda.synchronize()
        step_ms = (time.perf_counter() - step_start) * 1000
        if ddp.is_master:
            if step % 100 == 0 or step == 1:
                print(
                    f"step {step:5d}/{config.max_steps} | "
                    f"loss {train_loss:.4f} | lr {lr:.2e} | "
                    f"grad_norm {grad_norm.item():.3f} | {step_ms:.0f} ms"
                )
            wandb.log(
                {
                    "train/loss": train_loss,
                    "train/lr": lr,
                    "train/grad_norm": grad_norm.item(),
                },
                step=step,
            )

        if step % config.eval_every == 0 or step == 1:
            # Throughput for the training interval, measured before evals run.
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            tokens_per_sec = (
                tokens_per_step
                * config.eval_every
                / (time.perf_counter() - last_log_time)
            )

            val_loss = evaluate(model, val_loader, config, ddp)
            hs = None
            if hellaswag is not None:
                hs = evaluate_hellaswag(
                    model, hellaswag, device, amp=lambda: autocast_ctx(device), ddp=ddp
                )
            last_log_time = time.perf_counter()  # exclude eval time from next interval

            is_best = val_loss < best_val
            if is_best:
                best_val = val_loss
                if ddp.is_master:
                    torch.save(model.state_dict(), config.save_path)

            if ddp.is_master:
                hs_str = f" | hswag {hs['acc_norm']:.4f}" if hs is not None else ""
                print(
                    f"eval  {step:5d}/{config.max_steps} | "
                    f"val_loss {val_loss:.4f} | ppl {math.exp(val_loss):.2f} | "
                    f"best {best_val:.4f}{' *' if is_best else ''}{hs_str} | "
                    f"{tokens_per_sec:,.0f} tok/s"
                )
                log_data = {
                    "val/loss": val_loss,
                    "val/perplexity": math.exp(val_loss),
                    "perf/tokens_per_sec": tokens_per_sec,
                }
                if hs is not None:
                    log_data["val/hellaswag_acc"] = hs["acc"]
                    log_data["val/hellaswag_acc_norm"] = hs["acc_norm"]
                wandb.log(log_data, step=step)

    if ddp.is_master:
        wandb.summary["best_val_loss"] = best_val
        wandb.finish()
    if ddp.enabled:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
