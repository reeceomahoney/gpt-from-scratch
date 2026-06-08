"""HellaSwag eval: the standard commonsense benchmark for GPT-2-class models.

Each example is a context plus 4 candidate endings; exactly one is the real
continuation. We score a candidate by the cross-entropy the model assigns to
its ending tokens (lower = more likely) and predict the lowest-loss candidate.
No generation, no finetuning -- a pure likelihood comparison, so it reuses the
training forward pass directly.

    uv run gpt_from_scratch/hellaswag.py --checkpoint gpt.pt

Two metrics, following the standard convention:
  - acc:      pick the candidate with the lowest summed ending loss
  - acc_norm: pick the lowest *average* (length-normalized) loss  <- headline

Reference acc_norm: random 25.0, GPT-2 124M ~29.5, GPT-3 124M ~33.7.
"""

import argparse
from contextlib import nullcontext
from typing import TYPE_CHECKING, Callable, ContextManager

import torch
import torch.distributed as dist
import torch.nn.functional as F

from gpt_from_scratch.model import GPT

# TYPE_CHECKING only: train.py imports this module, so a runtime import is circular.
if TYPE_CHECKING:
    from gpt_from_scratch.train import DDPInfo

# Zero-arg factory yielding a fresh autocast/null context per forward.
AmpFactory = Callable[[], ContextManager]

# HF parquet mirror (no trust_remote_code), cached under ~/.cache/huggingface.
DATASET = "Rowan/hellaswag"


def load_val(split: str = "validation"):
    """Load a HellaSwag split (10,042 labelled examples for validation)."""
    import datasets

    return datasets.load_dataset(DATASET, split=split)


def render_example(example: dict, tokenizer) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Build the 4 candidate sequences for one example.

    Returns (tokens, mask, label):
      tokens (4, T): ctx + ending, right-padded to the longest candidate
      mask   (4, T): 1 on ending tokens, 0 on ctx and padding
      label:         index of the correct ending
    """
    ctx_tokens = tokenizer.encode(example["ctx"])
    rows, masks = [], []
    for ending in example["endings"]:
        # Leading space: BPE merges the boundary the way it appears in text.
        end_tokens = tokenizer.encode(" " + ending)
        rows.append(torch.cat([ctx_tokens, end_tokens]))
        masks.append(
            torch.cat([torch.zeros_like(ctx_tokens), torch.ones_like(end_tokens)])
        )

    max_len = max(len(r) for r in rows)
    tokens = torch.zeros(4, max_len, dtype=torch.long)
    mask = torch.zeros(4, max_len, dtype=torch.long)
    for i, (row, m) in enumerate(zip(rows, masks)):
        tokens[i, : len(row)] = row
        mask[i, : len(m)] = m
    return tokens, mask, int(example["label"])


@torch.no_grad()
def score_example(
    model: GPT,
    tokens: torch.Tensor,
    mask: torch.Tensor,
    device: str,
    amp: AmpFactory = nullcontext,
) -> tuple[int, int]:
    """Return (pred, pred_norm): the summed- and average-loss argmin candidates.

    Right-padding is safe: causal attention means real ending tokens never see
    the pad tokens to their right, and the loss mask zeroes the pad positions.
    """
    tokens, mask = tokens.to(device), mask.to(device)
    with amp():
        logits = model.forward_tokens(tokens)  # (4, T, vocab)

    # Logits at position t predict token t+1, so align by shifting one over.
    shift_logits = logits[:, :-1, :].float()
    shift_tokens = tokens[:, 1:]
    shift_mask = mask[:, 1:]

    losses = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_tokens.reshape(-1),
        reduction="none",
    ).view(tokens.size(0), -1)
    losses = losses * shift_mask

    sum_loss = losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    return int(sum_loss.argmin()), int(avg_loss.argmin())


@torch.no_grad()
def evaluate_hellaswag(
    model: GPT,
    dataset,
    device: str,
    amp: AmpFactory = nullcontext,
    limit: int | None = None,
    ddp: "DDPInfo | None" = None,
) -> dict:
    """Accuracy over the dataset, sharded round-robin across DDP ranks.

    Each rank scores examples where ``index % world_size == rank`` and the
    counts are summed across ranks, so the result matches a single-process run.
    """
    was_training = model.training
    model.eval()
    rank = ddp.rank if ddp is not None and ddp.enabled else 0
    world = ddp.world_size if ddp is not None and ddp.enabled else 1

    n = correct = correct_norm = 0
    for i, example in enumerate(dataset):
        if limit is not None and i >= limit:
            break
        if i % world != rank:  # this example belongs to another rank
            continue
        tokens, mask, label = render_example(example, model.tokenizer)
        pred, pred_norm = score_example(model, tokens, mask, device, amp)
        n += 1
        correct += pred == label
        correct_norm += pred_norm == label

    if ddp is not None and ddp.enabled:
        totals = torch.tensor([n, correct, correct_norm], device=device)
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        n, correct, correct_norm = (int(v) for v in totals.tolist())

    if was_training:
        model.train()
    return {"n": n, "acc": correct / n, "acc_norm": correct_norm / n}


def main():
    # Imported lazily so the module-level import in train.py stays acyclic.
    from gpt_from_scratch.eval import load_model
    from gpt_from_scratch.train import TrainConfig, autocast_ctx

    parser = argparse.ArgumentParser(description="HellaSwag eval for a trained GPT.")
    parser.add_argument("--checkpoint", default="gpt.pt", help="path to model weights")
    parser.add_argument(
        "--limit", type=int, default=None, help="cap examples (default: all 10,042)"
    )
    args = parser.parse_args()

    config = TrainConfig()
    model = load_model(args.checkpoint, config.device)

    dataset = load_val()
    result = evaluate_hellaswag(
        model,
        dataset,
        config.device,
        amp=lambda: autocast_ctx(config.device),
        limit=args.limit,
    )
    print(
        f"hellaswag ({result['n']} examples) | "
        f"acc {result['acc']:.4f} | acc_norm {result['acc_norm']:.4f}"
    )


if __name__ == "__main__":
    main()
