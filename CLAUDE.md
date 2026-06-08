# gpt-from-scratch

A from-scratch GPT-2 (~124M) trained on FineWeb-Edu, with DDP multi-GPU training.

## Layout

- `model.py` — GPT model, config, transformer block, generation.
- `data.py` — tokenize FineWeb-Edu into binary shards; `TokenLoader` streams windows.
- `train.py` — training loop, DDP, eval, wandb logging. Run via `torchrun`.
- `eval.py` — load a checkpoint, report val loss/perplexity, sample.
- `hellaswag.py` — HellaSwag benchmark; runs standalone or during training.
- `tokenizer.py` — GPT-2 BPE (tiktoken) and byte tokenizers.

## Conventions

- **Keep comments short.** Avoid long or multi-line comments; add one only where the
  code is genuinely unclear, and explain *why*, not *what*. Prefer clear code and
  names over commentary. Same for docstrings — concise.
- Lint/format with `uv run pre-commit run --files <files>` (ruff + ty).
- Tests: `uv run pytest`.
