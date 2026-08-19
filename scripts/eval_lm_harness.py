"""Evaluate an Ultron checkpoint with EleutherAI's lm-evaluation-harness.

Usage:
    accelerate launch scripts/eval_lm_harness.py
    accelerate launch scripts/eval_lm_harness.py \
        --tasks arc_easy,hellaswag \
        --limit 50
"""

import argparse
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import UltronConfig
from model import UltronModel, load_ultron_state_dict

DEFAULT_TASKS = (
    "arc_easy",
    "arc_challenge",
    "hellaswag",
    "openbookqa",
    "piqa",
    "winogrande",
)


def parse_tasks(value: str) -> list[str]:
    """Parse a comma-separated task list, preserving order and removing duplicates."""
    tasks = list(
        dict.fromkeys(task.strip() for task in value.split(",") if task.strip())
    )
    if not tasks:
        raise ValueError("--tasks must contain at least one task")
    return tasks


def normalize_limit(limit: int) -> int | None:
    """Interpret zero as a complete evaluation and reject negative limits."""
    if limit < 0:
        raise ValueError("--limit cannot be negative")
    return limit or None


def select_accuracy(metrics: dict) -> float | None:
    """Return the preferred accuracy metric without discarding a valid zero."""
    for key in ("acc,none", "acc_norm,none", "acc"):
        value = metrics.get(key)
        if value is not None:
            return value
    return None


def load_base_model(
    checkpoint_dir: Path,
    config: UltronConfig,
    *,
    device: torch.device | str = "cpu",
    print_fn: Callable[[str], None] = print,
) -> UltronModel:
    """Instantiate Ultron and strictly load an Accelerate checkpoint."""
    model = UltronModel(config)
    weight_file = None
    for relative_path in (
        "model.safetensors",
        "pytorch_model.bin",
        "pytorch_model/model.safetensors",
        "pytorch_model/pytorch_model.bin",
    ):
        candidate = checkpoint_dir / relative_path
        if candidate.exists():
            weight_file = candidate
            break

    if weight_file is None:
        raise FileNotFoundError(f"No checkpoint weights found in '{checkpoint_dir}'")

    print_fn(f"Loading weights from: {weight_file}")
    if weight_file.suffix == ".safetensors":
        from safetensors.torch import load_file

        state_dict = load_file(str(weight_file), device="cpu")
    else:
        state_dict = torch.load(
            weight_file,
            map_location="cpu",
            weights_only=True,
        )

    missing, unexpected = load_ultron_state_dict(model, state_dict)
    if missing:
        print_fn(f"Expected tied-weight alias omitted: {missing}")
    if unexpected:
        print_fn(f"Unexpected checkpoint keys: {unexpected}")
    return model.to(device).eval()


def save_results(results: dict, output_path: Path) -> None:
    """Persist only task metrics, converting harness-specific value types."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    with temporary_path.open("w") as handle:
        json.dump(results.get("results", {}), handle, indent=2, default=str)
    temporary_path.replace(output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ultron EleutherAI lm-evaluation-harness suite"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("accelerate_checkpoint"),
    )
    parser.add_argument(
        "--tasks",
        default=",".join(DEFAULT_TASKS),
        help="Comma-separated lm-evaluation-harness task names",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Samples per task; use 0 for complete task splits",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs/pre_training_checkpoint_eval.json"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        task_list = parse_tasks(args.tasks)
        eval_limit = normalize_limit(args.limit)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    if not any(
        key in os.environ
        for key in (
            "ACCELERATE_TORCH_DEVICE",
            "ACCELERATE_PROCESS_ID",
            "LOCAL_RANK",
            "ACCELERATE_MIXED_PRECISION",
        )
    ):
        raise RuntimeError("Run with: accelerate launch scripts/eval_lm_harness.py ...")

    # Heavy benchmark dependencies and distributed state are initialized only
    # for an actual evaluation, keeping helper functions CPU-testable.
    from accelerate import Accelerator
    from lm_eval.evaluator import simple_evaluate
    from lm_eval.models.huggingface import HFLM
    from transformers import AutoTokenizer

    accelerator = Accelerator()
    config = UltronConfig()
    model = load_base_model(
        args.checkpoint_dir,
        config,
        device=accelerator.device,
        print_fn=accelerator.print,
    )

    accelerator.print(f"Loading tokenizer ({config.tokenizer_name})...")
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    if tokenizer is None or tokenizer.vocab_size != config.vocab_size:
        raise RuntimeError(
            f"Tokenizer vocabulary ({getattr(tokenizer, 'vocab_size', None)}) does not match "
            f"model configuration ({config.vocab_size})"
        )

    from typing import Any, cast

    evaluation_model = HFLM(
        pretrained=cast(Any, model),
        tokenizer=tokenizer,
        batch_size=1,
        device=str(accelerator.device),
    )
    accelerator.print(
        "\nRunning EleutherAI lm-evaluation-harness across "
        f"{task_list} (limit: {eval_limit or 'FULL'})...\n"
    )
    results = simple_evaluate(
        model=evaluation_model,
        tasks=task_list,
        limit=eval_limit,
    )

    accelerator.print("=" * 50)
    accelerator.print("ELEUTHERAI LM-EVALUATION-HARNESS REPORT")
    accelerator.print("=" * 50)
    for task_name, metrics in results.get("results", {}).items():
        accuracy = select_accuracy(metrics)
        if accuracy is None:
            accelerator.print(f"• {task_name:<25} : {metrics}")
        else:
            accelerator.print(f"• {task_name:<25} : {accuracy * 100:.2f}%")
    accelerator.print("=" * 50)

    if accelerator.is_main_process:
        save_results(results, args.output)
        accelerator.print(f"Results saved to '{args.output}'")
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
