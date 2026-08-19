"""Supervised Fine-Tuning (SFT) Entry Point for Ultron (113M).

Loads pretrained base model weights and fine-tunes on conversational instruction
shards with prompt loss masking using Accelerate and mixed precision.
"""

import argparse
import os
from pathlib import Path
from typing import Any

import torch

from config import UltronConfig
from model import UltronModel, load_ultron_state_dict
from sft_dataset import get_sft_dataloaders
from telemetry import UltronTelemetry
from trainer import UltronTrainer


def load_base_weights(model: UltronModel, checkpoint_dir: str | Path) -> None:
    """Load model weights from an Accelerate pretraining checkpoint directory."""
    path = Path(checkpoint_dir)
    candidates = (
        path / "model.safetensors",
        path / "pytorch_model.bin",
        path / "pytorch_model" / "model.safetensors",
        path / "pytorch_model" / "pytorch_model.bin",
    )
    weight_path = next((p for p in candidates if p.is_file()), None)
    if weight_path is None:
        raise FileNotFoundError(
            f"No base model weights found in '{checkpoint_dir}'. "
            f"Expected one of {[c.name for c in candidates]}."
        )

    if weight_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state_dict = load_file(str(weight_path))
    else:
        state_dict = torch.load(str(weight_path), map_location="cpu", weights_only=True)

    missing, unexpected = load_ultron_state_dict(model, state_dict)
    if missing or unexpected:
        print(
            f"Base weights loaded with missing keys: {missing}, unexpected keys: {unexpected}"
        )
    else:
        print(f"Successfully loaded base weights from '{weight_path}'.")


def build_sft_config(args: Any) -> UltronConfig:
    """Build UltronConfig tailored for SFT."""
    config = UltronConfig()
    config.learning_rate = args.lr
    config.min_lr = args.min_lr
    config.warmup_steps = args.warmup_steps
    config.eval_interval = args.eval_interval
    config.eval_batches = args.eval_batches
    if args.max_steps is not None:
        config.max_steps = args.max_steps
    if args.mode == "test":
        config.is_test_mode = True
        config.max_steps = min(config.max_steps, 5)
        config.eval_interval = 2
        config.eval_batches = 2
    return config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ultron (113M) Supervised Fine-Tuning (SFT)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "fresh", "continue", "test"],
        help="Run mode: 'train'/'fresh' for full SFT run, 'continue' to resume, 'test' for quick smoke test.",
    )
    parser.add_argument(
        "--base-checkpoint",
        type=str,
        default="accelerate_checkpoint",
        help="Path to pretrained base checkpoint directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="accelerate_sft_checkpoint",
        help="Directory to save fine-tuned SFT checkpoints.",
    )
    parser.add_argument(
        "--shard-dir",
        type=str,
        default="shards_sft",
        help="Directory containing tokenized SFT shards.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Peak learning rate for SFT (default: 2e-4).",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=2e-5,
        help="Minimum learning rate after decay (default: 2e-5).",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=100,
        help="Warmup steps for SFT (default: 100).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=2500,
        help="Maximum training steps for SFT.",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=100,
        help="Evaluation interval in steps.",
    )
    parser.add_argument(
        "--eval-batches",
        type=int,
        default=20,
        help="Number of dev batches to sample during evaluation.",
    )
    args = parser.parse_args()

    config = build_sft_config(args)
    accelerator = UltronTelemetry.setup_accelerator_trackers(
        config,
        args,
        checkpoint_dir=args.output_dir,
        task_type="sft",
        project_name="ultron-sft",
    )

    accelerator.print(f"--- Starting Ultron-113M SFT ({args.mode.upper()} mode) ---")
    accelerator.print(
        f"Base Checkpoint: {args.base_checkpoint} -> Output Checkpoint: {args.output_dir}"
    )

    # Initialize model
    model = UltronModel(config)
    if os.path.exists(args.base_checkpoint):
        try:
            load_base_weights(model, args.base_checkpoint)
        except Exception as err:
            accelerator.print(
                f"Warning: Could not load base weights ({err}). Training from scratch/init."
            )
    else:
        accelerator.print(
            f"Base checkpoint '{args.base_checkpoint}' not found; initializing fresh weights."
        )

    optimizer_muon, optimizer_adamw = model.configure_optimizers(config.learning_rate)

    train_loader, dev_loader = get_sft_dataloaders(
        config, accelerator, shard_dir=args.shard_dir
    )

    (
        model,
        optimizer_muon,
        optimizer_adamw,
    ) = accelerator.prepare(
        model,
        optimizer_muon,
        optimizer_adamw,
    )

    trainer = UltronTrainer(
        model=model,
        optimizer_muon=optimizer_muon,
        optimizer_adamw=optimizer_adamw,
        train_loader=train_loader,
        dev_loader=dev_loader,
        config=config,
        accelerator=accelerator,
    )
    trainer.accelerate_dir = args.output_dir

    if args.mode == "continue":
        trainer.load_checkpoint()

    trainer.train()
    accelerator.print("--- SFT Training Completed Successfully ---")


if __name__ == "__main__":
    main()
