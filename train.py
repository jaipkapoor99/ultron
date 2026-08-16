"""
Ultron (113M) pre-training script

Usage:
    accelerate launch train.py [--mode=fresh|continue|test] [--max-steps=N]
"""

import argparse
import os

import torch

from config import UltronConfig
from dataset import get_dataloaders
from model import UltronModel


def build_config(args):
    """Build a run configuration from parsed CLI arguments."""
    config = UltronConfig.from_metadata()
    config.is_test_mode = args.mode == "test"
    if args.max_steps is not None:
        if args.max_steps <= 0:
            raise ValueError("--max-steps must be greater than zero")
        config.max_steps = args.max_steps
    elif args.mode == "test":
        config.max_steps = 100
    return config


def main():
    parser = argparse.ArgumentParser(description="Ultron (113M) Pre-training")
    parser.add_argument("--mode", type=str, choices=["fresh", "continue", "test"], default="continue", help="Training execution mode: 'fresh', 'continue' (default), or 'test'")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional max training steps override (e.g. --max-steps=1000)")
    args = parser.parse_args()

    if not any(k in os.environ for k in ["ACCELERATE_TORCH_DEVICE", "ACCELERATE_PROCESS_ID", "LOCAL_RANK", "ACCELERATE_MIXED_PRECISION"]):
        raise RuntimeError("train.py must be launched using HuggingFace Accelerate!\nRun: accelerate launch train.py [--mode=fresh|continue|test] [--max-steps=N]")

    try:
        config = build_config(args)
    except ValueError as error:
        parser.error(str(error))

    from telemetry import UltronTelemetry
    accelerator = UltronTelemetry.setup_accelerator_trackers(config, args)

    train_loader, dev_loader = get_dataloaders(config, accelerator)

    model = UltronModel(config)
    torch.set_float32_matmul_precision('high')

    # Muon (2D weight matrices) + AdamW (embeddings, norms, biases)
    optimizer_muon, optimizer_adamw = model.configure_optimizers(config.learning_rate)

    model = torch.compile(model)
    model, optimizer_muon, optimizer_adamw = accelerator.prepare(model, optimizer_muon, optimizer_adamw)

    from trainer import UltronTrainer
    trainer = UltronTrainer(model, optimizer_muon, optimizer_adamw, train_loader, dev_loader, config, accelerator)

    # Load checkpoint depending on --mode=fresh|continue|test
    if args.mode == "test" or args.mode == "fresh":
        pass
    elif args.mode == "continue":
        # print_rich removed
        trainer.load_checkpoint()

    trainer.train()
    accelerator.end_training()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
