"""Run a complete validation pass over the leakage-safe dev partition.

Usage:
    accelerate launch scripts/validate.py
    accelerate launch scripts/validate.py --checkpoint accelerate_checkpoint
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import UltronConfig
from dataset import get_dataloaders
from model import UltronModel, load_ultron_state_dict
from telemetry import ValidationTelemetry


def sequence_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Return one mean next-token loss per sequence."""
    token_losses = F.cross_entropy(
        logits.flatten(0, 1),
        targets.flatten(),
        reduction="none",
    ).view_as(targets)
    return token_losses.mean(dim=1)


def masked_cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculate token cross-entropy ignoring -1 target masks."""
    loss = F.cross_entropy(
        logits.flatten(0, 1),
        targets.flatten(),
        ignore_index=-1,
        reduction="sum",
    )
    active_tokens = (targets != -1).sum().to(loss.dtype)
    return loss, active_tokens


def load_checkpoint_weights(model: UltronModel, checkpoint_dir: Path) -> Path:
    """Load model weights from an Accelerate checkpoint directory."""
    candidates = (
        checkpoint_dir / "model.safetensors",
        checkpoint_dir / "pytorch_model.bin",
        checkpoint_dir / "pytorch_model" / "model.safetensors",
        checkpoint_dir / "pytorch_model" / "pytorch_model.bin",
    )
    weight_path = next((path for path in candidates if path.is_file()), None)
    if weight_path is None:
        expected = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"No model weights found. Expected one of: {expected}")

    if weight_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state_dict = load_file(weight_path, device="cpu")
    else:
        state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
    load_ultron_state_dict(model, state_dict)
    return weight_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sft",
        "--instruct",
        action="store_true",
        dest="sft",
        help="Run validation on SFT / instruction dev set instead of pretraining dev set",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Accelerate checkpoint directory (default: accelerate_sft_checkpoint if --sft else accelerate_checkpoint)",
    )
    parser.add_argument(
        "--shard-dir",
        type=str,
        default=None,
        help="Directory containing binary dataset shards",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON result path (default: logs/sft_validation.json if --sft else logs/full_validation.json)",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile the model before evaluation",
    )
    parser.add_argument(
        "--wandb-project",
        default=None,
        help="W&B project for the dedicated full-validation run",
    )
    parser.add_argument(
        "--wandb-run-name",
        default=None,
        help="Optional W&B run name; defaults to a timestamped name",
    )
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint or Path(
        "accelerate_sft_checkpoint" if args.sft else "accelerate_checkpoint"
    )
    output_path = args.output or Path(
        "logs/sft_validation.json" if args.sft else "logs/full_validation.json"
    )
    wandb_project = args.wandb_project or (
        "ultron-sft-validation" if args.sft else ValidationTelemetry.PROJECT_NAME
    )

    config = UltronConfig.from_metadata()
    accelerator = ValidationTelemetry.setup_accelerator(
        config,
        project_name=wandb_project,
        run_name=args.wandb_run_name,
    )

    if args.sft:
        from sft_dataset import get_sft_dataloaders

        shard_dir = args.shard_dir or "shards_sft"
        _, dev_loader = get_sft_dataloaders(config, accelerator, shard_dir=shard_dir)
    else:
        _, dev_loader = get_dataloaders(config, accelerator)

    model = UltronModel(config)
    weight_path = load_checkpoint_weights(model, checkpoint_dir)
    if args.compile:
        model = torch.compile(model)
    model = accelerator.prepare(model)
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    total_sequences = 0
    raw_dataset: Any = dev_loader.dataset
    num_dev_sequences = (
        len(raw_dataset)
        if hasattr(raw_dataset, "__len__")
        else len(dev_loader) * config.B
    )
    validation_telemetry = ValidationTelemetry(
        total_sequences=num_dev_sequences,
        sequence_length=config.T,
        accelerator=accelerator,
    )
    try:
        with torch.inference_mode():
            for inputs, targets in dev_loader:
                with accelerator.autocast():
                    logits = model(inputs).logits
                    if args.sft:
                        batch_loss, active_tokens = masked_cross_entropy(
                            logits, targets
                        )
                        reduced = accelerator.reduce(
                            torch.stack((batch_loss, active_tokens)),
                            reduction="sum",
                        )
                        total_loss += reduced[0].item()
                        total_tokens += int(reduced[1].item())
                        total_sequences += inputs.size(0) * accelerator.num_processes
                    else:
                        sequence_losses = sequence_cross_entropy(logits, targets)
                        sequence_losses = accelerator.gather_for_metrics(
                            sequence_losses
                        )
                        total_loss += sequence_losses.double().sum().item()
                        total_sequences += sequence_losses.numel()

                current_mean_loss = (
                    (total_loss / max(1, total_tokens))
                    if args.sft
                    else (total_loss / max(1, total_sequences))
                )
                validation_telemetry.update(
                    total_sequences,
                    mean_loss=current_mean_loss,
                )
    finally:
        validation_telemetry.close()

    if total_sequences == 0:
        raise RuntimeError("Validation dataloader produced no samples.")

    mean_loss = (
        (total_loss / max(1, total_tokens))
        if args.sft
        else (total_loss / max(1, total_sequences))
    )
    result = {
        "validation_scope": "full_sft" if args.sft else "full_pretraining",
        "loss": mean_loss,
        "perplexity": math.exp(mean_loss),
        "sequences": total_sequences,
        "tokens": total_tokens if args.sft else (total_sequences * config.T),
        "checkpoint": str(weight_path),
        "elapsed_seconds": validation_telemetry.elapsed_seconds,
        "average_tokens_per_second": (validation_telemetry.average_tokens_per_second),
    }
    validation_telemetry.finish(
        loss=mean_loss,
        perplexity=result["perplexity"],
    )

    try:
        if accelerator.is_main_process:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_output = output_path.with_suffix(output_path.suffix + ".tmp")
            with temporary_output.open("w") as file:
                json.dump(result, file, indent=2)
            os.replace(temporary_output, output_path)
            accelerator.print(
                f"Full validation | loss {mean_loss:.6f} | "
                f"perplexity {result['perplexity']:.4f} | "
                f"{result['tokens']:,} tokens | "
                f"{result['average_tokens_per_second']:,.0f} tok/s"
            )
            accelerator.print(f"Saved results to {output_path}")
        accelerator.wait_for_everyone()
    finally:
        accelerator.end_training()


if __name__ == "__main__":
    main()
