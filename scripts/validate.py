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
        "--checkpoint",
        type=Path,
        default=Path("accelerate_checkpoint"),
        help="Accelerate checkpoint directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs/full_validation.json"),
        help="JSON result path",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile the model before evaluation",
    )
    parser.add_argument(
        "--wandb-project",
        default=ValidationTelemetry.PROJECT_NAME,
        help="W&B project for the dedicated full-validation run",
    )
    parser.add_argument(
        "--wandb-run-name",
        default=None,
        help="Optional W&B run name; defaults to a timestamped name",
    )
    args = parser.parse_args()

    config = UltronConfig.from_metadata()
    accelerator = ValidationTelemetry.setup_accelerator(
        config,
        project_name=args.wandb_project,
        run_name=args.wandb_run_name,
    )
    _, dev_loader = get_dataloaders(config, accelerator)

    model = UltronModel(config)
    weight_path = load_checkpoint_weights(model, args.checkpoint)
    if args.compile:
        model = torch.compile(model)
    model = accelerator.prepare(model)
    model.eval()

    total_loss = 0.0
    total_sequences = 0
    validation_telemetry = ValidationTelemetry(
        total_sequences=len(dev_loader.dataset),
        sequence_length=config.T,
        accelerator=accelerator,
    )
    try:
        with torch.inference_mode():
            for inputs, targets in dev_loader:
                with accelerator.autocast():
                    logits = model(inputs).logits
                    sequence_losses = sequence_cross_entropy(logits, targets)
                sequence_losses = accelerator.gather_for_metrics(sequence_losses)
                total_loss += sequence_losses.double().sum().item()
                total_sequences += sequence_losses.numel()
                validation_telemetry.update(
                    total_sequences,
                    mean_loss=total_loss / total_sequences,
                )
    finally:
        validation_telemetry.close()

    if total_sequences == 0:
        raise RuntimeError("Validation dataloader produced no samples.")

    mean_loss = total_loss / total_sequences
    result = {
        "validation_scope": "full",
        "loss": mean_loss,
        "perplexity": math.exp(mean_loss),
        "sequences": total_sequences,
        "tokens": total_sequences * config.T,
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
            args.output.parent.mkdir(parents=True, exist_ok=True)
            temporary_output = args.output.with_suffix(args.output.suffix + ".tmp")
            with temporary_output.open("w") as file:
                json.dump(result, file, indent=2)
            os.replace(temporary_output, args.output)
            accelerator.print(
                f"Full validation | loss {mean_loss:.6f} | "
                f"perplexity {result['perplexity']:.4f} | "
                f"{result['tokens']:,} tokens | "
                f"{result['average_tokens_per_second']:,.0f} tok/s"
            )
            accelerator.print(f"Saved results to {args.output}")
        accelerator.wait_for_everyone()
    finally:
        accelerator.end_training()


if __name__ == "__main__":
    main()
