"""
Ultron (113M) checkpoint uploader script

Uploads the complete local `accelerate_checkpoint/` directory—including model,
optimizer, scheduler, scaler, and RNG state—to Hugging Face Hub.

Usage:
    python3 scripts/upload_checkpoint.py [--repo-id=USER/REPO] [--private]
"""

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, login
from rich.console import Console

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import UltronConfig

console = Console()


def main() -> None:
    config = UltronConfig()
    parser = argparse.ArgumentParser(
        description="Upload Ultron checkpoint to Hugging Face Hub"
    )
    parser.add_argument(
        "--sft",
        "--instruct",
        action="store_true",
        dest="sft",
        help="Upload the SFT / Instruct checkpoint instead of base pre-training checkpoint",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Target Hugging Face Repository ID (default: dynamically derived from config)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Path to local checkpoint directory (default: accelerate_sft_checkpoint if --sft else accelerate_checkpoint)",
    )
    parser.add_argument(
        "--private", action="store_true", help="Set target repository to private"
    )
    args = parser.parse_args()

    target_repo_id = (
        args.repo_id
        if args.repo_id is not None
        else (config.hf_instruct_repo_id if args.sft else config.hf_repo_id)
    )
    target_checkpoint_dir = (
        args.checkpoint_dir
        if args.checkpoint_dir is not None
        else ("accelerate_sft_checkpoint" if args.sft else "accelerate_checkpoint")
    )

    token = os.environ.get("HF_TOKEN")
    if not token:
        console.print(
            "[bold yellow]⚠️ HF_TOKEN environment variable not set. Attempting login using stored credentials...[/bold yellow]"
        )
    else:
        login(token=token)

    api = HfApi(token=token)

    if not os.path.exists(target_checkpoint_dir):
        console.print(
            f"[bold red]❌ Error: Checkpoint directory '{target_checkpoint_dir}' does not exist![/bold red]"
        )
        return

    console.print(
        f"[bold cyan]🤗 Initializing Hugging Face repository:[/bold cyan] [bold white]{target_repo_id}[/bold white]"
    )
    api.create_repo(repo_id=target_repo_id, exist_ok=True, private=args.private)

    checkpoint_path = Path(target_checkpoint_dir)
    files_to_upload = sorted([p for p in checkpoint_path.iterdir() if p.is_file()])

    total_mb = sum(p.stat().st_size for p in files_to_upload) / (1024 * 1024)
    console.print(
        f"[bold blue]📦 Uploading {len(files_to_upload)} files ({total_mb:.1f} MB) from '{target_checkpoint_dir}'...[/bold blue]"
    )

    for idx, file_path in enumerate(files_to_upload, 1):
        file_name = file_path.name
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        console.print(
            f"  [{idx}/{len(files_to_upload)}] Uploading [bold white]{file_name}[/bold white] ({file_size_mb:.1f} MB)..."
        )
        api.upload_file(
            path_or_fileobj=str(file_path),
            path_in_repo=file_name,
            repo_id=target_repo_id,
            commit_message=f"Upload {file_name}",
        )
        console.print(
            f"  [bold green]✓ {file_name} committed successfully![/bold green]"
        )

    console.print(
        "[bold green]✅ All checkpoint files uploaded and committed![/bold green]"
    )

    # 2. Upload Model Card if present
    model_card_path = checkpoint_path / "README.md"
    if model_card_path.is_file():
        console.print(
            f"[bold blue]📄 Syncing model card '{model_card_path}' to Hugging Face Hub...[/bold blue]"
        )
        api.upload_file(
            path_or_fileobj=str(model_card_path),
            path_in_repo="README.md",
            repo_id=target_repo_id,
            commit_message="Sync Hugging Face Hub model card",
        )
        console.print("[bold green]✅ Model card synced successfully![/bold green]")

    metadata_path = Path("metadata.yaml")
    if metadata_path.is_file():
        console.print("[bold blue]⚙️ Syncing metadata.yaml configuration...[/bold blue]")
        api.upload_file(
            path_or_fileobj=str(metadata_path),
            path_in_repo="metadata.yaml",
            repo_id=target_repo_id,
            commit_message="Sync model metadata.yaml",
        )
        console.print("[bold green]✅ metadata.yaml synced successfully![/bold green]")

    console.print(
        f"\n[bold green]🚀 Checkpoint successfully deployed to:[/bold green] https://huggingface.co/{target_repo_id}\n"
    )


if __name__ == "__main__":
    main()
