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

from huggingface_hub import HfApi, login
from rich.console import Console

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import UltronConfig

console = Console()

def main():
    config = UltronConfig()
    parser = argparse.ArgumentParser(description="Upload Ultron checkpoint to Hugging Face Hub")
    parser.add_argument("--repo-id", type=str, default=config.hf_repo_id, help=f"Target Hugging Face Repository ID (default: {config.hf_repo_id})")
    parser.add_argument("--checkpoint-dir", type=str, default="accelerate_checkpoint", help="Path to local checkpoint directory")
    parser.add_argument("--private", action="store_true", help="Set target repository to private")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        console.print("[bold yellow]⚠️ HF_TOKEN environment variable not set. Attempting login using stored credentials...[/bold yellow]")
    else:
        login(token=token)

    api = HfApi(token=token)

    if not os.path.exists(args.checkpoint_dir):
        console.print(f"[bold red]❌ Error: Checkpoint directory '{args.checkpoint_dir}' does not exist![/bold red]")
        return

    console.print(f"[bold cyan]🤗 Initializing Hugging Face repository:[/bold cyan] [bold white]{args.repo_id}[/bold white]")
    api.create_repo(repo_id=args.repo_id, exist_ok=True, private=args.private)

    # Upload every checkpoint file: this repository is intentionally resumable,
    # not an inference-only weight export.
    console.print(f"[bold blue]📦 Uploading complete training state from '{args.checkpoint_dir}'...[/bold blue]")
    api.upload_folder(
        folder_path=args.checkpoint_dir,
        repo_id=args.repo_id,
        repo_type="model",
        commit_message="Upload Ultron 113M Accelerate checkpoint weights and state"
    )
    console.print("[bold green]✅ Checkpoint files uploaded successfully![/bold green]")

    # 2. Upload Model Card (accelerate_checkpoint/README.md)
    model_card_path = os.path.join(args.checkpoint_dir, "README.md")
    if os.path.exists(model_card_path):
        console.print(f"[bold blue]📄 Syncing model card '{model_card_path}' to Hugging Face Hub...[/bold blue]")
        api.upload_file(
            path_or_fileobj=model_card_path,
            path_in_repo="README.md",
            repo_id=args.repo_id,
            commit_message="Sync Hugging Face Hub model card"
        )
        console.print("[bold green]✅ Model card synced successfully![/bold green]")

    if os.path.exists("metadata.yaml"):
        console.print("[bold blue]⚙️ Syncing metadata.yaml configuration...[/bold blue]")
        api.upload_file(
            path_or_fileobj="metadata.yaml",
            path_in_repo="metadata.yaml",
            repo_id=args.repo_id,
            commit_message="Sync model metadata.yaml"
        )
        console.print("[bold green]✅ metadata.yaml synced successfully![/bold green]")

    console.print(f"\n[bold green]🚀 Checkpoint successfully deployed to:[/bold green] https://huggingface.co/{args.repo_id}\n")

if __name__ == "__main__":
    main()
