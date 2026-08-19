"""Upload Model Cards (README.md) to Hugging Face Hub for Ultron Base and Instruct models."""

import os
from pathlib import Path

from huggingface_hub import HfApi, login
from rich.console import Console

console = Console()


def upload_model_cards() -> None:
    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token, add_to_git_credential=True)

    api = HfApi()

    cards = [
        (
            "model_cards/BASE_MODEL_CARD.md",
            "jaipkapoor99/ultron-113m",
            "Ultron-113M (Base)",
        ),
        (
            "model_cards/INSTRUCT_MODEL_CARD.md",
            "jaipkapoor99/ultron-113m-instruct",
            "Ultron-113M-Instruct",
        ),
    ]

    for local_file, repo_id, display_name in cards:
        path = Path(local_file)
        if not path.exists():
            console.print(f"[red]Error: Model card '{local_file}' not found.[/red]")
            continue

        console.print(
            f"📄 Uploading Model Card for [bold cyan]{display_name}[/bold cyan] to [bold green]{repo_id}[/bold green]..."
        )
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"docs: add Hugging Face Hub model card for {display_name}",
        )
        console.print(
            f"✓ [bold green]{display_name} Model Card published successfully![/bold green]"
        )


if __name__ == "__main__":
    upload_model_cards()
