"""Validate and upload the complete FineWeb-Edu shard set."""

import argparse
import json
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi
from rich.console import Console

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import UltronConfig

console = Console()


def validate_complete_shard_set(shards_dir: Path) -> dict:
    """Validate the exact-resume state and every committed shard pair."""
    state_path = shards_dir / "tokenization_state.json"
    if not state_path.is_file():
        raise RuntimeError(f"Missing exact tokenization state: {state_path}")
    with state_path.open() as file:
        state = json.load(file)

    shard_count = state.get("max_shards")
    if (
        not isinstance(shard_count, int)
        or state.get("next_shard") != shard_count
        or state.get("committed_tokens")
        != shard_count * state.get("shard_size_tokens", 0)
    ):
        raise RuntimeError(
            "Tokenization is incomplete; upload is allowed only after every "
            "configured shard has been atomically committed."
        )

    shard_size_tokens = state["shard_size_tokens"]
    expected_bytes = shard_size_tokens * 2
    for index in range(shard_count):
        shard_path = shards_dir / f"fineweb_edu_shard_{index:04d}.bin"
        meta_path = shards_dir / f"fineweb_edu_shard_{index:04d}_meta.json"
        if not shard_path.is_file() or shard_path.stat().st_size != expected_bytes:
            raise RuntimeError(f"Shard {index:04d} is missing or has an invalid size.")
        if not meta_path.is_file():
            raise RuntimeError(f"Metadata for shard {index:04d} is missing.")
        with meta_path.open() as file:
            metadata = json.load(file)
        if (
            metadata.get("shard_index") != index
            or metadata.get("tokens") != shard_size_tokens
            or metadata.get("dtype") != "uint16"
            or metadata.get("dataset_revision") != state.get("dataset_revision")
            or metadata.get("tokenizer_revision") != state.get("tokenizer_revision")
        ):
            raise RuntimeError(f"Metadata for shard {index:04d} is inconsistent.")
    return state


def main() -> None:
    config = UltronConfig()
    parser = argparse.ArgumentParser(
        description="Upload Ultron FineWeb-Edu binary dataset shards to Hugging Face Hub"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=config.hf_dataset_repo_id,
        help=f"Target Hugging Face Dataset Repo ID (default: {config.hf_dataset_repo_id})",
    )
    parser.add_argument(
        "--shards-dir",
        type=str,
        default="shards_edu",
        help="Path to local shards directory",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Set target dataset repository to private",
    )
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    shards_dir = Path(args.shards_dir).resolve()
    if not shards_dir.is_dir():
        raise FileNotFoundError(f"Shards directory does not exist: {shards_dir}")
    state = validate_complete_shard_set(shards_dir)

    shard_count = state["max_shards"]
    total_gib = state["committed_tokens"] * 2 / (1024**3)
    console.print(
        f"[bold cyan]🤗 Target Hugging Face Dataset Repository:[/bold cyan] [bold white]{args.repo_id}[/bold white]"
    )
    console.print(
        f"Validated [bold yellow]{shard_count}[/bold yellow] binary shards "
        f"and metadata pairs ({total_gib:.2f} GiB)."
    )

    api.create_repo(
        repo_id=args.repo_id, repo_type="dataset", exist_ok=True, private=args.private
    )

    console.print(
        f"[bold blue]🚀 Resumably uploading '{shards_dir}' to the Datasets Hub...[/bold blue]"
    )
    directory_name = shards_dir.name
    api.upload_large_folder(
        folder_path=shards_dir.parent,
        repo_id=args.repo_id,
        repo_type="dataset",
        revision="main",
        private=args.private,
        allow_patterns=[
            f"{directory_name}/fineweb_edu_shard_*.bin",
            f"{directory_name}/fineweb_edu_shard_*_meta.json",
        ],
    )
    console.print(
        f"[bold green]🎉 All dataset shards uploaded successfully to https://huggingface.co/datasets/{args.repo_id} ![/bold green]"
    )


if __name__ == "__main__":
    main()
