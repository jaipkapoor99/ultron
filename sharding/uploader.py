"""Validation and uploading logic for pretraining and SFT shards to Hugging Face Hub."""

import json
import os
from pathlib import Path

from huggingface_hub import HfApi
from rich.console import Console

console = Console()


def validate_complete_shard_set(shards_dir: Path, is_sft: bool = False) -> dict:
    """Validate the exact-resume state and every committed shard pair."""
    state_path = shards_dir / "tokenization_state.json"
    if not state_path.is_file():
        raise RuntimeError(f"Missing exact tokenization state: {state_path}")
    with state_path.open() as file:
        state = json.load(file)

    shard_count = state.get("max_shards")
    if isinstance(shard_count, int) and (
        state.get("next_shard") != shard_count
        or state.get("committed_tokens")
        != shard_count * state.get("shard_size_tokens", 0)
    ):
        raise RuntimeError(
            "Tokenization is incomplete; upload is allowed only after every "
            "configured shard has been atomically committed."
        )

    shard_count = state.get("next_shard", 0)
    shard_size_tokens = state["shard_size_tokens"]
    prefix = "smoltalk_shard" if is_sft else "fineweb_edu_shard"

    for index in range(shard_count):
        if is_sft:
            inputs_path = shards_dir / f"{prefix}_{index:04d}_inputs.bin"
            targets_path = shards_dir / f"{prefix}_{index:04d}_targets.bin"
            meta_path = shards_dir / f"{prefix}_{index:04d}_meta.json"
            if (
                not inputs_path.is_file()
                or inputs_path.stat().st_size != shard_size_tokens * 2
            ):
                raise RuntimeError(
                    f"SFT inputs shard {index:04d} is missing or has invalid size."
                )
            if (
                not targets_path.is_file()
                or targets_path.stat().st_size != shard_size_tokens * 4
            ):
                raise RuntimeError(
                    f"SFT targets shard {index:04d} is missing or has invalid size."
                )
        else:
            shard_path = shards_dir / f"{prefix}_{index:04d}.bin"
            meta_path = shards_dir / f"{prefix}_{index:04d}_meta.json"
            if (
                not shard_path.is_file()
                or shard_path.stat().st_size != shard_size_tokens * 2
            ):
                raise RuntimeError(
                    f"Shard {index:04d} is missing or has an invalid size."
                )

        if not meta_path.is_file():
            raise RuntimeError(f"Metadata for shard {index:04d} is missing.")
        with meta_path.open() as file:
            metadata = json.load(file)
        if is_sft:
            if (
                metadata.get("shard_index") != index
                or metadata.get("tokens") != shard_size_tokens
                or metadata.get("inputs_dtype") != "uint16"
                or metadata.get("targets_dtype") != "int32"
            ):
                raise RuntimeError(
                    f"Metadata for SFT shard {index:04d} is inconsistent."
                )
        else:
            if (
                metadata.get("shard_index") != index
                or metadata.get("tokens") != shard_size_tokens
                or metadata.get("dtype") != "uint16"
                or metadata.get("dataset_revision") != state.get("dataset_revision")
                or metadata.get("tokenizer_revision") != state.get("tokenizer_revision")
            ):
                raise RuntimeError(f"Metadata for shard {index:04d} is inconsistent.")
    return state


def upload_dataset_shards(
    target_repo: str,
    shards_dir: str | Path,
    is_sft: bool = False,
    private: bool = False,
) -> None:
    """Validate and resumably upload binary dataset shards to Hugging Face Hub."""
    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    resolved_shards_dir = Path(shards_dir).resolve()
    if not resolved_shards_dir.is_dir():
        raise FileNotFoundError(
            f"Shards directory does not exist: {resolved_shards_dir}"
        )
    state = validate_complete_shard_set(resolved_shards_dir, is_sft=is_sft)

    shard_count = state["next_shard"]
    multiplier = 6 if is_sft else 2
    total_gib = state["committed_tokens"] * multiplier / (1024**3)
    console.print(
        f"[bold cyan]🤗 Target Hugging Face Dataset Repository:[/bold cyan] [bold white]{target_repo}[/bold white]"
    )
    console.print(
        f"Validated [bold yellow]{shard_count}[/bold yellow] binary shards "
        f"and metadata pairs ({total_gib:.2f} GiB)."
    )

    api.create_repo(
        repo_id=target_repo,
        repo_type="dataset",
        exist_ok=True,
        private=private,
    )

    console.print(
        f"[bold blue]🚀 Resumably uploading '{resolved_shards_dir}' to the Datasets Hub...[/bold blue]"
    )
    directory_name = resolved_shards_dir.name
    allow_patterns = (
        [
            f"{directory_name}/smoltalk_shard_*.bin",
            f"{directory_name}/smoltalk_shard_*_meta.json",
        ]
        if is_sft
        else [
            f"{directory_name}/fineweb_edu_shard_*.bin",
            f"{directory_name}/fineweb_edu_shard_*_meta.json",
        ]
    )

    api.upload_large_folder(
        folder_path=resolved_shards_dir.parent,
        repo_id=target_repo,
        repo_type="dataset",
        revision="main",
        private=private,
        allow_patterns=allow_patterns,
    )
    console.print(
        f"[bold green]🎉 All dataset shards uploaded successfully to https://huggingface.co/datasets/{target_repo} ![/bold green]"
    )
