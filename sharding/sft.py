"""SmolTalk SFT dataset tokenization, loss masking, and sharding."""

import json
import signal
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset

from config import UltronConfig
from sharding.base import (
    STATE_SCHEMA_VERSION,
    atomic_json_write,
    atomic_numpy_write,
    atomic_shard_write,
    close_iterator,
)
from telemetry import TokenizationTelemetry
from tokenizer import UltronTokenizer


def is_tokenization_complete(
    state: dict[str, Any],
    shard_size_tokens: int,
    max_shards: int | None,
) -> bool:
    if max_shards is None:
        return False
    return (
        state["next_shard"] == max_shards
        and state["committed_tokens"] == max_shards * shard_size_tokens
    )


def load_sft_resume_state(
    output_dir: Path,
    shard_size_tokens: int,
    max_shards: int | None,
    config: UltronConfig,
) -> tuple[dict[str, Any] | None, tuple[list[int], list[int]]]:
    state_file = output_dir / "tokenization_state.json"
    meta_files = sorted(output_dir.glob("smoltalk_shard_*_meta.json"))

    if not state_file.exists():
        if meta_files:
            raise RuntimeError(
                f"Found {len(meta_files)} shard metadata files in {output_dir} "
                "without an exact resume checkpoint (tokenization_state.json)."
            )
        return None, ([], [])

    state = json.loads(state_file.read_text())
    if state.get("schema_version") != STATE_SCHEMA_VERSION:
        raise RuntimeError(f"Unsupported state schema: {state.get('schema_version')}")

    expected_config = {
        "dataset_id": config.sft_dataset_id,
        "dataset_config": config.sft_dataset_config,
        "dataset_split": config.sft_dataset_split,
        "tokenizer_name": config.tokenizer_name,
        "sequence_length": config.T,
    }
    for key, expected_val in expected_config.items():
        if state.get(key) != expected_val:
            raise RuntimeError(
                f"SFT tokenization configuration mismatch for '{key}': "
                f"expected {expected_val!r}, found {state.get(key)!r} in checkpoint."
            )

    pending_file = state.get("pending_tokens_file")
    pending_inputs: list[int] = []
    pending_targets: list[int] = []
    if pending_file:
        pending_path = output_dir / pending_file
        if not pending_path.is_file():
            raise RuntimeError(f"Pending SFT token buffer is missing: {pending_path}")
        loaded = np.load(pending_path, allow_pickle=False)
        pending_inputs = loaded[0].tolist()
        pending_targets = loaded[1].tolist()

    return state, (pending_inputs, pending_targets)


def save_sft_resume_state(
    output_dir: Path,
    state: dict[str, Any],
    pending_inputs: list[int],
    pending_targets: list[int],
    previous_pending_file: str | None = None,
) -> None:
    pending_filename = None
    if pending_inputs:
        pending_filename = f".pending_sft_tokens_{state['next_shard']:04d}.npy"
        pending_array = np.array([pending_inputs, pending_targets], dtype=np.int32)
        atomic_numpy_write(output_dir / pending_filename, pending_array)

    state["pending_tokens_file"] = pending_filename
    atomic_json_write(output_dir / "tokenization_state.json", state)

    if previous_pending_file and previous_pending_file != pending_filename:
        old_path = output_dir / previous_pending_file
        if old_path.exists():
            old_path.unlink()


def process_messages_batch(
    batch: dict[str, list[Any]],
    tokenizer: UltronTokenizer,
) -> tuple[list[int], list[int]]:
    """Convert conversational message rows into token inputs and masked targets."""
    batch_inputs: list[int] = []
    batch_targets: list[int] = []

    messages_column = batch.get("messages", [])
    for conversation in messages_column:
        if not conversation:
            continue
        inputs, targets = tokenizer.encode_chat_conversation(conversation)
        batch_inputs.extend(inputs)
        batch_targets.extend(targets)

    return batch_inputs, batch_targets


def tokenize_sft(
    output_dir: str = "shards_sft",
    shard_size_tokens: int = 5_000_000,
    max_shards: int | None = None,
    dataset_name: str | None = None,
    dataset_config: str | None = None,
    dataset_split: str | None = None,
    config: UltronConfig | None = None,
) -> None:
    """Download, tokenize, mask loss, and write SFT shards to disk."""
    config = config or UltronConfig()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dataset_name = dataset_name or config.sft_dataset_id
    dataset_config = dataset_config or config.sft_dataset_config
    dataset_split = dataset_split or config.sft_dataset_split

    tokenizer = UltronTokenizer(config.tokenizer_name)

    state, (pending_inputs, pending_targets) = load_sft_resume_state(
        output_path, shard_size_tokens, max_shards, config
    )

    if state is None:
        state = {
            "schema_version": STATE_SCHEMA_VERSION,
            "dataset_id": dataset_name,
            "dataset_config": dataset_config,
            "dataset_split": dataset_split,
            "tokenizer_name": config.tokenizer_name,
            "sequence_length": config.T,
            "shard_size_tokens": shard_size_tokens,
            "max_shards": max_shards,
            "next_shard": 0,
            "committed_tokens": 0,
            "source_documents_consumed": 0,
            "pending_tokens_file": None,
        }

    if is_tokenization_complete(state, shard_size_tokens, max_shards):
        print(
            f"All {max_shards} requested SFT shards are already complete in {output_path}."
        )
        return

    current_inputs: list[int] = list(pending_inputs)
    current_targets: list[int] = list(pending_targets)
    next_shard: int = int(state.get("next_shard") or 0)
    committed_tokens: int = int(state.get("committed_tokens") or 0)
    docs_consumed: int = int(state.get("source_documents_consumed") or 0)

    target_total = max_shards * shard_size_tokens if max_shards is not None else None
    telemetry = TokenizationTelemetry(
        target_tokens=target_total,
        start_tokens=committed_tokens + len(current_inputs),
    )

    stop_requested = False

    def handle_interrupt(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, handle_interrupt)
    signal.signal(signal.SIGTERM, handle_interrupt)

    telemetry.print_message(
        f"Loading streaming SFT dataset: {dataset_name} ({dataset_config})..."
    )
    dataset = load_dataset(
        dataset_name,
        dataset_config,
        split=dataset_split,
        streaming=True,
    )
    if docs_consumed > 0:
        telemetry.print_message(
            f"Fast-forwarding stream past {docs_consumed:,} consumed conversations..."
        )
        dataset = dataset.skip(docs_consumed)

    iterator = iter(dataset.iter(batch_size=500))

    try:
        while not stop_requested:
            if max_shards is not None and next_shard >= max_shards:
                break

            try:
                batch = next(iterator)
            except StopIteration:
                telemetry.print_message("End of streaming instruction dataset reached.")
                break

            new_inputs, new_targets = process_messages_batch(batch, tokenizer)
            current_inputs.extend(new_inputs)
            current_targets.extend(new_targets)
            batch_doc_count = len(batch.get("messages", []))
            docs_consumed += batch_doc_count

            telemetry.update(
                added_tokens=len(new_inputs),
                current_total=committed_tokens + len(current_inputs),
                shard_info=f"shard {next_shard:04d}",
            )

            # Write full shards
            while len(current_inputs) >= shard_size_tokens:
                if max_shards is not None and next_shard >= max_shards:
                    break

                shard_inputs = np.array(
                    current_inputs[:shard_size_tokens], dtype=np.uint16
                )
                shard_targets = np.array(
                    current_targets[:shard_size_tokens], dtype=np.int32
                )

                current_inputs = current_inputs[shard_size_tokens:]
                current_targets = current_targets[shard_size_tokens:]

                shard_inputs_path = (
                    output_path / f"smoltalk_shard_{next_shard:04d}_inputs.bin"
                )
                shard_targets_path = (
                    output_path / f"smoltalk_shard_{next_shard:04d}_targets.bin"
                )
                shard_meta_path = (
                    output_path / f"smoltalk_shard_{next_shard:04d}_meta.json"
                )

                atomic_shard_write(shard_inputs_path, shard_inputs)
                atomic_shard_write(shard_targets_path, shard_targets)

                loss_tokens_count = int(np.count_nonzero(shard_targets != -1))
                loss_ratio = round(loss_tokens_count / shard_size_tokens, 4)
                metadata = {
                    "shard_index": next_shard,
                    "tokens": shard_size_tokens,
                    "loss_tokens": loss_tokens_count,
                    "loss_token_ratio": loss_ratio,
                    "inputs_dtype": "uint16",
                    "targets_dtype": "int32",
                    "dataset_id": dataset_name,
                    "dataset_config": dataset_config,
                    "tokenizer_name": config.tokenizer_name,
                    "sequence_length": config.T,
                }
                atomic_json_write(shard_meta_path, metadata)

                telemetry.print_message(
                    f"Committed SFT shard {next_shard:04d} "
                    f"({shard_size_tokens:,} tokens, {loss_ratio * 100:.1f}% loss active)"
                )

                next_shard += 1
                committed_tokens += shard_size_tokens

                pending_val = state.get("pending_tokens_file")
                prev_pending: str | None = str(pending_val) if pending_val else None
                state.update(
                    {
                        "next_shard": next_shard,
                        "committed_tokens": committed_tokens,
                        "source_documents_consumed": docs_consumed,
                    }
                )
                save_sft_resume_state(
                    output_path,
                    state,
                    current_inputs,
                    current_targets,
                    previous_pending_file=prev_pending,
                )

    finally:
        close_iterator(iterator)
        telemetry.close()

    # Save final residual state
    final_pending = state.get("pending_tokens_file")
    prev_pending_final: str | None = str(final_pending) if final_pending else None
    state.update(
        {
            "next_shard": next_shard,
            "committed_tokens": committed_tokens,
            "source_documents_consumed": docs_consumed,
        }
    )
    save_sft_resume_state(
        output_path,
        state,
        current_inputs,
        current_targets,
        previous_pending_file=prev_pending_final,
    )
    print(
        f"\nSFT Tokenization run complete: {next_shard} committed shards "
        f"({committed_tokens:,} tokens) in {output_path}."
    )
