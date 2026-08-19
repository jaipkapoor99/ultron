"""Tokenize FineWeb-Edu into deterministic, atomically committed shards.

Resume checkpoints store the streaming dataset's native cursor plus the token
buffer left after the last committed shard. Dataset and tokenizer revisions
are pinned on the first run, so a restart reproduces the same stream.
"""

import argparse
import glob
import json
import os
import signal
import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset
from huggingface_hub import HfApi
from transformers import AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import UltronConfig
from telemetry import TokenizationTelemetry

STATE_SCHEMA_VERSION = 1


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json_write(path: Path, payload: dict) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w") as file:
        json.dump(payload, file, indent=2)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary_path, path)
    _fsync_directory(path.parent)


def _atomic_numpy_write(path: Path, values: np.ndarray) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("wb") as file:
        np.save(file, values, allow_pickle=False)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary_path, path)
    _fsync_directory(path.parent)


def _atomic_shard_write(path: Path, values: np.ndarray) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("wb") as file:
        values.tofile(file)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary_path, path)
    _fsync_directory(path.parent)


def _validate_committed_shards(
    output_dir: Path,
    next_shard: int,
    shard_size_tokens: int,
) -> None:
    expected_bytes = shard_size_tokens * np.dtype(np.uint16).itemsize
    for shard_index in range(next_shard):
        shard_path = output_dir / f"fineweb_edu_shard_{shard_index:04d}.bin"
        meta_path = output_dir / f"fineweb_edu_shard_{shard_index:04d}_meta.json"
        if not shard_path.is_file() or shard_path.stat().st_size != expected_bytes:
            raise RuntimeError(
                f"Committed shard {shard_index:04d} is missing or has an "
                f"invalid size; expected {expected_bytes:,} bytes."
            )
        if not meta_path.is_file():
            raise RuntimeError(f"Metadata is missing for shard {shard_index:04d}.")
        with meta_path.open() as file:
            metadata = json.load(file)
        if (
            metadata.get("shard_index") != shard_index
            or metadata.get("tokens") != shard_size_tokens
            or metadata.get("dtype") != "uint16"
        ):
            raise RuntimeError(f"Metadata is invalid for shard {shard_index:04d}.")


def _load_resume_state(
    output_dir: Path,
    shard_size_tokens: int,
    max_shards: int,
    config: UltronConfig,
) -> tuple[dict | None, np.ndarray]:
    state_path = output_dir / "tokenization_state.json"
    if not state_path.exists():
        existing_outputs = glob.glob(str(output_dir / "fineweb_edu_shard_*"))
        if existing_outputs:
            raise RuntimeError(
                "Shard files exist without an exact resume checkpoint. "
                "Move or delete them before starting."
            )
        return None, np.empty(0, dtype=np.uint16)

    with state_path.open() as file:
        state = json.load(file)
    expected = {
        "schema_version": STATE_SCHEMA_VERSION,
        "dataset_id": config.dataset_id,
        "dataset_config": config.dataset_config,
        "dataset_split": config.dataset_split,
        "shard_size_tokens": shard_size_tokens,
        "max_shards": max_shards,
    }
    mismatches = {
        key: (state.get(key), value)
        for key, value in expected.items()
        if state.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Resume checkpoint configuration mismatch: {mismatches}")

    next_shard = state["next_shard"]
    _validate_committed_shards(output_dir, next_shard, shard_size_tokens)

    pending_path = output_dir / state["pending_tokens_file"]
    if not pending_path.is_file():
        raise RuntimeError(f"Resume token buffer is missing: {pending_path}")
    pending_tokens = np.load(pending_path, allow_pickle=False)
    if (
        pending_tokens.dtype != np.uint16
        or len(pending_tokens) != state["pending_tokens"]
    ):
        raise RuntimeError("Resume token buffer does not match checkpoint metadata.")
    return state, pending_tokens


def _save_resume_state(
    output_dir: Path,
    state: dict,
    pending_tokens: list[int],
    previous_pending_file: str | None,
) -> None:
    pending_array = np.asarray(pending_tokens, dtype=np.uint16)
    pending_name = f".pending_tokens_{state['next_shard']:04d}.npy"
    _atomic_numpy_write(output_dir / pending_name, pending_array)

    state = {
        **state,
        "pending_tokens": len(pending_array),
        "pending_tokens_file": pending_name,
    }
    _atomic_json_write(output_dir / "tokenization_state.json", state)

    if previous_pending_file and previous_pending_file != pending_name:
        previous_path = output_dir / previous_pending_file
        if previous_path.exists():
            previous_path.unlink()


def _new_state(
    config: UltronConfig,
    dataset_revision: str,
    tokenizer_revision: str,
    shard_size_tokens: int,
    max_shards: int,
) -> dict:
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "dataset_id": config.dataset_id,
        "dataset_config": config.dataset_config,
        "dataset_split": config.dataset_split,
        "dataset_revision": dataset_revision,
        "tokenizer_name": config.tokenizer_name,
        "tokenizer_revision": tokenizer_revision,
        "shard_size_tokens": shard_size_tokens,
        "max_shards": max_shards,
        "next_shard": 0,
        "source_documents_consumed": 0,
        "committed_tokens": 0,
    }


def _is_tokenization_complete(
    state: dict,
    shard_size_tokens: int,
    max_shards: int,
) -> bool:
    """Return whether the requested shard set is fully committed."""
    return (
        state["next_shard"] == max_shards
        and state["committed_tokens"] == max_shards * shard_size_tokens
    )


def _close_iterator(iterator) -> None:
    """Best-effort cancellation of an exhausted streaming iterator."""
    close = getattr(iterator, "close", None)
    if callable(close):
        close()


def main(shard_size_tokens: int = 100_000_000, max_shards: int = 100) -> None:
    config = UltronConfig()
    output_dir = Path("shards_edu")
    output_dir.mkdir(parents=True, exist_ok=True)

    state, pending_array = _load_resume_state(
        output_dir,
        shard_size_tokens,
        max_shards,
        config,
    )

    if state is None:
        api = HfApi()
        dataset_info = api.dataset_info(config.dataset_id)
        tokenizer_info = api.model_info(config.tokenizer_name)
        if dataset_info.sha is None or tokenizer_info.sha is None:
            raise RuntimeError("Failed to resolve dataset or tokenizer revision sha.")
        dataset_revision = str(dataset_info.sha)
        tokenizer_revision = str(tokenizer_info.sha)
        state = _new_state(
            config,
            dataset_revision,
            tokenizer_revision,
            shard_size_tokens,
            max_shards,
        )
        _save_resume_state(output_dir, state, [], previous_pending_file=None)
        state["pending_tokens"] = 0
        state["pending_tokens_file"] = ".pending_tokens_0000.npy"

    if _is_tokenization_complete(state, shard_size_tokens, max_shards):
        print(
            f"Pre-tokenization already complete: "
            f"{state['committed_tokens']:,} tokens committed."
        )
        return

    tokenizer = AutoTokenizer.from_pretrained(
        state["tokenizer_name"],
        revision=state["tokenizer_revision"],
    )
    if tokenizer is None or tokenizer.vocab_size is None:
        raise ValueError("Failed to load tokenizer or vocabulary size.")
    if tokenizer.vocab_size > np.iinfo(np.uint16).max:
        raise ValueError("Tokenizer vocabulary does not fit in uint16 shards.")
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("Tokenizer must define an EOS token.")

    raw_dataset = load_dataset(
        state["dataset_id"],
        name=state["dataset_config"],
        split=state["dataset_split"],
        streaming=True,
        revision=state["dataset_revision"],
    )
    documents_consumed = state["source_documents_consumed"]
    dataset_state = state.get("dataset_state")
    if dataset_state is not None:
        raw_dataset.load_state_dict(dataset_state)
        print(
            "Restored native dataset cursor at "
            f"{documents_consumed:,} source documents."
        )
    elif documents_consumed:
        raise RuntimeError(
            "Resume checkpoint is missing its native dataset cursor. "
            "Start again with an empty shards directory."
        )
    iterator = iter(raw_dataset)

    shard_index = state["next_shard"]
    current_tokens = pending_array.tolist()
    target_total_tokens = max_shards * shard_size_tokens
    committed_tokens = state["committed_tokens"]

    print("=== FINEWEB-EDU PRE-TOKENIZATION PIPELINE ===")
    print(f"Dataset Revision  : {state['dataset_revision']}")
    print(f"Tokenizer Revision: {state['tokenizer_revision']}")
    print(f"Committed Shards  : {shard_index}/{max_shards}")
    print(f"Source Documents  : {documents_consumed:,}")
    print(f"Buffered Tokens   : {len(current_tokens):,}")

    telemetry = TokenizationTelemetry(
        target_tokens=target_total_tokens,
        start_tokens=committed_tokens + len(current_tokens),
    )
    stop_requested = False

    def handle_interrupt(_signum, _frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, handle_interrupt)
    signal.signal(signal.SIGTERM, handle_interrupt)

    batch_size = 1000
    try:
        while shard_index < max_shards and not stop_requested:
            batch_docs = []
            for _ in range(batch_size):
                if stop_requested:
                    break
                try:
                    batch_docs.append(next(iterator)["text"])
                    documents_consumed += 1
                except StopIteration:
                    break
            if not batch_docs:
                break

            encodings = tokenizer.backend_tokenizer.encode_batch(
                batch_docs,
                add_special_tokens=False,
            )
            added_tokens = 0
            for encoding in encodings:
                document_tokens = encoding.ids
                current_tokens.extend(document_tokens)
                current_tokens.append(eos_token_id)
                added_tokens += len(document_tokens) + 1
            telemetry.update(
                added_tokens=added_tokens,
                current_total=committed_tokens + len(current_tokens),
            )

            while len(current_tokens) >= shard_size_tokens and shard_index < max_shards:
                shard_values = np.asarray(
                    current_tokens[:shard_size_tokens],
                    dtype=np.uint16,
                )
                current_tokens = current_tokens[shard_size_tokens:]

                shard_path = output_dir / f"fineweb_edu_shard_{shard_index:04d}.bin"
                meta_path = (
                    output_dir / f"fineweb_edu_shard_{shard_index:04d}_meta.json"
                )
                _atomic_shard_write(shard_path, shard_values)
                _atomic_json_write(
                    meta_path,
                    {
                        "shard_index": shard_index,
                        "tokens": shard_size_tokens,
                        "vocab_size": tokenizer.vocab_size,
                        "dtype": "uint16",
                        "dataset_revision": state["dataset_revision"],
                        "tokenizer_revision": state["tokenizer_revision"],
                    },
                )

                shard_index += 1
                committed_tokens = shard_index * shard_size_tokens
                previous_pending_file = state.get("pending_tokens_file")
                state.update(
                    {
                        "next_shard": shard_index,
                        "source_documents_consumed": documents_consumed,
                        "committed_tokens": committed_tokens,
                        "dataset_state": raw_dataset.state_dict(),
                    }
                )
                _save_resume_state(
                    output_dir,
                    state,
                    current_tokens,
                    previous_pending_file=previous_pending_file,
                )
                state["pending_tokens_file"] = f".pending_tokens_{shard_index:04d}.npy"
                telemetry.print_message(
                    f"✓ Atomically committed shard {shard_index - 1:04d}"
                )
    finally:
        _close_iterator(iterator)
        telemetry.close()

    if stop_requested:
        print(
            "\nStopped safely. Uncommitted work will be replayed from the "
            "last exact checkpoint."
        )
    else:
        print(f"\nPre-tokenization complete: {committed_tokens:,} tokens committed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-size-tokens", type=int, default=100_000_000)
    parser.add_argument("--max-shards", type=int, default=100)
    arguments = parser.parse_args()
    if arguments.shard_size_tokens <= 0 or arguments.max_shards <= 0:
        parser.error("shard size and shard count must be greater than zero")
    main(arguments.shard_size_tokens, arguments.max_shards)
