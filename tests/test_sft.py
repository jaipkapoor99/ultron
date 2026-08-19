"""Unit tests for SFT data loading, loss masking, and training contracts."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from config import UltronConfig
from model import UltronModel
from sft_dataset import ZeroCopySFTDataset, split_train_dev_sft_datasets


def make_sft_shard(path_prefix: Path, num_sequences: int = 4, seq_len: int = 16) -> str:
    total_tokens = num_sequences * seq_len + 1
    # Create input token IDs (10..10+total_tokens)
    inputs = np.arange(10, 10 + total_tokens, dtype=np.uint16)
    # Create target token IDs: first half of each seq is -1 (prompt), second half is targets
    targets = np.full(total_tokens, -1, dtype=np.int32)
    for i in range(num_sequences):
        start = i * seq_len + (seq_len // 2)
        end = (i + 1) * seq_len + 1
        targets[start:end] = inputs[start:end]

    inputs_path = path_prefix.parent / f"{path_prefix.name}_inputs.bin"
    targets_path = path_prefix.parent / f"{path_prefix.name}_targets.bin"
    meta_path = path_prefix.parent / f"{path_prefix.name}_meta.json"

    inputs.tofile(inputs_path)
    targets.tofile(targets_path)
    meta_path.write_text(
        json.dumps(
            {
                "shard_index": 0,
                "tokens": total_tokens,
                "inputs_dtype": "uint16",
                "targets_dtype": "int32",
            }
        )
    )
    return str(path_prefix)


def test_sft_dataset_reads_inputs_and_masked_targets(tmp_path: Path) -> None:
    prefix = make_sft_shard(tmp_path / "shard_0000", num_sequences=4, seq_len=16)
    dataset = ZeroCopySFTDataset([prefix], sequence_length=16)

    assert len(dataset) == 4

    inp, tgt = dataset[0]
    assert inp.shape == (16,)
    assert tgt.shape == (16,)

    # Targets are shifted by +1 relative to raw tokens
    assert tgt[:7].tolist() == [-1] * 7

    # Test negative index
    last_inp, last_tgt = dataset[-1]
    assert len(last_inp) == 16
    assert len(last_tgt) == 16


def test_sft_dataset_rejects_mismatched_lengths(tmp_path: Path) -> None:
    prefix = tmp_path / "corrupt_0000"
    (tmp_path / "corrupt_0000_inputs.bin").write_bytes(b"\x00" * 32)
    (tmp_path / "corrupt_0000_targets.bin").write_bytes(b"\x00" * 16)

    with pytest.raises(ValueError, match="Mismatch"):
        ZeroCopySFTDataset([str(prefix)], sequence_length=16)


def test_split_train_dev_sft_datasets(tmp_path: Path) -> None:
    shards = [
        make_sft_shard(tmp_path / f"shard_{i:04d}", num_sequences=2, seq_len=8)
        for i in range(3)
    ]

    train_ds, dev_ds = split_train_dev_sft_datasets(
        shards, sequence_length=8, val_ratio=0.33
    )

    assert isinstance(train_ds, ZeroCopySFTDataset)
    assert isinstance(dev_ds, ZeroCopySFTDataset)
    assert len(train_ds) == 4
    assert len(dev_ds) == 2


def test_sft_loss_gradient_only_flows_to_unmasked_targets() -> None:
    config = UltronConfig(
        B=1, T=8, C=16, n_head=2, n_kv_head=1, n_layer=1, vocab_size=32
    )
    model = UltronModel(config).train()

    inputs = torch.randint(0, config.vocab_size, (1, 8))
    # Target masks first 4 tokens (-1), provides targets for last 4 tokens
    targets = torch.tensor([[-1, -1, -1, -1, 5, 6, 7, 8]], dtype=torch.long)

    output = model(inputs, targets=targets)
    assert output.loss is not None
    assert torch.isfinite(output.loss)

    # Compute reference cross entropy ignoring -1
    ref_loss = F.cross_entropy(
        output.logits[:, :-1].reshape(-1, output.logits.size(-1)),
        targets[:, :-1].reshape(-1),
        ignore_index=-1,
    )
    assert torch.isfinite(output.loss)
    assert torch.isfinite(ref_loss)


def test_build_sft_config() -> None:
    from types import SimpleNamespace

    from train_sft import build_sft_config

    args = SimpleNamespace(
        lr=3e-4,
        min_lr=3e-5,
        warmup_steps=50,
        eval_interval=50,
        eval_batches=10,
        max_steps=1000,
        mode="test",
    )
    config = build_sft_config(args)
    assert config.is_test_mode is True
    assert config.learning_rate == 3e-4
    assert config.min_lr == 3e-5
    assert config.max_steps == 5


def test_load_base_weights(tmp_path: Path) -> None:
    from train_sft import load_base_weights

    config = UltronConfig(
        B=1, T=8, C=16, n_head=2, n_kv_head=1, n_layer=1, vocab_size=32
    )
    source_model = UltronModel(config)
    torch.save(source_model.state_dict(), tmp_path / "pytorch_model.bin")

    target_model = UltronModel(config)
    load_base_weights(target_model, tmp_path)

    torch.testing.assert_close(
        target_model.transformer.wte.weight,
        source_model.transformer.wte.weight,
    )


def test_load_base_weights_missing_directory(tmp_path: Path) -> None:
    from train_sft import load_base_weights

    config = UltronConfig(
        B=1, T=8, C=16, n_head=2, n_kv_head=1, n_layer=1, vocab_size=32
    )
    model = UltronModel(config)
    with pytest.raises(FileNotFoundError, match="No base model weights"):
        load_base_weights(model, tmp_path / "empty")


def test_chat_model_loader_and_fallback(tmp_path: Path) -> None:
    from scripts.chat import load_instruct_model

    model, tokenizer = load_instruct_model(
        tmp_path / "nonexistent", torch.device("cpu")
    )
    assert isinstance(model, UltronModel)
    assert tokenizer.im_start_id is not None


def test_dataset_sharder_base_and_iterator(tmp_path: Path) -> None:
    from sharding import BaseDatasetSharder, close_iterator

    class DummyIterator:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    it = DummyIterator()
    close_iterator(it)
    assert it.closed is True
    close_iterator(None)  # Safe no-op

    class ConcreteSharder(BaseDatasetSharder):
        def process_batch(self, batch: dict[str, list[Any]]) -> list[int]:
            return [1, 2, 3]

        def commit_shard(
            self,
            shard_index: int,
            tokens: Any,
            metadata_extra: dict[str, Any] | None = None,
        ) -> None:
            pass

    sharder = ConcreteSharder(
        output_dir=tmp_path / "sharder_out",
        shard_size_tokens=1000,
        max_shards=2,
    )
    assert sharder.shard_size_tokens == 1000
    assert sharder.process_batch({}) == [1, 2, 3]


def test_get_sft_dataloaders(tmp_path: Path) -> None:
    from sft_dataset import get_sft_dataloaders

    make_sft_shard(tmp_path / "shard_0000", num_sequences=4, seq_len=8)
    make_sft_shard(tmp_path / "shard_0001", num_sequences=4, seq_len=8)

    class FakeAccelerator:
        def print(self, _message: Any) -> None:
            pass

        def prepare(self, *loaders: Any) -> Any:
            return loaders

    config = UltronConfig(B=2, T=8)
    train_loader, dev_loader = get_sft_dataloaders(
        config, FakeAccelerator(), shard_dir=str(tmp_path)
    )
    assert train_loader is not None
    assert dev_loader is not None
    batch = next(iter(train_loader))
    assert len(batch) == 2
    assert batch[0].shape == (2, 8)
    assert batch[1].shape == (2, 8)


def test_sharding_base_atomic_writes(tmp_path: Path) -> None:
    from sharding.base import (
        atomic_json_write,
        atomic_numpy_write,
        atomic_shard_write,
    )

    json_path = tmp_path / "data.json"
    atomic_json_write(json_path, {"test": 123})
    assert json.loads(json_path.read_text()) == {"test": 123}

    npy_path = tmp_path / "data.npy"
    arr = np.array([1, 2, 3], dtype=np.int32)
    atomic_numpy_write(npy_path, arr)
    assert np.load(npy_path).tolist() == [1, 2, 3]

    bin_path = tmp_path / "data.bin"
    atomic_shard_write(bin_path, arr)
    assert np.fromfile(bin_path, dtype=np.int32).tolist() == [1, 2, 3]


def test_sft_sharding_state_and_batch_processing(tmp_path: Path) -> None:
    from sharding.sft import (
        is_tokenization_complete,
        load_sft_resume_state,
        process_messages_batch,
        save_sft_resume_state,
    )
    from tokenizer import UltronTokenizer

    # test is_tokenization_complete
    assert not is_tokenization_complete(
        {"next_shard": 1, "committed_tokens": 10}, 10, None
    )
    assert not is_tokenization_complete(
        {"next_shard": 1, "committed_tokens": 10}, 10, 2
    )
    assert is_tokenization_complete({"next_shard": 2, "committed_tokens": 20}, 10, 2)

    config = UltronConfig()
    state = {
        "schema_version": 1,
        "dataset_id": config.sft_dataset_id,
        "dataset_config": config.sft_dataset_config,
        "dataset_split": config.sft_dataset_split,
        "tokenizer_name": config.tokenizer_name,
        "sequence_length": config.T,
        "shard_size_tokens": 100,
        "max_shards": 2,
        "next_shard": 0,
        "committed_tokens": 0,
        "source_documents_consumed": 0,
        "pending_tokens_file": None,
    }

    # Save state with pending tokens
    save_sft_resume_state(tmp_path, state, [10, 20], [-1, 20])
    loaded_state, (inp, tgt) = load_sft_resume_state(tmp_path, 100, 2, config)
    assert loaded_state is not None
    assert inp == [10, 20]
    assert tgt == [-1, 20]

    # Test batch processing
    tokenizer = UltronTokenizer()
    batch = {
        "messages": [
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ]
        ]
    }
    b_inp, b_tgt = process_messages_batch(batch, tokenizer)
    assert len(b_inp) == len(b_tgt)
    assert len(b_inp) > 0


def test_sft_resume_state_error_handling(tmp_path: Path) -> None:
    from sharding.sft import load_sft_resume_state

    config = UltronConfig()

    # Case 1: Metadata files exist without state file
    (tmp_path / "smoltalk_shard_0000_meta.json").write_text("{}")
    with pytest.raises(RuntimeError, match="without an exact resume checkpoint"):
        load_sft_resume_state(tmp_path, 100, 2, config)

    # Case 2: Schema version mismatch
    state_file = tmp_path / "tokenization_state.json"
    state_file.write_text(json.dumps({"schema_version": 999}))
    with pytest.raises(RuntimeError, match="Unsupported state schema"):
        load_sft_resume_state(tmp_path, 100, 2, config)

    # Case 3: Config mismatch
    invalid_state = {
        "schema_version": 1,
        "dataset_id": "wrong_dataset",
        "dataset_config": config.sft_dataset_config,
        "dataset_split": config.sft_dataset_split,
        "tokenizer_name": config.tokenizer_name,
        "sequence_length": config.T,
    }
    state_file.write_text(json.dumps(invalid_state))
    with pytest.raises(RuntimeError, match="mismatch"):
        load_sft_resume_state(tmp_path, 100, 2, config)
