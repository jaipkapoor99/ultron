"""Exact-resume and atomic-shard helper tests."""

import json
from typing import Any

import numpy as np
import pytest

from config import UltronConfig
from scripts.tokenize_dataset import (
    _atomic_json_write,
    _atomic_shard_write,
    _is_tokenization_complete,
    _load_resume_state,
    _new_state,
    _save_resume_state,
)


def make_state(shard_size: int = 8, max_shards: int = 2) -> dict[str, Any]:
    return _new_state(
        config=UltronConfig(),
        dataset_revision="dataset-sha",
        tokenizer_revision="tokenizer-sha",
        shard_size_tokens=shard_size,
        max_shards=max_shards,
    )


def test_resume_state_round_trip_preserves_exact_pending_tokens(
    tmp_path: Any,
) -> None:
    state = make_state()
    pending = [3, 5, 8, 13]
    _save_resume_state(tmp_path, state, pending, previous_pending_file=None)

    restored_state, restored_pending = _load_resume_state(
        tmp_path,
        8,
        2,
        UltronConfig(),
    )

    assert restored_state is not None
    assert restored_state["source_documents_consumed"] == 0
    assert restored_pending.tolist() == pending


def test_committed_shard_size_is_validated(tmp_path: Any) -> None:
    state = make_state()
    values = np.arange(8, dtype=np.uint16)
    _atomic_shard_write(tmp_path / "fineweb_edu_shard_0000.bin", values)
    _atomic_json_write(
        tmp_path / "fineweb_edu_shard_0000_meta.json",
        {"shard_index": 0, "tokens": 8, "dtype": "uint16"},
    )
    state.update(
        {
            "next_shard": 1,
            "committed_tokens": 8,
            "source_documents_consumed": 2,
        }
    )
    _save_resume_state(tmp_path, state, [21], previous_pending_file=None)

    restored_state, restored_pending = _load_resume_state(
        tmp_path,
        8,
        2,
        UltronConfig(),
    )

    assert restored_state is not None
    assert restored_state["next_shard"] == 1
    assert restored_pending.tolist() == [21]

    (tmp_path / "fineweb_edu_shard_0000.bin").write_bytes(b"truncated")
    with pytest.raises(RuntimeError, match="invalid size"):
        _load_resume_state(tmp_path, 8, 2, UltronConfig())


def test_outputs_without_resume_state_are_rejected(tmp_path: Any) -> None:
    metadata = tmp_path / "fineweb_edu_shard_0000_meta.json"
    metadata.write_text(json.dumps({"shard_index": 0}))

    with pytest.raises(RuntimeError, match="without an exact resume checkpoint"):
        _load_resume_state(tmp_path, 8, 2, UltronConfig())


def test_complete_state_is_detected_without_reopening_dataset() -> None:
    state = make_state()
    state.update({"next_shard": 2, "committed_tokens": 16})

    assert _is_tokenization_complete(state, 8, 2)
    assert not _is_tokenization_complete(state, 8, 3)


def test_resume_rejects_configuration_drift(tmp_path: Any) -> None:
    state = make_state()
    _save_resume_state(tmp_path, state, [], previous_pending_file=None)
    changed_config = UltronConfig(dataset_config="different-config")

    with pytest.raises(RuntimeError, match="configuration mismatch"):
        _load_resume_state(tmp_path, 8, 2, changed_config)


def test_resume_rejects_missing_or_corrupt_pending_buffer(tmp_path: Any) -> None:
    state = make_state()
    _save_resume_state(tmp_path, state, [1, 2], previous_pending_file=None)
    pending_path = tmp_path / ".pending_tokens_0000.npy"

    pending_path.unlink()
    with pytest.raises(RuntimeError, match="buffer is missing"):
        _load_resume_state(tmp_path, 8, 2, UltronConfig())

    np.save(pending_path, np.array([1, 2], dtype=np.int64))
    with pytest.raises(RuntimeError, match="does not match"):
        _load_resume_state(tmp_path, 8, 2, UltronConfig())


def test_resume_rejects_inconsistent_committed_metadata(tmp_path: Any) -> None:
    state = make_state()
    _atomic_shard_write(
        tmp_path / "fineweb_edu_shard_0000.bin",
        np.arange(8, dtype=np.uint16),
    )
    _atomic_json_write(
        tmp_path / "fineweb_edu_shard_0000_meta.json",
        {"shard_index": 7, "tokens": 8, "dtype": "uint16"},
    )
    state.update({"next_shard": 1, "committed_tokens": 8})
    _save_resume_state(tmp_path, state, [], previous_pending_file=None)

    with pytest.raises(RuntimeError, match="Metadata is invalid"):
        _load_resume_state(tmp_path, 8, 2, UltronConfig())


def test_new_pending_checkpoint_removes_superseded_buffer(tmp_path: Any) -> None:
    state = make_state()
    _save_resume_state(tmp_path, state, [1], previous_pending_file=None)
    old_pending = tmp_path / ".pending_tokens_0000.npy"
    assert old_pending.exists()

    state["next_shard"] = 1
    _save_resume_state(
        tmp_path,
        state,
        [2, 3],
        previous_pending_file=old_pending.name,
    )

    assert not old_pending.exists()
    assert np.load(
        tmp_path / ".pending_tokens_0001.npy",
        allow_pickle=False,
    ).tolist() == [2, 3]
    assert not list(tmp_path.glob("*.tmp"))
