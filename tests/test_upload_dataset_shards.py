"""Dataset uploader validation tests."""

import json
from typing import Any

import numpy as np
import pytest

from sharding import validate_complete_shard_set


def write_complete_set(
    directory: Any, shard_count: int = 2, shard_size: int = 8
) -> dict[str, Any]:
    state = {
        "max_shards": shard_count,
        "next_shard": shard_count,
        "shard_size_tokens": shard_size,
        "committed_tokens": shard_count * shard_size,
        "dataset_revision": "dataset-sha",
        "tokenizer_revision": "tokenizer-sha",
    }
    (directory / "tokenization_state.json").write_text(json.dumps(state))
    for index in range(shard_count):
        np.arange(shard_size, dtype=np.uint16).tofile(
            directory / f"fineweb_edu_shard_{index:04d}.bin"
        )
        metadata = {
            "shard_index": index,
            "tokens": shard_size,
            "dtype": "uint16",
            "dataset_revision": "dataset-sha",
            "tokenizer_revision": "tokenizer-sha",
        }
        (directory / f"fineweb_edu_shard_{index:04d}_meta.json").write_text(
            json.dumps(metadata)
        )
    return state


def test_complete_shard_set_is_accepted(tmp_path: Any) -> None:
    expected = write_complete_set(tmp_path)

    assert validate_complete_shard_set(tmp_path) == expected


def test_complete_sft_shard_set_is_accepted(tmp_path: Any) -> None:
    state = {
        "max_shards": 2,
        "next_shard": 2,
        "shard_size_tokens": 8,
        "committed_tokens": 16,
    }
    (tmp_path / "tokenization_state.json").write_text(json.dumps(state))
    for index in range(2):
        np.arange(8, dtype=np.uint16).tofile(
            tmp_path / f"smoltalk_shard_{index:04d}_inputs.bin"
        )
        np.arange(8, dtype=np.int32).tofile(
            tmp_path / f"smoltalk_shard_{index:04d}_targets.bin"
        )
        metadata = {
            "shard_index": index,
            "tokens": 8,
            "inputs_dtype": "uint16",
            "targets_dtype": "int32",
        }
        (tmp_path / f"smoltalk_shard_{index:04d}_meta.json").write_text(
            json.dumps(metadata)
        )

    assert validate_complete_shard_set(tmp_path, is_sft=True) == state


def test_incomplete_shard_set_is_rejected(tmp_path: Any) -> None:
    state = write_complete_set(tmp_path)
    state["next_shard"] = 1
    (tmp_path / "tokenization_state.json").write_text(json.dumps(state))

    with pytest.raises(RuntimeError, match="incomplete"):
        validate_complete_shard_set(tmp_path)


def test_inconsistent_metadata_is_rejected(tmp_path: Any) -> None:
    write_complete_set(tmp_path)
    metadata_path = tmp_path / "fineweb_edu_shard_0001_meta.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["dataset_revision"] = "wrong"
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(RuntimeError, match="inconsistent"):
        validate_complete_shard_set(tmp_path)


def test_missing_state_is_rejected(tmp_path: Any) -> None:
    with pytest.raises(RuntimeError, match="Missing exact tokenization state"):
        validate_complete_shard_set(tmp_path)


@pytest.mark.parametrize("missing_kind", ["shard", "metadata"])
def test_missing_committed_files_are_rejected(tmp_path: Any, missing_kind: str) -> None:
    write_complete_set(tmp_path)
    if missing_kind == "shard":
        (tmp_path / "fineweb_edu_shard_0001.bin").unlink()
        expected = "missing or has an invalid size"
    else:
        (tmp_path / "fineweb_edu_shard_0001_meta.json").unlink()
        expected = "Metadata.*missing"

    with pytest.raises(RuntimeError, match=expected):
        validate_complete_shard_set(tmp_path)


def test_wrong_shard_size_is_rejected(tmp_path: Any) -> None:
    write_complete_set(tmp_path)
    (tmp_path / "fineweb_edu_shard_0001.bin").write_bytes(b"short")

    with pytest.raises(RuntimeError, match="invalid size"):
        validate_complete_shard_set(tmp_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dtype", "int32"),
        ("tokens", 7),
        ("shard_index", 99),
        ("tokenizer_revision", "wrong"),
    ],
)
def test_every_metadata_contract_field_is_validated(
    tmp_path: Any,
    field: str,
    value: Any,
) -> None:
    write_complete_set(tmp_path)
    path = tmp_path / "fineweb_edu_shard_0001_meta.json"
    metadata = json.loads(path.read_text())
    metadata[field] = value
    path.write_text(json.dumps(metadata))

    with pytest.raises(RuntimeError, match="inconsistent"):
        validate_complete_shard_set(tmp_path)


def test_upload_dataset_shards_invokes_hf_api(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sharding.uploader import upload_dataset_shards

    write_complete_set(tmp_path)

    calls: list[str] = []

    class FakeHfApi:
        def __init__(self, token: str | None = None) -> None:
            pass

        def create_repo(
            self, repo_id: str, repo_type: str, exist_ok: bool, private: bool
        ) -> None:
            calls.append(f"create_repo:{repo_id}")

        def upload_large_folder(self, **kwargs: Any) -> None:
            calls.append("upload_large_folder")

    monkeypatch.setattr("sharding.uploader.HfApi", FakeHfApi)

    upload_dataset_shards(
        target_repo="test/repo",
        shards_dir=tmp_path,
        is_sft=False,
        private=False,
    )
    assert "create_repo:test/repo" in calls
    assert "upload_large_folder" in calls
