"""Complete training-state uploader regression tests."""

import sys
from typing import Any

from scripts import upload_checkpoint


def test_checkpoint_uploader_does_not_filter_training_state(
    tmp_path: Any, monkeypatch: Any
) -> None:
    checkpoint_dir = tmp_path / "accelerate_checkpoint"
    checkpoint_dir.mkdir()
    expected_files = {
        "model.safetensors",
        "optimizer.bin",
        "optimizer_1.bin",
        "scheduler.bin",
        "random_states_0.pkl",
        "training_state.json",
    }
    for filename in expected_files:
        (checkpoint_dir / filename).touch()

    class FakeApi:
        def __init__(self) -> None:
            self.uploaded_files: list[dict[str, Any]] = []

        def create_repo(self, **_kwargs: Any) -> None:
            pass

        def upload_file(self, **kwargs: Any) -> None:
            self.uploaded_files.append(kwargs)

    api = FakeApi()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setattr(upload_checkpoint, "login", lambda token: None)
    monkeypatch.setattr(upload_checkpoint, "HfApi", lambda token: api)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "upload_checkpoint.py",
            "--repo-id=test/ultron",
            f"--checkpoint-dir={checkpoint_dir}",
        ],
    )

    upload_checkpoint.main()

    assert {path.name for path in checkpoint_dir.iterdir()} == expected_files
    uploaded_names = {call["path_in_repo"] for call in api.uploaded_files}
    assert expected_files.issubset(uploaded_names)


def test_checkpoint_uploader_stops_before_hub_writes_when_directory_missing(
    tmp_path: Any,
    monkeypatch: Any,
) -> None:
    class FakeApi:
        def __init__(self) -> None:
            self.created = False
            self.uploaded = False

        def create_repo(self, **_kwargs: Any) -> None:
            self.created = True

        def upload_file(self, **_kwargs: Any) -> None:
            self.uploaded = True

    api = FakeApi()
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(upload_checkpoint, "HfApi", lambda token: api)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "upload_checkpoint.py",
            "--repo-id=test/ultron",
            "--checkpoint-dir=missing",
        ],
    )

    upload_checkpoint.main()

    assert api.created is False
    assert api.uploaded is False
