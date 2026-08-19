"""Complete training-state uploader regression tests."""

import sys

from scripts import upload_checkpoint


def test_checkpoint_uploader_does_not_filter_training_state(tmp_path, monkeypatch):
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
        def __init__(self):
            self.upload_kwargs = None

        def create_repo(self, **_kwargs):
            pass

        def upload_folder(self, **kwargs):
            self.upload_kwargs = kwargs

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
    assert api.upload_kwargs is not None
    assert api.upload_kwargs["folder_path"] == str(checkpoint_dir)
    assert "allow_patterns" not in api.upload_kwargs
    assert "ignore_patterns" not in api.upload_kwargs


def test_checkpoint_uploader_stops_before_hub_writes_when_directory_missing(
    tmp_path,
    monkeypatch,
):
    class FakeApi:
        def __init__(self):
            self.created = False
            self.uploaded = False

        def create_repo(self, **_kwargs):
            self.created = True

        def upload_folder(self, **_kwargs):
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
