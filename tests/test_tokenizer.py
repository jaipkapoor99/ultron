"""Tests for unified UltronTokenizer and atomic file operations."""

import json
from pathlib import Path

import numpy as np
import pytest

from tokenizer import (
    UltronTokenizer,
    atomic_json_write,
    atomic_numpy_write,
    atomic_shard_write,
)


def test_atomic_file_writes(tmp_path: Path) -> None:
    json_path = tmp_path / "test.json"
    atomic_json_write(json_path, {"key": "value"})
    assert json.loads(json_path.read_text(encoding="utf-8")) == {"key": "value"}
    assert not json_path.with_suffix(".json.tmp").exists()

    npy_path = tmp_path / "test.npy"
    data = np.array([1, 2, 3, 4], dtype=np.uint16)
    atomic_numpy_write(npy_path, data)
    assert np.load(npy_path).tolist() == [1, 2, 3, 4]
    assert not npy_path.with_suffix(".npy.tmp").exists()

    bin_path = tmp_path / "test.bin"
    atomic_shard_write(bin_path, data)
    assert np.fromfile(bin_path, dtype=np.uint16).tolist() == [1, 2, 3, 4]
    assert not bin_path.with_suffix(".bin.tmp").exists()


class FakeHfTokenizer:
    def __init__(self) -> None:
        self.eos_token_id = 0
        self.bos_token_id = 1
        self.pad_token_id = 2
        self.chat_template = None
        self._vocab = {
            "<|im_start|>": 10,
            "<|im_end|>": 11,
            "<|endoftext|>": 0,
            "system\n": 20,
            "user\n": 21,
            "assistant\n": 22,
            "hello": 30,
            "world": 31,
            "hi": 32,
        }

    def __len__(self) -> int:
        return 100

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._vocab[token]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        tokens = [self._vocab.get(word, 99) for word in text.split()]
        return tokens or [99]

    def decode(self, tokens: list[int], skip_special_tokens: bool = False) -> str:
        inv = {v: k for k, v in self._vocab.items()}
        return " ".join(inv.get(t, "<unk>") for t in tokens)


def test_ultron_tokenizer_loss_masking(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = UltronTokenizer.__new__(UltronTokenizer)
    tokenizer.hf_tokenizer = FakeHfTokenizer()
    tokenizer.im_start_id = 10
    tokenizer.im_end_id = 11
    tokenizer.eos_token_id = 0
    tokenizer.bos_token_id = 1
    tokenizer.pad_token_id = 2

    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]

    inputs, targets = tokenizer.encode_chat_conversation(
        messages, system_prompt="world"
    )

    assert len(inputs) == len(targets)

    # All tokens belonging to system and user turns must be -1 in targets
    # Assistant content tokens must be > -1
    # Check that there are masked tokens (-1) and unmasked tokens (> 0)
    assert -1 in targets
    unmasked = [t for t in targets if t != -1]
    assert len(unmasked) > 0
    # The unmasked targets must match the corresponding slice of inputs
    for inp, tgt in zip(inputs, targets, strict=True):
        if tgt != -1:
            assert inp == tgt


def test_apply_chat_template_fallback() -> None:
    tokenizer = UltronTokenizer.__new__(UltronTokenizer)
    tokenizer.hf_tokenizer = FakeHfTokenizer()
    tokenizer.im_start_id = 10
    tokenizer.im_end_id = 11
    tokenizer.eos_token_id = 0
    tokenizer.bos_token_id = 1
    tokenizer.pad_token_id = 2

    formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": "hello"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    assert "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n" in formatted
