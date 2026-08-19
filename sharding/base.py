"""Base abstractions and atomic I/O utilities for dataset sharding."""

import json
import os
import signal
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from config import UltronConfig
from tokenizer import UltronTokenizer

STATE_SCHEMA_VERSION = 1


def fsync_directory(path: Path) -> None:
    """Flush directory metadata changes to physical storage."""
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    """Atomically commit a JSON file to disk."""
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary_path, path)
    fsync_directory(path.parent)


def atomic_numpy_write(path: Path, values: np.ndarray) -> None:
    """Atomically commit an unpickled NumPy array to disk."""
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("wb") as file:
        np.save(file, values, allow_pickle=False)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary_path, path)
    fsync_directory(path.parent)


def atomic_shard_write(path: Path, values: np.ndarray) -> None:
    """Atomically commit a raw binary shard array to disk."""
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("wb") as file:
        values.tofile(file)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary_path, path)
    fsync_directory(path.parent)


def close_iterator(iterator: Any) -> None:
    """Best-effort cancellation of an exhausted or interrupted streaming iterator."""
    close = getattr(iterator, "close", None)
    if callable(close):
        close()


class BaseDatasetSharder(ABC):
    """Base class for streaming dataset tokenizers and sharding pipelines."""

    def __init__(
        self,
        output_dir: str | Path,
        shard_size_tokens: int = 100_000_000,
        max_shards: int | None = None,
        config: UltronConfig | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.shard_size_tokens = shard_size_tokens
        self.max_shards = max_shards
        self.config = config or UltronConfig()
        self.tokenizer = UltronTokenizer(self.config.tokenizer_name)
        self.stop_requested = False

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._setup_signals()

    def _setup_signals(self) -> None:
        def handle_interrupt(_signum: int, _frame: Any) -> None:
            self.stop_requested = True

        signal.signal(signal.SIGINT, handle_interrupt)
        signal.signal(signal.SIGTERM, handle_interrupt)

    @abstractmethod
    def process_batch(
        self, batch: dict[str, list[Any]]
    ) -> list[int] | tuple[list[int], list[int]]:
        """Tokenize a raw document batch into token streams."""
        raise NotImplementedError

    @abstractmethod
    def commit_shard(
        self,
        shard_index: int,
        tokens: Any,
        metadata_extra: dict[str, Any] | None = None,
    ) -> None:
        """Write a full shard and its metadata atomically to disk."""
        raise NotImplementedError
