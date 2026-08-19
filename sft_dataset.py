"""Zero-Copy Memory-Mapped Dataset for Supervised Fine-Tuning (SFT).

Provides fast, forkserver-safe loading of paired (inputs, targets) binary shards
where user/system prompts are pre-masked with -1 for loss calculation.
"""

import glob
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler

from config import UltronConfig
from dataset import EpochRandomSampler


class ZeroCopySFTDataset(Dataset):
    """Memory-mapped SFT dataset with process-local, lazily opened shard views."""

    def __init__(
        self,
        shard_prefixes: Sequence[str | Path],
        sequence_length: int = 1024,
    ) -> None:
        self.shard_prefixes = [str(p) for p in shard_prefixes]
        self.T = sequence_length

        if self.T <= 0:
            raise ValueError(f"sequence_length must be > 0, got {self.T}")

        self.shard_lengths: list[int] = []
        self.shard_windows: list[int] = []
        self.shard_offsets: list[tuple[int, int]] = []
        self.shard_ends: list[int] = []

        total_windows = 0
        for prefix in self.shard_prefixes:
            inputs_file = f"{prefix}_inputs.bin"
            if not os.path.exists(inputs_file):
                raise FileNotFoundError(f"Missing SFT inputs shard: {inputs_file}")
            targets_file = f"{prefix}_targets.bin"
            if not os.path.exists(targets_file):
                raise FileNotFoundError(f"Missing SFT targets shard: {targets_file}")

            token_count = os.path.getsize(inputs_file) // 2  # uint16 (2 bytes)
            target_tokens = os.path.getsize(targets_file) // 4  # int32 (4 bytes)
            if token_count != target_tokens:
                raise ValueError(
                    f"Mismatch between inputs ({token_count}) and targets ({target_tokens}) in {prefix}"
                )

            windows = max(0, (token_count - 1) // self.T)
            start_window = total_windows
            total_windows += windows

            self.shard_lengths.append(token_count)
            self.shard_windows.append(windows)
            self.shard_offsets.append((start_window, total_windows))
            self.shard_ends.append(total_windows)

        self._total_windows = total_windows
        self._inputs_memmaps: dict[str, np.memmap] = {}
        self._targets_memmaps: dict[str, np.memmap] = {}
        self._memmap_owner_pid: int | None = None

    def __len__(self) -> int:
        return self._total_windows

    def _ensure_memmaps(self) -> None:
        current_pid = os.getpid()
        if self._memmap_owner_pid != current_pid:
            self._inputs_memmaps = {}
            self._targets_memmaps = {}
            self._memmap_owner_pid = current_pid

    def _get_shard_memmaps(self, shard_index: int) -> tuple[np.memmap, np.memmap]:
        self._ensure_memmaps()
        prefix = self.shard_prefixes[shard_index]
        if prefix not in self._inputs_memmaps:
            self._inputs_memmaps[prefix] = np.memmap(
                f"{prefix}_inputs.bin",
                dtype=np.uint16,
                mode="r",
            )
            self._targets_memmaps[prefix] = np.memmap(
                f"{prefix}_targets.bin",
                dtype=np.int32,
                mode="r",
            )
        return self._inputs_memmaps[prefix], self._targets_memmaps[prefix]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0:
            index += self._total_windows
        if not (0 <= index < self._total_windows):
            raise IndexError(
                f"Index {index} out of range for SFT dataset with {self._total_windows} windows"
            )

        shard_idx = int(np.searchsorted(self.shard_ends, index, side="right"))
        start_window = self.shard_offsets[shard_idx][0]
        window_in_shard = index - start_window

        offset = window_in_shard * self.T
        inputs_mmap, targets_mmap = self._get_shard_memmaps(shard_idx)

        inp = inputs_mmap[offset : offset + self.T].astype(np.int64)
        tgt = targets_mmap[offset + 1 : offset + self.T + 1].astype(np.int64)

        return torch.from_numpy(inp), torch.from_numpy(tgt)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_inputs_memmaps"] = {}
        state["_targets_memmaps"] = {}
        state["_memmap_owner_pid"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._inputs_memmaps = {}
        self._targets_memmaps = {}
        self._memmap_owner_pid = None


def split_train_dev_sft_datasets(
    shard_prefixes: Sequence[str | Path],
    sequence_length: int = 1024,
    val_ratio: float = 0.05,
) -> tuple[Any, Any]:
    """Split SFT shards cleanly at shard boundaries between training and validation."""
    shards = list(shard_prefixes)
    if not shards:
        raise ValueError("No SFT shard prefixes provided.")

    if len(shards) > 1:
        num_dev_shards = max(1, int(len(shards) * val_ratio))
        train_shards = shards[:-num_dev_shards]
        dev_shards = shards[-num_dev_shards:]
        return (
            ZeroCopySFTDataset(train_shards, sequence_length=sequence_length),
            ZeroCopySFTDataset(dev_shards, sequence_length=sequence_length),
        )

    full_ds = ZeroCopySFTDataset(shards, sequence_length=sequence_length)
    dev_size = max(1, int(len(full_ds) * val_ratio))
    train_size = len(full_ds) - dev_size
    train_ds, dev_ds = torch.utils.data.random_split(
        full_ds,
        [train_size, dev_size],
        generator=torch.Generator().manual_seed(42),
    )
    return train_ds, dev_ds


def get_sft_dataloaders(
    config: UltronConfig,
    accelerator: Any,
    shard_dir: str = "shards_sft",
) -> tuple[DataLoader, DataLoader]:
    """Discover SFT shards and instantiate reproducible DataLoaders."""
    meta_files = sorted(glob.glob(os.path.join(shard_dir, "*_meta.json")))
    if not meta_files:
        raise FileNotFoundError(
            f"No SFT shard metadata files found in '{shard_dir}'. "
            f"Run scripts/tokenize_sft_dataset.py first."
        )

    shard_prefixes = [
        os.path.splitext(meta_path)[0].replace("_meta", "") for meta_path in meta_files
    ]

    train_ds: Any
    dev_ds: Any
    train_ds, dev_ds = split_train_dev_sft_datasets(
        shard_prefixes,
        sequence_length=config.T,
    )

    accelerator.print(
        f"Initialized SFT dataset: {len(train_ds):,} training sequences, "
        f"{len(dev_ds):,} dev sequences (context T={config.T})."
    )

    train_sampler = EpochRandomSampler(train_ds, seed=config.data_seed)
    dev_sampler = SequentialSampler(dev_ds)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.B,
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
        multiprocessing_context="forkserver",
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=config.B,
        sampler=dev_sampler,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
        multiprocessing_context="forkserver",
    )

    return accelerator.prepare(train_loader, dev_loader)
