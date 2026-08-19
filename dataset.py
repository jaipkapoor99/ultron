"""Memory-mapped sharded token dataset.

Shards remain on disk and each sample allocates only the requested token
window while converting uint16 token IDs to the int64 dtype PyTorch
embeddings require.
"""

import glob
import os
from bisect import bisect_right
from collections.abc import Generator, Sized

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler, SequentialSampler, Subset

from config import UltronConfig


class EpochRandomSampler(Sampler[int]):
    """Deterministically shuffle indices using a distinct permutation per epoch."""

    def __init__(self, data_source: Sized, seed: int) -> None:
        self.data_source = data_source
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch cannot be negative")
        self.epoch = epoch

    def __iter__(self) -> Generator[int]:
        indices = np.arange(len(self.data_source), dtype=np.int32)
        np.random.default_rng(self.seed + self.epoch).shuffle(indices)
        return (int(index) for index in indices)

    def __len__(self) -> int:
        return len(self.data_source)


class ZeroCopyShardedDataset(Dataset):
    """Memory-mapped dataset with process-local, lazily opened shard views."""

    def __init__(self, bin_shards, sequence_length=1024, step=None) -> None:
        self.bin_shards = list(bin_shards)
        self.T = sequence_length
        self.step = sequence_length if step is None else step
        if self.T <= 0 or self.step <= 0:
            raise ValueError("sequence_length and step must be greater than zero")

        # NumPy memmaps must never be included in the pickled dataset state.
        # Python 3.14 uses forkserver by default, so DataLoader workers receive
        # this object through pickle. Serializing open memmaps can copy the
        # complete corpus into RAM.
        self._shard_memmaps = {}
        self._memmap_owner_pid = None
        self.shard_offsets = []
        # pyrefly: ignore [unknown-name]
        self.shard_ends: list[int] = []
        total_sequences = 0

        for shard_path in self.bin_shards:
            num_tokens = os.path.getsize(shard_path) // 2  # uint16 = 2 bytes
            num_seqs = max(0, (num_tokens - (self.T + 1)) // self.step + 1)
            self.shard_offsets.append((total_sequences, total_sequences + num_seqs))
            total_sequences += num_seqs
            self.shard_ends.append(total_sequences)

        self.total_sequences = total_sequences

    def __getstate__(self):
        """Strip process-local mappings before spawn/forkserver serialization."""
        state = self.__dict__.copy()
        state["_shard_memmaps"] = {}
        state["_memmap_owner_pid"] = None
        return state

    def _get_shard_memmap(self, shard_idx: int) -> np.memmap:
        """Open a shard lazily and cache it only in the current process."""
        current_pid = os.getpid()
        if self._memmap_owner_pid != current_pid:
            self._shard_memmaps = {}
            self._memmap_owner_pid = current_pid

        mmap = self._shard_memmaps.get(shard_idx)
        if mmap is None:
            mmap = np.memmap(
                self.bin_shards[shard_idx],
                dtype=np.uint16,
                mode="r",
            )
            self._shard_memmaps[shard_idx] = mmap
        return mmap

    def __len__(self):
        return self.total_sequences

    def __getitem__(self, index: int):
        idx = index
        if idx < 0:
            idx += self.total_sequences
        if idx < 0 or idx >= self.total_sequences:
            raise IndexError(
                f"Index {idx} out of range for dataset size {self.total_sequences}"
            )

        shard_idx = bisect_right(self.shard_ends, idx)
        start_seq, _ = self.shard_offsets[shard_idx]
        seq_idx_in_shard = idx - start_seq
        token_start = seq_idx_in_shard * self.step

        # Convert the requested uint16 disk slice to int64 for embeddings.
        mmap = self._get_shard_memmap(shard_idx)
        chunk = mmap[token_start : token_start + self.T + 1].astype(np.int64)

        x = torch.from_numpy(chunk[: self.T])
        y = torch.from_numpy(chunk[1 : self.T + 1])
        return x, y


def split_train_dev_datasets(bin_shards, sequence_length, step=None):
    """Create leakage-safe train/dev datasets.

    Multiple-shard corpora are split at a shard boundary. A single-shard
    corpus uses contiguous windows with a gap large enough to prevent the
    train and validation windows from sharing tokens.
    """
    step = sequence_length if step is None else step
    if len(bin_shards) >= 2:
        train_shard_count = min(
            len(bin_shards) - 1,
            max(1, round(0.95 * len(bin_shards))),
        )
        train_ds = ZeroCopyShardedDataset(
            bin_shards[:train_shard_count],
            sequence_length=sequence_length,
            step=step,
        )
        dev_ds = ZeroCopyShardedDataset(
            bin_shards[train_shard_count:],
            sequence_length=sequence_length,
            step=step,
        )
        return train_ds, dev_ds

    full_ds = ZeroCopyShardedDataset(
        bin_shards,
        sequence_length=sequence_length,
        step=step,
    )
    split_index = int(0.95 * len(full_ds))
    overlap_gap = (sequence_length + step - 1) // step
    dev_start = min(len(full_ds), split_index + overlap_gap)
    return (
        Subset(full_ds, range(split_index)),
        Subset(full_ds, range(dev_start, len(full_ds))),
    )


def get_dataloaders(config: UltronConfig, accelerator):
    bin_shards = sorted(
        glob.glob("shards/fineweb_shard_*.bin")
        + glob.glob("shards_edu/fineweb_edu_shard_*.bin")
    )

    if not bin_shards:
        if os.path.exists("fineweb_tokens.bin"):
            bin_shards = ["fineweb_tokens.bin"]
        else:
            raise FileNotFoundError(
                "No binary dataset shards found! Run 'python tokenize_dataset.py' first."
            )

    accelerator.print(
        f"Loading {len(bin_shards)} binary shard(s) via memory mapping..."
    )

    train_ds, dev_ds = split_train_dev_datasets(
        bin_shards,
        sequence_length=config.T,
        step=config.T,
    )
    if len(train_ds) == 0 or len(dev_ds) == 0:
        raise ValueError(
            "Dataset is too small for a leakage-safe train/dev split. "
            "Provide at least two shards or a larger token file."
        )
    accelerator.print(
        f"Sequences Available: {len(train_ds):,} train / {len(dev_ds):,} dev"
    )

    train_sampler = EpochRandomSampler(train_ds, seed=config.data_seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=config.B,
        sampler=train_sampler,
        num_workers=2,
        pin_memory=False,
        drop_last=True,
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=config.B,
        sampler=SequentialSampler(dev_ds),
        num_workers=2,
        pin_memory=False,
    )

    train_loader, dev_loader = accelerator.prepare(train_loader, dev_loader)

    return train_loader, dev_loader
