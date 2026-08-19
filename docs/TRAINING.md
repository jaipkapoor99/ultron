# Ultron-113M Training & Data Pipeline

This document explains the data tokenization, memory-mapping, distributed training loop, and checkpoint resume contracts for **Ultron-113M**.

______________________________________________________________________

## 1. Pre-tokenization & Sharding (`scripts/tokenize_dataset.py`)

- **Dataset Source**: `HuggingFaceFW/fineweb-edu` (`sample-10BT` subset).
- **Streaming Pipeline**: Streams source documents and tokenizes in batches using Rust-backed `backend_tokenizer.encode_batch` from `HuggingFaceTB/SmolLM2-135M` (~4.34M tokens/s).
- **Atomic Sharding**: Tokens are serialized into 100M-token binary shards (`uint16` dtype) accompanied by atomic JSON metadata (`fineweb_edu_shard_XXXX_meta.json`).
- **Exact Resume State**: The pipeline writes `tokenization_state.json` recording dataset/tokenizer revisions, exact document cursors, and uncommitted token buffers (`.pending_tokens_XXXX.npy`).

### Running Tokenization

```bash
# Tokenize 100 binary shards (10B tokens total)
python scripts/tokenize_dataset.py --shard-size-tokens 100000000 --max-shards 100
```

______________________________________________________________________

## 2. Memory-Mapped Dataset & Sampling Geometry (`dataset.py`)

- **Zero-Copy Memory Mapping**: Shards remain on disk and are opened lazily per worker process via `np.memmap(..., mode='r')`.
- **Forkserver Safety**: Under Python 3.14 `forkserver`, open file descriptors and memmaps are stripped during pickle serialization (`__getstate__`) and re-instantiated lazily inside DataLoader workers.
- **Non-Overlapping Windows**: Sequences are sampled with a stride equal to context length ($T = 1024$), guaranteeing 0 token overlap between adjacent samples.
- **Leakage-Safe Train/Dev Split**: Multiple-shard datasets are partitioned strictly at shard boundaries (e.g., 19 shards for training, 1 shard for dev evaluation).
- **Epoch Shuffling**: `EpochRandomSampler` applies seeded, deterministic permutations per epoch (`data_seed + epoch`), ensuring exact reproducibility upon checkpoint resumption.

______________________________________________________________________

## 3. Pre-training Loop & Optimization (`trainer.py`, `train.py`)

### Training Hyperparameters

| Hyperparameter               | Value             | Description                                                              |
| :--------------------------- | :---------------- | :----------------------------------------------------------------------- |
| **Global Batch Size**        | 65,536 tokens     | Micro-batch 16 $\\times$ 4 gradient accumulation $\\times$ 1,024 context |
| **Total Optimization Steps** | 152,587           | Reaching ~10.0 Billion tokens                                            |
| **Precision**                | BFloat16 (`bf16`) | Native mixed precision via PyTorch AMP / Accelerate                      |
| **Muon Learning Rate**       | `0.04`            | For 2D weight matrices                                                   |
| **AdamW Learning Rate**      | `1.2e-3`          | For 1D vectors and embedding tables                                      |
| **Schedule**                 | WSD               | Warmup (200 steps) $\\to$ Stable (80%) $\\to$ Linear Decay (20%)         |
| **Graph Compilation**        | `torch.compile`   | Inductor backend with `high` float32 matmul precision                    |

### Launch Commands

```bash
# Fresh training run
accelerate launch train.py --mode=fresh

# Resume from existing checkpoint
accelerate launch train.py --mode=continue

# Fast smoke test run (100 steps)
accelerate launch train.py --mode=test
```

______________________________________________________________________

## 4. Exact Resume Contract

Checkpoint recovery restores the complete multi-component training state:

Checkpoint recovery restores the complete multi-component training state:

1. Model weights ([`accelerate_checkpoint/model.safetensors`](../accelerate_checkpoint/))
1. Optimizer states for both Muon and AdamW
1. RNG seeds for PyTorch, NumPy, and Python
1. Sampler shuffle epoch and batch offset derived from `step`
1. Rotating validation cursor (`dev_batch_cursor`)
1. W&B experiment tracking identity (`resume="allow"`)
