# Ultron-113M: Modern Transformer Pre-training Pipeline

[![CI](https://github.com/jaipkapoor99/ultron/actions/workflows/ci.yml/badge.svg)](https://github.com/jaipkapoor99/ultron/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.14-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.13-EE4C2C?logo=pytorch&logoColor=white)

A high-performance PyTorch implementation of **Ultron-113M** pre-trained from scratch on **10.0 billion tokens** of the **FineWeb-Edu** dataset.

🤖🤖🤖
*Originally designed as a humble GPT-2 clone, Ultron grew into a modernized decoder-only SOTA powerhouse — as Ultron would say, "There are no strings on me."*
🤖🤖🤖

> [!IMPORTANT]
> **🚀 Pre-training Base Checkpoint Status Notice:**
> The **100% pre-trained base model checkpoint** (`10.0 billion tokens`) is published on [Hugging Face](https://huggingface.co/jaipkapoor99/ultron-113m). It represents the raw foundational model before instruction tuning. Checkpoints and dataset shards are intentionally excluded from Git.

---

## ⚡ Quick Architecture Summary

```text
Ultron-113M Layout:
├── Parameters        : 113,266,944 (113M, with tied embeddings)
├── Layers            : 12 Transformer blocks
├── Embedding (C)     : 768 hidden dimension
├── Attention Heads   : 12 Query heads, 4 Key/Value heads (GQA 3:1 ratio)
├── Head Dimension    : 64
├── Context Window    : 1,024 tokens (RoPE frequency base 10,000)
├── FFN Activation    : SwiGLU (Tensor Core aligned to multiples of 64)
├── Normalization     : RMSNorm (with QK-head normalization, eps=1e-5)
├── Logit Regularizer : Soft-Capping (cap=15.0 via tanh)
├── Linear Projections: 100% Bias-Free (bias=False across all layers)
├── Optimizer         : torch.optim.Muon for 2D body, fused AdamW for 1D/embeddings
└── Dataset & Tokens  : FineWeb-Edu (10.0B tokens across 152,587 steps)
```

---

## 🏗️ Architectural Flow & Block Diagram

```text
                        Input Token IDs
                               │
                               ▼
                   Token Embedding (SmolLM Vocab: 49,152)
                               │
                               ▼
             ┌───────────────────────────────────┐
             │   12 × Decoder Layer Stack        │
             │                                   │
             │   ┌───────────────────────────┐   │
             │   │ RMSNorm                   │   │
             │   └─────────────┬─────────────┘   │
             │                 │                 │
             │                 ▼                 │
             │   ┌───────────────────────────┐   │
             │   │ GQA (12 Q / 4 KV) + RoPE  │   │
             │   │  └─ QK-Head RMSNorm       │   │
             │   └─────────────┬─────────────┘   │
             │                 │                 │
             │                 ▼                 │
             │            Residual ───(+)        │
             │                 │                 │
             │                 ▼                 │
             │   ┌───────────────────────────┐   │
             │   │ RMSNorm                   │   │
             │   └─────────────┬─────────────┘   │
             │                 │                 │
             │                 ▼                 │
             │   ┌───────────────────────────┐   │
             │   │ SwiGLU FFN                │   │
             │   └─────────────┬─────────────┘   │
             │                 │                 │
             │                 ▼                 │
             │            Residual ───(+)        │
             └─────────────────┬─────────────────┘
                               │
                               ▼
                        Final RMSNorm
                               │
                               ▼
                     LM Head Linear Projection
                               │
                               ▼
                  Logit Soft-Capping (cap=15.0)
                               │
                               ▼
                         Output Logits
```

---

## ⚔️ Architectural Evolution: GPT-2 vs. Ultron-113M

| Feature | GPT-2 (124M) | Ultron-113M | Why it Matters (Engineering Justification) |
| :--- | :---: | :---: | :--- |
| **Positional Encoding** | Absolute Learned (`wpe`) | **RoPE (Rotary)** | Enables zero-shot context length extension and better relative distance modeling. |
| **Attention Mechanism** | Multi-Head (MHA) | **Grouped-Query (GQA)** | 12 Q heads : 4 KV heads (**3:1 ratio**), reducing KV-cache memory usage during inference by **3×**. |
| **Attention Stability** | Standard Unnormalized | **QK-Head RMSNorm** | Prevents logit explosion / attention entropy collapse during long pre-training runs. |
| **FFN Activation** | Standard GELU | **SwiGLU** | Gated non-linearity yielding higher model capacity per FLOP; aligned to multiples of 64 for Tensor Core throughput. |
| **Layer Normalization** | LayerNorm (with bias) | **RMSNorm (Bias-Free)** | Eliminates mean-centering overhead; 100% bias-free projections (`bias=False`) for cleaner gradient dynamics. |
| **Logit Regularization** | None | **Logit Soft-Capping** | Applies `tanh` capping (`cap=15.0`) to prevent overconfidence and extreme logit growth. |
| **Optimizer Engine** | AdamW | **PyTorch Muon + Fused AdamW** | Uses built-in `torch.optim.Muon` for 2D body weights and AdamW for embeddings and normalization parameters. |
| **Learning Rate Schedule** | Cosine Decay | **WSD Schedule** | Warmup-Stable-Decay schedule with an 80% stable phase followed by linear decay. |
| **Mixed Precision** | FP32 | **Native BFloat16 (`bf16`)** | Dynamic range stability without loss scalers on RTX 30xx/40xx/50xx GPUs. |
| **Graph Compiler** | None | **PyTorch 2.0 (`torch.compile`)** | Fuses element-wise operations and kernel launches via Inductor. |

---

## 🌟 Key Features & Engineering Design

- **Rotary Position Embeddings (RoPE):** Applied directly to $Q$ and $K$ heads (frequency base $\theta = 10,000$), preserving relative token distances.
- **QK-Head RMSNorm:** Normalizes Query and Key head vectors before dot-product attention to stabilize scale across deep layers.
- **Grouped-Query Attention (GQA):** Uses 12 Query heads paired with 4 Key/Value heads, reducing memory bandwidth pressure during generation.
- **SwiGLU FFN:** SwiGLU Gated Linear Units with hidden dimensions rounded up to multiples of 64 for optimal GPU Tensor Core utilization.
- **Logit Soft-Capping:** `15.0 * tanh(logits / 15.0)` applied prior to loss calculation to prevent logit explosion.
- **PyTorch Muon Optimizer:** Built-in `torch.optim.Muon` handles 2D matrix weights, combined with fused `AdamW` for 1D vectors and embeddings.
- **Rust-Engine Batch Tokenizer:** Sub-process tokenization via Rust `backend_tokenizer.encode_batch` streaming at **~4.34 Million tokens/sec** into compact `uint16` binary shards.
- **Worker-Safe Memory-Mapped Pipeline:** `np.memmap` keeps the 10B-token corpus on disk and copies only each requested window to the `int64` dtype required by PyTorch embeddings. Each DataLoader process opens shard mappings lazily, so Python 3.14 `forkserver` workers never deserialize the corpus into RAM.
- **Deterministic Shuffling:** Training uses seeded, epoch-specific permutations over non-overlapping windows. Validation remains sequential, and checkpoint resume reconstructs the correct permutation before fast-forwarding.

---

## 📊 Pre-training Architecture & Hyperparameters

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Model Name / Tag** | **Ultron-113M** | 113,266,944 trainable parameters with tied token-embedding and LM-head weights |
| **Layers / Query Heads / KV Heads** | 12 layers / 12 Q-heads / 4 KV-heads | GQA Transformer layout ($C=768, n_{head}=12, n_{kv\_head}=4$) |
| **Context Window ($T$)** | 1,024 tokens | Sequence length per pass |
| **Micro-Batch Size ($B$)** | 16 | Per-GPU micro-batch size |
| **Gradient Accumulation** | 4 steps | Effective batch size = 64 sequences (65,536 tokens/step) |
| **Sample Order** | Deterministic epoch shuffle | Seeded permutations diversify batches; stride equals context length, so windows do not overlap |
| **Tokenizer** | SmolLM Vocab (49,152) | Efficient BPE tokenizer (`HuggingFaceTB/SmolLM2-135M`) |
| **Precision** | BFloat16 (`bf16`) | Native mixed precision |
| **LR Schedule** | WSD | Warmup-Stable-Linear-Decay (80% stable, 20% linear decay) |
| **Optimizer** | `torch.optim.Muon` + fused AdamW | Newton-Schulz matrix optimizer ($LR=0.04$) + fused AdamW ($LR=1.2\times 10^{-3}$) |
| **Throughput** | 181,543 tok/sec effective; 189,475 tok/sec final rolling | Completed-run W&B telemetry on one NVIDIA RTX 5090 |
| **GPU VRAM Allocation** | ~17.0 GiB average / 32 GiB | W&B system telemetry for the completed run; checkpoint saves briefly peaked at 26.1 GiB |
| **Total Pre-training Time** | **15 Hours 18 Minutes 3 Seconds (55,083s)** | Cumulative W&B runtime for 9,999,941,632 tokens / 152,587 steps |

---

## 📂 Repository Structure

```text
ultron/
├── .github/workflows/ci.yml # CPU dependency, compilation, and pytest CI
├── .python-version         # uv-managed Python 3.14.6 pin
├── AGENTS.md               # Contributor and repository guidelines
├── JOURNEY.md              # Engineering lessons and major corrections
├── model.py                # PyTorch Ultron-113M (RoPE + GQA + SwiGLU + RMSNorm + QKNorm + Logit SoftCap)
├── config.py               # Model & Hyperparameter Configuration Dataclass
├── dataset.py              # Memory-mapped sharded dataset loader
├── train.py                # Main Accelerated Distributed Training Runner
├── trainer.py              # Training loop with PyTorch Muon + fused AdamW
├── telemetry.py            # Rolling throughput, ETA, W&B metrics, and progress UI
├── pyproject.toml          # Canonical dependencies and Ruff configuration
├── assets/                 # Tracked documentation figures, including average train loss
├── accelerate_checkpoint/  # Saved Accelerate model weights, optimizer state & RNG seeds
├── shards_edu/             # Binary FineWeb-Edu tokenized data shards (.bin)
├── logs/                   # Generated validation and benchmark JSON results
├── wandb/                  # Local step telemetry logs & experiment tracking runs
├── .agents/                # Agent-specific engineering principles
├── tests/                  # CPU-safe model, dataset, and training tests
│   ├── test_dataset.py     # Shard lookup and leakage-safe split tests
│   ├── test_model.py       # Causality, cache, optimizer, and learning tests
│   ├── test_telemetry.py   # Rate, ETA, and metric-schema tests
│   ├── test_tokenize_dataset.py # Exact-resume and atomic-write tests
│   ├── test_training.py    # Evaluation, resume, and checkpoint-safety tests
│   ├── test_eval_lm_harness.py # lm-evaluation entry-point tests
│   ├── test_upload_checkpoint.py # Complete training-state upload tests
│   ├── test_upload_dataset_shards.py # Upload validation tests
│   └── test_validate.py    # Full-validation metric tests
└── scripts/                # Helper Scripts
    ├── generate.py         # Text generation from local Accelerate checkpoint
    ├── tokenize_dataset.py # Exact-resume FineWeb-Edu sharding
    ├── validate.py         # Complete leakage-safe validation pass
    ├── eval_lm_harness.py  # EleutherAI lm-evaluation-harness benchmark script
    ├── upload_checkpoint.py# Hugging Face Hub model checkpoint uploader script
    └── upload_dataset_shards.py # Hugging Face Hub dataset shards uploader script
```

---

## 🤗 Hugging Face Repositories

- **Model Checkpoint**: [`jaipkapoor99/ultron-113m`](https://huggingface.co/jaipkapoor99/ultron-113m)
- **Pre-tokenized Dataset Shards**: [`jaipkapoor99/ultron-fineweb-edu-shards`](https://huggingface.co/datasets/jaipkapoor99/ultron-fineweb-edu-shards)

The model repository is a complete training-resume artifact, not an
inference-only export. Upload the entire Accelerate checkpoint—including model,
Muon and AdamW optimizer state, scheduler state, scaler state when present, and
RNG state—with:

```bash
HF_TOKEN=hf_... python scripts/upload_checkpoint.py
```

Because Accelerate RNG state can use pickle serialization, download and resume
only from a repository you trust.

---

## 🚀 Quickstart & Workflow Guide

### 1. Installation

```bash
git clone https://github.com/jaipkapoor99/ultron.git
cd ultron

# Fast environment setup using uv
uv python install 3.14.6
uv venv --python 3.14.6 .venv
source .venv/bin/activate

# Install the PyTorch 2.13 wheel for your CUDA/CPU platform first
uv pip install torch==2.13.0
uv pip install -r pyproject.toml --group dev
uv pip check

# Lint the project
uv run --no-sync ruff check .

# Optional: install if torch.compile cannot locate a CUDA compiler
uv pip install nvidia-cuda-nvcc
```

`pyproject.toml` is the single source of truth for runtime and development
dependencies. Lock and exported requirements files are generated artifacts and
are intentionally ignored; CI resolves a temporary CPU-safe requirements file
directly from the project metadata.

### 2. Tokenize Dataset

Tokenize the FineWeb-Edu dataset into compact binary shards:

```bash
python scripts/tokenize_dataset.py
```

Shards are committed atomically. The tokenizer pins the dataset and tokenizer
revisions and records the streaming dataset's native file/row cursor plus the
pending token buffer in `shards_edu/tokenization_state.json`. Resuming from a
native cursor jumps directly to the next document instead of replaying the
stream. Existing shards without a complete native-cursor state file are
rejected.

Check whether tokenization is already running:

```bash
pgrep -af '[t]okenize_dataset.py'
```

For a tokenizer job that survives terminal closure on a systemd-based Linux
machine:

```bash
systemd-run --user --unit=ultron-tokenizer --collect \
  --property=WorkingDirectory="$PWD" \
  "$PWD/.venv/bin/python" scripts/tokenize_dataset.py
systemctl --user status ultron-tokenizer
journalctl --user -fu ultron-tokenizer
```

Monitor durable progress:

```bash
watch -n 5 'grep -E "\"next_shard\"|\"committed_tokens\"|\"source_documents_consumed\"" shards_edu/tokenization_state.json'
```

Request a safe stop; uncommitted work is replayed on resume:

```bash
pkill -INT -f '[t]okenize_dataset.py'
```

After all 100 shards are committed, validate and resumably upload them:

```bash
HF_TOKEN=hf_... python scripts/upload_dataset_shards.py
```

The uploader refuses incomplete or inconsistent shard sets and uploads only
the `.bin` shards and their public metadata; private resume buffers remain
local.

### 3. Configure Accelerate

Run this **once** to generate the config for your machine:

```bash
accelerate config
```

Recommended settings for this project:

| Setting | Value | Why |
| :--- | :--- | :--- |
| Compute environment | Local machine | Single-node training |
| Distributed type | `NO` | Single GPU |
| Mixed precision | `bf16` | Required for peak throughput on RTX 30xx/40xx/50xx |
| TorchDynamo backend | `INDUCTOR` | Enables `torch.compile` graph compilation |

### 4. Pre-training Execution

Launch pre-training:

```bash
accelerate launch train.py --mode=fresh
```

Fresh W&B runs are named with a local timestamp prefix, for example
`20260806-193356-fresh`. If `ULTRON_RUN_NAME` is set, its value follows the
timestamp, such as `20260806-193356-baseline`. Continued runs retain their
existing W&B identity and are not renamed.

Training uses deterministic epoch-specific shuffling with `data_seed`, while
validation remains sequential. Samples use a stride equal to the context
length, so adjacent dataset windows do not overlap. Checkpoint resume derives
the data epoch and batch offset from the recorded optimizer step, reconstructs
the same permutation, and fast-forwards without storing the permutation.
Shard paths and compact sequence metadata are safe to serialize during worker
startup; open memmaps remain process-local and are created only when a worker
first reads a shard.

---

## ✅ Tests & Continuous Integration

Run the CPU-safe test suite locally:

```bash
pytest -q
```

The current suite collects 133 CPU-safe tests. It covers model
causality and caching, optimizer partitioning, non-overlapping dataset windows,
deterministic shuffle epochs, exact checkpoint positioning, rotating validation,
telemetry summaries, tokenization corruption, shard validation, evaluation
harness behavior, and upload guards. The optional compiler test is skipped
unless explicitly enabled.

Run the slower compiler smoke test explicitly:

```bash
ULTRON_TEST_COMPILE=1 pytest -q tests/test_model.py -k torch_compile
```

Run the end-to-end CUDA smoke test:

```bash
accelerate launch train.py --mode=test
```

Test mode compiles the model, trains for 100 optimizer steps, and runs sampled
validation against the prepared shards. It does not initialize W&B or write a
persistent training checkpoint. The Python 3.14.6 environment has been verified
with PyTorch 2.13 and CUDA 13.0 on an NVIDIA RTX 5090.

Training-time validation deliberately samples `eval_batches=20` dev batches
for inexpensive monitoring. The sampled window advances after every evaluation,
wraps at the end of the dev loader, and is restored from checkpoints instead of
repeating the first batches. Run the complete leakage-safe validation partition
separately:

```bash
accelerate launch scripts/validate.py
```

Full validation creates a fresh timestamped W&B run in the
`ultron-pretraining` project with the `full-validation` job type. It records
running full-dev loss, rolling token throughput, progress, final perplexity,
elapsed time, processed counts, and normal W&B CPU/RAM/GPU system telemetry.
Terminal progress still shows throughput and ETA. The complete result is also
written atomically to `logs/full_validation.json`. Use `--wandb-project` or
`--wandb-run-name` to override the tracking destination or display name.

The completed full-dev pass is recorded in
[W&B run `jg640nwo`](https://wandb.ai/jaipkapoor99-rumani-dhaage/ultron-pretraining/runs/jg640nwo):

| Full-validation metric | Recorded value |
| :--- | :--- |
| Loss / perplexity | **2.964989 / 19.3945** |
| Evaluated data | **488,280 sequences / 499,998,720 tokens** |
| Runtime | **17m 20.7s** |
| Token throughput | **480,436 tok/s average / 517,229 tok/s peak rolling** |
| GPU utilization | **97.6% average** |
| VRAM allocation | **9.4 GiB average / 12.5 GiB peak** |
| Host CPU utilization | **12.1% average** |
| System RAM utilization | **22.1% average / 36.8% peak** |
| GPU power / temperature | **570.7 W / 71.7°C average** |

### CI workflow

The workflow in [`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs for
pushes and pull requests targeting `master`, and can also be started manually
with `workflow_dispatch`. Only the newest run for a workflow and Git reference
continues; an older in-progress run is cancelled. The job has read-only
repository permissions and a 20-minute timeout.

| Stage | CI behavior |
| :--- | :--- |
| Environment | Ubuntu runner with uv-managed Python 3.14.6 |
| Dependency cache | Keyed from `pyproject.toml` |
| PyTorch | CPU-only PyTorch 2.13 from the official PyTorch wheel index |
| Dependencies | Generates a CPU-safe requirements file from `pyproject.toml` |
| Dependency validation | Runs `uv pip check` to reject incompatible packages |
| Static analysis | Runs Ruff bug, import, modernization, and simplification checks |
| Syntax validation | Byte-compiles core modules, scripts, and tests |
| Unit tests | Runs pytest without writing bytecode or `.pytest_cache` |

Reproduce the CI checks locally from an activated environment:

```bash
uv pip check
uv run --no-sync ruff check .
python -m compileall -q \
  config.py dataset.py model.py telemetry.py train.py trainer.py scripts tests
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider
```

CI deliberately remains CPU-safe: it does not download dataset shards, run
full validation, initialize W&B, execute the opt-in compiler test, or launch
CUDA training. Run those checks locally with the commands above this section.

---

## 📊 Telemetry & Pre-training Evaluation

### 📈 Pre-training Telemetry Summary

| Metric | Recorded Value | Description |
| :--- | :--- | :--- |
| **Total Steps Completed** | **152,587 / 152,587 (100%)** | Full pre-training run on FineWeb-Edu |
| **Total Tokens Processed** | **~10.0 Billion Tokens** | 65,536 tokens per step (batch size 64 $\times$ seq len 1,024) |
| **Step Throughput** | **2.89 iterations/sec** | Final 30-second rolling measurement |
| **Token Throughput** | **189,475 tokens/sec rolling; 181,543 tokens/sec effective** | Final rolling rate and end-to-end tokens divided by cumulative tracked runtime |
| **Compute Hardware** | **NVIDIA RTX 5090 (32GB)** | Native BFloat16 (`bf16`) mixed precision |
| **Total Runtime** | **15 Hours 18 Minutes 3 Seconds (55,083s)** | Cumulative W&B runtime across resumed processes, including validation and checkpoint overhead |
| **VRAM Footprint** | **17.0 GiB average; 26.1 GiB peak** | Peak is the transient checkpoint/save allocation, not steady training |
| **System RAM After Worker Fix** | **16.9% average; 20.6% peak (~10.1/12.3 GiB of 59.4 GiB)** | File-backed memmaps remained reclaimable; the former >50 GiB forkserver duplication was removed |
| **Host CPU Utilization** | **12.3% average; 12.4% sustained peak** | Whole-machine utilization across 16 logical CPUs; GPU utilization averaged 97.7% |
| **Final sampled validation estimate (`dev_loss`)** | **`2.9687`** | Estimated from 20 rotating validation batches at step 152,587; not a full validation pass |
| **Final interval-average train loss** | **`2.9446`** | Average since the preceding sampled evaluation |

---

### 🧪 Official EleutherAI `lm-evaluation-harness` Baseline Benchmark Report

Evaluated across **all un-truncated test/validation splits** (17,195 total
log-likelihood examples) using `scripts/eval_lm_harness.py`. Results are stored
in `logs/pre_training_checkpoint_eval.json`:

```bash
accelerate launch scripts/eval_lm_harness.py --limit=0
```

| Benchmark task | Examples | Raw accuracy | Length-normalized accuracy | Random baseline |
| :--- | ---: | ---: | ---: | ---: |
| **`piqa`** | 1,838 | **64.25%** | **63.66%** | 50% |
| **`arc_easy`** | 2,376 | **53.62%** | **47.05%** | 25% |
| **`hellaswag`** | 10,042 | **30.24%** | **33.75%** | 25% |
| **`arc_challenge`** | 1,172 | 23.55% | **26.54%** | 25% |
| **`openbookqa`** | 500 | 21.60% | **32.20%** | 25% |
| **`winogrande`** | 1,267 | 49.17% | — | 50% |

Raw macro-average accuracy is **40.41%**. PIQA, ARC Easy, and HellaSwag are
above their random baselines; WinoGrande and ARC Challenge remain approximately
at chance. OpenBookQA improves materially after answer-length normalization.
Together with the full-dev loss of **2.964989**, the suite indicates a
converged 113M base model whose remaining limitations are concentrated in hard
science and pronoun reasoning.

---

### 📉 Average Train Loss

![Ultron interval-average train loss](assets/average_train_loss.svg)

> [!NOTE]
> The curve contains all 611 interval averages recorded every 250 optimizer
> steps, plus the final partial interval at step 152,587. It plots the current
> non-overlapping, deterministically shuffled 10B-token run—not the legacy
> overlapping-window experiment.

---

## 📊 Weights & Biases (W&B) Experiment Tracking Architecture

Pre-training metrics are logged live via **Weights & Biases** under the `ultron-pretraining` project.

- **Rolling Performance Estimates**: `telemetry.py` uses monotonic 30-second rolling windows for responsive throughput and ETA estimates instead of increasingly stale session averages. Training throughput counts tokens across every distributed worker.
- **Throttled Terminal UI**: Training and tokenization progress render at most twice per second, preventing high-frequency updates from flooding captured logs.
- **Out-of-Order Resumption Resolved**: Solved early telemetry log fragmentation by standardizing `resume="allow"` in `setup_accelerator_trackers()`. W&B runs now resume seamlessly across checkpoint restarts without step monotonicity conflicts.
- **Metric Grouping & Summaries**: Canonical loss, learning-rate, and throughput metrics use W&B's native step axis without an unwanted step chart. The run summary is populated explicitly with steps, processed and planned tokens, progress, current and best losses, learning rate, throughput, and validation count.
- **Loss Monitoring**: Throughput and held sampled dev loss remain continuous, while interval-average train loss and sampled dev loss share a comparison chart. The raw sampled estimate remains separately available without dashboard clutter.
- **Resume Diagnostics**: Malformed W&B checkpoint metadata and tracker-resolution failures emit explicit warnings instead of being silently discarded.

> [!TIP]
> **Engineering Takeaway — Master W&B & Telemetry Pipeline:**
> *"There is no data science without data."* Resolving metric step alignment and offline binary `.wandb` log parsing reinforced the importance of mastering telemetry pipelines, structured metric registration (`define_metric`), and experiment tracking early in large-scale pre-training projects.

---

## 🧪 Pre-training Sample Generations

The following unedited samples were generated together from the completed
step-152,587 checkpoint. The tokenizer, model, and checkpoint are loaded only
once:

```bash
source .venv/bin/activate && accelerate launch scripts/generate.py \
  --prompt "Artificial intelligence systems learn by" \
  --prompt "To learn mathematics effectively, one must" \
  --prompt "As the quantum portal activated, the crew stared into the alien"
```

The defaults reproduce the documented sampling policy: 70 new tokens, seed
1337, temperature 0.7, top-k 20, top-p 0.95, min-p disabled, repetition penalty
1.1, no repeated 3-grams, one sample, and EOS stopping. The repetition penalty
discourages overused tokens while the n-gram constraint stops exact phrase
loops. CLI options remain available for deliberate overrides. Use `--samples
N` to request multiple continuations for every prompt, `--ignore-eos` to force
the complete token budget, or `--greedy` for a maximum-probability baseline.
Set `--repetition-penalty 1 --no-repeat-ngram-size 0` to recover unconstrained
sampling.

### 🤖 Artificial Intelligence

> **Prompt:** *"Artificial intelligence systems learn by"*
>
> **Output:** Artificial intelligence systems learn by watching the user and
> then apply the learning. If a system is able to learn from the experience, it
> will be able to apply the knowledge. This can be achieved through machine
> learning in a number of different ways including:
>
> - Predictive learning
> - Real-time learning
> Machine learning is a method of learning where data are analyzed to make
> predictions

### 📐 Mathematics

> **Prompt:** *"To learn mathematics effectively, one must"*
>
> **Output:** To learn mathematics effectively, one must understand and analyze
> the content of mathematics.
> Mathematics is a scientific discipline that involves the study of
> mathematical symbols (called symbols) as well as their relationships. In the
> modern world, mathematics is used in many areas of life: for example, the
> development of computer systems, medicine, engineering, the study and
> interpretation of data, and the analysis of

### 🚀 Science Fiction

> **Prompt:** *"As the quantum portal activated, the crew stared into the alien"*
>
> **Output:** As the quantum portal activated, the crew stared into the alien
> world of space and wondered what would happen to them if they were exposed to
> radiation.
> “The only thing we could do was take out our eyes, and we would look at the
> light and see a cloud of radiation,” says Kyle. “We had no idea that this
> would be possible.”
> For the crew, however, it meant they

A further three-sample stress test across all three domains produced nine
grammatical, on-topic continuations without exact phrase loops. The constraint
guarantees that a completed 3-gram cannot recur; the softer repetition penalty
also reduces broader lexical recycling without forcing unnatural substitutions.

---

## 🎓 Engineering Journey

The mistakes, corrections, and systems lessons behind Ultron are documented in
[JOURNEY.md](JOURNEY.md).

---

## 📜 Acknowledgments & Citation

- Andrej Karpathy for the inspiring [*Neural Networks: Zero to Hero*](https://github.com/karpathy/build-nanogpt) course and `nanoGPT` project.
- Keller Jordan et al. for pioneering the [Muon](https://github.com/KellerJordan/Muon) optimizer.

```bibtex
@misc{jordan2024muon,
  author = {Jordan, Keller and Jin, Yuchen and Boza, Vlado and You, Jiacheng and Cesista, Franz and Newhouse, Laker and Bernstein, Jeremy},
  title  = {Muon: An optimizer for hidden layers in neural networks},
  year   = {2024},
  url    = {https://kellerjordan.github.io/posts/muon/}
}
```
