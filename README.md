# Ultron-113M: Modern Transformer Pre-training Pipeline

[![CI](https://github.com/jaipkapoor99/ultron/actions/workflows/ci.yml/badge.svg)](https://github.com/jaipkapoor99/ultron/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.14-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.13-EE4C2C?logo=pytorch&logoColor=white)
![Ruff](https://img.shields.io/badge/Linter-Ruff-black?logo=ruff&logoColor=white)
![Pyrefly](https://img.shields.io/badge/Type_Check-Pyrefly-blue)

A high-performance PyTorch implementation of **Ultron-113M**, pre-trained from scratch on **10.0 billion tokens** of the FineWeb-Edu corpus on a single NVIDIA RTX 5090.

🤖 _"There are no strings on me."_ 🤖

> [!IMPORTANT]
> **🚀 Open-Source Model Artifacts Available:**
>
> - **Pre-trained Base Checkpoint**: [`jaipkapoor99/ultron-113m`](https://huggingface.co/jaipkapoor99/ultron-113m) (10B FineWeb-Edu pretraining).
> - **Fine-Tuned Instruct Checkpoint**: [`jaipkapoor99/ultron-113m-instruct`](https://huggingface.co/jaipkapoor99/ultron-113m-instruct) (SmolTalk SFT instruction alignment).
> - **Pre-tokenized Datasets**: [`jaipkapoor99/ultron-fineweb-edu-shards`](https://huggingface.co/datasets/jaipkapoor99/ultron-fineweb-edu-shards) & [`jaipkapoor99/ultron-smoltalk-shards`](https://huggingface.co/datasets/jaipkapoor99/ultron-smoltalk-shards).

---

## ⚡ Specifications at a Glance

| Property              | Specification                                                                   |
| :-------------------- | :------------------------------------------------------------------------------ |
| **Parameter Count**   | **113.3 Million** (tied embeddings)                                             |
| **Architecture**      | 12 Layers, 768 Embedding Dimension, Pre-RMSNorm                                 |
| **Attention**         | **Grouped-Query Attention (GQA)**: 12 Query Heads : 4 KV Heads (3:1 ratio)      |
| **Position Encoding** | **RoPE (Rotary)**, base $\theta = 10,000$, 1,024 Context Length                 |
| **Stability**         | **QK-Head RMSNorm** + **Logit Soft-Capping** (15.0 via tanh)                    |
| **FeedForward**       | **SwiGLU** (Tensor Core aligned to multiples of 64)                             |
| **Optimization**      | **PyTorch Muon** (2D body matrices) + **Fused AdamW** (1D vectors & embeddings) |
| **Pre-training**      | 10 Billion tokens (FineWeb-Edu), 189k tok/s peak, Final Dev Loss: **2.965** (19.39 PPL) |
| **Fine-Tuning (SFT)** | 164M tokens (SmolTalk), 192k tok/s peak, Final Dev Loss: **1.466** (4.33 PPL)  |

---

## 🚀 Quickstart

### 1. Setup Environment

```bash
git clone https://github.com/jaipkapoor99/ultron.git
cd ultron

uv venv --python 3.14.6 .venv
source .venv/bin/activate

# Install PyTorch 2.13 and dependencies
uv pip install torch==2.13.0
uv pip install -r pyproject.toml --group dev
```

### 2. Code Quality & Testing

```bash
# Linting, formatting check, and static type checking
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync pyrefly check .

# Run full test suite (151 unit and contract tests)
uv run --no-sync python -m pytest -q
```

### 3. Training & Interactive Chat

```bash
# 1. Pre-training (from scratch on FineWeb-Edu)
accelerate launch train.py --mode=fresh

# 2. Supervised Fine-Tuning (from base checkpoint on SmolTalk)
accelerate launch train_sft.py --mode=fresh

# 3. Interactive Terminal Chat with Ultron-113M-Instruct
python scripts/chat.py --checkpoint-dir=accelerate_sft_checkpoint
```

---

## 🧪 Benchmark Highlights

Evaluated across **17,195 un-truncated zero-shot examples** using EleutherAI's `lm-evaluation-harness`:

| Benchmark | Base Model (Norm) | **Instruct SFT (Norm)** | Baseline (Random) |
| :--- | :---: | :---: | :---: |
| **PIQA** | **63.66%** | **61.86%** | 50.00% |
| **Winogrande** | 49.17% | **50.83%** | 50.00% |
| **ARC-Easy** | **47.05%** | **45.29%** | 25.00% |
| **HellaSwag** | **33.75%** | **32.83%** | 25.00% |
| **OpenBookQA** | **32.20%** | **29.80%** | 25.00% |
| **ARC-Challenge** | **26.54%** | **25.26%** | 25.00% |
| **Macro Average** | **40.41%** | **38.58%** | 33.33% |

> [!NOTE]
> **A Note on 113M Parameter Scale & Realistic Expectations:**
> While knowledge recall in a 113M parameter model may seem modest or prone to entity hallucinations compared to multi-billion parameter cloud models, this is the honest reality of the sub-200M parameter regime on a 10B token budget. Ultron-113M establishes clean linguistic syntax, solid commonsense reasoning, and disciplined ChatML turn-taking in a lightweight footprint that executes locally in milliseconds with under 250MB VRAM.

---

## 📚 Documentation & Deep Dives

To keep this overview concise, detailed deep-dives are organized into dedicated documentation:

- 🏛️ **[Architecture & Math Deep-Dive](docs/ARCHITECTURE.md)**: RoPE, GQA, SwiGLU, QK-Norm, Muon optimizer math, and GPT-2 vs Ultron evolution table.
- ⚙️ **[Training & Data Pipeline](docs/TRAINING.md)**: Rust streaming tokenization, zero-copy memory mapping, forkserver safety, and exact checkpoint resume contract.
- 📊 **[Benchmarks & Telemetry Report](docs/BENCHMARKS.md)**: Full EleutherAI evaluation logs, W&B telemetry, full validation pass, and qualitative completions.
- 🗺️ **[Engineering Journey](JOURNEY.md)**: Lessons learned, bug investigations, and systems breakthroughs.
- 📜 **[Contributor & Repository Guidelines](AGENTS.md)**: Development workflows, invariants, and coding standards.

---

## 📜 Acknowledgments & Citation

- Andrej Karpathy for [_Neural Networks: Zero to Hero_
- Andrej Karpathy for [_Neural Networks: Zero to Hero_](https://github.com/karpathy/build-nanogpt) and `nanoGPT`.
- Keller Jordan et al. for pioneering the [Muon](https://github.com/KellerJordan/Muon) optimizer.

```bibtex
@misc{jordan2024muon,
  author = {Jordan, Keller and Jin, Yuchen and Boza, Vlado and You, Jiacheng and Cesista, Franz and Newhouse, Laker and Bernstein, Jeremy},
  title  = {Muon: An optimizer for hidden layers in neural networks},
  year   = {2024},
  url    = {https://kellerjordan.github.io/posts/muon/}
}
```
