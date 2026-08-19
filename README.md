# Ultron-113M: Modern Transformer Pre-training Pipeline

[![CI](https://github.com/jaipkapoor99/ultron/actions/workflows/ci.yml/badge.svg)](https://github.com/jaipkapoor99/ultron/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.14-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.13-EE4C2C?logo=pytorch&logoColor=white)
![Ruff](https://img.shields.io/badge/Linter-Ruff-black?logo=ruff&logoColor=white)
![Pyrefly](https://img.shields.io/badge/Type_Check-Pyrefly-blue)

A high-performance PyTorch implementation of **Ultron-113M**, pre-trained from scratch on **10.0 billion tokens** of the FineWeb-Edu corpus on a single NVIDIA RTX 5090.

🤖 _"There are no strings on me."_ 🤖

> [!IMPORTANT]
> **🚀 Base Model Artifacts Available:**
>
> - **Pre-trained Weights**: [`jaipkapoor99/ultron-113m`](https://huggingface.co/jaipkapoor99/ultron-113m) on Hugging Face Hub (complete resumable checkpoint).
> - **Pre-tokenized Shards**: [`jaipkapoor99/ultron-fineweb-edu-shards`](https://huggingface.co/datasets/jaipkapoor99/ultron-fineweb-edu-shards) (100M-token binary shards).

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
| **Throughput**        | **189,475 tokens/sec** peak rolling (~15.3 hours for 10B tokens on 1× RTX 5090) |
| **Validation Loss**   | **2.965** (19.39 perplexity on 500M held-out tokens)                            |

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

# Run CPU-safe test suite (133 tests with coverage enforcement)
uv run --no-sync python -m pytest -q --cov --cov-report=term-missing
```

### 3. Training & Inference

```bash
# Pre-training
accelerate launch train.py --mode=fresh

# Text Generation
accelerate launch scripts/generate.py \
  --prompt "Artificial intelligence systems learn by" \
  --samples 4
```

---

## 🧪 Benchmark Highlights

Evaluated across **17,195 un-truncated zero-shot / few-shot examples** using EleutherAI's `lm-evaluation-harness`:

| Benchmark         | Accuracy (Length-Norm) | Baseline (Random) |
| :---------------- | :--------------------: | :---------------: |
| **PIQA**          |       **63.66%**       |      50.00%       |
| **ARC-Easy**      |       **47.05%**       |      25.00%       |
| **HellaSwag**     |       **33.75%**       |      25.00%       |
| **OpenBookQA**    |       **32.20%**       |      25.00%       |
| **ARC-Challenge** |       **26.54%**       |      25.00%       |

Macro-average benchmark accuracy reaches **40.41%**, establishing strong foundational reasoning in a compact 113M footprint.

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
