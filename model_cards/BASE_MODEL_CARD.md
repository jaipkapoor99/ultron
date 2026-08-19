---
language:
  - en
license: mit
tags:
  - pytorch
  - causal-lm
  - muon
  - modern-transformer
  - rope
  - gqa
  - swiglu
  - fineweb-edu
datasets:
  - HuggingFaceFW/fineweb-edu
pipeline_tag: text-generation
inference: false
model-index:
  - name: Ultron-113M
    results:
      - task:
          type: text-generation
          name: Text Generation
        dataset:
          name: PIQA
          type: piqa
        metrics:
          - name: Accuracy (Norm)
            type: acc_norm
            value: 63.66
      - task:
          type: text-generation
          name: Text Generation
        dataset:
          name: ARC Easy
          type: arc_easy
        metrics:
          - name: Accuracy (Norm)
            type: acc_norm
            value: 47.05
      - task:
          type: text-generation
          name: Text Generation
        dataset:
          name: HellaSwag
          type: hellaswag
        metrics:
          - name: Accuracy (Norm)
            type: acc_norm
            value: 33.75
      - task:
          type: text-generation
          name: Text Generation
        dataset:
          name: OpenBookQA
          type: openbookqa
        metrics:
          - name: Accuracy (Norm)
            type: acc_norm
            value: 32.20
      - task:
          type: text-generation
          name: Text Generation
        dataset:
          name: ARC Challenge
          type: arc_challenge
        metrics:
          - name: Accuracy (Norm)
            type: acc_norm
            value: 26.54
      - task:
          type: text-generation
          name: Text Generation
        dataset:
          name: Winogrande
          type: winogrande
        metrics:
          - name: Accuracy
            type: acc
            value: 49.17
---

# Ultron-113M (Base Pre-trained Model)

**Ultron-113M** is a modern decoder-only language model pre-trained from scratch on **10.0 billion tokens** of the `HuggingFaceFW/fineweb-edu` (sample-10BT) corpus on a single NVIDIA RTX 5090.

🤖 _"There are no strings on me."_ 🤖

- **GitHub Repository**: [jaipkapoor99/ultron](https://github.com/jaipkapoor99/ultron)
- **Fine-Tuned Instruct Model**: [`jaipkapoor99/ultron-113m-instruct`](https://huggingface.co/jaipkapoor99/ultron-113m-instruct)
- **Pre-tokenized Dataset Shards**: [`jaipkapoor99/ultron-fineweb-edu-shards`](https://huggingface.co/datasets/jaipkapoor99/ultron-fineweb-edu-shards)

---

## ⚡ Model Specifications

| Parameter | Specification |
| :--- | :--- |
| **Total Parameters** | **113.3 Million** (Weight-Tied Embeddings & LM Head) |
| **Layers / Hidden Dim** | 12 Layers, $d_{\text{model}} = 768$ |
| **Attention Mechanism** | **Grouped-Query Attention (GQA)**: 12 Query Heads, 4 Key/Value Heads (3:1 ratio) |
| **FeedForward Network** | **SwiGLU** ($d_{\text{ffn}} = 2048$, Tensor Core aligned to multiples of 64) |
| **Positional Encoding** | **RoPE (Rotary Position Embeddings)**, base $\theta = 10,000$, Context $T = 1024$ |
| **Normalization & Stability** | **Pre-RMSNorm** + **QK-Head RMSNorm** + **Logit Soft-Capping** (15.0 via $\tanh$) |
| **Optimizer** | **PyTorch Muon** (2D body weight matrices, $\text{lr}=0.04$) + **Fused AdamW** (1D parameters & embeddings, $\text{lr}=1.2 \times 10^{-3}$) |
| **Tokenizer** | SmolLM2 Byte-Level BPE (`HuggingFaceTB/SmolLM2-135M`, vocab size: 49,152) |
| **Training Budget** | 10.0 Billion tokens across 152,587 optimizer steps |
| **Throughput** | **189,475 tokens/sec** peak rolling on 1× NVIDIA RTX 5090 (~15.3 hours total runtime) |
| **Validation Loss** | **2.965** (19.39 Perplexity on 500M held-out tokens) |

---

## 🧪 Benchmark Results

Evaluated across **17,195 un-truncated zero-shot / few-shot examples** using EleutherAI's `lm-evaluation-harness`:

| Benchmark Task | Metric | Value | Random Baseline |
| :--- | :--- | :---: | :---: |
| **`piqa`** | Accuracy (Length-Norm) | **63.66%** | 50.00% |
| **`arc_easy`** | Accuracy (Length-Norm) | **47.05%** | 25.00% |
| **`hellaswag`** | Accuracy (Length-Norm) | **33.75%** | 25.00% |
| **`openbookqa`** | Accuracy (Length-Norm) | **32.20%** | 25.00% |
| **`arc_challenge`** | Accuracy (Length-Norm) | **26.54%** | 25.00% |
| **`winogrande`** | Accuracy | **49.17%** | 50.00% |
| **Macro Average** | **Accuracy (Norm)** | **40.41%** | 33.33% |

---

## 🚀 How to Use

Clone the official repository to run text completion:

```bash
git clone https://github.com/jaipkapoor99/ultron.git
cd ultron
uv venv --python 3.14.6 .venv && source .venv/bin/activate
uv pip install torch==2.13.0
uv pip install -r pyproject.toml --group dev

# Generate completions
python scripts/generate.py \
  --prompt "Artificial intelligence systems learn by" \
  --samples 4
```

---

## 📜 Citation

```bibtex
@misc{ultron2026,
  author = {Jai Kapoor},
  title = {Ultron-113M: Modern Transformer Pre-training and Instruction Alignment from Scratch},
  year = {2026},
  publisher = {Hugging Face},
  journal = {Hugging Face Hub repository},
  howpublished = {\url{https://huggingface.co/jaipkapoor99/ultron-113m}}
}
```
