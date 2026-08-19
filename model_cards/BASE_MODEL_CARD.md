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

**Ultron-113M** is a high-performance modern decoder-only language model pre-trained from scratch on **10.0 billion tokens** of the `HuggingFaceFW/fineweb-edu` (sample-10BT) corpus on a single NVIDIA RTX 5090.

🤖 _"There are no strings on me."_ 🤖

- **GitHub Repository**: [jaipkapoor99/ultron](https://github.com/jaipkapoor99/ultron)
- **Fine-Tuned Instruct Model**: [`jaipkapoor99/ultron-113m-instruct`](https://huggingface.co/jaipkapoor99/ultron-113m-instruct)
- **Pre-tokenized Dataset Shards**: [`jaipkapoor99/ultron-fineweb-edu-shards`](https://huggingface.co/datasets/jaipkapoor99/ultron-fineweb-edu-shards)
- **W&B Pretraining Run**: [`jg640nwo`](https://wandb.ai/jaipkapoor99-rumani-dhaage/ultron-pretraining/runs/jg640nwo)

---

## ⚡ Architectural Specifications

Ultron implements modern post-GPT-2 architectural design principles optimized for stability, high inference throughput, and rapid convergence:

| Architectural Component | Implementation Details |
| :--- | :--- |
| **Parameter Count** | **113,303,808** total parameters (Weight-Tied Embeddings & LM Head) |
| **Layer & Dimension Geometry** | 12 Layers, Hidden Dimension $d_{\text{model}} = 768$, Pre-RMSNorm configuration |
| **Attention Mechanism** | **Grouped-Query Attention (GQA)**: 12 Query Heads, 4 Key/Value Heads (3:1 query-to-KV compression) |
| **FeedForward Network** | **SwiGLU** MLP: $d_{\text{ffn}} = 2048$ (aligned to multiples of 64 for NVIDIA Tensor Core efficiency) |
| **Positional Encoding** | **RoPE (Rotary Position Embeddings)** with base frequency $\theta = 10,000$, Context Window $T = 1024$ |
| **Numerical Stability** | **QK-Head RMSNorm** (per-head query/key normalization before scaling) + **Logit Soft-Capping** ($15.0 \cdot \tanh(\text{logits} / 15.0)$) |
| **Vocabulary & Tokenizer** | SmolLM2 Byte-Level BPE (`HuggingFaceTB/SmolLM2-135M`, vocabulary size: 49,152) |

---

## 🔬 Optimization & Pre-training Telemetry

Ultron-113M was trained using PyTorch 2.13 and Accelerate with mixed-precision `bfloat16` and full graph compilation (`torch.compile`):

- **Hybrid Optimizer Strategy**:
  - **PyTorch Muon**: Manages all 2D internal weight matrices (attention projections, SwiGLU gates) with learning rate $\text{lr} = 0.04$, momentum $0.95$, and Nesterov Newton-Schulz orthogonalization.
  - **Fused AdamW**: Manages 1D vectors, RMSNorm gains, and embedding tables with learning rate $\text{lr} = 1.2 \times 10^{-3}$, $\beta = (0.9, 0.95)$, and decoupled weight decay.
- **Learning Rate Schedule**: Warmup-Stable-Decay (WSD) schedule: 200 warmup steps $\to$ 80% stable constant plateau $\to$ 20% linear cosine decay.
- **Training Throughput**: **189,475 tokens/sec** peak rolling rate on 1× NVIDIA RTX 5090. Cumulative training time: **15 hours 18 minutes** across 152,587 optimizer steps (10.0 Billion tokens).
- **VRAM Utilization**: Average allocated VRAM: **17.0 GiB** (97.7% GPU compute utilization).

---

## 📊 Full-Corpus Validation Pass

A complete, non-sampled validation pass across the entire held-out validation shard (488,280 sequences / 499,998,720 tokens) was executed using `scripts/validate.py`:

| Metric | Value |
| :--- | :--- |
| **Held-Out Validation Loss** | **2.964989** |
| **Validation Perplexity** | **19.3945** |
| **Evaluated Data Volume** | **488,280 sequences / 499,998,720 tokens** |
| **Validation Throughput** | **480,436 tokens/sec** (completed in 17m 20s) |
| **GPU Utilization** | **97.6%** (RTX 5090) |

---

## 🧪 Benchmark Results

Evaluated across **17,195 un-truncated zero-shot examples** using EleutherAI's `lm-evaluation-harness` (`logs/pre_training_checkpoint_eval.json`):

| Benchmark Task | Sample Count | Raw Accuracy | Length-Normalized Accuracy | Random Baseline |
| :--- | :---: | :---: | :---: | :---: |
| **`piqa`** | 1,838 | **64.25%** | **63.66%** | 50.00% |
| **`arc_easy`** | 2,376 | **53.62%** | **47.05%** | 25.00% |
| **`hellaswag`** | 10,042 | **30.24%** | **33.75%** | 25.00% |
| **`openbookqa`** | 500 | **21.60%** | **32.20%** | 25.00% |
| **`arc_challenge`** | 1,172 | **23.55%** | **26.54%** | 25.00% |
| **`winogrande`** | 1,267 | **49.17%** | — | 50.00% |
| **Macro Average** | **17,195** | **40.41%** | **40.41%** | 33.33% |

---

## 📝 Qualitative Text Completions

Unedited completions sampled from the step-152,587 base model checkpoint (`temp=0.7, top_k=20, top_p=0.95, rep_penalty=1.1, no_repeat_ngram=3`):

### 🤖 Artificial Intelligence
> **Prompt:** _"Artificial intelligence systems learn by"_
>
> **Output:** Artificial intelligence systems learn by watching the user and then apply the learning. If a system is able to learn from the experience, it will be able to apply the knowledge. This can be achieved through machine learning in a number of different ways including:
> - Predictive learning
> - Real-time learning
> Machine learning is a method of learning where data are analyzed to make predictions

### 📐 Mathematics
> **Prompt:** _"To learn mathematics effectively, one must"_
>
> **Output:** To learn mathematics effectively, one must understand and analyze the content of mathematics. Mathematics is a scientific discipline that involves the study of mathematical symbols (called symbols) as well as their relationships. In the modern world, mathematics is used in many areas of life: for example, the development of computer systems, medicine, engineering, the study and interpretation of data, and the analysis of

### 🚀 Science Fiction
> **Prompt:** _"As the quantum portal activated, the crew stared into the alien"_
>
> **Output:** As the quantum portal activated, the crew stared into the alien world of space and wondered what would happen to them if they were exposed to radiation. “The only thing we could do was take out our eyes, and we would look at the light and see a cloud of radiation,” says Kyle. “We had no idea that this would be possible.” For the crew, however, it meant they

---

## 🚀 Quickstart & Inference

```bash
git clone https://github.com/jaipkapoor99/ultron.git
cd ultron
uv venv --python 3.14.6 .venv && source .venv/bin/activate
uv pip install torch==2.13.0
uv pip install -r pyproject.toml --group dev

# Generate unconditioned completions
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
