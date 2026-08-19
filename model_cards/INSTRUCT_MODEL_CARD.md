---
language:
  - en
license: mit
tags:
  - pytorch
  - causal-lm
  - muon
  - modern-transformer
  - chatml
  - instruction-tuned
  - sft
  - smoltalk
datasets:
  - HuggingFaceTB/smoltalk
pipeline_tag: text-generation
inference: false
model-index:
  - name: Ultron-113M-Instruct
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
            value: 61.86
      - task:
          type: text-generation
          name: Text Generation
        dataset:
          name: Winogrande
          type: winogrande
        metrics:
          - name: Accuracy
            type: acc
            value: 50.83
      - task:
          type: text-generation
          name: Text Generation
        dataset:
          name: ARC Easy
          type: arc_easy
        metrics:
          - name: Accuracy (Norm)
            type: acc_norm
            value: 45.29
      - task:
          type: text-generation
          name: Text Generation
        dataset:
          name: HellaSwag
          type: hellaswag
        metrics:
          - name: Accuracy (Norm)
            type: acc_norm
            value: 32.83
      - task:
          type: text-generation
          name: Text Generation
        dataset:
          name: OpenBookQA
          type: openbookqa
        metrics:
          - name: Accuracy (Norm)
            type: acc_norm
            value: 29.80
      - task:
          type: text-generation
          name: Text Generation
        dataset:
          name: ARC Challenge
          type: arc_challenge
        metrics:
          - name: Accuracy (Norm)
            type: acc_norm
            value: 25.26
---

# Ultron-113M-Instruct (Supervised Fine-Tuned Model)

**Ultron-113M-Instruct** is an instruction-tuned language model derived from **Ultron-113M** via Supervised Fine-Tuning (SFT) across **163.84 Million tokens** of the `HuggingFaceTB/smoltalk` conversational corpus.

🤖 _"There are no strings on me."_ 🤖

- **GitHub Repository**: [jaipkapoor99/ultron](https://github.com/jaipkapoor99/ultron)
- **Base Pre-trained Model**: [`jaipkapoor99/ultron-113m`](https://huggingface.co/jaipkapoor99/ultron-113m)
- **Pre-tokenized SFT Dataset Shards**: [`jaipkapoor99/ultron-smoltalk-shards`](https://huggingface.co/datasets/jaipkapoor99/ultron-smoltalk-shards)

---

## ⚡ Specifications at a Glance

| Parameter | Specification |
| :--- | :--- |
| **Total Parameters** | **113.3 Million** (Weight-Tied Embeddings & LM Head) |
| **Architecture** | 12 Layers, $d_{\text{model}} = 768$, Pre-RMSNorm, RoPE ($\theta=10,000$), SwiGLU, QK-Norm |
| **Attention** | **Grouped-Query Attention (GQA)**: 12 Query Heads : 4 KV Heads (3:1 ratio) |
| **Chat Template** | **ChatML** (`<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n...<|im_end|>`) |
| **Fine-Tuning Budget** | 163.84 Million tokens across 2,500 optimizer steps (Peak: 192k tok/s) |
| **Full SFT Dev Loss** | **1.4662** (Perplexity: **4.3328**) across 43,938 held-out dev sequences (31.0M target tokens) |
| **Tokenizer** | SmolLM2 Byte-Level BPE (`HuggingFaceTB/SmolLM2-135M`, vocab size: 49,152) |

---

## 🧪 Benchmark Highlights

Evaluated across **17,195 zero-shot examples** using EleutherAI's `lm-evaluation-harness`:

| Benchmark Task | Base Model (Norm) | **Instruct SFT (Norm)** | Random Baseline |
| :--- | :---: | :---: | :---: |
| **`piqa`** | **63.66%** | **61.86%** | 50.00% |
| **`winogrande`** | 49.17% | **50.83%** | 50.00% |
| **`arc_easy`** | **47.05%** | **45.29%** | 25.00% |
| **`hellaswag`** | **33.75%** | **32.83%** | 25.00% |
| **`openbookqa`** | **32.20%** | **29.80%** | 25.00% |
| **`arc_challenge`** | **26.54%** | **25.26%** | 25.00% |
| **Macro Average** | **40.41%** | **38.58%** | 33.33% |

---

## 💬 Interactive Chat

To chat directly with Ultron-113M-Instruct in the terminal:

```bash
git clone https://github.com/jaipkapoor99/ultron.git
cd ultron
uv venv --python 3.14.6 .venv && source .venv/bin/activate
uv pip install torch==2.13.0
uv pip install -r pyproject.toml --group dev

# Start terminal chat session
python scripts/chat.py --checkpoint-dir=accelerate_sft_checkpoint
```

### Sample Conversation

```text
You > hello
Ultron > Hello! How can I help you today?

You > Explain photosynthesis in one sentence.
Ultron > Photosynthesis is the process by which green plants use sunlight, water, and carbon dioxide to create oxygen and energy in the form of sugar.
```

---

## 🧠 Note on 113M Parameter Scale

While entity memory and deep encyclopedic recall in a 113M model are naturally constrained compared to multi-billion parameter foundation models, this reflects the physical scaling bounds of sub-200M models on a 10B token budget. Ultron-113M-Instruct demonstrates disciplined conversational turn taking, strict ChatML stopping, and concise formatting in a sub-250MB memory footprint.

---

## 📜 Citation

```bibtex
@misc{ultron_instruct2026,
  author = {Jai Kapoor},
  title = {Ultron-113M-Instruct: Lightweight Instruction-Tuned Transformer},
  year = {2026},
  publisher = {Hugging Face},
  journal = {Hugging Face Hub repository},
  howpublished = {\url{https://huggingface.co/jaipkapoor99/ultron-113m-instruct}}
}
```
