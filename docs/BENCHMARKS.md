# Ultron-113M Evaluation & Benchmarks

This document records the evaluation benchmarks, validation reports, telemetry summaries, and qualitative generation samples for **Ultron-113M**.

---

## 1. EleutherAI `lm-evaluation-harness` Report

Evaluated across **17,195 total log-likelihood examples** without sample limits using `scripts/eval_lm_harness.py`. Results are stored in `logs/pre_training_checkpoint_eval.json`:

```bash
accelerate launch scripts/eval_lm_harness.py --limit=0
```

| Benchmark Task      | Examples | Raw Accuracy | Length-Normalized Accuracy | Random Baseline |
| :------------------ | -------: | -----------: | -------------------------: | --------------: |
| **`piqa`**          |    1,838 |   **64.25%** |                 **63.66%** |          50.00% |
| **`arc_easy`**      |    2,376 |   **53.62%** |                 **47.05%** |          25.00% |
| **`hellaswag`**     |   10,042 |   **30.24%** |                 **33.75%** |          25.00% |
| **`arc_challenge`** |    1,172 |       23.55% |                 **26.54%** |          25.00% |
| **`openbookqa`**    |      500 |       21.60% |                 **32.20%** |          25.00% |
| **`winogrande`**    |    1,267 |       49.17% |                          — |          50.00% |

- **Macro-Average Accuracy**: **40.41%**
- PIQA, ARC-Easy, and HellaSwag show substantial gains over random chance, confirming foundational reasoning and world knowledge in a 113M parameter envelope.

---

## 2. Full-Corpus Validation Pass (`scripts/validate.py`)

A full, non-sampled validation pass over the held-out validation shard (488,280 sequences / 499,998,720 tokens) was completed and logged to [W&B run `jg640nwo`](https://wandb.ai/jaipkapoor99-rumani-dhaage/ultron-pretraining/runs/jg640nwo):

```bash
accelerate launch scripts/validate.py
```

| Metric                       | Value                                        |
| :--------------------------- | :------------------------------------------- |
| **Validation Loss**          | **2.964989**                                 |
| **Validation Perplexity**    | **19.3945**                                  |
| **Total Evaluated Data**     | **488,280 sequences / 499,998,720 tokens**   |
| **Runtime**                  | **17 minutes 20.7 seconds**                  |
| **Average Token Throughput** | **480,436 tokens/sec** (Peak: 517,229 tok/s) |
| **Average GPU Utilization**  | **97.6%** (RTX 5090)                         |
| **VRAM Usage**               | **9.4 GiB average / 12.5 GiB peak**          |

---

## 3. Pre-training Telemetry Summary

| Metric                       | Value                        | Details                                   |
| :--------------------------- | :--------------------------- | :---------------------------------------- |
| **Total Steps**              | **152,587 / 152,587 (100%)** | 10.0 Billion tokens                       |
| **Effective Throughput**     | **181,543 tokens/sec**       | End-to-end tokens / total tracked time    |
| **Peak Rolling Throughput**  | **189,475 tokens/sec**       | 30s rolling rate on NVIDIA RTX 5090       |
| **Total Training Runtime**   | **15h 18m 03s (55,083s)**    | Cumulative W&B tracked runtime            |
| **Final Train Loss (avg)**   | **2.9446**                   | Interval-average loss over last 250 steps |
| **Final Dev Loss (sampled)** | **2.9687**                   | 20 rotating validation batches            |

### Training Loss Curve

![Ultron interval-average train loss](../assets/average_train_loss.svg)

---

## 4. Qualitative Sample Generations (`scripts/generate.py`)

Unedited completions sampled from the step-152,587 base model checkpoint (`temp=0.7, top_k=20, top_p=0.95, rep_penalty=1.1, no_repeat_ngram=3`):

### 🤖 Artificial Intelligence

> **Prompt:** _"Artificial intelligence systems learn by"_
>
> **Output:** Artificial intelligence systems learn by watching the user and then apply the learning. If a system is able to learn from the experience, it will be able to apply the knowledge. This can be achieved through machine learning in a number of different ways including:
>
> - Predictive learning
> - Real-time learning
>   Machine learning is a method of learning where data are analyzed to make predictions

### 📐 Mathematics

> **Prompt:** _"To learn mathematics effectively, one must"_
>
> **Output:** To learn mathematics effectively, one must understand and analyze the content of mathematics. Mathematics is a scientific discipline that involves the study of mathematical symbols (called symbols) as well as their relationships. In the modern world, mathematics is used in many areas of life: for example, the development of computer systems, medicine, engineering, the study and interpretation of data, and the analysis of

### 🚀 Science Fiction

> **Prompt:** _"As the quantum portal activated, the crew stared into the alien"_
> **Output:** To learn mathematics effectively, one must understand and analyze the content of mathematics. Mathematics is a scientific discipline that involves the study of mathematical symbols (called symbols) as well as their relationships. In the modern world, mathematics is used in many areas of life: for example, the development of computer systems, medicine, engineering, the study and interpretation of data, and the analysis of

### 🚀 Science Fiction

> **Prompt:** _"As the quantum portal activated, the crew stared into the alien"_
>
> **Output:** As the quantum portal activated, the crew stared into the alien world of space and wondered what would happen to them if they were exposed to radiation. “The only thing we could do was take out our eyes, and we would look at the light and see a cloud of radiation,” says Kyle. “We had no idea that this would be possible.” For the crew, however, it meant they
