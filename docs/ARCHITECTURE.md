# Ultron-113M Architecture Specification

This document details the architectural design, layer layout, parameter choices, and engineering justifications of **Ultron-113M**.

---

## 1. Architectural Flow & Block Diagram

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

## 2. GPT-2 vs. Ultron-113M

| Feature                    |       GPT-2 (124M)       |            Ultron-113M            | Why it Matters (Engineering Justification)                                                                          |
| :------------------------- | :----------------------: | :-------------------------------: | :------------------------------------------------------------------------------------------------------------------ |
| **Positional Encoding**    | Absolute Learned (`wpe`) |         **RoPE (Rotary)**         | Enables zero-shot context length extension and better relative distance modeling.                                   |
| **Attention Mechanism**    |     Multi-Head (MHA)     |      **Grouped-Query (GQA)**      | 12 Q heads : 4 KV heads (**3:1 ratio**), reducing KV-cache memory usage during inference by **3×**.                 |
| **Attention Stability**    |  Standard Unnormalized   |        **QK-Head RMSNorm**        | Prevents logit explosion / attention entropy collapse during long pre-training runs.                                |
| **FFN Activation**         |      Standard GELU       |            **SwiGLU**             | Gated non-linearity yielding higher model capacity per FLOP; aligned to multiples of 64 for Tensor Core throughput. |
| **Layer Normalization**    |  LayerNorm (with bias)   |      **RMSNorm (Bias-Free)**      | Eliminates mean-centering overhead; 100% bias-free projections (`bias=False`) for cleaner gradient dynamics.        |
| **Logit Regularization**   |           None           |      **Logit Soft-Capping**       | Applies `tanh` capping (`cap=15.0`) to prevent overconfidence and extreme logit growth.                             |
| **Optimizer Engine**       |          AdamW           |  **PyTorch Muon + Fused AdamW**   | Uses built-in `torch.optim.Muon` for 2D body weights and AdamW for embeddings and normalization parameters.         |
| **Learning Rate Schedule** |       Cosine Decay       |         **WSD Schedule**          | Warmup-Stable-Decay schedule with an 80% stable phase followed by linear decay.                                     |
| **Mixed Precision**        |           FP32           |   **Native BFloat16 (`bf16`)**    | Dynamic range stability without loss scalers on modern CUDA GPUs.                                                   |
| **Graph Compiler**         |           None           | **PyTorch 2.0 (`torch.compile`)** | Fuses element-wise operations and kernel launches via Inductor.                                                     |

---

## 3. Core Component Deep-Dive

### 3.1 Rotary Position Embeddings (RoPE)

RoPE encodes token positions directly into the Query and Key head vectors via complex rotation:
\[
R*{\Theta, m}^d = \text{diag}\left(R*{\theta*1, m}, R*{\theta*2, m}, \dots, R*{\theta\_{d/2}, m}\right)
\]
Ultron uses a base frequency of $\theta = 10,000$, enabling reliable relative distance modeling across the 1,024-token context window.

### 3.2 Grouped-Query Attention (GQA) & QK-Head Normalization

Ultron deploys 12 Query heads ($n_{head} = 12$) paired with 4 Key/Value heads ($n_{kv\_head} = 4$), achieving a 3:1 GQA compression ratio. Prior to the dot-product computation, Queries and Keys are normalized via head-wise RMSNorm:
\[
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{\text{RMSNorm}(Q)\text{RMSNorm}(K)^T}{\sqrt{d_k}}\right)V
\]
This prevents attention logit blow-up in deep layers and stabilizes mixed-precision training.

### 3.3 SwiGLU FeedForward Network

The feedforward blocks use Swish-Gated Linear Units:
\[
\text{SwiGLU}(x) = \left(\text{SiLU}(x W_1) \otimes (x W_3)\right) W_2
\]
The hidden dimension is rounded up to the nearest multiple of 64 for optimal GPU Tensor Core alignment.

### 3.4 Logit Soft-Capping

Before computing cross-entropy loss or sampling next-token logits:
\[
\text{Logits}\_{\text{capped}} = 15.0 \times \tanh\left(\frac{\text{Logits}}{15.0}\right)
\]
This Gemma-2-style regularization restricts extreme probability divergence and prevents overconfidence during pre-training.

### 3.5 Hybrid Muon + AdamW Optimizer

- **`torch.optim.Muon`**: Handles 2D matrix weights (attention and MLP projections) using Newton-Schulz matrix iterations for orthogonalized parameter updates ($LR = 0.04$).
- **Fused `AdamW`**: Handles 1D vectors, RMSNorm gains, and token embeddings ($LR = 1.2 \times 10^{-3}$).
