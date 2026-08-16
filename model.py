"""
Ultron (113M) PyTorch Model Module
Implements:
1. PyTorch scaled-dot-product attention with automatic backend selection
2. SwiGLU Gated FeedForward Network (LLaMA 3 / Qwen 2.5 architecture)
3. Pre-LayerNorm Residual Skip Connections
4. Weight Tying (lm_head.weight = wte.weight)
5. 1 / sqrt(2*N) Residual Parameter Initialization
6. KV Caching for efficient autoregressive generation
7. Grouped-Query Attention (GQA)
8. Vocab Size padding (multiple of 64) for Tensor Cores
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .config import UltronConfig
except ImportError:
    from config import UltronConfig


@dataclass
class UltronOutput:
    """Output class for UltronModel containing logits, optional loss, and optional KV cache."""
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    past_key_values: tuple | None = None


def load_ultron_state_dict(model: UltronModel, state_dict: dict):
    """Load an Ultron checkpoint and reject incompatible key sets.

    Safetensors stores only one copy of tied parameters, so exactly one of the
    embedding/LM-head aliases may be absent. Every other missing or unexpected
    key is treated as an incompatible checkpoint.
    """
    cleaned = {key.removeprefix("_orig_mod."): value for key, value in state_dict.items()}
    incompatible = model.load_state_dict(cleaned, strict=False)
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    tied_aliases = {"transformer.wte.weight", "lm_head.weight"}

    invalid_missing = missing - tied_aliases
    if invalid_missing or unexpected or len(missing & tied_aliases) > 1:
        raise RuntimeError(
            "Incompatible Ultron checkpoint: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )

    model.tie_weights()
    return sorted(missing), sorted(unexpected)

def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """ Apply Rotary Position Embedding (RoPE) to input tensor x """
    cos = cos.to(x.dtype)
    sin = sin.to(x.dtype)
    d = x.shape[-1] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos[..., :d] - x2 * sin[..., :d]
    y2 = x1 * sin[..., :d] + x2 * cos[..., :d]
    return torch.cat([y1, y2], dim=-1)

class RotaryEmbedding(nn.Module):
    """ Rotary Position Embedding (RoPE) Module (LLaMA 3 / Qwen 2.5 standard) """
    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int, offset: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len + offset > self.cos_cached.size(2):
            self._build_cache(seq_len + offset)
        return self.cos_cached[:, :, offset:seq_len+offset, :], self.sin_cached[:, :, offset:seq_len+offset, :]

class CausalSelfAttention(nn.Module):
    """Grouped-Query Attention with RoPE using PyTorch SDPA."""
    def __init__(self, config: UltronConfig):
        super().__init__()
        assert config.C % config.n_head == 0
        assert config.n_head % config.n_kv_head == 0
        
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.num_queries_per_kv = config.n_head // config.n_kv_head
        self.head_dim = config.head_dim
        self.C = config.C
        self.dropout_p = config.dropout

        q_dim = self.n_head * self.head_dim
        kv_dim = self.n_kv_head * self.head_dim
        
        self.c_attn = nn.Linear(config.C, q_dim + 2 * kv_dim, bias=False)
        self.c_proj = nn.Linear(config.C, config.C, bias=False)
        self.c_proj.RESIDUAL_SCALE_INIT = 1
        
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x: torch.Tensor, rot_emb: tuple[torch.Tensor, torch.Tensor] | None = None, past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        B, T, C = x.size()
        qkv = self.c_attn(x)
        
        q_dim = self.n_head * self.head_dim
        kv_dim = self.n_kv_head * self.head_dim
        
        q, k, v = qkv.split([q_dim, kv_dim, kv_dim], dim=2)
        
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        
        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if rot_emb is not None:
            cos, sin = rot_emb
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)
            
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)
            
        new_past_key_value = (k, v)

        if self.num_queries_per_kv > 1:
            # Expand GQA heads without repeat_interleave.
            k_len = k.size(2)
            k = k.unsqueeze(2).expand(B, self.n_kv_head, self.num_queries_per_kv, k_len, self.head_dim).reshape(B, self.n_head, k_len, self.head_dim)
            v = v.unsqueeze(2).expand(B, self.n_kv_head, self.num_queries_per_kv, k_len, self.head_dim).reshape(B, self.n_head, k_len, self.head_dim)
            
        dropout_p = self.dropout_p if self.training else 0.0
        
        # PyTorch selects the best available SDPA backend for the device and dtype.
        is_causal = self.training or q.size(2) > 1
        y = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, dropout_p=dropout_p)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y), new_past_key_value

class SwiGLUMLP(nn.Module):
    """ Modern SwiGLU FeedForward Network (used in LLaMA 3, Qwen 2.5, & Mistral) """
    def __init__(self, config: UltronConfig):
        super().__init__()
        hidden_dim = int(2 * (4 * config.C) / 3)
        hidden_dim = 64 * ((hidden_dim + 63) // 64) # Round to multiple of 64 for Tensor Cores
        
        self.w1 = nn.Linear(config.C, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.C, bias=False)
        self.w3 = nn.Linear(config.C, hidden_dim, bias=False)
        self.w2.RESIDUAL_SCALE_INIT = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class RMSNorm(nn.Module):
    """ Root Mean Square Layer Normalization (RMSNorm) - LLaMA / Qwen 2.5 standard """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return output * self.weight

class Block(nn.Module):
    """Transformer block with SDPA attention and a SwiGLU MLP."""
    def __init__(self, config: UltronConfig):
        super().__init__()
        self.ln_1 = RMSNorm(config.C)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.C)
        self.mlp  = SwiGLUMLP(config)

    def forward(self, x: torch.Tensor, rot_emb: tuple[torch.Tensor, torch.Tensor] | None = None, past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        attn_out, present_key_value = self.attn(self.ln_1(x), rot_emb=rot_emb, past_key_value=past_key_value)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, present_key_value

class UltronModel(nn.Module):
    """Full Ultron (113M) language model with RMSNorm, RoPE, and GQA."""
    def __init__(self, config: UltronConfig):
        super().__init__()
        self.config = config
        
        # Padded vocab size for Tensor Core efficiency
        self.vocab_size = math.ceil(config.vocab_size / 64) * 64
        
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(self.vocab_size, config.C),
                "drop": nn.Dropout(config.dropout),
                "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                "ln_f": RMSNorm(config.C),
            }
        )
        self.lm_head = nn.Linear(config.C, self.vocab_size, bias=False)

        self.rotary_emb = RotaryEmbedding(config.head_dim, max_seq_len=config.T, base=config.rope_base)
            
        # Weight Tying
        self.transformer.wte.weight = self.lm_head.weight
        
        self.apply(self._init_weights)

    @property
    def device(self):
        return next(self.parameters()).device

    def tie_weights(self):
        self.transformer.wte.weight = self.lm_head.weight

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'RESIDUAL_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None, use_cache: bool = False, past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None):
        _batch_size, T = idx.size()
        
        past_length = past_key_values[0][0].size(2) if past_key_values is not None else 0
        assert T + past_length <= self.config.T, f"Cannot forward sequence length {T + past_length}, model block size is {self.config.T}"
        
        tok_emb = self.transformer.wte(idx)
        x = self.transformer.drop(tok_emb)
        rot_emb = self.rotary_emb(x, T, offset=past_length)
        
        present_key_values = [] if use_cache else None
        
        for i, block in enumerate(self.transformer.h):
            past_kv = past_key_values[i] if past_key_values is not None else None
            
            if getattr(self.config, 'gradient_checkpointing', False) and self.training:
                # Cache is not supported with gradient checkpointing
                x, _ = torch.utils.checkpoint.checkpoint(block, x, rot_emb, None, use_reentrant=False)
            else:
                x, present_kv = block(x, rot_emb=rot_emb, past_key_value=past_kv)
                
            if use_cache:
                present_key_values.append(present_kv)
                
        x = self.transformer.ln_f(x)
        
        logits = self.lm_head(x)
        if self.config.logit_softcap > 0.0:
            logits = self.config.logit_softcap * torch.tanh(logits / self.config.logit_softcap)
            
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            
        if use_cache:
            return UltronOutput(logits=logits, loss=loss, past_key_values=present_key_values)
        return UltronOutput(logits=logits, loss=loss)

    def configure_optimizers(self, learning_rate: float):
        """Hybrid Muon + AdamW optimizer setup.
        
        Muon handles 2D weight matrices (attention, MLP projections).
        AdamW handles everything else (embeddings, norms, biases).
        """
        partitions = self.partition_optimizer_parameters()
        muon_params = partitions["muon"]
        adamw_decay_params = partitions["adamw_decay"]
        adamw_nodecay_params = partitions["adamw_nodecay"]
        
        adamw_groups = [
            {"params": adamw_decay_params, "weight_decay": 0.1},
            {"params": adamw_nodecay_params, "weight_decay": 0.0}
        ]
        
        optimizer_muon = torch.optim.Muon(
            muon_params,
            lr=0.04,
            momentum=0.95,
            weight_decay=0.0,
        )
        optimizer_adamw = torch.optim.AdamW(adamw_groups, lr=learning_rate, betas=(0.9, 0.95), fused=True)
        return optimizer_muon, optimizer_adamw

    def partition_optimizer_parameters(self):
        """Assign every trainable parameter to exactly one optimizer group."""
        partitions = {
            "muon": [],
            "adamw_decay": [],
            "adamw_nodecay": [],
        }
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if name == "transformer.wte.weight":
                partitions["adamw_nodecay"].append(parameter)
            elif parameter.ndim == 2:
                partitions["muon"].append(parameter)
            else:
                partitions["adamw_decay"].append(parameter)
        return partitions

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        token_selector: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """Generate tokens with KV caching while delegating decoding policy.

        ``UltronModel`` owns autoregressive execution and cache management.
        Callers own the policy that selects a token from the next-token logits.
        Without a selector, generation is greedy.
        """
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens cannot be negative")
        if idx.ndim != 2 or idx.size(1) == 0:
            raise ValueError("idx must have shape (batch, sequence) with a non-empty sequence")
        if max_new_tokens >= self.config.T:
            raise ValueError("max_new_tokens must be smaller than the context length")
        if eos_token_id is not None and not 0 <= eos_token_id < self.config.vocab_size:
            raise ValueError("eos_token_id must be inside the tokenizer vocabulary")

        max_prompt_len = max(1, self.config.T - max_new_tokens)
        if idx.size(1) > max_prompt_len:
            idx = idx[:, -max_prompt_len:]

        if max_new_tokens == 0:
            return idx

        if token_selector is None:
            def token_selector(logits, _tokens):
                return torch.argmax(
                            logits,
                            dim=-1,
                            keepdim=True,
                        )

        past_key_values = None
        finished = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device)
        for _ in range(max_new_tokens):
            # Pre-fill with the prompt, then decode one token at a time.
            idx_cond = idx if past_key_values is None else idx[:, -1:]
                
            out = self(idx_cond, use_cache=True, past_key_values=past_key_values)
            logits = out.logits[:, -1, :self.config.vocab_size] # Strip padded vocab
            past_key_values = out.past_key_values

            idx_next = token_selector(logits, idx)
            if idx_next.shape != (idx.size(0), 1):
                raise ValueError("token_selector must return shape (batch, 1)")
            if idx_next.dtype != torch.long:
                raise TypeError("token_selector must return torch.long token IDs")
            if idx_next.min().item() < 0 or idx_next.max().item() >= self.config.vocab_size:
                raise ValueError("token_selector returned an ID outside the vocabulary")

            if eos_token_id is not None:
                idx_next = torch.where(
                    finished.unsqueeze(1),
                    torch.full_like(idx_next, eos_token_id),
                    idx_next,
                )
                finished |= idx_next.squeeze(1).eq(eos_token_id)

            idx = torch.cat((idx, idx_next), dim=1)
            if finished.all():
                break

        return idx
