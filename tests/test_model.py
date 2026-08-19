"""CPU-safe unit tests for the Ultron model and checkpoint contract."""

import os
import sys
from typing import Any

import pytest
import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import UltronConfig
from model import (
    RMSNorm,
    UltronModel,
    apply_rotary_emb,
    load_ultron_state_dict,
)


def tiny_config(**overrides: Any) -> UltronConfig:
    values = {
        "B": 2,
        "T": 32,
        "C": 32,
        "n_head": 4,
        "n_kv_head": 2,
        "n_layer": 2,
        "vocab_size": 128,
        "dropout": 0.0,
    }
    values.update(overrides)
    return UltronConfig(**values)


@pytest.fixture
def model() -> UltronModel:
    torch.manual_seed(0)
    return UltronModel(tiny_config()).eval()


def test_config_defaults() -> None:
    config = UltronConfig()
    assert config.C == 768
    assert config.n_head == 12
    assert config.n_kv_head == 4
    assert config.head_dim == 64
    assert config.vocab_size == 49152
    assert config.grad_accum_steps == 4
    assert config.eval_batches == 20


def test_documented_parameter_count() -> None:
    with torch.device("meta"):
        model = UltronModel(UltronConfig())
    assert sum(parameter.numel() for parameter in model.parameters()) == 113_266_944


def test_forward_shape_and_loss(model: UltronModel) -> None:
    inputs = torch.randint(0, model.config.vocab_size, (2, 12))
    targets = torch.randint(0, model.config.vocab_size, (2, 12))
    output = model(inputs, targets=targets)

    assert output.logits.shape == (2, 12, model.config.vocab_size)
    assert output.loss is not None
    assert torch.isfinite(output.loss)


def test_forward_rejects_context_longer_than_configured(
    model: UltronModel,
) -> None:
    inputs = torch.randint(
        0,
        model.config.vocab_size,
        (1, model.config.T + 1),
    )

    with pytest.raises(AssertionError, match="Cannot forward sequence length"):
        model(inputs)


def test_loss_ignores_masked_targets(model: UltronModel) -> None:
    inputs = torch.randint(0, model.config.vocab_size, (1, 6))
    targets = torch.randint(0, model.config.vocab_size, (1, 6))
    targets[:, -2:] = -1

    output = model(inputs, targets=targets)
    expected = F.cross_entropy(
        output.logits[:, :-2].reshape(-1, output.logits.size(-1)),
        targets[:, :-2].reshape(-1),
    )

    torch.testing.assert_close(output.loss, expected)


def test_future_tokens_do_not_change_prefix_logits(model: UltronModel) -> None:
    inputs = torch.randint(0, model.config.vocab_size, (2, 12))
    changed = inputs.clone()
    changed[:, 7:] = torch.randint(0, model.config.vocab_size, changed[:, 7:].shape)

    with torch.no_grad():
        original_logits = model(inputs).logits
        changed_logits = model(changed).logits

    torch.testing.assert_close(original_logits[:, :7], changed_logits[:, :7])


def test_cached_decoding_matches_full_forward(model: UltronModel) -> None:
    inputs = torch.randint(0, model.config.vocab_size, (2, 12))

    with torch.no_grad():
        full_logits = model(inputs).logits
        cache = None
        cached_logits = []
        for position in range(inputs.size(1)):
            output = model(
                inputs[:, position : position + 1],
                use_cache=True,
                past_key_values=cache,
            )
            cache = output.past_key_values
            cached_logits.append(output.logits)

    torch.testing.assert_close(
        torch.cat(cached_logits, dim=1),
        full_logits,
        rtol=1e-5,
        atol=1e-6,
    )


def test_checkpoint_loader_allows_only_tied_alias(model: UltronModel) -> None:
    state_dict = dict(model.state_dict())
    state_dict.pop("lm_head.weight")

    restored = UltronModel(tiny_config())
    missing, unexpected = load_ultron_state_dict(restored, state_dict)

    assert missing == ["lm_head.weight"]
    assert unexpected == []
    assert (
        restored.transformer.wte.weight.data_ptr() == restored.lm_head.weight.data_ptr()
    )
    torch.testing.assert_close(
        restored.transformer.wte.weight,
        model.transformer.wte.weight,
    )


def test_checkpoint_loader_rejects_other_missing_keys(model: UltronModel) -> None:
    state_dict = dict(model.state_dict())
    state_dict.pop("transformer.h.0.attn.c_attn.weight")

    with pytest.raises(RuntimeError, match="Incompatible Ultron checkpoint"):
        load_ultron_state_dict(UltronModel(tiny_config()), state_dict)


def test_checkpoint_loader_accepts_compiled_prefixes(model: UltronModel) -> None:
    state_dict = {
        f"_orig_mod.{key}": value for key, value in model.state_dict().items()
    }
    restored = UltronModel(tiny_config())

    missing, unexpected = load_ultron_state_dict(restored, state_dict)

    assert missing == []
    assert unexpected == []
    torch.testing.assert_close(
        restored.transformer.wte.weight,
        model.transformer.wte.weight,
    )


def test_checkpoint_loader_rejects_unexpected_keys(model: UltronModel) -> None:
    state_dict = dict(model.state_dict())
    state_dict["not_a_real_parameter"] = torch.zeros(1)

    with pytest.raises(RuntimeError, match="unexpected"):
        load_ultron_state_dict(UltronModel(tiny_config()), state_dict)


def test_optimizer_partition_is_complete_and_disjoint(model: UltronModel) -> None:
    partitions = model.partition_optimizer_parameters()
    grouped = [parameter for group in partitions.values() for parameter in group]
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]

    assert len(grouped) == len(trainable)
    assert len({id(parameter) for parameter in grouped}) == len(grouped)
    assert {id(parameter) for parameter in grouped} == {
        id(parameter) for parameter in trainable
    }
    assert model.transformer.wte.weight in partitions["adamw_nodecay"]
    assert all(parameter.ndim == 2 for parameter in partitions["muon"])
    assert all(parameter.ndim < 2 for parameter in partitions["adamw_decay"])


def test_uses_official_muon(model: UltronModel) -> None:
    optimizer_muon, optimizer_adamw = model.configure_optimizers(
        model.config.learning_rate
    )
    assert isinstance(optimizer_muon, torch.optim.Muon)
    assert optimizer_muon.param_groups[0]["nesterov"] is True
    assert optimizer_muon.param_groups[0]["ns_steps"] == 5
    assert optimizer_muon.param_groups[0]["weight_decay"] == 0.0
    assert [group["weight_decay"] for group in optimizer_adamw.param_groups] == [
        0.1,
        0.0,
    ]


def test_generation_preserves_batch_and_returns_valid_token_ids(
    model: UltronModel,
) -> None:
    torch.manual_seed(11)
    prompt = torch.randint(0, model.config.vocab_size, (2, 5))

    generated = model.generate(prompt, max_new_tokens=3)

    assert generated.shape == (2, 8)
    assert torch.equal(generated[:, :5], prompt)
    assert generated.min() >= 0
    assert generated.max() < model.config.vocab_size


def test_generation_delegates_token_selection(model: UltronModel) -> None:
    prompt = torch.randint(0, model.config.vocab_size, (2, 5))
    selected_token = 17

    generated = model.generate(
        prompt,
        max_new_tokens=3,
        token_selector=lambda _logits, tokens: torch.full(
            (tokens.size(0), 1),
            selected_token,
            dtype=torch.long,
            device=tokens.device,
        ),
    )

    assert generated[:, -3:].tolist() == [[selected_token] * 3] * 2


def test_generation_stops_when_every_sequence_reaches_eos(
    model: UltronModel,
) -> None:
    prompt = torch.randint(0, model.config.vocab_size, (2, 5))
    eos_token_id = 3

    generated = model.generate(
        prompt,
        max_new_tokens=10,
        token_selector=lambda _logits, tokens: torch.full(
            (tokens.size(0), 1),
            eos_token_id,
            dtype=torch.long,
            device=tokens.device,
        ),
        eos_token_id=eos_token_id,
    )

    assert generated.shape == (2, 6)
    assert generated[:, -1].tolist() == [eos_token_id, eos_token_id]


def test_generation_rejects_invalid_selector_output(model: UltronModel) -> None:
    prompt = torch.randint(0, model.config.vocab_size, (2, 5))

    with pytest.raises(ValueError, match="shape"):
        model.generate(
            prompt,
            max_new_tokens=1,
            token_selector=lambda _logits, _tokens: torch.zeros(
                2,
                dtype=torch.long,
            ),
        )


def test_repeated_batch_can_be_learned() -> None:
    torch.manual_seed(7)
    config = tiny_config(C=16, n_head=2, n_kv_head=1, n_layer=1, vocab_size=32)
    model = UltronModel(config).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-2)
    inputs = torch.arange(12).remainder(config.vocab_size).unsqueeze(0)
    targets = torch.roll(inputs, shifts=-1, dims=1)

    with torch.no_grad():
        initial_loss = F.cross_entropy(
            model(inputs).logits.flatten(0, 1),
            targets.flatten(),
        )

    for _ in range(20):
        optimizer.zero_grad()
        loss = model(inputs, targets=targets).loss
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_loss = model(inputs, targets=targets).loss

    assert final_loss < initial_loss * 0.5


def test_rmsnorm_has_unit_mean_square() -> None:
    norm = RMSNorm(64)
    output = norm(torch.randn(2, 10, 64))
    mean_square = output.pow(2).mean(-1)
    torch.testing.assert_close(
        mean_square,
        torch.ones_like(mean_square),
        rtol=1e-3,
        atol=1e-3,
    )


def test_rotary_embedding_preserves_shape(model: UltronModel) -> None:
    query = torch.randn(2, model.config.n_head, 16, model.config.head_dim)
    cosine, sine = model.rotary_emb(query, 16)
    rotated = apply_rotary_emb(query, cosine, sine)
    assert rotated.shape == query.shape


@pytest.mark.parametrize(
    "overrides",
    [
        {"C": 30, "n_head": 4, "n_kv_head": 2},
        {"C": 32, "n_head": 4, "n_kv_head": 3},
    ],
)
def test_config_rejects_incompatible_attention_dimensions(
    overrides: dict[str, Any],
) -> None:
    with pytest.raises(AssertionError):
        tiny_config(**overrides)


@pytest.mark.skipif(
    os.environ.get("ULTRON_TEST_COMPILE") != "1",
    reason="set ULTRON_TEST_COMPILE=1 to run the slower compiler smoke test",
)
def test_torch_compile_forward(model: UltronModel) -> None:
    compiled = torch.compile(model)
    inputs = torch.randint(0, model.config.vocab_size, (2, 12))
    assert compiled(inputs).logits.shape == (2, 12, model.config.vocab_size)
