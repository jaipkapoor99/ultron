"""Full-validation metric and W&B configuration tests."""

import sys
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn.functional as F

from config import UltronConfig
from model import UltronModel
from scripts.validate import load_checkpoint_weights, sequence_cross_entropy
from telemetry import ValidationTelemetry


def tiny_config() -> UltronConfig:
    return UltronConfig(
        B=1,
        T=16,
        C=16,
        n_head=2,
        n_kv_head=1,
        n_layer=1,
        vocab_size=32,
    )


def test_sequence_cross_entropy_matches_per_sequence_reference() -> None:
    torch.manual_seed(3)
    logits = torch.randn(2, 4, 7)
    targets = torch.randint(0, 7, (2, 4))

    losses = sequence_cross_entropy(logits, targets)
    expected = torch.stack(
        [
            F.cross_entropy(logits[index], targets[index])
            for index in range(len(targets))
        ]
    )

    torch.testing.assert_close(losses, expected)


def test_checkpoint_weight_loader_rejects_empty_directory(
    tmp_path: Any,
) -> None:
    with pytest.raises(FileNotFoundError, match="No model weights found"):
        load_checkpoint_weights(
            UltronModel(tiny_config()),
            tmp_path,
        )


def test_checkpoint_weight_loader_accepts_pytorch_state_dict(
    tmp_path: Any,
) -> None:
    source = UltronModel(tiny_config())
    weight_path = tmp_path / "pytorch_model.bin"
    torch.save(source.state_dict(), weight_path)
    restored = UltronModel(tiny_config())

    selected = load_checkpoint_weights(restored, tmp_path)

    assert selected == weight_path
    torch.testing.assert_close(
        restored.transformer.wte.weight,
        source.transformer.wte.weight,
    )


def test_full_validation_defines_wandb_metrics(monkeypatch: Any) -> None:
    definitions = []
    fake_wandb = SimpleNamespace(
        define_metric=lambda name, **options: definitions.append((name, options))
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    accelerator = SimpleNamespace(is_main_process=True)

    ValidationTelemetry._define_wandb_metrics(accelerator)

    by_name = dict(definitions)
    assert by_name["validation/*"] == {}
    assert by_name["validation/loss"]["summary"] == "min"
    assert by_name["validation/tokens_per_sec"]["summary"] == "max"
