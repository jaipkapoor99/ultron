"""CPU-safe tests for the EleutherAI evaluation entry point."""

import json
from typing import Any

import pytest
import torch

from config import UltronConfig
from model import UltronModel
from scripts.eval_lm_harness import (
    DEFAULT_TASKS,
    build_parser,
    load_base_model,
    normalize_limit,
    parse_tasks,
    save_results,
    select_accuracy,
)


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


def test_default_cli_tasks_match_declared_suite() -> None:
    args = build_parser().parse_args([])

    assert parse_tasks(args.tasks) == list(DEFAULT_TASKS)
    assert args.limit == 50


def test_task_parser_trims_and_deduplicates() -> None:
    assert parse_tasks(" arc_easy,hellaswag,arc_easy, ") == [
        "arc_easy",
        "hellaswag",
    ]


def test_task_parser_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="at least one"):
        parse_tasks(" , ")


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0, None), (1, 1), (50, 50)],
)
def test_limit_normalization(value: int, expected: int | None) -> None:
    assert normalize_limit(value) == expected


def test_limit_normalization_rejects_negative_values() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        normalize_limit(-1)


@pytest.mark.parametrize(
    ("metrics", "expected"),
    [
        ({"acc,none": 0.25, "acc_norm,none": 0.5}, 0.25),
        ({"acc_norm,none": 0.5}, 0.5),
        ({"acc": 0.75}, 0.75),
        ({"acc,none": 0.0}, 0.0),
        ({"exact_match": 1.0}, None),
    ],
)
def test_accuracy_selection(metrics: dict[str, Any], expected: float | None) -> None:
    assert select_accuracy(metrics) == expected


def test_loader_rejects_empty_checkpoint_directory(tmp_path: Any) -> None:
    with pytest.raises(FileNotFoundError, match="No checkpoint weights found"):
        load_base_model(tmp_path, tiny_config())


def test_loader_accepts_pytorch_checkpoint_on_cpu(tmp_path: Any) -> None:
    source = UltronModel(tiny_config())
    torch.save(source.state_dict(), tmp_path / "pytorch_model.bin")
    messages = []

    restored = load_base_model(
        tmp_path,
        tiny_config(),
        print_fn=messages.append,
    )

    assert not restored.training
    assert restored.device.type == "cpu"
    assert messages[0].startswith("Loading weights from:")
    torch.testing.assert_close(
        restored.transformer.wte.weight,
        source.transformer.wte.weight,
    )


def test_loader_rejects_incompatible_checkpoint(tmp_path: Any) -> None:
    source = UltronModel(tiny_config())
    state_dict = source.state_dict()
    state_dict.pop("transformer.h.0.attn.c_attn.weight")
    torch.save(state_dict, tmp_path / "pytorch_model.bin")

    with pytest.raises(RuntimeError, match="Incompatible Ultron checkpoint"):
        load_base_model(tmp_path, tiny_config())


def test_results_are_saved_atomically_and_stringified(tmp_path: Any) -> None:
    output_path = tmp_path / "nested" / "results.json"

    save_results(
        {
            "results": {
                "arc_easy": {
                    "acc,none": 0.5,
                    "alias": object(),
                }
            },
            "ignored": "harness metadata",
        },
        output_path,
    )

    saved = json.loads(output_path.read_text())
    assert saved["arc_easy"]["acc,none"] == 0.5
    assert isinstance(saved["arc_easy"]["alias"], str)
    assert not output_path.with_suffix(".json.tmp").exists()
