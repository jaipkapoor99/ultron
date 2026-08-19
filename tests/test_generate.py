"""CPU-safe tests for generation policy and checkpoint metadata."""

import json
from typing import Any

import pytest
import torch

from config import UltronConfig
from scripts.generate import (
    apply_repetition_penalty,
    ban_repeated_ngrams,
    build_parser,
    filter_logits,
    load_checkpoint_metadata,
    select_next_token,
    validate_args,
)


def test_checkpoint_metadata_restores_exact_config(tmp_path: Any) -> None:
    config = UltronConfig(C=384, n_head=6, n_kv_head=2, n_layer=4)
    (tmp_path / "training_state.json").write_text(
        json.dumps(
            {
                "step": 12,
                "model_config": config.to_metadata(),
            }
        )
    )

    restored, state = load_checkpoint_metadata(tmp_path)

    assert restored.to_metadata() == config.to_metadata()
    assert state["step"] == 12


def test_legacy_checkpoint_metadata_warns_and_uses_defaults(
    tmp_path: Any,
) -> None:
    (tmp_path / "training_state.json").write_text(json.dumps({"step": 12}))

    with pytest.warns(RuntimeWarning, match="no saved model_config"):
        config, state = load_checkpoint_metadata(tmp_path)

    assert config.to_metadata() == UltronConfig().to_metadata()
    assert state["step"] == 12


def test_greedy_selection_ignores_sampling_controls() -> None:
    logits = torch.tensor([[0.0, 4.0, 1.0]])
    tokens = torch.tensor([[0]])

    selected = select_next_token(
        logits,
        tokens,
        greedy=True,
        temperature=0.0,
        top_k=0,
        top_p=1.0,
        min_p=0.0,
        repetition_penalty=1.0,
        no_repeat_ngram_size=0,
    )

    assert selected.tolist() == [[1]]


def test_cli_accepts_multiple_prompts() -> None:
    args = build_parser().parse_args(
        [
            "--prompt",
            "First prompt",
            "--prompt",
            "Second prompt",
            "--samples",
            "3",
        ]
    )

    assert args.prompts == ["First prompt", "Second prompt"]
    assert args.samples == 3


def test_cli_defaults_match_recommended_sampling_policy() -> None:
    args = build_parser().parse_args([])

    assert args.max_tokens == 70
    assert args.samples == 1
    assert args.seed == 1337
    assert args.greedy is False
    assert args.temperature == 0.7
    assert args.top_k == 20
    assert args.top_p == 0.95
    assert args.min_p == 0.0
    assert args.repetition_penalty == 1.1
    assert args.no_repeat_ngram_size == 3
    assert args.ignore_eos is False


def test_seeded_sampling_is_reproducible() -> None:
    logits = torch.zeros(2, 8)
    tokens = torch.tensor([[0, 1], [2, 3]])
    first_generator = torch.Generator().manual_seed(42)
    second_generator = torch.Generator().manual_seed(42)
    kwargs = {
        "greedy": False,
        "temperature": 0.7,
        "top_k": 5,
        "top_p": 0.9,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "no_repeat_ngram_size": 0,
    }

    first = select_next_token(
        logits,
        tokens,
        generator=first_generator,
        **kwargs,
    )
    second = select_next_token(
        logits,
        tokens,
        generator=second_generator,
        **kwargs,
    )

    assert torch.equal(first, second)


def test_top_k_filter_retains_only_requested_candidates() -> None:
    logits = torch.tensor([[1.0, 4.0, 3.0, 2.0]])

    filtered = filter_logits(logits, top_k=2, top_p=1.0, min_p=0.0)

    assert torch.isfinite(filtered).tolist() == [[False, True, True, False]]


def test_repetition_penalty_reduces_seen_positive_logits() -> None:
    logits = torch.tensor([[1.0, 4.0, 3.0]])
    tokens = torch.tensor([[1]])

    penalized = apply_repetition_penalty(logits, tokens, penalty=2.0)

    assert penalized.tolist() == [[1.0, 2.0, 3.0]]


def test_no_repeat_ngram_bans_only_repeated_continuations() -> None:
    logits = torch.zeros(2, 8)
    tokens = torch.tensor(
        [
            [1, 2, 3, 1, 2],
            [4, 5, 6, 4, 7],
        ]
    )

    filtered = ban_repeated_ngrams(logits, tokens, ngram_size=3)

    assert torch.isneginf(filtered[0, 3])
    assert torch.isfinite(filtered[0, 4])
    assert torch.isfinite(filtered[1]).all()


def test_no_repeat_ngram_changes_token_selection() -> None:
    logits = torch.tensor([[0.0, 1.0, 2.0, 10.0]])
    tokens = torch.tensor([[1, 2, 3, 1, 2]])

    selected = select_next_token(
        logits,
        tokens,
        greedy=True,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        min_p=0.0,
        repetition_penalty=1.0,
        no_repeat_ngram_size=3,
    )

    assert selected.tolist() == [[2]]


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--samples", "0"], "--samples"),
        (["--temperature", "0"], "--temperature"),
        (["--top-p", "0"], "--top-p"),
        (["--min-p", "2"], "--min-p"),
        (["--repetition-penalty", "0.9"], "--repetition-penalty"),
        (["--no-repeat-ngram-size", "-1"], "--no-repeat-ngram-size"),
    ],
)
def test_invalid_cli_values_are_rejected(arguments: list[str], message: str) -> None:
    args = build_parser().parse_args(arguments)

    with pytest.raises(ValueError, match=message):
        validate_args(args)
