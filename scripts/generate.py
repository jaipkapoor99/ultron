"""Generate text from a local Ultron Accelerate checkpoint.

Examples:
    accelerate launch scripts/generate.py --prompt "The theory of relativity"
    accelerate launch scripts/generate.py --prompt "Once upon a time" --samples 4
    accelerate launch scripts/generate.py --prompt "Water freezes at" --greedy
"""

import argparse
import json
import os
import sys
import warnings
from collections.abc import Callable
from dataclasses import fields
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from transformers import AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import UltronConfig
from model import UltronModel, load_ultron_state_dict


def load_checkpoint_metadata(
    checkpoint_dir: Path,
) -> tuple[UltronConfig, dict]:
    """Load the exact saved configuration, with a legacy-checkpoint fallback."""
    state_file = checkpoint_dir / "training_state.json"
    state = {}
    if state_file.exists():
        with state_file.open() as handle:
            state = json.load(handle)

    saved_config = state.get("model_config")
    if saved_config is None:
        warnings.warn(
            "Checkpoint has no saved model_config; falling back to the current "
            "UltronConfig defaults. Future checkpoints record the exact config.",
            RuntimeWarning,
            stacklevel=2,
        )
        return UltronConfig(), state

    init_fields = {field.name for field in fields(UltronConfig) if field.init}
    unknown = set(saved_config) - init_fields
    if unknown:
        raise RuntimeError(
            f"Checkpoint model_config contains unknown fields: {sorted(unknown)}"
        )
    return UltronConfig.from_metadata(**saved_config), state


def load_model_weights(
    model: UltronModel,
    checkpoint_dir: Path,
    print_fn: Callable[[str], None] = print,
) -> UltronModel:
    """Load and validate model weights from an Accelerate checkpoint."""
    weight_file = None
    for relative_path in (
        "model.safetensors",
        "pytorch_model.bin",
        "pytorch_model/model.safetensors",
        "pytorch_model/pytorch_model.bin",
    ):
        candidate = checkpoint_dir / relative_path
        if candidate.exists():
            weight_file = candidate
            break

    if weight_file is None:
        raise FileNotFoundError(
            f"No model weights found in '{checkpoint_dir}'. Expected "
            "model.safetensors or pytorch_model.bin."
        )

    print_fn(f"Loading weights from: {weight_file}")
    if weight_file.suffix == ".safetensors":
        from safetensors.torch import load_file

        state_dict = load_file(str(weight_file), device="cpu")
    else:
        state_dict = torch.load(weight_file, map_location="cpu", weights_only=True)

    missing, unexpected = load_ultron_state_dict(model, state_dict)
    if missing:
        print_fn(f"Expected tied-weight alias omitted: {missing}")
    if unexpected:
        print_fn(f"Unexpected keys: {unexpected}")
    return model


def apply_repetition_penalty(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    penalty: float,
) -> torch.Tensor:
    """Apply the standard sign-aware repetition penalty to prior token IDs."""
    if penalty == 1.0:
        return logits
    scores = torch.gather(logits, 1, token_ids)
    scores = torch.where(scores < 0, scores * penalty, scores / penalty)
    return logits.scatter(1, token_ids, scores)


def ban_repeated_ngrams(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    ngram_size: int,
) -> torch.Tensor:
    """Prevent any completed n-gram from being generated a second time."""
    if ngram_size == 0:
        return logits

    filtered = logits.clone()
    sequence_length = token_ids.size(1)
    prefix_length = ngram_size - 1
    if sequence_length < prefix_length:
        return filtered

    for batch_index in range(token_ids.size(0)):
        prefix = token_ids[batch_index, sequence_length - prefix_length :]
        banned_tokens = []
        final_start = sequence_length - ngram_size
        for start in range(final_start + 1):
            if torch.equal(
                token_ids[batch_index, start : start + prefix_length],
                prefix,
            ):
                banned_tokens.append(token_ids[batch_index, start + prefix_length])
        if banned_tokens:
            filtered[batch_index, torch.stack(banned_tokens)] = -torch.inf
    return filtered


def filter_logits(
    logits: torch.Tensor,
    *,
    top_k: int,
    top_p: float,
    min_p: float,
) -> torch.Tensor:
    """Apply top-k, nucleus, and min-p filters while retaining one token."""
    filtered = logits.clone()
    if top_k > 0:
        threshold = torch.topk(
            filtered,
            min(top_k, filtered.size(-1)),
            dim=-1,
        ).values[:, -1:]
        filtered.masked_fill_(filtered < threshold, -torch.inf)

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(
            filtered,
            descending=True,
            dim=-1,
        )
        cumulative = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove = cumulative > top_p
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = False
        sorted_logits.masked_fill_(remove, -torch.inf)
        filtered = torch.full_like(filtered, -torch.inf).scatter(
            1,
            sorted_indices,
            sorted_logits,
        )

    if min_p > 0.0:
        threshold = filtered.max(dim=-1, keepdim=True).values + torch.log(
            torch.tensor(min_p, dtype=filtered.dtype, device=filtered.device)
        )
        filtered.masked_fill_(filtered < threshold, -torch.inf)
    return filtered


def select_next_token(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    *,
    greedy: bool,
    temperature: float,
    top_k: int,
    top_p: float,
    min_p: float,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Select one token per row; this is the complete decoding policy."""
    logits = apply_repetition_penalty(
        logits.float(),
        token_ids,
        repetition_penalty,
    )
    logits = ban_repeated_ngrams(
        logits,
        token_ids,
        no_repeat_ngram_size,
    )
    if greedy:
        return torch.argmax(logits, dim=-1, keepdim=True)

    filtered = filter_logits(
        logits / temperature,
        top_k=top_k,
        top_p=top_p,
        min_p=min_p,
    )
    probabilities = F.softmax(filtered, dim=-1)
    return torch.multinomial(
        probabilities,
        num_samples=1,
        generator=generator,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ultron text generation")
    parser.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
        help="Prompt to continue; repeat this option for multiple prompts",
    )
    parser.add_argument("--max-tokens", type=int, default=70)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=3)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("accelerate_checkpoint"),
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.max_tokens < 0:
        raise ValueError("--max-tokens cannot be negative")
    if args.samples <= 0:
        raise ValueError("--samples must be greater than zero")
    if not args.greedy and args.temperature <= 0:
        raise ValueError("--temperature must be greater than zero when sampling")
    if args.top_k < 0:
        raise ValueError("--top-k cannot be negative")
    if not 0 < args.top_p <= 1:
        raise ValueError("--top-p must be in (0, 1]")
    if not 0 <= args.min_p <= 1:
        raise ValueError("--min-p must be in [0, 1]")
    if args.repetition_penalty < 1:
        raise ValueError("--repetition-penalty must be at least 1")
    if args.no_repeat_ngram_size < 0:
        raise ValueError("--no-repeat-ngram-size cannot be negative")


def main() -> None:
    args = build_parser().parse_args()
    try:
        validate_args(args)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    if not any(
        key in os.environ
        for key in (
            "ACCELERATE_TORCH_DEVICE",
            "ACCELERATE_PROCESS_ID",
            "LOCAL_RANK",
            "ACCELERATE_MIXED_PRECISION",
        )
    ):
        raise RuntimeError(
            "Run with: accelerate launch scripts/generate.py [arguments]"
        )

    accelerator = Accelerator()
    accelerator.print(f"Device : {accelerator.device}")

    config, training_state = load_checkpoint_metadata(args.checkpoint)
    accelerator.print(f"Loading tokenizer ({config.tokenizer_name})...")
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    if tokenizer is None or tokenizer.vocab_size != config.vocab_size:
        raise RuntimeError(
            f"Tokenizer vocabulary ({getattr(tokenizer, 'vocab_size', None)}) does not match "
            f"checkpoint configuration ({config.vocab_size})"
        )

    model = load_model_weights(
        UltronModel(config),
        args.checkpoint,
        accelerator.print,
    )
    model = accelerator.prepare(model)
    model.eval()

    if "step" in training_state:
        accelerator.print(
            f"Step   : {training_state['step']:,} / "
            f"{training_state.get('max_steps', config.max_steps):,}"
        )

    generator = torch.Generator(device=accelerator.device)
    generator.manual_seed(args.seed)

    def selector(logits, tokens):
        return select_next_token(
            logits,
            tokens,
            greedy=args.greedy,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            min_p=args.min_p,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            generator=generator,
        )

    eos_token_id = None if args.ignore_eos else tokenizer.eos_token_id

    accelerator.print(
        "\nMode   : "
        + (
            "greedy"
            if args.greedy
            else (
                f"sampled (seed={args.seed}, temperature={args.temperature}, "
                f"top-k={args.top_k}, top-p={args.top_p}, min-p={args.min_p}, "
                f"repetition-penalty={args.repetition_penalty}, "
                f"no-repeat-ngram={args.no_repeat_ngram_size})"
            )
        )
    )

    unwrapped_model = accelerator.unwrap_model(model)
    prompts = args.prompts or ["Hello, my name is"]
    for prompt_index, prompt in enumerate(prompts, start=1):
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        max_prompt_length = max(1, config.T - args.max_tokens)
        if input_ids.size(1) > max_prompt_length:
            accelerator.print(
                f"Prompt {prompt_index} truncated from {input_ids.size(1)} to "
                f"{max_prompt_length} tokens."
            )
            input_ids = input_ids[:, -max_prompt_length:]
        input_ids = input_ids.repeat(args.samples, 1).to(accelerator.device)

        accelerator.print(f"\nPrompt {prompt_index}: {prompt!r}")
        with torch.no_grad(), accelerator.autocast():
            output_ids = unwrapped_model.generate(
                input_ids,
                max_new_tokens=args.max_tokens,
                token_selector=selector,
                eos_token_id=eos_token_id,
            )

        prompt_length = input_ids.size(1)
        for sample_index, token_ids in enumerate(output_ids, start=1):
            continuation = tokenizer.decode(
                token_ids[prompt_length:],
                skip_special_tokens=True,
            )
            accelerator.print(f"\n{'=' * 70}")
            accelerator.print(f"Prompt {prompt_index}, sample {sample_index}")
            accelerator.print(f"{prompt}{continuation}")
    accelerator.print("=" * 70)


if __name__ == "__main__":
    main()
