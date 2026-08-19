"""Ultron Tokenizer & Atomic Shard Utilities.

Provides a unified tokenizer wrapper for pretraining and SFT, supporting
ChatML formatting, prompt loss masking, special token constants, and atomic
file I/O.
"""

import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoTokenizer

from config import UltronConfig


def fsync_directory(path: Path) -> None:
    """Flush directory metadata changes to physical storage."""
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    """Atomically commit a JSON file to disk."""
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary_path, path)
    fsync_directory(path.parent)


def atomic_numpy_write(path: Path, values: np.ndarray) -> None:
    """Atomically commit an unpickled NumPy array to disk."""
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("wb") as file:
        np.save(file, values, allow_pickle=False)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary_path, path)
    fsync_directory(path.parent)


def atomic_shard_write(path: Path, values: np.ndarray) -> None:
    """Atomically commit a raw binary shard array to disk."""
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("wb") as file:
        values.tofile(file)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary_path, path)
    fsync_directory(path.parent)


class UltronTokenizer:
    """Unified tokenizer interface for Ultron pretraining and SFT."""

    IM_START = "<|im_start|>"
    IM_END = "<|im_end|>"
    END_OF_TEXT = "<|endoftext|>"

    def __init__(self, tokenizer_name_or_path: str | None = None) -> None:
        name = tokenizer_name_or_path or UltronConfig.tokenizer_name
        self.hf_tokenizer: Any = AutoTokenizer.from_pretrained(
            name, model_max_length=int(1e9)
        )

        convert = getattr(self.hf_tokenizer, "convert_tokens_to_ids", None)
        raw_start = convert(self.IM_START) if callable(convert) else 0
        raw_end = convert(self.IM_END) if callable(convert) else 0

        self.im_start_id: int = raw_start if isinstance(raw_start, int) else 0
        self.im_end_id: int = raw_end if isinstance(raw_end, int) else 0
        self.eos_token_id: int = int(getattr(self.hf_tokenizer, "eos_token_id", 0))
        self.bos_token_id: int | None = getattr(self.hf_tokenizer, "bos_token_id", None)
        pad_id = getattr(self.hf_tokenizer, "pad_token_id", None)
        self.pad_token_id: int = (
            pad_id if isinstance(pad_id, int) else self.eos_token_id
        )

    @property
    def vocab_size(self) -> int:
        return len(self.hf_tokenizer)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Encode raw text into token IDs."""
        return self.hf_tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, tokens: Sequence[int], skip_special_tokens: bool = False) -> str:
        """Decode token IDs into text."""
        return self.hf_tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> str | list[int]:
        """Format a list of conversational messages into a ChatML string or token list."""
        if (
            hasattr(self.hf_tokenizer, "chat_template")
            and self.hf_tokenizer.chat_template
        ):
            return self.hf_tokenizer.apply_chat_template(
                messages,
                tokenize=tokenize,
                add_generation_prompt=add_generation_prompt,
            )

        # Standard fallback ChatML rendering
        formatted = ""
        for message in messages:
            role = message["role"]
            content = message["content"]
            formatted += f"{self.IM_START}{role}\n{content}{self.IM_END}\n"
        if add_generation_prompt:
            formatted += f"{self.IM_START}assistant\n"

        if tokenize:
            return self.encode(formatted, add_special_tokens=False)
        return formatted

    def encode_chat_conversation(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
    ) -> tuple[list[int], list[int]]:
        """Encode a multi-turn conversation with target loss masking.

        Returns:
            tuple[list[int], list[int]]: (input_ids, target_ids) where user/system
            tokens in target_ids are set to -1 (masked) and assistant response tokens
            contain the ground-truth token IDs for supervised fine-tuning.
        """
        all_messages: list[dict[str, str]] = []
        if system_prompt:
            all_messages.append({"role": "system", "content": system_prompt})
        elif messages and messages[0].get("role") != "system":
            all_messages.append(
                {
                    "role": "system",
                    "content": "You are Ultron, a helpful, precise, and respectful AI assistant.",
                }
            )

        all_messages.extend(messages)

        input_ids: list[int] = []
        target_ids: list[int] = []

        for message in all_messages:
            role = message["role"]
            content = message["content"]

            header = f"{self.IM_START}{role}\n"
            body = f"{content}{self.IM_END}\n"

            header_tokens = self.encode(header, add_special_tokens=False)
            body_tokens = self.encode(body, add_special_tokens=False)

            turn_inputs = header_tokens + body_tokens
            input_ids.extend(turn_inputs)

            if role == "assistant":
                # Mask header (we don't train model to generate '<|im_start|>assistant\n')
                # But compute loss on content + '<|im_end|>\n'
                turn_targets = [-1] * len(header_tokens) + body_tokens
            else:
                # Mask all tokens for system and user turns
                turn_targets = [-1] * len(turn_inputs)

            target_ids.extend(turn_targets)

        return input_ids, target_ids
