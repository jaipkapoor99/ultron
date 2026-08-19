"""CLI Entry Point for Tokenizing SmolTalk SFT Dataset."""

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sharding.sft import tokenize_sft


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tokenize SmolTalk instruction dataset into deterministic SFT shards."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="shards_sft",
        help="Directory to write binary SFT shards and metadata.",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=5_000_000,
        help="Number of tokens per binary SFT shard (default: 5,000,000).",
    )
    parser.add_argument(
        "--max-shards",
        type=int,
        default=None,
        help="Maximum number of SFT shards to generate.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Hugging Face instruction dataset identifier.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Dataset subset configuration (default: 'all').",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    tokenize_sft(
        output_dir=args.output_dir,
        shard_size_tokens=args.shard_size,
        max_shards=args.max_shards,
        dataset_name=args.dataset,
        dataset_config=args.config,
    )


if __name__ == "__main__":
    main()
