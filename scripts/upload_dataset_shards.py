"""CLI Entry Point for Uploading Ultron Shards to Hugging Face Hub."""

import argparse
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import UltronConfig
from sharding.uploader import (
    upload_dataset_shards,
)


def main() -> None:
    config = UltronConfig()
    parser = argparse.ArgumentParser(
        description="Upload Ultron dataset shards to Hugging Face Hub"
    )
    parser.add_argument(
        "--sft",
        action="store_true",
        help="Upload SFT instruction shards (shards_sft) instead of pretraining shards",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Target Hugging Face Dataset Repo ID",
    )
    parser.add_argument(
        "--shards-dir",
        type=str,
        default=None,
        help="Path to local shards directory",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Set target dataset repository to private",
    )
    args = parser.parse_args()

    default_repo = (
        config.hf_sft_dataset_repo_id if args.sft else config.hf_dataset_repo_id
    )
    default_dir = "shards_sft" if args.sft else "shards_edu"

    target_repo = args.repo_id or default_repo
    shards_dir = Path(args.shards_dir or default_dir).resolve()

    upload_dataset_shards(
        target_repo=target_repo,
        shards_dir=shards_dir,
        is_sft=args.sft,
        private=args.private,
    )


if __name__ == "__main__":
    main()
