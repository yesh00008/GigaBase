"""Download the common-pile/comma_v0.1_training_dataset repository."""

import argparse
import os
from typing import Optional

from huggingface_hub import login, snapshot_download


def download_comma_dataset(
    token: Optional[str] = None,
    save_path: str = "data/raw/comma_dataset",
    revision: Optional[str] = None,
    max_workers: int = 8,
):
    os.makedirs(save_path, exist_ok=True)

    if token:
        login(token=token)

    print("Downloading dataset repository snapshot...")

    local_path = snapshot_download(
        repo_id="common-pile/comma_v0.1_training_dataset",
        repo_type="dataset",
        token=token,
        local_dir=save_path,
        local_dir_use_symlinks=False,
        resume_download=True,
        revision=revision,
        max_workers=max_workers,
    )

    print(f"Dataset downloaded to {local_path}")

    return local_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download common-pile/comma_v0.1_training_dataset")
    parser.add_argument("--token", type=str, help="Hugging Face authentication token")
    parser.add_argument(
        "--save_path",
        type=str,
        default="data/raw/comma_dataset",
        help="Path to save the dataset (default: data/raw/comma_dataset)",
    )
    parser.add_argument("--revision", type=str, default=None, help="Dataset revision to download")
    parser.add_argument("--max-workers", type=int, default=8, help="Parallel download workers")

    args = parser.parse_args()

    download_comma_dataset(
        token=args.token,
        save_path=args.save_path,
        revision=args.revision,
        max_workers=args.max_workers,
    )