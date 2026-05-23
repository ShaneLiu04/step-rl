"""
Pre-download Qwen2.5-7B-Instruct using ModelScope (fast in China).
"""

import argparse

from modelscope import snapshot_download


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--local_dir", type=str, default="./models/Qwen2.5-7B-Instruct")
    args = parser.parse_args()

    print(f"Downloading {args.model_id} via ModelScope...")
    print(f"Target directory: {args.local_dir}")

    snapshot_download(
        model_id=args.model_id,
        cache_dir=args.local_dir,
    )
    print(f"Download complete: {args.local_dir}")


if __name__ == "__main__":
    main()
