import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Fetch FunASR SenseVoiceSmall model for RL rewards.")
    parser.add_argument("--model_id", type=str, default="FunAudioLLM/SenseVoiceSmall", help="HF model id or local path")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="checkpoints/funasr/SenseVoiceSmall",
        help="Directory to store the model files",
    )
    parser.add_argument("--offline", action="store_true", help="Do not download, only validate cache_dir")
    return parser.parse_args()


def main():
    args = parse_args()
    cache_dir = os.path.abspath(args.cache_dir)
    if args.offline:
        if not os.path.exists(cache_dir):
            raise SystemExit(f"Offline mode requested but cache_dir missing: {cache_dir}")
        print(f"Offline mode: using cached model at {cache_dir}")
        return

    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:  # noqa: BLE001
        raise SystemExit("huggingface_hub is required. Install with: pip install huggingface_hub") from exc

    snapshot_download(
        repo_id=args.model_id,
        local_dir=cache_dir,
        local_dir_use_symlinks=False,
    )
    print(f"Downloaded {args.model_id} to {cache_dir}")


if __name__ == "__main__":
    main()
