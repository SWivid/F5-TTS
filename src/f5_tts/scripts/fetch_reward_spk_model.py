import argparse
import os
import tarfile
import zipfile


def parse_args():
    parser = argparse.ArgumentParser(description="Fetch WeSpeaker model for RL rewards.")
    parser.add_argument(
        "--repo_id",
        type=str,
        default="openspeech/wespeaker-models",
        help="Hugging Face repo containing WeSpeaker model archives",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="cnceleb_resnet34.zip",
        help="Primary model archive filename to download",
    )
    parser.add_argument(
        "--fallback_filename",
        type=str,
        default="voxceleb_resnet34.zip",
        help="Fallback archive filename if the primary is missing",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="checkpoints/wespeaker/cnceleb_resnet34",
        help="Directory to store the extracted model files",
    )
    parser.add_argument("--offline", action="store_true", help="Do not download, only validate cache_dir")
    return parser.parse_args()


def _extract_archive(archive_path: str, cache_dir: str) -> None:
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            zip_ref.extractall(cache_dir)
        return
    if tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path, "r:*") as tar_ref:
            tar_ref.extractall(cache_dir)
        return
    raise SystemExit(f"Unsupported archive format: {archive_path}")


def main():
    args = parse_args()
    cache_dir = os.path.abspath(args.cache_dir)
    if args.offline:
        if not os.path.exists(cache_dir):
            raise SystemExit(f"Offline mode requested but cache_dir missing: {cache_dir}")
        print(f"Offline mode: using cached model at {cache_dir}")
        return

    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:  # noqa: BLE001
        raise SystemExit("huggingface_hub is required. Install with: pip install huggingface_hub") from exc

    os.makedirs(cache_dir, exist_ok=True)
    download_dir = os.path.dirname(cache_dir)
    os.makedirs(download_dir, exist_ok=True)
    try:
        archive_path = hf_hub_download(
            repo_id=args.repo_id,
            filename=args.filename,
            local_dir=download_dir,
            local_dir_use_symlinks=False,
        )
    except Exception as exc:  # noqa: BLE001
        if not args.fallback_filename:
            raise
        print(f"Primary archive '{args.filename}' not found, trying '{args.fallback_filename}'.")
        archive_path = hf_hub_download(
            repo_id=args.repo_id,
            filename=args.fallback_filename,
            local_dir=download_dir,
            local_dir_use_symlinks=False,
        )
    _extract_archive(archive_path, cache_dir)
    print(f"Downloaded WeSpeaker model to {cache_dir}")


if __name__ == "__main__":
    main()
