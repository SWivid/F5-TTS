import argparse
import os
import tarfile
import urllib.request
import zipfile


DEFAULT_URL = "https://wenet.org.cn/downloads?models=wespeaker&version=cnceleb_resnet34.zip"


def parse_args():
    parser = argparse.ArgumentParser(description="Fetch WeSpeaker model for RL rewards.")
    parser.add_argument("--url", type=str, default=DEFAULT_URL, help="Download URL for the WeSpeaker model zip")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="checkpoints/wespeaker/chinese",
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

    os.makedirs(cache_dir, exist_ok=True)
    archive_path = os.path.join(cache_dir, "wespeaker_model_archive.zip")
    urllib.request.urlretrieve(args.url, archive_path)
    _extract_archive(archive_path, cache_dir)
    print(f"Downloaded WeSpeaker model to {cache_dir}")


if __name__ == "__main__":
    main()
