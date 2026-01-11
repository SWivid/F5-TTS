import argparse
import json
from pathlib import Path

import soundfile as sf


def parse_args():
    parser = argparse.ArgumentParser(description="Build a small audio pack for integration tests.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="tests/assets/audio_pack",
        help="Output directory for audio pack",
    )
    parser.add_argument(
        "--hf_dataset",
        type=str,
        default="hf-internal-testing/librispeech_asr_dummy",
        help="HF dataset name to stream from",
    )
    parser.add_argument("--hf_split", type=str, default="validation", help="HF dataset split")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to export")
    parser.add_argument("--local_dir", type=str, default="", help="Local directory with audio files")
    return parser.parse_args()


def build_from_local(output_dir: Path, local_dir: Path, num_samples: int) -> None:
    audio_files = sorted([p for p in local_dir.iterdir() if p.suffix.lower() in {".wav", ".flac"}])
    if not audio_files:
        raise SystemExit(f"No audio files found in {local_dir}")
    meta_path = output_dir / "metadata.jsonl"
    with meta_path.open("w", encoding="utf-8") as meta_file:
        for idx, audio_path in enumerate(audio_files[:num_samples]):
            target_name = f"sample_{idx:02d}{audio_path.suffix.lower()}"
            target_path = output_dir / target_name
            target_path.write_bytes(audio_path.read_bytes())
            meta_file.write(json.dumps({"audio": target_name, "text": ""}) + "\n")


def build_from_hf(output_dir: Path, dataset_name: str, split: str, num_samples: int) -> None:
    try:
        from datasets import load_dataset
    except Exception as exc:  # noqa: BLE001
        raise SystemExit("datasets is required. Install with: pip install datasets") from exc

    dataset = load_dataset(dataset_name, split=split, streaming=True)
    meta_path = output_dir / "metadata.jsonl"
    with meta_path.open("w", encoding="utf-8") as meta_file:
        for idx, item in enumerate(dataset):
            if idx >= num_samples:
                break
            audio = item["audio"]["array"]
            sr = item["audio"]["sampling_rate"]
            text = item.get("text", "")
            target_name = f"sample_{idx:02d}.wav"
            target_path = output_dir / target_name
            sf.write(target_path, audio, sr)
            meta_file.write(json.dumps({"audio": target_name, "text": text}) + "\n")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.local_dir:
        build_from_local(output_dir, Path(args.local_dir), args.num_samples)
    elif args.hf_dataset:
        build_from_hf(output_dir, args.hf_dataset, args.hf_split, args.num_samples)
    else:
        raise SystemExit("Provide either --local_dir or --hf_dataset")

    print(f"Audio pack written to {output_dir}")


if __name__ == "__main__":
    main()
