import concurrent.futures
import multiprocessing
import os
import shutil
import signal
import subprocess  # For invoking ffprobe
import sys
from contextlib import contextmanager


sys.path.append(os.getcwd())

import argparse
import csv
import json
from importlib.resources import files
from pathlib import Path

import torchaudio
from datasets.arrow_writer import ArrowWriter
from tqdm import tqdm

from f5_tts.model.utils import convert_char_to_pinyin


PRETRAINED_VOCAB_PATH = files("f5_tts").joinpath("../../data/Emilia_ZH_EN_pinyin/vocab.txt")


def is_csv_wavs_format(input_dataset_dir):
    fpath = Path(input_dataset_dir)
    metadata = fpath / "metadata.csv"
    wavs = fpath / "wavs"
    return metadata.exists() and metadata.is_file() and wavs.exists() and wavs.is_dir()


# Configuration constants
BATCH_SIZE = 100  # Batch size for text conversion
MAX_WORKERS = max(1, multiprocessing.cpu_count() - 1)  # Leave one CPU free
THREAD_NAME_PREFIX = "AudioProcessor"
CHUNK_SIZE = 100  # Number of files to process per worker batch

executor = None  # Global executor for cleanup


@contextmanager
def graceful_exit():
    """Context manager for graceful shutdown on signals"""

    def signal_handler(signum, frame):
        print("\nReceived signal to terminate. Cleaning up...")
        if executor is not None:
            print("Shutting down executor...")
            executor.shutdown(wait=False, cancel_futures=True)
        sys.exit(1)

    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        yield
    finally:
        if executor is not None:
            executor.shutdown(wait=False)


def process_audio_file(audio_path, text, polyphone):
    """Process a single audio file by checking its existence and extracting duration."""
    if not Path(audio_path).exists():
        print(f"audio {audio_path} not found, skipping")
        return None
    try:
        audio_duration = get_audio_duration(audio_path)
        if audio_duration <= 0:
            raise ValueError(f"Duration {audio_duration} is non-positive.")
        return (audio_path, text, audio_duration)
    except Exception as e:
        print(f"Warning: Failed to process {audio_path} due to error: {e}. Skipping corrupt file.")
        return None


def batch_convert_texts(texts, polyphone, batch_size=BATCH_SIZE):
    """Convert a list of texts to pinyin in batches."""
    converted_texts = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        converted_batch = convert_char_to_pinyin(batch, polyphone=polyphone)
        converted_texts.extend(converted_batch)
    return converted_texts


def prepare_csv_wavs_dir(input_dir, num_workers=None):
    global executor
    assert is_csv_wavs_format(input_dir), f"not csv_wavs format: {input_dir}"
    input_dir = Path(input_dir)
    metadata_path = input_dir / "metadata.csv"
    audio_path_text_pairs = read_audio_text_pairs(metadata_path.as_posix())

    polyphone = True
    total_files = len(audio_path_text_pairs)

    # Use provided worker count or calculate optimal number
    worker_count = num_workers if num_workers is not None else min(MAX_WORKERS, total_files)
    print(f"\nProcessing {total_files} audio files using {worker_count} workers...")

    with graceful_exit():
        # Initialize thread pool with optimized settings
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=worker_count, thread_name_prefix=THREAD_NAME_PREFIX
        ) as exec:
            executor = exec
            results = []

            # Process files in chunks for better efficiency
            for i in range(0, len(audio_path_text_pairs), CHUNK_SIZE):
                chunk = audio_path_text_pairs[i : i + CHUNK_SIZE]
                # Submit futures in order
                chunk_futures = [executor.submit(process_audio_file, pair[0], pair[1], polyphone) for pair in chunk]

                # Iterate over futures in the original submission order to preserve ordering
                for future in tqdm(
                    chunk_futures,
                    total=len(chunk),
                    desc=f"Processing chunk {i // CHUNK_SIZE + 1}/{(total_files + CHUNK_SIZE - 1) // CHUNK_SIZE}",
                ):
                    try:
                        result = future.result()
                        if result is not None:
                            results.append(result)
                    except Exception as e:
                        print(f"Error processing file: {e}")

            executor = None

    # Filter out failed results
    processed = [res for res in results if res is not None]
    if not processed:
        raise RuntimeError("No valid audio files were processed!")

    # Batch process text conversion
    raw_texts = [item[1] for item in processed]
    converted_texts = batch_convert_texts(raw_texts, polyphone, batch_size=BATCH_SIZE)

    # Prepare final results
    sub_result = []
    durations = []
    vocab_set = set()

    for (audio_path, _, duration), conv_text in zip(processed, converted_texts):
        sub_result.append({"audio_path": audio_path, "text": conv_text, "duration": duration})
        durations.append(duration)
        vocab_set.update(list(conv_text))

    return sub_result, durations, vocab_set


def get_audio_duration(audio_path, timeout=5):
    """
    Get the duration of an audio file in seconds using ffmpeg's ffprobe.
    Falls back to torchaudio.load() if ffprobe fails.
    """
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            audio_path,
        ]
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True, timeout=timeout
        )
        duration_str = result.stdout.strip()
        if duration_str:
            return float(duration_str)
        raise ValueError("Empty duration string from ffprobe.")
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, ValueError) as e:
        print(f"Warning: ffprobe failed for {audio_path} with error: {e}. Falling back to torchaudio.")
        try:
            audio, sample_rate = torchaudio.load(audio_path)
            return audio.shape[1] / sample_rate
        except Exception as e:
            raise RuntimeError(f"Both ffprobe and torchaudio failed for {audio_path}: {e}")


def read_audio_text_pairs(csv_file_path):
    audio_text_pairs = []

    parent = Path(csv_file_path).parent
    with open(csv_file_path, mode="r", newline="", encoding="utf-8-sig") as csvfile:
        reader = csv.reader(csvfile, delimiter="|")
        next(reader)  # Skip the header row
        for row in reader:
            if len(row) >= 2:
                audio_file = row[0].strip()  # First column: audio file path
                text = row[1].strip()  # Second column: text
                audio_file_path = parent / audio_file
                audio_text_pairs.append((audio_file_path.as_posix(), text))

    return audio_text_pairs


def save_prepped_dataset(out_dir, result, duration_list, text_vocab_set, is_finetune):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    print(f"\nSaving to {out_dir} ...")

    # Save dataset with improved batch size for better I/O performance
    raw_arrow_path = out_dir / "raw.arrow"
    with ArrowWriter(path=raw_arrow_path.as_posix(), writer_batch_size=100) as writer:
        for line in tqdm(result, desc="Writing to raw.arrow ..."):
            writer.write(line)

    # Save durations to JSON
    dur_json_path = out_dir / "duration.json"
    with open(dur_json_path.as_posix(), "w", encoding="utf-8") as f:
        json.dump({"duration": duration_list}, f, ensure_ascii=False)

    # Handle vocab file - write only once based on finetune flag
    voca_out_path = out_dir / "vocab.txt"
    if is_finetune:
        file_vocab_finetune = PRETRAINED_VOCAB_PATH.as_posix()
        shutil.copy2(file_vocab_finetune, voca_out_path)
    else:
        with open(voca_out_path.as_posix(), "w") as f:
            for vocab in sorted(text_vocab_set):
                f.write(vocab + "\n")

    dataset_name = out_dir.stem
    print(f"\nFor {dataset_name}, sample count: {len(result)}")
    print(f"For {dataset_name}, vocab size is: {len(text_vocab_set)}")
    print(f"For {dataset_name}, total {sum(duration_list) / 3600:.2f} hours")


def prepare_and_save_set(inp_dir, out_dir, is_finetune: bool = True, num_workers: int = None):
    if is_finetune:
        assert PRETRAINED_VOCAB_PATH.exists(), f"pretrained vocab.txt not found: {PRETRAINED_VOCAB_PATH}"
    sub_result, durations, vocab_set = prepare_csv_wavs_dir(inp_dir, num_workers=num_workers)
    save_prepped_dataset(out_dir, sub_result, durations, vocab_set, is_finetune)


def cli():
    try:
        # Before processing, check if ffprobe is available.
        if shutil.which("ffprobe") is None:
            print(
                "Warning: ffprobe is not available. Duration extraction will rely on torchaudio (which may be slower)."
            )

        # Usage examples in help text
        parser = argparse.ArgumentParser(
            description="Prepare and save dataset.",
            epilog="""
Examples:
    # For fine-tuning (default):
    python prepare_csv_wavs.py /input/dataset/path /output/dataset/path
    
    # For pre-training:
    python prepare_csv_wavs.py /input/dataset/path /output/dataset/path --pretrain
    
    # With custom worker count:
    python prepare_csv_wavs.py /input/dataset/path /output/dataset/path --workers 4
            """,
        )
        parser.add_argument("inp_dir", type=str, help="Input directory containing the data.")
        parser.add_argument("out_dir", type=str, help="Output directory to save the prepared data.")
        parser.add_argument("--pretrain", action="store_true", help="Enable for new pretrain, otherwise is a fine-tune")
        parser.add_argument("--workers", type=int, help=f"Number of worker threads (default: {MAX_WORKERS})")
        args = parser.parse_args()

        prepare_and_save_set(args.inp_dir, args.out_dir, is_finetune=not args.pretrain, num_workers=args.workers)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user. Cleaning up...")
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
