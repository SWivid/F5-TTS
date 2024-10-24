import os
import sys

sys.path.append(os.getcwd())

import argparse
import csv
import json
import shutil
from importlib.resources import files
from pathlib import Path

import torchaudio
from tqdm import tqdm
from datasets.arrow_writer import ArrowWriter

from f5_tts.model.utils import (
    convert_char_to_pinyin,
)


PRETRAINED_VOCAB_PATH = files("f5_tts").joinpath("../../data/Emilia_ZH_EN_pinyin/vocab.txt")


def is_csv_wavs_format(input_dataset_dir):
    fpath = Path(input_dataset_dir)
    metadata = fpath / "metadata.csv"
    wavs = fpath / "wavs"
    return metadata.exists() and metadata.is_file() and wavs.exists() and wavs.is_dir()


def prepare_csv_wavs_dir(input_dir):
    assert is_csv_wavs_format(input_dir), f"not csv_wavs format: {input_dir}"
    input_dir = Path(input_dir)
    metadata_path = input_dir / "metadata.csv"
    audio_path_text_pairs = read_audio_text_pairs(metadata_path.as_posix())

    sub_result, durations = [], []
    vocab_set = set()
    polyphone = True
    for audio_path, text in audio_path_text_pairs:
        if not Path(audio_path).exists():
            print(f"audio {audio_path} not found, skipping")
            continue
        audio_duration = get_audio_duration(audio_path)
        # assume tokenizer = "pinyin"  ("pinyin" | "char")
        text = convert_char_to_pinyin([text], polyphone=polyphone)[0]
        sub_result.append({"audio_path": audio_path, "text": text, "duration": audio_duration})
        durations.append(audio_duration)
        vocab_set.update(list(text))

    return sub_result, durations, vocab_set


def get_audio_duration(audio_path):
    audio, sample_rate = torchaudio.load(audio_path)
    num_channels = audio.shape[0]
    return audio.shape[1] / (sample_rate * num_channels)


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
    # save preprocessed dataset to disk
    out_dir.mkdir(exist_ok=True, parents=True)
    print(f"\nSaving to {out_dir} ...")

    # dataset = Dataset.from_dict({"audio_path": audio_path_list, "text": text_list, "duration": duration_list})  # oom
    # dataset.save_to_disk(f"{out_dir}/raw", max_shard_size="2GB")
    raw_arrow_path = out_dir / "raw.arrow"
    with ArrowWriter(path=raw_arrow_path.as_posix(), writer_batch_size=1) as writer:
        for line in tqdm(result, desc="Writing to raw.arrow ..."):
            writer.write(line)

    # dup a json separately saving duration in case for DynamicBatchSampler ease
    dur_json_path = out_dir / "duration.json"
    with open(dur_json_path.as_posix(), "w", encoding="utf-8") as f:
        json.dump({"duration": duration_list}, f, ensure_ascii=False)

    # vocab map, i.e. tokenizer
    # add alphabets and symbols (optional, if plan to ft on de/fr etc.)
    # if tokenizer == "pinyin":
    #     text_vocab_set.update([chr(i) for i in range(32, 127)] + [chr(i) for i in range(192, 256)])
    voca_out_path = out_dir / "vocab.txt"
    with open(voca_out_path.as_posix(), "w") as f:
        for vocab in sorted(text_vocab_set):
            f.write(vocab + "\n")

    if is_finetune:
        file_vocab_finetune = PRETRAINED_VOCAB_PATH.as_posix()
        shutil.copy2(file_vocab_finetune, voca_out_path)
    else:
        with open(voca_out_path, "w") as f:
            for vocab in sorted(text_vocab_set):
                f.write(vocab + "\n")

    dataset_name = out_dir.stem
    print(f"\nFor {dataset_name}, sample count: {len(result)}")
    print(f"For {dataset_name}, vocab size is: {len(text_vocab_set)}")
    print(f"For {dataset_name}, total {sum(duration_list)/3600:.2f} hours")


def prepare_and_save_set(inp_dir, out_dir, is_finetune: bool = True):
    if is_finetune:
        assert PRETRAINED_VOCAB_PATH.exists(), f"pretrained vocab.txt not found: {PRETRAINED_VOCAB_PATH}"
    sub_result, durations, vocab_set = prepare_csv_wavs_dir(inp_dir)
    save_prepped_dataset(out_dir, sub_result, durations, vocab_set, is_finetune)


def cli():
    # finetune: python scripts/prepare_csv_wavs.py /path/to/input_dir /path/to/output_dir_pinyin
    # pretrain: python scripts/prepare_csv_wavs.py /path/to/output_dir_pinyin --pretrain
    parser = argparse.ArgumentParser(description="Prepare and save dataset.")
    parser.add_argument("inp_dir", type=str, help="Input directory containing the data.")
    parser.add_argument("out_dir", type=str, help="Output directory to save the prepared data.")
    parser.add_argument("--pretrain", action="store_true", help="Enable for new pretrain, otherwise is a fine-tune")

    args = parser.parse_args()

    prepare_and_save_set(args.inp_dir, args.out_dir, is_finetune=not args.pretrain)


if __name__ == "__main__":
    cli()
