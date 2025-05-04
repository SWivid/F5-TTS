# generate audio text map for WenetSpeech4TTS
# evaluate for vocab size

import os
import sys


sys.path.append(os.getcwd())

import json
from concurrent.futures import ProcessPoolExecutor
from importlib.resources import files

import torchaudio
from datasets import Dataset
from tqdm import tqdm

from f5_tts.model.utils import convert_char_to_pinyin


def deal_with_sub_path_files(dataset_path, sub_path):
    print(f"Dealing with: {sub_path}")

    text_dir = os.path.join(dataset_path, sub_path, "txts")
    audio_dir = os.path.join(dataset_path, sub_path, "wavs")
    text_files = os.listdir(text_dir)

    audio_paths, texts, durations = [], [], []
    for text_file in tqdm(text_files):
        with open(os.path.join(text_dir, text_file), "r", encoding="utf-8") as file:
            first_line = file.readline().split("\t")
        audio_nm = first_line[0]
        audio_path = os.path.join(audio_dir, audio_nm + ".wav")
        text = first_line[1].strip()

        audio_paths.append(audio_path)

        if tokenizer == "pinyin":
            texts.extend(convert_char_to_pinyin([text], polyphone=polyphone))
        elif tokenizer == "char":
            texts.append(text)

        audio, sample_rate = torchaudio.load(audio_path)
        durations.append(audio.shape[-1] / sample_rate)

    return audio_paths, texts, durations


def main():
    assert tokenizer in ["pinyin", "char"]

    audio_path_list, text_list, duration_list = [], [], []

    executor = ProcessPoolExecutor(max_workers=max_workers)
    futures = []
    for dataset_path in dataset_paths:
        sub_items = os.listdir(dataset_path)
        sub_paths = [item for item in sub_items if os.path.isdir(os.path.join(dataset_path, item))]
        for sub_path in sub_paths:
            futures.append(executor.submit(deal_with_sub_path_files, dataset_path, sub_path))
    for future in tqdm(futures, total=len(futures)):
        audio_paths, texts, durations = future.result()
        audio_path_list.extend(audio_paths)
        text_list.extend(texts)
        duration_list.extend(durations)
    executor.shutdown()

    if not os.path.exists("data"):
        os.makedirs("data")

    print(f"\nSaving to {save_dir} ...")
    dataset = Dataset.from_dict({"audio_path": audio_path_list, "text": text_list, "duration": duration_list})
    dataset.save_to_disk(f"{save_dir}/raw", max_shard_size="2GB")  # arrow format

    with open(f"{save_dir}/duration.json", "w", encoding="utf-8") as f:
        json.dump(
            {"duration": duration_list}, f, ensure_ascii=False
        )  # dup a json separately saving duration in case for DynamicBatchSampler ease

    print("\nEvaluating vocab size (all characters and symbols / all phonemes) ...")
    text_vocab_set = set()
    for text in tqdm(text_list):
        text_vocab_set.update(list(text))

    # add alphabets and symbols (optional, if plan to ft on de/fr etc.)
    if tokenizer == "pinyin":
        text_vocab_set.update([chr(i) for i in range(32, 127)] + [chr(i) for i in range(192, 256)])

    with open(f"{save_dir}/vocab.txt", "w") as f:
        for vocab in sorted(text_vocab_set):
            f.write(vocab + "\n")
    print(f"\nFor {dataset_name}, sample count: {len(text_list)}")
    print(f"For {dataset_name}, vocab size is: {len(text_vocab_set)}\n")


if __name__ == "__main__":
    max_workers = 32

    tokenizer = "pinyin"  # "pinyin" | "char"
    polyphone = True
    dataset_choice = 1  # 1: Premium, 2: Standard, 3: Basic

    dataset_name = (
        ["WenetSpeech4TTS_Premium", "WenetSpeech4TTS_Standard", "WenetSpeech4TTS_Basic"][dataset_choice - 1]
        + "_"
        + tokenizer
    )
    dataset_paths = [
        "<SOME_PATH>/WenetSpeech4TTS/Basic",
        "<SOME_PATH>/WenetSpeech4TTS/Standard",
        "<SOME_PATH>/WenetSpeech4TTS/Premium",
    ][-dataset_choice:]
    save_dir = str(files("f5_tts").joinpath("../../")) + f"/data/{dataset_name}"
    print(f"\nChoose Dataset: {dataset_name}, will save to {save_dir}\n")

    main()

    # Results (if adding alphabets with accents and symbols):
    # WenetSpeech4TTS       Basic     Standard     Premium
    # samples count       3932473      1941220      407494
    # pinyin vocab size      1349         1348        1344   (no polyphone)
    #                           -            -        1459   (polyphone)
    # char   vocab size      5264         5219        5042

    # vocab size may be slightly different due to jieba tokenizer and pypinyin (e.g. way of polyphoneme)
    # please be careful if using pretrained model, make sure the vocab.txt is same
