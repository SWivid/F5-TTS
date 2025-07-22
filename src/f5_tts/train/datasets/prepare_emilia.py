# Emilia Dataset: https://huggingface.co/datasets/amphion/Emilia-Dataset/tree/fc71e07
# if use updated new version, i.e. WebDataset, feel free to modify / draft your own script

# generate audio text map for Emilia ZH & EN
# evaluate for vocab size

import os
import sys


sys.path.append(os.getcwd())

import json
from concurrent.futures import ProcessPoolExecutor
from importlib.resources import files
from pathlib import Path

from datasets.arrow_writer import ArrowWriter
from tqdm import tqdm

from f5_tts.model.utils import convert_char_to_pinyin, repetition_found


out_zh = {
    "ZH_B00041_S06226",
    "ZH_B00042_S09204",
    "ZH_B00065_S09430",
    "ZH_B00065_S09431",
    "ZH_B00066_S09327",
    "ZH_B00066_S09328",
}
zh_filters = ["い", "て"]
# seems synthesized audios, or heavily code-switched
out_en = {
    "EN_B00013_S00913",
    "EN_B00042_S00120",
    "EN_B00055_S04111",
    "EN_B00061_S00693",
    "EN_B00061_S01494",
    "EN_B00061_S03375",
    "EN_B00059_S00092",
    "EN_B00111_S04300",
    "EN_B00100_S03759",
    "EN_B00087_S03811",
    "EN_B00059_S00950",
    "EN_B00089_S00946",
    "EN_B00078_S05127",
    "EN_B00070_S04089",
    "EN_B00074_S09659",
    "EN_B00061_S06983",
    "EN_B00061_S07060",
    "EN_B00059_S08397",
    "EN_B00082_S06192",
    "EN_B00091_S01238",
    "EN_B00089_S07349",
    "EN_B00070_S04343",
    "EN_B00061_S02400",
    "EN_B00076_S01262",
    "EN_B00068_S06467",
    "EN_B00076_S02943",
    "EN_B00064_S05954",
    "EN_B00061_S05386",
    "EN_B00066_S06544",
    "EN_B00076_S06944",
    "EN_B00072_S08620",
    "EN_B00076_S07135",
    "EN_B00076_S09127",
    "EN_B00065_S00497",
    "EN_B00059_S06227",
    "EN_B00063_S02859",
    "EN_B00075_S01547",
    "EN_B00061_S08286",
    "EN_B00079_S02901",
    "EN_B00092_S03643",
    "EN_B00096_S08653",
    "EN_B00063_S04297",
    "EN_B00063_S04614",
    "EN_B00079_S04698",
    "EN_B00104_S01666",
    "EN_B00061_S09504",
    "EN_B00061_S09694",
    "EN_B00065_S05444",
    "EN_B00063_S06860",
    "EN_B00065_S05725",
    "EN_B00069_S07628",
    "EN_B00083_S03875",
    "EN_B00071_S07665",
    "EN_B00071_S07665",
    "EN_B00062_S04187",
    "EN_B00065_S09873",
    "EN_B00065_S09922",
    "EN_B00084_S02463",
    "EN_B00067_S05066",
    "EN_B00106_S08060",
    "EN_B00073_S06399",
    "EN_B00073_S09236",
    "EN_B00087_S00432",
    "EN_B00085_S05618",
    "EN_B00064_S01262",
    "EN_B00072_S01739",
    "EN_B00059_S03913",
    "EN_B00069_S04036",
    "EN_B00067_S05623",
    "EN_B00060_S05389",
    "EN_B00060_S07290",
    "EN_B00062_S08995",
}
en_filters = ["ا", "い", "て"]


def deal_with_audio_dir(audio_dir):
    audio_jsonl = audio_dir.with_suffix(".jsonl")
    sub_result, durations = [], []
    vocab_set = set()
    bad_case_zh = 0
    bad_case_en = 0
    with open(audio_jsonl, "r") as f:
        lines = f.readlines()
        for line in tqdm(lines, desc=f"{audio_jsonl.stem}"):
            obj = json.loads(line)
            text = obj["text"]
            if obj["language"] == "zh":
                if obj["wav"].split("/")[1] in out_zh or any(f in text for f in zh_filters) or repetition_found(text):
                    bad_case_zh += 1
                    continue
                else:
                    text = text.translate(
                        str.maketrans({",": "，", "!": "！", "?": "？"})
                    )  # not "。" cuz much code-switched
            if obj["language"] == "en":
                if (
                    obj["wav"].split("/")[1] in out_en
                    or any(f in text for f in en_filters)
                    or repetition_found(text, length=4)
                ):
                    bad_case_en += 1
                    continue
            if tokenizer == "pinyin":
                text = convert_char_to_pinyin([text], polyphone=polyphone)[0]
            duration = obj["duration"]
            sub_result.append({"audio_path": str(audio_dir.parent / obj["wav"]), "text": text, "duration": duration})
            durations.append(duration)
            vocab_set.update(list(text))
    return sub_result, durations, vocab_set, bad_case_zh, bad_case_en


def main():
    assert tokenizer in ["pinyin", "char"]
    result = []
    duration_list = []
    text_vocab_set = set()
    total_bad_case_zh = 0
    total_bad_case_en = 0

    # process raw data
    executor = ProcessPoolExecutor(max_workers=max_workers)
    futures = []
    for lang in langs:
        dataset_path = Path(os.path.join(dataset_dir, lang))
        [
            futures.append(executor.submit(deal_with_audio_dir, audio_dir))
            for audio_dir in dataset_path.iterdir()
            if audio_dir.is_dir()
        ]
    for futures in tqdm(futures, total=len(futures)):
        sub_result, durations, vocab_set, bad_case_zh, bad_case_en = futures.result()
        result.extend(sub_result)
        duration_list.extend(durations)
        text_vocab_set.update(vocab_set)
        total_bad_case_zh += bad_case_zh
        total_bad_case_en += bad_case_en
    executor.shutdown()

    # save preprocessed dataset to disk
    if not os.path.exists(f"{save_dir}"):
        os.makedirs(f"{save_dir}")
    print(f"\nSaving to {save_dir} ...")

    # dataset = Dataset.from_dict({"audio_path": audio_path_list, "text": text_list, "duration": duration_list})  # oom
    # dataset.save_to_disk(f"{save_dir}/raw", max_shard_size="2GB")
    with ArrowWriter(path=f"{save_dir}/raw.arrow") as writer:
        for line in tqdm(result, desc="Writing to raw.arrow ..."):
            writer.write(line)
        writer.finalize()

    # dup a json separately saving duration in case for DynamicBatchSampler ease
    with open(f"{save_dir}/duration.json", "w", encoding="utf-8") as f:
        json.dump({"duration": duration_list}, f, ensure_ascii=False)

    # vocab map, i.e. tokenizer
    # add alphabets and symbols (optional, if plan to ft on de/fr etc.)
    # if tokenizer == "pinyin":
    #     text_vocab_set.update([chr(i) for i in range(32, 127)] + [chr(i) for i in range(192, 256)])
    with open(f"{save_dir}/vocab.txt", "w") as f:
        for vocab in sorted(text_vocab_set):
            f.write(vocab + "\n")

    print(f"\nFor {dataset_name}, sample count: {len(result)}")
    print(f"For {dataset_name}, vocab size is: {len(text_vocab_set)}")
    print(f"For {dataset_name}, total {sum(duration_list) / 3600:.2f} hours")
    if "ZH" in langs:
        print(f"Bad zh transcription case: {total_bad_case_zh}")
    if "EN" in langs:
        print(f"Bad en transcription case: {total_bad_case_en}\n")


if __name__ == "__main__":
    max_workers = 32

    tokenizer = "pinyin"  # "pinyin" | "char"
    polyphone = True

    langs = ["ZH", "EN"]
    dataset_dir = "<SOME_PATH>/Emilia_Dataset/raw"
    dataset_name = f"Emilia_{'_'.join(langs)}_{tokenizer}"
    save_dir = str(files("f5_tts").joinpath("../../")) + f"/data/{dataset_name}"
    print(f"\nPrepare for {dataset_name}, will save to {save_dir}\n")

    main()

    # Emilia               ZH & EN
    # samples count       37837916   (after removal)
    # pinyin vocab size       2543   (polyphone)
    # total duration      95281.87   (hours)
    # bad zh asr cnt        230435   (samples)
    # bad eh asr cnt         37217   (samples)

    # vocab size may be slightly different due to jieba tokenizer and pypinyin (e.g. way of polyphoneme)
    # please be careful if using pretrained model, make sure the vocab.txt is same
