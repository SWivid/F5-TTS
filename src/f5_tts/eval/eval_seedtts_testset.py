# Evaluate with Seed-TTS testset

import argparse
import json
import os
import sys


sys.path.append(os.getcwd())

import multiprocessing as mp
from importlib.resources import files

import numpy as np

from f5_tts.eval.utils_eval import get_seed_tts_test, run_asr_wer, run_sim


rel_path = str(files("f5_tts").joinpath("../../"))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--eval_task", type=str, default="wer", choices=["sim", "wer"])
    parser.add_argument("-l", "--lang", type=str, default="en", choices=["zh", "en"])
    parser.add_argument("-g", "--gen_wav_dir", type=str, required=True)
    parser.add_argument("-n", "--gpu_nums", type=int, default=8, help="Number of GPUs to use")
    parser.add_argument("--local", action="store_true", help="Use local custom checkpoint directory")
    return parser.parse_args()


def main():
    args = get_args()
    eval_task = args.eval_task
    lang = args.lang
    gen_wav_dir = args.gen_wav_dir
    metalst = rel_path + f"/data/seedtts_testset/{lang}/meta.lst"  # seed-tts testset

    # NOTE. paraformer-zh result will be slightly different according to the number of gpus, cuz batchsize is different
    #       zh 1.254 seems a result of 4 workers wer_seed_tts
    gpus = list(range(args.gpu_nums))
    test_set = get_seed_tts_test(metalst, gen_wav_dir, gpus)

    local = args.local
    if local:  # use local custom checkpoint dir
        if lang == "zh":
            asr_ckpt_dir = "../checkpoints/funasr"  # paraformer-zh dir under funasr
        elif lang == "en":
            asr_ckpt_dir = "../checkpoints/Systran/faster-whisper-large-v3"
    else:
        asr_ckpt_dir = ""  # auto download to cache dir
    wavlm_ckpt_dir = "../checkpoints/UniSpeech/wavlm_large_finetune.pth"

    # --------------------------------------------------------------------------

    full_results = []
    metrics = []

    if eval_task == "wer":
        with mp.Pool(processes=len(gpus)) as pool:
            args = [(rank, lang, sub_test_set, asr_ckpt_dir) for (rank, sub_test_set) in test_set]
            results = pool.map(run_asr_wer, args)
            for r in results:
                full_results.extend(r)
    elif eval_task == "sim":
        with mp.Pool(processes=len(gpus)) as pool:
            args = [(rank, sub_test_set, wavlm_ckpt_dir) for (rank, sub_test_set) in test_set]
            results = pool.map(run_sim, args)
            for r in results:
                full_results.extend(r)
    else:
        raise ValueError(f"Unknown metric type: {eval_task}")

    result_path = f"{gen_wav_dir}/_{eval_task}_results.jsonl"
    with open(result_path, "w") as f:
        for line in full_results:
            metrics.append(line[eval_task])
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
        metric = round(np.mean(metrics), 5)
        f.write(f"\n{eval_task.upper()}: {metric}\n")

    print(f"\nTotal {len(metrics)} samples")
    print(f"{eval_task.upper()}: {metric}")
    print(f"{eval_task.upper()} results saved to {result_path}")


if __name__ == "__main__":
    main()
