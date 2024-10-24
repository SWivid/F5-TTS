# Evaluate with Seed-TTS testset

import sys
import os

sys.path.append(os.getcwd())

import multiprocessing as mp
from importlib.resources import files

import numpy as np

from f5_tts.eval.utils_eval import (
    get_seed_tts_test,
    run_asr_wer,
    run_sim,
)

rel_path = str(files("f5_tts").joinpath("../../"))


eval_task = "wer"  # sim | wer
lang = "zh"  # zh | en
metalst = rel_path + f"/data/seedtts_testset/{lang}/meta.lst"  # seed-tts testset
# gen_wav_dir = rel_path + f"/data/seedtts_testset/{lang}/wavs"  # ground truth wavs
gen_wav_dir = "PATH_TO_GENERATED"  # generated wavs


# NOTE. paraformer-zh result will be slightly different according to the number of gpus, cuz batchsize is different
#       zh 1.254 seems a result of 4 workers wer_seed_tts
gpus = [0, 1, 2, 3, 4, 5, 6, 7]
test_set = get_seed_tts_test(metalst, gen_wav_dir, gpus)

local = False
if local:  # use local custom checkpoint dir
    if lang == "zh":
        asr_ckpt_dir = "../checkpoints/funasr"  # paraformer-zh dir under funasr
    elif lang == "en":
        asr_ckpt_dir = "../checkpoints/Systran/faster-whisper-large-v3"
else:
    asr_ckpt_dir = ""  # auto download to cache dir

wavlm_ckpt_dir = "../checkpoints/UniSpeech/wavlm_large_finetune.pth"


# --------------------------- WER ---------------------------

if eval_task == "wer":
    wers = []

    with mp.Pool(processes=len(gpus)) as pool:
        args = [(rank, lang, sub_test_set, asr_ckpt_dir) for (rank, sub_test_set) in test_set]
        results = pool.map(run_asr_wer, args)
        for wers_ in results:
            wers.extend(wers_)

    wer = round(np.mean(wers) * 100, 3)
    print(f"\nTotal {len(wers)} samples")
    print(f"WER      : {wer}%")


# --------------------------- SIM ---------------------------

if eval_task == "sim":
    sim_list = []

    with mp.Pool(processes=len(gpus)) as pool:
        args = [(rank, sub_test_set, wavlm_ckpt_dir) for (rank, sub_test_set) in test_set]
        results = pool.map(run_sim, args)
        for sim_ in results:
            sim_list.extend(sim_)

    sim = round(sum(sim_list) / len(sim_list), 3)
    print(f"\nTotal {len(sim_list)} samples")
    print(f"SIM      : {sim}")
