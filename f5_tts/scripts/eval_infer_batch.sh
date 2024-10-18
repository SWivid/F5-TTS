#!/bin/bash

# e.g. F5-TTS, 16 NFE
accelerate launch scripts/eval_infer_batch.py -n "F5TTS_Base" -t "seedtts_test_zh" -nfe 16
accelerate launch scripts/eval_infer_batch.py -n "F5TTS_Base" -t "seedtts_test_en" -nfe 16
accelerate launch scripts/eval_infer_batch.py -n "F5TTS_Base" -t "ls_pc_test_clean" -nfe 16

# e.g. Vanilla E2 TTS, 32 NFE
accelerate launch scripts/eval_infer_batch.py -n "E2TTS_Base" -t "seedtts_test_zh" -o "midpoint" -ss 0
accelerate launch scripts/eval_infer_batch.py -n "E2TTS_Base" -t "seedtts_test_en" -o "midpoint" -ss 0
accelerate launch scripts/eval_infer_batch.py -n "E2TTS_Base" -t "ls_pc_test_clean" -o "midpoint" -ss 0

# etc.
