
# Evaluation

Install packages for evaluation:

```bash
pip install -e .[eval]
```

## Generating Samples for Evaluation

### Prepare Test Datasets

1. *Seed-TTS testset*: Download from [seed-tts-eval](https://github.com/BytedanceSpeech/seed-tts-eval).
2. *LibriSpeech test-clean*: Download from [OpenSLR](http://www.openslr.org/12/).
3. Unzip the downloaded datasets and place them in the `data/` directory.
4. Our filtered LibriSpeech-PC 4-10s subset: `data/librispeech_pc_test_clean_cross_sentence.lst`

### Batch Inference for Test Set

To run batch inference for evaluations, execute the following commands:

```bash
# if not setup accelerate config yet
accelerate config

# if only perform inference
bash src/f5_tts/eval/eval_infer_batch.sh --infer-only

# if inference and with corresponding evaluation, setup the following tools first
bash src/f5_tts/eval/eval_infer_batch.sh
```

## Objective Evaluation on Generated Results

### Download Evaluation Model Checkpoints

1. Chinese ASR Model: [Paraformer-zh](https://huggingface.co/funasr/paraformer-zh)
2. English ASR Model: [Faster-Whisper](https://huggingface.co/Systran/faster-whisper-large-v3)
3. WavLM Model: Download from [Google Drive](https://drive.google.com/file/d/1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP/view).

> [!NOTE]  
> ASR model will be automatically downloaded if `--local` not set for evaluation scripts.  
> Otherwise, you should update the `asr_ckpt_dir` path values in `eval_librispeech_test_clean.py` or `eval_seedtts_testset.py`.
> 
> WavLM model must be downloaded and your `wavlm_ckpt_dir` path updated in `eval_librispeech_test_clean.py` and `eval_seedtts_testset.py`.

### Objective Evaluation Examples

Update the path with your batch-inferenced results, and carry out WER / SIM / UTMOS evaluations:
```bash
# Evaluation [WER] for Seed-TTS test [ZH] set
python src/f5_tts/eval/eval_seedtts_testset.py --eval_task wer --lang zh --gen_wav_dir <GEN_WAV_DIR> --gpu_nums 8

# Evaluation [SIM] for LibriSpeech-PC test-clean (cross-sentence)
python src/f5_tts/eval/eval_librispeech_test_clean.py --eval_task sim --gen_wav_dir <GEN_WAV_DIR> --librispeech_test_clean_path <TEST_CLEAN_PATH>

# Evaluation [UTMOS]. --ext: Audio extension
python src/f5_tts/eval/eval_utmos.py --audio_dir <WAV_DIR> --ext wav
```

> [!NOTE]  
> Evaluation results can also be found in `_*_results.jsonl` files saved in `<GEN_WAV_DIR>`/`<WAV_DIR>`.
