
# F5-TTS
### <a href="https://swivid.github.io/F5-TTS/">Demo</a>
Official code for "A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching"

## Installation

```bash
pip install -r requirements.txt
```

## Dataset

```bash
# prepare custom dataset up to your need
# download corresponding dataset first, and fill in the path in scripts
python scripts/prepare_emilia.py
python scripts/prepare_wenetspeech4tts.py
```

## Training

```bash
# setup accelerate config, e.g. use multi-gpu ddp, fp16
# will be to: ~/.cache/huggingface/accelerate/default_config.yaml     
accelerate config
accelerate launch test_train.py
```

## Inference
Pretrained model ckpts. https://huggingface.co/SWivid/F5-TTS
```bash
# single test inference
# modify the config up to your need,
# e.g. fix_duration (the total length of prompt + to_generate, currently support up to 30s)
#      nfe_step     (larger takes more time to do more precise inference ode)
#      ode_method   (switch to 'midpoint' for better compatibility with small nfe_step, )
#                   ( though 'midpoint' is 2nd-order ode solver, slower compared to 1st-order 'Euler')
python test_infer_single.py
```


## Evaluation

download seedtts testset. https://github.com/BytedanceSpeech/seed-tts-eval \
download test-clean. http://www.openslr.org/12/ \
uzip and place under data/, and fill in the path of test-clean in `test_infer_batch.py` \
our librispeech-pc 4-10s subset is already under data/ in this repo

zh asr model ckpt. https://huggingface.co/funasr/paraformer-zh \
en asr model ckpt. https://huggingface.co/Systran/faster-whisper-large-v3 \
wavlm model ckpt. https://drive.google.com/file/d/1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP/view \
fill in the path of ckpts in `test_infer_batch.py`
```bash
# batch inference for evaluations
accelerate config  # if not set before
bash test_infer_batch.sh
```

faster-whisper if cuda11,     
`pip install --force-reinstall ctranslate2==3.24.0`  
(recommended) `pip install faster-whisper==0.10.1`,     
otherwise may encounter asr failure (output abnormal repetition)
```bash
# evaluation for Seed-TTS test set
python scripts/eval_seedtts_testset.py

# evaluation for LibriSpeech-PC test-clean cross sentence
python scripts/eval_librispeech_test_clean.py
```

## Appreciation

- <a href="https://arxiv.org/abs/2406.18009">E2-TTS</a> brilliant work, simple and effective
- <a href="https://arxiv.org/abs/2407.05361">Emilia</a>, <a href="https://arxiv.org/abs/2406.05763">WenetSpeech4TTS</a> valuable datasets
- <a href="https://github.com/lucidrains/e2-tts-pytorch">lucidrains</a> initial CFM structure</a> with also <a href="https://github.com/bfs18">bfs18</a> for discussion</a>
- <a href="https://arxiv.org/abs/2403.03206">SD3</a> & <a href="https://github.com/huggingface/diffusers">Huggingface diffusers</a> DiT and MMDiT code structure
- <a href="https://github.com/modelscope/FunASR">FunASR</a>, <a href="https://github.com/SYSTRAN/faster-whisper">faster-whisper</a> & <a href="https://github.com/microsoft/UniSpeech">UniSpeech</a> for evaluation tools
- <a href="https://github.com/rtqichen/torchdiffeq">torchdiffeq</a> as ODE solver, <a href="https://huggingface.co/charactr/vocos-mel-24khz">Vocos</a> as vocoder
