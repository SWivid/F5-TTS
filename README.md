
# F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching

[![arXiv](https://img.shields.io/badge/arXiv-2410.06885-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2410.06885)
[![demo](https://img.shields.io/badge/GitHub-Demo%20page-blue.svg)](https://swivid.github.io/F5-TTS/)
[![space](https://img.shields.io/badge/🤗-Space%20demo-yellow)](https://huggingface.co/spaces/mrfakename/E2-F5-TTS) \
**F5-TTS**: Diffusion Transformer with ConvNeXt V2, faster trained and inference. \
**E2 TTS**: Flat-UNet Transformer, closest reproduction.\
**Sway Sampling**: Inference-time flow step sampling strategy, greatly improves performance

## Installation
Clone this repository.
```bash
git clone git@github.com:SWivid/F5-TTS.git
cd F5-TTS
```
Install packages.
```bash
pip install -r requirements.txt
```

## Prepare Dataset
Example data processing scripts for Emilia and Wenetspeech4TTS, and you may tailor your own one along with a Dataset class in `model/dataset.py`.
```bash
# prepare custom dataset up to your need
# download corresponding dataset first, and fill in the path in scripts

# Prepare the Emilia dataset
python scripts/prepare_emilia.py

# Prepare the Wenetspeech4TTS dataset
python scripts/prepare_wenetspeech4tts.py
```

## Training
Once your datasets are prepared, you can start the training process.
```bash
# setup accelerate config, e.g. use multi-gpu ddp, fp16
# will be to: ~/.cache/huggingface/accelerate/default_config.yaml     
accelerate config
accelerate launch test_train.py
```

## Inference
To inference with pretrained models, download the checkpoints from [🤗](https://huggingface.co/SWivid/F5-TTS).

### Single Inference
You can test single inference using the following command. Before running the command, modify the config up to your need.
```bash
# modify the config up to your need,
# e.g. fix_duration (the total length of prompt + to_generate, currently support up to 30s)
#      nfe_step     (larger takes more time to do more precise inference ode)
#      ode_method   (switch to 'midpoint' for better compatibility with small nfe_step, )
#                   ( though 'midpoint' is 2nd-order ode solver, slower compared to 1st-order 'Euler')
python test_infer_single.py
```
### Speech Edit
To test speech editing capabilities, use the following command.
```
python test_infer_single_edit.py
```

## Evaluation
### Prepare Test Datasets
1. Seed-TTS test set: Download from [seed-tts-eval](https://github.com/BytedanceSpeech/seed-tts-eval).
2. LibriSpeech test-clean: Download from [OpenSLR](http://www.openslr.org/12/).
3. Unzip the downloaded datasets and place them in the data/ directory.
4. Update the path for the test-clean data in `test_infer_batch.py`
5. Our filtered LibriSpeech-PC 4-10s subset is already under data/ in this repo

### Batch Inference for Test Set
To run batch inference for evaluations, execute the following commands:
```bash
# batch inference for evaluations
accelerate config  # if not set before
bash test_infer_batch.sh
```

### Download Evaluation Model Checkpoints
1. Chinese ASR Model: [Paraformer-zh](https://huggingface.co/funasr/paraformer-zh)
2. English ASR Model: [Faster-Whisper](https://huggingface.co/Systran/faster-whisper-large-v3)
3. WavLM Model: Download from [Google Drive](https://drive.google.com/file/d/1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP/view).

### Objective Evaluation
**Some Notes**\
For faster-whisper with CUDA 11: \
`pip install --force-reinstall ctranslate2==3.24.0`\
(Recommended) To avoid possible ASR failures, such as abnormal repetitions in output:\
`pip install faster-whisper==0.10.1`

Update the path with your batch-inferenced results, and carry out WER / SIM evaluations:
```bash
# Evaluation for Seed-TTS test set
python scripts/eval_seedtts_testset.py

# Evaluation for LibriSpeech-PC test-clean (cross-sentence)
python scripts/eval_librispeech_test_clean.py
```

## Acknowledgements

- <a href="https://arxiv.org/abs/2406.18009">E2-TTS</a> brilliant work, simple and effective
- <a href="https://arxiv.org/abs/2407.05361">Emilia</a>, <a href="https://arxiv.org/abs/2406.05763">WenetSpeech4TTS</a> valuable datasets
- <a href="https://github.com/lucidrains/e2-tts-pytorch">lucidrains</a> initial CFM structure</a> with also <a href="https://github.com/bfs18">bfs18</a> for discussion</a>
- <a href="https://arxiv.org/abs/2403.03206">SD3</a> & <a href="https://github.com/huggingface/diffusers">Huggingface diffusers</a> DiT and MMDiT code structure
- <a href="https://github.com/rtqichen/torchdiffeq">torchdiffeq</a> as ODE solver, <a href="https://huggingface.co/charactr/vocos-mel-24khz">Vocos</a> as vocoder
- <a href="https://x.com/realmrfakename">mrfakename</a> huggingface space demo ~
- <a href="https://github.com/modelscope/FunASR">FunASR</a>, <a href="https://github.com/SYSTRAN/faster-whisper">faster-whisper</a> & <a href="https://github.com/microsoft/UniSpeech">UniSpeech</a> for evaluation tools
- <a href="https://github.com/MahmoudAshraf97/ctc-forced-aligner">ctc-forced-aligner</a> for speech edit test

## Citation
```
@article{chen-etal-2024-f5tts,
      title={F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching}, 
      author={Yushen Chen and Zhikang Niu and Ziyang Ma and Keqi Deng and Chunhui Wang and Jian Zhao and Kai Yu and Xie Chen},
      journal={arXiv preprint arXiv:2410.06885},
      year={2024},
}
```
## LICENSE
Our code is released under MIT License.
