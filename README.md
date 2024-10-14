# F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching

[![arXiv](https://img.shields.io/badge/arXiv-2410.06885-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2410.06885)
[![demo](https://img.shields.io/badge/GitHub-Demo%20page-blue.svg)](https://swivid.github.io/F5-TTS/)
[![space](https://img.shields.io/badge/🤗-Space%20demo-yellow)](https://huggingface.co/spaces/mrfakename/E2-F5-TTS)

**F5-TTS**: Diffusion Transformer with ConvNeXt V2, faster trained and inference.

**E2 TTS**: Flat-UNet Transformer, closest reproduction.

**Sway Sampling**: Inference-time flow step sampling strategy, greatly improves performance

## Installation

Clone the repository:

```bash
git clone https://github.com/SWivid/F5-TTS.git
cd F5-TTS
```

Install packages:

```bash
pip install -r requirements.txt
```

Install torch with your CUDA version, e.g. :

```bash
pip install torch==2.3.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip install torchaudio==2.3.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
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
An initial guidance on Finetuning #57.

## Inference

To run inference with pretrained models, download the checkpoints from [🤗 Hugging Face](https://huggingface.co/SWivid/F5-TTS).

Currently support up to 30s generation, which is the **TOTAL** length of prompt audio and the generated. Batch inference with chunks is supported by Gradio APP now. 
- To avoid inference failure, make sure you have seen through following instructions.
- Uppercased letters will be uttered letter by letter, so use lowercased letter for normal words. 
- Add some spaces (blank: " ") or punctuations (e.g. "," ".") to explicitly introduce some pauses.

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
### Speech Editing

To test speech editing capabilities, use the following command.

```bash
python test_infer_single_edit.py
```

### Gradio App

You can launch a Gradio app (web interface) to launch a GUI for inference.

First, make sure you have the dependencies installed (`pip install -r requirements.txt`). Then, install the Gradio app dependencies:

```bash
pip install -r requirements_gradio.txt
```

After installing the dependencies, launch the app (will load ckpt from Huggingface, you may set `ckpt_path` to local file in `gradio_app.py`):

```bash
python gradio_app.py
```

You can specify the port/host:

```bash
python gradio_app.py --port 7860 --host 0.0.0.0
```

Or launch a share link:

```bash
python gradio_app.py --share
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

**Some Notes**

For faster-whisper with CUDA 11:

```bash
pip install --force-reinstall ctranslate2==3.24.0
```

(Recommended) To avoid possible ASR failures, such as abnormal repetitions in output:

```bash
pip install faster-whisper==0.10.1
```

Update the path with your batch-inferenced results, and carry out WER / SIM evaluations:
```bash
# Evaluation for Seed-TTS test set
python scripts/eval_seedtts_testset.py

# Evaluation for LibriSpeech-PC test-clean (cross-sentence)
python scripts/eval_librispeech_test_clean.py
```

## Acknowledgements

- [E2-TTS](https://arxiv.org/abs/2406.18009) brilliant work, simple and effective
- [Emilia](https://arxiv.org/abs/2407.05361), [WenetSpeech4TTS](https://arxiv.org/abs/2406.05763) valuable datasets
- [lucidrains](https://github.com/lucidrains) initial CFM structure with also [bfs18](https://github.com/bfs18) for discussion
- [SD3](https://arxiv.org/abs/2403.03206) & [Hugging Face diffusers](https://github.com/huggingface/diffusers) DiT and MMDiT code structure
- [torchdiffeq](https://github.com/rtqichen/torchdiffeq) as ODE solver, [Vocos](https://huggingface.co/charactr/vocos-mel-24khz) as vocoder
- [mrfakename](https://x.com/realmrfakename) huggingface space demo ~
- [FunASR](https://github.com/modelscope/FunASR), [faster-whisper](https://github.com/SYSTRAN/faster-whisper), [UniSpeech](https://github.com/microsoft/UniSpeech) for evaluation tools
- [ctc-forced-aligner](https://github.com/MahmoudAshraf97/ctc-forced-aligner) for speech edit test

## Citation
```
@article{chen-etal-2024-f5tts,
      title={F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching}, 
      author={Yushen Chen and Zhikang Niu and Ziyang Ma and Keqi Deng and Chunhui Wang and Jian Zhao and Kai Yu and Xie Chen},
      journal={arXiv preprint arXiv:2410.06885},
      year={2024},
}
```
## License

Our code is released under MIT License.