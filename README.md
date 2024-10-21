# F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching

[![python](https://img.shields.io/badge/Python-3.10-brightgreen)](https://github.com/SWivid/F5-TTS)
[![arXiv](https://img.shields.io/badge/arXiv-2410.06885-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2410.06885)
[![demo](https://img.shields.io/badge/GitHub-Demo%20page-orange.svg)](https://swivid.github.io/F5-TTS/)
[![hfspace](https://img.shields.io/badge/🤗-Space%20demo-yellow)](https://huggingface.co/spaces/mrfakename/E2-F5-TTS)
[![msspace](https://img.shields.io/badge/🤖-Space%20demo-blue)](https://modelscope.cn/studios/modelscope/E2-F5-TTS)
[![lab](https://img.shields.io/badge/X--LANCE-Lab-grey?labelColor=lightgrey)](https://x-lance.sjtu.edu.cn/)
<img src="https://github.com/user-attachments/assets/12d7749c-071a-427c-81bf-b87b91def670" alt="Watermark" style="width: 40px; height: auto">

**F5-TTS**: Diffusion Transformer with ConvNeXt V2, faster trained and inference.

**E2 TTS**: Flat-UNet Transformer, closest reproduction from [paper](https://arxiv.org/abs/2406.18009).

**Sway Sampling**: Inference-time flow step sampling strategy, greatly improves performance

### Thanks to all the contributors !

## Installation

Clone the repository:

```bash
git clone https://github.com/SWivid/F5-TTS.git
cd F5-TTS
```

Install torch with your CUDA version, e.g. :

```bash
pip install torch==2.3.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip install torchaudio==2.3.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
```

Install other packages:

```bash
pip install -r requirements.txt
```

**[Optional]**: We provide [Dockerfile](https://github.com/SWivid/F5-TTS/blob/main/Dockerfile) and you can use the following command to build it.
```bash
docker build -t f5tts:v1 .
```

### Development

When making a pull request, please use pre-commit to ensure code quality:

```bash
pip install pre-commit
pre-commit install
```

This will run linters and formatters automatically before each commit.

Manually run using: 

```bash
pre-commit run --all-files
```

Note: Some model components have linting exceptions for E722 to accommodate tensor notation


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

## Training & Finetuning

Once your datasets are prepared, you can start the training process.

```bash
# setup accelerate config, e.g. use multi-gpu ddp, fp16
# will be to: ~/.cache/huggingface/accelerate/default_config.yaml     
accelerate config
accelerate launch train.py
```
An initial guidance on Finetuning [#57](https://github.com/SWivid/F5-TTS/discussions/57).

Gradio UI finetuning with `finetune_gradio.py` see [#143](https://github.com/SWivid/F5-TTS/discussions/143).

### Wandb Logging

By default, the training script does NOT use logging (assuming you didn't manually log in using `wandb login`).

To turn on wandb logging, you can either:

1. Manually login with `wandb login`: Learn more [here](https://docs.wandb.ai/ref/cli/wandb-login)
2. Automatically login programmatically by setting an environment variable: Get an API KEY at https://wandb.ai/site/ and set the environment variable as follows:

On Mac & Linux:

```
export WANDB_API_KEY=<YOUR WANDB API KEY>
```

On Windows:

```
set WANDB_API_KEY=<YOUR WANDB API KEY>
```
Moreover, if you couldn't access Wandb and want to log metrics offline, you can the environment variable as follows:

```
export WANDB_MODE=offline
```

## Inference

The pretrained model checkpoints can be reached at [🤗 Hugging Face](https://huggingface.co/SWivid/F5-TTS) and [🤖 Model Scope](https://www.modelscope.cn/models/SWivid/F5-TTS_Emilia-ZH-EN), or automatically downloaded with `inference-cli` and `gradio_app`.

Currently support 30s for a single generation, which is the **TOTAL** length of prompt audio and the generated. Batch inference with chunks is supported by `inference-cli` and `gradio_app`. 
- To avoid possible inference failures, make sure you have seen through the following instructions.
- A longer prompt audio allows shorter generated output. The part longer than 30s cannot be generated properly. Consider using a prompt audio <15s.
- Uppercased letters will be uttered letter by letter, so use lowercased letters for normal words. 
- Add some spaces (blank: " ") or punctuations (e.g. "," ".") to explicitly introduce some pauses. If first few words skipped in code-switched generation (cuz different speed with different languages), this might help.

### CLI Inference

Either you can specify everything in `inference-cli.toml` or override with flags. Leave `--ref_text ""` will have ASR model transcribe the reference audio automatically (use extra GPU memory). If encounter network error, consider use local ckpt, just set `ckpt_path` in `inference-cli.py`

for change model use `--ckpt_file` to specify the model you want to load,  
for change vocab.txt use `--vocab_file` to provide your vocab.txt file.

```bash
python inference-cli.py \
--model "F5-TTS" \
--ref_audio "tests/ref_audio/test_en_1_ref_short.wav" \
--ref_text "Some call me nature, others call me mother nature." \
--gen_text "I don't really care what you call me. I've been a silent spectator, watching species evolve, empires rise and fall. But always remember, I am mighty and enduring. Respect me and I'll nurture you; ignore me and you shall face the consequences."

python inference-cli.py \
--model "E2-TTS" \
--ref_audio "tests/ref_audio/test_zh_1_ref_short.wav" \
--ref_text "对，这就是我，万人敬仰的太乙真人。" \
--gen_text "突然，身边一阵笑声。我看着他们，意气风发地挺直了胸膛，甩了甩那稍显肉感的双臂，轻笑道，我身上的肉，是为了掩饰我爆棚的魅力，否则，岂不吓坏了你们呢？"

# Multi voice
python inference-cli.py -c samples/story.toml
```

### Gradio App
Currently supported features:
- Chunk inference
- Podcast Generation
- Multiple Speech-Type Generation

You can launch a Gradio app (web interface) to launch a GUI for inference (will load ckpt from Huggingface, you may set `ckpt_path` to local file in `gradio_app.py`). Currently load ASR model, F5-TTS and E2 TTS all in once, thus use more GPU memory than `inference-cli`.

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

### Speech Editing

To test speech editing capabilities, use the following command.

```bash
python speech_edit.py
```

## Evaluation

### Prepare Test Datasets

1. Seed-TTS test set: Download from [seed-tts-eval](https://github.com/BytedanceSpeech/seed-tts-eval).
2. LibriSpeech test-clean: Download from [OpenSLR](http://www.openslr.org/12/).
3. Unzip the downloaded datasets and place them in the data/ directory.
4. Update the path for the test-clean data in `scripts/eval_infer_batch.py`
5. Our filtered LibriSpeech-PC 4-10s subset is already under data/ in this repo

### Batch Inference for Test Set

To run batch inference for evaluations, execute the following commands:

```bash
# batch inference for evaluations
accelerate config  # if not set before
bash scripts/eval_infer_batch.sh
```

### Download Evaluation Model Checkpoints

1. Chinese ASR Model: [Paraformer-zh](https://huggingface.co/funasr/paraformer-zh)
2. English ASR Model: [Faster-Whisper](https://huggingface.co/Systran/faster-whisper-large-v3)
3. WavLM Model: Download from [Google Drive](https://drive.google.com/file/d/1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP/view).

### Objective Evaluation

Install packages for evaluation:

```bash
pip install -r requirements_eval.txt
```

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
- [FunASR](https://github.com/modelscope/FunASR), [faster-whisper](https://github.com/SYSTRAN/faster-whisper), [UniSpeech](https://github.com/microsoft/UniSpeech) for evaluation tools
- [ctc-forced-aligner](https://github.com/MahmoudAshraf97/ctc-forced-aligner) for speech edit test
- [mrfakename](https://x.com/realmrfakename) huggingface space demo ~
- [f5-tts-mlx](https://github.com/lucasnewman/f5-tts-mlx/tree/main) Implementation of F5-TTS, with the MLX framework.

## Citation
If our work and codebase is useful for you, please cite as:
```
@article{chen-etal-2024-f5tts,
      title={F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching}, 
      author={Yushen Chen and Zhikang Niu and Ziyang Ma and Keqi Deng and Chunhui Wang and Jian Zhao and Kai Yu and Xie Chen},
      journal={arXiv preprint arXiv:2410.06885},
      year={2024},
}
```
## License

Our code is released under MIT License. The pre-trained models are licensed under the CC-BY-NC license due to the training data Emilia, which is an in-the-wild dataset. Sorry for any inconvenience this may cause.
