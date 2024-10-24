# Training

## Prepare Dataset

Example data processing scripts for Emilia and Wenetspeech4TTS, and you may tailor your own one along with a Dataset class in `src/f5_tts/model/dataset.py`.

### 1. Datasets used for pretrained models
Download corresponding dataset first, and fill in the path in scripts.

```bash
# Prepare the Emilia dataset
python src/f5_tts/train/datasets/prepare_emilia.py

# Prepare the Wenetspeech4TTS dataset
python src/f5_tts/train/datasets/prepare_wenetspeech4tts.py
```

### 2. Create custom dataset with metadata.csv
Use guidance see [#57 here](https://github.com/SWivid/F5-TTS/discussions/57#discussioncomment-10959029).

```bash
python src/f5_tts/train/datasets/prepare_csv_wavs.py
```

## Training & Finetuning

Once your datasets are prepared, you can start the training process.

### 1. Training script used for pretrained model

```bash
# setup accelerate config, e.g. use multi-gpu ddp, fp16
# will be to: ~/.cache/huggingface/accelerate/default_config.yaml     
accelerate config
accelerate launch src/f5_tts/train/train.py
```

### 2. Finetuning practice
Discussion board for Finetuning [#57](https://github.com/SWivid/F5-TTS/discussions/57).

Gradio UI training/finetuning with `src/f5_tts/train/finetune_gradio.py` see [#143](https://github.com/SWivid/F5-TTS/discussions/143).

### 3. Wandb Logging

The `wandb/` dir will be created under path you run training/finetuning scripts.

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
