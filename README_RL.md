# RL Two-Stage Guide (Warmup + GRPO)

This guide documents the two-stage RL workflow:
1) Stage 1 warmup (gaussian_nll pretrain)
2) Stage 2 GRPO fine-tune with rewards

It includes checkpoint naming/handling, minimal commands, and W&B logging tips.

## Requirements

- Use the uv venv: `/home/mithex/work/tts/.venv/bin/python`
- W&B online logging (set `WANDB_DISABLE_SERVICE=1` if sockets are blocked)
- Reward models:
  - FunASR SenseVoiceSmall
  - WeSpeaker cnceleb_resnet34
- Install RL extras (uses the GitHub WeSpeaker source):
  ```bash
  ./.venv/bin/python -m pip install -e ".[rl]"
  ```
- If you prefer explicit installs:
  ```bash
  ./.venv/bin/python -m pip install "wespeaker @ git+https://github.com/wenet-e2e/wespeaker.git" funasr
  ./.venv/bin/python -m pip install huggingface_hub
  ```
- WeSpeaker reward loading is fbank-only; if your WeSpeaker config uses other frontends,
  install the needed dependencies (e.g., `s3prl`, `whisper`, `peft`) or switch to an fbank model.

## Dataset layout

`train.py` uses `data/<dataset_name>_<tokenizer>` for `CustomDataset`:
- `data/<dataset>_<tokenizer>/raw` (HF dataset saved via `save_to_disk`)
- `data/<dataset>_<tokenizer>/duration.json`

If you want a tiny smoke-test dataset:

```bash
./.venv/bin/python - <<'PY'
from datasets import load_dataset, Dataset
from pathlib import Path
import soundfile as sf
import json

out_root = Path("data/mini_rl_custom")
wav_dir = out_root / "wavs"
wav_dir.mkdir(parents=True, exist_ok=True)

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", split="validation")
text_key = "text" if "text" in ds.column_names else ds.column_names[-1]

items = []
durations = []
for idx, row in enumerate(ds):
    if idx >= 8:
        break
    audio = row["audio"]["array"]
    sr = row["audio"]["sampling_rate"]
    text = row[text_key]
    wav_path = wav_dir / f"sample_{idx:02d}.wav"
    sf.write(wav_path, audio, sr)
    duration = len(audio) / sr
    items.append({"audio_path": str(wav_path), "text": text, "duration": duration})
    durations.append(duration)

Dataset.from_list(items).save_to_disk(str(out_root / "raw"))
with (out_root / "duration.json").open("w", encoding="utf-8") as f:
    json.dump({"duration": durations}, f)
PY
```

Use `datasets.name=mini_rl` with `model.tokenizer=custom` so the loader reads
`data/mini_rl_custom`.

## Reward models (Stage 2)

FunASR:
```bash
./.venv/bin/python -m f5_tts.scripts.fetch_reward_asr_model \
  --cache_dir checkpoints/funasr/SenseVoiceSmall
```

WeSpeaker (HF archive; fallback supported):
```bash
./.venv/bin/python -m f5_tts.scripts.fetch_reward_spk_model \
  --cache_dir checkpoints/wespeaker/cnceleb_resnet34
```

Set WeSpeaker model_dir to:
```
checkpoints/wespeaker/cnceleb_resnet34/cnceleb_resnet34
```

## Stage 1: Warmup (gaussian_nll)

Checkpoint naming/handling:
- Trainer loads `pretrained_*.pt` or `pretrained_*.safetensors` in the save dir
  if no `model_*.pt` exists.
- It writes `model_last.pt` and `model_<update>.pt`.

Download a pretrained base checkpoint and place it in the warmup save dir:
```bash
./.venv/bin/python - <<'PY'
from cached_path import cached_path
from pathlib import Path
import shutil

ckpt = cached_path("hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors")
out_dir = Path("ckpts/mini_rl_warmup")
out_dir.mkdir(parents=True, exist_ok=True)
dst = out_dir / ("pretrained_" + Path(ckpt).name)
if not dst.exists():
    shutil.copy2(ckpt, dst)
print(dst)
PY
```

Warmup command (8GB GPU friendly):
```bash
WANDB_MODE=online WANDB_DISABLE_SERVICE=1 \
WANDB_DIR=$PWD/.wandb WANDB_CACHE_DIR=$PWD/.wandb_cache WANDB_CONFIG_DIR=$PWD/.wandb_config \
CUDA_VISIBLE_DEVICES=0 ACCELERATE_MIXED_PRECISION=bf16 PYTORCH_ALLOC_CONF=expandable_segments:True \
./.venv/bin/python -m f5_tts.train.train -cn F5TTS_v1_Base \
datasets.name=mini_rl datasets.batch_size_per_gpu=1 datasets.batch_size_type=sample datasets.num_workers=1 \
model.tokenizer=custom model.tokenizer_path=$PWD/data/Emilia_ZH_EN_pinyin/vocab.txt \
model.output_dist=gaussian model.objective=gaussian_nll model.sample_from_dist=false model.use_rl_head=true \
model.arch.checkpoint_activations=true \
optim.epochs=1 optim.learning_rate=1e-5 optim.num_warmup_updates=0 optim.grad_accumulation_steps=1 \
optim.bnb_optimizer=true \
ckpts.save_dir=ckpts/mini_rl_warmup ckpts.log_samples=false
```

Output: `ckpts/mini_rl_warmup/model_last.pt`

## Stage 2: GRPO

Checkpoint naming/handling:
- GRPOTrainer only loads `model_last.pt` / `model_*.pt` from its save dir.
- It does not load `pretrained_*` by default.

To start from the warmup model, copy it into the GRPO save dir:
```bash
mkdir -p ckpts/mini_rl_grpo
cp ckpts/mini_rl_warmup/model_last.pt ckpts/mini_rl_grpo/model_last.pt
```

GRPO command (bnb8bit supported):
```bash
WANDB_MODE=online WANDB_DISABLE_SERVICE=1 \
WANDB_DIR=$PWD/.wandb WANDB_CACHE_DIR=$PWD/.wandb_cache WANDB_CONFIG_DIR=$PWD/.wandb_config \
CUDA_VISIBLE_DEVICES=0 ACCELERATE_MIXED_PRECISION=bf16 PYTORCH_ALLOC_CONF=expandable_segments:True \
./.venv/bin/python -m f5_tts.train.train_rl \
datasets.name=mini_rl datasets.batch_size_per_gpu=1 datasets.batch_size_type=sample datasets.num_workers=1 \
model.tokenizer=custom model.tokenizer_path=$PWD/data/Emilia_ZH_EN_pinyin/vocab.txt \
model.output_dist=gaussian model.objective=grpo model.use_rl_head=true model.arch.checkpoint_activations=true \
optim.epochs=1 optim.learning_rate=1e-6 optim.num_warmup_updates=0 optim.grad_accumulation_steps=1 \
optim.bnb_optimizer=true \
ckpts.save_dir=ckpts/mini_rl_grpo ckpts.log_samples=false \
rl.steps=2 rl.repeat_count=1 rl.mini_repeat_count=1 rl.prompt_frac_range='[0.1,0.1]' \
rl.prompt_length_mode=min \
rl.cfg_strength=1.0 rl.sway_sampling_coef=null rl.kl_weight=1.0 \
rl.rewards.providers.0.config.model_dir=checkpoints/wespeaker/cnceleb_resnet34/cnceleb_resnet34 \
rl.rewards.providers.1.config.model_id=$PWD/checkpoints/funasr/SenseVoiceSmall
```

Output: `ckpts/mini_rl_grpo/model_last.pt`

Prompt length modes:
- `min` (default): keep batch prompt length equal to the minimum sampled value (matches F5R behavior).
- `per_sample`: keep per-sample prompt lengths; prompts are padded and `lens` is passed to `forward_rl`.

## W&B logging

GRPO now logs useful metrics:
- `loss`, `loss/kl`, `loss/pro_adv`
- `reward/mean`, `reward/std`, `reward/min`, `reward/max`
- `reward/<component>` per provider (e.g., `reward/wespeaker_sim.sim`, `reward/funasr_wer.acc`)

To visualize:
```bash
wandb login
wandb sync .wandb/wandb
```

Or use the run URLs printed in the console output.

## Troubleshooting

- `num_workers=0` is now safe; the trainer only enables `persistent_workers` when `num_workers > 0`.
- If WeSpeaker errors during import, install via the GitHub source and use an fbank-based model.
- If FunASR is missing, install `.[reward_funasr]` or `funasr` directly.
- If checkpoint saves fail on a full disk, point `ckpts.save_dir` to a larger volume.
