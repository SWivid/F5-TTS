# RL Two-Stage Training (Warmup + GRPO)

This guide covers the RL workflow for F5-TTS:
- Stage 1: Gaussian NLL warmup (probabilistic head pretrain)
- Stage 2: GRPO fine-tune with reward models

It includes dataset layout, reward model setup, and minimal launch commands.

## Requirements

- Use your uv venv (examples use `./.venv/bin/python`; adjust if your venv lives elsewhere).
- Install RL extras:
  ```bash
  ./.venv/bin/python -m pip install -e ".[rl]"
  ```
- Optional trackers:
  ```bash
  ./.venv/bin/python -m pip install -e ".[trackio]"
  ```
- Reward models:
  - FunASR SenseVoiceSmall
  - WeSpeaker cnceleb_resnet34 (fbank frontend)

## Dataset layout

`train.py` uses `data/<dataset_name>_<tokenizer>` for `CustomDataset`:
- `data/<dataset>_<tokenizer>/raw` (HF dataset saved via `save_to_disk`)
- `data/<dataset>_<tokenizer>/duration.json`

### Option A: quick dummy dataset (HF internal)

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

### Option B: small LibriSpeech subset (streaming)

```bash
./.venv/bin/python - <<'PY'
from datasets import load_dataset, Dataset
from pathlib import Path
import json
import soundfile as sf

out_root = Path("data/mini_rl_custom")
wav_dir = out_root / "wavs"
wav_dir.mkdir(parents=True, exist_ok=True)

ds = load_dataset("librispeech_asr", "clean", split="train.100", streaming=True)
items = []
durations = []
for idx, row in enumerate(ds):
    if idx >= 64:
        break
    audio = row["audio"]["array"]
    sr = row["audio"]["sampling_rate"]
    text = row.get("text", "")
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

## Reward model assets

FunASR:
```bash
./.venv/bin/python -m f5_tts.scripts.fetch_reward_asr_model \
  --cache_dir checkpoints/funasr/SenseVoiceSmall
```

WeSpeaker (HF archive; fbank frontend):
```bash
./.venv/bin/python -m f5_tts.scripts.fetch_reward_spk_model \
  --cache_dir checkpoints/wespeaker/cnceleb_resnet34
```

WeSpeaker `model_dir` should point to:
```
checkpoints/wespeaker/cnceleb_resnet34/cnceleb_resnet34
```

## Stage 1: Warmup (gaussian_nll)

Download a pretrained base checkpoint and place it in the warmup dir:

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

Warmup command:

```bash
CUDA_VISIBLE_DEVICES=0 \
./.venv/bin/python -m f5_tts.train.train -cn F5TTS_v1_Base \
datasets.name=mini_rl datasets.batch_size_per_gpu=2 datasets.batch_size_type=sample datasets.num_workers=2 \
model.tokenizer=custom model.tokenizer_path=$PWD/data/Emilia_ZH_EN_pinyin/vocab.txt \
model.output_dist=gaussian model.objective=gaussian_nll model.sample_from_dist=false model.use_rl_head=true \
model.arch.checkpoint_activations=false \
optim.epochs=1 optim.learning_rate=1e-5 optim.num_warmup_updates=0 optim.grad_accumulation_steps=1 optim.bnb_optimizer=false \
ckpts.save_dir=ckpts/mini_rl_warmup ckpts.save_per_updates=1000 ckpts.keep_last_n_checkpoints=0 ckpts.log_samples=false ckpts.logger=null
```

Checkpoint behavior:
- If no `model_*.pt` exists, the trainer loads `pretrained_*.safetensors` from the save dir.
- It writes `model_last.pt` and `model_<update>.pt`.

## Stage 2: GRPO

Copy the warmup checkpoint into the GRPO directory:

```bash
mkdir -p ckpts/mini_rl_grpo
cp ckpts/mini_rl_warmup/model_last.pt ckpts/mini_rl_grpo/model_last.pt
```

GRPO command:

```bash
CUDA_VISIBLE_DEVICES=0 ACCELERATE_MIXED_PRECISION=bf16 PYTORCH_ALLOC_CONF=expandable_segments:True \
./.venv/bin/python -m f5_tts.train.train_rl \
datasets.name=mini_rl datasets.batch_size_per_gpu=1 datasets.batch_size_type=sample datasets.num_workers=2 \
model.tokenizer=custom model.tokenizer_path=$PWD/data/Emilia_ZH_EN_pinyin/vocab.txt \
model.output_dist=gaussian model.objective=grpo model.use_rl_head=true model.arch.checkpoint_activations=false \
optim.epochs=1 optim.learning_rate=1e-6 optim.num_warmup_updates=0 optim.grad_accumulation_steps=1 optim.bnb_optimizer=true \
ckpts.save_dir=ckpts/mini_rl_grpo ckpts.save_per_updates=1000 ckpts.keep_last_n_checkpoints=0 ckpts.log_samples=false ckpts.logger=wandb \
rl.steps=30 rl.repeat_count=1 rl.mini_repeat_count=1 rl.prompt_frac_range='[0.1,0.3]' rl.prompt_length_mode=min \
rl.cfg_strength=2.0 rl.sway_sampling_coef=-1.0 rl.kl_weight=1.0 \
rl.ref_model_ckpt=$PWD/ckpts/mini_rl_warmup/model_last.pt \
rl.rewards.providers.0.config.model_dir=$PWD/checkpoints/wespeaker/cnceleb_resnet34/cnceleb_resnet34 \
rl.rewards.providers.0.config.device=cpu \
rl.rewards.providers.1.config.model_id=$PWD/checkpoints/funasr/SenseVoiceSmall \
rl.rewards.providers.1.config.device=cpu
```

Sample logging:
- Set `ckpts.log_samples=true` to save `update_*_gen.wav` / `update_*_ref.wav` under `ckpts/.../samples`
  at each `ckpts.save_per_updates` interval.

## RL knobs (quick reference)

- `rl.steps`: number of diffusion/inference steps per GRPO rollout. Higher values improve
  audio quality and reward signal (ASR/WER), but increase compute and memory.
- `rl.steps_plus_one`: opt-in to use `steps + 1` integration points in `forward_rl`. Default is `false`
  for F5R parity; set `true` if you want RL rollouts to match non-RL step count.
- `rl.prompt_length_mode`: `min` (F5R parity), `per_sample`, or `range`. `range` uses the sampled
  fraction directly so prompt length respects the lower bound in `prompt_frac_range`.
- `wer_mode`: `char | word` (default: `char`, matching F5R).
- `ref_source`: `text | audio` (default: `text`; set `audio` to match ASR-vs-ASR reward in F5R).

## Logging

W&B logs include:
- `loss`, `loss/kl`, `loss/pro_adv`
- `reward/mean`, `reward/std`, `reward/min`, `reward/max`
- `reward/speaker_similarity/cosine`
- `reward/asr/char_error_rate` or `reward/asr/word_error_rate` (depends on `wer_mode`)

Trackio (drop-in alternative):
```bash
./.venv/bin/python -m pip install -e ".[trackio]"
```
Then set `ckpts.logger=trackio` and view logs locally with:
```bash
trackio show
```

## Implementation parity notes

These details intentionally match the F5R reference code:
- Gaussian loss adds `t^2 * ln_sig` to regularize variance over time.
- GRPO uses Gaussian density weighting (not log-prob) for advantage shaping.
- ODE integration randomly skips gradients on some steps for speed.

## Troubleshooting

- `num_workers=0` is supported; `persistent_workers` is only enabled when `num_workers > 0`.
- If WeSpeaker fails to import, install the GitHub source and use an fbank model.
- If FunASR is missing, install `.[reward_funasr]` or `funasr==1.3.0`.
- If WER is flat, increase `rl.steps` (very low values often produce poor audio).
- On low disk, keep only the final checkpoint: `ckpts.keep_last_n_checkpoints=0`.
