#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
LOGGER="${LOGGER:-wandb}" # wandb | trackio | null

DATASET_NAME="${DATASET_NAME:-librispeech_asr}"
HF_DATASET_ID="${HF_DATASET_ID:-openslr/librispeech_asr}"
HF_CONFIG="${HF_CONFIG:-clean}"
HF_SPLIT="${HF_SPLIT:-train.100}"
NUM_SAMPLES="${NUM_SAMPLES:-512}"

TOKENIZER_PATH="${TOKENIZER_PATH:-$PWD/data/Emilia_ZH_EN_pinyin/vocab.txt}"
WARMUP_DIR="${WARMUP_DIR:-ckpts/rl_warmup}"
GRPO_DIR="${GRPO_DIR:-ckpts/rl_grpo}"
BASE_CKPT_URL="${BASE_CKPT_URL:-hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors}"

NUM_WORKERS="${NUM_WORKERS:-2}"
BATCH_SIZE="${BATCH_SIZE:-1}"
USE_BNB="${USE_BNB:-true}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-1}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-1}"

RL_STEPS="${RL_STEPS:-30}"
PROMPT_FRAC_RANGE="${PROMPT_FRAC_RANGE:-[0.1,0.3]}"
PROMPT_LENGTH_MODE="${PROMPT_LENGTH_MODE:-min}"
CFG_STRENGTH="${CFG_STRENGTH:-2.0}"
KL_WEIGHT="${KL_WEIGHT:-1.0}"
WER_MODE="${WER_MODE:-char}"
REF_SOURCE="${REF_SOURCE:-audio}"

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
export ACCELERATE_MIXED_PRECISION="${ACCELERATE_MIXED_PRECISION:-bf16}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

if [[ "${LOGGER}" == "trackio" ]]; then
  export GRADIO_SERVER_PORT="${GRADIO_SERVER_PORT:-7860}"
fi

dataset_root="data/${DATASET_NAME}_custom"
if [[ ! -f "${dataset_root}/duration.json" ]]; then
  "${PYTHON_BIN}" - <<PY
from datasets import load_dataset, Dataset
from pathlib import Path
import json
import soundfile as sf

out_root = Path("${dataset_root}")
wav_dir = out_root / "wavs"
wav_dir.mkdir(parents=True, exist_ok=True)

ds = load_dataset("${HF_DATASET_ID}", "${HF_CONFIG}", split="${HF_SPLIT}", streaming=True)
items = []
durations = []
for idx, row in enumerate(ds):
    if idx >= int("${NUM_SAMPLES}"):
        break
    audio = row["audio"]["array"]
    sr = row["audio"]["sampling_rate"]
    text = row.get("text", "")
    wav_path = wav_dir / f"sample_{idx:05d}.wav"
    sf.write(wav_path, audio, sr)
    duration = len(audio) / sr
    items.append({"audio_path": str(wav_path), "text": text, "duration": duration})
    durations.append(duration)

Dataset.from_list(items).save_to_disk(str(out_root / "raw"))
with (out_root / "duration.json").open("w", encoding="utf-8") as f:
    json.dump({"duration": durations}, f)
print(f"Wrote {len(items)} samples to {out_root}")
PY
fi

mkdir -p "${WARMUP_DIR}"
if [[ ! -f "${WARMUP_DIR}/model_last.pt" ]]; then
  "${PYTHON_BIN}" - <<PY
from cached_path import cached_path
import torch

try:
    from safetensors.torch import load_file
except Exception as exc:  # noqa: BLE001
    raise SystemExit("safetensors is required to convert the base checkpoint. Install with: pip install safetensors") from exc

ckpt = cached_path("${BASE_CKPT_URL}")
state = load_file(ckpt)
torch.save({"ema_model_state_dict": state}, "${WARMUP_DIR}/model_last.pt")
print("Prepared warmup checkpoint:", "${WARMUP_DIR}/model_last.pt")
PY
fi

"${PYTHON_BIN}" -m f5_tts.scripts.fetch_reward_asr_model \
  --cache_dir checkpoints/funasr/SenseVoiceSmall
"${PYTHON_BIN}" -m f5_tts.scripts.fetch_reward_spk_model \
  --cache_dir checkpoints/wespeaker/cnceleb_resnet34

"${PYTHON_BIN}" -m f5_tts.train.train -cn F5TTS_v1_Base \
  datasets.name="${DATASET_NAME}" datasets.batch_size_per_gpu="${BATCH_SIZE}" datasets.batch_size_type=sample datasets.num_workers="${NUM_WORKERS}" \
  model.tokenizer=custom model.tokenizer_path="${TOKENIZER_PATH}" \
  model.output_dist=gaussian model.objective=gaussian_nll model.sample_from_dist=false model.use_rl_head=true \
  model.arch.checkpoint_activations=true \
  optim.epochs="${STAGE1_EPOCHS}" optim.learning_rate=1e-5 optim.num_warmup_updates=0 optim.grad_accumulation_steps=1 optim.bnb_optimizer="${USE_BNB}" \
  ckpts.save_dir="${WARMUP_DIR}" ckpts.save_per_updates=50 ckpts.keep_last_n_checkpoints=0 ckpts.log_samples=false ckpts.logger="${LOGGER}"

mkdir -p "${GRPO_DIR}"
cp -f "${WARMUP_DIR}/model_last.pt" "${GRPO_DIR}/model_last.pt"

"${PYTHON_BIN}" -m f5_tts.train.train_rl \
  datasets.name="${DATASET_NAME}" datasets.batch_size_per_gpu="${BATCH_SIZE}" datasets.batch_size_type=sample datasets.num_workers="${NUM_WORKERS}" \
  model.tokenizer=custom model.tokenizer_path="${TOKENIZER_PATH}" \
  model.output_dist=gaussian model.objective=grpo model.use_rl_head=true model.arch.checkpoint_activations=true \
  optim.epochs="${STAGE2_EPOCHS}" optim.learning_rate=1e-6 optim.num_warmup_updates=0 optim.grad_accumulation_steps=1 optim.bnb_optimizer="${USE_BNB}" \
  ckpts.save_dir="${GRPO_DIR}" ckpts.save_per_updates=50 ckpts.keep_last_n_checkpoints=0 ckpts.log_samples=false ckpts.logger="${LOGGER}" \
  rl.steps="${RL_STEPS}" rl.repeat_count=1 rl.mini_repeat_count=1 rl.prompt_frac_range="${PROMPT_FRAC_RANGE}" rl.prompt_length_mode="${PROMPT_LENGTH_MODE}" \
  rl.cfg_strength="${CFG_STRENGTH}" rl.sway_sampling_coef=null rl.kl_weight="${KL_WEIGHT}" \
  rl.ref_model_ckpt="${WARMUP_DIR}/model_last.pt" \
  rl.rewards.providers.0.config.model_dir="$PWD/checkpoints/wespeaker/cnceleb_resnet34/cnceleb_resnet34" \
  rl.rewards.providers.0.config.device=cpu \
  rl.rewards.providers.1.config.model_id="$PWD/checkpoints/funasr/SenseVoiceSmall" \
  rl.rewards.providers.1.config.device=cpu \
  rl.rewards.providers.1.config.wer_mode="${WER_MODE}" \
  rl.rewards.providers.1.config.ref_source="${REF_SOURCE}"
