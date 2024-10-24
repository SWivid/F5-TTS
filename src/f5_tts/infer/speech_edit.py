import os

import torch
import torch.nn.functional as F
import torchaudio
from vocos import Vocos

from f5_tts.model import CFM, UNetT, DiT
from f5_tts.model.utils import (
    get_tokenizer,
    convert_char_to_pinyin,
)
from f5_tts.infer.utils_infer import (
    load_checkpoint,
    save_spectrogram,
)

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


# --------------------- Dataset Settings -------------------- #

target_sample_rate = 24000
n_mel_channels = 100
hop_length = 256
target_rms = 0.1

tokenizer = "pinyin"
dataset_name = "Emilia_ZH_EN"


# ---------------------- infer setting ---------------------- #

seed = None  # int | None

exp_name = "F5TTS_Base"  # F5TTS_Base | E2TTS_Base
ckpt_step = 1200000

nfe_step = 32  # 16, 32
cfg_strength = 2.0
ode_method = "euler"  # euler | midpoint
sway_sampling_coef = -1.0
speed = 1.0

if exp_name == "F5TTS_Base":
    model_cls = DiT
    model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)

elif exp_name == "E2TTS_Base":
    model_cls = UNetT
    model_cfg = dict(dim=1024, depth=24, heads=16, ff_mult=4)

ckpt_path = f"ckpts/{exp_name}/model_{ckpt_step}.safetensors"
output_dir = "tests"

# [leverage https://github.com/MahmoudAshraf97/ctc-forced-aligner to get char level alignment]
# pip install git+https://github.com/MahmoudAshraf97/ctc-forced-aligner.git
# [write the origin_text into a file, e.g. tests/test_edit.txt]
# ctc-forced-aligner --audio_path "src/f5_tts/infer/examples/basic/basic_ref_en.wav" --text_path "tests/test_edit.txt" --language "zho" --romanize --split_size "char"
# [result will be saved at same path of audio file]
# [--language "zho" for Chinese, "eng" for English]
# [if local ckpt, set --alignment_model "../checkpoints/mms-300m-1130-forced-aligner"]

audio_to_edit = "src/f5_tts/infer/examples/basic/basic_ref_en.wav"
origin_text = "Some call me nature, others call me mother nature."
target_text = "Some call me optimist, others call me realist."
parts_to_edit = [
    [1.42, 2.44],
    [4.04, 4.9],
]  # stard_ends of "nature" & "mother nature", in seconds
fix_duration = [
    1.2,
    1,
]  # fix duration for "optimist" & "realist", in seconds

# audio_to_edit = "src/f5_tts/infer/examples/basic/basic_ref_zh.wav"
# origin_text = "对，这就是我，万人敬仰的太乙真人。"
# target_text = "对，那就是你，万人敬仰的太白金星。"
# parts_to_edit = [[0.84, 1.4], [1.92, 2.4], [4.26, 6.26], ]
# fix_duration = None  # use origin text duration


# -------------------------------------------------#

use_ema = True

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Vocoder model
local = False
if local:
    vocos_local_path = "../checkpoints/charactr/vocos-mel-24khz"
    vocos = Vocos.from_hparams(f"{vocos_local_path}/config.yaml")
    state_dict = torch.load(f"{vocos_local_path}/pytorch_model.bin", weights_only=True, map_location=device)
    vocos.load_state_dict(state_dict)

    vocos.eval()
else:
    vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")

# Tokenizer
vocab_char_map, vocab_size = get_tokenizer(dataset_name, tokenizer)

# Model
model = CFM(
    transformer=model_cls(**model_cfg, text_num_embeds=vocab_size, mel_dim=n_mel_channels),
    mel_spec_kwargs=dict(
        target_sample_rate=target_sample_rate,
        n_mel_channels=n_mel_channels,
        hop_length=hop_length,
    ),
    odeint_kwargs=dict(
        method=ode_method,
    ),
    vocab_char_map=vocab_char_map,
).to(device)

model = load_checkpoint(model, ckpt_path, device, use_ema=use_ema)

# Audio
audio, sr = torchaudio.load(audio_to_edit)
if audio.shape[0] > 1:
    audio = torch.mean(audio, dim=0, keepdim=True)
rms = torch.sqrt(torch.mean(torch.square(audio)))
if rms < target_rms:
    audio = audio * target_rms / rms
if sr != target_sample_rate:
    resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
    audio = resampler(audio)
offset = 0
audio_ = torch.zeros(1, 0)
edit_mask = torch.zeros(1, 0, dtype=torch.bool)
for part in parts_to_edit:
    start, end = part
    part_dur = end - start if fix_duration is None else fix_duration.pop(0)
    part_dur = part_dur * target_sample_rate
    start = start * target_sample_rate
    audio_ = torch.cat((audio_, audio[:, round(offset) : round(start)], torch.zeros(1, round(part_dur))), dim=-1)
    edit_mask = torch.cat(
        (
            edit_mask,
            torch.ones(1, round((start - offset) / hop_length), dtype=torch.bool),
            torch.zeros(1, round(part_dur / hop_length), dtype=torch.bool),
        ),
        dim=-1,
    )
    offset = end * target_sample_rate
# audio = torch.cat((audio_, audio[:, round(offset):]), dim = -1)
edit_mask = F.pad(edit_mask, (0, audio.shape[-1] // hop_length - edit_mask.shape[-1] + 1), value=True)
audio = audio.to(device)
edit_mask = edit_mask.to(device)

# Text
text_list = [target_text]
if tokenizer == "pinyin":
    final_text_list = convert_char_to_pinyin(text_list)
else:
    final_text_list = [text_list]
print(f"text  : {text_list}")
print(f"pinyin: {final_text_list}")

# Duration
ref_audio_len = 0
duration = audio.shape[-1] // hop_length

# Inference
with torch.inference_mode():
    generated, trajectory = model.sample(
        cond=audio,
        text=final_text_list,
        duration=duration,
        steps=nfe_step,
        cfg_strength=cfg_strength,
        sway_sampling_coef=sway_sampling_coef,
        seed=seed,
        edit_mask=edit_mask,
    )
print(f"Generated mel: {generated.shape}")

# Final result
generated = generated.to(torch.float32)
generated = generated[:, ref_audio_len:, :]
generated_mel_spec = generated.permute(0, 2, 1)
generated_wave = vocos.decode(generated_mel_spec.cpu())
if rms < target_rms:
    generated_wave = generated_wave * rms / target_rms

save_spectrogram(generated_mel_spec[0].cpu().numpy(), f"{output_dir}/speech_edit_out.png")
torchaudio.save(f"{output_dir}/speech_edit_out.wav", generated_wave, target_sample_rate)
print(f"Generated wav: {generated_wave.shape}")
