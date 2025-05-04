import os


os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # for MPS device compatibility

from importlib.resources import files

import torch
import torch.nn.functional as F
import torchaudio
from cached_path import cached_path
from hydra.utils import get_class
from omegaconf import OmegaConf

from f5_tts.infer.utils_infer import load_checkpoint, load_vocoder, save_spectrogram
from f5_tts.model import CFM
from f5_tts.model.utils import convert_char_to_pinyin, get_tokenizer


device = (
    "cuda"
    if torch.cuda.is_available()
    else "xpu"
    if torch.xpu.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


# ---------------------- infer setting ---------------------- #

seed = None  # int | None

exp_name = "F5TTS_v1_Base"  # F5TTS_v1_Base | E2TTS_Base
ckpt_step = 1250000

nfe_step = 32  # 16, 32
cfg_strength = 2.0
ode_method = "euler"  # euler | midpoint
sway_sampling_coef = -1.0
speed = 1.0
target_rms = 0.1


model_cfg = OmegaConf.load(str(files("f5_tts").joinpath(f"configs/{exp_name}.yaml")))
model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
model_arc = model_cfg.model.arch

dataset_name = model_cfg.datasets.name
tokenizer = model_cfg.model.tokenizer

mel_spec_type = model_cfg.model.mel_spec.mel_spec_type
target_sample_rate = model_cfg.model.mel_spec.target_sample_rate
n_mel_channels = model_cfg.model.mel_spec.n_mel_channels
hop_length = model_cfg.model.mel_spec.hop_length
win_length = model_cfg.model.mel_spec.win_length
n_fft = model_cfg.model.mel_spec.n_fft


# ckpt_path = str(files("f5_tts").joinpath("../../")) + f"/ckpts/{exp_name}/model_{ckpt_step}.safetensors"
ckpt_path = str(cached_path(f"hf://SWivid/F5-TTS/{exp_name}/model_{ckpt_step}.safetensors"))
output_dir = "tests"


# [leverage https://github.com/MahmoudAshraf97/ctc-forced-aligner to get char level alignment]
# pip install git+https://github.com/MahmoudAshraf97/ctc-forced-aligner.git
# [write the origin_text into a file, e.g. tests/test_edit.txt]
# ctc-forced-aligner --audio_path "src/f5_tts/infer/examples/basic/basic_ref_en.wav" --text_path "tests/test_edit.txt" --language "zho" --romanize --split_size "char"
# [result will be saved at same path of audio file]
# [--language "zho" for Chinese, "eng" for English]
# [if local ckpt, set --alignment_model "../checkpoints/mms-300m-1130-forced-aligner"]

audio_to_edit = str(files("f5_tts").joinpath("infer/examples/basic/basic_ref_en.wav"))
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
if mel_spec_type == "vocos":
    vocoder_local_path = "../checkpoints/charactr/vocos-mel-24khz"
elif mel_spec_type == "bigvgan":
    vocoder_local_path = "../checkpoints/bigvgan_v2_24khz_100band_256x"
vocoder = load_vocoder(vocoder_name=mel_spec_type, is_local=local, local_path=vocoder_local_path)

# Tokenizer
vocab_char_map, vocab_size = get_tokenizer(dataset_name, tokenizer)

# Model
model = CFM(
    transformer=model_cls(**model_arc, text_num_embeds=vocab_size, mel_dim=n_mel_channels),
    mel_spec_kwargs=dict(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mel_channels=n_mel_channels,
        target_sample_rate=target_sample_rate,
        mel_spec_type=mel_spec_type,
    ),
    odeint_kwargs=dict(
        method=ode_method,
    ),
    vocab_char_map=vocab_char_map,
).to(device)

dtype = torch.float32 if mel_spec_type == "bigvgan" else None
model = load_checkpoint(model, ckpt_path, device, dtype=dtype, use_ema=use_ema)

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
audio = torch.cat((audio_, audio[:, round(offset) :]), dim=-1)
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
    gen_mel_spec = generated.permute(0, 2, 1)
    if mel_spec_type == "vocos":
        generated_wave = vocoder.decode(gen_mel_spec).cpu()
    elif mel_spec_type == "bigvgan":
        generated_wave = vocoder(gen_mel_spec).squeeze(0).cpu()

    if rms < target_rms:
        generated_wave = generated_wave * rms / target_rms

    save_spectrogram(gen_mel_spec[0].cpu().numpy(), f"{output_dir}/speech_edit_out.png")
    torchaudio.save(f"{output_dir}/speech_edit_out.wav", generated_wave, target_sample_rate)
    print(f"Generated wav: {generated_wave.shape}")
