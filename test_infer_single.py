import os
import re

import torch
import torchaudio
from einops import rearrange
from ema_pytorch import EMA
from vocos import Vocos

from model import CFM, UNetT, DiT, MMDiT
from model.utils import (
    get_tokenizer, 
    convert_char_to_pinyin, 
    save_spectrogram,
)

device = "cuda" if torch.cuda.is_available() else "cpu"


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
cfg_strength = 2.
ode_method = 'euler'  # euler | midpoint
sway_sampling_coef = -1.
speed = 1.
fix_duration = 27  # None (will linear estimate. if code-switched, consider fix) | float (total in seconds, include ref audio) 

if exp_name == "F5TTS_Base":
    model_cls = DiT
    model_cfg = dict(dim = 1024, depth = 22, heads = 16, ff_mult = 2, text_dim = 512, conv_layers = 4)

elif exp_name == "E2TTS_Base":
    model_cls = UNetT
    model_cfg = dict(dim = 1024, depth = 24, heads = 16, ff_mult = 4)

checkpoint = torch.load(f"ckpts/{exp_name}/model_{ckpt_step}.pt", map_location=device)
output_dir = "tests"

ref_audio = "tests/ref_audio/test_en_1_ref_short.wav"
ref_text = "Some call me nature, others call me mother nature."
gen_text = "I don't really care what you call me. I've been a silent spectator, watching species evolve, empires rise and fall. But always remember, I am mighty and enduring. Respect me and I'll nurture you; ignore me and you shall face the consequences."

# ref_audio = "tests/ref_audio/test_zh_1_ref_short.wav"
# ref_text = "对，这就是我，万人敬仰的太乙真人。"
# gen_text = "突然，身边一阵笑声。我看着他们，意气风发地挺直了胸膛，甩了甩那稍显肉感的双臂，轻笑道：\"我身上的肉，是为了掩饰我爆棚的魅力，否则，岂不吓坏了你们呢？\""


# -------------------------------------------------#

use_ema = True

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Vocoder model
local = False
if local:
    vocos_local_path = "../checkpoints/charactr/vocos-mel-24khz"
    vocos = Vocos.from_hparams(f"{vocos_local_path}/config.yaml")
    state_dict = torch.load(f"{vocos_local_path}/pytorch_model.bin", map_location=device)
    vocos.load_state_dict(state_dict)
    vocos.eval()
else:
    vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")

# Tokenizer
vocab_char_map, vocab_size = get_tokenizer(dataset_name, tokenizer)

# Model
model = CFM(
    transformer = model_cls(
        **model_cfg,
        text_num_embeds = vocab_size, 
        mel_dim = n_mel_channels
    ),
    mel_spec_kwargs = dict(
        target_sample_rate = target_sample_rate, 
        n_mel_channels = n_mel_channels,
        hop_length = hop_length,
    ),
    odeint_kwargs = dict(
        method = ode_method,
    ),
    vocab_char_map = vocab_char_map,
).to(device)

if use_ema == True:
    ema_model = EMA(model, include_online_model = False).to(device)
    ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
    ema_model.copy_params_from_ema_to_model()
else:
    model.load_state_dict(checkpoint['model_state_dict'])

# Audio
audio, sr = torchaudio.load(ref_audio)
rms = torch.sqrt(torch.mean(torch.square(audio)))
if rms < target_rms:
    audio = audio * target_rms / rms
if sr != target_sample_rate:
    resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
    audio = resampler(audio)
audio = audio.to(device)

# Text
text_list = [ref_text + gen_text]
if tokenizer == "pinyin":
    final_text_list = convert_char_to_pinyin(text_list)
else:
    final_text_list = [text_list]
print(f"text  : {text_list}")
print(f"pinyin: {final_text_list}")

# Duration
ref_audio_len = audio.shape[-1] // hop_length
if fix_duration is not None:
    duration = int(fix_duration * target_sample_rate / hop_length)
else:  # simple linear scale calcul
    zh_pause_punc = r"。，、；：？！"
    ref_text_len = len(ref_text) + len(re.findall(zh_pause_punc, ref_text))
    gen_text_len = len(gen_text) + len(re.findall(zh_pause_punc, gen_text))
    duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / speed)

# Inference
with torch.inference_mode():
    generated, trajectory = model.sample(
        cond = audio,
        text = final_text_list,
        duration = duration,
        steps = nfe_step,
        cfg_strength = cfg_strength,
        sway_sampling_coef = sway_sampling_coef,
        seed = seed,
    )
print(f"Generated mel: {generated.shape}")

# Final result
generated = generated[:, ref_audio_len:, :]
generated_mel_spec = rearrange(generated, '1 n d -> 1 d n')
generated_wave = vocos.decode(generated_mel_spec.cpu())
if rms < target_rms:
    generated_wave = generated_wave * rms / target_rms

save_spectrogram(generated_mel_spec[0].cpu().numpy(), f"{output_dir}/test_single.png")
torchaudio.save(f"{output_dir}/test_single.wav", generated_wave, target_sample_rate)
print(f"Generated wav: {generated_wave.shape}")
