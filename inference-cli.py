import argparse
import codecs
import re
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import tomli
import torch
import torchaudio
import tqdm
from cached_path import cached_path
from einops import rearrange
from pydub import AudioSegment, silence
from transformers import pipeline
from vocos import Vocos

from model import CFM, DiT, MMDiT, UNetT
from model.utils import (convert_char_to_pinyin, get_tokenizer,
                         load_checkpoint, save_spectrogram)

parser = argparse.ArgumentParser(
    prog="python3 inference-cli.py",
    description="Commandline interface for E2/F5 TTS with Advanced Batch Processing.",
    epilog="Specify  options above  to override  one or more settings from config.",
)
parser.add_argument(
    "-c",
    "--config",
    help="Configuration file. Default=cli-config.toml",
    default="inference-cli.toml",
)
parser.add_argument(
    "-m",
    "--model",
    help="F5-TTS | E2-TTS",
)
parser.add_argument(
    "-r",
    "--ref_audio",
    type=str,
    help="Reference audio file < 15 seconds."
)
parser.add_argument(
    "-s",
    "--ref_text",
    type=str,
    default="666",
    help="Subtitle for the reference audio."
)
parser.add_argument(
    "-t",
    "--gen_text",
    type=str,
    help="Text to generate.",
)
parser.add_argument(
    "-f",
    "--gen_file",
    type=str,
    help="File with text to generate. Ignores --text",
)
parser.add_argument(
    "-o",
    "--output_dir",
    type=str,
    help="Path to output folder..",
)
parser.add_argument(
    "--remove_silence",
    help="Remove silence.",
)
parser.add_argument(
    "--load_vocoder_from_local",
    action="store_true",
    help="load vocoder from local. Default: ../checkpoints/charactr/vocos-mel-24khz",
)
args = parser.parse_args()

config = tomli.load(open(args.config, "rb"))

ref_audio = args.ref_audio if args.ref_audio else config["ref_audio"]
ref_text = args.ref_text if args.ref_text != "666" else config["ref_text"]
gen_text = args.gen_text if args.gen_text else config["gen_text"]
gen_file = args.gen_file if args.gen_file else config["gen_file"]
if gen_file:
    gen_text = codecs.open(gen_file, "r", "utf-8").read()
output_dir = args.output_dir if args.output_dir else config["output_dir"]
model = args.model if args.model else config["model"]
remove_silence = args.remove_silence if args.remove_silence else config["remove_silence"]
wave_path = Path(output_dir)/"out.wav"
spectrogram_path = Path(output_dir)/"out.png"
vocos_local_path = "../checkpoints/charactr/vocos-mel-24khz"

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

if args.load_vocoder_from_local:
    print(f"Load vocos from local path {vocos_local_path}")
    vocos = Vocos.from_hparams(f"{vocos_local_path}/config.yaml")
    state_dict = torch.load(f"{vocos_local_path}/pytorch_model.bin", map_location=device)
    vocos.load_state_dict(state_dict)
    vocos.eval()
else:
    print("Donwload Vocos from huggingface charactr/vocos-mel-24khz")
    vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")

print(f"Using {device} device")

# --------------------- Settings -------------------- #

target_sample_rate = 24000
n_mel_channels = 100
hop_length = 256
target_rms = 0.1
nfe_step = 32  # 16, 32
cfg_strength = 2.0
ode_method = "euler"
sway_sampling_coef = -1.0
speed = 1.0
# fix_duration = 27  # None or float (duration in seconds)
fix_duration = None

def load_model(repo_name, exp_name, model_cls, model_cfg, ckpt_step):
    ckpt_path = f"ckpts/{exp_name}/model_{ckpt_step}.pt" # .pt | .safetensors
    if not Path(ckpt_path).exists():
        ckpt_path = str(cached_path(f"hf://SWivid/{repo_name}/{exp_name}/model_{ckpt_step}.safetensors"))
    vocab_char_map, vocab_size = get_tokenizer("Emilia_ZH_EN", "pinyin")
    model = CFM(
        transformer=model_cls(
            **model_cfg, text_num_embeds=vocab_size, mel_dim=n_mel_channels
        ),
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

    model = load_checkpoint(model, ckpt_path, device, use_ema = True)

    return model


# load models
F5TTS_model_cfg = dict(
    dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4
)
E2TTS_model_cfg = dict(dim=1024, depth=24, heads=16, ff_mult=4)


def chunk_text(text, max_chars=135):
    """
    Splits the input text into chunks, each with a maximum number of characters.
    Args:
        text (str): The text to be split.
        max_chars (int): The maximum number of characters per chunk.
    Returns:
        List[str]: A list of text chunks.
    """
    chunks = []
    current_chunk = ""
    # Split the text into sentences based on punctuation followed by whitespace
    sentences = re.split(r'(?<=[;:,.!?])\s+|(?<=[；：，。！？])', text)

    for sentence in sentences:
        if len(current_chunk.encode('utf-8')) + len(sentence.encode('utf-8')) <= max_chars:
            current_chunk += sentence + " " if sentence and len(sentence[-1].encode('utf-8')) == 1 else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " " if sentence and len(sentence[-1].encode('utf-8')) == 1 else sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def infer_batch(ref_audio, ref_text, gen_text_batches, model, remove_silence, cross_fade_duration=0.15):
    if model == "F5-TTS":
        ema_model = load_model(model, "F5TTS_Base", DiT, F5TTS_model_cfg, 1200000)
    elif model == "E2-TTS":
        ema_model = load_model(model, "E2TTS_Base", UNetT, E2TTS_model_cfg, 1200000)

    audio, sr = ref_audio
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    rms = torch.sqrt(torch.mean(torch.square(audio)))
    if rms < target_rms:
        audio = audio * target_rms / rms
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
        audio = resampler(audio)
    audio = audio.to(device)

    generated_waves = []
    spectrograms = []

    for i, gen_text in enumerate(tqdm.tqdm(gen_text_batches)):
        # Prepare the text
        if len(ref_text[-1].encode('utf-8')) == 1:
            ref_text = ref_text + " "
        text_list = [ref_text + gen_text]
        final_text_list = convert_char_to_pinyin(text_list)

        # Calculate duration
        ref_audio_len = audio.shape[-1] // hop_length
        zh_pause_punc = r"。，、；：？！"
        ref_text_len = len(ref_text.encode('utf-8')) + 3 * len(re.findall(zh_pause_punc, ref_text))
        gen_text_len = len(gen_text.encode('utf-8')) + 3 * len(re.findall(zh_pause_punc, gen_text))
        duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / speed)

        # inference
        with torch.inference_mode():
            generated, _ = ema_model.sample(
                cond=audio,
                text=final_text_list,
                duration=duration,
                steps=nfe_step,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef,
            )

        generated = generated[:, ref_audio_len:, :]
        generated_mel_spec = rearrange(generated, "1 n d -> 1 d n")
        generated_wave = vocos.decode(generated_mel_spec.cpu())
        if rms < target_rms:
            generated_wave = generated_wave * rms / target_rms

        # wav -> numpy
        generated_wave = generated_wave.squeeze().cpu().numpy()
        
        generated_waves.append(generated_wave)
        spectrograms.append(generated_mel_spec[0].cpu().numpy())

    # Combine all generated waves with cross-fading
    if cross_fade_duration <= 0:
        # Simply concatenate
        final_wave = np.concatenate(generated_waves)
    else:
        final_wave = generated_waves[0]
        for i in range(1, len(generated_waves)):
            prev_wave = final_wave
            next_wave = generated_waves[i]

            # Calculate cross-fade samples, ensuring it does not exceed wave lengths
            cross_fade_samples = int(cross_fade_duration * target_sample_rate)
            cross_fade_samples = min(cross_fade_samples, len(prev_wave), len(next_wave))

            if cross_fade_samples <= 0:
                # No overlap possible, concatenate
                final_wave = np.concatenate([prev_wave, next_wave])
                continue

            # Overlapping parts
            prev_overlap = prev_wave[-cross_fade_samples:]
            next_overlap = next_wave[:cross_fade_samples]

            # Fade out and fade in
            fade_out = np.linspace(1, 0, cross_fade_samples)
            fade_in = np.linspace(0, 1, cross_fade_samples)

            # Cross-faded overlap
            cross_faded_overlap = prev_overlap * fade_out + next_overlap * fade_in

            # Combine
            new_wave = np.concatenate([
                prev_wave[:-cross_fade_samples],
                cross_faded_overlap,
                next_wave[cross_fade_samples:]
            ])

            final_wave = new_wave

    with open(wave_path, "wb") as f:
        sf.write(f.name, final_wave, target_sample_rate)
        # Remove silence
        if remove_silence:
            aseg = AudioSegment.from_file(f.name)
            non_silent_segs = silence.split_on_silence(aseg, min_silence_len=1000, silence_thresh=-50, keep_silence=500)
            non_silent_wave = AudioSegment.silent(duration=0)
            for non_silent_seg in non_silent_segs:
                non_silent_wave += non_silent_seg
            aseg = non_silent_wave
            aseg.export(f.name, format="wav")
        print(f.name)

    # Create a combined spectrogram
    combined_spectrogram = np.concatenate(spectrograms, axis=1)
    save_spectrogram(combined_spectrogram, spectrogram_path)
    print(spectrogram_path)


def infer(ref_audio_orig, ref_text, gen_text, model, remove_silence, cross_fade_duration=0.15):

    print(gen_text)

    print("Converting audio...")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        aseg = AudioSegment.from_file(ref_audio_orig)

        non_silent_segs = silence.split_on_silence(aseg, min_silence_len=1000, silence_thresh=-50, keep_silence=1000)
        non_silent_wave = AudioSegment.silent(duration=0)
        for non_silent_seg in non_silent_segs:
            non_silent_wave += non_silent_seg
        aseg = non_silent_wave

        audio_duration = len(aseg)
        if audio_duration > 15000:
            print("Audio is over 15s, clipping to only first 15s.")
            aseg = aseg[:15000]
        aseg.export(f.name, format="wav")
        ref_audio = f.name

    if not ref_text.strip():
        print("No reference text provided, transcribing reference audio...")
        pipe = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-large-v3-turbo",
            torch_dtype=torch.float16,
            device=device,
        )
        ref_text = pipe(
            ref_audio,
            chunk_length_s=30,
            batch_size=128,
            generate_kwargs={"task": "transcribe"},
            return_timestamps=False,
        )["text"].strip()
        print("Finished transcription")
    else:
        print("Using custom reference text...")

    # Add the functionality to ensure it ends with ". "
    if not ref_text.endswith(". ") and not ref_text.endswith("。"):
        if ref_text.endswith("."):
            ref_text += " "
        else:
            ref_text += ". "

    # Split the input text into batches
    audio, sr = torchaudio.load(ref_audio)
    max_chars = int(len(ref_text.encode('utf-8')) / (audio.shape[-1] / sr) * (25 - audio.shape[-1] / sr))
    gen_text_batches = chunk_text(gen_text, max_chars=max_chars)
    print('ref_text', ref_text)
    for i, gen_text in enumerate(gen_text_batches):
        print(f'gen_text {i}', gen_text)
    
    print(f"Generating audio using {model} in {len(gen_text_batches)} batches, loading models...")
    return infer_batch((audio, sr), ref_text, gen_text_batches, model, remove_silence, cross_fade_duration)
    

infer(ref_audio, ref_text, gen_text, model, remove_silence)
