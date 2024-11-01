import argparse
import codecs
import os
import re
from importlib.resources import files
from pathlib import Path

import numpy as np
import soundfile as sf
import tomli
from cached_path import cached_path

from f5_tts.infer.utils_infer import (
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
)
from f5_tts.model import DiT, UNetT

parser = argparse.ArgumentParser(
    prog="python3 infer-cli.py",
    description="Commandline interface for E2/F5 TTS with Advanced Batch Processing.",
    epilog="Specify options above to override one or more settings from config.",
)
parser.add_argument(
    "-c",
    "--config",
    help="Configuration file. Default=infer/examples/basic/basic.toml",
    default=os.path.join(files("f5_tts").joinpath("infer/examples/basic"), "basic.toml"),
)
parser.add_argument(
    "-m",
    "--model",
    help="F5-TTS | E2-TTS",
)
parser.add_argument(
    "-p",
    "--ckpt_file",
    help="The Checkpoint .pt",
)
parser.add_argument(
    "-v",
    "--vocab_file",
    help="The vocab .txt",
)
parser.add_argument("-r", "--ref_audio", type=str, help="Reference audio file < 15 seconds.")
parser.add_argument("-s", "--ref_text", type=str, default="666", help="Subtitle for the reference audio.")
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
parser.add_argument("--vocoder_name", type=str, default="vocos", choices=["vocos", "bigvgan"], help="vocoder name")
parser.add_argument(
    "--load_vocoder_from_local",
    action="store_true",
    help="load vocoder from local. Default: ../checkpoints/charactr/vocos-mel-24khz",
)
parser.add_argument(
    "--speed",
    type=float,
    default=1.0,
    help="Adjust the speed of the audio generation (default: 1.0)",
)
args = parser.parse_args()

config = tomli.load(open(args.config, "rb"))

ref_audio = args.ref_audio if args.ref_audio else config["ref_audio"]
ref_text = args.ref_text if args.ref_text != "666" else config["ref_text"]
gen_text = args.gen_text if args.gen_text else config["gen_text"]
gen_file = args.gen_file if args.gen_file else config["gen_file"]

# patches for pip pkg user
if "infer/examples/" in ref_audio:
    ref_audio = str(files("f5_tts").joinpath(f"{ref_audio}"))
if "infer/examples/" in gen_file:
    gen_file = str(files("f5_tts").joinpath(f"{gen_file}"))
if "voices" in config:
    for voice in config["voices"]:
        voice_ref_audio = config["voices"][voice]["ref_audio"]
        if "infer/examples/" in voice_ref_audio:
            config["voices"][voice]["ref_audio"] = str(files("f5_tts").joinpath(f"{voice_ref_audio}"))

if gen_file:
    gen_text = codecs.open(gen_file, "r", "utf-8").read()
output_dir = args.output_dir if args.output_dir else config["output_dir"]
model = args.model if args.model else config["model"]
ckpt_file = args.ckpt_file if args.ckpt_file else ""
vocab_file = args.vocab_file if args.vocab_file else ""
remove_silence = args.remove_silence if args.remove_silence else config["remove_silence"]
speed = args.speed
wave_path = Path(output_dir) / "infer_cli_out.wav"
# spectrogram_path = Path(output_dir) / "infer_cli_out.png"
if args.vocoder_name == "vocos":
    vocoder_local_path = "../checkpoints/vocos-mel-24khz"
elif args.vocoder_name == "bigvgan":
    vocoder_local_path = "../checkpoints/bigvgan_v2_24khz_100band_256x"
mel_spec_type = args.vocoder_name

vocoder = load_vocoder(vocoder_name=mel_spec_type, is_local=args.load_vocoder_from_local, local_path=vocoder_local_path)


# load models
if model == "F5-TTS":
    model_cls = DiT
    model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
    if ckpt_file == "":
        if args.vocoder_name == "vocos":
            repo_name = "F5-TTS"
            exp_name = "F5TTS_Base"
            ckpt_step = 1200000
            ckpt_file = str(cached_path(f"hf://SWivid/{repo_name}/{exp_name}/model_{ckpt_step}.safetensors"))
            # ckpt_file = f"ckpts/{exp_name}/model_{ckpt_step}.pt"  # .pt | .safetensors; local path
        elif args.vocoder_name == "bigvgan":
            repo_name = "F5-TTS"
            exp_name = "F5TTS_Base_bigvgan"
            ckpt_step = 1250000
            ckpt_file = str(cached_path(f"hf://SWivid/{repo_name}/{exp_name}/model_{ckpt_step}.pt"))

elif model == "E2-TTS":
    model_cls = UNetT
    model_cfg = dict(dim=1024, depth=24, heads=16, ff_mult=4)
    if ckpt_file == "":
        repo_name = "E2-TTS"
        exp_name = "E2TTS_Base"
        ckpt_step = 1200000
        ckpt_file = str(cached_path(f"hf://SWivid/{repo_name}/{exp_name}/model_{ckpt_step}.safetensors"))
        # ckpt_file = f"ckpts/{exp_name}/model_{ckpt_step}.pt"  # .pt | .safetensors; local path
    elif args.vocoder_name == "bigvgan":  # TODO: need to test
        repo_name = "F5-TTS"
        exp_name = "F5TTS_Base_bigvgan"
        ckpt_step = 1250000
        ckpt_file = str(cached_path(f"hf://SWivid/{repo_name}/{exp_name}/model_{ckpt_step}.pt"))


print(f"Using {model}...")
ema_model = load_model(model_cls, model_cfg, ckpt_file, mel_spec_type=args.vocoder_name, vocab_file=vocab_file)


def main_process(ref_audio, ref_text, text_gen, model_obj, mel_spec_type, remove_silence, speed):
    main_voice = {"ref_audio": ref_audio, "ref_text": ref_text}
    if "voices" not in config:
        voices = {"main": main_voice}
    else:
        voices = config["voices"]
        voices["main"] = main_voice
    for voice in voices:
        voices[voice]["ref_audio"], voices[voice]["ref_text"] = preprocess_ref_audio_text(
            voices[voice]["ref_audio"], voices[voice]["ref_text"]
        )
        print("Voice:", voice)
        print("Ref_audio:", voices[voice]["ref_audio"])
        print("Ref_text:", voices[voice]["ref_text"])

    generated_audio_segments = []
    reg1 = r"(?=\[\w+\])"
    chunks = re.split(reg1, text_gen)
    reg2 = r"\[(\w+)\]"
    for text in chunks:
        if not text.strip():
            continue
        match = re.match(reg2, text)
        if match:
            voice = match[1]
        else:
            print("No voice tag found, using main.")
            voice = "main"
        if voice not in voices:
            print(f"Voice {voice} not found, using main.")
            voice = "main"
        text = re.sub(reg2, "", text)
        gen_text = text.strip()
        ref_audio = voices[voice]["ref_audio"]
        ref_text = voices[voice]["ref_text"]
        print(f"Voice: {voice}")
        audio, final_sample_rate, spectragram = infer_process(
            ref_audio, ref_text, gen_text, model_obj, vocoder, mel_spec_type=mel_spec_type, speed=speed
        )
        generated_audio_segments.append(audio)

    if generated_audio_segments:
        final_wave = np.concatenate(generated_audio_segments)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(wave_path, "wb") as f:
            sf.write(f.name, final_wave, final_sample_rate)
            # Remove silence
            if remove_silence:
                remove_silence_for_generated_wav(f.name)
            print(f.name)


def main():
    main_process(ref_audio, ref_text, gen_text, ema_model, mel_spec_type, remove_silence, speed)


if __name__ == "__main__":
    main()
