import argparse
import codecs
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import tomli
from cached_path import cached_path

from model import DiT, UNetT
from model.utils_infer import (
    load_vocoder,
    load_model,
    preprocess_ref_audio_text,
    infer_process,
    remove_silence_for_generated_wav,
)


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
ckpt_file = args.ckpt_file if args.ckpt_file else ""
vocab_file = args.vocab_file if args.vocab_file else ""
remove_silence = args.remove_silence if args.remove_silence else config["remove_silence"]
wave_path = Path(output_dir) / "out.wav"
spectrogram_path = Path(output_dir) / "out.png"
vocos_local_path = "../checkpoints/charactr/vocos-mel-24khz"

vocos = load_vocoder(is_local=args.load_vocoder_from_local, local_path=vocos_local_path)


# load models
if model == "F5-TTS":
    model_cls = DiT
    model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
    if ckpt_file == "":
        repo_name = "F5-TTS"
        exp_name = "F5TTS_Base"
        ckpt_step = 1200000
        ckpt_file = str(cached_path(f"hf://SWivid/{repo_name}/{exp_name}/model_{ckpt_step}.safetensors"))
        # ckpt_file = f"ckpts/{exp_name}/model_{ckpt_step}.pt"  # .pt | .safetensors; local path

elif model == "E2-TTS":
    model_cls = UNetT
    model_cfg = dict(dim=1024, depth=24, heads=16, ff_mult=4)
    if ckpt_file == "":
        repo_name = "E2-TTS"
        exp_name = "E2TTS_Base"
        ckpt_step = 1200000
        ckpt_file = str(cached_path(f"hf://SWivid/{repo_name}/{exp_name}/model_{ckpt_step}.safetensors"))
        # ckpt_file = f"ckpts/{exp_name}/model_{ckpt_step}.pt"  # .pt | .safetensors; local path

print(f"Using {model}...")
ema_model = load_model(model_cls, model_cfg, ckpt_file, vocab_file)


def main_process(ref_audio, ref_text, text_gen, model_obj, remove_silence):
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
        audio, final_sample_rate, spectragram = infer_process(ref_audio, ref_text, gen_text, model_obj)
        generated_audio_segments.append(audio)

    if generated_audio_segments:
        final_wave = np.concatenate(generated_audio_segments)
        with open(wave_path, "wb") as f:
            sf.write(f.name, final_wave, final_sample_rate)
            # Remove silence
            if remove_silence:
                remove_silence_for_generated_wav(f.name)
            print(f.name)


main_process(ref_audio, ref_text, gen_text, ema_model, remove_silence)
