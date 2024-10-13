import os
import re
import torch
import torchaudio
import gradio as gr
import numpy as np
import tempfile
from einops import rearrange
from vocos import Vocos
from pydub import AudioSegment
from model import CFM, UNetT, DiT, MMDiT
from cached_path import cached_path
from model.utils import (
    load_checkpoint,
    get_tokenizer,
    convert_char_to_pinyin,
    save_spectrogram,
)
from transformers import pipeline
import librosa
import click

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

print(f"Using {device} device")

pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3-turbo",
    torch_dtype=torch.float16,
    device=device,
)

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


def load_model(exp_name, model_cls, model_cfg, ckpt_step):
    ckpt_path = str(cached_path(f"hf://SWivid/F5-TTS/{exp_name}/model_{ckpt_step}.safetensors"))
    # ckpt_path = f"ckpts/{exp_name}/model_{ckpt_step}.pt"  # .pt | .safetensors
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

F5TTS_ema_model = load_model(
    "F5TTS_Base", DiT, F5TTS_model_cfg, 1200000
)
E2TTS_ema_model = load_model(
    "E2TTS_Base", UNetT, E2TTS_model_cfg, 1200000
)


def infer(ref_audio_orig, ref_text, gen_text, exp_name, remove_silence):
    print(gen_text)
    if len(gen_text) > 200:
        raise gr.Error("Please keep your text under 200 chars.")
    gr.Info("Converting audio...")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        aseg = AudioSegment.from_file(ref_audio_orig)
        audio_duration = len(aseg)
        if audio_duration > 15000:
            gr.Warning("Audio is over 15s, clipping to only first 15s.")
            aseg = aseg[:15000]
        aseg.export(f.name, format="wav")
        ref_audio = f.name
    if exp_name == "F5-TTS":
        ema_model = F5TTS_ema_model
    elif exp_name == "E2-TTS":
        ema_model = E2TTS_ema_model

    if not ref_text.strip():
        gr.Info("No reference text provided, transcribing reference audio...")
        ref_text = outputs = pipe(
            ref_audio,
            chunk_length_s=30,
            batch_size=128,
            generate_kwargs={"task": "transcribe"},
            return_timestamps=False,
        )["text"].strip()
        gr.Info("Finished transcription")
    else:
        gr.Info("Using custom reference text...")
    audio, sr = torchaudio.load(ref_audio)

    rms = torch.sqrt(torch.mean(torch.square(audio)))
    if rms < target_rms:
        audio = audio * target_rms / rms
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
        audio = resampler(audio)
    audio = audio.to(device)

    # Prepare the text
    text_list = [ref_text + gen_text]
    final_text_list = convert_char_to_pinyin(text_list)

    # Calculate duration
    ref_audio_len = audio.shape[-1] // hop_length
    # if fix_duration is not None:
    #     duration = int(fix_duration * target_sample_rate / hop_length)
    # else:
    zh_pause_punc = r"。，、；：？！"
    ref_text_len = len(ref_text) + len(re.findall(zh_pause_punc, ref_text))
    gen_text_len = len(gen_text) + len(re.findall(zh_pause_punc, gen_text))
    duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / speed)

    # inference
    gr.Info(f"Generating audio using {exp_name}")
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
    gr.Info("Running vocoder")
    vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")
    generated_wave = vocos.decode(generated_mel_spec.cpu())
    if rms < target_rms:
        generated_wave = generated_wave * rms / target_rms

    # wav -> numpy
    generated_wave = generated_wave.squeeze().cpu().numpy()

    if remove_silence:
        gr.Info("Removing audio silences... This may take a moment")
        non_silent_intervals = librosa.effects.split(generated_wave, top_db=30)
        non_silent_wave = np.array([])
        for interval in non_silent_intervals:
            start, end = interval
            non_silent_wave = np.concatenate(
                [non_silent_wave, generated_wave[start:end]]
            )
        generated_wave = non_silent_wave

    # spectogram
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_spectrogram:
        spectrogram_path = tmp_spectrogram.name
        save_spectrogram(generated_mel_spec[0].cpu().numpy(), spectrogram_path)

    return (target_sample_rate, generated_wave), spectrogram_path


with gr.Blocks() as app:
    gr.Markdown(
        """
# E2/F5 TTS

This is a local web UI for F5 TTS, based on the unofficial [online demo](https://huggingface.co/spaces/mrfakename/E2-F5-TTS). This app supports the following TTS models:

* [F5-TTS](https://arxiv.org/abs/2410.06885) (A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching)
* [E2-TTS](https://arxiv.org/abs/2406.18009) (Embarrassingly Easy Fully Non-Autoregressive Zero-Shot TTS)

The checkpoints support English and Chinese.

If you're having issues, try converting your reference audio to WAV or MP3, clipping it to 15s, and shortening your prompt.

**NOTE: Reference text will be automatically transcribed with Whisper if not provided. For best results, keep your reference clips short (<15s). Ensure the audio is fully uploaded before generating.**
"""
    )

    ref_audio_input = gr.Audio(label="Reference Audio", type="filepath")
    gen_text_input = gr.Textbox(label="Text to Generate (max 200 chars.)", lines=4)
    model_choice = gr.Radio(
        choices=["F5-TTS", "E2-TTS"], label="Choose TTS Model", value="F5-TTS"
    )
    generate_btn = gr.Button("Synthesize", variant="primary")
    with gr.Accordion("Advanced Settings", open=False):
        ref_text_input = gr.Textbox(
            label="Reference Text",
            info="Leave blank to automatically transcribe the reference audio. If you enter text it will override automatic transcription.",
            lines=2,
        )
        remove_silence = gr.Checkbox(
            label="Remove Silences",
            info="The model tends to produce silences, especially on longer audio. We can manually remove silences if needed. Note that this is an experimental feature and may produce strange results. This will also increase generation time.",
            value=True,
        )

    audio_output = gr.Audio(label="Synthesized Audio")
    spectrogram_output = gr.Image(label="Spectrogram")

    generate_btn.click(
        infer,
        inputs=[
            ref_audio_input,
            ref_text_input,
            gen_text_input,
            model_choice,
            remove_silence,
        ],
        outputs=[audio_output, spectrogram_output],
    )


@click.command()
@click.option("--port", "-p", default=None, type=int, help="Port to run the app on")
@click.option("--host", "-H", default=None, help="Host to run the app on")
@click.option(
    "--share",
    "-s",
    default=False,
    is_flag=True,
    help="Share the app via Gradio share link",
)
@click.option("--api", "-a", default=True, is_flag=True, help="Allow API access")
def main(port, host, share, api):
    global app
    print(f"Starting app...")
    app.queue(api_open=api).launch(
        server_name=host, server_port=port, share=share, show_api=api
    )


if __name__ == "__main__":
    main()
