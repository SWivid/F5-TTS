import os
import re
import torch
import torchaudio
import gradio as gr
import numpy as np
import tempfile
from einops import rearrange
from vocos import Vocos
from pydub import AudioSegment, silence
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
import soundfile as sf

SPLIT_WORDS = [
    "but", "however", "nevertheless", "yet", "still",
    "therefore", "thus", "hence", "consequently",
    "moreover", "furthermore", "additionally",
    "meanwhile", "alternatively", "otherwise",
    "namely", "specifically", "for example", "such as",
    "in fact", "indeed", "notably",
    "in contrast", "on the other hand", "conversely",
    "in conclusion", "to summarize", "finally"
]

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

def split_text_into_batches(text, max_chars=200, split_words=SPLIT_WORDS):
    if len(text.encode('utf-8')) <= max_chars:
        return [text]
    if text[-1] not in ['。', '.', '!', '！', '?', '？']:
        text += '.'
        
    sentences = re.split('([。.!?！？])', text)
    sentences = [''.join(i) for i in zip(sentences[0::2], sentences[1::2])]
    
    batches = []
    current_batch = ""
    
    def split_by_words(text):
        words = text.split()
        current_word_part = ""
        word_batches = []
        for word in words:
            if len(current_word_part.encode('utf-8')) + len(word.encode('utf-8')) + 1 <= max_chars:
                current_word_part += word + ' '
            else:
                if current_word_part:
                    # Try to find a suitable split word
                    for split_word in split_words:
                        split_index = current_word_part.rfind(' ' + split_word + ' ')
                        if split_index != -1:
                            word_batches.append(current_word_part[:split_index].strip())
                            current_word_part = current_word_part[split_index:].strip() + ' '
                            break
                    else:
                        # If no suitable split word found, just append the current part
                        word_batches.append(current_word_part.strip())
                        current_word_part = ""
                current_word_part += word + ' '
        if current_word_part:
            word_batches.append(current_word_part.strip())
        return word_batches

    for sentence in sentences:
        if len(current_batch.encode('utf-8')) + len(sentence.encode('utf-8')) <= max_chars:
            current_batch += sentence
        else:
            # If adding this sentence would exceed the limit
            if current_batch:
                batches.append(current_batch)
                current_batch = ""
            
            # If the sentence itself is longer than max_chars, split it
            if len(sentence.encode('utf-8')) > max_chars:
                # First, try to split by colon
                colon_parts = sentence.split(':')
                if len(colon_parts) > 1:
                    for part in colon_parts:
                        if len(part.encode('utf-8')) <= max_chars:
                            batches.append(part)
                        else:
                            # If colon part is still too long, split by comma
                            comma_parts = re.split('[,，]', part)
                            if len(comma_parts) > 1:
                                current_comma_part = ""
                                for comma_part in comma_parts:
                                    if len(current_comma_part.encode('utf-8')) + len(comma_part.encode('utf-8')) <= max_chars:
                                        current_comma_part += comma_part + ','
                                    else:
                                        if current_comma_part:
                                            batches.append(current_comma_part.rstrip(','))
                                        current_comma_part = comma_part + ','
                                if current_comma_part:
                                    batches.append(current_comma_part.rstrip(','))
                            else:
                                # If no comma, split by words
                                batches.extend(split_by_words(part))
                else:
                    # If no colon, split by comma
                    comma_parts = re.split('[,，]', sentence)
                    if len(comma_parts) > 1:
                        current_comma_part = ""
                        for comma_part in comma_parts:
                            if len(current_comma_part.encode('utf-8')) + len(comma_part.encode('utf-8')) <= max_chars:
                                current_comma_part += comma_part + ','
                            else:
                                if current_comma_part:
                                    batches.append(current_comma_part.rstrip(','))
                                current_comma_part = comma_part + ','
                        if current_comma_part:
                            batches.append(current_comma_part.rstrip(','))
                    else:
                        # If no comma, split by words
                        batches.extend(split_by_words(sentence))
            else:
                current_batch = sentence
    
    if current_batch:
        batches.append(current_batch)
    
    return batches

def infer_batch(ref_audio, ref_text, gen_text_batches, exp_name, remove_silence, progress=gr.Progress()):
    if exp_name == "F5-TTS":
        ema_model = F5TTS_ema_model
    elif exp_name == "E2-TTS":
        ema_model = E2TTS_ema_model

    audio, sr = torchaudio.load(ref_audio)
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

    for i, gen_text in enumerate(progress.tqdm(gen_text_batches)):
        # Prepare the text
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
        
        vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")
        generated_wave = vocos.decode(generated_mel_spec.cpu())
        if rms < target_rms:
            generated_wave = generated_wave * rms / target_rms

        # wav -> numpy
        generated_wave = generated_wave.squeeze().cpu().numpy()
        
        generated_waves.append(generated_wave)
        spectrograms.append(generated_mel_spec[0].cpu().numpy())

    # Combine all generated waves
    final_wave = np.concatenate(generated_waves)

    # Remove silence
    if remove_silence:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            sf.write(f.name, final_wave, target_sample_rate)
            aseg = AudioSegment.from_file(f.name)
            non_silent_segs = silence.split_on_silence(aseg, min_silence_len=1000, silence_thresh=-50, keep_silence=500)
            non_silent_wave = AudioSegment.silent(duration=0)
            for non_silent_seg in non_silent_segs:
                non_silent_wave += non_silent_seg
            aseg = non_silent_wave
            aseg.export(f.name, format="wav")
            final_wave, _ = torchaudio.load(f.name)
        final_wave = final_wave.squeeze().cpu().numpy()

    # Create a combined spectrogram
    combined_spectrogram = np.concatenate(spectrograms, axis=1)
    
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_spectrogram:
        spectrogram_path = tmp_spectrogram.name
        save_spectrogram(combined_spectrogram, spectrogram_path)

    return (target_sample_rate, final_wave), spectrogram_path

def infer(ref_audio_orig, ref_text, gen_text, exp_name, remove_silence, custom_split_words):
    if not custom_split_words.strip():
        custom_words = [word.strip() for word in custom_split_words.split(',')]
        global SPLIT_WORDS
        SPLIT_WORDS = custom_words

    print(gen_text)

    gr.Info("Converting audio...")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        aseg = AudioSegment.from_file(ref_audio_orig)

        non_silent_segs = silence.split_on_silence(aseg, min_silence_len=1000, silence_thresh=-50, keep_silence=500)
        non_silent_wave = AudioSegment.silent(duration=0)
        for non_silent_seg in non_silent_segs:
            non_silent_wave += non_silent_seg
        aseg = non_silent_wave

        audio_duration = len(aseg)
        if audio_duration > 15000:
            gr.Warning("Audio is over 15s, clipping to only first 15s.")
            aseg = aseg[:15000]
        aseg.export(f.name, format="wav")
        ref_audio = f.name

    if not ref_text.strip():
        gr.Info("No reference text provided, transcribing reference audio...")
        ref_text = pipe(
            ref_audio,
            chunk_length_s=30,
            batch_size=128,
            generate_kwargs={"task": "transcribe"},
            return_timestamps=False,
        )["text"].strip()
        gr.Info("Finished transcription")
    else:
        gr.Info("Using custom reference text...")

    # Split the input text into batches
    if len(ref_text.encode('utf-8')) == len(ref_text) and len(gen_text.encode('utf-8')) == len(gen_text):
        max_chars = 400-len(ref_text.encode('utf-8'))
    else:
        max_chars = 300-len(ref_text.encode('utf-8'))
    gen_text_batches = split_text_into_batches(gen_text, max_chars=max_chars)
    print('ref_text', ref_text)
    for i, gen_text in enumerate(gen_text_batches):
        print(f'gen_text {i}', gen_text)
    
    gr.Info(f"Generating audio using {exp_name} in {len(gen_text_batches)} batches")
    return infer_batch(ref_audio, ref_text, gen_text_batches, exp_name, remove_silence)
    
def generate_podcast(script, speaker1_name, ref_audio1, ref_text1, speaker2_name, ref_audio2, ref_text2, exp_name, remove_silence):
    # Split the script into speaker blocks
    speaker_pattern = re.compile(f"^({re.escape(speaker1_name)}|{re.escape(speaker2_name)}):", re.MULTILINE)
    speaker_blocks = speaker_pattern.split(script)[1:]  # Skip the first empty element
    
    generated_audio_segments = []
    
    for i in range(0, len(speaker_blocks), 2):
        speaker = speaker_blocks[i]
        text = speaker_blocks[i+1].strip()
        
        # Determine which speaker is talking
        if speaker == speaker1_name:
            ref_audio = ref_audio1
            ref_text = ref_text1
        elif speaker == speaker2_name:
            ref_audio = ref_audio2
            ref_text = ref_text2
        else:
            continue  # Skip if the speaker is neither speaker1 nor speaker2
        
        # Generate audio for this block
        audio, _ = infer(ref_audio, ref_text, text, exp_name, remove_silence)
        
        # Convert the generated audio to a numpy array
        sr, audio_data = audio
        
        # Save the audio data as a WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            sf.write(temp_file.name, audio_data, sr)
            audio_segment = AudioSegment.from_wav(temp_file.name)
        
        generated_audio_segments.append(audio_segment)
        
        # Add a short pause between speakers
        pause = AudioSegment.silent(duration=500)  # 500ms pause
        generated_audio_segments.append(pause)
    
    # Concatenate all audio segments
    final_podcast = sum(generated_audio_segments)
    
    # Export the final podcast
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        podcast_path = temp_file.name
        final_podcast.export(podcast_path, format="wav")
    
    return podcast_path

with gr.Blocks() as app:
    gr.Markdown(
        """
# E2/F5 TTS with Advanced Batch Processing

This is a local web UI for F5 TTS with advanced batch processing support, based on the unofficial [online demo](https://huggingface.co/spaces/mrfakename/E2-F5-TTS). This app supports the following TTS models:

* [F5-TTS](https://arxiv.org/abs/2410.06885) (A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching)
* [E2 TTS](https://arxiv.org/abs/2406.18009) (Embarrassingly Easy Fully Non-Autoregressive Zero-Shot TTS)

The checkpoints support English and Chinese.

If you're having issues, try converting your reference audio to WAV or MP3, clipping it to 15s, and shortening your prompt.

**NOTE: Reference text will be automatically transcribed with Whisper if not provided. For best results, keep your reference clips short (<15s). Ensure the audio is fully uploaded before generating.**
"""
    )

    ref_audio_input = gr.Audio(label="Reference Audio", type="filepath")
    gen_text_input = gr.Textbox(label="Text to Generate", lines=10)
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
        split_words_input = gr.Textbox(
            label="Custom Split Words",
            info="Enter custom words to split on, separated by commas. Leave blank to use default list.",
            lines=2,
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
            split_words_input,
        ],
        outputs=[audio_output, spectrogram_output],
    )
    
    gr.Markdown(
        """
# Podcast Generation

Supported by [RootingInLoad](https://github.com/RootingInLoad)
"""
    )
    with gr.Tab("Podcast Generation"):
        speaker1_name = gr.Textbox(label="Speaker 1 Name")
        ref_audio_input1 = gr.Audio(label="Reference Audio (Speaker 1)", type="filepath")
        ref_text_input1 = gr.Textbox(label="Reference Text (Speaker 1)", lines=2)
        
        speaker2_name = gr.Textbox(label="Speaker 2 Name")
        ref_audio_input2 = gr.Audio(label="Reference Audio (Speaker 2)", type="filepath")
        ref_text_input2 = gr.Textbox(label="Reference Text (Speaker 2)", lines=2)
        
        script_input = gr.Textbox(label="Podcast Script", lines=10, 
                                  placeholder="Enter the script with speaker names at the start of each block, e.g.:\nSean: How did you start studying...\n\nMeghan: I came to my interest in technology...\nIt was a long journey...\n\nSean: That's fascinating. Can you elaborate...")
        
        podcast_model_choice = gr.Radio(
            choices=["F5-TTS", "E2-TTS"], label="Choose TTS Model", value="F5-TTS"
        )
        podcast_remove_silence = gr.Checkbox(
            label="Remove Silences",
            value=True,
        )
        generate_podcast_btn = gr.Button("Generate Podcast", variant="primary")
        podcast_output = gr.Audio(label="Generated Podcast")
    
    def podcast_generation(script, speaker1, ref_audio1, ref_text1, speaker2, ref_audio2, ref_text2, model, remove_silence):
        return generate_podcast(script, speaker1, ref_audio1, ref_text1, speaker2, ref_audio2, ref_text2, model, remove_silence)
    
    generate_podcast_btn.click(
        podcast_generation,
        inputs=[
            script_input,
            speaker1_name,
            ref_audio_input1,
            ref_text_input1,
            speaker2_name,
            ref_audio_input2,
            ref_text_input2,
            podcast_model_choice,
            podcast_remove_silence,
        ],
        outputs=podcast_output,
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
