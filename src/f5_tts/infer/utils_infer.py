# A unified script for inference process
# Make adjustments inside functions, and consider both gradio and cli scripts if need to change func output format
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np # Ensure numpy is imported for concatenate
import soundfile as sf # For saving intermediate audio


os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # for MPS device compatibility
sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../../third_party/BigVGAN/")

import hashlib
import re
import tempfile
from importlib.resources import files

import matplotlib


matplotlib.use("Agg")

import matplotlib.pylab as plt
import numpy as np
import torch
import torchaudio
import tqdm
from huggingface_hub import hf_hub_download
from pydub import AudioSegment, silence
from transformers import pipeline
from vocos import Vocos

from f5_tts.model import CFM
from f5_tts.model.utils import convert_char_to_pinyin, get_tokenizer


_ref_audio_cache = {}
_ref_text_cache = {}

device = (
    "cuda"
    if torch.cuda.is_available()
    else "xpu"
    if torch.xpu.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# -----------------------------------------

target_sample_rate = 24000
n_mel_channels = 100
hop_length = 256
win_length = 1024
n_fft = 1024
mel_spec_type = "vocos"
target_rms = 0.1
cross_fade_duration = 0.15
ode_method = "euler"
nfe_step = 32  # 16, 32
cfg_strength = 2.0
sway_sampling_coef = -1.0
speed = 1.0
fix_duration = None

# -----------------------------------------


# chunk text into smaller pieces


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
    sentences = re.split(r"(?<=[;:,.!?])\s+|(?<=[；：，。！？])", text)

    for sentence in sentences:
        if len(current_chunk.encode("utf-8")) + len(sentence.encode("utf-8")) <= max_chars:
            current_chunk += sentence + " " if sentence and len(sentence[-1].encode("utf-8")) == 1 else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " " if sentence and len(sentence[-1].encode("utf-8")) == 1 else sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


# load vocoder
def load_vocoder(vocoder_name="vocos", is_local=False, local_path="", device=device, hf_cache_dir=None):
    if vocoder_name == "vocos":
        # vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device)
        if is_local:
            print(f"Load vocos from local path {local_path}")
            config_path = f"{local_path}/config.yaml"
            model_path = f"{local_path}/pytorch_model.bin"
        else:
            print("Download Vocos from huggingface charactr/vocos-mel-24khz")
            repo_id = "charactr/vocos-mel-24khz"
            config_path = hf_hub_download(repo_id=repo_id, cache_dir=hf_cache_dir, filename="config.yaml")
            model_path = hf_hub_download(repo_id=repo_id, cache_dir=hf_cache_dir, filename="pytorch_model.bin")
        vocoder = Vocos.from_hparams(config_path)
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        from vocos.feature_extractors import EncodecFeatures

        if isinstance(vocoder.feature_extractor, EncodecFeatures):
            encodec_parameters = {
                "feature_extractor.encodec." + key: value
                for key, value in vocoder.feature_extractor.encodec.state_dict().items()
            }
            state_dict.update(encodec_parameters)
        vocoder.load_state_dict(state_dict)
        vocoder = vocoder.eval().to(device)
    elif vocoder_name == "bigvgan":
        try:
            from third_party.BigVGAN import bigvgan
        except ImportError:
            print("You need to follow the README to init submodule and change the BigVGAN source code.")
        if is_local:
            # download generator from https://huggingface.co/nvidia/bigvgan_v2_24khz_100band_256x/tree/main
            vocoder = bigvgan.BigVGAN.from_pretrained(local_path, use_cuda_kernel=False)
        else:
            vocoder = bigvgan.BigVGAN.from_pretrained(
                "nvidia/bigvgan_v2_24khz_100band_256x", use_cuda_kernel=False, cache_dir=hf_cache_dir
            )

        vocoder.remove_weight_norm()
        vocoder = vocoder.eval().to(device)
    return vocoder


# load asr pipeline

asr_pipe = None


def initialize_asr_pipeline(device: str = device, dtype=None):
    if dtype is None:
        dtype = (
            torch.float16
            if "cuda" in device
            and torch.cuda.get_device_properties(device).major >= 7
            and not torch.cuda.get_device_name().endswith("[ZLUDA]")
            else torch.float32
        )
    global asr_pipe
    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3-turbo",
        torch_dtype=dtype,
        device=device,
    )


# transcribe


def transcribe(ref_audio, language=None):
    global asr_pipe
    if asr_pipe is None:
        initialize_asr_pipeline(device=device)
    return asr_pipe(
        ref_audio,
        chunk_length_s=30,
        batch_size=128,
        generate_kwargs={"task": "transcribe", "language": language} if language else {"task": "transcribe"},
        return_timestamps=False,
    )["text"].strip()


# load model checkpoint for inference


def load_checkpoint(model, ckpt_path, device: str, dtype=None, use_ema=True):
    if dtype is None:
        dtype = (
            torch.float16
            if "cuda" in device
            and torch.cuda.get_device_properties(device).major >= 7
            and not torch.cuda.get_device_name().endswith("[ZLUDA]")
            else torch.float32
        )
    model = model.to(dtype)

    ckpt_type = ckpt_path.split(".")[-1]
    if ckpt_type == "safetensors":
        from safetensors.torch import load_file

        checkpoint = load_file(ckpt_path, device=device)
    else:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)

    if use_ema:
        if ckpt_type == "safetensors":
            checkpoint = {"ema_model_state_dict": checkpoint}
        checkpoint["model_state_dict"] = {
            k.replace("ema_model.", ""): v
            for k, v in checkpoint["ema_model_state_dict"].items()
            if k not in ["initted", "step"]
        }

        # patch for backward compatibility, 305e3ea
        for key in ["mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window"]:
            if key in checkpoint["model_state_dict"]:
                del checkpoint["model_state_dict"][key]

        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        if ckpt_type == "safetensors":
            checkpoint = {"model_state_dict": checkpoint}
        model.load_state_dict(checkpoint["model_state_dict"])

    del checkpoint
    torch.cuda.empty_cache()

    return model.to(device)


# load model for inference


def load_model(
    model_cls,
    model_cfg,
    ckpt_path,
    mel_spec_type=mel_spec_type,
    vocab_file="",
    ode_method=ode_method,
    use_ema=True,
    device=device,
):
    if vocab_file == "":
        vocab_file = str(files("f5_tts").joinpath("infer/examples/vocab.txt"))
    tokenizer = "custom"

    print("\nvocab : ", vocab_file)
    print("token : ", tokenizer)
    print("model : ", ckpt_path, "\n")

    vocab_char_map, vocab_size = get_tokenizer(vocab_file, tokenizer)
    model = CFM(
        transformer=model_cls(**model_cfg, text_num_embeds=vocab_size, mel_dim=n_mel_channels),
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

    return model


def remove_silence_edges(audio, silence_threshold=-42):
    # Remove silence from the start
    non_silent_start_idx = silence.detect_leading_silence(audio, silence_threshold=silence_threshold)
    audio = audio[non_silent_start_idx:]

    # Remove silence from the end
    non_silent_end_duration = audio.duration_seconds
    for ms in reversed(audio):
        if ms.dBFS > silence_threshold:
            break
        non_silent_end_duration -= 0.001
    trimmed_audio = audio[: int(non_silent_end_duration * 1000)]

    return trimmed_audio


# preprocess reference audio and text


def preprocess_ref_audio_text(ref_audio_orig, ref_text, show_info=print):
    show_info("Converting audio...")

    # Compute a hash of the reference audio file
    with open(ref_audio_orig, "rb") as audio_file:
        audio_data = audio_file.read()
        audio_hash = hashlib.md5(audio_data).hexdigest()

    global _ref_audio_cache

    if audio_hash in _ref_audio_cache:
        show_info("Using cached preprocessed reference audio...")
        ref_audio = _ref_audio_cache[audio_hash]

    else:  # first pass, do preprocess
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            aseg = AudioSegment.from_file(ref_audio_orig)

            # 1. try to find long silence for clipping
            non_silent_segs = silence.split_on_silence(
                aseg, min_silence_len=1000, silence_thresh=-50, keep_silence=1000, seek_step=10
            )
            non_silent_wave = AudioSegment.silent(duration=0)
            for non_silent_seg in non_silent_segs:
                if len(non_silent_wave) > 6000 and len(non_silent_wave + non_silent_seg) > 12000:
                    show_info("Audio is over 12s, clipping short. (1)")
                    break
                non_silent_wave += non_silent_seg

            # 2. try to find short silence for clipping if 1. failed
            if len(non_silent_wave) > 12000:
                non_silent_segs = silence.split_on_silence(
                    aseg, min_silence_len=100, silence_thresh=-40, keep_silence=1000, seek_step=10
                )
                non_silent_wave = AudioSegment.silent(duration=0)
                for non_silent_seg in non_silent_segs:
                    if len(non_silent_wave) > 6000 and len(non_silent_wave + non_silent_seg) > 12000:
                        show_info("Audio is over 12s, clipping short. (2)")
                        break
                    non_silent_wave += non_silent_seg

            aseg = non_silent_wave

            # 3. if no proper silence found for clipping
            if len(aseg) > 12000:
                aseg = aseg[:12000]
                show_info("Audio is over 12s, clipping short. (3)")

            aseg = remove_silence_edges(aseg) + AudioSegment.silent(duration=50)
            aseg.export(f.name, format="wav")
            ref_audio = f.name

        # Cache the processed reference audio
        _ref_audio_cache[audio_hash] = ref_audio

    if not ref_text.strip():
        global _ref_text_cache
        if audio_hash in _ref_text_cache:
            # Use cached asr transcription
            show_info("Using cached reference text...")
            ref_text = _ref_text_cache[audio_hash]
        else:
            show_info("No reference text provided, transcribing reference audio...")
            ref_text = transcribe(ref_audio)
            # Cache the transcribed text (not caching custom ref_text, enabling users to do manual tweak)
            _ref_text_cache[audio_hash] = ref_text
    else:
        show_info("Using custom reference text...")

    # Ensure ref_text ends with a proper sentence-ending punctuation
    if not ref_text.endswith(". ") and not ref_text.endswith("。"):
        if ref_text.endswith("."):
            ref_text += " "
        else:
            ref_text += ". "

    print("\nref_text  ", ref_text)

    return ref_audio, ref_text


# infer process: chunk text -> infer batches [i.e. infer_batch_process()]


def infer_process(
    ref_audio,
    ref_text,
    gen_text,
    model_obj,
    vocoder,
    mel_spec_type=mel_spec_type,
    show_info=print,
    progress=tqdm,
    target_rms=target_rms,
    cross_fade_duration=cross_fade_duration,
    nfe_step=nfe_step,
    cfg_strength=cfg_strength,
    sway_sampling_coef=sway_sampling_coef,
    speed=speed,
    fix_duration=fix_duration,
    device=device,
    skip_on_error: bool = False,
    save_intermediate_every_n_chunks: int = 0,
    output_file_path: str = None,
):
    # Split the input text into batches
    audio, sr = torchaudio.load(ref_audio)
    max_chars = int(len(ref_text.encode("utf-8")) / (audio.shape[-1] / sr) * (22 - audio.shape[-1] / sr) * speed)
    gen_text_batches = chunk_text(gen_text, max_chars=max_chars)
    total_chunks = len(gen_text_batches)
    for i, current_gen_text in enumerate(gen_text_batches):
        print(f"gen_text {i}", current_gen_text)
    print("\n")

    show_info(f"Generating audio in {total_chunks} batches...")
    return next(
        infer_batch_process(
            (audio, sr),
            ref_text,
            gen_text_batches,
            model_obj,
            vocoder,
            total_chunks=total_chunks,
            mel_spec_type=mel_spec_type,
            progress=progress,
            target_rms=target_rms,
            cross_fade_duration=cross_fade_duration,
            nfe_step=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            speed=speed,
            fix_duration=fix_duration,
            device=device,
            skip_on_error=skip_on_error,
            save_intermediate_every_n_chunks=save_intermediate_every_n_chunks,
            output_file_path=output_file_path,
        )
    )


# infer batches


def infer_batch_process(
    ref_audio,
    ref_text,
    gen_text_batches,
    model_obj,
    vocoder,
    total_chunks=1, # Default to 1 if not provided, though infer_process should always provide it
    mel_spec_type="vocos",
    progress=tqdm,
    target_rms=0.1,
    cross_fade_duration=0.15,
    nfe_step=32,
    cfg_strength=2.0,
    sway_sampling_coef=-1,
    speed=1,
    fix_duration=None,
    device=None,
    streaming=False,
    chunk_size=2048,
    skip_on_error: bool = False,
    show_info=print,
    save_intermediate_every_n_chunks: int = 0,
    output_file_path: str = None,
):
    start_time = time.time()
    current_chunk_num = 0
    intermediate_save_part_count = 1

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

    if len(ref_text[-1].encode("utf-8")) == 1:
        ref_text = ref_text + " "

    def process_batch(gen_text):
        try:
            local_speed = speed
            if len(gen_text.encode("utf-8")) < 10:
                local_speed = 0.3

            # Prepare the text
            text_list = [ref_text + gen_text]
            final_text_list = convert_char_to_pinyin(text_list)

            ref_audio_len = audio.shape[-1] // hop_length
            if fix_duration is not None:
                duration = int(fix_duration * target_sample_rate / hop_length)
            else:
                # Calculate duration
                ref_text_len = len(ref_text.encode("utf-8"))
                gen_text_len = len(gen_text.encode("utf-8"))
                duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / local_speed)

            # inference
            with torch.inference_mode():
                generated, _ = model_obj.sample(
                    cond=audio,
                    text=final_text_list,
                    duration=duration,
                    steps=nfe_step,
                    cfg_strength=cfg_strength,
                    sway_sampling_coef=sway_sampling_coef,
                )
                del _

                generated = generated.to(torch.float32)  # generated mel spectrogram
                generated = generated[:, ref_audio_len:, :]
                generated = generated.permute(0, 2, 1)
                if mel_spec_type == "vocos":
                    generated_wave = vocoder.decode(generated)
                elif mel_spec_type == "bigvgan":
                    generated_wave = vocoder(generated)
                if rms < target_rms:
                    generated_wave = generated_wave * rms / target_rms

                # wav -> numpy
                generated_wave = generated_wave.squeeze().cpu().numpy()

                if streaming:
                    for j in range(0, len(generated_wave), chunk_size):
                        yield generated_wave[j : j + chunk_size], target_sample_rate
                else:
                    generated_cpu = generated[0].cpu().numpy()
                    del generated
                    yield generated_wave, generated_cpu
        except Exception as e:
            show_info(f"\n--- Error processing chunk: '{gen_text[:100]}...' ---")
            show_info(f"Exception: {e}\n")
            if skip_on_error:
                if streaming:
                    # For streaming, we might yield a special error signal or just nothing for this chunk
                    # Yielding None, None might be problematic if consumer expects (data, rate)
                    # For now, let's make it yield nothing for this failed chunk in streaming if skipped
                    return # Effectively yields nothing from this generator for this chunk
                else:
                    yield None, None # Placeholder for non-streaming
            else:
                raise e

    def format_eta(seconds):
        if seconds < 0: seconds = 0
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"

    if streaming:
        if save_intermediate_every_n_chunks > 0:
            show_info("Note: Intermediate saving is not supported in streaming mode.")
        # TODO: ETA for streaming is a bit more complex due to yield inside process_batch
        # For now, just show chunk progress for streaming
        pbar = gen_text_batches
        if progress is not None and hasattr(progress, "tqdm"):
            pbar = progress.tqdm(gen_text_batches, total=total_chunks, desc="Processing chunks")
        
        for i, gen_text in enumerate(pbar):
            current_chunk_num = i + 1
            if progress is not None:
                if hasattr(progress, "tqdm"): # Check if it's a tqdm instance
                    elapsed_time = time.time() - start_time
                    avg_time_per_chunk = elapsed_time / current_chunk_num if current_chunk_num > 0 else 0
                    eta_seconds = avg_time_per_chunk * (total_chunks - current_chunk_num) if avg_time_per_chunk > 0 else 0
                    pbar.set_description(f"Streaming chunk {current_chunk_num}/{total_chunks}, ETA: {format_eta(eta_seconds)}")
                elif callable(progress): # Check if it's gr.Progress
                    # ETA calculation for gr.Progress in streaming might be tricky if chunks yield multiple times
                    # For simplicity, updating gr.Progress description per main chunk text
                    progress(current_chunk_num / total_chunks, desc=f"Streaming chunk {current_chunk_num}/{total_chunks}")

            for chunk_data in process_batch(gen_text):
                yield chunk_data
    else: # Non-streaming path
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_batch, gen_text) for gen_text in gen_text_batches]
            
            pbar = futures
            if progress is not None and hasattr(progress, "tqdm"):
                pbar = progress.tqdm(futures, total=total_chunks, desc="Processing chunks")

            for i, future in enumerate(pbar):
                current_chunk_num = i + 1
                try:
                    result = future.result()
                except Exception as e: # Exception from process_batch if skip_on_error is False
                    show_info(f"\n--- Critical error in chunk {current_chunk_num}/{total_chunks} (processing stopped) ---")
                    # The actual text chunk is not readily available here without extra plumbing
                    # The error in process_batch (if skip_on_error=False) would have already printed details.
                    raise e


                if progress is not None:
                    elapsed_time = time.time() - start_time
                    avg_time_per_chunk = elapsed_time / current_chunk_num if current_chunk_num > 0 else 0
                    eta_seconds = avg_time_per_chunk * (total_chunks - current_chunk_num) if avg_time_per_chunk > 0 else 0
                    
                    if hasattr(progress, "tqdm"): # tqdm
                        pbar.set_description(f"Processing chunk {current_chunk_num}/{total_chunks}, ETA: {format_eta(eta_seconds)}")
                    elif callable(progress): # gr.Progress
                        progress(current_chunk_num / total_chunks, desc=f"Processing chunk {current_chunk_num}/{total_chunks}, ETA: {format_eta(eta_seconds)}")
                
                if result and result != (None, None): # Result is not a placeholder
                    generated_wave, generated_mel_spec = next(result) # process_batch yields once
                    generated_waves.append(generated_wave)
                    spectrograms.append(generated_mel_spec)

                    # Intermediate saving logic
                    if save_intermediate_every_n_chunks > 0 and \
                       output_file_path is not None and \
                       len(generated_waves) > 0 and \
                       len([w for w in generated_waves if w is not None]) % save_intermediate_every_n_chunks == 0:
                        
                        current_valid_waves = [w for w in generated_waves if w is not None]
                        if current_valid_waves: # Ensure there's something to save
                            base, ext = os.path.splitext(output_file_path)
                            intermediate_filename = f"{base}_intermediate_part_{intermediate_save_part_count}{ext}"
                            
                            # Concatenate all valid waves collected so far for this intermediate save
                            # This part needs to be careful with cross-fading if applied intermediately
                            # For simplicity, intermediate saves will be simple concatenations.
                            # Cross-fading is applied only to the final output.
                            concatenated_intermediate_wave = np.concatenate(current_valid_waves)
                            
                            try:
                                sf.write(intermediate_filename, concatenated_intermediate_wave, target_sample_rate)
                                show_info(f"\nIntermediate audio (part {intermediate_save_part_count}) saved: {intermediate_filename}")
                                intermediate_save_part_count += 1
                            except Exception as write_e:
                                show_info(f"\nFailed to save intermediate audio: {write_e}")

                elif result == (None, None) and skip_on_error: # Placeholder due to skipped error
                    generated_waves.append(None) # Keep None to maintain chunk count if needed, filtered later
                    spectrograms.append(None)
                # If result is None but not (None,None) or skip_on_error is False, it implies an issue or no output.

        valid_waves = [w for w in generated_waves if w is not None]
        valid_spectrograms = [s for s in spectrograms if s is not None]

        if not valid_waves:
            show_info("No valid audio segments generated.")
            yield None, target_sample_rate, None
            return

        if cross_fade_duration <= 0:
            # Simply concatenate
            final_wave = np.concatenate(valid_waves)
        else:
            # Combine all generated waves with cross-fading
            # This logic should apply to the final set of valid_waves
            if len(valid_waves) == 1:
                final_wave = valid_waves[0]
            else:
                final_wave = valid_waves[0]
                for idx_vw in range(1, len(valid_waves)):
                    prev_wave = final_wave
                    next_wave = valid_waves[idx_vw]

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
                    new_wave = np.concatenate(
                        [prev_wave[:-cross_fade_samples], cross_faded_overlap, next_wave[cross_fade_samples:]]
                    )
                    final_wave = new_wave
        
        if not valid_spectrograms:
            combined_spectrogram = None
        else:
            # Create a combined spectrogram
            combined_spectrogram = np.concatenate(valid_spectrograms, axis=1)

        yield final_wave, target_sample_rate, combined_spectrogram


# remove silence from generated wav


def remove_silence_for_generated_wav(filename):
    aseg = AudioSegment.from_file(filename)
    non_silent_segs = silence.split_on_silence(
        aseg, min_silence_len=1000, silence_thresh=-50, keep_silence=500, seek_step=10
    )
    non_silent_wave = AudioSegment.silent(duration=0)
    for non_silent_seg in non_silent_segs:
        non_silent_wave += non_silent_seg
    aseg = non_silent_wave
    aseg.export(filename, format="wav")


# save spectrogram


def save_spectrogram(spectrogram, path):
    plt.figure(figsize=(12, 4))
    plt.imshow(spectrogram, origin="lower", aspect="auto")
    plt.colorbar()
    plt.savefig(path)
    plt.close()
