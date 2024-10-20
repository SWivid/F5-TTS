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

class F5TTS:
    def __init__(self, model_type="F5-TTS", ckpt_file="", vocab_file="", ode_method="euler", use_ema=True, local_vocoder=False, device=None):
        # Initialize parameters
        self.final_wave = None
        self.target_sample_rate = 24000
        self.n_mel_channels = 100
        self.hop_length = 256
        self.target_rms = 0.1

        # Set device
        self.device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        
        # Load models
        self.load_vecoder_model(local_vocoder)
        self.load_ema_model(model_type, ckpt_file, vocab_file, ode_method, use_ema)

    def load_vecoder_model(self, load_vocoder_from_local):
        if load_vocoder_from_local:
            vocos_local_path = "../checkpoints/charactr/vocos-mel-24khz"
            print(f"Loading vocoder from local path {vocos_local_path}")
            self.vocos = Vocos.from_hparams(f"{vocos_local_path}/config.yaml")
            state_dict = torch.load(f"{vocos_local_path}/pytorch_model.bin", map_location=self.device)
            self.vocos.load_state_dict(state_dict)
        else:
            path_huggingface = "charactr/vocos-mel-24khz"
            print(f"Downloading vocoder from huggingface {path_huggingface}")
            self.vocos = Vocos.from_pretrained(path_huggingface)
        self.vocos.eval()

    def load_ema_model(self, model_type, ckpt_file, vocab_file, ode_method, use_ema):
        if model_type == "F5-TTS":
            if not ckpt_file:
                ckpt_file = str(cached_path("hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.safetensors"))
            model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
            model_cls = DiT
        elif model_type == "E2-TTS":
            if not ckpt_file:
                ckpt_file = str(cached_path("hf://SWivid/E2-TTS/E2TTS_Base/model_1200000.safetensors"))
            model_cfg = dict(dim=1024, depth=24, heads=16, ff_mult=4)
            model_cls = UNetT
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.ema_model = self.load_model(model_cls, model_cfg, ckpt_file, vocab_file, ode_method, use_ema)

    def load_model(self, model_cls, model_cfg, ckpt_path, file_vocab, ode_method, use_ema):
        tokenizer = "pinyin" if file_vocab == "" else "custom"
        file_vocab = file_vocab or "Emilia_ZH_EN"
        
        vocab_char_map, vocab_size = get_tokenizer(file_vocab, tokenizer)
        model = CFM(
            transformer=model_cls(
                **model_cfg, text_num_embeds=vocab_size, mel_dim=self.n_mel_channels
            ),
            mel_spec_kwargs=dict(
                target_sample_rate=self.target_sample_rate,
                n_mel_channels=self.n_mel_channels,
                hop_length=self.hop_length,
            ),
            odeint_kwargs=dict(method=ode_method),
            vocab_char_map=vocab_char_map,
        ).to(self.device)

        return load_checkpoint(model, ckpt_path, self.device, use_ema=use_ema)

    def chunk_text(self, text, max_chars=135):
        chunks = []
        current_chunk = ""
        sentences = re.split(r'(?<=[;:,.!?])\s+|(?<=[；：，。！？])', text)

        for sentence in sentences:
            if len(current_chunk.encode('utf-8')) + len(sentence.encode('utf-8')) <= max_chars:
                current_chunk += sentence + (" " if sentence and len(sentence[-1].encode('utf-8')) == 1 else "")
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + (" " if sentence and len(sentence[-1].encode('utf-8')) == 1 else "")

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def process_voice(self, ref_audio_orig, ref_text):
        print("Converting", ref_audio_orig)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            aseg = AudioSegment.from_file(ref_audio_orig)
    
            non_silent_segs = silence.split_on_silence(aseg, min_silence_len=1000, silence_thresh=-50, keep_silence=1000)
            non_silent_wave = AudioSegment.silent(duration=0)
            for non_silent_seg in non_silent_segs:
                non_silent_wave += non_silent_seg
            aseg = non_silent_wave
    
            if len(aseg) > 15000:
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
                device=self.device,
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
        return ref_audio, ref_text    
    
    def remove_silence_from_audio(self, audio_path):
        aseg = AudioSegment.from_file(audio_path)
        non_silent_segs = silence.split_on_silence(aseg, min_silence_len=1000, silence_thresh=-50, keep_silence=500)
        non_silent_wave = AudioSegment.silent(duration=0)
        for non_silent_seg in non_silent_segs:
            non_silent_wave += non_silent_seg
        non_silent_wave.export(audio_path, format="wav")
        print("Silence removed from audio.")

    def export_wav(self,wav, file_wave, remove_silence=False):
        if remove_silence:
            self.remove_silence_from_audio(file_wave)

        sf.write(file_wave, wav, self.target_sample_rate)
        
    def export_spectrogram(self,spect, file_spect):   
        save_spectrogram(spect, file_spect)
   
    def infer_batch(self, ref_audio, ref_text, gen_text_batches, sway_sampling_coef=-1, cfg_strength=2, nfe_step=32, speed=1.0, fix_duration=None, cross_fade_duration=0.15):
        audio, sr = ref_audio
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
	    
        rms = torch.sqrt(torch.mean(torch.square(audio)))
        if rms < self.target_rms:
            audio = audio * self.target_rms / rms
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            audio = resampler(audio)
        audio = audio.to(self.device)
	    
        generated_waves = []
        spectrograms = []
	    
        if len(ref_text[-1].encode('utf-8')) == 1:
            ref_text = ref_text + " "
        for gen_text in tqdm.tqdm(gen_text_batches):
            # Prepare the text
            text_list = [ref_text + gen_text]
            final_text_list = convert_char_to_pinyin(text_list)
	    
            # Calculate duration
            ref_audio_len = audio.shape[-1] // self.hop_length
            ref_text_len = len(ref_text.encode('utf-8'))
            gen_text_len = len(gen_text.encode('utf-8'))
            duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / speed)
	    
            # inference
            with torch.inference_mode():
                generated, _ = self.ema_model.sample(
                    cond=audio,
                    text=final_text_list,
                    duration=duration,
                    steps=nfe_step,
                    cfg_strength=cfg_strength,
                    sway_sampling_coef=sway_sampling_coef,
                )
	    
            generated = generated.to(torch.float32)
            generated = generated[:, ref_audio_len:, :]
            generated_mel_spec = rearrange(generated, "1 n d -> 1 d n")
            generated_wave = self.vocos.decode(generated_mel_spec.cpu())
            if rms < self.target_rms:
                generated_wave = generated_wave * rms / self.target_rms
	    
            # wav -> numpy
            generated_wave = generated_wave.squeeze().cpu().numpy()
            
            generated_waves.append(generated_wave)
            spectrograms.append(generated_mel_spec[0].cpu().numpy())
	    
        # Combine all generated waves with cross-fading
        final_wave = self.combine_waves_with_crossfade(generated_waves, cross_fade_duration)
	    
        # Create a combined spectrogram
        combined_spectrogram = np.concatenate(spectrograms, axis=1)
	    
        return final_wave, self.target_sample_rate, combined_spectrogram

    def combine_waves_with_crossfade(self, generated_waves, cross_fade_duration):
        if cross_fade_duration <= 0:
            return np.concatenate(generated_waves)

        final_wave = generated_waves[0]
        for i in range(1, len(generated_waves)):
            prev_wave = final_wave
            next_wave = generated_waves[i]

            cross_fade_samples = int(cross_fade_duration * self.target_sample_rate)
            cross_fade_samples = min(cross_fade_samples, len(prev_wave), len(next_wave))

            if cross_fade_samples <= 0:
                final_wave = np.concatenate([prev_wave, next_wave])
                continue

            prev_overlap = prev_wave[-cross_fade_samples:]
            next_overlap = next_wave[:cross_fade_samples]

            fade_out = np.linspace(1, 0, cross_fade_samples)
            fade_in = np.linspace(0, 1, cross_fade_samples)

            cross_faded_overlap = prev_overlap * fade_out + next_overlap * fade_in

            final_wave = np.concatenate([
                prev_wave[:-cross_fade_samples],
                cross_faded_overlap,
                next_wave[cross_fade_samples:]
            ])

        return final_wave
    
    def infer(self, ref_file, ref_text, gen_text, sway_sampling_coef=-1, cfg_strength=2, nfe_step=32, speed=1.0, fix_duration=None, remove_silence=False, file_wave=None, file_spect=None, cross_fade_duration=0.15):
 
        # Ensure ref_text ends with ". " or "。"
        if not ref_text.endswith(". ") and not ref_text.endswith("。"):
            ref_text += ". " if ref_text.endswith(".") else ". "
	    
        # Split the input text into batches
        audio, sr = torchaudio.load(ref_file)
        max_chars = int(len(ref_text.encode('utf-8')) / (audio.shape[-1] / sr) * (25 - audio.shape[-1] / sr))
        gen_text_batches = self.chunk_text(gen_text, max_chars=max_chars)
        for i, gen_text in enumerate(gen_text_batches):
            print(f'gen_text {i}', gen_text)

        wav,sr,spect = self.infer_batch((audio, sr), ref_text, gen_text_batches, sway_sampling_coef, cfg_strength, nfe_step, speed, fix_duration, cross_fade_duration)
        
        if file_wave is not None:
           self.export_wav(wav,file_wave,remove_silence)
            
        if file_spect is not None:
           self.export_spectrogram(spect,file_spect)

        return wav,sr,spect   