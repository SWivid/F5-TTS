from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

from .splitter import ArticleSplitter, SentenceSegment
from .generator import AudioGenerator, VoiceConfig  # type: ignore
from .concatenator import FileConcatenator
from .config import Config, VoiceConfig as VoiceCfg
import re
import hashlib


def _sanitize_for_filename(text: str, max_len: int = 60) -> str:
    # Basic slugify: remove punctuation, lowercase, spaces to underscores
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r"[^A-Za-z0-9\s]", "", text)
    text = text.strip().lower()
    text = re.sub(r"\s+", "_", text)
    if max_len and len(text) > max_len:
        text = text[:max_len]
    return text
def slugify_text(text: str, max_len: int = 60) -> str:
    return _sanitize_for_filename(text, max_len=max_len)
from pydub import AudioSegment


def _get_ref_text(ref_audio: str, voice_cfg: VoiceCfg | None = None) -> str:
    """Load ref_text from config or from companion .txt file."""
    ref_text = voice_cfg.ref_text if voice_cfg else ""
    if not ref_text:
        txt = os.path.splitext(ref_audio)[0] + ".txt"
        if os.path.exists(txt):
            try:
                ref_text = open(txt, "r", encoding="utf-8").read().strip()
            except Exception:
                ref_text = ""
    if not ref_text:
        root_speech = Path(__file__).resolve().parents[1] / "speech.txt"
        if root_speech.exists():
            try:
                ref_text = open(root_speech, "r", encoding="utf-8").read().strip()
            except Exception:
                ref_text = ""
    return ref_text or ""


def _apply_polyphone_replacements(text: str, polyphone_dict: Dict[str, str] | None) -> str:
    """Apply polyphone replacements using homophone characters.

    Example: "偏好" -> "偏浩" (hao4 instead of hao3)
    """
    if not polyphone_dict:
        return text

    result = text
    for word, replacement in polyphone_dict.items():
        if word in result:
            result = result.replace(word, replacement)
    return result


def _convert_nums_to_chinese(text: str) -> str:
    """Convert Arabic numerals to Chinese characters using cn2an.

    Example: "123" -> "一二三", "2024年" -> "二零二四年", "99.9" -> "九十九点九"
    """
    import cn2an
    try:
        # cn2an.transform with "an2cn" converts all numbers in text to Chinese
        return cn2an.transform(text, "an2cn")
    except Exception:
        # Fallback: return original text if cn2an fails
        return text


class GenerationPipeline:
    def __init__(self, config: Config, workers: int = 4):
        self.config = config
        self.workers = workers
        self.splitter = ArticleSplitter(max_length=config.max_sentence_length)
        self.audio_gen = None
        self.concater = FileConcatenator()
        self.voices = config.voices
        self._gpu_lock = threading.Lock()  # Protect GPU inference on Metal

    def _format_duration(self, seconds: float) -> str:
        """Format duration in seconds to human-readable string."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        if hours > 0:
            return f"{hours}h {minutes}m {secs:.2f}s"
        elif minutes > 0:
            return f"{minutes}m {secs:.2f}s"
        else:
            return f"{secs:.2f}s"

    def _load_article(self) -> str:
        path = Path(self.config.input_article)
        if not path.exists():
            raise FileNotFoundError(f"Article not found: {self.config.input_article}")
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def _get_audio_path(self, audio_dir: Path, voice_name: str | None, text: str, speed: float | None = None) -> Path:
        """Generate deterministic path for audio file based on text content and speed.

        Uses SHA1 hash to ensure cache hits even when text or speed changes.
        """
        voice = voice_name or "main"
        # Include speed in cache key if specified
        cache_key = f"{voice}_{text}_{speed if speed is not None else ''}"
        hash_suffix = hashlib.sha1(cache_key.encode('utf-8')).hexdigest()
        return audio_dir / f"{voice}_{hash_suffix}.wav"

    def _postprocess_audio(self, audio_path: str) -> float:
        """Add silence tail and fade out to audio file."""
        try:
            seg_audio = AudioSegment.from_wav(audio_path)
            tail_ms = 150
            fade_ms = 30 if len(seg_audio) > 2 * 30 else 0
            seg_audio = seg_audio + AudioSegment.silent(duration=tail_ms)
            if fade_ms:
                seg_audio = seg_audio.fade_out(fade_ms)
            seg_audio.export(audio_path, format="wav")
            return len(seg_audio) / 1000.0
        except Exception:
            return len(AudioSegment.from_wav(audio_path)) / 1000.0

    def _generate_segment(self, seg: SentenceSegment, speech_types: Dict[str, Tuple[str, str, float]], audio_dir: Path) -> Tuple[int, str, float, str, dict]:
        """Generate audio for a single segment (runs in worker thread).

        Returns: (index, audio_path, duration, original_text, params_dict)
        """
        ref_audio, ref_text, v_speed = speech_types.get(seg.voice_name, speech_types.get("main"))

        # Use segment-level speed if specified, otherwise fall back to voice/config speed
        final_speed = seg.speed if seg.speed is not None else v_speed

        # Get voice-level parameters (override global if set)
        voice_cfg = self.voices.get(seg.voice_name) if self.voices else None
        nfe_step = voice_cfg.nfe_step if voice_cfg and voice_cfg.nfe_step else self.config.nfe_step
        cfg_strength = voice_cfg.cfg_strength if voice_cfg and voice_cfg.cfg_strength else self.config.cfg_strength
        target_rms = voice_cfg.target_rms if voice_cfg and voice_cfg.target_rms else self.config.target_rms

        params = {
            "speed": final_speed,
            "nfe_step": nfe_step,
            "cfg_strength": cfg_strength,
            "target_rms": target_rms,
        }

        # Cache key based on original text and speed
        audio_path = self._get_audio_path(audio_dir, seg.voice_name, seg.text, final_speed)

        if audio_path.exists():
            duration = self._postprocess_audio(str(audio_path))
            return seg.index, str(audio_path), duration, seg.text, params

        # Generate with temporary file to avoid conflicts
        with tempfile.NamedTemporaryFile(suffix=".wav", dir=audio_dir, delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Apply polyphone replacements (homophone substitution)
            gen_text = _apply_polyphone_replacements(seg.text, self.config.polyphone_dict)
            # Convert Arabic numerals to Chinese characters
            gen_text = _convert_nums_to_chinese(gen_text)

            # Use lock for GPU inference on Metal (not thread-safe)
            with self._gpu_lock:
                wav, sr, spec = self.audio_gen._tts.infer(
                    ref_file=ref_audio,
                    ref_text=ref_text,
                    gen_text=gen_text,
                    file_wave=tmp_path,
                    nfe_step=nfe_step,
                    cfg_strength=cfg_strength,
                    speed=final_speed,
                    target_rms=target_rms,
                )
            # Rename to final path
            os.rename(tmp_path, str(audio_path))
            duration = self._postprocess_audio(str(audio_path))
            # Return: index, path, duration, original_text, params
            return seg.index, str(audio_path), duration, seg.text, params
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    def run(self) -> Tuple[str, str]:
        total_start_time = time.time()
        article_text = self._load_article()
        segments = self.splitter.split(article_text)
        if not segments:
            raise ValueError("No segments produced from article.")

        # Build voices map
        referenced = set(s.voice_name for s in segments if s.voice_name)
        voices_for_tts = {}
        for vkey in referenced:
            if self.voices and vkey in self.voices:
                voices_for_tts[vkey] = self.voices[vkey]
        if not voices_for_tts:
            if self.voices and "main" in self.voices:
                voices_for_tts["main"] = self.voices["main"]

        # Build TTS config and initialize model
        from .config import Config as TTSConfig
        tts_config = TTSConfig(
            input_article=self.config.input_article,
            output_dir=self.config.output_dir,
            max_sentence_length=self.config.max_sentence_length,
            model_name=self.config.model_name,
            nfe_step=self.config.nfe_step,
            cfg_strength=self.config.cfg_strength,
            speed=self.config.speed,
            voices=voices_for_tts,
        )
        self.audio_gen = AudioGenerator(tts_config)
        self.audio_gen.initialize_model()

        if self.audio_gen._tts is None:
            raise RuntimeError("Failed to initialize TTS model; aborting generation.")

        # Output directories
        audio_dir = Path(self.config.output_dir) / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        # Build speech types map (ref_audio, ref_text, speed per voice)
        unique_voices = sorted({s.voice_name for s in segments if s.voice_name})
        use_multispeech = len(unique_voices) > 1
        speech_types: Dict[str, Tuple[str, str, float]] = {}

        for v in unique_voices:
            vc = (self.config.voices or {}).get(v)
            if vc is None:
                vc = (self.config.voices or {}).get("main")
            if vc is None:
                raise RuntimeError(f"No reference audio configured for voice '{v}'")
            ref_text = _get_ref_text(vc.ref_audio, vc)
            speech_types[v] = (vc.ref_audio, ref_text, vc.speed if vc.speed is not None else self.config.speed)

        # Handle single voice fallback for missing voices
        default_ref = speech_types.get("main")
        if default_ref is None and speech_types:
            default_ref = list(speech_types.values())[0]

        # Generate all segments
        # Collect segment metadata: index -> (path, duration, text, voice_name, params)
        index_to_audio: Dict[int, Tuple[str, float, str, str, dict]] = {}
        total_segments = len(segments)

        if use_multispeech and len(segments) > 1:
            # Parallel generation for multi-voice mode
            completed = 0
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                futures = {
                    executor.submit(self._generate_segment, seg, speech_types, audio_dir): seg
                    for seg in segments
                }
                for future in as_completed(futures):
                    seg = futures[future]
                    completed += 1
                    print(f"\rProgress: {completed}/{total_segments} ({completed*100//total_segments}%)", end="", flush=True)
                    try:
                        idx, path, duration, text, params = future.result()
                        index_to_audio[idx] = (path, duration, text, seg.voice_name or "main", params)
                    except Exception as e:
                        print(f"\nError generating segment {seg.index}: {e}")
                        raise
            print()  # New line after progress
        else:
            # Sequential for single voice or single segment
            for i, seg in enumerate(segments, 1):
                print(f"\rProgress: {i}/{total_segments} ({i*100//total_segments}%)", end="", flush=True)
                idx, path, duration, text, params = self._generate_segment(seg, speech_types or {"main": default_ref}, audio_dir)
                index_to_audio[idx] = (path, duration, text, seg.voice_name or "main", params)
            print()  # New line after progress

        # Sort by index to maintain order
        sorted_indices = sorted(index_to_audio.keys())
        per_segment_audio_paths = [index_to_audio[i][0] for i in sorted_indices]

        # Concatenate all audio and build metadata
        final_audio = Path(self.config.output_dir) / "final_audio.wav"

        # Build segment metadata with timing info
        segments_metadata: List[Dict] = []
        current_time = 0.0

        if per_segment_audio_paths:
            from pydub import AudioSegment as _AS
            combined = _AS.empty()
            last_path = None

            for idx in sorted_indices:
                path, duration, text, voice_name, params = index_to_audio[idx]
                start_time = current_time
                end_time = current_time + duration

                # Add to combined audio
                if last_path is not None and path == last_path:
                    combined += _AS.silent(duration=200)
                    current_time += 0.2  # Add 200ms silence gap
                    end_time = current_time

                combined += _AS.from_wav(path)
                last_path = path

                # Build segment metadata with params
                segments_metadata.append({
                    "index": idx,
                    "voice": voice_name,
                    "text": text,
                    "start_time": round(start_time, 3),
                    "end_time": round(end_time, 3),
                    "duration": round(duration, 3),
                    "audio_file": Path(path).name,
                    "speed": params["speed"],
                    "nfe_step": params["nfe_step"],
                    "cfg_strength": params["cfg_strength"],
                    "target_rms": params["target_rms"],
                })

                current_time += duration

            combined.export(str(final_audio), format="wav")

        # Generate metadata JSON file
        metadata = {
            "source_file": self.config.input_article,
            "output_audio": "final_audio.wav",
            "total_duration": round(current_time, 3),
            "segment_count": len(segments_metadata),
            "created_at": datetime.now().isoformat(),
            "model": self.config.model_name,
            "speed": self.config.speed,
            "nfe_step": self.config.nfe_step,
            "cfg_strength": self.config.cfg_strength,
            "target_rms": self.config.target_rms,
            "segments": segments_metadata,
        }

        metadata_path = Path(self.config.output_dir) / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        print(f"Metadata saved to: {metadata_path}")

        # Print total generation time
        total_time = time.time() - total_start_time
        print(f"\n========================================")
        print(f"Total generation time: {self._format_duration(total_time)}")
        print("========================================")

        return str(final_audio), ""
