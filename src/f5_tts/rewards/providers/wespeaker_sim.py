from __future__ import annotations

import importlib.util
import os
import sys
import types
from typing import Any

import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.compliance.kaldi as kaldi

from f5_tts.rewards.types import RewardInput, RewardOutput, RewardProvider
from f5_tts.rewards.utils import RewardCache, audio_hash, resolve_device


class WeSpeakerSimProvider(RewardProvider):
    name = "wespeaker_sim"
    required_extras = ["reward_wespeaker"]

    def setup(self, cfg: dict[str, Any] | None = None) -> None:
        cfg = cfg or {}
        self.model_dir = cfg.get("model_dir", "checkpoints/wespeaker/cnceleb_resnet34/cnceleb_resnet34")
        self.device = resolve_device(cfg.get("device", "auto"))
        cache_enabled = cfg.get("cache_enabled", True)
        cache_dir = cfg.get("cache_dir") if cache_enabled else None
        self.cache = RewardCache(cache_dir, enabled=cache_enabled)
        self._model = None
        self._resample_rate = 16000
        self._fbank_args = {
            "num_mel_bins": 80,
            "frame_length": 25,
            "frame_shift": 10,
            "dither": 0.0,
            "window_type": "hamming",
        }

    def _ensure_wespeaker_package(self) -> None:
        spec = importlib.util.find_spec("wespeaker")
        if spec is None or not spec.submodule_search_locations:
            raise ImportError(
                "WeSpeaker is required for WeSpeakerSimProvider. Install with: pip install -e '.[reward_wespeaker]'"
            )
        if "wespeaker" not in sys.modules:
            # Avoid wespeaker.__init__ side effects; we only need package paths.
            pkg = types.ModuleType("wespeaker")
            pkg.__path__ = list(spec.submodule_search_locations)
            sys.modules["wespeaker"] = pkg

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        self._ensure_wespeaker_package()
        # Treat any import/setup error as missing optional deps to keep the user-facing hint consistent.
        try:
            import yaml
            from wespeaker.models.speaker_model import get_speaker_model
            from wespeaker.utils.checkpoint import load_checkpoint
        except Exception as exc:  # noqa: BLE001
            raise ImportError(
                "WeSpeaker dependencies are missing. Install with: pip install -e '.[reward_wespeaker]'"
            ) from exc

        config_path = os.path.join(self.model_dir, "config.yaml")
        checkpoint_path = os.path.join(self.model_dir, "avg_model.pt")
        if not os.path.exists(config_path) or not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"WeSpeaker model_dir missing config.yaml or avg_model.pt: {self.model_dir}")

        with open(config_path, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        dataset_args = config.get("dataset_args", {})
        frontend_type = dataset_args.get("frontend", "fbank")
        if frontend_type != "fbank":
            raise RuntimeError(f"WeSpeaker frontend '{frontend_type}' requires extra dependencies.")

        fbank_args = dataset_args.get("fbank_args", {})
        self._fbank_args = {
            "num_mel_bins": fbank_args.get("num_mel_bins", 80),
            "frame_length": fbank_args.get("frame_length", 25),
            "frame_shift": fbank_args.get("frame_shift", 10),
            "dither": fbank_args.get("dither", 0.0),
            "window_type": fbank_args.get("window_type", "hamming"),
        }
        self._resample_rate = dataset_args.get("resample_rate", 16000)
        model = get_speaker_model(config["model"])(**config.get("model_args", {}))
        load_checkpoint(model, checkpoint_path)
        model.eval()
        self._model = model.to(self.device)

    def supports_language(self, lang: str | None) -> bool:
        return True

    def _compute_fbank(self, pcm: torch.Tensor, sample_rate: int) -> torch.Tensor:
        feats = kaldi.fbank(
            pcm,
            num_mel_bins=self._fbank_args["num_mel_bins"],
            frame_length=self._fbank_args["frame_length"],
            frame_shift=self._fbank_args["frame_shift"],
            sample_frequency=sample_rate,
            dither=self._fbank_args["dither"],
            window_type=self._fbank_args["window_type"],
        )
        return feats - feats.mean(dim=0)

    def _embedding(self, audio: torch.Tensor, sample_rate: int, key_prefix: str) -> torch.Tensor:
        self._ensure_model()
        audio = audio.detach().cpu()
        if audio.ndim > 1:
            audio = audio.mean(dim=0)
        key = f"{key_prefix}_{audio_hash(audio, sample_rate)}"
        cached = self.cache.get(key)
        if cached is not None:
            return cached.to(self.device)
        pcm = audio.unsqueeze(0).to(torch.float32)
        if sample_rate != self._resample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self._resample_rate)
            pcm = resampler(pcm)
        feats = self._compute_fbank(pcm, sample_rate=self._resample_rate).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self._model(feats)
            outputs = outputs[-1] if isinstance(outputs, tuple) else outputs
        embedding = outputs.squeeze(0)
        self.cache.set(key, embedding.detach().cpu())
        return embedding

    def compute(self, batch: list[RewardInput]) -> list[RewardOutput]:
        if not batch:
            return []

        outputs: list[RewardOutput] = []
        for item in batch:
            if item.speaker_ref is None:
                raise ValueError("WeSpeakerSimProvider requires speaker_ref in RewardInput")
            emb_gen = self._embedding(item.audio, item.sample_rate, "gen")
            emb_ref = self._embedding(item.speaker_ref, item.sample_rate, "ref")
            sim = F.cosine_similarity(emb_gen, emb_ref, dim=-1)
            total = sim.to(dtype=torch.float32).cpu()
            outputs.append(
                RewardOutput(
                    total_reward=total,
                    components={"sim": total},
                    logs={},
                )
            )
        return outputs
