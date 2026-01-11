from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
import torchaudio

from f5_tts.rewards.types import RewardInput, RewardOutput, RewardProvider
from f5_tts.rewards.utils import RewardCache, audio_hash, resolve_device


class WeSpeakerSimProvider(RewardProvider):
    name = "wespeaker_sim"
    required_extras = ["reward_wespeaker"]

    def setup(self, cfg: dict[str, Any] | None = None) -> None:
        cfg = cfg or {}
        self.model_dir = cfg.get("model_dir", "src/rl/wespeaker/chinese")
        self.device = resolve_device(cfg.get("device", "auto"))
        self.cache = RewardCache(cfg.get("cache_dir"), enabled=cfg.get("cache_enabled", True))
        self._model = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            from wespeaker.cli.speaker import Speaker
        except Exception as exc:  # noqa: BLE001
            raise ImportError(
                "WeSpeaker is required for WeSpeakerSimProvider. "
                "Install with: pip install f5-tts[reward_wespeaker]"
            ) from exc

        class SpeakerEmbedding(Speaker):
            def extract_embedding_from_pcm(self, pcm: torch.Tensor, sample_rate: int):
                pcm = pcm.to(torch.float)
                if sample_rate != self.resample_rate:
                    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.resample_rate)
                    pcm = resampler(pcm)
                feats = self.compute_fbank(pcm, sample_rate=self.resample_rate, cmn=True)
                feats = feats.unsqueeze(0).to(self.device)
                with torch.no_grad():
                    outputs = self.model(feats)
                    outputs = outputs[-1] if isinstance(outputs, tuple) else outputs
                return outputs

        self._model = SpeakerEmbedding(self.model_dir)
        self._model.device = torch.device(self.device)

    def supports_language(self, lang: str | None) -> bool:
        return True

    def _embedding(self, audio: torch.Tensor, sample_rate: int, key_prefix: str) -> torch.Tensor:
        self._ensure_model()
        audio = audio.detach().cpu()
        if audio.ndim > 1:
            audio = audio.mean(dim=0)
        key = f"{key_prefix}_{audio_hash(audio, sample_rate)}"
        cached = self.cache.get(key)
        if cached is not None:
            return cached.to(self.device)
        embedding = self._model.extract_embedding_from_pcm(audio.unsqueeze(0), sample_rate).squeeze(0)
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
