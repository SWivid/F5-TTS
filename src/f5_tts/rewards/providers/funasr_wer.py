from __future__ import annotations

from typing import Any

import torch
import torchaudio

from f5_tts.rewards.types import RewardInput, RewardOutput, RewardProvider
from f5_tts.rewards.utils import RewardCache, audio_hash, resolve_device


def _edit_distance(r: list[str], h: list[str]) -> int:
    d = torch.zeros((len(r) + 1, len(h) + 1), dtype=torch.int32)
    d[:, 0] = torch.arange(len(r) + 1)
    d[0, :] = torch.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                d[i, j] = d[i - 1, j - 1]
            else:
                substitute = d[i - 1, j - 1] + 1
                insert = d[i, j - 1] + 1
                delete = d[i - 1, j] + 1
                d[i, j] = min(substitute, insert, delete)
    return int(d[len(r), len(h)].item())


def _wer(ref: str, hyp: str, mode: str = "word") -> float:
    if mode == "word":
        r = ref.split()
        h = hyp.split()
    elif mode == "char":
        r = list(ref)
        h = list(hyp)
    else:
        raise ValueError(f"Unsupported WER mode '{mode}'. Expected 'word' or 'char'.")
    if not r:
        return 0.0 if not h else 1.0
    return _edit_distance(r, h) / max(1, len(r))


class FunASRWERProvider(RewardProvider):
    name = "funasr_wer"
    required_extras = ["reward_funasr"]

    def setup(self, cfg: dict[str, Any] | None = None) -> None:
        cfg = cfg or {}
        self.model_id = cfg.get("model_id", "FunAudioLLM/SenseVoiceSmall")
        self.lang = cfg.get("lang", "auto")
        self.wer_mode = cfg.get("wer_mode", "char")
        self.ref_source = cfg.get("ref_source", "text")
        if self.ref_source not in {"text", "audio"}:
            raise ValueError(f"Unsupported ref_source '{self.ref_source}'. Expected 'text' or 'audio'.")
        self.device = resolve_device(cfg.get("device", "auto"))
        self.batch_size = cfg.get("batch_size", 8)
        self.cache = RewardCache(cfg.get("cache_dir"), enabled=cfg.get("cache_enabled", True))
        self._model = None
        self._postprocess = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        # Treat any import/setup error as missing optional deps to keep the user-facing hint consistent.
        try:
            from funasr import AutoModel
            from funasr.utils.postprocess_utils import rich_transcription_postprocess
        except Exception as exc:  # noqa: BLE001
            raise ImportError(
                "FunASR is required for FunASRWERProvider. Install with: pip install f5-tts[reward_funasr]"
            ) from exc
        self._postprocess = rich_transcription_postprocess
        self._model = AutoModel(model=self.model_id, device=self.device, disable_update=True)

    def supports_language(self, lang: str | None) -> bool:
        return True

    def _prepare_audio(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        audio = audio.detach().cpu()
        if audio.ndim > 1:
            audio = audio.mean(dim=0)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            audio = resampler(audio)
        return audio

    def _infer_texts(self, audios: list[torch.Tensor], lang: str) -> list[str]:
        self._ensure_model()
        batch_size = min(self.batch_size, len(audios)) if self.batch_size else len(audios)
        results = self._model.inference(
            input=audios,
            cache={},
            language=lang,
            use_itn=True,
            disable_pbar=True,
            batch_size=batch_size,
        )
        return [self._postprocess(res["text"]) for res in results]

    def compute(self, batch: list[RewardInput]) -> list[RewardOutput]:
        if not batch:
            return []

        lang = self.lang
        batch_langs = [item.lang for item in batch if item.lang]
        if batch_langs and all(item.lang == batch_langs[0] for item in batch if item.lang):
            lang = batch_langs[0]

        gen_audios = []
        gen_keys = []
        for item in batch:
            prepared = self._prepare_audio(item.audio, item.sample_rate)
            key = f"gen_{self.model_id}_{lang}_{audio_hash(prepared, 16000)}"
            gen_audios.append(prepared)
            gen_keys.append(key)

        gen_texts: list[str] = []
        pending_audios = []
        pending_indices = []
        for idx, key in enumerate(gen_keys):
            cached = self.cache.get(key)
            if cached is None:
                pending_audios.append(gen_audios[idx])
                pending_indices.append(idx)
                gen_texts.append("")
            else:
                gen_texts.append(cached)

        if pending_audios:
            inferred = self._infer_texts(pending_audios, lang)
            for idx, text in zip(pending_indices, inferred):
                gen_texts[idx] = text
                self.cache.set(gen_keys[idx], text)

        ref_texts: list[str] = []
        ref_audios = []
        ref_keys = []
        ref_indices = []
        for idx, item in enumerate(batch):
            use_audio_ref = self.ref_source == "audio"
            if not use_audio_ref and item.text:
                ref_texts.append(item.text)
                continue
            if item.speaker_ref is not None:
                ref_audio = self._prepare_audio(item.speaker_ref, item.sample_rate)
                key = f"ref_{self.model_id}_{lang}_{audio_hash(ref_audio, 16000)}"
                cached = self.cache.get(key)
                if cached is None:
                    ref_texts.append("")
                    ref_audios.append(ref_audio)
                    ref_keys.append(key)
                    ref_indices.append(idx)
                else:
                    ref_texts.append(cached)
            elif item.text:
                ref_texts.append(item.text)
            else:
                ref_texts.append("")

        if ref_audios:
            inferred_ref = self._infer_texts(ref_audios, lang)
            for idx, text, key in zip(ref_indices, inferred_ref, ref_keys):
                ref_texts[idx] = text
                self.cache.set(key, text)

        outputs: list[RewardOutput] = []
        for ref_text, gen_text in zip(ref_texts, gen_texts):
            wer_value = _wer(ref_text, gen_text, mode=self.wer_mode) if ref_text else 1.0
            acc = 1.0 - wer_value
            total = torch.tensor(acc, dtype=torch.float32)
            outputs.append(
                RewardOutput(
                    total_reward=total,
                    components={
                        "wer": torch.tensor(wer_value, dtype=torch.float32),
                        "acc": total,
                    },
                    logs={"ref_text": ref_text, "gen_text": gen_text},
                )
            )
        return outputs
