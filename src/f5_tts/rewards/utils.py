from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import torch


def audio_hash(audio: torch.Tensor, sample_rate: int | None = None) -> str:
    if audio.is_cuda:
        audio = audio.detach().cpu()
    data = audio.contiguous().numpy().tobytes()
    if sample_rate is not None:
        data = str(sample_rate).encode("ascii") + data
    return hashlib.md5(data).hexdigest()


class RewardCache:
    def __init__(self, cache_dir: str | None = None, enabled: bool = True):
        self.enabled = enabled
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.memory: dict[str, Any] = {}
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

    def _path_for_key(self, key: str) -> Path:
        assert self.cache_dir is not None
        return self.cache_dir / f"{key}.pt"

    def get(self, key: str) -> Any | None:
        if not self.enabled:
            return None
        if key in self.memory:
            return self.memory[key]
        if self.cache_dir:
            path = self._path_for_key(key)
            if path.exists():
                value = torch.load(path)
                self.memory[key] = value
                return value
        return None

    def set(self, key: str, value: Any) -> None:
        if not self.enabled:
            return
        self.memory[key] = value
        if self.cache_dir:
            torch.save(value, self._path_for_key(key))


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
