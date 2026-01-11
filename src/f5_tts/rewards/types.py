from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class RewardInput:
    audio: torch.Tensor
    text: str | None = None
    speaker_ref: torch.Tensor | None = None
    sample_rate: int = 24000
    meta: dict[str, Any] = field(default_factory=dict)
    lang: str | None = None


@dataclass
class RewardOutput:
    total_reward: torch.Tensor
    components: dict[str, torch.Tensor] = field(default_factory=dict)
    logs: dict[str, Any] = field(default_factory=dict)


class RewardProvider:
    name = "base"
    required_extras: list[str] = []

    def setup(self, cfg: dict[str, Any] | None = None) -> None:
        return None

    def supports_language(self, lang: str | None) -> bool:
        return True

    def compute(self, batch: list[RewardInput]) -> list[RewardOutput]:
        raise NotImplementedError
