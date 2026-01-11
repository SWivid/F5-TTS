from __future__ import annotations

from typing import Any

import torch

from f5_tts.rewards.registry import RewardRegistry
from f5_tts.rewards.types import RewardInput, RewardOutput, RewardProvider


class RewardCombiner:
    def __init__(
        self,
        providers: list[RewardProvider],
        weights: list[float] | None = None,
        mode: str = "sum",
        eps: float = 1e-4,
    ):
        if mode not in ("sum", "normalized_sum", "rank_shaping"):
            raise ValueError(f"mode must be 'sum', 'normalized_sum', or 'rank_shaping', got {mode}")
        self.providers = providers
        self.weights = weights or [1.0] * len(providers)
        if len(self.weights) != len(self.providers):
            raise ValueError("weights must match providers length")
        self.mode = mode
        self.eps = eps

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "RewardCombiner":
        providers_cfg = cfg.get("providers", [])
        providers = []
        weights = []
        for item in providers_cfg:
            provider = RewardRegistry.create(item)
            providers.append(provider)
            weights.append(item.get("weight", 1.0))
        return cls(
            providers=providers,
            weights=weights,
            mode=cfg.get("mode", "sum"),
            eps=cfg.get("eps", 1e-4),
        )

    def _stack_provider_rewards(self, outputs: list[list[RewardOutput]]) -> list[torch.Tensor]:
        reward_tensors = []
        for provider_outputs in outputs:
            reward_tensors.append(torch.stack([out.total_reward for out in provider_outputs], dim=0))
        return reward_tensors

    def _rank_shaping(self, rewards: torch.Tensor) -> torch.Tensor:
        if rewards.numel() <= 1:
            return torch.zeros_like(rewards)
        ranks = torch.argsort(torch.argsort(rewards))
        scaled = ranks.float() / max(1, rewards.numel() - 1)
        return scaled - scaled.mean()

    def compute(self, batch: list[RewardInput]) -> list[RewardOutput]:
        if not batch:
            return []
        provider_outputs = [provider.compute(batch) for provider in self.providers]
        reward_vectors = self._stack_provider_rewards(provider_outputs)

        shaped = []
        for idx, rewards in enumerate(reward_vectors):
            if self.mode == "sum":
                shaped.append(rewards)
            elif self.mode == "normalized_sum":
                mean = rewards.mean()
                std = rewards.std(unbiased=False)
                shaped.append((rewards - mean) / (std + self.eps))
            else:
                shaped.append(self._rank_shaping(rewards))

        combined = torch.zeros_like(shaped[0])
        for weight, shaped_rewards in zip(self.weights, shaped):
            combined = combined + shaped_rewards * weight

        outputs: list[RewardOutput] = []
        for i in range(len(batch)):
            components: dict[str, torch.Tensor] = {}
            logs: dict[str, Any] = {}
            for provider, provider_reward, provider_output in zip(self.providers, reward_vectors, provider_outputs):
                components[f"{provider.name}.total"] = provider_reward[i]
                components.update(
                    {f"{provider.name}.{k}": v for k, v in provider_output[i].components.items()}
                )
                if provider_output[i].logs:
                    logs[provider.name] = provider_output[i].logs
            outputs.append(
                RewardOutput(
                    total_reward=combined[i],
                    components=components,
                    logs=logs,
                )
            )
        return outputs
