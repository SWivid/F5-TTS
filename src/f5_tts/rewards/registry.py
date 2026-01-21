from __future__ import annotations

import importlib
from typing import Any, Type

from f5_tts.rewards.types import RewardProvider


_DEFAULT_PROVIDERS = {
    "funasr_wer": "f5_tts.rewards.providers.funasr_wer:FunASRWERProvider",
    "wespeaker_sim": "f5_tts.rewards.providers.wespeaker_sim:WeSpeakerSimProvider",
}


def _load_class(path: str) -> Type[RewardProvider]:
    if ":" in path:
        module_name, class_name = path.split(":", 1)
    else:
        module_name, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


class RewardRegistry:
    _providers: dict[str, Type[RewardProvider]] = {}

    @classmethod
    def register(cls, name: str, provider_cls: Type[RewardProvider]) -> None:
        cls._providers[name] = provider_cls

    @classmethod
    def resolve(cls, name: str) -> Type[RewardProvider]:
        if name in cls._providers:
            return cls._providers[name]
        if name in _DEFAULT_PROVIDERS:
            provider_cls = _load_class(_DEFAULT_PROVIDERS[name])
            cls.register(name, provider_cls)
            return provider_cls
        if "." in name or ":" in name:
            provider_cls = _load_class(name)
            cls.register(provider_cls.name, provider_cls)
            return provider_cls
        raise KeyError(f"Unknown reward provider '{name}'")

    @classmethod
    def create(cls, spec: str | dict[str, Any]) -> RewardProvider:
        if isinstance(spec, str):
            provider_cls = cls.resolve(spec)
            provider = provider_cls()
            provider.setup({})
            return provider

        name = spec.get("name") or spec.get("provider")
        if not name:
            raise KeyError("Reward provider spec must include 'name' or 'provider'")
        provider_cls = cls.resolve(name)
        provider = provider_cls()
        provider_cfg = spec.get("config", {})
        provider.setup(provider_cfg)
        return provider
