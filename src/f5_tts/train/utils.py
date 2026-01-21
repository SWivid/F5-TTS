from __future__ import annotations

import torch


def resolve_mixed_precision(mode: str | None) -> str | None:
    if mode is None:
        return None
    mode = str(mode).lower()
    if mode in {"no", "none"}:
        return "no"
    if mode == "auto":
        if torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                return "bf16"
            return "fp16"
        return "no"
    if mode in {"fp16", "bf16"}:
        return mode
    raise ValueError(f"Unsupported mixed_precision mode '{mode}'. Expected auto|no|fp16|bf16.")


def apply_tf32(enabled: bool) -> None:
    if not enabled or not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
