"""Device resolution shared by the inference, API, socket and finetune entry points."""

import torch


def is_npu_available() -> bool:
    """Whether an Ascend NPU is usable through a ``torch_npu`` build.

    ``torch.npu`` only exists once ``torch_npu`` has been imported, so the import is
    attempted lazily and stays invisible to CUDA/XPU/MPS/CPU-only installs.
    """
    if not hasattr(torch, "npu"):
        try:
            import torch_npu  # noqa: F401
        except ImportError:
            return False
    npu = getattr(torch, "npu", None)
    return npu is not None and bool(npu.is_available())


def get_default_device() -> str:
    """Return the accelerator to use when the caller does not pick one explicitly.

    ``torch.accelerator`` (PyTorch >= 2.6) reports whichever backend is registered and
    available, so CUDA, Ascend NPU, XPU and future backends are covered without a
    per-backend branch. Builds without it fall back to explicit probes.
    """
    accelerator = getattr(torch, "accelerator", None)
    if accelerator is not None and hasattr(accelerator, "current_accelerator"):
        try:
            current = accelerator.current_accelerator(check_available=True)
        except RuntimeError:  # no accelerator available in this build
            current = None
        if current is not None:
            return current.type
    if torch.cuda.is_available():
        return "cuda"
    if is_npu_available():
        return "npu"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
