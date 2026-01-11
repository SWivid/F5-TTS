from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path

import pytest
import torch

from f5_tts.model import Trainer
from f5_tts.model.backbones.dit import DiT
from f5_tts.model.cfm import CFM
from f5_tts.model.utils import load_state_dict_compat
from f5_tts.rewards import RewardCombiner, RewardInput, RewardOutput, RewardProvider, RewardRegistry
from f5_tts.rewards.providers.funasr_wer import FunASRWERProvider, _wer
from f5_tts.rewards.providers.wespeaker_sim import WeSpeakerSimProvider
from f5_tts.rl.trainer_grpo import GRPOTrainer, sample_prompt_spans


def _make_dit(output_dist: str = "deterministic") -> DiT:
    return DiT(
        dim=16,
        depth=2,
        heads=2,
        ff_mult=2,
        mel_dim=8,
        text_num_embeds=256,
        text_dim=8,
        conv_layers=0,
        output_dist=output_dist,
    )


def _make_cfm(output_dist: str, objective: str) -> CFM:
    mel_spec_kwargs = dict(
        n_fft=16,
        hop_length=4,
        win_length=16,
        n_mel_channels=8,
        target_sample_rate=24000,
        mel_spec_type="vocos",
    )
    return CFM(
        transformer=_make_dit(output_dist=output_dist),
        mel_spec_kwargs=mel_spec_kwargs,
        vocab_char_map=None,
        objective=objective,
        output_dist=output_dist,
    )


def test_deterministic_forward():
    model = _make_dit(output_dist="deterministic")
    assert not hasattr(model, "proj_out_ln_sig")
    x = torch.randn(2, 4, 8)
    cond = torch.randn(2, 4, 8)
    text = torch.zeros(2, 3, dtype=torch.long)
    time = torch.rand(2)
    out = model(x=x, cond=cond, text=text, time=time)
    assert out.shape == x.shape


def test_gaussian_forward_prob():
    model = _make_dit(output_dist="gaussian")
    assert hasattr(model, "proj_out_ln_sig")
    x = torch.randn(2, 4, 8)
    cond = torch.randn(2, 4, 8)
    text = torch.zeros(2, 3, dtype=torch.long)
    time = torch.rand(2)
    mu, ln_sig = model.forward_prob(x=x, cond=cond, text=text, time=time)
    assert mu.shape == x.shape
    assert ln_sig.shape == x.shape


def test_soft_load_deterministic_into_gaussian():
    det_model = _make_dit(output_dist="deterministic")
    gauss_model = _make_dit(output_dist="gaussian")
    load_state_dict_compat(gauss_model, det_model.state_dict(), output_dist="gaussian")


def test_soft_load_warns_on_missing_ln_sig():
    det_model = _make_dit(output_dist="deterministic")
    gauss_model = _make_dit(output_dist="gaussian")
    with pytest.warns(RuntimeWarning, match="proj_out_ln_sig"):
        load_state_dict_compat(gauss_model, det_model.state_dict(), output_dist="gaussian")


def test_gaussian_loss_gradients():
    model = _make_cfm(output_dist="gaussian", objective="gaussian_nll")
    inp = torch.randn(2, 4, 8)
    text = torch.zeros(2, 3, dtype=torch.long)
    loss, _, _ = model(inp, text=text)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.transformer.proj_out_ln_sig.weight.grad is not None


class DummyRewardProvider(RewardProvider):
    name = "dummy_reward"

    def compute(self, batch: list[RewardInput]) -> list[RewardOutput]:
        outputs = []
        for item in batch:
            reward = item.audio.mean().to(dtype=torch.float32)
            outputs.append(RewardOutput(total_reward=reward, components={"mean": reward}, logs={}))
        return outputs


class DummyVocoder:
    def decode(self, mel: torch.Tensor) -> torch.Tensor:
        return mel.mean(dim=1)


class DummyDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, idx):
        mel = torch.randn(8, 6)
        return {"mel_spec": mel, "text": "hi"}


def test_grpo_single_step_updates_params(tmp_path):
    torch.manual_seed(0)
    model = _make_cfm(output_dist="gaussian", objective="grpo")
    before = model.transformer.proj_out.weight.detach().clone()
    combiner = RewardCombiner([DummyRewardProvider()])
    trainer = GRPOTrainer(
        model,
        reward_combiner=combiner,
        epochs=1,
        learning_rate=1e-3,
        num_warmup_updates=1,
        save_per_updates=1000,
        keep_last_n_checkpoints=0,
        checkpoint_path=str(tmp_path),
        batch_size_per_gpu=2,
        batch_size_type="sample",
        max_samples=2,
        grad_accumulation_steps=1,
        max_grad_norm=1.0,
        logger=None,
        mel_spec_type="vocos",
        vocoder=DummyVocoder(),
        repeat_count=1,
        mini_repeat_count=1,
        prompt_frac_range=(0.5, 0.5),
        steps=3,
        cfg_strength=1.0,
        sway_sampling_coef=None,
    )
    trainer.train(DummyDataset(), num_workers=0)
    after = model.transformer.proj_out.weight.detach()
    assert not torch.equal(before, after)


def test_registry_import_path():
    provider = RewardRegistry.create({"name": f"{__name__}:DummyRewardProvider"})
    assert isinstance(provider, DummyRewardProvider)


def test_rewards_import_does_not_pull_optional_deps():
    import f5_tts.rewards  # noqa: F401

    assert "funasr" not in sys.modules
    assert "wespeaker" not in sys.modules


def test_prompt_length_mode_behavior():
    seq_len = torch.tensor([10, 10])
    frac = torch.tensor([0.6, 0.6])
    rand = torch.tensor([0.1, 0.9])
    start_min, _, _ = sample_prompt_spans(seq_len, frac, mode="min", rand=rand)
    start_per, _, _ = sample_prompt_spans(seq_len, frac, mode="per_sample", rand=rand)
    assert start_min[0].item() == start_min[1].item()
    assert start_per[0].item() != start_per[1].item()


def test_forward_rl_preserves_eval_mode():
    model = _make_cfm(output_dist="gaussian", objective="grpo")
    model.eval()
    cond = torch.randn(1, 2, 8)
    text = ["hi"]
    duration = torch.tensor([2])
    model.forward_rl(cond=cond, text=text, duration=duration, steps=2, cfg_strength=0.0, set_train=False)
    assert model.training is False


def _write_wespeaker_stub(tmp_path: Path, frontend: str = "fbank") -> Path:
    pkg_root = tmp_path / "wespeaker"
    models_dir = pkg_root / "models"
    utils_dir = pkg_root / "utils"
    models_dir.mkdir(parents=True, exist_ok=True)
    utils_dir.mkdir(parents=True, exist_ok=True)
    (pkg_root / "__init__.py").write_text("", encoding="utf-8")
    (models_dir / "__init__.py").write_text("", encoding="utf-8")
    (utils_dir / "__init__.py").write_text("", encoding="utf-8")
    (models_dir / "speaker_model.py").write_text(
        "\n".join(
            [
                "import torch",
                "",
                "class DummyModel(torch.nn.Module):",
                "    def __init__(self, **kwargs):",
                "        super().__init__()",
                "        self.weight = torch.nn.Parameter(torch.ones(1))",
                "",
                "    def forward(self, feats):",
                "        return torch.ones((feats.size(0), 256), device=feats.device)",
                "",
                "def get_speaker_model(name):",
                "    return DummyModel",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (utils_dir / "checkpoint.py").write_text(
        "\n".join(
            [
                "def load_checkpoint(model, path):",
                "    return model",
                "",
            ]
        ),
        encoding="utf-8",
    )

    model_dir = tmp_path / "wespeaker_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "avg_model.pt").write_bytes(b"stub")
    config = "\n".join(
        [
            "model: ResNet34",
            "model_args: {}",
            "dataset_args:",
            f"  frontend: {frontend}",
            "  fbank_args:",
            "    num_mel_bins: 80",
            "    frame_length: 25",
            "    frame_shift: 10",
            "    dither: 0.0",
            "    window_type: hamming",
            "  resample_rate: 16000",
            "",
        ]
    )
    (model_dir / "config.yaml").write_text(config, encoding="utf-8")
    return model_dir


def _install_funasr_stub(monkeypatch, texts: list[str] | None = None) -> None:
    texts = texts or ["hello world"]

    class DummyAutoModel:
        def __init__(self, model, device, disable_update=True):
            self.model = model
            self.device = device

        def inference(self, input, cache, language, use_itn, disable_pbar, batch_size):
            return [{"text": texts[idx % len(texts)]} for idx in range(len(input))]

    funasr = types.ModuleType("funasr")
    funasr.__path__ = []
    funasr.AutoModel = DummyAutoModel
    utils = types.ModuleType("funasr.utils")
    utils.__path__ = []
    post = types.ModuleType("funasr.utils.postprocess_utils")

    def rich_transcription_postprocess(text: str) -> str:
        return text

    post.rich_transcription_postprocess = rich_transcription_postprocess
    funasr.utils = utils
    utils.postprocess_utils = post

    monkeypatch.setitem(sys.modules, "funasr", funasr)
    monkeypatch.setitem(sys.modules, "funasr.utils", utils)
    monkeypatch.setitem(sys.modules, "funasr.utils.postprocess_utils", post)


def test_trainer_allows_num_workers_zero(tmp_path, monkeypatch):
    model = _make_cfm(output_dist="deterministic", objective="mse")
    trainer = Trainer(
        model,
        epochs=1,
        learning_rate=1e-4,
        num_warmup_updates=0,
        save_per_updates=1000,
        keep_last_n_checkpoints=0,
        checkpoint_path=str(tmp_path),
        batch_size_per_gpu=1,
        batch_size_type="sample",
        max_samples=1,
        grad_accumulation_steps=1,
        max_grad_norm=1.0,
        logger=None,
        log_samples=False,
        mel_spec_type="vocos",
    )
    monkeypatch.setattr(trainer, "save_checkpoint", lambda *args, **kwargs: None)
    trainer.train(DummyDataset(), num_workers=0)


def test_wespeaker_requires_package(monkeypatch):
    original_find_spec = importlib.util.find_spec

    def _missing_spec(name, *args, **kwargs):
        if name == "wespeaker":
            return None
        return original_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", _missing_spec)
    provider = WeSpeakerSimProvider()
    provider.setup({})
    with pytest.raises(ImportError, match="WeSpeaker is required"):
        provider._ensure_model()


def test_wespeaker_rejects_non_fbank(tmp_path, monkeypatch):
    model_dir = _write_wespeaker_stub(tmp_path, frontend="s3prl")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    for key in [name for name in sys.modules if name.startswith("wespeaker")]:
        sys.modules.pop(key, None)
    provider = WeSpeakerSimProvider()
    provider.setup({"model_dir": str(model_dir), "device": "cpu", "cache_enabled": False})
    with pytest.raises(RuntimeError, match="frontend"):
        provider._ensure_model()


def test_wespeaker_fbank_stub_runs(tmp_path, monkeypatch):
    model_dir = _write_wespeaker_stub(tmp_path, frontend="fbank")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    for key in [name for name in sys.modules if name.startswith("wespeaker")]:
        sys.modules.pop(key, None)
    provider = WeSpeakerSimProvider()
    provider.setup({"model_dir": str(model_dir), "device": "cpu", "cache_enabled": False})
    audio = torch.randn(16000)
    batch = [
        RewardInput(
            audio=audio,
            text="hi",
            speaker_ref=audio,
            sample_rate=16000,
            meta={},
        )
    ]
    outputs = provider.compute(batch)
    assert torch.isfinite(outputs[0].total_reward)


def test_wespeaker_respects_device(tmp_path, monkeypatch):
    model_dir = _write_wespeaker_stub(tmp_path, frontend="fbank")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    for key in [name for name in sys.modules if name.startswith("wespeaker")]:
        sys.modules.pop(key, None)
    provider = WeSpeakerSimProvider()
    provider.setup({"model_dir": str(model_dir), "device": "cpu", "cache_enabled": False})
    provider._ensure_model()
    assert provider._model.weight.device.type == "cpu"


def test_funasr_stub_runs_and_respects_device(monkeypatch):
    _install_funasr_stub(monkeypatch, texts=["hello"])
    provider = FunASRWERProvider()
    provider.setup({"model_id": "stub", "device": "cpu", "cache_enabled": False})
    audio = torch.randn(16000)
    batch = [
        RewardInput(
            audio=audio,
            text="hello",
            speaker_ref=None,
            sample_rate=16000,
            meta={},
        )
    ]
    outputs = provider.compute(batch)
    assert outputs[0].total_reward.device.type == "cpu"
    assert provider._model.device == "cpu"


def test_funasr_reward_values(monkeypatch):
    _install_funasr_stub(monkeypatch, texts=["hello world"])
    provider = FunASRWERProvider()
    provider.setup({"model_id": "stub", "device": "cpu", "cache_enabled": False})
    audio = torch.randn(16000)
    batch = [
        RewardInput(
            audio=audio,
            text="hello world",
            speaker_ref=None,
            sample_rate=16000,
            meta={},
        )
    ]
    outputs = provider.compute(batch)
    assert outputs[0].components["wer"].item() == 0.0
    assert outputs[0].components["acc"].item() == 1.0
    assert outputs[0].total_reward.item() == 1.0


def test_funasr_wer_modes_punctuation_case():
    ref = "hello world"
    hyp = "hello, world."
    word = _wer(ref, hyp, mode="word")
    char = _wer(ref, hyp, mode="char")
    assert word == 1.0
    assert 0.0 < char < word


def test_funasr_default_wer_mode_is_char():
    provider = FunASRWERProvider()
    provider.setup({"model_id": "stub", "device": "cpu", "cache_enabled": False})
    assert provider.wer_mode == "char"


def test_wespeaker_similarity_identical_audio(tmp_path, monkeypatch):
    model_dir = _write_wespeaker_stub(tmp_path, frontend="fbank")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    for key in [name for name in sys.modules if name.startswith("wespeaker")]:
        sys.modules.pop(key, None)
    provider = WeSpeakerSimProvider()
    provider.setup({"model_dir": str(model_dir), "device": "cpu", "cache_enabled": False})
    audio = torch.randn(16000)
    batch = [
        RewardInput(
            audio=audio,
            text="hi",
            speaker_ref=audio,
            sample_rate=16000,
            meta={},
        )
    ]
    outputs = provider.compute(batch)
    assert outputs[0].components["sim"].item() >= 0.99


@pytest.mark.integration
def test_audio_pack_metadata():
    pack_dir = Path(__file__).parent / "assets" / "audio_pack"
    metadata = pack_dir / "metadata.jsonl"
    if not metadata.exists():
        pytest.skip("audio pack not present")
    line = metadata.read_text(encoding="utf-8").splitlines()[0]
    audio_name = json.loads(line).get("audio")
    assert (pack_dir / audio_name).exists()
