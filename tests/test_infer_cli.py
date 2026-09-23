import runpy
import sys
from types import ModuleType
from unittest.mock import Mock

import hydra.utils
import numpy as np
import pytest
import soundfile as sf


@pytest.mark.parametrize(
    "option, default, configured",
    [
        ("cross_fade_duration", 0.15, 0.25),
        ("cfg_strength", 2.0, 3.0),
        ("sway_sampling_coef", -1.0, -0.5),
    ],
)
@pytest.mark.parametrize("case", ["zero_default", "zero_config", "config", "default", "override"])
def test_inference_option_precedence(monkeypatch, tmp_path, option, default, configured, case):
    # Keep the CLI parser, TOML/model configuration loading, main(), and WAV output real.
    # Replace model/audio processing boundaries so no weights or GPU are required.
    utils = ModuleType("f5_tts.infer.utils_infer")
    defaults = {
        "cfg_strength": 2.0,
        "cross_fade_duration": 0.15,
        "device": "cpu",
        "fix_duration": None,
        "mel_spec_type": "vocos",
        "nfe_step": 32,
        "speed": 1.0,
        "sway_sampling_coef": -1.0,
        "target_rms": 0.1,
    }
    utils.__dict__.update(defaults)
    waveform = np.zeros(240, dtype=np.float32)
    inference = Mock(return_value=(waveform, 24000, None))
    utils.infer_process = inference
    utils.load_model = Mock()
    utils.load_vocoder = Mock()
    utils.preprocess_ref_audio_text = Mock(return_value=("reference.wav", "Reference text."))
    utils.remove_silence_for_generated_wav = Mock()
    monkeypatch.setitem(sys.modules, utils.__name__, utils)

    downloads = ModuleType("cached_path")
    downloads.cached_path = Mock(side_effect=AssertionError("Unexpected checkpoint download"))
    monkeypatch.setitem(sys.modules, downloads.__name__, downloads)
    monkeypatch.setattr(hydra.utils, "get_class", Mock(return_value=object))

    config = tmp_path / "inference.toml"
    config_text = 'ckpt_file = "local-model.pt"\ngen_text = "Generated text."\n'
    if case in {"zero_config", "config", "override"}:
        config_text += f"{option} = {configured}\n"
    config.write_text(config_text, encoding="utf-8")
    argv = [
        "f5-tts_infer-cli",
        "--config",
        str(config),
        "--output_dir",
        str(tmp_path),
        "--output_file",
        "output.wav",
    ]
    if case.startswith("zero"):
        argv.extend([f"--{option}", "0"])
        expected = 0.0
    elif case == "override":
        argv.extend([f"--{option}", "0.5"])
        expected = 0.5
    else:
        expected = configured if case == "config" else default
    monkeypatch.setattr(sys, "argv", argv)

    runpy.run_module("f5_tts.infer.infer_cli", run_name="__main__")

    inference.assert_called_once()
    assert inference.call_args.args[2] == "Generated text."
    assert inference.call_args.kwargs[option] == expected
    audio, sample_rate = sf.read(tmp_path / "output.wav")
    assert sample_rate == 24000
    np.testing.assert_array_equal(audio, waveform)
    downloads.cached_path.assert_not_called()
