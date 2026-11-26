"""The last vocab line keeps every character when the file has no final newline."""

import importlib.util
from pathlib import Path


def _get_tokenizer():
    path = Path(__file__).resolve().parents[1] / "src" / "f5_tts" / "model" / "utils.py"
    spec = importlib.util.spec_from_file_location("f5_tts_model_utils_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_tokenizer


def test_last_vocab_entry_keeps_its_last_character(tmp_path: Path) -> None:
    get_tokenizer = _get_tokenizer()
    vocab = tmp_path / "vocab.txt"
    vocab.write_bytes(b" \nni3")

    vocab_char_map, vocab_size = get_tokenizer(str(vocab), tokenizer="custom")

    assert vocab_char_map[" "] == 0
    assert vocab_char_map["ni3"] == 1
    assert vocab_size == 2

    vocab.write_bytes(b" \nni3\n")
    vocab_char_map, vocab_size = get_tokenizer(str(vocab), tokenizer="custom")
    assert vocab_char_map["ni3"] == 1
    assert vocab_size == 2

    vocab.write_bytes(b" \nx")
    vocab_char_map, _ = get_tokenizer(str(vocab), tokenizer="custom")
    assert vocab_char_map["x"] == 1
