from __future__ import annotations

import os
import random
from collections import defaultdict
from importlib.resources import files

import torch
from torch.nn.utils.rnn import pad_sequence

import jieba
from pypinyin import lazy_pinyin, Style

import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import torchaudio

# seed everything


def seed_everything(seed=0):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# helpers


def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


# tensor helpers


def lens_to_mask(t: int["b"], length: int | None = None) -> bool["b n"]:  # noqa: F722 F821
    if not exists(length):
        length = t.amax()

    seq = torch.arange(length, device=t.device)
    return seq[None, :] < t[:, None]


def mask_from_start_end_indices(seq_len: int["b"], start: int["b"], end: int["b"]):  # noqa: F722 F821
    max_seq_len = seq_len.max().item()
    seq = torch.arange(max_seq_len, device=start.device).long()
    start_mask = seq[None, :] >= start[:, None]
    end_mask = seq[None, :] < end[:, None]
    return start_mask & end_mask


def mask_from_frac_lengths(seq_len: int["b"], frac_lengths: float["b"]):  # noqa: F722 F821
    lengths = (frac_lengths * seq_len).long()
    max_start = seq_len - lengths

    rand = torch.rand_like(frac_lengths)
    start = (max_start * rand).long().clamp(min=0)
    end = start + lengths

    return mask_from_start_end_indices(seq_len, start, end)


def maybe_masked_mean(t: float["b n d"], mask: bool["b n"] = None) -> float["b d"]:  # noqa: F722
    if not exists(mask):
        return t.mean(dim=1)

    t = torch.where(mask[:, :, None], t, torch.tensor(0.0, device=t.device))
    num = t.sum(dim=1)
    den = mask.float().sum(dim=1)

    return num / den.clamp(min=1.0)


# simple utf-8 tokenizer, since paper went character based
def list_str_to_tensor(text: list[str], padding_value=-1) -> int["b nt"]:  # noqa: F722
    list_tensors = [torch.tensor([*bytes(t, "UTF-8")]) for t in text]  # ByT5 style
    text = pad_sequence(list_tensors, padding_value=padding_value, batch_first=True)
    return text


# char tokenizer, based on custom dataset's extracted .txt file
def list_str_to_idx(
    text: list[str] | list[list[str]],
    vocab_char_map: dict[str, int],  # {char: idx}
    padding_value=-1,
) -> int["b nt"]:  # noqa: F722
    list_idx_tensors = [torch.tensor([vocab_char_map.get(c, 0) for c in t]) for t in text]  # pinyin or char style
    text = pad_sequence(list_idx_tensors, padding_value=padding_value, batch_first=True)
    return text


# Get tokenizer


def get_tokenizer(dataset_name, tokenizer: str = "pinyin"):
    """
    tokenizer   - "pinyin" do g2p for only chinese characters, need .txt vocab_file
                - "char" for char-wise tokenizer, need .txt vocab_file
                - "byte" for utf-8 tokenizer
                - "custom" if you're directly passing in a path to the vocab.txt you want to use
    vocab_size  - if use "pinyin", all available pinyin types, common alphabets (also those with accent) and symbols
                - if use "char", derived from unfiltered character & symbol counts of custom dataset
                - if use "byte", set to 256 (unicode byte range)
    """
    if tokenizer in ["pinyin", "char"]:
        tokenizer_path = os.path.join(files("f5_tts").joinpath("../../data"), f"{dataset_name}_{tokenizer}/vocab.txt")
        with open(tokenizer_path, "r", encoding="utf-8") as f:
            vocab_char_map = {}
            for i, char in enumerate(f):
                vocab_char_map[char[:-1]] = i
        vocab_size = len(vocab_char_map)
        assert vocab_char_map[" "] == 0, "make sure space is of idx 0 in vocab.txt, cuz 0 is used for unknown char"

    elif tokenizer == "byte":
        vocab_char_map = None
        vocab_size = 256

    elif tokenizer == "custom":
        with open(dataset_name, "r", encoding="utf-8") as f:
            vocab_char_map = {}
            for i, char in enumerate(f):
                vocab_char_map[char[:-1]] = i
        vocab_size = len(vocab_char_map)

    return vocab_char_map, vocab_size


# convert char to pinyin


def convert_char_to_pinyin(text_list, polyphone=True):
    final_text_list = []
    god_knows_why_en_testset_contains_zh_quote = str.maketrans(
        {"“": '"', "”": '"', "‘": "'", "’": "'"}
    )  # in case librispeech (orig no-pc) test-clean
    custom_trans = str.maketrans({";": ","})  # add custom trans here, to address oov
    for text in text_list:
        char_list = []
        text = text.translate(god_knows_why_en_testset_contains_zh_quote)
        text = text.translate(custom_trans)
        for seg in jieba.cut(text):
            seg_byte_len = len(bytes(seg, "UTF-8"))
            if seg_byte_len == len(seg):  # if pure alphabets and symbols
                if char_list and seg_byte_len > 1 and char_list[-1] not in " :'\"":
                    char_list.append(" ")
                char_list.extend(seg)
            elif polyphone and seg_byte_len == 3 * len(seg):  # if pure chinese characters
                seg = lazy_pinyin(seg, style=Style.TONE3, tone_sandhi=True)
                for c in seg:
                    if c not in "。，、；：？！《》【】—…":
                        char_list.append(" ")
                    char_list.append(c)
            else:  # if mixed chinese characters, alphabets and symbols
                for c in seg:
                    if ord(c) < 256:
                        char_list.extend(c)
                    else:
                        if c not in "。，、；：？！《》【】—…":
                            char_list.append(" ")
                            char_list.extend(lazy_pinyin(c, style=Style.TONE3, tone_sandhi=True))
                        else:  # if is zh punc
                            char_list.append(c)
        final_text_list.append(char_list)

    return final_text_list


# filter func for dirty data with many repetitions


def repetition_found(text, length=2, tolerance=10):
    pattern_count = defaultdict(int)
    for i in range(len(text) - length + 1):
        pattern = text[i : i + length]
        pattern_count[pattern] += 1
    for pattern, count in pattern_count.items():
        if count > tolerance:
            return True
    return False


def normalize_and_colorize_spectrogram(mel_org):
    mel_min, mel_max = mel_org.min(), mel_org.max()
    mel_norm = (mel_org - mel_min) / (mel_max - mel_min + 1e-8)
    mel_colored = plt.get_cmap("viridis")(mel_norm.detach().cpu().numpy())[:, :, :3]
    mel_colored = np.transpose(mel_colored, (2, 0, 1))
    return mel_colored


def export_audio(file_out, wav, target_sample_rate):
    sf.write(file_out, wav, samplerate=target_sample_rate)


def export_mel(mel_colored_hwc, file_out):
    plt.imsave(file_out, mel_colored_hwc)


def gen_sample(model, vocos, file_wav_org, text_inputs, hop_length, nfe_step, cfg_strength, sway_sampling_coef):
    audio, sr = torchaudio.load(file_wav_org)
    audio = audio.to("cuda")
    ref_audio_len = audio.shape[-1] // hop_length
    text = [text_inputs[0] + [" . "] + text_inputs[0]]
    duration = int((audio.shape[1] / 256) * 2.0)
    with torch.inference_mode():
        generated_gen, _ = model.sample(
            cond=audio,
            text=text,
            duration=duration,
            steps=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
        )
    generated_gen = generated_gen.to(torch.float32)
    generated_gen = generated_gen[:, ref_audio_len:, :]
    generated_mel_spec_gen = generated_gen.permute(0, 2, 1)
    generated_wave_gen = vocos.decode(generated_mel_spec_gen.cpu())
    generated_wave_gen = generated_wave_gen.squeeze().cpu().numpy()
    return generated_wave_gen, generated_mel_spec_gen


def get_sample(
    vocos,
    model,
    file_path_samples,
    global_step,
    mel_org,
    text_inputs,
    target_sample_rate,
    hop_length,
    nfe_step,
    cfg_strength,
    sway_sampling_coef,
):
    generated_wave_org = vocos.decode(mel_org.unsqueeze(0).cpu())
    generated_wave_org = generated_wave_org.squeeze().cpu().numpy()
    file_wav_org = os.path.join(file_path_samples, f"step_{global_step}_org.wav")
    export_audio(file_wav_org, generated_wave_org, target_sample_rate)
    generated_wave_gen, generated_mel_spec_gen = gen_sample(
        model, vocos, file_wav_org, text_inputs, hop_length, nfe_step, cfg_strength, sway_sampling_coef
    )
    file_wav_gen = os.path.join(file_path_samples, f"step_{global_step}_gen.wav")
    export_audio(file_wav_gen, generated_wave_gen, target_sample_rate)
    mel_org = normalize_and_colorize_spectrogram(mel_org)
    mel_gen = normalize_and_colorize_spectrogram(generated_mel_spec_gen[0])
    file_gen_org = os.path.join(file_path_samples, f"step_{global_step}_org.png")
    export_mel(np.transpose(mel_org, (1, 2, 0)), file_gen_org)
    file_gen_gen = os.path.join(file_path_samples, f"step_{global_step}_gen.png")
    export_mel(np.transpose(mel_gen, (1, 2, 0)), file_gen_gen)
    return generated_wave_org, generated_wave_gen, mel_org, mel_gen
