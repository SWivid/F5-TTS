# Copyright (c) 2024 Tsinghua Univ. (authors: Xingchen Song)
#               2025               （authors: Yuekai Zhang）
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified from https://github.com/xingchensong/S3Tokenizer/blob/main/s3tokenizer/cli.py
""" Example Usage
torchrun --nproc_per_node=1 \
benchmark.py --output-dir $log_dir \
--batch-size $batch_size \
--enable-warmup \
--split-name $split_name \
--model-path $F5_TTS_HF_DOWNLOAD_PATH/$model/model_1200000.pt \
--vocab-file $F5_TTS_HF_DOWNLOAD_PATH/$model/vocab.txt \
--vocoder-trt-engine-path $vocoder_trt_engine_path \
--backend-type $backend_type \
--tllm-model-dir $F5_TTS_TRT_LLM_ENGINE_PATH || exit 1
"""

import argparse
import json
import os
import time
from typing import Dict, List, Union

import datasets
import jieba
import tensorrt as trt
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchaudio
from datasets import load_dataset
from f5_tts_trtllm import F5TTS
from huggingface_hub import hf_hub_download
from pypinyin import Style, lazy_pinyin
from tensorrt_llm._utils import trt_dtype_to_torch
from tensorrt_llm.logger import logger
from tensorrt_llm.runtime.session import Session, TensorInfo
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from vocos import Vocos


torch.manual_seed(0)


def get_args():
    parser = argparse.ArgumentParser(description="extract speech code")
    parser.add_argument(
        "--split-name",
        type=str,
        default="wenetspeech4tts",
        choices=["wenetspeech4tts", "test_zh", "test_en", "test_hard"],
        help="huggingface dataset split name",
    )
    parser.add_argument("--output-dir", required=True, type=str, help="dir to save result")
    parser.add_argument(
        "--vocab-file",
        required=True,
        type=str,
        help="vocab file",
    )
    parser.add_argument(
        "--model-path",
        required=True,
        type=str,
        help="model path, to load text embedding",
    )
    parser.add_argument(
        "--tllm-model-dir",
        required=True,
        type=str,
        help="tllm model dir",
    )
    parser.add_argument(
        "--batch-size",
        required=True,
        type=int,
        help="batch size (per-device) for inference",
    )
    parser.add_argument("--num-workers", type=int, default=0, help="workers for dataloader")
    parser.add_argument("--prefetch", type=int, default=None, help="prefetch for dataloader")
    parser.add_argument(
        "--vocoder",
        default="vocos",
        type=str,
        help="vocoder name",
    )
    parser.add_argument(
        "--vocoder-trt-engine-path",
        default=None,
        type=str,
        help="vocoder trt engine path",
    )
    parser.add_argument("--enable-warmup", action="store_true")
    parser.add_argument("--remove-input-padding", action="store_true")
    parser.add_argument("--use-perf", action="store_true", help="use nvtx to record performance")
    parser.add_argument("--backend-type", type=str, default="triton", choices=["trt", "pytorch"], help="backend type")
    args = parser.parse_args()
    return args


def padded_mel_batch(ref_mels, max_seq_len):
    padded_ref_mels = []
    for mel in ref_mels:
        # pad along the last dimension
        padded_ref_mel = F.pad(mel, (0, 0, 0, max_seq_len - mel.shape[0]), value=0)
        padded_ref_mels.append(padded_ref_mel)
    padded_ref_mels = torch.stack(padded_ref_mels)
    return padded_ref_mels


def data_collator(batch, vocab_char_map, device="cuda", use_perf=False):
    if use_perf:
        torch.cuda.nvtx.range_push("data_collator")
    target_sample_rate = 24000
    target_rms = 0.1
    ids, ref_mel_list, ref_mel_len_list, estimated_reference_target_mel_len, reference_target_texts_list = (
        [],
        [],
        [],
        [],
        [],
    )
    for i, item in enumerate(batch):
        item_id, prompt_text, target_text = (
            item["id"],
            item["prompt_text"],
            item["target_text"],
        )
        ids.append(item_id)
        reference_target_texts_list.append(prompt_text + target_text)

        ref_audio_org, ref_sr = (
            item["prompt_audio"]["array"],
            item["prompt_audio"]["sampling_rate"],
        )
        ref_audio_org = torch.from_numpy(ref_audio_org).unsqueeze(0).float()
        ref_rms = torch.sqrt(torch.mean(torch.square(ref_audio_org)))
        if ref_rms < target_rms:
            ref_audio_org = ref_audio_org * target_rms / ref_rms

        if ref_sr != target_sample_rate:
            resampler = torchaudio.transforms.Resample(ref_sr, target_sample_rate)
            ref_audio = resampler(ref_audio_org)
        else:
            ref_audio = ref_audio_org

        if use_perf:
            torch.cuda.nvtx.range_push(f"mel_spectrogram {i}")
        ref_mel = mel_spectrogram(ref_audio, vocoder="vocos", device="cuda")
        if use_perf:
            torch.cuda.nvtx.range_pop()
        ref_mel = ref_mel.squeeze()
        ref_mel_len = ref_mel.shape[0]
        assert ref_mel.shape[1] == 100

        ref_mel_list.append(ref_mel)
        ref_mel_len_list.append(ref_mel_len)

        estimated_reference_target_mel_len.append(
            int(ref_mel.shape[0] * (1 + len(target_text.encode("utf-8")) / len(prompt_text.encode("utf-8"))))
        )

    max_seq_len = max(estimated_reference_target_mel_len)
    ref_mel_batch = padded_mel_batch(ref_mel_list, max_seq_len)
    ref_mel_len_batch = torch.LongTensor(ref_mel_len_list)

    pinyin_list = convert_char_to_pinyin(reference_target_texts_list, polyphone=True)
    text_pad_sequence = list_str_to_idx(pinyin_list, vocab_char_map)

    for i, item in enumerate(text_pad_sequence):
        text_pad_sequence[i] = F.pad(
            item, (0, estimated_reference_target_mel_len[i] - len(item)), mode="constant", value=-1
        )
        text_pad_sequence[i] += 1  # WAR: 0 is reserved for padding token, hard coding in F5-TTS
    text_pad_sequence = pad_sequence(text_pad_sequence, padding_value=-1, batch_first=True).to(device)
    text_pad_sequence = F.pad(
        text_pad_sequence, (0, max_seq_len - text_pad_sequence.shape[1]), mode="constant", value=-1
    )
    if use_perf:
        torch.cuda.nvtx.range_pop()
    return {
        "ids": ids,
        "ref_mel_batch": ref_mel_batch,
        "ref_mel_len_batch": ref_mel_len_batch,
        "text_pad_sequence": text_pad_sequence,
        "estimated_reference_target_mel_len": estimated_reference_target_mel_len,
    }


def init_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    print(
        "Inference on multiple gpus, this gpu {}".format(local_rank)
        + ", rank {}, world_size {}".format(rank, world_size)
    )
    torch.cuda.set_device(local_rank)
    # Initialize process group with explicit device IDs
    dist.init_process_group(
        "nccl",
    )
    return world_size, local_rank, rank


def get_tokenizer(vocab_file_path: str):
    """
    tokenizer   - "pinyin" do g2p for only chinese characters, need .txt vocab_file
                - "char" for char-wise tokenizer, need .txt vocab_file
                - "byte" for utf-8 tokenizer
                - "custom" if you're directly passing in a path to the vocab.txt you want to use
    vocab_size  - if use "pinyin", all available pinyin types, common alphabets (also those with accent) and symbols
                - if use "char", derived from unfiltered character & symbol counts of custom dataset
                - if use "byte", set to 256 (unicode byte range)
    """
    with open(vocab_file_path, "r", encoding="utf-8") as f:
        vocab_char_map = {}
        for i, char in enumerate(f):
            vocab_char_map[char[:-1]] = i
    vocab_size = len(vocab_char_map)
    return vocab_char_map, vocab_size


def convert_char_to_pinyin(reference_target_texts_list, polyphone=True):
    final_reference_target_texts_list = []
    custom_trans = str.maketrans(
        {";": ",", "“": '"', "”": '"', "‘": "'", "’": "'"}
    )  # add custom trans here, to address oov

    def is_chinese(c):
        return "\u3100" <= c <= "\u9fff"  # common chinese characters

    for text in reference_target_texts_list:
        char_list = []
        text = text.translate(custom_trans)
        for seg in jieba.cut(text):
            seg_byte_len = len(bytes(seg, "UTF-8"))
            if seg_byte_len == len(seg):  # if pure alphabets and symbols
                if char_list and seg_byte_len > 1 and char_list[-1] not in " :'\"":
                    char_list.append(" ")
                char_list.extend(seg)
            elif polyphone and seg_byte_len == 3 * len(seg):  # if pure east asian characters
                seg_ = lazy_pinyin(seg, style=Style.TONE3, tone_sandhi=True)
                for i, c in enumerate(seg):
                    if is_chinese(c):
                        char_list.append(" ")
                    char_list.append(seg_[i])
            else:  # if mixed characters, alphabets and symbols
                for c in seg:
                    if ord(c) < 256:
                        char_list.extend(c)
                    elif is_chinese(c):
                        char_list.append(" ")
                        char_list.extend(lazy_pinyin(c, style=Style.TONE3, tone_sandhi=True))
                    else:
                        char_list.append(c)
        final_reference_target_texts_list.append(char_list)

    return final_reference_target_texts_list


def list_str_to_idx(
    text: Union[List[str], List[List[str]]],
    vocab_char_map: Dict[str, int],  # {char: idx}
    padding_value=-1,
):
    list_idx_tensors = [torch.tensor([vocab_char_map.get(c, 0) for c in t]) for t in text]  # pinyin or char style
    # text = pad_sequence(list_idx_tensors, padding_value=padding_value, batch_first=True)
    return list_idx_tensors


def load_vocoder(
    vocoder_name="vocos", is_local=False, local_path="", device="cuda", hf_cache_dir=None, vocoder_trt_engine_path=None
):
    if vocoder_name == "vocos":
        if vocoder_trt_engine_path is not None:
            vocoder = VocosTensorRT(engine_path=vocoder_trt_engine_path)
        else:
            # vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device)
            if is_local:
                print(f"Load vocos from local path {local_path}")
                config_path = f"{local_path}/config.yaml"
                model_path = f"{local_path}/pytorch_model.bin"
            else:
                print("Download Vocos from huggingface charactr/vocos-mel-24khz")
                repo_id = "charactr/vocos-mel-24khz"
                config_path = hf_hub_download(repo_id=repo_id, cache_dir=hf_cache_dir, filename="config.yaml")
                model_path = hf_hub_download(repo_id=repo_id, cache_dir=hf_cache_dir, filename="pytorch_model.bin")
            vocoder = Vocos.from_hparams(config_path)
            state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
            from vocos.feature_extractors import EncodecFeatures

            if isinstance(vocoder.feature_extractor, EncodecFeatures):
                encodec_parameters = {
                    "feature_extractor.encodec." + key: value
                    for key, value in vocoder.feature_extractor.encodec.state_dict().items()
                }
                state_dict.update(encodec_parameters)
            vocoder.load_state_dict(state_dict)
            vocoder = vocoder.eval().to(device)
    elif vocoder_name == "bigvgan":
        raise NotImplementedError("BigVGAN is not implemented yet")
    return vocoder


def mel_spectrogram(waveform, vocoder="vocos", device="cuda"):
    if vocoder == "vocos":
        mel_stft = torchaudio.transforms.MelSpectrogram(
            sample_rate=24000,
            n_fft=1024,
            win_length=1024,
            hop_length=256,
            n_mels=100,
            power=1,
            center=True,
            normalized=False,
            norm=None,
        ).to(device)
    mel = mel_stft(waveform.to(device))
    mel = mel.clamp(min=1e-5).log()
    return mel.transpose(1, 2)


class VocosTensorRT:
    def __init__(self, engine_path="./vocos_vocoder.plan", stream=None):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(TRT_LOGGER, namespace="")
        logger.info(f"Loading vae engine from {engine_path}")
        self.engine_path = engine_path
        with open(engine_path, "rb") as f:
            engine_buffer = f.read()
        self.session = Session.from_serialized_engine(engine_buffer)
        self.stream = stream if stream is not None else torch.cuda.current_stream().cuda_stream

    def decode(self, mels):
        mels = mels.contiguous()
        inputs = {"mel": mels}
        output_info = self.session.infer_shapes([TensorInfo("mel", trt.DataType.FLOAT, mels.shape)])
        outputs = {
            t.name: torch.empty(tuple(t.shape), dtype=trt_dtype_to_torch(t.dtype), device="cuda") for t in output_info
        }
        ok = self.session.run(inputs, outputs, self.stream)

        assert ok, "Runtime execution failed for vae session"

        samples = outputs["waveform"]
        return samples


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    assert torch.cuda.is_available()
    world_size, local_rank, rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}")

    vocab_char_map, vocab_size = get_tokenizer(args.vocab_file)

    tllm_model_dir = args.tllm_model_dir
    config_file = os.path.join(tllm_model_dir, "config.json")
    with open(config_file) as f:
        config = json.load(f)
    if args.backend_type == "trt":
        model = F5TTS(
            config, debug_mode=False, tllm_model_dir=tllm_model_dir, model_path=args.model_path, vocab_size=vocab_size
        )
    elif args.backend_type == "pytorch":
        import sys

        sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../../../../src/")
        from f5_tts.infer.utils_infer import load_model
        from f5_tts.model import DiT

        F5TTS_model_cfg = dict(
            dim=1024,
            depth=22,
            heads=16,
            ff_mult=2,
            text_dim=512,
            conv_layers=4,
            pe_attn_head=1,
            text_mask_padding=False,
        )
        model = load_model(DiT, F5TTS_model_cfg, args.model_path)

    vocoder = load_vocoder(
        vocoder_name=args.vocoder, device=device, vocoder_trt_engine_path=args.vocoder_trt_engine_path
    )

    dataset = load_dataset(
        "yuekai/seed_tts",
        split=args.split_name,
        trust_remote_code=True,
    )

    def add_estimated_duration(example):
        prompt_audio_len = example["prompt_audio"]["array"].shape[0]
        scale_factor = 1 + len(example["target_text"]) / len(example["prompt_text"])
        estimated_duration = prompt_audio_len * scale_factor
        example["estimated_duration"] = estimated_duration / example["prompt_audio"]["sampling_rate"]
        return example

    dataset = dataset.map(add_estimated_duration)
    dataset = dataset.sort("estimated_duration", reverse=True)
    if args.use_perf:
        # dataset_list = [dataset.select(range(1)) for i in range(16)]  # seq_len 1000
        dataset_list_short = [dataset.select([24]) for i in range(8)]  # seq_len 719
        # dataset_list_long = [dataset.select([23]) for i in range(8)] # seq_len 2002
        # dataset = datasets.concatenate_datasets(dataset_list_short + dataset_list_long)
        dataset = datasets.concatenate_datasets(dataset_list_short)
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    else:
        # This would disable shuffling
        sampler = None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch,
        collate_fn=lambda x: data_collator(x, vocab_char_map, use_perf=args.use_perf),
    )

    total_steps = len(dataset)

    if args.enable_warmup:
        for batch in dataloader:
            ref_mels, ref_mel_lens = batch["ref_mel_batch"].to(device), batch["ref_mel_len_batch"].to(device)
            text_pad_seq = batch["text_pad_sequence"].to(device)
            total_mel_lens = batch["estimated_reference_target_mel_len"]
            if args.backend_type == "trt":
                _ = model.sample(
                    text_pad_seq, ref_mels, ref_mel_lens, total_mel_lens, remove_input_padding=args.remove_input_padding
                )
            elif args.backend_type == "pytorch":
                with torch.inference_mode():
                    text_pad_seq -= 1
                    text_pad_seq[text_pad_seq == -2] = -1
                    total_mel_lens = torch.tensor(total_mel_lens, device=device)
                    generated, _ = model.sample(
                        cond=ref_mels,
                        text=text_pad_seq,
                        duration=total_mel_lens,
                        steps=16,
                        cfg_strength=2.0,
                        sway_sampling_coef=-1,
                    )

    if rank == 0:
        progress_bar = tqdm(total=total_steps, desc="Processing", unit="wavs")

    decoding_time = 0
    vocoder_time = 0
    total_duration = 0
    if args.use_perf:
        torch.cuda.cudart().cudaProfilerStart()
    total_decoding_time = time.time()
    for batch in dataloader:
        if args.use_perf:
            torch.cuda.nvtx.range_push("data sample")
        ref_mels, ref_mel_lens = batch["ref_mel_batch"].to(device), batch["ref_mel_len_batch"].to(device)
        text_pad_seq = batch["text_pad_sequence"].to(device)
        total_mel_lens = batch["estimated_reference_target_mel_len"]

        if args.use_perf:
            torch.cuda.nvtx.range_pop()
        if args.backend_type == "trt":
            generated, cost_time = model.sample(
                text_pad_seq,
                ref_mels,
                ref_mel_lens,
                total_mel_lens,
                remove_input_padding=args.remove_input_padding,
                use_perf=args.use_perf,
            )
        elif args.backend_type == "pytorch":
            total_mel_lens = torch.tensor(total_mel_lens, device=device)
            with torch.inference_mode():
                start_time = time.time()
                text_pad_seq -= 1
                text_pad_seq[text_pad_seq == -2] = -1
                generated, _ = model.sample(
                    cond=ref_mels,
                    text=text_pad_seq,
                    duration=total_mel_lens,
                    lens=ref_mel_lens,
                    steps=16,
                    cfg_strength=2.0,
                    sway_sampling_coef=-1,
                )
                cost_time = time.time() - start_time
        decoding_time += cost_time
        vocoder_start_time = time.time()
        for i, gen in enumerate(generated):
            gen = gen[ref_mel_lens[i] : total_mel_lens[i], :].unsqueeze(0)
            gen_mel_spec = gen.permute(0, 2, 1).to(torch.float32)
            if args.vocoder == "vocos":
                if args.use_perf:
                    torch.cuda.nvtx.range_push("vocoder decode")
                generated_wave = vocoder.decode(gen_mel_spec).cpu()
                if args.use_perf:
                    torch.cuda.nvtx.range_pop()
            else:
                generated_wave = vocoder(gen_mel_spec).squeeze(0).cpu()
            target_rms = 0.1
            target_sample_rate = 24_000
            # if ref_rms_list[i] < target_rms:
            #     generated_wave = generated_wave * ref_rms_list[i] / target_rms
            rms = torch.sqrt(torch.mean(torch.square(generated_wave)))
            if rms < target_rms:
                generated_wave = generated_wave * target_rms / rms
            utt = batch["ids"][i]
            torchaudio.save(
                f"{args.output_dir}/{utt}.wav",
                generated_wave,
                target_sample_rate,
            )
            total_duration += generated_wave.shape[1] / target_sample_rate
        vocoder_time += time.time() - vocoder_start_time
        if rank == 0:
            progress_bar.update(world_size * len(batch["ids"]))
    total_decoding_time = time.time() - total_decoding_time
    if rank == 0:
        progress_bar.close()
    rtf = total_decoding_time / total_duration
    s = f"RTF: {rtf:.4f}\n"
    s += f"total_duration: {total_duration:.3f} seconds\n"
    s += f"({total_duration / 3600:.2f} hours)\n"
    s += f"DiT time: {decoding_time:.3f} seconds ({decoding_time / 3600:.2f} hours)\n"
    s += f"Vocoder time: {vocoder_time:.3f} seconds ({vocoder_time / 3600:.2f} hours)\n"
    s += f"total decoding time: {total_decoding_time:.3f} seconds ({total_decoding_time / 3600:.2f} hours)\n"
    s += f"batch size: {args.batch_size}\n"
    print(s)

    with open(f"{args.output_dir}/rtf.txt", "w") as f:
        f.write(s)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
