# Copyright (c) 2024 Tsinghua Univ. (authors: Xingchen Song)
#               2025                (authors: Yuekai Zhang)
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
--model-path $CKPT_DIR/$model/model_1200000.pt \
--vocab-file $CKPT_DIR/$model/vocab.txt \
--vocoder-trt-engine-path $vocoder_trt_engine_path \
--backend-type $backend_type \
--tllm-model-dir $TRTLLM_ENGINE_DIR || exit 1
"""

import argparse
import importlib
import json
import os
import sys
import time

import datasets
import tensorrt as trt
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchaudio
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from tensorrt_llm._utils import trt_dtype_to_torch
from tensorrt_llm.logger import logger
from tensorrt_llm.runtime.session import Session, TensorInfo
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from vocos import Vocos


sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../../../../src/")

from f5_tts.eval.utils_eval import padded_mel_batch
from f5_tts.model.modules import get_vocos_mel_spectrogram
from f5_tts.model.utils import convert_char_to_pinyin, get_tokenizer, list_str_to_idx


F5TTS = importlib.import_module("model_repo_f5_tts.f5_tts.1.f5_tts_trtllm").F5TTS

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


def data_collator(batch, vocab_char_map, device="cuda", use_perf=False):
    if use_perf:
        torch.cuda.nvtx.range_push("data_collator")
    target_sample_rate = 24000
    target_rms = 0.1
    (
        ids,
        ref_rms_list,
        ref_mel_list,
        ref_mel_len_list,
        estimated_reference_target_mel_len,
        reference_target_texts_list,
    ) = (
        [],
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
        ref_rms_list.append(ref_rms)
        if ref_rms < target_rms:
            ref_audio_org = ref_audio_org * target_rms / ref_rms

        if ref_sr != target_sample_rate:
            resampler = torchaudio.transforms.Resample(ref_sr, target_sample_rate)
            ref_audio = resampler(ref_audio_org)
        else:
            ref_audio = ref_audio_org

        if use_perf:
            torch.cuda.nvtx.range_push(f"mel_spectrogram {i}")
        ref_audio = ref_audio.to("cuda")
        ref_mel = get_vocos_mel_spectrogram(ref_audio).squeeze(0)
        if use_perf:
            torch.cuda.nvtx.range_pop()
        ref_mel_len = ref_mel.shape[-1]
        assert ref_mel.shape[0] == 100

        ref_mel_list.append(ref_mel)
        ref_mel_len_list.append(ref_mel_len)

        estimated_reference_target_mel_len.append(
            int(ref_mel_len * (1 + len(target_text.encode("utf-8")) / len(prompt_text.encode("utf-8"))))
        )

    ref_mel_batch = padded_mel_batch(ref_mel_list)
    ref_mel_len_batch = torch.LongTensor(ref_mel_len_list)

    pinyin_list = convert_char_to_pinyin(reference_target_texts_list, polyphone=True)
    text_pad_sequence = list_str_to_idx(pinyin_list, vocab_char_map)

    if use_perf:
        torch.cuda.nvtx.range_pop()
    return {
        "ids": ids,
        "ref_rms_list": ref_rms_list,
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


class VocosTensorRT:
    def __init__(self, engine_path="./vocos_vocoder.plan", stream=None):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(TRT_LOGGER, namespace="")
        logger.info(f"Loading vocoder engine from {engine_path}")
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

    vocab_char_map, vocab_size = get_tokenizer(args.vocab_file, "custom")

    tllm_model_dir = args.tllm_model_dir
    with open(os.path.join(tllm_model_dir, "config.json")) as f:
        tllm_model_config = json.load(f)
    if args.backend_type == "trt":
        model = F5TTS(
            tllm_model_config,
            debug_mode=False,
            tllm_model_dir=tllm_model_dir,
            model_path=args.model_path,
            vocab_size=vocab_size,
        )
    elif args.backend_type == "pytorch":
        from f5_tts.infer.utils_infer import load_model
        from f5_tts.model import DiT

        pretrained_config = tllm_model_config["pretrained_config"]
        pt_model_config = dict(
            dim=pretrained_config["hidden_size"],
            depth=pretrained_config["num_hidden_layers"],
            heads=pretrained_config["num_attention_heads"],
            ff_mult=pretrained_config["ff_mult"],
            text_dim=pretrained_config["text_dim"],
            text_mask_padding=pretrained_config["text_mask_padding"],
            conv_layers=pretrained_config["conv_layers"],
            pe_attn_head=pretrained_config["pe_attn_head"],
            # attn_backend="flash_attn",
            # attn_mask_enabled=True,
        )
        model = load_model(DiT, pt_model_config, args.model_path)

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
            cond_pad_seq = F.pad(ref_mels, (0, 0, 0, max(total_mel_lens) - ref_mels.shape[1], 0, 0))
            if args.backend_type == "trt":
                _ = model.sample(
                    text_pad_seq,
                    cond_pad_seq,
                    ref_mel_lens,
                    total_mel_lens,
                    remove_input_padding=args.remove_input_padding,
                )
            elif args.backend_type == "pytorch":
                total_mel_lens = torch.tensor(total_mel_lens, device=device)
                with torch.inference_mode():
                    generated, _ = model.sample(
                        cond=ref_mels,
                        text=text_pad_seq,
                        duration=total_mel_lens,
                        steps=32,
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
        cond_pad_seq = F.pad(ref_mels, (0, 0, 0, max(total_mel_lens) - ref_mels.shape[1], 0, 0))
        if args.use_perf:
            torch.cuda.nvtx.range_pop()
        if args.backend_type == "trt":
            generated, cost_time = model.sample(
                text_pad_seq,
                cond_pad_seq,
                ref_mel_lens,
                total_mel_lens,
                remove_input_padding=args.remove_input_padding,
                use_perf=args.use_perf,
            )
        elif args.backend_type == "pytorch":
            total_mel_lens = torch.tensor(total_mel_lens, device=device)
            with torch.inference_mode():
                start_time = time.time()
                generated, _ = model.sample(
                    cond=ref_mels,
                    text=text_pad_seq,
                    duration=total_mel_lens,
                    lens=ref_mel_lens,
                    steps=32,
                    cfg_strength=2.0,
                    sway_sampling_coef=-1,
                )
                cost_time = time.time() - start_time
        decoding_time += cost_time
        vocoder_start_time = time.time()
        target_rms = 0.1
        target_sample_rate = 24000
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

            if batch["ref_rms_list"][i] < target_rms:
                generated_wave = generated_wave * batch["ref_rms_list"][i] / target_rms

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
