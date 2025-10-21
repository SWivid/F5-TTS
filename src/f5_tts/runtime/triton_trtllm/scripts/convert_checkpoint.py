import argparse
import json
import os
import re
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import safetensors.torch
import torch
from tensorrt_llm import str_dtype_to_torch
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.convert_utils import split, split_matrix_tp


def split_q_tp(v, n_head, n_hidden, tensor_parallel, rank):
    split_v = split(v, tensor_parallel, rank, dim=1)
    return split_v.contiguous()


def split_q_bias_tp(v, n_head, n_hidden, tensor_parallel, rank):
    split_v = split(v, tensor_parallel, rank, dim=0)
    return split_v.contiguous()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pytorch_ckpt", type=str, default="./ckpts/model_last.pt")
    parser.add_argument(
        "--output_dir", type=str, default="./tllm_checkpoint", help="The path to save the TensorRT-LLM checkpoint"
    )
    parser.add_argument("--tp_size", type=int, default=1, help="N-way tensor parallelism size")
    parser.add_argument("--cp_size", type=int, default=1, help="Context parallelism size")
    parser.add_argument("--pp_size", type=int, default=1, help="N-way pipeline parallelism size")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--fp8_linear", action="store_true", help="Whether use FP8 for linear layers")
    parser.add_argument(
        "--workers", type=int, default=1, help="The number of workers for converting checkpoint in parallel"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="F5TTS_Custom",
        choices=[
            "F5TTS_v1_Base",
            "F5TTS_Base",
            "F5TTS_v1_Small",
            "F5TTS_Small",
        ],  # if set, overwrite the below hyperparams
    )
    parser.add_argument("--hidden_size", type=int, default=1024, help="The hidden size of DiT")
    parser.add_argument("--depth", type=int, default=22, help="The number of DiTBlock layers")
    parser.add_argument("--num_heads", type=int, default=16, help="The number of heads of attention module")
    parser.add_argument("--dim_head", type=int, default=64, help="The dimension of attention head")
    parser.add_argument("--ff_mult", type=int, default=2, help="The FFN intermediate dimension multiplier")
    parser.add_argument("--text_dim", type=int, default=512, help="The output dimension of text encoder")
    parser.add_argument(
        "--text_mask_padding",
        type=lambda x: x.lower() == "true",
        choices=[True, False],
        default=True,
        help="Whether apply padding mask for conv layers in text encoder",
    )
    parser.add_argument("--conv_layers", type=int, default=4, help="The number of conv layers of text encoder")
    parser.add_argument("--pe_attn_head", type=int, default=None, help="The number of attn head that apply pos emb")
    args = parser.parse_args()

    # overwrite if --model_name ordered
    if args.model_name == "F5TTS_v1_Base":
        args.hidden_size = 1024
        args.depth = 22
        args.num_heads = 16
        args.dim_head = 64
        args.ff_mult = 2
        args.text_dim = 512
        args.text_mask_padding = True
        args.conv_layers = 4
        args.pe_attn_head = None
    elif args.model_name == "F5TTS_Base":
        args.hidden_size = 1024
        args.depth = 22
        args.num_heads = 16
        args.dim_head = 64
        args.ff_mult = 2
        args.text_dim = 512
        args.text_mask_padding = False
        args.conv_layers = 4
        args.pe_attn_head = 1
    elif args.model_name == "F5TTS_v1_Small":
        args.hidden_size = 768
        args.depth = 18
        args.num_heads = 12
        args.dim_head = 64
        args.ff_mult = 2
        args.text_dim = 512
        args.text_mask_padding = True
        args.conv_layers = 4
        args.pe_attn_head = None
    elif args.model_name == "F5TTS_Small":
        args.hidden_size = 768
        args.depth = 18
        args.num_heads = 12
        args.dim_head = 64
        args.ff_mult = 2
        args.text_dim = 512
        args.text_mask_padding = False
        args.conv_layers = 4
        args.pe_attn_head = 1

    return args


def convert_pytorch_dit_to_trtllm_weight(args, mapping, dtype="float32", use_ema=True):
    weights = {}
    tik = time.time()
    torch_dtype = str_dtype_to_torch(dtype)
    tensor_parallel = mapping.tp_size

    ckpt_path = args.pytorch_ckpt
    ckpt_type = ckpt_path.split(".")[-1]
    if ckpt_type == "safetensors":
        from safetensors.torch import load_file

        model_params = load_file(ckpt_path)
    else:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model_params = ckpt["ema_model_state_dict"] if use_ema else ckpt["model_state_dict"]

    prefix = "ema_model.transformer." if use_ema else "transformer."
    if any(k.startswith(prefix) for k in model_params.keys()):
        model_params = {
            key[len(prefix) :] if key.startswith(prefix) else key: value
            for key, value in model_params.items()
            if key.startswith(prefix)
        }

    pytorch_to_trtllm_name = {
        r"^time_embed\.time_mlp\.0\.(weight|bias)$": r"time_embed.mlp1.\1",
        r"^time_embed\.time_mlp\.2\.(weight|bias)$": r"time_embed.mlp2.\1",
        r"^input_embed\.conv_pos_embed\.conv1d\.0\.(weight|bias)$": r"input_embed.conv_pos_embed.conv1d1.\1",
        r"^input_embed\.conv_pos_embed\.conv1d\.2\.(weight|bias)$": r"input_embed.conv_pos_embed.conv1d2.\1",
        r"^transformer_blocks\.(\d+)\.attn\.to_out\.0\.(weight|bias)$": r"transformer_blocks.\1.attn.to_out.\2",
        r"^transformer_blocks\.(\d+)\.ff\.ff\.0\.0\.(weight|bias)$": r"transformer_blocks.\1.ff.project_in.\2",
        r"^transformer_blocks\.(\d+)\.ff\.ff\.2\.(weight|bias)$": r"transformer_blocks.\1.ff.ff.\2",
    }

    def get_trtllm_name(pytorch_name):
        for pytorch_name_pattern, trtllm_name_replacement in pytorch_to_trtllm_name.items():
            trtllm_name_if_matched = re.sub(pytorch_name_pattern, trtllm_name_replacement, pytorch_name)
            if trtllm_name_if_matched != pytorch_name:
                return trtllm_name_if_matched
        return pytorch_name

    weights = dict()
    for name, param in model_params.items():
        if name == "input_embed.conv_pos_embed.conv1d.0.weight" or name == "input_embed.conv_pos_embed.conv1d.2.weight":
            weights[get_trtllm_name(name)] = param.contiguous().to(torch_dtype).unsqueeze(-1)
        else:
            weights[get_trtllm_name(name)] = param.contiguous().to(torch_dtype)

    assert len(weights) == len(model_params)

    # new_prefix = "f5_transformer."
    new_prefix = ""
    weights = {new_prefix + key: value for key, value in weights.items()}
    import math

    scale_factor = math.pow(64, -0.25)
    for k, v in weights.items():
        if re.match("^transformer_blocks.*.attn.to_k.weight$", k):
            weights[k] *= scale_factor
            weights[k] = split_q_tp(v, args.num_heads, args.hidden_size, tensor_parallel, mapping.tp_rank)

        elif re.match("^transformer_blocks.*.attn.to_k.bias$", k):
            weights[k] *= scale_factor
            weights[k] = split_q_bias_tp(v, args.num_heads, args.hidden_size, tensor_parallel, mapping.tp_rank)

        elif re.match("^transformer_blocks.*.attn.to_q.weight$", k):
            weights[k] = split_q_tp(v, args.num_heads, args.hidden_size, tensor_parallel, mapping.tp_rank)
            weights[k] *= scale_factor

        elif re.match("^transformer_blocks.*.attn.to_q.bias$", k):
            weights[k] = split_q_bias_tp(v, args.num_heads, args.hidden_size, tensor_parallel, mapping.tp_rank)
            weights[k] *= scale_factor

        elif re.match("^transformer_blocks.*.attn.to_v.weight$", k):
            weights[k] = split_q_tp(v, args.num_heads, args.hidden_size, tensor_parallel, mapping.tp_rank)

        elif re.match("^transformer_blocks.*.attn.to_v.bias$", k):
            weights[k] = split_q_bias_tp(v, args.num_heads, args.hidden_size, tensor_parallel, mapping.tp_rank)

        elif re.match("^transformer_blocks.*.attn.to_out.weight$", k):
            weights[k] = split_matrix_tp(v, tensor_parallel, mapping.tp_rank, dim=1)

    tok = time.time()
    t = time.strftime("%H:%M:%S", time.gmtime(tok - tik))
    print(f"Weights loaded. Total time: {t}")
    return weights


def save_config(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    config = {
        "architecture": "F5TTS",  # set the same as in ../patch/__init__.py
        "dtype": args.dtype,
        "hidden_size": args.hidden_size,
        "num_hidden_layers": args.depth,
        "num_attention_heads": args.num_heads,
        "dim_head": args.dim_head,
        "dropout": 0.0,  # inference-only
        "ff_mult": args.ff_mult,
        "mel_dim": 100,
        "text_dim": args.text_dim,
        "text_mask_padding": args.text_mask_padding,
        "conv_layers": args.conv_layers,
        "pe_attn_head": args.pe_attn_head,
        "mapping": {
            "world_size": args.cp_size * args.tp_size * args.pp_size,
            "cp_size": args.cp_size,
            "tp_size": args.tp_size,
            "pp_size": args.pp_size,
        },
    }
    if args.fp8_linear:
        config["quantization"] = {
            "quant_algo": "FP8",
            # TODO: add support for exclude modules.
            # "exclude_modules": "*final_layer*",
        }

    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)


def covert_and_save(args, rank):
    if rank == 0:
        save_config(args)

    mapping = Mapping(
        world_size=args.cp_size * args.tp_size * args.pp_size,
        rank=rank,
        cp_size=args.cp_size,
        tp_size=args.tp_size,
        pp_size=args.pp_size,
    )

    weights = convert_pytorch_dit_to_trtllm_weight(args, mapping, dtype=args.dtype)

    safetensors.torch.save_file(weights, os.path.join(args.output_dir, f"rank{rank}.safetensors"))


def execute(workers, func, args):
    if workers == 1:
        for rank, f in enumerate(func):
            f(args, rank)
    else:
        with ThreadPoolExecutor(max_workers=workers) as p:
            futures = [p.submit(f, args, rank) for rank, f in enumerate(func)]
            exceptions = []
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    traceback.print_exc()
                    exceptions.append(e)
            assert len(exceptions) == 0, "Checkpoint conversion failed, please check error log."


def main():
    args = parse_arguments()
    world_size = args.cp_size * args.tp_size * args.pp_size

    assert args.pp_size == 1, "PP is not supported yet."

    tik = time.time()
    if args.pytorch_ckpt is None:
        return
    print("Start execute")
    execute(args.workers, [covert_and_save] * world_size, args)

    tok = time.time()
    t = time.strftime("%H:%M:%S", time.gmtime(tok - tik))
    print(f"Total time of converting checkpoints: {t}")


if __name__ == "__main__":
    main()
