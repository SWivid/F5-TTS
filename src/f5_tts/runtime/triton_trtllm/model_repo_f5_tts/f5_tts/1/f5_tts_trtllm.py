import math
import os
import time
from functools import wraps
from typing import List, Optional

import tensorrt as trt
import tensorrt_llm
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorrt_llm._utils import str_dtype_to_torch, trt_dtype_to_torch
from tensorrt_llm.logger import logger
from tensorrt_llm.runtime.session import Session
from torch.nn.utils.rnn import pad_sequence


def remove_tensor_padding(input_tensor, input_tensor_lengths=None):
    # Audio tensor case: batch, seq_len, feature_len
    # position_ids case: batch, seq_len
    assert input_tensor_lengths is not None, "input_tensor_lengths must be provided for 3D input_tensor"

    # Initialize a list to collect valid sequences
    valid_sequences = []

    for i in range(input_tensor.shape[0]):
        valid_length = input_tensor_lengths[i]
        valid_sequences.append(input_tensor[i, :valid_length])

    # Concatenate all valid sequences along the batch dimension
    output_tensor = torch.cat(valid_sequences, dim=0).contiguous()
    return output_tensor


class TextEmbedding(nn.Module):
    def __init__(
        self, text_num_embeds, text_dim, mask_padding=True, conv_layers=0, conv_mult=2, precompute_max_pos=4096
    ):
        super().__init__()
        self.text_embed = nn.Embedding(text_num_embeds + 1, text_dim)  # use 0 as filler token
        self.mask_padding = mask_padding
        self.register_buffer("freqs_cis", precompute_freqs_cis(text_dim, precompute_max_pos), persistent=False)
        self.text_blocks = nn.Sequential(*[ConvNeXtV2Block(text_dim, text_dim * conv_mult) for _ in range(conv_layers)])

    def forward(self, text, seq_len, drop_text=False):
        text = text + 1
        text = text[:, :seq_len]  # curtail if character tokens are more than the mel spec tokens
        text = F.pad(text, (0, seq_len - text.shape[1]), value=0)
        if self.mask_padding:
            text_mask = text == 0

        if drop_text:  # cfg for text
            text = torch.zeros_like(text)

        text = self.text_embed(text)  # b n -> b n d
        text = text + self.freqs_cis[:seq_len, :]
        if self.mask_padding:
            text = text.masked_fill(text_mask.unsqueeze(-1).expand(-1, -1, text.size(-1)), 0.0)
            for block in self.text_blocks:
                text = block(text)
                text = text.masked_fill(text_mask.unsqueeze(-1).expand(-1, -1, text.size(-1)), 0.0)
        else:
            text = self.text_blocks(text)

        return text


class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class ConvNeXtV2Block(nn.Module):
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        dilation: int = 1,
    ):
        super().__init__()
        padding = (dilation * (7 - 1)) // 2
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size=7, padding=padding, groups=dim, dilation=dilation
        )  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = x.transpose(1, 2)  # b n d -> b d n
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # b d n -> b n d
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        return residual + x


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, theta_rescale_factor=1.0):
    # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
    # has some connection to NTK literature
    # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
    # https://github.com/lucidrains/rotary-embedding-torch/blob/main/rotary_embedding_torch/rotary_embedding_torch.py
    theta *= theta_rescale_factor ** (dim / (dim - 2))
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cos = torch.cos(freqs)  # real part
    freqs_sin = torch.sin(freqs)  # imaginary part
    return torch.cat([freqs_cos, freqs_sin], dim=-1)


def get_text_embed_dict(ckpt_path, use_ema=True):
    ckpt_type = ckpt_path.split(".")[-1]
    if ckpt_type == "safetensors":
        from safetensors.torch import load_file

        checkpoint = load_file(ckpt_path)
    else:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    if use_ema:
        if ckpt_type == "safetensors":
            checkpoint = {"ema_model_state_dict": checkpoint}
        checkpoint["model_state_dict"] = {
            k.replace("ema_model.", ""): v
            for k, v in checkpoint["ema_model_state_dict"].items()
            if k not in ["initted", "step"]
        }
    else:
        if ckpt_type == "safetensors":
            checkpoint = {"model_state_dict": checkpoint}
    model_params = checkpoint["model_state_dict"]

    text_embed_dict = {}
    for key in model_params.keys():
        # transformer.text_embed.text_embed.weight -> text_embed.weight
        if "text_embed" in key:
            text_embed_dict[key.replace("transformer.text_embed.", "")] = model_params[key]
    return text_embed_dict


class F5TTS(object):
    def __init__(
        self,
        config,
        debug_mode=True,
        stream: Optional[torch.cuda.Stream] = None,
        tllm_model_dir: Optional[str] = None,
        model_path: Optional[str] = None,
        vocab_size: Optional[int] = None,
    ):
        self.dtype = config["pretrained_config"]["dtype"]

        rank = tensorrt_llm.mpi_rank()
        world_size = config["pretrained_config"]["mapping"]["world_size"]
        cp_size = config["pretrained_config"]["mapping"]["cp_size"]
        tp_size = config["pretrained_config"]["mapping"]["tp_size"]
        pp_size = config["pretrained_config"]["mapping"]["pp_size"]
        assert pp_size == 1
        self.mapping = tensorrt_llm.Mapping(
            world_size=world_size, rank=rank, cp_size=cp_size, tp_size=tp_size, pp_size=1, gpus_per_node=1
        )

        local_rank = rank % self.mapping.gpus_per_node
        self.device = torch.device(f"cuda:{local_rank}")

        torch.cuda.set_device(self.device)

        self.stream = stream
        if self.stream is None:
            self.stream = torch.cuda.Stream(self.device)
        torch.cuda.set_stream(self.stream)

        engine_file = os.path.join(tllm_model_dir, f"rank{rank}.engine")
        logger.info(f"Loading engine from {engine_file}")
        with open(engine_file, "rb") as f:
            engine_buffer = f.read()

        assert engine_buffer is not None

        self.session = Session.from_serialized_engine(engine_buffer)

        self.debug_mode = debug_mode

        self.inputs = {}
        self.outputs = {}
        self.buffer_allocated = False

        expected_tensor_names = ["noise", "cond", "time", "rope_cos", "rope_sin", "input_lengths", "denoised"]

        found_tensor_names = [self.session.engine.get_tensor_name(i) for i in range(self.session.engine.num_io_tensors)]
        if not self.debug_mode and set(expected_tensor_names) != set(found_tensor_names):
            logger.error(
                f"The following expected tensors are not found: {set(expected_tensor_names).difference(set(found_tensor_names))}"
            )
            logger.error(
                f"Those tensors in engine are not expected: {set(found_tensor_names).difference(set(expected_tensor_names))}"
            )
            logger.error(f"Expected tensor names: {expected_tensor_names}")
            logger.error(f"Found tensor names: {found_tensor_names}")
            raise RuntimeError("Tensor names in engine are not the same as expected.")
        if self.debug_mode:
            self.debug_tensors = list(set(found_tensor_names) - set(expected_tensor_names))

        self.max_mel_len = 4096
        self.text_embedding = TextEmbedding(
            text_num_embeds=vocab_size,
            text_dim=config["pretrained_config"]["text_dim"],
            mask_padding=config["pretrained_config"]["text_mask_padding"],
            conv_layers=config["pretrained_config"]["conv_layers"],
            precompute_max_pos=self.max_mel_len,
        ).to(self.device)
        self.text_embedding.load_state_dict(get_text_embed_dict(model_path), strict=True)

        self.n_mel_channels = config["pretrained_config"]["mel_dim"]
        self.head_dim = config["pretrained_config"]["dim_head"]
        self.base_rescale_factor = 1.0
        self.interpolation_factor = 1.0
        base = 10000.0 * self.base_rescale_factor ** (self.head_dim / (self.head_dim - 2))
        inv_freq = 1.0 / (base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        freqs = torch.outer(torch.arange(self.max_mel_len, dtype=torch.float32), inv_freq) / self.interpolation_factor
        self.freqs = freqs.repeat_interleave(2, dim=-1).unsqueeze(0)
        self.rope_cos = self.freqs.cos().half()
        self.rope_sin = self.freqs.sin().half()

        self.nfe_steps = 32
        epss = {
            5: [0, 2, 4, 8, 16, 32],
            6: [0, 2, 4, 6, 8, 16, 32],
            7: [0, 2, 4, 6, 8, 16, 24, 32],
            10: [0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32],
            12: [0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32],
            16: [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32],
        }
        t = 1 / 32 * torch.tensor(epss.get(self.nfe_steps, list(range(self.nfe_steps + 1))), dtype=torch.float32)
        time_step = 1 - torch.cos(torch.pi * t / 2)
        delta_t = torch.diff(time_step)

        freq_embed_dim = 256  # Warning: hard coding 256 here
        time_expand = torch.zeros((1, self.nfe_steps, freq_embed_dim), dtype=torch.float32)
        half_dim = freq_embed_dim // 2
        emb_factor = math.log(10000) / (half_dim - 1)
        emb_factor = 1000.0 * torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb_factor)
        for i in range(self.nfe_steps):
            emb = time_step[i] * emb_factor
            time_expand[:, i, :] = torch.cat((emb.sin(), emb.cos()), dim=-1)
        self.time_expand = time_expand.to(self.device)
        self.delta_t = torch.cat((delta_t, delta_t), dim=0).contiguous().to(self.device)

    def _tensor_dtype(self, name):
        # return torch dtype given tensor name for convenience
        dtype = trt_dtype_to_torch(self.session.engine.get_tensor_dtype(name))
        return dtype

    def _setup(self, batch_size, seq_len):
        for i in range(self.session.engine.num_io_tensors):
            name = self.session.engine.get_tensor_name(i)
            if self.session.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                shape = list(self.session.engine.get_tensor_shape(name))
                shape[0] = batch_size
                shape[1] = seq_len
                self.outputs[name] = torch.empty(shape, dtype=self._tensor_dtype(name), device=self.device)

        self.buffer_allocated = True

    def cuda_stream_guard(func):
        """Sync external stream and set current stream to the one bound to the session. Reset on exit."""

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            external_stream = torch.cuda.current_stream()
            if external_stream != self.stream:
                external_stream.synchronize()
                torch.cuda.set_stream(self.stream)
            ret = func(self, *args, **kwargs)
            if external_stream != self.stream:
                self.stream.synchronize()
                torch.cuda.set_stream(external_stream)
            return ret

        return wrapper

    @cuda_stream_guard
    def forward(
        self,
        noise: torch.Tensor,
        cond: torch.Tensor,
        time_expand: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        input_lengths: torch.Tensor,
        delta_t: torch.Tensor,
        use_perf: bool = False,
    ):
        if use_perf:
            torch.cuda.nvtx.range_push("flow matching")
        cfg_strength = 2.0
        batch_size = noise.shape[0]
        half_batch = batch_size // 2
        noise_half = noise[:half_batch]  # Store the initial half of noise

        input_type = str_dtype_to_torch(self.dtype)

        # Keep a copy of the initial tensors
        cond = cond.to(input_type)
        rope_cos = rope_cos.to(input_type)
        rope_sin = rope_sin.to(input_type)
        input_lengths = input_lengths.to(str_dtype_to_torch("int32"))

        # Instead of iteratively updating noise within a single model context,
        # we'll do a single forward pass for each iteration with fresh context setup
        for i in range(self.nfe_steps):
            # Re-setup the buffers for clean execution
            self._setup(batch_size, noise.shape[1])
            if not self.buffer_allocated:
                raise RuntimeError("Buffer not allocated, please call setup first!")

            # Re-create combined noises for this iteration
            current_noise = torch.cat([noise_half, noise_half], dim=0).to(input_type)

            # Get time step for this iteration
            current_time = time_expand[:, i].to(input_type)

            # Create fresh input dictionary for this iteration
            current_inputs = {
                "noise": current_noise,
                "cond": cond,
                "time": current_time,
                "rope_cos": rope_cos,
                "rope_sin": rope_sin,
                "input_lengths": input_lengths,
            }

            # Update inputs and set shapes
            self.inputs.clear()  # Clear previous inputs
            self.inputs.update(**current_inputs)
            self.session.set_shapes(self.inputs)

            if use_perf:
                torch.cuda.nvtx.range_push(f"execute {i}")
            ok = self.session.run(self.inputs, self.outputs, self.stream.cuda_stream)
            assert ok, "Failed to execute model"
            # self.session.context.execute_async_v3(self.stream.cuda_stream)
            if use_perf:
                torch.cuda.nvtx.range_pop()
            # Process results
            t_scale = delta_t[i].unsqueeze(0).to(input_type)

            # Extract predictions
            pred_cond = self.outputs["denoised"][:half_batch]
            pred_uncond = self.outputs["denoised"][half_batch:]

            # Apply classifier-free guidance with safeguards
            guidance = pred_cond + (pred_cond - pred_uncond) * cfg_strength
            # Calculate update for noise
            noise_half = noise_half + guidance * t_scale
        if use_perf:
            torch.cuda.nvtx.range_pop()
        return noise_half

    def sample(
        self,
        text_pad_sequence: torch.Tensor,
        cond_pad_sequence: torch.Tensor,
        ref_mel_len_batch: torch.Tensor,
        estimated_reference_target_mel_len: List[int],
        remove_input_padding: bool = False,
        use_perf: bool = False,
    ):
        if use_perf:
            torch.cuda.nvtx.range_push("text embedding")
        batch = text_pad_sequence.shape[0]
        max_seq_len = cond_pad_sequence.shape[1]

        # get text_embed one by one to avoid misalignment
        text_and_drop_embedding_list = []
        for i in range(batch):
            text_embedding_i = self.text_embedding(
                text_pad_sequence[i].unsqueeze(0).to(self.device),
                estimated_reference_target_mel_len[i],
                drop_text=False,
            )
            text_embedding_drop_i = self.text_embedding(
                text_pad_sequence[i].unsqueeze(0).to(self.device),
                estimated_reference_target_mel_len[i],
                drop_text=True,
            )
            text_and_drop_embedding_list.extend([text_embedding_i[0], text_embedding_drop_i[0]])

        # pad separately computed text_embed to form batch with max_seq_len
        text_and_drop_embedding = pad_sequence(
            text_and_drop_embedding_list,
            batch_first=True,
            padding_value=0,
        )
        text_embedding = text_and_drop_embedding[0::2]
        text_embedding_drop = text_and_drop_embedding[1::2]

        noise = torch.randn_like(cond_pad_sequence).to(self.device)
        rope_cos = self.rope_cos[:, :max_seq_len, :].float().repeat(batch, 1, 1)
        rope_sin = self.rope_sin[:, :max_seq_len, :].float().repeat(batch, 1, 1)

        cat_mel_text = torch.cat(
            (
                cond_pad_sequence,
                text_embedding,
            ),
            dim=-1,
        )
        cat_mel_text_drop = torch.cat(
            (
                torch.zeros((batch, max_seq_len, self.n_mel_channels), dtype=torch.float32).to(self.device),
                text_embedding_drop,
            ),
            dim=-1,
        )

        time_expand = self.time_expand.repeat(2 * batch, 1, 1).contiguous()

        # Convert estimated_reference_target_mel_len to tensor
        input_lengths = torch.tensor(estimated_reference_target_mel_len, dtype=torch.int32)

        # combine above along the batch dimension
        inputs = {
            "noise": torch.cat((noise, noise), dim=0).contiguous(),
            "cond": torch.cat((cat_mel_text, cat_mel_text_drop), dim=0).contiguous(),
            "time_expand": time_expand,
            "rope_cos": torch.cat((rope_cos, rope_cos), dim=0).contiguous(),
            "rope_sin": torch.cat((rope_sin, rope_sin), dim=0).contiguous(),
            "input_lengths": torch.cat((input_lengths, input_lengths), dim=0).contiguous(),
            "delta_t": self.delta_t,
        }
        if use_perf and remove_input_padding:
            torch.cuda.nvtx.range_push("remove input padding")
        if remove_input_padding:
            max_seq_len = inputs["cond"].shape[1]
            inputs["noise"] = remove_tensor_padding(inputs["noise"], inputs["input_lengths"])
            inputs["cond"] = remove_tensor_padding(inputs["cond"], inputs["input_lengths"])
            # for time_expand, convert from B,D to B,T,D by repeat
            inputs["time_expand"] = inputs["time_expand"].unsqueeze(1).repeat(1, max_seq_len, 1, 1)
            inputs["time_expand"] = remove_tensor_padding(inputs["time_expand"], inputs["input_lengths"])
            inputs["rope_cos"] = remove_tensor_padding(inputs["rope_cos"], inputs["input_lengths"])
            inputs["rope_sin"] = remove_tensor_padding(inputs["rope_sin"], inputs["input_lengths"])
        if use_perf and remove_input_padding:
            torch.cuda.nvtx.range_pop()
        for key in inputs:
            inputs[key] = inputs[key].to(self.device)
        if use_perf:
            torch.cuda.nvtx.range_pop()
        start_time = time.time()
        denoised = self.forward(**inputs, use_perf=use_perf)
        cost_time = time.time() - start_time
        if use_perf and remove_input_padding:
            torch.cuda.nvtx.range_push("remove input padding output")
        if remove_input_padding:
            denoised_list = []
            start_idx = 0
            for i in range(batch):
                denoised_list.append(denoised[start_idx : start_idx + inputs["input_lengths"][i]])
                start_idx += inputs["input_lengths"][i]
            if use_perf and remove_input_padding:
                torch.cuda.nvtx.range_pop()
            return denoised_list, cost_time
        return denoised, cost_time
