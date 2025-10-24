from __future__ import annotations

import os
import sys
from collections import OrderedDict

import numpy as np
import tensorrt as trt
from tensorrt_llm._common import default_net

from ..._utils import str_dtype_to_trt
from ...functional import (
    Tensor,
    concat,
    constant,
    expand,
    shape,
    slice,
    unsqueeze,
)
from ...layers import Linear
from ...module import Module, ModuleList
from ...plugin import current_all_reduce_helper
from ..modeling_utils import PretrainedConfig, PretrainedModel
from .modules import AdaLayerNormZero_Final, ConvPositionEmbedding, DiTBlock, TimestepEmbedding


current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
sys.path.append(parent_dir)


class InputEmbedding(Module):
    def __init__(self, mel_dim, text_dim, out_dim):
        super().__init__()
        self.proj = Linear(mel_dim * 2 + text_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)

    def forward(self, x, cond, mask=None):
        x = self.proj(concat([x, cond], dim=-1))
        return self.conv_pos_embed(x, mask=mask) + x


class F5TTS(PretrainedModel):
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.dtype = str_dtype_to_trt(config.dtype)

        self.time_embed = TimestepEmbedding(config.hidden_size)
        self.input_embed = InputEmbedding(config.mel_dim, config.text_dim, config.hidden_size)

        self.dim = config.hidden_size
        self.depth = config.num_hidden_layers
        self.transformer_blocks = ModuleList(
            [
                DiTBlock(
                    dim=self.dim,
                    heads=config.num_attention_heads,
                    dim_head=config.dim_head,
                    ff_mult=config.ff_mult,
                    dropout=config.dropout,
                    pe_attn_head=config.pe_attn_head,
                )
                for _ in range(self.depth)
            ]
        )

        self.norm_out = AdaLayerNormZero_Final(config.hidden_size)  # final modulation
        self.proj_out = Linear(config.hidden_size, config.mel_dim)

    def forward(
        self,
        noise,  # nosied input audio
        cond,  # masked cond audio
        time,  # time step
        rope_cos,
        rope_sin,
        input_lengths,
        scale=1.0,
    ):
        if default_net().plugin_config.remove_input_padding:
            mask = None
        else:
            N = shape(noise, 1)
            B = shape(noise, 0)
            seq_len_2d = concat([1, N])
            max_position_embeddings = 4096
            # create position ids
            position_ids_buffer = constant(np.expand_dims(np.arange(max_position_embeddings).astype(np.int32), 0))
            tmp_position_ids = slice(position_ids_buffer, starts=[0, 0], sizes=seq_len_2d)
            tmp_position_ids = expand(tmp_position_ids, concat([B, N]))  # [B, N]
            tmp_input_lengths = unsqueeze(input_lengths, 1)  # [B, 1]
            tmp_input_lengths = expand(tmp_input_lengths, concat([B, N]))  # [B, N]
            mask = tmp_position_ids < tmp_input_lengths  # [B, N]
            mask = mask.cast("int32")

        t = self.time_embed(time)
        x = self.input_embed(noise, cond, mask=mask)
        for block in self.transformer_blocks:
            x = block(x, t, rope_cos=rope_cos, rope_sin=rope_sin, input_lengths=input_lengths, scale=scale, mask=mask)
        denoise = self.proj_out(self.norm_out(x, t))
        denoise.mark_output("denoised", self.dtype)
        return denoise

    def prepare_inputs(self, **kwargs):
        max_batch_size = kwargs["max_batch_size"]
        batch_size_range = [2, 2, max_batch_size]
        mel_size = self.config.mel_dim
        max_seq_len = 3000  # 4096
        num_frames_range = [mel_size * 2, max_seq_len * 2, max_seq_len * max_batch_size]
        concat_feature_dim = mel_size + self.config.text_dim
        freq_embed_dim = 256  # Warning: hard coding 256 here
        head_dim = self.config.dim_head
        mapping = self.config.mapping
        if mapping.tp_size > 1:
            current_all_reduce_helper().set_workspace_tensor(mapping, 1)
        if default_net().plugin_config.remove_input_padding:
            noise = Tensor(
                name="noise",
                dtype=self.dtype,
                shape=[-1, mel_size],
                dim_range=OrderedDict(
                    [
                        ("num_frames", [num_frames_range]),
                        ("n_mels", [mel_size]),
                    ]
                ),
            )
            cond = Tensor(
                name="cond",
                dtype=self.dtype,
                shape=[-1, concat_feature_dim],
                dim_range=OrderedDict(
                    [
                        ("num_frames", [num_frames_range]),
                        ("embeded_length", [concat_feature_dim]),
                    ]
                ),
            )
            time = Tensor(
                name="time",
                dtype=self.dtype,
                shape=[-1, freq_embed_dim],
                dim_range=OrderedDict(
                    [
                        ("num_frames", [num_frames_range]),
                        ("freq_dim", [freq_embed_dim]),
                    ]
                ),
            )
            rope_cos = Tensor(
                name="rope_cos",
                dtype=self.dtype,
                shape=[-1, head_dim],
                dim_range=OrderedDict(
                    [
                        ("num_frames", [num_frames_range]),
                        ("head_dim", [head_dim]),
                    ]
                ),
            )
            rope_sin = Tensor(
                name="rope_sin",
                dtype=self.dtype,
                shape=[-1, head_dim],
                dim_range=OrderedDict(
                    [
                        ("num_frames", [num_frames_range]),
                        ("head_dim", [head_dim]),
                    ]
                ),
            )

        else:
            noise = Tensor(
                name="noise",
                dtype=self.dtype,
                shape=[-1, -1, mel_size],
                dim_range=OrderedDict(
                    [
                        ("batch_size", [batch_size_range]),
                        ("max_duratuion", [[100, max_seq_len // 2, max_seq_len]]),
                        ("n_mels", [mel_size]),
                    ]
                ),
            )
            cond = Tensor(
                name="cond",
                dtype=self.dtype,
                shape=[-1, -1, concat_feature_dim],
                dim_range=OrderedDict(
                    [
                        ("batch_size", [batch_size_range]),
                        ("max_duratuion", [[100, max_seq_len // 2, max_seq_len]]),
                        ("embeded_length", [concat_feature_dim]),
                    ]
                ),
            )
            time = Tensor(
                name="time",
                dtype=self.dtype,
                shape=[-1, freq_embed_dim],
                dim_range=OrderedDict(
                    [
                        ("batch_size", [batch_size_range]),
                        ("freq_dim", [freq_embed_dim]),
                    ]
                ),
            )
            rope_cos = Tensor(
                name="rope_cos",
                dtype=self.dtype,
                shape=[-1, -1, head_dim],
                dim_range=OrderedDict(
                    [
                        ("batch_size", [batch_size_range]),
                        ("max_duratuion", [[100, max_seq_len // 2, max_seq_len]]),
                        ("head_dim", [head_dim]),
                    ]
                ),
            )
            rope_sin = Tensor(
                name="rope_sin",
                dtype=self.dtype,
                shape=[-1, -1, head_dim],
                dim_range=OrderedDict(
                    [
                        ("batch_size", [batch_size_range]),
                        ("max_duratuion", [[100, max_seq_len // 2, max_seq_len]]),
                        ("head_dim", [head_dim]),
                    ]
                ),
            )
        input_lengths = Tensor(
            name="input_lengths",
            dtype=trt.int32,
            shape=[-1],
            dim_range=OrderedDict([("batch_size", [batch_size_range])]),
        )
        return {
            "noise": noise,
            "cond": cond,
            "time": time,
            "rope_cos": rope_cos,
            "rope_sin": rope_sin,
            "input_lengths": input_lengths,
        }
