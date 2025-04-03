"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""

from __future__ import annotations
import sys
import os
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
sys.path.append(parent_dir)
import math
import numpy as np
import torch
from torch import nn
import tensorrt as trt
from collections import OrderedDict
from ..._utils import str_dtype_to_trt, trt_dtype_to_str, trt_dtype_to_np
from ...plugin import current_all_reduce_helper
from ..modeling_utils import PretrainedConfig, PretrainedModel
from ...functional import (Tensor, allgather, arange, chunk, concat, constant,
                           cos, exp, expand, shape, silu, sin, slice, split,
                           unsqueeze, squeeze, cast)
from ...module import Module, ModuleList
from tensorrt_llm._common import default_net
from ...layers import Linear

from .modules import (
    TimestepEmbedding,
    # ConvNeXtV2Block,
    ConvPositionEmbedding,
    DiTBlock,
    AdaLayerNormZero_Final,
    # precompute_freqs_cis, get_pos_embed_indices,
)

# Text embedding
# class TextEmbedding(Module):
#     def __init__(self, text_num_embeds, text_dim, conv_layers = 0, conv_mult = 2):
#         super().__init__()
#         self.text_embed = nn.Embedding(text_num_embeds + 1, text_dim)  # use 0 as filler token

#         if conv_layers > 0:
#             self.extra_modeling = True
#             self.precompute_max_pos = 4096  # ~44s of 24khz audio
#             self.register_buffer("freqs_cis", precompute_freqs_cis(text_dim, self.precompute_max_pos), persistent=False)
#             self.text_blocks = nn.Sequential(*[ConvNeXtV2Block(text_dim, text_dim * conv_mult) for _ in range(conv_layers)])
#         else:
#             self.extra_modeling = False

#     def forward(self, text: int['b nt'], seq_len):
#         text = self.text_embed(text) # b n -> b n d

#         # possible extra modeling
#         if self.extra_modeling:
#             # sinus pos emb
#             pos_idx = get_pos_embed_indices(torch.zeros(1, dtype=torch.int32), seq_len, max_pos=self.precompute_max_pos)
#             # convnextv2 blocks
#             text = self.text_blocks(text + self.freqs_cis[pos_idx])

#         return text

class InputEmbedding(Module):
    def __init__(self, mel_dim, text_dim, out_dim):
        super().__init__()
        self.proj = Linear(mel_dim * 2 + text_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim = out_dim)

    def forward(self, x: float['b n d'], cond: float['b n d'], drop_audio_cond = False):
        # if drop_audio_cond:  # cfg for cond audio
        x = self.proj(concat([x, cond], dim = -1))
        return self.conv_pos_embed(x) + x
    
# Transformer backbone using DiT blocks
# class F5TTS(PretrainedModel):
#     def __init__(self, config: PretrainedConfig):
#         super().__init__(config)
#         self.f5_transformer = DiT_transformer(config)
#         self.dtype = str_dtype_to_trt(config.dtype)
#         self.cfg_strength = 2

#     def forward(self,
#                 noise: float['b n d'],  # nosied input audio
#                 cond: float['b n d'],  # masked cond audio
#                 cond_drop: float['b n d'],
#                 time: float['b n'],  # time step
#                 rope_cos: float['b n d'],
#                 rope_sin: float['b n d'],
#                 t_scale: float['b'],
#                 mask: bool['b n'] | None = None):
        
#         pred = self.f5_transformer(x = noise, cond = cond, cond_drop = cond_drop, time = time, rope_cos = rope_cos, rope_sin = rope_sin, mask = mask)
#         pred, pred1 = chunk(pred, 2, dim = 0), chunk works only for static tensor
#         # cfg_strength = constant(np.array([self.cfg_strength], dtype = np.float32)).cast(noise.dtype)
#         # noise = noise + (pred_cond + (pred_cond - pred_uncond) * cfg_strength) * t_scale
#         noise.mark_output('denoised', self.dtype)
#         return noise



class F5TTS(PretrainedModel):
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.dtype = str_dtype_to_trt(config.dtype)

        self.time_embed = TimestepEmbedding(config.hidden_size) # √
        if config.text_dim is None:
            text_dim = config.mel_dim
        self.input_embed = InputEmbedding(config.mel_dim, config.text_dim, config.hidden_size)

        self.dim = config.hidden_size
        self.depth = config.num_hidden_layers
        self.transformer_blocks = ModuleList(
            [
                DiTBlock(
                    dim = self.dim,
                    heads = config.num_attention_heads,
                    dim_head = config.dim_head,
                    ff_mult = config.ff_mult,
                    dropout = config.dropout
                )
                for _ in range(self.depth)
            ]
        )
        
        self.norm_out = AdaLayerNormZero_Final(config.hidden_size)  # final modulation
        self.proj_out = Linear(config.hidden_size, config.mel_dim)

    def forward(
            self,
            noise: float['b n d'],  # nosied input audio
            cond: float['b n d'],  # masked cond audio
            time: float['b n'],  # time step
            rope_cos: float['b n d'] ,
            rope_sin: float['b n d'],
            input_lengths: int['b'],
            scale = 1.0
    ):
        t = self.time_embed(time)
        x = self.input_embed(noise, cond)
        # x = concat([self.input_embed(x, cond), self.input_embed(x, cond_drop)], dim = 0)
        
        for block in self.transformer_blocks:
            x = block(x, t, rope_cos = rope_cos, rope_sin = rope_sin, input_lengths=input_lengths, scale = scale)
        denoise = self.proj_out(self.norm_out(x, t))
        denoise.mark_output('denoised', self.dtype)
        return denoise

    def prepare_inputs(self, **kwargs):
        max_batch_size = kwargs['max_batch_size']
        batch_size_range = [2, 2, max_batch_size]
        mel_size = 100
        max_seq_len = 3000
        num_frames_range = [200, 2 * max_seq_len, max_seq_len * max_batch_size]
        hidden_size = 512
        concat_feature_dim = mel_size + hidden_size
        freq_embed_dim=256
        head_dim = 64
        mapping = self.config.mapping
        if mapping.tp_size > 1:
            current_all_reduce_helper().set_workspace_tensor(mapping, 1)
        if default_net().plugin_config.remove_input_padding:
            noise = Tensor(
                name='noise',
                dtype=self.dtype,
                shape=[-1, mel_size],
                dim_range=OrderedDict([
                    ('num_frames', [num_frames_range]),
                    ('n_mels', [mel_size]),
                ]))
            cond = Tensor(
                name='cond',
                dtype=self.dtype,
                shape=[-1, concat_feature_dim],
                dim_range=OrderedDict([
                    ('num_frames', [num_frames_range]),
                    ('embeded_length', [concat_feature_dim]),
            ]))
            time = Tensor(name='time',
                                dtype=self.dtype,
                                shape=[-1, freq_embed_dim],
                                dim_range=OrderedDict([
                                    ('num_frames', [num_frames_range]),
                                    ('freq_dim', [freq_embed_dim]),
                                ]))
            rope_cos = Tensor(name='rope_cos',
                                dtype=self.dtype,
                                shape=[-1, head_dim],
                                dim_range=OrderedDict([
                                    ('num_frames', [num_frames_range]),
                                    ('head_dim', [head_dim]),
                                ]))
            rope_sin = Tensor(name='rope_sin',
                                dtype=self.dtype,
                                shape=[-1, head_dim],
                                dim_range=OrderedDict([
                                    ('num_frames', [num_frames_range]),
                                    ('head_dim', [head_dim]),
                                ]))

        else:
            noise = Tensor(
                name='noise',
                dtype=self.dtype,
                shape=[-1, -1, mel_size],
                dim_range=OrderedDict([
                    ('batch_size', [batch_size_range]),
                    ('max_duratuion', [[100, max_seq_len // 2, max_seq_len]]),
                    ('n_mels', [mel_size]),
                ]))
            cond = Tensor(
                name='cond',
                dtype=self.dtype,
                shape=[-1, -1, concat_feature_dim],
                dim_range=OrderedDict([
                    ('batch_size', [batch_size_range]),
                    ('max_duratuion', [[100, max_seq_len // 2, max_seq_len]]),
                    ('embeded_length', [concat_feature_dim]),
            ]))
            print(233333333333333333333333333333333333333333333333333, batch_size_range)
            time = Tensor(name='time',
                                dtype=self.dtype,
                                shape=[-1, freq_embed_dim],
                                dim_range=OrderedDict([
                                    ('batch_size', [batch_size_range]),
                                    ('freq_dim', [freq_embed_dim]),
                                ]))
            rope_cos = Tensor(name='rope_cos',
                                dtype=self.dtype,
                                shape=[-1, -1, head_dim],
                                dim_range=OrderedDict([
                                    ('batch_size', [batch_size_range]),
                                    ('max_duratuion', [[100, max_seq_len // 2, max_seq_len]]),
                                    ('head_dim', [head_dim]),
                                ]))
            rope_sin = Tensor(name='rope_sin',
                                dtype=self.dtype,
                                shape=[-1, -1, head_dim],
                                dim_range=OrderedDict([
                                    ('batch_size', [batch_size_range]),
                                    ('max_duratuion', [[100, max_seq_len // 2, max_seq_len]]),
                                    ('head_dim', [head_dim]),
                                ]))
        input_lengths = Tensor(
            name="input_lengths",
            dtype=trt.int32,
            shape=[-1],
            dim_range=OrderedDict([("batch_size", [batch_size_range])]),
        )
        return {'noise': noise, 'cond': cond, 'time': time, 'rope_cos': rope_cos, 'rope_sin': rope_sin, 'input_lengths': input_lengths}