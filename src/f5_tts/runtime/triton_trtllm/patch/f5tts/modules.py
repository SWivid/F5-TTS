from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from tensorrt_llm._common import default_net

from ..._utils import str_dtype_to_trt, trt_dtype_to_np
from ...functional import (
    Tensor,
    bert_attention,
    cast,
    chunk,
    concat,
    constant,
    expand,
    expand_dims,
    expand_dims_like,
    expand_mask,
    gelu,
    matmul,
    permute,
    shape,
    silu,
    slice,
    softmax,
    squeeze,
    unsqueeze,
    view,
)
from ...layers import ColumnLinear, Conv1d, LayerNorm, Linear, Mish, RowLinear
from ...module import Module


class FeedForward(Module):
    def __init__(self, dim, dim_out=None, mult=4, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        self.project_in = Linear(dim, inner_dim)
        self.ff = Linear(inner_dim, dim_out)

    def forward(self, x):
        return self.ff(gelu(self.project_in(x)))


class AdaLayerNormZero(Module):
    def __init__(self, dim):
        super().__init__()

        self.linear = Linear(dim, dim * 6)
        self.norm = LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb=None):
        emb = self.linear(silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = chunk(emb, 6, dim=1)
        x = self.norm(x)
        ones = constant(np.ones(1, dtype=np.float32)).cast(x.dtype)
        if default_net().plugin_config.remove_input_padding:
            x = x * (ones + scale_msa) + shift_msa
        else:
            x = x * (ones + unsqueeze(scale_msa, 1)) + unsqueeze(shift_msa, 1)
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormZero_Final(Module):
    def __init__(self, dim):
        super().__init__()

        self.linear = Linear(dim, dim * 2)

        self.norm = LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb):
        emb = self.linear(silu(emb))
        scale, shift = chunk(emb, 2, dim=1)
        ones = constant(np.ones(1, dtype=np.float32)).cast(x.dtype)
        if default_net().plugin_config.remove_input_padding:
            x = self.norm(x) * (ones + scale) + shift
        else:
            x = self.norm(x) * unsqueeze((ones + scale), 1)
            x = x + unsqueeze(shift, 1)
        return x


class ConvPositionEmbedding(Module):
    def __init__(self, dim, kernel_size=31, groups=16):
        super().__init__()
        assert kernel_size % 2 != 0
        self.conv1d1 = Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2)
        self.conv1d2 = Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2)
        self.mish = Mish()

    def forward(self, x, mask=None):  # noqa: F722
        if default_net().plugin_config.remove_input_padding:
            x = unsqueeze(x, 0)
        x = permute(x, [0, 2, 1])
        x = self.mish(self.conv1d2(self.mish(self.conv1d1(x))))
        out = permute(x, [0, 2, 1])
        if default_net().plugin_config.remove_input_padding:
            out = squeeze(out, 0)
        return out


class Attention(Module):
    def __init__(
        self,
        processor: AttnProcessor,
        dim: int,
        heads: int = 16,
        dim_head: int = 64,
        dropout: float = 0.0,
        context_dim: Optional[int] = None,  # if not None -> joint attention
        context_pre_only=None,
    ):
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("Attention equires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        self.processor = processor

        self.dim = dim  # hidden_size
        self.heads = heads
        self.inner_dim = dim_head * heads
        self.dropout = dropout
        self.attention_head_size = dim_head
        self.context_dim = context_dim
        self.context_pre_only = context_pre_only
        self.tp_size = 1
        self.num_attention_heads = heads // self.tp_size
        self.num_attention_kv_heads = heads // self.tp_size  # 8
        self.dtype = str_dtype_to_trt("float32")
        self.attention_hidden_size = self.attention_head_size * self.num_attention_heads
        self.to_q = ColumnLinear(
            dim,
            self.tp_size * self.num_attention_heads * self.attention_head_size,
            bias=True,
            dtype=self.dtype,
            tp_group=None,
            tp_size=self.tp_size,
        )
        self.to_k = ColumnLinear(
            dim,
            self.tp_size * self.num_attention_heads * self.attention_head_size,
            bias=True,
            dtype=self.dtype,
            tp_group=None,
            tp_size=self.tp_size,
        )
        self.to_v = ColumnLinear(
            dim,
            self.tp_size * self.num_attention_heads * self.attention_head_size,
            bias=True,
            dtype=self.dtype,
            tp_group=None,
            tp_size=self.tp_size,
        )

        if self.context_dim is not None:
            self.to_k_c = Linear(context_dim, self.inner_dim)
            self.to_v_c = Linear(context_dim, self.inner_dim)
            if self.context_pre_only is not None:
                self.to_q_c = Linear(context_dim, self.inner_dim)

        self.to_out = RowLinear(
            self.tp_size * self.num_attention_heads * self.attention_head_size,
            dim,
            bias=True,
            dtype=self.dtype,
            tp_group=None,
            tp_size=self.tp_size,
        )

        if self.context_pre_only is not None and not self.context_pre_only:
            self.to_out_c = Linear(self.inner_dim, dim)

    def forward(
        self,
        x,  # noised input x
        rope_cos,
        rope_sin,
        input_lengths,
        c=None,  # context c
        scale=1.0,
        rope=None,
        c_rope=None,  # rotary position embedding for c
    ) -> torch.Tensor:
        if c is not None:
            return self.processor(self, x, c=c, input_lengths=input_lengths, scale=scale, rope=rope, c_rope=c_rope)
        else:
            return self.processor(
                self, x, rope_cos=rope_cos, rope_sin=rope_sin, input_lengths=input_lengths, scale=scale
            )


def rotate_every_two_3dim(tensor: Tensor) -> Tensor:
    shape_tensor = concat(
        [shape(tensor, i) / 2 if i == (tensor.ndim() - 1) else shape(tensor, i) for i in range(tensor.ndim())]
    )
    if default_net().plugin_config.remove_input_padding:
        assert tensor.ndim() == 2
        x1 = slice(tensor, [0, 0], shape_tensor, [1, 2])
        x2 = slice(tensor, [0, 1], shape_tensor, [1, 2])
        x1 = expand_dims(x1, 2)
        x2 = expand_dims(x2, 2)
        zero = constant(np.ascontiguousarray(np.zeros([1], dtype=trt_dtype_to_np(tensor.dtype))))
        x2 = zero - x2
        x = concat([x2, x1], 2)
        out = view(x, concat([shape(x, 0), shape(x, 1) * 2]))
    else:
        assert tensor.ndim() == 3

        x1 = slice(tensor, [0, 0, 0], shape_tensor, [1, 1, 2])
        x2 = slice(tensor, [0, 0, 1], shape_tensor, [1, 1, 2])
        x1 = expand_dims(x1, 3)
        x2 = expand_dims(x2, 3)
        zero = constant(np.ascontiguousarray(np.zeros([1], dtype=trt_dtype_to_np(tensor.dtype))))
        x2 = zero - x2
        x = concat([x2, x1], 3)
        out = view(x, concat([shape(x, 0), shape(x, 1), shape(x, 2) * 2]))

    return out


def apply_rotary_pos_emb_3dim(x, rope_cos, rope_sin):
    if default_net().plugin_config.remove_input_padding:
        rot_dim = shape(rope_cos, -1)  # 64
        new_t_shape = concat([shape(x, 0), rot_dim])  # (-1, 64)
        x_ = slice(x, [0, 0], new_t_shape, [1, 1])
        end_dim = shape(x, -1) - shape(rope_cos, -1)
        new_t_unrotated_shape = concat([shape(x, 0), end_dim])  # (2, -1, 960)
        x_unrotated = slice(x, concat([0, rot_dim]), new_t_unrotated_shape, [1, 1])
        out = concat([x_ * rope_cos + rotate_every_two_3dim(x_) * rope_sin, x_unrotated], dim=-1)
    else:
        rot_dim = shape(rope_cos, 2)  # 64
        new_t_shape = concat([shape(x, 0), shape(x, 1), rot_dim])  # (2, -1, 64)
        x_ = slice(x, [0, 0, 0], new_t_shape, [1, 1, 1])
        end_dim = shape(x, 2) - shape(rope_cos, 2)
        new_t_unrotated_shape = concat([shape(x, 0), shape(x, 1), end_dim])  # (2, -1, 960)
        x_unrotated = slice(x, concat([0, 0, rot_dim]), new_t_unrotated_shape, [1, 1, 1])
        out = concat([x_ * rope_cos + rotate_every_two_3dim(x_) * rope_sin, x_unrotated], dim=-1)
    return out


class AttnProcessor:
    def __init__(self):
        pass

    def __call__(
        self,
        attn,
        x,  # noised input x
        rope_cos,
        rope_sin,
        input_lengths,
        scale=1.0,
        rope=None,
    ) -> torch.FloatTensor:
        query = attn.to_q(x)
        key = attn.to_k(x)
        value = attn.to_v(x)
        # k,v,q all (2,1226,1024)
        query = apply_rotary_pos_emb_3dim(query, rope_cos, rope_sin)
        key = apply_rotary_pos_emb_3dim(key, rope_cos, rope_sin)

        # attention
        inner_dim = key.shape[-1]
        norm_factor = math.sqrt(attn.attention_head_size)
        q_scaling = 1.0 / norm_factor
        mask = None
        if not default_net().plugin_config.remove_input_padding:
            N = shape(x, 1)
            B = shape(x, 0)
            seq_len_2d = concat([1, N])
            max_position_embeddings = 4096
            # create position ids
            position_ids_buffer = constant(np.expand_dims(np.arange(max_position_embeddings).astype(np.int32), 0))
            tmp_position_ids = slice(position_ids_buffer, starts=[0, 0], sizes=seq_len_2d)
            tmp_position_ids = expand(tmp_position_ids, concat([B, N]))  # BxL
            tmp_input_lengths = unsqueeze(input_lengths, 1)  # Bx1
            tmp_input_lengths = expand(tmp_input_lengths, concat([B, N]))  # BxL
            mask = tmp_position_ids < tmp_input_lengths  # BxL
            mask = mask.cast("int32")

        if default_net().plugin_config.bert_attention_plugin:
            qkv = concat([query, key, value], dim=-1)
            # TRT plugin mode
            assert input_lengths is not None
            if default_net().plugin_config.remove_input_padding:
                qkv = qkv.view(concat([-1, 3 * inner_dim]))
                max_input_length = constant(
                    np.zeros(
                        [
                            2048,
                        ],
                        dtype=np.int32,
                    )
                )
            else:
                max_input_length = None
            context = bert_attention(
                qkv,
                input_lengths,
                attn.num_attention_heads,
                attn.attention_head_size,
                q_scaling=q_scaling,
                max_input_length=max_input_length,
            )
        else:
            assert not default_net().plugin_config.remove_input_padding

            def transpose_for_scores(x):
                new_x_shape = concat([shape(x, 0), shape(x, 1), attn.num_attention_heads, attn.attention_head_size])

                y = x.view(new_x_shape)
                y = y.transpose(1, 2)
                return y

            def transpose_for_scores_k(x):
                new_x_shape = concat([shape(x, 0), shape(x, 1), attn.num_attention_heads, attn.attention_head_size])

                y = x.view(new_x_shape)
                y = y.permute([0, 2, 3, 1])
                return y

            query = transpose_for_scores(query)
            key = transpose_for_scores_k(key)
            value = transpose_for_scores(value)

            attention_scores = matmul(query, key, use_fp32_acc=False)

            if mask is not None:
                attention_mask = expand_mask(mask, shape(query, 2))
                attention_mask = cast(attention_mask, attention_scores.dtype)
                attention_scores = attention_scores + attention_mask

            attention_probs = softmax(attention_scores, dim=-1)

            context = matmul(attention_probs, value, use_fp32_acc=False).transpose(1, 2)
            context = context.view(concat([shape(context, 0), shape(context, 1), attn.attention_hidden_size]))
        context = attn.to_out(context)
        if mask is not None:
            mask = mask.view(concat([shape(mask, 0), shape(mask, 1), 1]))
            mask = expand_dims_like(mask, context)
            mask = cast(mask, context.dtype)
            context = context * mask
        return context


# DiT Block
class DiTBlock(Module):
    def __init__(self, dim, heads, dim_head, ff_mult=2, dropout=0.1):
        super().__init__()

        self.attn_norm = AdaLayerNormZero(dim)
        self.attn = Attention(
            processor=AttnProcessor(),
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
        )

        self.ff_norm = LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, mult=ff_mult, dropout=dropout)

    def forward(
        self, x, t, rope_cos, rope_sin, input_lengths, scale=1.0, rope=ModuleNotFoundError
    ):  # x: noised input, t: time embedding
        # pre-norm & modulation for attention input
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, emb=t)
        # attention
        # norm ----> (2,1226,1024)
        attn_output = self.attn(x=norm, rope_cos=rope_cos, rope_sin=rope_sin, input_lengths=input_lengths, scale=scale)

        # process attention output for input x
        if default_net().plugin_config.remove_input_padding:
            x = x + gate_msa * attn_output
        else:
            x = x + unsqueeze(gate_msa, 1) * attn_output
        ones = constant(np.ones(1, dtype=np.float32)).cast(x.dtype)
        if default_net().plugin_config.remove_input_padding:
            norm = self.ff_norm(x) * (ones + scale_mlp) + shift_mlp
        else:
            norm = self.ff_norm(x) * (ones + unsqueeze(scale_mlp, 1)) + unsqueeze(shift_mlp, 1)
            # norm = self.ff_norm(x) * (ones + scale_mlp) + shift_mlp
        ff_output = self.ff(norm)
        if default_net().plugin_config.remove_input_padding:
            x = x + gate_mlp * ff_output
        else:
            x = x + unsqueeze(gate_mlp, 1) * ff_output

        return x


class TimestepEmbedding(Module):
    def __init__(self, dim, freq_embed_dim=256, dtype=None):
        super().__init__()
        # self.time_embed = SinusPositionEmbedding(freq_embed_dim)
        self.mlp1 = Linear(freq_embed_dim, dim, bias=True, dtype=dtype)
        self.mlp2 = Linear(dim, dim, bias=True, dtype=dtype)

    def forward(self, timestep):
        t_freq = self.mlp1(timestep)
        t_freq = silu(t_freq)
        t_emb = self.mlp2(t_freq)
        return t_emb
