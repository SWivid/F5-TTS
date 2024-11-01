from __future__ import annotations
from functools import lru_cache
import math
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

import numpy as np

from einops import rearrange, repeat


# rotary positional embedding related

device = "cuda" if torch.cuda.is_available() else "cpu"

class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        use_xpos: bool = False,
        scale_base: int = 512,
        interpolation_factor: float = 1.0,
        base: float = 10000.0,
        base_rescale_factor: float = 1.0,
    ):
        super().__init__()
        base *= base_rescale_factor ** (dim / (dim - 2))

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.inv_freq = inv_freq
        # self.register_buffer('inv_freq', inv_freq)


        assert interpolation_factor >= 1.0
        self.interpolation_factor = interpolation_factor

        if not use_xpos:
            self.scale = None
            return

        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)

        self.scale_base = scale_base
        self.scale = scale

    def forward_from_seq_len(self, seq_len: int) -> tuple[torch.Tensor, float]:
        t = torch.arange(seq_len)
        return self(t)

    def __call__(self, t: torch.Tensor) -> tuple[torch.Tensor, float]:
        max_pos = t.max() + 1

        freqs = (
            torch.einsum("i , j -> i j", t.to(self.inv_freq.dtype), self.inv_freq)
            / self.interpolation_factor
        )
        freqs = torch.stack((freqs, freqs), axis=-1)
        freqs = rearrange(freqs, "... d r -> ... (d r)")

        if self.scale is None:
            return freqs, 1.0

        power = (t - (max_pos // 2)) / self.scale_base
        scale = self.scale ** rearrange(power, "n -> n 1")
        scale = torch.stack((scale, scale), axis=-1)
        scale = rearrange(scale, "... d r -> ... (d r)")

        return freqs, scale


def precompute_freqs_cis(
    dim: int, end: int, theta: float = 10000.0, theta_rescale_factor=1.0
):
    freqs = 1.0 / (
        theta ** (torch.arange(0, dim, 2)[: (dim // 2)].to(torch.float32) / dim)
    )
    t = torch.arange(end)  # type: ignore
    freqs = torch.outer(t, freqs).to(torch.float32)  # type: ignore
    freqs_cos = freqs.cos()  # real part
    freqs_sin = freqs.sin()  # imaginary part
    return torch.cat([freqs_cos, freqs_sin], axis=-1)


def get_pos_embed_indices(start, length, max_pos, scale=1.0):
    # length = length if isinstance(length, int) else length.max()
    scale = scale * torch.ones_like(start, dtype=torch.long).to(device)
    arange = torch.arange(length, dtype=torch.long).to(device)
    pos = start.unsqueeze(1).to(device) + (arange * scale.unsqueeze(1))
    pos.to(torch.long)
    pos = torch.clamp(pos, max=max_pos - 1)
    return pos


def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = [torch.squeeze(s, dim=-1) for s in torch.split(x, 1, dim=-1)]
    x = torch.stack([-x2, x1], dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


def apply_rotary_pos_emb(t, freqs, scale=1):
    device = t.device
    rot_dim, seq_len = freqs.shape[-1], t.shape[-2]
    freqs = freqs[-seq_len:, :].to(device)
    if isinstance(scale, torch.Tensor):
        scale = scale[-seq_len:, :].to(device)
    if t.ndim == 4 and freqs.ndim == 3:
        freqs = rearrange(freqs, "b n d -> b 1 n d").to(device)
    t, t_unrotated = t[..., :rot_dim].to(device), t[..., rot_dim:].to(device)
    rotated = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)
    out = torch.cat((rotated, t_unrotated), axis=-1)
    return out


# mel spectrogram


@lru_cache(maxsize=None)
def mel_filters(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    f_min: float = 0,
    f_max: Optional[float] = None,
    norm: Optional[str] = None,
    mel_scale: str = "htk",
) -> torch.Tensor:
    def hz_to_mel(freq, mel_scale="htk"):
        if mel_scale == "htk":
            return 2595.0 * math.log10(1.0 + freq / 700.0)

        # slaney scale
        f_min, f_sp = 0.0, 200.0 / 3
        mels = (freq - f_min) / f_sp
        min_log_hz = 1000.0
        min_log_mel = (min_log_hz - f_min) / f_sp
        logstep = math.log(6.4) / 27.0
        if freq >= min_log_hz:
            mels = min_log_mel + math.log(freq / min_log_hz) / logstep
        return mels

    def mel_to_hz(mels, mel_scale="htk"):
        if mel_scale == "htk":
            return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

        # slaney scale
        f_min, f_sp = 0.0, 200.0 / 3
        freqs = f_min + f_sp * mels
        min_log_hz = 1000.0
        min_log_mel = (min_log_hz - f_min) / f_sp
        logstep = math.log(6.4) / 27.0
        log_t = mels >= min_log_mel
        freqs[log_t] = min_log_hz * torch.exp(logstep * (mels[log_t] - min_log_mel))
        return freqs

    f_max = f_max or sample_rate / 2

    # generate frequency points

    n_freqs = n_fft // 2 + 1
    all_freqs = torch.linspace(0, sample_rate // 2, n_freqs)

    # convert frequencies to mel and back to hz

    m_min = hz_to_mel(f_min, mel_scale)
    m_max = hz_to_mel(f_max, mel_scale)
    m_pts = torch.linspace(m_min, m_max, n_mels + 2)
    f_pts = mel_to_hz(m_pts, mel_scale)

    # compute slopes for filterbank

    f_diff = f_pts[1:] - f_pts[:-1]
    slopes = torch.expand_dims(f_pts, 0) - torch.expand_dims(all_freqs, 1)

    # calculate overlapping triangular filters

    down_slopes = (-slopes[:, :-2]) / f_diff[:-1]
    up_slopes = slopes[:, 2:] / f_diff[1:]
    filterbank = torch.maximum(
        torch.zeros_like(down_slopes), torch.minimum(down_slopes, up_slopes)
    )

    if norm == "slaney":
        enorm = 2.0 / (f_pts[2 : n_mels + 2] - f_pts[:n_mels])
        filterbank *= torch.expand_dims(enorm, 0)

    filterbank = filterbank.moveaxis(0, 1)
    return filterbank


@lru_cache(maxsize=None)
def hanning(size):
    return torch.Tensor(np.hanning(size + 1)[:-1])


class MelSpec(nn.Module):
    def __init__(
        self,
        filter_length=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=100,
        target_sample_rate=24_000,
        normalize=False,
        power=1,
        norm=None,
        center=True,
    ):
        super().__init__()
        self.n_mel_channels = n_mel_channels

        self.mel_stft = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sample_rate,
            n_fft=filter_length,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mel_channels,
            power=power,
            center=center,
            normalized=normalize,
            norm=norm,
        ).to(device)
        
        self.mel_stft.spectrogram.register_buffer('window', self.mel_stft.spectrogram.window)
        self.mel_stft.mel_scale.register_buffer('fb', self.mel_stft.mel_scale.fb)

    def forward(self, inp):
        if len(inp.shape) == 3:
            inp = inp.squeeze(1)  # 'b 1 nw -> b nw'

        assert len(inp.shape) == 2


        mel = self.mel_stft(inp)
        mel = mel.clamp(min=1e-5).log()
        return mel


# sinusoidal position embedding


class SinusPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def __call__(self, x, scale=1000):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim) * -emb)
        emb = scale * torch.expand_dims(x, axis=1) * torch.expand_dims(emb, axis=0)
        emb = torch.concatenate([emb.sin(), emb.cos()], axis=-1)
        return emb


# convolutional position embedding


class Rearrange(nn.Module):
    def __init__(self, pattern):
        super().__init__()
        self.pattern = pattern

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return rearrange(x, self.pattern)


class ConvPositionEmbedding(nn.Module):
    def __init__(self, dim, kernel_size=31, groups=16):
        super().__init__()
        self.conv1d = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
        )

    def forward(self, x, mask=None):
        if mask is not None:
            mask = mask[..., None]
            x = x * mask

        # Transpose to (batch_size, channels, sequence_length)
        x = x.transpose(1, 2)
        x = self.conv1d(x)
        # Transpose back to (batch_size, sequence_length, channels)
        x = x.transpose(1, 2)

        if mask is not None:
            x = x * mask

        return x


# global response normalization


class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros((1, 1, dim)))
        self.beta = nn.Parameter(torch.zeros((1, 1, dim)))

    def __call__(self, x):
        Gx = torch.linalg.norm(x, ord=2, axis=1, keepdims=True)
        Nx = Gx / (Gx.mean(axis=-1, keepdims=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


# ConvNeXt-v2 block


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
        self.pwconv1 = nn.Linear(
            dim, intermediate_dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = x.transpose(1, 2)
        x = self.dwconv(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        return residual + x


# AdaLayerNormZero
# return with modulated x for attn input, and params for later mlp modulation


class AdaLayerNormZero(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 6)
        self.norm = nn.LayerNorm(dim, affine=False, eps=1e-6)

    def __call__(self, x: torch.Tensor, emb: torch.Tensor | None = None) -> torch.Tensor:
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.split(
            emb, 6, axis=1
        )

        x = self.norm(x) * (1 + torch.expand_dims(scale_msa, axis=1)) + torch.expand_dims(
            shift_msa, axis=1
        )
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


# AdaLayerNormZero for final layer
# return only with modulated x for attn input, cuz no more mlp modulation


class AdaLayerNormZero_Final(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 2)
        self.norm = nn.LayerNorm(dim, affine=False, eps=1e-6)

    def __call__(self, x: torch.Tensor, emb: torch.Tensor | None = None) -> torch.Tensor:
        emb = self.linear(self.silu(emb))
        scale, shift = torch.split(emb, 2, axis=1)

        x = self.norm(x) * (1 + torch.expand_dims(scale, axis=1)) + torch.expand_dims(
            shift, axis=1
        )
        return x


# feed forward


class FeedForward(nn.Module):
    def __init__(
        self, dim, dim_out=None, mult=4, dropout=0.0, approximate: str = "none"
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        activation = nn.GELU(approximate=approximate)
        project_in = nn.Sequential(nn.Linear(dim, inner_dim), activation)
        self.ff = nn.Sequential(
            project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out)
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff(x)


# attention


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.dim = dim
        self.heads = heads
        self.inner_dim = dim_head * heads
        self.dropout = dropout

        self.to_q = nn.Linear(dim, self.inner_dim)
        self.to_k = nn.Linear(dim, self.inner_dim)
        self.to_v = nn.Linear(dim, self.inner_dim)

        self.to_out = nn.Sequential(nn.Linear(self.inner_dim, dim), nn.Dropout(dropout))

    def __call__(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        rope: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        # `sample` projections.
        query = self.to_q(x)
        key = self.to_k(x)
        value = self.to_v(x)

        # apply rotary position embedding
        if rope is not None:
            freqs, xpos_scale = rope
            q_xpos_scale, k_xpos_scale = (
                (
                    xpos_scale,
                    xpos_scale**-1.0,
                )
                if xpos_scale is not None
                else (1.0, 1.0)
            )

            query = apply_rotary_pos_emb(query, freqs, q_xpos_scale)
            key = apply_rotary_pos_emb(key, freqs, k_xpos_scale)

        # attention
        query = rearrange(query, "b n (h d) -> b h n d", h=self.heads)
        key = rearrange(key, "b n (h d) -> b h n d", h=self.heads)
        value = rearrange(value, "b n (h d) -> b h n d", h=self.heads)

        # mask. e.g. inference got a batch with different target durations, mask out the padding
        if mask is not None:
            attn_mask = mask
            attn_mask = rearrange(attn_mask, "b n -> b () () n")
            attn_mask = repeat(attn_mask, "b () () n -> b h () n", h=self.heads)
        else:
            attn_mask = None

        # scale_factor = 1 / math.sqrt(query.shape[-1])

        
        x = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask, dropout_p=self.dropout, is_causal=False
        )
        x = rearrange(x, "b h n d -> b n (h d)")

        # linear proj
        x = self.to_out(x)

        if attn_mask is not None:
            mask = rearrange(mask, "b n -> b n 1")
            x = x.masked_fill(~mask, 0.0)

        return x


# time step conditioning embedding


class TimestepEmbedding(nn.Module):
    def __init__(self, dim, freq_embed_dim=256):
        super().__init__()
        self.time_embed = SinusPositionEmbedding(freq_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(freq_embed_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )

    def __call__(self, timestep: torch.Tensor) -> torch.Tensor:
        time_hidden = self.time_embed(timestep)
        time = self.time_mlp(time_hidden)  # b d
        return time