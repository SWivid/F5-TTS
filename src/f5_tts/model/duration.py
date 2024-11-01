# adapted from https://github.com/lucasnewman/f5-tts-mlx/blob/main/f5_tts_mlx/duration.py

"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


from einops import rearrange, repeat
import einx

# from f5_tts.model.backbones.dit import TextEmbedding, ConvPositionEmbedding
from f5_tts.model.mlx_modules import (
    Attention,
    # AttnProcessor, 
    FeedForward, 
    MelSpec,
    ConvPositionEmbedding,
    ConvNeXtV2Block,
    precompute_freqs_cis, 
    get_pos_embed_indices,
    RotaryEmbedding
    )
# from f5_tts.model.modules import MelSpec

from f5_tts.model.utils import (
    exists,
    default,
    list_str_to_idx,
    list_str_to_tensor,
    lens_to_mask,
    maybe_masked_mean,
)

device = "cuda" if torch.cuda.is_available() else "cpu"

SAMPLE_RATE = 24_000
HOP_LENGTH = 256
SAMPLES_PER_SECOND = SAMPLE_RATE / HOP_LENGTH



class Rearrange(nn.Module):
    def __init__(self, pattern: str):
        super().__init__()
        self.pattern = pattern

    def __call__(self, x: torch.array) -> torch.array:
        return rearrange(x, self.pattern)


class DurationInputEmbedding(nn.Module):
    def __init__(self, mel_dim, text_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(mel_dim + text_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)

    def __call__(
        self,
        x: float["b n d"],
        text_embed: float["b n d"],
    ):
        x = self.proj(torch.concatenate((x, text_embed), axis=-1))
        x = self.conv_pos_embed(x) + x
        return x


# Duration block


class DurationBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, ff_mult=4, dropout=0.1):
        super().__init__()

        self.attn_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
        )

        self.ff_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(
            dim=dim, mult=ff_mult, dropout=dropout, approximate="tanh"
        )

    def __call__(self, x, mask=None, rope=None):
        norm = self.attn_norm(x)

        # attention
        attn_output = self.attn(x=norm, mask=mask, rope=rope)

        # process attention output for input x
        x = x + attn_output

        norm = self.ff_norm(x)
        ff_output = self.ff(norm)
        x = x + ff_output

        return x


class DurationTransformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth=8,
        heads=8,
        dim_head=64,
        dropout=0.0,
        ff_mult=4,
        mel_dim=100,
        text_num_embeds=256,
        text_dim=None,
        conv_layers=0,
    ):
        super().__init__()

        if text_dim is None:
            text_dim = mel_dim
        self.text_embed = TextEmbedding(
            text_num_embeds, text_dim, conv_layers=conv_layers
        )
        self.input_embed = DurationInputEmbedding(mel_dim, text_dim, dim)

        self.rotary_embed = RotaryEmbedding(dim_head)

        self.dim = dim
        self.depth = depth

        self.transformer_blocks = nn.ModuleList([
            DurationBlock(
                dim=dim,
                heads=heads,
                dim_head=dim_head,
                ff_mult=ff_mult,
                dropout=dropout,
            )
            for _ in range(depth)
        ])

        self.norm_out = nn.RMSNorm(dim)

    def __call__(
        self,
        x: float["b n d"],  # nosied input audio
        text: int["b nt"],  # text
        mask: bool["b n"] | None = None,
    ):
        seq_len = x.shape[1]

        text_embed = self.text_embed(text, seq_len)

        x = self.input_embed(x, text_embed)

        rope = self.rotary_embed.forward_from_seq_len(seq_len)

        for block in self.transformer_blocks:
            x = block(x, mask=mask, rope=rope)

        x = self.norm_out(x)

        return x


class DurationPredictor(nn.Module):
    def __init__(
        self,
        transformer: DurationTransformer,
        num_channels=None,
        mel_spec_kwargs: dict = dict(),
        vocab_char_map: dict[str, int] | None = None,
    ):
        super().__init__()

        # mel spec

        self._mel_spec = MelSpec(**mel_spec_kwargs)
        num_channels = default(num_channels, self._mel_spec.n_mel_channels)
        self.num_channels = num_channels

        self.transformer = transformer
        dim = transformer.dim
        self.dim = dim

        self.dim = dim

        # vocab map for tokenization
        self._vocab_char_map = vocab_char_map

        # to prediction

        self.to_pred = nn.Sequential(
            nn.Linear(dim, 1, bias=False), nn.Softplus(), Rearrange("... 1 -> ...")
        )

    def __call__(
        self,
        inp: torch.Tensor["b n d"] | torch.Tensor["b nw"],  # mel or raw wave
        text: torch.Tensor | list[str],
        *,
        lens: torch.Tensor["b"] | None = None,
        return_loss=False,
    ):
        # handle raw wave
        if inp.ndim == 2:
            inp = self._mel_spec(inp)
            inp = rearrange(inp, "b d n -> b n d")
            assert inp.shape[-1] == self.num_channels

        batch, seq_len = inp.shape[:2]

        # handle text as string
        if isinstance(text, list):
            if exists(self._vocab_char_map):
                text = list_str_to_idx(text, self._vocab_char_map)
            else:
                text = list_str_to_tensor(text)
            assert text.shape[0] == batch

        if seq_len < text.shape[1]:
            seq_len = text.shape[1]
            inp = F.pad(inp, [(0, 0), (0, seq_len - inp.shape[1]), (0, 0)])

        # lens and mask
        if not exists(lens):
            lens = torch.full((batch,), seq_len)

        if seq_len < text.shape[1]:
            seq_len = text.shape[1]
            inp = F.pad(inp, [(0, 0), (0, seq_len - inp.shape[1]), (0, 0)])

        mask = lens_to_mask(lens, length=seq_len)
        

        # if returning a loss, mask out randomly from an index and have it predict the duration

        if return_loss:
            rand_frac_index = torch.random.uniform(0, 1, (batch,))
            rand_index = (rand_frac_index * lens).astype(torch.int32)

            seq = torch.arange(seq_len)
            mask &= einx.less("n, b -> b n", seq, rand_index)

        # attending
        
        mask.to(device)
        inp.to(device)

        inp = torch.where(
            repeat(mask, "b n -> b n d", d=self.num_channels).to(device), inp, torch.zeros_like(inp)
        )

        x = self.transformer(inp, text=text)

        x = maybe_masked_mean(x, mask)

        pred = self.to_pred(x)

        # return the prediction if not returning loss

        if not return_loss:
            return pred

        # loss

        duration = lens.to(pred.dtype) / SAMPLES_PER_SECOND

        return nn.losses.l1_loss(pred, duration)
    

class TextEmbedding(nn.Module):
    def __init__(self, text_num_embeds, text_dim, conv_layers=0, conv_mult=2):
        super().__init__()
        self.text_embed = nn.Embedding(
            text_num_embeds + 1, text_dim
        ).to(device)  # use 0 as filler token

        if conv_layers > 0:
            self.extra_modeling = True
            self.precompute_max_pos = 4096  # ~44s of 24khz audio
            self._freqs_cis = precompute_freqs_cis(text_dim, self.precompute_max_pos).to(device)
            self.text_blocks = nn.Sequential(
                *[
                    ConvNeXtV2Block(text_dim, text_dim * conv_mult)
                    for _ in range(conv_layers)
                ]
            )
        else:
            self.extra_modeling = False

    def forward(self, text: int["b nt"], seq_len, drop_text=False):
        batch, text_len = text.shape[0], text.shape[1]
        text = (
            text + 1
        )  # use 0 as filler token. preprocess of batch pad -1, see list_str_to_idx()
        text = text[
            :, :seq_len
        ]  # curtail if character tokens are more than the mel spec tokens
        pad_size = seq_len - text_len
        text = F.pad(text, (0, pad_size), value=0).to(device)  

        if drop_text:  # cfg for text
            text = torch.zeros_like(text)

        text = self.text_embed(text)  # b n -> b n d

        # possible extra modeling
        if self.extra_modeling:
            # sinus pos emb
            batch_start = torch.zeros((batch,), dtype=torch.long)
            pos_idx = get_pos_embed_indices(
                batch_start, seq_len, max_pos=self.precompute_max_pos
            ).to(torch.long).to(device)

            text_pos_embed = self._freqs_cis[pos_idx]
            text = text + text_pos_embed

            # convnextv2 blocks
            text = self.text_blocks(text)

        return text


# noised input audio and context mixing embedding

class InputEmbedding(nn.Module):
    def __init__(self, mel_dim, text_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(mel_dim * 2 + text_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)

    def forward(
        self,
        x: float["b n d"],
        cond: float["b n d"],
        text_embed: float["b n d"],
        drop_audio_cond=False,
    ):
        if drop_audio_cond:  # cfg for cond audio
            cond = torch.zeros_like(cond)

        x = self.proj(torch.concatenate((x, cond, text_embed), axis=-1))
        x = self.conv_pos_embed(x) + x
        return x
    