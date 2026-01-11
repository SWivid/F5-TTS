"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""
# ruff: noqa: F722 F821

from __future__ import annotations

from random import random
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torchdiffeq import odeint

from f5_tts.model.modules import MelSpec
from f5_tts.model.utils import (
    default,
    exists,
    get_epss_timesteps,
    lens_to_mask,
    list_str_to_idx,
    list_str_to_tensor,
    mask_from_frac_lengths,
)
from f5_tts.rl.difftools import odeint_rl


class CFM(nn.Module):
    def __init__(
        self,
        transformer: nn.Module,
        sigma=0.0,
        odeint_kwargs: dict = dict(
            # atol = 1e-5,
            # rtol = 1e-5,
            method="euler"  # 'midpoint'
        ),
        audio_drop_prob=0.3,
        cond_drop_prob=0.2,
        num_channels=None,
        mel_spec_module: nn.Module | None = None,
        mel_spec_kwargs: dict = dict(),
        frac_lengths_mask: tuple[float, float] = (0.7, 1.0),
        vocab_char_map: dict[str:int] | None = None,
        objective: str = "mse",
        output_dist: str | None = None,
        sample_from_dist: bool = False,
    ):
        super().__init__()

        self.frac_lengths_mask = frac_lengths_mask

        # mel spec
        self.mel_spec = default(mel_spec_module, MelSpec(**mel_spec_kwargs))
        num_channels = default(num_channels, self.mel_spec.n_mel_channels)
        self.num_channels = num_channels

        # classifier-free guidance
        self.audio_drop_prob = audio_drop_prob
        self.cond_drop_prob = cond_drop_prob

        # transformer
        self.transformer = transformer
        dim = transformer.dim
        self.dim = dim
        if output_dist is None:
            output_dist = getattr(transformer, "output_dist", "deterministic")
        if output_dist not in ("deterministic", "gaussian"):
            raise ValueError(f"output_dist must be 'deterministic' or 'gaussian', got {output_dist}")
        self.output_dist = output_dist
        if objective not in ("mse", "gaussian_nll", "grpo"):
            raise ValueError(f"objective must be 'mse', 'gaussian_nll', or 'grpo', got {objective}")
        self.objective = objective
        self.sample_from_dist = sample_from_dist

        # conditional flow related
        self.sigma = sigma

        # sampling related
        self.odeint_kwargs = odeint_kwargs

        # vocab map for tokenization
        self.vocab_char_map = vocab_char_map

    @property
    def device(self):
        return next(self.parameters()).device

    def forward_prob(
        self,
        x: float["b n d"],
        cond: float["b n d"],
        text: int["b nt"],
        time: float["b"] | float[""],
        mask: bool["b n"] | None = None,
        drop_audio_cond: bool = False,
        drop_text: bool = False,
    ):
        if not hasattr(self.transformer, "forward_prob"):
            raise RuntimeError("transformer does not support forward_prob")
        return self.transformer.forward_prob(
            x=x,
            cond=cond,
            text=text,
            time=time,
            mask=mask,
            drop_audio_cond=drop_audio_cond,
            drop_text=drop_text,
        )

    @torch.no_grad()
    def sample(
        self,
        cond: float["b n d"] | float["b nw"],
        text: int["b nt"] | list[str],
        duration: int | int["b"],
        *,
        lens: int["b"] | None = None,
        steps=32,
        cfg_strength=1.0,
        sway_sampling_coef=None,
        seed: int | None = None,
        max_duration=4096,
        vocoder: Callable[[float["b d n"]], float["b nw"]] | None = None,
        use_epss=True,
        no_ref_audio=False,
        duplicate_test=False,
        t_inter=0.1,
        edit_mask=None,
        sample_from_dist: bool | None = None,
    ):
        self.eval()
        # raw wave

        if cond.ndim == 2:
            cond = self.mel_spec(cond)
            cond = cond.permute(0, 2, 1)
            assert cond.shape[-1] == self.num_channels

        cond = cond.to(next(self.parameters()).dtype)

        batch, cond_seq_len, device = *cond.shape[:2], cond.device
        if not exists(lens):
            lens = torch.full((batch,), cond_seq_len, device=device, dtype=torch.long)

        # text

        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch

        # duration

        cond_mask = lens_to_mask(lens)
        if edit_mask is not None:
            cond_mask = cond_mask & edit_mask

        if isinstance(duration, int):
            duration = torch.full((batch,), duration, device=device, dtype=torch.long)

        duration = torch.maximum(
            torch.maximum((text != -1).sum(dim=-1), lens) + 1, duration
        )  # duration at least text/audio prompt length plus one token, so something is generated
        duration = duration.clamp(max=max_duration)
        max_duration = duration.amax()

        # duplicate test corner for inner time step oberservation
        if duplicate_test:
            test_cond = F.pad(cond, (0, 0, cond_seq_len, max_duration - 2 * cond_seq_len), value=0.0)

        cond = F.pad(cond, (0, 0, 0, max_duration - cond_seq_len), value=0.0)
        if no_ref_audio:
            cond = torch.zeros_like(cond)

        cond_mask = F.pad(cond_mask, (0, max_duration - cond_mask.shape[-1]), value=False)
        cond_mask = cond_mask.unsqueeze(-1)
        step_cond = torch.where(
            cond_mask, cond, torch.zeros_like(cond)
        )  # allow direct control (cut cond audio) with lens passed in

        if batch > 1:
            mask = lens_to_mask(duration)
        else:  # save memory and speed up, as single inference need no mask currently
            mask = None

        use_dist_sample = self.sample_from_dist if sample_from_dist is None else sample_from_dist

        # neural ode

        def fn(t, x):
            # at each step, conditioning is fixed
            # step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))

            # predict flow (cond)
            if cfg_strength < 1e-5:
                if self.output_dist == "gaussian":
                    mu, ln_sig = self.forward_prob(
                        x=x,
                        cond=step_cond,
                        text=text,
                        time=t,
                        mask=mask,
                        drop_audio_cond=False,
                        drop_text=False,
                    )
                    pred = mu
                    if use_dist_sample:
                        pred = mu + torch.randn_like(mu) * torch.exp(ln_sig)
                else:
                    pred = self.transformer(
                        x=x,
                        cond=step_cond,
                        text=text,
                        time=t,
                        mask=mask,
                        drop_audio_cond=False,
                        drop_text=False,
                        cache=True,
                    )
                return pred

            # predict flow (cond and uncond), for classifier-free guidance
            if self.output_dist == "gaussian":
                mu_cfg, ln_sig_cfg = self.transformer.forward_prob(
                    x=x,
                    cond=step_cond,
                    text=text,
                    time=t,
                    mask=mask,
                    cfg_infer=True,
                    cache=True,
                )
                mu, null_mu = torch.chunk(mu_cfg, 2, dim=0)
                ln_sig, null_ln_sig = torch.chunk(ln_sig_cfg, 2, dim=0)
                pred = mu
                null_pred = null_mu
                if use_dist_sample:
                    pred = mu + torch.randn_like(mu) * torch.exp(ln_sig)
                    null_pred = null_mu + torch.randn_like(null_mu) * torch.exp(null_ln_sig)
            else:
                pred_cfg = self.transformer(
                    x=x,
                    cond=step_cond,
                    text=text,
                    time=t,
                    mask=mask,
                    cfg_infer=True,
                    cache=True,
                )
                pred, null_pred = torch.chunk(pred_cfg, 2, dim=0)
            return pred + (pred - null_pred) * cfg_strength

        # noise input
        # to make sure batch inference result is same with different batch size, and for sure single inference
        # still some difference maybe due to convolutional layers
        y0 = []
        for dur in duration:
            if exists(seed):
                torch.manual_seed(seed)
            y0.append(torch.randn(dur, self.num_channels, device=self.device, dtype=step_cond.dtype))
        y0 = pad_sequence(y0, padding_value=0, batch_first=True)

        t_start = 0

        # duplicate test corner for inner time step oberservation
        if duplicate_test:
            t_start = t_inter
            y0 = (1 - t_start) * y0 + t_start * test_cond
            steps = int(steps * (1 - t_start))

        if t_start == 0 and use_epss:  # use Empirically Pruned Step Sampling for low NFE
            t = get_epss_timesteps(steps, device=self.device, dtype=step_cond.dtype)
        else:
            t = torch.linspace(t_start, 1, steps + 1, device=self.device, dtype=step_cond.dtype)
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)

        trajectory = odeint(fn, y0, t, **self.odeint_kwargs)
        self.transformer.clear_cache()

        sampled = trajectory[-1]
        out = sampled
        out = torch.where(cond_mask, cond, out)

        if exists(vocoder):
            out = out.permute(0, 2, 1)
            out = vocoder(out)

        return out, trajectory

    def forward(
        self,
        inp: float["b n d"] | float["b nw"],  # mel or raw wave
        text: int["b nt"] | list[str],
        *,
        lens: int["b"] | None = None,
        noise_scheduler: str | None = None,
    ):
        # handle raw wave
        if inp.ndim == 2:
            inp = self.mel_spec(inp)
            inp = inp.permute(0, 2, 1)
            assert inp.shape[-1] == self.num_channels

        batch, seq_len, dtype, device, _σ1 = *inp.shape[:2], inp.dtype, self.device, self.sigma

        # handle text as string
        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch

        # lens and mask
        if not exists(lens):  # if lens not acquired by trainer from collate_fn
            lens = torch.full((batch,), seq_len, device=device)
        mask = lens_to_mask(lens, length=seq_len)

        # get a random span to mask out for training conditionally
        frac_lengths = torch.zeros((batch,), device=self.device).float().uniform_(*self.frac_lengths_mask)
        rand_span_mask = mask_from_frac_lengths(lens, frac_lengths)

        if exists(mask):
            rand_span_mask &= mask

        # mel is x1
        x1 = inp

        # x0 is gaussian noise
        x0 = torch.randn_like(x1)

        # time step
        time = torch.rand((batch,), dtype=dtype, device=self.device)
        # TODO. noise_scheduler

        # sample xt (φ_t(x) in the paper)
        t = time.unsqueeze(-1).unsqueeze(-1)
        φ = (1 - t) * x0 + t * x1
        flow = x1 - x0

        # only predict what is within the random mask span for infilling
        cond = torch.where(rand_span_mask[..., None], torch.zeros_like(x1), x1)

        # transformer and cfg training with a drop rate
        drop_audio_cond = random() < self.audio_drop_prob  # p_drop in voicebox paper
        if random() < self.cond_drop_prob:  # p_uncond in voicebox paper
            drop_audio_cond = True
            drop_text = True
        else:
            drop_text = False

        # apply mask will use more memory; might adjust batchsize or batchsampler long sequence threshold
        if self.objective == "gaussian_nll":
            if self.output_dist != "gaussian":
                raise RuntimeError("gaussian_nll objective requires output_dist='gaussian'")
            mu, ln_sig = self.forward_prob(
                x=φ,
                cond=cond,
                text=text,
                time=time,
                mask=mask,
                drop_audio_cond=drop_audio_cond,
                drop_text=drop_text,
            )
            loss = F.mse_loss(mu, flow, reduction="none") / (2 * (torch.exp(ln_sig) ** 2) + 1e-6) + ln_sig
            # Match F5R gaussian objective: extra t^2 * ln_sig term for timestep-scaled variance regularization.
            loss += (t * t) * ln_sig
            loss = loss[rand_span_mask]
            if not torch.isfinite(loss).all():
                raise RuntimeError(
                    f"Non-finite gaussian loss detected. "
                    f"loss={loss.detach().mean().item()} "
                    f"mu_range=({mu.min().item()}, {mu.max().item()}) "
                    f"ln_sig_range=({ln_sig.min().item()}, {ln_sig.max().item()}) "
                    f"time_range=({time.min().item()}, {time.max().item()})"
                )
            return loss.mean(), cond, mu

        if self.objective == "grpo":
            raise RuntimeError("grpo objective should use forward_rl and GRPOTrainer")

        if self.output_dist == "gaussian" and hasattr(self.transformer, "forward_prob"):
            pred, _ = self.forward_prob(
                x=φ,
                cond=cond,
                text=text,
                time=time,
                mask=mask,
                drop_audio_cond=drop_audio_cond,
                drop_text=drop_text,
            )
        else:
            pred = self.transformer(
                x=φ, cond=cond, text=text, time=time, drop_audio_cond=drop_audio_cond, drop_text=drop_text, mask=mask
            )

        # flow matching loss
        loss = F.mse_loss(pred, flow, reduction="none")
        loss = loss[rand_span_mask]

        return loss.mean(), cond, pred

    def forward_rl(
        self,
        cond: float["b n d"],
        text: int["b nt"] | list[str],
        duration: int | int["b"],
        *,
        lens: int["b"] | None = None,
        steps=32,
        cfg_strength=1.0,
        sway_sampling_coef=None,
        seed: int | None = None,
        max_duration=4096,
        vocoder: Callable[[float["b d n"]], float["b nw"]] | None = None,
        no_ref_audio=False,
        duplicate_test=False,
        t_inter=0.1,
        edit_mask=None,
        set_train: bool = True,
    ):
        if self.output_dist != "gaussian":
            raise RuntimeError("forward_rl requires output_dist='gaussian'")
        if set_train:
            self.train()

        batch, cond_seq_len, device = cond.size(0), cond.size(1), cond.device
        if not exists(lens):
            lens = torch.full((batch,), cond_seq_len, device=device, dtype=torch.long)

        # text
        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch

        if exists(text):
            text_lens = (text != -1).sum(dim=-1)
            lens = torch.maximum(text_lens, lens)

        cond_mask = lens_to_mask(lens)
        if edit_mask is not None:
            cond_mask = cond_mask & edit_mask

        if isinstance(duration, int):
            duration = torch.full((batch,), duration, device=device, dtype=torch.long)

        duration = torch.maximum(lens + 1, duration)
        duration = duration.clamp(max=max_duration)
        max_duration = duration.amax()

        if duplicate_test:
            test_cond = F.pad(cond, (0, 0, cond_seq_len, max_duration - 2 * cond_seq_len), value=0.0)

        cond = F.pad(cond, (0, 0, 0, max_duration - cond_seq_len), value=0.0)
        cond_mask = F.pad(cond_mask, (0, max_duration - cond_mask.shape[-1]), value=False)
        cond_mask = cond_mask.unsqueeze(-1)
        step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))

        if batch > 1:
            mask = lens_to_mask(duration)
        else:
            mask = None

        if no_ref_audio:
            cond = torch.zeros_like(cond)

        def fn(t, x):
            mu, ln_sig = self.forward_prob(
                x=x,
                cond=step_cond,
                text=text,
                time=t,
                mask=mask,
                drop_audio_cond=False,
                drop_text=False,
            )
            pred = mu + torch.randn_like(mu) * torch.exp(ln_sig)
            if cfg_strength < 1e-5:
                return pred, mu, ln_sig

            null_mu, null_ln_sig = self.forward_prob(
                x=x,
                cond=step_cond,
                text=text,
                time=t,
                mask=mask,
                drop_audio_cond=True,
                drop_text=True,
            )
            null_pred = null_mu + torch.randn_like(null_mu) * torch.exp(null_ln_sig)
            pred_cfg = pred + (pred - null_pred) * cfg_strength
            return pred_cfg, mu, ln_sig

        y0 = []
        for dur in duration:
            if exists(seed):
                torch.manual_seed(seed)
            y0.append(torch.randn(dur, self.num_channels, device=self.device))
        y0 = pad_sequence(y0, padding_value=0, batch_first=True)

        t_start = 0
        if duplicate_test:
            t_start = t_inter
            y0 = (1 - t_start) * y0 + t_start * test_cond
            steps = int(steps * (1 - t_start))

        t = torch.linspace(t_start, 1, steps, device=self.device)
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)

        trajectory, pro_result = odeint_rl(fn, y0, t, **self.odeint_kwargs)
        if hasattr(self.transformer, "clear_cache"):
            self.transformer.clear_cache()

        sampled = trajectory[-1]
        out = torch.where(cond_mask, cond, sampled)

        if exists(vocoder):
            out = out.permute(0, 2, 1)
            out = vocoder(out)

        return out, trajectory, pro_result
