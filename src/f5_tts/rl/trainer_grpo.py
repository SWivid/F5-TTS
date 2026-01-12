from __future__ import annotations

import copy
import gc
import math
import os
from typing import Any

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from ema_pytorch import EMA
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from tqdm import tqdm

from f5_tts.infer.utils_infer import load_vocoder
from f5_tts.model import CFM
from f5_tts.model.dataset import DynamicBatchSampler, collate_fn
from f5_tts.model.utils import default, exists, load_state_dict_compat, mask_from_start_end_indices
from f5_tts.rewards import RewardCombiner, RewardInput


def sample_prompt_spans(seq_len, frac_lengths, mode: str = "min", rand: torch.Tensor | None = None):
    max_start = (frac_lengths * seq_len).long()
    if mode == "range":
        # Use the sampled fraction directly so the prompt length respects the configured lower bound.
        start = max_start.clamp(min=0)
    else:
        # F5R parity: sample within [0, frac*len] and optionally collapse to batch min.
        rand = (
            torch.rand_like(frac_lengths)
            if rand is None
            else rand.to(device=frac_lengths.device, dtype=frac_lengths.dtype)
        )
        start = (max_start * rand).long().clamp(min=0)
        if mode == "min":
            start = torch.min(start, dim=-1, keepdim=True).values.repeat(start.size(0))
        elif mode != "per_sample":
            raise ValueError(f"prompt_length_mode must be 'min', 'per_sample', or 'range', got {mode}")
    prompt_idx = mask_from_start_end_indices(seq_len, (0 * start).long(), start)
    trg_idx = mask_from_start_end_indices(seq_len, start, seq_len)
    return start, prompt_idx, trg_idx


class GRPOTrainer:
    def __init__(
        self,
        model: CFM,
        reward_combiner: RewardCombiner,
        epochs: int,
        learning_rate: float,
        num_warmup_updates: int = 20000,
        save_per_updates: int = 1000,
        keep_last_n_checkpoints: int = -1,
        checkpoint_path: str | None = None,
        batch_size_per_gpu: int = 32,
        batch_size_type: str = "sample",
        max_samples: int = 32,
        grad_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        noise_scheduler: str | None = None,
        duration_predictor: torch.nn.Module | None = None,
        logger: str | None = "wandb",
        wandb_project: str = "CFM-TTS",
        wandb_run_name: str = "grpo_run",
        wandb_resume_id: str | None = None,
        last_per_updates: int | None = None,
        accelerate_kwargs: dict | None = None,
        ema_kwargs: dict | None = None,
        mel_spec_type: str = "vocos",
        vocoder: Any | None = None,
        repeat_count: int = 8,
        mini_repeat_count: int = 1,
        prompt_frac_range: tuple[float, float] = (0.1, 0.3),
        steps: int = 30,
        steps_plus_one: bool = False,
        cfg_strength: float = 2.0,
        sway_sampling_coef: float | None = -1.0,
        kl_weight: float = 1.0,
        ref_model: CFM | None = None,
        ref_model_ckpt: str | None = None,
        ref_model_use_ema: bool = True,
        allow_extra_keys: bool = False,
        bnb_optimizer: bool = False,
        prompt_length_mode: str = "min",
    ):
        if accelerate_kwargs is None:
            accelerate_kwargs = {}
        if ema_kwargs is None:
            ema_kwargs = {}

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

        self.logger = logger
        self._trackio = None
        if self.logger == "trackio":
            try:
                import trackio as trackio_module
            except Exception as exc:  # noqa: BLE001
                raise ImportError(
                    "Trackio is required for logger='trackio'. Install with: pip install f5-tts[trackio]"
                ) from exc
            self._trackio = trackio_module
        elif self.logger == "wandb":
            try:
                import wandb  # noqa: F401
            except Exception:
                self.logger = None

        self.accelerator = Accelerator(
            log_with=self.logger if self.logger == "wandb" else None,
            kwargs_handlers=[ddp_kwargs],
            gradient_accumulation_steps=grad_accumulation_steps,
            **accelerate_kwargs,
        )

        if self.logger == "wandb":
            init_kwargs = {"wandb": {"resume": "allow", "name": wandb_run_name}}
            if exists(wandb_resume_id):
                init_kwargs["wandb"]["id"] = wandb_resume_id
            reward_providers = [provider.name for provider in reward_combiner.providers]
            tracker_config = {
                "epochs": epochs,
                "learning_rate": learning_rate,
                "num_warmup_updates": num_warmup_updates,
                "batch_size_per_gpu": batch_size_per_gpu,
                "batch_size_type": batch_size_type,
                "max_samples": max_samples,
                "grad_accumulation_steps": grad_accumulation_steps,
                "max_grad_norm": max_grad_norm,
                "gpus": self.accelerator.num_processes,
                "noise_scheduler": noise_scheduler,
                "repeat_count": repeat_count,
                "mini_repeat_count": mini_repeat_count,
                "prompt_frac_range": prompt_frac_range,
                "prompt_length_mode": prompt_length_mode,
                "steps": steps,
                "cfg_strength": cfg_strength,
                "sway_sampling_coef": sway_sampling_coef,
                "kl_weight": kl_weight,
                "reward_mode": reward_combiner.mode,
                "reward_weights": reward_combiner.weights,
                "reward_providers": reward_providers,
            }
            self.accelerator.init_trackers(
                project_name=wandb_project,
                init_kwargs=init_kwargs,
                config=tracker_config,
            )
        elif self.logger == "trackio":
            reward_providers = [provider.name for provider in reward_combiner.providers]
            tracker_config = {
                "epochs": epochs,
                "learning_rate": learning_rate,
                "num_warmup_updates": num_warmup_updates,
                "batch_size_per_gpu": batch_size_per_gpu,
                "batch_size_type": batch_size_type,
                "max_samples": max_samples,
                "grad_accumulation_steps": grad_accumulation_steps,
                "max_grad_norm": max_grad_norm,
                "gpus": self.accelerator.num_processes,
                "noise_scheduler": noise_scheduler,
                "repeat_count": repeat_count,
                "mini_repeat_count": mini_repeat_count,
                "prompt_frac_range": prompt_frac_range,
                "prompt_length_mode": prompt_length_mode,
                "steps": steps,
                "cfg_strength": cfg_strength,
                "sway_sampling_coef": sway_sampling_coef,
                "kl_weight": kl_weight,
                "reward_mode": reward_combiner.mode,
                "reward_weights": reward_combiner.weights,
                "reward_providers": reward_providers,
            }
            self._trackio.init(
                project=wandb_project,
                name=wandb_run_name,
                config=tracker_config,
                space_id=os.getenv("TRACKIO_SPACE_ID"),
                dataset_id=os.getenv("TRACKIO_DATASET_ID"),
                embed=False,
            )

        self.model = model
        self.reward_combiner = reward_combiner

        if self.is_main:
            self.ema_model = EMA(model, include_online_model=False, **ema_kwargs)
            self.ema_model.to(self.accelerator.device)

        self.epochs = epochs
        self.num_warmup_updates = num_warmup_updates
        self.save_per_updates = save_per_updates
        self.keep_last_n_checkpoints = keep_last_n_checkpoints
        self.last_per_updates = default(last_per_updates, save_per_updates)
        self.checkpoint_path = default(checkpoint_path, "ckpts/grpo_run")

        self.batch_size_per_gpu = batch_size_per_gpu
        self.batch_size_type = batch_size_type
        self.max_samples = max_samples
        self.grad_accumulation_steps = grad_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.vocoder_name = mel_spec_type
        self.vocoder = vocoder
        self.repeat_count = repeat_count
        self.mini_repeat_count = mini_repeat_count
        self.prompt_frac_range = prompt_frac_range
        self.steps = steps
        self.steps_plus_one = steps_plus_one
        self.cfg_strength = cfg_strength
        self.sway_sampling_coef = sway_sampling_coef
        self.kl_weight = kl_weight
        self.allow_extra_keys = allow_extra_keys
        self.prompt_length_mode = prompt_length_mode

        self.noise_scheduler = noise_scheduler
        self.duration_predictor = duration_predictor

        if bnb_optimizer:
            import bitsandbytes as bnb

            self.optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=learning_rate)
        else:
            self.optimizer = AdamW(model.parameters(), lr=learning_rate)
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)

        self.ref_model = self._init_ref_model(ref_model, ref_model_ckpt, ref_model_use_ema)
        self.ref_model.eval()
        self.ref_model = self.accelerator.prepare(self.ref_model)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def _log(self, payload: dict[str, float], step: int) -> None:
        if self.logger == "wandb":
            if self.accelerator.trackers:
                self.accelerator.log(payload, step=step)
                return
            try:
                import wandb

                if wandb.run is not None:
                    wandb.log(payload, step=step)
            except Exception:
                pass
        elif self.logger == "trackio":
            self._trackio.log(payload, step=step)

    def _format_reward_key(self, key: str) -> str:
        if "." in key:
            provider, metric = key.split(".", 1)
        else:
            provider, metric = "reward", key
        provider_label = {
            "wespeaker_sim": "speaker_similarity",
            "funasr_wer": "asr",
        }.get(provider, provider)
        metric_label = {
            ("wespeaker_sim", "sim"): "cosine",
            ("wespeaker_sim", "total"): "total",
            ("funasr_wer", "wer"): "word_error_rate",
            ("funasr_wer", "acc"): "accuracy",
            ("funasr_wer", "total"): "total",
        }.get((provider, metric), metric)
        return f"reward/{provider_label}/{metric_label}"

    def _init_ref_model(self, ref_model: CFM | None, ckpt_path: str | None, use_ema: bool) -> CFM:
        if ref_model is not None:
            return ref_model
        model_copy = copy.deepcopy(self.accelerator.unwrap_model(self.model))
        if ckpt_path:
            from f5_tts.infer.utils_infer import load_checkpoint

            model_copy = load_checkpoint(model_copy, ckpt_path, device="cpu", use_ema=use_ema)
        for param in model_copy.parameters():
            param.requires_grad = False
        return model_copy

    def save_checkpoint(self, update: int, last: bool = False) -> None:
        self.accelerator.wait_for_everyone()
        if not self.is_main:
            return
        checkpoint = dict(
            model_state_dict=self.accelerator.unwrap_model(self.model).state_dict(),
            optimizer_state_dict=self.optimizer.state_dict(),
            ema_model_state_dict=self.ema_model.state_dict(),
            scheduler_state_dict=self.scheduler.state_dict(),
            update=update,
        )
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        if last:
            self.accelerator.save(checkpoint, f"{self.checkpoint_path}/model_last.pt")
        else:
            if self.keep_last_n_checkpoints == 0:
                return
            self.accelerator.save(checkpoint, f"{self.checkpoint_path}/model_{update}.pt")
            if self.keep_last_n_checkpoints > 0:
                checkpoints = [
                    f
                    for f in os.listdir(self.checkpoint_path)
                    if f.startswith("model_") and f.endswith(".pt") and f != "model_last.pt"
                ]
                checkpoints.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
                while len(checkpoints) > self.keep_last_n_checkpoints:
                    oldest_checkpoint = checkpoints.pop(0)
                    os.remove(os.path.join(self.checkpoint_path, oldest_checkpoint))

    def load_checkpoint(self) -> int:
        if (
            not exists(self.checkpoint_path)
            or not os.path.exists(self.checkpoint_path)
            or not any(filename.endswith((".pt", ".safetensors")) for filename in os.listdir(self.checkpoint_path))
        ):
            return 0

        self.accelerator.wait_for_everyone()
        if "model_last.pt" in os.listdir(self.checkpoint_path):
            latest_checkpoint = "model_last.pt"
        else:
            checkpoints = [f for f in os.listdir(self.checkpoint_path) if f.startswith("model_")]
            checkpoints.sort(key=lambda x: int("".join(filter(str.isdigit, x))))
            latest_checkpoint = checkpoints[-1]

        checkpoint = torch.load(f"{self.checkpoint_path}/{latest_checkpoint}", map_location="cpu")
        output_dist = getattr(self.accelerator.unwrap_model(self.model).transformer, "output_dist", "deterministic")
        if self.is_main:
            load_state_dict_compat(
                self.ema_model,
                checkpoint["ema_model_state_dict"],
                allow_extra_keys=self.allow_extra_keys,
                output_dist=output_dist,
            )

        if "update" in checkpoint:
            load_state_dict_compat(
                self.accelerator.unwrap_model(self.model),
                checkpoint["model_state_dict"],
                allow_extra_keys=self.allow_extra_keys,
                output_dist=output_dist,
            )
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if self.scheduler:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            update = checkpoint["update"]
        else:
            checkpoint["model_state_dict"] = {
                k.replace("ema_model.", ""): v
                for k, v in checkpoint["ema_model_state_dict"].items()
                if k not in ["initted", "step"]
            }
            load_state_dict_compat(
                self.accelerator.unwrap_model(self.model),
                checkpoint["model_state_dict"],
                allow_extra_keys=self.allow_extra_keys,
                output_dist=output_dist,
            )
            update = 0

        del checkpoint
        gc.collect()
        return update

    def _kl_divergence(self, gen, ref):
        gen_mu, gen_sig = gen
        ref_mu, ref_sig = ref
        kl = ref_sig - gen_sig
        kl += (torch.exp(gen_sig) ** 2 + F.mse_loss(gen_mu, ref_mu, reduction="none")) / (2 * (torch.exp(ref_sig) ** 2))
        return kl

    def _get_kl(self, gen_pros, ref_pros):
        if not gen_pros or not ref_pros:
            return torch.tensor(0.0, device=self.model.device)
        loss = 0
        for gen, ref in zip(gen_pros, ref_pros):
            loss = loss + self._kl_divergence(gen[1:3], ref[1:3])
        return loss

    def _ensure_vocoder(self):
        if self.vocoder is None:
            self.vocoder = load_vocoder(vocoder_name=self.vocoder_name)

    def _decode_audio(self, mel: torch.Tensor) -> torch.Tensor:
        self._ensure_vocoder()
        mel = mel.to(torch.float32)
        if self.vocoder_name == "vocos":
            return self.vocoder.decode(mel)
        audio = self.vocoder(mel)
        if audio.dim() == 3 and audio.size(1) == 1:
            audio = audio.squeeze(1)
        return audio

    def train(self, train_dataset: Dataset, num_workers: int = 16, resumable_with_seed: int | None = None) -> None:
        if exists(resumable_with_seed):
            generator = torch.Generator()
            generator.manual_seed(resumable_with_seed)
        else:
            generator = None

        if self.batch_size_type == "sample":
            train_dataloader = DataLoader(
                train_dataset,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=num_workers > 0,
                batch_size=self.batch_size_per_gpu,
                shuffle=True,
                generator=generator,
            )
        elif self.batch_size_type == "frame":
            self.accelerator.even_batches = False
            sampler = SequentialSampler(train_dataset)
            batch_sampler = DynamicBatchSampler(
                sampler,
                self.batch_size_per_gpu,
                max_samples=self.max_samples,
                random_seed=resumable_with_seed,
                drop_residual=False,
                repeat_count=self.repeat_count,
                mini_repeat_count=self.mini_repeat_count,
            )
            train_dataloader = DataLoader(
                train_dataset,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=num_workers > 0,
                batch_sampler=batch_sampler,
            )
        else:
            raise ValueError(f"batch_size_type must be 'sample' or 'frame', got {self.batch_size_type}")

        warmup_updates = self.num_warmup_updates * self.accelerator.num_processes
        total_updates = max(1, math.ceil(len(train_dataloader) / self.grad_accumulation_steps) * self.epochs)
        if warmup_updates <= 0:
            decay_updates = max(1, total_updates)
            self.scheduler = LinearLR(self.optimizer, start_factor=1.0, end_factor=1e-8, total_iters=decay_updates)
        elif total_updates <= warmup_updates:
            self.scheduler = LinearLR(self.optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_updates)
        else:
            decay_updates = total_updates - warmup_updates
            warmup_scheduler = LinearLR(self.optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_updates)
            decay_scheduler = LinearLR(self.optimizer, start_factor=1.0, end_factor=1e-8, total_iters=decay_updates)
            self.scheduler = SequentialLR(
                self.optimizer, schedulers=[warmup_scheduler, decay_scheduler], milestones=[warmup_updates]
            )
        train_dataloader, self.scheduler = self.accelerator.prepare(train_dataloader, self.scheduler)
        start_update = self.load_checkpoint()
        global_update = start_update

        if exists(resumable_with_seed):
            orig_epoch_step = len(train_dataloader)
            start_step = start_update * self.grad_accumulation_steps
            skipped_epoch = int(start_step // orig_epoch_step)
            skipped_batch = start_step % orig_epoch_step
            skipped_dataloader = self.accelerator.skip_first_batches(train_dataloader, num_batches=skipped_batch)
        else:
            skipped_epoch = 0

        for epoch in range(skipped_epoch, self.epochs):
            self.model.train()
            if exists(resumable_with_seed) and epoch == skipped_epoch:
                progress_bar_initial = math.ceil(skipped_batch / self.grad_accumulation_steps)
                current_dataloader = skipped_dataloader
            else:
                progress_bar_initial = 0
                current_dataloader = train_dataloader

            if hasattr(train_dataloader, "batch_sampler") and hasattr(train_dataloader.batch_sampler, "set_epoch"):
                train_dataloader.batch_sampler.set_epoch(epoch)

            progress_bar = tqdm(
                range(math.ceil(len(train_dataloader) / self.grad_accumulation_steps)),
                desc=f"Epoch {epoch + 1}/{self.epochs}",
                unit="update",
                disable=not self.accelerator.is_local_main_process,
                initial=progress_bar_initial,
            )

            for batch in current_dataloader:
                with self.accelerator.accumulate(self.model):
                    text_inputs = batch["text"]
                    mel_spec = batch["mel"].permute(0, 2, 1)
                    mel_lengths = batch["mel_lengths"]
                    text_len = max(len(item) for item in text_inputs)
                    if text_len > max(mel_lengths):
                        continue

                    dur_loss = None
                    if self.duration_predictor is not None:
                        dur_loss = self.duration_predictor(mel_spec, lens=batch.get("durations"))

                    frac_lengths = torch.zeros((mel_spec.size(0),), device=self.model.device)
                    frac_lengths = frac_lengths.float().uniform_(*self.prompt_frac_range)
                    prompt_lens, prompt_idx, trg_idx = sample_prompt_spans(
                        mel_lengths, frac_lengths, mode=self.prompt_length_mode
                    )
                    if self.prompt_length_mode == "per_sample":
                        max_prompt_len = int(prompt_lens.max().item())
                        prompt_audio = mel_spec.new_zeros((mel_spec.size(0), max_prompt_len, mel_spec.size(-1)))
                        for idx, prompt_len in enumerate(prompt_lens.tolist()):
                            if prompt_len:
                                prompt_audio[idx, :prompt_len, :] = mel_spec[idx, :prompt_len, :]
                        prompt_lens_arg = prompt_lens
                    else:
                        prompt_idx = prompt_idx.unsqueeze(-1).repeat(1, 1, mel_spec.size(-1))
                        prompt_audio = mel_spec[prompt_idx].view(mel_spec.size(0), -1, mel_spec.size(-1))
                        prompt_lens_arg = None

                    out, _, pro_result = self.model.forward_rl(
                        cond=prompt_audio,
                        text=text_inputs,
                        duration=mel_lengths,
                        lens=prompt_lens_arg,
                        steps=self.steps,
                        steps_plus_one=self.steps_plus_one,
                        cfg_strength=self.cfg_strength,
                        sway_sampling_coef=self.sway_sampling_coef,
                    )
                    with torch.no_grad():
                        _, _, ref_pro_result = self.ref_model.forward_rl(
                            cond=prompt_audio,
                            text=text_inputs,
                            duration=mel_lengths,
                            lens=prompt_lens_arg,
                            steps=self.steps,
                            steps_plus_one=self.steps_plus_one,
                            cfg_strength=self.cfg_strength,
                            sway_sampling_coef=self.sway_sampling_coef,
                            set_train=False,
                        )

                    pro_result_sample = [item[:-1] for item in pro_result if item[-1]]
                    ref_pro_result_sample = [item[:-1] for item in ref_pro_result if item[-1]]

                    gen_mel = out.to(torch.float32).permute(0, 2, 1)
                    ref_mel = mel_spec.to(torch.float32).permute(0, 2, 1)
                    gen_audio = self._decode_audio(gen_mel).cpu()
                    ref_audio = self._decode_audio(ref_mel).cpu()

                    reward_inputs = []
                    for idx, text_item in enumerate(text_inputs):
                        reward_inputs.append(
                            RewardInput(
                                audio=gen_audio[idx],
                                text=text_item,
                                speaker_ref=ref_audio[idx],
                                sample_rate=self.model.mel_spec.target_sample_rate,
                                meta={},
                            )
                        )
                    reward_outputs = self.reward_combiner.compute(reward_inputs)
                    rewards = torch.stack([out.total_reward for out in reward_outputs]).to(self.model.device)

                    rewards_all = self.accelerator.gather_for_metrics(rewards.detach())
                    mean = rewards_all.mean()
                    std = rewards_all.std(unbiased=False)
                    advantages = (rewards - mean) / (std + 1e-4)

                    # Keep upstream F5R behavior: weight advantages by Gaussian density (not log-prob).
                    # This preserves parity with the reference implementation and avoids unintended RL changes.
                    pro_advantages = []
                    for x, mu, log_sig in pro_result_sample:
                        p = torch.exp(-F.mse_loss(mu, x, reduction="none") / (2 * (torch.exp(log_sig) ** 2)))
                        p = p / torch.exp(log_sig)
                        pro_advantages.append(p)
                    if pro_advantages:
                        pro_advantages = torch.stack(pro_advantages, dim=1)
                        advantages = advantages.view(advantages.size(0), 1, 1, 1)
                        pro_advantages = pro_advantages * advantages
                        trg_idx = trg_idx[:, None, :, None].repeat(
                            1, pro_advantages.size(1), 1, pro_advantages.size(-1)
                        )
                        pro_advantages = pro_advantages[trg_idx]
                        pro_advantages = pro_advantages.mean()
                    else:
                        pro_advantages = torch.tensor(0.0, device=self.model.device)

                    loss_kl = self._get_kl(pro_result_sample, ref_pro_result_sample).mean()
                    loss = -pro_advantages + self.kl_weight * loss_kl
                    self.accelerator.backward(loss)

                    if self.max_grad_norm > 0 and self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                if self.accelerator.sync_gradients:
                    if self.is_main:
                        self.ema_model.update()
                    global_update += 1
                    progress_bar.update(1)
                    progress_bar.set_postfix(update=str(global_update), loss=loss.item())

                if self.is_main and self.accelerator.sync_gradients:
                    log_payload = {
                        "loss": loss.item(),
                        "lr": self.scheduler.get_last_lr()[0],
                        "loss/kl": loss_kl.item(),
                        "loss/pro_adv": pro_advantages.item(),
                        "reward/mean": mean.item(),
                        "reward/std": std.item(),
                        "reward/min": rewards_all.min().item(),
                        "reward/max": rewards_all.max().item(),
                    }
                    if dur_loss is not None:
                        log_payload["loss/duration"] = dur_loss.item()
                    component_values: dict[str, list[torch.Tensor]] = {}
                    for output in reward_outputs:
                        for key, value in output.components.items():
                            component_values.setdefault(key, []).append(value.detach().float().cpu())
                    for key, values in component_values.items():
                        if values:
                            log_payload[self._format_reward_key(key)] = torch.stack(values).mean().item()
                    self._log(log_payload, step=global_update)

                if global_update % self.last_per_updates == 0 and self.accelerator.sync_gradients:
                    self.save_checkpoint(global_update, last=True)

                if global_update % self.save_per_updates == 0 and self.accelerator.sync_gradients:
                    self.save_checkpoint(global_update)

        self.save_checkpoint(global_update, last=True)
        self.accelerator.end_training()
        if self.logger == "trackio":
            self._trackio.finish()
