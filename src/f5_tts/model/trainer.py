from __future__ import annotations

import os
import gc
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    print("TensorBoard is not installed")

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from torch.optim.lr_scheduler import LinearLR, SequentialLR

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

from ema_pytorch import EMA

from f5_tts.model import CFM
from f5_tts.model.utils import exists, default
from f5_tts.model.dataset import DynamicBatchSampler, collate_fn

import numpy as np
import matplotlib.pyplot as plt
# trainer

# audio imports
import torchaudio
import soundfile as sf
from vocos import Vocos
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# -----------------------------------------
target_sample_rate = 24000
hop_length = 256
nfe_step = 16
cfg_strength = 2.0
sway_sampling_coef = -1.0
# -----------------------------------------


class Trainer:
    def __init__(
        self,
        model: CFM,
        epochs,
        learning_rate,
        num_warmup_updates=20000,
        save_per_updates=1000,
        checkpoint_path=None,
        batch_size=32,
        batch_size_type: str = "sample",
        max_samples=32,
        grad_accumulation_steps=1,
        max_grad_norm=1.0,
        noise_scheduler: str | None = None,
        duration_predictor: torch.nn.Module | None = None,
        logger: str = "wandb",  # Add logger parameter wandb,tensorboard , none
        log_dir: str = "logs",  # Add log directory parameter
        wandb_project="test_e2-tts",
        wandb_run_name="test_run",
        wandb_resume_id: str = None,
        last_per_steps=None,
        accelerate_kwargs: dict = dict(),
        ema_kwargs: dict = dict(),
        bnb_optimizer: bool = False,
        export_samples=False,
    ):
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

        self.logger = logger
        if self.logger == "wandb":
            self.accelerator = Accelerator(
                log_with="wandb",
                kwargs_handlers=[ddp_kwargs],
                gradient_accumulation_steps=grad_accumulation_steps,
                **accelerate_kwargs,
            )

            if exists(wandb_resume_id):
                init_kwargs = {"wandb": {"resume": "allow", "name": wandb_run_name, "id": wandb_resume_id}}
            else:
                init_kwargs = {"wandb": {"resume": "allow", "name": wandb_run_name}}

            self.accelerator.init_trackers(
                project_name=wandb_project,
                init_kwargs=init_kwargs,
                config={
                    "epochs": epochs,
                    "learning_rate": learning_rate,
                    "num_warmup_updates": num_warmup_updates,
                    "batch_size": batch_size,
                    "batch_size_type": batch_size_type,
                    "max_samples": max_samples,
                    "grad_accumulation_steps": grad_accumulation_steps,
                    "max_grad_norm": max_grad_norm,
                    "gpus": self.accelerator.num_processes,
                    "noise_scheduler": noise_scheduler,
                },
            )
        elif self.logger == "tensorboard":
            self.accelerator = Accelerator(
                kwargs_handlers=[ddp_kwargs],
                gradient_accumulation_steps=grad_accumulation_steps,
                **accelerate_kwargs,
            )
            if self.is_main:
                path_log_dir = os.path.join(log_dir, wandb_project)
                os.makedirs(path_log_dir, exist_ok=True)
                existing_folders = [folder for folder in os.listdir(path_log_dir) if folder.startswith("exp")]
                next_number = len(existing_folders) + 2
                folder_name = f"exp{next_number}"
                folder_path = os.path.join(path_log_dir, folder_name)
                os.makedirs(folder_path, exist_ok=True)

                self.writer = SummaryWriter(log_dir=folder_path)

        # export audio and mel
        self.export_samples = export_samples
        if self.export_samples:
            self.path_ckpts_project = checkpoint_path

            self.vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")
            self.vocos.to("cpu")
            self.file_path_samples = os.path.join(self.path_ckpts_project, "samples")
            os.makedirs(self.file_path_samples, exist_ok=True)

        self.model = model

        if self.is_main:
            self.ema_model = EMA(model, include_online_model=False, **ema_kwargs)
            self.ema_model.to(self.accelerator.device)

        self.epochs = epochs
        self.num_warmup_updates = num_warmup_updates
        self.save_per_updates = save_per_updates
        self.last_per_steps = default(last_per_steps, save_per_updates * grad_accumulation_steps)
        self.checkpoint_path = default(checkpoint_path, "ckpts/test_e2-tts")

        self.batch_size = batch_size
        self.batch_size_type = batch_size_type
        self.max_samples = max_samples
        self.grad_accumulation_steps = grad_accumulation_steps
        self.max_grad_norm = max_grad_norm

        self.noise_scheduler = noise_scheduler

        self.duration_predictor = duration_predictor

        if bnb_optimizer:
            import bitsandbytes as bnb

            self.optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=learning_rate)
        else:
            self.optimizer = AdamW(model.parameters(), lr=learning_rate)
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def save_checkpoint(self, step, last=False):
        self.accelerator.wait_for_everyone()
        if self.is_main:
            checkpoint = dict(
                model_state_dict=self.accelerator.unwrap_model(self.model).state_dict(),
                optimizer_state_dict=self.accelerator.unwrap_model(self.optimizer).state_dict(),
                ema_model_state_dict=self.ema_model.state_dict(),
                scheduler_state_dict=self.scheduler.state_dict(),
                step=step,
            )
            if not os.path.exists(self.checkpoint_path):
                os.makedirs(self.checkpoint_path)
            if last:
                self.accelerator.save(checkpoint, f"{self.checkpoint_path}/model_last.pt")
                print(f"Saved last checkpoint at step {step}")
            else:
                self.accelerator.save(checkpoint, f"{self.checkpoint_path}/model_{step}.pt")

    def load_checkpoint(self):
        if (
            not exists(self.checkpoint_path)
            or not os.path.exists(self.checkpoint_path)
            or not os.listdir(self.checkpoint_path)
        ):
            return 0

        self.accelerator.wait_for_everyone()
        if "model_last.pt" in os.listdir(self.checkpoint_path):
            latest_checkpoint = "model_last.pt"
        else:
            latest_checkpoint = sorted(
                [f for f in os.listdir(self.checkpoint_path) if f.endswith(".pt")],
                key=lambda x: int("".join(filter(str.isdigit, x))),
            )[-1]
        # checkpoint = torch.load(f"{self.checkpoint_path}/{latest_checkpoint}", map_location=self.accelerator.device)  # rather use accelerator.load_state ಥ_ಥ
        checkpoint = torch.load(f"{self.checkpoint_path}/{latest_checkpoint}", weights_only=True, map_location="cpu")

        if self.is_main:
            self.ema_model.load_state_dict(checkpoint["ema_model_state_dict"])

        if "step" in checkpoint:
            self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint["model_state_dict"])
            self.accelerator.unwrap_model(self.optimizer).load_state_dict(checkpoint["optimizer_state_dict"])
            if self.scheduler:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            step = checkpoint["step"]
        else:
            checkpoint["model_state_dict"] = {
                k.replace("ema_model.", ""): v
                for k, v in checkpoint["ema_model_state_dict"].items()
                if k not in ["initted", "step"]
            }
            self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint["model_state_dict"])
            step = 0

        del checkpoint
        gc.collect()
        return step

    def log(self, metrics, step):
        """Unified logging method for both WandB and TensorBoard"""
        if self.logger == "none":
            return
        if self.logger == "wandb":
            self.accelerator.log(metrics, step=step)
        elif self.is_main:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, step)

    def export_add_log(self, global_step, mel_org, text_inputs):
        try:
            generated_wave_org = self.vocos.decode(mel_org.unsqueeze(0).cpu())
            generated_wave_org = generated_wave_org.squeeze().cpu().numpy()
            file_wav_org = os.path.join(self.file_path_samples, f"step_{global_step}_org.wav")
            sf.write(file_wav_org, generated_wave_org, target_sample_rate)

            audio, sr = torchaudio.load(file_wav_org)
            audio = audio.to("cuda")

            ref_audio_len = audio.shape[-1] // hop_length
            text = [text_inputs[0] + [" . "] + text_inputs[0]]
            duration = int((audio.shape[1] / 256) * 2.0)

            with torch.inference_mode():
                generated_gen, _ = self.model.sample(
                    cond=audio,
                    text=text,
                    duration=duration,
                    steps=nfe_step,
                    cfg_strength=cfg_strength,
                    sway_sampling_coef=sway_sampling_coef,
                )

            generated_gen = generated_gen.to(torch.float32)
            generated_gen = generated_gen[:, ref_audio_len:, :]
            generated_mel_spec_gen = generated_gen.permute(0, 2, 1)
            generated_wave_gen = self.vocos.decode(generated_mel_spec_gen.cpu())
            generated_wave_gen = generated_wave_gen.squeeze().cpu().numpy()
            file_wav_gen = os.path.join(self.file_path_samples, f"step_{global_step}_gen.wav")
            sf.write(file_wav_gen, generated_wave_gen, target_sample_rate)

            if self.logger == "tensorboard":
                self.writer.add_audio("Audio/original", generated_wave_org, global_step, sample_rate=target_sample_rate)

                self.writer.add_audio("Audio/generate", generated_wave_gen, global_step, sample_rate=target_sample_rate)

            mel_org = mel_org
            mel_min, mel_max = mel_org.min(), mel_org.max()
            mel_norm = (mel_org - mel_min) / (mel_max - mel_min + 1e-8)
            mel_colored = plt.get_cmap("viridis")(mel_norm.detach().cpu().numpy())[:, :, :3]
            mel_colored = np.transpose(mel_colored, (2, 0, 1))

            if self.logger == "tensorboard":
                self.writer.add_image("Mel/oginal", mel_colored, global_step, dataformats="CHW")

            mel_colored_hwc = np.transpose(mel_colored, (1, 2, 0))
            file_gen_org = os.path.join(self.file_path_samples, f"step_{global_step}_org.png")
            plt.imsave(file_gen_org, mel_colored_hwc)

            mel_gen = generated_mel_spec_gen[0]
            mel_min, mel_max = mel_gen.min(), mel_gen.max()
            mel_norm = (mel_gen - mel_min) / (mel_max - mel_min + 1e-8)
            mel_colored = plt.get_cmap("viridis")(mel_norm.detach().cpu().numpy())[:, :, :3]
            mel_colored = np.transpose(mel_colored, (2, 0, 1))

            if self.logger == "tensorboard":
                self.writer.add_image("Mel/generate", mel_colored, global_step, dataformats="CHW")

            mel_colored_hwc = np.transpose(mel_colored, (1, 2, 0))
            file_gen_gen = os.path.join(self.file_path_samples, f"step_{global_step}_gen.png")
            plt.imsave(file_gen_gen, mel_colored_hwc)

        except Exception as e:
            print("An error occurred:", e)

    def train(self, train_dataset: Dataset, num_workers=16, resumable_with_seed: int = None):
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
                persistent_workers=True,
                batch_size=self.batch_size,
                shuffle=True,
                generator=generator,
            )
        elif self.batch_size_type == "frame":
            self.accelerator.even_batches = False
            sampler = SequentialSampler(train_dataset)
            batch_sampler = DynamicBatchSampler(
                sampler, self.batch_size, max_samples=self.max_samples, random_seed=resumable_with_seed, drop_last=False
            )
            train_dataloader = DataLoader(
                train_dataset,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=True,
                batch_sampler=batch_sampler,
            )
        else:
            raise ValueError(f"batch_size_type must be either 'sample' or 'frame', but received {self.batch_size_type}")

        #  accelerator.prepare() dispatches batches to devices;
        #  which means the length of dataloader calculated before, should consider the number of devices
        warmup_steps = (
            self.num_warmup_updates * self.accelerator.num_processes
        )  # consider a fixed warmup steps while using accelerate multi-gpu ddp
        # otherwise by default with split_batches=False, warmup steps change with num_processes
        total_steps = len(train_dataloader) * self.epochs / self.grad_accumulation_steps
        decay_steps = total_steps - warmup_steps
        warmup_scheduler = LinearLR(self.optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps)
        decay_scheduler = LinearLR(self.optimizer, start_factor=1.0, end_factor=1e-8, total_iters=decay_steps)
        self.scheduler = SequentialLR(
            self.optimizer, schedulers=[warmup_scheduler, decay_scheduler], milestones=[warmup_steps]
        )
        train_dataloader, self.scheduler = self.accelerator.prepare(
            train_dataloader, self.scheduler
        )  # actual steps = 1 gpu steps / gpus
        start_step = self.load_checkpoint()
        global_step = start_step

        if exists(resumable_with_seed):
            orig_epoch_step = len(train_dataloader)
            skipped_epoch = int(start_step // orig_epoch_step)
            skipped_batch = start_step % orig_epoch_step
            skipped_dataloader = self.accelerator.skip_first_batches(train_dataloader, num_batches=skipped_batch)
        else:
            skipped_epoch = 0

        for epoch in range(skipped_epoch, self.epochs):
            self.model.train()
            if exists(resumable_with_seed) and epoch == skipped_epoch:
                progress_bar = tqdm(
                    skipped_dataloader,
                    desc=f"Epoch {epoch+1}/{self.epochs}",
                    unit="step",
                    disable=not self.accelerator.is_local_main_process,
                    initial=skipped_batch,
                    total=orig_epoch_step,
                )
            else:
                progress_bar = tqdm(
                    train_dataloader,
                    desc=f"Epoch {epoch+1}/{self.epochs}",
                    unit="step",
                    disable=not self.accelerator.is_local_main_process,
                )

            for batch in progress_bar:
                with self.accelerator.accumulate(self.model):
                    text_inputs = batch["text"]
                    mel_spec = batch["mel"].permute(0, 2, 1)
                    mel_lengths = batch["mel_lengths"]

                    # TODO. add duration predictor training
                    if self.duration_predictor is not None and self.accelerator.is_local_main_process:
                        dur_loss = self.duration_predictor(mel_spec, lens=batch.get("durations"))
                        self.accelerator.log({"duration loss": dur_loss.item()}, step=global_step)

                    loss, cond, pred = self.model(
                        mel_spec, text=text_inputs, lens=mel_lengths, noise_scheduler=self.noise_scheduler
                    )

                    # save 4 audio per save step
                    if (
                        self.accelerator.is_local_main_process
                        and self.export_samples
                        and global_step % (int(self.save_per_updates * 0.25) * self.grad_accumulation_steps) == 0
                    ):
                        self.export_add_log(global_step, batch["mel"][0], text_inputs)

                    self.accelerator.backward(loss)

                    if self.max_grad_norm > 0 and self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                if self.is_main:
                    self.ema_model.update()

                global_step += 1

                if self.accelerator.is_local_main_process:
                    self.log({"loss": loss.item(), "lr": self.scheduler.get_last_lr()[0]}, step=global_step)

                progress_bar.set_postfix(step=str(global_step), loss=loss.item())

                if global_step % (self.save_per_updates * self.grad_accumulation_steps) == 0:
                    self.save_checkpoint(global_step)

                if global_step % self.last_per_steps == 0:
                    self.save_checkpoint(global_step, last=True)

        self.save_checkpoint(global_step, last=True)

        self.accelerator.end_training()
