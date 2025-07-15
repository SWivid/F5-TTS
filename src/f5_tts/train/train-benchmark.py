from colorama import init as colorama_init
from colorama import Fore
from colorama import Style

import time
import argparse
import os
import shutil
from importlib.resources import files

from cached_path import cached_path

from f5_tts.model import CFM, DiT, Trainer, UNetT
from f5_tts.model.dataset import load_dataset
from f5_tts.model.utils import get_tokenizer


# -------------------------- Dataset Settings --------------------------- #
target_sample_rate = 24000
n_mel_channels = 100
hop_length = 256
win_length = 1024
n_fft = 1024
mel_spec_type = "vocos"
dataset_name = "train-benchmark"


def main():

    checkpoint_path = str(files("f5_tts").joinpath(f"../../ckpts/{dataset_name}"))

    wandb_resume_id = False
    model_cls = DiT
    model_cfg = dict(
        dim=1024,
        depth=22,
        heads=16,
        ff_mult=2,
        text_dim=512,
        conv_layers=4,
    )
    ckpt_path = str(cached_path("hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors"))

    if not os.path.isdir(checkpoint_path):
      os.makedirs(checkpoint_path, exist_ok=True)
    else:
      for file in os.listdir(checkpoint_path):
        if file.endswith(".pt"):
          os.remove(os.path.join(checkpoint_path, file))  
    
    file_checkpoint = os.path.basename(ckpt_path)
    if not file_checkpoint.startswith("pretrained_"):  # Change: Add 'pretrained_' prefix to copied model
        file_checkpoint = "pretrained_" + file_checkpoint
    file_checkpoint = os.path.join(checkpoint_path, file_checkpoint)
    if not os.path.isfile(file_checkpoint):
        shutil.copy2(ckpt_path, file_checkpoint)
        print("copy checkpoint for finetune")

    tokenizer = "pinyin"
    vocab_char_map, vocab_size = get_tokenizer(dataset_name, tokenizer)

    mel_spec_kwargs = dict(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mel_channels=n_mel_channels,
        target_sample_rate=target_sample_rate,
        mel_spec_type=mel_spec_type,
    )

    model = CFM(
        transformer=model_cls(**model_cfg, text_num_embeds=vocab_size, mel_dim=n_mel_channels),
        mel_spec_kwargs=mel_spec_kwargs,
        vocab_char_map=vocab_char_map,
    )

    trainer = Trainer(
        model,
        20,
        1e-05,
        num_warmup_updates=100,
        save_per_updates=500,
        keep_last_n_checkpoints=0,
        checkpoint_path=checkpoint_path,
        batch_size_per_gpu=3583,
        batch_size_type="frame",
        max_samples=5,
        grad_accumulation_steps=1,
        max_grad_norm=1,
        logger=None,
        wandb_project=dataset_name,
        wandb_run_name="F5TTS_v1_Base",
        wandb_resume_id=None,
        log_samples=False,
        last_per_updates=100,
        bnb_optimizer=False,
    )

    train_dataset = load_dataset(dataset_name, tokenizer, mel_spec_kwargs=mel_spec_kwargs)
    
    training_time = trainer.train(
        train_dataset,
        resumable_with_seed=666,  # seed for shuffling dataset
    )                
                                                                                         
    lastpt_file = os.path.join(checkpoint_path, "model_last.pt")
    if os.path.isfile(lastpt_file):
        os.remove(lastpt_file)
        
    print(f"\nProcessing time {training_time:.2f} sec")
    colorama_init()
    print(f"F5-TTS Performance score: {Fore.GREEN}{1e05 / training_time:.2f}{Style.RESET_ALL} F5PS")   


if __name__ == "__main__":
    main()