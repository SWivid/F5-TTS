import argparse
import os
import shutil

from cached_path import cached_path
from f5_tts.model import CFM, UNetT, DiT, Trainer
from f5_tts.model.utils import get_tokenizer
from f5_tts.model.dataset import load_dataset
from importlib.resources import files

import torch._dynamo
torch._dynamo.config.capture_scalar_outputs = True

"""
accelerate launch --mixed_precision=fp16 /home/prasais/projects/xttsv2/F5-TTS/src/f5_tts/train/finetune_cli.py --exp_name F5
TTS_Small --learning_rate 1e-07 --batch_size_per_gpu 12000 --batch_size_type frame --max_samples 64 --grad_accumulation_steps 1 --max_grad_norm 1 --epochs 150
000 --num_warmup_updates 814 --save_per_updates 1628 --keep_last_n_checkpoints 5 --last_per_updates 2000 --dataset_name indic_r --tokenizer char --log_samples
 --logger tensorboard 
 
"""
# -------------------------- Dataset Settings --------------------------- #
target_sample_rate = 24000
n_mel_channels = 100
hop_length = 256
win_length = 1024
n_fft = 1024
mel_spec_type = "vocos"  # 'vocos' or 'bigvgan'


# -------------------------- Argument Parsing --------------------------- #
def parse_args():
    # batch_size_per_gpu = 1000 settting for gpu 8GB
    # batch_size_per_gpu = 1600 settting for gpu 12GB
    # batch_size_per_gpu = 2000 settting for gpu 16GB
    # batch_size_per_gpu = 3200 settting for gpu 24GB

    # num_warmup_updates = 300 for 5000 sample about 10 hours

    # change save_per_updates , last_per_updates change this value what you need  ,

    parser = argparse.ArgumentParser(description="Train CFM Model")

    parser.add_argument(
        "--exp_name", type=str, default="F5TTS_Base", choices=["F5TTS_Base", "F5TTS_Small", "E2TTS_Base"], help="Experiment name"
    )
    parser.add_argument("--dataset_name", type=str, default="Emilia_ZH_EN", help="Name of the dataset to use")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for training")
    parser.add_argument("--batch_size_per_gpu", type=int, default=3200, help="Batch size per GPU")
    parser.add_argument(
        "--batch_size_type", type=str, default="frame", choices=["frame", "sample"], help="Batch size type"
    )
    parser.add_argument("--max_samples", type=int, default=64, help="Max sequences per batch")
    parser.add_argument("--grad_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--num_warmup_updates", type=int, default=300, help="Warmup updates")
    parser.add_argument("--save_per_updates", type=int, default=10000, help="Save checkpoint every X updates")
    parser.add_argument(
        "--keep_last_n_checkpoints",
        type=int,
        default=-1,
        help="-1 to keep all, 0 to not save intermediate, > 0 to keep last N checkpoints",
    )
    parser.add_argument("--last_per_updates", type=int, default=50000, help="Save last checkpoint every X updates")
    parser.add_argument("--finetune", action="store_true", help="Use Finetune")
    parser.add_argument("--pretrain", type=str, default=None, help="the path to the checkpoint")
    parser.add_argument(
        "--tokenizer", type=str, default="pinyin", choices=["pinyin", "char", "custom"], help="Tokenizer type"
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Path to custom tokenizer vocab file (only used if tokenizer = 'custom')",
    )
    parser.add_argument(
        "--log_samples",
        action="store_true",
        help="Log inferenced samples per ckpt save updates",
    )
    parser.add_argument("--logger", type=str, default=None, choices=["wandb", "tensorboard"], help="logger")
    parser.add_argument(
        "--bnb_optimizer",
        action="store_true",
        help="Use 8-bit Adam optimizer from bitsandbytes",
    )

    return parser.parse_args()


# -------------------------- Training Settings -------------------------- #


def main():
    args = parse_args()

    checkpoint_path = str(files("f5_tts").joinpath(f"../../ckpts/{args.dataset_name}"))


    wandb_resume_id = None
    model_cls = DiT
    model_cfg = dict(dim=768, depth=18, heads=12, ff_mult=2, text_dim=512, conv_layers=4)
        # if args.pretrain is None:
    ckpt_path = str("/home/prasais/projects/xttsv2/F5-TTS/ckpts/indic_r/pretrained_model_2500000.pt")
    file_checkpoint = os.path.basename(ckpt_path)

    # Use the tokenizer and tokenizer_path provided in the command line arguments
    # tokenizer = args.tokenizer

    tokenizer = "char"
    tokenizer_path = "indic_r"
    # tokenizer_path = "/home/prasais/projects/xttsv2/F5-TTS/data/indic_r_char"
    vocab_char_map, vocab_size = get_tokenizer(tokenizer_path, tokenizer)
    # vocab_char_map, vocab_size = get_tokenizer("indic_r", "char")


    print("\nvocab : ", vocab_size)
    print("\nvocoder : ", mel_spec_type)

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
        args.epochs,
        args.learning_rate,
        num_warmup_updates=args.num_warmup_updates,
        save_per_updates=args.save_per_updates,
        keep_last_n_checkpoints=args.keep_last_n_checkpoints,
        checkpoint_path=checkpoint_path,
        batch_size=args.batch_size_per_gpu,
        batch_size_type=args.batch_size_type,
        max_samples=args.max_samples,
        grad_accumulation_steps=args.grad_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        logger=args.logger,
        wandb_project=args.dataset_name,
        wandb_run_name=args.exp_name,
        wandb_resume_id=wandb_resume_id,
        log_samples=args.log_samples,
        last_per_updates=args.last_per_updates,
        bnb_optimizer=args.bnb_optimizer,
    )

    train_dataset = load_dataset(args.dataset_name, tokenizer, mel_spec_kwargs=mel_spec_kwargs)

    trainer.train(
        train_dataset,
        resumable_with_seed=666,  # seed for shuffling dataset
    )


if __name__ == "__main__":
    main()
