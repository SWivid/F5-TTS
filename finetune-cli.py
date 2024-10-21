import argparse
from model import CFM, UNetT, DiT, Trainer
from model.utils import get_tokenizer
from model.dataset import load_dataset
from cached_path import cached_path
import shutil
import os

# -------------------------- Dataset Settings --------------------------- #
target_sample_rate = 24000
n_mel_channels = 100
hop_length = 256


# -------------------------- Argument Parsing --------------------------- #
def parse_args():
    parser = argparse.ArgumentParser(description="Train CFM Model")

    parser.add_argument(
        "--exp_name", type=str, default="F5TTS_Base", choices=["F5TTS_Base", "E2TTS_Base"], help="Experiment name"
    )
    parser.add_argument("--dataset_name", type=str, default="Emilia_ZH_EN", help="Name of the dataset to use")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for training")
    parser.add_argument("--batch_size_per_gpu", type=int, default=256, help="Batch size per GPU")
    parser.add_argument(
        "--batch_size_type", type=str, default="frame", choices=["frame", "sample"], help="Batch size type"
    )
    parser.add_argument("--max_samples", type=int, default=16, help="Max sequences per batch")
    parser.add_argument("--grad_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--num_warmup_updates", type=int, default=5, help="Warmup steps")
    parser.add_argument("--save_per_updates", type=int, default=10, help="Save checkpoint every X steps")
    parser.add_argument("--last_per_steps", type=int, default=10, help="Save last checkpoint every X steps")
    parser.add_argument("--finetune", type=bool, default=True, help="Use Finetune")

    parser.add_argument(
        "--tokenizer", type=str, default="pinyin", choices=["pinyin", "char", "custom"], help="Tokenizer type"
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Path to custom tokenizer vocab file (only used if tokenizer = 'custom')",
    )

    return parser.parse_args()


# -------------------------- Training Settings -------------------------- #


def main():
    args = parse_args()

    # Model parameters based on experiment name
    if args.exp_name == "F5TTS_Base":
        wandb_resume_id = None
        model_cls = DiT
        model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
        if args.finetune:
            ckpt_path = str(cached_path("hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.pt"))
    elif args.exp_name == "E2TTS_Base":
        wandb_resume_id = None
        model_cls = UNetT
        model_cfg = dict(dim=1024, depth=24, heads=16, ff_mult=4)
        if args.finetune:
            ckpt_path = str(cached_path("hf://SWivid/E2-TTS/E2TTS_Base/model_1200000.pt"))

    if args.finetune:
        path_ckpt = os.path.join("ckpts", args.dataset_name)
        if not os.path.isdir(path_ckpt):
            os.makedirs(path_ckpt, exist_ok=True)
            shutil.copy2(ckpt_path, os.path.join(path_ckpt, os.path.basename(ckpt_path)))

    checkpoint_path = os.path.join("ckpts", args.dataset_name)

    # Use the tokenizer and tokenizer_path provided in the command line arguments
    tokenizer = args.tokenizer
    if tokenizer == "custom":
        if not args.tokenizer_path:
            raise ValueError("Custom tokenizer selected, but no tokenizer_path provided.")
        tokenizer_path = args.tokenizer_path
    else:
        tokenizer_path = args.dataset_name

    vocab_char_map, vocab_size = get_tokenizer(tokenizer_path, tokenizer)

    mel_spec_kwargs = dict(
        target_sample_rate=target_sample_rate,
        n_mel_channels=n_mel_channels,
        hop_length=hop_length,
    )

    e2tts = CFM(
        transformer=model_cls(**model_cfg, text_num_embeds=vocab_size, mel_dim=n_mel_channels),
        mel_spec_kwargs=mel_spec_kwargs,
        vocab_char_map=vocab_char_map,
    )

    trainer = Trainer(
        e2tts,
        args.epochs,
        args.learning_rate,
        num_warmup_updates=args.num_warmup_updates,
        save_per_updates=args.save_per_updates,
        checkpoint_path=checkpoint_path,
        batch_size=args.batch_size_per_gpu,
        batch_size_type=args.batch_size_type,
        max_samples=args.max_samples,
        grad_accumulation_steps=args.grad_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        wandb_project="CFM-TTS",
        wandb_run_name=args.exp_name,
        wandb_resume_id=wandb_resume_id,
        last_per_steps=args.last_per_steps,
    )

    train_dataset = load_dataset(args.dataset_name, tokenizer, mel_spec_kwargs=mel_spec_kwargs)
    trainer.train(
        train_dataset,
        resumable_with_seed=666,  # seed for shuffling dataset
    )


if __name__ == "__main__":
    main()
