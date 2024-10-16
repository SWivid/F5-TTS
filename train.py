import os
import argparse
import logging
from model import CFM, UNetT, DiT, Trainer
from model.utils import get_tokenizer
from model.dataset import load_dataset

# Set up logging
logging.basicConfig(level=logging.INFO)

# -------------------------- Dataset Settings --------------------------- #

target_sample_rate = 24000
n_mel_channels = 100
hop_length = 256

# -------------------------- Command-Line Argument Parsing -------------------------- #

def parse_args():
    parser = argparse.ArgumentParser(description="Train TTS Model")
    parser.add_argument("--exp_name", type=str, default="F5TTS_Base", help="Experiment name")
    parser.add_argument("--learning_rate", type=float, default=7.5e-5, help="Learning rate")
    parser.add_argument("--batch_size_per_gpu", type=int, default=38400, help="Batch size per GPU")
    parser.add_argument("--tokenizer", type=str, default="pinyin", choices=['pinyin', 'char', 'custom'], help="Tokenizer type")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to custom tokenizer if using 'custom'")
    parser.add_argument("--dataset_name", type=str, default="Emilia_ZH_EN", help="Dataset name")
    # Add other parameters as needed
    return parser.parse_args()

# -------------------------- Training Settings -------------------------- #

def main():
    args = parse_args()

    # Load tokenizer
    if args.tokenizer == "custom":
        if not os.path.exists(args.tokenizer_path):
            raise FileNotFoundError(f"Custom tokenizer path {args.tokenizer_path} does not exist.")
        tokenizer_path = args.tokenizer_path
    else:
        tokenizer_path = args.dataset_name
    
    try:
        vocab_char_map, vocab_size = get_tokenizer(tokenizer_path, args.tokenizer)
    except Exception as e:
        logging.error(f"Error loading tokenizer: {e}")
        return

    mel_spec_kwargs = dict(
        target_sample_rate=target_sample_rate,
        n_mel_channels=n_mel_channels,
        hop_length=hop_length,
    )

    # Model configuration based on experiment name
    if args.exp_name == "F5TTS_Base":
        model_cls = DiT
        model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
    elif args.exp_name == "E2TTS_Base":
        model_cls = UNetT
        model_cfg = dict(dim=1024, depth=24, heads=16, ff_mult=4)
    else:
        logging.error("Invalid experiment name specified.")
        return

    e2tts = CFM(
        transformer=model_cls(
            **model_cfg,
            text_num_embeds=vocab_size,
            mel_dim=n_mel_channels
        ),
        mel_spec_kwargs=mel_spec_kwargs,
        vocab_char_map=vocab_char_map,
    )

    trainer = Trainer(
        e2tts,
        epochs=11,
        learning_rate=args.learning_rate,
        num_warmup_updates=20000,
        save_per_updates=50000,
        checkpoint_path=f'ckpts/{args.exp_name}',
        batch_size=args.batch_size_per_gpu,
        batch_size_type="frame",
        max_samples=64,
        grad_accumulation_steps=1,
        max_grad_norm=1.0,
        wandb_project="CFM-TTS",
        wandb_run_name=args.exp_name,
        last_per_steps=5000,
    )

    # Load dataset
    try:
        train_dataset = load_dataset(args.dataset_name, args.tokenizer, mel_spec_kwargs=mel_spec_kwargs)
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        return

    logging.info("Starting training...")
    trainer.train(train_dataset, resumable_with_seed=666)  # seed for shuffling dataset
    logging.info("Training completed.")

if __name__ == '__main__':
    main()
