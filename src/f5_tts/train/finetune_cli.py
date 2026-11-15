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
mel_spec_type = "vocos"  # 'vocos' or 'bigvgan'


# -------------------------- Argument Parsing --------------------------- #
def parse_args():
    parser = argparse.ArgumentParser(description="Train CFM Model")

    parser.add_argument(
        "--exp_name",
        type=str,
        default="F5TTS_v1_Base",
        choices=["F5TTS_v1_Base", "F5TTS_Base", "E2TTS_Base"],
        help="Experiment name",
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
    parser.add_argument("--num_warmup_updates", type=int, default=20000, help="Warmup updates")
    parser.add_argument("--save_per_updates", type=int, default=50000, help="Save checkpoint every N updates")
    parser.add_argument(
        "--keep_last_n_checkpoints",
        type=int,
        default=-1,
        help="-1 to keep all, 0 to not save intermediate, > 0 to keep last N checkpoints",
    )
    parser.add_argument("--last_per_updates", type=int, default=5000, help="Save last checkpoint every N updates")
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
    parser.add_argument("--logger", type=str, default=None, choices=[None, "wandb", "tensorboard"], help="logger")
    parser.add_argument(
        "--bnb_optimizer",
        action="store_true",
        help="Use 8-bit Adam optimizer from bitsandbytes",
    )
    parser.add_argument(
        "--peft_method",
        type=str,
        default="none",
        choices=["none", "lora", "loha"],
        help="Parameter-efficient finetune adapter (frozen base, trainable adapter). 'none' = full finetune.",
    )
    parser.add_argument(
        "--peft_rank",
        type=int,
        default=8,
        help="Adapter rank r. Sensible: LoRA r=16-32, LoHa r=4-8 (LyCORIS rule r<=sqrt(dim)).",
    )
    parser.add_argument(
        "--peft_alpha",
        type=int,
        default=8,
        help="Adapter scaling alpha. Common choice: alpha = r (LoRA) or alpha = r (LoHa).",
    )
    parser.add_argument(
        "--peft_target_modules",
        type=str,
        default=None,
        help=(
            "Comma-separated module name suffixes to adapt. "
            "Default targets DiT/UNetT attention+FFN linears: 'to_q,to_k,to_v,to_out.0,ff.ff.0.0,ff.ff.2'."
        ),
    )

    return parser.parse_args()


# -------------------------- Training Settings -------------------------- #


def main():
    args = parse_args()

    checkpoint_path = str(files("f5_tts").joinpath(f"../../ckpts/{args.dataset_name}"))

    # Model parameters based on experiment name

    if args.exp_name == "F5TTS_v1_Base":
        wandb_resume_id = None
        model_cls = DiT
        model_cfg = dict(
            dim=1024,
            depth=22,
            heads=16,
            ff_mult=2,
            text_dim=512,
            conv_layers=4,
        )
        if args.finetune:
            if args.pretrain is None:
                ckpt_path = str(cached_path("hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors"))
            else:
                ckpt_path = args.pretrain

    elif args.exp_name == "F5TTS_Base":
        wandb_resume_id = None
        model_cls = DiT
        model_cfg = dict(
            dim=1024,
            depth=22,
            heads=16,
            ff_mult=2,
            text_dim=512,
            text_mask_padding=False,
            conv_layers=4,
            pe_attn_head=1,
        )
        if args.finetune:
            if args.pretrain is None:
                ckpt_path = str(cached_path("hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.pt"))
            else:
                ckpt_path = args.pretrain

    elif args.exp_name == "E2TTS_Base":
        wandb_resume_id = None
        model_cls = UNetT
        model_cfg = dict(
            dim=1024,
            depth=24,
            heads=16,
            ff_mult=4,
            text_mask_padding=False,
            pe_attn_head=1,
        )
        if args.finetune:
            if args.pretrain is None:
                ckpt_path = str(cached_path("hf://SWivid/E2-TTS/E2TTS_Base/model_1200000.pt"))
            else:
                ckpt_path = args.pretrain

    # PEFT runs need the base loaded INTO the bare model before adapter wrap (handled below).
    # Skip the pretrained-copy mechanic: load_checkpoint would otherwise try to load the
    # un-prefixed pretrained state into the PEFT-wrapped (prefixed) model.
    if args.finetune and args.peft_method == "none":
        if not os.path.isdir(checkpoint_path):
            os.makedirs(checkpoint_path, exist_ok=True)

        file_checkpoint = os.path.basename(ckpt_path)
        if not file_checkpoint.startswith("pretrained_"):  # Change: Add 'pretrained_' prefix to copied model
            file_checkpoint = "pretrained_" + file_checkpoint
        file_checkpoint = os.path.join(checkpoint_path, file_checkpoint)
        if not os.path.isfile(file_checkpoint):
            shutil.copy2(ckpt_path, file_checkpoint)
            print("copy checkpoint for finetune")

    # Use the tokenizer and tokenizer_path provided in the command line arguments

    tokenizer = args.tokenizer
    if tokenizer == "custom":
        if not args.tokenizer_path:
            raise ValueError("Custom tokenizer selected, but no tokenizer_path provided.")
        tokenizer_path = args.tokenizer_path
    else:
        tokenizer_path = args.dataset_name

    vocab_char_map, vocab_size = get_tokenizer(tokenizer_path, tokenizer)

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

    # --- PEFT: build adapter config + pre-load base weights ---
    peft_config = None
    if args.peft_method != "none":
        if args.peft_target_modules:
            targets = [s.strip() for s in args.peft_target_modules.split(",")]
        else:
            # DiT/UNetT share attention (to_q/k/v/out.0) and FFN (ff.ff.0.0 / ff.ff.2) module naming.
            targets = ["to_q", "to_k", "to_v", "to_out.0", "ff.ff.0.0", "ff.ff.2"]
        # Always exclude AdaLN-Zero modulation + final zero-init linears — adapting them breaks
        # F5-TTS init contract (NaN within first steps).
        excludes = [
            "attn_norm",
            "ff_norm",
            "norm_out",
            "proj_out",
            "time_embed",
            "text_embed",
            "input_embed",
            "long_skip_connection",
        ]
        if args.peft_method == "lora":
            from peft import LoraConfig

            peft_config = LoraConfig(
                r=args.peft_rank,
                lora_alpha=args.peft_alpha,
                target_modules=targets,
                exclude_modules=excludes,
                lora_dropout=0.0,
                bias="none",
            )
        elif args.peft_method == "loha":
            from peft import LoHaConfig

            peft_config = LoHaConfig(
                r=args.peft_rank,
                alpha=args.peft_alpha,
                target_modules=targets,
                exclude_modules=excludes,
                rank_dropout=0.0,
                module_dropout=0.0,
                use_effective_conv2d=False,
            )

        if args.finetune:
            import torch
            from safetensors.torch import load_file as _load_safetensors

            if ckpt_path.endswith(".safetensors"):
                sd = _load_safetensors(ckpt_path)
            else:
                sd_pt = torch.load(ckpt_path, weights_only=True, map_location="cpu")
                if "ema_model_state_dict" in sd_pt:
                    sd = {
                        k.replace("ema_model.", ""): v
                        for k, v in sd_pt["ema_model_state_dict"].items()
                        if k not in ("initted", "update", "step")
                    }
                elif "model_state_dict" in sd_pt:
                    sd = sd_pt["model_state_dict"]
                else:
                    sd = sd_pt
            for k in ("mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window"):
                sd.pop(k, None)
            missing, unexpected = model.load_state_dict(sd, strict=False)
            print(f"PEFT pretrain loaded from {ckpt_path}: missing={len(missing)} unexpected={len(unexpected)}")

    trainer = Trainer(
        model,
        args.epochs,
        args.learning_rate,
        num_warmup_updates=args.num_warmup_updates,
        save_per_updates=args.save_per_updates,
        keep_last_n_checkpoints=args.keep_last_n_checkpoints,
        checkpoint_path=checkpoint_path,
        batch_size_per_gpu=args.batch_size_per_gpu,
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
        peft_config=peft_config,
    )

    train_dataset = load_dataset(args.dataset_name, tokenizer, mel_spec_kwargs=mel_spec_kwargs)

    trainer.train(
        train_dataset,
        resumable_with_seed=666,  # seed for shuffling dataset
    )


if __name__ == "__main__":
    main()
