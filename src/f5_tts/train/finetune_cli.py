import os
import shutil
import hydra

from cached_path import cached_path
from f5_tts.model import CFM, UNetT, DiT, Trainer
from f5_tts.model.utils import get_tokenizer
from f5_tts.model.dataset import load_dataset
from importlib.resources import files


@hydra.main(config_path=os.path.join("..", "configs"), config_name=None)
def main(cfg):
    tokenizer = cfg.model.tokenizer
    mel_spec_type = cfg.model.mel_spec.mel_spec_type
    exp_name = f"finetune_{cfg.model.name}_{mel_spec_type}_{cfg.model.tokenizer}_{cfg.datasets.name}"

    # set text tokenizer
    if tokenizer != "custom":
        tokenizer_path = cfg.datasets.name
    else:
        tokenizer_path = cfg.model.tokenizer_path
    vocab_char_map, vocab_size = get_tokenizer(tokenizer_path, tokenizer)

    print("\nvocab : ", vocab_size)
    print("\nvocoder : ", mel_spec_type)

    # Model parameters based on experiment name
    if "F5TTS" in cfg.model.name:
        model_cls = DiT
        ckpt_path = cfg.ckpts.pretain_ckpt_path or str(cached_path("hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.pt"))
    elif "E2TTS" in cfg.model.name:
        model_cls = UNetT
        ckpt_path = cfg.ckpts.pretain_ckpt_path or str(cached_path("hf://SWivid/F5-TTS/E2TTS_Base/model_1200000.pt"))
    wandb_resume_id = None

    checkpoint_path = str(files("f5_tts").joinpath(f"../../{cfg.ckpts.save_dir}"))

    if not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path, exist_ok=True)

    file_checkpoint = os.path.join(checkpoint_path, os.path.basename(ckpt_path))
    if not os.path.isfile(file_checkpoint):
        shutil.copy2(ckpt_path, file_checkpoint)
        print("copy checkpoint for finetune")

    model = CFM(
        transformer=model_cls(**cfg.model.arch, text_num_embeds=vocab_size, mel_dim=cfg.model.mel_spec.n_mel_channels),
        mel_spec_kwargs=cfg.model.mel_spec,
        vocab_char_map=vocab_char_map,
    )

    trainer = Trainer(
        model,
        epochs=cfg.optim.epochs,
        learning_rate=cfg.optim.learning_rate,
        num_warmup_updates=cfg.optim.num_warmup_updates,
        save_per_updates=cfg.ckpts.save_per_updates,
        checkpoint_path=checkpoint_path,
        batch_size=cfg.datasets.batch_size_per_gpu,
        batch_size_type=cfg.datasets.batch_size_type,
        max_samples=cfg.datasets.max_samples,
        grad_accumulation_steps=cfg.optim.grad_accumulation_steps,
        max_grad_norm=cfg.optim.max_grad_norm,
        logger=cfg.ckpts.logger,
        wandb_project=cfg.datasets.name,
        wandb_run_name=exp_name,
        wandb_resume_id=wandb_resume_id,
        log_samples=True,
        last_per_steps=cfg.ckpts.last_per_steps,
        bnb_optimizer=cfg.optim.bnb_optimizer,
        mel_spec_type=mel_spec_type,
        is_local_vocoder=cfg.model.mel_spec.is_local_vocoder,
        local_vocoder_path=cfg.model.mel_spec.local_vocoder_path,
    )

    train_dataset = load_dataset(cfg.datasets.name, tokenizer, mel_spec_kwargs=cfg.model.mel_spec)

    trainer.train(
        train_dataset,
        num_workers=cfg.datasets.num_workers,
        resumable_with_seed=666,  # seed for shuffling dataset
    )


if __name__ == "__main__":
    main()
