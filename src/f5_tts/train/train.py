# training script.

import os
from importlib.resources import files

import hydra
from omegaconf import OmegaConf

from f5_tts.model import CFM, Trainer
from f5_tts.model.dataset import load_dataset
from f5_tts.model.utils import get_tokenizer
from f5_tts.train.utils import apply_tf32, resolve_mixed_precision


os.chdir(str(files("f5_tts").joinpath("../..")))  # change working directory to root of project (local editable)


@hydra.main(version_base="1.3", config_path=str(files("f5_tts").joinpath("configs")), config_name=None)
def main(model_cfg):
    model_cls = hydra.utils.get_class(f"f5_tts.model.{model_cfg.model.backbone}")
    model_arc = OmegaConf.to_container(model_cfg.model.arch, resolve=True)
    output_dist = model_cfg.model.get("output_dist", "deterministic")
    if "output_dist" not in model_arc:
        model_arc["output_dist"] = output_dist
    if "use_rl_head" not in model_arc and "use_rl_head" in model_cfg.model:
        model_arc["use_rl_head"] = model_cfg.model.use_rl_head
    tokenizer = model_cfg.model.tokenizer
    mel_spec_type = model_cfg.model.mel_spec.mel_spec_type

    exp_name = f"{model_cfg.model.name}_{mel_spec_type}_{model_cfg.model.tokenizer}_{model_cfg.datasets.name}"
    wandb_resume_id = None

    # set text tokenizer
    if tokenizer != "custom":
        tokenizer_path = model_cfg.datasets.name
    else:
        tokenizer_path = model_cfg.model.tokenizer_path
    vocab_char_map, vocab_size = get_tokenizer(tokenizer_path, tokenizer)

    # set model
    model = CFM(
        transformer=model_cls(**model_arc, text_num_embeds=vocab_size, mel_dim=model_cfg.model.mel_spec.n_mel_channels),
        mel_spec_kwargs=model_cfg.model.mel_spec,
        vocab_char_map=vocab_char_map,
        objective=model_cfg.model.get("objective", "mse"),
        output_dist=output_dist,
        sample_from_dist=model_cfg.model.get("sample_from_dist", False),
    )

    accelerate_kwargs = OmegaConf.to_container(model_cfg.optim.get("accelerate_kwargs", {}), resolve=True) or {}
    mixed_precision = resolve_mixed_precision(model_cfg.optim.get("mixed_precision"))
    if mixed_precision is not None:
        accelerate_kwargs["mixed_precision"] = mixed_precision
    apply_tf32(bool(model_cfg.optim.get("tf32", False)))

    # init trainer
    trainer = Trainer(
        model,
        epochs=model_cfg.optim.epochs,
        learning_rate=model_cfg.optim.learning_rate,
        num_warmup_updates=model_cfg.optim.num_warmup_updates,
        save_per_updates=model_cfg.ckpts.save_per_updates,
        keep_last_n_checkpoints=model_cfg.ckpts.keep_last_n_checkpoints,
        checkpoint_path=str(files("f5_tts").joinpath(f"../../{model_cfg.ckpts.save_dir}")),
        batch_size_per_gpu=model_cfg.datasets.batch_size_per_gpu,
        batch_size_type=model_cfg.datasets.batch_size_type,
        max_samples=model_cfg.datasets.max_samples,
        grad_accumulation_steps=model_cfg.optim.grad_accumulation_steps,
        max_grad_norm=model_cfg.optim.max_grad_norm,
        logger=model_cfg.ckpts.logger,
        wandb_project="CFM-TTS",
        wandb_run_name=exp_name,
        wandb_resume_id=wandb_resume_id,
        last_per_updates=model_cfg.ckpts.last_per_updates,
        log_samples=model_cfg.ckpts.log_samples,
        bnb_optimizer=model_cfg.optim.bnb_optimizer,
        mel_spec_type=mel_spec_type,
        is_local_vocoder=model_cfg.model.vocoder.is_local,
        local_vocoder_path=model_cfg.model.vocoder.local_path,
        model_cfg_dict=OmegaConf.to_container(model_cfg, resolve=True),
        allow_extra_keys=model_cfg.ckpts.get("allow_extra_keys", False),
        accelerate_kwargs=accelerate_kwargs,
    )

    train_dataset = load_dataset(model_cfg.datasets.name, tokenizer, mel_spec_kwargs=model_cfg.model.mel_spec)
    trainer.train(
        train_dataset,
        num_workers=model_cfg.datasets.num_workers,
        resumable_with_seed=666,  # seed for shuffling dataset
    )


if __name__ == "__main__":
    main()
