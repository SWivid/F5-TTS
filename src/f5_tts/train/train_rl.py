import os
from importlib.resources import files

import hydra
from omegaconf import OmegaConf

from f5_tts.model import CFM
from f5_tts.model.dataset import load_dataset
from f5_tts.model.utils import get_tokenizer
from f5_tts.rewards import RewardCombiner
from f5_tts.rl.trainer_grpo import GRPOTrainer


os.chdir(str(files("f5_tts").joinpath("../..")))  # change working directory to root of project (local editable)


@hydra.main(version_base="1.3", config_path=str(files("f5_tts").joinpath("configs")), config_name="F5TTS_RL")
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
    exp_name = f"{model_cfg.model.name}_{mel_spec_type}_{model_cfg.model.tokenizer}_{model_cfg.datasets.name}_rl"

    if tokenizer != "custom":
        tokenizer_path = model_cfg.datasets.name
    else:
        tokenizer_path = model_cfg.model.tokenizer_path
    vocab_char_map, vocab_size = get_tokenizer(tokenizer_path, tokenizer)

    model = CFM(
        transformer=model_cls(**model_arc, text_num_embeds=vocab_size, mel_dim=model_cfg.model.mel_spec.n_mel_channels),
        mel_spec_kwargs=model_cfg.model.mel_spec,
        vocab_char_map=vocab_char_map,
        objective=model_cfg.model.get("objective", "grpo"),
        output_dist=output_dist,
        sample_from_dist=model_cfg.model.get("sample_from_dist", False),
    )

    reward_combiner = RewardCombiner.from_config(model_cfg.rl.rewards)

    trainer = GRPOTrainer(
        model,
        reward_combiner=reward_combiner,
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
        last_per_updates=model_cfg.ckpts.last_per_updates,
        mel_spec_type=mel_spec_type,
        log_samples=model_cfg.ckpts.log_samples,
        repeat_count=model_cfg.rl.repeat_count,
        mini_repeat_count=model_cfg.rl.mini_repeat_count,
        prompt_frac_range=tuple(model_cfg.rl.prompt_frac_range),
        steps=model_cfg.rl.steps,
        steps_plus_one=model_cfg.rl.get("steps_plus_one", False),
        cfg_strength=model_cfg.rl.cfg_strength,
        sway_sampling_coef=model_cfg.rl.sway_sampling_coef,
        kl_weight=model_cfg.rl.kl_weight,
        ref_model_ckpt=model_cfg.rl.ref_model_ckpt,
        ref_model_use_ema=model_cfg.rl.ref_model_use_ema,
        allow_extra_keys=model_cfg.ckpts.get("allow_extra_keys", False),
        bnb_optimizer=model_cfg.optim.get("bnb_optimizer", False),
        prompt_length_mode=model_cfg.rl.get("prompt_length_mode", "min"),
    )

    train_dataset = load_dataset(model_cfg.datasets.name, tokenizer, mel_spec_kwargs=model_cfg.model.mel_spec)
    trainer.train(
        train_dataset,
        num_workers=model_cfg.datasets.num_workers,
        resumable_with_seed=666,
    )


if __name__ == "__main__":
    main()
