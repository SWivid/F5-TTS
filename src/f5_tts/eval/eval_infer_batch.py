import os
import sys


sys.path.append(os.getcwd())

import argparse
import time
from importlib.resources import files

import torch
import torchaudio
from accelerate import Accelerator
from hydra.utils import get_class
from omegaconf import OmegaConf
from tqdm import tqdm

from f5_tts.eval.utils_eval import (
    get_inference_prompt,
    get_librispeech_test_clean_metainfo,
    get_seedtts_testset_metainfo,
)
from f5_tts.infer.utils_infer import load_checkpoint, load_vocoder
from f5_tts.model import CFM
from f5_tts.model.utils import get_tokenizer


accelerator = Accelerator()
device = f"cuda:{accelerator.process_index}"


use_ema = True
target_rms = 0.1


rel_path = str(files("f5_tts").joinpath("../../"))


def main():
    parser = argparse.ArgumentParser(description="batch inference")

    parser.add_argument("-s", "--seed", default=None, type=int)
    parser.add_argument("-n", "--expname", required=True)
    parser.add_argument("-c", "--ckptstep", default=1250000, type=int)

    parser.add_argument("-nfe", "--nfestep", default=32, type=int)
    parser.add_argument("-o", "--odemethod", default="euler")
    parser.add_argument("-ss", "--swaysampling", default=-1, type=float)

    parser.add_argument("-t", "--testset", required=True)

    args = parser.parse_args()

    seed = args.seed
    exp_name = args.expname
    ckpt_step = args.ckptstep

    nfe_step = args.nfestep
    ode_method = args.odemethod
    sway_sampling_coef = args.swaysampling

    testset = args.testset

    infer_batch_size = 1  # max frames. 1 for ddp single inference (recommended)
    cfg_strength = 2.0
    speed = 1.0
    use_truth_duration = False
    no_ref_audio = False

    model_cfg = OmegaConf.load(str(files("f5_tts").joinpath(f"configs/{exp_name}.yaml")))
    model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
    model_arc = model_cfg.model.arch

    dataset_name = model_cfg.datasets.name
    tokenizer = model_cfg.model.tokenizer

    mel_spec_type = model_cfg.model.mel_spec.mel_spec_type
    target_sample_rate = model_cfg.model.mel_spec.target_sample_rate
    n_mel_channels = model_cfg.model.mel_spec.n_mel_channels
    hop_length = model_cfg.model.mel_spec.hop_length
    win_length = model_cfg.model.mel_spec.win_length
    n_fft = model_cfg.model.mel_spec.n_fft

    if testset == "ls_pc_test_clean":
        metalst = rel_path + "/data/librispeech_pc_test_clean_cross_sentence.lst"
        librispeech_test_clean_path = "<SOME_PATH>/LibriSpeech/test-clean"  # test-clean path
        metainfo = get_librispeech_test_clean_metainfo(metalst, librispeech_test_clean_path)

    elif testset == "seedtts_test_zh":
        metalst = rel_path + "/data/seedtts_testset/zh/meta.lst"
        metainfo = get_seedtts_testset_metainfo(metalst)

    elif testset == "seedtts_test_en":
        metalst = rel_path + "/data/seedtts_testset/en/meta.lst"
        metainfo = get_seedtts_testset_metainfo(metalst)

    # path to save genereted wavs
    output_dir = (
        f"{rel_path}/"
        f"results/{exp_name}_{ckpt_step}/{testset}/"
        f"seed{seed}_{ode_method}_nfe{nfe_step}_{mel_spec_type}"
        f"{f'_ss{sway_sampling_coef}' if sway_sampling_coef else ''}"
        f"_cfg{cfg_strength}_speed{speed}"
        f"{'_gt-dur' if use_truth_duration else ''}"
        f"{'_no-ref-audio' if no_ref_audio else ''}"
    )

    # -------------------------------------------------#

    prompts_all = get_inference_prompt(
        metainfo,
        speed=speed,
        tokenizer=tokenizer,
        target_sample_rate=target_sample_rate,
        n_mel_channels=n_mel_channels,
        hop_length=hop_length,
        mel_spec_type=mel_spec_type,
        target_rms=target_rms,
        use_truth_duration=use_truth_duration,
        infer_batch_size=infer_batch_size,
    )

    # Vocoder model
    local = False
    if mel_spec_type == "vocos":
        vocoder_local_path = "../checkpoints/charactr/vocos-mel-24khz"
    elif mel_spec_type == "bigvgan":
        vocoder_local_path = "../checkpoints/bigvgan_v2_24khz_100band_256x"
    vocoder = load_vocoder(vocoder_name=mel_spec_type, is_local=local, local_path=vocoder_local_path)

    # Tokenizer
    vocab_char_map, vocab_size = get_tokenizer(dataset_name, tokenizer)

    # Model
    model = CFM(
        transformer=model_cls(**model_arc, text_num_embeds=vocab_size, mel_dim=n_mel_channels),
        mel_spec_kwargs=dict(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mel_channels=n_mel_channels,
            target_sample_rate=target_sample_rate,
            mel_spec_type=mel_spec_type,
        ),
        odeint_kwargs=dict(
            method=ode_method,
        ),
        vocab_char_map=vocab_char_map,
    ).to(device)

    ckpt_path = rel_path + f"/ckpts/{exp_name}/model_{ckpt_step}.pt"
    if not os.path.exists(ckpt_path):
        print("Loading from self-organized training checkpoints rather than released pretrained.")
        ckpt_path = rel_path + f"/{model_cfg.ckpts.save_dir}/model_{ckpt_step}.pt"
    dtype = torch.float32 if mel_spec_type == "bigvgan" else None
    model = load_checkpoint(model, ckpt_path, device, dtype=dtype, use_ema=use_ema)

    if not os.path.exists(output_dir) and accelerator.is_main_process:
        os.makedirs(output_dir)

    # start batch inference
    accelerator.wait_for_everyone()
    start = time.time()

    with accelerator.split_between_processes(prompts_all) as prompts:
        for prompt in tqdm(prompts, disable=not accelerator.is_local_main_process):
            utts, ref_rms_list, ref_mels, ref_mel_lens, total_mel_lens, final_text_list = prompt
            ref_mels = ref_mels.to(device)
            ref_mel_lens = torch.tensor(ref_mel_lens, dtype=torch.long).to(device)
            total_mel_lens = torch.tensor(total_mel_lens, dtype=torch.long).to(device)

            # Inference
            with torch.inference_mode():
                generated, _ = model.sample(
                    cond=ref_mels,
                    text=final_text_list,
                    duration=total_mel_lens,
                    lens=ref_mel_lens,
                    steps=nfe_step,
                    cfg_strength=cfg_strength,
                    sway_sampling_coef=sway_sampling_coef,
                    no_ref_audio=no_ref_audio,
                    seed=seed,
                )
                # Final result
                for i, gen in enumerate(generated):
                    gen = gen[ref_mel_lens[i] : total_mel_lens[i], :].unsqueeze(0)
                    gen_mel_spec = gen.permute(0, 2, 1).to(torch.float32)
                    if mel_spec_type == "vocos":
                        generated_wave = vocoder.decode(gen_mel_spec).cpu()
                    elif mel_spec_type == "bigvgan":
                        generated_wave = vocoder(gen_mel_spec).squeeze(0).cpu()

                    if ref_rms_list[i] < target_rms:
                        generated_wave = generated_wave * ref_rms_list[i] / target_rms
                    torchaudio.save(f"{output_dir}/{utts[i]}.wav", generated_wave, target_sample_rate)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        timediff = time.time() - start
        print(f"Done batch inference in {timediff / 60:.2f} minutes.")


if __name__ == "__main__":
    main()
