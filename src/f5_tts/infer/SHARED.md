<!-- omit in toc -->
# Shared Model Cards

<!-- omit in toc -->
### **Prerequisites of using**
- This document is serving as a quick lookup table for the community training/finetuning result, with various language support.
- The models in this repository are open source and are based on voluntary contributions from contributors.
- The use of models must be conditioned on respect for the respective creators. The convenience brought comes from their efforts.

<!-- omit in toc -->
### **Welcome to share here**
- Have a pretrained/finetuned result: model checkpoint (pruned best to facilitate inference, i.e. leave only `ema_model_state_dict`) and corresponding vocab file (for tokenization).
- Host a public [huggingface model repository](https://huggingface.co/new) and upload the model related files.
- Make a pull request adding a model card to the current page, i.e. `src\f5_tts\infer\SHARED.md`.

<!-- omit in toc -->
### Supported Languages
- [Multilingual](#multilingual)
    - [F5-TTS Base @ pretrain @ zh \& en](#f5-tts-base--pretrain--zh--en)
- [English](#english)
- [Finnish](#finnish)
    - [Finnish Common\_Voice Vox\_Populi @ finetune @ fi](#finnish-common_voice-vox_populi--finetune--fi)
- [French](#french)
    - [French LibriVox @ finetune @ fr](#french-librivox--finetune--fr)
- [Japanese](#japanese)
    - [F5-TTS Japanese @ pretrain/finetune @ ja](#f5-tts-japanese--pretrainfinetune--ja)
- [Mandarin](#mandarin)
- [Spanish](#spanish)
    - [F5-TTS Spanish @ pretrain/finetune @ es](#f5-tts-spanish--pretrainfinetune--es)


## Multilingual

#### F5-TTS Base @ pretrain @ zh & en
|Model|🤗Hugging Face|Data (Hours)|Model License|
|:---:|:------------:|:-----------:|:-------------:|
|F5-TTS Base|[ckpt & vocab](https://huggingface.co/SWivid/F5-TTS/tree/main/F5TTS_Base)|[Emilia 95K zh&en](https://huggingface.co/datasets/amphion/Emilia-Dataset/tree/fc71e07)|cc-by-nc-4.0|

```bash
MODEL_CKPT: hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.safetensors
VOCAB_FILE: hf://SWivid/F5-TTS/F5TTS_Base/vocab.txt
```

*Other infos, e.g. Author info, Github repo, Link to some sampled results, Usage instruction, Tutorial (Blog, Video, etc.) ...*


## English


## Finnish

#### Finnish Common_Voice Vox_Populi @ finetune @ fi
|Model|🤗Hugging Face|Data|Model License|
|:---:|:------------:|:-----------:|:-------------:|
|F5-TTS Finnish|[ckpt & vocab](https://huggingface.co/AsmoKoskinen/F5-TTS_Finnish_Model)|[Common Voice](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0), [Vox Populi](https://huggingface.co/datasets/facebook/voxpopuli)|cc-by-nc-4.0|

```bash
MODEL_CKPT: hf://AsmoKoskinen/F5-TTS_Finish_Model/model_common_voice_fi_vox_populi_fi_20241206.safetensors
VOCAB_FILE: hf://AsmoKoskinen/F5-TTS_Finish_Model/vocab.txt
```


## French

#### French LibriVox @ finetune @ fr
|Model|🤗Hugging Face|Data (Hours)|Model License|
|:---:|:------------:|:-----------:|:-------------:|
|F5-TTS French|[ckpt & vocab](https://huggingface.co/RASPIAUDIO/F5-French-MixedSpeakers-reduced)|[LibriVox](https://librivox.org/)|cc-by-nc-4.0|

```bash
MODEL_CKPT: hf://RASPIAUDIO/F5-French-MixedSpeakers-reduced/model_last_reduced.pt
VOCAB_FILE: hf://RASPIAUDIO/F5-French-MixedSpeakers-reduced/vocab.txt
```

- [Online Inference with Hugging Face Space](https://huggingface.co/spaces/RASPIAUDIO/f5-tts_french).
- [Tutorial video to train a new language model](https://www.youtube.com/watch?v=UO4usaOojys).
- [Discussion about this training can be found here](https://github.com/SWivid/F5-TTS/issues/434).


## Japanese

#### F5-TTS Japanese @ pretrain/finetune @ ja
|Model|🤗Hugging Face|Data (Hours)|Model License|
|:---:|:------------:|:-----------:|:-------------:|
|F5-TTS Japanese|[ckpt & vocab](https://huggingface.co/Jmica/F5TTS/tree/main/JA_8500000)|[Emilia 1.7k JA](https://huggingface.co/datasets/amphion/Emilia-Dataset/tree/fc71e07) & [Galgame Dataset 5.4k](https://huggingface.co/datasets/OOPPEENN/Galgame_Dataset)|cc-by-nc-4.0|

```bash
MODEL_CKPT: hf://Jmica/F5TTS/JA_8500000/model_8499660.pt
VOCAB_FILE: hf://Jmica/F5TTS/JA_8500000/vocab_updated.txt
```


## Mandarin


## Spanish

#### F5-TTS Spanish @ pretrain/finetune @ es
|Model|🤗Hugging Face|Data (Hours)|Model License|
|:---:|:------------:|:-----------:|:-------------:|
|F5-TTS Spanish|[ckpt & vocab](https://huggingface.co/jpgallegoar/F5-Spanish)|[Voxpopuli](https://huggingface.co/datasets/facebook/voxpopuli) & Crowdsourced & TEDx, 218 hours|cc0-1.0|

- @jpgallegoar [GitHub repo](https://github.com/jpgallegoar/Spanish-F5), Jupyter Notebook and Gradio usage for Spanish model.
