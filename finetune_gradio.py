import os
import sys

import tempfile
import random
from transformers import pipeline
import gradio as gr
import torch
import gc
import click
import torchaudio
from glob import glob
import librosa
import numpy as np
from scipy.io import wavfile
import shutil
import time

import json
from model.utils import convert_char_to_pinyin
import signal
import psutil
import platform
import subprocess
from datasets.arrow_writer import ArrowWriter
from datasets import Dataset as Dataset_
from api import F5TTS
from safetensors.torch import save_file

training_process = None
system = platform.system()
python_executable = sys.executable or "python"
tts_api = None
last_checkpoint = ""
last_device = ""

path_data = "data"

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

pipe = None


# Load metadata
def get_audio_duration(audio_path):
    """Calculate the duration of an audio file."""
    audio, sample_rate = torchaudio.load(audio_path)
    num_channels = audio.shape[0]
    return audio.shape[1] / (sample_rate * num_channels)


def clear_text(text):
    """Clean and prepare text by lowering the case and stripping whitespace."""
    return text.lower().strip()


def get_rms(
    y,
    frame_length=2048,
    hop_length=512,
    pad_mode="constant",
):  # https://github.com/RVC-Boss/GPT-SoVITS/blob/main/tools/slicer2.py
    padding = (int(frame_length // 2), int(frame_length // 2))
    y = np.pad(y, padding, mode=pad_mode)

    axis = -1
    # put our new within-frame axis at the end for  now
    out_strides = y.strides + tuple([y.strides[axis]])
    # Reduce the shape on the framing axis
    x_shape_trimmed = list(y.shape)
    x_shape_trimmed[axis] -= frame_length - 1
    out_shape = tuple(x_shape_trimmed) + tuple([frame_length])
    xw = np.lib.stride_tricks.as_strided(y, shape=out_shape, strides=out_strides)
    if axis < 0:
        target_axis = axis - 1
    else:
        target_axis = axis + 1
    xw = np.moveaxis(xw, -1, target_axis)
    # Downsample along the target axis
    slices = [slice(None)] * xw.ndim
    slices[axis] = slice(0, None, hop_length)
    x = xw[tuple(slices)]

    # Calculate power
    power = np.mean(np.abs(x) ** 2, axis=-2, keepdims=True)

    return np.sqrt(power)


class Slicer:  # https://github.com/RVC-Boss/GPT-SoVITS/blob/main/tools/slicer2.py
    def __init__(
        self,
        sr: int,
        threshold: float = -40.0,
        min_length: int = 2000,
        min_interval: int = 300,
        hop_size: int = 20,
        max_sil_kept: int = 2000,
    ):
        if not min_length >= min_interval >= hop_size:
            raise ValueError("The following condition must be satisfied: min_length >= min_interval >= hop_size")
        if not max_sil_kept >= hop_size:
            raise ValueError("The following condition must be satisfied: max_sil_kept >= hop_size")
        min_interval = sr * min_interval / 1000
        self.threshold = 10 ** (threshold / 20.0)
        self.hop_size = round(sr * hop_size / 1000)
        self.win_size = min(round(min_interval), 4 * self.hop_size)
        self.min_length = round(sr * min_length / 1000 / self.hop_size)
        self.min_interval = round(min_interval / self.hop_size)
        self.max_sil_kept = round(sr * max_sil_kept / 1000 / self.hop_size)

    def _apply_slice(self, waveform, begin, end):
        if len(waveform.shape) > 1:
            return waveform[:, begin * self.hop_size : min(waveform.shape[1], end * self.hop_size)]
        else:
            return waveform[begin * self.hop_size : min(waveform.shape[0], end * self.hop_size)]

    # @timeit
    def slice(self, waveform):
        if len(waveform.shape) > 1:
            samples = waveform.mean(axis=0)
        else:
            samples = waveform
        if samples.shape[0] <= self.min_length:
            return [waveform]
        rms_list = get_rms(y=samples, frame_length=self.win_size, hop_length=self.hop_size).squeeze(0)
        sil_tags = []
        silence_start = None
        clip_start = 0
        for i, rms in enumerate(rms_list):
            # Keep looping while frame is silent.
            if rms < self.threshold:
                # Record start of silent frames.
                if silence_start is None:
                    silence_start = i
                continue
            # Keep looping while frame is not silent and silence start has not been recorded.
            if silence_start is None:
                continue
            # Clear recorded silence start if interval is not enough or clip is too short
            is_leading_silence = silence_start == 0 and i > self.max_sil_kept
            need_slice_middle = i - silence_start >= self.min_interval and i - clip_start >= self.min_length
            if not is_leading_silence and not need_slice_middle:
                silence_start = None
                continue
            # Need slicing. Record the range of silent frames to be removed.
            if i - silence_start <= self.max_sil_kept:
                pos = rms_list[silence_start : i + 1].argmin() + silence_start
                if silence_start == 0:
                    sil_tags.append((0, pos))
                else:
                    sil_tags.append((pos, pos))
                clip_start = pos
            elif i - silence_start <= self.max_sil_kept * 2:
                pos = rms_list[i - self.max_sil_kept : silence_start + self.max_sil_kept + 1].argmin()
                pos += i - self.max_sil_kept
                pos_l = rms_list[silence_start : silence_start + self.max_sil_kept + 1].argmin() + silence_start
                pos_r = rms_list[i - self.max_sil_kept : i + 1].argmin() + i - self.max_sil_kept
                if silence_start == 0:
                    sil_tags.append((0, pos_r))
                    clip_start = pos_r
                else:
                    sil_tags.append((min(pos_l, pos), max(pos_r, pos)))
                    clip_start = max(pos_r, pos)
            else:
                pos_l = rms_list[silence_start : silence_start + self.max_sil_kept + 1].argmin() + silence_start
                pos_r = rms_list[i - self.max_sil_kept : i + 1].argmin() + i - self.max_sil_kept
                if silence_start == 0:
                    sil_tags.append((0, pos_r))
                else:
                    sil_tags.append((pos_l, pos_r))
                clip_start = pos_r
            silence_start = None
        # Deal with trailing silence.
        total_frames = rms_list.shape[0]
        if silence_start is not None and total_frames - silence_start >= self.min_interval:
            silence_end = min(total_frames, silence_start + self.max_sil_kept)
            pos = rms_list[silence_start : silence_end + 1].argmin() + silence_start
            sil_tags.append((pos, total_frames + 1))
        # Apply and return slices.
        ####音频+起始时间+终止时间
        if len(sil_tags) == 0:
            return [[waveform, 0, int(total_frames * self.hop_size)]]
        else:
            chunks = []
            if sil_tags[0][0] > 0:
                chunks.append([self._apply_slice(waveform, 0, sil_tags[0][0]), 0, int(sil_tags[0][0] * self.hop_size)])
            for i in range(len(sil_tags) - 1):
                chunks.append(
                    [
                        self._apply_slice(waveform, sil_tags[i][1], sil_tags[i + 1][0]),
                        int(sil_tags[i][1] * self.hop_size),
                        int(sil_tags[i + 1][0] * self.hop_size),
                    ]
                )
            if sil_tags[-1][1] < total_frames:
                chunks.append(
                    [
                        self._apply_slice(waveform, sil_tags[-1][1], total_frames),
                        int(sil_tags[-1][1] * self.hop_size),
                        int(total_frames * self.hop_size),
                    ]
                )
            return chunks


# terminal
def terminate_process_tree(pid, including_parent=True):
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        # Process already terminated
        return

    children = parent.children(recursive=True)
    for child in children:
        try:
            os.kill(child.pid, signal.SIGTERM)  # or signal.SIGKILL
        except OSError:
            pass
    if including_parent:
        try:
            os.kill(parent.pid, signal.SIGTERM)  # or signal.SIGKILL
        except OSError:
            pass


def terminate_process(pid):
    if system == "Windows":
        cmd = f"taskkill /t /f /pid {pid}"
        os.system(cmd)
    else:
        terminate_process_tree(pid)


def start_training(
    dataset_name="",
    exp_name="F5TTS_Base",
    learning_rate=1e-4,
    batch_size_per_gpu=400,
    batch_size_type="frame",
    max_samples=64,
    grad_accumulation_steps=1,
    max_grad_norm=1.0,
    epochs=11,
    num_warmup_updates=200,
    save_per_updates=400,
    last_per_steps=800,
    finetune=True,
    file_checkpoint_train="",
    tokenizer_type="pinyin",
    tokenizer_file="",
    mixed_precision="fp16",
):
    global training_process, tts_api

    if tts_api is not None:
        del tts_api
        gc.collect()
        torch.cuda.empty_cache()
        tts_api = None

    path_project = os.path.join(path_data, dataset_name)

    if not os.path.isdir(path_project):
        yield (
            f"There is not project with name {dataset_name}",
            gr.update(interactive=True),
            gr.update(interactive=False),
        )
        return

    file_raw = os.path.join(path_project, "raw.arrow")
    if not os.path.isfile(file_raw):
        yield f"There is no file {file_raw}", gr.update(interactive=True), gr.update(interactive=False)
        return

    # Check if a training process is already running
    if training_process is not None:
        return "Train run already!", gr.update(interactive=False), gr.update(interactive=True)

    yield "start train", gr.update(interactive=False), gr.update(interactive=False)

    # Command to run the training script with the specified arguments

    if tokenizer_file == "":
        if dataset_name.endswith("_pinyin"):
            tokenizer_type = "pinyin"
        elif dataset_name.endswith("_char"):
            tokenizer_type = "char"
    else:
        tokenizer_file = "custom"

    dataset_name = dataset_name.replace("_pinyin", "").replace("_char", "")

    if mixed_precision != "none":
        fp16 = f"--mixed_precision={mixed_precision}"
    else:
        fp16 = ""

    cmd = (
        f"accelerate launch {fp16} finetune-cli.py --exp_name {exp_name} "
        f"--learning_rate {learning_rate} "
        f"--batch_size_per_gpu {batch_size_per_gpu} "
        f"--batch_size_type {batch_size_type} "
        f"--max_samples {max_samples} "
        f"--grad_accumulation_steps {grad_accumulation_steps} "
        f"--max_grad_norm {max_grad_norm} "
        f"--epochs {epochs} "
        f"--num_warmup_updates {num_warmup_updates} "
        f"--save_per_updates {save_per_updates} "
        f"--last_per_steps {last_per_steps} "
        f"--dataset_name {dataset_name}"
    )
    if finetune:
        cmd += f" --finetune {finetune}"

    if file_checkpoint_train != "":
        cmd += f" --file_checkpoint_train {file_checkpoint_train}"

    if tokenizer_file != "":
        cmd += f" --tokenizer_path {tokenizer_file}"

    cmd += f" --tokenizer {tokenizer_type}"

    print(cmd)

    try:
        # Start the training process
        training_process = subprocess.Popen(cmd, shell=True)

        time.sleep(5)
        yield "train start", gr.update(interactive=False), gr.update(interactive=True)

        # Wait for the training process to finish
        training_process.wait()
        time.sleep(1)

        if training_process is None:
            text_info = "train stop"
        else:
            text_info = "train complete !"

    except Exception as e:  # Catch all exceptions
        # Ensure that we reset the training process variable in case of an error
        text_info = f"An error occurred: {str(e)}"

    training_process = None

    yield text_info, gr.update(interactive=True), gr.update(interactive=False)


def stop_training():
    global training_process
    if training_process is None:
        return "Train not run !", gr.update(interactive=True), gr.update(interactive=False)
    terminate_process_tree(training_process.pid)
    training_process = None
    return "train stop", gr.update(interactive=True), gr.update(interactive=False)


def get_list_projects():
    project_list = []
    for folder in os.listdir("data"):
        path_folder = os.path.join("data", folder)
        if not os.path.isdir(path_folder):
            continue
        folder = folder.lower()
        if folder == "emilia_zh_en_pinyin":
            continue
        project_list.append(folder)

    projects_selelect = None if not project_list else project_list[-1]

    return project_list, projects_selelect


def create_data_project(name, tokenizer_type):
    name += "_" + tokenizer_type
    os.makedirs(os.path.join(path_data, name), exist_ok=True)
    os.makedirs(os.path.join(path_data, name, "dataset"), exist_ok=True)
    project_list, projects_selelect = get_list_projects()
    return gr.update(choices=project_list, value=name)


def transcribe(file_audio, language="english"):
    global pipe

    if pipe is None:
        pipe = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-large-v3-turbo",
            torch_dtype=torch.float16,
            device=device,
        )

    text_transcribe = pipe(
        file_audio,
        chunk_length_s=30,
        batch_size=128,
        generate_kwargs={"task": "transcribe", "language": language},
        return_timestamps=False,
    )["text"].strip()
    return text_transcribe


def transcribe_all(name_project, audio_files, language, user=False, progress=gr.Progress()):
    path_project = os.path.join(path_data, name_project)
    path_dataset = os.path.join(path_project, "dataset")
    path_project_wavs = os.path.join(path_project, "wavs")
    file_metadata = os.path.join(path_project, "metadata.csv")

    if not user:
        if audio_files is None:
            return "You need to load an audio file."

    if os.path.isdir(path_project_wavs):
        shutil.rmtree(path_project_wavs)

    if os.path.isfile(file_metadata):
        os.remove(file_metadata)

    os.makedirs(path_project_wavs, exist_ok=True)

    if user:
        file_audios = [
            file
            for format in ("*.wav", "*.ogg", "*.opus", "*.mp3", "*.flac")
            for file in glob(os.path.join(path_dataset, format))
        ]
        if file_audios == []:
            return "No audio file was found in the dataset."
    else:
        file_audios = audio_files

    alpha = 0.5
    _max = 1.0
    slicer = Slicer(24000)

    num = 0
    error_num = 0
    data = ""
    for file_audio in progress.tqdm(file_audios, desc="transcribe files", total=len((file_audios))):
        audio, _ = librosa.load(file_audio, sr=24000, mono=True)

        list_slicer = slicer.slice(audio)
        for chunk, start, end in progress.tqdm(list_slicer, total=len(list_slicer), desc="slicer files"):
            name_segment = os.path.join(f"segment_{num}")
            file_segment = os.path.join(path_project_wavs, f"{name_segment}.wav")

            tmp_max = np.abs(chunk).max()
            if tmp_max > 1:
                chunk /= tmp_max
            chunk = (chunk / tmp_max * (_max * alpha)) + (1 - alpha) * chunk
            wavfile.write(file_segment, 24000, (chunk * 32767).astype(np.int16))

            try:
                text = transcribe(file_segment, language)
                text = text.lower().strip().replace('"', "")

                data += f"{name_segment}|{text}\n"

                num += 1
            except:  # noqa: E722
                error_num += 1

    with open(file_metadata, "w", encoding="utf-8-sig") as f:
        f.write(data)

    if error_num != []:
        error_text = f"\nerror files : {error_num}"
    else:
        error_text = ""

    return f"transcribe complete samples : {num}\npath : {path_project_wavs}{error_text}"


def format_seconds_to_hms(seconds):
    hours = int(seconds / 3600)
    minutes = int((seconds % 3600) / 60)
    seconds = seconds % 60
    return "{:02d}:{:02d}:{:02d}".format(hours, minutes, int(seconds))


def create_metadata(name_project, ch_tokenizer, progress=gr.Progress()):
    path_project = os.path.join(path_data, name_project)
    path_project_wavs = os.path.join(path_project, "wavs")
    file_metadata = os.path.join(path_project, "metadata.csv")
    file_raw = os.path.join(path_project, "raw.arrow")
    file_duration = os.path.join(path_project, "duration.json")
    file_vocab = os.path.join(path_project, "vocab.txt")

    if not os.path.isfile(file_metadata):
        return "The file was not found in " + file_metadata, ""

    with open(file_metadata, "r", encoding="utf-8-sig") as f:
        data = f.read()

    audio_path_list = []
    text_list = []
    duration_list = []

    count = data.split("\n")
    lenght = 0
    result = []
    error_files = []
    text_vocab_set = set()
    for line in progress.tqdm(data.split("\n"), total=count):
        sp_line = line.split("|")
        if len(sp_line) != 2:
            continue
        name_audio, text = sp_line[:2]

        file_audio = os.path.join(path_project_wavs, name_audio + ".wav")

        if not os.path.isfile(file_audio):
            error_files.append([file_audio, "error path"])
            continue

        try:
            duration = get_audio_duration(file_audio)
        except Exception as e:
            error_files.append([file_audio, "duration"])
            print(f"Error processing {file_audio}: {e}")
            continue

        if duration < 1 and duration > 25:
            error_files.append([file_audio, "duration < 1 and > 25 "])
            continue
        if len(text) < 4:
            error_files.append([file_audio, "very small text len 3"])
            continue

        text = clear_text(text)
        text = convert_char_to_pinyin([text], polyphone=True)[0]

        audio_path_list.append(file_audio)
        duration_list.append(duration)
        text_list.append(text)

        result.append({"audio_path": file_audio, "text": text, "duration": duration})

        if ch_tokenizer:
            text_vocab_set.update(list(text))

        lenght += duration

    if duration_list == []:
        return f"Error: No audio files found in the specified path : {path_project_wavs}", ""

    min_second = round(min(duration_list), 2)
    max_second = round(max(duration_list), 2)

    with ArrowWriter(path=file_raw, writer_batch_size=1) as writer:
        for line in progress.tqdm(result, total=len(result), desc="prepare data"):
            writer.write(line)

    with open(file_duration, "w") as f:
        json.dump({"duration": duration_list}, f, ensure_ascii=False)

    new_vocal = ""
    if not ch_tokenizer:
        file_vocab_finetune = "data/Emilia_ZH_EN_pinyin/vocab.txt"
        if not os.path.isfile(file_vocab_finetune):
            return "Error: Vocabulary file 'Emilia_ZH_EN_pinyin' not found!"
        shutil.copy2(file_vocab_finetune, file_vocab)

        with open(file_vocab, "r", encoding="utf-8-sig") as f:
            vocab_char_map = {}
            for i, char in enumerate(f):
                vocab_char_map[char[:-1]] = i
        vocab_size = len(vocab_char_map)

    else:
        with open(file_vocab, "w", encoding="utf-8-sig") as f:
            for vocab in sorted(text_vocab_set):
                f.write(vocab + "\n")
                new_vocal += vocab + "\n"
        vocab_size = len(text_vocab_set)

    if error_files != []:
        error_text = "\n".join([" = ".join(item) for item in error_files])
    else:
        error_text = ""

    return (
        f"prepare complete \nsamples : {len(text_list)}\ntime data : {format_seconds_to_hms(lenght)}\nmin sec : {min_second}\nmax sec : {max_second}\nfile_arrow : {file_raw}\nvocab : {vocab_size}\n{error_text}",
        new_vocal,
    )


def check_user(value):
    return gr.update(visible=not value), gr.update(visible=value)


def calculate_train(
    name_project,
    batch_size_type,
    max_samples,
    learning_rate,
    num_warmup_updates,
    save_per_updates,
    last_per_steps,
    finetune,
):
    path_project = os.path.join(path_data, name_project)
    file_duraction = os.path.join(path_project, "duration.json")

    if not os.path.isfile(file_duraction):
        return (
            1000,
            max_samples,
            num_warmup_updates,
            save_per_updates,
            last_per_steps,
            "project not found !",
            learning_rate,
        )

    with open(file_duraction, "r") as file:
        data = json.load(file)

    duration_list = data["duration"]
    samples = len(duration_list)
    hours = sum(duration_list) / 3600

    # if torch.cuda.is_available():
    # gpu_properties = torch.cuda.get_device_properties(0)
    # total_memory = gpu_properties.total_memory / (1024**3)
    # elif torch.backends.mps.is_available():
    # total_memory = psutil.virtual_memory().available / (1024**3)

    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        total_memory = 0
        for i in range(gpu_count):
            gpu_properties = torch.cuda.get_device_properties(i)
            total_memory += gpu_properties.total_memory / (1024**3)  # in GB

    elif torch.backends.mps.is_available():
        gpu_count = 1
        total_memory = psutil.virtual_memory().available / (1024**3)

    if batch_size_type == "frame":
        batch = int(total_memory * 0.5)
        batch = (lambda num: num + 1 if num % 2 != 0 else num)(batch)
        batch_size_per_gpu = int(38400 / batch)
    else:
        batch_size_per_gpu = int(total_memory / 8)
        batch_size_per_gpu = (lambda num: num + 1 if num % 2 != 0 else num)(batch_size_per_gpu)
        batch = batch_size_per_gpu

    if batch_size_per_gpu <= 0:
        batch_size_per_gpu = 1

    if samples < 64:
        max_samples = int(samples * 0.25)
    else:
        max_samples = 64

    num_warmup_updates = int(samples * 0.05)
    save_per_updates = int(samples * 0.10)
    last_per_steps = int(save_per_updates * 5)

    max_samples = (lambda num: num + 1 if num % 2 != 0 else num)(max_samples)
    num_warmup_updates = (lambda num: num + 1 if num % 2 != 0 else num)(num_warmup_updates)
    save_per_updates = (lambda num: num + 1 if num % 2 != 0 else num)(save_per_updates)
    last_per_steps = (lambda num: num + 1 if num % 2 != 0 else num)(last_per_steps)

    total_hours = hours
    mel_hop_length = 256
    mel_sampling_rate = 24000

    # target
    wanted_max_updates = 1000000

    # train params
    gpus = gpu_count
    frames_per_gpu = batch_size_per_gpu  # 8 * 38400 = 307200
    grad_accum = 1

    # intermediate
    mini_batch_frames = frames_per_gpu * grad_accum * gpus
    mini_batch_hours = mini_batch_frames * mel_hop_length / mel_sampling_rate / 3600
    updates_per_epoch = total_hours / mini_batch_hours
    # steps_per_epoch = updates_per_epoch * grad_accum
    epochs = wanted_max_updates / updates_per_epoch

    if finetune:
        learning_rate = 1e-5
    else:
        learning_rate = 7.5e-5

    return (
        batch_size_per_gpu,
        max_samples,
        num_warmup_updates,
        save_per_updates,
        last_per_steps,
        samples,
        learning_rate,
        int(epochs),
    )


def extract_and_save_ema_model(checkpoint_path: str, new_checkpoint_path: str, safetensors: bool) -> str:
    try:
        checkpoint = torch.load(checkpoint_path)
        print("Original Checkpoint Keys:", checkpoint.keys())

        ema_model_state_dict = checkpoint.get("ema_model_state_dict", None)
        if ema_model_state_dict is None:
            return "No 'ema_model_state_dict' found in the checkpoint."

        if safetensors:
            new_checkpoint_path = new_checkpoint_path.replace(".pt", ".safetensors")
            save_file(ema_model_state_dict, new_checkpoint_path)
        else:
            new_checkpoint_path = new_checkpoint_path.replace(".safetensors", ".pt")
            new_checkpoint = {"ema_model_state_dict": ema_model_state_dict}
            torch.save(new_checkpoint, new_checkpoint_path)

        return f"New checkpoint saved at: {new_checkpoint_path}"

    except Exception as e:
        return f"An error occurred: {e}"


def vocab_check(project_name):
    name_project = project_name
    path_project = os.path.join(path_data, name_project)

    file_metadata = os.path.join(path_project, "metadata.csv")

    file_vocab = "data/Emilia_ZH_EN_pinyin/vocab.txt"
    if not os.path.isfile(file_vocab):
        return f"the file {file_vocab} not found !"

    with open(file_vocab, "r", encoding="utf-8-sig") as f:
        data = f.read()
        vocab = data.split("\n")
        vocab = set(vocab)

    if not os.path.isfile(file_metadata):
        return f"the file {file_metadata} not found !"

    with open(file_metadata, "r", encoding="utf-8-sig") as f:
        data = f.read()

    miss_symbols = []
    miss_symbols_keep = {}
    for item in data.split("\n"):
        sp = item.split("|")
        if len(sp) != 2:
            continue

        text = sp[1].lower().strip()

        for t in text:
            if t not in vocab and t not in miss_symbols_keep:
                miss_symbols.append(t)
                miss_symbols_keep[t] = t
    if miss_symbols == []:
        info = "You can train using your language !"
    else:
        info = f"The following symbols are missing in your language : {len(miss_symbols)}\n\n" + "\n".join(miss_symbols)

    return info


def get_random_sample_prepare(project_name):
    name_project = project_name
    path_project = os.path.join(path_data, name_project)
    file_arrow = os.path.join(path_project, "raw.arrow")
    if not os.path.isfile(file_arrow):
        return "", None
    dataset = Dataset_.from_file(file_arrow)
    random_sample = dataset.shuffle(seed=random.randint(0, 1000)).select([0])
    text = "[" + " , ".join(["' " + t + " '" for t in random_sample["text"][0]]) + "]"
    audio_path = random_sample["audio_path"][0]
    return text, audio_path


def get_random_sample_transcribe(project_name):
    name_project = project_name
    path_project = os.path.join(path_data, name_project)
    file_metadata = os.path.join(path_project, "metadata.csv")
    if not os.path.isfile(file_metadata):
        return "", None

    data = ""
    with open(file_metadata, "r", encoding="utf-8-sig") as f:
        data = f.read()

    list_data = []
    for item in data.split("\n"):
        sp = item.split("|")
        if len(sp) != 2:
            continue
        list_data.append([os.path.join(path_project, "wavs", sp[0] + ".wav"), sp[1]])

    if list_data == []:
        return "", None

    random_item = random.choice(list_data)

    return random_item[1], random_item[0]


def get_random_sample_infer(project_name):
    text, audio = get_random_sample_transcribe(project_name)
    return (
        text,
        text,
        audio,
    )


def infer(file_checkpoint, exp_name, ref_text, ref_audio, gen_text, nfe_step):
    global last_checkpoint, last_device, tts_api

    if not os.path.isfile(file_checkpoint):
        return None, "checkpoint not found!"

    if training_process is not None:
        device_test = "cpu"
    else:
        device_test = None

    if last_checkpoint != file_checkpoint or last_device != device_test:
        if last_checkpoint != file_checkpoint:
            last_checkpoint = file_checkpoint
        if last_device != device_test:
            last_device = device_test

        tts_api = F5TTS(model_type=exp_name, ckpt_file=file_checkpoint, device=device_test)

        print("update", device_test, file_checkpoint)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        tts_api.infer(gen_text=gen_text, ref_text=ref_text, ref_file=ref_audio, nfe_step=nfe_step, file_wave=f.name)
        return f.name, tts_api.device


def check_finetune(finetune):
    return gr.update(interactive=finetune), gr.update(interactive=finetune), gr.update(interactive=finetune)


def get_checkpoints_project(project_name, is_gradio=True):
    if project_name is None:
        return [], ""
    project_name = project_name.replace("_pinyin", "").replace("_char", "")
    path_project_ckpts = os.path.join("ckpts", project_name)

    if os.path.isdir(path_project_ckpts):
        files_checkpoints = glob(os.path.join(path_project_ckpts, "*.pt"))
        files_checkpoints = sorted(
            files_checkpoints,
            key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0])
            if os.path.basename(x) != "model_last.pt"
            else float("inf"),
        )
    else:
        files_checkpoints = []

    selelect_checkpoint = None if not files_checkpoints else files_checkpoints[0]

    if is_gradio:
        return gr.update(choices=files_checkpoints, value=selelect_checkpoint)

    return files_checkpoints, selelect_checkpoint


def get_gpu_stats():
    gpu_stats = ""

    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_properties = torch.cuda.get_device_properties(i)
            total_memory = gpu_properties.total_memory / (1024**3)  # in GB
            allocated_memory = torch.cuda.memory_allocated(i) / (1024**2)  # in MB
            reserved_memory = torch.cuda.memory_reserved(i) / (1024**2)  # in MB

            gpu_stats += (
                f"GPU {i} Name: {gpu_name}\n"
                f"Total GPU memory (GPU {i}): {total_memory:.2f} GB\n"
                f"Allocated GPU memory (GPU {i}): {allocated_memory:.2f} MB\n"
                f"Reserved GPU memory (GPU {i}): {reserved_memory:.2f} MB\n\n"
            )

    elif torch.backends.mps.is_available():
        gpu_count = 1
        gpu_stats += "MPS GPU\n"
        total_memory = psutil.virtual_memory().total / (
            1024**3
        )  # Total system memory (MPS doesn't have its own memory)
        allocated_memory = 0
        reserved_memory = 0

        gpu_stats += (
            f"Total system memory: {total_memory:.2f} GB\n"
            f"Allocated GPU memory (MPS): {allocated_memory:.2f} MB\n"
            f"Reserved GPU memory (MPS): {reserved_memory:.2f} MB\n"
        )

    else:
        gpu_stats = "No GPU available"

    return gpu_stats


def get_cpu_stats():
    cpu_usage = psutil.cpu_percent(interval=1)
    memory_info = psutil.virtual_memory()
    memory_used = memory_info.used / (1024**2)
    memory_total = memory_info.total / (1024**2)
    memory_percent = memory_info.percent

    pid = os.getpid()
    process = psutil.Process(pid)
    nice_value = process.nice()

    cpu_stats = (
        f"CPU Usage: {cpu_usage:.2f}%\n"
        f"System Memory: {memory_used:.2f} MB used / {memory_total:.2f} MB total ({memory_percent}% used)\n"
        f"Process Priority (Nice value): {nice_value}"
    )

    return cpu_stats


def get_combined_stats():
    gpu_stats = get_gpu_stats()
    cpu_stats = get_cpu_stats()
    combined_stats = f"### GPU Stats\n{gpu_stats}\n\n### CPU Stats\n{cpu_stats}"
    return combined_stats


with gr.Blocks() as app:
    gr.Markdown(
        """
# E2/F5 TTS AUTOMATIC FINETUNE 

This is a local web UI for F5 TTS with advanced batch processing support. This app supports the following TTS models:

* [F5-TTS](https://arxiv.org/abs/2410.06885) (A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching)
* [E2 TTS](https://arxiv.org/abs/2406.18009) (Embarrassingly Easy Fully Non-Autoregressive Zero-Shot TTS)

The checkpoints support English and Chinese.

for tutorial and updates check here (https://github.com/SWivid/F5-TTS/discussions/143)
"""
    )

    with gr.Row():
        projects, projects_selelect = get_list_projects()
        tokenizer_type = gr.Radio(label="Tokenizer Type", choices=["pinyin", "char"], value="pinyin")
        project_name = gr.Textbox(label="project name", value="my_speak")
        bt_create = gr.Button("create new project")

    cm_project = gr.Dropdown(choices=projects, value=projects_selelect, label="Project", allow_custom_value=True)

    bt_create.click(fn=create_data_project, inputs=[project_name, tokenizer_type], outputs=[cm_project])

    with gr.Tabs():
        with gr.TabItem("transcribe Data"):
            ch_manual = gr.Checkbox(label="audio from path", value=False)

            mark_info_transcribe = gr.Markdown(
                """```plaintext    
     Place your 'wavs' folder and 'metadata.csv' file in the {your_project_name}' directory. 
                 
     my_speak/
     │
     └── dataset/
         ├── audio1.wav
         └── audio2.wav
         ...
     ```""",
                visible=False,
            )

            audio_speaker = gr.File(label="voice", type="filepath", file_count="multiple")
            txt_lang = gr.Text(label="Language", value="english")
            bt_transcribe = bt_create = gr.Button("transcribe")
            txt_info_transcribe = gr.Text(label="info", value="")
            bt_transcribe.click(
                fn=transcribe_all,
                inputs=[cm_project, audio_speaker, txt_lang, ch_manual],
                outputs=[txt_info_transcribe],
            )
            ch_manual.change(fn=check_user, inputs=[ch_manual], outputs=[audio_speaker, mark_info_transcribe])

            random_sample_transcribe = gr.Button("random sample")

            with gr.Row():
                random_text_transcribe = gr.Text(label="Text")
                random_audio_transcribe = gr.Audio(label="Audio", type="filepath")

            random_sample_transcribe.click(
                fn=get_random_sample_transcribe,
                inputs=[cm_project],
                outputs=[random_text_transcribe, random_audio_transcribe],
            )

        with gr.TabItem("prepare Data"):
            gr.Markdown(
                """```plaintext    
     place all your wavs folder and your metadata.csv file in {your name project}                                 
     my_speak/
     │
     ├── wavs/
     │   ├── audio1.wav
     │   └── audio2.wav
     |   ...
     │
     └── metadata.csv
      
     file format metadata.csv

     audio1|text1
     audio2|text1
     ...

     ```"""
            )
            ch_tokenizern = gr.Checkbox(label="create vocabulary from dataset", value=False)
            bt_prepare = bt_create = gr.Button("prepare")
            txt_info_prepare = gr.Text(label="info", value="")
            txt_vocab_prepare = gr.Text(label="vocab", value="")
            bt_prepare.click(
                fn=create_metadata, inputs=[cm_project, ch_tokenizern], outputs=[txt_info_prepare, txt_vocab_prepare]
            )

            random_sample_prepare = gr.Button("random sample")

            with gr.Row():
                random_text_prepare = gr.Text(label="Pinyin")
                random_audio_prepare = gr.Audio(label="Audio", type="filepath")

            random_sample_prepare.click(
                fn=get_random_sample_prepare, inputs=[cm_project], outputs=[random_text_prepare, random_audio_prepare]
            )

        with gr.TabItem("train Data"):
            with gr.Row():
                bt_calculate = bt_create = gr.Button("Auto Settings")
                lb_samples = gr.Label(label="samples")
                batch_size_type = gr.Radio(label="Batch Size Type", choices=["frame", "sample"], value="frame")

            with gr.Row():
                ch_finetune = bt_create = gr.Checkbox(label="finetune", value=True)
                tokenizer_file = gr.Textbox(label="Tokenizer File", value="")
                file_checkpoint_train = gr.Textbox(label="Pretrain Model", value="")

            with gr.Row():
                exp_name = gr.Radio(label="Model", choices=["F5TTS_Base", "E2TTS_Base"], value="F5TTS_Base")
                learning_rate = gr.Number(label="Learning Rate", value=1e-5, step=1e-5)

            with gr.Row():
                batch_size_per_gpu = gr.Number(label="Batch Size per GPU", value=1000)
                max_samples = gr.Number(label="Max Samples", value=64)

            with gr.Row():
                grad_accumulation_steps = gr.Number(label="Gradient Accumulation Steps", value=1)
                max_grad_norm = gr.Number(label="Max Gradient Norm", value=1.0)

            with gr.Row():
                epochs = gr.Number(label="Epochs", value=10)
                num_warmup_updates = gr.Number(label="Warmup Updates", value=5)

            with gr.Row():
                save_per_updates = gr.Number(label="Save per Updates", value=10)
                last_per_steps = gr.Number(label="Last per Steps", value=50)

            with gr.Row():
                mixed_precision = gr.Radio(label="mixed_precision", choices=["none", "fp16", "fpb16"], value="none")
                start_button = gr.Button("Start Training")
                stop_button = gr.Button("Stop Training", interactive=False)

            txt_info_train = gr.Text(label="info", value="")
            start_button.click(
                fn=start_training,
                inputs=[
                    cm_project,
                    exp_name,
                    learning_rate,
                    batch_size_per_gpu,
                    batch_size_type,
                    max_samples,
                    grad_accumulation_steps,
                    max_grad_norm,
                    epochs,
                    num_warmup_updates,
                    save_per_updates,
                    last_per_steps,
                    ch_finetune,
                    file_checkpoint_train,
                    tokenizer_type,
                    tokenizer_file,
                    mixed_precision,
                ],
                outputs=[txt_info_train, start_button, stop_button],
            )
            stop_button.click(fn=stop_training, outputs=[txt_info_train, start_button, stop_button])

            bt_calculate.click(
                fn=calculate_train,
                inputs=[
                    cm_project,
                    batch_size_type,
                    max_samples,
                    learning_rate,
                    num_warmup_updates,
                    save_per_updates,
                    last_per_steps,
                    ch_finetune,
                ],
                outputs=[
                    batch_size_per_gpu,
                    max_samples,
                    num_warmup_updates,
                    save_per_updates,
                    last_per_steps,
                    lb_samples,
                    learning_rate,
                    epochs,
                ],
            )

            ch_finetune.change(
                check_finetune, inputs=[ch_finetune], outputs=[file_checkpoint_train, tokenizer_file, tokenizer_type]
            )

        with gr.TabItem("reduse checkpoint"):
            txt_path_checkpoint = gr.Text(label="path checkpoint :")
            txt_path_checkpoint_small = gr.Text(label="path output :")
            ch_safetensors = gr.Checkbox(label="safetensors", value="")
            txt_info_reduse = gr.Text(label="info", value="")
            reduse_button = gr.Button("reduse")
            reduse_button.click(
                fn=extract_and_save_ema_model,
                inputs=[txt_path_checkpoint, txt_path_checkpoint_small, ch_safetensors],
                outputs=[txt_info_reduse],
            )

        with gr.TabItem("vocab check"):
            check_button = gr.Button("check vocab")
            txt_info_check = gr.Text(label="info", value="")
            check_button.click(fn=vocab_check, inputs=[cm_project], outputs=[txt_info_check])

        with gr.TabItem("test model"):
            exp_name = gr.Radio(label="Model", choices=["F5-TTS", "E2-TTS"], value="F5-TTS")
            list_checkpoints, checkpoint_select = get_checkpoints_project(projects_selelect, False)

            nfe_step = gr.Number(label="n_step", value=32)

            with gr.Row():
                cm_checkpoint = gr.Dropdown(
                    choices=list_checkpoints, value=checkpoint_select, label="checkpoints", allow_custom_value=True
                )
                bt_checkpoint_refresh = gr.Button("refresh")

            random_sample_infer = gr.Button("random sample")

            ref_text = gr.Textbox(label="ref text")
            ref_audio = gr.Audio(label="audio ref", type="filepath")
            gen_text = gr.Textbox(label="gen text")
            random_sample_infer.click(
                fn=get_random_sample_infer, inputs=[cm_project], outputs=[ref_text, gen_text, ref_audio]
            )

            with gr.Row():
                txt_info_gpu = gr.Textbox("", label="device")
                check_button_infer = gr.Button("infer")

            gen_audio = gr.Audio(label="audio gen", type="filepath")

            check_button_infer.click(
                fn=infer,
                inputs=[cm_checkpoint, exp_name, ref_text, ref_audio, gen_text, nfe_step],
                outputs=[gen_audio, txt_info_gpu],
            )

            bt_checkpoint_refresh.click(fn=get_checkpoints_project, inputs=[cm_project], outputs=[cm_checkpoint])
            cm_project.change(fn=get_checkpoints_project, inputs=[cm_project], outputs=[cm_checkpoint])

        with gr.TabItem("system info"):
            output_box = gr.Textbox(label="GPU and CPU Information", lines=20)

            def update_stats():
                return get_combined_stats()

            update_button = gr.Button("Update Stats")
            update_button.click(fn=update_stats, outputs=output_box)

            def auto_update():
                yield gr.update(value=update_stats())

            gr.update(fn=auto_update, inputs=[], outputs=output_box)


@click.command()
@click.option("--port", "-p", default=None, type=int, help="Port to run the app on")
@click.option("--host", "-H", default=None, help="Host to run the app on")
@click.option(
    "--share",
    "-s",
    default=False,
    is_flag=True,
    help="Share the app via Gradio share link",
)
@click.option("--api", "-a", default=True, is_flag=True, help="Allow API access")
def main(port, host, share, api):
    global app
    print("Starting app...")
    app.queue(api_open=api).launch(server_name=host, server_port=port, share=share, show_api=api)


if __name__ == "__main__":
    main()
