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
    # put our new within-frame axis at the end for now
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
):
    global training_process, tts_api

    if tts_api is not None:
        del tts_api
        gc.collect()
        torch.cuda.empty_cache()

    path_project = os.path.join(path_data, dataset_name + "_pinyin")

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
    cmd = (
        f"accelerate launch finetune-cli.py --exp_name {exp_name} "
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


def create_data_project(name):
    name += "_pinyin"
    os.makedirs(os.path.join(path_data, name), exist_ok=True)
    os.makedirs(os.path.join(path_data, name, "dataset"), exist_ok=True)


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
    name_project += "_pinyin"
    path_project = os.path.join(path_data, name_project)
    path_dataset = os.path.join(path_project, "dataset")
    path_project_wavs = os.path.join(path_project, "wavs")
    file_metadata = os.path.join(path_project, "metadata.csv")

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

    with open(file_metadata, "w", encoding="utf-8") as f:
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


def create_metadata(name_project, progress=gr.Progress()):
    name_project += "_pinyin"
    path_project = os.path.join(path_data, name_project)
    path_project_wavs = os.path.join(path_project, "wavs")
    file_metadata = os.path.join(path_project, "metadata.csv")
    file_raw = os.path.join(path_project, "raw.arrow")
    file_duration = os.path.join(path_project, "duration.json")
    file_vocab = os.path.join(path_project, "vocab.txt")

    if not os.path.isfile(file_metadata):
        return "The file was not found in " + file_metadata

    with open(file_metadata, "r", encoding="utf-8") as f:
        data = f.read()

    audio_path_list = []
    text_list = []
    duration_list = []

    count = data.split("\n")
    lenght = 0
    result = []
    error_files = []
    for line in progress.tqdm(data.split("\n"), total=count):
        sp_line = line.split("|")
        if len(sp_line) != 2:
            continue
        name_audio, text = sp_line[:2]

        file_audio = os.path.join(path_project_wavs, name_audio + ".wav")

        if not os.path.isfile(file_audio):
            error_files.append(file_audio)
            continue

        duraction = get_audio_duration(file_audio)
        if duraction < 2 and duraction > 15:
            continue
        if len(text) < 4:
            continue

        text = clear_text(text)
        text = convert_char_to_pinyin([text], polyphone=True)[0]

        audio_path_list.append(file_audio)
        duration_list.append(duraction)
        text_list.append(text)

        result.append({"audio_path": file_audio, "text": text, "duration": duraction})

        lenght += duraction

    if duration_list == []:
        error_files_text = "\n".join(error_files)
        return f"Error: No audio files found in the specified path : \n{error_files_text}"

    min_second = round(min(duration_list), 2)
    max_second = round(max(duration_list), 2)

    with ArrowWriter(path=file_raw, writer_batch_size=1) as writer:
        for line in progress.tqdm(result, total=len(result), desc="prepare data"):
            writer.write(line)

    with open(file_duration, "w", encoding="utf-8") as f:
        json.dump({"duration": duration_list}, f, ensure_ascii=False)

    file_vocab_finetune = "data/Emilia_ZH_EN_pinyin/vocab.txt"
    if not os.path.isfile(file_vocab_finetune):
        return "Error: Vocabulary file 'Emilia_ZH_EN_pinyin' not found!"
    shutil.copy2(file_vocab_finetune, file_vocab)

    if error_files != []:
        error_text = "error files\n" + "\n".join(error_files)
    else:
        error_text = ""

    return f"prepare complete \nsamples : {len(text_list)}\ntime data : {format_seconds_to_hms(lenght)}\nmin sec : {min_second}\nmax sec : {max_second}\nfile_arrow : {file_raw}\n{error_text}"


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
    name_project += "_pinyin"
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

    if torch.cuda.is_available():
        gpu_properties = torch.cuda.get_device_properties(0)
        total_memory = gpu_properties.total_memory / (1024**3)
    elif torch.backends.mps.is_available():
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

    if finetune:
        learning_rate = 1e-5
    else:
        learning_rate = 7.5e-5

    return batch_size_per_gpu, max_samples, num_warmup_updates, save_per_updates, last_per_steps, samples, learning_rate


def extract_and_save_ema_model(checkpoint_path: str, new_checkpoint_path: str) -> None:
    try:
        checkpoint = torch.load(checkpoint_path)
        print("Original Checkpoint Keys:", checkpoint.keys())

        ema_model_state_dict = checkpoint.get("ema_model_state_dict", None)

        if ema_model_state_dict is not None:
            new_checkpoint = {"ema_model_state_dict": ema_model_state_dict}
            torch.save(new_checkpoint, new_checkpoint_path)
            return f"New checkpoint saved at: {new_checkpoint_path}"
        else:
            return "No 'ema_model_state_dict' found in the checkpoint."

    except Exception as e:
        return f"An error occurred: {e}"


def vocab_check(project_name):
    name_project = project_name + "_pinyin"
    path_project = os.path.join(path_data, name_project)

    file_metadata = os.path.join(path_project, "metadata.csv")

    file_vocab = "data/Emilia_ZH_EN_pinyin/vocab.txt"
    if not os.path.isfile(file_vocab):
        return f"the file {file_vocab} not found !"

    with open(file_vocab, "r", encoding="utf-8") as f:
        data = f.read()

    vocab = data.split("\n")

    if not os.path.isfile(file_metadata):
        return f"the file {file_metadata} not found !"

    with open(file_metadata, "r", encoding="utf-8") as f:
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
    name_project = project_name + "_pinyin"
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
    name_project = project_name + "_pinyin"
    path_project = os.path.join(path_data, name_project)
    file_metadata = os.path.join(path_project, "metadata.csv")
    if not os.path.isfile(file_metadata):
        return "", None

    data = ""
    with open(file_metadata, "r", encoding="utf-8") as f:
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


def infer(project_name, file_checkpoint, exp_name, ref_text, ref_audio, gen_text, nfe_step):
    global last_checkpoint, last_device, tts_api

    if not os.path.isfile(file_checkpoint):
        return None

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
        return f.name


with gr.Blocks() as app:
    with gr.Row():
        project_name = gr.Textbox(label="project name", value="my_speak")
        bt_create = gr.Button("create new project")

    bt_create.click(fn=create_data_project, inputs=[project_name])

    with gr.Tabs():
        with gr.TabItem("transcribe Data"):
            ch_manual = gr.Checkbox(label="user", value=False)

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
                inputs=[project_name, audio_speaker, txt_lang, ch_manual],
                outputs=[txt_info_transcribe],
            )
            ch_manual.change(fn=check_user, inputs=[ch_manual], outputs=[audio_speaker, mark_info_transcribe])

            random_sample_transcribe = gr.Button("random sample")

            with gr.Row():
                random_text_transcribe = gr.Text(label="Text")
                random_audio_transcribe = gr.Audio(label="Audio", type="filepath")

            random_sample_transcribe.click(
                fn=get_random_sample_transcribe,
                inputs=[project_name],
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

            bt_prepare = bt_create = gr.Button("prepare")
            txt_info_prepare = gr.Text(label="info", value="")
            bt_prepare.click(fn=create_metadata, inputs=[project_name], outputs=[txt_info_prepare])

            random_sample_prepare = gr.Button("random sample")

            with gr.Row():
                random_text_prepare = gr.Text(label="Pinyin")
                random_audio_prepare = gr.Audio(label="Audio", type="filepath")

            random_sample_prepare.click(
                fn=get_random_sample_prepare, inputs=[project_name], outputs=[random_text_prepare, random_audio_prepare]
            )

        with gr.TabItem("train Data"):
            with gr.Row():
                bt_calculate = bt_create = gr.Button("Auto Settings")
                ch_finetune = bt_create = gr.Checkbox(label="finetune", value=True)
                lb_samples = gr.Label(label="samples")
                batch_size_type = gr.Radio(label="Batch Size Type", choices=["frame", "sample"], value="frame")

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
                start_button = gr.Button("Start Training")
                stop_button = gr.Button("Stop Training", interactive=False)

            txt_info_train = gr.Text(label="info", value="")
            start_button.click(
                fn=start_training,
                inputs=[
                    project_name,
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
                ],
                outputs=[txt_info_train, start_button, stop_button],
            )
            stop_button.click(fn=stop_training, outputs=[txt_info_train, start_button, stop_button])
            bt_calculate.click(
                fn=calculate_train,
                inputs=[
                    project_name,
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
                ],
            )

        with gr.TabItem("reduse checkpoint"):
            txt_path_checkpoint = gr.Text(label="path checkpoint :")
            txt_path_checkpoint_small = gr.Text(label="path output :")
            txt_info_reduse = gr.Text(label="info", value="")
            reduse_button = gr.Button("reduse")
            reduse_button.click(
                fn=extract_and_save_ema_model,
                inputs=[txt_path_checkpoint, txt_path_checkpoint_small],
                outputs=[txt_info_reduse],
            )

        with gr.TabItem("vocab check experiment"):
            check_button = gr.Button("check vocab")
            txt_info_check = gr.Text(label="info", value="")
            check_button.click(fn=vocab_check, inputs=[project_name], outputs=[txt_info_check])

        with gr.TabItem("test model"):
            exp_name = gr.Radio(label="Model", choices=["F5-TTS", "E2-TTS"], value="F5-TTS")
            nfe_step = gr.Number(label="n_step", value=32)
            file_checkpoint_pt = gr.Textbox(label="Checkpoint", value="")

            random_sample_infer = gr.Button("random sample")

            ref_text = gr.Textbox(label="ref text")
            ref_audio = gr.Audio(label="audio ref", type="filepath")
            gen_text = gr.Textbox(label="gen text")
            random_sample_infer.click(
                fn=get_random_sample_infer, inputs=[project_name], outputs=[ref_text, gen_text, ref_audio]
            )
            check_button_infer = gr.Button("infer")
            gen_audio = gr.Audio(label="audio gen", type="filepath")

            check_button_infer.click(
                fn=infer,
                inputs=[project_name, file_checkpoint_pt, exp_name, ref_text, ref_audio, gen_text, nfe_step],
                outputs=[gen_audio],
            )


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
