# ruff: noqa: E402
# Above allows ruff to ignore E402: module level import not at top of file

import gc
import json
import re
import tempfile
from collections import OrderedDict
from functools import lru_cache
from importlib.resources import files

import click
import gradio as gr
import numpy as np
import soundfile as sf
import torch
import torchaudio
from cached_path import cached_path
from transformers import AutoModelForCausalLM, AutoTokenizer


try:
    import spaces

    USING_SPACES = True
except ImportError:
    USING_SPACES = False


def gpu_decorator(func):
    if USING_SPACES:
        return spaces.GPU(func)
    else:
        return func


from f5_tts.infer.utils_infer import (
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
    save_spectrogram,
)
from f5_tts.model import DiT, UNetT


DEFAULT_TTS_MODEL = "F5-TTS_v1"
tts_model_choice = DEFAULT_TTS_MODEL

DEFAULT_TTS_MODEL_CFG = [
    "hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors",
    "hf://SWivid/F5-TTS/F5TTS_v1_Base/vocab.txt",
    json.dumps(dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)),
]


# load models

vocoder = load_vocoder()


def load_f5tts():
    ckpt_path = str(cached_path(DEFAULT_TTS_MODEL_CFG[0]))
    F5TTS_model_cfg = json.loads(DEFAULT_TTS_MODEL_CFG[2])
    return load_model(DiT, F5TTS_model_cfg, ckpt_path)


def load_e2tts():
    ckpt_path = str(cached_path("hf://SWivid/E2-TTS/E2TTS_Base/model_1200000.safetensors"))
    E2TTS_model_cfg = dict(dim=1024, depth=24, heads=16, ff_mult=4, text_mask_padding=False, pe_attn_head=1)
    return load_model(UNetT, E2TTS_model_cfg, ckpt_path)


def load_custom(ckpt_path: str, vocab_path="", model_cfg=None):
    ckpt_path, vocab_path = ckpt_path.strip(), vocab_path.strip()
    if ckpt_path.startswith("hf://"):
        ckpt_path = str(cached_path(ckpt_path))
    if vocab_path.startswith("hf://"):
        vocab_path = str(cached_path(vocab_path))
    if model_cfg is None:
        model_cfg = json.loads(DEFAULT_TTS_MODEL_CFG[2])
    elif isinstance(model_cfg, str):
        model_cfg = json.loads(model_cfg)
    return load_model(DiT, model_cfg, ckpt_path, vocab_file=vocab_path)


F5TTS_ema_model = load_f5tts()
E2TTS_ema_model = load_e2tts() if USING_SPACES else None
custom_ema_model, pre_custom_path = None, ""

chat_model_state = None
chat_tokenizer_state = None


@gpu_decorator
def chat_model_inference(messages, model, tokenizer):
    """Generate response using Qwen"""
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.95,
    )

    generated_ids = [
        output_ids[len(input_ids) :] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


@gpu_decorator
def load_text_from_file(file):
    if file:
        with open(file, "r", encoding="utf-8") as f:
            text = f.read().strip()
    else:
        text = ""
    return gr.update(value=text)


@lru_cache(maxsize=100)  # NOTE. need to ensure params of infer() hashable
@gpu_decorator
def infer(
    ref_audio_orig,
    ref_text,
    gen_text,
    model,
    remove_silence,
    seed,
    cross_fade_duration=0.15,
    nfe_step=32,
    speed=1,
    show_info=gr.Info,
):
    if not ref_audio_orig:
        gr.Warning("Please provide reference audio.")
        return gr.update(), gr.update(), ref_text

    # Set inference seed
    if seed < 0 or seed > 2**31 - 1:
        gr.Warning("Seed must in range 0 ~ 2147483647. Using random seed instead.")
        seed = np.random.randint(0, 2**31 - 1)
    torch.manual_seed(seed)
    used_seed = seed

    if not gen_text.strip():
        gr.Warning("Please enter text to generate or upload a text file.")
        return gr.update(), gr.update(), ref_text

    ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_orig, ref_text, show_info=show_info)

    if model == DEFAULT_TTS_MODEL:
        ema_model = F5TTS_ema_model
    elif model == "E2-TTS":
        global E2TTS_ema_model
        if E2TTS_ema_model is None:
            show_info("Loading E2-TTS model...")
            E2TTS_ema_model = load_e2tts()
        ema_model = E2TTS_ema_model
    elif isinstance(model, tuple) and model[0] == "Custom":
        assert not USING_SPACES, "Only official checkpoints allowed in Spaces."
        global custom_ema_model, pre_custom_path
        if pre_custom_path != model[1]:
            show_info("Loading Custom TTS model...")
            custom_ema_model = load_custom(model[1], vocab_path=model[2], model_cfg=model[3])
            pre_custom_path = model[1]
        ema_model = custom_ema_model

    final_wave, final_sample_rate, combined_spectrogram = infer_process(
        ref_audio,
        ref_text,
        gen_text,
        ema_model,
        vocoder,
        cross_fade_duration=cross_fade_duration,
        nfe_step=nfe_step,
        speed=speed,
        show_info=show_info,
        progress=gr.Progress(),
    )

    # Remove silence
    if remove_silence:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            sf.write(f.name, final_wave, final_sample_rate)
            remove_silence_for_generated_wav(f.name)
            final_wave, _ = torchaudio.load(f.name)
        final_wave = final_wave.squeeze().cpu().numpy()

    # Save the spectrogram
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_spectrogram:
        spectrogram_path = tmp_spectrogram.name
        save_spectrogram(combined_spectrogram, spectrogram_path)

    return (final_sample_rate, final_wave), spectrogram_path, ref_text, used_seed


with gr.Blocks() as app_credits:
    gr.Markdown("""
# Credits

* [mrfakename](https://github.com/fakerybakery) for the original [online demo](https://huggingface.co/spaces/mrfakename/E2-F5-TTS)
* [RootingInLoad](https://github.com/RootingInLoad) for initial chunk generation and podcast app exploration
* [jpgallegoar](https://github.com/jpgallegoar) for multiple speech-type generation & voice chat
""")
with gr.Blocks() as app_tts:
    gr.Markdown("# Batched TTS")
    ref_audio_input = gr.Audio(label="Reference Audio", type="filepath")
    with gr.Row():
        gen_text_input = gr.Textbox(
            label="Text to Generate",
            lines=10,
            max_lines=40,
            scale=4,
        )
        gen_text_file = gr.File(label="Load Text to Generate from File (.txt)", file_types=[".txt"], scale=1)
    generate_btn = gr.Button("Synthesize", variant="primary")
    with gr.Accordion("Advanced Settings", open=False):
        with gr.Row():
            ref_text_input = gr.Textbox(
                label="Reference Text",
                info="Leave blank to automatically transcribe the reference audio. If you enter text or upload a file, it will override automatic transcription.",
                lines=2,
                scale=4,
            )
            ref_text_file = gr.File(label="Load Reference Text from File (.txt)", file_types=[".txt"], scale=1)
        with gr.Row():
            randomize_seed = gr.Checkbox(
                label="Randomize Seed",
                info="Check to use a random seed for each generation. Uncheck to use the seed specified.",
                value=True,
                scale=3,
            )
            seed_input = gr.Number(show_label=False, value=0, precision=0, scale=1)
            with gr.Column(scale=4):
                remove_silence = gr.Checkbox(
                    label="Remove Silences",
                    info="If undesired long silence(s) produced, turn on to automatically detect and crop.",
                    value=False,
                )
        speed_slider = gr.Slider(
            label="Speed",
            minimum=0.3,
            maximum=2.0,
            value=1.0,
            step=0.1,
            info="Adjust the speed of the audio.",
        )
        nfe_slider = gr.Slider(
            label="NFE Steps",
            minimum=4,
            maximum=64,
            value=32,
            step=2,
            info="Set the number of denoising steps.",
        )
        cross_fade_duration_slider = gr.Slider(
            label="Cross-Fade Duration (s)",
            minimum=0.0,
            maximum=1.0,
            value=0.15,
            step=0.01,
            info="Set the duration of the cross-fade between audio clips.",
        )

    audio_output = gr.Audio(label="Synthesized Audio")
    spectrogram_output = gr.Image(label="Spectrogram")

    @gpu_decorator
    def basic_tts(
        ref_audio_input,
        ref_text_input,
        gen_text_input,
        remove_silence,
        randomize_seed,
        seed_input,
        cross_fade_duration_slider,
        nfe_slider,
        speed_slider,
    ):
        if randomize_seed:
            seed_input = np.random.randint(0, 2**31 - 1)

        audio_out, spectrogram_path, ref_text_out, used_seed = infer(
            ref_audio_input,
            ref_text_input,
            gen_text_input,
            tts_model_choice,
            remove_silence,
            seed=seed_input,
            cross_fade_duration=cross_fade_duration_slider,
            nfe_step=nfe_slider,
            speed=speed_slider,
        )
        return audio_out, spectrogram_path, ref_text_out, used_seed

    gen_text_file.upload(
        load_text_from_file,
        inputs=[gen_text_file],
        outputs=[gen_text_input],
    )

    ref_text_file.upload(
        load_text_from_file,
        inputs=[ref_text_file],
        outputs=[ref_text_input],
    )

    generate_btn.click(
        basic_tts,
        inputs=[
            ref_audio_input,
            ref_text_input,
            gen_text_input,
            remove_silence,
            randomize_seed,
            seed_input,
            cross_fade_duration_slider,
            nfe_slider,
            speed_slider,
        ],
        outputs=[audio_output, spectrogram_output, ref_text_input, seed_input],
    )


def parse_speechtypes_text(gen_text):
    # Pattern to find {str} or {"name": str, "seed": int, "speed": float}
    pattern = r"(\{.*?\})"

    # Split the text by the pattern
    tokens = re.split(pattern, gen_text)

    segments = []

    current_type_dict = {
        "name": "Regular",
        "seed": -1,
        "speed": 1.0,
    }

    for i in range(len(tokens)):
        if i % 2 == 0:
            # This is text
            text = tokens[i].strip()
            if text:
                current_type_dict["text"] = text
                segments.append(current_type_dict)
        else:
            # This is type
            type_str = tokens[i].strip()
            try:  # if type dict
                current_type_dict = json.loads(type_str)
            except json.decoder.JSONDecodeError:
                type_str = type_str[1:-1]  # remove brace {}
                current_type_dict = {"name": type_str, "seed": -1, "speed": 1.0}

    return segments


with gr.Blocks() as app_multistyle:
    # New section for multistyle generation
    gr.Markdown(
        """
    # Multiple Speech-Type Generation

    This section allows you to generate multiple speech types or multiple people's voices. Enter your text in the format shown below, or upload a .txt file with the same format. The system will generate speech using the appropriate type. If unspecified, the model will use the regular speech type. The current speech type will be used until the next speech type is specified.
    """
    )

    with gr.Row():
        gr.Markdown(
            """
            **Example Input:** <br>
            {Regular} Hello, I'd like to order a sandwich please. <br>
            {Surprised} What do you mean you're out of bread? <br>
            {Sad} I really wanted a sandwich though... <br>
            {Angry} You know what, darn you and your little shop! <br>
            {Whisper} I'll just go back home and cry now. <br>
            {Shouting} Why me?!
            """
        )

        gr.Markdown(
            """
            **Example Input 2:** <br>
            {"name": "Speaker1_Happy", "seed": -1, "speed": 1} Hello, I'd like to order a sandwich please. <br>
            {"name": "Speaker2_Regular", "seed": -1, "speed": 1} Sorry, we're out of bread. <br>
            {"name": "Speaker1_Sad", "seed": -1, "speed": 1} I really wanted a sandwich though... <br>
            {"name": "Speaker2_Whisper", "seed": -1, "speed": 1} I'll give you the last one I was hiding.
            """
        )

    gr.Markdown(
        'Upload different audio clips for each speech type. The first speech type is mandatory. You can add additional speech types by clicking the "Add Speech Type" button.'
    )

    # Regular speech type (mandatory)
    with gr.Row(variant="compact") as regular_row:
        with gr.Column(scale=1, min_width=160):
            regular_name = gr.Textbox(value="Regular", label="Speech Type Name")
            regular_insert = gr.Button("Insert Label", variant="secondary")
        with gr.Column(scale=3):
            regular_audio = gr.Audio(label="Regular Reference Audio", type="filepath")
        with gr.Column(scale=3):
            regular_ref_text = gr.Textbox(label="Reference Text (Regular)", lines=4)
            with gr.Row():
                regular_seed_slider = gr.Slider(
                    show_label=False, minimum=-1, maximum=999, value=-1, step=1, info="Seed, -1 for random"
                )
                regular_speed_slider = gr.Slider(
                    show_label=False, minimum=0.3, maximum=2.0, value=1.0, step=0.1, info="Adjust the speed"
                )
        with gr.Column(scale=1, min_width=160):
            regular_ref_text_file = gr.File(label="Load Reference Text from File (.txt)", file_types=[".txt"])

    # Regular speech type (max 100)
    max_speech_types = 100
    speech_type_rows = [regular_row]
    speech_type_names = [regular_name]
    speech_type_audios = [regular_audio]
    speech_type_ref_texts = [regular_ref_text]
    speech_type_ref_text_files = [regular_ref_text_file]
    speech_type_seeds = [regular_seed_slider]
    speech_type_speeds = [regular_speed_slider]
    speech_type_delete_btns = [None]
    speech_type_insert_btns = [regular_insert]

    # Additional speech types (99 more)
    for i in range(max_speech_types - 1):
        with gr.Row(variant="compact", visible=False) as row:
            with gr.Column(scale=1, min_width=160):
                name_input = gr.Textbox(label="Speech Type Name")
                insert_btn = gr.Button("Insert Label", variant="secondary")
                delete_btn = gr.Button("Delete Type", variant="stop")
            with gr.Column(scale=3):
                audio_input = gr.Audio(label="Reference Audio", type="filepath")
            with gr.Column(scale=3):
                ref_text_input = gr.Textbox(label="Reference Text", lines=4)
                with gr.Row():
                    seed_input = gr.Slider(
                        show_label=False, minimum=-1, maximum=999, value=-1, step=1, info="Seed. -1 for random"
                    )
                    speed_input = gr.Slider(
                        show_label=False, minimum=0.3, maximum=2.0, value=1.0, step=0.1, info="Adjust the speed"
                    )
            with gr.Column(scale=1, min_width=160):
                ref_text_file_input = gr.File(label="Load Reference Text from File (.txt)", file_types=[".txt"])
        speech_type_rows.append(row)
        speech_type_names.append(name_input)
        speech_type_audios.append(audio_input)
        speech_type_ref_texts.append(ref_text_input)
        speech_type_ref_text_files.append(ref_text_file_input)
        speech_type_seeds.append(seed_input)
        speech_type_speeds.append(speed_input)
        speech_type_delete_btns.append(delete_btn)
        speech_type_insert_btns.append(insert_btn)

    # Global logic for all speech types
    for i in range(max_speech_types):
        speech_type_audios[i].clear(
            lambda: [None, None],
            None,
            [speech_type_ref_texts[i], speech_type_ref_text_files[i]],
        )
        speech_type_ref_text_files[i].upload(
            load_text_from_file,
            inputs=[speech_type_ref_text_files[i]],
            outputs=[speech_type_ref_texts[i]],
        )

    # Button to add speech type
    add_speech_type_btn = gr.Button("Add Speech Type")

    # Keep track of autoincrement of speech types, no roll back
    speech_type_count = 1

    # Function to add a speech type
    def add_speech_type_fn():
        row_updates = [gr.update() for _ in range(max_speech_types)]
        global speech_type_count
        if speech_type_count < max_speech_types:
            row_updates[speech_type_count] = gr.update(visible=True)
            speech_type_count += 1
        else:
            gr.Warning("Exhausted maximum number of speech types. Consider restart the app.")
        return row_updates

    add_speech_type_btn.click(add_speech_type_fn, outputs=speech_type_rows)

    # Function to delete a speech type
    def delete_speech_type_fn():
        return gr.update(visible=False), None, None, None, None

    # Update delete button clicks and ref text file changes
    for i in range(1, len(speech_type_delete_btns)):
        speech_type_delete_btns[i].click(
            delete_speech_type_fn,
            outputs=[
                speech_type_rows[i],
                speech_type_names[i],
                speech_type_audios[i],
                speech_type_ref_texts[i],
                speech_type_ref_text_files[i],
            ],
        )

    # Text input for the prompt
    with gr.Row():
        gen_text_input_multistyle = gr.Textbox(
            label="Text to Generate",
            lines=10,
            max_lines=40,
            scale=4,
            placeholder="Enter the script with speaker names (or emotion types) at the start of each block, e.g.:\n\n{Regular} Hello, I'd like to order a sandwich please.\n{Surprised} What do you mean you're out of bread?\n{Sad} I really wanted a sandwich though...\n{Angry} You know what, darn you and your little shop!\n{Whisper} I'll just go back home and cry now.\n{Shouting} Why me?!",
        )
        gen_text_file_multistyle = gr.File(label="Load Text to Generate from File (.txt)", file_types=[".txt"], scale=1)

    def make_insert_speech_type_fn(index):
        def insert_speech_type_fn(current_text, speech_type_name, speech_type_seed, speech_type_speed):
            current_text = current_text or ""
            if not speech_type_name:
                gr.Warning("Please enter speech type name before insert.")
                return current_text
            speech_type_dict = {
                "name": speech_type_name,
                "seed": speech_type_seed,
                "speed": speech_type_speed,
            }
            updated_text = current_text + json.dumps(speech_type_dict) + " "
            return updated_text

        return insert_speech_type_fn

    for i, insert_btn in enumerate(speech_type_insert_btns):
        insert_fn = make_insert_speech_type_fn(i)
        insert_btn.click(
            insert_fn,
            inputs=[gen_text_input_multistyle, speech_type_names[i], speech_type_seeds[i], speech_type_speeds[i]],
            outputs=gen_text_input_multistyle,
        )

    with gr.Accordion("Advanced Settings", open=True):
        with gr.Row():
            with gr.Column():
                show_cherrypick_multistyle = gr.Checkbox(
                    label="Show Cherry-pick Interface",
                    info="Turn on to show interface, picking seeds from previous generations.",
                    value=False,
                )
            with gr.Column():
                remove_silence_multistyle = gr.Checkbox(
                    label="Remove Silences",
                    info="Turn on to automatically detect and crop long silences.",
                    value=True,
                )

    # Generate button
    generate_multistyle_btn = gr.Button("Generate Multi-Style Speech", variant="primary")

    # Output audio
    audio_output_multistyle = gr.Audio(label="Synthesized Audio")

    # Used seed gallery
    cherrypick_interface_multistyle = gr.Textbox(
        label="Cherry-pick Interface",
        lines=10,
        max_lines=40,
        show_copy_button=True,
        interactive=False,
        visible=False,
    )

    # Logic control to show/hide the cherrypick interface
    show_cherrypick_multistyle.change(
        lambda is_visible: gr.update(visible=is_visible),
        show_cherrypick_multistyle,
        cherrypick_interface_multistyle,
    )

    # Function to load text to generate from file
    gen_text_file_multistyle.upload(
        load_text_from_file,
        inputs=[gen_text_file_multistyle],
        outputs=[gen_text_input_multistyle],
    )

    @gpu_decorator
    def generate_multistyle_speech(
        gen_text,
        *args,
    ):
        speech_type_names_list = args[:max_speech_types]
        speech_type_audios_list = args[max_speech_types : 2 * max_speech_types]
        speech_type_ref_texts_list = args[2 * max_speech_types : 3 * max_speech_types]
        remove_silence = args[3 * max_speech_types]
        # Collect the speech types and their audios into a dict
        speech_types = OrderedDict()

        ref_text_idx = 0
        for name_input, audio_input, ref_text_input in zip(
            speech_type_names_list, speech_type_audios_list, speech_type_ref_texts_list
        ):
            if name_input and audio_input:
                speech_types[name_input] = {"audio": audio_input, "ref_text": ref_text_input}
            else:
                speech_types[f"@{ref_text_idx}@"] = {"audio": "", "ref_text": ""}
            ref_text_idx += 1

        # Parse the gen_text into segments
        segments = parse_speechtypes_text(gen_text)

        # For each segment, generate speech
        generated_audio_segments = []
        current_type_name = "Regular"
        inference_meta_data = ""

        for segment in segments:
            name = segment["name"]
            seed_input = segment["seed"]
            speed = segment["speed"]
            text = segment["text"]

            if name in speech_types:
                current_type_name = name
            else:
                gr.Warning(f"Type {name} is not available, will use Regular as default.")
                current_type_name = "Regular"

            try:
                ref_audio = speech_types[current_type_name]["audio"]
            except KeyError:
                gr.Warning(f"Please provide reference audio for type {current_type_name}.")
                return [None] + [speech_types[name]["ref_text"] for name in speech_types] + [None]
            ref_text = speech_types[current_type_name].get("ref_text", "")

            if seed_input == -1:
                seed_input = np.random.randint(0, 2**31 - 1)

            # Generate or retrieve speech for this segment
            audio_out, _, ref_text_out, used_seed = infer(
                ref_audio,
                ref_text,
                text,
                tts_model_choice,
                remove_silence,
                seed=seed_input,
                cross_fade_duration=0,
                speed=speed,
                show_info=print,  # no pull to top when generating
            )
            sr, audio_data = audio_out

            generated_audio_segments.append(audio_data)
            speech_types[current_type_name]["ref_text"] = ref_text_out
            inference_meta_data += json.dumps(dict(name=name, seed=used_seed, speed=speed)) + f" {text}\n"

        # Concatenate all audio segments
        if generated_audio_segments:
            final_audio_data = np.concatenate(generated_audio_segments)
            return (
                [(sr, final_audio_data)]
                + [speech_types[name]["ref_text"] for name in speech_types]
                + [inference_meta_data]
            )
        else:
            gr.Warning("No audio generated.")
            return [None] + [speech_types[name]["ref_text"] for name in speech_types] + [None]

    generate_multistyle_btn.click(
        generate_multistyle_speech,
        inputs=[
            gen_text_input_multistyle,
        ]
        + speech_type_names
        + speech_type_audios
        + speech_type_ref_texts
        + [
            remove_silence_multistyle,
        ],
        outputs=[audio_output_multistyle] + speech_type_ref_texts + [cherrypick_interface_multistyle],
    )

    # Validation function to disable Generate button if speech types are missing
    def validate_speech_types(gen_text, regular_name, *args):
        speech_type_names_list = args

        # Collect the speech types names
        speech_types_available = set()
        if regular_name:
            speech_types_available.add(regular_name)
        for name_input in speech_type_names_list:
            if name_input:
                speech_types_available.add(name_input)

        # Parse the gen_text to get the speech types used
        segments = parse_speechtypes_text(gen_text)
        speech_types_in_text = set(segment["name"] for segment in segments)

        # Check if all speech types in text are available
        missing_speech_types = speech_types_in_text - speech_types_available

        if missing_speech_types:
            # Disable the generate button
            return gr.update(interactive=False)
        else:
            # Enable the generate button
            return gr.update(interactive=True)

    gen_text_input_multistyle.change(
        validate_speech_types,
        inputs=[gen_text_input_multistyle, regular_name] + speech_type_names,
        outputs=generate_multistyle_btn,
    )


with gr.Blocks() as app_chat:
    gr.Markdown(
        """
# Voice Chat
Have a conversation with an AI using your reference voice!
1. Upload a reference audio clip and optionally its transcript (via text or .txt file).
2. Load the chat model.
3. Record your message through your microphone or type it.
4. The AI will respond using the reference voice.
"""
    )

    chat_model_name_list = [
        "Qwen/Qwen2.5-3B-Instruct",
        "microsoft/Phi-4-mini-instruct",
    ]

    @gpu_decorator
    def load_chat_model(chat_model_name):
        show_info = gr.Info
        global chat_model_state, chat_tokenizer_state
        if chat_model_state is not None:
            chat_model_state = None
            chat_tokenizer_state = None
            gc.collect()
            torch.cuda.empty_cache()

        show_info(f"Loading chat model: {chat_model_name}")
        chat_model_state = AutoModelForCausalLM.from_pretrained(chat_model_name, torch_dtype="auto", device_map="auto")
        chat_tokenizer_state = AutoTokenizer.from_pretrained(chat_model_name)
        show_info(f"Chat model {chat_model_name} loaded successfully!")

        return gr.update(visible=False), gr.update(visible=True)

    if USING_SPACES:
        load_chat_model(chat_model_name_list[0])

    chat_model_name_input = gr.Dropdown(
        choices=chat_model_name_list,
        value=chat_model_name_list[0],
        label="Chat Model Name",
        info="Enter the name of a HuggingFace chat model",
        allow_custom_value=not USING_SPACES,
    )
    load_chat_model_btn = gr.Button("Load Chat Model", variant="primary", visible=not USING_SPACES)
    chat_interface_container = gr.Column(visible=USING_SPACES)

    chat_model_name_input.change(
        lambda: gr.update(visible=True),
        None,
        load_chat_model_btn,
        show_progress="hidden",
    )
    load_chat_model_btn.click(
        load_chat_model, inputs=[chat_model_name_input], outputs=[load_chat_model_btn, chat_interface_container]
    )

    with chat_interface_container:
        with gr.Row():
            with gr.Column():
                ref_audio_chat = gr.Audio(label="Reference Audio", type="filepath")
            with gr.Column():
                with gr.Accordion("Advanced Settings", open=False):
                    with gr.Row():
                        ref_text_chat = gr.Textbox(
                            label="Reference Text",
                            info="Optional: Leave blank to auto-transcribe",
                            lines=2,
                            scale=3,
                        )
                        ref_text_file_chat = gr.File(
                            label="Load Reference Text from File (.txt)", file_types=[".txt"], scale=1
                        )
                    with gr.Row():
                        randomize_seed_chat = gr.Checkbox(
                            label="Randomize Seed",
                            value=True,
                            info="Uncheck to use the seed specified.",
                            scale=3,
                        )
                        seed_input_chat = gr.Number(show_label=False, value=0, precision=0, scale=1)
                    remove_silence_chat = gr.Checkbox(
                        label="Remove Silences",
                        value=True,
                    )
                    system_prompt_chat = gr.Textbox(
                        label="System Prompt",
                        value="You are not an AI assistant, you are whoever the user says you are. You must stay in character. Keep your responses concise since they will be spoken out loud.",
                        lines=2,
                    )

        chatbot_interface = gr.Chatbot(label="Conversation", type="messages")

        with gr.Row():
            with gr.Column():
                audio_input_chat = gr.Microphone(
                    label="Speak your message",
                    type="filepath",
                )
                audio_output_chat = gr.Audio(autoplay=True)
            with gr.Column():
                text_input_chat = gr.Textbox(
                    label="Type your message",
                    lines=1,
                )
                send_btn_chat = gr.Button("Send Message")
                clear_btn_chat = gr.Button("Clear Conversation")

        # Modify process_audio_input to generate user input
        @gpu_decorator
        def process_audio_input(conv_state, audio_path, text):
            """Handle audio or text input from user"""

            if not audio_path and not text.strip():
                return conv_state

            if audio_path:
                text = preprocess_ref_audio_text(audio_path, text)[1]
            if not text.strip():
                return conv_state

            conv_state.append({"role": "user", "content": text})
            return conv_state

        # Use model and tokenizer from state to get text response
        @gpu_decorator
        def generate_text_response(conv_state, system_prompt):
            """Generate text response from AI"""

            system_prompt_state = [{"role": "system", "content": system_prompt}]
            response = chat_model_inference(system_prompt_state + conv_state, chat_model_state, chat_tokenizer_state)

            conv_state.append({"role": "assistant", "content": response})
            return conv_state

        @gpu_decorator
        def generate_audio_response(conv_state, ref_audio, ref_text, remove_silence, randomize_seed, seed_input):
            """Generate TTS audio for AI response"""
            if not conv_state or not ref_audio:
                return None, ref_text, seed_input

            last_ai_response = conv_state[-1]["content"]
            if not last_ai_response or conv_state[-1]["role"] != "assistant":
                return None, ref_text, seed_input

            if randomize_seed:
                seed_input = np.random.randint(0, 2**31 - 1)

            audio_result, _, ref_text_out, used_seed = infer(
                ref_audio,
                ref_text,
                last_ai_response,
                tts_model_choice,
                remove_silence,
                seed=seed_input,
                cross_fade_duration=0.15,
                speed=1.0,
                show_info=print,  # show_info=print no pull to top when generating
            )
            return audio_result, ref_text_out, used_seed

        def clear_conversation():
            """Reset the conversation"""
            return [], None

        ref_text_file_chat.upload(
            load_text_from_file,
            inputs=[ref_text_file_chat],
            outputs=[ref_text_chat],
        )

        for user_operation in [audio_input_chat.stop_recording, text_input_chat.submit, send_btn_chat.click]:
            user_operation(
                process_audio_input,
                inputs=[chatbot_interface, audio_input_chat, text_input_chat],
                outputs=[chatbot_interface],
            ).then(
                generate_text_response,
                inputs=[chatbot_interface, system_prompt_chat],
                outputs=[chatbot_interface],
            ).then(
                generate_audio_response,
                inputs=[
                    chatbot_interface,
                    ref_audio_chat,
                    ref_text_chat,
                    remove_silence_chat,
                    randomize_seed_chat,
                    seed_input_chat,
                ],
                outputs=[audio_output_chat, ref_text_chat, seed_input_chat],
            ).then(
                lambda: [None, None],
                None,
                [audio_input_chat, text_input_chat],
            )

        # Handle clear button or system prompt change and reset conversation
        for user_operation in [clear_btn_chat.click, system_prompt_chat.change, chatbot_interface.clear]:
            user_operation(
                clear_conversation,
                outputs=[chatbot_interface, audio_output_chat],
            )


with gr.Blocks() as app:
    gr.Markdown(
        f"""
# E2/F5 TTS

This is {"a local web UI for [F5 TTS](https://github.com/SWivid/F5-TTS)" if not USING_SPACES else "an online demo for [F5-TTS](https://github.com/SWivid/F5-TTS)"} with advanced batch processing support. This app supports the following TTS models:

* [F5-TTS](https://arxiv.org/abs/2410.06885) (A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching)
* [E2 TTS](https://arxiv.org/abs/2406.18009) (Embarrassingly Easy Fully Non-Autoregressive Zero-Shot TTS)

The checkpoints currently support English and Chinese.

If you're having issues, try converting your reference audio to WAV or MP3, clipping it to 12s with  ✂  in the bottom right corner (otherwise might have non-optimal auto-trimmed result).

**NOTE: Reference text will be automatically transcribed with Whisper if not provided. For best results, keep your reference clips short (<12s). Ensure the audio is fully uploaded before generating.**
"""
    )

    last_used_custom = files("f5_tts").joinpath("infer/.cache/last_used_custom_model_info_v1.txt")

    def load_last_used_custom():
        try:
            custom = []
            with open(last_used_custom, "r", encoding="utf-8") as f:
                for line in f:
                    custom.append(line.strip())
            return custom
        except FileNotFoundError:
            last_used_custom.parent.mkdir(parents=True, exist_ok=True)
            return DEFAULT_TTS_MODEL_CFG

    def switch_tts_model(new_choice):
        global tts_model_choice
        if new_choice == "Custom":  # override in case webpage is refreshed
            custom_ckpt_path, custom_vocab_path, custom_model_cfg = load_last_used_custom()
            tts_model_choice = ("Custom", custom_ckpt_path, custom_vocab_path, custom_model_cfg)
            return (
                gr.update(visible=True, value=custom_ckpt_path),
                gr.update(visible=True, value=custom_vocab_path),
                gr.update(visible=True, value=custom_model_cfg),
            )
        else:
            tts_model_choice = new_choice
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

    def set_custom_model(custom_ckpt_path, custom_vocab_path, custom_model_cfg):
        global tts_model_choice
        tts_model_choice = ("Custom", custom_ckpt_path, custom_vocab_path, custom_model_cfg)
        with open(last_used_custom, "w", encoding="utf-8") as f:
            f.write(custom_ckpt_path + "\n" + custom_vocab_path + "\n" + custom_model_cfg + "\n")

    with gr.Row():
        if not USING_SPACES:
            choose_tts_model = gr.Radio(
                choices=[DEFAULT_TTS_MODEL, "E2-TTS", "Custom"], label="Choose TTS Model", value=DEFAULT_TTS_MODEL
            )
        else:
            choose_tts_model = gr.Radio(
                choices=[DEFAULT_TTS_MODEL, "E2-TTS"], label="Choose TTS Model", value=DEFAULT_TTS_MODEL
            )
        custom_ckpt_path = gr.Dropdown(
            choices=[DEFAULT_TTS_MODEL_CFG[0]],
            value=load_last_used_custom()[0],
            allow_custom_value=True,
            label="Model: local_path | hf://user_id/repo_id/model_ckpt",
            visible=False,
        )
        custom_vocab_path = gr.Dropdown(
            choices=[DEFAULT_TTS_MODEL_CFG[1]],
            value=load_last_used_custom()[1],
            allow_custom_value=True,
            label="Vocab: local_path | hf://user_id/repo_id/vocab_file",
            visible=False,
        )
        custom_model_cfg = gr.Dropdown(
            choices=[
                DEFAULT_TTS_MODEL_CFG[2],
                json.dumps(
                    dict(
                        dim=1024,
                        depth=22,
                        heads=16,
                        ff_mult=2,
                        text_dim=512,
                        text_mask_padding=False,
                        conv_layers=4,
                        pe_attn_head=1,
                    )
                ),
                json.dumps(
                    dict(
                        dim=768,
                        depth=18,
                        heads=12,
                        ff_mult=2,
                        text_dim=512,
                        text_mask_padding=False,
                        conv_layers=4,
                        pe_attn_head=1,
                    )
                ),
            ],
            value=load_last_used_custom()[2],
            allow_custom_value=True,
            label="Config: in a dictionary form",
            visible=False,
        )

    choose_tts_model.change(
        switch_tts_model,
        inputs=[choose_tts_model],
        outputs=[custom_ckpt_path, custom_vocab_path, custom_model_cfg],
        show_progress="hidden",
    )
    custom_ckpt_path.change(
        set_custom_model,
        inputs=[custom_ckpt_path, custom_vocab_path, custom_model_cfg],
        show_progress="hidden",
    )
    custom_vocab_path.change(
        set_custom_model,
        inputs=[custom_ckpt_path, custom_vocab_path, custom_model_cfg],
        show_progress="hidden",
    )
    custom_model_cfg.change(
        set_custom_model,
        inputs=[custom_ckpt_path, custom_vocab_path, custom_model_cfg],
        show_progress="hidden",
    )

    gr.TabbedInterface(
        [app_tts, app_multistyle, app_chat, app_credits],
        ["Basic-TTS", "Multi-Speech", "Voice-Chat", "Credits"],
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
@click.option(
    "--root_path",
    "-r",
    default=None,
    type=str,
    help='The root path (or "mount point") of the application, if it\'s not served from the root ("/") of the domain. Often used when the application is behind a reverse proxy that forwards requests to the application, e.g. set "/myapp" or full URL for application served at "https://example.com/myapp".',
)
@click.option(
    "--inbrowser",
    "-i",
    is_flag=True,
    default=False,
    help="Automatically launch the interface in the default web browser",
)
def main(port, host, share, api, root_path, inbrowser):
    global app
    print("Starting app...")
    app.queue(api_open=api).launch(
        server_name=host,
        server_port=port,
        share=share,
        show_api=api,
        root_path=root_path,
        inbrowser=inbrowser,
    )


if __name__ == "__main__":
    if not USING_SPACES:
        main()
    else:
        app.queue().launch()
