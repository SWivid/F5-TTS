import re
import torch
import torchaudio
import gradio as gr
import numpy as np
import tempfile
from einops import rearrange
from vocos import Vocos
from pydub import AudioSegment, silence
from model import CFM, UNetT, DiT, MMDiT
from cached_path import cached_path
from model.utils import (
    load_checkpoint,
    get_tokenizer,
    convert_char_to_pinyin,
    save_spectrogram,
)
from transformers import pipeline
import click
import soundfile as sf

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

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

print(f"Using {device} device")

pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3-turbo",
    torch_dtype=torch.float16,
    device=device,
)
vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")

# --------------------- Settings -------------------- #

target_sample_rate = 24000
n_mel_channels = 100
hop_length = 256
target_rms = 0.1
nfe_step = 32  # 16, 32
cfg_strength = 2.0
ode_method = "euler"
sway_sampling_coef = -1.0
speed = 1.0
fix_duration = None


def load_model(repo_name, exp_name, model_cls, model_cfg, ckpt_step):
    ckpt_path = str(cached_path(f"hf://SWivid/{repo_name}/{exp_name}/model_{ckpt_step}.safetensors"))
    # ckpt_path = f"ckpts/{exp_name}/model_{ckpt_step}.pt"  # .pt | .safetensors
    vocab_char_map, vocab_size = get_tokenizer("Emilia_ZH_EN", "pinyin")
    model = CFM(
        transformer=model_cls(
            **model_cfg, text_num_embeds=vocab_size, mel_dim=n_mel_channels
        ),
        mel_spec_kwargs=dict(
            target_sample_rate=target_sample_rate,
            n_mel_channels=n_mel_channels,
            hop_length=hop_length,
        ),
        odeint_kwargs=dict(
            method=ode_method,
        ),
        vocab_char_map=vocab_char_map,
    ).to(device)

    model = load_checkpoint(model, ckpt_path, device, use_ema = True)

    return model


# load models
F5TTS_model_cfg = dict(
    dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4
)
E2TTS_model_cfg = dict(dim=1024, depth=24, heads=16, ff_mult=4)

F5TTS_ema_model = load_model(
    "F5-TTS", "F5TTS_Base", DiT, F5TTS_model_cfg, 1200000
)
E2TTS_ema_model = load_model(
    "E2-TTS", "E2TTS_Base", UNetT, E2TTS_model_cfg, 1200000
)

def chunk_text(text, max_chars=135):
    """
    Splits the input text into chunks, each with a maximum number of characters.

    Args:
        text (str): The text to be split.
        max_chars (int): The maximum number of characters per chunk.

    Returns:
        List[str]: A list of text chunks.
    """
    chunks = []
    current_chunk = ""
    # Split the text into sentences based on punctuation followed by whitespace
    sentences = re.split(r'(?<=[;:,.!?])\s+|(?<=[；：，。！？])', text)

    for sentence in sentences:
        if len(current_chunk.encode('utf-8')) + len(sentence.encode('utf-8')) <= max_chars:
            current_chunk += sentence + " " if sentence and len(sentence[-1].encode('utf-8')) == 1 else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " " if sentence and len(sentence[-1].encode('utf-8')) == 1 else sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

@gpu_decorator
def infer_batch(ref_audio, ref_text, gen_text_batches, exp_name, remove_silence, cross_fade_duration=0.15, progress=gr.Progress()):
    if exp_name == "F5-TTS":
        ema_model = F5TTS_ema_model
    elif exp_name == "E2-TTS":
        ema_model = E2TTS_ema_model

    audio, sr = ref_audio
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    rms = torch.sqrt(torch.mean(torch.square(audio)))
    if rms < target_rms:
        audio = audio * target_rms / rms
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
        audio = resampler(audio)
    audio = audio.to(device)

    generated_waves = []
    spectrograms = []

    for i, gen_text in enumerate(progress.tqdm(gen_text_batches)):
        # Prepare the text
        if len(ref_text[-1].encode('utf-8')) == 1:
            ref_text = ref_text + " "
        text_list = [ref_text + gen_text]
        final_text_list = convert_char_to_pinyin(text_list)

        # Calculate duration
        ref_audio_len = audio.shape[-1] // hop_length
        zh_pause_punc = r"。，、；：？！"
        ref_text_len = len(ref_text.encode('utf-8')) + 3 * len(re.findall(zh_pause_punc, ref_text))
        gen_text_len = len(gen_text.encode('utf-8')) + 3 * len(re.findall(zh_pause_punc, gen_text))
        duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / speed)

        # inference
        with torch.inference_mode():
            generated, _ = ema_model.sample(
                cond=audio,
                text=final_text_list,
                duration=duration,
                steps=nfe_step,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef,
            )

        generated = generated[:, ref_audio_len:, :]
        generated_mel_spec = rearrange(generated, "1 n d -> 1 d n")
        generated_wave = vocos.decode(generated_mel_spec.cpu())
        if rms < target_rms:
            generated_wave = generated_wave * rms / target_rms

        # wav -> numpy
        generated_wave = generated_wave.squeeze().cpu().numpy()
        
        generated_waves.append(generated_wave)
        spectrograms.append(generated_mel_spec[0].cpu().numpy())

    # Combine all generated waves with cross-fading
    if cross_fade_duration <= 0:
        # Simply concatenate
        final_wave = np.concatenate(generated_waves)
    else:
        final_wave = generated_waves[0]
        for i in range(1, len(generated_waves)):
            prev_wave = final_wave
            next_wave = generated_waves[i]

            # Calculate cross-fade samples, ensuring it does not exceed wave lengths
            cross_fade_samples = int(cross_fade_duration * target_sample_rate)
            cross_fade_samples = min(cross_fade_samples, len(prev_wave), len(next_wave))

            if cross_fade_samples <= 0:
                # No overlap possible, concatenate
                final_wave = np.concatenate([prev_wave, next_wave])
                continue

            # Overlapping parts
            prev_overlap = prev_wave[-cross_fade_samples:]
            next_overlap = next_wave[:cross_fade_samples]

            # Fade out and fade in
            fade_out = np.linspace(1, 0, cross_fade_samples)
            fade_in = np.linspace(0, 1, cross_fade_samples)

            # Cross-faded overlap
            cross_faded_overlap = prev_overlap * fade_out + next_overlap * fade_in

            # Combine
            new_wave = np.concatenate([
                prev_wave[:-cross_fade_samples],
                cross_faded_overlap,
                next_wave[cross_fade_samples:]
            ])

            final_wave = new_wave

    # Remove silence
    if remove_silence:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            sf.write(f.name, final_wave, target_sample_rate)
            aseg = AudioSegment.from_file(f.name)
            non_silent_segs = silence.split_on_silence(aseg, min_silence_len=1000, silence_thresh=-50, keep_silence=500)
            non_silent_wave = AudioSegment.silent(duration=0)
            for non_silent_seg in non_silent_segs:
                non_silent_wave += non_silent_seg
            aseg = non_silent_wave
            aseg.export(f.name, format="wav")
            final_wave, _ = torchaudio.load(f.name)
        final_wave = final_wave.squeeze().cpu().numpy()

    # Create a combined spectrogram
    combined_spectrogram = np.concatenate(spectrograms, axis=1)
    
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_spectrogram:
        spectrogram_path = tmp_spectrogram.name
        save_spectrogram(combined_spectrogram, spectrogram_path)

    return (target_sample_rate, final_wave), spectrogram_path

@gpu_decorator
def infer(ref_audio_orig, ref_text, gen_text, exp_name, remove_silence, cross_fade_duration=0.15):

    print(gen_text)

    gr.Info("Converting audio...")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        aseg = AudioSegment.from_file(ref_audio_orig)

        non_silent_segs = silence.split_on_silence(
            aseg, min_silence_len=1000, silence_thresh=-50, keep_silence=1000
        )
        non_silent_wave = AudioSegment.silent(duration=0)
        for non_silent_seg in non_silent_segs:
            non_silent_wave += non_silent_seg
        aseg = non_silent_wave

        audio_duration = len(aseg)
        if audio_duration > 15000:
            gr.Warning("Audio is over 15s, clipping to only first 15s.")
            aseg = aseg[:15000]
        aseg.export(f.name, format="wav")
        ref_audio = f.name

    if not ref_text.strip():
        gr.Info("No reference text provided, transcribing reference audio...")
        ref_text = pipe(
            ref_audio,
            chunk_length_s=30,
            batch_size=128,
            generate_kwargs={"task": "transcribe"},
            return_timestamps=False,
        )["text"].strip()
        gr.Info("Finished transcription")
    else:
        gr.Info("Using custom reference text...")

    # Add the functionality to ensure it ends with ". "
    if not ref_text.endswith(". "):
        if ref_text.endswith("."):
            ref_text += " "
        else:
            ref_text += ". "

    audio, sr = torchaudio.load(ref_audio)

    # Use the new chunk_text function to split gen_text
    max_chars = int(len(ref_text.encode('utf-8')) / (audio.shape[-1] / sr) * (25 - audio.shape[-1] / sr))
    gen_text_batches = chunk_text(gen_text, max_chars=max_chars)
    print('ref_text', ref_text)
    for i, batch_text in enumerate(gen_text_batches):
        print(f'gen_text {i}', batch_text)
    
    gr.Info(f"Generating audio using {exp_name} in {len(gen_text_batches)} batches")
    return infer_batch((audio, sr), ref_text, gen_text_batches, exp_name, remove_silence, cross_fade_duration)


@gpu_decorator
def generate_podcast(script, speaker1_name, ref_audio1, ref_text1, speaker2_name, ref_audio2, ref_text2, exp_name, remove_silence):
    # Split the script into speaker blocks
    speaker_pattern = re.compile(f"^({re.escape(speaker1_name)}|{re.escape(speaker2_name)}):", re.MULTILINE)
    speaker_blocks = speaker_pattern.split(script)[1:]  # Skip the first empty element
    
    generated_audio_segments = []
    
    for i in range(0, len(speaker_blocks), 2):
        speaker = speaker_blocks[i]
        text = speaker_blocks[i+1].strip()
        
        # Determine which speaker is talking
        if speaker == speaker1_name:
            ref_audio = ref_audio1
            ref_text = ref_text1
        elif speaker == speaker2_name:
            ref_audio = ref_audio2
            ref_text = ref_text2
        else:
            continue  # Skip if the speaker is neither speaker1 nor speaker2
        
        # Generate audio for this block
        audio, _ = infer(ref_audio, ref_text, text, exp_name, remove_silence)
        
        # Convert the generated audio to a numpy array
        sr, audio_data = audio
        
        # Save the audio data as a WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            sf.write(temp_file.name, audio_data, sr)
            audio_segment = AudioSegment.from_wav(temp_file.name)
        
        generated_audio_segments.append(audio_segment)
        
        # Add a short pause between speakers
        pause = AudioSegment.silent(duration=500)  # 500ms pause
        generated_audio_segments.append(pause)
    
    # Concatenate all audio segments
    final_podcast = sum(generated_audio_segments)
    
    # Export the final podcast
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        podcast_path = temp_file.name
        final_podcast.export(podcast_path, format="wav")
    
    return podcast_path

def parse_speechtypes_text(gen_text):
    # Pattern to find (Emotion)
    pattern = r'\((.*?)\)'

    # Split the text by the pattern
    tokens = re.split(pattern, gen_text)

    segments = []

    current_emotion = 'Regular'

    for i in range(len(tokens)):
        if i % 2 == 0:
            # This is text
            text = tokens[i].strip()
            if text:
                segments.append({'emotion': current_emotion, 'text': text})
        else:
            # This is emotion
            emotion = tokens[i].strip()
            current_emotion = emotion

    return segments

def update_speed(new_speed):
    global speed
    speed = new_speed
    return f"Speed set to: {speed}"

with gr.Blocks() as app_credits:
    gr.Markdown("""
# Credits

* [mrfakename](https://github.com/fakerybakery) for the original [online demo](https://huggingface.co/spaces/mrfakename/E2-F5-TTS)
* [RootingInLoad](https://github.com/RootingInLoad) for the podcast generation
* [jpgallegoar](https://github.com/jpgallegoar) for multiple speech-type generation
""")
with gr.Blocks() as app_tts:
    gr.Markdown("# Batched TTS")
    ref_audio_input = gr.Audio(label="Reference Audio", type="filepath")
    gen_text_input = gr.Textbox(label="Text to Generate", lines=10)
    model_choice = gr.Radio(
        choices=["F5-TTS", "E2-TTS"], label="Choose TTS Model", value="F5-TTS"
    )
    generate_btn = gr.Button("Synthesize", variant="primary")
    with gr.Accordion("Advanced Settings", open=False):
        ref_text_input = gr.Textbox(
            label="Reference Text",
            info="Leave blank to automatically transcribe the reference audio. If you enter text it will override automatic transcription.",
            lines=2,
        )
        remove_silence = gr.Checkbox(
            label="Remove Silences",
            info="The model tends to produce silences, especially on longer audio. We can manually remove silences if needed. Note that this is an experimental feature and may produce strange results. This will also increase generation time.",
            value=False,
        )
        speed_slider = gr.Slider(
            label="Speed",
            minimum=0.3,
            maximum=2.0,
            value=speed,
            step=0.1,
            info="Adjust the speed of the audio.",
        )
        cross_fade_duration_slider = gr.Slider(
            label="Cross-Fade Duration (s)",
            minimum=0.0,
            maximum=1.0,
            value=0.15,
            step=0.01,
            info="Set the duration of the cross-fade between audio clips.",
        )
    speed_slider.change(update_speed, inputs=speed_slider)

    audio_output = gr.Audio(label="Synthesized Audio")
    spectrogram_output = gr.Image(label="Spectrogram")

    generate_btn.click(
        infer,
        inputs=[
            ref_audio_input,
            ref_text_input,
            gen_text_input,
            model_choice,
            remove_silence,
            cross_fade_duration_slider,
        ],
        outputs=[audio_output, spectrogram_output],
    )
    
with gr.Blocks() as app_podcast:
    gr.Markdown("# Podcast Generation")
    speaker1_name = gr.Textbox(label="Speaker 1 Name")
    ref_audio_input1 = gr.Audio(label="Reference Audio (Speaker 1)", type="filepath")
    ref_text_input1 = gr.Textbox(label="Reference Text (Speaker 1)", lines=2)
    
    speaker2_name = gr.Textbox(label="Speaker 2 Name")
    ref_audio_input2 = gr.Audio(label="Reference Audio (Speaker 2)", type="filepath")
    ref_text_input2 = gr.Textbox(label="Reference Text (Speaker 2)", lines=2)
    
    script_input = gr.Textbox(label="Podcast Script", lines=10, 
                                placeholder="Enter the script with speaker names at the start of each block, e.g.:\nSean: How did you start studying...\n\nMeghan: I came to my interest in technology...\nIt was a long journey...\n\nSean: That's fascinating. Can you elaborate...")
    
    podcast_model_choice = gr.Radio(
        choices=["F5-TTS", "E2-TTS"], label="Choose TTS Model", value="F5-TTS"
    )
    podcast_remove_silence = gr.Checkbox(
        label="Remove Silences",
        value=True,
    )
    generate_podcast_btn = gr.Button("Generate Podcast", variant="primary")
    podcast_output = gr.Audio(label="Generated Podcast")

    def podcast_generation(script, speaker1, ref_audio1, ref_text1, speaker2, ref_audio2, ref_text2, model, remove_silence):
        return generate_podcast(script, speaker1, ref_audio1, ref_text1, speaker2, ref_audio2, ref_text2, model, remove_silence)

    generate_podcast_btn.click(
        podcast_generation,
        inputs=[
            script_input,
            speaker1_name,
            ref_audio_input1,
            ref_text_input1,
            speaker2_name,
            ref_audio_input2,
            ref_text_input2,
            podcast_model_choice,
            podcast_remove_silence,
        ],
        outputs=podcast_output,
    )

def parse_emotional_text(gen_text):
    # Pattern to find (Emotion)
    pattern = r'\((.*?)\)'

    # Split the text by the pattern
    tokens = re.split(pattern, gen_text)

    segments = []

    current_emotion = 'Regular'

    for i in range(len(tokens)):
        if i % 2 == 0:
            # This is text
            text = tokens[i].strip()
            if text:
                segments.append({'emotion': current_emotion, 'text': text})
        else:
            # This is emotion
            emotion = tokens[i].strip()
            current_emotion = emotion

    return segments

with gr.Blocks() as app_emotional:
    # New section for emotional generation
    gr.Markdown(
        """
    # Multiple Speech-Type Generation

    This section allows you to upload different audio clips for each speech type. 'Regular' emotion is mandatory. You can add additional speech types by clicking the "Add Speech Type" button. Enter your text in the format shown below, and the system will generate speech using the appropriate emotions. If unspecified, the model will use the regular speech type. The current speech type will be used until the next speech type is specified.

    **Example Input:**

    (Regular) Hello, I'd like to order a sandwich please. (Surprised) What do you mean you're out of bread? (Sad) I really wanted a sandwich though... (Angry) You know what, darn you and your little shop, you suck! (Whisper) I'll just go back home and cry now. (Shouting) Why me?!
    """
    )

    gr.Markdown("Upload different audio clips for each speech type. 'Regular' emotion is mandatory. You can add additional speech types by clicking the 'Add Speech Type' button.")

    # Regular speech type (mandatory)
    with gr.Row():
        regular_name = gr.Textbox(value='Regular', label='Speech Type Name', interactive=False)
        regular_audio = gr.Audio(label='Regular Reference Audio', type='filepath')
        regular_ref_text = gr.Textbox(label='Reference Text (Regular)', lines=2)

    # Additional speech types (up to 9 more)
    max_speech_types = 10
    speech_type_names = []
    speech_type_audios = []
    speech_type_ref_texts = []
    speech_type_delete_btns = []

    for i in range(max_speech_types - 1):
        with gr.Row():
            name_input = gr.Textbox(label='Speech Type Name', visible=False)
            audio_input = gr.Audio(label='Reference Audio', type='filepath', visible=False)
            ref_text_input = gr.Textbox(label='Reference Text', lines=2, visible=False)
            delete_btn = gr.Button("Delete", variant="secondary", visible=False)
        speech_type_names.append(name_input)
        speech_type_audios.append(audio_input)
        speech_type_ref_texts.append(ref_text_input)
        speech_type_delete_btns.append(delete_btn)

    # Button to add speech type
    add_speech_type_btn = gr.Button("Add Speech Type")

    # Keep track of current number of speech types
    speech_type_count = gr.State(value=0)

    # Function to add a speech type
    def add_speech_type_fn(speech_type_count):
        if speech_type_count < max_speech_types - 1:
            speech_type_count += 1
            # Prepare updates for the components
            name_updates = []
            audio_updates = []
            ref_text_updates = []
            delete_btn_updates = []
            for i in range(max_speech_types - 1):
                if i < speech_type_count:
                    name_updates.append(gr.update(visible=True))
                    audio_updates.append(gr.update(visible=True))
                    ref_text_updates.append(gr.update(visible=True))
                    delete_btn_updates.append(gr.update(visible=True))
                else:
                    name_updates.append(gr.update())
                    audio_updates.append(gr.update())
                    ref_text_updates.append(gr.update())
                    delete_btn_updates.append(gr.update())
        else:
            # Optionally, show a warning
            # gr.Warning("Maximum number of speech types reached.")
            name_updates = [gr.update() for _ in range(max_speech_types - 1)]
            audio_updates = [gr.update() for _ in range(max_speech_types - 1)]
            ref_text_updates = [gr.update() for _ in range(max_speech_types - 1)]
            delete_btn_updates = [gr.update() for _ in range(max_speech_types - 1)]
        return [speech_type_count] + name_updates + audio_updates + ref_text_updates + delete_btn_updates

    add_speech_type_btn.click(
        add_speech_type_fn,
        inputs=speech_type_count,
        outputs=[speech_type_count] + speech_type_names + speech_type_audios + speech_type_ref_texts + speech_type_delete_btns
    )

    # Function to delete a speech type
    def make_delete_speech_type_fn(index):
        def delete_speech_type_fn(speech_type_count):
            # Prepare updates
            name_updates = []
            audio_updates = []
            ref_text_updates = []
            delete_btn_updates = []

            for i in range(max_speech_types - 1):
                if i == index:
                    name_updates.append(gr.update(visible=False, value=''))
                    audio_updates.append(gr.update(visible=False, value=None))
                    ref_text_updates.append(gr.update(visible=False, value=''))
                    delete_btn_updates.append(gr.update(visible=False))
                else:
                    name_updates.append(gr.update())
                    audio_updates.append(gr.update())
                    ref_text_updates.append(gr.update())
                    delete_btn_updates.append(gr.update())

            speech_type_count = max(0, speech_type_count - 1)

            return [speech_type_count] + name_updates + audio_updates + ref_text_updates + delete_btn_updates

        return delete_speech_type_fn

    for i, delete_btn in enumerate(speech_type_delete_btns):
        delete_fn = make_delete_speech_type_fn(i)
        delete_btn.click(
            delete_fn,
            inputs=speech_type_count,
            outputs=[speech_type_count] + speech_type_names + speech_type_audios + speech_type_ref_texts + speech_type_delete_btns
        )

    # Text input for the prompt
    gen_text_input_emotional = gr.Textbox(label="Text to Generate", lines=10)

    # Model choice
    model_choice_emotional = gr.Radio(
        choices=["F5-TTS", "E2-TTS"], label="Choose TTS Model", value="F5-TTS"
    )

    with gr.Accordion("Advanced Settings", open=False):
        remove_silence_emotional = gr.Checkbox(
            label="Remove Silences",
            value=True,
        )

    # Generate button
    generate_emotional_btn = gr.Button("Generate Emotional Speech", variant="primary")

    # Output audio
    audio_output_emotional = gr.Audio(label="Synthesized Audio")
    @gpu_decorator
    def generate_emotional_speech(
        regular_audio,
        regular_ref_text,
        gen_text,
        *args,
    ):
        num_additional_speech_types = max_speech_types - 1
        speech_type_names_list = args[:num_additional_speech_types]
        speech_type_audios_list = args[num_additional_speech_types:2 * num_additional_speech_types]
        speech_type_ref_texts_list = args[2 * num_additional_speech_types:3 * num_additional_speech_types]
        model_choice = args[3 * num_additional_speech_types]
        remove_silence = args[3 * num_additional_speech_types + 1]

        # Collect the speech types and their audios into a dict
        speech_types = {'Regular': {'audio': regular_audio, 'ref_text': regular_ref_text}}

        for name_input, audio_input, ref_text_input in zip(speech_type_names_list, speech_type_audios_list, speech_type_ref_texts_list):
            if name_input and audio_input:
                speech_types[name_input] = {'audio': audio_input, 'ref_text': ref_text_input}

        # Parse the gen_text into segments
        segments = parse_speechtypes_text(gen_text)

        # For each segment, generate speech
        generated_audio_segments = []
        current_emotion = 'Regular'

        for segment in segments:
            emotion = segment['emotion']
            text = segment['text']

            if emotion in speech_types:
                current_emotion = emotion
            else:
                # If emotion not available, default to Regular
                current_emotion = 'Regular'

            ref_audio = speech_types[current_emotion]['audio']
            ref_text = speech_types[current_emotion].get('ref_text', '')

            # Generate speech for this segment
            audio, _ = infer(ref_audio, ref_text, text, model_choice, remove_silence, 0)
            sr, audio_data = audio

            generated_audio_segments.append(audio_data)

        # Concatenate all audio segments
        if generated_audio_segments:
            final_audio_data = np.concatenate(generated_audio_segments)
            return (sr, final_audio_data)
        else:
            gr.Warning("No audio generated.")
            return None

    generate_emotional_btn.click(
        generate_emotional_speech,
        inputs=[
            regular_audio,
            regular_ref_text,
            gen_text_input_emotional,
        ] + speech_type_names + speech_type_audios + speech_type_ref_texts + [
            model_choice_emotional,
            remove_silence_emotional,
        ],
        outputs=audio_output_emotional,
    )

    # Validation function to disable Generate button if speech types are missing
    def validate_speech_types(
        gen_text,
        regular_name,
        *args
    ):
        num_additional_speech_types = max_speech_types - 1
        speech_type_names_list = args[:num_additional_speech_types]

        # Collect the speech types names
        speech_types_available = set()
        if regular_name:
            speech_types_available.add(regular_name)
        for name_input in speech_type_names_list:
            if name_input:
                speech_types_available.add(name_input)

        # Parse the gen_text to get the speech types used
        segments = parse_emotional_text(gen_text)
        speech_types_in_text = set(segment['emotion'] for segment in segments)

        # Check if all speech types in text are available
        missing_speech_types = speech_types_in_text - speech_types_available

        if missing_speech_types:
            # Disable the generate button
            return gr.update(interactive=False)
        else:
            # Enable the generate button
            return gr.update(interactive=True)

    gen_text_input_emotional.change(
        validate_speech_types,
        inputs=[gen_text_input_emotional, regular_name] + speech_type_names,
        outputs=generate_emotional_btn
    )
with gr.Blocks() as app:
    gr.Markdown(
        """
# E2/F5 TTS

This is a local web UI for F5 TTS with advanced batch processing support. This app supports the following TTS models:

* [F5-TTS](https://arxiv.org/abs/2410.06885) (A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching)
* [E2 TTS](https://arxiv.org/abs/2406.18009) (Embarrassingly Easy Fully Non-Autoregressive Zero-Shot TTS)

The checkpoints support English and Chinese.

If you're having issues, try converting your reference audio to WAV or MP3, clipping it to 15s, and shortening your prompt.

**NOTE: Reference text will be automatically transcribed with Whisper if not provided. For best results, keep your reference clips short (<15s). Ensure the audio is fully uploaded before generating.**
"""
    )
    gr.TabbedInterface([app_tts, app_podcast, app_emotional, app_credits], ["TTS", "Podcast", "Multi-Style", "Credits"])

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
    print(f"Starting app...")
    app.queue(api_open=api).launch(
        server_name=host, server_port=port, share=share, show_api=api
    )


if __name__ == "__main__":
    main()