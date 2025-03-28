# Inference

The pretrained model checkpoints can be reached at [🤗 Hugging Face](https://huggingface.co/SWivid/F5-TTS) and [🤖 Model Scope](https://www.modelscope.cn/models/SWivid/F5-TTS_Emilia-ZH-EN), or will be automatically downloaded when running inference scripts.

**More checkpoints with whole community efforts can be found in [SHARED.md](SHARED.md), supporting more languages.**

Currently support **30s for a single** generation, which is the **total length** (same logic if `fix_duration`) including both prompt and output audio. However, `infer_cli` and `infer_gradio` will automatically do chunk generation for longer text. Long reference audio will be **clip short to ~12s**.

To avoid possible inference failures, make sure you have seen through the following instructions.

- Use reference audio <12s and leave proper silence space (e.g. 1s) at the end. Otherwise there is a risk of truncating in the middle of word, leading to suboptimal generation.
- <ins>Uppercased letters</ins> (best with form like K.F.C.) will be uttered letter by letter, and lowercased letters used for common words. 
- Add some spaces (blank: " ") or punctuations (e.g. "," ".") <ins>to explicitly introduce some pauses</ins>.
- If English punctuation marks the end of a sentence, make sure there is a space " " after it. Otherwise not regarded as when chunk.
- <ins>Preprocess numbers</ins> to Chinese letters if you want to have them read in Chinese, otherwise in English.
- If the generation output is blank (pure silence), <ins>check for ffmpeg installation</ins>.
- Try <ins>turn off `use_ema` if using an early-stage</ins> finetuned checkpoint (which goes just few updates).


## Gradio App

Currently supported features:

- Basic TTS with Chunk Inference
- Multi-Style / Multi-Speaker Generation
- Voice Chat powered by Qwen2.5-3B-Instruct
- [Custom inference with more language support](SHARED.md)

The cli command `f5-tts_infer-gradio` equals to `python src/f5_tts/infer/infer_gradio.py`, which launches a Gradio APP (web interface) for inference.

The script will load model checkpoints from Huggingface. You can also manually download files and update the path to `load_model()` in `infer_gradio.py`. Currently only load TTS models first, will load ASR model to do transcription if `ref_text` not provided, will load LLM model if use Voice Chat.

More flags options:

```bash
# Automatically launch the interface in the default web browser
f5-tts_infer-gradio --inbrowser

# Set the root path of the application, if it's not served from the root ("/") of the domain
# For example, if the application is served at "https://example.com/myapp"
f5-tts_infer-gradio --root_path "/myapp"
```

Could also be used as a component for larger application:
```python
import gradio as gr
from f5_tts.infer.infer_gradio import app

with gr.Blocks() as main_app:
    gr.Markdown("# This is an example of using F5-TTS within a bigger Gradio app")

    # ... other Gradio components

    app.render()

main_app.launch()
```


## CLI Inference

The cli command `f5-tts_infer-cli` equals to `python src/f5_tts/infer/infer_cli.py`, which is a command line tool for inference.

The script will load model checkpoints from Huggingface. You can also manually download files and use `--ckpt_file` to specify the model you want to load, or directly update in `infer_cli.py`.

For change vocab.txt use `--vocab_file` to provide your `vocab.txt` file.

Basically you can inference with flags:
```bash
# Leave --ref_text "" will have ASR model transcribe (extra GPU memory usage)
f5-tts_infer-cli \
--model F5TTS_v1_Base \
--ref_audio "ref_audio.wav" \
--ref_text "The content, subtitle or transcription of reference audio." \
--gen_text "Some text you want TTS model generate for you."

# Use BigVGAN as vocoder. Currently only support F5TTS_Base. 
f5-tts_infer-cli --model F5TTS_Base --vocoder_name bigvgan --load_vocoder_from_local

# Use custom path checkpoint, e.g.
f5-tts_infer-cli --ckpt_file ckpts/F5TTS_v1_Base/model_1250000.safetensors

# More instructions
f5-tts_infer-cli --help
```

And a `.toml` file would help with more flexible usage.

```bash
f5-tts_infer-cli -c custom.toml
```

For example, you can use `.toml` to pass in variables, refer to `src/f5_tts/infer/examples/basic/basic.toml`:

```toml
# F5TTS_v1_Base | E2TTS_Base
model = "F5TTS_v1_Base"
ref_audio = "infer/examples/basic/basic_ref_en.wav"
# If an empty "", transcribes the reference audio automatically.
ref_text = "Some call me nature, others call me mother nature."
gen_text = "I don't really care what you call me. I've been a silent spectator, watching species evolve, empires rise and fall. But always remember, I am mighty and enduring."
# File with text to generate. Ignores the text above.
gen_file = ""
remove_silence = false
output_dir = "tests"
```

You can also leverage `.toml` file to do multi-style generation, refer to `src/f5_tts/infer/examples/multi/story.toml`.

```toml
# F5TTS_v1_Base | E2TTS_Base
model = "F5TTS_v1_Base"
ref_audio = "infer/examples/multi/main.flac"
# If an empty "", transcribes the reference audio automatically.
ref_text = ""
gen_text = ""
# File with text to generate. Ignores the text above.
gen_file = "infer/examples/multi/story.txt"
remove_silence = true
output_dir = "tests"

[voices.town]
ref_audio = "infer/examples/multi/town.flac"
ref_text = ""

[voices.country]
ref_audio = "infer/examples/multi/country.flac"
ref_text = ""
```
You should mark the voice with `[main]` `[town]` `[country]` whenever you want to change voice, refer to `src/f5_tts/infer/examples/multi/story.txt`.

## Socket Real-time Service

Real-time voice output with chunk stream:

```bash
# Start socket server
python src/f5_tts/socket_server.py

# If PyAudio not installed
sudo apt-get install portaudio19-dev
pip install pyaudio

# Communicate with socket client
python src/f5_tts/socket_client.py
```

## Speech Editing

To test speech editing capabilities, use the following command:

```bash
python src/f5_tts/infer/speech_edit.py
```

