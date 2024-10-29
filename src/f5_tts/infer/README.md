# Inference

The pretrained model checkpoints can be reached at [🤗 Hugging Face](https://huggingface.co/SWivid/F5-TTS) and [🤖 Model Scope](https://www.modelscope.cn/models/SWivid/F5-TTS_Emilia-ZH-EN), or will be automatically downloaded when running inference scripts.

Currently support **30s for a single** generation, which is the **total length** including both prompt and output audio. However, you can provide `infer_cli` and `infer_gradio` with longer text, will automatically do chunk generation. Long reference audio will be **clip short to ~15s**.

To avoid possible inference failures, make sure you have seen through the following instructions.

- Use reference audio <15s and leave some silence (e.g. 1s) at the end. Otherwise there is a risk of truncating in the middle of word, leading to suboptimal generation.
- Uppercased letters will be uttered letter by letter, so use lowercased letters for normal words. 
- Add some spaces (blank: " ") or punctuations (e.g. "," ".") to explicitly introduce some pauses.
- Preprocess numbers to Chinese letters if you want to have them read in Chinese, otherwise in English.


## Gradio App

Currently supported features:

- Basic TTS with Chunk Inference
- Multi-Style / Multi-Speaker Generation
- Voice Chat powered by Qwen2.5-3B-Instruct

The cli command `f5-tts_infer-gradio` equals to `python src/f5_tts/infer/infer_gradio.py`, which launches a Gradio APP (web interface) for inference.

The script will load model checkpoints from Huggingface. You can also manually download files and update the path to `load_model()` in `infer_gradio.py`. Currently only load TTS models first, will load ASR model to do transcription if `ref_text` not provided, will load LLM model if use Voice Chat.

Could also be used as a component for larger application.
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
--model "F5-TTS" \
--ref_audio "ref_audio.wav" \
--ref_text "The content, subtitle or transcription of reference audio." \
--gen_text "Some text you want TTS model generate for you."
```

And a `.toml` file would help with more flexible usage.

```bash
f5-tts_infer-cli -c custom.toml
```

For example, you can use `.toml` to pass in variables, refer to `src/f5_tts/infer/examples/basic/basic.toml`:

```toml
# F5-TTS | E2-TTS
model = "F5-TTS"
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
# F5-TTS | E2-TTS
model = "F5-TTS"
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

## Speech Editing

To test speech editing capabilities, use the following command:

```bash
python src/f5_tts/infer/speech_edit.py
```