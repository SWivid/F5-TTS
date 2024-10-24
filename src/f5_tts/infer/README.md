## Inference

The pretrained model checkpoints can be reached at [🤗 Hugging Face](https://huggingface.co/SWivid/F5-TTS) and [🤖 Model Scope](https://www.modelscope.cn/models/SWivid/F5-TTS_Emilia-ZH-EN), or will be automatically downloaded when running inference scripts.

Currently support **30s for a single** generation, which is the **total length** including both prompt and output audio. However, you can leverage `infer_cli` and `infer_gradio` for longer text, will automatically do chunk generation. Long reference audio will be clip short to ~15s.

To avoid possible inference failures, make sure you have seen through the following instructions.

- Uppercased letters will be uttered letter by letter, so use lowercased letters for normal words. 
- Add some spaces (blank: " ") or punctuations (e.g. "," ".") to explicitly introduce some pauses.
- Preprocess numbers to Chinese letters if you want to have them read in Chinese, otherwise in English.

# TODO 👇 ...

### CLI Inference

It is possible to use cli `f5-tts_infer-cli` for following commands.

Either you can specify everything in `inference-cli.toml` or override with flags. Leave `--ref_text ""` will have ASR model transcribe the reference audio automatically (use extra GPU memory). If encounter network error, consider use local ckpt, just set `ckpt_file` in `inference-cli.py`

for change model use `--ckpt_file` to specify the model you want to load,  
for change vocab.txt use `--vocab_file` to provide your vocab.txt file.

```bash
# switch to the main directory
cd f5_tts

python inference-cli.py \
--model "F5-TTS" \
--ref_audio "tests/ref_audio/test_en_1_ref_short.wav" \
--ref_text "Some call me nature, others call me mother nature." \
--gen_text "I don't really care what you call me. I've been a silent spectator, watching species evolve, empires rise and fall. But always remember, I am mighty and enduring. Respect me and I'll nurture you; ignore me and you shall face the consequences."

python inference-cli.py \
--model "E2-TTS" \
--ref_audio "tests/ref_audio/test_zh_1_ref_short.wav" \
--ref_text "对，这就是我，万人敬仰的太乙真人。" \
--gen_text "突然，身边一阵笑声。我看着他们，意气风发地挺直了胸膛，甩了甩那稍显肉感的双臂，轻笑道，我身上的肉，是为了掩饰我爆棚的魅力，否则，岂不吓坏了你们呢？"

# Multi voice
# https://github.com/SWivid/F5-TTS/pull/146#issue-2595207852
python inference-cli.py -c samples/story.toml
```

### Gradio App
Currently supported features:
- Chunk inference
- Podcast Generation
- Multiple Speech-Type Generation
- Voice Chat powered by Qwen2.5-3B-Instruct

It is possible to use cli `f5-tts_infer-gradio` for following commands.

You can launch a Gradio app (web interface) to launch a GUI for inference (will load ckpt from Huggingface, you may also use local file in `gradio_app.py`). Currently load ASR model, F5-TTS and E2 TTS all in once, thus use more GPU memory than `inference-cli`.

```bash
python f5_tts/gradio_app.py
```

You can specify the port/host:

```bash
python f5_tts/gradio_app.py --port 7860 --host 0.0.0.0
```

Or launch a share link:

```bash
python f5_tts/gradio_app.py --share
```

```python
import gradio as gr
from f5_tts.gradio_app import app

with gr.Blocks() as main_app:
    gr.Markdown("# This is an example of using F5-TTS within a bigger Gradio app")

    # ... other Gradio components

    app.render()

main_app.launch()
```

### Speech Editing

To test speech editing capabilities, use the following command.

```bash
python f5_tts/speech_edit.py
```