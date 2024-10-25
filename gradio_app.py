import sys
import subprocess
import os
import warnings

warnings.warn("Warning: The Gradio app has moved to `src/f5_tts/infer/infer_gradio.py`. Please update your scripts accordingly.")

subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', 'f5_tts', 'infer', 'infer_gradio.py')])
