import gradio as gr
from f5_tts.gradio_app import app

with gr.Blocks() as other_app:
    app.render()

other_app.launch()
