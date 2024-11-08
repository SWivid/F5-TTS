# ruff: noqa: E402
# Above allows ruff to ignore E402: module level import not at top of file

import re
import tempfile

import click
import gradio as gr
import numpy as np
import soundfile as sf
import torchaudio
from cached_path import cached_path
from transformers import AutoModelForCausalLM, AutoTokenizer
from num2words import num2words

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


from f5_tts.model import DiT, UNetT
from f5_tts.infer.utils_infer import (
    load_vocoder,
    load_model,
    preprocess_ref_audio_text,
    infer_process,
    remove_silence_for_generated_wav,
    save_spectrogram,
)

vocoder = load_vocoder()


# load models
F5TTS_model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
F5TTS_ema_model = load_model(
    DiT, F5TTS_model_cfg, str(cached_path("hf://jpgallegoar/F5-Spanish/model_1200000.safetensors"))
)

chat_model_state = None
chat_tokenizer_state = None


@gpu_decorator
def generate_response(messages, model, tokenizer):
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

def traducir_numero_a_texto(texto):
    texto_separado = re.sub(r'([A-Za-z])(\d)', r'\1 \2', texto)
    texto_separado = re.sub(r'(\d)([A-Za-z])', r'\1 \2', texto_separado)
    
    def reemplazar_numero(match):
        numero = match.group()
        return num2words(int(numero), lang='es')

    texto_traducido = re.sub(r'\b\d+\b', reemplazar_numero, texto_separado)

    return texto_traducido

@gpu_decorator
def infer(
    ref_audio_orig, ref_text, gen_text, model, remove_silence, cross_fade_duration=0.15, speed=1, show_info=gr.Info
):
    ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_orig, ref_text, show_info=show_info)

    ema_model = F5TTS_ema_model

    if not gen_text.startswith(" "):
        gen_text = " " + gen_text
    if not gen_text.endswith(". "):
        gen_text += ". "

    gen_text = gen_text.lower()
    gen_text = traducir_numero_a_texto(gen_text)

    final_wave, final_sample_rate, combined_spectrogram = infer_process(
        ref_audio,
        ref_text,
        gen_text,
        ema_model,
        vocoder,
        cross_fade_duration=cross_fade_duration,
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

    return (final_sample_rate, final_wave), spectrogram_path


with gr.Blocks() as app_credits:
    gr.Markdown("""
# Créditos

* [mrfakename](https://github.com/fakerybakery) por el [demo online original](https://huggingface.co/spaces/mrfakename/E2-F5-TTS)
* [RootingInLoad](https://github.com/RootingInLoad) por la generación inicial de fragmentos y exploración de la aplicación de podcast
* [jpgallegoar](https://github.com/jpgallegoar) por la generación de múltiples tipos de habla, chat de voz y afinación en español
""")


with gr.Blocks() as app_tts:
    gr.Markdown("# TTS por Lotes")
    ref_audio_input = gr.Audio(label="Audio de Referencia", type="filepath")
    gen_text_input = gr.Textbox(label="Texto para Generar", lines=10)
    model_choice = gr.Radio(choices=["F5-TTS"], label="Seleccionar Modelo TTS", value="F5-TTS")
    generate_btn = gr.Button("Sintetizar", variant="primary")
    with gr.Accordion("Configuraciones Avanzadas", open=False):
        ref_text_input = gr.Textbox(
            label="Texto de Referencia",
            info="Deja en blanco para transcribir automáticamente el audio de referencia. Si ingresas texto, sobrescribirá la transcripción automática.",
            lines=2,
        )
        remove_silence = gr.Checkbox(
            label="Eliminar Silencios",
            info="El modelo tiende a producir silencios, especialmente en audios más largos. Podemos eliminar manualmente los silencios si es necesario. Ten en cuenta que esta es una característica experimental y puede producir resultados extraños. Esto también aumentará el tiempo de generación.",
            value=False,
        )
        speed_slider = gr.Slider(
            label="Velocidad",
            minimum=0.3,
            maximum=2.0,
            value=1.0,
            step=0.1,
            info="Ajusta la velocidad del audio.",
        )
        cross_fade_duration_slider = gr.Slider(
            label="Duración del Cross-Fade (s)",
            minimum=0.0,
            maximum=1.0,
            value=0.15,
            step=0.01,
            info="Establece la duración del cross-fade entre clips de audio.",
        )

    audio_output = gr.Audio(label="Audio Sintetizado")
    spectrogram_output = gr.Image(label="Espectrograma")

    generate_btn.click(
        infer,
        inputs=[
            ref_audio_input,
            ref_text_input,
            gen_text_input,
            model_choice,
            remove_silence,
            cross_fade_duration_slider,
            speed_slider,
        ],
        outputs=[audio_output, spectrogram_output],
    )


def parse_speechtypes_text(gen_text):
    # Pattern to find {speechtype}
    pattern = r"\{(.*?)\}"

    # Split the text by the pattern
    tokens = re.split(pattern, gen_text)

    segments = []

    current_style = "Regular"

    for i in range(len(tokens)):
        if i % 2 == 0:
            # This is text
            text = tokens[i].strip()
            if text:
                segments.append({"style": current_style, "text": text})
        else:
            # This is style
            style = tokens[i].strip()
            current_style = style

    return segments


with gr.Blocks() as app_multistyle:
    # New section for multistyle generation
    gr.Markdown(
        """
    # Generación de Múltiples Tipos de Habla

    Esta sección te permite generar múltiples tipos de habla o las voces de múltiples personas. Ingresa tu texto en el formato mostrado a continuación, y el sistema generará el habla utilizando el tipo apropiado. Si no se especifica, el modelo utilizará el tipo de habla regular. El tipo de habla actual se usará hasta que se especifique el siguiente tipo de habla.
    """
    )

    with gr.Row():
        gr.Markdown(
            """
            **Entrada de Ejemplo:**                                                                      
            {Regular} Hola, me gustaría pedir un sándwich, por favor.                                                         
            {Sorprendido} ¿Qué quieres decir con que no tienen pan?                                                                      
            {Triste} Realmente quería un sándwich...                                                              
            {Enojado} ¡Sabes qué, maldición a ti y a tu pequeña tienda!                                                                       
            {Susurro} Solo volveré a casa y lloraré ahora.                                                                           
            {Gritando} ¿Por qué yo?!                                                                         
            """
        )

        gr.Markdown(
            """
            **Entrada de Ejemplo 2:**                                                                                
            {Speaker1_Feliz} Hola, me gustaría pedir un sándwich, por favor.                                                            
            {Speaker2_Regular} Lo siento, nos hemos quedado sin pan.                                                                                
            {Speaker1_Triste} Realmente quería un sándwich...                                                                             
            {Speaker2_Susurro} Te daré el último que estaba escondiendo.                                                                     
            """
        )

    gr.Markdown(
        "Sube diferentes clips de audio para cada tipo de habla. El primer tipo de habla es obligatorio. Puedes agregar tipos de habla adicionales haciendo clic en el botón 'Agregar Tipo de Habla'."
    )

    # Regular speech type (mandatory)
    with gr.Row():
        with gr.Column():
            regular_name = gr.Textbox(value="Regular", label="Nombre del Tipo de Habla")
            regular_insert = gr.Button("Insertar", variant="secondary")
        regular_audio = gr.Audio(label="Audio de Referencia Regular", type="filepath")
        regular_ref_text = gr.Textbox(label="Texto de Referencia (Regular)", lines=2)

    # Additional speech types (up to 99 more)
    max_speech_types = 100
    speech_type_rows = []
    speech_type_names = [regular_name]
    speech_type_audios = []
    speech_type_ref_texts = []
    speech_type_delete_btns = []
    speech_type_insert_btns = []
    speech_type_insert_btns.append(regular_insert)

    for i in range(max_speech_types - 1):
        with gr.Row(visible=False) as row:
            with gr.Column():
                name_input = gr.Textbox(label="Nombre del Tipo de Habla")
                delete_btn = gr.Button("Eliminar", variant="secondary")
                insert_btn = gr.Button("Insertar", variant="secondary")
            audio_input = gr.Audio(label="Audio de Referencia", type="filepath")
            ref_text_input = gr.Textbox(label="Texto de Referencia", lines=2)
        speech_type_rows.append(row)
        speech_type_names.append(name_input)
        speech_type_audios.append(audio_input)
        speech_type_ref_texts.append(ref_text_input)
        speech_type_delete_btns.append(delete_btn)
        speech_type_insert_btns.append(insert_btn)

    # Button to add speech type
    add_speech_type_btn = gr.Button("Agregar Tipo de Habla")

    # Keep track of current number of speech types
    speech_type_count = gr.State(value=0)

    # Function to add a speech type
    def add_speech_type_fn(speech_type_count):
        if speech_type_count < max_speech_types - 1:
            speech_type_count += 1
            # Prepare updates for the rows
            row_updates = []
            for i in range(max_speech_types - 1):
                if i < speech_type_count:
                    row_updates.append(gr.update(visible=True))
                else:
                    row_updates.append(gr.update())
        else:
            # Optionally, show a warning
            row_updates = [gr.update() for _ in range(max_speech_types - 1)]
        return [speech_type_count] + row_updates

    add_speech_type_btn.click(
        add_speech_type_fn, inputs=speech_type_count, outputs=[speech_type_count] + speech_type_rows
    )

    # Function to delete a speech type
    def make_delete_speech_type_fn(index):
        def delete_speech_type_fn(speech_type_count):
            # Prepare updates
            row_updates = []

            for i in range(max_speech_types - 1):
                if i == index:
                    row_updates.append(gr.update(visible=False))
                else:
                    row_updates.append(gr.update())

            speech_type_count = max(0, speech_type_count - 1)

            return [speech_type_count] + row_updates

        return delete_speech_type_fn

    # Update delete button clicks
    for i, delete_btn in enumerate(speech_type_delete_btns):
        delete_fn = make_delete_speech_type_fn(i)
        delete_btn.click(delete_fn, inputs=speech_type_count, outputs=[speech_type_count] + speech_type_rows)

    # Text input for the prompt
    gen_text_input_multistyle = gr.Textbox(
        label="Texto para Generar",
        lines=10,
        placeholder="Ingresa el guion con los nombres de los hablantes (o tipos de emociones) al inicio de cada bloque, por ejemplo:\n\n{Regular} Hola, me gustaría pedir un sándwich, por favor.\n{Sorprendido} ¿Qué quieres decir con que no tienen pan?\n{Triste} Realmente quería un sándwich...\n{Enojado} ¡Sabes qué, maldición a ti y a tu pequeña tienda!\n{Susurro} Solo volveré a casa y lloraré ahora.\n{Gritando} ¿Por qué yo?!",
    )

    def make_insert_speech_type_fn(index):
        def insert_speech_type_fn(current_text, speech_type_name):
            current_text = current_text or ""
            speech_type_name = speech_type_name or "Ninguno"
            updated_text = current_text + f"{{{speech_type_name}}} "
            return gr.update(value=updated_text)

        return insert_speech_type_fn

    for i, insert_btn in enumerate(speech_type_insert_btns):
        insert_fn = make_insert_speech_type_fn(i)
        insert_btn.click(
            insert_fn,
            inputs=[gen_text_input_multistyle, speech_type_names[i]],
            outputs=gen_text_input_multistyle,
        )

    # Model choice
    model_choice_multistyle = gr.Radio(choices=["F5-TTS"], label="Seleccionar Modelo TTS", value="F5-TTS")

    with gr.Accordion("Configuraciones Avanzadas", open=False):
        remove_silence_multistyle = gr.Checkbox(
            label="Eliminar Silencios",
            value=False,
        )

    # Generate button
    generate_multistyle_btn = gr.Button("Generar Habla Multi-Estilo", variant="primary")

    # Output audio
    audio_output_multistyle = gr.Audio(label="Audio Sintetizado")

    @gpu_decorator
    def generate_multistyle_speech(
        regular_audio,
        regular_ref_text,
        gen_text,
        *args,
    ):
        num_additional_speech_types = max_speech_types - 1
        speech_type_names_list = args[:num_additional_speech_types]
        speech_type_audios_list = args[num_additional_speech_types : 2 * num_additional_speech_types]
        speech_type_ref_texts_list = args[2 * num_additional_speech_types : 3 * num_additional_speech_types]
        model_choice = args[3 * num_additional_speech_types]
        remove_silence = args[3 * num_additional_speech_types + 1]

        # Collect the speech types and their audios into a dict
        speech_types = {"Regular": {"audio": regular_audio, "ref_text": regular_ref_text}}

        for name_input, audio_input, ref_text_input in zip(
            speech_type_names_list, speech_type_audios_list, speech_type_ref_texts_list
        ):
            if name_input and audio_input:
                speech_types[name_input] = {"audio": audio_input, "ref_text": ref_text_input}

        # Parse the gen_text into segments
        segments = parse_speechtypes_text(gen_text)

        # For each segment, generate speech
        generated_audio_segments = []
        current_style = "Regular"

        for segment in segments:
            style = segment["style"]
            text = segment["text"]

            if style in speech_types:
                current_style = style
            else:
                # If style not available, default to Regular
                current_style = "Regular"

            ref_audio = speech_types[current_style]["audio"]
            ref_text = speech_types[current_style].get("ref_text", "")

            # Generate speech for this segment
            audio, _ = infer(
                ref_audio, ref_text, text, model_choice, remove_silence, 0, show_info=print
            )  # show_info=print no pull to top when generating
            sr, audio_data = audio

            generated_audio_segments.append(audio_data)

        # Concatenate all audio segments
        if generated_audio_segments:
            final_audio_data = np.concatenate(generated_audio_segments)
            return (sr, final_audio_data)
        else:
            gr.Warning("No se generó ningún audio.")
            return None

    generate_multistyle_btn.click(
        generate_multistyle_speech,
        inputs=[
            regular_audio,
            regular_ref_text,
            gen_text_input_multistyle,
        ]
        + speech_type_names
        + speech_type_audios
        + speech_type_ref_texts
        + [
            model_choice_multistyle,
            remove_silence_multistyle,
        ],
        outputs=audio_output_multistyle,
    )

    # Validation function to disable Generate button if speech types are missing
    def validate_speech_types(gen_text, regular_name, *args):
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
        segments = parse_speechtypes_text(gen_text)
        speech_types_in_text = set(segment["style"] for segment in segments)

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
# Chat de Voz
¡Mantén una conversación con una IA usando tu voz de referencia! 
1. Sube un clip de audio de referencia y opcionalmente su transcripción.
2. Carga el modelo de chat.
3. Graba tu mensaje a través de tu micrófono.
4. La IA responderá usando la voz de referencia.
"""
    )

    if not USING_SPACES:
        load_chat_model_btn = gr.Button("Cargar Modelo de Chat", variant="primary")

        chat_interface_container = gr.Column(visible=False)

        @gpu_decorator
        def load_chat_model():
            global chat_model_state, chat_tokenizer_state
            if chat_model_state is None:
                show_info = gr.Info
                show_info("Cargando modelo de chat...")
                model_name = "Qwen/Qwen2.5-3B-Instruct"
                chat_model_state = AutoModelForCausalLM.from_pretrained(
                    model_name, torch_dtype="auto", device_map="auto"
                )
                chat_tokenizer_state = AutoTokenizer.from_pretrained(model_name)
                show_info("Modelo de chat cargado.")

            return gr.update(visible=False), gr.update(visible=True)

        load_chat_model_btn.click(load_chat_model, outputs=[load_chat_model_btn, chat_interface_container])

    else:
        chat_interface_container = gr.Column()

        if chat_model_state is None:
            model_name = "Qwen/Qwen2.5-3B-Instruct"
            chat_model_state = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
            chat_tokenizer_state = AutoTokenizer.from_pretrained(model_name)

    with chat_interface_container:
        with gr.Row():
            with gr.Column():
                ref_audio_chat = gr.Audio(label="Audio de Referencia", type="filepath")
            with gr.Column():
                with gr.Accordion("Configuraciones Avanzadas", open=False):
                    model_choice_chat = gr.Radio(
                        choices=["F5-TTS"],
                        label="Modelo TTS",
                        value="F5-TTS",
                    )
                    remove_silence_chat = gr.Checkbox(
                        label="Eliminar Silencios",
                        value=True,
                    )
                    ref_text_chat = gr.Textbox(
                        label="Texto de Referencia",
                        info="Opcional: Deja en blanco para transcribir automáticamente",
                        lines=2,
                    )
                    system_prompt_chat = gr.Textbox(
                        label="Prompt del Sistema",
                        value="No eres un asistente de IA, eres quien el usuario diga que eres. Debes mantenerte en personaje. Mantén tus respuestas concisas ya que serán habladas en voz alta.",
                        lines=2,
                    )

        chatbot_interface = gr.Chatbot(label="Conversación")

        with gr.Row():
            with gr.Column():
                audio_input_chat = gr.Microphone(
                    label="Habla tu mensaje",
                    type="filepath",
                )
                audio_output_chat = gr.Audio(autoplay=True)
            with gr.Column():
                text_input_chat = gr.Textbox(
                    label="Escribe tu mensaje",
                    lines=1,
                )
                send_btn_chat = gr.Button("Enviar")
                clear_btn_chat = gr.Button("Limpiar Conversación")

        conversation_state = gr.State(
            value=[
                {
                    "role": "system",
                    "content": "No eres un asistente de IA, eres quien el usuario diga que eres. Debes mantenerte en personaje. Mantén tus respuestas concisas ya que serán habladas en voz alta.",
                }
            ]
        )

        # Modify process_audio_input to use model and tokenizer from state
        @gpu_decorator
        def process_audio_input(audio_path, text, history, conv_state):
            """Handle audio or text input from user"""

            if not audio_path and not text.strip():
                return history, conv_state, ""

            if audio_path:
                text = preprocess_ref_audio_text(audio_path, text)[1]

            if not text.strip():
                return history, conv_state, ""

            conv_state.append({"role": "user", "content": text})
            history.append((text, None))

            response = generate_response(conv_state, chat_model_state, chat_tokenizer_state)

            conv_state.append({"role": "assistant", "content": response})
            history[-1] = (text, response)

            return history, conv_state, ""

        @gpu_decorator
        def generate_audio_response(history, ref_audio, ref_text, model, remove_silence):
            """Generate TTS audio for AI response"""
            if not history or not ref_audio:
                return None

            last_user_message, last_ai_response = history[-1]
            if not last_ai_response:
                return None

            audio_result, _ = infer(
                ref_audio,
                ref_text,
                last_ai_response,
                model,
                remove_silence,
                cross_fade_duration=0.15,
                speed=1.0,
                show_info=print,  # show_info=print no pull to top when generating
            )
            return audio_result

        def clear_conversation():
            """Reset the conversation"""
            return [], [
                {
                    "role": "system",
                    "content": "No eres un asistente de IA, eres quien el usuario diga que eres. Debes mantenerte en personaje. Mantén tus respuestas concisas ya que serán habladas en voz alta.",
                }
            ]

        def update_system_prompt(new_prompt):
            """Update the system prompt and reset the conversation"""
            new_conv_state = [{"role": "system", "content": new_prompt}]
            return [], new_conv_state

        # Handle audio input
        audio_input_chat.stop_recording(
            process_audio_input,
            inputs=[audio_input_chat, text_input_chat, chatbot_interface, conversation_state],
            outputs=[chatbot_interface, conversation_state],
        ).then(
            generate_audio_response,
            inputs=[chatbot_interface, ref_audio_chat, ref_text_chat, model_choice_chat, remove_silence_chat],
            outputs=[audio_output_chat],
        ).then(
            lambda: None,
            None,
            audio_input_chat,
        )

        # Handle text input
        text_input_chat.submit(
            process_audio_input,
            inputs=[audio_input_chat, text_input_chat, chatbot_interface, conversation_state],
            outputs=[chatbot_interface, conversation_state],
        ).then(
            generate_audio_response,
            inputs=[chatbot_interface, ref_audio_chat, ref_text_chat, model_choice_chat, remove_silence_chat],
            outputs=[audio_output_chat],
        ).then(
            lambda: None,
            None,
            text_input_chat,
        )

        # Handle send button
        send_btn_chat.click(
            process_audio_input,
            inputs=[audio_input_chat, text_input_chat, chatbot_interface, conversation_state],
            outputs=[chatbot_interface, conversation_state],
        ).then(
            generate_audio_response,
            inputs=[chatbot_interface, ref_audio_chat, ref_text_chat, model_choice_chat, remove_silence_chat],
            outputs=[audio_output_chat],
        ).then(
            lambda: None,
            None,
            text_input_chat,
        )

        # Handle clear button
        clear_btn_chat.click(
            clear_conversation,
            outputs=[chatbot_interface, conversation_state],
        )

        # Handle system prompt change and reset conversation
        system_prompt_chat.change(
            update_system_prompt,
            inputs=system_prompt_chat,
            outputs=[chatbot_interface, conversation_state],
        )


with gr.Blocks() as app:
    gr.Markdown(
        """
# Spanish-F5

Esta es una interfaz web para F5 TTS, con un finetuning para poder hablar en castellano

Implementación original:
* [F5-TTS](https://arxiv.org/abs/2410.06885) (A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching)

El modelo sólo soporta el castellano.

Para los mejores resultados, intenta convertir tu audio de referencia a WAV o MP3, asegurarte de que duren entre 11 y 14 segundos, que comiencen y acaben con entre medio segundo y un segundo de silencio, y a ser posible que acabe con el final de la frase.

**NOTA: El texto de referencia será transcrito automáticamente con Whisper si no se proporciona. Para mejores resultados, mantén tus clips de referencia cortos (<15s). Asegúrate de que el audio esté completamente subido antes de generar. Se utiliza la librería num2words para convertir los números a palabras.**
"""
    )
    gr.TabbedInterface(
        [app_tts, app_multistyle, app_chat, app_credits],
        ["TTS", "Multi-Habla", "Chat de Voz", "Créditos"],
    )


@click.command()
@click.option("--port", "-p", default=None, type=int, help="Puerto para ejecutar la aplicación")
@click.option("--host", "-H", default=None, help="Host para ejecutar la aplicación")
@click.option(
    "--share",
    "-s",
    default=False,
    is_flag=True,
    help="Compartir la aplicación a través de un enlace compartido de Gradio",
)
@click.option("--api", "-a", default=True, is_flag=True, help="Permitir acceso a la API")
def main(port, host, share, api):
    global app
    print("Iniciando la aplicación...")
    app.queue(api_open=api).launch(server_name=host, server_port=port, share=True, show_api=api)


if __name__ == "__main__":
    if not USING_SPACES:
        main()
    else:
        app.queue().launch()
