import socket
import struct
import torch
import torchaudio
import anthropic
import gc
import traceback
from threading import Thread
import os

from f5_tts.infer.utils_infer import infer_batch_process, preprocess_ref_audio_text, load_vocoder, load_model
from f5_tts.model.backbones.dit import DiT

class TTSStreamingProcessor:
    def __init__(self, ckpt_file, vocab_file, ref_audio, ref_text, device=None, dtype=torch.float32):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        
        # Load the TTS model
        self.model = load_model(
            model_cls=DiT,
            model_cfg=dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4),
            ckpt_path=ckpt_file,
            mel_spec_type="vocos",
            vocab_file=vocab_file,
            ode_method="euler",
            use_ema=True,
            device=self.device,
        ).to(self.device, dtype=dtype)

        # Load the vocoder
        self.vocoder = load_vocoder(is_local=False)
        self.sampling_rate = 24000  
        
        # Set reference audio and text
        self.ref_audio = ref_audio
        self.ref_text = ref_text
        
        # System prompt for Claude
        self.system_prompt = """You are an AI friend who always responds in formal Mongolian language using Cyrillic script. 
        Your responses should be modern, friendly with a fun vibe tailored for GenZ and concise (1-2 sentences maximum).
        But it must maintain proper Mongolian grammar and full words.
        Respond directly in all lower-case Cyrillic Mongolian without any additional commentary or translations.
        For example, if asked "Sain baina uu?" you might respond with "Сайн сайн, та сайн байна уу?"."""

    def get_haiku_response(self, user_text):
        """Get response from Claude Haiku and convert to Mongolian Cyrillic"""
        try:
            message = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=100,
                system=self.system_prompt,
                messages=[{
                    "role": "user", 
                    "content": user_text
                }]
            )
            # Extract the actual text from the TextBlock
            response_text = message.content[0].text if isinstance(message.content, list) else message.content
            print(f"Raw response: {response_text}")
            return response_text

        except Exception as e:
            print(f"Error getting Haiku response: {e}")
            return "Уучлаарай, алдаа гарлаа."


    def generate_stream(self, text, play_steps_in_s=0.5):
        """Generate audio in chunks and yield them."""
        try:
            # Get Mongolian response from Haiku
            mongolian_response = self.get_haiku_response(text)
            print(f"Processing Mongolian text: {mongolian_response}")
            mongolian_response = mongolian_response.lower()
            print(mongolian_response)

            # Preprocess reference audio and text
            ref_audio, ref_text = preprocess_ref_audio_text(self.ref_audio, self.ref_text)
            audio, sr = torchaudio.load(ref_audio)

            # Run inference - pass the text directly since it's now a string
            audio_chunk, final_sample_rate, _ = infer_batch_process(
                (audio, sr),
                ref_text,
                [mongolian_response],  # Still wrap in list as infer_batch_process expects a batch
                self.model,
                self.vocoder,
                device=self.device,
            )

            # Break the generated audio into chunks and send them
            chunk_size = int(final_sample_rate * play_steps_in_s)
            for i in range(0, len(audio_chunk), chunk_size):
                chunk = audio_chunk[i:i + chunk_size]
                if len(chunk) == 0:
                    break
                packed_audio = struct.pack(f"{len(chunk)}f", *chunk)
                yield packed_audio

            # Send remaining audio if any
            if len(audio_chunk) % chunk_size != 0:
                remaining = audio_chunk[-(len(audio_chunk) % chunk_size):]
                packed_audio = struct.pack(f"{len(remaining)}f", *remaining)
                yield packed_audio

        except Exception as e:
            print(f"Error in generate_stream: {e}")
            traceback.print_exc()

def handle_client(client_socket, processor):
    try:
        while True:
            data = client_socket.recv(1024).decode("utf-8")
            if not data:
                break

            try:
                text = data.strip()
                for audio_chunk in processor.generate_stream(text):
                    client_socket.sendall(audio_chunk)
                client_socket.sendall(b"END_OF_AUDIO")

            except Exception as e:
                print(f"Error during processing: {e}")
                traceback.print_exc()
                break

    except Exception as e:
        print(f"Error handling client: {e}")
        traceback.print_exc()
    finally:
        client_socket.close()

def start_server(host, port, processor):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((host, port))
    server.listen(5)
    print(f"Server listening on {host}:{port}")

    while True:
        client_socket, addr = server.accept()
        print(f"Accepted connection from {addr}")
        client_handler = Thread(target=handle_client, args=(client_socket, processor))
        client_handler.start()

if __name__ == "__main__":
    try:
        # Load the model and vocoder using the provided files
        ckpt_file = "ckpts/mn_tts/model_last.pt"  # pointing your checkpoint "ckpts/model/model_1096.pt"
        vocab_file = "data/mn_tts_char/vocab.txt"  # Add vocab file path if needed
        ref_audio = "data/mn_tts_char/wavs/john.mp3"  # add ref audio"./tests/ref_audio/reference.wav"
        ref_text = "Hello, my name is John. I am happy to be sharing this news with all of you."

        processor = TTSStreamingProcessor(
            ckpt_file=ckpt_file,
            vocab_file=vocab_file,
            ref_audio=ref_audio,
            ref_text=ref_text,
            dtype=torch.float32,
        )

        start_server("0.0.0.0", 9998, processor)
    except KeyboardInterrupt:
        gc.collect()