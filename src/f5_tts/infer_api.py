import argparse
import gc
import torch
import torchaudio
import traceback
from importlib.resources import files
from fastapi import FastAPI, HTTPException, Response, Query
from pydantic import BaseModel
import base64
import io
from cached_path import cached_path
from fastapi.responses import StreamingResponse

from infer.utils_infer import infer_batch_process, preprocess_ref_audio_text, load_vocoder, load_model
from model.backbones.dit import DiT

# Initialize FastAPI App
app = FastAPI()

class TTSStreamingProcessor:
    def __init__(self, ckpt_file, vocab_file, ref_audio, ref_text, device=None, dtype=torch.float32):
        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "xpu"
            if torch.xpu.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        # Load the model using the provided checkpoint and vocab files
        self.model = load_model(
            model_cls=DiT,
            model_cfg=dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4),
            ckpt_path=ckpt_file,
            mel_spec_type="vocos",  # or "bigvgan" depending on vocoder
            vocab_file=vocab_file,
            ode_method="euler",
            use_ema=True,
            device=self.device,
        ).to(self.device, dtype=dtype)

        # Load the vocoder
        self.vocoder = load_vocoder(is_local=False)

        # Set sampling rate for streaming
        self.sampling_rate = 24000  # Consistency with client

        # Set reference audio and text
        self.ref_audio = ref_audio
        self.ref_text = ref_text

        # Warm up the model
        self._warm_up()

    def _warm_up(self):
        """Warm up the model with a dummy input to ensure it's ready for real-time processing."""
        print("Warming up the model...")
        ref_audio, ref_text = preprocess_ref_audio_text(self.ref_audio, self.ref_text)
        audio, sr = torchaudio.load(ref_audio)
        gen_text = "Warm-up text for the model."

        # Pass the vocoder as an argument here
        infer_batch_process((audio, sr), ref_text, [gen_text], self.model, self.vocoder, device=self.device)
        print("Warm-up completed.")

    def generate_audio(self, text):
        """Generate audio for the given text and return it as a WAV file."""
        # Preprocess the reference audio and text
        ref_audio, ref_text = preprocess_ref_audio_text(self.ref_audio, self.ref_text)

        # Load reference audio
        audio, sr = torchaudio.load(ref_audio)

        # Run inference for the input text
        audio_chunk, final_sample_rate, _ = infer_batch_process(
            (audio, sr),
            ref_text,
            [text],
            self.model,
            self.vocoder,
            device=self.device,  # Pass vocoder here
        )

        # Convert audio array to bytes (WAV format)
        audio_buffer = io.BytesIO()
        torchaudio.save(audio_buffer, torch.tensor(audio_chunk).unsqueeze(0), final_sample_rate, format="wav")
        audio_buffer.seek(0)


        return audio_buffer


# Define input data model for API requests
class TTSRequest(BaseModel):
    text: str
    response_type: str = Query("json", description="Response format: json, file, stream")


# Initialize processor globally
processor = None

@app.on_event("startup")
def load_model_on_startup():
    """Load the model when the server starts"""
    global processor
    args = parser.parse_args()

    try:
        processor = TTSStreamingProcessor(
            ckpt_file=args.ckpt_file,
            vocab_file=args.vocab_file,
            ref_audio=args.ref_audio,
            ref_text=args.ref_text,
            device=args.device,
            dtype=args.dtype,
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        traceback.print_exc()
        processor = None


@app.post("/tts/")
async def text_to_speech(request: TTSRequest):
    """
    Converts text to speech and returns the audio in different formats.
    """
    try:
        if processor is None:
            raise HTTPException(status_code=500, detail="TTS Processor not initialized")

        # Generate audio buffer
        audio_buffer = processor.generate_audio(request.text)
        chunk_size = 1024  # Stream in 1024-byte chunks

        # (A) JSON-encoded Base64 (default)
        if request.response_type == "json":
            audio_base64 = base64.b64encode(audio_buffer.read()).decode("utf-8")
            return {"audio_base64": audio_base64, "message": "TTS generated successfully"}

        # (B) Return WAV File (File Download Mode)
        elif request.response_type == "file":
            audio_buffer.seek(0)
            return Response(content=audio_buffer.read(), media_type="audio/wav",
                            headers={"Content-Disposition": "attachment; filename=output.wav"})

        # (C) Stream Audio in Small Chunks (Real-Time Playback)
        elif request.response_type == "stream":
            def audio_stream():
                audio_buffer.seek(0)

                # **Ensure WAV header is sent first**
                wav_header = audio_buffer.read(44)  # First 44 bytes = WAV header
                yield wav_header

                # **Stream the rest of the audio in chunks**
                while True:
                    chunk = audio_buffer.read(chunk_size)
                    if not chunk:
                        print("End of audio stream")
                        break  # Stop when all audio is sent
                    print(f"Streaming chunk of size {len(chunk)} bytes")
                    yield chunk

            return StreamingResponse(audio_stream(), media_type="audio/wav") 

        else:
            raise HTTPException(status_code=400, detail="Invalid response_type. Choose 'json', 'file', or 'stream'.")

    except Exception as e:
        print(f"Server Error: {e}")  # Log error on server side
        raise HTTPException(status_code=500, detail=f"Error generating speech: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)

    parser.add_argument(
        "--ckpt_file",
        default=str(cached_path("hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.safetensors")),
        help="Path to the model checkpoint file",
    )
    parser.add_argument(
        "--vocab_file",
        default="",
        help="Path to the vocab file if customized",
    )

    parser.add_argument(
        "--ref_audio",
        default=str(files("f5_tts").joinpath("infer/examples/basic/basic_ref_en.wav")),
        help="Reference audio to provide model with speaker characteristics",
    )
    parser.add_argument(
        "--ref_text",
        default="",
        help="Reference audio subtitle, leave empty to auto-transcribe",
    )

    parser.add_argument("--device", default=None, help="Device to run the model on")
    parser.add_argument("--dtype", default=torch.float32, help="Data type to use for model inference")

    args = parser.parse_args()

    # Start FastAPI server
    uvicorn.run(app, host=args.host, port=args.port)
