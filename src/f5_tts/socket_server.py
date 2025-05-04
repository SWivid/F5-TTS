import argparse
import gc
import logging
import queue
import socket
import struct
import threading
import traceback
import wave
from importlib.resources import files

import numpy as np
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from hydra.utils import get_class
from omegaconf import OmegaConf

from f5_tts.infer.utils_infer import (
    chunk_text,
    infer_batch_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioFileWriterThread(threading.Thread):
    """Threaded file writer to avoid blocking the TTS streaming process."""

    def __init__(self, output_file, sampling_rate):
        super().__init__()
        self.output_file = output_file
        self.sampling_rate = sampling_rate
        self.queue = queue.Queue()
        self.stop_event = threading.Event()
        self.audio_data = []

    def run(self):
        """Process queued audio data and write it to a file."""
        logger.info("AudioFileWriterThread started.")
        with wave.open(self.output_file, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sampling_rate)

            while not self.stop_event.is_set() or not self.queue.empty():
                try:
                    chunk = self.queue.get(timeout=0.1)
                    if chunk is not None:
                        chunk = np.int16(chunk * 32767)
                        self.audio_data.append(chunk)
                        wf.writeframes(chunk.tobytes())
                except queue.Empty:
                    continue

    def add_chunk(self, chunk):
        """Add a new chunk to the queue."""
        self.queue.put(chunk)

    def stop(self):
        """Stop writing and ensure all queued data is written."""
        self.stop_event.set()
        self.join()
        logger.info("Audio writing completed.")


class TTSStreamingProcessor:
    def __init__(self, model, ckpt_file, vocab_file, ref_audio, ref_text, device=None, dtype=torch.float32):
        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "xpu"
            if torch.xpu.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        model_cfg = OmegaConf.load(str(files("f5_tts").joinpath(f"configs/{model}.yaml")))
        self.model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
        self.model_arc = model_cfg.model.arch
        self.mel_spec_type = model_cfg.model.mel_spec.mel_spec_type
        self.sampling_rate = model_cfg.model.mel_spec.target_sample_rate

        self.model = self.load_ema_model(ckpt_file, vocab_file, dtype)
        self.vocoder = self.load_vocoder_model()

        self.update_reference(ref_audio, ref_text)
        self._warm_up()
        self.file_writer_thread = None
        self.first_package = True

    def load_ema_model(self, ckpt_file, vocab_file, dtype):
        return load_model(
            self.model_cls,
            self.model_arc,
            ckpt_path=ckpt_file,
            mel_spec_type=self.mel_spec_type,
            vocab_file=vocab_file,
            ode_method="euler",
            use_ema=True,
            device=self.device,
        ).to(self.device, dtype=dtype)

    def load_vocoder_model(self):
        return load_vocoder(vocoder_name=self.mel_spec_type, is_local=False, local_path=None, device=self.device)

    def update_reference(self, ref_audio, ref_text):
        self.ref_audio, self.ref_text = preprocess_ref_audio_text(ref_audio, ref_text)
        self.audio, self.sr = torchaudio.load(self.ref_audio)

        ref_audio_duration = self.audio.shape[-1] / self.sr
        ref_text_byte_len = len(self.ref_text.encode("utf-8"))
        self.max_chars = int(ref_text_byte_len / (ref_audio_duration) * (25 - ref_audio_duration))
        self.few_chars = int(ref_text_byte_len / (ref_audio_duration) * (25 - ref_audio_duration) / 2)
        self.min_chars = int(ref_text_byte_len / (ref_audio_duration) * (25 - ref_audio_duration) / 4)

    def _warm_up(self):
        logger.info("Warming up the model...")
        gen_text = "Warm-up text for the model."
        for _ in infer_batch_process(
            (self.audio, self.sr),
            self.ref_text,
            [gen_text],
            self.model,
            self.vocoder,
            progress=None,
            device=self.device,
            streaming=True,
        ):
            pass
        logger.info("Warm-up completed.")

    def generate_stream(self, text, conn):
        text_batches = chunk_text(text, max_chars=self.max_chars)
        if self.first_package:
            text_batches = chunk_text(text_batches[0], max_chars=self.few_chars) + text_batches[1:]
            text_batches = chunk_text(text_batches[0], max_chars=self.min_chars) + text_batches[1:]
            self.first_package = False

        audio_stream = infer_batch_process(
            (self.audio, self.sr),
            self.ref_text,
            text_batches,
            self.model,
            self.vocoder,
            progress=None,
            device=self.device,
            streaming=True,
            chunk_size=2048,
        )

        # Reset the file writer thread
        if self.file_writer_thread is not None:
            self.file_writer_thread.stop()
        self.file_writer_thread = AudioFileWriterThread("output.wav", self.sampling_rate)
        self.file_writer_thread.start()

        for audio_chunk, _ in audio_stream:
            if len(audio_chunk) > 0:
                logger.info(f"Generated audio chunk of size: {len(audio_chunk)}")

                # Send audio chunk via socket
                conn.sendall(struct.pack(f"{len(audio_chunk)}f", *audio_chunk))

                # Write to file asynchronously
                self.file_writer_thread.add_chunk(audio_chunk)

        logger.info("Finished sending audio stream.")
        conn.sendall(b"END")  # Send end signal

        # Ensure all audio data is written before exiting
        self.file_writer_thread.stop()


def handle_client(conn, processor):
    try:
        with conn:
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            while True:
                data = conn.recv(1024)
                if not data:
                    processor.first_package = True
                    break
                data_str = data.decode("utf-8").strip()
                logger.info(f"Received text: {data_str}")

                try:
                    processor.generate_stream(data_str, conn)
                except Exception as inner_e:
                    logger.error(f"Error during processing: {inner_e}")
                    traceback.print_exc()
                    break
    except Exception as e:
        logger.error(f"Error handling client: {e}")
        traceback.print_exc()


def start_server(host, port, processor):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen()
        logger.info(f"Server started on {host}:{port}")
        while True:
            conn, addr = s.accept()
            logger.info(f"Connected by {addr}")
            handle_client(conn, processor)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=9998)

    parser.add_argument(
        "--model",
        default="F5TTS_v1_Base",
        help="The model name, e.g. F5TTS_v1_Base",
    )
    parser.add_argument(
        "--ckpt_file",
        default=str(hf_hub_download(repo_id="SWivid/F5-TTS", filename="F5TTS_v1_Base/model_1250000.safetensors")),
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

    try:
        # Initialize the processor with the model and vocoder
        processor = TTSStreamingProcessor(
            model=args.model,
            ckpt_file=args.ckpt_file,
            vocab_file=args.vocab_file,
            ref_audio=args.ref_audio,
            ref_text=args.ref_text,
            device=args.device,
            dtype=args.dtype,
        )

        # Start the server
        start_server(args.host, args.port, processor)

    except KeyboardInterrupt:
        gc.collect()
