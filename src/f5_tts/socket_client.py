import asyncio
import logging
import socket
import time

import numpy as np
import pyaudio


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def listen_to_F5TTS(text, server_ip="localhost", server_port=9998):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    await asyncio.get_event_loop().run_in_executor(None, client_socket.connect, (server_ip, int(server_port)))

    start_time = time.time()
    first_chunk_time = None

    async def play_audio_stream():
        nonlocal first_chunk_time
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paFloat32, channels=1, rate=24000, output=True, frames_per_buffer=2048)

        try:
            while True:
                data = await asyncio.get_event_loop().run_in_executor(None, client_socket.recv, 8192)
                if not data:
                    break
                if data == b"END":
                    logger.info("End of audio received.")
                    break

                audio_array = np.frombuffer(data, dtype=np.float32)
                stream.write(audio_array.tobytes())

                if first_chunk_time is None:
                    first_chunk_time = time.time()

        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

        logger.info(f"Total time taken: {time.time() - start_time:.4f} seconds")

    try:
        data_to_send = f"{text}".encode("utf-8")
        await asyncio.get_event_loop().run_in_executor(None, client_socket.sendall, data_to_send)
        await play_audio_stream()

    except Exception as e:
        logger.error(f"Error in listen_to_F5TTS: {e}")

    finally:
        client_socket.close()


if __name__ == "__main__":
    text_to_send = "As a Reader assistant, I'm familiar with new technology. which are key to its improved performance in terms of both training speed and inference efficiency. Let's break down the components"

    asyncio.run(listen_to_F5TTS(text_to_send))
