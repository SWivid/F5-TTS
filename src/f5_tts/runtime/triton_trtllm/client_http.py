# Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import argparse

import numpy as np
import requests
import soundfile as sf


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--server-url",
        type=str,
        default="localhost:8000",
        help="Address of the server",
    )

    parser.add_argument(
        "--reference-audio",
        type=str,
        default="../../infer/examples/basic/basic_ref_en.wav",
        help="Path to a single audio file. It can't be specified at the same time with --manifest-dir",
    )

    parser.add_argument(
        "--reference-text",
        type=str,
        default="Some call me nature, others call me mother nature.",
        help="",
    )

    parser.add_argument(
        "--target-text",
        type=str,
        default="I don't really care what you call me. I've been a silent spectator, watching species evolve, empires rise and fall. But always remember, I am mighty and enduring.",
        help="",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="f5_tts",
        choices=["f5_tts", "spark_tts"],
        help="triton model_repo module name to request",
    )

    parser.add_argument(
        "--output-audio",
        type=str,
        default="output.wav",
        help="Path to save the output audio",
    )
    return parser.parse_args()


def prepare_request(
    samples,
    reference_text,
    target_text,
    sample_rate=16000,
    audio_save_dir: str = "./",
):
    assert len(samples.shape) == 1, "samples should be 1D"
    lengths = np.array([[len(samples)]], dtype=np.int32)
    samples = samples.reshape(1, -1).astype(np.float32)

    data = {
        "inputs": [
            {"name": "reference_wav", "shape": samples.shape, "datatype": "FP32", "data": samples.tolist()},
            {
                "name": "reference_wav_len",
                "shape": lengths.shape,
                "datatype": "INT32",
                "data": lengths.tolist(),
            },
            {"name": "reference_text", "shape": [1, 1], "datatype": "BYTES", "data": [reference_text]},
            {"name": "target_text", "shape": [1, 1], "datatype": "BYTES", "data": [target_text]},
        ]
    }

    return data


def load_audio(wav_path, target_sample_rate=16000):
    assert target_sample_rate == 16000, "hard coding in server"
    if isinstance(wav_path, dict):
        samples = wav_path["array"]
        sample_rate = wav_path["sampling_rate"]
    else:
        samples, sample_rate = sf.read(wav_path)
    if sample_rate != target_sample_rate:
        from scipy.signal import resample

        num_samples = int(len(samples) * (target_sample_rate / sample_rate))
        samples = resample(samples, num_samples)
    return samples, target_sample_rate


if __name__ == "__main__":
    args = get_args()
    server_url = args.server_url
    if not server_url.startswith(("http://", "https://")):
        server_url = f"http://{server_url}"

    url = f"{server_url}/v2/models/{args.model_name}/infer"
    samples, sr = load_audio(args.reference_audio)
    assert sr == 16000, "sample rate hardcoded in server"

    samples = np.array(samples, dtype=np.float32)
    data = prepare_request(samples, args.reference_text, args.target_text)

    rsp = requests.post(
        url, headers={"Content-Type": "application/json"}, json=data, verify=False, params={"request_id": "0"}
    )
    result = rsp.json()
    audio = result["outputs"][0]["data"]
    audio = np.array(audio, dtype=np.float32)
    sf.write(args.output_audio, audio, 24000, "PCM_16")
