#!/usr/bin/env python3
# Copyright      2022  Xiaomi Corp.        (authors: Fangjun Kuang)
#                2023  Nvidia              (authors: Yuekai Zhang)
#                2023  Recurrent.ai        (authors: Songtao Shi)
# See LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This script supports to load dataset from huggingface and sends it to the server
for decoding, in parallel.

Usage:
num_task=2

# For offline F5-TTS
python3 client_grpc.py \
    --server-addr localhost \
    --model-name f5_tts \
    --num-tasks $num_task \
    --huggingface-dataset yuekai/seed_tts \
    --split-name test_zh \
    --log-dir ./log_concurrent_tasks_${num_task}

# For offline Spark-TTS-0.5B
python3 client_grpc.py \
    --server-addr localhost \
    --model-name spark_tts \
    --num-tasks $num_task \
    --huggingface-dataset yuekai/seed_tts \
    --split-name wenetspeech4tts \
    --log-dir ./log_concurrent_tasks_${num_task}
"""

import argparse
import asyncio
import json
import os
import time
import types
from pathlib import Path

import numpy as np
import soundfile as sf
import tritonclient
import tritonclient.grpc.aio as grpcclient
from tritonclient.utils import np_to_triton_dtype


def write_triton_stats(stats, summary_file):
    with open(summary_file, "w") as summary_f:
        model_stats = stats["model_stats"]
        # write a note, the log is from triton_client.get_inference_statistics(), to better human readability
        summary_f.write(
            "The log is parsing from triton_client.get_inference_statistics(), to better human readability. \n"
        )
        summary_f.write("To learn more about the log, please refer to: \n")
        summary_f.write("1. https://github.com/triton-inference-server/server/blob/main/docs/user_guide/metrics.md \n")
        summary_f.write("2. https://github.com/triton-inference-server/server/issues/5374 \n\n")
        summary_f.write(
            "To better improve throughput, we always would like let requests wait in the queue for a while, and then execute them with a larger batch size. \n"
        )
        summary_f.write(
            "However, there is a trade-off between the increased queue time and the increased batch size. \n"
        )
        summary_f.write(
            "You may change 'max_queue_delay_microseconds' and 'preferred_batch_size' in the model configuration file to achieve this. \n"
        )
        summary_f.write(
            "See https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#delayed-batching for more details. \n\n"
        )
        for model_state in model_stats:
            if "last_inference" not in model_state:
                continue
            summary_f.write(f"model name is {model_state['name']} \n")
            model_inference_stats = model_state["inference_stats"]
            total_queue_time_s = int(model_inference_stats["queue"]["ns"]) / 1e9
            total_infer_time_s = int(model_inference_stats["compute_infer"]["ns"]) / 1e9
            total_input_time_s = int(model_inference_stats["compute_input"]["ns"]) / 1e9
            total_output_time_s = int(model_inference_stats["compute_output"]["ns"]) / 1e9
            summary_f.write(
                f"queue time {total_queue_time_s:<5.2f} s, compute infer time {total_infer_time_s:<5.2f} s, compute input time {total_input_time_s:<5.2f} s, compute output time {total_output_time_s:<5.2f} s \n"  # noqa
            )
            model_batch_stats = model_state["batch_stats"]
            for batch in model_batch_stats:
                batch_size = int(batch["batch_size"])
                compute_input = batch["compute_input"]
                compute_output = batch["compute_output"]
                compute_infer = batch["compute_infer"]
                batch_count = int(compute_infer["count"])
                assert compute_infer["count"] == compute_output["count"] == compute_input["count"]
                compute_infer_time_ms = int(compute_infer["ns"]) / 1e6
                compute_input_time_ms = int(compute_input["ns"]) / 1e6
                compute_output_time_ms = int(compute_output["ns"]) / 1e6
                summary_f.write(
                    f"execuate inference with batch_size {batch_size:<2} total {batch_count:<5} times, total_infer_time {compute_infer_time_ms:<9.2f} ms, avg_infer_time {compute_infer_time_ms:<9.2f}/{batch_count:<5}={compute_infer_time_ms / batch_count:.2f} ms, avg_infer_time_per_sample {compute_infer_time_ms:<9.2f}/{batch_count:<5}/{batch_size}={compute_infer_time_ms / batch_count / batch_size:.2f} ms \n"  # noqa
                )
                summary_f.write(
                    f"input {compute_input_time_ms:<9.2f} ms, avg {compute_input_time_ms / batch_count:.2f} ms, "  # noqa
                )
                summary_f.write(
                    f"output {compute_output_time_ms:<9.2f} ms, avg {compute_output_time_ms / batch_count:.2f} ms \n"  # noqa
                )


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--server-addr",
        type=str,
        default="localhost",
        help="Address of the server",
    )

    parser.add_argument(
        "--server-port",
        type=int,
        default=8001,
        help="Grpc port of the triton server, default is 8001",
    )

    parser.add_argument(
        "--reference-audio",
        type=str,
        default=None,
        help="Path to a single audio file. It can't be specified at the same time with --manifest-dir",
    )

    parser.add_argument(
        "--reference-text",
        type=str,
        default="",
        help="",
    )

    parser.add_argument(
        "--target-text",
        type=str,
        default="",
        help="",
    )

    parser.add_argument(
        "--huggingface-dataset",
        type=str,
        default="yuekai/seed_tts",
        help="dataset name in huggingface dataset hub",
    )

    parser.add_argument(
        "--split-name",
        type=str,
        default="wenetspeech4tts",
        choices=["wenetspeech4tts", "test_zh", "test_en", "test_hard"],
        help="dataset split name, default is 'test'",
    )

    parser.add_argument(
        "--manifest-path",
        type=str,
        default=None,
        help="Path to the manifest dir which includes wav.scp trans.txt files.",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="f5_tts",
        choices=["f5_tts", "spark_tts"],
        help="triton model_repo module name to request: transducer for k2, attention_rescoring for wenet offline, streaming_wenet for wenet streaming, infer_pipeline for paraformer large offline",
    )

    parser.add_argument(
        "--num-tasks",
        type=int,
        default=1,
        help="Number of concurrent tasks for sending",
    )

    parser.add_argument(
        "--log-interval",
        type=int,
        default=5,
        help="Controls how frequently we print the log.",
    )

    parser.add_argument(
        "--compute-wer",
        action="store_true",
        default=False,
        help="""True to compute WER.
        """,
    )

    parser.add_argument(
        "--log-dir",
        type=str,
        required=False,
        default="./tmp",
        help="log directory",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Inference batch_size per request for offline mode.",
    )

    return parser.parse_args()


def load_audio(wav_path, target_sample_rate=24000):
    assert target_sample_rate == 24000, "hard coding in server"
    if isinstance(wav_path, dict):
        waveform = wav_path["array"]
        sample_rate = wav_path["sampling_rate"]
    else:
        waveform, sample_rate = sf.read(wav_path)
    if sample_rate != target_sample_rate:
        from scipy.signal import resample

        num_samples = int(len(waveform) * (target_sample_rate / sample_rate))
        waveform = resample(waveform, num_samples)
    return waveform, target_sample_rate


async def send(
    manifest_item_list: list,
    name: str,
    triton_client: tritonclient.grpc.aio.InferenceServerClient,
    protocol_client: types.ModuleType,
    log_interval: int,
    model_name: str,
    padding_duration: int = None,
    audio_save_dir: str = "./",
    save_sample_rate: int = 24000,
):
    total_duration = 0.0
    latency_data = []
    task_id = int(name[5:])

    print(f"manifest_item_list: {manifest_item_list}")
    for i, item in enumerate(manifest_item_list):
        if i % log_interval == 0:
            print(f"{name}: {i}/{len(manifest_item_list)}")
        waveform, sample_rate = load_audio(item["audio_filepath"], target_sample_rate=24000)
        duration = len(waveform) / sample_rate
        lengths = np.array([[len(waveform)]], dtype=np.int32)

        reference_text, target_text = item["reference_text"], item["target_text"]

        estimated_target_duration = duration / len(reference_text) * len(target_text)

        if padding_duration:
            # padding to nearset 10 seconds
            samples = np.zeros(
                (
                    1,
                    padding_duration
                    * sample_rate
                    * ((int(estimated_target_duration + duration) // padding_duration) + 1),
                ),
                dtype=np.float32,
            )

            samples[0, : len(waveform)] = waveform
        else:
            samples = waveform

        samples = samples.reshape(1, -1).astype(np.float32)

        inputs = [
            protocol_client.InferInput("reference_wav", samples.shape, np_to_triton_dtype(samples.dtype)),
            protocol_client.InferInput("reference_wav_len", lengths.shape, np_to_triton_dtype(lengths.dtype)),
            protocol_client.InferInput("reference_text", [1, 1], "BYTES"),
            protocol_client.InferInput("target_text", [1, 1], "BYTES"),
        ]
        inputs[0].set_data_from_numpy(samples)
        inputs[1].set_data_from_numpy(lengths)

        input_data_numpy = np.array([reference_text], dtype=object)
        input_data_numpy = input_data_numpy.reshape((1, 1))
        inputs[2].set_data_from_numpy(input_data_numpy)

        input_data_numpy = np.array([target_text], dtype=object)
        input_data_numpy = input_data_numpy.reshape((1, 1))
        inputs[3].set_data_from_numpy(input_data_numpy)

        outputs = [protocol_client.InferRequestedOutput("waveform")]

        sequence_id = 100000000 + i + task_id * 10
        start = time.time()
        response = await triton_client.infer(model_name, inputs, request_id=str(sequence_id), outputs=outputs)

        audio = response.as_numpy("waveform").reshape(-1)

        end = time.time() - start

        audio_save_path = os.path.join(audio_save_dir, f"{item['target_audio_path']}.wav")
        sf.write(audio_save_path, audio, save_sample_rate, "PCM_16")

        actual_duration = len(audio) / save_sample_rate
        latency_data.append((end, actual_duration))
        total_duration += actual_duration

    return total_duration, latency_data


def load_manifests(manifest_path):
    with open(manifest_path, "r") as f:
        manifest_list = []
        for line in f:
            assert len(line.strip().split("|")) == 4
            utt, prompt_text, prompt_wav, gt_text = line.strip().split("|")
            utt = Path(utt).stem
            # gt_wav = os.path.join(os.path.dirname(manifest_path), "wavs", utt + ".wav")
            if not os.path.isabs(prompt_wav):
                prompt_wav = os.path.join(os.path.dirname(manifest_path), prompt_wav)
            manifest_list.append(
                {
                    "audio_filepath": prompt_wav,
                    "reference_text": prompt_text,
                    "target_text": gt_text,
                    "target_audio_path": utt,
                }
            )
    return manifest_list


def split_data(data, k):
    n = len(data)
    if n < k:
        print(f"Warning: the length of the input list ({n}) is less than k ({k}). Setting k to {n}.")
        k = n

    quotient = n // k
    remainder = n % k

    result = []
    start = 0
    for i in range(k):
        if i < remainder:
            end = start + quotient + 1
        else:
            end = start + quotient

        result.append(data[start:end])
        start = end

    return result


async def main():
    args = get_args()
    url = f"{args.server_addr}:{args.server_port}"

    triton_client = grpcclient.InferenceServerClient(url=url, verbose=False)
    protocol_client = grpcclient

    if args.reference_audio:
        args.num_tasks = 1
        args.log_interval = 1
        manifest_item_list = [
            {
                "reference_text": args.reference_text,
                "target_text": args.target_text,
                "audio_filepath": args.reference_audio,
                "target_audio_path": "test",
            }
        ]
    elif args.huggingface_dataset:
        import datasets

        dataset = datasets.load_dataset(
            args.huggingface_dataset,
            split=args.split_name,
            trust_remote_code=True,
        )
        manifest_item_list = []
        for i in range(len(dataset)):
            manifest_item_list.append(
                {
                    "audio_filepath": dataset[i]["prompt_audio"],
                    "reference_text": dataset[i]["prompt_text"],
                    "target_audio_path": dataset[i]["id"],
                    "target_text": dataset[i]["target_text"],
                }
            )
    else:
        manifest_item_list = load_manifests(args.manifest_path)

    args.num_tasks = min(args.num_tasks, len(manifest_item_list))
    manifest_item_list = split_data(manifest_item_list, args.num_tasks)

    os.makedirs(args.log_dir, exist_ok=True)
    tasks = []
    start_time = time.time()
    for i in range(args.num_tasks):
        task = asyncio.create_task(
            send(
                manifest_item_list[i],
                name=f"task-{i}",
                triton_client=triton_client,
                protocol_client=protocol_client,
                log_interval=args.log_interval,
                model_name=args.model_name,
                audio_save_dir=args.log_dir,
                padding_duration=1,
                save_sample_rate=24000,
            )
        )
        tasks.append(task)

    ans_list = await asyncio.gather(*tasks)

    end_time = time.time()
    elapsed = end_time - start_time

    total_duration = 0.0
    latency_data = []
    for ans in ans_list:
        total_duration += ans[0]
        latency_data += ans[1]

    rtf = elapsed / total_duration

    s = f"RTF: {rtf:.4f}\n"
    s += f"total_duration: {total_duration:.3f} seconds\n"
    s += f"({total_duration / 3600:.2f} hours)\n"
    s += f"processing time: {elapsed:.3f} seconds ({elapsed / 3600:.2f} hours)\n"

    latency_list = [chunk_end for (chunk_end, chunk_duration) in latency_data]
    latency_ms = sum(latency_list) / float(len(latency_list)) * 1000.0
    latency_variance = np.var(latency_list, dtype=np.float64) * 1000.0
    s += f"latency_variance: {latency_variance:.2f}\n"
    s += f"latency_50_percentile_ms: {np.percentile(latency_list, 50) * 1000.0:.2f}\n"
    s += f"latency_90_percentile_ms: {np.percentile(latency_list, 90) * 1000.0:.2f}\n"
    s += f"latency_95_percentile_ms: {np.percentile(latency_list, 95) * 1000.0:.2f}\n"
    s += f"latency_99_percentile_ms: {np.percentile(latency_list, 99) * 1000.0:.2f}\n"
    s += f"average_latency_ms: {latency_ms:.2f}\n"

    print(s)
    if args.manifest_path:
        name = Path(args.manifest_path).stem
    elif args.split_name:
        name = args.split_name
    with open(f"{args.log_dir}/rtf-{name}.txt", "w") as f:
        f.write(s)

    stats = await triton_client.get_inference_statistics(model_name="", as_json=True)
    write_triton_stats(stats, f"{args.log_dir}/stats_summary-{name}.txt")

    metadata = await triton_client.get_model_config(model_name=args.model_name, as_json=True)
    with open(f"{args.log_dir}/model_config-{name}.json", "w") as f:
        json.dump(metadata, f, indent=4)


if __name__ == "__main__":
    asyncio.run(main())
