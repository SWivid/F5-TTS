## Triton Inference Serving Best Practice for F5 TTS

### Quick Start
Directly launch the service using docker compose.
```sh
# TODO: support F5TTS_v1_Base
MODEL=F5TTS_Base docker compose up
```

### Build Image
Build the docker image from scratch. 
```sh
docker build . -f Dockerfile.server -t soar97/triton-f5-tts:24.12
```

### Create Docker Container
```sh
your_mount_dir=/mnt:/mnt
docker run -it --name "f5-server" --gpus all --net host -v $your_mount_dir --shm-size=2g soar97/triton-f5-tts:24.12
```

### Export Models to TensorRT-LLM and Launch Server
Inside docker container, we would follow the official guide of TensorRT-LLM to build qwen and whisper TensorRT-LLM engines. See [here](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/whisper).

```sh
bash build_server.sh
```

### Benchmark using Dataset
```sh
num_task=2
python3 client.py --num-tasks $num_task --huggingface-dataset yuekai/seed_tts --split-name wenetspeech4tts
```

### Benchmark Results
Decoding on a single L20 GPU, using 26 different prompt_audio/target_text pairs.

| Model | Note   | Concurrency | Avg Latency     | RTF | 
|-------|-----------|-----------------------|---------|--|
| F5-TTS Base (Vocos) | [Code Commit](https://github.com/yuekaizhang/sherpa/tree/329ab3c573252e835844bea38505c6b43e994cf4/triton/f5_tts) | 1                   | 253 ms | 0.0394|

### Credits
1. [F5-TTS](https://github.com/SWivid/F5-TTS)
2. [F5-TTS-TRTLLM](https://github.com/Bigfishering/f5-tts-trtllm)