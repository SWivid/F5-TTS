## Triton Inference Serving Best Practice for F5-TTS

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
bash run.sh 0 4 F5TTS_Base
```

### HTTP Client
```sh
python3 client_http.py
```

### Benchmark using Client-Server Mode
```sh
num_task=2
python3 client_grpc.py --num-tasks $num_task --huggingface-dataset yuekai/seed_tts --split-name wenetspeech4tts
```

### Benchmark using Offline TRT-LLM Mode
```sh
batch_size=1
split_name=wenetspeech4tts
backend_type=trt
log_dir=./log_benchmark_batch_size_${batch_size}_${split_name}_${backend_type}
rm -r $log_dir
ln -s model_repo_f5_tts/f5_tts/1/f5_tts_trtllm.py ./
torchrun --nproc_per_node=1 \
benchmark.py --output-dir $log_dir \
--batch-size $batch_size \
--enable-warmup \
--split-name $split_name \
--model-path $F5_TTS_HF_DOWNLOAD_PATH/$model/model_1200000.pt \
--vocab-file $F5_TTS_HF_DOWNLOAD_PATH/$model/vocab.txt \
--vocoder-trt-engine-path $vocoder_trt_engine_path \
--backend-type $backend_type \
--tllm-model-dir $F5_TTS_TRT_LLM_ENGINE_PATH || exit 1
```

### Benchmark Results
Decoding on a single L20 GPU, using 26 different prompt_audio & target_text pairs, 16 NFE.

| Model               | Concurrency    | Avg Latency | RTF    | Mode            |
|---------------------|----------------|-------------|--------|-----------------|
| F5-TTS Base (Vocos) | 2              | 253 ms      | 0.0394 | Client-Server   |
| F5-TTS Base (Vocos) | 1 (Batch_size) | -           | 0.0402 | Offline TRT-LLM |
| F5-TTS Base (Vocos) | 1 (Batch_size) | -           | 0.1467 | Offline Pytorch |

### Credits
1. [F5-TTS-TRTLLM](https://github.com/Bigfishering/f5-tts-trtllm)
