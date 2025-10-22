## Triton Inference Serving Best Practice for F5-TTS

### Setup
#### Option 1: Quick Start
```sh
# Directly launch the service using docker compose
MODEL=F5TTS_v1_Base docker compose up
```

#### Option 2: Build from scratch
```sh
# Build the docker image
docker build . -f Dockerfile.server -t soar97/triton-f5-tts:24.12

# Create Docker Container
your_mount_dir=/mnt:/mnt
docker run -it --name "f5-server" --gpus all --net host -v $your_mount_dir --shm-size=2g soar97/triton-f5-tts:24.12
```

### Build TensorRT-LLM Engines and Launch Server
Inside docker container, we would follow the official guide of TensorRT-LLM to build qwen and whisper TensorRT-LLM engines. See [here](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/core/whisper).
```sh
# F5TTS_v1_Base | F5TTS_Base | F5TTS_v1_Small | F5TTS_Small
bash run.sh 0 4 F5TTS_v1_Base
```
> [!NOTE]  
> If use custom checkpoint, set `ckpt_file` and `vocab_file` in `run.sh`.  
> Remember to used matched model version (`F5TTS_v1_*` for v1, `F5TTS_*` for v0).
> 
> If use checkpoint of different structure, see `scripts/convert_checkpoint.py`, and perform modification if necessary.

> [!IMPORTANT]  
> If train or finetune with fp32, add `--dtype float32` flag when converting checkpoint in `run.sh` phase 1.

### HTTP Client
```sh
python3 client_http.py
```

### Benchmarking
#### Using Client-Server Mode
```sh
# bash run.sh 5 5 F5TTS_v1_Base
num_task=2
python3 client_grpc.py --num-tasks $num_task --huggingface-dataset yuekai/seed_tts --split-name wenetspeech4tts
```

#### Using Offline TRT-LLM Mode
```sh
# bash run.sh 7 7 F5TTS_v1_Base
batch_size=1
split_name=wenetspeech4tts
backend_type=trt
log_dir=./tests/benchmark_batch_size_${batch_size}_${split_name}_${backend_type}
rm -r $log_dir
torchrun --nproc_per_node=1 \
benchmark.py --output-dir $log_dir \
--batch-size $batch_size \
--enable-warmup \
--split-name $split_name \
--model-path $ckpt_file \
--vocab-file $vocab_file \
--vocoder-trt-engine-path $VOCODER_TRT_ENGINE_PATH \
--backend-type $backend_type \
--tllm-model-dir $TRTLLM_ENGINE_DIR || exit 1
```

### Benchmark Results
Decoding on a single L20 GPU, using 26 different prompt_audio & target_text pairs, 16 NFE.

| Model               | Concurrency    | Avg Latency | RTF    | Mode            |
|---------------------|----------------|-------------|--------|-----------------|
| F5-TTS Base (Vocos) | 2              | 253 ms      | 0.0394 | Client-Server   |
| F5-TTS Base (Vocos) | 1 (Batch_size) | -           | 0.0402 | Offline TRT-LLM |
| F5-TTS Base (Vocos) | 1 (Batch_size) | -           | 0.1467 | Offline Pytorch |

### Credits
1. [Yuekai Zhang](https://github.com/yuekaizhang)
2. [F5-TTS-TRTLLM](https://github.com/Bigfishering/f5-tts-trtllm)
