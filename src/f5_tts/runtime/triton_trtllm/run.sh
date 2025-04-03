stage=$1
stop_stage=$2
model=$3 # F5TTS_Base
if [ -z "$model" ]; then
    echo "Model is none, using default model F5TTS_Base"
    model=F5TTS_Base
fi
echo "Start stage: $stage, Stop stage: $stop_stage, Model: $model"
export CUDA_VISIBLE_DEVICES=0

F5_TTS_HF_DOWNLOAD_PATH=./F5-TTS
F5_TTS_TRT_LLM_CHECKPOINT_PATH=./trtllm_ckpt
F5_TTS_TRT_LLM_ENGINE_PATH=./f5_trt_llm_engine

vocoder_trt_engine_path=vocos_vocoder.plan
model_repo=./model_repo

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    echo "Downloading f5 tts from huggingface"
    huggingface-cli download SWivid/F5-TTS --local-dir $F5_TTS_HF_DOWNLOAD_PATH

fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    echo "Converting checkpoint"
    python3 ./scripts/convert_checkpoint.py \
        --timm_ckpt "$F5_TTS_HF_DOWNLOAD_PATH/$model/model_1200000.pt" \
        --output_dir "$F5_TTS_TRT_LLM_CHECKPOINT_PATH" --model_name $model
    python_package_path=/usr/local/lib/python3.12/dist-packages
    cp -r patch/* $python_package_path/tensorrt_llm/models
    trtllm-build --checkpoint_dir $F5_TTS_TRT_LLM_CHECKPOINT_PATH \
      --max_batch_size 8 \
      --output_dir $F5_TTS_TRT_LLM_ENGINE_PATH --remove_input_padding disable
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    echo "Exporting vocos vocoder"
    onnx_vocoder_path=vocos_vocoder.onnx
    python3 scripts/export_vocoder_to_onnx.py --vocoder vocos --output-path $onnx_vocoder_path
    bash scripts/export_vocos_trt.sh $onnx_vocoder_path $vocoder_trt_engine_path
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    echo "Building triton server"
    rm -r $model_repo
    cp -r ./model_repo_f5_tts $model_repo
    python3 scripts/fill_template.py -i $model_repo/f5_tts/config.pbtxt vocab:$F5_TTS_HF_DOWNLOAD_PATH/$model/vocab.txt,model:$F5_TTS_HF_DOWNLOAD_PATH/$model/model_1200000.pt,trtllm:$F5_TTS_TRT_LLM_ENGINE_PATH,vocoder:vocos
    cp $vocoder_trt_engine_path $model_repo/vocoder/1/vocoder.plan
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    echo "Starting triton server"
    tritonserver --model-repository=$model_repo
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
    echo "Testing triton server"
    num_task=1
    log_dir=./log_concurrent_tasks_${num_task}
    rm -r $log_dir
    python3 client_grpc.py --num-tasks $num_task --huggingface-dataset yuekai/seed_tts --split-name wenetspeech4tts --log-dir $log_dir
fi

if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
    echo "Testing http client"
    audio=../../infer/examples/basic/basic_ref_en.wav
    reference_text="Some call me nature, others call me mother nature."
    target_text="I don't really care what you call me. I've been a silent spectator, watching species evolve, empires rise and fall. But always remember, I am mighty and enduring."
    python3 client_http.py --reference-audio $audio --reference-text "$reference_text" --target-text "$target_text"
fi

if [ $stage -le 7 ] && [ $stop_stage -ge 7 ]; then
    echo "TRT-LLM: offline decoding benchmark test"
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
fi

if [ $stage -le 8 ] && [ $stop_stage -ge 8 ]; then
    echo "Native Pytorch: offline decoding benchmark test"
    pip install -r requirements-pytorch.txt
    batch_size=1
    split_name=wenetspeech4tts
    backend_type=pytorch
    log_dir=./log_benchmark_batch_size_${batch_size}_${split_name}_${backend_type}
    rm -r $log_dir
    ln -s model_repo_f5_tts/f5_tts/1/f5_tts_trtllm.py ./
    torchrun --nproc_per_node=1 \
    benchmark.py --output-dir $log_dir \
    --batch-size $batch_size \
    --split-name $split_name \
    --enable-warmup \
    --model-path $F5_TTS_HF_DOWNLOAD_PATH/$model/model_1200000.pt \
    --vocab-file $F5_TTS_HF_DOWNLOAD_PATH/$model/vocab.txt \
    --backend-type $backend_type \
    --tllm-model-dir $F5_TTS_TRT_LLM_ENGINE_PATH || exit 1
fi