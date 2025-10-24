stage=$1
stop_stage=$2
model=$3  # F5TTS_v1_Base | F5TTS_Base | F5TTS_v1_Small | F5TTS_Small
if [ -z "$model" ]; then
    model=F5TTS_v1_Base
fi
echo "Start stage: $stage, Stop stage: $stop_stage, Model: $model"
export CUDA_VISIBLE_DEVICES=0

CKPT_DIR=../../../../ckpts
TRTLLM_CKPT_DIR=$CKPT_DIR/$model/trtllm_ckpt
TRTLLM_ENGINE_DIR=$CKPT_DIR/$model/trtllm_engine

VOCODER_ONNX_PATH=$CKPT_DIR/vocos_vocoder.onnx
VOCODER_TRT_ENGINE_PATH=$CKPT_DIR/vocos_vocoder.plan
MODEL_REPO=./model_repo

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    echo "Downloading F5-TTS from huggingface"
    huggingface-cli download SWivid/F5-TTS $model/model_*.* $model/vocab.txt --local-dir $CKPT_DIR
fi

ckpt_file=$(ls $CKPT_DIR/$model/model_*.* 2>/dev/null | sort -V | tail -1)  # default select latest update
vocab_file=$CKPT_DIR/$model/vocab.txt

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    echo "Converting checkpoint"
    python3 scripts/convert_checkpoint.py \
        --pytorch_ckpt $ckpt_file \
        --output_dir $TRTLLM_CKPT_DIR --model_name $model
    python_package_path=/usr/local/lib/python3.12/dist-packages
    cp -r patch/* $python_package_path/tensorrt_llm/models
    trtllm-build --checkpoint_dir $TRTLLM_CKPT_DIR \
      --max_batch_size 8 \
      --output_dir $TRTLLM_ENGINE_DIR --remove_input_padding disable
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    echo "Exporting vocos vocoder"
    python3 scripts/export_vocoder_to_onnx.py --vocoder vocos --output-path $VOCODER_ONNX_PATH
    bash scripts/export_vocos_trt.sh $VOCODER_ONNX_PATH $VOCODER_TRT_ENGINE_PATH
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    echo "Building triton server"
    rm -r $MODEL_REPO
    cp -r ./model_repo_f5_tts $MODEL_REPO
    python3 scripts/fill_template.py -i $MODEL_REPO/f5_tts/config.pbtxt vocab:$vocab_file,model:$ckpt_file,trtllm:$TRTLLM_ENGINE_DIR,vocoder:vocos
    cp $VOCODER_TRT_ENGINE_PATH $MODEL_REPO/vocoder/1/vocoder.plan
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    echo "Starting triton server"
    tritonserver --model-repository=$MODEL_REPO
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
    echo "Testing triton server"
    num_task=1
    split_name=wenetspeech4tts
    log_dir=./tests/client_grpc_${model}_concurrent_${num_task}_${split_name}
    rm -r $log_dir
    python3 client_grpc.py --num-tasks $num_task --huggingface-dataset yuekai/seed_tts --split-name $split_name --log-dir $log_dir
fi

if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
    echo "Testing http client"
    audio=../../infer/examples/basic/basic_ref_en.wav
    reference_text="Some call me nature, others call me mother nature."
    target_text="I don't really care what you call me. I've been a silent spectator, watching species evolve, empires rise and fall. But always remember, I am mighty and enduring."
    python3 client_http.py --reference-audio $audio --reference-text "$reference_text" --target-text "$target_text" --output-audio "./tests/client_http_$model.wav"
fi

if [ $stage -le 7 ] && [ $stop_stage -ge 7 ]; then
    echo "TRT-LLM: offline decoding benchmark test"
    batch_size=2
    split_name=wenetspeech4tts
    backend_type=trt
    log_dir=./tests/benchmark_${model}_batch_size_${batch_size}_${split_name}_${backend_type}
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
fi

if [ $stage -le 8 ] && [ $stop_stage -ge 8 ]; then
    echo "Native Pytorch: offline decoding benchmark test"
    if ! python3 -c "import f5_tts" &> /dev/null; then
        pip install -e ../../../../
    fi
    batch_size=1  # set attn_mask_enabled=True if batching in actual use case
    split_name=wenetspeech4tts
    backend_type=pytorch
    log_dir=./tests/benchmark_${model}_batch_size_${batch_size}_${split_name}_${backend_type}
    rm -r $log_dir
    torchrun --nproc_per_node=1 \
    benchmark.py --output-dir $log_dir \
    --batch-size $batch_size \
    --split-name $split_name \
    --enable-warmup \
    --model-path $ckpt_file \
    --vocab-file $vocab_file \
    --backend-type $backend_type \
    --tllm-model-dir $TRTLLM_ENGINE_DIR || exit 1
fi