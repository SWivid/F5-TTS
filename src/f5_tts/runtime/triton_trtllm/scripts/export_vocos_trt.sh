#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

TRTEXEC="/usr/src/tensorrt/bin/trtexec"

ONNX_PATH=$1
ENGINE_PATH=$2
echo "ONNX_PATH: $ONNX_PATH"
echo "ENGINE_PATH: $ENGINE_PATH"
PRECISION="fp32"


MIN_BATCH_SIZE=1
OPT_BATCH_SIZE=1
MAX_BATCH_SIZE=8

MIN_INPUT_LENGTH=1
OPT_INPUT_LENGTH=1000
MAX_INPUT_LENGTH=3000

MEL_MIN_SHAPE="${MIN_BATCH_SIZE}x100x${MIN_INPUT_LENGTH}"
MEL_OPT_SHAPE="${OPT_BATCH_SIZE}x100x${OPT_INPUT_LENGTH}"
MEL_MAX_SHAPE="${MAX_BATCH_SIZE}x100x${MAX_INPUT_LENGTH}"

${TRTEXEC} \
    --minShapes="mel:${MEL_MIN_SHAPE}" \
    --optShapes="mel:${MEL_OPT_SHAPE}" \
    --maxShapes="mel:${MEL_MAX_SHAPE}" \
    --onnx=${ONNX_PATH} \
    --saveEngine=${ENGINE_PATH}

