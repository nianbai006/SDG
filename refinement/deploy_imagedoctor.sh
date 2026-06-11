#!/bin/bash
# ImageDoctor service deployment script
# Used by mode3: ImageDoctor-feedback editing
#
# ImageDoctor (GYX97/ImageDoctor) makeuse  HuggingFace Transformers load
# Deploy via the bundled imagedoctor_server.py as an OpenAI-compatible HTTP API
# Supports device_map="auto" to span multiple GPUs

# ==================== argconfig ====================
MODEL_PATH=${MODEL_PATH:-"GYX97/ImageDoctor"}
PORT=${PORT:-17141}
TP_SIZE=${TP_SIZE:-2}
HOST=${HOST:-"0.0.0.0"}
MODEL_NAME=${MODEL_NAME:-"ImageDoctor"}
PYTHON=${PYTHON:-"python"}

# ==================== NCCL config ====================
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=WARN
export NCCL_IB_ECE_ENABLE=0
export NCCL_RUNTIME_CONNECT=0
export NCCL_NVLS_ENABLE=0

# export http_proxy=""
# export https_proxy=""
# export HTTP_PROXY=""
# export HTTPS_PROXY=""
# export no_proxy="localhost,127.0.0.1,0.0.0.0"

# ==================== launchservice ====================
echo "=========================================="
echo "Starting ImageDoctor Server"
echo "=========================================="
echo "Model:  $MODEL_PATH"
echo "Port:   $PORT"
echo "TP:     $TP_SIZE"
echo "Name:   $MODEL_NAME"
echo "=========================================="

$PYTHON imagedoctor_server.py \
    --model_path $MODEL_PATH \
    --port $PORT \
    --host $HOST
