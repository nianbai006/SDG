#!/bin/bash
# SDG service deployment script
# Used by mode4: SDG-feedback editing

# ==================== argconfig ====================
MODEL_PATH=${MODEL_PATH:-"${SDG_CKPT}/sdg_detector_merged"}
PORT=${PORT:-17142}
TP_SIZE=${TP_SIZE:-2}
HOST=${HOST:-"0.0.0.0"}
MODEL_NAME=${MODEL_NAME:-"sdg-detector"}
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
echo "Starting SDG Server"
echo "=========================================="
echo "Model:  $MODEL_PATH"
echo "Port:   $PORT"
echo "TP:     $TP_SIZE"
echo "Name:   $MODEL_NAME"
echo "=========================================="

$PYTHON -m sglang.launch_server \
    --model-path $MODEL_PATH \
    --port $PORT \
    --tp $TP_SIZE \
    --host $HOST \
    --trust-remote-code \
    --served-model-name $MODEL_NAME \
    --chat-template qwen2-vl
