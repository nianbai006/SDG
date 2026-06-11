#!/bin/bash
# DenseGRPO: Launch training on Server B (8 GPUs)
#
# Prerequisites:
#   1. BBox server running on Server A (port 17142)
#   2. UR2 server running on Server A (port 17141)
#   3. Update YAML config with correct server IPs
#
# Usage: bash scripts/dense_grpo/train_B.sh [config_path]
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CONFIG="${1:-flux1_dev_exp1_AB.yaml}"

# NCCL settings for long reward computation
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=3600
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export NCCL_IB_ECE_ENABLE=0
export NCCL_RUNTIME_CONNECT=0
export NCCL_NVLS_ENABLE=0

# Cache
export HF_HOME=${SDG_HOME}/../cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Proxy: keep existing proxy settings and exclude reward server addresses.
SERVER_A_IP="${SERVER_A_IP:-localhost}"
export no_proxy="localhost,127.0.0.1,0.0.0.0,${SERVER_A_IP}"

RUN_NAME="dense_grpo_$(date +%m%d_%H%M)"
export WANDB_NAME="${RUN_NAME}"

echo "=============================================="
echo "DenseGRPO Training"
echo "Config: ${CONFIG}"
echo "Server A (rewards): ${SERVER_A_IP}"
echo "Run: ${RUN_NAME}"
echo "=============================================="

ff-train "${CONFIG}"

echo ">>> Training done: ${RUN_NAME}"
