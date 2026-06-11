#!/bin/bash
# ============================================================
# V5 GRPO v5 shuffled — 2 nodes x 8 GPUs = 16 GPUs
# Based on v4_safe, with:
#   - Shuffled GRPO data
#   - SFT base: V5 shuffled 16gpu jitter checkpoint-5360
# Usage:
#   Master: MASTER_ADDR=MASTER_NODE_IP bash train_grpo.sh 0
#   Worker: MASTER_ADDR=MASTER_NODE_IP bash train_grpo.sh 1
# ============================================================
set -euo pipefail

export PYTHONPATH=${PYTHONPATH:-}:${SDG_HOME}/ms-swift/

NNODES=2
NPROC_PER_NODE=8
MASTER_ADDR=${MASTER_ADDR:?Set MASTER_ADDR to the master node IP}
MASTER_PORT=${MASTER_PORT:-29513}

NODE_RANK_ARG=${1:-}
if [ "$NODE_RANK_ARG" = "0" ]; then
    NODE_RANK=0; NODE_DESC="Master"
elif [ "$NODE_RANK_ARG" = "1" ]; then
    NODE_RANK=1; NODE_DESC="Worker"
else
    echo "Usage: bash $0 <0|1>"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_LOGGING_LEVEL=WARNING
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=bond0
export GLOO_SOCKET_IFNAME=bond0
export NCCL_IB_ECE_ENABLE=0
export NCCL_RUNTIME_CONNECT=0
export NCCL_NVLS_ENABLE=0

PROJECT_ROOT="${SDG_HOME}"
EXPERIMENT_ROOT="${SDG_HOME}/sdg_detector/scripts"

DATASET_PATH="${SDG_DATA}/sdg30k/annotations/grpo_think_shuffled.jsonl"
MODEL_PATH="${PROJECT_ROOT}/outputs/2.5v5/sft_2.5v5_0322_0147_jitter10_shuffled_lr3e-5-16gpu-freezevit-thinkpe_imp/v0-20260322-014754/checkpoint-5360"
PLUGIN_PATH="${EXPERIMENT_ROOT}/plugin.py"

RUN_NAME="grpo_2.5v5_$(date +%m%d_%H%M)_v5_shuffled_16gpu"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/2.5v5/grpo/${RUN_NAME}"

export WANDB_PROJECT="${WANDB_PROJECT:-SDG}"
export WANDB_NAME="${RUN_NAME}"
export WANDB_TAGS="grpo,qwen3vl,sdg,2.5v5,v5,combined,hungarian,shuffled,16gpu"
export WANDB_NOTES="GRPO v5: v4_safe config + shuffled data + shuffled SFT base (16gpu)"
export WANDB_DIR="${OUTPUT_DIR}"

mkdir -p "${OUTPUT_DIR}"

echo "============================================"
echo "V5 GRPO v5 shuffled — Node $NODE_RANK_ARG ($NODE_DESC)"
echo "  Reward: SDG_combined_v3 (single)"
echo "  Plugin: ${PLUGIN_PATH}"
echo "  Model:  ${MODEL_PATH}"
echo "  Data:   ${DATASET_PATH} (shuffled)"
echo "  Beta:   0.01"
echo "  LR:     5e-6"
echo "  IFACE:  bond0"
echo "============================================"

MAX_PIXELS=1003520 \
NNODES=$NNODES \
NODE_RANK=$NODE_RANK \
MASTER_ADDR=$MASTER_ADDR \
MASTER_PORT=$MASTER_PORT \
NPROC_PER_NODE=$NPROC_PER_NODE \
swift rlhf \
    --rlhf_type grpo \
    --model "$MODEL_PATH" \
    --external_plugins "$PLUGIN_PATH" \
    --reward_funcs SDG_combined_v3 \
    --dataset "$DATASET_PATH" \
    --load_from_cache_file true \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.5 \
    --vllm_tensor_parallel_size 8 \
    --torch_dtype bfloat16 \
    --system "You are a helpful assistant. " \
    --num_train_epochs 2 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --learning_rate 5e-6 \
    --save_steps 200 \
    --save_total_limit 200000 \
    --logging_steps 1 \
    --output_dir "${OUTPUT_DIR}" \
    --gradient_accumulation_steps 2 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --max_completion_length 4096 \
    --vllm_max_model_len 6096 \
    --num_generations 8 \
    --beta 0.01 \
    --attn_impl flash_attn \
    --deepspeed zero2 \
    --report_to wandb \
    --vllm_mm_processor_cache_gb 0 \
    --template qwen3_vl

echo "GRPO v5 shuffled completed! Node $NODE_RANK_ARG ($NODE_DESC). Output: ${OUTPUT_DIR}"
