#!/bin/bash
# ============================================================
# V5 Jitter SFT — 2 nodes x 8 GPUs = 16 GPUs
# Data is globally shuffled (sft_jitter_3epoch_shuffled.jsonl)
# Usage:
#   Master: MASTER_ADDR=MASTER_NODE_IP bash train_sft.sh 0
#   Worker: MASTER_ADDR=MASTER_NODE_IP bash train_sft.sh 1
# ============================================================
set -e

NNODES=2
NPROC_PER_NODE=8
MASTER_ADDR=${MASTER_ADDR:?Set MASTER_ADDR to the master node IP}
MASTER_PORT=${MASTER_PORT:-29501}

NODE_RANK_ARG=$1
if [ "$NODE_RANK_ARG" == "0" ]; then
    NODE_RANK=0; NODE_DESC="Master"
elif [ "$NODE_RANK_ARG" == "1" ]; then
    NODE_RANK=1; NODE_DESC="Worker"
else
    echo "Usage: bash $0 <0|1>"; exit 1
fi

export NCCL_IB_DISABLE=0
export NCCL_DEBUG=WARN
export NCCL_IB_ECE_ENABLE=0
export NCCL_RUNTIME_CONNECT=0
export NCCL_NVLS_ENABLE=0

lr=3e-5

PROJECT_ROOT="${SDG_HOME}"
EXPERIMENT_ROOT="${SDG_HOME}/sdg_detector/scripts"

RUN_NAME="sft_2.5v5_$(date +%m%d_%H%M)_jitter10_shuffled_lr${lr}-16gpu-freezevit-thinkpe_imp"
CHECKPOINT_PATH="${PROJECT_ROOT}/outputs/2.5v5"
DATASET_PATH="${SDG_DATA}/sdg30k/annotations/sft_jitter_shuffled.jsonl"
MODEL_PATH="Qwen/Qwen3-VL-4B-Instruct"
OUTPUT_DIR="${CHECKPOINT_PATH}/${RUN_NAME}"

export WANDB_PROJECT="${WANDB_PROJECT:-SDG}"
export WANDB_NAME="${RUN_NAME}"
export WANDB_TAGS="swift,sft,qwen3vl,sdg,2.5v5,jitter,shuffled,16gpu,thinkpe_imp"
export WANDB_NOTES="V5 jitter±10, globally shuffled, 16GPU 2-node, 3 epochs in data"
export WANDB_DIR="${OUTPUT_DIR}"

mkdir -p ${OUTPUT_DIR}

echo "============================================"
echo "V5 Jitter SFT (16 GPU) — Node $NODE_RANK_ARG ($NODE_DESC)"
echo "============================================"
echo "Run Name:   ${RUN_NAME}"
echo "Output Dir: ${OUTPUT_DIR}"
echo "Dataset:    ${DATASET_PATH} (85770 samples, shuffled)"
echo "LR:         ${lr}"
echo "Epochs:     1 (3 epochs pre-baked in data)"
echo "============================================"

MAX_PIXELS=1280000 \
MIN_PIXELS=200704 \
NNODES=$NNODES \
NODE_RANK=$NODE_RANK \
MASTER_ADDR=$MASTER_ADDR \
MASTER_PORT=$MASTER_PORT \
NPROC_PER_NODE=${NPROC_PER_NODE} \
swift sft \
    --model "${MODEL_PATH}" \
    --dataset "${DATASET_PATH}" \
    --train_type full \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate ${lr} \
    --gradient_accumulation_steps 1 \
    --eval_steps 2000 \
    --save_steps 2000 \
    --save_total_limit 10 \
    --logging_steps 1 \
    --max_length 5100 \
    --output_dir "${OUTPUT_DIR}" \
    --warmup_ratio 0.05 \
    --lr_scheduler_type cosine \
    --dataloader_num_workers 4 \
    --gradient_checkpointing true \
    --freeze_vit true \
    --attn_impl flash_attn \
    --report_to wandb \
    --deepspeed zero2 \
    --use_hf true

echo "SFT completed! Node $NODE_RANK_ARG ($NODE_DESC). Output: ${OUTPUT_DIR}"
