#!/bin/bash
# Run this on Server A: starts BBox + UR2 sglang servers
set -euo pipefail

export HF_HOME=${SDG_HOME}/../cache/huggingface
export no_proxy='localhost,127.0.0.1,0.0.0.0'


# Kill old servers
pkill -9 -f sglang 2>/dev/null || true
sleep 3
echo "Old processes cleaned"

# BBox server: GPUs 0-3, port 17142
tmux kill-session -t bbox_server 2>/dev/null || true
tmux new-session -d -s bbox_server "
export CUDA_VISIBLE_DEVICES=0,1,2,3
export no_proxy='localhost,127.0.0.1,0.0.0.0'
${SDG_HOME}/../envs/sglang/bin/python -m sglang.launch_server \
    --model-path ${SDG_CKPT}/sdg_detector_merged \
    --port 17142 --tp 4 --host 0.0.0.0 --trust-remote-code \
    --served-model-name sdg-detector --chat-template qwen2-vl \
    --api-key flowgrpo --mem-fraction-static 0.85 2>&1 | tee /tmp/bbox_server.log
"

# UR2 server: GPUs 4-7, port 17141
tmux kill-session -t ur2_server 2>/dev/null || true
tmux new-session -d -s ur2_server "
export CUDA_VISIBLE_DEVICES=4,5,6,7
export no_proxy='localhost,127.0.0.1,0.0.0.0'
${SDG_HOME}/../envs/sglang/bin/python -m sglang.launch_server \
    --model-path CodeGoat24/UnifiedReward-2.0-qwen3vl-2b \
    --port 17141 --tp 4 --host 0.0.0.0 --trust-remote-code \
    --api-key flowgrpo --mem-fraction-static 0.85 2>&1 | tee /tmp/ur2_server.log
"

echo "================================================"
echo "BBox server: tmux attach -t bbox_server (port 17142)"
echo "UR2 server:  tmux attach -t ur2_server  (port 17141)"
echo "================================================"
echo "Wait ~2 min for servers to be ready, then check:"
echo "  grep 'fired up' /tmp/bbox_server.log /tmp/ur2_server.log"
