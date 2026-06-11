#!/bin/bash
# Eval FLUX.1-dev experiment: gen + reference T2I metrics + forensic, every 60 steps
# ENV vars: EXP, SAVES_ROOT, STEPS
set -uo pipefail

cd ${SDG_HOME}/Flow-Factory
export PATH=${SDG_HOME}/../envs/flow-factory/bin:$PATH
export PYTHONPATH=${SDG_HOME}/Flow-Factory/src:${PYTHONPATH:-}


FF_PY=${SDG_HOME}/../envs/flow-factory/bin/python
FG_PY=${SDG_HOME}/../envs/flow_grpo/bin/python
SCRIPTS=scripts/dense_grpo
EVAL_ROOT=${SDG_HOME}/Flow-Factory/logs/eval/drawbench

EXP="${EXP:?must set EXP}"
SAVES_ROOT="${SAVES_ROOT:-${SDG_HOME}/Flow-Factory/saves/dense_grpo}"
STEPS="${STEPS:-50 100 150 200 250 300 350 400}"

echo "=== EXP=$EXP STEPS=$STEPS ==="

for step in $STEPS; do
    echo ""; echo "=== step=$step ==="
    LORA=$SAVES_ROOT/$EXP/checkpoints/checkpoint-$step
    [ ! -d "$LORA" ] && echo "SKIP missing" && continue

    OUT=$EVAL_ROOT/$EXP/checkpoint-$step
    IMG=$OUT/images
    RJ=$OUT/results.json

    # Gen
    if [ ! -f "$IMG/0998.png" ]; then
        echo "[1/3] Generating..."
        $FF_PY $SCRIPTS/eval_generate.py --model_type flux1 --lora_path "$LORA" --output_dir "$IMG" --num_gpus 8
    else
        echo "[1/3] Images exist"
    fi

    # reference T2I metrics
    if [ ! -f "$RJ" ]; then
        echo "[2/3] Metrics..."
        bash $SCRIPTS/eval_metrics_split.sh "$IMG" "$RJ" 0
    else
        echo "[2/3] Metrics exist"
    fi

    # Forensic
    has_fc=$(python3 -c "import json; d=json.load(open('$RJ')); print('yes' if 'forensic_chat' in d else 'no')" 2>/dev/null || echo "no")
    if [ "$has_fc" = "no" ]; then
        echo "[3/3] Forensic..."
        $FG_PY $SCRIPTS/eval_forensic_chat.py --images_dir "$IMG" --output_file "$RJ" --device cuda:0
    else
        echo "[3/3] Forensic exists"
    fi
    echo "=== DONE step=$step ==="
done
echo "=== ALL DONE ==="
