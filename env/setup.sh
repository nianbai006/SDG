#!/usr/bin/env bash
# One-shot environment setup for the SDG open-source release.
#
# Creates up-to-six conda environments, one per pipeline component. Envs are
# created at ${SDG_HOME}/../envs/<name>/ via `conda create -p` — this matches
# the absolute paths embedded in the shipped scripts (e.g. start_servers_A.sh
# does `${SDG_HOME}/../envs/sglang/bin/python ...`).
#
# Usage:
#   bash env/setup.sh                  # all six envs (default)
#   bash env/setup.sh core             # only sdg-core
#   bash env/setup.sh core sglang      # subset
#   bash env/setup.sh quickstart       # alias: core + sglang + flow-factory
#
# Targets ↔ paper sections:
#
#   target           env path (under ${SDG_HOME}/../envs/)   used by
#   ---------------- --------------------------------------- ----------------------------
#   core             sdg-core/                               §3 dataset + §5 SDG-Eval (Table 1)
#   flow-factory     flow-factory/                           §4 BoxFlow-GRPO training (Table 4)
#   flow_grpo        flow_grpo/                              §4 reference T2I metrics (Table 4)
#   sglang           sglang/                                 §4 reward serving on Server A

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")"/.. && pwd)"
export SDG_HOME="${SDG_HOME:-$REPO_ROOT}"
export SDG_DATA="${SDG_DATA:-$REPO_ROOT/data}"
export SDG_CKPT="${SDG_CKPT:-$REPO_ROOT/checkpoints}"
export ENVS_ROOT="${ENVS_ROOT:-$SDG_HOME/../envs}"

echo "SDG_HOME=$SDG_HOME"
echo "SDG_DATA=$SDG_DATA"
echo "SDG_CKPT=$SDG_CKPT"
echo "ENVS_ROOT=$ENVS_ROOT"
mkdir -p "$ENVS_ROOT"

# Map target → (env_name, requirements_file)
declare -A REQS=(
    [core]="sdg-core:requirements.txt"
    [flow-factory]="flow-factory:requirements_flow_factory.txt"
    [flow_grpo]="flow_grpo:requirements_flow_grpo.txt"
    [sglang]="sglang:requirements_sglang.txt"
)

ALL_TARGETS=(core flow-factory flow_grpo sglang)
QUICKSTART=(core sglang flow-factory)

create_env() {
    local target="$1"
    local spec="${REQS[$target]:-}"
    if [[ -z "$spec" ]]; then
        echo "[setup] unknown target: $target" >&2
        exit 1
    fi
    local name="${spec%%:*}"
    local reqs="${spec##*:}"
    local prefix="$ENVS_ROOT/$name"

    if [[ -d "$prefix" ]]; then
        echo "[setup] '$name' exists at $prefix — updating in place."
    else
        echo "[setup] creating '$name' at $prefix"
        conda create -y -p "$prefix" python=3.10
    fi
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$prefix"
    pip install --upgrade pip
    pip install -r "$REPO_ROOT/env/$reqs"
    # Flow-Factory needs editable install    of the local source tree.
    if [[ "$name" == "flow-factory" && -f "$SDG_HOME/boxflow_grpo/pyproject.toml" ]]; then
        pip install -e "$SDG_HOME/boxflow_grpo"
    fi
    conda deactivate
}

if [[ $# -eq 0 ]] || [[ "$1" == "all" ]]; then
    TARGETS=("${ALL_TARGETS[@]}")
elif [[ "$1" == "quickstart" ]]; then
    TARGETS=("${QUICKSTART[@]}")
else
    TARGETS=("$@")
fi

echo "[setup] targets: ${TARGETS[*]}"
for t in "${TARGETS[@]}"; do
    create_env "$t"
done

cat <<EOF

[setup] done. Created envs:
$(ls "$ENVS_ROOT" 2>/dev/null | sed 's/^/  - /')

Next steps:
  Download data/checkpoints from https://huggingface.co/your-org/sdg
  export SDG_DATA=$SDG_HOME/data
  export SDG_CKPT=$SDG_HOME/checkpoints
EOF
