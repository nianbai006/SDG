# Environment Setup

The pipeline spans **four** isolated conda environments. They're isolated
because their dependencies clash (different `transformers` / `torch` /
`vllm` / `sglang` pins). Each env is created at
`${SDG_HOME}/../envs/<name>/` so the absolute paths baked into the shipped
shell scripts work without modification.

## Environment ↔ paper section

| env name        | requirements file                       | what it runs                                                                  | paper                |
|-----------------|-----------------------------------------|-------------------------------------------------------------------------------|----------------------|
| `sdg-core`      | `requirements.txt`                      | SDG detector training / inference, SDG-Eval metrics, refinement Phase 1 + 2   | §3, §5 (Table 1)     |
| `flow-factory`  | `requirements_flow_factory.txt`         | BoxFlow-GRPO `ff-train`, PickScore                                            | §4 (Table 4)         |
| `flow_grpo`     | `requirements_flow_grpo.txt`            | ImageReward + CLIPScore + LAION aesthetic reference metrics                   | §4 (Table 4)         |
| `sglang`        | `requirements_sglang.txt`               | SGLang reward server (SDG bbox + UR2) on Server A                             | §4 (RL training)     |

## Quickstart

```bash
# Smallest env set that runs inference + SDG-Eval after downloading assets.
bash env/setup.sh core

# To also reproduce Table 4 (BoxFlow-GRPO):
bash env/setup.sh quickstart        # alias for: core + sglang + flow-factory

# Full pipeline (all four envs, ~20 GB on disk):
bash env/setup.sh                   # default = all
bash env/setup.sh all               # explicit

# Custom subset:
bash env/setup.sh core flow-factory
```

The script is idempotent — re-running on an existing env just updates
packages.

## Why four?

| conflict                                      | explains the split                            |
|-----------------------------------------------|-----------------------------------------------|
| `transformers` 4.46 (Qwen3-VL) vs 4.40 (ImageReward) | core / flow_grpo forks                  |
| `vllm` + `sglang` heavy CUDA pins             | sglang isolated for serving                   |
| `flow-factory` patches `diffusers` source     | flow-factory isolated                         |

## Disk + compute

- All envs together occupy **~20 GB**.
- `sdg-core`, `flow-factory`, `sglang` each pull torch 2.x + CUDA wheels (~5–8 GB).
- `flow_grpo` is smaller (~2–4 GB).
- Installation time on a fresh machine: **~20 min** for the full set, ~5 min
  for `core` alone.

## Verifying

```bash
# Each env should have a usable python:
for e in sdg-core flow-factory flow_grpo sglang; do
    "$ENVS_ROOT/$e/bin/python" -c "import sys; print('$e', sys.version)"
done
```

Data and checkpoints are intentionally not committed to Git. Download the
released assets from HuggingFace and set `SDG_DATA` / `SDG_CKPT` accordingly:

- `P1n3/SDG-30K`
- `P1n3/sdg-detector-sft`
- `P1n3/sdg-detector-grpo` (merged GRPO detector checkpoint)
- `P1n3/boxflow-grpo-flux-lora`
