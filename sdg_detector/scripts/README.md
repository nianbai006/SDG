# Canonical Training Scripts

These are the **exact** shell launchers used to produce the SDG checkpoints
released with the project. They are the load-bearing scripts for the paper's
detector training runs.

## What you get

| script                            | output                                   | hardware            |
|-----------------------------------|------------------------------------------|---------------------|
| `train_sft.sh`          | `sdg_detector_sft` checkpoint  | 16 × A100-80G (2×8) |
| `train_grpo.sh`         | `sdg_detector_grpo_lora` checkpoint | 16 × A100-80G (2×8) |
| `prepare_coord_jitter_data.py`    | the SFT input JSONL with ±10 px jitter applied per epoch | CPU |

Run from a 2-node cluster (one master, one worker), passing `0` on the master
and `1` on the worker:

```bash
# Master node
bash train_sft.sh 0
# Worker node
bash train_sft.sh 1
```

GRPO is launched the same way **after** the SFT checkpoint exists.

## Configuration knobs

Both scripts read paths from environment variables, which `env/setup.sh` exports:

| variable            | meaning                                            |
|---------------------|----------------------------------------------------|
| `SDG_HOME`          | repository root                                    |
| `SDG_DATA`          | dataset root (downloaded SDG-30K)                  |
| `SDG_CKPT`          | checkpoint root (downloaded SDG-SFT for Stage 2)   |
| `MASTER_ADDR`       | the master node's reachable IP                     |
| `MASTER_PORT`       | rendezvous port (default 29501 / 29513)            |

## Cluster notes

Set `MASTER_ADDR` before launch. If your cluster needs a custom NCCL build,
set `LD_PRELOAD` yourself; the scripts do not hard-code internal NCCL paths.

## See also

- `train_sft.sh` and `train_grpo.sh` both wrap the ms-swift framework.
- Download released checkpoints and datasets from `P1n3/SDG-30K`,
  `P1n3/sdg-detector-sft`, and the merged GRPO detector
  `P1n3/sdg-detector-grpo`.
