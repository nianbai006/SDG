# BoxFlow-GRPO — Spatial-Reward RL for Diffusion Alignment

Implementation    of **BoxFlow-GRPO** (paper Section 4.2): a Group-Relative Policy
Optimization variant that converts the SDG detector's predicted bounding boxes
into spatially-localized reward maps and aligns FLUX.1-dev via LoRA.

This directory is a fork    of the open-source [Flow-Factory](README_UPSTREAM.md)
codebase (RL training framework for diffusion / flow-matching models). The
paper's BoxFlow-GRPO algorithm is implemented as the **`dense-grpo`** trainer
plus the **`CombinedUR2BBoxReward`** reward — both ship with this fork.

## Layout

boxflow_grpo/
├── src/flow_factory/         core library
│   ├── trainers/
│   │   ├── dense_grpo.py     ← BoxFlow-GRPO trainer (paper §4.2)
│   │   ├── grpo.py
│   │   └── awm.py
│   ├── rewards/
│   │   ├── dense_grpo_rewards.py   ← CombinedUR2BBoxReward (paper §4.2 reward)
│   │   ├── ur2_scalar.py
│   │   ├── imagedoctor.py
│   │   └── ...
│   ├── models/               adapters for FLUX, SD3, Z-Image, Qwen-Image, etc.
│   ├── samples/              sampling utilities
│   ├── advantage/            group / dense advantage computation
│   ├── data_utils/           prompt-batch sampling
│   └── ...
├── flux1_dev_exp1_AB.yaml    paper run config (Table 4 row "Ours")
├── scripts/                  launch wrappers
├── pyproject.toml            install via `pip install -e .` (registers `ff-train`)
├── LICENSE                   Apache-2.0 (Flow-Factory)
├── README_UPSTREAM.md        original Flow-Factory README, retained for attribution
└── README.md                 this file

## Canonical paper run (Table 4 row "Ours")

```bash
pip install -e .
ff-train flux1_dev_exp1_AB.yaml
```

Key hyperparameters (mirrors `flux1_dev_exp1_AB.yaml`):

| field             | value                                 |
| ----------------- | ------------------------------------- |
| base model        | `black-forest-labs/FLUX.1-dev`      |
| trainer           | `dense-grpo`                        |
| reward            | `CombinedUR2BBoxReward`             |
| α (artifact)     | 0.5                                   |
| α (misalignment) | 0.05                                  |
| LoRA rank / α    | r=64, α=128                          |
| resolution        | 512×512                              |
| inference steps   | 10 (ODE-SDE hybrid, SDE window [0,5]) |
| guidance scale    | 3.5                                   |
| group size        | 16                                    |
| epochs            | 600 (paper run saved at epoch 570)    |
| hardware          | 8 × A100-80G                         |

## Deployment

Two nodes, 8 × A100-80G each:

- **Server A — reward services.** `CombinedUR2BBoxReward` calls two
  HTTP services: a UnifiedReward-2.0 scalar scorer (GPUs 4-7, port 17141)
  and the SDG bbox detector (GPUs 0-3, port 17142, loads the released
  merged SDG-GRPO checkpoint). Both boot via one command:

  ```bash
  BBOX_MODEL=$SDG_CKPT/sdg-detector-grpo bash scripts/dense_grpo/start_servers_A.sh
  ```
- **Server B — training.** Once the reward endpoints are reachable,
  launch RL training (8 GPUs):

  ```bash
  bash scripts/dense_grpo/train_B.sh           # uses flux1_dev_exp1_AB.yaml
  ```

The training config points at the A-side endpoints via `reward_url`
fields — update them to A's reachable IP before launching.

## Method outline (paper §4.2)

```
┌─ FLUX.1-dev (LoRA) ──→ image x_t
│                          │
│                          ▼
│                  SDG detector (reward server)
│                          │
│                  boxes {(b_i, type_i, desc_i)}
│                          │
│                          ▼
│           reward map  R(x, y) = Σ_i w_i · g(x, y; b_i)
│                          │
│                          ▼
│   per-location advantage  A_loc = R - 𝔼_{group}[R]
│                          │
└────── GRPO update ←──────┘
```

`w_i` weights artifact / misalignment differently (`α_artifact = 0.5`,
`α_misalignment = 0.05`). `g(·)` is a sof t mask that lights up the box
interior. See `src/flow_factory/rewards/dense_grpo_rewards.py` for the exact
implementation.

## Released LoRA

The BoxFlow-GRPO LoRA checkpoint is released at
[`P1n3/boxflow-grpo-flux-lora`](https://huggingface.co/P1n3/boxflow-grpo-flux-lora).
After download, the expected local path is
`$SDG_CKPT/boxflow-grpo-flux-lora`.

## Notes

- Heavy training artifacts (`wandb/`, `logs/`, `saves/`) are excluded from the
  Git repository.
- The Flow-Factory library supports many other diffusion / flow-matching
  models (SD3, Z-Image, Qwen-Image, FLUX-Kontext, …); only the FLUX.1-dev
  config is load-bearing for the paper's headline result.
- `README_UPSTREAM.md` is the original Flow-Factory README, retained for
  attribution.
