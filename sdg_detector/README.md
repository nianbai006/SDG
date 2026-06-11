# SDG Detector — Structured Defect Grounding (Qwen3-VL-4B)

Two-stage training pipeline for the SDG defect detector reported in Section 5
   of the paper. Outputs structured
`<think>...</think><answer>[{box_2d, label, desc}]</answer>` predictions over
text-to-image samples.

Both stages drive the **`ms-swift`** CLI; we ship no in-tree trainer. The
main launchers live in `scripts/`. All ablation, model-size variant, and
exploratory launchers have been pruned.

## Layout

```
sdg_detector/
├── README.md
├── configs/                    DeepSpeed ZeRO-2 launch config
├── inference/
│   ├── eval_qwen.py            batched inference (SGLang)
│   ├── eval_gemini_thinkpe_imp.py   Gemini-3-Pro zero-shot baseline (Table 1)
│   ├── eval_gpt_thinkpe_imp.py      GPT-5.4    zero-shot baseline (Table 1)
│   └── visualize_predictions_qwen.py
├── scripts/
│   ├── train_sft.sh       Stage 1 — paper SFT run (16 × A100-80G)
│   ├── train_grpo.sh      Stage 2 — paper GRPO run
│   ├── plugin.py                 ms-swift reward plugin (composite reward)
│   ├── prepare_coord_jitter_data.py jitter ±10 px, repeat 3 epochs, shuffle
│   └── README.md
├── data_prep/                  build SDG-30K JSONLs from raw annotations
├── preprocess/                 Gemini-3-Pro distillation + bbox normalization
│   └── swift/                  ms-swift dataset format converters
└── utils/                      jsonl balancing, filtering, viz helpers
```

## Quick start

Download `P1n3/SDG-30K`, `P1n3/sdg-detector-sft`, and the directly loadable
GRPO detector checkpoint `P1n3/sdg-detector-grpo`. The GRPO checkpoint is a
merged full checkpoint, not a LoRA adapter.

```bash
# Inference with the released GRPO detector checkpoint.
python inference/eval_qwen.py \
    --model_path        $SDG_CKPT/sdg-detector-grpo \
    --eval_dataset_path $SDG_DATA/SDG-30K/annotations/test.jsonl \
    --output_dir        predictions/ \
    --tp_size 1
```

## Reproducing training

### Stage 1 — SFT (1 epoch, 5,360 steps, 16 × A100-80G, LR 3e-5)

```bash
# Master node:
bash scripts/train_sft.sh 0
# Worker node:
bash scripts/train_sft.sh 1
```

Coordinate jitter (±10 px, per-epoch resampling) is applied at data prep time
by `scripts/prepare_coord_jitter_data.py`. Output: `sft_jitter_shuffled.jsonl`
(85,770 samples).

### Stage 2 — GRPO (2 epochs, 16 × A100-80G, LR 5e-6, 8 rollouts/prompt)

```bash
# Master:
bash scripts/train_grpo.sh 0
# Worker:
bash scripts/train_grpo.sh 1
```

Composite reward `0.6 × DIoU + 0.25 × DescCos + 0.15 × ImpAcc` is implemented
in `scripts/plugin.py` (an `ms-swift` external plugin loaded via
`--external_plugins`). The same plugin is reused by the BoxFlow-GRPO reward
server (`../boxflow_grpo/`).

## Baselines (Table 1 zero-shot rows)

```bash
python inference/eval_gpt_thinkpe_imp.py    \
    --image_dir $SDG_DATA/sample_200/images \
    --output    pred_gpt.jsonl
python inference/eval_gemini_thinkpe_imp.py \
    --image_dir $SDG_DATA/sample_200/images \
    --output    pred_gemini.jsonl
```

Both call the respective public APIs; set `OPENAI_API_KEY` / `GEMINI_API_KEY`
in your environment.

## Output format

```
<think>
Step 1: Caption Understanding ...
Step 2: Visual Analysis & Defect Spotting ...
Step 3: Localization ...
</think>
<answer>
[{"box_2d": [y0, x0, y1, x1], "label": "artifact", "desc": "..."},
 ...]
</answer>
```

`box_2d` uses the `[0, 1000]` normalized coordinate convention (top, left,
bottom, right).

## Notes

- Paths in shipped scripts use `${SDG_HOME}` / `${SDG_DATA}` / `${SDG_CKPT}`
  placeholders; export them via `env/setup.sh` before running.
- `env/setup.sh core` installs the verified `ms-swift` stack. Put
  `$ENVS_ROOT/sdg-core/bin` on `PATH` so the `swift` CLI resolves.
- The released GRPO detector checkpoint is already merged with the SFT
  checkpoint and can be loaded directly with `transformers`; do not load it as
  a PEFT adapter.
