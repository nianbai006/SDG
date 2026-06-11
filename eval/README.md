# SDG-Eval — Evaluation Suite

Implements the five SDG-Eval metrics reported in Section 5    of the paper:

| metric        | description                                                     |
|---------------|-----------------------------------------------------------------|
| `DetTypeF1`   | Per-defect-type detection F1 (artifact / misalignment)          |
| `ClnAcc`      | Clean-image classification accuracy                              |
| `BoxF1@0.1`   | Bounding-box F1 at IoU ≥ 0.1                                    |
| `BoxF1@0.5`   | Bounding-box F1 at IoU ≥ 0.5                                    |
| `DescCos`     | Description cosine similarity (Qwen3-Embedding-0.6B)            |
| `ImpAcc`      | Importance-bucket accuracy (Negligible / Moderate / Critical)   |

Hungarian matching uses 1 − DIoU as the assignment cost (Section 4    of the paper).
Human inter-annotator agreement on the 1,151-image test split — `BoxF1@0.5 = 0.278`
(artifact) and `0.409` (misalignment) — is the localization upper bound.

## Layout

```
eval/
├── code/                      # SDG-Eval metric implementations
│   ├── evaluate.py            # main runner
│   ├── compute_label_desc_metrics.py
│   ├── eval_imdoc_richhf.py   # ImageDoctor cross-dataset evaluation
│   └── eval_sdg_richhf.py     # SDG cross-dataset evaluation
└── README.md
```

## Usage

```bash
# Compute all metrics on the downloaded sample.
python code/evaluate.py \
    --pred predictions/predictions.jsonl \
    --gt   $SDG_DATA/sample_200/annotations.jsonl
```

## Cross-dataset transfer

`code/eval_sdg_richhf.py` runs the SDG detector on RichHF-18K and reports a
1-to-1 metric comparison with ImageDoctor. It requires the RichHF-18K test split
and the released SDG assets (`P1n3/SDG-30K`, `P1n3/sdg-detector-sft`, and the
merged GRPO detector `P1n3/sdg-detector-grpo`).
