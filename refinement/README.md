# Defect-Guided Refinement (Section 5 — `sec:reflection_results`)

Reproduces the paper's GPT-Image-1.5 refinement experiment (Table 5).
Three editor configurations are compared on identical inputs:

| config         | feedback to GPT-Image-1.5                                       |
|----------------|-----------------------------------------------------------------|
| **Fixed**      | original image + caption only (no defect feedback)              |
| **ImageDoctor**| + artifact / misalignment heatmaps + ImageDoctor text feedback  |
| **SDG (ours)** | + bbox-annotated overlay + SDG text feedback                    |

Per Section 5, two annotators independently and blindly assign **Good /
Same / Bad** labels comparing SDG against each baseline on 873 valid
samples retained after GPT-Image-1.5 filtering. The paper reports only
this GSB metric — no automated quality scores are claimed.

## Layout (flat)

```
refinement/
├── README.md                   this file
├── config.py                   global mode-to-service mapping
│
│   ── Phase 1 — feedback inference ──────────────────────────────────
├── infer_imagedoctor.py        ImageDoctor predictions (heatmaps + text)
├── infer_sdg.py                SDG predictions (boxes + text)
├── deploy_imagedoctor.sh       boots the ImageDoctor service
├── deploy_sdg.sh               boots the SDG detector service
├── imagedoctor_server.py       FastAPI wrapper for ImageDoctor
│
│   ── Phase 2 — GPT-Image-1.5 editing ──────────────────────────────
├── edit_gpt_image.py           main editor (SDG / IMDOC modes; ±text ablation)
├── spatial_guide.py            parse bbox list from SDG / ImageDoctor responses
│
│   ── Phase 3 — GSB annotation bundle ──────────────────────────────
├── prepare_human_eval.py       set up the human GSB annotation zip
│
│   ── shared helpers ───────────────────────────────────────────────
├── data_loader.py              test-set / prediction loaders + matchers
├── prompt_builder.py           Fixed / ImageDoctor / SDG edit-prompt builders
├── visualization.py            bbox / heatmap overlay renderers
└── vlm_client.py               HTTP client for ImageDoctor / SDG endpoints
```

## End-to-end reproduction

All paths in the example below assume `pwd` is `refinement/`. Download the
released SDG assets from HuggingFace first (`P1n3/SDG-30K`,
`P1n3/sdg-detector-sft`, and the merged GRPO detector
`P1n3/sdg-detector-grpo`).
Set `IMAGE_EDIT_API_URL`, `IMAGE_EDIT_API_KEY`, and `IMAGE_EDIT_MODEL` for the
image editing API used by `edit_gpt_image.py`.

```bash
# Phase 1A — boot the SDG detector and ImageDoctor reward services.
bash deploy_sdg.sh           &
bash deploy_imagedoctor.sh   &

# Phase 1B — run feedback inference on the 200-image sample.
python infer_sdg.py \
    --test_data_path $SDG_DATA/sample_200/annotations.jsonl \
    --output_dir     outputs/sdg
python infer_imagedoctor.py \
    --test_data_path $SDG_DATA/sample_200/annotations.jsonl \
    --output_dir     outputs/imdoc

# Phase 2 — run the three paper conditions (Fixed / ImageDoctor / SDG)
#           plus the two *_notext appendix ablations.
python edit_gpt_image.py --mode fixed \
    --test_data $SDG_DATA/sample_200/annotations.jsonl \
    --output_dir outputs/v2_gpt_image_fixed_full
python edit_gpt_image.py --mode imdoc \
    --imagedoctor_predictions outputs/imdoc/inference_results.jsonl \
    --output_dir outputs/v2_gpt_image_imdoc_full
python edit_gpt_image.py --mode sdg \
    --sdg_predictions outputs/sdg/inference_results.jsonl \
    --output_dir outputs/v2_gpt_image_sdg_full
python edit_gpt_image.py --mode imdoc --no_text \
    --imagedoctor_predictions outputs/imdoc/inference_results.jsonl \
    --output_dir outputs/v2_gpt_image_imdoc_full_notext
python edit_gpt_image.py --mode sdg --no_text \
    --sdg_predictions outputs/sdg/inference_results.jsonl \
    --output_dir outputs/v2_gpt_image_sdg_full_notext

# Phase 3 — build the human GSB annotation bundle.
python prepare_human_eval.py
```

The Phase 3 output is a zip containing the source / Fixed / ImageDoctor /
SDG images plus a JSON manifest with randomized order; annotators see
randomized triples and assign Good / Same / Bad labels. Aggregating those
labels yields the SDG-vs-baseline GSB rates reported in Table 5.

## Notes

- GPT-Image-1.5 is the editor for **all three** conditions; no
  Qwen-Image-Edit or other editor variants are part of the paper.
- The `*_notext` ablations are reported in the appendix.
- The paper reports only GSB (Good / Same / Bad). No automated reward /
  quality scores (PickScore, ImageReward, HPSv3, DeQA, …) are claimed for
  refinement, so those evaluators are not included in this repository.
