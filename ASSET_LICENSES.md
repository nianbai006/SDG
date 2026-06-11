# Asset Licenses

This file separates the repository code license from the licenses of datasets,
model checkpoints, generated images, APIs, and upstream components.

Repository source code is licensed under Apache-2.0. Large assets are not
committed to Git and should be downloaded from the project HuggingFace
placeholder:

```text
https://huggingface.co/your-org/sdg
```

## Released Project Assets

| asset | location | license |
|---|---|---|
| SDG-30K dataset | HuggingFace placeholder | CC-BY-NC-4.0, subject to upstream asset constraints |
| `sample_200` data sample | HuggingFace placeholder | CC-BY-NC-4.0, subject to upstream asset constraints |
| `sdg_detector_sft` | HuggingFace placeholder | CC-BY-NC-4.0; derivative of `Qwen/Qwen3-VL-4B-Instruct` |
| `sdg_detector_grpo_lora` | HuggingFace placeholder | CC-BY-NC-4.0; derivative of `Qwen/Qwen3-VL-4B-Instruct` |
| `boxflow_grpo_flux_lora` | HuggingFace placeholder | FLUX.1-dev Non-Commercial License; derivative of `black-forest-labs/FLUX.1-dev` |

## Upstream Models

| asset | use | upstream license / terms |
|---|---|---|
| Qwen3-VL-4B-Instruct | SDG detector base model | Apache-2.0 |
| Qwen3-Embedding-0.6B | description similarity for eval/reward | Apache-2.0 |
| FLUX.1-dev | BoxFlow-GRPO base diffusion model | FLUX.1-dev Non-Commercial License |
| Qwen-Image-Edit-2511 | refinement backbone for related experiments | Apache-2.0 |

## T2I Generators Used To Build SDG-30K

| generator | upstream license / terms |
|---|---|
| FLUX.2-dev | FLUX Non-Commercial License |
| Z-Image-Turbo | Apache-2.0 |
| LongCat-Image | Apache-2.0 |
| SANA-1.5-1.6B | NVIDIA Open Model License / model-specific terms |

## Prompt And Evaluation Resources

| asset | use | upstream license / terms |
|---|---|---|
| Pick-a-Pic | source prompt corpus | MIT |
| ImageDoctor | baseline / refinement comparison | upstream public release terms |
| Flow-GRPO | baseline / related RL training | upstream public release terms |
| UnifiedReward-2.0 | reward model | upstream public release terms |
| PickScore, CLIPScore, HPSv3, DeQA, Forensic-Chat, DrawBench, RichHF-18K | baselines and evaluation resources | respective upstream licenses / provider terms |

## API-Only Services

The following services are used only through their official APIs and are not
redistributed in this repository:

- Gemini for annotation processing and baseline evaluation.
- GPT-family models for zero-shot baselines.
- Image generation/editing APIs for refinement experiments.

Users are responsible for complying with each provider's terms of service and
for supplying their own API credentials.
