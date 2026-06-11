"""
Prompt builders for the GPT-Image-1.5 refinement pipeline.

Three prompts back the paper's Table 5 configurations:
- ``build_fixed_prompt``           — Fixed (caption-only) baseline
- ``build_prompt_from_imagedoctor`` — ImageDoctor heatmap + text feedback
- ``build_prompt_from_sdg``         — SDG (ours) bbox + text feedback
"""
from typing import Optional


def build_fixed_prompt(caption: str) -> str:
    """Caption-only edit prompt used by the Fixed baseline."""
    return (
        f"Improve the image quality, fix any visual artifacts, distortions, "
        f"and ensure the image accurately matches the following description: {caption}"
    )


def build_prompt_from_imagedoctor(
    caption: str,
    prediction: str,
    use_heatmap_info: bool = False,
    heatmap_regions: Optional[list] = None,
    original_only: bool = False,
) -> str:
    """ImageDoctor edit prompt. Forwards ImageDoctor's full <think>/<answer>
    response as textual feedback; ``original_only=True`` omits the heatmap
    accompaniment language (used in the *_notext ablation).
    """
    if not prediction:
        return build_fixed_prompt(caption)

    base = (
        f"Based on feedback from an external evaluation model, "
        f"improve the qualityof this image, fix any visual artifacts, distortions, "
        f"and ensure the image accurately matches the following description: {caption}\n\n"
    )
    if original_only:
        return (
            base
            + "Below is the textual feedback from the external model:\n\n"
            + prediction
        )
    return (
        base
        + "Below is the textual feedback from the external model, along with the "
          "corresponding artifact and misalignment heatmaps (highlighting the "
          "problematic regions):\n\n"
        + prediction
    )


def build_prompt_from_sdg(
    caption: str,
    sdg_response: str,
    original_only: bool = False,
) -> str:
    """SDG edit prompt. Forwards SDG's full <think>/<answer> response as
    textual feedback; ``original_only=True`` omits the bbox-overlay
    accompaniment language (used in the *_notext ablation).
    """
    if not sdg_response:
        return build_fixed_prompt(caption)

    base = (
        f"Based on feedback from an external evaluation model, "
        f"improve the qualityof this image, fix any visual artifacts, distortions, "
        f"and ensure the image accurately matches the following description: {caption}\n\n"
    )
    if original_only:
        return (
            base
            + "Below is the textual feedback from the external model:\n\n"
            + sdg_response
        )
    return (
        base
        + "Below is the textual feedback from the external model, along with the "
          "bounding box annotations on the second image (highlighting the defect "
          "regions):\n\n"
        + sdg_response
    )
