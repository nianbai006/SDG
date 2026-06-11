# src/flow_factory/rewards/dense_grpo_rewards.py
"""
Combined UnifiedReward2 + QwenVL3 BBox reward model for DenseGRPO.

Evaluates images concurrently via two sglang servers:
- UnifiedReward2: produces scalar quality scores (alignment/coherence/style)
- QwenVL3 BBox V2: detects defect bounding boxes (misalignment/artifact)

The scalar rewards go into standard advantage computation;
the bboxes are passed via extra_info for pixel-level reward map construction.

Ported from flow_grpo/rewards.py:combined_unifiedreward2_bbox_v2
"""
from __future__ import annotations

import re
import json
import asyncio
import base64
import logging
import os
from io import BytesIO
from typing import Dict, Any, Optional, List, Tuple

import torch
import numpy as np
from PIL import Image
from accelerate import Accelerator

from .abc import PointwiseRewardModel, RewardModelOutput
from ..hparams import RewardArguments

logger = logging.getLogger(__name__)

# Suppress verbose HTTP logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


# =====================================================================
# Utility functions
# =====================================================================

def pil_image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string for API."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image;base64,{encoded}"


def parse_answer_block_v2(response_text: str) -> list:
    """
    Parse the <answer> block from model response to extract bboxes.

    Returns:
        List   bbox dicts with keys: box_2d, label, desc/description, importance
    """
    answer_match = re.search(r'<answer>\s*(.*?)\s*</answer>', response_text, re.DOTALL)
    if not answer_match:
        return []

    answer_content = answer_match.group(1).strip()
    try:
        json_match = re.search(r'\[.*\]', answer_content, re.DOTALL)
        if json_match:
            bboxes = json.loads(json_match.group(0))
            if isinstance(bboxes, list):
                return bboxes
        return []
    except json.JSONDecodeError:
        return []


def extract_unified_scores(text_outputs: List[str]) -> List[float]:
    """
    Extract UnifiedReward2 scores from text outputs.

    Parses Alignment/Coherence/Style scores and computes weighted average:
    score = (0.8 * alignment + 0.1 * coherence + 0.1 * style) / 5.0
    """
    scores = []
    for text in text_outputs:
        try:
            alignment_match = re.search(r'Alignment Score[^:]*:\s*([\d.]+)', text)
            coherence_match = re.search(r'Coherence Score[^:]*:\s*([\d.]+)', text)
            style_match = re.search(r'Style Score[^:]*:\s*([\d.]+)', text)

            if alignment_match and coherence_match and style_match:
                alignment = float(alignment_match.group(1))
                coherence = float(coherence_match.group(1))
                style = float(style_match.group(1))
                avg_score = alignment * 0.8 + coherence * 0.1 + style * 0.1
                scores.append(avg_score / 5.0)
            else:
                scores.append(0.0)
        except Exception as e:
            logger.warning(f"UnifiedReward2 score extraction error: {e}, text: {text[:200]}")
            scores.append(0.0)
    return scores


# =====================================================================
# Prompt templates for BBox detection
# =====================================================================

SFT_THINKPE_IMP_TEMPLATE = """You are an AI image quality evaluator. You will be given **one image** to analyze.

### Definitions

**Misalignment**: Areas where the image content does NOT match the text caption, including:
- Missing objects: Objects mentioned in caption but not present in image
- Extra objects: Objects present in image but not mentioned in caption
- Wrong attributes: Incorrect color, size, material, count, or other properties
- Wrong spatial relationships: Incorrect positions, orientations, or arrangements

**Artifact**: Visual defects in the generated image, including:
- Distorted anatomy: Malformed hands, extra/missing limbs, wrong number    of fingers
- Duplicated/missing parts: Repeated or absent body parts, objects
- Warped geometry: Perspective errors, impossible shapes
- Texture issues: Melted, smeared, or overly smooth textures
- Unnatural edges: Jagged, broken, or blurry boundaries
- Garbled text: Unreadable or malformed text/letters
- Lighting inconsistencies: Wrong shadows, reflections, or light sources

Text Caption: {caption}

**Goal**: Produce a detailed analysis    of the image quality and output bounding boxes with severity scores for all detected issues.

### Strict Output Rules
Output **TWO blocks in this exact order**:
1) `<think>` - Your detailed analysis
2) `<answer>` - JSON list    of bounding boxes

### Think Format (STRICT)
<think>
### Step 1: Caption Understanding
- Briefly summarize what the caption requires (subject, key attributes, actions, setting, style/composition if mentioned).

### Step 2: Visual Analysis & Defect Spotting (Issue Summary)
- Describe the quality issues you observe in the image.
- Each bullet MUST include:
  (a) the issue category (artifact or misalignment)
  (b) what is affected
  (c) concrete visual evidence

### Step 3: Localization (Box-by-Box Grounding)
- Provide a detailed, precise localization statement for EACH defect instance.
</think>

### Answer Format (for <answer>)
Return a JSON list:
[
    {{"box_2d": [x0, y0, x1, y1], "label": "misalignment"|"artifact", "description": "brief description    of the issue", "importance": N}}
]

Bounding box coordinates are in normalized 0-1000 space: [x0, y0, x1, y1].
If there are no issues, output an empty list.

### Importance Scoring
For EACH box, assign an integer importance score from 1 to 100:
- 90-100: Critical defect, immediately obvious, ruins the image.
- 70-89: Major defect, clearly visible at normal viewing distance.
- 40-69: Moderate defect, noticeable on closer inspection.
- 15-39: Minor defect, only visible on careful examination.
- 1-14: Negligible defect, barely perceptible.

Now analyze the image and produce your output:
"""

BBOX_PROMPT_TEMPLATES = {
    "thinkpe_imp": SFT_THINKPE_IMP_TEMPLATE,
}


# =====================================================================
# CombinedUR2BBoxReward
# =====================================================================

class CombinedUR2BBoxReward(PointwiseRewardModel):
    """
    Combined reward using UnifiedReward2 for base score + QwenVL3 BBox V2 for spatial penalty.

    Base reward: UnifiedReward2 weighted average score (0-1)
    Extra info: bboxes from QwenVL3 for pixel-level reward map construction

    Server setup (on dedicated reward server):
    - GPU 0-3: QwenVL3 BBox V2 (port 17142)
    - GPU 4-7: UnifiedReward2 (port 17141)
    """
    required_fields: tuple = ('prompt', 'image')
    use_tensor_inputs: bool = False

    def __init__(self, config: RewardArguments, accelerator: Accelerator):
        super().__init__(config=config, accelerator=accelerator)

        # Read server URLs from extra_kwargs
        self.ur2_url = config.extra_kwargs.get('ur2_server_url', 'http://localhost:17141/v1')
        self.bbox_url = config.extra_kwargs.get('bbox_server_url', 'http://localhost:17142/v1')
        self.bbox_template_name = config.extra_kwargs.get('bbox_prompt_template', 'thinkpe_imp')
        self.bbox_template = BBOX_PROMPT_TEMPLATES.get(self.bbox_template_name, SFT_THINKPE_IMP_TEMPLATE)

        # Temporarily clear proxy env vars for local connections
        from openai import AsyncOpenAI
        saved_proxy = {k: os.environ.pop(k, None) for k in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"]}

        self.unified_client = AsyncOpenAI(
            base_url=self.ur2_url, api_key="flowgrpo", max_retries=10, timeout=180.0,
        )
        self.bbox_client = AsyncOpenAI(
            base_url=self.bbox_url, api_key="flowgrpo", max_retries=10, timeout=180.0,
        )

        # Restore proxy for wandb etc.
        for k, v in saved_proxy.items():
            if v is not None:
                os.environ[k] = v

        logger.info(f"CombinedUR2BBoxReward initialized: UR2={self.ur2_url}, BBox={self.bbox_url}, template={self.bbox_template_name}")

    async def _evaluate_unified(self, prompt: str, image: Image.Image) -> str:
        """Evaluate a single image with UnifiedReward2."""
        question = (
            "You are presented with a generated image and its associated text caption. "
            "Your task is to analyze the image across multiple dimensions in relation to the caption. Specifically:\n"
            "Provide overall assessments for the image along the following axes (each rated from 1 to 5):\n"
            "- Alignment Score: How well the image matches the caption in terms    of content.\n"
            "- Coherence Score: How logically consistent the image is (absence    of visual glitches, object distortions, etc.).\n"
            "- Style Score: How aesthetically appealing the image looks, regardless    of caption accuracy.\n\n"
            "Output your evaluation using the format below:\n\n"
            "Alignment Score (1-5): X\n"
            "Coherence Score (1-5): Y\n"
            "Style Score (1-5): Z\n\n"
            "Your task is provided as follows:\n"
            f"Text Caption: [{prompt}]"
        )
        image_base64 = pil_image_to_base64(image)
        try:
            response = await self.unified_client.chat.completions.create(
                model="CodeGoat24/UnifiedReward-2.0-qwen3vl-2b",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_base64}},
                        {"type": "text", "text": question},
                    ],
                }],
                temperature=0,
                max_tokens=512,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.warning(f"UnifiedReward2 API call failed: {e}")
            return ""

    async def _evaluate_bbox(self, caption: str, image: Image.Image) -> dict:
        """Evaluate a single image with QwenVL3 BBox V2."""
        image_base64 = pil_image_to_base64(image.resize((512, 512)))
        prompt = self.bbox_template.format(caption=caption)

        try:
            response = await asyncio.wait_for(
                self.bbox_client.chat.completions.create(
                    model="sdg-detector",
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_base64}},
                            {"type": "text", "text": prompt},
                        ],
                    }],
                    temperature=0,
                    max_tokens=4096,
                ),
                timeout=180.0,
            )
            response_text = response.choices[0].message.content
            bboxes = parse_answer_block_v2(response_text)
            return {"bboxes": bboxes, "raw_response": response_text}
        except asyncio.TimeoutError:
            logger.warning("Timeout evaluating bbox after 180s")
            return {"bboxes": [], "raw_response": "Timeout"}
        except Exception as e:
            logger.warning(f"Error evaluating bbox: {e}")
            return {"bboxes": [], "raw_response": str(e)}

    async def _evaluate_batch_concurrent(self, images: List[Image.Image], prompts: List[str]):
        """Evaluate both UR2 and BBox concurrently for all images."""
        unified_tasks = [self._evaluate_unified(p, img) for p, img in zip(prompts, images)]
        bbox_tasks = [self._evaluate_bbox(p, img) for p, img in zip(prompts, images)]

        unified_results, bbox_results = await asyncio.gather(
            asyncio.gather(*unified_tasks),
            asyncio.gather(*bbox_tasks),
        )
        return unified_results, bbox_results

    def __call__(
        self,
        prompt: List[str],
        image: Optional[List[Image.Image]] = None,
        **kwargs,
    ) -> RewardModelOutput:
        """
        Compute combined UR2 + BBox rewards.

        Returns:
            RewardModelOutput with:
              - rewards: UR2 scalar scores (batch_size,)
              - extra_info: {'bboxes': List[List[dict]]} for pixel reward map construction
        """
        # Resize images for UR2
        images = [img.resize((512, 512)) for img in image]

        # Run concurrent evaluation
        unified_text_outputs, bbox_results = asyncio.run(
            self._evaluate_batch_concurrent(images, prompt)
        )

        # Extract UR2 scores
        base_scores = extract_unified_scores(unified_text_outputs)

        # Extract bboxes
        all_bboxes = [r.get("bboxes", []) for r in bbox_results]

        # Log stats
        num_timeout = sum(1 for r in bbox_results if r.get("raw_response") == "Timeout")
        num_success = len(bbox_results) - num_timeout
        logger.info(f"CombinedUR2BBox: {num_success}/{len(bbox_results)} success, {num_timeout} timeout, "
                     f"mean_score={np.mean(base_scores):.3f}")

        return RewardModelOutput(
            rewards=base_scores,
            extra_info={
                'bboxes': all_bboxes,
                'bbox_timeout_count': num_timeout,
                'bbox_success_count': num_success,
            },
        )

class BBoxOnlyReward(PointwiseRewardModel):
    """
    BBox-only reward: no UR2 scalar score, only spatial defect detection.
    Returns a constant scalar reward (default 0.0) + bboxes for pixel reward map.
    """
    required_fields: tuple = ('prompt', 'image')
    use_tensor_inputs: bool = False

    def __init__(self, config: RewardArguments, accelerator: Accelerator):
        super().__init__(config=config, accelerator=accelerator)

        self.bbox_url = config.extra_kwargs.get('bbox_server_url', 'http://localhost:17142/v1')
        self.bbox_template_name = config.extra_kwargs.get('bbox_prompt_template', 'thinkpe_imp')
        self.bbox_template = BBOX_PROMPT_TEMPLATES.get(self.bbox_template_name, SFT_THINKPE_IMP_TEMPLATE)
        self.base_scalar_reward = config.extra_kwargs.get('base_scalar_reward', 0.0)

        from openai import AsyncOpenAI
        saved_proxy = {k: os.environ.pop(k, None) for k in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"]}
        self.bbox_client = AsyncOpenAI(
            base_url=self.bbox_url, api_key="flowgrpo", max_retries=10, timeout=180.0,
        )
        for k, v in saved_proxy.items():
            if v is not None:
                os.environ[k] = v

        logger.info(f"BBoxOnlyReward initialized: BBox={self.bbox_url}, base_reward={self.base_scalar_reward}")

    async def _evaluate_bbox(self, caption: str, image: Image.Image) -> dict:
        image_base64 = pil_image_to_base64(image.resize((512, 512)))
        prompt = self.bbox_template.format(caption=caption)
        try:
            response = await asyncio.wait_for(
                self.bbox_client.chat.completions.create(
                    model="sdg-detector",
                    messages=[{"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": image_base64}},
                        {"type": "text", "text": prompt},
                    ]}],
                    temperature=0, max_tokens=4096,
                ),
                timeout=180.0,
            )
            bboxes = parse_answer_block_v2(response.choices[0].message.content)
            return {"bboxes": bboxes, "raw_response": response.choices[0].message.content}
        except asyncio.TimeoutError:
            logger.warning("BBoxOnly: timeout after 180s")
            return {"bboxes": [], "raw_response": "Timeout"}
        except Exception as e:
            logger.warning(f"BBoxOnly: error: {e}")
            return {"bboxes": [], "raw_response": str(e)}

    def __call__(self, prompt: List[str], image: Optional[List[Image.Image]] = None, **kwargs) -> RewardModelOutput:
        images = [img.resize((512, 512)) for img in image]

        async def _run_all():
            return await asyncio.gather(*[self._evaluate_bbox(p, img) for p, img in zip(prompt, images)])

        bbox_results = asyncio.run(_run_all())

        all_bboxes = [r.get("bboxes", []) for r in bbox_results]
        base_scores = [self.base_scalar_reward] * len(prompt)

        num_timeout = sum(1 for r in bbox_results if r.get("raw_response") == "Timeout")
        logger.info(f"BBoxOnly: {len(bbox_results)-num_timeout}/{len(bbox_results)} success, {num_timeout} timeout")

        return RewardModelOutput(
            rewards=base_scores,
            extra_info={
                'bboxes': all_bboxes,
                'bbox_timeout_count': num_timeout,
                'bbox_success_count': len(bbox_results) - num_timeout,
            },
        )


class BBoxScalarReward(BBoxOnlyReward):
    """
    BBox-to-scalar reward: computes bboxes, builds pixel reward map, returns map mean as scalar.
    Converts dense spatial penalty into a single scalar reward for standard GRPO.
    """

    def __init__(self, config: RewardArguments, accelerator: Accelerator):
        super().__init__(config=config, accelerator=accelerator)
        self.alpha_artifact = config.extra_kwargs.get('alpha_artifact', 0.5)
        self.alpha_misalignment = config.extra_kwargs.get('alpha_misalignment', 0.05)
        self.importance_weighting = config.extra_kwargs.get('importance_weighting', True)
        self.latent_size = config.extra_kwargs.get('latent_size', 64)
        self.image_size = config.extra_kwargs.get('image_size', 512)
        logger.info(f"BBoxScalarReward: alpha_art={self.alpha_artifact}, alpha_mis={self.alpha_misalignment}")

    def __call__(self, prompt: List[str], image: Optional[List[Image.Image]] = None, **kwargs) -> RewardModelOutput:
        from ..utils.pixel_reward_map import create_pixel_reward_map

        images = [img.resize((512, 512)) for img in image]

        async def _run_all():
            return await asyncio.gather(*[self._evaluate_bbox(p, img) for p, img in zip(prompt, images)])

        bbox_results = asyncio.run(_run_all())
        all_bboxes = [r.get("bboxes", []) for r in bbox_results]

        # Build pixel map per sample, take mean as scalar reward
        scores = []
        for bboxes in all_bboxes:
            rmap = create_pixel_reward_map(
                bboxes=bboxes, scalar_reward=0.0,
                latent_size=self.latent_size, image_size=self.image_size,
                alpha_artifact=self.alpha_artifact,
                alpha_misalignment=self.alpha_misalignment,
                importance_weighting=self.importance_weighting,
            )
            scores.append(rmap.mean().item())

        num_timeout = sum(1 for r in bbox_results if r.get("raw_response") == "Timeout")
        logger.info(f"BBoxScalar: {len(bbox_results)-num_timeout}/{len(bbox_results)} success, "
                     f"mean_score={np.mean(scores):.4f}")

        return RewardModelOutput(rewards=scores, extra_info={'bboxes': all_bboxes})


class CombinedImageDoctorBBoxReward(BBoxOnlyReward):
    """
    Combined: local ImageDoctor scalar score + remote BBox server for spatial grounding.
    Scalar reward = ImageDoctor's (alignment * 0.8 + aesthetic * 0.1 + plausibility * 0.1).
    Bboxes = remote BBox server results (stored in extra_info).
    """

    def __init__(self, config: RewardArguments, accelerator: Accelerator):
        super().__init__(config=config, accelerator=accelerator)
        from .imagedoctor import _ImageDoctorBackend
        ckpt = config.extra_kwargs.get('imagedoctor_checkpoint', 'GYX97/ImageDoctor')
        self.imagedoctor = _ImageDoctorBackend(
            checkpoint=ckpt, dtype=torch.bfloat16, device=accelerator.device,
        )
        logger.info(f"CombinedImageDoctorBBoxReward: ckpt={ckpt} bbox={self.bbox_url}")

    def __call__(self, prompt: List[str], image: Optional[List[Image.Image]] = None, **kwargs) -> RewardModelOutput:
        images = [img.resize((512, 512)) for img in image]

        async def _run_bbox():
            return await asyncio.gather(*[self._evaluate_bbox(p, img) for p, img in zip(prompt, images)])

        bbox_results = asyncio.run(_run_bbox())
        all_bboxes = [r.get("bboxes", []) for r in bbox_results]

        scores = []
        for img, p in zip(images, prompt):
            try:
                s, _ = self.imagedoctor.process(img, p, return_heatmap=False)
                scores.append(float(s))
            except Exception as e:
                logger.warning(f"ImageDoctor error: {e}, using 0.5")
                scores.append(0.5)

        num_timeout = sum(1 for r in bbox_results if r.get("raw_response") == "Timeout")
        logger.info(f"CombinedID+BBox: {len(bbox_results)-num_timeout}/{len(bbox_results)} bbox success, "
                     f"id_mean={np.mean(scores):.4f}")

        return RewardModelOutput(
            rewards=scores,
            extra_info={
                'bboxes': all_bboxes,
                'bbox_timeout_count': num_timeout,
                'bbox_success_count': len(bbox_results) - num_timeout,
            },
        )

