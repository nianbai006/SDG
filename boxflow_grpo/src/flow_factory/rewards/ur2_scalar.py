# src/flow_factory/rewards/ur2_scalar.py
"""
UR2 Scalar Reward Model.

Queries UnifiedReward2 sglang server with the same Alignment/Coherence/Style prompt
as CombinedUR2BBoxReward, parses the text response, and returns a scalar reward:
    score = (0.8 * alignment + 0.1 * coherence + 0.1 * style) / 5.0

This is the scalar-only counterpart    of CombinedUR2BBoxReward — same reward signal
but without BBox spatial penalty, so standard (non-dense) GRPO can use it directly.
"""
from __future__ import annotations

import asyncio
import os
from typing import List, Optional

import httpx
import numpy as np
from accelerate import Accelerator
from PIL import Image

from .abc import PointwiseRewardModel, RewardModelOutput
from .dense_grpo_rewards import extract_unified_scores
from ..hparams import RewardArguments
from ..utils.image import pil_image_to_base64
from ..utils.logger_utils import setup_logger

logger = setup_logger(__name__)


UR2_QUESTION = (
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
    "Text Caption: [{prompt}]"
)


class UR2ScalarReward(PointwiseRewardModel):
    """
    Scalar UR2 reward, parsing Alignment/Coherence/Style scores from UR2 text output.

    Reward = (0.8 * alignment + 0.1 * coherence + 0.1 * style) / 5.0

    Extra kwargs:
        ur2_server_url (str): Base URL for UR2 sglang server. Default: http://localhost:17141/v1
        api_key (str): API key. Default: "flowgrpo"
        vlm_model (str): Model name. Default: "CodeGoat24/UnifiedReward-2.0-qwen3vl-2b"
        max_retries (int): Default: 10
        timeout (float): Default: 180.0
    """
    required_fields: tuple = ('prompt', 'image')
    use_tensor_inputs: bool = False

    def __init__(self, config: RewardArguments, accelerator: Accelerator):
        super().__init__(config=config, accelerator=accelerator)

        self.ur2_url = config.extra_kwargs.get('ur2_server_url', 'http://localhost:17141/v1')
        self.api_key = config.extra_kwargs.get('api_key', 'flowgrpo')
        self.vlm_model = config.extra_kwargs.get('vlm_model', 'CodeGoat24/UnifiedReward-2.0-qwen3vl-2b')
        self.max_retries = config.extra_kwargs.get('max_retries', 10)
        self.timeout = config.extra_kwargs.get('timeout', 180.0)

        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(
            base_url=self.ur2_url,
            api_key=self.api_key,
            max_retries=self.max_retries,
            timeout=self.timeout,
            http_client=httpx.AsyncClient(trust_env=False),
        )

        logger.info(f"UR2ScalarReward initialized: url={self.ur2_url}, model={self.vlm_model}")

    async def _evaluate_unified(self, prompt: str, image: Image.Image) -> str:
        image_base64 = pil_image_to_base64(image)
        try:
            response = await self.client.chat.completions.create(
                model=self.vlm_model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_base64}},
                        {"type": "text", "text": UR2_QUESTION.format(prompt=prompt)},
                    ],
                }],
                temperature=0,
                max_tokens=512,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.warning(f"UR2ScalarReward API call failed: {e}")
            return ""

    async def _evaluate_batch(self, images: List[Image.Image], prompts: List[str]) -> List[str]:
        tasks = [self._evaluate_unified(p, img) for p, img in zip(prompts, images)]
        return await asyncio.gather(*tasks)

    def __call__(
        self,
        prompt: List[str],
        image: Optional[List[Image.Image]] = None,
        **kwargs,
    ) -> RewardModelOutput:
        # Resize to match training resolution used by CombinedUR2BBoxReward
        images = [img.resize((512, 512)) for img in image]
        text_outputs = asyncio.run(self._evaluate_batch(images, prompt))
        scores = extract_unified_scores(text_outputs)
        logger.info(f"UR2ScalarReward: mean_score={np.mean(scores):.3f}, "
                    f"n_failed={sum(1 for s in scores if s == 0.0)}")
        return RewardModelOutput(rewards=scores)
