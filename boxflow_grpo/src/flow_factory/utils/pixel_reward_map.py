# src/flow_factory/utils/pixel_reward_map.py
"""
Pixel-level reward map construction for DenseGRPO.

Creates spatial reward maps R_D(h,w) = R + R_P(h,w) where:
- R: scalar reward (e.g., from UnifiedReward2)
- R_P(h,w): pixel-level penalty based on detected defect bboxes

Ported from flow_grpo/qwen3vl_bbox_scorer.py:create_pixel_reward_map
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

import torch


def create_pixel_reward_map(
    bboxes: list,
    scalar_reward: float,
    latent_size: int = 64,
    image_size: int = 512,
    alpha: float = 0.5,
    alpha_artifact: Optional[float] = None,
    alpha_misalignment: Optional[float] = None,
    area_normalize_artifact: bool = False,
    area_normalize_misalignment: bool = False,
    importance_weighting: bool = False,
    min_penalty: float = 0.0,
    stack_same_label: bool = False,
) -> torch.Tensor:
    """
    Create pixel-level reward map for DenseGRPO.

    R_D(h,w) = R + R_P(h,w)

    Where:
        R: scalar reward from calculate_reward
        R_P(h,w): pixel-level penalty (-alpha per unique label at that pixel)

    Note: Same-label boxes that overlap do NOT accumulate penalty.
          Only different-label boxes overlapping will accumulate penalty.

    Args:
        bboxes: List   bbox dicts with 'box_2d' key [x0,y0,x1,y1] in 0-1000 coords
                and optional 'label' key ("misalignment" or "artifact")
        scalar_reward: Base scalar reward
        latent_size: Size    of latent space (assumes square)
        image_size: Original image size
        alpha: Default penalty value for defect regions
        alpha_artifact: Penalty value specifically for artifact regions
        alpha_misalignment: Penalty value specifically for misalignment regions
        area_normalize_artifact: Normalize artifact penalty by coverage ratio
        area_normalize_misalignment: Normalize misalignment penalty by coverage ratio
        importance_weighting: If True, scale penalty by normalized importance (importance/100).
                              Overlapping same-label bboxes use max importance per pixel.
        min_penalty: Floor for penalty values

    Returns:
        reward_map: (H, W) tensor    of pixel-level rewards
    """
    if alpha_artifact is None:
        alpha_artifact = alpha
    if alpha_misalignment is None:
        alpha_misalignment = alpha

    reward_map = torch.full((latent_size, latent_size), scalar_reward)

    if not bboxes:
        return reward_map

    scale = latent_size / image_size

    # Group bboxes by label
    if importance_weighting:
        label_masks: Dict[str, torch.Tensor] = defaultdict(
            lambda: torch.zeros((latent_size, latent_size), dtype=torch.float32)
        )
    else:
        label_masks: Dict[str, torch.Tensor] = defaultdict(
            lambda: torch.zeros((latent_size, latent_size), dtype=torch.bool)
        )

    for bbox_info in bboxes:
        if not isinstance(bbox_info, dict):
            continue
        if "box_2d" not in bbox_info:
            continue

        box_2d = bbox_info["box_2d"]
        if not isinstance(box_2d, (list, tuple)) or len(box_2d) != 4:
            continue

        try:
            x0, y0, x1, y1 = box_2d
        except (ValueError, TypeError):
            continue

        label = bbox_info.get("label", "artifact")

        # Convert from 0-1000 to latent coords
        lx0 = int(x0 * image_size / 1000 * scale)
        ly0 = int(y0 * image_size / 1000 * scale)
        lx1 = int(x1 * image_size / 1000 * scale)
        ly1 = int(y1 * image_size / 1000 * scale)
        lx0, lx1 = max(0, lx0), min(latent_size, lx1)
        ly0, ly1 = max(0, ly0), min(latent_size, ly1)

        if lx1 > lx0 and ly1 > ly0:
            if importance_weighting:
                imp = bbox_info.get("importance")
                try:
                    imp_weight = (float(imp) / 100.0) if imp is not None else 1.0
                except (TypeError, ValueError):
                    imp_weight = 1.0
                if stack_same_label:
                    label_masks[label][ly0:ly1, lx0:lx1] += imp_weight
                else:
                    label_masks[label][ly0:ly1, lx0:lx1] = torch.max(
                        label_masks[label][ly0:ly1, lx0:lx1],
                        torch.tensor(imp_weight),
                    )
            else:
                if stack_same_label:
                    # Use int count for stacking
                    label_masks[label] = label_masks[label].to(torch.int32) if label_masks[label].dtype == torch.bool else label_masks[label]
                    label_masks[label][ly0:ly1, lx0:lx1] += 1
                else:
                    label_masks[label][ly0:ly1, lx0:lx1] = True

    # Apply penalty for each label's mask
    for label, mask in label_masks.items():
        if label == "misalignment":
            penalty = alpha_misalignment
            if area_normalize_misalignment:
                active = mask > 0 if importance_weighting else mask
                coverage = active.sum().item() / (latent_size * latent_size)
                penalty *= (1.0 - coverage)
            penalty = max(penalty, min_penalty)
            if importance_weighting:
                pixel_penalty = penalty * mask
                pixel_penalty = torch.where(
                    mask > 0, torch.clamp(pixel_penalty, min=min_penalty), pixel_penalty
                )
                reward_map -= pixel_penalty
            else:
                # Handle both bool (no stack) and int32 (stack) mask
                bool_mask = mask > 0 if mask.dtype != torch.bool else mask
                if stack_same_label and mask.dtype != torch.bool:
                    reward_map -= penalty * mask.to(torch.float32)
                else:
                    reward_map[bool_mask] -= penalty
        else:
            penalty = alpha_artifact
            if area_normalize_artifact:
                active = mask > 0 if importance_weighting else mask
                coverage = active.sum().item() / (latent_size * latent_size)
                penalty *= (1.0 - coverage)
            penalty = max(penalty, min_penalty)
            if importance_weighting:
                pixel_penalty = penalty * mask
                pixel_penalty = torch.where(
                    mask > 0, torch.clamp(pixel_penalty, min=min_penalty), pixel_penalty
                )
                reward_map -= pixel_penalty
            else:
                # Handle both bool (no stack) and int32 (stack) mask
                bool_mask = mask > 0 if mask.dtype != torch.bool else mask
                if stack_same_label and mask.dtype != torch.bool:
                    reward_map -= penalty * mask.to(torch.float32)
                else:
                    reward_map[bool_mask] -= penalty

    return reward_map


def batch_create_pixel_reward_maps(
    scalar_rewards: List[float],
    all_bboxes: List[List[Dict]],
    latent_size: int = 64,
    image_size: int = 512,
    alpha: float = 0.5,
    alpha_artifact: Optional[float] = None,
    alpha_misalignment: Optional[float] = None,
    min_penalty: float = 0.02,
    importance_weighting: bool = True,
    area_normalize_artifact: bool = False,
    area_normalize_misalignment: bool = False,
    per_sample_group_stds: Optional[List[float]] = None,
    device: Optional[torch.device] = None,
    stack_same_label: bool = False,
) -> torch.Tensor:
    """
    Create pixel reward maps for a batch    of samples.

    Args:
        scalar_rewards: List    of scalar rewards per sample
        all_bboxes: List   bbox lists per sample
        per_sample_group_stds: Per-sample group std for adaptive penalty scaling.
                               If provided, alpha is scaled by group_std.
        device: Target device

    Returns:
        reward_maps: (B, H, W) tensor
    """
    maps = []
    for i, (sr, bboxes) in enumerate(zip(scalar_rewards, all_bboxes)):
        # Adaptive alpha: scale by per-group std
        adaptive_alpha = alpha
        if per_sample_group_stds is not None and i < len(per_sample_group_stds):
            adaptive_alpha = alpha * per_sample_group_stds[i]

        # Scale per-type alpha by adaptive factor
        adaptive_alpha_artifact = (alpha_artifact if alpha_artifact is not None else alpha)
        adaptive_alpha_misalignment = (alpha_misalignment if alpha_misalignment is not None else alpha)
        if per_sample_group_stds is not None and i < len(per_sample_group_stds):
            scale = per_sample_group_stds[i]
            adaptive_alpha_artifact *= scale
            adaptive_alpha_misalignment *= scale

        rmap = create_pixel_reward_map(
            bboxes=bboxes if bboxes is not None else [],
            scalar_reward=sr,
            latent_size=latent_size,
            image_size=image_size,
            alpha=adaptive_alpha,
            alpha_artifact=adaptive_alpha_artifact,
            alpha_misalignment=adaptive_alpha_misalignment,
            min_penalty=min_penalty,
            importance_weighting=importance_weighting,
            area_normalize_artifact=area_normalize_artifact,
            area_normalize_misalignment=area_normalize_misalignment,
            stack_same_label=stack_same_label,
        )
        maps.append(rmap)

    result = torch.stack(maps)
    if device is not None:
        result = result.to(device)
    return result
