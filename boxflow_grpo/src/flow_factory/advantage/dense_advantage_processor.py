# src/flow_factory/advantage/dense_advantage_processor.py
"""
Dense (spatial) advantage processor for DenseGRPO.

Computes per-pixel group-normalized advantages from pixel reward maps.
Supports distributed training via accelerator.gather/scatter.

Ported from flow_grpo/scripts/train_sd3_dense_v2.py (lines 1220-1258)
"""
from __future__ import annotations

import logging
from typing import List, Optional

import torch
import numpy as np
from accelerate import Accelerator

logger = logging.getLogger(__name__)


class DenseAdvantageProcessor:
    """
    Computes spatial (pixel-level) advantages for DenseGRPO.

    Given pixel reward maps (B, H, W) and prompt group IDs, performs:
    1. Expand to (B, T, H, W) for all training timesteps
    2. Gather across GPUs for full group visibility
    3. Per-pixel group normalization with σ_min floor
    4. Scatter back to local rank
    """

    def __init__(self, log_func=None, verbose: bool = True):
        self.log_func = log_func
        self.verbose = verbose

    def compute_dense_advantages(
        self,
        accelerator: Accelerator,
        reward_maps: torch.Tensor,
        unique_ids: List[str],
        num_timesteps: int,
        sigma_min: float = 0.1,
        smooth_kernel: int = 0,
    ) -> torch.Tensor:
        """
        Compute pixel-level advantages with per-group per-pixel normalization.

        Args:
            accelerator: Accelerator instance for distributed ops
            reward_maps: (B_local, H, W) pixel reward maps
            unique_ids: List    of prompt unique_id strings for grouping (B_local,)
            num_timesteps: Number    of training timesteps T
            sigma_min: Minimum std floor for stable normalization

        Returns:
            advantages_spatial: (B_local, T, H, W) dense advantages
        """
        device = reward_maps.device
        B_local, H, W = reward_maps.shape

        # Expand to (B, T, H, W) — same reward map for all timesteps
        reward_maps_t = reward_maps.unsqueeze(1).expand(-1, num_timesteps, -1, -1).contiguous()

        # Gather across all processes
        gathered_reward_maps = accelerator.gather(reward_maps_t)  # (B_global, T, H, W)

        # Gather unique_ids for grouping
        # Encode unique_ids as integer hash tensors for gathering
        unique_id_hashes = torch.tensor(
            [hash(uid) % (2**31) for uid in unique_ids],
            dtype=torch.long, device=device,
        )
        gathered_hashes = accelerator.gather(unique_id_hashes)  # (B_global,)

        # Per-group per-pixel normalization
        unique_groups = torch.unique(gathered_hashes)
        advantages_global = torch.zeros_like(gathered_reward_maps)

        group_sizes = []
        for group_id in unique_groups:
            mask = (gathered_hashes == group_id)
            group_maps = gathered_reward_maps[mask]  # (G, T, H, W)
            group_sizes.append(int(mask.sum()))

            mean_per_pixel = group_maps.mean(dim=0, keepdim=True)  # (1, T, H, W)
            std_per_pixel = group_maps.std(dim=0, keepdim=True)    # (1, T, H, W)

            # σ_min floor for stable normalization
            std_per_pixel = torch.clamp(std_per_pixel, min=sigma_min)

            advantages_global[mask] = (group_maps - mean_per_pixel) / std_per_pixel

        # Optional: Gaussian smoothing to reduce extreme advantage peaks
        if smooth_kernel > 0:
            import torch.nn.functional as F
            k = smooth_kernel if smooth_kernel % 2 == 1 else smooth_kernel + 1
            sigma = k / 3.0
            coords = torch.arange(k, dtype=advantages_global.dtype, device=advantages_global.device) - k // 2
            gauss_1d = torch.exp(-0.5 * (coords / sigma) ** 2)
            gauss_2d = gauss_1d[:, None] * gauss_1d[None, :]
            gauss_2d = gauss_2d / gauss_2d.sum()
            weight = gauss_2d.unsqueeze(0).unsqueeze(0)  # (1,1,k,k)
            B_g, T, H, W = advantages_global.shape
            adv_flat = advantages_global.reshape(B_g * T, 1, H, W)
            adv_flat = F.pad(adv_flat, [k // 2] * 4, mode='reflect')
            adv_flat = F.conv2d(adv_flat, weight)
            advantages_global = adv_flat.reshape(B_g, T, H, W)

        # Scatter back to local rank
        advantages_local = (
            advantages_global
            .reshape(accelerator.num_processes, -1, *advantages_global.shape[1:])
            [accelerator.process_index]
        )

        # Log stats
        if self.verbose and accelerator.is_local_main_process:
            logger.info(
                f"DenseAdvantage: {len(unique_groups)} groups, "
                f"sizes={group_sizes}, sigma_min={sigma_min}, "
                f"adv_mean={advantages_local.mean():.4f}, adv_std={advantages_local.std():.4f}"
            )

        return advantages_local

    def compute_per_group_scalar_stds(
        self,
        accelerator: Accelerator,
        scalar_rewards: torch.Tensor,
        unique_ids: List[str],
    ) -> List[float]:
        """
        Compute per-sample group std    of scalar rewards (for adaptive penalty).

        Args:
            accelerator: Accelerator instance
            scalar_rewards: (B_local,) scalar rewards
            unique_ids: List    of prompt unique_id strings (B_local,)

        Returns:
            per_sample_group_stds: List    of group std values (B_local,)
        """
        device = scalar_rewards.device

        # Gather scalar rewards and unique_ids
        gathered_rewards = accelerator.gather(scalar_rewards.contiguous())
        unique_id_hashes = torch.tensor(
            [hash(uid) % (2**31) for uid in unique_ids],
            dtype=torch.long, device=device,
        )
        gathered_hashes = accelerator.gather(unique_id_hashes)

        # Compute per-group std
        per_sample_std = torch.zeros_like(gathered_rewards)
        for group_id in torch.unique(gathered_hashes):
            mask = (gathered_hashes == group_id)
            group_rewards = gathered_rewards[mask]
            group_std = group_rewards.std().item() if mask.sum() > 1 else 0.0
            per_sample_std[mask] = group_std

        # Scatter back to local rank
        local_stds = (
            per_sample_std
            .reshape(accelerator.num_processes, -1)
            [accelerator.process_index]
        )

        return local_stds.tolist()
