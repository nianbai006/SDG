# src/flow_factory/trainers/dense_grpo.py
"""
DenseGRPO Trainer — pixel-level advantage GRPO for diffusion models.

Extends GRPOTrainer with:
- Spatial log_prob (B, H, W) via DenseFlowMatchEulerDiscreteSDEScheduler
- Pixel reward maps from UR2 scalar + BBox spatial detections
- Per-pixel group-normalized advantage with σ_min floor
- Reference-aligned spatial ratio for policy loss
- No advantage clipping

Ported from flow_grpo/scripts/train_sd3_dense_v2.py
"""
from __future__ import annotations

import os
import tempfile
from typing import List, Dict, Optional, Any, Union
from functools import partial
from collections import defaultdict

import torch
import numpy as np
from PIL import Image, ImageDraw
import tqdm as tqdm_
tqdm = partial(tqdm_.tqdm, dynamic_ncols=True)

from .abc import BaseTrainer
from .grpo import GRPOTrainer
from ..hparams import GRPOTrainingArguments
from ..samples import BaseSample
from ..utils.base import filter_kwargs, create_generator, create_generator_by_prompt
from ..logger.formatting import LogImage
from ..utils.logger_utils import setup_logger
from ..utils.trajectory_collector import compute_trajectory_indices
from ..utils.pixel_reward_map import create_pixel_reward_map, batch_create_pixel_reward_maps
from ..advantage.dense_advantage_processor import DenseAdvantageProcessor

logger = setup_logger(__name__)


class DenseGRPOTrainer(GRPOTrainer):
    """
    DenseGRPO Trainer with pixel-level advantages.

    Key differences from standard GRPO:
    - Rewards are spatial (H, W) maps, not scalars
    - Advantages are spatial (B, T, H, W), computed per-pixel per-group
    - Policy ratio is spatial via reference-aligned decomposition
    - No advantage clipping (spatial info makes it unnecessary)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Read DenseGRPO-specific parameters from extra_kwargs
        ek = self.training_args.extra_kwargs
        self.spatial_alpha = ek.get('spatial_grpo_alpha', 0.5)
        self.spatial_alpha_artifact = ek.get('spatial_grpo_alpha_artifact', self.spatial_alpha)
        self.spatial_alpha_misalignment = ek.get('spatial_grpo_alpha_misalignment', self.spatial_alpha)
        self.sigma_min = ek.get('dense_sigma_min', 0.1)
        self.min_penalty = ek.get('dense_min_penalty', 0.02)
        self.importance_weighting = ek.get('bbox_importance_weighting', True)
        self.stack_same_label = ek.get('bbox_stack_same_label', False)
        self.area_normalize_artifact = ek.get('area_normalize_artifact', False)
        self.area_normalize_misalignment = ek.get('area_normalize_misalignment', False)
        self.pixel_ratio_mode = ek.get('pixel_ratio_mode', 'direct')  # 'direct' or 'detached'
        self.adv_smooth_kernel = ek.get('adv_smooth_kernel', 0)  # 0=disabled, odd int=gaussian kernel size
        self.fixed_noise_seed = ek.get('fixed_noise_seed', False)  # True=same noise every epoch
        self.disable_adaptive_alpha = ek.get('disable_adaptive_alpha', False)  # True=skip per-group std scaling

        # Replace scheduler with Dense variant for spatial log_prob
        from ..scheduler.dense_flow_match_euler_discrete import DenseFlowMatchEulerDiscreteSDEScheduler
        old_scheduler = self.adapter.scheduler
        new_scheduler = DenseFlowMatchEulerDiscreteSDEScheduler.from_scheduler(old_scheduler)
        self.adapter.scheduler = new_scheduler
        self.adapter.pipeline.scheduler = new_scheduler

        # Initialize dense advantage processor
        self.dense_advantage_processor = DenseAdvantageProcessor(
            log_func=self.log_data,
            verbose=True,
        )

        logger.info(
            f"DenseGRPO initialized: alpha={self.spatial_alpha}, sigma_min={self.sigma_min}, "
            f"min_penalty={self.min_penalty}, importance_weighting={self.importance_weighting}, "
            f"pixel_ratio_mode={self.pixel_ratio_mode}, adv_smooth_kernel={self.adv_smooth_kernel}, "
            f"fixed_noise_seed={self.fixed_noise_seed}"
        )

    # =========================== Wandb Visual Logging ============================
    def _log_dense_visuals(
        self,
        samples: List[BaseSample],
        reward_maps: torch.Tensor,
        scalar_rewards: List[float],
        all_bboxes: List[List[dict]],
        advantages_spatial: torch.Tensor,
    ):
        """Log generated images, bbox overlays, reward map heatmaps, and advantage maps to wandb."""
        if not self.accelerator.is_main_process:
            return

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.cm as cm

        num_vis = min(8, len(samples))
        log_images = []
        resolution = self.training_args.resolution
        if isinstance(resolution, (list, tuple)):
            resolution = resolution[0]

        for i in range(num_vis):
            s = samples[i]
            prompt = s.prompt if hasattr(s, 'prompt') and s.prompt else f"sample_{i}"
            prompt_short = prompt[:80]
            sr = scalar_rewards[i]

            # 1. Generated image
            if hasattr(s, 'image') and s.image is not None:
                pil_img = s.image if isinstance(s.image, Image.Image) else None
                if pil_img is None and isinstance(s.image, (torch.Tensor, np.ndarray)):
                    img_np = s.image
                    if isinstance(img_np, torch.Tensor):
                        img_np = img_np.float().cpu().numpy()
                    if img_np.ndim == 3 and img_np.shape[0] in (1, 3, 4):
                        img_np = img_np.transpose(1, 2, 0)
                    if img_np.max() <= 1.0:
                        img_np = (img_np * 255).astype(np.uint8)
                    else:
                        img_np = img_np.astype(np.uint8)
                    pil_img = Image.fromarray(img_np).resize((resolution, resolution))
            else:
                pil_img = None

            if pil_img is not None:
                # 2. BBox overlay
                bbox_img = pil_img.copy()
                draw = ImageDraw.Draw(bbox_img)
                bboxes = all_bboxes[i] if i < len(all_bboxes) else []
                for bbox_info in bboxes:
                    if 'box_2d' not in bbox_info:
                        continue
                    x0, y0, x1, y1 = bbox_info['box_2d']
                    # Convert 0-1000 to image coords
                    px0 = int(x0 / 1000 * resolution)
                    py0 = int(y0 / 1000 * resolution)
                    px1 = int(x1 / 1000 * resolution)
                    py1 = int(y1 / 1000 * resolution)
                    # Guard against invalid bboxes (x1<x0 or y1<y0) from VLM
                    if px1 < px0:
                        px0, px1 = px1, px0
                    if py1 < py0:
                        py0, py1 = py1, py0
                    if px1 == px0 or py1 == py0:
                        continue  # Skip degenerate bboxes
                    label = bbox_info.get('label', 'artifact')
                    imp = bbox_info.get('importance', '')
                    color = 'red' if label == 'misalignment' else 'blue'
                    draw.rectangle([px0, py0, px1, py1], outline=color, width=2)
                    draw.text((px0, max(0, py0 - 12)), f"{label[:4]} {imp}", fill=color)

                log_images.append(LogImage(
                    pil_img,
                    caption=f"{prompt_short} | reward={sr:.3f}",
                ))
                log_images.append(LogImage(
                    bbox_img,
                    caption=f"[BBOX] {prompt_short} | {len(bboxes)} boxes",
                ))

            # 3. Reward map heatmap
            rmap = reward_maps[i].cpu().numpy()  # (H, W)
            rmap_norm = (rmap - rmap.min()) / (rmap.max() - rmap.min() + 1e-8)
            rmap_colored = cm.jet(rmap_norm)[:, :, :3]
            rmap_colored = (rmap_colored * 255).astype(np.uint8)
            rmap_pil = Image.fromarray(rmap_colored).resize((resolution, resolution), Image.NEAREST)

            if pil_img is not None:
                blended = Image.blend(pil_img.convert('RGB'), rmap_pil, alpha=0.4)
                log_images.append(LogImage(
                    blended,
                    caption=f"[RMAP] {prompt_short} | min={rmap.min():.3f} max={rmap.max():.3f}",
                ))

            # 4. Spatial advantage heatmap (first timestep)
            adv_map = advantages_spatial[i, 0].cpu().numpy()  # (H, W) for t=0
            adv_abs_max = max(abs(adv_map.min()), abs(adv_map.max()), 1e-8)
            adv_norm = (adv_map + adv_abs_max) / (2 * adv_abs_max)  # map to [0, 1], 0.5 = neutral
            adv_colored = cm.RdBu_r(adv_norm)[:, :, :3]
            adv_colored = (adv_colored * 255).astype(np.uint8)
            adv_pil = Image.fromarray(adv_colored).resize((resolution, resolution), Image.NEAREST)
            log_images.append(LogImage(
                adv_pil,
                caption=f"[ADV] {prompt_short} | mean={adv_map.mean():.3f} std={adv_map.std():.3f}",
            ))

        # Log all images + scalar summary
        log_data = {
            'dense_grpo/visuals': log_images,
            'dense_grpo/reward_mean': np.mean(scalar_rewards),
            'dense_grpo/reward_std': np.std(scalar_rewards),
            'dense_grpo/reward_map_mean': reward_maps.mean().item(),
            'dense_grpo/reward_map_std': reward_maps.std().item(),
            'dense_grpo/adv_spatial_mean': advantages_spatial.mean().item(),
            'dense_grpo/adv_spatial_std': advantages_spatial.std().item(),
            'dense_grpo/num_bboxes_mean': np.mean([len(b) for b in all_bboxes]),
            'dense_grpo/num_bboxes_total': sum(len(b) for b in all_bboxes),
            'dense_grpo/nonempty_bbox_ratio': np.mean([1 if len(b) > 0 else 0 for b in all_bboxes]),
        }
        self.log_data(log_data, step=self.step)

    # =========================== Main Loop ============================
    def start(self):
        """Main training loop — skip initial eval (epoch=0)."""
        while self.should_continue_training():
            self.adapter.scheduler.set_seed(self.epoch + self.training_args.seed)

            # Save checkpoint
            if (
                self.log_args.save_freq > 0 and
                self.epoch % self.log_args.save_freq == 0 and
                self.log_args.save_dir
            ):
                save_dir = os.path.join(
                    self.log_args.save_dir,
                    str(self.log_args.run_name),
                    'checkpoints',
                )
                self.save_checkpoint(save_dir, epoch=self.epoch)

            # Evaluation — skip epoch 0
            if (
                self.eval_args.eval_freq > 0 and
                self.epoch % self.eval_args.eval_freq == 0 and
                self.epoch > 0
            ):
                self.evaluate()

            samples = self.sample()
            self.optimize(samples)

            self.adapter.ema_step(step=self.epoch)

            self.epoch += 1

    # =========================== Sampling ============================
    def sample(self) -> List[BaseSample]:
        """Generate rollouts with spatial log_prob collection."""
        self.adapter.rollout()
        self.reward_buffer.clear()
        samples = []
        data_iter = iter(self.dataloader)
        trajectory_indices = compute_trajectory_indices(
            train_timestep_indices=self.adapter.scheduler.train_timesteps,
            num_inference_steps=self.training_args.num_inference_steps,
        )

        with torch.no_grad(), self.autocast():
            for batch_index in tqdm(
                range(self.training_args.num_batches_per_epoch),
                desc=f'Epoch {self.epoch} Sampling',
                disable=not self.show_progress_bar,
            ):
                batch = next(data_iter)
                sample_kwargs = {
                    **self.training_args,
                    'compute_log_prob': True,
                    'trajectory_indices': trajectory_indices,
                    'extra_call_back_kwargs': ['log_prob_spatial'],  # DenseGRPO: capture spatial log_prob
                    **batch,
                }
                if self.fixed_noise_seed:
                    # Same initial noise per prompt across all epochs (deterministic z_T)
                    sample_kwargs['generator'] = create_generator_by_prompt(
                        batch['prompt'], self.training_args.seed,
                    )
                sample_kwargs = filter_kwargs(self.adapter.inference, **sample_kwargs)
                sample_batch = self.adapter.inference(**sample_kwargs)
                samples.extend(sample_batch)
                self.reward_buffer.add_samples(sample_batch)

        return samples

    # =========================== Optimization ============================
    def optimize(self, samples: List[BaseSample]) -> None:
        """Main training loop with dense (pixel-level) advantages."""

        # ---- Step 1: Finalize rewards and extract bboxes ----
        rewards = self.reward_buffer.finalize(store_to_samples=True, split='all')

        # Also compute standard scalar advantages (for logging/comparison)
        scalar_advantages = self.compute_advantages(samples, rewards, store_to_samples=True)

        # ---- Step 2: Extract bboxes and scalar rewards from samples ----
        scalar_rewards_list = []
        all_bboxes_list = []
        unique_ids = []

        for s in samples:
            # Scalar reward from extra_kwargs['rewards'] dict {name: value}
            reward_dict = s.extra_kwargs.get('rewards', {})
            if isinstance(reward_dict, dict):
                # Average all reward values
                vals = []
                for v in reward_dict.values():
                    if isinstance(v, torch.Tensor):
                        vals.append(v.item())
                    elif isinstance(v, (int, float)):
                        vals.append(float(v))
                sr = float(np.mean(vals)) if vals else 0.5
            elif isinstance(reward_dict, torch.Tensor):
                sr = reward_dict.mean().item()
            else:
                sr = 0.5
            scalar_rewards_list.append(sr)

            # Bboxes: stored in sample.extra_kwargs by the reward model
            # CombinedUR2BBoxReward stores them in extra_info during __call__,
            # but RewardProcessor only saves scalar rewards to samples.
            # We need to get bboxes from the reward buffer's cached extra_info.
            bboxes = s.extra_kwargs.get('bboxes', [])
            all_bboxes_list.append(bboxes if bboxes else [])

            unique_ids.append(s.unique_id)

        # ---- Step 3: Compute adaptive alpha via per-group scalar std ----
        if self.disable_adaptive_alpha:
            per_sample_group_stds = None
        else:
            scalar_rewards_tensor = torch.tensor(scalar_rewards_list, device=self.accelerator.device, dtype=torch.float32)
            per_sample_group_stds = self.dense_advantage_processor.compute_per_group_scalar_stds(
                self.accelerator, scalar_rewards_tensor, unique_ids,
            )

        # ---- Step 4: Build pixel reward maps ----
        # Get latent spatial size from first sample's log_prob_spatial
        latent_H, latent_W = 64, 64  # default
        for s in samples:
            lps = s.extra_kwargs.get('log_prob_spatial')
            if lps is not None:
                if lps.ndim == 3:
                    # Image model: per-sample (T', H, W)
                    latent_H, latent_W = lps.shape[-2], lps.shape[-1]
                elif lps.ndim == 2:
                    # Sequence model (FLUX): per-sample (T', L) where L = H * W
                    # Infer H = W = sqrt(L)
                    import math
                    L = lps.shape[-1]
                    side = int(math.isqrt(L))
                    logger.info(f"Sequence model detected: lps.shape={lps.shape}, L={L}, sqrt={side}, side²={side*side}")
                    if side * side == L:
                        latent_H, latent_W = side, side
                    else:
                        # Non-perfect-square: find closest factors
                        for h in range(side, 0, -1):
                            if L % h == 0:
                                latent_H, latent_W = h, L // h
                                break
                break

        reward_maps = batch_create_pixel_reward_maps(
            scalar_rewards=scalar_rewards_list,
            all_bboxes=all_bboxes_list,
            latent_size=latent_H,
            image_size=self.training_args.resolution[0] if isinstance(self.training_args.resolution, (list, tuple)) else self.training_args.resolution,
            alpha=self.spatial_alpha,
            alpha_artifact=self.spatial_alpha_artifact,
            alpha_misalignment=self.spatial_alpha_misalignment,
            min_penalty=self.min_penalty,
            importance_weighting=self.importance_weighting,
            area_normalize_artifact=self.area_normalize_artifact,
            area_normalize_misalignment=self.area_normalize_misalignment,
            per_sample_group_stds=per_sample_group_stds,
            device=self.accelerator.device,
            stack_same_label=self.stack_same_label,
        )  # (B_local, H, W)

        # ---- Step 5: Compute dense advantages ----
        num_train_timesteps = len(self.adapter.scheduler.train_timesteps)
        advantages_spatial = self.dense_advantage_processor.compute_dense_advantages(
            accelerator=self.accelerator,
            reward_maps=reward_maps,
            unique_ids=unique_ids,
            num_timesteps=num_train_timesteps,
            sigma_min=self.sigma_min,
            smooth_kernel=self.adv_smooth_kernel,
        )  # (B_local, T, H, W)

        # Store dense advantages to samples for batching
        for i, s in enumerate(samples):
            s.extra_kwargs['advantage_spatial'] = advantages_spatial[i]  # (T, H, W)

        # ---- Step 5b: Log per-type bbox penalty stats ----
        if self.accelerator.is_main_process:
            from ..utils.pixel_reward_map import create_pixel_reward_map
            artifact_penalties, misalign_penalties = [], []
            for i, (sr, bboxes) in enumerate(zip(scalar_rewards_list, all_bboxes_list)):
                if not bboxes:
                    continue
                bboxes_valid = [b for b in bboxes if isinstance(b, dict)]
                art_bboxes = [b for b in bboxes_valid if b.get('label', 'artifact') != 'misalignment']
                mis_bboxes = [b for b in bboxes_valid if b.get('label') == 'misalignment']
                latent_sz = reward_maps.shape[-1]
                img_sz = self.training_args.resolution
                if isinstance(img_sz, (list, tuple)):
                    img_sz = img_sz[0]
                if art_bboxes:
                    art_map = create_pixel_reward_map(
                        art_bboxes, 0.0, latent_sz, img_sz,
                        alpha=self.spatial_alpha_artifact,
                        importance_weighting=self.importance_weighting,
                    )
                    artifact_penalties.append(-art_map.mean().item())
                if mis_bboxes:
                    mis_map = create_pixel_reward_map(
                        mis_bboxes, 0.0, latent_sz, img_sz,
                        alpha=self.spatial_alpha_misalignment,
                        importance_weighting=self.importance_weighting,
                    )
                    misalign_penalties.append(-mis_map.mean().item())

            penalty_log = {
                'dense_grpo/artifact_penalty_mean': np.mean(artifact_penalties) if artifact_penalties else 0.0,
                'dense_grpo/artifact_penalty_count': len(artifact_penalties),
                'dense_grpo/misalign_penalty_mean': np.mean(misalign_penalties) if misalign_penalties else 0.0,
                'dense_grpo/misalign_penalty_count': len(misalign_penalties),
            }
            self.log_data(penalty_log, step=self.step)

        # ---- Step 5c: Log visuals to wandb ----
        self._log_dense_visuals(
            samples, reward_maps, scalar_rewards_list, all_bboxes_list, advantages_spatial,
        )

        # ---- Step 6: Training loop ----
        for inner_epoch in range(self.training_args.num_inner_epochs):
            perm_gen = create_generator(self.training_args.seed, self.epoch, inner_epoch)
            perm = torch.randperm(len(samples), generator=perm_gen)
            shuffled_samples = [samples[i] for i in perm]

            sample_batches = [
                BaseSample.stack(shuffled_samples[i:i + self.training_args.per_device_batch_size])
                for i in range(0, len(shuffled_samples), self.training_args.per_device_batch_size)
            ]

            self.adapter.train()
            loss_info = defaultdict(list)

            with self.autocast():
                for batch_idx, batch in enumerate(tqdm(
                    sample_batches,
                    total=len(sample_batches),
                    desc=f'Epoch {self.epoch} Training',
                    position=0,
                    disable=not self.show_progress_bar,
                )):
                    latents_index_map = batch['latent_index_map']
                    log_probs_index_map = batch['log_prob_index_map']
                    callback_index_map = batch['callback_index_map'][0]

                    for idx, timestep_index in enumerate(tqdm(
                        self.adapter.scheduler.train_timesteps,
                        desc=f'Epoch {self.epoch} Timestep',
                        position=1, leave=False,
                        disable=not self.show_progress_bar,
                    )):
                        with self.accelerator.accumulate(*self.adapter.trainable_components):
                            # 1. Prepare inputs
                            old_log_prob = batch['log_probs'][:, log_probs_index_map[timestep_index]]
                            num_timesteps_total = batch['timesteps'].shape[1]
                            t = batch['timesteps'][:, timestep_index]
                            t_next = (
                                batch['timesteps'][:, timestep_index + 1]
                                if timestep_index + 1 < num_timesteps_total
                                else torch.tensor(0, device=self.accelerator.device)
                            )
                            latents = batch['all_latents'][:, latents_index_map[timestep_index]]
                            next_latents = batch['all_latents'][:, latents_index_map[timestep_index + 1]]

                            forward_inputs = {
                                **self.training_args,
                                't': t,
                                't_next': t_next,
                                'latents': latents,
                                'next_latents': next_latents,
                                'compute_log_prob': True,
                                'noise_level': self.adapter.scheduler.noise_level,
                                **batch,
                            }
                            forward_inputs = filter_kwargs(self.adapter.forward, **forward_inputs)

                            # 2. Forward pass — request spatial log_prob
                            return_kwargs = ['log_prob', 'log_prob_spatial', 'dt']
                            if self.enable_kl_loss:
                                return_kwargs.append('next_latents_mean')
                                if self.training_args.kl_type == 'v-based':
                                    return_kwargs.append('noise_pred')

                            forward_inputs['return_kwargs'] = return_kwargs
                            output = self.adapter.forward(**forward_inputs)

                            # 3. Dense advantage for this timestep (B, H, W) — NO clamp
                            adv = batch['advantage_spatial'][:, idx]  # (B, H, W)

                            # 4. Pixel ratio computation
                            log_prob_spatial = output.log_prob_spatial  # (B, H, W) or (B, L) for sequence models
                            old_log_prob_spatial = batch['log_prob_spatial'][:, callback_index_map[timestep_index]]

                            # For sequence models (FLUX): reshape (B, L) → (B, H, W) to match advantage
                            if log_prob_spatial.ndim == 2 and adv.ndim == 3:
                                H, W = adv.shape[-2], adv.shape[-1]
                                log_prob_spatial = log_prob_spatial[:, :H*W].reshape(-1, H, W)
                                old_log_prob_spatial = old_log_prob_spatial[:, :H*W].reshape(-1, H, W)

                            # Scalar ratio for logging
                            ratio_mean = torch.exp(output.log_prob - old_log_prob)  # (B,)

                            if self.pixel_ratio_mode == 'detached':
                                # flow_grpo reference-aligned decomposition:
                                # pixel ratio = detached(scalar_ratio) * exp(log_prob_spatial) / detach(exp(log_prob_spatial))
                                si = ratio_mean.detach().reshape(-1, *([1] * (log_prob_spatial.ndim - 1)))
                                pitheta = torch.exp(log_prob_spatial)
                                normpitheta = pitheta.detach()
                                ratio = si * pitheta / (normpitheta + 1e-10)  # (B, H, W)
                            else:
                                # Direct pixel ratio (default)
                                ratio = torch.exp(log_prob_spatial - old_log_prob_spatial)  # (B, H, W)

                            # 5. PPO-style clipped loss (spatial)
                            ratio_clip_range = self.training_args.clip_range
                            unclipped_loss = -adv * ratio
                            clipped_loss = -adv * torch.clamp(
                                ratio,
                                1.0 + ratio_clip_range[0],
                                1.0 + ratio_clip_range[1],
                            )
                            # Reduce spatial dims then batch
                            policy_loss = (
                                torch.maximum(unclipped_loss, clipped_loss)
                                .mean(dim=tuple(range(1, clipped_loss.ndim)))
                                .mean()
                            )

                            loss = policy_loss

                            # 6. KL divergence (same as standard GRPO, scalar)
                            if self.enable_kl_loss:
                                with torch.no_grad(), self.adapter.use_ref_parameters():
                                    ref_forward_inputs = forward_inputs.copy()
                                    ref_forward_inputs['compute_log_prob'] = False
                                    if self.training_args.kl_type == 'v-based':
                                        ref_forward_inputs['return_kwargs'] = ['noise_pred']
                                        ref_output = self.adapter.forward(**ref_forward_inputs)
                                        kl_div = torch.mean(
                                            ((output.noise_pred - ref_output.noise_pred) ** 2),
                                            dim=tuple(range(1, output.noise_pred.ndim)), keepdim=True,
                                        )
                                    elif self.training_args.kl_type == 'x-based':
                                        ref_forward_inputs['return_kwargs'] = ['next_latents_mean']
                                        ref_output = self.adapter.forward(**ref_forward_inputs)
                                        kl_div = torch.mean(
                                            ((output.next_latents_mean - ref_output.next_latents_mean) ** 2),
                                            dim=tuple(range(1, output.next_latents_mean.ndim)), keepdim=True,
                                        )

                                kl_div = torch.mean(kl_div)
                                kl_loss = self.training_args.kl_beta * kl_div
                                loss += kl_loss
                                loss_info['kl_div'].append(kl_div.detach())
                                loss_info['kl_loss'].append(kl_loss.detach())

                            # 7. Log info
                            loss_info['ratio_mean'].append(ratio_mean.mean().detach())
                            loss_info['ratio_mean_min'].append(ratio_mean.min().detach())
                            loss_info['ratio_mean_max'].append(ratio_mean.max().detach())
                            # Pixel ratio stats (the actual ratio used in loss)
                            loss_info['pixel_ratio_mean'].append(ratio.mean().detach())
                            loss_info['pixel_ratio_min'].append(ratio.min().detach())
                            loss_info['pixel_ratio_max'].append(ratio.max().detach())
                            loss_info['pixel_ratio_std'].append(ratio.std().detach())
                            loss_info['policy_loss'].append(policy_loss.detach())
                            loss_info['loss'].append(loss.detach())
                            loss_info['adv_spatial_mean'].append(adv.mean().detach())
                            loss_info['adv_spatial_std'].append(adv.std().detach())

                            # Clip frac based on pixel ratio
                            clip_frac_high = torch.mean((ratio > 1.0 + ratio_clip_range[1]).float())
                            clip_frac_low = torch.mean((ratio < 1.0 + ratio_clip_range[0]).float())
                            loss_info["clip_frac_high"].append(clip_frac_high.detach())
                            loss_info["clip_frac_low"].append(clip_frac_low.detach())
                            loss_info['clip_frac_total'].append((clip_frac_high + clip_frac_low).detach())

                            # 8. Backward and optimizer step
                            self.accelerator.backward(loss)
                            if self.accelerator.sync_gradients:
                                grad_norm = self.accelerator.clip_grad_norm_(
                                    self.adapter.get_trainable_parameters(),
                                    self.training_args.max_grad_norm,
                                )
                                self.optimizer.step()
                                self.optimizer.zero_grad()
                                loss_info = {
                                    k: torch.stack(v).mean()
                                    for k, v in loss_info.items()
                                }
                                loss_info = self.accelerator.reduce(loss_info, reduction="mean")
                                loss_info['grad_norm'] = grad_norm
                                self.log_data(
                                    {f'train/{k}': v for k, v in loss_info.items()},
                                    step=self.step,
                                )
                                self.step += 1
                                loss_info = defaultdict(list)
