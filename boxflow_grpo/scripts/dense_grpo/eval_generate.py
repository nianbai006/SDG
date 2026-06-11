#!/usr/bin/env python
"""
Generate DrawBench images from a checkpoint (base or LoRA-merged).
Supports SD3.5 and FLUX2-klein with multi-GPU parallel generation.

Usage:
    # FLUX2-klein with LoRA
    python eval_generate.py --model_type flux2klein --lora_path /path/to/checkpoint --output_dir /path/to/out --num_gpus 8

    # SD3.5 with LoRA
    python eval_generate.py --model_type sd3 --lora_path /path/to/checkpoint --output_dir /path/to/out --num_gpus 8

    # FLUX2-klein base (no LoRA)
    python eval_generate.py --model_type flux2klein --output_dir /path/to/out --num_gpus 8
"""
import argparse
import os
import sys
import torch
import numpy as np
from multiprocessing import Process, set_start_method

BASE_MODELS = {
    "sd3": "stabilityai/stable-diffusion-3.5-medium",
    "flux2klein": "black-forest-labs/FLUX.2-klein-4B",
    "flux2klein-base": "black-forest-labs/FLUX.2-klein-base-4B",
    "flux1": "black-forest-labs/FLUX.1-dev",
}

DEFAULTS = {
    "sd3": {"resolution": 512, "num_steps": 40, "guidance_scale": 4.5},
    "flux2klein": {"resolution": 512, "num_steps": 28, "guidance_scale": 4.0},
    "flux2klein-base": {"resolution": 512, "num_steps": 28, "guidance_scale": 4.0},
    "flux1": {"resolution": 512, "num_steps": 28, "guidance_scale": 3.5},
}

PROMPTS_FILE = "${SDG_HOME}/flow_grpo/dataset/drawbench/test.txt"


def load_prompts(path, max_samples=999):
    with open(path) as f:
        return [l.strip() for l in f if l.strip()][:max_samples]


def gpu_worker(physical_gpu, prompt_indices, model_type, base_model, lora_path,
               output_dir, resolution, num_steps, guidance_scale, prompts):
    """Single GPU worker: load model, generate assigned prompts."""
    # Each worker gets exclusive access to one physical GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
    import torch
    from peft import PeftModel

    device = "cuda:0"

    if model_type in ("flux2klein", "flux2klein-base"):
        from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline
        pipe = Flux2KleinPipeline.from_pretrained(base_model, low_cpu_mem_usage=False, torch_dtype=torch.bfloat16)
    elif model_type == "flux1":
        from diffusers import FluxPipeline
        pipe = FluxPipeline.from_pretrained(base_model, torch_dtype=torch.bfloat16)
    else:
        from diffusers import StableDiffusion3Pipeline
        pipe = StableDiffusion3Pipeline.from_pretrained(base_model, torch_dtype=torch.bfloat16)

    if lora_path:
        pipe.transformer = PeftModel.from_pretrained(pipe.transformer, lora_path, is_trainable=False)
        pipe.transformer = pipe.transformer.merge_and_unload()

    pipe = pipe.to(device)

    for idx in prompt_indices:
        img_path = os.path.join(output_dir, f"{idx:04d}.png")
        if os.path.exists(img_path):
            continue
        img = pipe(
            prompt=prompts[idx],
            height=resolution,
            width=resolution,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
        ).images[0]
        img.save(img_path)

    del pipe
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", required=True, choices=["sd3", "flux2klein", "flux2klein-base", "flux1"])
    parser.add_argument("--lora_path", default=None, help="LoRA checkpoint dir (omit for base model)")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_gpus", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--guidance_scale", type=float, default=None)
    parser.add_argument("--prompts_file", default=PROMPTS_FILE)
    args = parser.parse_args()

    base_model = BASE_MODELS[args.model_type]
    defaults = DEFAULTS[args.model_type]
    resolution = args.resolution or defaults["resolution"]
    num_steps = args.num_steps or defaults["num_steps"]
    guidance_scale = args.guidance_scale or defaults["guidance_scale"]

    prompts = load_prompts(args.prompts_file)
    os.makedirs(args.output_dir, exist_ok=True)

    # Check which images already exist
    remaining = [i for i in range(len(prompts)) if not os.path.exists(os.path.join(args.output_dir, f"{i:04d}.png"))]
    if not remaining:
        print(f"All {len(prompts)} images already exist, skipping generation.")
        return

    print(f"Generating {len(remaining)}/{len(prompts)} images with {args.num_gpus} GPUs")
    print(f"  Model: {base_model}")
    print(f"  LoRA: {args.lora_path or 'None (base model)'}")
    print(f"  Resolution: {resolution}, Steps: {num_steps}, CFG: {guidance_scale}")

    # Split indices across GPUs
    # Resolve physical GPU IDs from CUDA_VISIBLE_DEVICES
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible:
        physical_gpus = [int(x) for x in visible.split(",")][:args.num_gpus]
    else:
        physical_gpus = list(range(args.num_gpus))

    chunks = np.array_split(remaining, len(physical_gpus))

    processes = []
    for physical_gpu, chunk in zip(physical_gpus, chunks):
        if len(chunk) == 0:
            continue
        p = Process(target=gpu_worker, args=(
            physical_gpu, chunk.tolist(), args.model_type, base_model, args.lora_path,
            args.output_dir, resolution, num_steps, guidance_scale, prompts,
        ))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Verify
    generated = sum(1 for i in range(len(prompts)) if os.path.exists(os.path.join(args.output_dir, f"{i:04d}.png")))
    print(f"Generation complete: {generated}/{len(prompts)} images in {args.output_dir}")


if __name__ == "__main__":
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass
    main()
