"""Evaluate HPSv3 scores on drawbench images for one or more model dirs."""
import os, sys, json, argparse
from pathlib import Path

os.environ.setdefault("HF_HOME", "${SDG_HOME}/../cache/huggingface")

import numpy as np
from hpsv3 import HPSv3RewardInferencer

DRAWBENCH = "${SDG_HOME}/flow_grpo/dataset/drawbench/test.txt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images_dir", type=str, help="Dir with 0000.png .. 0998.png")
    ap.add_argument("--outfile", default=None)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    # Load prompts
    with open(DRAWBENCH) as f:
        prompts = [l.strip() for l in f if l.strip()]
    print(f"Loaded {len(prompts)} prompts from drawbench")

    # Collect images
    img_dir = Path(args.images_dir)
    pairs = []
    for i, p in enumerate(prompts):
        img = img_dir / f"{i:04d}.png"
        if img.exists():
            pairs.append((str(img), p))
    print(f"Found {len(pairs)} images")

    print("Loading HPSv3 model...")
    inferencer = HPSv3RewardInferencer(device=args.device)
    print("Loaded.")

    scores = []
    B = args.batch_size
    import time
    t0 = time.time()
    for i in range(0, len(pairs), B):
        batch = pairs[i:i+B]
        imgs = [p[0] for p in batch]
        prs = [p[1] for p in batch]
        rewards = inferencer.reward(imgs, prs)
        for r in rewards:
            scores.append(float(r[0].item()))
        if i // B % 5 == 0:
            el = time.time() - t0
            rate = (i + len(batch)) / el
            print(f"  {i+len(batch)}/{len(pairs)}  {rate:.2f} img/s", flush=True)

    m, s = float(np.mean(scores)), float(np.std(scores))
    print(f"\n=> {args.images_dir}\n   HPSv3 mean={m:.4f} std={s:.4f}")

    out = args.outfile or str(img_dir.parent / "hpsv3_results.json")
    data = {"mean": m, "std": s, "scores": scores, "n": len(scores)}
    with open(out, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
