#!/usr/bin/env python
"""Single-GPU metric evaluation on pre-generated images. Avoids multiprocessing issues on A800."""
import argparse, os, json, torch, numpy as np
from PIL import Image
from tqdm import tqdm

def load_images_and_prompts(images_dir, prompts_file, max_samples=1000):
    with open(prompts_file) as f:
        prompts = [l.strip() for l in f if l.strip()][:max_samples]
    images = []
    for i in range(len(prompts)):
        p = os.path.join(images_dir, f"{i:04d}.png")
        if os.path.exists(p):
            images.append(Image.open(p).convert("RGB"))
        else:
            images.append(None)
    valid = [(p, img) for p, img in zip(prompts, images) if img is not None]
    return [v[0] for v in valid], [v[1] for v in valid]

def eval_imagereward(prompts, images, device):
    import ImageReward as RM
    model = RM.load("ImageReward-v1.0", device=device)
    scores = []
    for p, img in tqdm(zip(prompts, images), total=len(prompts), desc="ImageReward"):
        scores.append(model.score(p, img))
    return scores

def eval_clipscore(prompts, images, device):
    from transformers import CLIPProcessor, CLIPModel
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    scores = []
    for p, img in tqdm(zip(prompts, images), total=len(prompts), desc="CLIPScore"):
        inputs = processor(text=[p], images=[img], return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            out = model(**inputs)
        scores.append(out.logits_per_image.item() / 100.0)
    return scores

def eval_pickscore(prompts, images, device):
    from transformers import AutoProcessor, AutoModel
    proc = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    model = AutoModel.from_pretrained("yuvalkirstain/PickScore_v1").eval().to(device)
    scores = []
    for p, img in tqdm(zip(prompts, images), total=len(prompts), desc="PickScore"):
        inputs = proc(images=[img], text=[p], return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            scores.append(model(**inputs).logits_per_image.item())
    return scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--prompts_file", default="${SDG_HOME}/flow_grpo/dataset/drawbench/test.txt")
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    prompts, images = load_images_and_prompts(args.images_dir, args.prompts_file)
    print(f"Loaded {len(images)} images")
    results = {}

    for name, fn in [("imagereward", eval_imagereward), ("clipscore", eval_clipscore),
                     ("pickscore", eval_pickscore)]:
        try:
            scores = fn(prompts, images, args.device)
            results[name] = {"mean": float(np.mean(scores)), "std": float(np.std(scores))}
            print(f"{name}: {results[name]['mean']:.4f} ± {results[name]['std']:.4f}")
        except Exception as e:
            print(f"{name} FAILED: {e}")

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)

    result_txt = args.output_file.replace(".json", ".txt").replace("results.json", "result.txt")
    with open(result_txt, "w") as f:
        f.write("EVALUATION SUMMARY\n" + "="*50 + "\n")
        for k, v in results.items():
            f.write(f"{k:20s}: {v['mean']:.4f} ± {v['std']:.4f}\n")
    print(f"\nSaved to {args.output_file}")
