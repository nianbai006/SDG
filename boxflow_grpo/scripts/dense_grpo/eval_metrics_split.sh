#!/bin/bash
# Run the reference T2I metrics on pre-generated images using two Python envs.
# flow_grpo:    imagereward + clipscore
# flow-factory: pickscore
#
# Usage: bash eval_metrics_split.sh <images_dir> <output_json> [gpu_id]
set -euo pipefail

IMAGES_DIR="$1"
OUTPUT_JSON="$2"
GPU_ID="${3:-0}"
PROMPTS="${SDG_HOME}/flow_grpo/dataset/drawbench/test.txt"

FF_PY=${SDG_HOME}/../envs/flow-factory/bin/python
FG_PY=${SDG_HOME}/../envs/flow_grpo/bin/python

OUT_DIR=$(dirname "$OUTPUT_JSON")
mkdir -p "$OUT_DIR"

PARTIAL_3="$OUT_DIR/.partial_ircp.json"
PARTIAL_PS="$OUT_DIR/.partial_pickscore.json"

echo "[METRICS] Images: $IMAGES_DIR"
echo "[METRICS] GPU: $GPU_ID"

# --- imagereward + clipscore via flow_grpo env ---
if [ ! -f "$PARTIAL_3" ]; then
echo "[METRICS] Running imagereward + clipscore (flow_grpo env)..."
CUDA_VISIBLE_DEVICES=$GPU_ID HF_HOME=${SDG_HOME}/../cache/huggingface \
  _EVAL_IMAGES_DIR="$IMAGES_DIR" _EVAL_PROMPTS="$PROMPTS" _EVAL_OUTPUT="$PARTIAL_3" \
  $FG_PY << 'PYEOF'
import os, json, torch, numpy as np
from PIL import Image
from tqdm import tqdm

images_dir = os.environ["_EVAL_IMAGES_DIR"]
prompts_file = os.environ["_EVAL_PROMPTS"]
output_file = os.environ["_EVAL_OUTPUT"]
device = "cuda:0"

with open(prompts_file) as f:
    prompts = [l.strip() for l in f if l.strip()][:999]
images = []
for i in range(len(prompts)):
    p = os.path.join(images_dir, f"{i:04d}.png")
    images.append(Image.open(p).convert("RGB") if os.path.exists(p) else None)
valid = [(p, img) for p, img in zip(prompts, images) if img is not None]
prompts_v, images_v = [v[0] for v in valid], [v[1] for v in valid]
print(f"Loaded {len(images_v)} images")

results = {}

# ImageReward
import ImageReward as RM
model = RM.load("ImageReward-v1.0", device=device)
scores = [model.score(p, img) for p, img in tqdm(zip(prompts_v, images_v), total=len(prompts_v), desc="ImageReward")]
m, s = float(np.mean(scores)), float(np.std(scores))
results["imagereward"] = {"mean": m, "std": s}
print(f"  imagereward: {m:.4f} +/- {s:.4f}")
del model; torch.cuda.empty_cache()

# CLIPScore
from transformers import CLIPProcessor, CLIPModel
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
scores = []
for p, img in tqdm(zip(prompts_v, images_v), total=len(prompts_v), desc="CLIPScore"):
    inputs = processor(text=[p], images=[img], return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        out = model(**inputs)
    scores.append(out.logits_per_image.item() / 30.0)
m, s = float(np.mean(scores)), float(np.std(scores))
results["clipscore"] = {"mean": m, "std": s}
print(f"  clipscore: {m:.4f} +/- {s:.4f}")
del model, processor; torch.cuda.empty_cache()

with open(output_file, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved metrics to {output_file}")
PYEOF
fi

# --- PickScore with flow-factory ---
if [ ! -f "$PARTIAL_PS" ]; then
echo "[METRICS] Running pickscore (flow-factory env)..."
CUDA_VISIBLE_DEVICES=$GPU_ID HF_HOME=${SDG_HOME}/../cache/huggingface \
  _EVAL_IMAGES_DIR="$IMAGES_DIR" _EVAL_PROMPTS="$PROMPTS" _EVAL_OUTPUT="$PARTIAL_PS" \
  $FF_PY << 'PYEOF'
import os, json, torch, numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

images_dir = os.environ["_EVAL_IMAGES_DIR"]
prompts_file = os.environ["_EVAL_PROMPTS"]
output_file = os.environ["_EVAL_OUTPUT"]
device = "cuda:0"

with open(prompts_file) as f:
    prompts = [l.strip() for l in f if l.strip()][:999]
images = []
for i in range(len(prompts)):
    p = os.path.join(images_dir, f"{i:04d}.png")
    images.append(Image.open(p).convert("RGB") if os.path.exists(p) else None)
valid = [(p, img) for p, img in zip(prompts, images) if img is not None]
prompts_v, images_v = [v[0] for v in valid], [v[1] for v in valid]

from transformers import CLIPProcessor, CLIPModel
proc = CLIPProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
model = CLIPModel.from_pretrained("yuvalkirstain/PickScore_v1").eval().to(device)
scores = []
for p, img in tqdm(zip(prompts_v, images_v), total=len(prompts_v), desc="PickScore"):
    img_inputs = proc(images=[img], return_tensors="pt", padding=True, truncation=True)
    img_inputs = {k: v.to(device) for k, v in img_inputs.items()}
    txt_inputs = proc(text=[p], return_tensors="pt", padding=True, truncation=True, max_length=77)
    txt_inputs = {k: v.to(device) for k, v in txt_inputs.items()}
    with torch.no_grad():
        img_out = model.get_image_features(**img_inputs)
        img_embs = img_out.pooler_output if hasattr(img_out, "pooler_output") else img_out
        img_embs = img_embs / img_embs.norm(p=2, dim=-1, keepdim=True)
        txt_out = model.get_text_features(**txt_inputs)
        txt_embs = txt_out.pooler_output if hasattr(txt_out, "pooler_output") else txt_out
        txt_embs = txt_embs / txt_embs.norm(p=2, dim=-1, keepdim=True)
        logit_scale = model.logit_scale.exp()
        score = (logit_scale * (txt_embs @ img_embs.T)).item() / 26.0
    scores.append(score)
m, s = float(np.mean(scores)), float(np.std(scores))
result = {"pickscore": {"mean": m, "std": s}}
print(f"  pickscore: {m:.4f} +/- {s:.4f}")

with open(output_file, "w") as f:
    json.dump(result, f, indent=2)
PYEOF
fi

# --- Merge results ---
echo "[METRICS] Merging results..."
$FG_PY -c "
import json
r = {}
with open('$PARTIAL_3') as f: r.update(json.load(f))
with open('$PARTIAL_PS') as f: r.update(json.load(f))
with open('$OUTPUT_JSON', 'w') as f: json.dump(r, f, indent=2)

# Also write human-readable summary
txt = '$OUTPUT_JSON'.replace('.json', '.txt')
with open(txt, 'w') as f:
    f.write('EVALUATION SUMMARY\n' + '='*50 + '\n')
    for k, v in r.items():
        f.write(f'{k:20s}: {v[\"mean\"]:.4f} +/- {v[\"std\"]:.4f}\n')
print(open(txt).read())
"

# Cleanup partials
rm -f "$PARTIAL_3" "$PARTIAL_PS"
echo "[METRICS] Done: $OUTPUT_JSON"
