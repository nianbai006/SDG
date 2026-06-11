#!/usr/bin/env python3
"""
GPT evaluation with thinkpe_imp prompt (description + importance).
Uses internal gRPC proxy (kess.framework). Output compatible with evalcode/evaluate.py.

Usage:
    python sdg_detector/inference/eval_gpt_thinkpe_imp.py \
        --eval_dataset_path experiments/gemini_ann_prompt_20260315/results/2.5v5/prepared/distilled_2.5v5_filepath_test.jsonl \
        --output_dir eval_results_gpt_thinkpe_imp \
        --num_workers 50
"""
import sys
sys.path.append('<author_utils>')

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

import os
import io
import json
import re
import time
import argparse
import logging
import threading
from typing import Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image
from tqdm import tqdm

from kess.framework import ClientOption, GrpcClient
from mmu.mmu_chat_gpt_pb2 import MmuChatGptRequest
from mmu.mmu_chat_gpt_pb2_grpc import MmuChatGptServiceStub
from mmu.media_common_pb2 import ImgUnit

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

# gRPC biz name (GPT-4.1 from metrics_common.py)
DEFAULT_BIZ_NAME = "dupenghui_e3fe3622_gpt-4.1"

# Thread-local gRPC client
_thread_local = threading.local()

def get_thread_client(biz_name):
    if not hasattr(_thread_local, 'client') or _thread_local.biz_name != biz_name:
        client_option = ClientOption(
            biz_def=biz_name,
            grpc_service_name='mmu-chat-gpt-service',
            grpc_stub_class=MmuChatGptServiceStub,
        )
        _thread_local.client = GrpcClient(client_option)
        _thread_local.biz_name = biz_name
    return _thread_local.client

# Prompt: thinkpe_imp with Qwen coordinate format [x0, y0, x1, y1]
PROMPT_TEMPLATE = """You are an AI image quality evaluator. You will be given **one image** to analyze.

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
  (b) what is affected (object/part)
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
- 90-100: Critical defect, immediately obvious, ruins the image (e.g., missing limb, completely wrong subject, large garbled text).
- 70-89: Major defect, clearly visible at normal viewing distance (e.g., extra fingers, noticeable texture melting, prominent unreadable text).
- 40-69: Moderate defect, noticeable on closer inspection (e.g., minor hand deformity, slight color mismatch, small distorted detail).
- 15-39: Minor defect, only visible on careful examination (e.g., tiny floating speck, slight boundary blur, minor lighting inconsistency).
- 1-14: Negligible defect, barely perceptible (e.g., sub-pixel aliasing, faint halo, minuscule texture irregularity).

### Description Style Guide
For EACH box, write `description` with richer detail:
- 18 to 45 words per description.
- Must mention the affected object/part.
- Must include at least one concrete evidence phrase (e.g., fused contours, missing separations, distorted geometry, incorrect lettering, absent requested object, incorrect identity traits).
- Do NOT use vague words like "weird/strange/bad".
- Do NOT include numeric coordinates in description.

Now analyze the image and produce your output:
"""


def read_image_bytes(image_path: str, max_size: int = 1024) -> Optional[bytes]:
    """Read image, resize if >max_size, return PNG bytes."""
    try:
        img = Image.open(image_path)
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.LANCZOS)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        return buf.getvalue()
    except Exception as e:
        logger.warning(f"Failed to read image {image_path}: {e}")
        return None


def call_gpt_grpc(biz_name, image_bytes, prompt, timeout=180, max_retries=10):
    """Call GPT via internal gRPC proxy."""
    client = get_thread_client(biz_name)

    for attempt in range(max_retries):
        try:
            request = MmuChatGptRequest(biz=biz_name)
            request.session_id = biz_name
            request.req_id = str(int(time.time() * 1000000))
            request.query = prompt
            request.img.append(ImgUnit(image=image_bytes))

            resp = client.Chat(request, timeout=timeout)
            if resp.state.code == 1:
                answer = resp.answer
                low = answer.lower()
                if not ("erro" in low or "ratelimitreached" in low or "badrequest" in low):
                    return answer
        except Exception as e:
            if attempt == max_retries - 1:
                logger.warning(f"gRPC failed after {max_retries} retries: {e}")
        time.sleep(0.2 * (attempt + 1))

    return None


def process_sample(args_tuple):
    """Process a single sample."""
    sample, biz_name = args_tuple

    filepath = sample.get('filepath') or sample.get('filename', '')
    caption = sample.get('caption', '')

    if not filepath or not os.path.exists(filepath):
        return filepath, caption, None

    image_bytes = read_image_bytes(filepath)
    if image_bytes is None:
        return filepath, caption, None

    prompt = PROMPT_TEMPLATE.format(caption=caption)
    raw = call_gpt_grpc(biz_name, image_bytes, prompt)
    return filepath, caption, raw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dataset_path", required=True)
    parser.add_argument("--output_dir", default="./eval_results_gpt_thinkpe_imp")
    parser.add_argument("--biz_name", default=DEFAULT_BIZ_NAME,
                        help="gRPC biz name for GPT model")
    parser.add_argument("--num_workers", type=int, default=50)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    eval_data = []
    with open(args.eval_dataset_path) as f:
        for line in f:
            if line.strip():
                eval_data.append(json.loads(line))

    if args.max_samples and args.max_samples < len(eval_data):
        eval_data = eval_data[:args.max_samples]

    logger.info(f"Loaded {len(eval_data)} samples, biz={args.biz_name}, workers={args.num_workers}")

    tasks = [(s, args.biz_name) for s in eval_data]

    pred_path = os.path.join(args.output_dir, "predictions.jsonl")
    success = fail = total_boxes = 0

    with open(pred_path, 'w') as fout:
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {executor.submit(process_sample, t): i for i, t in enumerate(tasks)}
            for future in tqdm(as_completed(futures), total=len(futures), desc="GPT eval"):
                filepath, caption, raw = future.result()

                if raw is None:
                    fail += 1
                    pred = {"filepath": filepath, "caption": caption, "response": ""}
                else:
                    success += 1
                    m = re.search(r'<answer>\s*(.*?)\s*</answer>', raw, re.DOTALL)
                    if m:
                        try:
                            bboxes = json.loads(m.group(1).strip())
                            if isinstance(bboxes, list):
                                total_boxes += len(bboxes)
                        except:
                            pass
                    pred = {"filepath": filepath, "caption": caption, "response": raw}

                fout.write(json.dumps(pred, ensure_ascii=False) + '\n')

    logger.info(f"Done: {success} success, {fail} fail, {total_boxes} total pred boxes")
    logger.info(f"Predictions saved to {pred_path}")

    config = {
        "biz_name": args.biz_name,
        "num_workers": args.num_workers,
        "prompt": "thinkpe_imp (description + importance, Qwen coords [x0,y0,x1,y1])",
        "total_samples": len(eval_data),
        "success": success,
        "fail": fail,
        "total_pred_boxes": total_boxes,
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(args.output_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)


if __name__ == "__main__":
    main()
