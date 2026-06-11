#!/usr/bin/env python3
"""
Gemini evaluation with thinkpe_imp prompt (description + importance).
Output format compatible with evalcode/evaluate.py.

Usage:
    python sdg_detector/inference/eval_gemini_thinkpe_imp.py \
        --eval_dataset_path experiments/gemini_ann_prompt_20260315/results/2.5v5/prepared/distilled_2.5v5_filepath_test.jsonl \
        --output_dir eval_results_gemini_thinkpe_imp \
        --model gemini-2.5-pro \
        --num_workers 50
"""
import os
import json
import re
import time
import argparse
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image
from io import BytesIO
from tqdm import tqdm

from google import genai
from google.genai import types

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

# Use thinkpe_imp template from constants.py
# Gemini outputs [y0, x0, y1, x1], so we adjust the coordinate instruction
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
    {{"box_2d": [y0, x0, y1, x1], "label": "misalignment"|"artifact", "description": "brief description    of the issue", "importance": N}}
]

Bounding box coordinates are in normalized 0-1000 space: [y0, x0, y1, x1].
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


def image_to_bytes(image_path: str, max_size: int = 1024) -> Optional[bytes]:
    """Read image file and return PNG bytes, resizing if needed."""
    try:
        img = Image.open(image_path)
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.LANCZOS)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        return buffer.getvalue()
    except Exception as e:
        logger.warning(f"Failed to read image {image_path}: {e}")
        return None


def parse_response(raw_text: str) -> List[Dict]:
    """Parse <answer> block from Gemini response, convert [y0,x0,y1,x1] → [x0,y0,x1,y1]."""
    if not raw_text:
        return []

    # Try <answer> tag first
    m = re.search(r'<answer>\s*(.*?)\s*</answer>', raw_text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    else:
        # Fallback: try ```json block
        m2 = re.search(r'```(?:json)?\s*(.*?)\s*```', raw_text, re.DOTALL)
        if m2:
            text = m2.group(1).strip()
        else:
            # Fallback: try bare JSON array
            start = raw_text.find('[')
            end = raw_text.rfind(']')
            if start != -1 and end > start:
                text = raw_text[start:end + 1]
            else:
                return []

    if not text:
        return []

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        try:
            import ast
            data = ast.literal_eval(text)
        except:
            return []

    if not isinstance(data, list):
        return []

    # Convert Gemini [y0,x0,y1,x1] → Qwen [x0,y0,x1,y1]
    result = []
    for item in data:
        if not isinstance(item, dict) or 'box_2d' not in item:
            continue
        box = item['box_2d']
        if not isinstance(box, list) or len(box) != 4:
            continue
        y0, x0, y1, x1 = box
        result.append({
            'box_2d': [x0, y0, x1, y1],  # Qwen format
            'label': item.get('label', ''),
            'description': item.get('description', '') or item.get('desc', ''),
            'importance': item.get('importance'),
        })
    return result


def call_gemini(client, model_name, image_bytes, caption, temperature=0.5, max_retries=3):
    """Call Gemini API with retries."""
    prompt = PROMPT_TEMPLATE.format(caption=caption)
    content_parts = [
        prompt,
        types.Part.from_bytes(mime_type='image/png', data=image_bytes),
    ]

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=content_parts,
                config=types.GenerateContentConfig(temperature=temperature),
            )
            if response and getattr(response, 'text', None):
                return response.text
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))
            else:
                logger.warning(f"API failed after {max_retries} retries: {e}")
    return None


def process_sample(args_tuple):
    """Process a single sample. Returns (filepath, caption, response, parsed_bboxes)."""
    sample, model_name, project, location, temperature = args_tuple

    filepath = sample.get('filepath') or sample.get('filename', '')
    caption = sample.get('caption', '')

    if not filepath or not os.path.exists(filepath):
        return filepath, caption, None, []

    image_bytes = image_to_bytes(filepath)
    if image_bytes is None:
        return filepath, caption, None, []

    # Each thread creates its own client
    client = genai.Client(vertexai=True, project=project, location=location)
    raw = call_gemini(client, model_name, image_bytes, caption, temperature)
    if raw is None:
        return filepath, caption, None, []

    bboxes = parse_response(raw)
    return filepath, caption, raw, bboxes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dataset_path", required=True)
    parser.add_argument("--output_dir", default="./eval_results_gemini_thinkpe_imp")
    parser.add_argument("--model", default="gemini-2.5-pro")
    parser.add_argument("--project", default=os.environ.get("GEMINI_PROJECT", ""))
    parser.add_argument("--location", default="global")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=50)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_retries", type=int, default=3)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    eval_data = []
    with open(args.eval_dataset_path) as f:
        for line in f:
            if line.strip():
                eval_data.append(json.loads(line))

    if args.max_samples and args.max_samples < len(eval_data):
        eval_data = eval_data[:args.max_samples]

    logger.info(f"Loaded {len(eval_data)} samples, model={args.model}, workers={args.num_workers}")

    # Prepare tasks
    tasks = [(s, args.model, args.project, args.location, args.temperature) for s in eval_data]

    # Run with thread pool
    predictions = []
    success = 0
    fail = 0
    total_boxes = 0

    pred_path = os.path.join(args.output_dir, "predictions.jsonl")
    with open(pred_path, 'w') as fout:
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {executor.submit(process_sample, t): i for i, t in enumerate(tasks)}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Gemini eval"):
                filepath, caption, raw, bboxes = future.result()

                if raw is None:
                    fail += 1
                    # Still write empty prediction for alignment
                    pred = {"filepath": filepath, "caption": caption, "response": ""}
                else:
                    success += 1
                    total_boxes += len(bboxes)
                    # Build response string with <think> and <answer> for compatibility
                    # Keep raw response as-is (it already has <think>/<answer> or bare JSON)
                    pred = {"filepath": filepath, "caption": caption, "response": raw}

                fout.write(json.dumps(pred, ensure_ascii=False) + '\n')

    logger.info(f"Done: {success} success, {fail} fail, {total_boxes} total pred boxes")
    logger.info(f"Predictions saved to {pred_path}")

    # Save config
    config = {
        "model": args.model,
        "project": args.project,
        "location": args.location,
        "temperature": args.temperature,
        "num_workers": args.num_workers,
        "prompt": "thinkpe_imp (description + importance)",
        "coord_format": "Gemini [y0,x0,y1,x1] → converted to Qwen [x0,y0,x1,y1] in parse",
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
