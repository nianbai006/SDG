#!/usr/bin/env python3
"""
One-shot: emit three outputs from a distilled JSONL:
  1. *_filepath_test.jsonl                  -- test split (raw format)
  2. *_filepath_train_swift_sft_think.jsonl  -- train split, Swift SFT format
  3. *_filepath_train_swift_grpo_think.jsonl -- train split, Swift GRPO format

Pipeline:
  1. split by "test"/"train" in filepath
  2. train -> SFT format (includes the assistant reply)
  3. train -> GRPO format (user prompt + GT boxes only)
  4. test -> emit the raw JSON directly

Usage:
  python tools/prepare_all_datasets.py --input path/to/input.jsonl
"""
import argparse
import ast
import json
import os
import sys

# ---- question template (inlined to avoid constants.py import conflicts) ----
# NOTE: thinkpe variant (matches constants.py SFT_pos_en_TEMPLATE / "thinkpe", 3-step)
SFT_pos_en_TEMPLATE = """You are an AI image quality evaluator. You will be given **one image** to analyze.

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

**Goal**: Produce a detailed analysis    of the image quality and output bounding boxes for all detected issues.

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
- You MAY merge similar issues into a single bullet (e.g., "hand deformities", "text rendering errors", "identity mismatch", "composition mismatch").
- Each bullet MUST include:
  (a) the issue category (artifact or misalignment, or both if needed)
  (b) what is affected (object/part, e.g., hands, face, text, background element)
  (c) concrete visual evidence (what specifically looks wrong/missing/mismatched)
- Do NOT mention numeric coordinates.

### Step 3: Localization (Box-by-Box Grounding)
- Provide a detailed, precise localization statement for EACH defect instance (one line per box).
- Do NOT mention numeric coordinates.

- Each localization line MUST include all    of the following information in natural language:
  1) Anchor: the exact object/part involved (e.g., fingertip, ring band, sign text area, ear cartilage, hairline).
  2) Position: image-based cues (image-left/right, upper/lower, center, near the border, foreground/background).
     - Avoid subject-centric left/right (e.g., "her left"). If you need left/right, use "image-left"/"image-right".
  3) Interaction cue (when applicable): holding/touching/overlapping/merging/at the seam/near the mouth/etc.
  4) Scale description: explicitly state whether the region is a tiny localized detail, a part-sized area, a large area, extends along an edge, or affects the whole image.
  5) Shape/orientation description: explicitly state whether it is compact, elongated, runs along a boundary/edge, wraps around an object, sits on an interface between two objects/materials, or crosses from one region into another.

- Do NOT invent new defects; each line must correspond to exactly one defect instance.
</think>

### Answer Format (for <answer>)
Return a JSON list:
[
    {{"box_2d": [x0, y0, x1, y1], "label": "misalignment"|"artifact", "description": "brief description    of the issue"}}
]

Bounding box coordinates are in normalized 0-1000 space: [x0, y0, x1, y1].
If there are no issues, output an empty list.

### DESC Style Guide
For EACH box, write `description` with richer detail:
- 18 to 45 words per description.
- Must mention the affected object/part.
- Must include at least one concrete evidence phrase (e.g., fused contours, missing separations, distorted geometry, incorrect lettering, absent requested object, incorrect identity traits).
- Do NOT use vague words like "weird/strange/bad".
- Do NOT include numeric coordinates in description.

Now analyze the image and produce your output:
"""

# 4-step variant (matches constants.py SFT_THINK4step_TEMPLATE / "think4step")
SFT_THINK4step_TEMPLATE = """You are an AI image quality evaluator. You will be given **one image** to analyze.

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

**Goal**: Produce a detailed analysis    of the image quality and output bounding boxes for all detected issues.

### Strict Output Rules
Output **TWO blocks in this exact order**:
1) `<think>` - Your detailed analysis
2) `<answer>` - JSON list    of bounding boxes

### Think Format (STRICT, 4 STEPS)
<think>
### Step 1: Caption Understanding
- Briefly summarize what the caption requires, including subject, key attributes, actions, setting, style, and major composition cues when relevant.

### Step 2: Whole-Image Quality Reading
- Analyze the image globally before focusing on individual defects.
- Cover several dimensions when applicable: semantic alignment, aesthetics, composition, realism/plausibility, detail consistency, lighting/material consistency, text rendering quality, and any subtle suspicious cues.
- This step MUST still be reasonably detailed even if the final answer is empty.
- You may mention possible mild issues, borderline concerns, or image strengths that are not strong enough to enter the final JSON.

### Step 3: Visual Analysis & Defect Spotting (Issue Summary)
- Describe the strongest visible issues you observe.
- You MAY merge similar issues into a single bullet.
- Step 3 may mention potential mild issues beyond the final JSON.
- If the final JSON is empty, still describe visible strengths and any weak or borderline concerns in detail instead    of giving a minimal answer.
- Do NOT mention numeric coordinates.

### Step 4: Localization (Box-by-Box Grounding)
- Provide one line for EACH final defect instance, and only for those final defect instances.
- Do NOT mention numeric coordinates.
- Each localization line should include:
  1) the affected object, part, or region
  2) image-relative position cues
  3) local visual evidence
  4) approximate scale or extent
  5) shape/orientation or boundary relationship when useful
- If there are no final defect instances, explicitly say that no annotated defect instances are grounded into the final answer.
</think>

### Answer Format (for <answer>)
Return a JSON list:
[
    {{"box_2d": [x0, y0, x1, y1], "label": "misalignment"|"artifact", "description": "brief description    of the issue"}}
]

Bounding box coordinates are in normalized 0-1000 space: [x0, y0, x1, y1].
If there are no issues, output an empty list.

### DESC Style Guide
For EACH box, write `description` in English with grounded but moderately concise detail:
- Prefer 12 to 30 words per description.
- Must mention the affected object, part, or region.
- Must include at least one concrete evidence phrase such as fused boundary, missing part, distorted geometry, wrong identity traits, missing requested object, incorrect attribute, unreadable text, implausible texture, or structural inconsistency.
- Avoid overly verbose scene-level narration.
- Do NOT use vague words like weird, strange, or bad.
- Do NOT include numeric coordinates in description.

### SPECIAL EMPTY-ANSWER EXPECTATION
If the final answer is empty:
- The <think> block should still be substantive.
- In Step 2 and Step 3, discuss overall image quality, strengths, and any subtle or borderline concerns you can infer from the image.
- The final <answer> must still be [].

Now analyze the image and produce your output:
"""

question_template_registry = {
    "thinkpe": SFT_pos_en_TEMPLATE,
    "think4step": SFT_THINK4step_TEMPLATE,
}

# ===================== helpers =====================

def normalize_box(box_item):
    """Extract [x0, y0, x1, y1] and cast to int"""
    box = None
    if isinstance(box_item, dict) and "box_2d" in box_item:
        box = box_item["box_2d"]
    elif isinstance(box_item, list):
        box = box_item
    if box and isinstance(box, list) and len(box) >= 4:
        return [int(v) for v in box[:4]]
    return None


def extract_answer_list(raw_response):
    if not raw_response or "<answer>" not in raw_response:
        return None
    start = raw_response.find("<answer>") + len("<answer>")
    end = raw_response.find("</answer>")
    if end == -1:
        return None
    content = raw_response[start:end].strip()
    if content.startswith("```json"):
        content = content[7:]
    elif content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(content)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
    return None


def validate_xyxy_boxes(answer_list):
    if answer_list is None or not isinstance(answer_list, list):
        return False
    for item in answer_list:
        if not isinstance(item, dict):
            return False
        label = (item.get("label") or "").lower()
        if label not in {"misalignment", "artifact"}:
            return False
        box = item.get("box_2d")
        if not isinstance(box, list) or len(box) != 4:
            return False
        if not all(isinstance(v, int) for v in box):
            return False
        x0, y0, x1, y1 = box
        if any(v < 0 or v > 1000 for v in (x0, y0, x1, y1)):
            return False
        if x0 >= x1 or y0 >= y1:
            return False
    return True


# ===================== format converters =====================

def make_sft_item(entry, question_template):
    """Convert to Swift SFT format (includes the assistant reply)"""
    image_path = entry.get("filepath")
    caption = entry.get("caption", "")
    raw_response = entry.get("ann_response") or entry.get("response", "")

    if not image_path or not raw_response:
        return None, "missing_field"
    if "<think>" not in raw_response or "</think>" not in raw_response:
        return None, "missing_think_block"

    answer_list = extract_answer_list(raw_response)
    if not validate_xyxy_boxes(answer_list):
        return None, "invalid_bbox"

    user_text = question_template.format(caption=caption)
    messages = [
        {"role": "user", "content": f"<image>{user_text}"},
        {"role": "assistant", "content": raw_response},
    ]
    return {"messages": messages, "images": [image_path]}, None


def _normalize_bbox_item(box_item):
    """Normalize a bbox entry into {"box_2d": [x0,y0,x1,y1], "label": ..., "description": ...}。

    supported input formats:
      - dict: {"box_2d": [...], "label": "...", "description": "..."}
      - list: [x0, y0, x1, y1]  (no label / description)
    """
    if isinstance(box_item, dict) and "box_2d" in box_item:
        box = box_item["box_2d"]
        if not (isinstance(box, list) and len(box) >= 4):
            return None
        result = {
            "box_2d": [int(v) for v in box[:4]],
            "label": box_item.get("label", ""),
            "description": box_item.get("description", "") or box_item.get("desc", ""),
        }
        if "importance" in box_item:
            result["importance"] = box_item["importance"]
        return result
    elif isinstance(box_item, list) and len(box_item) >= 4:
        return {
            "box_2d": [int(v) for v in box_item[:4]],
            "label": "",
            "description": "",
        }
    return None


def make_grpo_item(entry, question_template):
    """Convert to Swift GRPO format (user prompt + GT boxes only; keeps description/label)"""
    image_path = entry.get("filepath")
    caption = entry.get("caption", "")

    if not image_path:
        return None, "missing_image_path"

    user_text = question_template.format(caption=caption)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": user_text},
            ],
        }
    ]

    # Use ann_translated_bboxes (from distillation, has description + importance).
    # Do NOT fall back to misalignment_bboxes/artifact_bboxes (model-generated, not human annotation).
    ann_tb = entry.get("ann_translated_bboxes", []) or []
    gt_misalignment = []
    gt_artifact = []
    if ann_tb and isinstance(ann_tb[0], dict) and "label" in ann_tb[0]:
        for item in ann_tb:
            normalized = _normalize_bbox_item(item)
            if normalized:
                label = (normalized.get("label") or "").lower()
                if label == "misalignment":
                    gt_misalignment.append(normalized)
                elif label == "artifact":
                    gt_artifact.append(normalized)

    # validate coordinates
    for bbox_dict in gt_misalignment + gt_artifact:
        x0, y0, x1, y1 = bbox_dict["box_2d"]
        if any(v < 0 or v > 1000 for v in (x0, y0, x1, y1)) or x0 >= x1 or y0 >= y1:
            return None, "invalid_bbox"

    return {
        "messages": messages,
        "gt_misalignment_bboxes": gt_misalignment,
        "gt_artifact_bboxes": gt_artifact,
    }, None


# ===================== main =====================

def main():
    parser = argparse.ArgumentParser(
        description="One-shot generation    of test / SFT / GRPO splits"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Input JSONL path",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory (defaults to the input directory)",
    )
    parser.add_argument(
        "--question_template", type=str, default="thinkpe",
        help="key in question_template_registry (default: thinkpe)",
    )
    args = parser.parse_args()

    question_template = question_template_registry.get(args.question_template)
    if not question_template:
        raise ValueError(f"Unknown question_template: {args.question_template}")

    input_path = args.input
    output_dir = args.output_dir or os.path.dirname(input_path)
    base_name = os.path.splitext(os.path.basename(input_path))[0]

    test_path = os.path.join(output_dir, f"{base_name}_filepath_test.jsonl")
    sft_path = os.path.join(output_dir, f"{base_name}_filepath_train_swift_sft_think.jsonl")
    grpo_path = os.path.join(output_dir, f"{base_name}_filepath_train_swift_grpo_think.jsonl")

    os.makedirs(output_dir, exist_ok=True)

    print(f"Input:  {input_path}")
    print(f"Output:")
    print(f"  Test:  {test_path}")
    print(f"  SFT:   {sft_path}")
    print(f"  GRPO:  {grpo_path}")
    print(f"Template: {args.question_template}")
    print()

    # stats
    total = 0
    no_filepath = 0
    skipped_neither = 0
    test_count = 0
    train_total = 0
    sft_count = 0
    grpo_count = 0
    sft_skip_reasons = {}
    grpo_skip_reasons = {}

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(test_path, "w", encoding="utf-8") as f_test, \
         open(sft_path, "w", encoding="utf-8") as f_sft, \
         open(grpo_path, "w", encoding="utf-8") as f_grpo:

        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1

            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            filepath = entry.get("filepath")
            if not filepath:
                no_filepath += 1
                continue

            lower_path = filepath.lower()

            # ---- test: filepath contains "test" ----
            if "test" in lower_path:
                f_test.write(json.dumps(entry, ensure_ascii=False) + "\n")
                test_count += 1
                continue

            # ---- train: filepath contains "train" ----
            if "train" not in lower_path:
                skipped_neither += 1
                continue

            train_total += 1

            # SFT
            sft_item, sft_reason = make_sft_item(entry, question_template)
            if sft_item:
                f_sft.write(json.dumps(sft_item, ensure_ascii=False) + "\n")
                sft_count += 1
            elif sft_reason:
                sft_skip_reasons[sft_reason] = sft_skip_reasons.get(sft_reason, 0) + 1

            # GRPO
            grpo_item, grpo_reason = make_grpo_item(entry, question_template)
            if grpo_item:
                f_grpo.write(json.dumps(grpo_item, ensure_ascii=False) + "\n")
                grpo_count += 1
            elif grpo_reason:
                grpo_skip_reasons[grpo_reason] = grpo_skip_reasons.get(grpo_reason, 0) + 1

    # ---- summary ----
    print("=" * 50)
    print("Done.")
    print("=" * 50)
    print(f"Total input:          {total}")
    if no_filepath:
        print(f"Missing filepath:     {no_filepath}")
    if skipped_neither:
        print(f"Neither train nor test:   {skipped_neither}")
    print(f"Test split:          {test_count}")
    print(f"Train split:          {train_total}")
    print(f"SFT output:        {sft_count}")
    print(f"GRPO output:       {grpo_count}")

    if sft_skip_reasons:
        print(f"\nSFT skipped reasons:")
        for reason, count in sorted(sft_skip_reasons.items(), key=lambda x: -x[1]):
            print(f"  - {reason}: {count}")

    if grpo_skip_reasons:
        print(f"\nGRPO skipped reasons:")
        for reason, count in sorted(grpo_skip_reasons.items(), key=lambda x: -x[1]):
            print(f"  - {reason}: {count}")

    print(f"\nOutput files:")
    print(f"  1. {test_path}")
    print(f"  2. {sft_path}")
    print(f"  3. {grpo_path}")


if __name__ == "__main__":
    main()
