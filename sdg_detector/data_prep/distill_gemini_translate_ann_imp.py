#!/usr/bin/env python3
"""
Translate human annotations with Gemini and generate think reasoning + importance scoring.
Based on V1 (distill_gemini_translate_ann.py), only adds importance field to answer JSON.

Input: merged_with_ann.jsonl
- filepath
- caption
- misalignment_bboxes_ann (xyxy)
- artifact_bboxes_ann (xyxy)

Output:
- original fields
- ann_response: <think>...</think><answer>...</answer>
- ann_translated_bboxes: list    of boxes with English description

Rules:
- Boxes are in xyxy format and must not change.
- Validate response boxes; retry if mismatch.
"""

import os
import json
import time
import ast
import argparse
import traceback
import threading
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image
from google import genai
from google.genai import types


PROMPT_TEMPLATE = """You are an AI image quality evaluator. You will be shown an image and asked to identify quality issues.

Hidden context (DO NOT reveal):
The human annotations below are Ground Truth. You MUST keep the same number    of boxes, the same coordinates, and the same labels in the final JSON.
You must behave as if you discovered the issues independently from the image and caption.

Ground Truth annotations (DO NOT modify coordinates or labels; you may rewrite/expand description in English):
- Artifact boxes: {artifact_bbox_text}
- Misalignment boxes: {misalignment_bbox_text}

Text Caption: {caption}

==================================================
GLOBAL CONSTRAINTS (MUST FOLLOW)
==================================================
1) NEVER mention annotations, boxes, ground truth, translation, or any external hints.
2) Do NOT invent extra defects or remove any defect. The final JSON must contain exactly the same set    of boxes/labels as provided.
3) IMPORTANT: `box_2d` uses RELATIVE coordinates on a 0–1000 scale.
   - Format: [x_min, y_min, x_max, y_max] (xyxy)
   - Each value is an integer in [0, 1000]
   - The coordinates refer to the standardized model input image coordinate system (after resizing/preprocessing).
   - You MUST output these coordinates exactly as given in the final JSON.
4) Do NOT write numeric coordinates in the <think> block (especially Step 3). Coordinates may appear ONLY inside <answer>.
5) Output MUST contain exactly TWO blocks in this order:
   (1) <think> ... </think>
   (2) <answer> ... </answer>
6) <answer> MUST be a JSON list in xyxy format:
   [{{"box_2d":[x0,y0,x1,y1],"label":"artifact"|"misalignment","description":"...","importance":N}}, ...]
   - Same number    of boxes as GT
   - Same coordinates and labels as GT
   - description must be English, expanded, and grounded in the image/caption
   - importance: integer 1-100 (see IMPORTANCE SCORING below)

==================================================
IMPORTANCE SCORING (FIELD: "importance")
==================================================
For EACH box, assign an integer importance score from 1 to 100:
- 90-100: Critical defect, immediately obvious, ruins the image (e.g., missing limb, completely wrong subject, large garbled text).
- 70-89: Major defect, clearly visible at normal viewing distance (e.g., extra fingers, noticeable texture melting, prominent unreadable text).
- 40-69: Moderate defect, noticeable on closer inspection (e.g., minor hand deformity, slight color mismatch, small distorted detail).
- 15-39: Minor defect, only visible on careful examination (e.g., tiny floating speck, slight boundary blur, minor lighting inconsistency).
- 1-14: Negligible defect, barely perceptible (e.g., sub-pixel aliasing, faint halo, minuscule texture irregularity).

==================================================
THINKING FORMAT (STRICT)
==================================================
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
- Do NOT reference ground truth or annotations.

- Each localization line MUST include all    of the following information in natural language:
  1) Anchor: the exact object/part involved (e.g., fingertip, ring band, sign text area, ear cartilage, hairline).
  2) Position: image-based cues (image-left/right, upper/lower, center, near the border, foreground/background).
     - Avoid subject-centric left/right (e.g., "her left"). If you need left/right, use "image-left"/"image-right".
  3) Interaction cue (when applicable): holding/touching/overlapping/merging/at the seam/near the mouth/etc.
  4) Scale description: explicitly state whether the region is a tiny localized detail, a part-sized area, a large area, extends along an edge, or affects the whole image.
  5) Shape/orientation description: explicitly state whether it is compact, elongated, runs along a boundary/edge, wraps around an object, sits on an interface between two objects/materials, or crosses from one region into another.

- Do NOT invent new defects; each line must correspond to exactly one defect instance.

==================================================
DESCRIPTION STYLE GUIDE (FOR <answer>)
==================================================
For EACH box in the final JSON, rewrite `description` into English with richer detail:
- 18 to 45 words per description.
- Must mention the affected object/part.
- Must include at least one concrete evidence phrase (e.g., fused contours, missing separations, distorted geometry, incorrect lettering, absent requested object, incorrect identity traits).
- Do NOT use vague words like "weird/strange/bad".
- Do NOT mention translation or ground truth.
- Do NOT include numeric coordinates in description.

==================================================
FEW-SHOT EXAMPLES (FOLLOW THE STYLE)
==================================================

[EXAMPLE A: NON-EMPTY GT]

Caption:
a rock musician sticking tongue out holding a sign that says hail Satan

Ground Truth annotations (hidden from the model's narrative; do not mention them):
- Artifact boxes:
  {{"box_2d":[777,604,796,627],"label":"artifact","description":"Ring and finger are fused together"}}
  {{"box_2d":[759,640,778,665],"label":"artifact","description":"Ring edge merges with the skin"}}
  {{"box_2d":[760,671,776,686],"label":"artifact","description":"Abnormally bumpy skin texture on the finger"}}
  {{"box_2d":[212,641,242,713],"label":"artifact","description":"Ring and finger appear melted together"}}
- Misalignment boxes:
  {{"box_2d":[417,144,563,370],"label":"misalignment","description":"The person does not resemble Jim Morrison"}}
  {{"box_2d":[267,493,734,813],"label":"misalignment","description":"The sign text is not "hail Satan""}}

Desired output:

<think>
### Step 1: Caption Understanding
- The caption asks for a rock musician sticking their tongue out while holding a sign that should clearly read “hail Satan”.

### Step 2: Visual Analysis & Defect Spotting (Issue Summary)
- Misalignment: The person does not convincingly match the specified celebrity identity; facial proportions and styling look like a generic rock musician rather than the named individual.
- Misalignment: The sign text is not rendered as the requested phrase; the lettering is distorted and the spelling is incorrect, reducing readability.
- Artifact: The hands and jewelry show generation artifacts; ring edges blend into skin and some local finger texture appears unnaturally bumpy, suggesting boundary fusion and inconsistent surface detail.

### Step 3: Localization (Box-by-Box Grounding)
- Box 1 (artifact): The defect is on the image-right hand gripping the sign near the fingertips, exactly at the ring–skin contact; it is a tiny localized region that sits on an interface and crosses the boundary between metal and finger texture.
- Box 2 (artifact): The defect is on the image-right hand around the mid-finger ring edge, along the ring’s outer contour where it touches skin; it is a tiny localized region that is compact and aligned along a boundary that should be sharp.
- Box 3 (artifact): The defect is just below the rings on the image-right fingers near the gripping area, on the skin surface; it is a tiny localized region that is compact and concentrated within a small patch rather than spreading across the whole hand.
- Box 4 (artifact): The defect is on the image-left hand holding the sign, at the ring-to-finger junction where the hand overlaps the sign edge; it is a tiny localized region that sits on an interface and appears smeared across the metal–flesh boundary.
- Box 5 (misalignment): The defect is centered on the upper-central face region that defines identity (eyes, nose, and overall facial structure); it is a part-sized region that is compact and concentrated on the head rather than a single small detail.
- Box 6 (misalignment): The defect is confined to the central sign surface where the red lettering appears, within the writing block; it is a part-sized region that is compact and concentrated in the text area rather than extending into the background.
</think>
<answer>
[
  {{“box_2d”:[777,604,796,627],”label”:”artifact”,”description”:”On the image-right hand near the fingertips, the ring appears fused into the finger, evidenced by a smeared metal-to-skin boundary at the contact interface where the outline should be clean and separated.”,”importance”:45}},
  {{“box_2d”:[759,640,778,665],”label”:”artifact”,”description”:”On the image-right hand around the mid-finger ring edge, the metal outline blends into skin, forming a compact localized fusion with ssoftened borders and continuous texture where a sharp ring boundary is expected.”,”importance”:40}},
  {{“box_2d”:[760,671,776,686],”label”:”artifact”,”description”:”Just below the rings on the image-right fingers, a small patch    of skin shows unnatural raised bumps and inconsistent shading, forming a tiny localized texture anomaly rather than coherent wrinkles or pores.”,”importance”:25}},
  {{“box_2d”:[212,641,242,713],”label”:”artifact”,”description”:”On the image-left hand holding the sign, the ring looks partially embedded into the finger, with distorted geometry and a boundary that appears to cross from glossy metal into flesh without a clear separation.”,”importance”:42}},
  {{“box_2d”:[417,144,563,370],”label”:”misalignment”,”description”:”The face in the upper-central area does not satisfy the caption’s identity requirement, since key facial proportions and styling do not resemble Jim Morrison and instead read as a generic rock musician.”,”importance”:82}},
  {{“box_2d”:[267,493,734,813],”label”:”misalignment”,”description”:”The sign’s central text block does not match the required phrase “hail Satan”, as the letters are misspelled and deformed, producing unreadable character shapes rather than a clear, correctly spelled sentence.”,”importance”:88}}
]
</answer>

[EXAMPLE B: EMPTY GT]

Caption:
a girl with pink curly hair wearing a witch hat

Ground Truth annotations:
- Artifact boxes: (none)
- Misalignment boxes: (none)

Desired output:

<think>
### Step 1: Caption Understanding
- The caption requires a girl with pink, curly hair wearing a witch hat.

### Step 2: Visual Analysis & Defect Spotting (Issue Summary)
- No clear artifacts or caption misalignments are visible; the hair color and curl texture are consistent, and the witch hat appears correctly placed and well-formed.

### Step 3: Localization (Box-by-Box Grounding)
- No defect instances are identified, so there are no localized regions to ground.
</think>
<answer>
[]
</answer>


==================================================
NOW DO THE TASK ON THE GIVEN IMAGE
==================================================
Now analyze the image and respond with exactly TWO blocks (<think> then <answer>).
"""


def image_to_bytes(pil_image: Image.Image) -> bytes:
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    return buffer.getvalue()


def format_bbox_text(bboxes):
    if not bboxes:
        return "None"
    formatted = []
    for item in bboxes:
        if isinstance(item, dict):
            box = item.get("box_2d")
            desc = item.get("description", "")
            label = item.get("label", "")
            if isinstance(box, list) and len(box) == 4:
                formatted.append({"box_2d": box, "label": label, "description": desc})
        elif isinstance(item, list) and len(item) == 4:
            formatted.append({"box_2d": item, "label": "", "description": ""})
    return json.dumps(formatted, ensure_ascii=False)


def extract_all_tag_blocks(text: str, tag: str):
    if not text:
        return []
    results = []
    start_tag = f"<{tag}>"
    end_tag = f"</{tag}>"
    pos = 0
    while True:
        s = text.find(start_tag, pos)
        if s == -1:
            break
        e = text.find(end_tag, s)
        if e == -1:
            break
        content = text[s + len(start_tag): e].strip()
        results.append(content)
        pos = e + len(end_tag)
    return results


def parse_answer_list(text: str):
    if not text:
        return None
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except Exception:
        pass

    try:
        result = ast.literal_eval(text)
        if isinstance(result, list):
            return result
    except Exception:
        pass

    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        snippet = text[start:end + 1]
        try:
            result = json.loads(snippet)
            if isinstance(result, list):
                return result
        except Exception:
            pass
        try:
            result = ast.literal_eval(snippet)
            if isinstance(result, list):
                return result
        except Exception:
            pass
    return None


def extract_boxes_from_list(bboxes):
    output = []
    for item in bboxes:
        if isinstance(item, dict) and "box_2d" in item:
            box = item.get("box_2d")
            if isinstance(box, list) and len(box) == 4:
                entry = {
                    "box_2d": box,
                    "label": item.get("label", ""),
                    "description": item.get("description", ""),
                }
                if "importance" in item:
                    entry["importance"] = item["importance"]
                output.append(entry)
        elif isinstance(item, list) and len(item) == 4:
            output.append({"box_2d": item, "label": "", "description": ""})
    return output


def normalize_bbox_list(bboxes):
    items = []
    for item in bboxes:
        if isinstance(item, dict) and "box_2d" in item:
            box = item.get("box_2d")
            label = item.get("label", "")
            if isinstance(box, list) and len(box) == 4:
                items.append((label, tuple(box)))
        elif isinstance(item, list) and len(item) == 4:
            items.append(("", tuple(item)))
    return sorted(items, key=lambda x: (x[0], x[1]))


def build_expected_bboxes(artifact_bboxes, misalignment_bboxes):
    expected = []
    for item in artifact_bboxes:
        if isinstance(item, dict):
            expected.append({"box_2d": item.get("box_2d", []), "label": "artifact"})
        elif isinstance(item, list) and len(item) == 4:
            expected.append({"box_2d": item, "label": "artifact"})
    for item in misalignment_bboxes:
        if isinstance(item, dict):
            expected.append({"box_2d": item.get("box_2d", []), "label": "misalignment"})
        elif isinstance(item, list) and len(item) == 4:
            expected.append({"box_2d": item, "label": "misalignment"})
    return expected


def parse_gemini_response(response_text: str):
    if response_text is None:
        return None, []
    thinks = extract_all_tag_blocks(response_text, "think")
    answers = extract_all_tag_blocks(response_text, "answer")
    think_content = thinks[0] if len(thinks) > 0 else None
    answer_content = answers[0] if len(answers) > 0 else ""
    parsed = parse_answer_list(answer_content)
    all_bboxes = extract_boxes_from_list(parsed or [])
    return think_content, all_bboxes


def response_is_valid(raw_text, expected_bboxes):
    if not raw_text:
        print("Validation failed: empty response")
        return False
    if "<think>" not in raw_text or "</think>" not in raw_text:
        print(raw_text)
        print("Validation failed: missing <think> block")
        return False
    think_content, parsed_bboxes = parse_gemini_response(raw_text)
    if not think_content:
        print("Validation failed: empty <think> content")
        return False
    if normalize_bbox_list(parsed_bboxes) != normalize_bbox_list(expected_bboxes):
        print("Validation failed: bbox mismatch")
        print(f"Expected: {expected_bboxes}")
        print(f"Parsed: {parsed_bboxes}")
        return False
    return True


def process_single_sample(args_tuple):
    index, line, config = args_tuple
    try:
        data = json.loads(line)
        image_path = data.get("filepath")
        caption = data.get("caption", "")
        artifact_bboxes_ann = data.get("artifact_bboxes_ann", [])
        misalignment_bboxes_ann = data.get("misalignment_bboxes_ann", [])

        if not image_path or not os.path.exists(image_path):
            if config["debug"]:
                print(f"Warning: Missing image for sample {index}: {image_path}")
            return {"_failed": True, "_index": index, "_reason": "missing_image"}

        expected_bboxes = build_expected_bboxes(artifact_bboxes_ann, misalignment_bboxes_ann)

        img = Image.open(image_path).convert("RGB")
        image_bytes = image_to_bytes(img)

        filled_prompt = PROMPT_TEMPLATE.format(
            
            artifact_bbox_text=format_bbox_text(artifact_bboxes_ann),
            misalignment_bbox_text=format_bbox_text(misalignment_bboxes_ann),
            caption=caption,
        )

        content_parts = [
            filled_prompt,
            types.Part.from_bytes(mime_type="image/png", data=image_bytes)
        ]

        client = genai.Client(
            vertexai=True,
            project=config["project"],
            location=config["location"],
        )

        last_err = None
        response = None
        raw_text = None
        for attempt in range(config["max_retries"]):
            try:
                response = client.models.generate_content(
                    model=config["model_name"],
                    contents=content_parts,
                    config=types.GenerateContentConfig(
                        temperature=1.0,
                        thinking_config=types.ThinkingConfig(thinking_level="high")
                    )
                )
                # Count total API calls
                api_counter = config.get("api_call_counter")
                api_lock = config.get("api_call_lock")
                if api_counter is not None and api_lock is not None:
                    with api_lock:
                        api_counter[0] += 1
                        if api_counter[0] % 50 == 0:
                            print(f"[API Stats] total calls: {api_counter[0]}")
                if response is None or not getattr(response, "text", None):
                    last_err = "empty_response"
                    print(f"Retry {attempt + 1}/{config['max_retries']} for sample {index}: empty_response")
                    time.sleep(config["retry_sleep"] * (attempt + 1))
                    continue

                raw_text = response.text
                if response_is_valid(raw_text, expected_bboxes):
                    last_err = None
                    break
                last_err = "validation_failed"
                print(f"Retry {attempt + 1}/{config['max_retries']} for sample {index}: validation_failed")
                time.sleep(config["retry_sleep"] * (attempt + 1))
            except Exception as e:
                last_err = e
                print(f"Retry {attempt + 1}/{config['max_retries']} for sample {index}: {e}")
                time.sleep(config["retry_sleep"] * (attempt + 1))

        if not raw_text:
            if config["debug"]:
                print(f"No result for sample {index}. last_err={last_err}")
            return {"_failed": True, "_index": index, "_reason": "empty_response"}

        think_content, all_bboxes = parse_gemini_response(raw_text)

        output_data = dict(data)
        output_data["ann_response"] = raw_text
        output_data["ann_translated_bboxes"] = all_bboxes

        return output_data

    except Exception as e:
        if config["debug"]:
            print(f"Error processing sample {index}: {e}")
            traceback.print_exc()
        return {"_failed": True, "_index": index, "_reason": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Translate human annotations with Gemini (xyxy, box-preserving).")
    parser.add_argument("--input_jsonl", type=str,
                        default="${SDG_DATA}/sdg30k/annotations/all_filtered_distilled_new_all_left.jsonl",
                        help="Path to input JSONL file")
    parser.add_argument("--output_jsonl", type=str,
                        default="${SDG_DATA}/sdg30k/annotations/all_filtered_distilled_new_all_left_distilled.jsonl",
                        help="Path to output JSONL file")
    parser.add_argument("--model", type=str, default="gemini-3-pro-preview",
                        help="Gemini model name")
    parser.add_argument("--project", type=str, default=os.environ.get("GEMINI_PROJECT", ""),
                        help="Google Cloud project ID")
    parser.add_argument("--location", type=str, default="global",
                        help="Google Cloud location")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode")
    parser.add_argument("--num_samples", type=int, default=20,
                        help="Number    of samples to process in debug mode")
    parser.add_argument("--start_index", type=int, default=0,
                        help="Start processing from this index (0-based)")
    parser.add_argument("--max_retries", type=int, default=100,
                        help="Maximum retries per sample")
    parser.add_argument("--retry_sleep", type=float, default=2.0,
                        help="Retry sleep seconds")
    parser.add_argument("--concurrency", type=int, default=200,
                        help="Number    of concurrent threads")
    args = parser.parse_args()

    print(f"Reading from {args.input_jsonl}...")
    with open(args.input_jsonl, "r") as f:
        lines = f.readlines()

    if args.start_index > 0:
        lines = lines[args.start_index:]
        print(f"Starting from index {args.start_index}, remaining {len(lines)} samples.")

    if args.debug:
        lines = lines[:args.num_samples]
        print(f"Debug mode: processing first {len(lines)} samples.")

    output_dir = os.path.dirname(args.output_jsonl)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    api_call_counter = [0]  # Wrap in a list to allow closure-side mutation
    api_call_lock = threading.Lock()

    config = {
        "model_name": args.model,
        "project": args.project,
        "location": args.location,
        "max_retries": args.max_retries,
        "retry_sleep": args.retry_sleep,
        "debug": args.debug,
        "api_call_counter": api_call_counter,
        "api_call_lock": api_call_lock,
    }

    print("Starting translation...")
    print(f"Model: {args.model}")
    print(f"Project: {args.project}")
    print(f"Concurrency: {args.concurrency}")

    buffered_results = []
    failed_items = []
    total_processed = 0
    st_time = time.time()
    save_every = 50

    tasks = [(i + args.start_index, line, config) for i, line in enumerate(lines)]

    with open(args.output_jsonl, "w") as f_out:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            future_to_idx = {executor.submit(process_single_sample, task): task[0] for task in tasks}
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                except Exception as e:
                    result = {"_failed": True, "_index": idx, "_reason": str(e)}

                if result is not None:
                    if result.get("_failed"):
                        failed_items.append(result)
                    else:
                        buffered_results.append(result)
                        total_processed += 1
                        if len(buffered_results) >= save_every:
                            for item in buffered_results:
                                f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
                            f_out.flush()
                            buffered_results = []
                            print(f"Already processed {total_processed} samples (autosave every {save_every})")

        if buffered_results:
            for item in buffered_results:
                f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
            f_out.flush()

    total_time = time.time() - st_time
    print(f"\nSaved {total_processed} results to {args.output_jsonl}")
    print(f"Failed: {len(failed_items)} samples")
    print(f"Total API calls: {api_call_counter[0]}")
    if failed_items:
        print("Failed sample indices:", [item["_index"] for item in failed_items[:20]], "..." if len(failed_items) > 20 else "")
    print(f"Total time: {total_time:.2f}s")
    if total_processed:
        print(f"Average time: {total_time / total_processed:.2f}s per sample")


if __name__ == "__main__":
    main()
