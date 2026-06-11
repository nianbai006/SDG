#!/usr/bin/env python3
"""
GPT-Image-1.5 editor — full version

SDG mode:  source image + bbox annotation image + caption + SDG textfeedback
IMDOC mode: source image + artifact heatmap + misalignment heatmap + caption + IMDOC textfeedback

feature:
- Load each predictions file directly (no three-way intersection required)
- --resume: auto-skip already processed samples
- ThreadPoolExecutor concurrency + token-bucket rate limiting
- Write each finished record to results.jsonl immediately (resume-friendly)

API: /images/generations, rate-limited to N calls/min
"""
import argparse
import base64
import json
import os
import sys
import time
import threading
import traceback
from io import BytesIO
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import requests
from PIL import Image

PARENT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PARENT_ROOT not in sys.path:
    sys.path.insert(0, PARENT_ROOT)

from spatial_guide import parse_bboxes_from_response
from visualization import draw_bboxes


# ==================== config ====================

API_URL = os.environ.get("IMAGE_EDIT_API_URL", "")
DEFAULT_MODEL = os.environ.get("IMAGE_EDIT_MODEL", "")


# ==================== Prompt ====================

NO_TEXT_RULE = """

CRITICAL RULE: Do NOT add any text, words, letters, captions, titles, watermarks, or written content onto the image. Even if the evaluation feedback suggests adding text or labels, ignore that instruction completely. The output must be a pure visual image with no text overlays."""

FIXED_PROMPT_TEMPLATE = """Based on the reference image provided, generate an improved version    of the image.

The image was AI-generated from this description: {caption}

Please improve the image quality, fix any visual artifacts, distortions, and ensure the image accurately matches the description: {caption}"""

SDG_PROMPT_TEMPLATE = """Based on the two reference images provided, generate an improved version    of the image.

Image 1: The original AI-generated image created from this description: {caption}
Image 2: The same image with red bounding boxes highlighting defect regions detected by a quality evaluation model.

Evaluation feedback:
{feedback}
{extra_rules}
Please fix the issues in the red-boxed regions while preserving the overall composition, style, and content    of the original image. The output should accurately match the description: {caption}"""

IMDOC_PROMPT_TEMPLATE = """Based on the three reference images provided, generate an improved version    of the image.

Image 1: The original AI-generated image created from this description: {caption}
Image 2: Artifact heatmap — brighter regions indicate visual artifacts (distortions, malformed shapes, texture issues)
Image 3: Misalignment heatmap — brighter regions indicate areas where the image does not match the caption

Evaluation feedback:
{feedback}
{extra_rules}
Please fix the highlighted regions while preserving the overall composition, style, and content    of the original image. The output should accurately match the description: {caption}"""


# ==================== Utility functions ====================

def _image_to_b64(pil_image: Image.Image) -> str:
    buf = BytesIO()
    pil_image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _scale_bboxes(bboxes: List[Dict], img_w: int, img_h: int) -> List[Dict]:
    scaled = []
    for bbox in bboxes:
        box = bbox.get("box_2d", [])
        if len(box) != 4:
            continue
        x0, y0, x1, y1 = box
        scaled.append({
            **bbox,
            "box_2d": [
                int(x0 * img_w / 1000),
                int(y0 * img_h / 1000),
                int(x1 * img_w / 1000),
                int(y1 * img_h / 1000),
            ],
        })
    return scaled


def _extract_feedback_text(response: str) -> str:
    """Extract the full textual feedback (think + answer) and strip the XML tags."""
    import re
    # strip <think>/<answer>-style tags but keep the inner text
    text = re.sub(r'</?(?:think|answer)>', '', response).strip()
    return text if text else response[:2000]


def _load_file_as_b64(path: str) -> Optional[str]:
    if not path or not os.path.exists(path):
        return None
    try:
        img = Image.open(path).convert("RGB")
        return _image_to_b64(img)
    except Exception:
        return None


# ==================== dataload ====================

def load_sdg_samples(predictions_path: str) -> List[Dict]:
    """Load every SDG prediction directly."""
    samples = []
    with open(predictions_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            fp = raw.get("filepath", "")
            if fp and os.path.exists(fp):
                samples.append({
                    "filepath": fp,
                    "caption": raw.get("caption", ""),
                    "response": raw.get("response", ""),
                })
    return samples


def load_imdoc_samples(predictions_path: str) -> List[Dict]:
    """Load every ImageDoctor prediction directly."""
    samples = []
    with open(predictions_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            fp = raw.get("filename", "") or raw.get("filepath", "")
            if fp and os.path.exists(fp):
                samples.append({
                    "filepath": fp,
                    "caption": raw.get("caption", ""),
                    "prediction": raw.get("prediction", ""),
                    "artifact_heatmap": raw.get("artifact_heatmap", ""),
                    "misalignment_heatmap": raw.get("misalignment_heatmap", ""),
                })
    return samples


def load_fixed_samples(test_data_path: str) -> List[Dict]:
    """Load every sampleof the test set (fixed-prompt mode)."""
    samples = []
    with open(test_data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            fp = raw.get("filepath", "")
            if fp and os.path.exists(fp):
                samples.append({
                    "filepath": fp,
                    "caption": raw.get("caption", ""),
                })
    return samples


def load_done_filepaths(output_dir: str) -> set:
    """Collect already-processed filepaths from existing results.jsonl and failed_samples.jsonl."""
    done = set()
    for fname in ["results.jsonl", "failed_samples.jsonl"]:
        path = os.path.join(output_dir, fname)
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    fp = rec.get("filepath", "")
                    if fp:
                        done.add(fp)
                except json.JSONDecodeError:
                    pass
    return done


# ==================== API calls ====================

def call_gpt_image(
    prompt: str,
    images_b64: List[str],
    api_key: str,
    model: str = DEFAULT_MODEL,
    size: str = "1024x1024",
    max_retries: int = 3,
    retry_sleep: float = 10.0,
) -> Optional[bytes]:
    """Call the GPT-Image-1.5 API and return the generated image bytes."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "prompt": prompt,
        "image": [f"data:image/png;base64,{b64}" for b64 in images_b64],
        "size": size,
        "n": 1,
        "quality": "high",
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=180)

            if resp.status_code == 429:
                wait = retry_sleep * (attempt + 1)
                time.sleep(wait)
                continue

            if resp.status_code != 200:
                # unrecoverable errors (e.g. safety filter) — do not retry
                if resp.status_code == 400:
                    return None
                time.sleep(retry_sleep)
                continue

            data = resp.json()
            content = data.get("data", {}).get("content", [])
            if not content:
                content = data.get("data", [])

            for item in content:
                if isinstance(item, dict):
                    b64_json = item.get("b64_json", "")
                    if b64_json:
                        return base64.b64decode(b64_json)
                    url = item.get("url", "")
                    if url:
                        img_resp = requests.get(url, timeout=60)
                        if img_resp.state_code == 200:
                            return img_resp.content

            time.sleep(retry_sleep)

        except Exception as e:
            time.sleep(retry_sleep)

    return None


# ==================== Rate Limiter ====================

class RateLimiter:
    """Token-bucket rate limiter; thread-safe."""
    def __init__(self, rpm: int):
        self.interval = 60.0 / rpm
        self.lock = threading.Lock()
        self.last_time = 0.0

    def acquire(self):
        with self.lock:
            now = time.time()
            wait = self.last_time + self.interval - now
            if wait > 0:
                time.sleep(wait)
            self.last_time = time.time()


def _unique_name(filepath: str) -> str:
    """Generate a unique filename from the path, e.g. zimage_test_test_000197."""
    parts = Path(filepath).parts
    if len(parts) >= 3:
        return f"{parts[-3]}_{parts[-2]}_{Path(filepath).stem}"
    elif len(parts) >= 2:
        return f"{parts[-2]}_{Path(filepath).stem}"
    return Path(filepath).stem


# ==================== Sample processing ====================

def process_fixed(sample: Dict, api_key: str, out_dir: Path, rate_limiter: RateLimiter, no_text: bool = False, model: str = DEFAULT_MODEL) -> Dict:
    fp = sample["filepath"]
    caption = sample["caption"]
    basename = _unique_name(fp)

    try:
        img = Image.open(fp).convert("RGB")
        orig_b64 = _image_to_b64(img)

        prompt = FIXED_PROMPT_TEMPLATE.format(caption=caption)

        rate_limiter.acquire()
        img_bytes = call_gpt_image(prompt=prompt, images_b64=[orig_b64], api_key=api_key, model=model)

        if img_bytes:
            save_path = str(out_dir / "edited_images" / f"{basename}.png")
            with open(save_path, "wb") as f:
                f.write(img_bytes)
            return {
                "ok": True,
                "record": {
                    "filepath": fp,
                    "caption": caption,
                    "edited_path": save_path,
                    "edit_prompt": prompt,
                },
            }
        return {"ok": False, "record": {"filepath": fp, "reason": "api_failed"}}

    except Exception as e:
        return {"ok": False, "record": {"filepath": fp, "reason": str(e)}}

def process_sdg(sample: Dict, api_key: str, out_dir: Path, rate_limiter: RateLimiter, no_text: bool = False, model: str = DEFAULT_MODEL) -> Dict:
    fp = sample["filepath"]
    caption = sample["caption"]
    response = sample["response"]
    basename = _unique_name(fp)

    try:
        img = Image.open(fp).convert("RGB")
        img_w, img_h = img.size
        orig_b64 = _image_to_b64(img)

        bboxes = parse_bboxes_from_response(response)
        if bboxes:
            scaled = _scale_bboxes(bboxes, img_w, img_h)
            bbox_img = draw_bboxes(img, scaled, line_width=3, color=(255, 0, 0))
        else:
            bbox_img = img.copy()
        bbox_b64 = _image_to_b64(bbox_img)

        feedback = _extract_feedback_text(response)
        extra_rules = NO_TEXT_RULE if no_text else ""
        prompt = SDG_PROMPT_TEMPLATE.format(caption=caption, feedback=feedback, extra_rules=extra_rules)

        rate_limiter.acquire()
        img_bytes = call_gpt_image(prompt=prompt, images_b64=[orig_b64, bbox_b64], api_key=api_key, model=model)

        if img_bytes:
            save_path = str(out_dir / "edited_images" / f"{basename}.png")
            with open(save_path, "wb") as f:
                f.write(img_bytes)
            return {
                "ok": True,
                "record": {
                    "filepath": fp,
                    "caption": caption,
                    "edited_path": save_path,
                    "edit_prompt": prompt,
                    "num_bboxes": len(bboxes),
                },
            }
        return {"ok": False, "record": {"filepath": fp, "reason": "api_failed"}}

    except Exception as e:
        return {"ok": False, "record": {"filepath": fp, "reason": str(e)}}


def process_imdoc(sample: Dict, api_key: str, out_dir: Path, rate_limiter: RateLimiter, no_text: bool = False, model: str = DEFAULT_MODEL) -> Dict:
    fp = sample["filepath"]
    caption = sample["caption"]
    prediction = sample["prediction"]
    basename = _unique_name(fp)

    try:
        img = Image.open(fp).convert("RGB")
        orig_b64 = _image_to_b64(img)

        images = [orig_b64]
        art_b64 = _load_file_as_b64(sample.get("artifact_heatmap", ""))
        mis_b64 = _load_file_as_b64(sample.get("misalignment_heatmap", ""))
        if art_b64:
            images.append(art_b64)
        if mis_b64:
            images.append(mis_b64)

        feedback = _extract_feedback_text(prediction)
        extra_rules = NO_TEXT_RULE if no_text else ""
        prompt = IMDOC_PROMPT_TEMPLATE.format(caption=caption, feedback=feedback, extra_rules=extra_rules)

        rate_limiter.acquire()
        img_bytes = call_gpt_image(prompt=prompt, images_b64=images, api_key=api_key, model=model)

        if img_bytes:
            save_path = str(out_dir / "edited_images" / f"{basename}.png")
            with open(save_path, "wb") as f:
                f.write(img_bytes)
            return {
                "ok": True,
                "record": {
                    "filepath": fp,
                    "caption": caption,
                    "edited_path": save_path,
                    "edit_prompt": prompt,
                    "has_art_hm": bool(art_b64),
                    "has_mis_hm": bool(mis_b64),
                },
            }
        return {"ok": False, "record": {"filepath": fp, "reason": "api_failed"}}

    except Exception as e:
        return {"ok": False, "record": {"filepath": fp, "reason": str(e)}}


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(description="GPT-Image-1.5 batch editor")
    parser.add_argument("--mode", choices=["sdg", "imdoc", "fixed"], required=True)
    parser.add_argument("--sdg_predictions", default="outputs/mode4_sdg/predictions.jsonl")
    parser.add_argument("--imagedoctor_predictions", default="outputs/mode3_imagedoctor/predictions.jsonl")
    parser.add_argument("--test_data", default="${SDG_DATA}/sdg30k/annotations/merged_all_filtered_distilled_filepath_test.jsonl",
                        help="test-set path for fixed mode")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--rpm", type=int, default=10, help="Requests per minute limit")
    parser.add_argument("--concurrency", type=int, default=5, help="concurrent thread count")
    parser.add_argument("--resume", action="store_true", help="skip already-processed samples")
    parser.add_argument("--no_text", action="store_true", help="add a rule that forbids overlaying text on the image")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="API model endpoint ID")
    args = parser.parse_args()

    if args.output_dir is None:
        suffix = f"_{args.mode}_full"
        if args.no_text:
            suffix += "_notext"
        args.output_dir = f"outputs/v2_gpt_image{suffix}"

    if not API_URL:
        print("Error: please set IMAGE_EDIT_API_URL")
        sys.exit(1)
    if not args.model:
        print("Error: please set IMAGE_EDIT_MODEL or pass --model")
        sys.exit(1)

    api_key = args.api_key or os.environ.get("IMAGE_EDIT_API_KEY", "")
    if not api_key:
        print("Error: please set IMAGE_EDIT_API_KEY or pass --api_key")
        sys.exit(1)

    # loadsample
    if args.mode == "sdg":
        samples = load_sdg_samples(args.sdg_predictions)
    elif args.mode == "imdoc":
        samples = load_imdoc_samples(args.imagedoctor_predictions)
    else:  # fixed
        samples = load_fixed_samples(args.test_data)

    if args.max_samples:
        samples = samples[:args.max_samples]

    # Resume: skip already-processed samples
    out_dir = Path(args.output_dir)
    (out_dir / "edited_images").mkdir(parents=True, exist_ok=True)

    if args.resume:
        done_fps = load_done_filepaths(args.output_dir)
        before = len(samples)
        samples = [s for s in samples if s["filepath"] not in done_fps]
        print(f"[resume] already processed {before - len(samples)} records, {len(samples)} remaining")

    print(f"[{args.mode}] samples: {len(samples)}")
    print(f"  output: {args.output_dir}")
    print(f"  RPM: {args.rpm}, concurrent: {args.concurrency}")

    if not samples:
        print("no samples to process!")
        return

    rate_limiter = RateLimiter(args.rpm)
    results_path = out_dir / "results.jsonl"
    failed_path = out_dir / "failed_samples.jsonl"
    write_lock = threading.Lock()

    success_count = [0]
    fail_count = [0]
    st_time = time.time()

    process_fn = {"sdg": process_sdg, "imdoc": process_imdoc, "fixed": process_fixed}[args.mode]

    # append-mode write (resume-friendly)
    with open(results_path, "a", encoding="utf-8") as f_ok, \
         open(failed_path, "a", encoding="utf-8") as f_fail:

        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            future_map = {}
            for sample in samples:
                fut = executor.submit(process_fn, sample, api_key, out_dir, rate_limiter, args.no_text, args.model)
                future_map[fut] = sample["filepath"]

            for future in as_completed(future_map):
                fp = future_map[future]
                basename = Path(fp).stem
                try:
                    result = future.result()
                except Exception as e:
                    result = {"ok": False, "record": {"filepath": fp, "reason": str(e)}}

                with write_lock:
                    if result["ok"]:
                        success_count[0] += 1
                        f_ok.write(json.dumps(result["record"], ensure_ascii=False) + "\n")
                        f_ok.flush()
                        total = success_count[0] + fail_count[0]
                        if total % 10 == 0:
                            print(f"[{total}/{len(samples)}] succeeded {success_count[0]}, failed {fail_count[0]}")
                    else:
                        fail_count[0] += 1
                        f_fail.write(json.dumps(result["record"], ensure_ascii=False) + "\n")
                        f_fail.flush()

    elapsed = time.time() - st_time
    print(f"\ndone! succeeded {success_count[0]}, failed {fail_count[0]}")
    print(f"elapsed: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"result: {results_path}")


if __name__ == "__main__":
    main()
