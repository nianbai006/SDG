"""
Phase 1 inference: SDG evaluation
- online: Call the SDG model for online evaluation
- offline: Use the existing ann_response field from the test set
- Parse the bbox info from the response
- draw bboxes onto the source image and save
- save inference_results.jsonl
"""
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

from data_loader import (
    load_test_data,
    load_sdg_predictions,
    match_sdg_to_test,
    filter_existing_images,
    make_unique_basename,
)
from vlm_client import VLMClient
from prompt_builder import build_prompt_from_sdg
from visualization import draw_bboxes


def _parse_bboxes_from_response(response: str) -> List[Dict]:
    """Parse the bbox list from the <answer> block    of a SDG response."""
    answer_match = re.search(r'<answer>\s*(.*?)\s*</answer>', response, re.DOTALL)
    if not answer_match:
        return []

    answer_text = answer_match.group(1).strip()
    try:
        start = answer_text.find('[')
        end = answer_text.rfind(']')
        if start == -1 or end == -1 or end <= start:
            return []
        data = json.loads(answer_text[start:end + 1])
        if isinstance(data, list):
            return data
    except (json.JSONDecodeError, ValueError):
        pass
    return []


def run_infer_mode4(
    test_data_path: str,
    output_dir: str,
    sdg_predictions_path: Optional[str] = None,
    sdg_server_url: str = "http://localhost:17142/v1",
    sdg_model_name: str = "sdg-detector",
    use_offline: bool = False,
    use_bbox: bool = False,
    max_samples: Optional[int] = None,
    filepath_prefix_map: Optional[Dict[str, str]] = None,
):
    """
    SDG inference stage: evaluate the image, parse bboxes, render the annotation image, save results.

    Supports three data sources:
        1. online mode: call SDG API for online inference
        2. offline mode (ann_response): use the existing ann_response field from the test set
        3. offline mode (predictions.jsonl): read predictions from the SDG predictions.jsonl

    output:
        {output_dir}/inference_results.jsonl
        {output_dir}/bbox_images/{basename}_bbox.png  (e.g.enable bbox)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    bbox_images_dir = output_path / "bbox_images"

    # loaddata
    data = load_test_data(test_data_path, max_samples=max_samples,
                          filepath_prefix_map=filepath_prefix_map)
    data = filter_existing_images(data, filepath_key="filepath")

    if use_offline and sdg_predictions_path:
        # offline mode: from  SDG predictions.jsonl loadprediction
        print("[infer_mode4] using offline SDG predictions (predictions.jsonl)...")
        sdg_data = load_sdg_predictions(
            sdg_predictions_path, filepath_prefix_map=filepath_prefix_map,
        )
        data = match_sdg_to_test(data, sdg_data)
        # parse response in   bboxes
        for sample in data:
            sample["sdg_bboxes"] = _parse_bboxes_from_response(
                sample.get("sdg_response", "")
            )
    elif use_offline:
        # offline mode: makeuse already has   of ann_response
        print("[infer_mode4] using offline SDG data (ann_response)...")
        for sample in data:
            sample["sdg_response"] = sample.get("ann_response", "")
            sample["sdg_bboxes"] = sample.get("ann_translated_bboxes", [])

        data = [s for s in data if s.get("sdg_response") and len(s["sdg_response"]) > 10]
        print(f"[infer_mode4] samples with ann_response: {len(data)}")
    else:
        # online mode
        print("[infer_mode4] using online SDG inference...")
        data = _run_sdg_online(data, sdg_server_url, sdg_model_name)

    if not data:
        print("[infer_mode4] no valid data!")
        return []

    print(f"[infer_mode4] validsample: {len(data)}")

    # process each record and save
    results = []
    results_path = output_path / "inference_results.jsonl"
    st_time = time.time()

    for i, sample in enumerate(data):
        filepath = sample["filepath"]
        caption = sample.get("caption", "")
        sdg_response = sample.get("sdg_response", "")
        sdg_bboxes = sample.get("sdg_bboxes", [])
        basename = make_unique_basename(filepath)

        if not sdg_response:
            continue

        if (i + 1) % 10 == 0:
            elapsed = time.time() - st_time
            avg = elapsed / (i + 1)
            eta = avg * (len(data) - i - 1)
            print(f"  progress: {i+1}/{len(data)}  elapsed {elapsed:.1f}s  mean {avg:.2f}s/sample  ETA {eta:.0f}s")

        # render the bbox annotation image and save
        bbox_image_path = ""
        if use_bbox and sdg_bboxes:
            bbox_images_dir.mkdir(parents=True, exist_ok=True)
            bbox_vis = draw_bboxes(filepath, sdg_bboxes)
            bbox_image_path = str(bbox_images_dir / f"{basename}_bbox.png")
            bbox_vis.save(bbox_image_path)

        # Build the edit prompt
        edit_prompt = build_prompt_from_sdg(
            caption=caption,
            sdg_response=sdg_response,
        )

        record = {
            "filepath": filepath,
            "caption": caption,
            "sdg_response": sdg_response,
            "bboxes": sdg_bboxes,
            "bbox_image": bbox_image_path,
            "edit_prompt": edit_prompt,
        }
        results.append(record)

        with open(results_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    total_time = time.time() - st_time
    print(f"[infer_mode4] done! saved {len(results)} results to {results_path}")
    print(f"  elapsed: {total_time:.1f}s  mean: {total_time/max(len(results),1):.2f}s/sample")
    return results


def _run_sdg_online(
    data: List[Dict],
    sdg_server_url: str,
    sdg_model_name: str,
) -> List[Dict]:
    """Online call to the SDG model to evaluate the image."""
    from config import SDG_SYSTEM_PROMPT, SDG_QUESTION_TEMPLATE

    vlm = VLMClient(
        base_url=sdg_server_url,
        model_name=sdg_model_name,
        max_new_tokens=2048,
        timeout=120,
    )

    if not vlm.check_health():
        print("[infer_mode4] error: SDG serviceunavailable!")
        print("  please deploy first: bash deploy_sdg.sh")
        return data

    print(f"[infer_mode4] SDG online inference ({sdg_model_name})...")
    st_time = time.time()

    for i, sample in enumerate(data):
        filepath = sample["filepath"]
        caption = sample.get("caption", "")

        if (i + 1) % 10 == 0:
            elapsed = time.time() - st_time
            avg = elapsed / (i + 1)
            eta = avg * (len(data) - i - 1)
            print(f"  SDG inference progress: {i+1}/{len(data)}  already use {elapsed:.1f}s  mean{avg:.2f}s/sample  ETA {eta:.0f}s")

        result = vlm.evaluate_image(
            image_path=filepath,
            caption=caption,
            system_prompt=SDG_SYSTEM_PROMPT,
            user_prompt_template=SDG_QUESTION_TEMPLATE,
        )

        if result["success"]:
            response = result["text"]
            sample["sdg_response"] = response
            sample["sdg_bboxes"] = _parse_bboxes_from_response(response)
        else:
            sample["sdg_response"] = ""
            sample["sdg_bboxes"] = []
            print(f"  SDG inference failed ({i}): {result['error'][:100]}")

    return data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase 1 — run SDG inference on a directory of images "
                    "and save the structured predictions used by GPT-Image-1.5 "
                    "as feedback (paper Section 5)."
    )
    parser.add_argument("--test_data_path", required=True,
                        help="Test-set JSONL (each row: {filepath, caption, ...})")
    parser.add_argument("--output_dir", required=True,
                        help="Where to write inference_results.jsonl + bbox overlays")
    parser.add_argument("--use_offline", action="store_true",
                        help="Use the test-set's existing ann_response field "
                             "instead of querying the SDG service")
    parser.add_argument("--predictions_path", default=None,
                        help="(offline-mode alt) read predictions from this jsonl")
    parser.add_argument("--server_url", default="http://localhost:17142/v1",
                        help="SDG detector OpenAI-compatible endpoint URL")
    parser.add_argument("--model_name", default="sdg-detector",
                        help="Served model name on the SDG endpoint")
    parser.add_argument("--use_bbox", action="store_true",
                        help="Render the bbox overlay image alongside text")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    run_infer_mode4(
        test_data_path=args.test_data_path,
        output_dir=args.output_dir,
        sdg_predictions_path=args.predictions_path,
        sdg_server_url=args.server_url,
        sdg_model_name=args.model_name,
        use_offline=args.use_offline,
        use_bbox=args.use_bbox,
        max_samples=args.max_samples,
    )
