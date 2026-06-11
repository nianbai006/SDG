"""
Phase 1 inference: ImageDoctor eval
- online: Call the ImageDoctor API to evaluate the image
- offline: Read existing predictions from predictions.jsonl
- Save text predictions + heatmap paths to inference_results.jsonl
- if offline mode supplies heatmaps, copy them to the output directory
"""
import json
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional

from data_loader import (
    load_test_data,
    load_imagedoctor_predictions,
    match_imagedoctor_to_test,
    filter_existing_images,
    make_unique_basename,
)
from vlm_client import VLMClient
from prompt_builder import build_prompt_from_imagedoctor
from visualization import get_heatmap_high_regions


def run_infer_mode3(
    test_data_path: str,
    output_dir: str,
    imagedoctor_path: Optional[str] = None,
    imagedoctor_server_url: str = "http://localhost:17141/v1",
    imagedoctor_model_name: str = "ImageDoctor",
    use_offline: bool = False,
    use_heatmap: bool = False,
    heatmap_threshold: float = 0.3,
    max_samples: Optional[int] = None,
    filepath_prefix_map: Optional[Dict[str, str]] = None,
):
    """
    ImageDoctor inference stage: evaluate the image and save predictions, heatmaps, and edit instructions.

    output:
        {output_dir}/inference_results.jsonl
        {output_dir}/heatmaps/{basename}_misalignment.png  (e.g.has)
        {output_dir}/heatmaps/{basename}_artifact.png      (e.g.has)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    heatmaps_dir = output_path / "heatmaps"

    # loaddata
    test_data = load_test_data(
        test_data_path, max_samples=max_samples,
        filepath_prefix_map=filepath_prefix_map,
    )
    test_data = filter_existing_images(test_data, filepath_key="filepath")

    if use_offline and imagedoctor_path:
        # offline mode
        print("[infer_mode3] using offline ImageDoctor predictions...")
        imagedoctor_data = load_imagedoctor_predictions(
            imagedoctor_path, filepath_prefix_map=filepath_prefix_map,
        )
        matched_data = match_imagedoctor_to_test(test_data, imagedoctor_data)
    else:
        # online mode
        print("[infer_mode3] using online ImageDoctor inference...")
        matched_data = _run_imagedoctor_online(
            test_data,
            imagedoctor_server_url=imagedoctor_server_url,
            imagedoctor_model_name=imagedoctor_model_name,
        )

    print(f"[infer_mode3] validsample: {len(matched_data)}")

    if not matched_data:
        print("[infer_mode3] no valid data!")
        return []

    # process each record and save
    results = []
    results_path = output_path / "inference_results.jsonl"
    st_time = time.time()

    for i, sample in enumerate(matched_data):
        filepath = sample["filepath"]
        caption = sample.get("caption", "")
        prediction = sample.get("imagedoctor_prediction", "")
        basename = make_unique_basename(filepath)

        if not prediction:
            continue

        if (i + 1) % 10 == 0:
            elapsed = time.time() - st_time
            avg = elapsed / (i + 1)
            eta = avg * (len(matched_data) - i - 1)
            print(f"  progress: {i+1}/{len(matched_data)}  elapsed {elapsed:.1f}s  mean {avg:.2f}s/sample  ETA {eta:.0f}s")

        # copy heatmaps into the output directory
        mis_heatmap_dst = ""
        art_heatmap_dst = ""

        mis_heatmap_src = sample.get("imagedoctor_misalignment_heatmap", "")
        art_heatmap_src = sample.get("imagedoctor_artifact_heatmap", "")

        if mis_heatmap_src and os.path.exists(mis_heatmap_src):
            heatmaps_dir.mkdir(parents=True, exist_ok=True)
            mis_heatmap_dst = str(heatmaps_dir / f"{basename}_misalignment.png")
            shutil.copy2(mis_heatmap_src, mis_heatmap_dst)

        if art_heatmap_src and os.path.exists(art_heatmap_src):
            heatmaps_dir.mkdir(parents=True, exist_ok=True)
            art_heatmap_dst = str(heatmaps_dir / f"{basename}_artifact.png")
            shutil.copy2(art_heatmap_src, art_heatmap_dst)

        # analyse heatmaps for use in the prompt
        heatmap_regions = None
        if use_heatmap:
            regions = []
            if mis_heatmap_dst and os.path.exists(mis_heatmap_dst):
                mis_regions = get_heatmap_high_regions(
                    mis_heatmap_dst, threshold=heatmap_threshold
                )
                regions.extend([f"misalignment in {r}" for r in mis_regions])
            if art_heatmap_dst and os.path.exists(art_heatmap_dst):
                art_regions = get_heatmap_high_regions(
                    art_heatmap_dst, threshold=heatmap_threshold
                )
                regions.extend([f"artifact in {r}" for r in art_regions])
            heatmap_regions = regions if regions else None

        # Build the edit prompt
        edit_prompt = build_prompt_from_imagedoctor(
            caption=caption,
            prediction=prediction,
            use_heatmap_info=use_heatmap,
            heatmap_regions=heatmap_regions,
        )

        record = {
            "filepath": filepath,
            "caption": caption,
            "prediction": prediction,
            "misalignment_heatmap": mis_heatmap_dst,
            "artifact_heatmap": art_heatmap_dst,
            "heatmap_regions": heatmap_regions,
            "edit_prompt": edit_prompt,
        }
        results.append(record)

        with open(results_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    total_time = time.time() - st_time
    print(f"[infer_mode3] done! saved {len(results)} results to {results_path}")
    print(f"  elapsed: {total_time:.1f}s  mean: {total_time/max(len(results),1):.2f}s/sample")
    return results


def _run_imagedoctor_online(
    data: List[Dict],
    imagedoctor_server_url: str,
    imagedoctor_model_name: str,
) -> List[Dict]:
    """Online call to the ImageDoctor model to evaluate the image."""
    from config import IMAGEDOCTOR_PROMPT

    vlm = VLMClient(
        base_url=imagedoctor_server_url,
        model_name=imagedoctor_model_name,
        max_new_tokens=4096,
        timeout=180,
    )

    if not vlm.check_health():
        print("[infer_mode3] error: ImageDoctor serviceunavailable!")
        print("  please deploy first: bash scripts/deploy_imagedoctor_server.sh")
        return data

    print(f"[infer_mode3] ImageDoctor online inference ({imagedoctor_model_name})...")
    st_time = time.time()

    for i, sample in enumerate(data):
        filepath = sample["filepath"]
        caption = sample.get("caption", "")

        if (i + 1) % 10 == 0:
            elapsed = time.time() - st_time
            avg = elapsed / (i + 1)
            eta = avg * (len(data) - i - 1)
            print(f"  ImageDoctor inference progress: {i+1}/{len(data)}  elapsed {elapsed:.1f}s  mean {avg:.2f}s/sample  ETA {eta:.0f}s")

        result = vlm.evaluate_image(
            image_path=filepath,
            caption=caption,
            system_prompt="You are a helpful assistant.",
            user_prompt_template=IMAGEDOCTOR_PROMPT,
        )

        if result["success"]:
            sample["imagedoctor_prediction"] = result["text"]
        else:
            sample["imagedoctor_prediction"] = ""
            print(f"  ImageDoctor inference failed ({i}): {result['error'][:100]}")

    return data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase 1 — run ImageDoctor inference on a directory of images "
                    "and save the heatmaps + text used by GPT-Image-1.5 as feedback "
                    "(paper Section 5)."
    )
    parser.add_argument("--test_data_path", required=True,
                        help="Test-set JSONL (each row: {filepath, caption, ...})")
    parser.add_argument("--output_dir", required=True,
                        help="Where to write inference_results.jsonl + heatmap PNGs")
    parser.add_argument("--use_offline", action="store_true",
                        help="Use a precomputed ImageDoctor predictions.jsonl "
                             "instead of querying the service")
    parser.add_argument("--predictions_path", default=None,
                        help="(offline-mode alt) read predictions from this jsonl")
    parser.add_argument("--server_url", default="http://localhost:17141/v1",
                        help="ImageDoctor OpenAI-compatible endpoint URL")
    parser.add_argument("--model_name", default="ImageDoctor",
                        help="Served model name on the ImageDoctor endpoint")
    parser.add_argument("--use_heatmap", action="store_true",
                        help="Copy the per-image heatmap PNGs into output_dir")
    parser.add_argument("--heatmap_threshold", type=float, default=0.3,
                        help="Threshold above which a heatmap pixel is 'highlighted'")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    run_infer_mode3(
        test_data_path=args.test_data_path,
        output_dir=args.output_dir,
        imagedoctor_path=args.predictions_path,
        imagedoctor_server_url=args.server_url,
        imagedoctor_model_name=args.model_name,
        use_offline=args.use_offline,
        use_heatmap=args.use_heatmap,
        heatmap_threshold=args.heatmap_threshold,
        max_samples=args.max_samples,
    )
