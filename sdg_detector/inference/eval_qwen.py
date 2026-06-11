"""
Evaluation Script for SFT-trained Qwen3-VL models (Qwen bbox format).
Evaluates bounding box prediction using IoU metrics.
Uses SGLang for fast batched inference.

Supports separate evaluation for misalignment and artifact bboxes.

Bbox format: [x0, y0, x1, y1] (Qwen format)
- x0, y0: top-left corner coordinates
- x1, y1: bottom-right corner coordinates
- All values normalized to 0-1000 range

Usage:
    python sdg_detector/inference/eval_qwen.py \\
        --model_path ./outputs/sft/checkpoint-xxx \\
        --eval_dataset_path /path/to/test.jsonl \\
        --output_dir ./eval_results
"""
import os
import json
import re
import ast
import argparse
import logging
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime

from PIL import Image
from tqdm import tqdm

from shapely.geometry import box
from shapely.ops import unary_union

# Import prompts
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train.constants import system_prompt_registry, question_template_registry


def multi_box_iou_union(gt_boxes: List[List[float]], pred_boxes: List[List[float]], eps: float = 1e-9) -> Tuple[float, bool]:
    """
    Calculate IoU using union    of boxes.
    Returns (iou_value, is_both_empty).
    """
    # Handle both empty case
    gt_empty = len(gt_boxes) == 0 or gt_boxes == [[0, 0, 0, 0]]
    pred_empty = len(pred_boxes) == 0 or pred_boxes == [[0, 0, 0, 0]]
    
    if gt_empty and pred_empty:
        return 1.0, True  # Both empty, perfect match
    
    if gt_empty or pred_empty:
        return 0.0, False  # One empty, one not
    
    try:
        gt_union = unary_union([box(*b) for b in gt_boxes])
        pr_union = unary_union([box(*b) for b in pred_boxes])
        inter = gt_union.intersection(pr_union).area
        union = gt_union.union(pr_union).area
        return inter / (union + eps), False
    except Exception:
        return 0.0, False


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True  # Force reconfiguration even if logging was already configured
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False  # Prevent duplicate logs from propagating to root logger
# Ensure we have exactly one console handler
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(console_handler)


def smart_resize(
    height: int, width: int, factor: int = 32,
    min_pixels: int = 256 * 32 * 32,
    max_pixels: int = 1280 * 32 * 32
) -> tuple:
    """
    Rescales the image so that the following conditions are met:
    1. Both dimensions (height and width) are divisible by 'factor'.
    2. The total number    of pixels is within the range ['min_pixels', 'max_pixels'].
    3. The aspect ratio    of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = (height * width / max_pixels) ** 0.5
        h_bar = max(factor, int(height / beta / factor) * factor)
        w_bar = max(factor, int(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = (min_pixels / height / width) ** 0.5
        h_bar = max(factor, int(height * beta / factor) * factor)
        w_bar = max(factor, int(width * beta / factor) * factor)
    return h_bar, w_bar


def parse_boxes_from_response(response: str) -> Tuple[List[Dict], List[Dict]]:
    """
    Parse bounding boxes from model response.
    Returns (misalignment_boxes, artifact_boxes).
    Each box is a dict with 'box_2d', 'label', 'desc'.
    """
    # Try to extract from <answer>...</answer>
    answer_match = re.search(r'<answer>\s*(.*?)\s*</answer>', response, re.DOTALL)
    if answer_match:
        boxes_str = answer_match.group(1)
    else:
        boxes_str = response
    
    # Try to parse JSON
    parsed_boxes = []
    
    # Try json.loads first
    try:
        # Find JSON array
        start = boxes_str.find('[')
        end = boxes_str.rfind(']')
        if start != -1 and end != -1 and end > start:
            result = json.loads(boxes_str[start:end+1])
            if isinstance(result, list):
                parsed_boxes = result
    except Exception:
        pass
    
    # If JSON failed, try to find individual box patterns
    if not parsed_boxes:
        # Pattern for structured boxes with label
        box_pattern = r'\{[^}]*"box_2d"\s*:\s*\[(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\][^}]*"label"\s*:\s*"([^"]+)"[^}]*\}'
        matches = re.findall(box_pattern, boxes_str, re.DOTALL)
        for match in matches:
            parsed_boxes.append({
                'box_2d': [float(match[0]), float(match[1]), float(match[2]), float(match[3])],
                'label': match[4],
                'desc': ''
            })
    
    # Separate by label, keep full dict format
    misalignment_boxes = []
    artifact_boxes = []
    misalignment_boxes_coords = []
    artifact_boxes_coords = []
    
    for item in parsed_boxes:
        if isinstance(item, dict) and 'box_2d' in item:
            box_coords = item['box_2d']
            if isinstance(box_coords, list) and len(box_coords) == 4:
                label = item.get('label', '').lower()
                box_entry = {
                    'box_2d': box_coords,
                    'label': item.get('label', ''),
                    'desc': item.get('desc', '')
                }
                if label == 'misalignment':
                    misalignment_boxes.append(box_entry)
                    misalignment_boxes_coords.append(box_coords)
                elif label == 'artifact':
                    artifact_boxes.append(box_entry)
                    artifact_boxes_coords.append(box_coords)
    
    # Return both formats: coords for IoU, full dicts for output
    return misalignment_boxes_coords, artifact_boxes_coords, misalignment_boxes, artifact_boxes


def extract_gt_boxes(sample: Dict):
    """
    Extract GT boxes from sample and convert to Qwen format.
    GT is stored in Gemini format [y0, x0, y1, x1], need to convert to Qwen format [x0, y0, x1, y1].
    Returns (misalignment_coords, artifact_coords, misalignment_raw, artifact_raw).
    - coords: List    of [x0, y0, x1, y1] for IoU calculation (converted to Qwen format)
    - raw: Original format with box_2d, label, desc for output
    """
    def gemini_to_qwen(box):
        """Convert Gemini [y0, x0, y1, x1] to Qwen [x0, y0, x1, y1]"""
        y0, x0, y1, x1 = box
        return [x0, y0, x1, y1]
    
    misalignment_coords = []
    artifact_coords = []
    
    # Get raw format (original dicts with box_2d, label, desc)
    gt_misalignment_raw = sample.get('gemini_misalignment_bboxes', [])
    gt_artifact_raw = sample.get('gemini_artifact_bboxes', [])
    
    for item in gt_misalignment_raw:
        if isinstance(item, dict) and 'box_2d' in item:
            # Convert from Gemini to Qwen format
            misalignment_coords.append(gemini_to_qwen(item['box_2d']))
        elif isinstance(item, list) and len(item) == 4:
            misalignment_coords.append(gemini_to_qwen(item))
    
    for item in gt_artifact_raw:
        if isinstance(item, dict) and 'box_2d' in item:
            # Convert from Gemini to Qwen format
            artifact_coords.append(gemini_to_qwen(item['box_2d']))
        elif isinstance(item, list) and len(item) == 4:
            artifact_coords.append(gemini_to_qwen(item))
    
    return misalignment_coords, artifact_coords, gt_misalignment_raw, gt_artifact_raw


def prepare_sample(sample: Dict, system_prompt: str, question_template: str, 
                   min_pixels: int, max_pixels: int) -> Optional[Dict]:
    """Prepare a single sample for inference."""
    # Either `filename` or `filepath` works here
    image_path = sample.get("filename") or sample.get("filepath")
    
    if not image_path or not os.path.exists(image_path):
        return None
    
    # Get GT boxes (coords for IoU, raw for output)
    gt_mis_coords, gt_art_coords, gt_mis_raw, gt_art_raw = extract_gt_boxes(sample)
    
    caption = sample.get("caption", "")
    question_text = question_template.format(caption=caption)
    
    return {
        "system_prompt": system_prompt,
        "question": question_text,
        "image_path": image_path,
        "gt_misalignment_boxes": gt_mis_coords,  # For IoU calc
        "gt_artifact_boxes": gt_art_coords,      # For IoU calc
        "gt_misalignment_bboxes": gt_mis_raw,    # Raw format for output
        "gt_artifact_bboxes": gt_art_raw,        # Raw format for output
        "caption": caption,
    }


def evaluate_with_sglang_offline(
    model_path: str,
    prepared_data: List[Dict],
    max_new_tokens: int = 512,
    tp_size: int = 1,
    save_predictions: bool = False,
    tokenizer_path: str = None,
    batch_size: int = 32,  # batch size argument
) -> Dict[str, Any]:
    """
    Evaluate using SGLang offline engine (no server needed).
    Uses batch inference for high throughput.
    """
    import sglang as sgl
    
    logger.info(f"Initializing SGLang offline engine with model: {model_path}")
    if tokenizer_path:
        logger.info(f"Using tokenizer from: {tokenizer_path}")
    
    # Initialize engine
    engine_kwargs = {
        "model_path": model_path,
        "tp_size": tp_size,
    }
    if tokenizer_path:
        engine_kwargs["tokenizer_path"] = tokenizer_path
    
    llm = sgl.Engine(**engine_kwargs)
    
    # Get processor for applying chat template
    from transformers import AutoProcessor
    processor_path = tokenizer_path if tokenizer_path else model_path
    processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)
    
    # Prepare prompts by applying chat template
    logger.info(f"Running batch inference on {len(prepared_data)} samples (batch_size={batch_size})...")
    
    prompts = []
    image_paths = []
    for data in prepared_data:
        messages = [
            {"role": "system", "content": data["system_prompt"]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": data["image_path"]},
                    {"type": "text", "text": data["question"]},
                ]
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # # Pre-fill an empty <think> block to force the model to emit <answer> directly
        # text = text + "<think>\nDue to efficiency constraints, I need to skip the thinking process and directly generate the answer.\n</think>\n"
        prompts.append(text)
        image_paths.append(data["image_path"])
    
    # Generate with images in batches to avoid OOM
    sampling_params = {"max_new_tokens": max_new_tokens, "temperature": 0}
    
    outputs = []
    total_batches = (len(prompts) + batch_size - 1) // batch_size
    
    pbar = tqdm(range(0, len(prompts), batch_size), total=total_batches, desc="Batch Inference")
    for i in pbar:
        batch_prompts = prompts[i:i+batch_size]
        batch_images = image_paths[i:i+batch_size]
        pbar.set_postfix({"samples": f"{min(i+batch_size, len(prompts))}/{len(prompts)}"})
        
        batch_outputs = llm.generate(
            batch_prompts,
            sampling_params,
            image_data=batch_images,
        )
        outputs.extend(batch_outputs)
    
    # Process results - separate metrics for each type
    # Track IoUs with and without both-empty samples
    misalignment_ious_with_empty = []  # Both empty = 1.0
    misalignment_ious_exclude_empty = []  # Exclude both-empty samples
    artifact_ious_with_empty = []
    artifact_ious_exclude_empty = []
    
    misalignment_both_empty = 0
    artifact_both_empty = 0
    total_both_empty = 0  # Both types are both-empty
    
    # Track GT empty counts
    misalignment_gt_empty = 0
    artifact_gt_empty = 0
    misalignment_pred_empty_when_gt_empty = 0
    artifact_pred_empty_when_gt_empty = 0
    
    misalignment_correct_05 = 0
    artifact_correct_05 = 0
    
    total_samples = 0
    predictions = []
    
    for data, output in zip(prepared_data, outputs):
        # Handle different output formats from SGLang
        if isinstance(output, dict):
            response_text = output.get("text", output.get("content", ""))
        elif isinstance(output, str):
            response_text = output
        else:
            response_text = str(output)
        
        # Parse predicted boxes (coords for IoU, full dicts for output)
        pred_misalignment, pred_artifact, pred_mis_raw, pred_art_raw = parse_boxes_from_response(response_text)
        
        gt_misalignment = data["gt_misalignment_boxes"]
        gt_artifact = data["gt_artifact_boxes"]
        gt_mis_raw = data.get("gt_misalignment_bboxes", [])
        gt_art_raw = data.get("gt_artifact_bboxes", [])
        
        # Track GT empty and prediction accuracy when GT is empty
        gt_mis_empty = len(gt_misalignment) == 0
        gt_art_empty = len(gt_artifact) == 0
        pred_mis_empty = len(pred_misalignment) == 0
        pred_art_empty = len(pred_artifact) == 0
        
        if gt_mis_empty:
            misalignment_gt_empty += 1
            if pred_mis_empty:
                misalignment_pred_empty_when_gt_empty += 1
        if gt_art_empty:
            artifact_gt_empty += 1
            if pred_art_empty:
                artifact_pred_empty_when_gt_empty += 1
        
        # Calculate IoU for misalignment
        mis_iou, mis_both_empty = multi_box_iou_union(gt_misalignment, pred_misalignment)
        misalignment_ious_with_empty.append(mis_iou)
        if mis_both_empty:
            misalignment_both_empty += 1
        else:
            misalignment_ious_exclude_empty.append(mis_iou)
        if mis_iou >= 0.5:
            misalignment_correct_05 += 1
        
        # Calculate IoU for artifact
        art_iou, art_both_empty = multi_box_iou_union(gt_artifact, pred_artifact)
        artifact_ious_with_empty.append(art_iou)
        if art_both_empty:
            artifact_both_empty += 1
        else:
            artifact_ious_exclude_empty.append(art_iou)
        if art_iou >= 0.5:
            artifact_correct_05 += 1
        
        # Check if both types are both-empty
        if mis_both_empty and art_both_empty:
            total_both_empty += 1
        
        total_samples += 1
        
        if save_predictions:
            predictions.append({
                "filename": data["image_path"],
                "caption": data["caption"],
                "gemini_misalignment_bboxes": gt_mis_raw,
                "gemini_artifact_bboxes": gt_art_raw,
                "pred_misalignment_bboxes": pred_mis_raw,
                "pred_artifact_bboxes": pred_art_raw,
                "misalignment_iou": mis_iou,
                "artifact_iou": art_iou,
                "misalignment_both_empty": mis_both_empty,
                "artifact_both_empty": art_both_empty,
                "response": response_text,
            })
    
    # Cleanup
    llm.shutdown()
    
    if total_samples == 0:
        return {
            "metrics": {},
            "predictions": predictions
        }
    
    # Calculate combined IoU (average    of both types)
    combined_ious_with_empty = [(m + a) / 2 for m, a in zip(misalignment_ious_with_empty, artifact_ious_with_empty)]
    
    # Helper function for safe mean
    def safe_mean(lst):
        return sum(lst) / len(lst) if lst else 0.0
    
    metrics = {
        # Misalignment metrics (with empty = 1.0)
        "misalignment_mean_iou": safe_mean(misalignment_ious_with_empty),
        "misalignment_mean_iou_exclude_empty": safe_mean(misalignment_ious_exclude_empty),
        "misalignment_acc@0.5": misalignment_correct_05 / total_samples,
        "misalignment_both_empty_count": misalignment_both_empty,
        "misalignment_both_empty_rate": misalignment_both_empty / total_samples,
        "misalignment_gt_empty_count": misalignment_gt_empty,
        "misalignment_gt_empty_rate": misalignment_gt_empty / total_samples,
        "misalignment_pred_empty_when_gt_empty": misalignment_pred_empty_when_gt_empty / misalignment_gt_empty if misalignment_gt_empty > 0 else 0.0,
        "misalignment_non_empty_count": len(misalignment_ious_exclude_empty),
        
        # Artifact metrics (with empty = 1.0)
        "artifact_mean_iou": safe_mean(artifact_ious_with_empty),
        "artifact_mean_iou_exclude_empty": safe_mean(artifact_ious_exclude_empty),
        "artifact_acc@0.5": artifact_correct_05 / total_samples,
        "artifact_both_empty_count": artifact_both_empty,
        "artifact_both_empty_rate": artifact_both_empty / total_samples,
        "artifact_gt_empty_count": artifact_gt_empty,
        "artifact_gt_empty_rate": artifact_gt_empty / total_samples,
        "artifact_pred_empty_when_gt_empty": artifact_pred_empty_when_gt_empty / artifact_gt_empty if artifact_gt_empty > 0 else 0.0,
        "artifact_non_empty_count": len(artifact_ious_exclude_empty),
        
        # Combined metrics
        "combined_mean_iou": safe_mean(combined_ious_with_empty),
        "total_samples": total_samples,
        "total_both_empty_count": total_both_empty,
        "total_both_empty_rate": total_both_empty / total_samples,
    }
    
    return {"metrics": metrics, "predictions": predictions}


def main():
    parser = argparse.ArgumentParser(description="Evaluate SFT-trained Qwen3-VL model using SGLang (offline mode)")
    
    # Model settings
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                        help="Path to tokenizer/processor (default: use model_path)")
    parser.add_argument("--tp_size", type=int, default=8,
                        help="Tensor parallel size")
    
    # Data paths
    parser.add_argument("--eval_dataset_path", type=str, required=True,
                        help="Path to evaluation JSONL file")
    parser.add_argument("--output_dir", type=str, default="./eval_results",
                        help="Directory to save evaluation results")
    
    # Image processing
    parser.add_argument("--max_pixels", type=int, default=1280*32*32)
    parser.add_argument("--min_pixels", type=int, default=256*32*32)
    
    # Evaluation settings
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number    of samples to evaluate")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for inference (reduce if OOM)")
    parser.add_argument("--system_prompt_template", type=str, default="default")
    parser.add_argument("--question_template", type=str, default="default")
    
    # Output settings
    parser.add_argument("--save_predictions", action="store_true",
                        help="Save individual predictions to file")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging to file
    log_file = os.path.join(args.output_dir, "eval.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info("=" * 60)
    logger.info("SGLang Evaluation Configuration (Offline Mode)")
    logger.info("=" * 60)
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    logger.info("=" * 60)
    
    # Load evaluation data
    logger.info(f"Loading evaluation data from {args.eval_dataset_path}")
    eval_data = []
    with open(args.eval_dataset_path, "r") as f:
        for line in f:
            if line.strip():
                eval_data.append(json.loads(line))
    
    if args.max_samples and args.max_samples < len(eval_data):
        import random
        eval_data = random.sample(eval_data, args.max_samples)
    
    logger.info(f"Loaded {len(eval_data)} evaluation samples")
    
    # Get prompts
    system_prompt = system_prompt_registry.get(args.system_prompt_template, "You are a helpful assistant.")
    question_template = question_template_registry.get(args.question_template, "")
    
    # Prepare samples
    logger.info("Preparing samples...")
    prepared_data = []
    failed_load = 0
    for sample in tqdm(eval_data, desc="Loading data"):
        try:
            data = prepare_sample(
                sample, system_prompt, question_template,
                args.min_pixels, args.max_pixels
            )
            if data:
                prepared_data.append(data)
            else:
                failed_load += 1
        except Exception as e:
            logger.warning(f"Error preparing sample: {e}")
            failed_load += 1
    
    logger.info(f"Prepared {len(prepared_data)} samples, {failed_load} failed to load")
    
    # Run evaluation
    logger.info("Starting evaluation...")
    start_time = datetime.now()
    
    results = evaluate_with_sglang_offline(
        model_path=args.model_path,
        prepared_data=prepared_data,
        max_new_tokens=args.max_new_tokens,
        tp_size=args.tp_size,
        save_predictions=args.save_predictions,
        tokenizer_path=args.tokenizer_path,
        batch_size=args.batch_size,
    )
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    metrics = results["metrics"]
    
    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("Evaluation Results (Separate Metrics)")
    logger.info("=" * 60)
    logger.info(f"  Model:      {args.model_path}")
    logger.info(f"  Dataset:    {args.eval_dataset_path}")
    logger.info(f"  Samples:    {metrics.get('total_samples', 0)}")
    logger.info(f"  Duration:   {duration:.1f}s")
    logger.info("-" * 60)
    logger.info("Misalignment Metrics:")
    logger.info(f"  Mean IoU (with empty=1):    {metrics.get('misalignment_mean_iou', 0):.4f}")
    logger.info(f"  Mean IoU (exclude empty):   {metrics.get('misalignment_mean_iou_exclude_empty', 0):.4f}")
    logger.info(f"  Acc@0.5:                    {metrics.get('misalignment_acc@0.5', 0):.4f}")
    logger.info(f"  GT Empty:                   {metrics.get('misalignment_gt_empty_count', 0)} ({metrics.get('misalignment_gt_empty_rate', 0):.2%})")
    logger.info(f"  Pred Empty | GT Empty:      {metrics.get('misalignment_pred_empty_when_gt_empty', 0):.2%}")
    logger.info(f"  Both Empty:                 {metrics.get('misalignment_both_empty_count', 0)} ({metrics.get('misalignment_both_empty_rate', 0):.2%})")
    logger.info(f"  Non-Empty Samples:          {metrics.get('misalignment_non_empty_count', 0)}")
    logger.info("-" * 60)
    logger.info("Artifact Metrics:")
    logger.info(f"  Mean IoU (with empty=1):    {metrics.get('artifact_mean_iou', 0):.4f}")
    logger.info(f"  Mean IoU (exclude empty):   {metrics.get('artifact_mean_iou_exclude_empty', 0):.4f}")
    logger.info(f"  Acc@0.5:                    {metrics.get('artifact_acc@0.5', 0):.4f}")
    logger.info(f"  GT Empty:                   {metrics.get('artifact_gt_empty_count', 0)} ({metrics.get('artifact_gt_empty_rate', 0):.2%})")
    logger.info(f"  Pred Empty | GT Empty:      {metrics.get('artifact_pred_empty_when_gt_empty', 0):.2%}")
    logger.info(f"  Both Empty:                 {metrics.get('artifact_both_empty_count', 0)} ({metrics.get('artifact_both_empty_rate', 0):.2%})")
    logger.info(f"  Non-Empty Samples:          {metrics.get('artifact_non_empty_count', 0)}")
    logger.info("-" * 60)
    logger.info("Combined Metrics:")
    logger.info(f"  Mean IoU:       {metrics.get('combined_mean_iou', 0):.4f}")
    logger.info(f"  Total Both Empty: {metrics.get('total_both_empty_count', 0)} ({metrics.get('total_both_empty_rate', 0):.2%})")
    logger.info("=" * 60)
    
    # Save metrics
    metrics_output = {
        "model_path": args.model_path,
        "eval_dataset_path": args.eval_dataset_path,
        "config": vars(args),
        "metrics": metrics,
        "duration_seconds": duration,
        "samples_per_second": metrics.get('total_samples', 0) / max(1, duration),
        "timestamp": datetime.now().isoformat(),
    }
    
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_output, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")
    
    # Save predictions if requested
    if args.save_predictions and results["predictions"]:
        predictions_path = os.path.join(args.output_dir, "predictions.jsonl")
        with open(predictions_path, "w") as f:
            for pred in results["predictions"]:
                f.write(json.dumps(pred) + "\n")
        logger.info(f"Predictions saved to {predictions_path}")
    
    # Print summary to console
    print("\n" + "=" * 70)
    print("Evaluation Summary")
    print("=" * 70)
    print(f"Misalignment - IoU: {metrics.get('misalignment_mean_iou', 0):.4f} (excl: {metrics.get('misalignment_mean_iou_exclude_empty', 0):.4f}), Acc@0.5: {metrics.get('misalignment_acc@0.5', 0):.4f}, Empty: {metrics.get('misalignment_both_empty_rate', 0):.1%}")
    print(f"Artifact     - IoU: {metrics.get('artifact_mean_iou', 0):.4f} (excl: {metrics.get('artifact_mean_iou_exclude_empty', 0):.4f}), Acc@0.5: {metrics.get('artifact_acc@0.5', 0):.4f}, Empty: {metrics.get('artifact_both_empty_rate', 0):.1%}")
    print(f"Combined     - IoU: {metrics.get('combined_mean_iou', 0):.4f}, Total Both Empty: {metrics.get('total_both_empty_rate', 0):.1%}")
    print(f"Samples: {metrics.get('total_samples', 0)}, Speed: {metrics.get('total_samples', 0)/max(1,duration):.2f} samples/s")
    print("=" * 70)


if __name__ == "__main__":
    main()
