#!/usr/bin/env python3
"""
SDG evaluation script for RichHF.
Use with SDG predictions and RichHF annotation JSONL files.
Prediction: judge defect presence by inspecting the JSON-array labels parsed from <answer> in `response`.
GT: judge per-class presence by whether the corresponding heatmap maximum exceeds the threshold (default 0.33).
"""

import json
import os
import argparse
import numpy as np
import cv2
import re
from tqdm import tqdm
from collections import defaultdict

def get_heatmap_max(path):
    """Read a heatmap and return its max value, normalized to [0, 1]."""
    if not path or not os.path.exists(path):
        return 0.0
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return 0.0
    
    # Normalize into [0, 1] based on dtype
    if img.dtype == np.uint8:
        return img.max() / 255.0
    elif img.dtype == np.uint16:
        return img.max() / 65535.0
    else:
        return float(img.max())

def calculate_metrics(tp, fp, fn, tn):
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    balanced_accuracy = (recall + specificity) / 2
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "specificity": specificity, "balanced_accuracy": balanced_accuracy, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}

def parse_sdg_response(text):
    """Extract the JSON array nested inside <answer>."""
    if not text:
        return []
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
    if not match:
        return []
    json_text = match.group(1).strip()
    try:
        bboxes = json.loads(json_text)
    except Exception:
        # Try ast.literal_eval as a fallback
        try:
            import ast
            bboxes = ast.literal_eval(json_text)
        except Exception:
            return []
    if isinstance(bboxes, list):
        return bboxes
    return []

def main():
    parser = argparse.ArgumentParser(description="Evaluate SDG RichHF predictions")
    parser.add_argument("--gt", default="${SDG_HOME}/eval/code/richhfGT.jsonl", help="GT JSONL file path")
    parser.add_argument("--pred", default="${SDG_HOME}/eval/code/sdg_richhf.jsonl", help="Predictions JSONL file path")
    parser.add_argument("--heatmap-threshold", type=float, default=0.33, help="Threshold for heatmap max value to determine defect presence")
    args = parser.parse_args()

    print(f"Loading GT from: {args.gt}")
    gt_data = []
    with open(args.gt, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                gt_data.append(json.loads(line))
                
    print(f"Loading Predictions from: {args.pred}")
    pred_data = []
    with open(args.pred, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                pred_data.append(json.loads(line))

    # based on basename matchpredictionand GT
    gt_dict = {}
    for gt in gt_data:
        basename = os.path.basename(gt.get("filename", ""))
        gt_dict[basename] = gt
        
    matched_gt = []
    matched_pred = []
    
    for pred in pred_data:
        basename = os.path.basename(pred.get("filepath", pred.get("filename", "")))
        if basename in gt_dict:
            matched_pred.append(pred)
            matched_gt.append(gt_dict[basename])
            
    print(f"Matched {len(matched_pred)} images between GT and Predictions.")
    
    # stats
    stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
    
    print("\nEvaluating inferences (GT: Heatmaps, Pred: Box labels)...")
    for gt, pred in tqdm(zip(matched_gt, matched_pred), total=len(matched_pred)):
        # extract GT heatmapof max value
        gt_art_path = gt.get("artifact_map_path", "")
        gt_mis_path = gt.get("misalignment_map_path", "")
        
        gt_art_max = get_heatmap_max(gt_art_path)
        gt_mis_max = get_heatmap_max(gt_mis_path)
        
        gt_art_flag = gt_art_max > args.heatmap_threshold
        gt_mis_flag = gt_mis_max > args.heatmap_threshold
        gt_overall_flag = gt_art_flag or gt_mis_flag
        
        # extract Pred bbox info
        pred_bboxes = parse_sdg_response(pred.get("response", ""))
        pred_art_flag = any(isinstance(b, dict) and b.get("label", "") in ["artifact", "both"] for b in pred_bboxes)
        pred_mis_flag = any(isinstance(b, dict) and b.get("label", "") in ["misalignment", "both"] for b in pred_bboxes)
        pred_overall_flag = pred_art_flag or pred_mis_flag
        
        # compute artifact metrics
        if gt_art_flag and pred_art_flag: stats["artifact"]["tp"] += 1
        elif gt_art_flag and not pred_art_flag: stats["artifact"]["fn"] += 1
        elif not gt_art_flag and pred_art_flag: stats["artifact"]["fp"] += 1
        else: stats["artifact"]["tn"] += 1
            
        # compute misalignment metrics
        if gt_mis_flag and pred_mis_flag: stats["misalignment"]["tp"] += 1
        elif gt_mis_flag and not pred_mis_flag: stats["misalignment"]["fn"] += 1
        elif not gt_mis_flag and pred_mis_flag: stats["misalignment"]["fp"] += 1
        else: stats["misalignment"]["tn"] += 1
            
        # compute overall metrics
        if gt_overall_flag and pred_overall_flag: stats["overall"]["tp"] += 1
        elif gt_overall_flag and not pred_overall_flag: stats["overall"]["fn"] += 1
        elif not gt_overall_flag and pred_overall_flag: stats["overall"]["fp"] += 1
        else: stats["overall"]["tn"] += 1

    # compute final metrics
    res_overall = calculate_metrics(**stats["overall"])
    res_artifact = calculate_metrics(**stats["artifact"])
    res_misalignment = calculate_metrics(**stats["misalignment"])

    print("\n" + "=" * 90)
    print("Summary")
    print("=" * 90)
    
    header = f"{'metrics':<35} {'Precision':>10} {'Recall':>10} {'Specificity':>12} {'Bal_Acc':>10} {'F1':>10}"
    sep = "-" * 95
    
    print()
    print(header)
    print(sep)
    print(f"{'image-level (overall)':<35} {res_overall['precision']:>10.4f} {res_overall['recall']:>10.4f} {res_overall['specificity']:>12.4f} {res_overall['balanced_accuracy']:>10.4f} {res_overall['f1']:>10.4f}")
    print(f"{'image-level (artifact)':<35} {res_artifact['precision']:>10.4f} {res_artifact['recall']:>10.4f} {res_artifact['specificity']:>12.4f} {res_artifact['balanced_accuracy']:>10.4f} {res_artifact['f1']:>10.4f}")
    print(f"{'image-level (misalignment)':<35} {res_misalignment['precision']:>10.4f} {res_misalignment['recall']:>10.4f} {res_misalignment['specificity']:>12.4f} {res_misalignment['balanced_accuracy']:>10.4f} {res_misalignment['f1']:>10.4f}")
    print(sep)
    
    print("\n======================================================================")
    print("Detailed detection matrices")
    print("======================================================================")
    for label, r in zip(["overall", "artifact", "misalignment"], [res_overall, res_artifact, res_misalignment]):
        print(f"\n  --- {label} ---")
        print(f"  TP={r['tp']}, FP={r['fp']}, FN={r['fn']}, TN={r['tn']}")
        print(f"  Accuracy:  {r['accuracy']:.4f}")
        print(f"  Precision: {r['precision']:.4f}")
        print(f"  Recall:    {r['recall']:.4f}")
        print(f"  Specificity:{r['specificity']:.4f}")
        print(f"  Bal_Acc:   {r['balanced_accuracy']:.4f}")
        print(f"  F1 Score:  {r['f1']:.4f}")

    # saveresult
    out_path = os.path.join(os.path.dirname(os.path.abspath(args.pred)), "eval_sdg_richhf_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        result_dict = {
            "overall": res_overall,
            "artifact": res_artifact,
            "misalignment": res_misalignment
        }
        json.dump(result_dict, f, indent=4)
        
    print(f"\nResults saved to: {out_path}")

if __name__ == "__main__":
    main()
