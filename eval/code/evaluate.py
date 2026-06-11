#!/usr/bin/env python3
"""
SDG evaluation script: multi-dimensional evaluation of GT vs predictions
Task: per-image defect detection (each image yields a list of bounding boxes with label and description)

datastructure:
  GT:  {"filepath", "caption", "ann_translated_bboxes": [{"box_2d": [x1,y1,x2,y2], "label": str, "description": str}]}
  Pred: {"filepath", "caption", "response": str}  -- the bbox list must be parsed from <answer>[...]</answer> inside `response`

Metrics (sorted by recommended priority):
  1. Image-Level Classification (defect / no-defect) - Accuracy, Precision, Recall, F1
  2. Label-Level Detection (artifact / misalignment classes) - Per-class Precision, Recall, F1
  3. Bbox-Level Detection (IoU matching) - AP, mAP
  4. Count-Based Metrics (prediction count vs GT count)
  5. Response quality (parse success rate, etc.)
python ${SDG_HOME}/eval/code/evaluate.py --pred jsonl
"""

import json
import os
import re
import sys
import numpy as np
from collections import defaultdict, Counter
from statistics import median


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def _try_parse_bbox_json(json_text):
    """Try parsing JSON text into a bbox list; return list or None."""
    json_text = json_text.strip()
    if not json_text:
        return []
    try:
        bboxes = json.loads(json_text)
    except json.JSONDecodeError:
        try:
            import ast
            bboxes = ast.literal_eval(json_text)
        except Exception:
            return None
    if isinstance(bboxes, list):
        valid_bboxes = []
        for b in bboxes:
            if isinstance(b, dict) and "box_2d" in b and "label" in b:
                valid_bboxes.append(b)
        return valid_bboxes
    return []


def parse_prediction_bboxes(response_text):
    """Parse the bbox list from a prediction's `response` field.

    Supported formats:
      1. <answer>JSON</answer>          (may be preceded by <think>...</think>)
      2. ```json\nJSON\n```             (Markdown JSON code block, possibly preceded by <think>...</think>)
      3. raw JSON array
    """
    if not response_text or not isinstance(response_text, str):
        return None  # parsefailed

    # ----- strategy 1: search inside the <answer>...</answer> block -----
    answer_match = re.search(r'<answer>\s*(.*?)\s*</answer>', response_text, re.DOTALL)
    if answer_match:
        answer_content = answer_match.group(1).strip()
        # the answer block may further wrap a ```json ... ``` fence
        inner_code = re.search(r'```(?:json)?\s*(.*?)\s*```', answer_content, re.DOTALL)
        if inner_code:
            answer_content = inner_code.group(1).strip()
        result = _try_parse_bbox_json(answer_content)
        if result is not None:
            return result

    # ----- strategy 2: search inside a ```json ... ``` markdown block -----
    code_match = re.search(r'```(?:json)?\s*(.*?)\s*```', response_text, re.DOTALL)
    if code_match:
        code_content = code_match.group(1).strip()
        result = _try_parse_bbox_json(code_content)
        if result is not None:
            return result

    # ----- strategy 3: try parsing the whole response as a JSON array -----
    stripped = response_text.strip()
    if stripped.startswith('['):
        result = _try_parse_bbox_json(stripped)
        if result is not None:
            return result

    return None  # all parsing strategies failed


def compute_iou(box1, box2):
    """Compute IoU between two bboxes; box format: [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, y2 - y1) * max(0, x2 - x1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def _labels_match(pred_label, gt_label):
    """Decide whether the prediction label matches the GT label.
    
    'both' matches 'artifact', 'misalignment', or another 'both'.
    """
    if pred_label == gt_label:
        return True
    if pred_label == "both" and gt_label in ("artifact", "misalignment"):
        return True
    if gt_label == "both" and pred_label in ("artifact", "misalignment"):
        return True
    return False


def _normalize_label(label):
    """Normalize 'both' to 'artifact' for per-label stats (both is counted in both classes)."""
    return label


def match_boxes(gt_bboxes, pred_bboxes, iou_threshold):
    """Greedy match by IoU desc; requires the same label."""
    matched_gt = set()
    matched_pred = set()
    matches = []

    iou_pairs = []
    for pi, pb in enumerate(pred_bboxes):
        if "box_2d" not in pb:
            continue
        for gi, gb in enumerate(gt_bboxes):
            if "box_2d" not in gb:
                continue
            iou = compute_iou(pb["box_2d"], gb["box_2d"])
            if iou >= iou_threshold:
                iou_pairs.append((iou, pi, gi))

    iou_pairs.sort(reverse=True)

    for iou_val, pi, gi in iou_pairs:
        if pi in matched_pred or gi in matched_gt:
            continue
        pred_label = pred_bboxes[pi].get("label", "")
        gt_label = gt_bboxes[gi].get("label", "")
        if _labels_match(pred_label, gt_label):
            matched_pred.add(pi)
            matched_gt.add(gi)
            matches.append(
                {
                    "pred_idx": pi,
                    "gt_idx": gi,
                    "iou": iou_val,
                    "pred_label": pred_label,
                    "gt_label": gt_label,
                }
            )

    unmatched_pred = [pi for pi, pb in enumerate(pred_bboxes) if "box_2d" in pb and pi not in matched_pred]
    unmatched_gt = [gi for gi, gb in enumerate(gt_bboxes) if "box_2d" in gb and gi not in matched_gt]
    return {
        "matches": matches,
        "unmatched_pred": unmatched_pred,
        "unmatched_gt": unmatched_gt,
    }


def normalize_desc(text):
    """Lightweight normalization of `desc` for the zero-dep semantic baseline."""
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize_desc(text):
    normalized = normalize_desc(text)
    if not normalized:
        return []
    return normalized.split()


def compute_desc_similarity(gt_desc, pred_desc):
    """Zero-dependency description similarity: token-level F1."""
    gt_tokens = tokenize_desc(gt_desc)
    pred_tokens = tokenize_desc(pred_desc)
    if not gt_tokens and not pred_tokens:
        return 1.0
    if not gt_tokens or not pred_tokens:
        return 0.0

    gt_counter = Counter(gt_tokens)
    pred_counter = Counter(pred_tokens)
    overlap = sum(min(gt_counter[t], pred_counter[t]) for t in gt_counter.keys() & pred_counter.keys())
    if overlap == 0:
        return 0.0

    precision = overlap / sum(pred_counter.values())
    recall = overlap / sum(gt_counter.values())
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


class TransformerEmbeddingScorer:
    """Cosine similarity using a local transformers text-embedding model."""

    def __init__(self, model_name_or_path, batch_size=32, max_length=256):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.model.eval()
        self.cache = {}

    def _mean_pool(self, last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = self.torch.sum(last_hidden_state * mask, dim=1)
        counts = self.torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    def encode_texts(self, texts):
        missing = [text for text in texts if text not in self.cache]
        if not missing:
            return

        with self.torch.no_grad():
            for start in range(0, len(missing), self.batch_size):
                batch = missing[start:start + self.batch_size]
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                outputs = self.model(**encoded)
                last_hidden_state = getattr(outputs, "last_hidden_state", None)
                if last_hidden_state is None and isinstance(outputs, (tuple, list)) and outputs:
                    last_hidden_state = outputs[0]
                if last_hidden_state is None:
                    raise RuntimeError("Embedding model did not return last_hidden_state")
                embeddings = self._mean_pool(last_hidden_state, encoded["attention_mask"])
                embeddings = self.torch.nn.functional.normalize(embeddings, p=2, dim=1)
                embeddings = embeddings.detach().cpu().numpy()
                for text, vec in zip(batch, embeddings):
                    self.cache[text] = vec

    def similarity(self, text_a, text_b):
        text_a = text_a if isinstance(text_a, str) else ""
        text_b = text_b if isinstance(text_b, str) else ""
        self.encode_texts([text_a, text_b])
        vec_a = self.cache[text_a]
        vec_b = self.cache[text_b]
        return float(np.dot(vec_a, vec_b))


def metric1_image_level(gt_data, pred_data, pred_parsed):
    """
    Metric 1: image-level classification (most recommended)
    Reduce to a binary task: does the image have any defect (any bbox = positive, no bbox = negative)?
    """
    print("\n" + "=" * 70)
    print("Metric 1: image-level binary classification (Image-Level Classification)")
    print("  Decide whether each image has a defect (i.e. has any bbox)")
    print("=" * 70)
    
    tp = fp = fn = tn = 0
    
    for i, (gt, pred) in enumerate(zip(gt_data, pred_data)):
        gt_has_defect = len(gt["ann_translated_bboxes"]) > 0
        
        parsed = pred_parsed[i]
        if parsed is None:
            # parse failed -> treat as predicting 'no defect'
            pred_has_defect = False
        else:
            pred_has_defect = len(parsed) > 0
        
        if gt_has_defect and pred_has_defect:
            tp += 1
        elif gt_has_defect and not pred_has_defect:
            fn += 1
        elif not gt_has_defect and pred_has_defect:
            fp += 1
        else:
            tn += 1
    
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n  Confusion matrix:")
    print(f"                predictionhasdefect    predictionnodefect")
    print(f"  GThasdefect      TP={tp:<8d}  FN={fn:<8d}")
    print(f"  GTnodefect      FP={fp:<8d}  TN={tn:<8d}")
    print(f"\n  total samples: {total}")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def metric2_label_level(gt_data, pred_data, pred_parsed):
    """
    Metric 2: per-class detection (Label-Level Detection)
    Evaluate image-level detection separately for the artifact and misalignment labels
    """
    print("\n" + "=" * 70)
    print("Metric 2: per-class detection (Label-Level Detection)")
    print("  Evaluate image-level artifact / misalignment detection separately")
    print("=" * 70)
    
    labels = ["artifact", "misalignment"]
    results = {}
    
    for label in labels:
        tp = fp = fn = tn = 0
        
        for i, (gt, pred) in enumerate(zip(gt_data, pred_data)):
            gt_has_label = any(b["label"] == label or b["label"] == "both" for b in gt["ann_translated_bboxes"])
            
            parsed = pred_parsed[i]
            if parsed is None:
                pred_has_label = False
            else:
                pred_has_label = any(b.get("label") == label or b.get("label") == "both" for b in parsed)
            
            if gt_has_label and pred_has_label:
                tp += 1
            elif gt_has_label and not pred_has_label:
                fn += 1
            elif not gt_has_label and pred_has_label:
                fp += 1
            else:
                tn += 1
        
        total = tp + fp + fn + tn
        accuracy = (tp + tn) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\n  --- {label} ---")
        print(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        
        results[label] = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
    
    return results


def metric3_bbox_level(gt_data, pred_data, pred_parsed, iou_threshold=0.5):
    """
    Metric 3: bbox-level detection (Precision / Recall via IoU matching)
    Greedy matching: assign pred bboxes to GT bboxes in IoU-descending order
    """
    print("\n" + "=" * 70)
    print(f"Metric 3: bbox-level detection (IoU threshold={iou_threshold})")
    print("  Compute per-bbox Precision / Recall / F1 via IoU matching")
    print("=" * 70)
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    # Per-class stats
    per_label_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    
    for i, (gt, pred) in enumerate(zip(gt_data, pred_data)):
        gt_bboxes = gt["ann_translated_bboxes"]
        parsed = pred_parsed[i]
        
        if parsed is None:
            pred_bboxes = []
        else:
            pred_bboxes = parsed

        matched = match_boxes(gt_bboxes, pred_bboxes, iou_threshold)

        for item in matched["matches"]:
            total_tp += 1
            per_label_stats[item["gt_label"]]["tp"] += 1

        for pi in matched["unmatched_pred"]:
            pb = pred_bboxes[pi]
            total_fp += 1
            per_label_stats[pb.get("label", "unknown")]["fp"] += 1

        for gi in matched["unmatched_gt"]:
            gb = gt_bboxes[gi]
            total_fn += 1
            per_label_stats[gb.get("label", "unknown")]["fn"] += 1
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n  Overall:")
    print(f"  TP={total_tp}, FP={total_fp}, FN={total_fn}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    
    for label in sorted(per_label_stats.keys()):
        s = per_label_stats[label]
        p = s["tp"] / (s["tp"] + s["fp"]) if (s["tp"] + s["fp"]) > 0 else 0
        r = s["tp"] / (s["tp"] + s["fn"]) if (s["tp"] + s["fn"]) > 0 else 0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0
        print(f"\n  --- {label} ---")
        print(f"  TP={s['tp']}, FP={s['fp']}, FN={s['fn']}")
        print(f"  Precision: {p:.4f}")
        print(f"  Recall:    {r:.4f}")
        print(f"  F1 Score:  {f:.4f}")
    
    return {"precision": precision, "recall": recall, "f1": f1, "per_label": dict(per_label_stats)}


def metric3_multi_iou(gt_data, pred_data, pred_parsed):
    """Bbox-level detection across multiple IoU thresholds."""
    print("\n" + "=" * 70)
    print("Metric 3b: bbox-level detection across multiple IoU thresholds")
    print("=" * 70)
    
    thresholds = [0.1, 0.2, 0.25, 0.3, 0.4, 0.5]
    results = {}
    
    for thr in thresholds:
        total_tp = 0
        total_fp = 0
        total_fn = 0
        per_label_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
        
        for i, (gt, pred) in enumerate(zip(gt_data, pred_data)):
            gt_bboxes = gt["ann_translated_bboxes"]
            parsed = pred_parsed[i]
            pred_bboxes = parsed if parsed is not None else []

            matched = match_boxes(gt_bboxes, pred_bboxes, thr)

            for item in matched["matches"]:
                total_tp += 1
                per_label_stats[item["gt_label"]]["tp"] += 1

            for pi in matched["unmatched_pred"]:
                pb = pred_bboxes[pi]
                total_fp += 1
                per_label_stats[pb.get("label", "unknown")]["fp"] += 1

            for gi in matched["unmatched_gt"]:
                gb = gt_bboxes[gi]
                total_fn += 1
                per_label_stats[gb.get("label", "unknown")]["fn"] += 1
        
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        label_bbox = {}
        for label in ["artifact", "misalignment"]:
            s = per_label_stats[label]
            p = s["tp"] / (s["tp"] + s["fp"]) if (s["tp"] + s["fp"]) > 0 else 0
            r = s["tp"] / (s["tp"] + s["fn"]) if (s["tp"] + s["fn"]) > 0 else 0
            lf = 2 * p * r / (p + r) if (p + r) > 0 else 0
            label_bbox[label] = {"precision": p, "recall": r, "f1": lf}
        
        results[thr] = {"precision": precision, "recall": recall, "f1": f1, "per_label": label_bbox}
        print(f"\n  IoU >= {thr}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
        for label in ["artifact", "misalignment"]:
            lb = label_bbox[label]
            print(f"    * {label}: P={lb['precision']:.4f}, R={lb['recall']:.4f}, F1={lb['f1']:.4f}")
    
    return results


def metric4_count_based(gt_data, pred_data, pred_parsed):
    """
    Metric 4: count-based stats (Count-Based Metrics)
    Compare GT and prediction bbox counts per image
    """
    print("\n" + "=" * 70)
    print("Metric 4: count-based stats (Count-Based Metrics)")
    print("=" * 70)
    
    gt_counts = []
    pred_counts = []
    abs_diffs = []
    
    for i, (gt, pred) in enumerate(zip(gt_data, pred_data)):
        gc = len(gt["ann_translated_bboxes"])
        parsed = pred_parsed[i]
        pc = len(parsed) if parsed is not None else 0
        
        gt_counts.append(gc)
        pred_counts.append(pc)
        abs_diffs.append(abs(gc - pc))
    
    gt_counts = np.array(gt_counts)
    pred_counts = np.array(pred_counts)
    abs_diffs = np.array(abs_diffs)
    
    # Compute MAE
    mae = np.mean(abs_diffs)
    # Exact-match rate
    exact_match = np.mean(gt_counts == pred_counts)
    # Correlation coefficient
    if np.std(gt_counts) > 0 and np.std(pred_counts) > 0:
        corr = np.corrcoef(gt_counts, pred_counts)[0, 1]
    else:
        corr = 0.0
    
    print(f"\n  GT bbox total:    {int(np.sum(gt_counts))}")
    print(f"  Pred bbox total:  {int(np.sum(pred_counts))}")
    print(f"  GT meaneach:     {np.mean(gt_counts):.2f}")
    print(f"  Pred meaneach:   {np.mean(pred_counts):.2f}")
    print(f"  MAE (mean absolute error): {mae:.4f}")
    print(f"  exact-match rate: {exact_match:.4f}")
    print(f"  correlation:      {corr:.4f}")
    
    # distribution
    print(f"\n  GT bbox countdistribution:")
    gt_counter = Counter(gt_counts)
    for k in sorted(gt_counter.keys()):
        print(f"    {int(k)} bbox(es): {gt_counter[k]} images")
    
    print(f"\n  Pred bbox countdistribution:")
    pred_counter = Counter(pred_counts)
    for k in sorted(pred_counter.keys()):
        print(f"    {int(k)} bbox(es): {pred_counter[k]} images")
    
    return {"mae": mae, "exact_match": exact_match, "correlation": corr}


def metric5_response_quality(gt_data, pred_data, pred_parsed):
    """
    Metric 5: response quality stats (Response Quality)
    - parse success rate
    - fractionof degenerate outputs (entirely repeated characters)
    """
    print("\n" + "=" * 70)
    print("Metric 5: response quality stats (Response Quality)")
    print("=" * 70)
    
    total = len(pred_data)
    parse_success = 0
    parse_fail = 0
    degenerate = 0
    
    for i, pred in enumerate(pred_data):
        resp = pred.get("response", "")
        
        # detect degenerate outputs (long runsof repeated characters)
        if len(resp) > 100:
            # check whether a single character is heavily repeated
            char_counter = Counter(resp)
            most_common_char, most_common_count = char_counter.most_common(1)[0]
            if most_common_count / len(resp) > 0.5 and most_common_char in "0123456789":
                degenerate += 1
                continue
            # check for unicode replacement characters
            if resp.count('\ufffd') > len(resp) * 0.3:
                degenerate += 1
                continue
        
        parsed = pred_parsed[i]
        if parsed is not None:
            parse_success += 1
        else:
            parse_fail += 1
    
    print(f"\n  total samples:    {total}")
    print(f"  parsesucceeded:     {parse_success} ({parse_success/total*100:.1f}%)")
    print(f"  parsefailed:     {parse_fail} ({parse_fail/total*100:.1f}%)")
    print(f"  degenerateoutput:     {degenerate} ({degenerate/total*100:.1f}%)")
    
    return {
        "total": total,
        "parse_success": parse_success,
        "parse_fail": parse_fail,
        "degenerate": degenerate,
        "success_rate": parse_success / total if total > 0 else 0
    }


def metric6_desc_semantic(
    gt_data,
    pred_data,
    pred_parsed,
    similarity_fn,
    metric_name,
    iou_threshold=0.5,
    sim_thresholds=(0.3, 0.5, 0.7),
):
    """
    Metric 6: description semantic consistency
    Compute desc similarity only on boxes that match in both IoU and label, avoiding contamination from localization errors.
    Also provide a detection-aware variant that penalizes missed and spurious detections.
    """
    print("\n" + "=" * 70)
    print(f"Metric 6: box-description semantic consistency [{metric_name}] (IoU threshold={iou_threshold})")
    print("  Compute desc similarity only on matched GT/Pred boxes; report a detection-penalized variant as well")
    print("=" * 70)

    matched_scores = []
    detection_aware_scores = []
    per_label_scores = defaultdict(list)
    threshold_hits = {thr: 0 for thr in sim_thresholds}
    case_records = []
    matched_pair_count = 0
    missing_pred_desc = 0
    missing_gt_desc = 0
    unmatched_gt_total = 0
    unmatched_pred_total = 0

    for gt, pred, parsed in zip(gt_data, pred_data, pred_parsed):
        gt_bboxes = gt["ann_translated_bboxes"]
        pred_bboxes = parsed if parsed is not None else []
        matched = match_boxes(gt_bboxes, pred_bboxes, iou_threshold)

        unmatched_gt_total += len(matched["unmatched_gt"])
        unmatched_pred_total += len(matched["unmatched_pred"])

        image_score_sum = 0.0
        image_norm = max(len(gt_bboxes), len(pred_bboxes), 1)

        for item in matched["matches"]:
            gt_box = gt_bboxes[item["gt_idx"]]
            pred_box = pred_bboxes[item["pred_idx"]]
            gt_desc = gt_box.get("description", "") or gt_box.get("desc", "")
            pred_desc = pred_box.get("description", "") or pred_box.get("desc", "")

            if not gt_desc:
                missing_gt_desc += 1
            if not pred_desc:
                missing_pred_desc += 1

            sim = similarity_fn(gt_desc, pred_desc)
            matched_scores.append(sim)
            per_label_scores[item["gt_label"]].append(sim)
            image_score_sum += sim
            matched_pair_count += 1

            for thr in sim_thresholds:
                if sim >= thr:
                    threshold_hits[thr] += 1

            case_records.append(
                {
                    "filepath": gt["filepath"],
                    "label": item["gt_label"],
                    "iou": item["iou"],
                    "desc_similarity": sim,
                    "gt_desc": gt_desc,
                    "pred_desc": pred_desc,
                }
            )

        detection_aware_scores.append(image_score_sum / image_norm)

    matched_mean = float(np.mean(matched_scores)) if matched_scores else 0.0
    matched_median = float(median(matched_scores)) if matched_scores else 0.0
    detection_aware_mean = float(np.mean(detection_aware_scores)) if detection_aware_scores else 0.0

    print(f"\n  matchboxcount:              {matched_pair_count}")
    print(f"  not yet match GT count:          {unmatched_gt_total}")
    print(f"  not yet match Pred count:        {unmatched_pred_total}")
    print(f"  missing GT descriptions:    {missing_gt_desc}")
    print(f"  missing pred descriptions:  {missing_pred_desc}")
    print(f"  Matched Mean Token-F1:   {matched_mean:.4f}")
    print(f"  Matched Median Token-F1: {matched_median:.4f}")
    print(f"  Detection-Aware Mean:    {detection_aware_mean:.4f}")

    per_label_results = {}
    for label in ["artifact", "misalignment"]:
        scores = per_label_scores[label]
        mean_score = float(np.mean(scores)) if scores else 0.0
        median_score = float(median(scores)) if scores else 0.0
        per_label_results[label] = {
            "matched_count": len(scores),
            "mean_similarity": mean_score,
            "median_similarity": median_score,
        }
        print(f"\n  --- {label} ---")
        print(f"  Matched Count:   {len(scores)}")
        print(f"  Mean Token-F1:   {mean_score:.4f}")
        print(f"  Median Token-F1: {median_score:.4f}")

    threshold_acc = {}
    for thr in sim_thresholds:
        acc = threshold_hits[thr] / matched_pair_count if matched_pair_count > 0 else 0.0
        threshold_acc[str(thr)] = acc
        print(f"  Acc@desc>={thr}:         {acc:.4f}")

    case_records.sort(key=lambda x: (x["desc_similarity"], x["iou"]))

    return {
        "backend": metric_name,
        "metric_name": metric_name,
        "iou_threshold": iou_threshold,
        "matched_count": matched_pair_count,
        "unmatched_gt_count": unmatched_gt_total,
        "unmatched_pred_count": unmatched_pred_total,
        "missing_gt_desc_count": missing_gt_desc,
        "missing_pred_desc_count": missing_pred_desc,
        "matched_mean_similarity": matched_mean,
        "matched_median_similarity": matched_median,
        "detection_aware_mean_similarity": detection_aware_mean,
        "acc_by_similarity_threshold": threshold_acc,
        "per_label": per_label_results,
        "worst_cases": case_records[:20],
    }


def export_desc_worst_cases(output_dir, results):
    for key, suffix in (
        ("desc_semantic_token_f1", "desc_semantic_token_f1_worst_cases.jsonl"),
        ("desc_semantic_embedding", "desc_semantic_embedding_worst_cases.jsonl"),
    ):
        if key not in results:
            continue
        out_path = os.path.join(output_dir, suffix)
        with open(out_path, "w", encoding="utf-8") as f:
            for item in results[key].get("worst_cases", []):
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Desc worst cases exported to: {out_path}")


import argparse

def main():
    parser = argparse.ArgumentParser(description="SDG Evaluation Script")
    parser.add_argument("--gt", required=True,
                        help="GT JSONL file path (e.g. data/sample_200/annotations.jsonl)")
    parser.add_argument("--pred", required=True,
                        help="Predictions JSONL file path produced by sdg_detector/inference/eval_qwen.py")
    parser.add_argument("--desc_iou_threshold", type=float, default=0.5, help="IoU threshold for desc semantic evaluation")
    parser.add_argument("--embedding_model", default="Qwen/Qwen3-Embedding-0.6B", help="Embedding model name or local path")
    args = parser.parse_args()
    
    gt_path = args.gt
    pred_path = args.pred
    
    print("loaddata...")
    gt_data = load_jsonl(gt_path)
    pred_data = load_jsonl(pred_path)
    
    print(f"GT samples:   {len(gt_data)}")
    print(f"Pred samples: {len(pred_data)}")
    
    assert len(gt_data) == len(pred_data), f"GT and Pred sample count mismatch: {len(gt_data)} vs {len(pred_data)}"
    
    # Validate that filepaths align
    for i, (g, p) in enumerate(zip(gt_data, pred_data)):
        assert g["filepath"] == p["filepath"], f"line {i} filepath mismatch: {g['filepath']} vs {p['filepath']}"
    
    print("All filepaths align; parsing predictions...")
    
    # Parse all predictions aheadof time
    pred_parsed = []
    for p in pred_data:
        pred_parsed.append(parse_prediction_bboxes(p.get("response", "")))
    
    # ==========================================
    # runeachevalmetrics
    # ==========================================
    
    all_results = {}
    
    # Run metric 5 first to gauge data quality
    all_results["response_quality"] = metric5_response_quality(gt_data, pred_data, pred_parsed)
    
    # Metric 1: image-level classification (most recommended)
    all_results["image_level"] = metric1_image_level(gt_data, pred_data, pred_parsed)
    
    # Metric 2: per-class detection
    all_results["label_level"] = metric2_label_level(gt_data, pred_data, pred_parsed)
    
    # Metric 3: bbox-level detection
    all_results["bbox_level_iou50"] = metric3_bbox_level(gt_data, pred_data, pred_parsed, iou_threshold=0.5)
    
    # Metric 3b: multiple IoU thresholds
    all_results["bbox_multi_iou"] = metric3_multi_iou(gt_data, pred_data, pred_parsed)
    
    # Metric 4: count-based stats
    all_results["count_based"] = metric4_count_based(gt_data, pred_data, pred_parsed)

    # Metric 6a: box-description semantic consistency (token F1)
    all_results["desc_semantic_token_f1"] = metric6_desc_semantic(
        gt_data,
        pred_data,
        pred_parsed,
        similarity_fn=compute_desc_similarity,
        metric_name="token_f1",
        iou_threshold=args.desc_iou_threshold,
    )

    # Metric 6b: box-description semantic consistency (embedding cosine)
    print(f"\nload embedding model: {args.embedding_model}")
    embedding_scorer = TransformerEmbeddingScorer(args.embedding_model)
    all_results["desc_semantic_embedding"] = metric6_desc_semantic(
        gt_data,
        pred_data,
        pred_parsed,
        similarity_fn=embedding_scorer.similarity,
        metric_name="embedding_cosine",
        iou_threshold=args.desc_iou_threshold,
    )
    
    # ==========================================
    # Summary
    # ==========================================
    print("\n" + "=" * 90)
    print("Summary")
    print("=" * 90)
    
    rq = all_results["response_quality"]
    il = all_results["image_level"]
    ll = all_results["label_level"]
    bl = all_results["bbox_level_iou50"]
    cb = all_results["count_based"]
    mi = all_results["bbox_multi_iou"]
    ds_token = all_results["desc_semantic_token_f1"]
    ds_embed = all_results["desc_semantic_embedding"]
    
    header = f"{'metrics':<45} {'Precision':>10} {'Recall':>10} {'F1':>10}"
    sep = "-" * 80
    
    print(f"\n  response parse success rate: {rq['success_rate']:.4f}  |  degenerate-output fraction: {rq['degenerate']/rq['total']:.4f}")
    print()
    print(header)
    print(sep)
    print(f"{'image-level (overall)':<45} {il['precision']:>10.4f} {il['recall']:>10.4f} {il['f1']:>10.4f}")
    for label in ["artifact", "misalignment"]:
        lr = ll[label]
        print(f"{'image-level (' + label + ')':<45} {lr['precision']:>10.4f} {lr['recall']:>10.4f} {lr['f1']:>10.4f}")
    print(sep)
    for thr in [0.1, 0.2, 0.25, 0.3, 0.4, 0.5]:
        m = mi[thr]
        print(f"{'Bbox-level overall (IoU>=' + str(thr) + ')':<45} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f}")
        for label in ["artifact", "misalignment"]:
            lb = m["per_label"][label]
            print(f"{'  * ' + label + ' (IoU>=' + str(thr) + ')':<45} {lb['precision']:>10.4f} {lb['recall']:>10.4f} {lb['f1']:>10.4f}")
    print(sep)
    print(f"  Count MAE: {cb['mae']:.4f}  |  exact-match rate: {cb['exact_match']:.4f}  |  correlation: {cb['correlation']:.4f}")
    print(f"  Desc Token Mean: {ds_token['matched_mean_similarity']:.4f}  |  Token Detection-Aware: {ds_token['detection_aware_mean_similarity']:.4f}")
    print(f"  Desc Token Acc@0.5: {ds_token['acc_by_similarity_threshold']['0.5']:.4f}")
    print(f"  Desc Embed Mean: {ds_embed['matched_mean_similarity']:.4f}  |  Embed Detection-Aware: {ds_embed['detection_aware_mean_similarity']:.4f}")
    print(f"  Desc Embed Acc@0.5: {ds_embed['acc_by_similarity_threshold']['0.5']:.4f}")
    print(sep)
    
    # Save results to JSON (in the same directory as the prediction file)
    import os
    output_path = os.path.join(os.path.dirname(os.path.abspath(pred_path)), "eval_results.json")
    
    # convert numpy type
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    serializable = json.loads(json.dumps(all_results, default=convert))
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)

    print(f"Detailed results saved to: {output_path}")
    export_desc_worst_cases(os.path.dirname(os.path.abspath(pred_path)), serializable)


if __name__ == "__main__":
    main()
