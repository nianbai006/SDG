#!/usr/bin/env python3
"""
Compute box-level Label F1 and Desc Cosine @ IoU>=0.1.

Label-agnostic matching: match by IoU only (no label constraint),
then evaluate label correctness and desc similarity on matched pairs.

This is different from evalcode's bbox_multi_iou which requires IoU+label match.

Usage:
    python compute_label_desc_metrics.py --eval_dir <dir_with_eval_results>
    python compute_label_desc_metrics.py --eval_dirs dir1 dir2 dir3 ...
    python compute_label_desc_metrics.py --all  # scan all dirs under EVAL_ROOT
"""
import json, os, re, sys, argparse
import numpy as np
from collections import defaultdict

GT_FILE = "${SDG_DATA}/sample_200/annotations.jsonl"
EVAL_ROOT = "${SDG_HOME}/experiments/sdg_eval"


def compute_iou(box1, box2, eps=1e-9):
    x0 = max(box1[0], box2[0]); y0 = max(box1[1], box2[1])
    x1 = min(box1[2], box2[2]); y1 = min(box1[3], box2[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    a1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    a2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    return inter / (a1 + a2 - inter + eps)


def _norm_label(l):
    l = l.lower().strip()
    if "misalign" in l: return "misalignment"
    if "artifact" in l: return "artifact"
    return l


def match_boxes_no_label(gt_bboxes, pred_bboxes, iou_threshold):
    """Match by IoU only, no label constraint. Greedy best-IoU matching."""
    pairs = []
    for pi, pb in enumerate(pred_bboxes):
        if "box_2d" not in pb: continue
        for gi, gb in enumerate(gt_bboxes):
            if "box_2d" not in gb: continue
            iou = compute_iou(pb["box_2d"], gb["box_2d"])
            if iou >= iou_threshold:
                pairs.append((iou, pi, gi))
    pairs.sort(reverse=True)
    matched_gt, matched_pred, matches = set(), set(), []
    for iou_val, pi, gi in pairs:
        if pi in matched_pred or gi in matched_gt:
            continue
        matched_pred.add(pi); matched_gt.add(gi)
        matches.append({"pred_idx": pi, "gt_idx": gi, "iou": iou_val})
    return matches


def parse_answer(text):
    """Extract bbox list from model output."""
    if not isinstance(text, str): return None
    m = re.search(r'<answer>\s*(.*?)\s*</answer>', text, re.DOTALL)
    if not m: m = re.search(r'<answer>\s*(.*)', text, re.DOTALL)
    if not m: return None
    raw = m.group(1).strip()
    if not raw: return []
    raw = re.sub(r',\s*]', ']', raw)
    raw = re.sub(r',\s*}', '}', raw)
    try:
        result = json.loads(raw)
        if isinstance(result, list): return result
    except: pass
    try:
        import ast
        result = ast.literal_eval(raw)
        if isinstance(result, list): return result
    except: pass
    return None


def load_embedding_model():
    """Load embedding model for desc cosine similarity."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    EMB_MODEL = "Qwen/Qwen3-Embedding-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(EMB_MODEL, trust_remote_code=True)
    model = AutoModel.from_pretrained(EMB_MODEL, trust_remote_code=True, torch_dtype=torch.float16).cuda().eval()
    return tokenizer, model


def encode_texts(texts, tokenizer, model):
    import torch
    if not texts: return np.array([])
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).expand(out.size()).float()
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    return pooled.cpu().numpy()


def cosine_sim(a, b):
    a_n = a / (np.linalg.norm(a) + 1e-9)
    b_n = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a_n, b_n))


def compute_for_experiment(exp_dir, gt_all, emb_tokenizer, emb_model, iou_threshold=0.1):
    """Compute label F1 and desc cosine for one experiment directory."""
    pred_file = os.path.join(exp_dir, "predictions_aligned.jsonl")
    gt_file = os.path.join(exp_dir, "GT_aligned.jsonl")

    if not os.path.exists(pred_file):
        pred_file = os.path.join(exp_dir, "predictions.jsonl")
    if not os.path.exists(pred_file):
        return None

    pred_lines = [json.loads(l) for l in open(pred_file) if l.strip()]

    if os.path.exists(gt_file):
        gt_lines = [json.loads(l) for l in open(gt_file) if l.strip()]
    else:
        gt_lines = []
        for p in pred_lines:
            fp = p.get("filepath", "")
            if fp in gt_all:
                gt_lines.append(gt_all[fp])

    if len(gt_lines) != len(pred_lines):
        # Align by filepath
        gt_map = {g["filepath"]: g for g in gt_lines}
        aligned_gt, aligned_pred = [], []
        for p in pred_lines:
            fp = p.get("filepath", "")
            if fp in gt_map:
                aligned_gt.append(gt_map[fp])
                aligned_pred.append(p)
        gt_lines, pred_lines = aligned_gt, aligned_pred

    # Label stats: per-label TP/FP/FN (label-agnostic IoU matching)
    label_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    all_gt_descs, all_pred_descs, all_desc_meta = [], [], []

    for gt, pred in zip(gt_lines, pred_lines):
        gt_bboxes = gt.get("ann_translated_bboxes", [])
        if not isinstance(gt_bboxes, list): gt_bboxes = []
        gt_bboxes = [b for b in gt_bboxes if isinstance(b, dict) and "box_2d" in b]

        response = pred.get("response", "")
        parsed = parse_answer(response)
        pred_bboxes = []
        if parsed:
            for b in parsed:
                if isinstance(b, dict) and "box_2d" in b:
                    pred_bboxes.append(b)

        matches = match_boxes_no_label(gt_bboxes, pred_bboxes, iou_threshold)
        matched_pred = {m["pred_idx"] for m in matches}
        matched_gt = {m["gt_idx"] for m in matches}

        # Label F1: only among IoU-matched pairs (not counting unmatched boxes)
        for m in matches:
            gt_label = _norm_label(gt_bboxes[m["gt_idx"]].get("label", ""))
            pred_label = _norm_label(pred_bboxes[m["pred_idx"]].get("label", ""))
            if gt_label == pred_label:
                label_stats[gt_label]["tp"] += 1
            else:
                label_stats[gt_label]["fn"] += 1
                label_stats[pred_label]["fp"] += 1

            # Desc similarity
            gt_desc = gt_bboxes[m["gt_idx"]].get("description", gt_bboxes[m["gt_idx"]].get("desc", ""))
            pred_desc = pred_bboxes[m["pred_idx"]].get("description", pred_bboxes[m["pred_idx"]].get("desc", ""))
            if gt_desc and pred_desc:
                all_gt_descs.append(gt_desc)
                all_pred_descs.append(pred_desc)
                all_desc_meta.append(gt_label)

    # Label F1
    label_f1 = {}
    for lbl in ["artifact", "misalignment"]:
        s = label_stats[lbl]
        p = s["tp"] / (s["tp"] + s["fp"]) if (s["tp"] + s["fp"]) > 0 else 0
        r = s["tp"] / (s["tp"] + s["fn"]) if (s["tp"] + s["fn"]) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        label_f1[lbl] = f1

    # Desc cosine similarity
    desc_sims = {"artifact": [], "misalignment": []}
    if all_gt_descs:
        gt_embs = encode_texts(all_gt_descs, emb_tokenizer, emb_model)
        pred_embs = encode_texts(all_pred_descs, emb_tokenizer, emb_model)
        for i, lbl in enumerate(all_desc_meta):
            sim = cosine_sim(gt_embs[i], pred_embs[i])
            desc_sims[lbl].append(sim)

    desc_cos_mean = {}
    for lbl in ["artifact", "misalignment"]:
        desc_cos_mean[lbl] = float(np.mean(desc_sims[lbl])) if desc_sims[lbl] else 0.0

    return {
        "art_label_f1": label_f1["artifact"],
        "mis_label_f1": label_f1["misalignment"],
        "art_desc_cos": desc_cos_mean["artifact"],
        "mis_desc_cos": desc_cos_mean["misalignment"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dirs", nargs="+", help="Specific eval directories")
    parser.add_argument("--all", action="store_true", help="Scan all dirs under EVAL_ROOT")
    parser.add_argument("--iou_threshold", type=float, default=0.1)
    args = parser.parse_args()

    gt_all = {json.loads(l)["filepath"]: json.loads(l) for l in open(GT_FILE) if l.strip()}

    print("Loading embedding model...", file=sys.stderr)
    emb_tokenizer, emb_model = load_embedding_model()

    if args.all:
        dirs = sorted([
            os.path.join(EVAL_ROOT, d) for d in os.listdir(EVAL_ROOT)
            if os.path.isdir(os.path.join(EVAL_ROOT, d))
        ])
    elif args.eval_dirs:
        dirs = args.eval_dirs
    else:
        print("Specify --eval_dirs or --all", file=sys.stderr)
        return

    print("experiment,art_label_f1,mis_label_f1,art_desc_cos,mis_desc_cos")
    for exp_dir in dirs:
        exp_name = os.path.basename(exp_dir)
        result = compute_for_experiment(exp_dir, gt_all, emb_tokenizer, emb_model, args.iou_threshold)
        if result:
            print(f"{exp_name},{result['art_label_f1']:.4f},{result['mis_label_f1']:.4f},{result['art_desc_cos']:.4f},{result['mis_desc_cos']:.4f}")
            print(f"  Done: {exp_name}", file=sys.stderr)
        else:
            print(f"  SKIP: {exp_name}", file=sys.stderr)


if __name__ == "__main__":
    main()
