"""
GRPO Plugin v3 for SDG experiments.

Single combined reward: SDG_combined_v3
  - Format as gate (fail => -1.0)
  - Hungarian matching (no IoU threshold, label-constrained)
  - R_box: DIoU-based, continuous
  - R_desc: continuous cosine similarity
  - R_imp: continuous importance error
  - total = 0.6*R_box + 0.25*R_desc + 0.15*R_imp (if format pass)
  - Empty handling only in R_box, not duplicated in desc/imp
"""

import json
import re
from collections import Counter
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from swift.rewards import ORM, orms
from swift.utils import get_logger

logger = get_logger()


# ============================================================
# Parsing
# ============================================================

def normalize_completion_text(completion: Any) -> str:
    """Best-effort extraction    of textual completion content.

    GRPO/vLLM can occasionally hand reward functions a non-string payload
    (e.g. None, structured content blocks, or response dicts). Treat any
    unrecognized shape as empty text so the sample gets a format penalty
    instead    of raising a noisy NoneType/regex error.
    """
    if completion is None:
        return ''
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts = []
        for item in completion:
            text = normalize_completion_text(item)
            if text:
                parts.append(text)
        return '\n'.join(parts)
    if isinstance(completion, dict):
        if isinstance(completion.get('text'), str):
            return completion['text']
        if 'content' in completion:
            return normalize_completion_text(completion['content'])
        if 'message' in completion:
            return normalize_completion_text(completion['message'])
        if 'choices' in completion:
            return normalize_completion_text(completion['choices'])
    return ''

def parse_answer_bboxes(response: str) -> Optional[List[Dict]]:
    response = normalize_completion_text(response)
    if not response or not isinstance(response, str):
        return None
    m = re.search(r'<answer>\s*(.*?)\s*</answer>', response, re.DOTALL)
    if not m:
        return None
    text = m.group(1).strip()
    code_m = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
    if code_m:
        text = code_m.group(1).strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list):
        return None
    return [item for item in data if isinstance(item, dict) and 'box_2d' in item and 'label' in item]


def extract_coords(bboxes):
    coords = []
    if bboxes is None:
        return coords
    if isinstance(bboxes, str):
        try:
            bboxes = json.loads(bboxes)
        except:
            return coords
    if not isinstance(bboxes, list):
        return coords
    for b in bboxes:
        if isinstance(b, dict) and 'box_2d' in b:
            coords.append(b['box_2d'])
        elif isinstance(b, list) and len(b) == 4:
            coords.append(b)
    return coords


def gemini_to_qwen(coords_list):
    return [[c[1], c[0], c[3], c[2]] for c in coords_list if isinstance(c, list) and len(c) == 4]


# ============================================================
# Metrics: DIoU, cosine sim
# ============================================================

def _diou(box1, box2, eps=1e-9):
    ix1 = max(box1[0], box2[0]); iy1 = max(box1[1], box2[1])
    ix2 = min(box1[2], box2[2]); iy2 = min(box1[3], box2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter
    iou = inter / union if union > eps else 0.0
    cx1 = (box1[0] + box1[2]) / 2; cy1 = (box1[1] + box1[3]) / 2
    cx2 = (box2[0] + box2[2]) / 2; cy2 = (box2[1] + box2[3]) / 2
    d2 = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2
    ex1 = min(box1[0], box2[0]); ey1 = min(box1[1], box2[1])
    ex2 = max(box1[2], box2[2]); ey2 = max(box1[3], box2[3])
    c2 = (ex2 - ex1) ** 2 + (ey2 - ey1) ** 2
    return iou - d2 / c2 if c2 > eps else iou


class _EmbeddingScorer:
    _instance = None

    @classmethod
    def get(cls, model_name="Qwen/Qwen3-Embedding-0.6B"):
        if cls._instance is None:
            import torch
            from transformers import AutoModel, AutoTokenizer
            logger.info(f"Loading embedding model: {model_name}")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModel.from_pretrained(
                model_name, trust_remote_code=True,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            ).to(device)
            model.eval()
            cls._instance = {
                "tokenizer": tokenizer, "model": model,
                "device": device, "torch": torch, "cache": {},
            }
        return cls._instance

    @classmethod
    def encode(cls, texts):
        s = cls.get()
        torch = s["torch"]
        missing = [t for t in texts if t not in s["cache"]]
        if not missing:
            return
        with torch.no_grad():
            for start in range(0, len(missing), 32):
                batch = missing[start:start + 32]
                encoded = s["tokenizer"](batch, padding=True, truncation=True, max_length=256, return_tensors="pt")
                encoded = {k: v.to(s["device"]) for k, v in encoded.items()}
                outputs = s["model"](**encoded)
                lhs = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
                mask = encoded["attention_mask"].unsqueeze(-1).expand(lhs.size()).float()
                emb = (lhs * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                for text, vec in zip(batch, emb.detach().cpu().numpy()):
                    s["cache"][text] = vec

    @classmethod
    def cosine_sim(cls, text_a, text_b):
        if not text_a or not text_b:
            return 0.0
        cls.encode([text_a, text_b])
        c = cls.get()["cache"]
        return float(np.dot(c[text_a], c[text_b]))


# ============================================================
# Hungarian matching (label-constrained, no IoU threshold)
# ============================================================

def hungarian_match(gt_bboxes, pred_bboxes):
    """Hungarian matching constrained by label. No IoU threshold.

    Returns: list    of (gt_idx, pred_idx, diou_score)
    """
    if not gt_bboxes or not pred_bboxes:
        return [], len(pred_bboxes), len(gt_bboxes)

    n_gt = len(gt_bboxes)
    n_pred = len(pred_bboxes)

    # Build cost matrix: -DIoU (minimize cost = maximize DIoU)
    # Set cost=2.0 (high) for label mismatch to prevent cross-label matching
    cost = np.full((n_gt, n_pred), 2.0)
    diou_matrix = np.zeros((n_gt, n_pred))

    for gi, gb in enumerate(gt_bboxes):
        for pi, pb in enumerate(pred_bboxes):
            g_label = gb.get('label', '').lower()
            p_label = pb.get('label', '').lower()
            # Allow matching only if labels match (or 'both')
            labels_ok = (g_label == p_label or
                         p_label == 'both' or g_label == 'both')
            if labels_ok and 'box_2d' in gb and 'box_2d' in pb:
                d = _diou(pb['box_2d'], gb['box_2d'])
                diou_matrix[gi, pi] = d
                cost[gi, pi] = -d  # negative because we minimize

    # Solve
    row_ind, col_ind = linear_sum_assignment(cost)

    matches = []
    matched_g = set()
    matched_p = set()
    for gi, pi in zip(row_ind, col_ind):
        if cost[gi, pi] < 1.5:  # Only accept label-compatible matches (cost < 2.0)
            matches.append((gi, pi, diou_matrix[gi, pi]))
            matched_g.add(gi)
            matched_p.add(pi)

    unmatched_pred = n_pred - len(matched_p)
    unmatched_gt = n_gt - len(matched_g)
    return matches, unmatched_pred, unmatched_gt


# ============================================================
# Format gate
# ============================================================

def check_format(completion):
    """Returns True if format is valid (think + answer + valid bboxes)."""
    completion = normalize_completion_text(completion)
    if not re.search(r'<think>', completion):
        return False
    parsed = parse_answer_bboxes(completion)
    if parsed is None:
        return False
    for item in parsed:
        box = item.get('box_2d')
        if not isinstance(box, list) or len(box) != 4:
            return False
        x0, y0, x1, y1 = box
        if not all(isinstance(v, (int, float)) for v in [x0, y0, x1, y1]):
            return False
        if not (x0 < x1 and y0 < y1):
            return False
        if 'label' not in item:
            return False
        if 'description' not in item and 'desc' not in item:
            return False
    return True


# ============================================================
# Build GT bboxes from GRPO data fields
# ============================================================

def build_gt_bboxes(gt_mis_raw, gt_art_raw):
    """Build list    of GT bbox dicts in Qwen format [x0,y0,x1,y1]."""
    gt_bboxes = []
    for items, label in [(gt_mis_raw, 'misalignment'), (gt_art_raw, 'artifact')]:
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, dict) and 'box_2d' in item:
                coords = item['box_2d']
                gt_bboxes.append({
                    'box_2d': [coords[1], coords[0], coords[3], coords[2]],
                    'label': label,
                    'description': item.get('description', '') or item.get('desc', ''),
                    'importance': item.get('importance', 50),
                })
            elif isinstance(item, list) and len(item) == 4:
                gt_bboxes.append({
                    'box_2d': [item[1], item[0], item[3], item[2]],
                    'label': label,
                    'description': '',
                    'importance': 50,
                })
    return gt_bboxes


def normalize_batch_field(values, batch_size):
    if values is None:
        return [[] for _ in range(batch_size)]
    if not isinstance(values, list):
        return [values for _ in range(batch_size)]
    if len(values) < batch_size:
        return values + ([[]] * (batch_size - len(values)))
    return values[:batch_size]


# ============================================================
# Combined reward v3
# ============================================================

class CombinedV3ORM(ORM):
    """Combined reward with format gate, Hungarian matching, continuous scores.

    if format_fail: total = -1.0
    else: total = 0.6*R_box + 0.25*R_desc + 0.15*R_imp

    R_box: DIoU-based with miss/false_alarm penalties
    R_desc: continuous cosine similarity, clip((cos-0.5)/0.4, 0, 1)
    R_imp: continuous importance error, clip(1 - |pred-gt|/50, 0, 1)

    Empty handling only in R_box. desc/imp = 0 when empty.
    """

    UNMATCHED_PENALTY = -0.5
    MISS_PENALTY = -0.8       # GT has + Pred empty
    FALSE_ALARM_PENALTY = -0.3  # GT empty + Pred has
    EMPTY_MATCH = 0.3         # Both empty

    W_BOX = 0.6
    W_DESC = 0.25
    W_IMP = 0.15

    def __init__(self):
        self._call_count = 0
        self._stats = {
            'total': [], 'r_box': [], 'r_desc': [], 'r_imp': [],
            'format_fail': 0, 'gt_empty_pred_empty': 0,
            'gt_has_pred_empty': 0, 'gt_empty_pred_has': 0,
            'gt_has_pred_has': 0, 'matched': 0, 'unmatched_pred': 0,
            'unmatched_gt': 0,
        }
        self._log_interval = 50  # log every 50 calls

    def _log_stats(self):
        s = self._stats
        n = max(len(s['total']), 1)
        logger.info(
            f"[v3 reward] calls={self._call_count} | "
            f"total={np.mean(s['total']):.3f} box={np.mean(s['r_box']):.3f} "
            f"desc={np.mean(s['r_desc']):.3f} imp={np.mean(s['r_imp']):.3f} | "
            f"fmt_fail={s['format_fail']}/{self._call_count} "
            f"ee={s['gt_empty_pred_empty']} he={s['gt_has_pred_empty']} "
            f"eh={s['gt_empty_pred_has']} hh={s['gt_has_pred_has']} | "
            f"matched={s['matched']} up={s['unmatched_pred']} ug={s['unmatched_gt']}"
        )
        # Reset running stats
        self._stats = {
            'total': [], 'r_box': [], 'r_desc': [], 'r_imp': [],
            'format_fail': 0, 'gt_empty_pred_empty': 0,
            'gt_has_pred_empty': 0, 'gt_empty_pred_has': 0,
            'gt_has_pred_has': 0, 'matched': 0, 'unmatched_pred': 0,
            'unmatched_gt': 0,
        }

    def __call__(self, completions,
                 gt_misalignment_bboxes=None, gt_artifact_bboxes=None,
                 **kwargs) -> List[float]:
        if completions is None:
            return []
        if not isinstance(completions, list):
            completions = [completions]

        rewards = []
        n = len(completions)
        gt_mis_list = normalize_batch_field(gt_misalignment_bboxes, n)
        gt_art_list = normalize_batch_field(gt_artifact_bboxes, n)

        for completion, gt_mis_raw, gt_art_raw in zip(completions, gt_mis_list, gt_art_list):
            try:
                self._call_count += 1
                completion_text = normalize_completion_text(completion)

                # Format gate
                if not check_format(completion_text):
                    rewards.append(-1.0)
                    self._stats['format_fail'] += 1
                    self._stats['total'].append(-1.0)
                    self._stats['r_box'].append(-1.0)
                    self._stats['r_desc'].append(0.0)
                    self._stats['r_imp'].append(0.0)
                    continue

                parsed = parse_answer_bboxes(completion_text)
                pred_bboxes = parsed if parsed is not None else []
                gt_bboxes = build_gt_bboxes(gt_mis_raw, gt_art_raw)

                gt_has = len(gt_bboxes) > 0
                pred_has = len(pred_bboxes) > 0

                # === R_box ===
                if not gt_has and not pred_has:
                    r_box = self.EMPTY_MATCH
                    r_desc = 0.0
                    r_imp = 0.0
                    self._stats['gt_empty_pred_empty'] += 1
                elif gt_has and not pred_has:
                    r_box = self.MISS_PENALTY
                    r_desc = 0.0
                    r_imp = 0.0
                    self._stats['gt_has_pred_empty'] += 1
                elif not gt_has and pred_has:
                    r_box = self.FALSE_ALARM_PENALTY
                    r_desc = 0.0
                    r_imp = 0.0
                    self._stats['gt_empty_pred_has'] += 1
                else:
                    self._stats['gt_has_pred_has'] += 1
                    # Hungarian matching
                    matches, unmatched_pred, unmatched_gt = hungarian_match(gt_bboxes, pred_bboxes)
                    self._stats['matched'] += len(matches)
                    self._stats['unmatched_pred'] += unmatched_pred
                    self._stats['unmatched_gt'] += unmatched_gt

                    # R_box: matched DIoU + unmatched penalty
                    box_sum = sum(diou for _, _, diou in matches)
                    box_sum += (unmatched_pred + unmatched_gt) * self.UNMATCHED_PENALTY
                    norm = max(len(gt_bboxes), len(pred_bboxes), 1)
                    r_box = max(-1.0, min(1.0, box_sum / norm))

                    # Batch encode descriptions
                    all_texts = []
                    for gi, pi, _ in matches:
                        gt_d = gt_bboxes[gi].get('description', '')
                        pred_d = pred_bboxes[pi].get('description', '') or pred_bboxes[pi].get('desc', '')
                        if gt_d:
                            all_texts.append(gt_d)
                        if pred_d:
                            all_texts.append(pred_d)
                    if all_texts:
                        _EmbeddingScorer.encode(all_texts)

                    # R_desc: continuous cosine, clip((cos-0.5)/0.4, 0, 1)
                    desc_sum = 0.0
                    for gi, pi, _ in matches:
                        gt_d = gt_bboxes[gi].get('description', '')
                        pred_d = pred_bboxes[pi].get('description', '') or pred_bboxes[pi].get('desc', '')
                        if gt_d and pred_d:
                            cos = _EmbeddingScorer.cosine_sim(gt_d, pred_d)
                            desc_sum += max(0.0, min(1.0, (cos - 0.5) / 0.4))
                    r_desc = desc_sum / norm

                    # R_imp: continuous, clip(1 - |pred-gt|/50, 0, 1)
                    imp_sum = 0.0
                    for gi, pi, _ in matches:
                        gt_imp = gt_bboxes[gi].get('importance', 50)
                        pred_imp = pred_bboxes[pi].get('importance')
                        if pred_imp is not None and isinstance(pred_imp, (int, float)):
                            imp_sum += max(0.0, min(1.0, 1.0 - abs(float(pred_imp) - float(gt_imp)) / 50.0))
                    r_imp = imp_sum / norm

                total = self.W_BOX * r_box + self.W_DESC * r_desc + self.W_IMP * r_imp
                rewards.append(float(total))

                self._stats['total'].append(total)
                self._stats['r_box'].append(r_box)
                self._stats['r_desc'].append(r_desc)
                self._stats['r_imp'].append(r_imp)

            except Exception as e:
                rewards.append(-1.0)
                self._stats['format_fail'] += 1
                self._stats['total'].append(-1.0)
                self._stats['r_box'].append(-1.0)
                self._stats['r_desc'].append(0.0)
                self._stats['r_imp'].append(0.0)
                logger.warning(f'CombinedV3ORM error: {e}')

        if self._call_count % self._log_interval == 0:
            self._log_stats()

        return rewards


orms['SDG_combined_v3'] = CombinedV3ORM
