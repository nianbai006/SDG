"""
Data-loading helpers
- load test-set JSONL
- load ImageDoctor prediction
- datamatch
"""
import json
import os
import ast
from pathlib import Path
from typing import List, Dict, Optional, Tuple


def extract_generator_name(filepath: str) -> str:
    """
    Extract the generator name from the file path (sana / flux2 / zimage / longcat).

    Path format: .../SDG/{generator}/test/test_NNNNNN.png
    Note: expects /<root>/{generator}/{train,test}/<file>.png

    Returns:
        Generator name, e.g. "zimage"; returns an empty string if it cannot be extracted.
    """
    path_parts = Path(filepath).parts
    for i, part in enumerate(path_parts):
        if part == "SDG" and i + 1 < len(path_parts):
            candidate = path_parts[i + 1]
            if candidate != "dataset":
                return candidate
    return ""


def make_unique_basename(filepath: str) -> str:
    """
    Build a generator-suffixed unique basename to avoid filename collisions across generators.

    Example: /path/SDG/zimage/test/test_000010.png -> "test_000010_zimage"
        /other/path/image.png -> "image"  (no suffix appended if extraction fails)
    """
    stem = Path(filepath).stem
    generator = extract_generator_name(filepath)
    if generator:
        return f"{stem}_{generator}"
    return stem


def _parse_bbox_string(bbox_str):
    """parsestringformat  bbox list。"""
    if isinstance(bbox_str, list):
        return bbox_str
    if isinstance(bbox_str, str):
        try:
            return ast.literal_eval(bbox_str)
        except Exception:
            return []
    return []


def load_test_data(
    jsonl_path: str,
    max_samples: Optional[int] = None,
    filepath_prefix_map: Optional[Dict[str, str]] = None,
) -> List[Dict]:
    """
    Load the test-set JSONL file.

    Args:
        jsonl_path: JSONL file path
        max_samples: maximum numberof samples
        filepath_prefix_map: filepath-prefix replacement map, e.g. {"${SDG_HOME}/../pine": "/data"}

    Returns:
        List[Dict]: samples; each contains:
            - filepath: image path
            - caption: textdescription
            - misalignment_bboxes_ann: annotation   of misalignment bbox list
            - artifact_bboxes_ann: annotation   of artifact bbox list
            - ann_response: SDG of full response text
            - ann_translated_bboxes: translated bbox list
    """
    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)

            # path mapping
            filepath = sample.get("filepath", "")
            if filepath_prefix_map:
                for old_prefix, new_prefix in filepath_prefix_map.items():
                    if filepath.startswith(old_prefix):
                        filepath = new_prefix + filepath[len(old_prefix):]
                        break
            sample["filepath"] = filepath

            # parse bbox string
            for key in ["misalignment_bboxes", "artifact_bboxes",
                        "misalignment_bboxes_ann", "artifact_bboxes_ann",
                        "ann_translated_bboxes"]:
                if key in sample and isinstance(sample[key], str):
                    sample[key] = _parse_bbox_string(sample[key])

            data.append(sample)

            if max_samples and len(data) >= max_samples:
                break

    print(f"[data_loader] loaddone {len(data)} recordstestsample from {jsonl_path}")
    return data


def load_imagedoctor_predictions(
    jsonl_path: str,
    max_samples: Optional[int] = None,
    filepath_prefix_map: Optional[Dict[str, str]] = None,
) -> List[Dict]:
    """
    load ImageDoctor predictionfile。

    Returns:
        List[Dict]: each record contains:
            - filename: image path
            - caption: textdescription
            - prediction: full prediction text (includes <think> and <answer>)
            - misalignment_heatmap: heatmap path
            - artifact_heatmap: heatmap path
    """
    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)

            # path mapping
            for key in ["filename", "misalignment_heatmap", "artifact_heatmap"]:
                val = sample.get(key, "")
                if val and filepath_prefix_map:
                    for old_prefix, new_prefix in filepath_prefix_map.items():
                        if val.startswith(old_prefix):
                            val = new_prefix + val[len(old_prefix):]
                            break
                    sample[key] = val

            data.append(sample)

            if max_samples and len(data) >= max_samples:
                break

    print(f"[data_loader] loaddone {len(data)} records ImageDoctor prediction from {jsonl_path}")
    return data


def match_imagedoctor_to_test(
    test_data: List[Dict],
    imagedoctor_data: List[Dict],
) -> List[Dict]:
    """
    Match ImageDoctor predictions to the test set by filename.

    Returns:
        List[Dict]: matched samples; each record carries both test-set and ImageDoctor info
    """
    # build the ImageDoctor index (by filename basename)
    id_index = {}
    for item in imagedoctor_data:
        basename = os.path.basename(item.get("filename", ""))
        if basename:
            id_index[basename] = item

    matched = []
    for sample in test_data:
        basename = os.path.basename(sample.get("filepath", ""))
        if basename in id_index:
            merged = {**sample}
            id_item = id_index[basename]
            merged["imagedoctor_prediction"] = id_item.get("prediction", "")
            merged["imagedoctor_misalignment_heatmap"] = id_item.get("misalignment_heatmap", "")
            merged["imagedoctor_artifact_heatmap"] = id_item.get("artifact_heatmap", "")
            matched.append(merged)

    print(f"[data_loader] matched {len(matched)} ImageDoctor records")
    return matched


def filter_existing_images(data: List[Dict], filepath_key: str = "filepath") -> List[Dict]:
    """Filter out samples whose image file does not exist."""
    valid = []
    missing = 0
    for sample in data:
        path = sample.get(filepath_key, "")
        if path and os.path.exists(path):
            valid.append(sample)
        else:
            missing += 1
    if missing:
        print(f"[data_loader] skipdone {missing} recordsimagefilenot existsof sample")
    return valid


def load_sdg_predictions(
    jsonl_path: str,
    max_samples: Optional[int] = None,
    filepath_prefix_map: Optional[Dict[str, str]] = None,
) -> List[Dict]:
    """
    load SDG predictionfile (predictions.jsonl)。

    Returns:
        List[Dict]: each record contains:
            - filepath: image path
            - caption: textdescription
            - response: full SDG response text (includes <think> and <answer>)
    """
    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)

            # path mapping
            for key in ["filepath"]:
                val = sample.get(key, "")
                if val and filepath_prefix_map:
                    for old_prefix, new_prefix in filepath_prefix_map.items():
                        if val.startswith(old_prefix):
                            val = new_prefix + val[len(old_prefix):]
                            break
                    sample[key] = val

            data.append(sample)

            if max_samples and len(data) >= max_samples:
                break

    print(f"[data_loader] loaddone {len(data)} records SDG prediction from {jsonl_path}")
    return data


def match_sdg_to_test(
    test_data: List[Dict],
    sdg_data: List[Dict],
) -> List[Dict]:
    """
    Match SDG predictions to the test set by filename.

    Returns:
        List[Dict]: matched samples; each record carries both test-set and SDG info
    """
    # build the SDG index (by filename basename)
    sdg_index = {}
    for item in sdg_data:
        basename = os.path.basename(item.get("filepath", ""))
        if basename:
            sdg_index[basename] = item

    matched = []
    for sample in test_data:
        basename = os.path.basename(sample.get("filepath", ""))
        if basename in sdg_index:
            merged = {**sample}
            sdg_item = sdg_index[basename]
            merged["sdg_response"] = sdg_item.get("response", "")
            matched.append(merged)

    print(f"[data_loader] matched {len(matched)} SDG records")
    return matched
