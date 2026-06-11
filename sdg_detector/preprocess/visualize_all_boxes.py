"""
Visualize every bbox type plus the source image and heatmaps
- artifact_bboxes: blue boxes (from artifact heatmap)
- misalignment_bboxes: red boxes (from misalignment heatmap)
- merged_bboxes: green boxes (from merged heatmap)
- gemini_bboxes: yellow boxes (legacy Gemini predictions)
- gemini_artifact_bboxes: cyan boxes (Gemini artifact predictions)
- gemini_misalignment_bboxes: magenta boxes (Gemini misalignment predictions)
"""

import os
import json
import argparse
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm


def load_heatmap(path):
    """Load heatmap (supports npy and image formats)"""
    if not path or not os.path.exists(path):
        return None
    if path.endswith('.npy'):
        return np.load(path)
    else:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        return img / 255.0


def convert_bbox_1000(bbox, img_h, img_w):
    """Convert normalized 0-1000 coords to absolute image pixels
    bbox format: [y0, x0, y1, x1]
    """
    y0, x0, y1, x1 = bbox
    x_min = int(x0 / 1000 * img_w)
    y_min = int(y0 / 1000 * img_h)
    x_max = int(x1 / 1000 * img_w)
    y_max = int(y1 / 1000 * img_h)
    return x_min, y_min, x_max, y_max


def extract_box_coords(bbox_item):
    """Extract coordinates from a bbox item; supports two formats:
    - new format: {"box_2d": [y0, x0, y1, x1], "label": "...", "desc": "..."}
    - legacy format: [y0, x0, y1, x1]
    Returns: [y0, x0, y1, x1] or None
    """
    if isinstance(bbox_item, dict):
        # new object format
        box = bbox_item.get("box_2d", [])
        if isinstance(box, list) and len(box) == 4:
            return box
        return None
    elif isinstance(bbox_item, list) and len(bbox_item) == 4:
        # legacy list format
        return bbox_item
    return None


def visualize_sample(data, output_path, target_size=768):
    """
    Visualize a single sample as a 2x2 image grid:
    - top-left: source + gemini_bboxes (yellow)
    - top-right: source + artifact (blue) + misalignment (red) + merged (green) bboxes
    - bottom-left: artifact heatmap
    - bottom-right: misalignment heatmap
    """
    # Get paths
    original_path = data.get("filename")
    artifact_map_path = data.get("artifact_map_path")
    misalignment_map_path = data.get("misalignment_map_path")
    
    # Get bboxes
    artifact_bboxes = data.get("artifact_bboxes", [])
    misalignment_bboxes = data.get("misalignment_bboxes", [])
    merged_bboxes = data.get("merged_bboxes", [])
    gemini_bboxes = data.get("gemini_bboxes", [])  # legacy: single list
    gemini_artifact_bboxes = data.get("gemini_artifact_bboxes", [])  # new format: separate artifact
    gemini_misalignment_bboxes = data.get("gemini_misalignment_bboxes", [])  # new format: separate misalignment
    
    # Load source image
    if not original_path or not os.path.exists(original_path):
        return False
    original_img = cv2.imread(original_path)
    if original_img is None:
        return False
    
    orig_h, orig_w = original_img.shape[:2]
    
    # Resize to a square canvas
    cell_size = target_size // 2
    
    # ============ top-left: source + Gemini predictions ============
    img_gemini = cv2.resize(original_img.copy(), (cell_size, cell_size))
    
    # Detect new vs legacy format
    use_new_format = len(gemini_artifact_bboxes) > 0 or len(gemini_misalignment_bboxes) > 0
    
    if use_new_format:
        # New format: gemini_artifact_bboxes (cyan) + gemini_misalignment_bboxes (magenta)
        for bbox_item in gemini_artifact_bboxes:
            bbox = extract_box_coords(bbox_item)
            if bbox is None or bbox == [0, 0, 0, 0]:
                continue
            x_min, y_min, x_max, y_max = convert_bbox_1000(bbox, cell_size, cell_size)
            # cyan (BGR: 255, 255, 0), inset
            cv2.rectangle(img_gemini, (x_min+3, y_min+3), (x_max+3, y_max+3), (255, 255, 0), 2)
        for bbox_item in gemini_misalignment_bboxes:
            bbox = extract_box_coords(bbox_item)
            if bbox is None or bbox == [0, 0, 0, 0]:
                continue
            x_min, y_min, x_max, y_max = convert_bbox_1000(bbox, cell_size, cell_size)
            # magenta (BGR: 255, 0, 255), outset
            cv2.rectangle(img_gemini, (x_min-3, y_min-3), (x_max-3, y_max-3), (255, 0, 255), 2)
        # Add label
        cv2.putText(img_gemini, "Gemini: Art(C) Mis(M)", (10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    else:
        # Legacy format: gemini_bboxes (yellow)
        for bbox_item in gemini_bboxes:
            bbox = extract_box_coords(bbox_item)
            if bbox is None or bbox == [0, 0, 0, 0]:
                continue
            x_min, y_min, x_max, y_max = convert_bbox_1000(bbox, cell_size, cell_size)
            cv2.rectangle(img_gemini, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)  # yellow (BGR)
        # Add label
        cv2.putText(img_gemini, "Gemini Pred (Yellow)", (10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # ============ top-right: source + all GT bboxes ============
    img_gt = cv2.resize(original_img.copy(), (cell_size, cell_size))
    # artifact_bboxes: blue (inset)
    for bbox in artifact_bboxes:
        x_min, y_min, x_max, y_max = convert_bbox_1000(bbox, cell_size, cell_size)
        x_min, y_min = x_min + 4, y_min + 4
        x_max, y_max = x_max + 4, y_max + 4
        cv2.rectangle(img_gt, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)  # blue
    # misalignment_bboxes: red (outset)
    for bbox in misalignment_bboxes:
        x_min, y_min, x_max, y_max = convert_bbox_1000(bbox, cell_size, cell_size)
        x_min, y_min = x_min - 4, y_min - 4
        x_max, y_max = x_max - 4, y_max - 4
        cv2.rectangle(img_gt, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)  # red
    # merged_bboxes: green (no offset)
    for bbox in merged_bboxes:
        x_min, y_min, x_max, y_max = convert_bbox_1000(bbox, cell_size, cell_size)
        cv2.rectangle(img_gt, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # green
    # Add label
    cv2.putText(img_gt, "GT: Art(B) Mis(R) Merge(G)", (10, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # ============ bottom-left: artifact heatmap ============
    artifact_map = load_heatmap(artifact_map_path)
    if artifact_map is not None:
        if artifact_map.max() <= 1.0:
            artifact_norm = (artifact_map * 255).astype(np.uint8)
        else:
            artifact_norm = artifact_map.astype(np.uint8)
        # Apply blue colormap
        img_artifact = cv2.applyColorMap(artifact_norm, cv2.COLORMAP_OCEAN)
        img_artifact = cv2.resize(img_artifact, (cell_size, cell_size))
    else:
        img_artifact = np.zeros((cell_size, cell_size, 3), dtype=np.uint8)
    cv2.putText(img_artifact, "Artifact Heatmap", (10, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # ============ bottom-right: misalignment heatmap ============
    misalignment_map = load_heatmap(misalignment_map_path)
    if misalignment_map is not None:
        if misalignment_map.max() <= 1.0:
            misalignment_norm = (misalignment_map * 255).astype(np.uint8)
        else:
            misalignment_norm = misalignment_map.astype(np.uint8)
        # Apply red colormap
        img_misalignment = cv2.applyColorMap(misalignment_norm, cv2.COLORMAP_HOT)
        img_misalignment = cv2.resize(img_misalignment, (cell_size, cell_size))
    else:
        img_misalignment = np.zeros((cell_size, cell_size, 3), dtype=np.uint8)
    cv2.putText(img_misalignment, "Misalignment Heatmap", (10, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # ============ Compose 2x2 grid ============
    top_row = np.hstack([img_gemini, img_gt])
    bottom_row = np.hstack([img_artifact, img_misalignment])
    final_img = np.vstack([top_row, bottom_row])
    
    # Append caption
    caption = data.get("caption", "")[:100]
    cv2.putText(final_img, f"Caption: {caption}", (10, target_size - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imwrite(output_path, final_img)
    return True


def main():
    parser = argparse.ArgumentParser(description='Visualize all bboxes with heatmaps')
    parser.add_argument('--jsonl', type=str, 
                        default="${SDG_DATA}/sdg30k/test/test_gemini_bbox_ref6.jsonl",
                        help='Path to input JSONL file')
    parser.add_argument('--num', type=int, default=100, 
                        help='Number    of samples to visualize')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: same as jsonl dir)')
    parser.add_argument('--size', type=int, default=768,
                        help='Output image size (will be size x size)')
    args = parser.parse_args()
    
    # Set up output directory
    if args.output_dir:
        out_dir = args.output_dir
    else:
        out_dir = os.path.join(os.path.dirname(args.jsonl), 'vis_all_boxes_bbox_ref6')
    os.makedirs(out_dir, exist_ok=True)
    
    # Read data
    with open(args.jsonl, 'r') as f:
        lines = f.readlines()
    
    print(f"Total samples: {len(lines)}")
    print(f"Visualizing first {min(args.num, len(lines))} samples...")
    print(f"Output directory: {out_dir}")
    
    success_count = 0
    for idx, line in enumerate(tqdm(lines[:args.num], desc="Visualizing")):
        data = json.loads(line)
        
        # Generate filename
        caption = data.get("caption", "no_caption")
        safe_caption = "".join(c if c.isalnum() or c in " _-" else "_" for c in caption)[:60]
        output_path = os.path.join(out_dir, f"{idx+1:04d}_{safe_caption}.jpg")
        
        if visualize_sample(data, output_path, args.size):
            success_count += 1
    
    print(f"\nDone! Successfully visualized {success_count}/{min(args.num, len(lines))} samples")
    print(f"Output saved to: {out_dir}")


if __name__ == '__main__':
    main()
