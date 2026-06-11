"""
Visualization script for evaluation predictions (Qwen bbox format version).
Draws GT and predicted bboxes on images with different colors for different types.
Generates an LMDB for browsing with all information.

Bbox formats:
- GT (from gemini): [y0, x0, y1, x1] (Gemini format)
- Pred (from model): [x0, y0, x1, y1] (Qwen format)

Colors:
- GT misalignment: red solid
- GT artifact: green solid  
- Pred misalignment: light red dashed
- Pred artifact: light green dashed
"""
import os
import sys
import json
import re
import argparse
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# Add parent directory to path for lmdb_process import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lmdb_process import LmdbProcesser


def parse_boxes_with_desc(response: str):
    """
    Parse bounding boxes with descriptions from model response.
    Returns (misalignment_boxes, artifact_boxes) where each box is a dict with 'box_2d', 'label', 'desc'.
    """
    # Try to extract from <answer>...</answer>
    answer_match = re.search(r'<answer>\s*(.*?)\s*</answer>', response, re.DOTALL)
    if answer_match:
        boxes_str = answer_match.group(1)
    else:
        boxes_str = response
    
    parsed_boxes = []
    
    # Try json.loads first
    try:
        start = boxes_str.find('[')
        end = boxes_str.rfind(']')
        if start != -1 and end != -1 and end > start:
            result = json.loads(boxes_str[start:end+1])
            if isinstance(result, list):
                parsed_boxes = result
    except Exception:
        pass
    
    # Separate by label
    misalignment_boxes = []
    artifact_boxes = []
    
    for item in parsed_boxes:
        if isinstance(item, dict) and 'box_2d' in item:
            label = item.get('label', '').lower()
            box_entry = {
                'box_2d': item['box_2d'],
                'label': item.get('label', ''),
                'desc': item.get('desc', '')
            }
            if label == 'misalignment':
                misalignment_boxes.append(box_entry)
            elif label == 'artifact':
                artifact_boxes.append(box_entry)
    
    return misalignment_boxes, artifact_boxes


def is_valid_box(box):
    """Check if box coordinates are valid (x1 < x2 and y1 < y2)."""
    x1, y1, x2, y2 = box
    return x1 < x2 and y1 < y2


def draw_box(draw, box, color, width=3, dashed=False):
    """Draw a bounding box (solid or dashed)."""
    x1, y1, x2, y2 = box
    
    # Skip invalid boxes
    if not is_valid_box(box):
        return
    if dashed:
        # Draw dashed rectangle
        dash_length = 10
        gap_length = 5
        # Top edge
        x = x1
        while x < x2:
            draw.line([(x, y1), (min(x + dash_length, x2), y1)], fill=color, width=width)
            x += dash_length + gap_length
        # Bottom edge
        x = x1
        while x < x2:
            draw.line([(x, y2), (min(x + dash_length, x2), y2)], fill=color, width=width)
            x += dash_length + gap_length
        # Left edge
        y = y1
        while y < y2:
            draw.line([(x1, y), (x1, min(y + dash_length, y2))], fill=color, width=width)
            y += dash_length + gap_length
        # Right edge
        y = y1
        while y < y2:
            draw.line([(x2, y), (x2, min(y + dash_length, y2))], fill=color, width=width)
            y += dash_length + gap_length
    else:
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)


def convert_box_coords_gemini(box, img_width, img_height, coord_scale=1000):
    """Convert Gemini format [y0, x0, y1, x1] normalized [0-1000] coords to pixel coords [x0, y0, x1, y1]."""
    y0, x0, y1, x1 = box  # Gemini format: [y0, x0, y1, x1]
    return [
        x0 / coord_scale * img_width,
        y0 / coord_scale * img_height,
        x1 / coord_scale * img_width,
        y1 / coord_scale * img_height
    ]


def convert_box_coords_qwen(box, img_width, img_height, coord_scale=1000):
    """Convert Qwen format [x0, y0, x1, y1] normalized [0-1000] coords to pixel coords."""
    x0, y0, x1, y1 = box  # Qwen format: [x0, y0, x1, y1]
    return [
        x0 / coord_scale * img_width,
        y0 / coord_scale * img_height,
        x1 / coord_scale * img_width,
        y1 / coord_scale * img_height
    ]


def draw_boxes_on_image(image, gt_mis_boxes, gt_art_boxes, pred_mis_boxes, pred_art_boxes):
    """Draw all boxes on image with different colors and index labels.
    
    GT boxes are in Gemini format [y0, x0, y1, x1].
    Pred boxes are in Qwen format [x0, y0, x1, y1].
    """
    draw = ImageDraw.Draw(image)
    width, height = image.size
    
    # Colors
    GT_MIS_COLOR = '#B71C1C'  # Red
    GT_ART_COLOR = '#1B5E20'  # Green
    PRED_MIS_COLOR = '#EF5350'  # Light Red
    PRED_ART_COLOR = '#66BB6A'  # Light Green
    # Calculate font size as 1/5    of image size (use smaller dimension)
    font_size = max(20, min(width, height) // 20)
    
    # Try to load a font for labels (try multiple paths)
    font = None
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Arial.ttf",
    ]
    for font_path in font_paths:
        try:
            font = ImageFont.truetype(font_path, font_size)
            break
        except:
            continue
    
    # If no TrueType font found, use default but scale via resize workaround
    if font is None:
        # Fallback: use default font and we'll just have small text
        font = ImageFont.load_default()
        # print(f"Warning: Could not load TrueType font, using default (small) font")
    
    def draw_label(draw, x, y, text, color, font):
        """Draw a label at position (x, y)."""
        # Draw text directly in the box color
        draw.text((x, y), text, fill=color, font=font)
    
    # Draw GT boxes (solid) with labels - use Gemini format conversion
    for idx, box in enumerate(gt_mis_boxes):
        pixel_box = convert_box_coords_gemini(box, width, height)
        draw_box(draw, pixel_box, GT_MIS_COLOR, width=3, dashed=False)
        # Draw label to the left    of the box
        x1, y1, x2, y2 = pixel_box
        label_x = max(0, x1 - font_size - 2)  # Position to the left    of box
        draw_label(draw, label_x, y1, str(idx), GT_MIS_COLOR, font)
    
    for idx, box in enumerate(gt_art_boxes):
        pixel_box = convert_box_coords_gemini(box, width, height)
        draw_box(draw, pixel_box, GT_ART_COLOR, width=3, dashed=False)
        x1, y1, x2, y2 = pixel_box
        label_x = max(0, x1 - font_size - 2)
        draw_label(draw, label_x, y1, str(idx), GT_ART_COLOR, font)
    
    # Draw Pred boxes (dashed) with labels - use Qwen format conversion
    for idx, box in enumerate(pred_mis_boxes):
        pixel_box = convert_box_coords_qwen(box, width, height)
        draw_box(draw, pixel_box, PRED_MIS_COLOR, width=2, dashed=True)
        x1, y1, x2, y2 = pixel_box
        label_x = max(0, x1 - font_size - 2)
        draw_label(draw, label_x, y1, str(idx), PRED_MIS_COLOR, font)
    
    for idx, box in enumerate(pred_art_boxes):
        pixel_box = convert_box_coords_qwen(box, width, height)
        draw_box(draw, pixel_box, PRED_ART_COLOR, width=2, dashed=True)
        x1, y1, x2, y2 = pixel_box
        label_x = max(0, x1 - font_size - 2)
        draw_label(draw, label_x, y1, str(idx), PRED_ART_COLOR, font)
    
    return image


def add_legend(image, has_gt_mis, has_gt_art, has_pred_mis, has_pred_art):
    """Add a legend to the image."""
    draw = ImageDraw.Draw(image)
    
    legend_items = []
    if has_gt_mis:
        legend_items.append(('GT Misalign', '#B71C1C'))
    if has_gt_art:
        legend_items.append(('GT Artifact', '#1B5E20'))
    if has_pred_mis:
        legend_items.append(('Pred Misalign', '#EF5350'))
    if has_pred_art:
        legend_items.append(('Pred Artifact', '#66BB6A'))
    
    if not legend_items:
        return image
    
    # Draw legend background
    legend_height = 25 * len(legend_items) + 10
    legend_width = 140
    draw.rectangle([5, 5, legend_width, legend_height], fill='white', outline='black')
    
    y = 10
    for text, color in legend_items:
        draw.rectangle([10, y, 30, y + 15], fill=color, outline='black')
        draw.text((35, y), text, fill='black')
        y += 25
    
    return image


def process_predictions(jsonl_path, output_pic_dir, output_lmdb_dir, max_samples=None):
    """Process predictions, generate visualizations and LMDB."""
    os.makedirs(output_pic_dir, exist_ok=True)
    
    # Load predictions
    predictions = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                predictions.append(json.loads(line))
    
    if max_samples and max_samples < len(predictions):
        predictions = predictions[:max_samples]
    
    print(f"Processing {len(predictions)} predictions...")
    
    # Prepare LMDB data
    lmdb_data = []
    
    for idx, pred in enumerate(tqdm(predictions, desc="Generating visualizations")):
        image_path = pred.get('filename', pred.get('image_path', ''))
        caption = pred.get('caption', '')
        
        # Get boxes - support both new and old field names
        # New format: gemini_*_bboxes / pred_*_bboxes (list    of dicts with box_2d, label, desc)
        # Old format: gt_*_boxes / pred_*_boxes (list    of coords)
        gt_mis_raw = pred.get('gemini_misalignment_bboxes', [])
        gt_art_raw = pred.get('gemini_artifact_bboxes', [])
        pred_mis_raw = pred.get('pred_misalignment_bboxes', [])
        pred_art_raw = pred.get('pred_artifact_bboxes', [])
        
        # Extract coordinates for drawing
        def extract_coords(boxes):
            coords = []
            for b in boxes:
                if isinstance(b, dict) and 'box_2d' in b:
                    coords.append(b['box_2d'])
                elif isinstance(b, list) and len(b) == 4:
                    coords.append(b)
            return coords
        
        gt_mis = extract_coords(gt_mis_raw) if gt_mis_raw else pred.get('gt_misalignment_boxes', [])
        gt_art = extract_coords(gt_art_raw) if gt_art_raw else pred.get('gt_artifact_boxes', [])
        pred_mis = extract_coords(pred_mis_raw) if pred_mis_raw else pred.get('pred_misalignment_boxes', [])
        pred_art = extract_coords(pred_art_raw) if pred_art_raw else pred.get('pred_artifact_boxes', [])
        
        # Load and process image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Cannot load image {image_path}: {e}")
            continue
        
        # Draw boxes
        vis_image = draw_boxes_on_image(
            image.copy(), gt_mis, gt_art, pred_mis, pred_art
        )
        
        # Add legend
        vis_image = add_legend(
            vis_image,
            len(gt_mis) > 0, len(gt_art) > 0,
            len(pred_mis) > 0, len(pred_art) > 0
        )
        
        safe_caption = caption.replace(' ', '-')[:80].replace('/', '-')
        # Save visualization
        vis_filename = f"{idx:04d}-{safe_caption}.jpg"
        vis_path = os.path.join(output_pic_dir, vis_filename)
        vis_image.save(vis_path, quality=95)
        
        # Get response text
        response_text = pred.get('response', '')
        
        # Use raw boxes with desc if available, otherwise parse from response
        if not pred_mis_raw and not pred_art_raw and response_text:
            pred_mis_raw, pred_art_raw = parse_boxes_with_desc(response_text)
        
        # Prepare LMDB entry with full desc format
        lmdb_entry = {
            'idx': idx,
            'caption': caption,
            # GT with descriptions
            'gemini_misalignment_bboxes': gt_mis_raw,
            'gemini_artifact_bboxes': gt_art_raw,
            # Pred with descriptions
            'pred_misalignment_bboxes': pred_mis_raw,
            'pred_artifact_bboxes': pred_art_raw,
            
            'filename': image_path,
            'response': response_text,
            
            'misalignment_iou': pred.get('misalignment_iou', 0),
            'artifact_iou': pred.get('artifact_iou', 0),
            'input_images': [{"img_path": vis_path.replace('_', '-')}],
        }
        lmdb_data.append(lmdb_entry)
    
    print(f"Saved {len(lmdb_data)} visualizations to {output_pic_dir}")
    
    # Create LMDB
    if output_lmdb_dir:
        print(f"Creating LMDB at {output_lmdb_dir}...")
        lmdb_writer = LmdbProcesser(output_lmdb_dir, mode="write")
        for entry in tqdm(lmdb_data, desc="Writing LMDB"):
            lmdb_writer.save_one_data(entry)
        lmdb_writer.end_save()
        print(f"LMDB created with {len(lmdb_data)} entries")
    
    return lmdb_data


def main():
    parser = argparse.ArgumentParser(description='Visualize evaluation predictions')
    parser.add_argument('--jsonl', type=str, required=True,
                        help='Path to predictions JSONL file')
    parser.add_argument('--output_pic', type=str, default=None,
                        help='Output directory for visualizations (default: {jsonl_dir}/pic)')
    parser.add_argument('--output_lmdb', type=str, default=None,
                        help='Output directory for LMDB (default: {jsonl_dir}/lmdb)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number    of samples to process')
    args = parser.parse_args()
    
    # Auto-infer output directories from jsonl path
    jsonl_dir = os.path.dirname(args.jsonl)
    output_pic = args.output_pic or os.path.join(jsonl_dir, 'pic')
    output_lmdb = args.output_lmdb or os.path.join(jsonl_dir, 'lmdb')
    
    process_predictions(
        jsonl_path=args.jsonl,
        output_pic_dir=output_pic,
        output_lmdb_dir=output_lmdb,
        max_samples=args.max_samples
    )


if __name__ == '__main__':
    main()
