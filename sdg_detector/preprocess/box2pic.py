import os
import json
import argparse
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from qwen_vl_utils import smart_resize


def draw_boxes_on_image(image_path, gt_boxes, pred_boxes, save_path):
    image = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(image)
    # Draw GT boxes in green
    for box in gt_boxes:
        draw.rectangle(box, outline='green', width=3)
    # Draw predicted boxes in red
    for box in pred_boxes:
        draw.rectangle(box, outline='red', width=3)
    image.save(save_path)


def main():
    parser = argparse.ArgumentParser(description='Preview GT and predicted boxes from jsonl')
    parser.add_argument('--jsonl', type=str,  default="${SDG_DATA}/sdg30k/test/test_with_think_gemini_prov3.jsonl")
    parser.add_argument('--num', type=int, default=955, help='Number    of images to preview')
    args = parser.parse_args()

    jsonl_path = args.jsonl
    num = args.num
    out_dir = os.path.dirname(jsonl_path)
    pic_dir = os.path.join(out_dir, 'pic_resized')
    os.makedirs(pic_dir, exist_ok=True)

    with open(jsonl_path, 'r') as f:
        lines = f.readlines()

    for idx, line in enumerate(lines[:num]):
        data = json.loads(line)
        image_path = data['filename']
        caption = data.get('caption', 'No caption')
        caption = caption.replace(' ', '_')[:90]  # Shorten caption for filename
        gt_boxes = data.get('bboxes', [])
        pred_boxes = data.get('gemini_bboxes', [])
        # Convert to (x1, y1, x2, y2) float tuples
        gt_boxes = [tuple(map(float, box)) for box in gt_boxes]
        pred_boxes = [tuple(map(float, box)) for box in pred_boxes]
        image = Image.open(image_path)
        width, height = image.size
        new_boxes = []
        new_boxes_gt = []
        # resize gt_boxes
        for box in gt_boxes:
            x1, y1, x2, y2 = box
            box = [
                x1 /512 * width,
                y1 /512 * height,  
                x2 /512 * width,
                y2 /512 * height
            ]
            new_boxes_gt.append(tuple(box))
        gt_boxes = new_boxes_gt
        # resize pred_boxes
        for box in pred_boxes:
            y1, x1, y2, x2 = box
            box = [
                x1 /1000 * width,
                y1 /1000 * height,
                x2 /1000 * width,
                y2 /1000 * height
            ]
            new_boxes.append(tuple(box))
        pred_boxes = new_boxes
        


        save_path = os.path.join(pic_dir, f'{idx+1}_{caption}.jpg')
        draw_boxes_on_image(image_path, gt_boxes, pred_boxes, save_path)
        print(f'Saved: {save_path}')

if __name__ == '__main__':
    main()
