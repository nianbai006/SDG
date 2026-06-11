#!/usr/bin/env python3
"""Prepare merged SDG JSONL for Swift GRPO training."""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from prepare_all_datasets import question_template_registry

def normalize_boxes(boxes) -> list[list[int]]:
    output = []
    for item in boxes or []:
        if isinstance(item, list) and len(item) == 4:
            output.append([int(v) for v in item])
        elif isinstance(item, dict) and "box_2d" in item:
            box = item["box_2d"]
            if isinstance(box, list) and len(box) == 4:
                output.append([int(v) for v in box])
    return output


def validate_xyxy_boxes(boxes: list[list[int]]) -> bool:
    for box in boxes:
        if len(box) != 4:
            return False
        x0, y0, x1, y1 = box
        if any(v < 0 or v > 1000 for v in (x0, y0, x1, y1)):
            return False
        if x0 >= x1 or y0 >= y1:
            return False
    return True


def process_item(item: dict, question_template: str):
    image_path = item.get("filepath") or item.get("filename") or item.get("image_path")
    caption = item.get("caption", "")

    if not image_path:
        return None, "missing_image_path"
    # if not os.path.exists(image_path):
    #     return None, "missing_image_file"

    user_text = question_template.format(caption=caption)
    
    # Structure from preprocess_swift_grpo_dataset_think.py
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": user_text},
            ],
        }
    ]

    gt_misalignment = normalize_boxes(item.get("misalignment_bboxes", []))
    gt_artifact = normalize_boxes(item.get("artifact_bboxes", []))

    if not validate_xyxy_boxes(gt_misalignment + gt_artifact):
        return None, "invalid_bbox"

    return {
        "messages": messages,
        "gt_misalignment_bboxes": gt_misalignment,
        "gt_artifact_bboxes": gt_artifact,
    }, None


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess merged JSONL for Swift GRPO (think).")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSONL file path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--question_template",
        type=str,
        default="think",
        help="Key in question_template_registry",
    )
    parser.add_argument("--concurrency", type=int, default=128, help="Number    of threads")
    args = parser.parse_args()

    question_template = question_template_registry.get(args.question_template)
    if not question_template:
        raise ValueError(f"Unknown question_template: {args.question_template}")

    dataset_path = args.input
    if args.output:
        output_path = args.output
    else:
        base_name = os.path.splitext(os.path.basename(dataset_path))[0]
        output_path = os.path.join(
            os.path.dirname(dataset_path), f"{base_name}_swift_grpo_think.jsonl"
        )

    print(f"Processing {dataset_path}...")
    print(f"Output: {output_path}")
    print(f"Question template: {args.question_template}")
    print(f"Concurrency: {args.concurrency} threads")

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    with open(dataset_path, "r") as f_in:
        lines = [line.strip() for line in f_in.readlines() if line.strip()]

    print(f"Total samples: {len(lines)}")

    results = []
    skipped = 0
    skipped_reasons: dict[str, int] = {}

    def process_line(line: str):
        try:
            raw_item = json.loads(line)
            return process_item(raw_item, question_template)
        except Exception:
            return None, "parse_error"

    # with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
    #     futures = {executor.submit(process_line, line): i for i, line in enumerate(lines)}
    #     for future in as_completed(futures):
    #         result, reason = future.result()
    #         if result:
    #             results.append(result)
    #         else:
    #             skipped += 1
    #             reason = reason or "unknown"
    #             skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1
    
    # Using simple loop to avoid tqdm dependency and ensure sequential print execution without conflicts
    for line in lines:
        result, reason = process_line(line)
        if result:
            results.append(result)
        else:
            skipped += 1
            reason = reason or "unknown"
            skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1

    with open(output_path, "w") as f_out:
        for item in results:
            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Successfully saved {len(results)} samples to {output_path}")
    if skipped > 0:
        print(f"Skipped {skipped} samples")
        for reason, count in sorted(skipped_reasons.items(), key=lambda x: (-x[1], x[0])):
            print(f"  - {reason}: {count}")


if __name__ == "__main__":
    main()
