"""
Convert Gemini bbox format [y0, x0, y1, x1] to Qwen bbox format [x0, y0, x1, y1].

Pipeline:
  1. Read JSONL data containing `raw_response`.
  2. Extract the <answer> block.
  3. Validate each bbox (range 0-1000, ordering y0<y1, x0<x1).
  4. Convert coordinate ordering.
  5. Write the converted records out.

Example:
  python convert_gemini_to_qwen_bbox.py \\
      --input  ${SDG_DATA}/sdg30k/annotations/test.jsonl \\
      --output ${SDG_DATA}/sdg30k/annotations/test_qwen.jsonl
"""

import json
import re
import argparse
from tqdm import tqdm


def extract_answer_block(raw_response: str) -> str:
    """Return the contents    of <answer>...</answer> in raw_response."""
    pattern = r'<answer>\s*(.*?)\s*</answer>'
    match = re.search(pattern, raw_response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def extract_think_block(raw_response: str) -> str:
    """Return the contents    of <think>...</think> in raw_response."""
    pattern = r'<think>\s*(.*?)\s*</think>'
    match = re.search(pattern, raw_response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def parse_bbox_list(answer_content: str) -> list:
    """Parse the bbox list inside the <answer> block."""
    try:
        bbox_list = json.loads(answer_content)
        if isinstance(bbox_list, list):
            return bbox_list
    except json.JSONDecodeError:
        pass

    # Fall back to extracting the first JSON array substring.
    try:
        array_match = re.search(r'\[.*\]', answer_content, re.DOTALL)
        if array_match:
            bbox_list = json.loads(array_match.group())
            if isinstance(bbox_list, list):
                return bbox_list
    except json.JSONDecodeError:
        pass

    return []


def validate_bbox(box: list, item_info: str) -> tuple:
    """
    Validate a bbox in Gemini format [y0, x0, y1, x1].

    Returns: (is_valid, error_messages)
    """
    errors = []

    if not isinstance(box, list) or len(box) != 4:
        errors.append(f"bbox is not a list    of 4 elements: {box}")
        return False, errors

    y0, x0, y1, x1 = box

    for i, val in enumerate([y0, x0, y1, x1]):
        if not isinstance(val, (int, float)):
            errors.append(f"non-numeric value: box[{i}] = {val}")

    if errors:
        return False, errors

    coords = {'y0': y0, 'x0': x0, 'y1': y1, 'x1': x1}
    for name, val in coords.items():
        if val < 0 or val > 1000:
            errors.append(f"{name}={val} out    of range [0, 1000]")

    if y0 >= y1:
        errors.append(f"y0={y0} >= y1={y1} (expected y0 < y1)")

    if x0 >= x1:
        errors.append(f"x0={x0} >= x1={x1} (expected x0 < x1)")

    return len(errors) == 0, errors


def convert_gemini_to_qwen_bbox(box: list) -> list:
    """
    Convert from Gemini to Qwen bbox ordering.
        Gemini: [y0, x0, y1, x1]
        Qwen:   [x0, y0, x1, y1]
    """
    y0, x0, y1, x1 = box
    return [x0, y0, x1, y1]


def convert_bbox_in_text(text: str) -> str:
    """
    Convert every [y0, x0, y1, x1] occurrence in `text` to [x0, y0, x1, y1].
    Matches integer-only patterns like `[220, 730, 550, 900]`.
    """
    def replace_bbox(match):
        try:
            bbox_str = match.group(0)
            bbox = json.loads(bbox_str)
            if isinstance(bbox, list) and len(bbox) == 4:
                if all(isinstance(x, (int, float)) for x in bbox):
                    y0, x0, y1, x1 = bbox
                    converted = [x0, y0, x1, y1]
                    return json.dumps(converted)
        except Exception:
            pass
        return match.group(0)

    pattern = r'\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]'
    return re.sub(pattern, replace_bbox, text)


def convert_answer_block(raw_response: str) -> str:
    """Convert all bboxes inside the <answer> block    of raw_response."""
    answer_content = extract_answer_block(raw_response)
    if not answer_content:
        return raw_response

    bbox_list = parse_bbox_list(answer_content)
    if not bbox_list:
        return raw_response

    converted_list = []
    for item in bbox_list:
        if isinstance(item, dict) and 'box_2d' in item:
            new_item = item.copy()
            new_item['box_2d'] = convert_gemini_to_qwen_bbox(item['box_2d'])
            converted_list.append(new_item)
        else:
            converted_list.append(item)

    new_answer_content = json.dumps(converted_list, ensure_ascii=False)
    pattern = r'<answer>\s*.*?\s*</answer>'
    new_answer_block = f'<answer>\n{new_answer_content}\n</answer>'
    return re.sub(pattern, new_answer_block, raw_response, flags=re.DOTALL)


def convert_think_block(raw_response: str) -> str:
    """Convert all bboxes inside the <think> block    of raw_response."""
    think_content = extract_think_block(raw_response)
    if not think_content:
        return raw_response

    new_think_content = convert_bbox_in_text(think_content)
    pattern = r'<think>\s*.*?\s*</think>'
    new_think_block = f'<think>\n{new_think_content}\n</think>'
    return re.sub(pattern, new_think_block, raw_response, flags=re.DOTALL)


def process_item(item: dict, stats: dict, line_num: int) -> dict:
    """Validate and convert a single record."""
    raw_response = item.get("raw_response", "")
    filename = item.get("filename", f"line_{line_num}")

    if not raw_response:
        stats['no_response'] += 1
        print(f"\n[skip] line {line_num}: missing raw_response, file: {filename}")
        return None

    answer_content = extract_answer_block(raw_response)
    bbox_list = parse_bbox_list(answer_content)

    has_invalid = False
    for i, bbox_item in enumerate(bbox_list):
        if isinstance(bbox_item, dict) and 'box_2d' in bbox_item:
            box = bbox_item['box_2d']
            is_valid, errors = validate_bbox(box, filename)
            if not is_valid:
                has_invalid = True
                stats['invalid_bbox_count'] += 1
                error_info = {
                    'filename': filename,
                    'line_num': line_num,
                    'bbox_index': i,
                    'bbox': box,
                    'label': bbox_item.get('label', 'unknown'),
                    'desc': bbox_item.get('desc', ''),
                    'errors': errors,
                }
                stats['invalid_details'].append(error_info)
                print(f"\n[invalid bbox] file: {filename}, line: {line_num}, bbox idx: {i}")
                print(f"  bbox: {box}")
                print(f"  label: {bbox_item.get('label', 'unknown')}")
                print(f"  errors: {', '.join(errors)}")

    if has_invalid:
        stats['items_with_invalid_bbox'] += 1

    new_item = item.copy()
    converted_response = convert_think_block(raw_response)
    converted_response = convert_answer_block(converted_response)
    new_item['raw_response'] = converted_response

    return new_item


def main():
    parser = argparse.ArgumentParser(
        description='Convert Gemini bbox format [y0,x0,y1,x1] to Qwen format [x0,y0,x1,y1].'
    )
    parser.add_argument('--input', type=str, required=True,
                        help='Input JSONL file path.')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSONL file path (default: {input_name}_qwen.jsonl).')
    parser.add_argument('--skip-invalid', action='store_true',
                        help='Skip records that contain any invalid bbox '
                             '(default: keep them but still convert).')
    args = parser.parse_args()

    if args.output:
        output_path = args.output
    else:
        import os
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        output_dir = os.path.dirname(args.input) or '.'
        output_path = os.path.join(output_dir, f"{base_name}_qwen2.jsonl")

    print(f"input file:  {args.input}")
    print(f"output file: {output_path}")
    print(f"skip invalid: {args.skip_invalid}")
    print("-" * 60)

    stats = {
        'total': 0,
        'processed': 0,
        'skipped': 0,
        'no_response': 0,
        'invalid_bbox_count': 0,
        'items_with_invalid_bbox': 0,
        'invalid_details': [],
    }

    import os
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    with open(args.input, 'r', encoding='utf-8') as f_in:
        lines = f_in.readlines()

    with open(output_path, 'w', encoding='utf-8') as f_out:
        for line_num, line in enumerate(tqdm(lines, desc="processing"), 1):
            if not line.strip():
                continue

            stats['total'] += 1

            try:
                item = json.loads(line)
                processed_item = process_item(item, stats, line_num)

                if processed_item is None:
                    stats['skipped'] += 1
                    continue

                if args.skip_invalid and any(
                    d['line_num'] == line_num for d in stats['invalid_details']
                ):
                    stats['skipped'] += 1
                    print(f"\n[skip] line {line_num}: contains invalid bbox "
                          f"(--skip-invalid), file: {item.get('filename', 'unknown')}")
                    continue

                f_out.write(json.dumps(processed_item, ensure_ascii=False) + '\n')
                stats['processed'] += 1

            except json.JSONDecodeError as e:
                print(f"\n[JSON parse error] line {line_num}: {e}")
                stats['skipped'] += 1
            except Exception as e:
                print(f"\n[process error] line {line_num}: {e}")
                stats['skipped'] += 1

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)
    print(f"total records:        {stats['total']}")
    print(f"processed:            {stats['processed']}")
    print(f"skipped:              {stats['skipped']}")
    print(f"  - no raw_response:  {stats['no_response']}")
    print("-" * 60)
    print(f"total invalid bboxes: {stats['invalid_bbox_count']}")
    print(f"records with invalid bbox: {stats['items_with_invalid_bbox']}")

    if stats['invalid_details']:
        print("\nInvalid bbox details:")
        print("-" * 60)
        for detail in stats['invalid_details']:
            print(f"  file:  {detail['filename']}")
            print(f"  line:  {detail['line_num']}, bbox idx: {detail['bbox_index']}")
            print(f"  bbox:  {detail['bbox']}")
            print(f"  label: {detail['label']}, desc: {detail['desc']}")
            print(f"  errors: {', '.join(detail['errors'])}")
            print()

    print(f"\noutput saved to: {output_path}")


if __name__ == "__main__":
    main()
