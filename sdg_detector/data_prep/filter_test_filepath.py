#!/usr/bin/env python3
"""
Filter JSONL records where `filepath` contains the substring "test".
"""
import argparse
import json
import os
from typing import Optional


def should_keep(filepath: Optional[str], keywords: list[str]) -> bool:
    if not filepath:
        return False
    lower_path = filepath.lower()
    return any(keyword in lower_path for keyword in keywords)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract JSONL records whose filepath contains 'test'."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="${SDG_DATA}/sdg30k/annotations/merged_all_filtered_distilled.jsonl",
        help="Input JSONL file path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSONL file path (used when a single keyword is provided)",
    )
    parser.add_argument(
        "--output_train",
        type=str,
        default=None,
        help="Output JSONL file path for train records",
    )
    parser.add_argument(
        "--output_test",
        type=str,
        default=None,
        help="Output JSONL file path for test records",
    )
    parser.add_argument(
        "--keywords",
        type=str,
        default="test,train",
        help="Comma-separated keywords to match in filepath",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number    of records to write",
    )
    args = parser.parse_args()

    input_path = args.input
    keywords = [k.strip().lower() for k in args.keywords.split(",") if k.strip()]
    if not keywords:
        raise ValueError("--keywords must contain at least one non-empty keyword")

    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_dir = os.path.dirname(input_path)

    if len(keywords) == 1:
        if args.output:
            output_path = args.output
        else:
            output_path = os.path.join(
                output_dir, f"{base_name}_filepath_{keywords[0]}.jsonl"
            )
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        output_paths = {keywords[0]: output_path}
    else:
        output_paths = {}
        if "train" in keywords:
            output_paths["train"] = args.output_train or os.path.join(
                output_dir, f"{base_name}_filepath_train.jsonl"
            )
        if "test" in keywords:
            output_paths["test"] = args.output_test or os.path.join(
                output_dir, f"{base_name}_filepath_test.jsonl"
            )
        for path in output_paths.values():
            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

    total = 0
    kept = 0
    skipped_no_filepath = 0
    kept_by_keyword = {key: 0 for key in output_paths}

    output_files = {key: open(path, "w") for key, path in output_paths.items()}
    try:
        with open(input_path, "r") as f_in:
            for line in f_in:
                line = line.strip()
                if not line:
                    continue
                total += 1
                try:
                    item = json.loads(line)
                except Exception:
                    continue

                filepath = item.get("filepath")
                if filepath is None:
                    skipped_no_filepath += 1
                    continue

                lower_path = filepath.lower()
                wrote = False
                for key, f_out in output_files.items():
                    if key in lower_path:
                        f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
                        kept_by_keyword[key] += 1
                        wrote = True
                if wrote:
                    kept += 1
                if args.limit is not None and kept >= args.limit:
                    break
    finally:
        for f_out in output_files.values():
            f_out.close()

    print(f"Input: {input_path}")
    for key, path in output_paths.items():
        print(f"Output ({key}): {path}")
    print(f"Total scanned: {total}")
    print(f"Kept total: {kept}")
    for key, count in kept_by_keyword.items():
        print(f"Kept {key}: {count}")
    if skipped_no_filepath:
        print(f"Skipped (missing filepath): {skipped_no_filepath}")


if __name__ == "__main__":
    main()
