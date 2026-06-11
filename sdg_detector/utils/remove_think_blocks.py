#!/usr/bin/env python3
"""
Remove <think>...</think> blocks from raw_response field in JSONL files.

Usage:
    python remove_think_blocks.py <input_jsonl> [output_jsonl]
    
If output_jsonl is not provided, it will be named as input_name_nothink.jsonl
"""

import json
import re
import sys
from pathlib import Path


def remove_think_block(text: str) -> str:
    """
    Remove content between <think> and </think>\n from the text.
    
    Args:
        text: Input text potentially containing <think>...</think> blocks
        
    Returns:
        Text with <think>...</think>\n blocks removed
    """
    # Pattern to match <think>...</think>\n
    # Using re.DOTALL to match across newlines
    pattern = r'<think>.*?</think>\n'
    cleaned_text = re.sub(pattern, '', text, flags=re.DOTALL)
    return cleaned_text


def process_jsonl(input_path: str, output_path: str):
    """
    Process JSONL file to remove <think> blocks from raw_response fields.
    
    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file
    """
    input_file = Path(input_path)
    output_file = Path(output_path)
    
    if not input_file.exists():
        print(f"Error: Input file {input_path} does not exist")
        sys.exit(1)
    
    processed_count = 0
    modified_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                data = json.loads(line)
                
                # Check if raw_response field exists
                if 'raw_response' in data and data['raw_response']:
                    original = data['raw_response']
                    cleaned = remove_think_block(original)
                    
                    if original != cleaned:
                        modified_count += 1
                        data['raw_response'] = cleaned
                
                # Write to output file
                outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                processed_count += 1
                
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num}: {e}")
                continue
    
    print(f"Processing complete:")
    print(f"  Total lines processed: {processed_count}")
    print(f"  Lines modified: {modified_count}")
    print(f"  Output written to: {output_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python remove_think_blocks.py <input_jsonl> [output_jsonl]")
        print("\nIf output_jsonl is not provided, it will be named as input_name_nothink.jsonl")
        sys.exit(1)
    
    input_path = sys.argv[1]
    
    # Generate output path if not provided
    if len(sys.argv) >= 3:
        output_path = sys.argv[2]
    else:
        input_file = Path(input_path)
        output_path = input_file.parent / f"{input_file.stem}_nothink{input_file.suffix}"
    
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print()
    
    process_jsonl(input_path, str(output_path))


if __name__ == "__main__":
    main()
