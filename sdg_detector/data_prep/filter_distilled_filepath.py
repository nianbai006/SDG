#!/usr/bin/env python3
"""
Find records in `source_file` that are NOT in `reference_file` based on `filepath`.
Usage:
    python filter_distilled_filepath.py --source all_filtered.jsonl --reference merged_all_filtered_distilled.jsonl --output missing_in_distilled.jsonl
"""
import argparse
import json
import os
import sys

def load_filepaths(jsonl_path):
    filepaths = set()
    if not os.path.exists(jsonl_path):
        print(f"Warning: File not found: {jsonl_path}")
        return filepaths
        
    print(f"Loading filepaths from {jsonl_path} ...")
    count = 0
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                data = json.loads(line)
                filepath = data.get('filepath')
                if filepath:
                    filepaths.add(filepath)
                    count += 1
            except:
                pass
    print(f"  Loaded {len(filepaths)} unique filepaths from {count} records.")
    return filepaths

def main():
    parser = argparse.ArgumentParser(description="Find records in source NOT in reference.")
    parser.add_argument('--source', required=True, help="Source JSONL file (to keep items from)")
    parser.add_argument('--reference', required=True, help="Reference JSONL file (items to exclude)")
    parser.add_argument('--output', required=True, help="Output JSONL file")
    args = parser.parse_args()
    
    # 1. Load reference keys
    ref_keys = load_filepaths(args.reference)
    
    # 2. Process source
    print(f"Scanning source {args.source} ...")
    kept_count = 0
    total_scanned = 0
    
    with open(args.source, 'r', encoding='utf-8') as fin, \
         open(args.output, 'w', encoding='utf-8') as fout:
        
        for line in fin:
            line = line.strip()
            if not line: continue
            
            total_scanned += 1
            try:
                data = json.loads(line)
                filepath = data.get('filepath')
                
                # If filepath is missing or NOT in reference, keep it?
                # Usually we want to find records that are "new" or "missing"
                # If filepath is None, we probably skip or keep depending on strictness.
                # Here we assume valid records have filepath.
                
                if filepath and filepath not in ref_keys:
                    fout.write(line + '\n')
                    kept_count += 1
            except:
                pass
                
    print(f"Done.")
    print(f"Scanned: {total_scanned}")
    print(f"Kept (Not in Reference): {kept_count}")
    print(f"Output: {args.output}")

if __name__ == "__main__":
    main()
