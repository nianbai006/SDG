#!/usr/bin/env python3
"""
Compute image resolution statistics from a JSONL file where each line is a JSON object
containing at least a `filename` field pointing to an image file.

Usage:
  python tools/compute_image_resolutions.py --input /path/to/file.jsonl [--csv out.csv]

Options:
  --test   : generate a small test dataset in ./_tmp_test_images and run the analysis

Outputs summary to stdout and optionally writes a CSV    of per-image resolutions.
"""
import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
import sys

try:
    from PIL import Image
except Exception as e:
    Image = None


def analyze(jsonl_path, write_csv=None, verbose=False):
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {jsonl_path}")

    total = 0
    missing = 0
    errors = 0
    res_counter = Counter()
    per_image = []

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                if verbose:
                    print(f"Line {i}: JSON parse error: {e}", file=sys.stderr)
                errors += 1
                continue
            if 'filename' not in obj:
                if verbose:
                    print(f"Line {i}: no 'filename' field", file=sys.stderr)
                errors += 1
                continue
            fn = obj['filename']
            total += 1
            try:
                if Image is None:
                    raise ImportError('Pillow not installed (PIL)')
                with Image.open(fn) as im:
                    w, h = im.size
                res = (w, h)
                res_counter[res] += 1
                per_image.append((fn, w, h))
            except FileNotFoundError:
                missing += 1
                if verbose:
                    print(f"Missing file: {fn}", file=sys.stderr)
            except Exception as e:
                errors += 1
                if verbose:
                    print(f"Error opening {fn}: {e}", file=sys.stderr)

    # Summary
    print(f"Total file entries processed: {total}")
    print(f"Missing image files: {missing}")
    print(f"Other errors (parse/open): {errors}")
    print(f"Unique resolutions: {len(res_counter)}")

    if res_counter:
        print('\nTop resolutions:')
        for res, cnt in res_counter.most_common(20):
            print(f"  {res[0]}x{res[1]} : {cnt}")

    if write_csv:
        import csv
        with open(write_csv, 'w', newline='', encoding='utf-8') as outf:
            writer = csv.writer(outf)
            writer.writerow(['filename','width','height'])
            for fn,w,h in per_image:
                writer.writerow([fn,w,h])
        print(f"Wrote per-image csv to: {write_csv}")

    # Return a dict for programmatic use
    return {
        'total': total,
        'missing': missing,
        'errors': errors,
        'unique_resolutions': len(res_counter),
        'res_counter': res_counter,
    }


def make_test_dataset(out_dir):
    """Create a small test dataset    of images and a JSONL file."""
    if Image is None:
        raise ImportError('Pillow not installed; required to run test')
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    images = [ (64,64), (128,128), (128,128), (256,128), (192,256) ]
    jsonl_path = out_dir / 'test.jsonl'
    with open(jsonl_path, 'w', encoding='utf-8') as jf:
        for i, (w,h) in enumerate(images, 1):
            fn = out_dir / f'image_{i}_{w}x{h}.png'
            im = Image.new('RGB', (w,h), (i*40 % 255, i*80 % 255, i*120 % 255))
            im.save(fn)
            jf.write(json.dumps({'filename': str(fn)}) + '\n')
        # add a missing file entry
        jf.write(json.dumps({'filename': str(out_dir / 'missing.png')}) + '\n')
    return jsonl_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', '-i', help='Path to JSONL file (each line must be JSON with a filename key)')
    p.add_argument('--csv', help='Optional output CSV path for per-image resolutions')
    p.add_argument('--test', action='store_true', help='Create a small test dataset and analyze it')
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args()

    if args.test:
        test_dir = Path.cwd() / '_tmp_test_images'
        print(f"Creating test dataset in: {test_dir}")
        jsonl = make_test_dataset(test_dir)
        print(f"Running analysis on: {jsonl}")
        analyze(jsonl, write_csv=args.csv, verbose=args.verbose)
        return

    if not args.input:
        p.print_help()
        return

    analyze(args.input, write_csv=args.csv, verbose=args.verbose)


if __name__ == '__main__':
    main()
