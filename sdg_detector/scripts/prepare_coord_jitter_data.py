#!/usr/bin/env python3
"""
Generate multi-epoch SFT data with coordinate jitter.

Each epoch applies random offsets to box_2d coords inside <answer>:
  - offset range: ±jitter (default ±10, 0-1000 scale)
  - each epoch uses a different random seed
  - coords are clamped to [0, 1000] with x0 < x1, y0 < y1
  - <think> block is unchanged (no coordinates)

Output: a single jsonl that concatenates `num_epochs` jittered copies.
Train with num_train_epochs=1 since the epochs are already pre-baked.
"""

import argparse
import json
import os
import random
import re


def jitter_bbox_in_answer(response, jitter_range, rng):
    """Apply a random offset to every box_2d inside the <answer> JSON."""
    m = re.search(r'(<answer>)(.*?)(</answer>)', response, re.DOTALL)
    if not m:
        return response
    try:
        bboxes = json.loads(m.group(2).strip())
    except:
        return response
    if not isinstance(bboxes, list):
        return response

    new_bboxes = []
    for b in bboxes:
        if not isinstance(b, dict) or 'box_2d' not in b:
            new_bboxes.append(b)
            continue
        box = list(b['box_2d'])
        # Apply jitter to each coordinate
        for i in range(4):
            box[i] = box[i] + rng.randint(-jitter_range, jitter_range)
            box[i] = max(0, min(1000, box[i]))
        # Ensure x0 < x1, y0 < y1
        if box[0] >= box[2]:
            box[0], box[2] = min(box[0], box[2]), max(box[0], box[2])
            if box[0] == box[2]:
                box[2] = min(box[0] + 1, 1000)
        if box[1] >= box[3]:
            box[1], box[3] = min(box[1], box[3]), max(box[1], box[3])
            if box[1] == box[3]:
                box[3] = min(box[1] + 1, 1000)
        new_b = dict(b)
        new_b['box_2d'] = box
        new_bboxes.append(new_b)

    new_json = json.dumps(new_bboxes, ensure_ascii=False, indent=2)
    return response[:m.start(2)] + "\n" + new_json + "\n" + response[m.end(2):]


def main():
    parser = argparse.ArgumentParser(
        description="Generate coordinate-jittered multi-epoch SFT data")
    parser.add_argument("--input_sft", type=str, required=True,
                        help="Standard SFT jsonl (single epoch)")
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number    of jittered epochs to generate")
    parser.add_argument("--jitter", type=int, default=10,
                        help="Max coordinate jitter per axis (0-1000 scale)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epoch0_original", action="store_true", default=True,
                        help="First epoch uses original coords (no jitter)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_jsonl) or '.', exist_ok=True)

    # Load all data
    data = []
    with open(args.input_sft) as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    print(f"Input: {len(data)} samples")
    print(f"Epochs: {args.num_epochs}, Jitter: ±{args.jitter}")
    print(f"Epoch 0 original: {args.epoch0_original}")

    total = 0
    jittered_count = 0

    with open(args.output_jsonl, 'w') as fout:
        for epoch in range(args.num_epochs):
            rng = random.Random(args.seed + epoch)
            is_original = (epoch == 0 and args.epoch0_original)

            for d in data:
                new_d = dict(d)
                msgs = list(new_d['messages'])
                new_msgs = []

                for msg in msgs:
                    new_msg = dict(msg)
                    if msg['role'] == 'assistant' and not is_original:
                        new_msg['content'] = jitter_bbox_in_answer(
                            msg['content'], args.jitter, rng)
                        jittered_count += 1
                    new_msgs.append(new_msg)

                new_d['messages'] = new_msgs
                fout.write(json.dumps(new_d, ensure_ascii=False) + '\n')
                total += 1

    print(f"Output: {total} samples ({args.num_epochs} epochs × {len(data)})")
    print(f"  Epoch 0 (original): {len(data)} samples")
    print(f"  Epoch 1-{args.num_epochs-1} (jittered): {jittered_count} assistant responses jittered")
    print(f"  -> {args.output_jsonl}")
    print(f"\nNote: Train with --num_train_epochs 1 (data already has {args.num_epochs} epochs)")


if __name__ == "__main__":
    main()
