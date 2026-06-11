import argparse
import json
import random
import os

def load_jsonl(path):
    data = []
    print(f"Loading {path}...")
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Warning: Failed to decode line in {path}")
    return data

def save_jsonl(data, path):
    print(f"Saving to {path}...")
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def main():
    parser = argparse.ArgumentParser(description="Mix Real data into Train and Test datasets.")
    parser.add_argument('--train_files', nargs='+', required=True, help="List    of train JSONL files")
    parser.add_argument('--test_files', nargs='+', required=True, help="List    of test JSONL files")
    parser.add_argument('--real_file', required=True, help="Path to real data JSONL file")
    parser.add_argument('--output_file', default="${SDG_DATA}/sdg30k/mix.jsonl", help="Output JSONL file")
    parser.add_argument('--num_real', type=int, required=True, help="Total number    of real samples to mix")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    

    # Load datasets
    train_data = []
    for p in args.train_files:
        train_data.extend(load_jsonl(p))
    
    test_data = []
    for p in args.test_files:
        test_data.extend(load_jsonl(p))
        
    real_data = load_jsonl(args.real_file)
    
    print(f"Loaded {len(train_data)} train samples.")
    print(f"Loaded {len(test_data)} test samples.")
    print(f"Loaded {len(real_data)} real samples.")
    
    # Select real samples
    if args.num_real > len(real_data):
        print(f"Warning: Requested {args.num_real} real samples, but only {len(real_data)} available. Using all.")
        selected_real = real_data
    else:
        # Shuffle real data before selecting to ensure random selection
        # (Make a copy to avoid modifying original list if we were to reuse it, though not needed here)
        real_data_shuffled = real_data[:]
        random.shuffle(real_data_shuffled)
        selected_real = real_data_shuffled[:args.num_real]
    
    print(f"Selected {len(selected_real)} real samples to mix.")

    # Normalize real data format to match train/test if needed
    # Check if train data has specific fields that real data might miss
    # Based on context, real.jsonl already has gemini_misalignment_bboxes etc.
    # But we'll ensure consistency just in case.
    
    # Calculate split based on original data ratio
    n_train = len(train_data)
    n_test = len(test_data)
    total_original = n_train + n_test
    
    if total_original == 0:
        print("Error: No train or test data found.")
        return

    train_ratio = n_train / total_original
    n_real_train = int(len(selected_real) * train_ratio)
    real_train = selected_real[:n_real_train]
    real_test = selected_real[n_real_train:]
    
    print(f"Splitting real data: {len(real_train)} for train, {len(real_test)} for test (Ratio: {train_ratio:.2f})")

    # Mix and shuffle
    mixed_train = train_data + real_train
    random.shuffle(mixed_train)
    
    mixed_test = test_data + real_test
    random.shuffle(mixed_test)
    
    # Determine output paths
    base_dir = os.path.dirname(args.output_file)
    base_name = os.path.basename(args.output_file)
    name_root, name_ext = os.path.splitext(base_name)
    
    out_train = os.path.join(base_dir, f"{name_root}_train{name_ext}")
    out_test = os.path.join(base_dir, f"{name_root}_test{name_ext}")
    
    if base_dir:
        os.makedirs(base_dir, exist_ok=True)
        
    save_jsonl(mixed_train, out_train)
    save_jsonl(mixed_test, out_test)
    
    print(f"Saved {len(mixed_train)} samples to {out_train}")
    print(f"Saved {len(mixed_test)} samples to {out_test}")

if __name__ == "__main__":
    main()
