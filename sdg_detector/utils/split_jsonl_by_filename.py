import json
import sys
import os

def split_jsonl(input_file):
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found.")
        return

    base_name, ext = os.path.splitext(input_file)
    output_richhf = f"{base_name}_richhf{ext}"
    output_other = f"{base_name}_other{ext}"

    count_richhf = 0
    count_other = 0

    print(f"Splitting '{input_file}'...")

    try:
        with open(input_file, 'r', encoding='utf-8') as f_in, \
             open(output_richhf, 'w', encoding='utf-8') as f_rich, \
             open(output_other, 'w', encoding='utf-8') as f_other:
            
            for line in f_in:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    filename = data.get("filename", "")
                    
                    # Check if "RichHF" is in the filename string
                    if "RichHF" in filename:
                        f_rich.write(line + "\n")
                        count_richhf += 1
                    else:
                        f_other.write(line + "\n")
                        count_other += 1
                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON line: {line[:50]}...")
                    continue
        
        print(f"Done.")
        print(f"Total processed: {count_richhf + count_other}")
        print(f"RichHF lines:    {count_richhf} -> {output_richhf}")
        print(f"Other lines:     {count_other} -> {output_other}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python split_jsonl_by_filename.py <input_jsonl_file>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    split_jsonl(input_path)
