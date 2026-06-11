import json
import sys
import os

def fix_bboxes(data, field):
    bboxes = data.get(field, [])
    if not bboxes:
        return False
    
    modified = False
    new_bboxes = []
    
    for box in bboxes:
        new_box = []
        box_modified = False
        for val in box:
            if isinstance(val, float):
                modified = True
                box_modified = True
                if val <= 1.0:
                    new_box.append(int(round(val * 1000)))
                else:
                    new_box.append(int(round(val)))
            else:
                new_box.append(val)
        new_bboxes.append(new_box)
    
    if modified:
        data[field] = new_bboxes
        
    return modified

def fix_file(input_path):
    temp_path = input_path + ".tmp"
    count_fixed = 0
    
    print(f"Processing {input_path} ...")
    
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(temp_path, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                f_out.write(line)
                continue
            
            modified = False
            if fix_bboxes(data, "gt_misalignment_bboxes"):
                modified = True
            
            if fix_bboxes(data, "gt_artifact_bboxes"):
                modified = True
            
            # Write back
            f_out.write(json.dumps(data) + "\n")
            if modified:
                count_fixed += 1
                if count_fixed < 5:
                    print(f"Fixed line example: {data.get('gt_misalignment_bboxes')} {data.get('gt_artifact_bboxes')}")

    print(f"Fixed {count_fixed} lines.")
    print(f"Replacing original file...")
    os.replace(temp_path, input_path)
    print("Done.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fix.py <jsonl_file>")
        sys.exit(1)
        
    fix_file(sys.argv[1])
