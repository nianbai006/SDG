import argparse
import os
import json


def extract_fields(input_file, output_file):
    """
    Reads a JSONL file and extracts specific fields into a new JSONL file.
    Fields to extract: filename, caption, gemini_misalignment_bboxes, gemini_artifact_bboxes, raw_response
    """
    print(f"Reading from {input_file}...")
    
    extracted_count = 0
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            try:
                data = json.loads(line)
                
                new_record = {
                    "filename": data.get("filename"),
                    "caption": data.get("caption"),
                    "gemini_misalignment_bboxes": data.get("gemini_misalignment_bboxes"),
                    "gemini_artifact_bboxes": data.get("gemini_artifact_bboxes"),
                    "raw_response": data.get("raw_response")
                }
                
                outfile.write(json.dumps(new_record, ensure_ascii=False) + "\n")
                extracted_count += 1
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line")
                continue

    print(f"Successfully extracted {extracted_count} records to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract specific fields from a JSONL file.")
    parser.add_argument("--input_file", help="Path to the input JSONL file",default="${SDG_DATA}/sdg30k/RichHF/train_gemini_bbox.jsonl")
    parser.add_argument("--output_file", help="Path to the output JSONL file",default="${SDG_DATA}/sdg30k/RichHF/train.jsonl")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist.")
    else:
        extract_fields(args.input_file, args.output_file)
