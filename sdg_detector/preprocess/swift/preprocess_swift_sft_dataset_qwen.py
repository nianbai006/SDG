import json
import os
import argparse
from PIL import Image
from tqdm import tqdm

# Define templates(kept consistent with GRPO)
Mix_Dataset_TEMPLATE = """You are an AI image quality evaluator. You will be given **one image** to analyze.

### Definitions

**Misalignment**: Areas where the image content does NOT match the text caption, including:
- Missing objects: Objects mentioned in caption but not present in image
- Extra objects: Objects present in image but not mentioned in caption
- Wrong attributes: Incorrect color, size, material, count, or other properties
- Wrong spatial relationships: Incorrect positions, orientations, or arrangements

**Artifact**: Visual defects in images, including:
- Distorted anatomy: Malformed hands, extra/missing limbs, wrong number    of fingers
- Duplicated/missing parts: Repeated or absent body parts, objects
- Warped geometry: Perspective errors, impossible shapes
- Texture issues: Melted, smeared, or overly smooth textures
- Unnatural edges: Jagged, broken, or blurry boundaries
- Garbled text: Unreadable or malformed text/letters
- Lighting inconsistencies: Wrong shadows, reflections, or light sources

Text Caption: {caption}

**Goal**: Carefully analyze this image to verify image quality and caption alignment. Examine every detail thoroughly to determine if there are any artifacts or misalignments.

### Strict Output Rules
Output **TWO blocks in this exact order**:
1) `<think>` - Your detailed analysis (must contain TWO clearly labeled sections)
2) `<answer>` - Bounding boxes in JSON list format (empty list if no issues found)

### Think Format (MUST have two sections with DETAILED analysis):
<think>
## Misalignment Analysis
1. **Caption breakdown**: List all key elements mentioned in the caption (objects, attributes, actions, relationships).
2. **Systematic check**: For each element, verify if it matches the image:
   - Object presence: Is it there? Is there anything extra?
   - Attributes: Color, size, count, material - are they correct?
   - Spatial relationships: Positions, orientations - are they correct?
3. **Findings**: Describe each mismatch in detail with its location in the image, or confirm no misalignments.

## Artifact Analysis
1. **Region-by-region scan**: Examine different parts    of the image systematically.
2. **Check each category**:
   - Anatomy: Any distorted hands, faces, limbs, fingers?
   - Geometry: Any warped shapes, impossible perspectives?
   - Textures: Any melted, smeared, or unnatural surfaces?
   - Edges: Any jagged, broken, or blurry boundaries?
   - Text: Any garbled or unreadable text?
   - Lighting: Any inconsistent shadows or reflections?
3. **Findings**: Describe each artifact in detail with its location, or confirm no artifacts found.
</think>

### Bounding Box Format (for <answer>):
A JSON list containing ALL detected issues. Each box must have THREE fields:
- "box_2d": [x0, y0, x1, y1] in normalized 0-1000 coordinate space
- "label": "misalignment" or "artifact"
- "description": A short description    of the issue (max 10 words)

### Example Output
<think>
## Misalignment Analysis
1. **Caption breakdown**: "Two cats watering roses in a greenhouse"
   - Objects: two cats, roses, greenhouse
   - Actions: watering
   - Count: two cats

2. **Systematic check**:
   - Cats: I can see THREE cats in the image, not two. There's an orange tabby on the left, a black-and-white cat on the right, and another cat head visible in the bottom-right corner.
   - Watering action: The cats are looking at or touching the roses, but there is no watering can, hose, or water visible. The "watering" action is not depicted.
   - Roses and greenhouse: Present and correct.

3. **Findings**:
   - Extra cat in bottom-right corner (count mismatch)
   - Missing watering action/tool in the center area

## Artifact Analysis
1. **Region-by-region scan**: Left cat, right cat, roses, background.

2. **Check each category**:
   - Anatomy: The right cat's paw reaching toward the rose appears distorted - it looks like a human hand with distinct fingers rather than a cat paw.
   - Textures: The rose stem has strange glowing blue lines that look like digital glitches.
   - Other areas appear normal.

3. **Findings**:
   - Distorted cat paw resembling human hand (center-right area)
   - Unnatural blue glitch lines on rose stem (lower center)
</think>
<answer>
[
  {{"box_2d": [200, 300, 700, 600], "label": "misalignment", "description": "car is blue not red"}},
  {{"box_2d": [150, 500, 300, 620], "label": "artifact", "description": "distorted wheel geometry"}},
  {{"box_2d": [700, 200, 900, 400], "label": "artifact", "description": "strange texture on building"}}
]
</answer>

If no issues found, output empty list in <answer>: []

Now analyze the image and produce your output:
"""

Mix_Dataset_TEMPLATEv2 = """You are an AI image quality evaluator. You will be given **one image** to analyze.

Text Caption: {caption}

### Definitions

**Misalignment**: Caption-image mismatch (missing/extra objects, wrong attributes, wrong spatial relationships)

**Artifact**: Visual defects (distorted anatomy, warped geometry, texture issues, unnatural edges, garbled text, lighting inconsistencies)


**Goal**: Carefully analyze this image to verify image quality and caption alignment. Examine every detail thoroughly to determine if there are any artifacts or misalignments.

### Strict Output Rules
Output **TWO blocks in this exact order**:
1) `<think>` - Your detailed analysis 
2) `<answer>` - Bounding boxes in JSON list format (empty list if no issues found)

### Block 1: Reasoning Process
<think>
## Image Quality Analysis

### Step 1: Caption Understanding
[Break down what the caption describes: objects, attributes, relationships, actions]

### Step 2: Misalignment Check
[Systematically compare caption vs image]
- Object presence: Are all mentioned objects present? Any extra objects?
- Attributes: Colors, sizes, counts - are they correct?
- Spatial relationships: Positions, orientations - are they correct?
- For each mismatch found: describe location and why it's wrong

### Step 3: Artifact Detection
[Scan the image region by region for visual defects]
- Anatomy check: Any distorted hands, faces, limbs?
- Geometry check: Any warped shapes, impossible perspectives?
- Texture check: Any melted, smeared, unnatural surfaces?
- Edge check: Any jagged, broken, blurry boundaries?
- Text check: Any garbled or unreadable text?
- For each artifact found: describe location and type    of defect

### Step 4: Bounding Box Placement
[For each issue identified, explain the exact coordinates]
- Issue 1: Located at [describe region], covering [x0, y0] to [x1, y1] because...
- Issue 2: ...
</think>

### Block 2: Bounding Boxes
<answer>
A JSON list containing ALL detected issues. Each box must have THREE fields:
- "box_2d": [x0, y0, x1, y1] in normalized 0-1000 coordinate space
- "label": "misalignment" or "artifact"
- "description": A short description    of the issue (max 10 words)
</answer>

### Example answer output
<answer>
[
  {{"box_2d": [x0, y0, x1, y1], "label": "misalignment" or "artifact", "description": "brief description    of the issue"}}
]
</answer>

If no issues found, explain why in <think> and output empty list in <answer>: []

Now analyze the image and produce your output:
"""

Mix_Dataset_TEMPLATE = Mix_Dataset_TEMPLATEv2

def process_item(item):
    """
    Process a single record into a Swift SFT-formatted dict
    Unlike GRPO, SFT must include the model response (raw_response)
    
    Format expected by Swift:
    - <image> placeholder used inside messages
    - image paths live in the separate `images` field
    """
    image_path = item["filename"]
    
    # Path check
    if not os.path.exists(image_path):
        # image_path = os.path.join("/your/root", image_path)
        pass

    try:
        with Image.open(image_path) as img:
            w, h = img.size
    except Exception as e:
        return None

    caption = item.get("caption", "")
    raw_response = item.get("raw_response", "")
    
    # Skip if raw_response is missing
    if not raw_response:
        return None
    
    user_text = Mix_Dataset_TEMPLATE.format(caption=caption)

    # Swift SFT format: <image> placeholder, image paths live in the separate `images` field
    messages = [
        {
            "role": "user", 
            "content": f"<image>{user_text}"  # <image> placeholder placed before the text
        },
        {
            "role": "assistant",
            "content": raw_response
        }
    ]

    return {
        "messages": messages,
        "images": [image_path],  # list    of image paths
    }

def main():
    parser = argparse.ArgumentParser(description='Preprocess dataset for Swift SFT training')
    parser.add_argument('--input', type=str, required=True,
                        help='Input JSONL file path')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSONL file path (default: {input_dir}/{input_name}_swift_sft.jsonl)')
    parser.add_argument('--concurrency', type=int, default=128,
                        help='Number    of concurrent threads')
    args = parser.parse_args()
    
    dataset_path = args.input
    
    # Auto-infer output path
    if args.output:
        output_path = args.output
    else:
        base_name = os.path.splitext(os.path.basename(dataset_path))[0]
        output_path = os.path.join(os.path.dirname(dataset_path), f"{base_name}_swift_sft.jsonl")
    
    print(f"Processing {dataset_path}...")
    print(f"Output: {output_path}")
    print(f"Concurrency: {args.concurrency} threads")
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Read all lines
    with open(dataset_path, 'r') as f_in:
        lines = [line.strip() for line in f_in.readlines() if line.strip()]
    
    print(f"Total samples: {len(lines)}")
    
    # Multi-threaded processing
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    results = []
    skipped = 0
    
    def process_line(line):
        try:
            raw_item = json.loads(line)
            return process_item(raw_item)
        except Exception as e:
            return None
    
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {executor.submit(process_line, line): i for i, line in enumerate(lines)}
        
        for future in tqdm(as_completed(futures), total=len(lines), desc="Processing"):
            result = future.result()
            if result:
                results.append(result)
            else:
                skipped += 1
    
    # Write output
    with open(output_path, 'w') as f_out:
        for item in results:
            f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Successfully saved {len(results)} samples to {output_path}")
    if skipped > 0:
        print(f"Skipped {skipped} samples (no raw_response or invalid)")

if __name__ == "__main__":
    main()
