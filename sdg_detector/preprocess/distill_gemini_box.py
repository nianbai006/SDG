"""
Distill thinking content using the Gemini API
Distillation logic mirrors distillqwen.py; Gemini config follows ditillgemini.py
"""

import os
import json
import time
import ast
import argparse
from tqdm import tqdm
from PIL import Image
from io import BytesIO
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from google import genai
from google.genai import types

# Configure Google credentials
# Prompt template
PROMPT_TEMPLATE = """(keep old if you want)"""
PROMPT_TEMPLATEv2 = """You are a multi-modal assistant. You will be given **three images in a fixed order**:
1) **Original Image**: the generated image to inspect (this is the ONLY image you should describe).
2) **Artifact Guide Image**: a visual hint that helps you locate likely artifact regions.
3) **Misalignment Guide Image**: a visual hint that helps you locate likely caption-image mismatch regions.

**Goal**: Produce a **high-quality step-by-step reasoning** that a careful evaluator would follow to identify **all artifact regions** and **all caption-image misalignment regions** in the **Original Image**, and then output the corresponding bounding boxes.

Text Caption: {question}

### Strict Output Rules
- Output **ONLY TWO blocks in this exact order**:
  1) `<think> ... </think>`
  2) `<answer> ... </answer>`
- Do **NOT** output anything else.
- Do **NOT** mention guide images, heatmaps, overlays, hints, or “the second/third image”.
- In `<think>`:
  - Describe only what is visible in the **Original Image**.
  - Do **NOT** include coordinates, bounding boxes, JSON, or labels.
- In `<answer>`:
  - Output **ONLY** a list    of bounding boxes in the format:
    `[[x1, y1, x2, y2], ...]`
  - If **no artifact or misalignment region exists**, output exactly:
    `[[0, 0, 0, 0]]`

### Reasoning Requirements (for `<think>`, step-by-step, natural language)

1) **Caption checklist**
   - Break the caption into key elements:
     main subjects, attributes (color/material/count), actions, scene context, and spatial relations.

2) **Global match scan**
   - Verify each element against the Original Image.
   - Identify missing objects, extra objects, incorrect attributes, or incorrect relationships as **misalignment candidates**.

3) **Artifact scan**
   - Carefully inspect the image for visual artifacts, including but not limited to:
     - distorted or implausible anatomy,
     - duplicated or missing body/object parts,
     - warped geometry or perspective,
     - melted, smeared, or overly smooth textures,
     - jagged, broken, or unnatural edges,
     - garbled or unreadable text,
     - inconsistent lighting or shadows,
     - broken or impossible reflections,
     - physically impossible interactions.

4) **Region-by-region reasoning (must be exhaustive)**
   - For **every suspicious region**, explicitly describe:
     - what you see (the visual symptom),
     - why it is problematic (artifact or caption-image misalignment),
     - roughly where it is in the scene (e.g., “around the left hand”, “top-right background”, “near the sign text”), **without using coordinates**.

### Final Output Instructions
- Put the complete reasoning process in `<think>...</think>`.
- Put the final bounding box list in `<answer>...</answer>`.
- Ensure the number    of regions described in `<think>` matches the number    of boxes in `<answer>`.

Now produce your output.
"""

PROMPT_TEMPLATEv3 = """You are a multi-modal assistant. You will be given **three images in a fixed order**:
1) **Original Image**: the generated image to inspect (this is the ONLY image you should describe).
2) **Artifact Guide Image**: a visual hint that helps you locate likely artifact regions.
3) **Misalignment Guide Image**: a visual hint that helps you locate likely caption-image mismatch regions.

**Goal**: Produce a **high-quality step-by-step reasoning** that a careful evaluator would follow to identify **all artifact regions** and **all caption-image misalignment regions** in the **Original Image**, and then output the corresponding bounding boxes.

Text Caption: {question}

### Strict Output Rules
- Output **ONLY TWO blocks in this exact order**:
  1) `<think> ... </think>`
  2) `<answer> ... </answer>`
- Do **NOT** output anything else.
- Do **NOT** mention guide images, heatmaps, overlays, hints, or “the second/third image”.
- In `<think>`:
  - Describe only what is visible in the **Original Image**.
  - Do **NOT** include coordinates, bounding boxes, JSON, or labels.
- In `<answer>`:
  - Output **ONLY** a list    of bounding boxes in the format:
    `[[y0, x0, y1, x1], ...]`
  - **Coordinate System (IMPORTANT)**:
    - Use **normalized 2D bounding boxes** in the **0–1000** coordinate space.
    - `y` is the vertical axis (top=0, bottom=1000); `x` is the horizontal axis (left=0, right=1000).
    - Each box must satisfy: `0 <= y0 < y1 <= 1000` and `0 <= x0 < x1 <= 1000`.
    - Use **integers** (round to nearest int).
  - If **no artifact or misalignment region exists**, output exactly:
    `[[0, 0, 0, 0]]`

### Reasoning Requirements (for `<think>`, step-by-step, natural language)

1) **Caption checklist**
   - Break the caption into key elements:
     main subjects, attributes (color/material/count), actions, scene context, and spatial relations.

2) **Global match scan**
   - Verify each element against the Original Image.
   - Identify missing objects, extra objects, incorrect attributes, or incorrect relationships as **misalignment candidates**.

3) **Artifact scan**
   - Carefully inspect the image for visual artifacts, including but not limited to:
     - distorted or implausible anatomy,
     - duplicated or missing body/object parts,
     - warped geometry or perspective,
     - melted, smeared, or overly smooth textures,
     - jagged, broken, or unnatural edges,
     - garbled or unreadable text,
     - inconsistent lighting or shadows,
     - broken or impossible reflections,
     - physically impossible interactions.

4) **Region-by-region reasoning (must be exhaustive)**
   - For **every suspicious region**, explicitly describe:
     - what you see (the visual symptom),
     - why it is problematic (artifact or caption-image misalignment),
     - roughly where it is in the scene (e.g., “around the left hand”, “top-right background”, “near the sign text”), **without using coordinates**.

### Final Output Instructions
- Put the complete reasoning process in `<think>...</think>`.
- Put the final bounding box list in `<answer>...</answer>`.
- Ensure the number    of regions described in `<think>` matches the number    of boxes in `<answer>`.

Now produce your output.
"""

PROMPT_TEMPLATEv4 = PROMPT_TEMPLATEv3


PROMPT_TEMPLATE = PROMPT_TEMPLATEv3

def local_image_to_bytes(image_path, max_size=1024):
    """Convert a local image to bytes and resize"""
    with Image.open(image_path) as img:
        width, height = img.size
        if max(width, height) > max_size:
            if width > height:
                new_width = max_size
                new_height = int(height * max_size / width)
            else:
                new_height = max_size
                new_width = int(width * max_size / height)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        buffer = BytesIO()
        img.save(buffer, format='PNG')
        return buffer.getvalue()


def extract_tag_block(text: str, tag: str):
    """Extract the contents inside <tag>...</tag>; return None if absent"""
    if not text:
        return None
    start_tag = f"<{tag}>"
    end_tag = f"</{tag}>"
    s = text.find(start_tag)
    e = text.find(end_tag)
    if s == -1 or e == -1 or e <= s:
        return None
    return text[s + len(start_tag): e].strip()


def extract_think_content(response_text: str):
    """Extract the <think> contents; fall back to the whole text if absent"""
    if response_text is None:
        return None
    think = extract_tag_block(response_text, "think")
    return think if think is not None else response_text.strip()


def parse_bboxes_from_answer(answer_text: str):
    """
    answer_text is expected to be [[x1,y1,x2,y2], ...] or [[0,0,0,0]]
    Returns None on parse failure
    """
    if answer_text is None:
        return None
    s = answer_text.strip()

    # First try json.loads
    try:
        b = json.loads(s)
        return b
    except Exception:
        pass

    # Then try the python literal evaluator
    try:
        b = ast.literal_eval(s)
        return b
    except Exception:
        return None


def extract_answer_bboxes(response_text: str):
    """Extract and parse the bbox list from <answer>...</answer>"""
    ans = extract_tag_block(response_text, "answer")
    if ans is None:
        return None
    return parse_bboxes_from_answer(ans)


class ThinkingDataset(Dataset):
    """Dataset used to distill thinking content"""

    def __init__(self, input_jsonl, output_jsonl, model_name="gemini-3-pro-preview",
                 project="", location="global", debug=False, num_samples=10,
                 max_retries=3, retry_sleep=2.0):
        self.input_jsonl = input_jsonl
        self.output_jsonl = output_jsonl
        self.model_name = model_name
        self.project = project
        self.location = location
        self.debug = debug
        self.max_retries = max_retries
        self.retry_sleep = retry_sleep

        # Note: do NOT create the client in __init__ (DataLoader multiprocessing forks/pickles it)
        self.client = None

        print(f"Reading from {input_jsonl}...")
        with open(input_jsonl, 'r') as f:
            self.lines = f.readlines()

        if debug:
            self.lines = self.lines[:num_samples]
            print(f"Debug mode: processing first {len(self.lines)} samples.")

        self.output_dir = os.path.dirname(output_jsonl)
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)

    def _get_client(self):
        if self.client is None:
            self.client = genai.Client(
                vertexai=True,
                project=self.project,
                location=self.location,
            )
        return self.client

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        """Process a single sample (Gemini call included)"""
        try:
            data = json.loads(self.lines[index])

            original_image = data.get("filename")
            artifact_image = data.get("artifact_map_path")
            misalignment_image = data.get("misalignment_map_path")
            caption = data.get("caption", "")

            # Verify the file exists
            for path in [original_image, artifact_image, misalignment_image]:
                if not path or not os.path.exists(path):
                    if self.debug:
                        print(f"Warning: Missing file for sample {index}: {path}")
                    return None

            # use the prompt
            filled_prompt = PROMPT_TEMPLATE.format(question=caption)

            # Build contents
            content_parts = [filled_prompt]
            for img_path in [original_image, artifact_image, misalignment_image]:
                image_bytes = local_image_to_bytes(img_path)
                content_parts.append(types.Part.from_bytes(mime_type='image/png', data=image_bytes))

            client = self._get_client()

            # Call Gemini (with simple retry)
            last_err = None
            response = None
            for attempt in range(self.max_retries):
                try:
                    response = client.models.generate_content(
                        model=self.model_name,
                        contents=content_parts,
                    )
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    time.sleep(self.retry_sleep * (attempt + 1))

            if response is None or not getattr(response, "text", None):
                if self.debug:
                    print(f"No result for sample {index}. last_err={last_err}")
                return None

            raw_text = response.text

            # Extract think + answer
            think_content = extract_think_content(raw_text)
            gemini_bboxes = extract_answer_bboxes(raw_text)

            # Even if parsing the answer fails, keep the raw response for downstream debugging
            # data["think"] = think_content
            data["gemini_bboxes"] = gemini_bboxes 
            data["raw_response"] = raw_text

            return data

        except Exception as e:
            if self.debug:
                print(f"Error processing sample {index}: {e}")
            return None


def collate_fn(batch):
    """Filter out None values"""
    return [item for item in batch if item is not None]


def main():
    parser = argparse.ArgumentParser(description="Distill thinking content using Gemini API.")
    parser.add_argument("--input_jsonl", type=str, default="${SDG_DATA}/sdg30k/test/test_with_bboxes.jsonl",
                        help="Path to input JSONL file")
    parser.add_argument("--output_jsonl", type=str, default="${SDG_DATA}/sdg30k/test/test_with_think_gemini_prov3.jsonl",
                        help="Path to output JSONL file")
    parser.add_argument("--model", type=str, default="gemini-3-pro-preview",
                        help="Gemini model name")
    parser.add_argument("--project", type=str, default=os.environ.get("GEMINI_PROJECT", ""),
                        help="Google Cloud project ID")
    parser.add_argument("--location", type=str, default="global",
                        help="Google Cloud location")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number    of samples to process in debug mode")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for processing")
    parser.add_argument("--num_workers", type=int, default=10,
                        help="Number    of worker processes")
    parser.add_argument("--distributed", action="store_true",
                        help="Use distributed processing")

    args = parser.parse_args()

    dataset = ThinkingDataset(
        input_jsonl=args.input_jsonl,
        output_jsonl=args.output_jsonl,
        model_name=args.model,
        project=args.project,
        location=args.location,
        debug=args.debug,
        num_samples=args.num_samples
    )

    if args.distributed:
        try:
            from util import get_mpi_info
            rank, size, local_rank = get_mpi_info()
            sampler = DistributedSampler(dataset, num_replicas=size, rank=rank, shuffle=False)
            print(f"Using distributed processing: rank={rank}, size={size}")
        except ImportError:
            print("Warning: util.get_mpi_info not available, using single process")
            sampler = None
    else:
        sampler = None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler=sampler,
        shuffle=False,
        collate_fn=collate_fn,
    )

    results = []
    total_processed = 0
    st_time = time.time()

    print("Starting processing...")
    print(f"Model: {args.model}")
    print(f"Project: {args.project}")
    print(f"Prompt: {PROMPT_TEMPLATE} (think + answer)")

    for batch in tqdm(dataloader, desc="Processing batches"):
        results.extend(batch)
        total_processed += len(batch)

        if total_processed and total_processed % 100 == 0:
            print(f"Already processed {total_processed} samples")

    total_time = time.time() - st_time

    with open(args.output_jsonl, 'w') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(results)} results to {args.output_jsonl}")
    print(f"Failed: {len(dataset) - len(results)} samples")
    print(f"Total time: {total_time:.2f}s")
    if total_processed:
        print(f"Average time: {total_time/total_processed:.2f}s per sample")


if __name__ == "__main__":
    main()
