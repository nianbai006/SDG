"""
Reflection editing project — global config
"""
import os
from pathlib import Path

# ==================== Paths ====================
PROJECT_ROOT = Path(__file__).parent

# data paths
SDG_ROOT = Path(os.environ.get(
    "SDG_ROOT",
    os.environ.get("SDG_DATA", "data")
))
TEST_DATA_PATH = SDG_ROOT / "sdg30k" / "annotations" / "merged_all_filtered_distilled_filepath_test.jsonl"
IMAGEDOCTOR_PREDICTIONS_PATH = PROJECT_ROOT / "outputs" / "mode3_imagedoctor" / "predictions.jsonl"
SDG_PREDICTIONS_PATH = Path(os.environ.get(
    "SDG_PREDICTIONS_PATH",
    str(PROJECT_ROOT / "outputs" / "mode4_sdg" / "predictions.jsonl")
))

# output directory
OUTPUT_ROOT = Path(os.environ.get(
    "OUTPUT_ROOT",
    str(PROJECT_ROOT / "outputs")
))

# ==================== Qwen Edit config ====================
QWEN_EDIT_MODEL_PATH = os.environ.get(
    "QWEN_EDIT_MODEL_PATH",
    "Qwen/Qwen-Image-Edit-2511"
)
QWEN_EDIT_DEVICE = os.environ.get("QWEN_EDIT_DEVICE", "cuda")
QWEN_EDIT_DTYPE = "bfloat16"

# default edit arguments
DEFAULT_NOISE_LEVEL = 1.0
DEFAULT_NUM_INFERENCE_STEPS = 40
DEFAULT_TRUE_CFG_SCALE = 4.0
DEFAULT_IMAGE_SIZE = 512
DEFAULT_SEED = 42

# ==================== generic VLM config (mode2: Qwen3-VL-8B) ====================
VLM_MODEL_PATH = os.environ.get(
    "VLM_MODEL_PATH",
    "Qwen/Qwen3-VL-8B-Instruct"
)
VLM_SERVER_URL = os.environ.get(
    "VLM_SERVER_URL",
    "http://localhost:17140/v1"
)
VLM_MODEL_NAME = os.environ.get(
    "VLM_MODEL_NAME",
    "Qwen3-VL-8B-Instruct"
)
VLM_MAX_NEW_TOKENS = 2048
VLM_TEMPERATURE = 0.0
VLM_TP_SIZE = int(os.environ.get("VLM_TP_SIZE", "1"))

# ==================== ImageDoctor config (mode3) ====================
IMAGEDOCTOR_MODEL_PATH = os.environ.get(
    "IMAGEDOCTOR_MODEL_PATH",
    "GYX97/ImageDoctor"
)
IMAGEDOCTOR_SERVER_URL = os.environ.get(
    "IMAGEDOCTOR_SERVER_URL",
    "http://localhost:17141/v1"
)
IMAGEDOCTOR_MODEL_NAME = os.environ.get(
    "IMAGEDOCTOR_MODEL_NAME",
    "ImageDoctor"
)
IMAGEDOCTOR_NUM_GPUS = int(os.environ.get("IMAGEDOCTOR_NUM_GPUS", "1"))

# ==================== SDG config (mode4) ====================
SDG_MODEL_PATH = os.environ.get(
    "SDG_MODEL_PATH",
    "${SDG_CKPT}/sdg_detector_merged"
)
SDG_SERVER_URL = os.environ.get(
    "SDG_SERVER_URL",
    "http://localhost:17142/v1"
)
SDG_MODEL_NAME = os.environ.get(
    "SDG_MODEL_NAME",
    "sdg-detector"
)
SDG_TP_SIZE = int(os.environ.get("SDG_TP_SIZE", "8"))

# ==================== Gemini config (mode5) ====================
GEMINI_MODEL_NAME = os.environ.get(
    "GEMINI_MODEL_NAME",
    "gemini-3.1-pro-preview"
)
GEMINI_PROJECT = os.environ.get(
    "GEMINI_PROJECT",
    ""
)
GEMINI_LOCATION = os.environ.get(
    "GEMINI_LOCATION",
    "global"
)
GEMINI_CREDENTIALS = os.environ.get(
    "GOOGLE_APPLICATION_CREDENTIALS",
    ""
)
GEMINI_CONCURRENCY = int(os.environ.get("GEMINI_CONCURRENCY", "50"))

# ==================== ImageDoctor Prompt ====================
IMAGEDOCTOR_PROMPT = """Given a caption and an image generated based on this caption, please analyze the provided image in detail. Evaluate it on various dimensions including Semantic Alignment (How well the image content corresponds to the caption), Aesthetics (composition, color usage, and overall artistic quality), Plausibility (realism and attention to detail), and Overall Impression (General subjective assessment    of the image's quality). For each evaluation dimension, provide a score between 0-1 and provide a concise rationale for the score. Use a chain- of -thought process to detail your reasoning steps, and enclose all potential important areas and detailed reasoning within <think> and </think> tags. The important areas are represented in following format: " I need to focus on the bounding box area. Proposed regions (xyxy): ..., which is an enumerated list in the exact format:1.[x1,y1,x2,y2];\\n2.[x1,y1,x2,y2];\\n3.[x1,y1,x2,y2]... Here, x1,y1 is the top-left corner, and x2,y2 is the bottom-right corner. Then, within the <answer> and </answer> tags, summarize your assessment in the following format: "Semantic Alignment score: ... \\nMisalignment Locations: ...\\nAesthetic score: ...\\nPlausibility score: ... nArtifact Locations: ...\\nOverall Impression score: ...". No additional text is allowed in the answer section.\\n\\n Your actual evaluation should be based on the quality    of the provided image.**\\n\\nYour task is provided as follows:\\nText Caption: [{caption}]"""

# ==================== SDG Prompt ====================
# Question template used by SDG (think variant)
SDG_SYSTEM_PROMPT = "You are a helpful assistant. "
SDG_QUESTION_TEMPLATE =  """You are an AI image quality evaluator. You will be given **one image** to analyze.

### Definitions

**Misalignment**: Areas where the image content does NOT match the text caption, including:
- Missing objects: Objects mentioned in caption but not present in image
- Extra objects: Objects present in image but not mentioned in caption
- Wrong attributes: Incorrect color, size, material, count, or other properties
- Wrong spatial relationships: Incorrect positions, orientations, or arrangements

**Artifact**: Visual defects in the generated image, including:
- Distorted anatomy: Malformed hands, extra/missing limbs, wrong number    of fingers
- Duplicated/missing parts: Repeated or absent body parts, objects
- Warped geometry: Perspective errors, impossible shapes
- Texture issues: Melted, smeared, or overly smooth textures
- Unnatural edges: Jagged, broken, or blurry boundaries
- Garbled text: Unreadable or malformed text/letters
- Lighting inconsistencies: Wrong shadows, reflections, or light sources

Text Caption: {caption}

**Goal**: Produce the full detailed think process and infer the bounding boxes.

### Strict Output Rules
Output **TWO blocks in this exact order**:
1) `<think>` - Your analysis (must contain TWO clearly labeled sections)
2) `<answer>` - JSON list    of bounding boxes

**IMPORTANT**: In your output, do NOT mention or refer to the provided reference boxes. Analyze the image as if you discovered the issues yourself.

### Think Format (Follow the numbered steps and headings below):
<think>
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

### Answer Format (for <answer>):
Return a JSON list:
[
    {{"box_2d": [x0, y0, x1, y1], "label": "misalignment"|"artifact", "desc": "short description"}}
]

Bounding box coordinates are in normalized 0-1000 space: [x0, y0, x1, y1].
If there are no issues, output an empty list.

Now analyze the image and produce your output:
"""

# ==================== generic VLM eval Prompt (mode2) ====================
VLM_EVAL_SYSTEM_PROMPT = "You are a helpful assistant."
VLM_EVAL_USER_PROMPT = """Analyze this AI-generated image and identify specific visual problems that need to be fixed.

Caption: {caption}

Please identify:
1. Any misalignment between the image and the caption (missing/wrong objects, wrong attributes)
2. Any visual artifacts (distorted hands, blurry areas, unrealistic textures)

For each problem, describe:
- What the problem is
- Where it is located in the image (e.g., upper-left, center, etc.)

Be concise and specific. Format your response as a numbered list    of problems."""

# ==================== Edit prompt template ====================
FIXED_EDIT_PROMPT = (
    "Improve the image quality, fix any visual artifacts, distortions, "
    "and ensure the image accurately matches the following description: {caption}"
)

FEEDBACK_EDIT_PROMPT = (
    "Fix the following issues in this image: {issues}. "
    "The image should match: {caption}"
)

# ==================== runconfig ====================
DEFAULT_BATCH_SIZE = 4
DEFAULT_MAX_SAMPLES = None
DEFAULT_NUM_ROUNDS = 1
