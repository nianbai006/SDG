"""Parse the bbox list from a SDG / SDG <think>/<answer> response.

Only ``parse_bboxes_from_response`` is reachable from the GPT-Image-1.5
refinement pipeline (`edit_gpt_image.py`). The mask- and text-rendering
helpers used by the deleted Qwen-Image-Edit / inpaint variants have been
removed.
"""
import json
import re
from typing import Dict, List


def parse_bboxes_from_response(response: str) -> List[Dict]:
    """Extract the bbox list from a SDG/SDG response's <answer> block.

    Returns a list of `{"box_2d": [x0, y0, x1, y1], ...}` dicts; entries
    without a well-formed `box_2d` are dropped silently.
    """
    if not response:
        return []
    answer_match = re.search(r'<answer>\s*(.*?)\s*</answer>', response, re.DOTALL)
    if not answer_match:
        return []
    answer_text = answer_match.group(1).strip()
    try:
        start = answer_text.find('[')
        end = answer_text.rfind(']')
        if start == -1 or end == -1 or end <= start:
            return []
        data = json.loads(answer_text[start:end + 1])
        if isinstance(data, list):
            return [
                item for item in data
                if isinstance(item, dict) and "box_2d" in item
                and isinstance(item["box_2d"], list) and len(item["box_2d"]) == 4
            ]
    except (json.JSONDecodeError, ValueError):
        pass
    return []
