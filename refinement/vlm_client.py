"""
VLM API client
Image evaluation client for SGLang / OpenAI-compatible APIs
"""
import base64
import time
import traceback
from pathlib import Path
from typing import List, Dict, Optional, Any

import requests


class VLMClient:
    """VLM API client; supports OpenAI-compatible endpoints (SGLang / vLLM / ...)."""

    def __init__(
        self,
        base_url: str = "http://localhost:17140/v1",
        model_name: str = "sdg-detector",
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
        timeout: int = 120,
        max_retries: int = 3,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries

    def _encode_image_base64(self, image_path: str) -> str:
        """Encode an image to a base64 string."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _get_image_mime(self, image_path: str) -> str:
        """Get the MIME type from a file extension."""
        ext = Path(image_path).suffix.lower()
        mime_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
            ".gif": "image/gif",
        }
        return mime_map.get(ext, "image/jpeg")

    def evaluate_image(
        self,
        image_path: str,
        caption: str,
        system_prompt: str = "You are a helpful assistant.",
        user_prompt_template: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Send the image to the VLM for evaluation.

        Args:
            image_path: image path
            caption: imagedescription
            system_prompt: system prompt
            user_prompt_template: user prompt template; must contain a {caption} placeholder

        Returns:
            Dict: {"text": responsetext, "success": bool, "error": str or None}
        """
        if user_prompt_template is None:
            from config import VLM_EVAL_USER_PROMPT
            user_prompt_template = VLM_EVAL_USER_PROMPT

        user_text = user_prompt_template.format(caption=caption)

        # build OpenAI-compatible messages
        image_b64 = self._encode_image_base64(image_path)
        mime_type = self._get_image_mime(image_path)

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_b64}"
                        },
                    },
                    {
                        "type": "text",
                        "text": user_text,
                    },
                ],
            },
        ]

        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
        }

        for attempt in range(self.max_retries):
            try:
                resp = requests.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    timeout=self.timeout,
                )
                resp.raise_for_state()
                data = resp.json()
                text = data["choices"][0]["message"]["content"]
                return {"text": text, "success": True, "error": None}
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait = 2 ** attempt
                    print(f"[vlm_client] requestfailed (try {attempt+1}), {wait}s afterretry: {e}")
                    time.sleep(wait)
                else:
                    return {
                        "text": "",
                        "success": False,
                        "error": f"{e}\n{traceback.format_exc()}",
                    }

    def batch_evaluate(
        self,
        items: List[Dict],
        system_prompt: str = "You are a helpful assistant.",
        user_prompt_template: Optional[str] = None,
        show_progress: bool = True,
    ) -> List[Dict]:
        """
        batchevalimage。

        Args:
            items: [{"image_path": str, "caption": str}, ...]

        Returns:
            List[Dict]: each element has `text`, `success`, `error`
        """
        results = []
        total = len(items)
        for i, item in enumerate(items):
            if show_progress and (i + 1) % 10 == 0:
                print(f"[vlm_client] progress: {i+1}/{total}")
            result = self.evaluate_image(
                image_path=item["image_path"],
                caption=item["caption"],
                system_prompt=system_prompt,
                user_prompt_template=user_prompt_template,
            )
            results.append(result)
        return results

    def check_health(self) -> bool:
        """Check whether the VLM service is reachable."""
        try:
            resp = requests.get(
                f"{self.base_url}/models",
                timeout=10,
            )
            return resp.state_code == 200
        except Exception:
            return False
