#!/usr/bin/env python3
"""
ImageDoctor custom HTTP API service (OpenAI-compatible format)

reference:
  - flow_grpo/flow_grpo/imagedoctor_scorer.py
  - SDG/comparison/imagedoctor/inference_batch.py

Usage:
    CUDA_VISIBLE_DEVICES=4,5 python imagedoctor_server.py \
        --port 17141 --model_path GYX97/ImageDoctor
"""
import os
import sys
import json
import math
import base64
import argparse
import traceback
from io import BytesIO
from http.server import BaseHTTPRequestHandler, HTTPServer

import torch
import numpy as np
from PIL import Image

# ================================================================
# Compat fix: newer transformers' AutoProcessor tries to load
# video_processor, but the ImageDoctor config (based on older Qwen2-VL)
# Missing video_processor_type causes ValueError.
# is monkey-patched out before importing AutoProcessor.
# ================================================================
try:
    import transformers.models.auto.video_processing_auto as _vp_mod
    _orig_vp_from_pretrained = _vp_mod.AutoVideoProcessor.from_pretrained

    @classmethod  # type: ignore[misc]
    def _patched_vp_from_pretrained(cls, *args, **kwargs):
        try:
            return _orig_vp_from_pretrained.__func__(cls, *args, **kwargs)
        except (ValueError, KeyError, OSError):
            # ImageDoctor does not support video; return None to let the processor skip
            return None

    _vp_mod.AutoVideoProcessor.from_pretrained = _patched_vp_from_pretrained
    print("[patch] AutoVideoProcessor.from_pretrained patched for ImageDoctor compatibility")
except Exception:
    pass

from transformers import AutoProcessor, AutoModelForCausalLM
from qwen_vl_utils import process_vision_info

# globals
PROCESSOR = None
MODEL = None
DEVICE = None


# ==================== Utility functions ====================

def resize_image(img: Image.Image, target_pixels: int = 512 * 512) -> Image.Image:
    """reference inference_batch.py    of resize logic"""
    r = math.sqrt(target_pixels / (img.width * img.height))
    new_size = (max(1, int(img.width * r)), max(1, int(img.height * r)))
    return img.resize(new_size, resample=Image.BICUBIC)


def decode_base64_image(data_url: str) -> Image.Image:
    """parse data:image/...;base64,xxxx format"""
    if data_url.startswith("data:image"):
        data_url = data_url.split(",", 1)[1]
    raw = base64.b64decode(data_url)
    return Image.open(BytesIO(raw)).convert("RGB")


def build_messages(image: Image.Image, task_prompt: str):
    """reference imagedoctor_scorer.py    of _build_messages"""
    return [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": task_prompt}
        ]
    }]


def extract_image_and_text(messages):
    """Extract the base64 image and the text from OpenAI-format messages."""
    img_b64 = None
    text = ""
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            for item in content:
                if item.get("type") == "image_url":
                    img_b64 = item["image_url"]["url"]
                elif item.get("type") == "text":
                    text = item["text"]
    return img_b64, text


# ==================== Inference ====================

@torch.no_grad()
def run_inference(img_b64: str, prompt_text: str) -> str:
    """
    reference imagedoctor_scorer.py    of __call__ method
    """
    # 1. decode & resize
    img = decode_base64_image(img_b64)
    img = resize_image(img)

    # 2. build messages
    messages = build_messages(img, prompt_text)
    text = PROCESSOR.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = PROCESSOR(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(DEVICE)

    # 3. generate
    outputs = MODEL.generate(
        **inputs,
        max_new_tokens=4096,
        use_cache=True,
    )

    # 4. decode output
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, outputs)]
    decoded = PROCESSOR.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0].strip()

    return decoded


# ==================== HTTP service ====================

class Handler(BaseHTTPRequestHandler):
    """Simple HTTP handler, compatible with OpenAI /v1/chat/completions."""

    def log_message(self, fmt, *args):
        # concise logging
        print(f"[{self.log_date_time_string()}] {fmt % args}")

    def _send_json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/health", "/v1/models"):
            self._send_json(200, {"state": "ok", "model": "ImageDoctor"})
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_response(404)
            self.end_headers()
            return

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)

        try:
            req = json.loads(body)
            img_b64, prompt_text = extract_image_and_text(req.get("messages", []))
            if not img_b64:
                raise ValueError("No image found in request messages")

            result = run_inference(img_b64, prompt_text)

            self._send_json(200, {
                "id": "chatcmpl-imagedoctor",
                "object": "chat.completion",
                "model": "ImageDoctor",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": result},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            })
        except Exception as e:
            traceback.print_exc()
            self._send_json(500, {"error": str(e)})


# ==================== launch ====================

def main():
    parser = argparse.ArgumentParser(description="ImageDoctor HTTP API Server")
    parser.add_argument("--model_path", type=str, default="GYX97/ImageDoctor")
    parser.add_argument("--port", type=int, default=17141)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    global PROCESSOR, MODEL, DEVICE

    print(f"Loading ImageDoctor from {args.model_path} ...")

    # Load processor (the monkey-patch is active from module top)
    PROCESSOR = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    # loadmodel (reference imagedoctor_scorer.py)
    MODEL = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    MODEL.eval()

    DEVICE = next(MODEL.parameters()).device
    print(f"Model loaded on {DEVICE}")
    print(f"Starting server on {args.host}:{args.port} ...")

    server = HTTPServer((args.host, args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        print("Server stopped.")


if __name__ == "__main__":
    main()
