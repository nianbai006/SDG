#!/usr/bin/env python
"""
Evaluate images with Forensic-Chat (real/fake detection) score.
Higher score = more "real-looking".

Usage:
    python eval_forensic_chat.py --images_dir /path/to/images --output_file /path/to/results.json [--device cuda:0]
"""
import argparse
import os
import json
import base64
from io import BytesIO

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL_PATH = "${SDG_HOME}/../models/Forensic-Chat"
PROMPTS_FILE = "${SDG_HOME}/flow_grpo/dataset/drawbench/test.txt"


def pil_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image;base64,{encoded}"


def extract_scores(output_logits, processor):
    vocab = processor.tokenizer.get_vocab()
    probs = output_logits[0, -1, :].float().cpu().numpy()
    fake_score = (probs[vocab['fake']] + probs[vocab['Fake']]) / 2
    real_score = (probs[vocab['Real']] + probs[vocab['real']]) / 2
    compare = np.array([fake_score, real_score])
    e_x = np.exp(compare - np.max(compare))
    return float(e_x[1] / e_x.sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--output_file", required=True, help="results.json (will merge forensic_chat key)")
    parser.add_argument("--prompts_file", default=PROMPTS_FILE)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_samples", type=int, default=999)
    args = parser.parse_args()

    # Load images
    with open(args.prompts_file) as f:
        prompts = [l.strip() for l in f if l.strip()][:args.max_samples]
    images = []
    for i in range(len(prompts)):
        p = os.path.join(args.images_dir, f"{i:04d}.png")
        if os.path.exists(p):
            images.append((i, Image.open(p).convert("RGB")))
    print(f"Loaded {len(images)} images")

    # Load model
    print(f"Loading Forensic-Chat model to {args.device}...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map=None,
    ).to(args.device)
    model.requires_grad_(False)
    processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True)
    print("Model loaded")

    # Score each image
    scores = []
    for idx, img in tqdm(images, desc="ForensicChat"):
        b64 = pil_image_to_base64(img)
        messages = [[{
            "role": "user",
            "content": [
                {"type": "image", "image": b64},
                {"type": "text", "text": (
                    "Analyze the provided image. "
                    "Decide whether it is a real photograph or AI-generated. "
                    "The first word must be either 'real' or 'fake'."
                )},
            ],
        }]]

        texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        inputs = inputs.to(args.device)

        with torch.no_grad():
            outputs = model(**inputs)

        score = extract_scores(outputs.logits, processor)
        scores.append(score)

    mean_score = float(np.mean(scores))
    std_score = float(np.std(scores))
    print(f"  forensic_chat: {mean_score:.4f} +/- {std_score:.4f}")

    # Merge into existing results.json
    result = {}
    if os.path.exists(args.output_file):
        with open(args.output_file) as f:
            result = json.load(f)
    result["forensic_chat"] = {"mean": mean_score, "std": std_score}

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {args.output_file}")


if __name__ == "__main__":
    main()
