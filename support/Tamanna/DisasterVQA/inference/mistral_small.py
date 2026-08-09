"""
Mistral Small 3.1 inference via a local vLLM server for the DisasterVQA benchmark.

Start vLLM before running:
    vllm serve mistralai/Mistral-Small-3.1-24B-Instruct-2503 --port 8000

Usage:
    python mistral_small.py \
        --input_json  dataset/disasterVQA_dataset.json \
        --output_json outputs/mistral_small_raw.json \
        --api_url     http://localhost:8000/v1/chat/completions \
        --model       mistralai/Mistral-Small-3.1-24B-Instruct-2503
"""

import argparse
import base64
import json
import logging
import sys
import time
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image, ImageFile

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
    force=True,
)


def load_dataset(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def encode_image_to_base64(image_path):
    try:
        with open(image_path, "rb") as image_file, BytesIO() as buf:
            img = Image.open(image_file)
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(buf, format="PNG")
            return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"
    except Exception as e:
        logging.error("Error encoding image %s: %s", image_path, e)
        return None


def build_prompt(question, question_type):
    if question_type == "Yes/No":
        format_hint = 'your response should be either "Yes" or "No".'
    elif question_type == "Multiple-Choice":
        format_hint = 'your response should be all that apply from "A", "B", "C", or "D".'
    elif question_type == "Open-Ended":
        format_hint = "your response should be one or two words only."
    else:
        format_hint = "your response should be clear."

    return (
        question + "\n\n"
        "You must provide your response in the following structured JSON format:\n"
        "{\n"
        f'  "Model_answer": "<{format_hint}>"\n'
        "}"
    )


def run_inference(api_url, model, image_base64, prompt):
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_base64}},
                ],
            }
        ],
        "temperature": 0.0,
    }
    response = requests.post(
        api_url,
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload),
    )
    if response.ok:
        return response.json()["choices"][0]["message"]["content"]
    raise RuntimeError(f"vLLM API error {response.status_code}: {response.text}")


def main():
    parser = argparse.ArgumentParser(description="Mistral Small (vLLM) inference for DisasterVQA.")
    parser.add_argument("--input_json", required=True, help="Path to disasterVQA_dataset.json.")
    parser.add_argument("--output_json", required=True, help="Output JSON file path.")
    parser.add_argument("--api_url", default="http://localhost:8000/v1/chat/completions", help="vLLM API endpoint.")
    parser.add_argument("--model", default="mistralai/Mistral-Small-3.1-24B-Instruct-2503", help="Model name.")
    args = parser.parse_args()

    data = load_dataset(args.input_json)
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)

    total = len(data)
    start_all = time.time()

    with open(args.output_json, "w", encoding="utf-8", buffering=1) as f:
        f.write("[\n")
        for idx, entry in enumerate(data, start=1):
            image_path = entry["image_path"]
            question = entry["question"]
            question_type = entry.get("question_type", "Unknown")
            groundtruth = entry.get("groundtruth_answer", "N/A")
            disaster_type = entry.get("disaster_type", "Unknown")

            logging.info("[%d/%d] Processing: %s", idx, total, image_path)

            try:
                image_base64 = encode_image_to_base64(image_path)
                if not image_base64:
                    raise ValueError("Base64 conversion failed")

                prompt = build_prompt(question, question_type)
                t0 = time.time()
                raw = run_inference(args.api_url, args.model, image_base64, prompt)
                logging.info("[%d/%d] Done in %.2fs", idx, total, time.time() - t0)
                model_answer = raw.strip()

            except Exception as e:
                logging.error("[%d] Failed: %s", idx, e)
                model_answer = f"Error: {e}"

            result = {
                "image_path": image_path,
                "question": question,
                "question_type": question_type,
                "disaster_type": disaster_type,
                "groundtruth_answer": groundtruth,
                "Model_answer": model_answer,
            }
            json.dump(result, f, indent=2)
            f.write(",\n" if idx != total else "\n")
        f.write("]\n")

    logging.info("Saved %d results to %s (%.2fs)", total, args.output_json, time.time() - start_all)


if __name__ == "__main__":
    main()
