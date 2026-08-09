"""
Llama 3.2 inference via Ollama for the DisasterVQA benchmark.

Requires Ollama running locally with the Llama 3.2 vision model pulled:
    ollama pull llama3.2-vision

Usage:
    python llama32.py \
        --input_json  dataset/disasterVQA_dataset.json \
        --output_json outputs/llama32_raw.json \
        --model       llama3.2-vision \
        --url         http://localhost:11434/api/generate
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
            return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        logging.error("Error encoding image %s: %s", image_path, e)
        return None


def build_prompt(question, question_type):
    if question_type == "Yes/No":
        format_hint = 'your response should be either "Yes" or "No".'
    elif question_type == "Multiple-Choice":
        format_hint = 'your response should be all choices that apply from "A", "B", "C", or "D".'
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


def run_inference(image_path, question, question_type, url, model, idx):
    encoded = encode_image_to_base64(image_path)
    if not encoded:
        return None, "Image encoding failed"

    prompt = build_prompt(question, question_type)
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [encoded],
        "stream": False,
        "options": {"temperature": 0.0, "num_ctx": 4096},
    }

    try:
        t0 = time.time()
        response = requests.post(url, data=json.dumps(payload))
        logging.info("[%d] Completed in %.2fs", idx, time.time() - t0)

        if response.status_code == 200:
            output = response.json().get("response", "")
            return output, None
        return None, f"HTTP {response.status_code}"
    except Exception as e:
        return None, str(e)


def main():
    parser = argparse.ArgumentParser(description="Llama 3.2 (Ollama) inference for DisasterVQA.")
    parser.add_argument("-i", "--input_json", required=True, help="Path to disasterVQA_dataset.json.")
    parser.add_argument("-o", "--output_json", required=True, help="Output JSON file path.")
    parser.add_argument("-m", "--model", required=True, help="Ollama model name (e.g. llama3.2-vision).")
    parser.add_argument("-u", "--url", default="http://localhost:11434/api/generate", help="Ollama API endpoint.")
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
            answer, error = run_inference(image_path, question, question_type, args.url, args.model, idx)

            result = {
                "image_path": image_path,
                "question": question,
                "question_type": question_type,
                "disaster_type": disaster_type,
                "groundtruth_answer": groundtruth,
                "Model_answer": answer if answer else f"Error: {error}",
            }
            json.dump(result, f, indent=2)
            f.write(",\n" if idx != total else "\n")
        f.write("]\n")

    logging.info("Saved %d results to %s (%.2fs)", total, args.output_json, time.time() - start_all)


if __name__ == "__main__":
    main()
