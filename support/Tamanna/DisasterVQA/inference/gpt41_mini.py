"""
GPT-4.1-mini inference via Azure OpenAI for the DisasterVQA benchmark.

Usage:
    python gpt41_mini.py \
        --input_json  dataset/disasterVQA_dataset.json \
        --output_json outputs/gpt41_mini_raw.json \
        --deployment  <your-deployment-name> \
        --endpoint    <your-azure-endpoint> \
        --api_key     <your-api-key>
"""

import argparse
import base64
import json
import logging
import sys
import time
from io import BytesIO
from pathlib import Path

from openai import AzureOpenAI
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


def run_inference(client, deployment, image_base64, prompt):
    try:
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            },
                        },
                    ],
                }
            ],
            max_completion_tokens=200,
            temperature=0,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"


def main():
    parser = argparse.ArgumentParser(description="Azure OpenAI GPT-4.1-mini inference for DisasterVQA.")
    parser.add_argument("--input_json", required=True, help="Path to disasterVQA_dataset.json.")
    parser.add_argument("--output_json", required=True, help="Output JSON file path.")
    parser.add_argument("--deployment", required=True, help="Azure OpenAI deployment name.")
    parser.add_argument("--endpoint", required=True, help="Azure OpenAI endpoint URL.")
    parser.add_argument("--api_key", required=True, help="Azure OpenAI API key.")
    args = parser.parse_args()

    data = load_dataset(args.input_json)
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)

    client = AzureOpenAI(
        api_key=args.api_key,
        api_version="2024-12-01-preview",
        azure_endpoint=args.endpoint,
    )

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

            image_base64 = encode_image_to_base64(image_path)
            if not image_base64:
                model_answer = "Error: image encoding failed"
            else:
                prompt = build_prompt(question, question_type)
                t0 = time.time()
                raw = run_inference(client, args.deployment, image_base64, prompt)
                logging.info("[%d/%d] Done in %.2fs", idx, total, time.time() - t0)
                model_answer = raw.strip()

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
