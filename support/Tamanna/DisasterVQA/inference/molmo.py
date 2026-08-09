"""
Molmo-7B-D inference (local HuggingFace model) for the DisasterVQA benchmark.

The model is downloaded automatically from HuggingFace on first run.
Requires a GPU with sufficient VRAM (~16 GB).

Usage:
    python molmo.py \
        --input_json  dataset/disasterVQA_dataset.json \
        --output_json outputs/molmo_raw.json
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from PIL import Image, ImageFile
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
    force=True,
)

MODEL_ID = "allenai/Molmo-7B-D-0924"


def load_model_and_processor():
    logging.info("Loading Molmo model and processor from %s ...", MODEL_ID)
    processor = AutoProcessor.from_pretrained(
        MODEL_ID, trust_remote_code=True, torch_dtype="auto", device_map="auto"
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, torch_dtype="auto", device_map="auto"
    )
    return model.eval(), processor


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


def run_inference(model, processor, image_path, prompt):
    image = Image.open(image_path).convert("RGB")
    inputs = processor.process(images=[image], text=prompt)
    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

    output = model.generate_from_batch(
        inputs,
        GenerationConfig(
            max_new_tokens=200,
            stop_strings=["<|endoftext|>"],
            temperature=0.0,
        ),
        tokenizer=processor.tokenizer,
    )
    generated_tokens = output[0, inputs["input_ids"].size(1):]
    return processor.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser(description="Molmo-7B-D inference for DisasterVQA.")
    parser.add_argument("--input_json", required=True, help="Path to disasterVQA_dataset.json.")
    parser.add_argument("--output_json", required=True, help="Output JSON file path.")
    args = parser.parse_args()

    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    logging.info("Loaded %d entries.", len(data))

    model, processor = load_model_and_processor()
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
                prompt = build_prompt(question, question_type)
                t0 = time.time()
                model_answer = run_inference(model, processor, image_path, prompt)
                logging.info("[%d/%d] Done in %.2fs", idx, total, time.time() - t0)
            except Exception as e:
                logging.error("[%d] Failed %s: %s", idx, image_path, e)
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
