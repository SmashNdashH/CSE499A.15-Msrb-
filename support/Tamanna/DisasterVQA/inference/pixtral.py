"""
Pixtral inference (local Mistral inference library) for the DisasterVQA benchmark.

Download the model weights before running:
    mistral-rs download --model pixtral-12b-2409 --path mistral_models/Pixtral

Usage:
    python pixtral.py \
        --input_json    dataset/disasterVQA_dataset.json \
        --output_json   outputs/pixtral_raw.json \
        --model_path    mistral_models/Pixtral \
        --tokenizer     mistral_models/Pixtral/tekken.json
"""

import argparse
import base64
import json
import logging
import sys
import time
from io import BytesIO
from pathlib import Path

from mistral_common.protocol.instruct.messages import ImageURLChunk, TextChunk, UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_inference.generate import generate
from mistral_inference.transformer import Transformer
from PIL import Image, ImageFile

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
    force=True,
)


def load_model_and_tokenizer(tokenizer_path, model_path):
    logging.info("Loading Pixtral model from %s ...", model_path)
    tokenizer = MistralTokenizer.from_file(tokenizer_path)
    model = Transformer.from_folder(model_path)
    return tokenizer, model


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


def run_inference(image_base64, prompt, tokenizer, model):
    completion_request = ChatCompletionRequest(
        messages=[
            UserMessage(
                content=[
                    ImageURLChunk(image_url=image_base64),
                    TextChunk(text=prompt),
                ]
            )
        ]
    )
    encoded = tokenizer.encode_chat_completion(completion_request)
    out_tokens, _ = generate(
        [encoded.tokens],
        model,
        images=[encoded.images],
        max_tokens=200,
        temperature=0.0,
        eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id,
    )
    return tokenizer.decode(out_tokens[0])


def main():
    parser = argparse.ArgumentParser(description="Pixtral inference for DisasterVQA.")
    parser.add_argument("--input_json", required=True, help="Path to disasterVQA_dataset.json.")
    parser.add_argument("--output_json", required=True, help="Output JSON file path.")
    parser.add_argument("--model_path", default="mistral_models/Pixtral", help="Path to Pixtral model folder.")
    parser.add_argument("--tokenizer", default="mistral_models/Pixtral/tekken.json", help="Path to tekken.json tokenizer.")
    args = parser.parse_args()

    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    logging.info("Loaded %d entries.", len(data))

    tokenizer, model = load_model_and_tokenizer(args.tokenizer, args.model_path)
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
                model_answer = run_inference(image_base64, prompt, tokenizer, model)
                logging.info("[%d/%d] Done in %.2fs", idx, total, time.time() - t0)
                model_answer = model_answer.strip()

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
