"""
LLM-as-judge post-processing for DisasterVQA raw model outputs.

Reads raw model outputs (one entry per question, one key per model) and uses
an Azure OpenAI judge to parse each answer into a clean, structured format:
  - Yes/No  → "Yes" or "No"
  - MCQ     → ["A"], ["A", "C"], etc.
  - Open    → {"decision": "Right|Wrong", "answer": "<phrase>"}

Supports resuming: if the output file already exists and contains valid JSON,
previously processed entries (matched by id) are skipped.

Usage:
    python judge_postprocess.py \
        --input           outputs/raw/all_models_raw.json \
        --output          outputs/judged/all_models_judged.json \
        --prompt-yesno    prompts/judge_binary.txt \
        --prompt-mcq      prompts/judge_mcq.txt \
        --prompt-open     prompts/judge_open_ended.txt \
        --endpoint        <azure-endpoint> \
        --api-key         <api-key> \
        --deployment-name gpt-4o
"""

import argparse
import json
import logging
import os
import re
import time

from openai import AzureOpenAI


def load_prompt(path):
    if not os.path.exists(path):
        logging.error("Prompt file not found: %s", path)
        raise FileNotFoundError(path)
    with open(path, "r") as f:
        return f.read()


def clean_model_output(raw):
    """Strip markdown code fences and surrounding whitespace."""
    return re.sub(r"```(?:json)?", "", raw).strip()


def format_choices(choices):
    return "\n".join(f"    {k}: {v}" for k, v in choices.items())


def call_judge(client, deployment, prompt):
    response = client.chat.completions.create(
        model=deployment,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=200,
        temperature=0,
    )
    try:
        content = response.choices[0].message.content
        return content.strip() if content else ""
    except Exception as e:
        logging.error("Judge call failed: %s", e)
        return ""


def main():
    parser = argparse.ArgumentParser(description="LLM-as-judge post-processing for DisasterVQA.")
    parser.add_argument("-i", "--input", required=True, help="Raw model outputs JSON.")
    parser.add_argument("-o", "--output", required=True, help="Judged outputs JSON.")
    parser.add_argument("--prompt-yesno", required=True, help="Yes/No judge prompt file.")
    parser.add_argument("--prompt-mcq", required=True, help="MCQ judge prompt file.")
    parser.add_argument("--prompt-open", required=True, help="Open-ended judge prompt file.")
    parser.add_argument("--api-key", help="Azure OpenAI API key (or set AZURE_OPENAI_API_KEY).")
    parser.add_argument("--endpoint", help="Azure OpenAI endpoint (or set AZURE_OPENAI_API_ENDPOINT).")
    parser.add_argument("--api-version", default="2024-12-01-preview", help="Azure OpenAI API version.")
    parser.add_argument("--deployment-name", default="gpt-4o", help="Judge deployment name.")
    parser.add_argument("--max-entries", type=int, default=None, help="Process only the first N entries.")
    parser.add_argument("--num-open-ended", type=int, default=None, help="Process only the first N Open-Ended entries.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    api_key = args.api_key or os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = args.endpoint or os.getenv("AZURE_OPENAI_API_ENDPOINT")
    if not api_key or not endpoint:
        logging.error("Azure OpenAI credentials missing. Provide --api-key / --endpoint or set environment variables.")
        raise SystemExit(1)

    client = AzureOpenAI(api_key=api_key, api_version=args.api_version, azure_endpoint=endpoint)

    prompt_yesno = load_prompt(args.prompt_yesno)
    prompt_mcq = load_prompt(args.prompt_mcq)
    prompt_open = load_prompt(args.prompt_open)

    # Resume support: load previously processed entries
    processed_ids = set()
    if os.path.exists(args.output):
        try:
            with open(args.output, "r") as f:
                existing = json.load(f)
            processed_ids = {e["id"] for e in existing}
            logging.info("Resuming: %d entries already processed.", len(processed_ids))
        except json.JSONDecodeError:
            pass

    with open(args.input, "r") as f:
        data = json.load(f)
    if args.max_entries:
        data = data[: args.max_entries]
    if args.num_open_ended:
        data = [e for e in data if e.get("question_type") == "Open-Ended"][: args.num_open_ended]

    reserved_keys = {
        "id", "image_id", "image_path", "question", "question_type",
        "disaster_type", "groundtruth_answer", "choices",
    }

    total = len(data)
    with open(args.output, "w") as f:
        f.write("[\n")

    for idx, entry in enumerate(data):
        if entry.get("id") in processed_ids:
            logging.info("Skipping ID %s (already processed)", entry["id"])
            continue

        out = entry.copy()
        q_type = entry.get("question_type")
        question = entry.get("question")
        gt = entry.get("groundtruth_answer")
        choices = entry.get("choices", {})

        for model_key, raw in entry.items():
            if model_key in reserved_keys:
                continue
            logging.info("Processing ID %s, model %s", entry.get("id"), model_key)
            cleaned = clean_model_output(raw)

            if q_type == "Yes/No":
                prompt = prompt_yesno.format(question=question, model_output=cleaned)
                resp = call_judge(client, args.deployment_name, prompt)
                out[model_key] = resp or ""

            elif q_type == "Multiple-Choice":
                block = format_choices(choices)
                prompt = prompt_mcq.format(
                    question=question, choices_formatted=block, model_output=cleaned
                )
                resp = call_judge(client, args.deployment_name, prompt)
                out[model_key] = json.loads(resp) if resp else []

            else:  # Open-Ended
                prompt = prompt_open.format(
                    question=question, model_output=cleaned, ground_truth=json.dumps(gt)
                )
                resp = call_judge(client, args.deployment_name, prompt)
                match = re.search(r"(\{[\s\S]*\})", resp) if resp else None
                out[model_key] = json.loads(match.group(1)) if match else []

            time.sleep(1)

        with open(args.output, "a") as f:
            json.dump(out, f, indent=2)
            f.write("," if idx < total - 1 else "")

    with open(args.output, "a") as f:
        f.write("]\n")

    logging.info("Saved post-processed output to %s", args.output)


if __name__ == "__main__":
    main()
