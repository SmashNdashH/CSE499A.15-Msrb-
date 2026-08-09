# Judge

Raw model responses contain formatting noise (markdown fences, extra text, etc.). The judge script uses an LLM to parse each raw response into a clean, structured format suitable for evaluation.

## Input format

Each inference script in `inference/` writes one JSON array where every entry looks like this:

```json
{
  "image_path": "DisasterVQA/MEDIC/img001.jpg",
  "question": "Is there visible flooding?",
  "question_type": "Yes/No",
  "disaster_type": "flood",
  "groundtruth_answer": "Yes",
  "Model_answer": "{\"Model_answer\": \"Yes, there is visible flooding.\"}"
}
```

Before running the judge you must **merge** all per-model files into a single JSON array. In the merged file each entry must have:

- All fields from `disasterVQA_dataset.json` for that question (`id`, `image_id`, `image_path`, `question`, `question_type`, `groundtruth_answer`, `choices`)
- One key per model — named after the model — whose value is the raw `Model_answer` string from that model's inference output

```json
[
  {
    "id": "imgq-1",
    "image_id": "img-1",
    "image_path": "DisasterVQA/MEDIC/img001.jpg",
    "question_type": "Yes/No",
    "question": "Is there visible flooding?",
    "groundtruth_answer": "Yes",
    "choices": {},
    "gpt-4o-mini": "{\"Model_answer\": \"Yes, there is visible flooding.\"}",
    "llama3.2-11b": "```json\n{\"Model_answer\": \"Yes\"}\n```"
  }
]
```

The judge identifies model keys as any key not in the reserved set: `id`, `image_id`, `image_path`, `question`, `question_type`, `disaster_type`, `groundtruth_answer`, `choices`.

## Output format

Each model's value is replaced with a clean, type-specific answer:

| Question type | Judged output |
|---|---|
| Yes/No | `"Yes"` or `"No"` |
| Multiple-Choice | `["A"]`, `["A", "C"]`, etc. |
| Open-Ended | `{"decision": "Right", "answer": "flooding"}` |

## Running the judge

```bash
pip install openai

python judge/judge_postprocess.py \
    --input           outputs/all_models_raw.json \
    --output          outputs/all_models_judged.json \
    --prompt-yesno    prompts/judge_binary.txt \
    --prompt-mcq      prompts/judge_mcq.txt \
    --prompt-open     prompts/judge_open_ended.txt \
    --endpoint        https://<resource>.openai.azure.com/ \
    --api-key         <api-key> \
    --deployment-name gpt-4o
```

Alternatively, set credentials via environment variables:

```bash
export AZURE_OPENAI_API_KEY=<api-key>
export AZURE_OPENAI_API_ENDPOINT=https://<resource>.openai.azure.com/

python judge/judge_postprocess.py \
    --input  outputs/all_models_raw.json \
    --output outputs/all_models_judged.json \
    --prompt-yesno prompts/judge_binary.txt \
    --prompt-mcq   prompts/judge_mcq.txt \
    --prompt-open  prompts/judge_open_ended.txt
```

## Resume support

If the run is interrupted, re-run the same command. Entries already present in the output file (matched by `id`) are skipped automatically.

## Additional options

| Flag | Description |
|---|---|
| `--max-entries N` | Process only the first N entries |
| `--num-open-ended N` | Process only the first N Open-Ended entries |
| `--deployment-name` | Azure deployment name (default: `gpt-4o`) |
| `--api-version` | Azure API version (default: `2024-12-01-preview`) |
