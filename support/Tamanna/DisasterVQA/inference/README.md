# Inference

Each script reads `disasterVQA_dataset.json`, runs the specified model on every image–question pair, and writes a raw output JSON where each entry contains a `"Model_answer"` field with the model's unprocessed response.

All scripts stream results line-by-line (buffered writes) so runs can be interrupted and restarted.

## Common output format

```json
[
  {
    "image_path": "...",
    "question": "...",
    "question_type": "Yes/No | Multiple-Choice | Open-Ended",
    "disaster_type": "...",
    "groundtruth_answer": "...",
    "Model_answer": "<raw model response>"
  },
  ...
]
```

---

## Cloud / API models (Azure OpenAI)

These two scripts use the Azure OpenAI SDK. Set your credentials via CLI flags or environment variables (`AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`).

### GPT-4o-mini

```bash
python inference/gpt4o_mini.py \
    --input_json  dataset/disasterVQA_dataset.json \
    --output_json outputs/gpt4o_mini_raw.json \
    --deployment  <deployment-name> \
    --endpoint    https://<resource>.openai.azure.com/ \
    --api_key     <api-key>
```

### GPT-4.1-mini

```bash
python inference/gpt41_mini.py \
    --input_json  dataset/disasterVQA_dataset.json \
    --output_json outputs/gpt41_mini_raw.json \
    --deployment  <deployment-name> \
    --endpoint    https://<resource>.openai.azure.com/ \
    --api_key     <api-key>
```

---

## Local models via Ollama

### Llama 3.2 Vision

Install [Ollama](https://ollama.com) and pull the model, then run:

```bash
ollama pull llama3.2-vision
pip install requests Pillow

python inference/llama32.py \
    --input_json  dataset/disasterVQA_dataset.json \
    --output_json outputs/llama32_raw.json \
    --model       llama3.2-vision \
    --url         http://localhost:11434/api/generate
```

---

## Local models via vLLM

### Mistral Small 3.1 (24B)

```bash
pip install vllm
vllm serve mistralai/Mistral-Small-3.1-24B-Instruct-2503 --port 8000

pip install requests Pillow
python inference/mistral_small.py \
    --input_json  dataset/disasterVQA_dataset.json \
    --output_json outputs/mistral_small_raw.json \
    --api_url     http://localhost:8000/v1/chat/completions \
    --model       mistralai/Mistral-Small-3.1-24B-Instruct-2503
```

### Qwen2.5-VL 32B AWQ

```bash
vllm serve Qwen/Qwen2.5-VL-32B-Instruct-AWQ --port 8000

python inference/qwen25_vl.py \
    --input_json  dataset/disasterVQA_dataset.json \
    --output_json outputs/qwen25_vl_raw.json \
    --api_base    http://localhost:8000/v1 \
    --model       Qwen/Qwen2.5-VL-32B-Instruct-AWQ
```

---

## Local models (HuggingFace)

### Molmo-7B-D

The model is downloaded automatically on first run (~15 GB). Requires a GPU.

```bash
pip install transformers torch accelerate Pillow

python inference/molmo.py \
    --input_json  dataset/disasterVQA_dataset.json \
    --output_json outputs/molmo_raw.json
```

### Pixtral 12B

Download weights using the Mistral CLI, then run:

```bash
pip install mistral-inference mistral-common Pillow

python inference/pixtral.py \
    --input_json  dataset/disasterVQA_dataset.json \
    --output_json outputs/pixtral_raw.json \
    --model_path  mistral_models/Pixtral \
    --tokenizer   mistral_models/Pixtral/tekken.json
```
