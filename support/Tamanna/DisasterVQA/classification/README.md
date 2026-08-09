# Classification

This folder contains the script and taxonomy used to assign each QA entry in DisasterVQA a humanitarian response category via an LLM-as-classifier pipeline. The output is the `llm_classification` field in `dataset/disasterVQA_allmodel_judge_outputs.json`.

## Taxonomy

`taxonomy.json` defines the humanitarian response framework used for classification. It covers two tiers:

**Situational Awareness (SA)**

| ID | Category |
|---|---|
| SA-1 | Hazard Type & Severity |
| SA-2 | Built Environment Damage |
| SA-3 | Utilities Status & Damage |
| SA-4 | Access & Inaccessibility |
| SA-5 | Movement Restrictions & Controls |
| SA-6 | Population/Asset Presence & Exposure |
| SA-7 | Water & Flood Characteristics |
| SA-8 | Environmental/Terrain Context |

**Actionable Tasks (AT)**

| ID | Category |
|---|---|
| AT-9  | Search and Rescue |
| AT-10 | Public Health and Medical Services |
| AT-11 | Water, Sanitation & Hygiene Needs |
| AT-12 | Debris Clearance & Earthmoving |
| AT-13 | Infrastructure Repair & Engineering Works |
| AT-14 | Utilities Restoration |
| AT-15 | Logistics Management and Resource Support |
| AT-16 | Public Safety and Security / Traffic Management |
| AT-17 | Firefighting |
| AT-18 | Oil and Hazardous Materials Response |

The taxonomy is grounded in international humanitarian frameworks: IASC Cluster Approach, Sphere Handbook, MIRA Guidance, INSARAG Guidelines, and FEMA National Response Framework.

## classify_humanitarian.py

Classifies each QA entry into one taxonomy category using Azure OpenAI. Each entry is processed individually with automatic retry and exponential backoff. The `llm_classification` field is appended in place.

```bash
pip install openai

python classification/classify_humanitarian.py \
    --input      dataset/disasterVQA_dataset.json \
    --output     dataset/disasterVQA_allmodel_judge_outputs.json \
    --deployment <your-deployment-name> \
    --endpoint   <your-azure-endpoint> \
    --api-key    <your-api-key>
```

**Optional flags:**

| Flag | Description |
|---|---|
| `--max-tokens` | Base token budget per call (default: 2400, auto-increased on retry) |
| `--temperature` | Sampling temperature (default: 0.0) |
| `--jsonl-log` | Path to log each API call as JSONL for debugging |
| `--log-level` | Logging verbosity: DEBUG, INFO, WARNING (default: INFO) |
| `--api-version` | Azure OpenAI API version (default: 2024-12-01-preview) |
