# Dataset

## disasterVQA_dataset.json

The benchmark dataset contains **4,405 QA pairs** across **1,395 unique disaster images**.

### Entry schema

```json
{
  "id": "unique entry identifier",
  "image_id": "source image identifier",
  "image_path": "relative path to the image file",
  "dataset_source": "medic | crisismmd | incidents1m",
  "disaster_type": "earthquake | flood | hurricane | fire | accidents | storm | wildfire | landslide | other",
  "region": "North America | South Asia | Europe | ... | null",
  "question_type": "Yes/No | Multiple-Choice | Open-Ended",
  "question": "the question text",
  "groundtruth_answer": "Yes/No string | list of correct choice letters | list of valid answer phrases",
  "choices": {"A": "...", "B": "...", "C": "...", "D": "..."},
  "crisis_info_type": "situational_awareness | actionable_task",
  "crisis_info_code": "SA-1 | SA-2 | ... | AT-9 | ..."
}
```

`choices` is only present for Multiple-Choice entries.  
`groundtruth_answer` is a list of synonyms for Open-Ended entries.  
`region` is null for entries where geographic origin could not be determined.

### Statistics

| Question type | Count | % |
|---|---|---|
| Yes/No | 2,153 | 48.9% |
| Multiple-Choice | 1,693 | 38.4% |
| Open-Ended | 559 | 12.7% |

| Source dataset | Images | QA pairs |
|---|---|---|
| MEDIC | 600 | 1,685 |
| Incidents1M | 377 | 1,423 |
| CrisisMMD | 418 | 1,297 |

## disasterVQA_allmodel_judge_outputs.json

Pre-computed, judge-processed outputs from all seven models evaluated in the paper, enriched with geographic region and humanitarian category classification. This file is the direct input to all evaluation scripts in `evaluation/`.

### Entry schema

```json
{
  "id": "...",
  "image_id": "...",
  "image_path": "...",
  "region": "North America | South Asia | Europe | ...",
  "disaster_type": "...",
  "dataset_source": "...",
  "question_type": "Yes/No | Multiple-Choice | Open-Ended",
  "question": "...",
  "groundtruth_answer": "...",
  "choices": {"A": "...", "B": "...", "C": "...", "D": "..."},
  "benchmarking_answers": {
    "gpt-4.1-mini":  "Yes",
    "gpt-4o-mini":   "No",
    "llama3.2-11b":  ["A", "B"],
    "mistral-small": {"decision": "Right", "answer": "flooding"},
    "molmo-7b-d":    "Yes",
    "pixtral":       ["A"],
    "qwen-2.5-vl":   {"decision": "Wrong", "answer": "fire"}
  },
  "llm_classification": {
    "tier": "situational_awareness",
    "category_id": "SA-2",
    "category_name": "Built Environment Damage",
    "confidence": 1.0,
    "rationale": "..."
  }
}
```

`choices` is only present for Multiple-Choice entries.

Judged answer types by question type:

| Question type | Type | Example |
|---|---|---|
| Yes/No | string | `"Yes"` |
| Multiple-Choice | list of letters | `["A", "C"]` |
| Open-Ended | dict with decision + answer phrase | `{"decision": "Right", "answer": "fire"}` |

Humanitarian categories in `llm_classification.category_name`:

| Category |
|---|
| Built Environment Damage |
| Hazard Type & Severity |
| Population & Asset Exposure |
| Other Useful Information |
| Access & Inaccessibility |
| Water & Flood Characteristics |
| Movement Restrictions & Controls |
| Utilities Status & Damage |
| Environmental/Terrain Context |
| Search and Rescue |
| Debris Clearance & Earthmoving |
| Firefighting |

