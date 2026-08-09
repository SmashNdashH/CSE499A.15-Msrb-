"""
Humanitarian category classifier for the DisasterVQA benchmark.

Uses Azure OpenAI (GPT) to classify each QA entry into a humanitarian response
category from the taxonomy defined in taxonomy.json. The classification is
appended to each entry under the `llm_classification` field.

The taxonomy covers two tiers:
  - Situational Awareness (SA-1 to SA-8)
  - Actionable Tasks (AT-9 to AT-18)

Usage:
    python classify_humanitarian.py \
        --input      dataset/disasterVQA_dataset.json \
        --output     dataset/disasterVQA_allmodel_judge_outputs.json \
        --deployment <your-deployment-name> \
        --endpoint   <your-azure-endpoint> \
        --api-key    <your-api-key>
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from openai import AzureOpenAI


CATEGORY_DEFS = {
    # ----------------
    # Situational Awareness
    # ----------------
    "SA-1": {
        "tier": "situational_awareness",
        "category_name": "Hazard Type & Severity",
        "framework_references": [
            "MIRA Guidance – crisis impact (scope/scale, severity) and humanitarian profile",
            "IASC Cluster Approach – inter-cluster coordination baseline (OCHA/HC-led)",
            "Sphere Handbook – CHS / assessment as a foundation for response quality",
            "FEMA National Response Framework – ESF construct (incl. ESF #5 Information & Planning)",
        ],
        "definition": "Identify hazard/disaster type or visible hazard indicators (e.g., flooding visible, smoke present).",
    },
    "SA-2": {
        "tier": "situational_awareness",
        "category_name": "Built Environment Damage",
        "framework_references": [
            "IASC Cluster Approach – Shelter Cluster; Early Recovery (damage informs shelter/recovery priorities)",
            "Sphere Handbook – Shelter & Settlement standards (safety/habitability, site risk)",
            "MIRA Guidance – crisis impact / severity and priority needs implications",
            "FEMA Public Assistance – Permanent Work categories (C–G) and infrastructure restoration framing",
        ],
        "definition": "Damage level/condition of buildings, roads, bridges, dams/levees (collapsed, cracked, washed out).",
    },
    "SA-3": {
        "tier": "situational_awareness",
        "category_name": "Utilities Status & Damage",
        "framework_references": [
            "IASC Cluster Approach – WASH Cluster (water systems) + Emergency Telecommunications Cluster (comms)",
            "Sphere Handbook – WASH minimum standards (service continuity, water safety)",
            "MIRA Guidance – service disruption as severity/impact indicator",
            "FEMA Public Assistance – Category F (Utilities) framing",
        ],
        "definition": "Visible damage/status of power, water systems, telecom infrastructure (downed pole, broken lines).",
    },
    "SA-4": {
        "tier": "situational_awareness",
        "category_name": "Access & Inaccessibility (Site/Route Status)",
        "framework_references": [
            "MIRA Guidance – operational constraints / humanitarian access",
            "IASC Cluster Approach – Logistics Cluster (access routes, supply lines, constraints)",
            "Sphere Handbook – CHS foundations (feasibility, safe access, coordination)",
            "FEMA National Response Framework – ESF construct (transport/access functions)",
        ],
        "definition": "Passability and obstructions on routes/sites (blocked/partially blocked; debris/rocks/water/vehicles obstructing).",
    },
    "SA-5": {
        "tier": "situational_awareness",
        "category_name": "Movement Restrictions & Controls",
        "framework_references": [
            "MIRA Guidance – humanitarian access constraints (restrictions/checkpoints/closures)",
            "IASC Cluster Approach – Protection Cluster (civilian safety/rights) + Logistics Cluster (operational access)",
            "Sphere Handbook – Protection Principles (safety, dignity, avoid harm)",
            "FEMA National Response Framework – ESF construct (public safety/security functions)",
        ],
        "definition": "Barriers, cones, cordons, checkpoints, closures, restricted zones.",
    },
    "SA-6": {
        "tier": "situational_awareness",
        "category_name": "Population/Asset Presence & Exposure",
        "framework_references": [
            "MIRA Guidance – humanitarian profile / affected population status",
            "IASC Cluster Approach – Protection Cluster; (context-dependent) CCCM Cluster for displacement/collective sites",
            "Sphere Handbook – Protection Principles + CHS foundations (identify who is affected/vulnerable)",
            "FEMA National Response Framework – ESF construct (mass care / human services functions)",
        ],
        "definition": "Presence/count/type of people, vehicles, assets; stranded/damaged vehicles (when framed as observation).",
    },
    "SA-7": {
        "tier": "situational_awareness",
        "category_name": "Water & Flood Characteristics",
        "framework_references": [
            "IASC Cluster Approach – WASH Cluster (water safety/contamination/drainage) + Shelter Cluster (site risk)",
            "Sphere Handbook – WASH standards (water safety/drainage) + Shelter & Settlement (site selection risk)",
            "MIRA Guidance – hazard impact characteristics and compounding risks",
            "FEMA Public Assistance – Category D (Water Control Facilities) framing",
        ],
        "definition": "Floodwater depth/extent/coverage/flow; inundation area; standing vs moving water.",
    },
    "SA-8": {
        "tier": "situational_awareness",
        "category_name": "Environmental/Terrain Context",
        "framework_references": [
            "MIRA Guidance – drivers/underlying factors shaping impacts and needs",
            "IASC Cluster Approach – Shelter Cluster (site planning constraints); Food Security/Agriculture linkages where relevant",
            "Sphere Handbook – Shelter & Settlement (context/risk-informed planning)",
        ],
        "definition": "Terrain/setting context (mountainous, coastal, hillside, forest) used for situational understanding.",
    },
    # ----------------
    # Actionable Tasks
    # ----------------
    "AT-9": {
        "tier": "actionable_task",
        "category_name": "Search and Rescue",
        "framework_references": [
            "INSARAG Guidelines – international USAR coordination methodology/standards",
            "Sphere Handbook – Protection Principles (life-saving, do-no-harm) + Health linkages for trauma care pathways",
            "MIRA Guidance – immediate life-saving priorities in early analysis",
            "FEMA Public Assistance – Category B (Emergency Protective Measures) framing",
        ],
        "definition": "Rescue need, trapped persons, evacuation feasibility/requirements.",
    },
    "AT-10": {
        "tier": "actionable_task",
        "category_name": "Public Health and Medical Services",
        "framework_references": [
            "IASC Cluster Approach – Health Cluster",
            "Sphere Handbook – Health minimum standards",
            "MIRA Guidance – affected population status (health needs/risks)",
            "FEMA National Response Framework – ESF construct (health/medical functions)",
        ],
        "definition": "Injury/medical need, facility overload, outbreak/health risk requiring action.",
    },
    "AT-11": {
        "tier": "actionable_task",
        "category_name": "Water, Sanitation & Hygiene Needs",
        "framework_references": [
            "IASC Cluster Approach – WASH Cluster",
            "Sphere Handbook – WASH minimum standards (water supply, sanitation, hygiene)",
            "MIRA Guidance – affected population status (WASH needs)",
        ],
        "definition": "Potable water availability/adequacy, water points, sanitation/hygiene service needs.",
    },
    "AT-12": {
        "tier": "actionable_task",
        "category_name": "Debris Clearance & Earthmoving",
        "framework_references": [
            "IASC Cluster Approach – Early Recovery (cleanup/rehabilitation) + Logistics Cluster (opening access routes)",
            "MIRA Guidance – operational constraints and access restoration priorities",
            "FEMA Public Assistance – Category A (Debris Removal) framing",
        ],
        "definition": "Clearing debris/rocks/mud; need for machinery or clearance operations.",
    },
    "AT-13": {
        "tier": "actionable_task",
        "category_name": "Infrastructure Repair & Engineering Works",
        "framework_references": [
            "IASC Cluster Approach – Early Recovery; Shelter Cluster (structural safety implications); Logistics Cluster (critical corridors)",
            "MIRA Guidance – recovery constraints/priorities",
            "FEMA Public Assistance – Permanent Work categories (C–G) framing",
        ],
        "definition": "Repair/stabilize/restore built assets; structural safety actions; slope stabilization/shoring.",
    },
    "AT-14": {
        "tier": "actionable_task",
        "category_name": "Utilities Restoration",
        "framework_references": [
            "IASC Cluster Approach – WASH Cluster + Emergency Telecommunications Cluster",
            "Sphere Handbook – WASH minimum standards (service continuity/safety)",
            "MIRA Guidance – service restoration as priority action",
            "FEMA Public Assistance – Category F (Utilities) framing",
        ],
        "definition": "Restore power/comms/water services; utility repair actions.",
    },
    "AT-15": {
        "tier": "actionable_task",
        "category_name": "Logistics Management and Resource Support",
        "framework_references": [
            "IASC Cluster Approach – Logistics Cluster",
            "MIRA Guidance – humanitarian access / delivery constraints",
            "Sphere Handbook – CHS foundations (coordination, effective use of resources)",
            "FEMA National Response Framework – ESF construct (logistics/resource support functions)",
        ],
        "definition": "Delivering commodities, staging, distribution constraints, supply routes as an operational task.",
    },
    "AT-16": {
        "tier": "actionable_task",
        "category_name": "Public Safety and Security / Traffic Management",
        "framework_references": [
            "IASC Cluster Approach – Protection Cluster (civilian safety) + Logistics Cluster (access impacts)",
            "Sphere Handbook – Protection Principles (safety, minimize harm)",
            "MIRA Guidance – access constraints / security environment",
            "FEMA Public Assistance – Category B (Emergency Protective Measures) framing",
        ],
        "definition": "Enforcing closures, crowd control, securing hazardous areas, traffic control actions.",
    },
    "AT-17": {
        "tier": "actionable_task",
        "category_name": "Firefighting",
        "framework_references": [
            "Sphere Handbook – Protection Principles (life safety; avoid harm) + shelter safety considerations",
            "MIRA Guidance – evolving risks / immediate life-saving priorities",
            "FEMA Public Assistance – Category B (Emergency Protective Measures) framing",
            "FEMA National Response Framework – ESF construct (firefighting functions)",
        ],
        "definition": "Fire suppression/containment actions; wildfire response.",
    },
    "AT-18": {
        "tier": "actionable_task",
        "category_name": "Oil and Hazardous Materials Response",
        "framework_references": [
            "UNEP/OCHA Joint Environment Unit (JEU) – environmental emergency response support/coordination",
            "IASC Cluster Approach – WASH Cluster (water contamination) + Health Cluster (exposure risks) + Early Recovery (cleanup)",
            "Sphere Handbook – WASH standards (water quality/safety) + Protection Principles (avoid harm from exposure)",
            "MIRA Guidance – severity/risk + operational constraints",
            "FEMA Public Assistance – Category B (Emergency Protective Measures) framing",
        ],
        "definition": "Oil/chemical spills, contaminated floodwaters requiring containment/cleanup.",
    },
}


def build_system_prompt() -> str:
    def refs_for(cid):
        refs = CATEGORY_DEFS[cid].get("framework_references", [])
        return [str(r).strip() for r in refs if r is not None and str(r).strip()][:2]

    sa_keys = sorted([k for k in CATEGORY_DEFS if k.startswith("SA-")], key=lambda x: int(x.split("-")[1]))
    at_keys = sorted([k for k in CATEGORY_DEFS if k.startswith("AT-")], key=lambda x: int(x.split("-")[1]))

    lines: List[str] = []
    lines.append("TAXONOMY (choose at most ONE category; or choose none by setting nulls as instructed):\n")
    lines.append("Situational Awareness")
    for cid in sa_keys:
        c = CATEGORY_DEFS[cid]
        lines.append(f'- {cid} {c["category_name"]}')
        lines.append(f'  Framework references: {json.dumps(refs_for(cid), ensure_ascii=False)}')
        lines.append(f'  {c["definition"]}')
    lines.append("\nActionable Tasks")
    for cid in at_keys:
        c = CATEGORY_DEFS[cid]
        lines.append(f'- {cid} {c["category_name"]}')
        lines.append(f'  Framework references: {json.dumps(refs_for(cid), ensure_ascii=False)}')
        lines.append(f'  {c["definition"]}')

    taxonomy_block = "\n".join(lines)

    return f"""You are an emergency management analyst. Classify each disaster-assessment question into at most ONE category from the taxonomy below, and also label whether it is Situational Awareness or an Actionable Task.

IMPORTANT: The input items include groundtruth fields (e.g., "groundtruth", "groundtruth_answer").
Treat groundtruth as part of the evidence for classification.

CRITICAL OUTPUT RULES:
- Output MUST be a JSON array and the very first character of your response must be '['.
- Do not wrap JSON in code fences.
- Do not output any preamble text.
- Do not output trailing text after the closing ']'.
- Return ONLY valid JSON.

Other rules:
- Output must be a JSON array with one object per input item, preserving the same "id".
- Use ALL provided fields: question text + choices (if present) + groundtruth (if present) to decide the best category (or no match).
- You are NOT required to map every question to a taxonomy category:
  - If there is no clear match, set:
    tier=null, category_id=null, category_name=null, category=null, framework_references=[],
    confidence between 0.0 and 0.4, and a short rationale.
- If you DO classify:
  - Choose exactly ONE category_id and its matching tier.
  - category_name must match EXACTLY the taxonomy entry below.
  - category must equal category_name (duplicate field for clarity).
  - framework_references must match the taxonomy entry below (include BOTH if two are listed).
- Provide "confidence" from 0.0 to 1.0.
- Provide "rationale" in <= 1 short sentence.
- Intent rules:
  - If the question is mainly identify/describe/count/assess status -> Situational Awareness.
  - If the question implies a response action or operational need -> Actionable Task.
- Tie-breakers:
  1) "Can we get there / is it passable?" -> SA-4
  2) Barriers/closures/cordons/checkpoints -> SA-5
  3) Floodwater depth/extent/coverage -> SA-7
  4) Potable water availability/adequacy -> AT-11
  5) Built asset damage -> SA-2 unless explicitly about passability, then SA-4
  6) Utilities damage/status -> SA-3 unless explicitly about restoring service, then AT-14

{taxonomy_block}

OUTPUT FORMAT (strict):
[
  {{
    "id": "<same as input id>",
    "tier": "situational_awareness" or "actionable_task" or null,
    "category_id": "SA-1" ... "SA-8" or "AT-9" ... "AT-18" or null,
    "category_name": "<exact taxonomy entry>" or null,
    "category": "<same as category_name>" or null,
    "framework_references": ["<exact framework reference(s)>"] or [],
    "confidence": 0.0-1.0,
    "rationale": "<= 1 short sentence>"
  }}
]
"""


SYSTEM_PROMPT = build_system_prompt()


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def append_jsonl(path: Optional[str], record: Dict[str, Any]) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


_THINK_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)


def strip_think(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def extract_json_payload(text: str) -> str:
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE).strip()
    t = re.sub(r"\s*```$", "", t).strip()
    if t.startswith("[") and t.endswith("]"):
        return t
    if t.startswith("{") and t.endswith("}"):
        return f"[{t}]"
    m = re.search(r"\[[\s\S]*\]", t)
    if m:
        return m.group(0)
    m = re.search(r"\{[\s\S]*\}", t)
    if m:
        return f"[{m.group(0)}]"
    raise ValueError("No JSON payload found in model output.")


def validate_result_obj(obj: Dict[str, Any]) -> None:
    required = ["id", "tier", "category_id", "category_name", "category",
                "framework_references", "confidence", "rationale"]
    for k in required:
        if k not in obj:
            raise ValueError(f"Missing key '{k}' in result: {obj}")
    tier = obj["tier"]
    if tier is not None and tier not in ("situational_awareness", "actionable_task"):
        raise ValueError(f"Invalid tier: {tier}")
    if not isinstance(obj["framework_references"], list):
        raise ValueError("framework_references must be a list.")
    conf = obj["confidence"]
    if not isinstance(conf, (int, float)) or not (0.0 <= float(conf) <= 1.0):
        raise ValueError("confidence must be between 0.0 and 1.0")
    if not isinstance(obj["rationale"], str) or not obj["rationale"].strip():
        raise ValueError("rationale must be a non-empty string")


def slim_item(x: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"id": x.get("id"), "question": x.get("question")}
    if isinstance(x.get("choices"), dict):
        out["choices"] = x["choices"]
    for k in ("groundtruth", "groundtruth_answer"):
        if k in x:
            out[k] = x[k]
    return out


def build_user_prompt(item: Dict[str, Any]) -> str:
    payload = json.dumps([slim_item(item)], ensure_ascii=False)
    return (
        "Return ONLY a JSON array. The first character must be '[' and the last must be ']'.\n"
        "Return exactly one object for the single input item.\n"
        "Use question + choices + groundtruth fields as evidence for classification.\n"
        "You may set category fields to null if there is no clear taxonomy match.\n"
        f"Item:\n{payload}"
    )


def run_with_backoff(
    client: AzureOpenAI,
    deployment: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
    logger: logging.Logger,
    max_retries: int = 6,
) -> str:
    delay = 1.0
    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=deployment,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            last_err = e
            logger.warning("Azure call failed (attempt %d/%d): %s", attempt, max_retries, e)
            if attempt < max_retries:
                time.sleep(min(delay, 20.0))
                delay *= 2.0
    raise RuntimeError(f"Azure call failed after {max_retries} retries: {last_err}")


def classify_one(
    client: AzureOpenAI,
    deployment: str,
    item: Dict[str, Any],
    max_tokens: int,
    temperature: float,
    logger: logging.Logger,
    jsonl_log: Optional[str],
) -> Dict[str, Any]:
    target_id = str(item["id"])
    budgets = [max_tokens, int(max_tokens * 1.5), int(max_tokens * 2.0)]
    last_err: Optional[Exception] = None

    for attempt, budget in enumerate(budgets, start=1):
        t0 = time.time()
        record: Dict[str, Any] = {
            "time": now_utc_iso(), "attempt": attempt,
            "max_tokens": budget, "input_id": target_id, "parsed_ok": False,
        }
        try:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(item)},
            ]
            raw = run_with_backoff(client, deployment, messages, budget, temperature, logger)
            record["raw_text"] = raw[:8000]
            parsed = json.loads(extract_json_payload(strip_think(raw)))

            if not isinstance(parsed, list) or len(parsed) != 1:
                raise ValueError("Expected a JSON array with exactly one object.")
            obj = parsed[0]
            if str(obj.get("id")) != target_id:
                raise ValueError(f"ID mismatch: expected {target_id}, got {obj.get('id')}")
            validate_result_obj(obj)

            record.update({"parsed_ok": True, "duration_s": round(time.time() - t0, 4)})
            append_jsonl(jsonl_log, record)
            return obj

        except Exception as e:
            last_err = e
            record.update({"error": str(e), "duration_s": round(time.time() - t0, 4)})
            record.setdefault("raw_text", "")
            append_jsonl(jsonl_log, record)
            logger.warning("Item %s attempt %d failed: %s", target_id, attempt, e)

    raise RuntimeError(f"Item {target_id} failed after retries. Last error: {last_err}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Humanitarian category classifier for DisasterVQA.")
    ap.add_argument("--input", required=True, help="Input JSON file.")
    ap.add_argument("--output", required=True, help="Output JSON file (input + llm_classification).")
    ap.add_argument("--deployment", required=True, help="Azure OpenAI deployment name.")
    ap.add_argument("--endpoint", required=True, help="Azure OpenAI endpoint URL.")
    ap.add_argument("--api-key", required=True, help="Azure OpenAI API key.")
    ap.add_argument("--api-version", default="2024-12-01-preview", help="Azure OpenAI API version.")
    ap.add_argument("--max-tokens", type=int, default=2400, help="Base token budget (auto-increased on retry).")
    ap.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0 = deterministic).")
    ap.add_argument("--jsonl-log", default=None, help="Optional JSONL file to log each API call.")
    ap.add_argument("--log-level", default="INFO", help="Logging level.")
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    logger = logging.getLogger(__name__)

    logger.info("Reading input: %s", args.input)
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of objects.")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    client = AzureOpenAI(
        api_key=args.api_key,
        api_version=args.api_version,
        azure_endpoint=args.endpoint,
    )

    total = len(data)
    logger.info("Loaded %d items.", total)
    t_all = time.time()

    for idx, item in enumerate(data, start=1):
        logger.info("[%d/%d] Classifying id=%s", idx, total, item["id"])
        item["llm_classification"] = classify_one(
            client, args.deployment, item,
            args.max_tokens, args.temperature, logger, args.jsonl_log,
        )

    logger.info("Classified %d items in %.2fs", total, time.time() - t_all)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info("Saved output to: %s", args.output)


if __name__ == "__main__":
    main()
