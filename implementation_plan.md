# Revise DATASET_QUALITY_SPEC.md to Match Actual Project Workspace

## Background

The current [DATASET_QUALITY_SPEC.md](file:///d:/CSE499AB_project/DATASET_QUALITY_SPEC.md) was authored as a general-purpose agent delegation spec. The user wants it audited against the **real workspace state** and updated to reflect how things actually work. The [sanitization review](file:///d:/CSE499AB_project/claude%20sonnet%204.6%20web%20reponse%20for%20sanitization.txt) from Claude also needs to be incorporated as Stage 4.

---

## Part 1 — Audit: What the Spec Gets Wrong About This Workspace

### 🔴 Critical Mismatches

| # | Spec Says | Reality | Impact |
|---|-----------|---------|--------|
| 1 | **JPEG quality 92** (§1.1) | [image_standardizer.py](file:///d:/CSE499AB_project/support/image_standardizer.py#L54) saves at **quality=95** | Spec and code disagree. Either the code or the spec must be the source of truth — not both. |
| 2 | **`processed_hashes.txt` with MD5 hashes** (§1.3) | The actual tracker is [processed_tracker.txt](file:///d:/CSE499AB_project/data/processed_tracker.txt) and stores **only base filenames** — no MD5, no timestamp, no status column | §1.4's prescribed format (`original_filename \| output_filename \| md5_hash \| timestamp \| status`) is completely unimplemented. |
| 3 | **Reject grayscale images** (§1.2) | `image_standardizer.py` converts ALL non-RGB to RGB (`img.convert('RGB')`) — it does not reject grayscale, it silently converts it | The spec warns against pseudo-RGB from repeated channels, but the code does it anyway. For xView2 satellite imagery this is fine (all are RGB PNGs), but the spec's language should match the code's behaviour. |
| 4 | **`max_output_tokens` set to 600** (§2.3) | [generate_ground_truth.py](file:///d:/CSE499AB_project/support/generate_ground_truth.py#L182-L188) does **not** set `max_output_tokens` at all in the `GenerateContentConfig` | Responses can be arbitrarily long. |
| 5 | **Temperature 0.3 or lower** (§2.1) | `generate_ground_truth.py` does **not** set temperature | Defaults to the model's default (typically 1.0 for Gemini Flash). |
| 6 | **Automated quality gates in Script 2** (§2.3) | `generate_ground_truth.py` has **zero** pre-QA checks: no header validation, no word count, no placeholder regex, no first-person filter | Every raw sample is written to `train_dataset.jsonl` unconditionally. |
| 7 | **Train/val/test split: 70/15/15** (§Cross-Cutting) | The original xView2 dataset already has a **75/25 train/test split** per [dataset metadata.json](file:///d:/CSE499AB_project/data/xView2%20Challenge%20Dataset%20-%20train%20and%20test/dataset%20metadata.json). The test set has **no labels** (1,866 unlabelled images). The spec ignores this entirely and proposes a fresh re-split. | You **cannot** use the xView2 test images for your pipeline — they have no ground-truth labels. Your entire pipeline operates exclusively on the 2,799 post-disaster images from the `train/` split. The train/val/test split must happen on your **generated JSONL**, not the source images. |
| 8 | **`verified_dataset.jsonl` is the final output** (§Summary, §3.5) | The sanitization script (Stage 4) outputs `final_training_dataset.jsonl`. `verified_dataset.jsonl` is an **intermediate** file, not the final one. The `dataset/` folder currently has no `final_training_dataset.jsonl` yet. | The spec's checklist treats `verified_dataset.jsonl` as the terminal artifact. It needs a Stage 4 section. |
| 9 | **Image path format: `processed_images/...`** (§2.6) | Correct — [generate_ground_truth.py line 153](file:///d:/CSE499AB_project/support/generate_ground_truth.py#L153) stores `"processed_images/{base_name}.jpg"`. But this is relative to `data/`, not to project root. The spec says "relative path from project root" which would be `data/processed_images/...` | The actual path in the JSONL works for the validator (which bases at `data/`) but the spec's description is technically misaligned. |
| 10 | **Disaster-type-specific prompt tuning** (§2.4) | `generate_ground_truth.py` uses a **single identical prompt** for all disaster types. No per-type emphasis string is injected. | The table in §2.4 is aspirational, not implemented. |

### 🟡 Moderate Mismatches

| # | Spec Says | Reality |
|---|-----------|---------|
| 11 | **Coverage: no single type exceeding 40%** (§2.5) | Current `train_dataset.jsonl`: hurricane-michael 27.7%, hurricane-harvey 25.9%, hurricane-florence 22.7%, hurricane-matthew 20.0%, guatemala-volcano 3.6%. All hurricanes. No earthquakes, no floods, no wildfires despite xView2 having socal-fire (823), midwest-flooding (279), mexico-earthquake (121), palu-tsunami (113). |
| 12 | **15% low-damage/intact scenes** (§2.5) | Not tracked. Many samples appear intact-only, but no automated accounting exists. |
| 13 | **Edit logging with dropdown reasons** (§3.3) | The [dataset_validator.py](file:///d:/CSE499AB_project/support/dataset_validator.py) has no edit logging at all — edits are saved silently with no diff, no editor ID, no reason. |
| 14 | **Annotator calibration round** (§3.1) | No calibration samples exist. |
| 15 | **Spec references PaliGemma-3B** (§header, §VLM-Family table) | PaliGemma-3B is not mentioned in [README.md](file:///d:/CSE499AB_project/README.md) — the README lists Qwen2-VL-7B (primary), Qwen2-VL-2B (ablation), LLaVA-1.5-7B, InternVL2-8B (benchmarks). |
| 16 | **Metadata template includes `Sensor Type: {satellite \| UAV \| aerial drone}`** (§2.2) | [parse_xbd_json](file:///d:/CSE499AB_project/support/generate_ground_truth.py#L68-L94) does not output sensor type. The xBD metadata doesn't carry this field. |

### ✅ What the Spec Gets Right

- The 3-key JSONL contract (`image`, `instruction`, `response`) — correctly implemented
- System prompt with the 3-section rescue schema — correctly implemented
- Multimodal payload (image + metadata text) — correctly implemented
- JPEG output format — correctly implemented
- RGB conversion — correctly implemented (though behaviour differs from spec)
- Relative image paths (no absolute) — correctly implemented
- Append-mode writing for resumable generation — correctly implemented
- The granular 429 error handling strategy — correctly implemented

---

## Part 2 — Proposed Changes to DATASET_QUALITY_SPEC.md

### [MODIFY] [DATASET_QUALITY_SPEC.md](file:///d:/CSE499AB_project/DATASET_QUALITY_SPEC.md)

The following sections will be revised:

#### §1.1 — Align JPEG quality to code (95, not 92)
Rationale: The code already saves at quality 95 and has been used for all 500 processed images. Changing the code retroactively would require reprocessing. Align the spec to the code.

#### §1.2 — Rewrite grayscale handling to match code behaviour
The xView2 source is exclusively RGB PNGs. The spec should document that `convert('RGB')` is applied to all images (which correctly handles RGBA), and note that pure grayscale sources are not expected from xView2 but should be skipped if encountered in future dataset expansion (e.g., SAR).

#### §1.3 — Demote MD5 hashing to "recommended for future expansion"
The current tracker works for a single-operator pipeline on xView2. Full MD5 hashing is good practice but hasn't been implemented. Mark it as a Phase 2 enhancement rather than a mandatory requirement.

#### §1.4 — Align tracker format to what actually exists
Document the current `processed_tracker.txt` format (plain base-filename list) and note the richer format as a recommended upgrade.

#### §2.1 — Add `temperature=0.3` and `max_output_tokens=600` as "MUST be set"
These are genuinely important and should be enforced. Flag them as implementation gaps that need to be patched in `generate_ground_truth.py` before the next batch run.

#### §2.2 — Remove `Sensor Type` from metadata template
xView2 doesn't carry this. Add a note that for future multi-source datasets (FloodNet, AIDER), sensor type should be re-introduced.

#### §2.3 — Mark pre-QA quality gates as "deferred to Stage 4 (Sanitizer)"
In this pipeline, the sanitizer script handles placeholder removal, key validation, and structural checks. The spec should reflect this actual division of labour rather than pretending Script 2 does it.

#### §2.4 — Mark disaster-specific prompts as "Phase 2 — not yet implemented"
Honest about current state. Add concrete implementation note.

#### §2.5 — Update distribution targets to reflect actual xView2 source breakdown
Add the actual disaster type inventory (10 types, heavily hurricane-skewed). Note that the current 500-image batch only covers 5 types due to the shuffled sampling approach. Flag that future batches must deliberately target under-represented types (socal-fire, midwest-flooding, mexico-earthquake, palu-tsunami, santa-rosa-wildfire).

#### §2.6 — Clarify image path semantics
Document that the `image` field is relative to the `data/` directory (e.g., `processed_images/xxx.jpg`), not to the project root.

#### §3.3 — Note that edit logging is not implemented
Mark as a known gap. The QA validator saves edits but does not log diffs.

#### §3.5 — Rename references from `verified_dataset.jsonl` to clarify it's intermediate

#### NEW §4 — Add Stage 4: Sanitizer & Instruction Augmentation
Incorporate the sanitization script from the [Claude review](file:///d:/CSE499AB_project/claude%20sonnet%204.6%20web%20reponse%20for%20sanitization.txt), including:
- Placeholder regex filtering
- Missing key protection
- Instruction diversity via round-robin
- Reproducibility via `random.seed(42)`
- Output shuffle
- Output file: `dataset/final_training_dataset.jsonl`

#### Train/Val/Test Split — Rewrite entirely
- Document that xView2's original 75/25 split is at the **source image** level, and the test set is **unlabelled** (unusable for our pipeline)
- Our pipeline operates on the train split only (2,799 post-disaster images, of which 500 are currently processed)
- The train/val/test split is performed on the **final JSONL** (`final_training_dataset.jsonl`), not on source images
- The split script (Stage 5) does not exist yet — mark as future work
- Split must be stratified by disaster **event** (not just type), ensuring no geographic tile appears in both train and test

#### Summary Checklist — Update to reflect 5-stage pipeline
Stages: Image Standardizer → Ground Truth Generation → QA Validator → Sanitizer → Split (future)

#### VLM-Family Table — Remove PaliGemma-3B
Align with README.md's model list.

---

## Verification Plan

### Manual Verification
- Review the updated spec against every file in `support/` to confirm no remaining misalignments
- Cross-check all file paths, variable names, and schema references against actual code

> [!IMPORTANT]
> This is a documentation-only change. No Python scripts are modified. The spec update is about honesty — aligning the document to what actually exists, flagging what doesn't, and incorporating the sanitizer as Stage 4.
