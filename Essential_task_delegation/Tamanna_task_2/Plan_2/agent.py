"""
agent.py
--------
Autonomous ReAct Agent for Disaster Damage Assessment.
Orchestrates perception, native-resolution visual retrieval, and EMS-98 knowledge grounding.

Author: Tamanna (Member B, Group 5)
Project: CSE499B DisasterM3 Capstone
"""

import json
import re
from typing import Dict, Any, List, Optional
import numpy as np
from tools import SegmentationLocatorTool, HighResVisualRetrieverTool, EMS98KnowledgeRAGTool
from vlm_engine import MultiVLMEngine
import config


class AgenticVisionRAG:
    """
    Autonomous Multimodal Agent that breaks down complex geospatial disaster inquiries,
    invokes specialized vision tools, inspects high-resolution visual crops, and compiles
    structured forensic damage reports.
    """

    def __init__(
        self,
        vlm_engine: MultiVLMEngine,
        seg_tool: Optional[SegmentationLocatorTool] = None,
        retriever_tool: Optional[HighResVisualRetrieverTool] = None,
        knowledge_tool: Optional[EMS98KnowledgeRAGTool] = None
    ):
        self.vlm = vlm_engine
        self.seg_tool = seg_tool or SegmentationLocatorTool()
        self.retriever = retriever_tool or HighResVisualRetrieverTool()
        self.knowledge = knowledge_tool or EMS98KnowledgeRAGTool()

    def run(
        self,
        image_input: Any,
        user_query: str = "Perform comprehensive building damage assessment and flag rescue triage priorities.",
        max_crops_to_inspect: int = 5,
        mask_override: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Executes the ReAct reasoning loop.
        """
        print("\n" + "=" * 65)
        print(f"🤖 [Agent Initiated] Model: {self.vlm.model_key}")
        print(f"📋 [User Query]: {user_query}")
        print("=" * 65)

        # ─── Step 1: PERCEPTION (Locate building footprints) ─────────────────
        print("\n[Thought 1] Locating all buildings and obtaining damage masks using segmentation tool...")
        seg_data = self.seg_tool.run(image_input, mask_override=mask_override)

        total_blds = seg_data["total_buildings"]
        counts = seg_data["counts"]
        print(f"[Observation 1] Detected {total_blds} buildings total: "
              f"{counts['Destroyed']} Destroyed, {counts['Damaged']} Damaged, {counts['Intact']} Intact.")

        # ─── Step 2: DYNAMIC TRIAGE (Prioritize critical targets) ────────────
        print("\n[Thought 2] Prioritizing severe structures (Destroyed/Damaged) for native-resolution Visual RAG...")
        critical_targets = [b for b in seg_data["buildings"] if b["class_id"] in [2, 3]]

        # If few damaged buildings, include a sample of intact buildings for balance
        if len(critical_targets) < max_crops_to_inspect:
            intact_samples = [b for b in seg_data["buildings"] if b["class_id"] == 1]
            critical_targets.extend(intact_samples[:(max_crops_to_inspect - len(critical_targets))])

        selected_targets = critical_targets[:max_crops_to_inspect]
        print(f"[Observation 2] Selected {len(selected_targets)} high-priority building crops for deep inspection.")

        # ─── Step 3: KNOWLEDGE GROUNDING (Consult EMS-98) ────────────────────
        print("\n[Thought 3] Consulting EMS-98 European Macroseismic Scale for engineering criteria...")
        ems_rules = self.knowledge.query("structural")
        print(f"[Observation 3] Ingested EMS-98 damage grading taxonomy and triage rules.")

        # ─── Step 4: VISUAL RAG INSPECTION & CONFLICT RESOLUTION ────────────
        inspections = []
        conflicts_detected = 0

        for b in selected_targets:
            b_id = b["building_id"]
            predicted_class = b["class_name"]

            # Retrieve sharp, un-downsampled 256x256 crop
            crop_rgb = self.retriever.crop_patch(image_input, b["bbox"])

        prompt = (
            f"You are an expert structural disaster damage engineer examining satellite imagery.\n"
            f"Inspect the un-downsampled 256x256 native crop of Building #{b_id}.\n"
            f"The segmentation model initially flagged this building as: '{predicted_class}'.\n\n"
            f"Required Analysis:\n"
            f"1. Features: Structural condition (roof collapse, rubble pile, intact boundaries, cracks).\n"
            f"2. EMS-98: Assign an official European Macroseismic Grade (Grade 1, Grade 2, Grade 3, Grade 4, or Grade 5).\n"
            f"3. Independent Class: State strictly 'Intact', 'Damaged', or 'Destroyed'.\n"
            f"4. Priority: Declare rescue priority ('Immediate', 'High', 'Standard', or 'Low').\n\n"
            f"Response Template:\n"
            f"Features: <concise observation>\n"
            f"EMS-98: <Grade 1-5>\n"
            f"Independent Class: <Intact / Damaged / Destroyed>\n"
            f"Priority: <Immediate / High / Standard / Low>"
        )

        inspections = []
        conflicts_detected = 0
        vlm_correct_count = 0
        seg_correct_count = 0
        corrected_errors = 0
        gt_evaluated_count = 0

        ems_to_class = {
            "Grade 1": "Intact",
            "Grade 2": "Damaged",
            "Grade 3": "Damaged",
            "Grade 4": "Destroyed",
            "Grade 5": "Destroyed"
        }

        for b in selected_targets:
            b_id = b["building_id"]
            predicted_class = b["class_name"]

            # Retrieve sharp, un-downsampled 256x256 crop
            crop_rgb = self.retriever.crop_patch(image_input, b["bbox"])

            vlm_text = self.vlm.inspect_crop(crop_rgb, prompt)
            is_mock_run = getattr(self.vlm, "is_mock", False)

            # 1. Extract EMS-98 Grade
            grade_match = re.search(r"Grade\s*([1-5])", vlm_text, re.IGNORECASE)
            extracted_grade = f"Grade {grade_match.group(1)}" if grade_match else "Unspecified"

            # 2. Extract Independent VLM Class Prediction
            class_match = re.search(r"Independent Class:\s*(Intact|Damaged|Destroyed)", vlm_text, re.IGNORECASE)
            if class_match:
                vlm_class = class_match.group(1).capitalize()
            elif extracted_grade in ems_to_class:
                vlm_class = ems_to_class[extracted_grade]
            else:
                # Fallback keyword extraction
                if re.search(r"\b(destroyed|collapse|rubble)\b", vlm_text, re.IGNORECASE):
                    vlm_class = "Destroyed"
                elif re.search(r"\b(damaged|partial|cracked)\b", vlm_text, re.IGNORECASE):
                    vlm_class = "Damaged"
                else:
                    vlm_class = "Intact"

            # 3. Scientific Inter-Model Consensus: Does VLM agree with Segmenter?
            is_disagree = (vlm_class.lower() != predicted_class.lower())
            if is_disagree:
                conflicts_detected += 1

            # 4. Ground Truth Verification (if ground truth mask or mask_override is available)
            gt_class = None
            seg_is_correct = None
            vlm_is_correct = None
            error_corrected = False

            if mask_override is not None:
                x, y, w, h = b["bbox"]
                gt_crop = mask_override[y:y+h, x:x+w]
                fg_gt = gt_crop[gt_crop > 0]
                if len(fg_gt) > 0:
                    gt_class_id = int(np.bincount(fg_gt).argmax())
                    gt_class = config.CLASS_MAPPING.get(gt_class_id, "Intact")
                    gt_evaluated_count += 1

                    seg_is_correct = (predicted_class == gt_class)
                    vlm_is_correct = (vlm_class == gt_class)

                    if seg_is_correct:
                        seg_correct_count += 1
                    if vlm_is_correct:
                        vlm_correct_count += 1

                    # Flag true correction: Segmenter was wrong, but VLM corrected it to the Ground Truth!
                    if (not seg_is_correct) and vlm_is_correct:
                        error_corrected = True
                        corrected_errors += 1

            inspections.append({
                "building_id": b_id,
                "bbox": b["bbox"],
                "centroid": b["centroid"],
                "area_pixels": b["area_pixels"],
                "segmenter_class": predicted_class,
                "vlm_class": vlm_class,
                "ems98_grade": extracted_grade,
                "gt_class": gt_class,
                "seg_correct": seg_is_correct,
                "vlm_correct": vlm_is_correct,
                "vlm_corrected_error": error_corrected,
                "conflict_with_segmenter": is_disagree,
                "inspection_text": vlm_text,
                "is_mock_unverified": is_mock_run,
                "crop_rgb": crop_rgb
            })

            if is_mock_run:
                status_icon = "⚠️ Mock Fallback"
            elif is_disagree:
                status_icon = f"⚠️ Conflict (Seg:{predicted_class} vs VLM:{vlm_class})"
            else:
                status_icon = f"✓ Verified ({predicted_class})"
            print(f"  [{status_icon}] Building #{b_id} -> {extracted_grade}")

        # ─── Step 5: SYNTHESIS & REPORT GENERATION ───────────────────────────
        print(f"\n[Thought 4] Compiling forensic report. Detected {conflicts_detected} classification conflicts.")
        if gt_evaluated_count > 0:
            print(f"[Ground Truth Audit] Evaluated {gt_evaluated_count} buildings against Ground Truth:")
            print(f"  - Segmenter Accuracy: {round(seg_correct_count / gt_evaluated_count * 100, 1)}%")
            print(f"  - VLM Accuracy:       {round(vlm_correct_count / gt_evaluated_count * 100, 1)}%")
            print(f"  - VLM Corrected Errors: {corrected_errors} buildings")

        # Compute rescue priority distribution
        immediate_rescues = sum(1 for i in inspections if "immediate" in i["inspection_text"].lower())

        report = {
            "query": user_query,
            "vlm_model_used": self.vlm.model_id,
            "is_mock_run": getattr(self.vlm, "is_mock", False),
            "spatial_census": {
                "total_buildings_detected": total_blds,
                "destroyed_count": counts["Destroyed"],
                "damaged_count": counts["Damaged"],
                "intact_count": counts["Intact"],
            },
            "inspection_summary": {
                "crops_evaluated": len(inspections),
                "conflicts_detected": conflicts_detected,
                "consensus_rate_pct": round((len(inspections) - conflicts_detected) / max(1, len(inspections)) * 100, 2) if not getattr(self.vlm, "is_mock", False) else None,
                "ground_truth_stats": {
                    "gt_crops_evaluated": gt_evaluated_count,
                    "seg_accuracy_pct": round(seg_correct_count / gt_evaluated_count * 100, 2) if gt_evaluated_count > 0 else None,
                    "vlm_accuracy_pct": round(vlm_correct_count / gt_evaluated_count * 100, 2) if (gt_evaluated_count > 0 and not getattr(self.vlm, "is_mock", False)) else None,
                    "vlm_corrected_errors": corrected_errors
                },
                "immediate_rescue_priorities": immediate_rescues
            },
            "detailed_inspections": inspections,
            "segmentation_mask": seg_data["pred_mask"]
        }

        print("✓ [Agent Execution Completed]")
        return report
