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
                f"You are an expert structural disaster engineer inspecting satellite imagery.\n"
                f"You are examining an un-downsampled 256x256 crop of Building #{b_id}.\n"
                f"The automated segmentation network predicted this building as: '{predicted_class}'.\n\n"
                f"Task:\n"
                f"1. Describe specific visual structural features (e.g. collapsed roof, wall cracks, debris, intact roofline).\n"
                f"2. Assign an official EMS-98 Grade (Grade 1, Grade 2, Grade 3, Grade 4, or Grade 5).\n"
                f"3. Declare rescue priority (Immediate, High, Standard, or Low).\n"
                f"4. State whether you Agree or Disagree with the segmenter's prediction.\n\n"
                f"Format:\n"
                f"Features: <details>\n"
                f"EMS-98: <Grade X>\n"
                f"Priority: <Triage level>\n"
                f"Agreement: <Agree / Disagree>"
            )

            vlm_text = self.vlm.inspect_crop(crop_rgb, prompt)

            # Parse agreement and grade
            is_disagree = "disagree" in vlm_text.lower()
            if is_disagree:
                conflicts_detected += 1

            # Extract EMS-98 grade
            grade_match = re.search(r"Grade\s*([1-5])", vlm_text, re.IGNORECASE)
            extracted_grade = f"Grade {grade_match.group(1)}" if grade_match else "Unspecified"

            inspections.append({
                "building_id": b_id,
                "bbox": b["bbox"],
                "centroid": b["centroid"],
                "area_pixels": b["area_pixels"],
                "initial_segmentation_class": predicted_class,
                "ems98_grade": extracted_grade,
                "inspection_text": vlm_text,
                "conflict_with_segmenter": is_disagree,
                "crop_rgb": crop_rgb
            })

            status_icon = "⚠️ Conflict" if is_disagree else "✓ Verified"
            print(f"  [{status_icon}] Building #{b_id} ({predicted_class}) -> {extracted_grade}")

        # ─── Step 5: SYNTHESIS & REPORT GENERATION ───────────────────────────
        print(f"\n[Thought 4] Compiling forensic report. Detected {conflicts_detected} classification conflicts.")

        # Compute rescue priority distribution
        immediate_rescues = sum(1 for i in inspections if "immediate" in i["inspection_text"].lower())

        report = {
            "query": user_query,
            "vlm_model_used": self.vlm.model_id,
            "spatial_census": {
                "total_buildings_detected": total_blds,
                "destroyed_count": counts["Destroyed"],
                "damaged_count": counts["Damaged"],
                "intact_count": counts["Intact"],
            },
            "inspection_summary": {
                "crops_evaluated": len(inspections),
                "conflicts_detected": conflicts_detected,
                "immediate_rescue_priorities": immediate_rescues
            },
            "detailed_inspections": inspections,
            "segmentation_mask": seg_data["pred_mask"]
        }

        print("✓ [Agent Execution Completed]")
        return report
