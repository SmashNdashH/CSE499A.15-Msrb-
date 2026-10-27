"""
agent.py
--------
Agent for Disaster Damage Assessment.
Segments the tile, crops each selected building at native resolution, asks the VLM for the
building's damage class on its own, then asks it for an EMS-98 grade and a rescue priority.

The VLM is never told the segmenter's class. A ground-truth mask, when one is given, is used only
to score the segmenter's and the VLM's classes; it never replaces the segmenter's prediction.

Author: Tamanna (Member B, Group 5)
Project: CSE499B DisasterM3 Capstone
"""

import re
from typing import Dict, Any, Optional
import numpy as np
from tools import SegmentationLocatorTool, HighResVisualRetrieverTool, EMS98KnowledgeRAGTool
from vlm_engine import MultiVLMEngine
import config

# Asked on its own, so the answer depends only on the crop.
CLASS_PROMPT = (
    "This is a satellite image crop centred on one building after a disaster.\n"
    "What is the damage level of the building in the centre?\n"
    "Answer with one word: Intact, Damaged or Destroyed."
)


def inspection_prompt(building_id: int) -> str:
    """The EMS-98 inspection prompt for one building. It never contains the segmenter's class."""
    return (
        f"You are a structural engineer examining a satellite image crop centred on Building #{building_id}.\n"
        "Describe the building's condition and grade it.\n"
        "Answer in this format:\n"
        "Features: <roof, walls, debris>\n"
        "EMS-98: <Grade 1 to Grade 5>\n"
        "Priority: <Immediate / High / Standard / Low>"
    )


def parse_class(answer: str) -> Optional[str]:
    """The damage class named in the answer, or None when it names none or more than one."""
    found = {m.lower() for m in re.findall(r"\b(intact|damaged|destroyed)\b", answer, re.IGNORECASE)}
    return found.pop().capitalize() if len(found) == 1 else None


class AgenticVisionRAG:
    """
    Agent that locates buildings with the segmentation tool, inspects native-resolution crops
    of the selected buildings with a VLM, and compiles a structured damage report.
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
        pred_mask: Optional[np.ndarray] = None,
        gt_mask: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Runs the agent on one tile.
        Args:
            image_input: File path to the after-image or an RGB numpy array.
            pred_mask: A segmenter prediction made earlier (0-3 per pixel). When None, the
                segmentation tool predicts one from the image.
            gt_mask: Ground truth (0-3 per pixel), used only to score the segmenter's and the
                VLM's classes. It is never passed to the segmentation tool.
        """
        is_mock_run = getattr(self.vlm, "is_mock", False)
        print("\n" + "=" * 65)
        print(f"[Agent Initiated] Model: {self.vlm.model_key}")
        print(f"[User Query]: {user_query}")
        print("=" * 65)

        # ─── Step 1: PERCEPTION (Locate building footprints) ─────────────────
        print("\n[Thought 1] Locating all buildings and obtaining damage masks using segmentation tool...")
        seg_data = self.seg_tool.run(image_input, mask_override=pred_mask)

        total_blds = seg_data["total_buildings"]
        counts = seg_data["counts"]
        print(f"[Observation 1] Detected {total_blds} buildings total: "
              f"{counts['Destroyed']} Destroyed, {counts['Damaged']} Damaged, {counts['Intact']} Intact.")

        # ─── Step 2: DYNAMIC TRIAGE (Prioritize critical targets) ────────────
        print("\n[Thought 2] Prioritizing severe structures (Destroyed/Damaged) for native-resolution crops...")
        critical_targets = [b for b in seg_data["buildings"] if b["class_id"] in [2, 3]]

        # If few damaged buildings, include a sample of intact buildings for balance
        if len(critical_targets) < max_crops_to_inspect:
            intact_samples = [b for b in seg_data["buildings"] if b["class_id"] == 1]
            critical_targets.extend(intact_samples[:(max_crops_to_inspect - len(critical_targets))])

        selected_targets = critical_targets[:max_crops_to_inspect]
        if selected_targets:
            print(f"[Observation 2] Selected {len(selected_targets)} building crops for inspection.")
        else:
            print("[Observation 2] No buildings found on this tile; nothing to inspect.")

        # ─── Step 3: KNOWLEDGE (EMS-98 table) ────────────────────────────────
        ems_rules = self.knowledge.query()
        print(f"[Observation 3] EMS-98 table loaded ({len(ems_rules)} grades); it is not put into the prompt.")

        # ─── Step 4: CROP INSPECTION ─────────────────────────────────────────
        inspections = []
        agreements = 0
        conflicts_detected = 0
        unparsed = 0
        gt_evaluated_count = 0
        no_gt_building = 0
        seg_correct_count = 0
        vlm_correct_count = 0
        corrected_errors = 0

        for b in selected_targets:
            b_id = b["building_id"]
            seg_class = b["class_name"]

            # Retrieve sharp, un-downsampled 256x256 crop
            crop_rgb = self.retriever.crop_patch(image_input, b["bbox"])

            # 1. The class, asked on its own
            class_answer = self.vlm.inspect_crop(crop_rgb, CLASS_PROMPT)
            vlm_class = None if is_mock_run else parse_class(class_answer)

            # 2. The EMS-98 inspection, with a prompt built for this building
            prompt = inspection_prompt(b_id)
            vlm_text = self.vlm.inspect_crop(crop_rgb, prompt)
            grade_match = re.search(r"Grade\s*([1-5])", vlm_text, re.IGNORECASE)
            extracted_grade = f"Grade {grade_match.group(1)}" if grade_match else "Unspecified"
            priority_match = re.search(r"Priority:\s*(Immediate|High|Standard|Low)", vlm_text, re.IGNORECASE)
            priority = priority_match.group(1).capitalize() if priority_match else "Unspecified"

            # 3. Agreement between the two models (not accuracy)
            if vlm_class is None:
                is_disagree = None
                unparsed += 1
            else:
                is_disagree = vlm_class != seg_class
                if is_disagree:
                    conflicts_detected += 1
                else:
                    agreements += 1

            # 4. Ground-truth check, from gt_mask only
            gt_class = None
            seg_is_correct = None
            vlm_is_correct = None
            error_corrected = False

            if gt_mask is not None:
                x, y, w, h = b["bbox"]
                gt_crop = gt_mask[y:y + h, x:x + w]
                fg_gt = gt_crop[(gt_crop >= 1) & (gt_crop <= 3)]
                if len(fg_gt) > 0:
                    gt_class = config.CLASS_MAPPING[int(np.bincount(fg_gt).argmax())]
                    gt_evaluated_count += 1
                    seg_is_correct = seg_class == gt_class
                    seg_correct_count += int(seg_is_correct)
                    if vlm_class is not None:
                        vlm_is_correct = vlm_class == gt_class
                        vlm_correct_count += int(vlm_is_correct)
                        # The segmenter was wrong and the VLM matched the ground truth
                        error_corrected = (not seg_is_correct) and vlm_is_correct
                        corrected_errors += int(error_corrected)
                else:
                    no_gt_building += 1

            inspections.append({
                "building_id": b_id,
                "bbox": b["bbox"],
                "centroid": b["centroid"],
                "area_pixels": b["area_pixels"],
                "segmenter_class": seg_class,
                "vlm_class_answer": class_answer,
                "vlm_class": vlm_class,
                "ems98_grade": extracted_grade,
                "priority": priority,
                "gt_class": gt_class,
                "seg_correct": seg_is_correct,
                "vlm_correct": vlm_is_correct,
                "vlm_corrected_error": error_corrected,
                "conflict_with_segmenter": is_disagree,
                "inspection_prompt": prompt,
                "inspection_text": vlm_text,
                "is_mock_unverified": is_mock_run,
                "crop_rgb": crop_rgb
            })

            if is_mock_run:
                status = "Mock VLM, not a real answer"
            elif vlm_class is None:
                status = f"VLM class unparsed ({class_answer[:40]!r})"
            elif is_disagree:
                status = f"Conflict (Seg:{seg_class} vs VLM:{vlm_class})"
            else:
                status = f"Agree ({seg_class})"
            print(f"  [{status}] Building #{b_id} -> {extracted_grade}")

        # ─── Step 5: SYNTHESIS & REPORT GENERATION ───────────────────────────
        print(f"\n[Thought 4] Compiling report: {agreements} agree, {conflicts_detected} conflict, "
              f"{unparsed} unparsed VLM class answers.")
        if gt_evaluated_count > 0:
            print(f"[Ground Truth Check] {gt_evaluated_count} inspected buildings have ground-truth building pixels "
                  f"in their box; {no_gt_building} don't:")
            print(f"  - Segmenter correct: {seg_correct_count}")
            print(f"  - VLM correct:       {vlm_correct_count}")
            print(f"  - VLM corrected the segmenter: {corrected_errors}")

        parsed = agreements + conflicts_detected
        report = {
            "query": user_query,
            "vlm_model_used": self.vlm.model_id,
            "is_mock_run": is_mock_run,
            "spatial_census": {
                "total_buildings_detected": total_blds,
                "destroyed_count": counts["Destroyed"],
                "damaged_count": counts["Damaged"],
                "intact_count": counts["Intact"],
            },
            "inspection_summary": {
                "crops_evaluated": len(inspections),
                "agreements": agreements,
                "conflicts_detected": conflicts_detected,
                "vlm_class_unparsed": unparsed,
                "agreement_with_segmenter_pct": round(agreements / parsed * 100, 2) if (parsed and not is_mock_run) else None,
                "ground_truth_stats": {
                    "gt_crops_evaluated": gt_evaluated_count,
                    "crops_without_gt_building": no_gt_building if gt_mask is not None else None,
                    "seg_accuracy_pct": round(seg_correct_count / gt_evaluated_count * 100, 2) if gt_evaluated_count > 0 else None,
                    "vlm_accuracy_pct": round(vlm_correct_count / gt_evaluated_count * 100, 2) if (gt_evaluated_count > 0 and not is_mock_run) else None,
                    "vlm_corrected_errors": corrected_errors if gt_evaluated_count > 0 else None
                },
                "immediate_rescue_priorities": sum(1 for i in inspections if i["priority"] == "Immediate")
            },
            "detailed_inspections": inspections,
            "segmentation_mask": seg_data["pred_mask"]
        }

        print("[Agent Execution Completed]")
        return report
