"""
tools.py
--------
Deterministic Tools for Agentic Vision RAG:
  1. SegmentationLocatorTool  - Perception: Locates building instances & predicts damage masks.
  2. HighResVisualRetrieverTool - Visual RAG: Extracts un-downsampled 256x256 crops.
  3. EMS98KnowledgeRAGTool     - Knowledge RAG: Retrieves European Macroseismic Scale standards.

Author: Tamanna (Member B, Group 5)
Project: CSE499B DisasterM3 Capstone
"""

import os
import cv2
import numpy as np
import torch
from typing import Dict, List, Any, Optional, Union
import config

try:
    import segmentation_models_pytorch as smp
    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False


class SegmentationLocatorTool:
    """
    Perception Tool:
    Runs the trained segmentation model (ResNet-50 U-Net) to detect building footprints
    and classify pixel-level disaster damage (0=Background, 1=Intact, 2=Damaged, 3=Destroyed).
    Extracts individual building bounding boxes, centroids, and areas using connected components.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        encoder_name: str = "resnet34"
    ):
        self.device = device
        self.encoder_name = encoder_name
        self.checkpoint_path = checkpoint_path or config.DEFAULT_WEIGHTS_PATH
        self.model = None

        if SMP_AVAILABLE:
            self._load_model()
        else:
            print("[Warning] segmentation_models_pytorch not installed. Running in mock/mask-only mode.")

    def _load_model(self):
        """Initializes and loads the trained segmentation network."""
        try:
            self.model = smp.Unet(
                encoder_name=self.encoder_name,
                encoder_weights=None,
                in_channels=3,
                classes=4
            )
            if os.path.exists(self.checkpoint_path):
                print(f"[SegmentationTool] Loading weights from: {self.checkpoint_path}")
                checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
                state_dict = checkpoint.get("state_dict", checkpoint)
                clean_state = {k.replace("model.", "").replace("module.", ""): v for k, v in state_dict.items()}
                self.model.load_state_dict(clean_state, strict=False)
            else:
                print(f"[SegmentationTool] Checkpoint not found at {self.checkpoint_path}. Initialized base architecture.")
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"[SegmentationTool] Error loading model: {e}. Model will fallback to synthetic/mask parsing.")
            self.model = None

    def predict_mask(self, image_rgb: np.ndarray) -> np.ndarray:
        """Executes forward pass to generate [H, W] predicted damage class map."""
        if self.model is None:
            # Fallback for testing when model weights are not loaded locally
            H, W = image_rgb.shape[:2]
            return np.zeros((H, W), dtype=np.uint8)

        # Standard ImageNet normalization matching training
        norm = image_rgb.astype(np.float32) / 255.0
        norm = (norm - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        tensor = torch.from_numpy(norm).permute(2, 0, 1).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        return pred

    def run(
        self,
        image_input: Union[str, np.ndarray],
        mask_override: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Locates buildings and extracts their spatial properties.
        Args:
            image_input: File path to image or RGB numpy array.
            mask_override: Optional precomputed mask (e.g. from validation dataset).
        Returns:
            Dictionary containing building instances, bounding boxes, and summary statistics.
        """
        if isinstance(image_input, str):
            img_bgr = cv2.imread(image_input)
            if img_bgr is None:
                raise FileNotFoundError(f"Could not read image from: {image_input}")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = image_input

        H, W = img_rgb.shape[:2]

        # Obtain segmentation mask
        if mask_override is not None:
            pred_mask = mask_override
        else:
            pred_mask = self.predict_mask(img_rgb)

        buildings = []
        building_id = 0

        # Extract individual building instances across foreground classes
        for cls_id in [1, 2, 3]:  # 1=Intact, 2=Damaged, 3=Destroyed
            binary = (pred_mask == cls_id).astype(np.uint8)
            if binary.sum() == 0:
                continue

            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                binary, connectivity=8
            )

            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area < config.MIN_BUILDING_AREA_PX or area > config.MAX_BUILDING_AREA_PX:
                    continue

                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]

                building_id += 1
                buildings.append({
                    "building_id": building_id,
                    "class_id": cls_id,
                    "class_name": config.CLASS_MAPPING[cls_id],
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "centroid": [int(centroids[i][0]), int(centroids[i][1])],
                    "area_pixels": int(area)
                })

        counts = {
            "Intact": sum(1 for b in buildings if b["class_id"] == 1),
            "Damaged": sum(1 for b in buildings if b["class_id"] == 2),
            "Destroyed": sum(1 for b in buildings if b["class_id"] == 3),
        }

        return {
            "image_size": [H, W],
            "total_buildings": len(buildings),
            "counts": counts,
            "buildings": buildings,
            "pred_mask": pred_mask
        }


class HighResVisualRetrieverTool:
    """
    Visual RAG Tool:
    Extracts high-resolution, un-downsampled crops from the original full-size satellite image.
    Bypasses the VLM's global resolution downsampling trap by feeding sharp 256x256 crops.
    """

    @staticmethod
    def crop_patch(
        image_input: Union[str, np.ndarray],
        bbox: List[int],
        pad: int = config.CROP_PADDING,
        target_size: int = config.CROP_SIZE
    ) -> np.ndarray:
        """
        Extracts a zoomed crop corresponding to a building bounding box.
        """
        if isinstance(image_input, str):
            img_bgr = cv2.imread(image_input)
            if img_bgr is None:
                raise FileNotFoundError(f"Failed to load image: {image_input}")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = image_input

        H, W = img_rgb.shape[:2]
        x, y, w, h = bbox

        # Add context padding around the building
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(W, x + w + pad)
        y2 = min(H, y + h + pad)

        crop = img_rgb[y1:y2, x1:x2]

        # Guard against zero-area crops
        if crop.shape[0] == 0 or crop.shape[1] == 0:
            crop = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        else:
            crop = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)

        return crop


class EMS98KnowledgeRAGTool:
    """
    Knowledge RAG Tool:
    Provides domain-specific structural damage definitions and triage recommendations
    according to the European Macroseismic Scale (EMS-98).
    """

    @staticmethod
    def query(keyword: str = "") -> Dict[str, Any]:
        """Queries the EMS-98 database for relevant damage criteria."""
        keyword = keyword.lower().strip()
        if not keyword:
            return config.EMS98_GRADES

        matches = {}
        for grade, data in config.EMS98_GRADES.items():
            if (keyword in grade.lower() or
                keyword in data["name"].lower() or
                keyword in data["damage_type"].lower() or
                keyword in data["rescue_priority"].lower()):
                matches[grade] = data

        return matches if matches else config.EMS98_GRADES
