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
        encoder_name: Optional[str] = None
    ):
        self.device = device
        self.encoder_name = encoder_name or config.DEFAULT_SEGMENTATION_ENCODER
        self.checkpoint_path = checkpoint_path or config.DEFAULT_WEIGHTS_PATH
        self.model = None

        model_role = "Team Champion (ResNet-50, mIoU 0.5449)" if "50" in self.encoder_name else "B1 Architecture Baseline (ResNet-34, mIoU 0.4619)"
        print(f"[SegmentationTool] Selected backbone: {self.encoder_name} [{model_role}]")

        if SMP_AVAILABLE:
            self._load_model()
        else:
            print("[Warning] segmentation_models_pytorch not installed. Running in mock/mask-only mode.")

    def _resolve_checkpoint(self) -> Optional[str]:
        """Resolves local checkpoint or downloads the Team Champion ResNet-50 weights from HF."""
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            return self.checkpoint_path

        # Candidate local paths
        candidates = [
            self.checkpoint_path,
            "/kaggle/input/disasterm3-unet-champion/final_model.pth",
            "/kaggle/input/disasterm3-unet-champion/best_model.pth",
            "/kaggle/working/final_model.pth",
            "/kaggle/working/best_model.pth",
            str(config.PLAN2_DIR.parents[1] / "Champion_Task" / "final_model.pth"),
        ]
        for c in candidates:
            if c and os.path.exists(c):
                return c

        # Attempt to auto-fetch official Team Champion from Hugging Face
        try:
            from huggingface_hub import hf_hub_download
            for repo_info in config.CHAMPION_HF_REPOS:
                print(f"[SegmentationTool] Downloading {repo_info['desc']} from HF ({repo_info['repo_id']})...")
                try:
                    downloaded = hf_hub_download(
                        repo_id=repo_info["repo_id"],
                        filename=repo_info["filename"],
                        token=config.HF_TOKEN,
                        local_dir="/kaggle/working" if config.IS_KAGGLE else str(config.PLAN2_DIR / "outputs")
                    )
                    if downloaded and os.path.exists(downloaded):
                        print(f"✓ Checkpoint successfully retrieved: {downloaded}")
                        return downloaded
                except Exception as dl_err:
                    print(f"Notice: HF repo {repo_info['repo_id']} download skipped: {dl_err}")
        except ImportError:
            pass

        return None

    def _load_model(self):
        """Initializes and loads the trained segmentation network with automatic tensor shape verification."""
        try:
            resolved_ckpt = self._resolve_checkpoint()
            clean_state = None
            detected_encoder = self.encoder_name

            if resolved_ckpt and os.path.exists(resolved_ckpt):
                print(f"[SegmentationTool] Loading weights from: {resolved_ckpt}")
                checkpoint = torch.load(resolved_ckpt, map_location=self.device)
                state_dict = checkpoint.get("state_dict", checkpoint)
                clean_state = {k.replace("model.", "").replace("module.", ""): v for k, v in state_dict.items()}

                # Auto-verify architecture against kernel dimension to eliminate layer shape mismatches:
                # ResNet-50 bottleneck conv1: [64, 64, 1, 1] (Team Champion)
                # ResNet-34 basicblock conv1: [64, 64, 3, 3] (B1 Baseline)
                for k, v in clean_state.items():
                    if "layer1.0.conv1.weight" in k:
                        if v.shape == torch.Size([64, 64, 1, 1]):
                            detected_encoder = "resnet50"
                            print("[SegmentationTool] Verified weights: ResNet-50 Bottleneck (Team Champion Model).")
                        elif v.shape == torch.Size([64, 64, 3, 3]):
                            detected_encoder = "resnet34"
                            print("[SegmentationTool] Verified weights: ResNet-34 BasicBlock (B1 Baseline Model).")
                        break

            self.encoder_name = detected_encoder
            self.model = smp.Unet(
                encoder_name=self.encoder_name,
                encoder_weights=None,
                in_channels=3,
                classes=4
            )

            if clean_state is not None:
                self.model.load_state_dict(clean_state, strict=False)
                print(f"[SegmentationTool] Successfully loaded {self.encoder_name} U-Net weights.")
            else:
                print(f"[SegmentationTool] Checkpoint not located. Initialized base {self.encoder_name} architecture.")

            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"[SegmentationTool] Error loading model: {e}. Model will fallback to synthetic/mask parsing.")
            self.model = None

    @staticmethod
    def compute_mask_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, float]:
        """
        Rigorously evaluates predicted segmentation mask against Ground Truth mask
        using the project's standard 4-class evaluation protocol (E-protocol).
        Classes: 0=Background, 1=Intact, 2=Damaged, 3=Destroyed.
        """
        metrics = {}
        ious = []
        for cls_id, cls_name in config.CLASS_MAPPING.items():
            pred_c = (pred_mask == cls_id)
            gt_c = (gt_mask == cls_id)
            intersection = np.logical_and(pred_c, gt_c).sum()
            union = np.logical_or(pred_c, gt_c).sum()
            iou = (intersection / union) if union > 0 else 1.0 if (pred_c.sum() == 0 and gt_c.sum() == 0) else 0.0
            ious.append(iou)
            metrics[f"{cls_name.lower()}_iou"] = float(round(iou, 4))

        metrics["miou"] = float(round(np.mean(ious), 4))

        # Calculate building-level macro metrics
        fg_pred = (pred_mask > 0)
        fg_gt = (gt_mask > 0)
        tp = np.logical_and(fg_pred, fg_gt).sum()
        fp = np.logical_and(fg_pred, ~fg_gt).sum()
        fn = np.logical_and(~fg_pred, fg_gt).sum()

        precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        f2 = (5 * precision * recall / (4 * precision + recall)) if (4 * precision + recall) > 0 else 0.0

        metrics["macro_precision"] = float(round(precision, 4))
        metrics["macro_recall"] = float(round(recall, 4))
        metrics["f1"] = float(round(f1, 4))
        metrics["f2"] = float(round(f2, 4))
        return metrics

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
