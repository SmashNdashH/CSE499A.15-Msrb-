"""
vlm_engine.py
-------------
Multi-Model Vision-Language Engine supporting:
  1. Team Fine-Tuned Model: AbrarAlam/disasterm3-qwen2.5vl7b-mergedFP
  2. Base Foundation Model: Qwen/Qwen2.5-VL-7B-Instruct
  3. Google Multimodal Benchmark: google/paligemma2-3b-pt-448 (or 10B)

Includes 4-bit NF4 quantization to fit within Kaggle T4 GPU (16 GB VRAM) using ~5.5 GB memory.

Author: Tamanna (Member B, Group 5)
Project: CSE499B DisasterM3 Capstone
"""

import sys
import torch
from PIL import Image
import numpy as np
from typing import Optional, Dict, Any, Union
import config

# Check for GPU and transformers availability
HAS_CUDA = torch.cuda.is_available()


class MultiVLMEngine:
    """
    Unified interface for loading and querying different VLMs under the same API.
    Handles device management, chat formatting, 4-bit quantization, and output decoding.
    """

    def __init__(
        self,
        model_key: str = "finetuned",
        custom_model_id: Optional[str] = None,
        load_in_4bit: bool = True,
        device: Optional[str] = None
    ):
        self.model_key = model_key
        meta = config.VLM_MODELS.get(model_key, config.VLM_MODELS["finetuned"])
        self.model_id = custom_model_id or meta["model_id"]
        self.model_type = meta["model_type"]
        self.device = device or ("cuda" if HAS_CUDA else "cpu")
        self.load_in_4bit = load_in_4bit and HAS_CUDA
        self.model = None
        self.processor = None

        print(f"[VLMEngine] Initializing '{meta['name']}' from Hugging Face ({self.model_id})...")
        self._load_model()

    def _load_model(self):
        """Loads processor and model weights with optional 4-bit NF4 quantization."""
        try:
            from transformers import AutoProcessor, BitsAndBytesConfig

            # Configure 4-bit quantization if on CUDA
            bnb_config = None
            if self.load_in_4bit:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
                print("[VLMEngine] 4-bit NormalFloat (NF4) quantization enabled (~5.5 GB VRAM).")

            if self.model_type == "qwen":
                from transformers import Qwen2_5_VLForConditionalGeneration
                self.processor = AutoProcessor.from_pretrained(self.model_id)
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    self.model_id,
                    quantization_config=bnb_config,
                    device_map="auto" if HAS_CUDA else None,
                    torch_dtype=torch.float16 if HAS_CUDA else torch.float32,
                    low_cpu_mem_usage=True
                )
            elif self.model_type == "paligemma":
                from transformers import PaliGemmaForConditionalGeneration
                self.processor = AutoProcessor.from_pretrained(self.model_id)
                self.model = PaliGemmaForConditionalGeneration.from_pretrained(
                    self.model_id,
                    quantization_config=bnb_config,
                    device_map="auto" if HAS_CUDA else None,
                    torch_dtype=torch.float16 if HAS_CUDA else torch.float32,
                    low_cpu_mem_usage=True
                )
            print(f"[VLMEngine] Successfully loaded {self.model_id}.")

        except Exception as e:
            print(f"[VLMEngine] Warning: Could not initialize model from Hugging Face: {e}")
            print("[VLMEngine] Running in fallback heuristic/mock mode for dry runs.")
            self.model = None
            self.processor = None

    def inspect_crop(self, crop_image_rgb: np.ndarray, prompt: str) -> str:
        """
        Sends an extracted 256x256 building crop along with an EMS-98 prompt to the VLM.
        """
        if self.model is None or self.processor is None:
            # Fallback heuristic response for local testing without GPU/HF access
            return self._mock_inspection(crop_image_rgb, prompt)

        pil_img = Image.fromarray(crop_image_rgb)

        try:
            if self.model_type == "qwen":
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": pil_img},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
                text_input = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = self.processor(
                    text=[text_input],
                    images=[pil_img],
                    return_tensors="pt"
                ).to(self.model.device)

                with torch.no_grad():
                    output_ids = self.model.generate(**inputs, max_new_tokens=300)
                generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
                response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                return response.strip()

            elif self.model_type == "paligemma":
                prompt_text = f"answer en {prompt}"
                inputs = self.processor(
                    text=prompt_text,
                    images=pil_img,
                    return_tensors="pt"
                ).to(self.model.device)

                with torch.no_grad():
                    output = self.model.generate(**inputs, max_new_tokens=180)
                response = self.processor.decode(output[0], skip_special_tokens=True)
                return response.replace(prompt_text, "").strip()

        except Exception as err:
            return f"[VLM Generation Error]: {err}"

    def query_global(self, full_image_rgb: np.ndarray, prompt: str) -> str:
        """
        Queries the model on the full downsampled image (used for Baseline Config A comparisons).
        """
        if self.model is None or self.processor is None:
            return "Baseline downsampled observation: Unable to clearly resolve individual building details due to pixel downsampling."

        pil_img = Image.fromarray(full_image_rgb)
        if self.model_type == "qwen":
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_img},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], images=[pil_img], return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                out = self.model.generate(**inputs, max_new_tokens=250)
            return self.processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
        else:
            prompt_text = f"answer en {prompt}"
            inputs = self.processor(text=prompt_text, images=pil_img, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                out = self.model.generate(**inputs, max_new_tokens=150)
            return self.processor.decode(out[0], skip_special_tokens=True).replace(prompt_text, "").strip()

    def _mock_inspection(self, crop: np.ndarray, prompt: str) -> str:
        """Deterministic mock response for offline debugging without active GPU."""
        # Simple color variance heuristic
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        std_dev = float(np.std(gray))

        if std_dev > 50:
            return (
                "Damage: Severe structural compromise, roof failure with extensive debris scatter.\n"
                "EMS-98: Grade 5\n"
                "Priority: Immediate\n"
                "Agreement: Agree"
            )
        elif std_dev > 30:
            return (
                "Damage: Moderate wall fractures and partial roof displacement.\n"
                "EMS-98: Grade 3\n"
                "Priority: High\n"
                "Agreement: Agree"
            )
        else:
            return (
                "Damage: Intact roofline with no visible macro-structural collapses.\n"
                "EMS-98: Grade 1\n"
                "Priority: Low\n"
                "Agreement: Disagree"
            )
