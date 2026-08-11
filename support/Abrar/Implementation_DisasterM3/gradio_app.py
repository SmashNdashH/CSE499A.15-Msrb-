import subprocess
subprocess.run(["pip", "install", "-q", "gradio", "qwen-vl-utils", "segmentation-models-pytorch", "opencv-python-headless"], check=True)

import gradio as gr
import torch
import gc
import cv2
import math
import numpy as np
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import segmentation_models_pytorch as smp
from huggingface_hub import hf_hub_download

# Global State
QWEN_MODEL_ID = "AbrarAlam/disasterm3-qwen2.5vl7b-mergedFP"
UNET_REPO_ID = "AbrarAlam/disasterm3-unet-checkpoints-2"
UNET_FILENAME = "best_model.pth"

qwen_model = None
qwen_processor = None
unet_model = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def clear_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def load_qwen():
    global qwen_model, qwen_processor, unet_model
    if unet_model is not None:
        print("Unloading U-Net to free VRAM...")
        unet_model = None
        clear_vram()
    
    if qwen_model is None:
        print("Loading Qwen2.5-VL...")
        qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            QWEN_MODEL_ID,
            device_map="auto",
            torch_dtype=torch.float16
        )
        qwen_processor = AutoProcessor.from_pretrained(QWEN_MODEL_ID)
        print("Qwen2.5-VL Loaded.")

def load_unet():
    global qwen_model, qwen_processor, unet_model
    if qwen_model is not None:
        print("Unloading Qwen2.5-VL to free VRAM...")
        qwen_model = None
        # We don't necessarily need to unload the processor, it's small.
        clear_vram()
    
    if unet_model is None:
        print("Loading U-Net...")
        unet_model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=4,
        )
        weights_path = hf_hub_download(repo_id=UNET_REPO_ID, filename=UNET_FILENAME)
        unet_model.load_state_dict(torch.load(weights_path, map_location=device))
        unet_model.to(device)
        unet_model.eval()
        print("U-Net Loaded.")

def generate_unet_collage_and_counts(image_path):
    load_unet()
    original_img = cv2.imread(image_path)
    if original_img is None:
        return None, 0, 0
    
    img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    
    img_resized = cv2.resize(img_rgb, (512, 512))
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = unet_model(img_tensor)
        preds = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
        
    # Count Intact (Class 1)
    intact_mask = np.where(preds == 1, 255, 0).astype(np.uint8)
    intact_mask_full = cv2.resize(intact_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    contours_intact, _ = cv2.findContours(intact_mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    intact_count = len([c for c in contours_intact if cv2.contourArea(c) > 100])
    
    # Count Damaged (Classes 2 & 3)
    damage_mask = np.where((preds == 2) | (preds == 3), 255, 0).astype(np.uint8)
    damage_mask_full = cv2.resize(damage_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    contours_dmg, _ = cv2.findContours(damage_mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_dmg_boxes = [cv2.boundingRect(c) for c in contours_dmg if cv2.contourArea(c) > 100]
    dmg_count = len(valid_dmg_boxes)
    
    # Create Collage
    valid_dmg_boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
    top_boxes = valid_dmg_boxes[:9]
    
    collage_pil = None
    if len(top_boxes) > 0:
        grid_size = math.ceil(math.sqrt(len(top_boxes)))
        cell_size = 1024 // max(grid_size, 1)
        collage = np.zeros((1024, 1024, 3), dtype=np.uint8)
        for idx, (x, y, bw, bh) in enumerate(top_boxes):
            pad_x, pad_y = int(bw * 0.1), int(bh * 0.1)
            x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
            x2, y2 = min(w, x + bw + pad_x), min(h, y + bh + pad_y)
            crop = img_rgb[y1:y2, x1:x2]
            collage_cell = cv2.resize(crop, (cell_size, cell_size))
            
            row, col = idx // grid_size, idx % grid_size
            start_y, start_x = row * cell_size, col * cell_size
            collage[start_y:start_y+cell_size, start_x:start_x+cell_size] = collage_cell
        collage_pil = Image.fromarray(collage)
    else:
        collage_pil = Image.fromarray(img_rgb)
        
    return collage_pil, intact_count, dmg_count


def run_qwen_inference(image, prompt):
    load_qwen()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    text = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = qwen_processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    
    generated_ids = qwen_model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    return qwen_processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def process_request(image_path, prompt, mode):
    if not image_path:
        return "Please upload an image.", None
        
    if mode == "Hybrid Pipeline (U-Net + VLM)":
        collage_img, intact_count, dmg_count = generate_unet_collage_and_counts(image_path)
        
        # Check if it's a direct counting question
        prompt_lower = prompt.lower()
        if "how many" in prompt_lower and ("damaged" in prompt_lower or "intact" in prompt_lower or "building" in prompt_lower):
            if "intact" in prompt_lower or "undamaged" in prompt_lower:
                answer = f"Based on U-Net spatial extraction, there are {intact_count} intact buildings."
            else:
                answer = f"Based on U-Net spatial extraction, there are {dmg_count} damaged buildings."
            return answer, collage_img
        
        # Otherwise, feed collage to Qwen
        answer = run_qwen_inference(collage_img, prompt)
        return answer, collage_img
    else:
        # Direct Mode
        img = Image.open(image_path).convert("RGB")
        answer = run_qwen_inference(img, prompt)
        return answer, img

# Custom CSS for aesthetics
css = """
body { font-family: 'Inter', sans-serif; }
.gradio-container { max-width: 900px !important; }
.card { border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); padding: 20px; }
.header { text-align: center; margin-bottom: 20px; }
.header h1 { font-weight: 800; color: #1E3A8A; }
.header p { color: #4B5563; }
"""

with gr.Blocks(css=css, title="DisasterM3 Interactive") as demo:
    with gr.Column(elem_classes="header"):
        gr.Markdown("# DisasterM3 Interactive Explorer")
        gr.Markdown("Test the fine-tuned **Qwen2.5-VL 7B** and the **Hybrid U-Net Pipeline** on disaster imagery.")
        
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="filepath", label="Upload Image")
            mode_input = gr.Radio(
                choices=["Direct VLM", "Hybrid Pipeline (U-Net + VLM)"],
                value="Direct VLM",
                label="Inference Mode",
                info="Dynamic VRAM management will swap models as needed."
            )
            prompt_input = gr.Textbox(
                label="Ask a question about the image",
                placeholder="e.g. What type of disaster is shown here?",
                lines=3
            )
            submit_btn = gr.Button("Analyze", variant="primary")
            
        with gr.Column(scale=1):
            output_text = gr.Markdown("### Results will appear here...")
            output_image = gr.Image(label="Processed View (Original or Collage)", interactive=False)
            
    gr.Markdown("### Quick Prompts / Conversation Starters")
    gr.Markdown("*These prompts match the DisasterM3 benchmark MCQ format the model was fine-tuned on.*")
    gr.Examples(
        examples=[
            ["Which of the following disaster types does this image show? A: Earthquake, B: Fire, C: Flood, D: Hurricane."],
            ["How many damaged or destroyed buildings can you identify in this image? A: 0, B: 1-5, C: 6-10, D: More than 10."],
            ["What is the land use type of the scene shown in this image? A: Residential, B: Commercial, C: Agricultural, D: Industrial."],
            ["Is the road in this image blocked by debris or structural damage? A: Yes, completely blocked, B: Partially blocked, C: Minor debris, D: No damage visible."],
            ["Are the bearing bodies (structural pillars/foundations) of the buildings severely damaged? A: Yes, B: No, C: Partially, D: Cannot determine."],
            ["Describe the extent of the damage in the scene and recommend immediate emergency responses."]
        ],
        inputs=prompt_input,
        label="Click a prompt to fill the textbox"
    )

    submit_btn.click(
        fn=process_request,
        inputs=[image_input, prompt_input, mode_input],
        outputs=[output_text, output_image]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=True)
