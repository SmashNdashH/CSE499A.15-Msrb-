import os
import glob
import torch
import json
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# --- Configuration ---
# Generic paths for repository consistency
IMAGE_ROOT = "./data/images/"
ASSESSMENT_ROOT = "./results/vlm_assessments/"
OUTPUT_ROOT = "./results/clip_scores/"

# Filter for specific image types (e.g., post-disaster states)
IMAGE_SUFFIX_FILTER = "*post_disaster*.png"

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on device: {device}")

def load_clip_model():
    print("Loading CLIP model...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

def get_text_chunks(processor, text, chunk_size=75):
    """
    Tokenizes text and splits into chunks to bypass CLIP's 77-token limit.
    """
    inputs = processor.tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs['input_ids'][0]
    chunks = []
    for i in range(0, len(input_ids), chunk_size):
        chunk_ids = input_ids[i : i + chunk_size]
        chunk_text = processor.tokenizer.decode(chunk_ids, skip_special_tokens=True)
        if chunk_text.strip():
            chunks.append(chunk_text)
    return chunks

def analyze_image_text(model, processor, image_path, full_text):
    try:
        image = Image.open(image_path)
        text_chunks = get_text_chunks(processor, full_text)
        
        if not text_chunks:
            return None

        chunk_raw_cosine = []
        chunk_clip_scores = []
        chunk_clip_100 = []

        # 1. PROCESS IMAGE ONCE OUTSIDE THE LOOP
        image_inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            image_features = model.get_image_features(**image_inputs)
            image_features_norm = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

        # 2. LOOP ONLY THE TEXT
        for chunk in text_chunks:
            text_inputs = processor(
                text=[chunk], 
                return_tensors="pt", 
                padding=True,
                truncation=True
            ).to(device)
            
            with torch.no_grad():
                text_features = model.get_text_features(**text_inputs)
                text_features_norm = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
                
                # A. RAW COSINE SIMILARITY 
                cosine_sim = torch.matmul(text_features_norm, image_features_norm.t()).item()
                chunk_raw_cosine.append(cosine_sim)

                # B. CLIPScore FORMULA: w * max(cos_sim, 0) where w = 2.5
                clip_score = 2.5 * max(cosine_sim, 0)
                chunk_clip_scores.append(clip_score)

                # C. PUBLICATION SCALE: CLIPScore * 100
                clip_100 = clip_score * 100
                chunk_clip_100.append(clip_100)

        if not chunk_raw_cosine:
            return None

        return {
            "num_chunks": len(chunk_raw_cosine),
            "raw_cosine_avg": float(np.mean(chunk_raw_cosine)),
            "clip_score_avg": float(np.mean(chunk_clip_scores)),
            "clip_100_avg": float(np.mean(chunk_clip_100))
        }

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def main():
    model, processor = load_clip_model()
    
    vlm_folders = [f for f in os.listdir(ASSESSMENT_ROOT) if os.path.isdir(os.path.join(ASSESSMENT_ROOT, f))]
    
    for vlm_name in vlm_folders:
        print(f"\n--- Processing VLM: {vlm_name} ---")
        
        vlm_input_path = os.path.join(ASSESSMENT_ROOT, vlm_name)
        vlm_output_path = os.path.join(OUTPUT_ROOT, vlm_name)
        os.makedirs(vlm_output_path, exist_ok=True)
        
        model_clip_100_avgs = []
        assessment_files = glob.glob(os.path.join(vlm_input_path, "*_assessment.txt"))
        
        for text_file in assessment_files:
            filename = os.path.basename(text_file)
            file_id = filename.split('_')[0]
            
            with open(text_file, 'r', encoding='utf-8') as f:
                assessment_text = f.read().strip()
                
            image_folder = os.path.join(IMAGE_ROOT, file_id)
            found_images = glob.glob(os.path.join(image_folder, IMAGE_SUFFIX_FILTER))
            
            if not found_images:
                continue
            image_path = found_images[0]
            
            stats = analyze_image_text(model, processor, image_path, assessment_text)
            
            if stats:
                stats["id"] = file_id
                stats["image_path"] = image_path
                
                output_filename = f"{file_id}_analysis.json"
                with open(os.path.join(vlm_output_path, output_filename), 'w') as out_f:
                    json.dump(stats, out_f, indent=4)
                
                model_clip_100_avgs.append(stats["clip_100_avg"])

                print(f"[{vlm_name}] ID: {file_id} | Chunks: {stats['num_chunks']} | "
                      f"Raw Cosine: {stats['raw_cosine_avg']:.4f} | CLIPScore (0-100): {stats['clip_100_avg']:.2f}")

        if model_clip_100_avgs:
            summary_data = {
                "model_name": vlm_name,
                "total_assessments": len(model_clip_100_avgs),
                "overall_clip_100_avg": float(np.mean(model_clip_100_avgs)),
                "best_clip_score": float(np.max(model_clip_100_avgs)),
                "worst_clip_score": float(np.min(model_clip_100_avgs))
            }
            with open(os.path.join(vlm_output_path, "summary_statistics.json"), 'w') as sum_f:
                json.dump(summary_data, sum_f, indent=4)
            print("-" * 78)
            print(f"Completed {vlm_name}. Final CLIPScore: {summary_data['overall_clip_100_avg']:.2f}")
            print("-" * 78)

if __name__ == "__main__":
    main()
