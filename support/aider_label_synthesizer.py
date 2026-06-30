import os
import glob
import json
import random
import time
import google.generativeai as genai
from tqdm import tqdm
from dotenv import load_dotenv

# --- CONFIGURATION ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_IMG_DIR = os.path.join(PROJECT_ROOT, "data", "aider", "processed_images")
OUTPUT_LABEL_DIR = os.path.join(PROJECT_ROOT, "data", "aider", "processed_labels")

# Create output directory
os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

# Load environment variables (API Keys)
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
GEMINI_API_KEYS = [
    os.getenv('GEMINI_API_KEY_1'),
    os.getenv('GEMINI_API_KEY_2'),
    os.getenv('GEMINI_API_KEY_3'),
    os.getenv('GEMINI_API_KEY_4'),
    os.getenv('GEMINI_API_KEY_5'),
    os.getenv('GEMINI_API_KEY_6'),
    os.getenv('GEMINI_API_KEY_7')
]
api_keys = [k for k in GEMINI_API_KEYS if k and k.strip()]
if not api_keys:
    print("ERROR: No GEMINI API Keys found in .env")
    exit(1)

current_key_idx = 0
genai.configure(api_key=api_keys[current_key_idx])

# We use Flash because this is a simple classification/structuring task
MODEL_NAME = 'gemini-flash-latest'
model = genai.GenerativeModel(MODEL_NAME)

SYSTEM_PROMPT = """You are a geospatial analysis AI. 
Your task is to analyze this aerial/disaster image and synthesize a structured JSON metadata report. 
We need to emulate an instance-segmentation format (xBD schema).
Even though this is an image-level classification dataset, look at the image and identify 1 to 5 distinct "features" (e.g., a building, a road, a tree line, a vehicle). 
For each feature you identify, assign a severity subtype from this exact list:
['destroyed', 'major-damage', 'minor-damage', 'no-damage', 'un-classified']

Respond ONLY with a valid JSON object using the exact schema below. Do not include markdown blocks (```json).

{
  "metadata": {
    "dataset": "AIDER",
    "disaster_type": "<fire|flood|traffic_incident|collapsed_building|normal>"
  },
  "features": {
    "xy": [
      {
        "properties": {
          "feature_type": "<e.g., building, road, vehicle>",
          "subtype": "<severity_from_list_above>"
        }
      }
    ]
  }
}"""

def rotate_key():
    global current_key_idx
    current_key_idx = (current_key_idx + 1) % len(api_keys)
    genai.configure(api_key=api_keys[current_key_idx])
    print(f"\n[Rotated to API Key #{current_key_idx + 1}]")

def upload_to_gemini(path, mime_type=None):
    try:
        return genai.upload_file(path, mime_type=mime_type)
    except Exception as e:
        print(f"Upload failed: {e}")
        return None

def process_image(img_path):
    base_name = os.path.basename(img_path).replace('.jpg', '')
    json_out_path = os.path.join(OUTPUT_LABEL_DIR, f"{base_name}.json")
    
    if os.path.exists(json_out_path):
        try:
            with open(json_out_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                if 'features' in existing_data:
                    return True # Skip already synthesized
        except Exception:
            pass # Continue to overwrite if malformed or simple
        
    gemini_file = None
    try:
        gemini_file = upload_to_gemini(img_path, mime_type="image/jpeg")
        if not gemini_file: return False

        # Prompt Gemini
        # We pass the filename so Gemini knows the AIDER ground-truth class context
        context_prompt = f"System Instruction:\n{SYSTEM_PROMPT}\n\nGround Truth Hint: The image filename is '{base_name}'."
        
        response = model.generate_content([gemini_file, context_prompt])
        
        # Parse output
        output = response.text.strip()
        # Clean markdown if model ignored instruction
        if output.startswith("```json"):
            output = output.replace("```json", "").replace("```", "").strip()
            
        json_data = json.loads(output)
        
        with open(json_out_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2)
            
        return True

    except Exception as e:
        error_msg = str(e).lower()
        if "429" in error_msg or "resource_exhausted" in error_msg or "quota" in error_msg:
            print(f"\nQuota exhausted! Rotating key...")
            rotate_key()
            time.sleep(2) # brief pause before retry
        elif "400" in error_msg or "invalid" in error_msg or "api_key_invalid" in error_msg:
            print(f"\nAPI Key invalid! Rotating key...")
            rotate_key()
        else:
            print(f"\nError parsing/generating {img_path}: {e}")
        return False
        
    finally:
        if gemini_file:
            try:
                genai.delete_file(gemini_file.name)
            except:
                pass

def main():
    img_files = glob.glob(os.path.join(INPUT_IMG_DIR, "*.jpg"))
    if not img_files:
        print(f"No images found in {INPUT_IMG_DIR}. Please run aider_standardizer.py first!")
        return

    print(f"Synthesizing pseudo-xBD JSON metadata for {len(img_files)} AIDER images...")
    
    # We process sequentially because of API rate limits
    for img_path in tqdm(img_files, desc="Synthesizing Labels"):
        success = False
        attempts = 0
        while not success and attempts < len(api_keys):
            success = process_image(img_path)
            attempts += 1
            if not success:
                time.sleep(1)

if __name__ == "__main__":
    main()