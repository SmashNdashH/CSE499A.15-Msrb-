import os
import glob
import json
import time
import subprocess
import sys

try:
    from google import genai
    from google.genai import types
    from PIL import Image
    from dotenv import load_dotenv
except ImportError:
    print("Missing dependencies. Installing google-genai, pillow, and python-dotenv...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "google-genai", "pillow", "python-dotenv"])
    from google import genai
    from google.genai import types
    from PIL import Image
    from dotenv import load_dotenv

# --- PATH CONFIGURATION ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))
PROCESSED_LABEL_DIR = os.path.join(PROJECT_ROOT, "data", "processed_labels")
PROCESSED_IMG_DIR = os.path.join(PROJECT_ROOT, "data", "processed_images")
DATASET_OUT_PATH = os.path.join(PROJECT_ROOT, "dataset", "train_dataset.jsonl")
os.makedirs(os.path.dirname(DATASET_OUT_PATH), exist_ok=True)

# --- CONFIGURATION ---
# ✏️ Paste your Google Gemini API keys here (or use .env file)
GEMINI_API_KEYS = [
    # 'paste_your_api_key_here',
]

# Extract all keys from .env that start with GEMINI_API_KEY
env_keys = [v.strip() for k, v in os.environ.items() if k.startswith("GEMINI_API_KEY") and v.strip()]
GEMINI_API_KEYS.extend(env_keys)

# Set the first valid key as the default env var (for compatibility)
valid_keys = [k for k in GEMINI_API_KEYS if k and k.strip()]
if valid_keys:
    os.environ['GEMINI_API_KEY'] = valid_keys[0]

MODELS = [
    'gemini-3.5-flash',
    'gemini-2.5-flash'
]
TARGET_SAMPLE_COUNT = 220  # The script will stop automatically once this total is reached

# --- SYSTEM PROMPT ---
SYSTEM_INSTRUCTION = """
You are an expert disaster response and structural engineering analyst. 
You will be provided with an aerial post-disaster image AND localized metadata annotations. 
Your task is to combine the visual evidence from the image with the hard numbers from the metadata to generate a highly professional, concise, and tactical rescue report. 
Do not hallucinate hazards, but explicitly mention severe visual hazards (like mudflows, floods, or blocked roads) even if the metadata does not explicitly list them.

You must strictly output your assessment following this schema without deviation:
### 1. Priority Zones (Geospatial Mapping)
[Identify areas containing structural failures or severe isolation where survivors are likely trapped.]
### 2. Structural Damage & Collapse Characterization
[Classify the observed architectural failures based on the provided data.]
### 3. Hazard Avoidance & Logistics Constraints
[Highlight secondary tactical risks visible in the image or noted in the data.]

Constraint: Do not include introductory or concluding pleasantries. Maintain an authoritative, objective, and operational tone.
"""

def parse_xbd_json(json_path):
    """Digs into the xBD JSON file and extracts damage counts."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    disaster_type = data['metadata'].get('disaster_type', 'unknown disaster')
    damage_counts = {
        'no-damage': 0, 'minor-damage': 0, 'major-damage': 0, 'destroyed': 0, 'un-classified': 0
    }
    
    features = data['features']['xy']
    total_buildings = len(features)
    
    for feature in features:
        damage_level = feature['properties'].get('subtype', 'un-classified')
        if damage_level in damage_counts:
            damage_counts[damage_level] += 1
            
    summary = (
        f"Disaster Type: {disaster_type}\n"
        f"Total Buildings Detected: {total_buildings}\n"
        f"Damage Assessment: {damage_counts['destroyed']} destroyed, "
        f"{damage_counts['major-damage']} major damage, "
        f"{damage_counts['minor-damage']} minor damage, "
        f"{damage_counts['no-damage']} intact."
    )
    return summary

def load_existing_progress():
    """Reads the existing JSONL dataset to see which images are already processed."""
    processed_images = set()
    if os.path.exists(DATASET_OUT_PATH):
        with open(DATASET_OUT_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        row = json.loads(line)
                        processed_images.add(row['image'])
                    except Exception:
                        continue
    return processed_images

def generate_dataset():
    # 1. Load progress tracker to allow safe reruns
    processed_history = load_existing_progress()
    current_count = len(processed_history)
    
    print(f"Current progress: {current_count}/{TARGET_SAMPLE_COUNT} samples already in dataset.")
    
    if current_count >= TARGET_SAMPLE_COUNT:
        print(f"Target of {TARGET_SAMPLE_COUNT} samples already met! Exiting.")
        return

    print(f"Scanning {PROCESSED_LABEL_DIR} for JSON metadata...")
    json_files = glob.glob(os.path.join(PROCESSED_LABEL_DIR, "*.json"))
    
    if not json_files:
        print("No JSON files found! Run your Image Standardizer script first to populate data directories.")
        return

    print(f"Found {len(json_files)} total local labels available. Initializing Multimodal Gemini API...")
    
    # Extract all available Gemini API keys from the configuration list
    api_keys = [k.strip() for k in GEMINI_API_KEYS if k and k.strip()]
    if not api_keys:
        print("ERROR: No GEMINI_API_KEY found. Please add them to your .env file.")
        return
    
    print(f"Loaded {len(api_keys)} Gemini API keys for rotation.")
    current_key_idx = 0
    client = genai.Client(api_key=api_keys[current_key_idx])
    
    current_model_idx = 0
    current_model = MODELS[current_model_idx]

    # Open in append mode ('a') so we don't wipe out previous progress
    with open(DATASET_OUT_PATH, 'a', encoding='utf-8') as f_out:
        for json_path in json_files:
            # Check if we hit the ceiling mid-loop
            if current_count >= TARGET_SAMPLE_COUNT:
                print(f"\nTarget reached: {current_count} samples generated.")
                break
                
            base_name = os.path.basename(json_path).replace('.json', '')
            image_path = os.path.join(PROCESSED_IMG_DIR, f"{base_name}.jpg")
            image_reference = f"processed_images/{base_name}.jpg"
            
            # Skip if already in the dataset file from a prior run
            if image_reference in processed_history:
                continue
                
            if not os.path.exists(image_path):
                print(f"WARNING: Skipping {base_name}, no matching processed JPEG found.")
                continue
            
            max_retries = (len(api_keys) * len(MODELS)) + 3
            retries = 0
            success = False
            
            while retries < max_retries and not success:
                try:
                    # 1. Parse Metadata
                    metadata_summary = parse_xbd_json(json_path)
                    
                    # 2. Open image payload
                    img = Image.open(image_path)
                    
                    # 3. Assemble prompt
                    prompt_text = (
                        f"Analyze this aerial view and identify priority zones for search and rescue operations.\n\n"
                        f"Metadata Annotations:\n{metadata_summary}"
                    )
                    
                    # 4. Multimodal API Call
                    response = client.models.generate_content(
                        model=current_model,
                        contents=[prompt_text, img],
                        config=types.GenerateContentConfig(
                            system_instruction=SYSTEM_INSTRUCTION
                        )
                    )
                    ground_truth = response.text.strip()
                    
                    # 5. Build standard instruction-response JSONL line
                    jsonl_row = {
                        "image": image_reference,
                        "instruction": "Analyze this aerial view and identify priority zones for search and rescue operations.",
                        "response": ground_truth
                    }
                    
                    f_out.write(json.dumps(jsonl_row) + '\n')
                    f_out.flush()  # Push directly to storage file
                    
                    processed_history.add(image_reference)
                    current_count += 1
                    print(f"[{current_count}/{TARGET_SAMPLE_COUNT}] SUCCESS: Documented {base_name}")
                    
                    success = True
                    # --- API RATE LIMIT PROTECTION ---
                    # Throttling to stay under typical RPM caps
                    time.sleep(4.3) 
                    
                except Exception as e:
                    err = str(e).lower()
                    
                    if "503" in err or "unavailable" in err:
                        print(f"503 Server Overloaded. Waiting 15 seconds before retry...")
                        time.sleep(15)
                        retries += 1
                        
                    elif "429" in err or "quota" in err:
                        # Print the raw error so we can inspect it for specific quota strings as recommended
                        print(f"Quota error encountered: {repr(e)}")
                        
                        if "requestsperday" in err or "generate_content_free_tier_requests" in err:
                            print(f"{current_model} daily quota exhausted for this key.")
                            current_model_idx += 1
                            
                            if current_model_idx >= len(MODELS):
                                print("All fallback models exhausted for this key. Rotating key...")
                                current_key_idx = (current_key_idx + 1) % len(api_keys)
                                client = genai.Client(api_key=api_keys[current_key_idx])
                                current_model_idx = 0 # reset models for new key
                                current_model = MODELS[current_model_idx]
                                print(f"Switched to API Key #{current_key_idx + 1} (out of {len(api_keys)}).")
                                retries += 1
                            else:
                                current_model = MODELS[current_model_idx]
                                print(f"Switching to fallback model: {current_model}")
                                retries += 1
                                
                        elif "requestsperminute" in err:
                            print("RPM exceeded. Waiting 60 seconds...")
                            time.sleep(60)
                            retries += 1
                            
                        elif "tokensperminute" in err:
                            print("TPM exceeded. Waiting 30 seconds...")
                            time.sleep(30)
                            retries += 1
                            
                        else:
                            print(f"Unknown 429/Quota error! Rotating API key...")
                            current_key_idx = (current_key_idx + 1) % len(api_keys)
                            client = genai.Client(api_key=api_keys[current_key_idx])
                            print(f"Switched to API Key #{current_key_idx + 1} (out of {len(api_keys)}).")
                            retries += 1
                            time.sleep(2)
                    else:
                        print(f"ERROR processing {base_name}: {e}")
                        break
                        
            if not success:
                print(f"Failed to process {base_name} after trying available fallback strategies. Cooling down for 45s...")
                time.sleep(45)

    print(f"\nExecution terminated. Current Total: {current_count} rows written to {DATASET_OUT_PATH}")

if __name__ == "__main__":
    valid_keys = [k for k in GEMINI_API_KEYS if k and k.strip()]
    if not valid_keys:
        print("ERROR: No GEMINI_API_KEY found in the script configuration.")
    else:
        generate_dataset()