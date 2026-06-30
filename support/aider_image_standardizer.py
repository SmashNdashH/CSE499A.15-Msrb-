#Synthetic JSON example (script generates this per image):
import os
import glob
import json
import shutil
import random  # Required for fixing sequential bias
import hashlib
from datetime import datetime, timezone
from PIL import Image

# --- PATH CONFIGURATION ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Point this to your raw extracted AIDER dataset folder (e.g., inside the 'Train' directory or root)
SOURCE_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw_dataset", "AIDER")

# Your project workspace directories
PROCESSED_IMG_DIR = os.path.join(PROJECT_ROOT, "data", "aider", "processed_images")
PROCESSED_LABEL_DIR = os.path.join(PROJECT_ROOT, "data", "aider", "processed_labels")

# The tracking file to prevent processing duplicates across multiple runs
TRACKER_FILE = os.path.join(PROJECT_ROOT, "data", "aider", "processed_tracker.txt")

# Create output directories if they don't exist
os.makedirs(PROCESSED_IMG_DIR, exist_ok=True)
os.makedirs(PROCESSED_LABEL_DIR, exist_ok=True)


def load_processed_history():
    """Loads the list of already processed image base names."""
    if not os.path.exists(TRACKER_FILE):
        return set()
    processed = set()
    with open(TRACKER_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split('|')
                base = parts[0].strip().replace('.png', '').replace('.jpg', '').replace('.jpeg', '')
                processed.add(base)
    return processed

def append_to_history(original, output, md5_hash, status):
    """Appends a processed image record to the tracker."""
    timestamp = datetime.now(timezone.utc).isoformat()
    line = f"{original} | {output} | {md5_hash} | {timestamp} | {status}"
    with open(TRACKER_FILE, 'a') as f:
        f.write(line + '\n')


def standardize_image_for_qwen(input_path, output_path, max_edge=1280):
    """Standardizes image to RGB and caps resolution to avoid OOM on T4 GPUs."""
    try:
        with Image.open(input_path) as img:
            # Skip naturally grayscale imagery per spec
            if img.mode == 'L':
                print(f"Skipping {input_path}: Grayscale image detected.")
                return False, "skipped_grayscale", "NONE"
                
            # Drop alpha/multispectral channels
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
            width, height = img.size
            # Dynamically downscale if it exceeds max_edge, preserving aspect ratio
            if max(width, height) > max_edge:
                scaling_factor = max_edge / float(max(width, height))
                new_size = (int(width * scaling_factor), int(height * scaling_factor))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
                
            # Minimum resolution gate (min edge must be at least 256)
            if min(img.size) < 256:
                print(f"Skipping {input_path}: Resolution below 256x256 after scaling ({img.size}).")
                return False, "skipped_too_small", "NONE"
                
            # Save cleanly as JPEG
            img.save(output_path, "JPEG", quality=95)
            
            # Integrity check
            Image.open(output_path).verify()
            
            # Compute MD5
            with open(output_path, "rb") as f:
                md5_hash = hashlib.md5(f.read()).hexdigest()
                
            return True, "success", md5_hash
            
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False, "failed_error", "NONE"


def build_aider_pilot_batch(target_count=None, class_filter=None):
    print(f"Scanning {SOURCE_DATA_DIR} for AIDER disaster imagery...")
    
    # 1. Discover all image files by walking through subfolders (classes)
    # Supported common image extensions in AIDER
    valid_extensions = ("*.jpg", "*.jpeg", "*.png")
    all_image_paths = []
    
    # Walk through the dataset class subdirectories
    for root, dirs, files in os.walk(SOURCE_DATA_DIR):
        # Infer class name from the folder name
        class_name = os.path.basename(root)
        
        if class_filter and class_filter.lower() != class_name.lower():
            continue
            
        for ext in valid_extensions:
            all_image_paths.extend(glob.glob(os.path.join(root, ext)))
            
    if not all_image_paths:
        print("No images found. Please verify your SOURCE_DATA_DIR path structural design.")
        return

    # 2. SHUFFLE TO PREVENT SEQUENTIAL/CLASS BIAS
    random.seed(42)
    random.shuffle(all_image_paths)
    print(f"Discovered {len(all_image_paths)} images. Shuffled to balance class exposure.")
    
    # 3. Load tracking history
    processed_history = load_processed_history()
    print(f"Found {len(processed_history)} images already processed in previous batches.")
    
    successful_samples = 0
    
    # 4. Process the dataset
    for img_path in all_image_paths:
        if target_count is not None and successful_samples >= target_count:
            break
            
        filename = os.path.basename(img_path)
        # Separate name and extension safely
        base_name, _ = os.path.splitext(filename)
        
        # Pull class assignment from folder layout
        class_label = os.path.basename(os.path.dirname(img_path))
        
        # Unique identifier includes class directory to prevent name collisions across folders
        unique_id = f"{class_label}_{base_name}"
        
        if unique_id in processed_history:
            continue
            
        # Target output filename setup
        output_filename = f"{unique_id}.jpg"
        output_img_path = os.path.join(PROCESSED_IMG_DIR, output_filename)
        
        # Standardize for Qwen processing
        success, status, md5_hash = standardize_image_for_qwen(img_path, output_img_path)
        
        if success:
            # Generate a companion classification meta file to match your pipeline's downstream logic
            label_filename = f"{unique_id}.json"
            output_label_path = os.path.join(PROCESSED_LABEL_DIR, label_filename)
            
            meta_label_data = {
                "dataset": "AIDER",
                "original_filename": filename,
                "assigned_class": class_label,
                "processed_timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            with open(output_label_path, 'w') as lf:
                json.dump(meta_label_data, lf, indent=4)
            
            # Log to tracking repository immediately
            append_to_history(filename, output_filename, md5_hash, status)
            processed_history.add(unique_id)
            
            successful_samples += 1
            print(f"[{successful_samples}] Processed Class [{class_label}]: {base_name}")
        else:
            append_to_history(filename, "NONE", "NONE", status)
            processed_history.add(unique_id)

    # 5. Final Synthesis Report
    print("\nAIDER Pipeline Segment Complete!")
    print(f"Images converted and saved to: {PROCESSED_IMG_DIR}")
    print(f"Synthetic labels saved to: {PROCESSED_LABEL_DIR}")
    print(f"Total historical entries recorded: {len(processed_history)}")


if __name__ == "__main__":
    # Set target_count to an integer (e.g., 100) for a mini pilot test, or None to process all elements
    build_aider_pilot_batch(target_count=None, class_filter=None)