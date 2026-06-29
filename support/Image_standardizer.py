import os
import glob
import json
import shutil
import random  # Required for fixing sequential bias
import hashlib
from datetime import datetime, timezone
from PIL import Image

# --- PATH CONFIGURATION ---
# Using raw strings (r"") for Windows paths to prevent backslash errors
SOURCE_IMG_DIR = r"D:\CSE499AB_project\data\xView2 Challenge Dataset - train and test\train\images"
SOURCE_LABEL_DIR = r"D:\CSE499AB_project\data\xView2 Challenge Dataset - train and test\train\labels"

# Your project workspace directories
PROCESSED_IMG_DIR = r"D:\CSE499AB_project\data\processed_images"
PROCESSED_LABEL_DIR = r"D:\CSE499AB_project\data\processed_labels"

# The tracking file to prevent processing duplicates across multiple runs
TRACKER_FILE = r"D:\CSE499AB_project\data\processed_tracker.txt"

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
                base = parts[0].strip().replace('.png', '')
                if len(parts) >= 5:
                    status = parts[4].strip()
                    # Treat anything recorded (success or skipped) as processed to avoid retries
                    processed.add(base)
                else:
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


def build_pilot_batch(target_count=100, disaster_filter=None):
    print(f"Scanning {SOURCE_IMG_DIR} for post-disaster images...")
    
    # 1. Find all post_disaster images
    search_pattern = os.path.join(SOURCE_IMG_DIR, "*_post_disaster.png")
    all_post_images = glob.glob(search_pattern)
    
    if disaster_filter:
        all_post_images = [img for img in all_post_images if disaster_filter in img]
        
    # 2. SHUFFLE TO PREVENT SEQUENTIAL BIAS
    # Seed 42 ensures that if teammates run this, they get the exact same random set
    random.seed(42)
    random.shuffle(all_post_images)
    print("Images shuffled to ensure diverse disaster coverage.")
    
    # 3. Load tracking history
    processed_history = load_processed_history()
    print(f"Found {len(processed_history)} images already processed in previous batches.")
    
    successful_samples = 0
    
    # 4. Process the batch
    for img_path in all_post_images:
        if target_count is not None and successful_samples >= target_count:
            break
            
        filename = os.path.basename(img_path)
        base_name = filename.replace(".png", "")
        
        # Skip if already processed in a previous run
        if base_name in processed_history:
            continue
            
        # Locate corresponding label
        label_filename = base_name + ".json"
        source_label_path = os.path.join(SOURCE_LABEL_DIR, label_filename)
        
        # Ensure label exists before processing image
        if not os.path.exists(source_label_path):
            continue 
            
        output_img_path = os.path.join(PROCESSED_IMG_DIR, base_name + ".jpg")
        success, status, md5_hash = standardize_image_for_qwen(img_path, output_img_path)
        
        if success:
            output_label_path = os.path.join(PROCESSED_LABEL_DIR, label_filename)
            shutil.copy2(source_label_path, output_label_path)
            
            # Log it to the tracker file immediately
            append_to_history(filename, os.path.basename(output_img_path), md5_hash, status)
            processed_history.add(base_name)
            
            successful_samples += 1
            print(f"[{successful_samples}/{target_count}] Processed: {base_name}")
        else:
            append_to_history(filename, "NONE", "NONE", status)
            processed_history.add(base_name)

    # 5. Final Report
    print("\nBatch Complete!")
    print(f"Images saved to: {PROCESSED_IMG_DIR}")
    print(f"Labels saved to: {PROCESSED_LABEL_DIR}")
    print(f"Total historical images processed so far: {len(processed_history)}")

# Run the extractor
if __name__ == "__main__":
    # Set target_count=None to process the entire raw dataset for 2D stratified sampling downstream
    build_pilot_batch(target_count=None, disaster_filter=None)