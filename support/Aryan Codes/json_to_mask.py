import os
import json
import numpy as np
import cv2
from shapely import wkt
from tqdm import tqdm

# Mapping from xView2 string classes to integer values
DAMAGE_DICT = {
    "no-damage": 1,
    "minor-damage": 2,
    "major-damage": 3,
    "destroyed": 4,
    "un-classified": 0
}

def create_masks_from_json(json_dir, output_dir):
    """
    Reads xView2 JSON files containing polygon coordinates and damage classes,
    and converts them into 1024x1024 PNG mask images.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Only process POST-disaster json files as per your requirement
    json_files = [f for f in os.listdir(json_dir) if f.endswith('post_disaster.json')]
    
    print(f"Found {len(json_files)} POST-disaster JSON files. Starting conversion...")
    
    for j_file in tqdm(json_files):
        json_path = os.path.join(json_dir, j_file)
        
        # We assume the mask should be the exact size of xBD images
        width, height = 1024, 1024
        
        # Create an empty black mask (Background = 0)
        mask = np.zeros((height, width), dtype=np.uint8)
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Iterate over all building polygons in the JSON
        if 'features' in data and 'xy' in data['features']:
            for feature in data['features']['xy']:
                properties = feature['properties']
                poly_wkt = feature['wkt']
                
                # Get the damage subtype string
                subtype = properties.get('subtype', 'un-classified')
                class_value = DAMAGE_DICT.get(subtype, 0)
                
                # Parse the Well-Known Text (WKT) polygon string using shapely
                polygon = wkt.loads(poly_wkt)
                
                # Extract the exterior coordinates of the polygon
                coords = np.array(polygon.exterior.coords)
                
                # OpenCV needs coordinates as integer (x, y) arrays of shape (N, 1, 2)
                pts = np.round(coords).astype(np.int32)
                pts = pts.reshape((-1, 1, 2))
                
                # Fill the polygon on the mask with the respective class value
                cv2.fillPoly(mask, [pts], class_value)
                
        # The output mask should have the exact same base name as the image/json
        # We save it as a PNG image in the output directory
        base_name = j_file.replace('.json', '.png')
        out_path = os.path.join(output_dir, base_name)
        
        cv2.imwrite(out_path, mask)
        
    print(f"Finished generating masks! Saved to: {output_dir}")

if __name__ == "__main__":
    # The directory where your JSON labels are currently sitting
    JSON_DIR = r'F:\NSU\cse499 a-b\xview\raw pre-post images'
    
    # Let's create a new folder just for the converted masks
    # so they don't get mixed up with the raw images.
    OUTPUT_MASK_DIR = r'F:\NSU\cse499 a-b\xview\masks_png'
    
    create_masks_from_json(JSON_DIR, OUTPUT_MASK_DIR)
