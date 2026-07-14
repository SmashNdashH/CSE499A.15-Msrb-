import cv2
import os

# 1. Define paths to check where the image is stored
path_options = [
    '/content/VRT/sample_data/collapsed_building_image0469.jpg' 
]

high_res_source = None
for path in path_options:
    if os.path.exists(path):
        high_res_source = path
        break

if high_res_source is None:
    raise FileNotFoundError("Could not find 'AIDAR_collapsed_building_image0464.jpg' in /content/ or /content/sample_data/. Please verify where the file was uploaded in your Colab file explorer panel on the left.")

# 2. Load the image safely
src_img = cv2.imread(high_res_source)
h, w, _ = src_img.shape
print(f"✅ Found source image at: {high_res_source}")
print(f"📐 Original High-Res Size: {w}x{h}")

# 3. Mathematically downscale it by 4x using a clean bicubic filter
target_w, target_h = w // 4, h // 4
low_res_img = cv2.resize(src_img, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
print(f"📉 Synthetically Downscaled Size: {target_w}x{target_h}")

# 4. Create the clean VRT input folder structure
vrt_input_dir = '/content/VRT/testsets/synthetic_disaster/000'
os.makedirs(vrt_input_dir, exist_ok=True)

# 5. Save it 4 times sequentially to simulate the VRT pseudo-video stream
for frame_idx in range(4):
    frame_name = f"{frame_idx:04d}.png"
    cv2.imwrite(os.path.join(vrt_input_dir, frame_name), low_res_img)

print("🎉 Success! Your synthetic low-resolution sequence is staged and ready for testing.")