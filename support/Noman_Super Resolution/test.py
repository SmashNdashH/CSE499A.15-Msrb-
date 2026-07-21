import os
import cv2
import matplotlib.pyplot as plt

# 1. Establish absolute pathways
original_path = '/content/VRT/sample_data/disaster_image.jpg'
upscaled_path = '/content/VRT/results/001_VRT_videosr_bi_REDS_6frames/000/0000.png'

# Load the images safely
original_img = cv2.imread(original_path)
upscaled_img = cv2.imread(upscaled_path)

# Convert from BGR to RGB for accurate plotting
original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
upscaled_img = cv2.cvtColor(upscaled_img, cv2.COLOR_BGR2RGB)

# 2. Dynamically calculate the image dimensions
h_org, w_org, _ = original_img.shape
h_up, w_up, _ = upscaled_img.shape

print(f"📐 Original Dimensions: {w_org}x{h_org}")
print(f"📐 Upscaled Dimensions: {w_up}x{h_up}")

import os
import cv2
import matplotlib.pyplot as plt

# 1. Establish absolute pathways
original_path = '/content/VRT/sample_data/disaster_image.jpg'
upscaled_path = '/content/VRT/results/001_VRT_videosr_bi_REDS_6frames/000/0000.png'

# Load the images safely
original_img = cv2.imread(original_path)
upscaled_img = cv2.imread(upscaled_path)

# Convert from BGR to RGB for accurate plotting
original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
upscaled_img = cv2.cvtColor(upscaled_img, cv2.COLOR_BGR2RGB)

# 2. Dynamically calculate the image dimensions
h_org, w_org, _ = original_img.shape
h_up, w_up, _ = upscaled_img.shape

print(f"📐 Original Dimensions: {w_org}x{h_org}")
print(f"📐 Upscaled Dimensions: {w_up}x{h_up}")

# 3. 100px patch directly from the absolute center
crop_size_org = 100
x_org = w_org // 2 - (crop_size_org // 2)
y_org = h_org // 2 - (crop_size_org // 2)
patch_org = original_img[y_org:y_org+crop_size_org, x_org:x_org+crop_size_org]

# Scale coordinates dynamically by the exact upscale factor for the high-res image
scale_factor_x = w_up // w_org
scale_factor_y = h_up // h_org

crop_size_up = crop_size_org * scale_factor_x
x_up = x_org * scale_factor_x
y_up = y_org * scale_factor_y
patch_up = upscaled_img[y_up:y_up+crop_size_up, x_up:x_up+crop_size_up]

# Final safety assertion to verify the arrays contain data before plotting
if patch_org.size == 0 or patch_up.size == 0:
    raise ValueError(f"Slicing error: Patch sizing computed as zero. Org patch shape: {patch_org.shape}, Up patch shape: {patch_up.shape}")

# 4. Plot side-by-side to expose the micro-structural differences
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(patch_org)
axes[0].set_title(f"Original Image Center Crop ({w_org}x{h_org})\n[Notice Pixel Blocks & Blurring]", fontsize=11)
axes[0].axis('off')

axes[1].imshow(patch_up)
axes[1].set_title(f"VRT Super-Resolved Center Crop ({w_up}x{h_up})\n[Reconstructed Sharp Textures]", fontsize=11)
axes[1].axis('off')

plt.tight_layout()
plt.show()
crop_size_org = 100
x_org = w_org // 2 - (crop_size_org // 2)
y_org = h_org // 2 - (crop_size_org // 2)
patch_org = original_img[y_org:y_org+crop_size_org, x_org:x_org+crop_size_org]

# Scale coordinates dynamically by the exact upscale factor for the high-res image
scale_factor_x = w_up // w_org
scale_factor_y = h_up // h_org

crop_size_up = crop_size_org * scale_factor_x
x_up = x_org * scale_factor_x
y_up = y_org * scale_factor_y
patch_up = upscaled_img[y_up:y_up+crop_size_up, x_up:x_up+crop_size_up]

# Final safety assertion to verify the arrays contain data before plotting
if patch_org.size == 0 or patch_up.size == 0:
    raise ValueError(f"Slicing error: Patch sizing computed as zero. Org patch shape: {patch_org.shape}, Up patch shape: {patch_up.shape}")

# 4. Plot side-by-side to expose the micro-structural differences
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(patch_org)
axes[0].set_title(f"Original Image Center Crop ({w_org}x{h_org})\n[Notice Pixel Blocks & Blurring]", fontsize=11)
axes[0].axis('off')

axes[1].imshow(patch_up)
axes[1].set_title(f"VRT Super-Resolved Center Crop ({w_up}x{h_up})\n[Reconstructed Sharp Textures]", fontsize=11)
axes[1].axis('off')

plt.tight_layout()
plt.show()
