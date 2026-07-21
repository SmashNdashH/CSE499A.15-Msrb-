import os
import shutil

# 1. Let Python automatically look for your image in the workspace
possible_paths = [
    '/content/sample_data/AIDAR_collapsed_building_image0464.jpg',
    '/content/VRT/sample_data/disaster_image.jpg',
    'sample_data/AIDAR_collapsed_building_image0464.jpg'
]

uploaded_image = None
for path in possible_paths:
    if os.path.exists(path):
        uploaded_image = path
        break

if uploaded_image is None:
    raise FileNotFoundError("Could not find the image. Please make sure it's fully uploaded inside the sample_data folder!")

print(f"Found your image at: {uploaded_image}")

# 2. Define the subfolder structure VRT expects
vrt_input_folder = '/content/VRT/testsets/custom_disaster/000'
os.makedirs(vrt_input_folder, exist_ok=True)

# 3. Duplicate the static frame 4 times to construct the pseudo-video clip sequence
for frame_idx in range(4):
    target_frame_name = f"{frame_idx:04d}.png"
    target_path = os.path.join(vrt_input_folder, target_frame_name)
    shutil.copy(uploaded_image, target_path)

print(f"Successfully staged your imagery data in: {vrt_input_folder}")
